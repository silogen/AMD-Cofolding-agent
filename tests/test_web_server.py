# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the chat socket of the web gateway.

The MCP server and the LLM are replaced by a fake agent, so these tests need
the web dependencies (FastAPI) but no GPU and no network.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import WebSocketDisconnect  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from reinvent4_agent.web import server  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return list(self.history)

    def import_history(self, messages: list[dict]) -> None:
        self.history = list(messages)


class FakeAgent:
    """Stands in for ``Agent``.

    A question starting with "quick" is answered at once; any other turn waits
    until the test ends it.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.backend = FakeBackend()

    async def chat(self, content: str) -> str:
        self.started.set()
        self.backend.history.append({"role": "user", "content": content})
        if not content.startswith("quick"):
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        answer = f"## Scored {content.split()[-1]}\n\nbody" if "heading" in content else f"answer to {content}"
        self.backend.history.append({"role": "assistant", "content": answer})
        return answer


@pytest.fixture
def agents(monkeypatch, tmp_path):
    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    made: list[FakeAgent] = []

    async def start(self) -> None:
        self.ready = True

    def new_agent(self, handler) -> FakeAgent:
        agent = FakeAgent()
        made.append(agent)
        return agent

    monkeypatch.setattr(server.AgentGateway, "start", start)
    monkeypatch.setattr(server.AgentGateway, "new_agent", new_agent)
    return made


def receive_until(ws, event_type):
    while True:
        event = ws.receive_json()
        if event["type"] == event_type:
            return event


# -- origin check ----------------------------------------------------------


@pytest.mark.parametrize(
    "origin, host",
    [
        (None, "localhost:8080"),
        ("http://localhost:8080", "localhost:8080"),
        ("http://127.0.0.1:5173", "localhost:8000"),
        ("http://node42:8080", "node42:8080"),
    ],
)
def test_the_ui_and_local_pages_may_open_the_chat(origin, host):
    assert server.origin_allowed(origin, host)


@pytest.mark.parametrize("origin", ["https://example.com", "null", "http://evil.localhost.example"])
def test_other_web_pages_may_not_open_the_chat(origin):
    assert not server.origin_allowed(origin, "localhost:8080")


def test_an_extra_origin_can_be_allowed(monkeypatch):
    monkeypatch.setenv("REINVENT4_WEB_ALLOWED_ORIGINS", "http://proxy.example:9000")
    assert server.origin_allowed("http://proxy.example:9000", "localhost:8080")


def test_a_foreign_origin_is_refused_by_the_socket(agents):
    headers = {"origin": "https://example.com"}
    with (
        TestClient(server.create_app()) as client,
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/api/chat", headers=headers) as ws,
    ):
        ws.receive_json()
    assert agents == []


# -- access token ----------------------------------------------------------


def test_without_a_token_configured_nothing_is_required(agents):
    with TestClient(server.create_app()) as client:
        assert client.get("/api/health").status_code == 200


def test_a_configured_token_is_required_on_http(agents, monkeypatch):
    monkeypatch.setenv("REINVENT4_WEB_TOKEN", "s3cret")
    with TestClient(server.create_app()) as client:
        assert client.get("/api/session").status_code == 401
        assert client.get("/api/session", headers={"X-DD-Token": "wrong"}).status_code == 401
        assert client.get("/api/session", headers={"X-DD-Token": "s3cret"}).status_code == 200


def test_a_configured_token_is_required_on_the_chat_socket(agents, monkeypatch):
    monkeypatch.setenv("REINVENT4_WEB_TOKEN", "s3cret")
    with TestClient(server.create_app()) as client:
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/chat") as ws:
            ws.receive_json()
        assert agents == []
        with client.websocket_connect("/api/chat", headers={"X-DD-Token": "s3cret"}) as ws:
            ws.send_text("[]")
            assert receive_until(ws, "notice")


# -- messages --------------------------------------------------------------


@pytest.mark.parametrize("frame", ["[]", "null", "5", "not json"])
def test_a_message_that_is_not_an_object_is_reported(agents, frame):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "ready")
        ws.send_text(frame)
        assert receive_until(ws, "notice")["message"] == "Malformed message."


# -- sessions --------------------------------------------------------------


def test_a_new_connection_starts_a_session(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "ready")
        session = receive_until(ws, "session")
        assert session["id"].startswith("s-")
        assert session["lastSeq"] == 0
        # Listed only once something was asked, so idle tabs leave no trace.
        assert client.get("/api/sessions").json() == []
        ws.send_json({"type": "user_message", "content": "quick hello"})
        receive_until(ws, "user_message")
        listed = client.get("/api/sessions").json()
        assert [s["id"] for s in listed] == [session["id"]]
        assert listed[0]["title"] == "quick hello"
        assert listed[0]["turns"] == 1


def test_a_disconnect_leaves_the_turn_running_and_a_reconnect_replays_it(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            session_id = receive_until(ws, "session")["id"]
            ws.send_json({"type": "user_message", "content": "Design a ligand"})
            assert receive_until(ws, "status")["state"] == "busy"
        time.sleep(0.2)
        assert not agents[0].cancelled

        with client.websocket_connect(f"/api/chat?session={session_id}") as ws:
            session = receive_until(ws, "session")
            assert session["busy"] is True
            replayed = receive_until(ws, "user_message")
            assert replayed["content"] == "Design a ligand"
            assert replayed["seq"] == 1
        assert len(agents) == 1


def test_a_reconnect_only_gets_the_events_it_missed(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            session_id = receive_until(ws, "session")["id"]
            ws.send_json({"type": "user_message", "content": "quick one"})
            answer = receive_until(ws, "assistant_message")
            idle = receive_until(ws, "status")
            assert idle["state"] == "idle"

        with client.websocket_connect(f"/api/chat?session={session_id}&after={answer['seq']}") as ws:
            receive_until(ws, "session")
            assert ws.receive_json() == idle


def test_cancel_stops_the_running_turn(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "session")
        ws.send_json({"type": "user_message", "content": "Design a ligand"})
        receive_until(ws, "status")
        ws.send_json({"type": "cancel"})
        assert receive_until(ws, "cancelled")
        assert receive_until(ws, "status")["state"] == "idle"
        assert agents[0].cancelled


def test_a_message_sets_the_reasoning_effort_of_its_turn(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "session")
        ws.send_json({"type": "user_message", "content": "quick one", "effort": "high"})
        assert receive_until(ws, "user_message")["effort"] == "high"
        receive_until(ws, "assistant_message")
        assert receive_until(ws, "status")["state"] == "idle"
        assert agents[0].backend.reasoning_effort == "high"
        ws.send_json({"type": "user_message", "content": "quick two"})
        assert "effort" not in receive_until(ws, "user_message")
        receive_until(ws, "assistant_message")
        assert agents[0].backend.reasoning_effort == "high"


@pytest.mark.parametrize("effort", ["max", "", 3, ["high"]])
def test_an_unknown_effort_is_refused(agents, effort):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session_id = receive_until(ws, "session")["id"]
        ws.send_json({"type": "user_message", "content": "quick one", "effort": effort})
        assert "Unknown effort" in receive_until(ws, "notice")["message"]
        assert client.get(f"/api/sessions/{session_id}/events").json()["events"] == []
        assert agents[0].backend.reasoning_effort is None


def test_a_second_question_during_a_turn_is_refused_to_that_browser_only(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session_id = receive_until(ws, "session")["id"]
        ws.send_json({"type": "user_message", "content": "Design a ligand"})
        receive_until(ws, "status")
        ws.send_json({"type": "user_message", "content": "and another"})
        assert "still working" in receive_until(ws, "notice")["message"]
        events = client.get(f"/api/sessions/{session_id}/events").json()["events"]
        assert [e["type"] for e in events] == ["user_message", "status"]
        ws.send_json({"type": "cancel"})
        receive_until(ws, "cancelled")


def test_a_session_survives_a_gateway_restart(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session_id = receive_until(ws, "session")["id"]
        ws.send_json({"type": "user_message", "content": "quick hello"})
        receive_until(ws, "assistant_message")
        receive_until(ws, "status")

    with TestClient(server.create_app()) as client:
        listed = client.get("/api/sessions").json()
        assert listed[0]["id"] == session_id
        assert listed[0]["title"] == "quick hello"
        assert listed[0]["turns"] == 1
        with client.websocket_connect(f"/api/chat?session={session_id}") as ws:
            receive_until(ws, "session")
            assert receive_until(ws, "assistant_message")["content"] == "answer to quick hello"
        restored = agents[-1].backend.history
        assert [m["role"] for m in restored] == ["user", "assistant"]


def test_an_unknown_session_starts_a_new_one_and_says_so(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat?session=s-20260101-000000-abcdef") as ws:
            session = receive_until(ws, "session")
            assert session["id"] != "s-20260101-000000-abcdef"
            assert "no longer exists" in receive_until(ws, "notice")["message"]


def test_the_first_answer_heading_names_the_session(agents):
    """A raw SMILES tells the user nothing; the answer's own heading does."""
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            session_id = receive_until(ws, "session")["id"]
            ws.send_json({"type": "user_message", "content": "quick heading c1ccccc1"})
            receive_until(ws, "assistant_message")
        assert client.get("/api/sessions").json()[0]["title"] == "Scored c1ccccc1"

        client.patch(f"/api/sessions/{session_id}", json={"title": "My run"})
        with client.websocket_connect(f"/api/chat?session={session_id}") as ws:
            receive_until(ws, "session")
            ws.send_json({"type": "user_message", "content": "quick heading CCO"})
            receive_until(ws, "assistant_message")
        assert client.get("/api/sessions").json()[0]["title"] == "My run"


def test_sessions_can_be_renamed_and_deleted(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            session_id = receive_until(ws, "session")["id"]
        assert client.patch(f"/api/sessions/{session_id}", json={"title": "BRD4 run"}).json()["title"] == "BRD4 run"
        assert client.patch(f"/api/sessions/{session_id}", json={"title": " "}).status_code == 422
        assert client.delete(f"/api/sessions/{session_id}").status_code == 200
        assert client.get("/api/sessions").json() == []
        assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_a_session_with_a_running_turn_cannot_be_deleted(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session_id = receive_until(ws, "session")["id"]
        ws.send_json({"type": "user_message", "content": "Design a ligand"})
        receive_until(ws, "status")
        assert client.delete(f"/api/sessions/{session_id}").status_code == 409
        ws.send_json({"type": "cancel"})
        receive_until(ws, "cancelled")


@pytest.mark.parametrize("bad", ["../etc", "s-1", "abc"])
def test_malformed_session_ids_are_not_found(agents, bad):
    with TestClient(server.create_app()) as client:
        assert client.get(f"/api/sessions/{bad}/events").status_code == 404


# -- direct tool calls -----------------------------------------------------


@pytest.fixture
def tools(agents, monkeypatch):
    calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        calls.append((name, arguments))
        if name == "get_ttp" and arguments.get("ttp_id") == "missing":
            raise ValueError("No TTP named missing")
        return '{"jobs": [{"job_id": "abc123", "state": "running"}]}'

    monkeypatch.setattr(server.AgentGateway, "call_tool", call_tool)
    monkeypatch.setattr(server.AgentGateway, "tool_names", property(lambda self: sorted(server.DIRECT_TOOLS | {"create_ttp"})))
    return calls


def test_a_panel_can_call_a_read_tool(tools):
    with TestClient(server.create_app()) as client:
        body = client.post("/api/tools/list_jobs", json={"arguments": {}}).json()
        assert body["result"]["jobs"][0]["job_id"] == "abc123"
        assert tools == [("list_jobs", {})]


def test_tools_that_create_or_delete_are_left_to_the_agent(tools):
    with TestClient(server.create_app()) as client:
        assert client.post("/api/tools/delete_ttp", json={"arguments": {}}).status_code == 403
        assert client.post("/api/tools/score_molecules", json={"arguments": {}}).status_code == 403
        assert tools == []


def test_a_failing_tool_reports_its_error(tools):
    with TestClient(server.create_app()) as client:
        response = client.post("/api/tools/get_ttp", json={"arguments": {"ttp_id": "missing"}})
        assert response.status_code == 502
        assert "No TTP named missing" in response.json()["detail"]


def test_tool_arguments_must_be_an_object(tools):
    with TestClient(server.create_app()) as client:
        assert client.post("/api/tools/list_jobs", json={"arguments": []}).status_code == 422


# -- job files -------------------------------------------------------------


def test_job_files_are_served_by_name_only(agents, monkeypatch, tmp_path):
    job = tmp_path / "jobs" / "job_abc123"
    job.mkdir(parents=True)
    (job / "results_1.csv").write_text("SMILES,total_score\nCCO,0.5\n")
    (job / "secret.txt").write_text("no")
    monkeypatch.setenv("REINVENT4_JOBS_DIR", str(tmp_path / "jobs"))
    with TestClient(server.create_app()) as client:
        assert client.get("/api/jobs/abc123/files/results_1.csv").text.startswith("SMILES")
        assert client.get("/api/jobs/abc123/files/secret.txt").status_code == 404
        assert client.get("/api/jobs/abc123/files/..%2Fsecret.txt").status_code == 404
        assert client.get("/api/jobs/ABC/files/results_1.csv").status_code == 404


# -- uploads ---------------------------------------------------------------

PDB = b"ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\nEND\n"


@pytest.fixture
def shared(agents, monkeypatch, tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    monkeypatch.setattr(server.shared_files, "SHARED_ROOT", root)
    return root


def test_an_uploaded_structure_is_stored_and_served(shared):
    with TestClient(server.create_app()) as client:
        response = client.post("/api/uploads?session=new&name=my%20pocket.pdb", content=PDB)
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "my_pocket.pdb"
        assert body["size"] == len(PDB)
        assert body["path"] == (shared / "uploads" / "new" / "my_pocket.pdb").as_posix()
        assert body["url"] == "/api/artifacts/uploads/new/my_pocket.pdb"
        assert client.get(body["url"]).content == PDB


def test_an_upload_goes_to_its_session_folder(shared):
    session_id = "s-20260101-000000-abcdef"
    with TestClient(server.create_app()) as client:
        body = client.post(f"/api/uploads?session={session_id}&name=ligands.sdf", content=b"x\n$$$$\n").json()
    assert (shared / "uploads" / session_id / "ligands.sdf").read_bytes() == b"x\n$$$$\n"
    assert body["url"] == f"/api/artifacts/uploads/{session_id}/ligands.sdf"


@pytest.mark.parametrize("name", ["../../etc/passwd.pdb", "..\\up.cif", "/abs/x.csv"])
def test_an_upload_name_cannot_leave_the_folder(shared, name):
    with TestClient(server.create_app()) as client:
        body = client.post("/api/uploads", params={"session": "new", "name": name}, content=PDB).json()
    stored = Path(body["path"])
    assert stored.parent == shared / "uploads" / "new"
    assert stored.is_file()


@pytest.mark.parametrize(
    "params",
    [
        {"session": "new", "name": "notes.txt"},
        {"session": "new", "name": "script.pdb.exe"},
        {"session": "new", "name": ".pdb"},
        {"session": "new"},
        {"session": "../x", "name": "a.pdb"},
        {"name": "a.pdb"},
    ],
)
def test_an_upload_needs_an_allowed_name_and_a_session(shared, params):
    with TestClient(server.create_app()) as client:
        assert client.post("/api/uploads", params=params, content=PDB).status_code == 422
    assert not (shared / "uploads").exists()


def test_an_empty_or_oversized_upload_is_refused(shared, monkeypatch):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    with TestClient(server.create_app()) as client:
        assert client.post("/api/uploads?session=new&name=a.pdb", content=b"").status_code == 422
        assert client.post("/api/uploads?session=new&name=a.pdb", content=PDB).status_code == 413


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, limit: int = -1) -> bytes:
        return self.data if limit < 0 else self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def rcsb(monkeypatch):
    requested: list[str] = []

    def urlopen(url, timeout=None):
        requested.append(url)
        if "0XXX" in url:
            raise server.urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if "9BAD" in url:
            raise server.urllib.error.URLError("no route to host")
        return FakeResponse(b"data_5UF0\n#\n")

    monkeypatch.setattr(server.urllib.request, "urlopen", urlopen)
    return requested


def test_a_pdb_entry_is_fetched_from_rcsb(shared, rcsb):
    with TestClient(server.create_app()) as client:
        response = client.post("/api/uploads/rcsb", json={"id": "5uf0", "session": "new"})
        assert response.status_code == 200
        body = response.json()
        assert rcsb == ["https://files.rcsb.org/download/5UF0.cif"]
        assert body["name"] == "5UF0.cif"
        assert body["url"] == "/api/artifacts/uploads/new/5UF0.cif"
        assert client.get(body["url"]).content == b"data_5UF0\n#\n"


@pytest.mark.parametrize(
    "body",
    [
        {"id": "BRD4", "session": "new"},
        {"id": "5uf", "session": "new"},
        {"id": 5, "session": "new"},
        {"id": "5UF0", "session": "x"},
        {"id": "5UF0", "session": 1},
    ],
)
def test_a_pdb_id_and_session_are_validated(shared, rcsb, body):
    with TestClient(server.create_app()) as client:
        assert client.post("/api/uploads/rcsb", json=body).status_code == 422
    assert rcsb == []


def test_rcsb_failures_are_reported(shared, rcsb):
    with TestClient(server.create_app()) as client:
        assert client.post("/api/uploads/rcsb", json={"id": "0XXX", "session": "new"}).status_code == 404
        response = client.post("/api/uploads/rcsb", json={"id": "9BAD", "session": "new"})
        assert response.status_code == 502
        assert "9BAD" in response.json()["detail"]