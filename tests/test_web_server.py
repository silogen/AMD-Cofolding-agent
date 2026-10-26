# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the chat socket of the web gateway.

The MCP server and the LLM are replaced by a fake agent, so these tests need
the web dependencies (FastAPI) but no GPU and no network.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("fastapi")

from fastapi import WebSocketDisconnect  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from reinvent4_agent.web import server  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.history: list[dict] = []

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
        self.backend.history.append({"role": "assistant", "content": f"answer to {content}"})
        return f"answer to {content}"


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
        assert [s["id"] for s in client.get("/api/sessions").json()] == [session["id"]]


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
        assert client.post("/api/tools/create_ttp", json={"arguments": {}}).status_code == 403
        assert client.post("/api/tools/delete_ttp", json={"arguments": {}}).status_code == 403
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
