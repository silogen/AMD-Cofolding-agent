# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The autonomy budget over the chat socket.

The loop itself is tested in ``test_web_autonomy.py`` without a server; this
file only pins the wire contract: the ``auto`` message, the ``auto`` event it
publishes, and ``auto_until`` in the session summary.  Needs the web
dependencies (FastAPI) but no GPU and no network.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from reinvent4_agent.web import server  # noqa: E402
from reinvent4_agent.web import sessions as web_sessions  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return list(self.history)

    def import_history(self, messages: list[dict]) -> None:
        self.history = list(messages)


class FakeAgent:
    """Answers at once, and says it is done as soon as it works on its own."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.backend = FakeBackend()

    async def chat(self, content: str) -> str:
        self.asked.append(content)
        await asyncio.sleep(0)
        if content.startswith("[AUTO]"):
            return "The run is still going, nothing new. AUTONOMY DONE"
        return "Launched the campaign."


@pytest.fixture
def agents(monkeypatch, tmp_path):
    monkeypatch.setenv("REINVENT4_WEB_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv(server.chat_models.CONTROL_DIR_ENV, raising=False)
    # Real continuations pause a minute between turns; the tests must not.
    monkeypatch.setattr(web_sessions, "AUTO_TURN_DELAY_S", 0.0)
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


def test_a_budget_is_announced_and_can_be_taken_back(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session = receive_until(ws, "session")
        assert session["auto_until"] is None

        ws.send_json({"type": "auto", "minutes": 60})
        started = receive_until(ws, "auto")
        assert started["until"] is not None
        # Live state, not conversation: it is neither numbered nor recorded.
        assert "seq" not in started

        ws.send_json({"type": "auto", "minutes": 0})
        assert receive_until(ws, "auto")["until"] is None


def test_a_browser_that_comes_back_sees_the_time_left(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            session_id = receive_until(ws, "session")["id"]
            ws.send_json({"type": "auto", "minutes": 60})
            until = receive_until(ws, "auto")["until"]

        with client.websocket_connect(f"/api/chat?session={session_id}") as ws:
            assert receive_until(ws, "session")["auto_until"] == until


def test_the_budget_is_capped(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "session")
        ws.send_json({"type": "auto", "minutes": 10_000})
        capped = receive_until(ws, "auto")["until"]
        ws.send_json({"type": "auto", "minutes": web_sessions.MAX_AUTONOMY_MINUTES})
        assert receive_until(ws, "auto")["until"] <= capped


@pytest.mark.parametrize("minutes", [-1, "60", None, True, [60]])
def test_a_budget_that_is_not_a_number_of_minutes_is_refused(agents, minutes):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "session")
        ws.send_json({"type": "auto", "minutes": minutes})
        assert "budget in minutes" in receive_until(ws, "notice")["message"]
        ws.send_json({"type": "auto", "minutes": 0})
        assert receive_until(ws, "auto")["until"] is None


def test_an_automatic_turn_opens_a_marked_turn_with_no_user_message(agents):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        session_id = receive_until(ws, "session")["id"]
        ws.send_json({"type": "auto", "minutes": 60})
        receive_until(ws, "auto")
        ws.send_json({"type": "user_message", "content": "Launch the campaign"})
        receive_until(ws, "assistant_message")
        assert receive_until(ws, "status")["state"] == "idle"
        # The automatic turn that follows, which says it is done.
        assert receive_until(ws, "status")["auto"] is True
        # The phrase is for the gateway: the conversation keeps the rest.
        answer = receive_until(ws, "assistant_message")["content"]
        assert "AUTONOMY" not in answer
        assert answer == "The run is still going, nothing new."
        assert receive_until(ws, "status")["state"] == "idle"
        # Saying it is done hands the budget back.
        assert receive_until(ws, "auto")["until"] is None

        events = client.get(f"/api/sessions/{session_id}/events").json()["events"]
        assert [e["type"] for e in events] == [
            "user_message",
            "status",
            "assistant_message",
            "status",
            "status",
            "assistant_message",
            "status",
        ]
        assert [e.get("auto") for e in events if e["type"] == "status" and e["state"] == "busy"] == [None, True]
        assert agents[0].asked[1].startswith("[AUTO]")
