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


class FakeAgent:
    """Stands in for ``Agent``: a turn waits until the test ends it."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def chat(self, content: str) -> str:
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return content


@pytest.fixture
def agents(monkeypatch):
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


# -- messages --------------------------------------------------------------


@pytest.mark.parametrize("frame", ["[]", "null", "5", "not json"])
def test_a_message_that_is_not_an_object_is_reported(agents, frame):
    with TestClient(server.create_app()) as client, client.websocket_connect("/api/chat") as ws:
        receive_until(ws, "ready")
        ws.send_text(frame)
        assert receive_until(ws, "error")["message"] == "Malformed message."


def test_a_disconnect_cancels_the_running_turn(agents):
    with TestClient(server.create_app()) as client:
        with client.websocket_connect("/api/chat") as ws:
            receive_until(ws, "ready")
            ws.send_json({"type": "user_message", "content": "Design a ligand"})
            assert receive_until(ws, "status")["state"] == "busy"
        # Checked before the app shuts down, because shutdown cancels every
        # task anyway and would hide a turn that the disconnect left running.
        deadline = time.monotonic() + 5
        while not agents[0].cancelled and time.monotonic() < deadline:
            time.sleep(0.05)
        assert agents[0].cancelled
