# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Retrying an LLM call that failed only because the endpoint was away."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from reinvent4_agent import agent as agent_module
from reinvent4_agent.agent import MAX_LLM_ATTEMPTS, Agent
from reinvent4_agent.backends import (
    ChatResult,
    EndpointError,
    LLMBackend,
    ToolResult,
    TransientEndpointError,
)

#: Shaped like the real report: a first line, then the diagnostics.
UNREACHABLE = TransientEndpointError(
    "Cannot reach LLM API at http://127.0.0.1:8001/v1\n"
    "\n"
    "  Cause:\n"
    "    APIConnectionError: Connection error.\n"
    "\n"
    "  Check:\n"
    "    - Confirm the base URL path matches the endpoint."
)


class _Flaky(LLMBackend):
    """Fails the first *failures* calls, then answers."""

    def __init__(self, failures: int, error: BaseException = UNREACHABLE) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def complete(self, tools: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return ChatResult(text="done")

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[ToolResult]) -> None:
        pass

    def repair_dangling_tool_use(self) -> None:
        pass

    def reset(self, system_prompt: str) -> None:
        pass


class _MCP:
    async def list_tools(self) -> list[dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise AssertionError("no tool should be called")


class _Handler:
    """Records only what this test cares about."""

    def __init__(self) -> None:
        self.notices: list[str] = []

    def on_thinking(self, text: str) -> None: pass
    def on_tool_call(self, name, arguments) -> None: pass
    def on_tool_result(self, name, result, is_error) -> None: pass
    def on_error(self, message: str) -> None: pass
    def on_notice(self, message: str) -> None: self.notices.append(message)
    def on_monitor_alert(self, alert) -> None: pass
    def on_monitor_response(self, text: str) -> None: pass


@pytest.fixture(autouse=True)
def quick_backoff(monkeypatch):
    """Keep the waiting out of the test suite's runtime."""
    monkeypatch.setattr(agent_module, "LLM_RETRY_DELAY", 0.01)


def test_a_turn_survives_an_endpoint_that_comes_back() -> None:
    backend = _Flaky(failures=1)
    handler = _Handler()

    answer = asyncio.run(Agent(_MCP(), backend, event_handler=handler).chat("go"))

    assert answer == "done"
    assert backend.calls == 2
    assert handler.notices == [
        f"The model endpoint is not answering, retrying (2 of {MAX_LLM_ATTEMPTS})..."
    ]


def test_an_endpoint_that_stays_away_fails_with_one_short_sentence() -> None:
    backend = _Flaky(failures=MAX_LLM_ATTEMPTS)

    with pytest.raises(TransientEndpointError) as excinfo:
        asyncio.run(Agent(_MCP(), backend).chat("go"))

    assert backend.calls == MAX_LLM_ATTEMPTS
    lines = str(excinfo.value).splitlines()
    assert lines[0] == (
        f"The model endpoint did not answer after {MAX_LLM_ATTEMPTS} attempts; "
        "it may be restarting, so try again in a moment."
    )
    # The diagnostics are kept, but below the first line, where the UI puts
    # them behind its detail toggle.
    assert "Cannot reach LLM API at http://127.0.0.1:8001/v1" in lines[1]
    assert "  Check:" in lines


def test_a_rejected_request_is_not_retried() -> None:
    backend = _Flaky(
        failures=MAX_LLM_ATTEMPTS,
        error=EndpointError("Endpoint rejected the request (HTTP 400)"),
    )

    with pytest.raises(EndpointError):
        asyncio.run(Agent(_MCP(), backend).chat("go"))

    assert backend.calls == 1


def test_stopping_the_turn_does_not_wait_for_the_next_attempt(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "LLM_RETRY_DELAY", 30.0)
    backend = _Flaky(failures=MAX_LLM_ATTEMPTS)

    async def scenario() -> None:
        turn = asyncio.create_task(Agent(_MCP(), backend).chat("go"))
        await asyncio.sleep(0.2)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn

    started = time.monotonic()
    asyncio.run(scenario())

    assert time.monotonic() - started < 5
    assert backend.calls == 1
