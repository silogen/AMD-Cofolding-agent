# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Limits that keep a turn from looping or flooding the model's context."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from reinvent4_agent import agent as agent_module
from reinvent4_agent.agent import MAX_IDENTICAL_FAILURES, Agent
from reinvent4_agent.backends import ChatResult, LLMBackend, ToolCall, ToolResult


class _ScriptedBackend(LLMBackend):
    """Asks for the same tool call every round and records what it is sent."""

    def __init__(self, arguments: dict[str, Any] | None = None) -> None:
        self.arguments = arguments or {"ttp_id": "brd4"}
        self.rounds = 0
        self.results: list[ToolResult] = []
        self.messages: list[str] = []
        self.repairs = 0

    def complete(self, tools: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.rounds += 1
        return ChatResult(
            tool_calls=[ToolCall(id=f"t{self.rounds}", name="get_ttp", arguments=self.arguments)],
            needs_tool_response=True,
        )

    def add_user_message(self, content: str) -> None:
        self.messages.append(content)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results.extend(results)

    def repair_dangling_tool_use(self) -> None:
        self.repairs += 1

    def reset(self, system_prompt: str) -> None:
        pass


class _MCP:
    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply
        self.calls = 0

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "get_ttp", "description": "", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls += 1
        if self.reply is None:
            raise ValueError("Tool 'get_ttp' returned error: no such TTP")
        return self.reply


def test_a_turn_stops_after_the_same_call_fails_repeatedly() -> None:
    backend = _ScriptedBackend()
    mcp = _MCP()
    answer = asyncio.run(Agent(mcp, backend).chat("show the TTP"))

    assert mcp.calls == MAX_IDENTICAL_FAILURES
    assert backend.rounds == MAX_IDENTICAL_FAILURES
    assert "failed 3 times" in answer


def test_the_failure_count_starts_again_each_turn() -> None:
    backend = _ScriptedBackend()
    mcp = _MCP()
    agent = Agent(mcp, backend)
    asyncio.run(agent.chat("first"))
    asyncio.run(agent.chat("second"))
    assert mcp.calls == 2 * MAX_IDENTICAL_FAILURES


def test_a_long_tool_result_is_cut_for_the_model_but_not_for_the_ui(monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "MAX_TOOL_RESULT_CHARS", 100)
    seen: list[str] = []

    class Handler:
        def on_thinking(self, text): pass
        def on_tool_call(self, name, arguments): pass
        def on_tool_result(self, name, result, is_error): seen.append(result)
        def on_error(self, message): pass
        def on_monitor_alert(self, alert): pass
        def on_monitor_response(self, text): pass

    class OneRound(_ScriptedBackend):
        def complete(self, tools, max_tokens):
            self.rounds += 1
            if self.rounds == 1:
                return super().complete(tools, max_tokens)
            return ChatResult(text="done")

    backend = OneRound()
    asyncio.run(Agent(_MCP(reply="x" * 500), backend, event_handler=Handler()).chat("go"))

    assert len(seen[0]) == 500
    sent = backend.results[0].content
    assert sent.startswith("x" * 100)
    assert "cut at 100 of 500 characters" in sent


def test_the_next_turn_waits_for_the_reply_of_a_stopped_one() -> None:
    release = threading.Event()
    order: list[str] = []

    class Slow(_ScriptedBackend):
        def complete(self, tools, max_tokens):
            self.rounds += 1
            if self.rounds == 1:
                release.wait(5)
                order.append("late reply stored")
                return ChatResult(text="late")
            order.append("second turn asked")
            return ChatResult(text="second")

        def add_user_message(self, content):
            order.append(f"user: {content}")

    async def scenario() -> str:
        agent = Agent(_MCP(), Slow())
        first = asyncio.create_task(agent.chat("first"))
        await asyncio.sleep(0.1)
        first.cancel()
        second = asyncio.create_task(agent.chat("second"))
        await asyncio.sleep(0.1)
        release.set()
        return await second

    assert asyncio.run(scenario()) == "second"
    assert order == ["user: first", "late reply stored", "user: second", "second turn asked"]
