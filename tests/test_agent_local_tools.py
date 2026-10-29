# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tools the host answers itself, beside the shared MCP server's."""

from __future__ import annotations

import asyncio
from typing import Any

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, ToolCall, ToolResult

LOCAL = {
    "name": "campaign_log",
    "description": "Keep notes.",
    "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}},
}


class FakeMCP:
    def __init__(self) -> None:
        self.called: list[str] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "list_jobs", "description": "", "input_schema": {"type": "object"}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.called.append(name)
        return "[]"


class ScriptedBackend:
    """Asks for one tool call, then answers; records the tools it was offered."""

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.offered: list[str] = []
        self.results: list[ToolResult] = []
        self.calls = 0

    def repair_dangling_tool_use(self) -> None:
        pass

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results.extend(results)

    def complete(self, tools: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.offered = [tool["name"] for tool in tools]
        self.calls += 1
        if self.calls == 1:
            call = ToolCall(id="t1", name=self.tool, arguments={"goal": "Soluble JAK3 binders"})
            return ChatResult(tool_calls=[call], needs_tool_response=True, input_tokens=1200)
        return ChatResult(text="Noted.", input_tokens=1500)


def test_a_local_tool_is_offered_and_answered_by_the_host():
    async def run() -> None:
        mcp = FakeMCP()
        backend = ScriptedBackend("campaign_log")
        agent = Agent(mcp, backend)  # type: ignore[arg-type]
        seen: list[dict[str, Any]] = []
        agent.local_tools["campaign_log"] = (LOCAL, lambda arguments: seen.append(arguments) or '{"ok": true}')

        answer = await agent.chat("Remember the goal")

        assert answer == "Noted."
        assert backend.offered == ["list_jobs", "campaign_log"]
        assert seen == [{"goal": "Soluble JAK3 binders"}]
        assert mcp.called == []
        assert backend.results[0].content == '{"ok": true}'

    asyncio.run(run())


def test_a_failing_local_tool_is_an_error_result_not_a_crash():
    async def run() -> None:
        backend = ScriptedBackend("campaign_log")
        agent = Agent(FakeMCP(), backend)  # type: ignore[arg-type]

        def refuse(arguments: dict[str, Any]) -> str:
            raise ValueError("Give at least one of 'goal', 'finding' or 'next'.")

        agent.local_tools["campaign_log"] = (LOCAL, refuse)

        assert await agent.chat("Remember the goal") == "Noted."
        assert backend.results[0].is_error
        assert "Give at least one of" in backend.results[0].content

    asyncio.run(run())


def test_the_context_size_is_that_of_the_latest_call():
    async def run() -> None:
        agent = Agent(FakeMCP(), ScriptedBackend("list_jobs"))  # type: ignore[arg-type]
        await agent.chat("How are the runs?")
        assert agent.context_tokens == 1500
        assert agent.total_input_tokens == 2700

    asyncio.run(run())
