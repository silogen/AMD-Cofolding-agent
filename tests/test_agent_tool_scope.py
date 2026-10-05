"""Tool choice stays optional, and allow-lists apply after name resolution."""

from __future__ import annotations

import asyncio
from typing import Any

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, LLMBackend, ToolCall, ToolResult


class _ScriptedBackend(LLMBackend):
    """One tool call, then a text reply. Records whether a call was forced."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.force_flags: list[bool] = []
        self.tool_results: list[ToolResult] | None = None

    def complete(
        self,
        tools: list[dict[str, Any]],
        max_tokens: int,
        *,
        force_tool_use: bool = False,
    ) -> ChatResult:
        self.force_flags.append(force_tool_use)
        if len(self.force_flags) == 1:
            return ChatResult(
                tool_calls=[
                    ToolCall(id="call_1", name=self.tool_name, arguments={})
                ],
                needs_tool_response=True,
            )
        return ChatResult(text="done", needs_tool_response=False)

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.tool_results = results

    def repair_dangling_tool_use(self) -> None:
        pass

    def reset(self, system_prompt: str) -> None:
        pass


class _CatalogMCP:
    def __init__(self) -> None:
        self.called: list[str] = []
        self._tools = [
            {"name": "read_results", "description": "", "input_schema": {}},
            {"name": "delete_job", "description": "", "input_schema": {}},
        ]

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def resolve_tool_name(self, name: str) -> str | None:
        catalog = {t["name"] for t in self._tools}
        if name in catalog:
            return name
        if name == "wipe_job":
            return "delete_job"
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.called.append(name)
        return "ok"


def _chat(agent: Agent, message: str = "hello") -> str:
    return asyncio.run(agent.chat(message))


def test_chat_does_not_force_a_tool_call() -> None:
    backend = _ScriptedBackend("read_results")
    agent = Agent(_CatalogMCP(), backend)

    assert _chat(agent) == "done"
    assert backend.force_flags == [False, False]


def test_resolved_tool_outside_allow_list_is_not_executed() -> None:
    backend = _ScriptedBackend("delete_job")
    mcp = _CatalogMCP()
    agent = Agent(mcp, backend, allowed_tools={"read_results"})

    _chat(agent)

    assert mcp.called == []
    assert backend.tool_results is not None
    assert backend.tool_results[0].is_error is True
    assert "not available" in backend.tool_results[0].content
    assert "delete_job" in backend.tool_results[0].content
    assert "read_results" in backend.tool_results[0].content


def test_alias_of_a_disallowed_tool_is_not_executed() -> None:
    backend = _ScriptedBackend("wipe_job")
    mcp = _CatalogMCP()
    agent = Agent(mcp, backend, allowed_tools={"read_results"})

    _chat(agent)

    assert mcp.called == []
    assert backend.tool_results is not None
    assert "delete_job" in backend.tool_results[0].content


def test_allowed_tool_is_executed() -> None:
    backend = _ScriptedBackend("read_results")
    mcp = _CatalogMCP()
    agent = Agent(mcp, backend, allowed_tools={"read_results"})

    assert _chat(agent) == "done"
    assert mcp.called == ["read_results"]
