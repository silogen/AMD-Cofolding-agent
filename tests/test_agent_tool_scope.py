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


# ---------------------------------------------------------------------------
# Translating a deployment's deny list into the agent's allow set
# ---------------------------------------------------------------------------
#
# Spelled as a deny list because that is what the deployment can state
# honestly: the conductor's writable TTP mount lets it create a base TTP, and
# withholding update_ttp/delete_ttp is what stops it rewriting one a campaign
# is already being measured against.


TOOLS = [{"name": n} for n in ("create_ttp", "update_ttp", "delete_ttp", "get_ttp")]


class TestDenyList:

    def test_nothing_denied_leaves_the_agent_unrestricted(self) -> None:
        # None rather than the full set, so a tool the MCP server gains at
        # runtime is not frozen out by a snapshot taken at startup.
        from reinvent4_agent.cli import permitted_tools

        assert permitted_tools(TOOLS, "") is None

    def test_denied_names_are_withheld_and_the_rest_allowed(self) -> None:
        from reinvent4_agent.cli import permitted_tools

        allowed = permitted_tools(TOOLS, "update_ttp,delete_ttp")

        assert allowed == {"create_ttp", "get_ttp"}

    def test_whitespace_and_empty_entries_are_tolerated(self) -> None:
        # It arrives from a YAML env var, where " a, b, " is ordinary.
        from reinvent4_agent.cli import permitted_tools

        assert permitted_tools(TOOLS, " update_ttp , , delete_ttp ") == {
            "create_ttp", "get_ttp"
        }

    def test_a_denied_name_the_server_does_not_serve_is_harmless(self) -> None:
        from reinvent4_agent.cli import permitted_tools

        assert permitted_tools(TOOLS, "no_such_tool") == {
            "create_ttp", "update_ttp", "delete_ttp", "get_ttp"
        }

    def test_the_deny_list_reaches_the_agent(self) -> None:
        """End to end: a withheld tool is refused rather than called."""
        from reinvent4_agent.cli import permitted_tools

        backend = _ScriptedBackend("delete_job")
        mcp = _CatalogMCP()
        agent = Agent(
            mcp, backend,
            allowed_tools=permitted_tools(
                [{"name": "delete_job"}, {"name": "read_results"}], "delete_job"
            ),
        )

        _chat(agent)

        assert mcp.called == []
