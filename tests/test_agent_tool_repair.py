"""Regression tests for dangling ``tool_use`` repair.

The Anthropic/Vertex (and OpenAI) APIs require every ``tool_use`` /
``tool_calls`` request to be immediately followed by matching tool
results.  If a turn crashes or is cancelled between appending the
assistant message and appending the tool results, the conversation is
left permanently invalid and every later request fails with HTTP 400.

These tests cover the two layers of defence:

1. ``Agent.chat`` always appends a tool result per tool call, even when a
   tool layer raises a ``BaseException`` (e.g. ``CancelledError``).
2. ``backend.repair_dangling_tool_use`` heals an already-corrupted
   history before the next request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reinvent4_agent.agent import Agent
from reinvent4_agent.backends import ChatResult, LLMBackend, ToolCall, ToolResult
from reinvent4_agent.backends.anthropic_backend import AnthropicBackend
from reinvent4_agent.backends.openai_backend import OpenAIBackend


class _FakeBackend(LLMBackend):
    """Backend that emits one tool call then records appended results."""

    def __init__(self) -> None:
        self.tool_results: list[ToolResult] | None = None
        self._calls = 0

    def complete(self, tools: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self._calls += 1
        return ChatResult(
            tool_calls=[ToolCall(id="toolu_1", name="boom", arguments={})],
            needs_tool_response=True,
        )

    def add_user_message(self, content: str) -> None:
        pass

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.tool_results = results

    def repair_dangling_tool_use(self) -> None:
        pass

    def reset(self, system_prompt: str) -> None:
        pass


class _FakeMCP:
    """MCP client whose tool call raises a (non-Exception) cancellation."""

    async def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "boom", "description": "", "input_schema": {}}]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise asyncio.CancelledError()


def test_chat_appends_tool_results_even_on_cancellation() -> None:
    backend = _FakeBackend()
    agent = Agent(_FakeMCP(), backend)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent.chat("hi"))

    assert backend.tool_results is not None
    assert [r.tool_call_id for r in backend.tool_results] == ["toolu_1"]
    assert backend.tool_results[0].is_error is True
    assert backend.tool_results[0].content == "Tool call aborted."


class _Block:
    """Minimal stand-in for an Anthropic SDK content block."""

    def __init__(self, type: str, id: str | None = None) -> None:
        self.type = type
        self.id = id


def test_anthropic_repairs_dangling_tool_use() -> None:
    backend = AnthropicBackend(model="x", api_key="dummy", system_prompt="sp")
    backend._messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            _Block("text"),
            _Block("tool_use", id="toolu_1"),
            _Block("tool_use", id="toolu_2"),
        ]},
    ]

    backend.repair_dangling_tool_use()

    last = backend._messages[-1]
    assert last["role"] == "user"
    ids = [b["tool_use_id"] for b in last["content"]]
    assert ids == ["toolu_1", "toolu_2"]
    assert all(b["is_error"] for b in last["content"])

    # Idempotent: a second call must not append again.
    backend.repair_dangling_tool_use()
    assert backend._messages[-1] is last


def test_anthropic_repair_noop_when_already_answered() -> None:
    backend = AnthropicBackend(model="x", api_key="dummy", system_prompt="sp")
    backend._messages = [
        {"role": "assistant", "content": [_Block("text")]},
    ]
    backend.repair_dangling_tool_use()
    assert len(backend._messages) == 1


def test_openai_repairs_dangling_tool_use() -> None:
    backend = OpenAIBackend(model="x", api_key="dummy", system_prompt="sp")
    backend._messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "boom", "arguments": "{}"}},
        ],
    })

    backend.repair_dangling_tool_use()

    tail = backend._messages[-1]
    assert tail["role"] == "tool"
    assert tail["tool_call_id"] == "call_1"
    assert tail["content"] == "Tool call aborted."
