"""Anthropic tool_choice vs extended thinking."""

from __future__ import annotations

from typing import Any

from reinvent4_agent.backends.anthropic_backend import AnthropicBackend

_MCP_TOOL = {
    "name": "list_scoring_components",
    "description": "list",
    "input_schema": {"type": "object", "properties": {}},
}


class _TextBlock:
    type = "text"
    text = "ok"


class _Response:
    content = [_TextBlock()]
    stop_reason = "end_turn"
    usage = None


def _backend(**kwargs: Any) -> AnthropicBackend:
    return AnthropicBackend(
        model="claude-sonnet-4",
        api_key="dummy",
        system_prompt="sp",
        **kwargs,
    )


def _stub_create(backend: AnthropicBackend) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> _Response:
        sent.append(kwargs)
        return _Response()

    backend.client.messages.create = create  # type: ignore[method-assign]
    return sent


def test_force_tool_use_without_thinking_sets_tool_choice_any() -> None:
    backend = _backend()
    sent = _stub_create(backend)

    backend.complete([_MCP_TOOL], 128, force_tool_use=True)

    assert sent[0]["tool_choice"] == {"type": "any"}
    assert "thinking" not in sent[0]


def test_force_tool_use_with_thinking_omits_tool_choice() -> None:
    backend = _backend(thinking_budget=1024)
    sent = _stub_create(backend)

    backend.complete([_MCP_TOOL], 128, force_tool_use=True)

    assert "tool_choice" not in sent[0]
    assert sent[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
