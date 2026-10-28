# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for reasoning models on the OpenAI backend.

On gpt-5 and the o-series the response cap covers the model's hidden
reasoning tokens as well as its reply, so a cap sized for a plain chat
model gets spent entirely on reasoning: the endpoint answers with
``finish_reason="length"``, no content and no tool calls, while usage
reports a large completion-token count.  Such a turn used to be read as
"the model has finished", so the agent returned an empty string and the
session went quiet with no error.
"""

from __future__ import annotations

from typing import Any

import pytest

from reinvent4_agent.backends import EndpointError
from reinvent4_agent.backends.openai_backend import (
    MIN_REASONING_TOKEN_CAP,
    OpenAIBackend,
)


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.type = "function"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[_ToolCall] | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _CompletionTokensDetails:
    def __init__(self, reasoning_tokens: int) -> None:
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, completion_tokens: int, reasoning_tokens: int) -> None:
        self.prompt_tokens = 1234
        self.completion_tokens = completion_tokens
        self.completion_tokens_details = _CompletionTokensDetails(
            reasoning_tokens
        )


class _Response:
    def __init__(self, choice: _Choice, usage: _Usage) -> None:
        self.choices = [choice]
        self.usage = usage


def _stub_backend(
    backend: OpenAIBackend,
    response: _Response,
) -> list[dict[str, Any]]:
    """Replace the network call with ``response``, recording the payload."""
    sent: list[dict[str, Any]] = []

    def _request(kwargs: dict[str, Any]) -> Any:
        sent.append(kwargs)
        return response

    backend._request = _request  # type: ignore[method-assign]
    return sent


def _backend(model: str, **kwargs: Any) -> OpenAIBackend:
    return OpenAIBackend(
        model=model,
        api_key="dummy",
        base_url="https://llm.example.com/v1",
        system_prompt="sp",
        **kwargs,
    )


@pytest.mark.parametrize(
    "model",
    ["gpt-5", "GPT-5.4", "gpt-5-mini", "o1", "o3-mini", "amd-gpt-5"],
)
def test_reasoning_models_get_max_completion_tokens(model: str) -> None:
    """No reasoning_effort needed: the model name is enough to switch."""
    backend = _backend(model)
    sent = _stub_backend(
        backend,
        _Response(_Choice(_Message("hi"), "stop"), _Usage(10, 0)),
    )

    backend.complete([], 4096)

    assert "max_tokens" not in sent[0]
    assert sent[0]["max_completion_tokens"] == MIN_REASONING_TOKEN_CAP


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-4o", "claude-x"])
def test_chat_models_keep_max_tokens_verbatim(model: str) -> None:
    backend = _backend(model)
    sent = _stub_backend(
        backend,
        _Response(_Choice(_Message("hi"), "stop"), _Usage(10, 0)),
    )

    backend.complete([], 4096)

    assert sent[0]["max_tokens"] == 4096


def test_explicit_cap_above_the_floor_is_respected() -> None:
    backend = _backend("gpt-5")
    sent = _stub_backend(
        backend,
        _Response(_Choice(_Message("hi"), "stop"), _Usage(10, 0)),
    )

    backend.complete([], 100_000)

    assert sent[0]["max_completion_tokens"] == 100_000


def test_reasoning_exhausted_cap_raises_instead_of_going_quiet() -> None:
    backend = _backend("gpt-5")
    _stub_backend(
        backend,
        _Response(
            _Choice(_Message(content=None), "length"),
            _Usage(completion_tokens=32768, reasoning_tokens=32768),
        ),
    )
    before = len(backend._messages)

    with pytest.raises(EndpointError) as excinfo:
        backend.complete([], 4096)

    report = str(excinfo.value)
    assert "empty response" in report
    assert "length" in report
    assert "32768" in report
    assert "--max-tokens" in report
    # A content-less assistant turn must not enter the history: it says
    # nothing and some endpoints reject it on the next request.
    assert len(backend._messages) == before


def test_tool_calls_are_run_when_finish_reason_is_not_tool_calls() -> None:
    """Some proxies report "stop" even when they emit tool calls."""
    backend = _backend("gpt-5")
    _stub_backend(
        backend,
        _Response(
            _Choice(
                _Message(
                    content=None,
                    tool_calls=[_ToolCall("call_1", "get_jobs", "{}")],
                ),
                "stop",
            ),
            _Usage(64, 32),
        ),
    )

    result = backend.complete([], 4096)

    assert result.needs_tool_response is True
    assert [tc.name for tc in result.tool_calls] == ["get_jobs"]
    assert result.reasoning_tokens == 32


def test_reasoning_tokens_are_reported() -> None:
    backend = _backend("gpt-5")
    _stub_backend(
        backend,
        _Response(
            _Choice(_Message("done"), "stop"),
            _Usage(completion_tokens=900, reasoning_tokens=850),
        ),
    )

    result = backend.complete([], 40_000)

    assert result.output_tokens == 900
    assert result.reasoning_tokens == 850
