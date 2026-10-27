# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for addressing a model that changes under a conversation.

The endpoint is replaced by a recorder, so these need the ``openai`` SDK but
no server::

    python -m pytest tests/test_backend_model_choice.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from reinvent4_agent.backends import ModelChoice  # noqa: E402
from reinvent4_agent.backends.openai_backend import OpenAIBackend  # noqa: E402

QWEN = ModelChoice(model="Qwen/Qwen3.8-27B")
GEMMA = ModelChoice(
    model="google/gemma-4-31B-it",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    supports_reasoning_effort=False,
)


def reply(text: str = "hello") -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text, tool_calls=None, model_extra={})
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, completion_tokens_details=None)
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def sent(monkeypatch):
    """Every request body the backend sends, instead of sending it."""
    recorded: list[dict] = []

    def create(**kwargs):
        recorded.append(kwargs)
        return reply()

    monkeypatch.setattr(
        OpenAIBackend,
        "_request",
        lambda self, kwargs: create(**kwargs),
    )
    return recorded


def backend(choices, **kwargs):
    """A backend whose next model comes from *choices*, one per request."""
    pending = list(choices)
    return OpenAIBackend(
        model=pending[0].model,
        base_url="http://localhost:8000/v1",
        model_choice=lambda: pending.pop(0),
        system_prompt="be useful",
        **kwargs,
    )


def test_a_swapped_model_is_addressed_on_the_next_turn(sent):
    llm = backend([QWEN, GEMMA])
    llm.add_user_message("first")
    llm.complete([], 4096)
    llm.add_user_message("second")
    llm.complete([], 4096)

    assert [body["model"] for body in sent] == ["Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"]
    assert llm.model == "google/gemma-4-31B-it"
    # The conversation itself is untouched by the swap.
    assert [m["role"] for m in llm.export_history()] == ["user", "assistant", "user", "assistant"]


def test_gemma_carries_enable_thinking_and_never_an_effort(sent):
    llm = backend([GEMMA], reasoning_effort="high")
    llm.add_user_message("first")
    llm.complete([], 4096)

    body = sent[0]
    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "reasoning_effort" not in body


def test_qwen_carries_its_effort_and_no_extra_body(sent):
    llm = backend([QWEN], reasoning_effort="medium")
    llm.add_user_message("first")
    llm.complete([], 4096)

    assert sent[0]["reasoning_effort"] == "medium"
    assert "extra_body" not in sent[0]


def test_without_a_chooser_the_constructed_model_is_used(sent):
    llm = OpenAIBackend(
        model="Qwen/Qwen3.8-27B",
        base_url="http://localhost:8000/v1",
        system_prompt="be useful",
    )
    llm.add_user_message("first")
    llm.complete([], 4096)
    assert sent[0]["model"] == "Qwen/Qwen3.8-27B"


def test_a_chooser_that_fails_leaves_the_last_model_standing(sent):
    def explode():
        raise OSError("the control directory went away")

    llm = OpenAIBackend(
        model="Qwen/Qwen3.8-27B",
        base_url="http://localhost:8000/v1",
        model_choice=explode,
        system_prompt="be useful",
    )
    llm.add_user_message("first")
    llm.complete([], 4096)
    assert sent[0]["model"] == "Qwen/Qwen3.8-27B"
