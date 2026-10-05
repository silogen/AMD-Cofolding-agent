"""Regression tests for endpoints that reject a history holding tool calls.

vLLM decodes ``function.arguments`` from JSON text into a Python object
before rendering the chat template, but DeepSeek-V3.1's own template
concatenates that value onto a string.  Every request made after the first
tool call therefore came back as::

    HTTP 400 - can only concatenate str (not "dict") to str

which stranded the turn: the tool had run, and the model could never be
told what it returned.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import openai
import pytest

from reinvent4_agent.backends import EndpointError
from reinvent4_agent.backends.openai_backend import OpenAIBackend

MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"


class _Message:
    def __init__(self, content: str | None = None) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = None


class _Usage:
    prompt_tokens = 100
    completion_tokens = 20
    completion_tokens_details = None


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [
            type("_Choice", (), {"message": _Message(content), "finish_reason": "stop"})()
        ]
        self.usage = _Usage()


def _bad_request(message: str) -> openai.BadRequestError:
    request = httpx.Request("POST", "http://dd-agentic-aim/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=None)


def _backend() -> OpenAIBackend:
    backend = OpenAIBackend(
        model=MODEL,
        api_key="dummy",
        base_url="http://dd-agentic-aim/v1",
        system_prompt="sp",
    )
    backend._messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "score_molecules",
                        "arguments": '{"smiles": ["CCO"]}',
                    },
                }
            ],
        }
    )
    backend._messages.append(
        {"role": "tool", "tool_call_id": "call_1", "content": '{"qed": 0.55}'}
    )
    return backend


def _record_requests(
    backend: OpenAIBackend,
    errors: list[str | None],
) -> list[dict[str, Any]]:
    """Stub the transport, raising ``errors`` in turn; record every payload.

    The payload is copied because a retry rewrites the dict in place.
    """
    sent: list[dict[str, Any]] = []

    def _request(kwargs: dict[str, Any]) -> Any:
        sent.append(copy.deepcopy(kwargs))
        error = errors.pop(0) if errors else None
        if error is not None:
            raise _bad_request(error)
        return _Response("Aspirin scores 0.55.")

    backend._request = _request  # type: ignore[method-assign]
    return sent


def _arguments(request: dict[str, Any]) -> str:
    assistant = next(m for m in request["messages"] if m["role"] == "assistant")
    return assistant["tool_calls"][0]["function"]["arguments"]


def test_rejected_history_is_retried_with_quoted_arguments() -> None:
    backend = _backend()
    sent = _record_requests(
        backend, ['can only concatenate str (not "dict") to str']
    )

    result = backend.complete([], 4096)

    assert result.text == "Aspirin scores 0.55."
    assert len(sent) == 2
    assert json.loads(_arguments(sent[0])) == {"smiles": ["CCO"]}
    # The retry sends the same JSON, wrapped so decoding it yields a string.
    assert json.loads(_arguments(sent[1])) == '{"smiles": ["CCO"]}'


def test_quoting_persists_for_the_rest_of_the_session() -> None:
    """Re-learning the encoding on every turn would double every request."""
    backend = _backend()
    sent = _record_requests(
        backend, ['can only concatenate str (not "list") to str']
    )

    backend.complete([], 4096)
    backend.add_user_message("and the molecular weight?")
    backend.complete([], 4096)

    assert len(sent) == 3
    assert json.loads(_arguments(sent[2])) == '{"smiles": ["CCO"]}'


def test_unrelated_rejection_is_reported() -> None:
    backend = _backend()
    sent = _record_requests(backend, ["model 'deepseek' does not exist"])

    with pytest.raises(EndpointError, match="does not exist"):
        backend.complete([], 4096)

    assert len(sent) == 1


def test_arguments_are_sent_unquoted_until_rejected() -> None:
    backend = _backend()
    sent = _record_requests(backend, [])

    backend.complete([], 4096)

    assert json.loads(_arguments(sent[0])) == {"smiles": ["CCO"]}
