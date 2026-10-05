"""Regression tests for tool calls that arrive as message text.

vLLM's ``mistral`` tool-call parser fails on Mistral Large 3
(``mistral_tool_parser.py`` logs "Error in extracting tool call from
response" with ``KeyError: 'name'``) and returns the model's raw
``[TOOL_CALLS]`` block as content.  The turn then carries no structured
tool calls, so the agent used to print the block and run nothing.
"""

from __future__ import annotations

import json
from typing import Any

from reinvent4_agent.backends.openai_backend import OpenAIBackend

MODEL = "mistralai/Mistral-Large-3-675B-Instruct-2512"

#: What the endpoint actually returned for "Score aspirin for QED and
#: molecular weight".  The nested ``components`` array is what defeats
#: the server-side regex: it matches that instead of the whole call.
TEXT_TOOL_CALL = (
    '[TOOL_CALLS]score_molecules{"smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"], '
    '"components": [{"type": "QED", "weight": 1}, '
    '{"type": "MolecularWeight", "weight": 1}]}'
)


class _Message:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 100
        self.completion_tokens = 20
        self.completion_tokens_details = None


class _Response:
    def __init__(self, message: _Message, finish_reason: str = "stop") -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = _Usage()


def _backend(model: str = MODEL) -> OpenAIBackend:
    return OpenAIBackend(
        model=model,
        api_key="dummy",
        base_url="http://dd-agentic-aim/v1",
        system_prompt="sp",
    )


def _stub(backend: OpenAIBackend, response: _Response) -> None:
    backend._request = lambda kwargs: response  # type: ignore[method-assign]


def test_text_tool_call_is_executed() -> None:
    backend = _backend()
    _stub(backend, _Response(_Message(TEXT_TOOL_CALL)))

    result = backend.complete([], 4096)

    assert result.needs_tool_response
    assert [tc.name for tc in result.tool_calls] == ["score_molecules"]
    assert result.tool_calls[0].arguments == {
        "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
        "components": [
            {"type": "QED", "weight": 1},
            {"type": "MolecularWeight", "weight": 1},
        ],
    }


def test_raw_block_is_not_shown_or_replayed() -> None:
    """The unparsed text must not reach the user or the next request."""
    backend = _backend()
    _stub(backend, _Response(_Message("Scoring aspirin.\n" + TEXT_TOOL_CALL)))

    result = backend.complete([], 4096)

    assert result.text == "Scoring aspirin."
    assistant = backend._messages[-1]
    assert assistant["content"] == "Scoring aspirin."
    assert assistant["tool_calls"][0]["function"]["name"] == "score_molecules"


def test_recovered_id_matches_mistral_tool_call_ids() -> None:
    """Mistral rejects the conversation unless ids are 9 alphanumerics."""
    backend = _backend()
    _stub(backend, _Response(_Message(TEXT_TOOL_CALL)))

    call_id = backend.complete([], 4096).tool_calls[0].id

    assert len(call_id) == 9
    assert call_id.isalnum()


def test_tool_results_can_answer_a_recovered_call() -> None:
    """The recovered id has to line up with the tool result that follows."""
    backend = _backend()
    _stub(backend, _Response(_Message(TEXT_TOOL_CALL)))

    result = backend.complete([], 4096)
    backend.repair_dangling_tool_use()

    assert backend._messages[-1] == {
        "role": "tool",
        "tool_call_id": result.tool_calls[0].id,
        "content": "Tool call aborted.",
    }


def test_multiple_text_tool_calls_are_all_recovered() -> None:
    backend = _backend()
    _stub(
        backend,
        _Response(
            _Message(
                '[TOOL_CALLS]get_qed{"smiles": "CCO"}'
                '[TOOL_CALLS]get_weight{"smiles": "CCO"}'
            )
        ),
    )

    result = backend.complete([], 4096)

    assert [tc.name for tc in result.tool_calls] == ["get_qed", "get_weight"]
    assert len({tc.id for tc in result.tool_calls}) == 2


def test_args_token_is_stripped_from_the_tool_name() -> None:
    """Tokenizer v13 puts [ARGS] between the name and the arguments."""
    backend = _backend()
    _stub(backend, _Response(_Message('[TOOL_CALLS]get_qed[ARGS]{"s": "CCO"}')))

    result = backend.complete([], 4096)

    assert result.tool_calls[0].name == "get_qed"
    assert result.tool_calls[0].arguments == {"s": "CCO"}


def test_pre_v11_json_array_layout_is_recovered() -> None:
    backend = _backend()
    _stub(
        backend,
        _Response(
            _Message(
                '[TOOL_CALLS] [{"name": "get_qed", '
                '"arguments": {"smiles": "CCO"}}]'
            )
        ),
    )

    result = backend.complete([], 4096)

    assert result.tool_calls[0].name == "get_qed"
    assert result.tool_calls[0].arguments == {"smiles": "CCO"}


def test_stringified_arguments_are_decoded() -> None:
    backend = _backend()
    _stub(
        backend,
        _Response(
            _Message(
                '[TOOL_CALLS][{"name": "get_qed", '
                '"arguments": "{\\"smiles\\": \\"CCO\\"}"}]'
            )
        ),
    )

    result = backend.complete([], 4096)

    assert result.tool_calls[0].arguments == {"smiles": "CCO"}


def test_unparsable_block_is_left_as_text() -> None:
    """Nothing to run: keep the text so the user sees what came back."""
    backend = _backend()
    _stub(backend, _Response(_Message("[TOOL_CALLS] not a tool call")))

    result = backend.complete([], 4096)

    assert result.tool_calls == []
    assert not result.needs_tool_response
    assert result.text == "[TOOL_CALLS] not a tool call"


def test_structured_tool_calls_are_left_alone() -> None:
    """A working parser must not go through the text path."""

    class _Function:
        name = "score_molecules"
        arguments = '{"smiles": ["CCO"]}'

    class _StructuredCall:
        id = "abcdefghi"
        type = "function"
        function = _Function()

    backend = _backend()
    _stub(
        backend,
        _Response(
            _Message("Scoring.", tool_calls=[_StructuredCall()]),
            finish_reason="tool_calls",
        ),
    )

    result = backend.complete([], 4096)

    assert [tc.id for tc in result.tool_calls] == ["abcdefghi"]
    assert result.text == "Scoring."


def test_dict_tool_arguments_are_parsed_and_stringified() -> None:
    """AIM/vLLM sometimes returns function.arguments as an object.

    Replaying that object in the next request makes the chat template do
    ``str + dict`` and reject the turn with HTTP 400.
    """

    class _Function:
        name = "score_molecules"
        arguments = {
            "smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            "components": [{"name": "QED"}, {"name": "MolecularWeight"}],
        }

    class _StructuredCall:
        id = "call_ds1"
        type = "function"
        function = _Function()

    backend = _backend(model="deepseek-ai/DeepSeek-V3.1-Terminus")
    _stub(
        backend,
        _Response(
            _Message(
                content={"text": "Scoring aspirin."},
                tool_calls=[_StructuredCall()],
            ),
            finish_reason="tool_calls",
        ),
    )

    result = backend.complete([], 4096)

    assert result.tool_calls[0].arguments["components"][0] == {"name": "QED"}
    assistant = backend._messages[-1]
    assert assistant["content"] == "Scoring aspirin."
    raw_args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(raw_args, str)
    assert json.loads(raw_args)["smiles"] == ["CC(=O)OC1=CC=CC=C1C(=O)O"]
