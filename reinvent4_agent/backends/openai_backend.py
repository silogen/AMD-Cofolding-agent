# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""OpenAI Chat Completions API backend.

Uses the ``openai`` SDK against any OpenAI-compatible endpoint: a
private LLM gateway, OpenAI itself, vLLM, Ollama, LiteLLM, and so on.

A gateway that publishes OpenAI models as Azure deployments is addressed
accordingly, so a gateway key — like an ``*.openai.azure.com`` host —
selects ``AzureOpenAI``: an
``api-version`` query parameter and a per-deployment base URL of
``<root>/openai/deployments/<model>``.  Every other endpoint uses the
plain ``openai.OpenAI`` client with the base URL as given, as does a URL
that explicitly names the ``/openai/v1`` route.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any
from urllib.parse import urlsplit

import openai
from openai.types.chat import ChatCompletionMessage

from reinvent4_agent.backends import (
    ChatResult,
    EndpointError,
    LLMBackend,
    ToolCall,
    ToolResult,
    connection_failure_report,
    empty_response_report,
    rejected_request_report,
)

logger = logging.getLogger(__name__)

#: Used when the Azure route is selected without an explicit version.
#: Function calling needs >= 2023-07-01-preview; reasoning parameters
#: need >= 2024-12-01-preview.
DEFAULT_API_VERSION = "2024-12-01-preview"

#: Reasoning models: they reject ``max_tokens``, and the cap they do take
#: covers hidden reasoning as well as the reply.  Matched on the model or
#: deployment id, which is often prefixed (``my-gpt-5``) or versioned
#: (``GPT-5.4``, ``o3-mini``).
_REASONING_MODEL_RE = re.compile(
    r"(?:^|[-._/])(gpt-?5|o[1345])(?:[-._]|$)",
    re.IGNORECASE,
)

#: Floor for the reasoning-model token cap.  A cap sized for a plain chat
#: model is easily consumed by reasoning alone, which truncates the turn
#: before a single visible token is produced.
MIN_REASONING_TOKEN_CAP = 32768

_DEPLOYMENTS_PATH = "/openai/deployments/"

#: OpenAI-compatible route some endpoints expose alongside the Azure
#: deployments one.  Naming it in the base URL opts out of the Azure
#: addressing that is otherwise inferred.
_GATEWAY_OPENAI_PATH = "/openai/v1"

#: Azure OpenAI resources, which are always addressed by deployment.
_AZURE_HOST_SUFFIX = ".openai.azure.com"

#: Token Mistral models emit ahead of a tool call.  It reaches the client
#: only when the server's own tool-call parser gave up on the output:
#: vLLM's ``mistral`` parser fails on Mistral Large 3 ("Error in
#: extracting tool call from response") and returns the block as content.
_MISTRAL_BOT_TOKEN = "[TOOL_CALLS]"

#: Separates the tool name from its arguments on tokenizer v13.  Servers
#: that decode without special tokens drop it, so treat it as optional.
_MISTRAL_ARGS_TOKEN = "[ARGS]"

#: Mistral validates tool-call ids as exactly nine alphanumerics and
#: rejects the whole conversation otherwise, so ids for recovered calls
#: have to match that shape rather than the usual ``call_*``.
_MISTRAL_TOOL_CALL_ID_LEN = 9

#: vLLM decodes tool-call arguments from the JSON text OpenAI clients send
#: into a Python object before it renders the chat template, because that
#: is what tool-use templates are expected to want.  DeepSeek-V3.1's own
#: template instead concatenates the value straight onto a string, so the
#: render raises and the whole conversation comes back as HTTP 400 as soon
#: as the history holds one assistant tool call.
_TEMPLATE_CONCAT_RE = re.compile(
    r'can only concatenate str \(not "(?:dict|list|int|float|bool)"\) to str'
)


class OpenAIBackend(LLMBackend):
    """OpenAI Chat Completions API backend.

    Parameters
    ----------
    model:
        Model name, or the deployment id on an Azure-style endpoint
        (e.g. ``gpt-5-mini``, ``GPT-4.1-mini``, ``o1``).
    base_url:
        API endpoint.  On a gateway this is the root
        (``https://llm-gateway.example.com``), since the deployment path
        is appended.  ``None`` → default ``api.openai.com``.
    api_key:
        API key / bearer token.  Falls back to ``gateway_key``.
    gateway_key:
        Gateway subscription key.  The gateway authenticates on the
        ``Ocp-Apim-Subscription-Key`` header; it is also passed as the
        bearer token because the SDK requires a key.
    api_version:
        Azure ``api-version``.  Defaults to :data:`DEFAULT_API_VERSION`
        whenever the Azure route is used.
    verify_ssl:
        Whether to verify TLS certificates.
    reasoning_effort:
        ``"low"``/``"medium"``/``"high"`` for reasoning models.  Reasoning
        models take their token cap as ``max_completion_tokens`` whether
        or not this is set, and that cap has to cover the reasoning
        tokens too — see :data:`MIN_REASONING_TOKEN_CAP`.
    system_prompt:
        System prompt for the conversation.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        gateway_key: str | None = None,
        api_version: str | None = None,
        verify_ssl: bool = True,
        reasoning_effort: str | None = None,
        system_prompt: str,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._verify_ssl = verify_ssl
        # gpt-5 and the o-series reject max_tokens, so send the parameter
        # they want up front: a proxy in front of the model may quietly
        # translate max_tokens instead of rejecting it, in which case
        # there is no error to adapt to.  complete() still switches either
        # way round if the endpoint disagrees.
        self._token_param = (
            "max_completion_tokens"
            if reasoning_effort or _is_reasoning_model(model)
            else "max_tokens"
        )

        default_headers: dict[str, str] = {}
        if gateway_key:
            default_headers["Ocp-Apim-Subscription-Key"] = gateway_key
            default_headers["user"] = _get_username()

        kwargs: dict[str, Any] = {}
        if default_headers:
            kwargs["default_headers"] = default_headers
        if not verify_ssl:
            # Must come from the SDK: it type-checks http_client against the
            # httpx distribution it was built against, which newer releases
            # ship as `httpx2`, so a plain `httpx.Client` is rejected.
            kwargs["http_client"] = openai.DefaultHttpxClient(verify=False)

        if _use_azure_route(base_url, gateway_key, api_version):
            if not base_url:
                raise ValueError(
                    "An Azure-style endpoint (api_version) requires a base "
                    "URL, e.g. https://llm-gateway.example.com — set "
                    "REINVENT4_AGENT_BASE_URL or --base-url."
                )
            # The Azure samples authenticate purely through the gateway
            # headers and send a placeholder key.
            self.api_version = api_version or DEFAULT_API_VERSION
            self.base_url = _deployment_url(base_url, model)
            self.client: openai.OpenAI = openai.AzureOpenAI(
                api_key=api_key or "dummy",
                api_version=self.api_version,
                base_url=self.base_url,
                **kwargs,
            )
        else:
            self.api_version = None
            self.base_url = base_url
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.OpenAI(
                api_key=api_key or gateway_key or "dummy",
                **kwargs,
            )

        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        # Set once an endpoint rejects a history that holds a tool call --
        # see _TEMPLATE_CONCAT_RE.
        self._quote_tool_arguments = False

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def complete(
        self,
        tools: list[dict[str, Any]],
        max_tokens: int,
        *,
        force_tool_use: bool = False,
    ) -> ChatResult:
        self.repair_dangling_tool_use()
        openai_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._history_messages(),
            self._token_param: self._token_cap(max_tokens),
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if openai_tools:
            kwargs["tools"] = openai_tools
            # DeepSeek-V3.1 (and some other OpenAI-compatible servers)
            # ignore tool_choice=auto and print a fake call as markdown.
            # Those calls are recovered from the message text. Opt in to
            # tool_choice=required only when the caller sets force_tool_use.
            kwargs["tool_choice"] = "required" if force_tool_use else "auto"

        try:
            response = self._request(kwargs)
        except openai.BadRequestError as exc:
            if not self._adapt_to_rejection(exc, kwargs):
                raise EndpointError(
                    rejected_request_report(
                        exc,
                        self._endpoint_label(),
                        hints=self._route_hints(),
                    )
                ) from exc
            try:
                response = self._request(kwargs)
            except openai.BadRequestError as retry_exc:
                raise EndpointError(
                    rejected_request_report(
                        retry_exc,
                        self._endpoint_label(),
                        hints=self._route_hints(),
                    )
                ) from retry_exc

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            args = _parse_tool_arguments(tc.function.arguments)
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
            )

        text = _message_content_for_history(message.content)
        recovered = False
        if not tool_calls and _MISTRAL_BOT_TOKEN in (text or ""):
            tool_calls, text = _recover_text_tool_calls(text or "")
            recovered = bool(tool_calls)
            if recovered:
                logger.warning(
                    "%s returned %d tool call(s) as text; the endpoint's "
                    "tool-call parser did not convert them. Reading them "
                    "from the message content.",
                    self.model,
                    len(tool_calls),
                )

        # Go by the tool calls themselves rather than the finish reason:
        # proxies in front of a model do not all report "tool_calls", and
        # ignoring the calls would strand the turn with nothing to say.
        needs_tool = len(tool_calls) > 0
        if needs_tool and choice.finish_reason != "tool_calls":
            logger.debug(
                "Endpoint reported finish_reason=%r alongside %d tool "
                "call(s); running them anyway.",
                choice.finish_reason,
                len(tool_calls),
            )

        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        reasoning_tokens = _reasoning_tokens(usage)

        # An empty turn would otherwise read as "the model is done", and
        # the agent loop would end the turn with nothing to show.  Leave
        # it out of the history too: a content-less assistant message
        # carries no information and some endpoints reject it.
        if not (text or "").strip() and not tool_calls:
            raise EndpointError(
                self._empty_response_report(
                    choice.finish_reason,
                    kwargs,
                    output_tokens,
                    reasoning_tokens,
                )
            )

        if choice.finish_reason == "length":
            logger.warning(
                "Response truncated at the token cap (%s=%s); the reply is "
                "incomplete.",
                self._token_param,
                kwargs.get(self._token_param),
            )

        if recovered:
            self._messages.append(_recovered_message_dict(text, tool_calls))
        else:
            self._messages.append(_message_to_dict(message))

        return ChatResult(
            text=text,
            tool_calls=tool_calls,
            thinking=None,
            needs_tool_response=needs_tool,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": _message_content_for_history(r.content) or "",
                }
            )

    def repair_dangling_tool_use(self) -> None:
        if not self._messages or self._messages[-1].get("role") != "assistant":
            return
        tool_calls = self._messages[-1].get("tool_calls") or []
        for tc in tool_calls:
            self._messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": "Tool call aborted.",
            })

    def reset(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages = [
            {"role": "system", "content": system_prompt},
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _endpoint_label(self) -> str:
        return self.base_url or "https://api.openai.com"

    def _route_hints(self) -> list[str]:
        """Advice for errors that point at the URL rather than the payload."""
        if self.api_version:
            return [
                f"'{self.model}' must be the Azure deployment id, and the "
                "request goes to "
                f"<base-url>{_DEPLOYMENTS_PATH}{self.model}/chat/completions"
                f"?api-version={self.api_version}.",
                "Drop --api-version to use the standard "
                f"{_GATEWAY_OPENAI_PATH} route instead.",
            ]
        return [
            "'Resource not found' means the endpoint serves neither this "
            f"route nor this model. List what it does serve with: curl -H "
            f"'Ocp-Apim-Subscription-Key: $LLM_GATEWAY_KEY' -H 'user: "
            f"{_get_username()}' {self._endpoint_label()}/models",
            "If that 404s too, the endpoint only exposes Azure "
            "deployments: set REINVENT4_AGENT_API_VERSION (e.g. "
            f"{DEFAULT_API_VERSION}) to switch to "
            f"<base-url>{_DEPLOYMENTS_PATH}<model>/chat/completions.",
            "A 'deployment not found' reply instead means part of the URL "
            "path was read as a deployment name — check "
            "REINVENT4_AGENT_BASE_URL.",
        ]

    def _history_messages(self) -> list[dict[str, Any]]:
        """The conversation in the shape this endpoint has accepted so far."""
        return [
            _history_message(m, quote_tool_arguments=self._quote_tool_arguments)
            for m in self._messages
        ]

    def _adapt_to_rejection(
        self,
        exc: openai.BadRequestError,
        kwargs: dict[str, Any],
    ) -> bool:
        """Rewrite ``kwargs`` for a retry, or report that nothing applies."""

        swapped = self._swap_token_param(exc)
        if swapped is not None:
            old, new = swapped
            kwargs[new] = self._token_cap(kwargs.pop(old))
            return True

        if self._quote_tool_arguments or not _TEMPLATE_CONCAT_RE.search(str(exc)):
            return False

        logger.info(
            "%s rejected a history containing a tool call because its chat "
            "template concatenates the decoded arguments; sending them as a "
            "JSON string instead.",
            self._endpoint_label(),
        )
        self._quote_tool_arguments = True
        kwargs["messages"] = self._history_messages()
        return True

    def _request(self, kwargs: dict[str, Any]) -> Any:
        """Send one request, translating transport errors into hints."""
        try:
            return self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as exc:
            raise EndpointError(
                connection_failure_report(
                    exc,
                    self._endpoint_label(),
                    verify_ssl=self._verify_ssl,
                    hints=[
                        "A gateway typically serves OpenAI models under "
                        "/openai/v1; Azure deployments need "
                        "--api-version instead.",
                        "If the endpoint uses a private CA, retry with "
                        "--no-verify-ssl.",
                    ],
                )
            ) from exc
        except openai.AuthenticationError as exc:
            raise EndpointError(
                rejected_request_report(
                    exc,
                    self._endpoint_label(),
                    hints=[
                        "Check LLM_GATEWAY_KEY / REINVENT4_AGENT_API_KEY.",
                        "A gateway may require the 'user' header to be "
                        "your own account id for service-application "
                        f"keys; it is currently sent as "
                        f"'{_get_username()}'.",
                    ],
                )
            ) from exc
        except openai.NotFoundError as exc:
            raise EndpointError(
                rejected_request_report(
                    exc,
                    self._endpoint_label(),
                    hints=self._route_hints(),
                )
            ) from exc

    def _token_cap(self, max_tokens: int) -> int:
        """The token cap to send, floored for reasoning models.

        On a reasoning model the cap bounds hidden reasoning *and* the
        reply, so a value sized for a plain chat model can be spent
        entirely on reasoning — the turn then comes back empty, and the
        longer the conversation grows the more reliably it does.
        """
        if self._token_param != "max_completion_tokens":
            return max_tokens
        if max_tokens >= MIN_REASONING_TOKEN_CAP:
            return max_tokens
        logger.info(
            "Raising the token cap from %d to %d for %s: the cap also has "
            "to cover the model's hidden reasoning tokens.",
            max_tokens,
            MIN_REASONING_TOKEN_CAP,
            self.model,
        )
        return MIN_REASONING_TOKEN_CAP

    def _empty_response_report(
        self,
        finish_reason: str | None,
        kwargs: dict[str, Any],
        output_tokens: int,
        reasoning_tokens: int,
    ) -> str:
        """Explain a turn that carried neither text nor a tool call."""
        cap = kwargs.get(self._token_param)
        hints: list[str] = []
        if finish_reason == "length":
            hints.append(
                "The turn hit the token cap. On a reasoning model that cap "
                "covers the hidden reasoning as well as the reply, so raise "
                "it well above the reply length you expect: --max-tokens "
                "65536 (or REINVENT4_AGENT_MAX_TOKENS)."
            )
            hints.append(
                "Spend fewer tokens on reasoning with --reasoning-effort "
                "low, and shorten the history with /reset."
            )
        elif finish_reason == "content_filter":
            hints.append(
                "The endpoint's content filter suppressed the reply; "
                "rephrase the request."
            )
        else:
            hints.append(
                "The endpoint accepted the request but returned no content. "
                "If it proxies the model, check whether it supports "
                f"tool/function calling for '{self.model}'."
            )
        return empty_response_report(
            self._endpoint_label(),
            model=self.model,
            stop_reason=finish_reason,
            token_cap=f"{self._token_param}={cap}",
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            hints=hints,
        )

    def _swap_token_param(
        self,
        exc: openai.BadRequestError,
    ) -> tuple[str, str] | None:
        """The token parameter to retry with, or ``None`` to give up.

        Which models take ``max_tokens`` and which take
        ``max_completion_tokens`` is endpoint-specific, so trust the
        rejection over the model name and switch in whichever direction
        it asks for.
        """
        detail = str(exc)
        swaps = {
            "max_tokens": "max_completion_tokens",
            "max_completion_tokens": "max_tokens",
        }
        wanted = swaps[self._token_param]
        if wanted not in detail:
            return None
        logger.info(
            "%s rejects %s; retrying with %s",
            self.model,
            self._token_param,
            wanted,
        )
        old, self._token_param = self._token_param, wanted
        return old, wanted

    @staticmethod
    def _convert_tools(
        mcp_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """MCP tool defs → OpenAI function-calling tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in mcp_tools
        ]


def _recover_text_tool_calls(content: str) -> tuple[list[ToolCall], str | None]:
    """Tool calls a server left in message text, plus the prose around them.

    A server whose tool-call parser fails returns the model's raw
    ``[TOOL_CALLS]`` block as content, so the turn arrives with no
    structured calls and the agent has nothing to run.  The call itself
    is well formed in that text, so read it instead of dropping the turn.
    Both Mistral layouts are accepted::

        [TOOL_CALLS]name{"arg": 1}          # tokenizer v11+
        [TOOL_CALLS][{"name": ..., "arguments": {...}}]   # earlier

    Returns no calls when nothing parses, leaving the text untouched for
    the caller to show as-is.
    """
    prose, marker, rest = content.partition(_MISTRAL_BOT_TOKEN)
    if not marker:
        return [], content
    tool_calls: list[ToolCall] = []
    for chunk in rest.split(_MISTRAL_BOT_TOKEN):
        tool_calls.extend(_chunk_tool_calls(chunk))
    if not tool_calls:
        return [], content
    return tool_calls, prose.strip() or None


def _chunk_tool_calls(chunk: str) -> list[ToolCall]:
    """Tool calls from one ``[TOOL_CALLS]``-delimited chunk of text."""
    decoded = _decode_json_prefix(chunk)
    if decoded is not None:
        entries = decoded if isinstance(decoded, list) else [decoded]
        return [
            call
            for entry in entries
            if (call := _entry_tool_call(entry)) is not None
        ]

    name, brace, arguments = chunk.partition("{")
    if not brace:
        return []
    name = name.replace(_MISTRAL_ARGS_TOKEN, "").strip()
    decoded = _decode_json_prefix(brace + arguments)
    if not name or not isinstance(decoded, dict):
        return []
    return [ToolCall(id=_tool_call_id(), name=name, arguments=decoded)]


def _entry_tool_call(entry: Any) -> ToolCall | None:
    """One ``{"name": ..., "arguments": ...}`` entry as a tool call."""
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    arguments = entry.get("arguments")
    if isinstance(arguments, str):
        arguments = _decode_json_prefix(arguments)
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return ToolCall(id=_tool_call_id(), name=name, arguments=arguments)


def _decode_json_prefix(text: str) -> Any | None:
    """The JSON object or array ``text`` starts with, else ``None``.

    Decoding only the prefix keeps whatever follows the arguments — a
    second call, a trailing sentence — from failing the parse.
    """
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    return value


def _tool_call_id() -> str:
    """An id for a recovered call, in the shape Mistral requires."""
    return uuid.uuid4().hex[:_MISTRAL_TOOL_CALL_ID_LEN]


def _recovered_message_dict(
    text: str | None,
    tool_calls: list[ToolCall],
) -> dict[str, Any]:
    """History entry for tool calls recovered from message text.

    The raw ``[TOOL_CALLS]`` text is replaced by the structured calls:
    sending it back as content would have the model read its own
    unparsed output as conversation.
    """
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    }


def _is_reasoning_model(model: str) -> bool:
    """Whether ``model`` names a reasoning model (gpt-5, o-series)."""
    return bool(_REASONING_MODEL_RE.search(model))


def _reasoning_tokens(usage: Any) -> int:
    """Hidden reasoning tokens from a usage object, 0 if not reported."""
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


def _use_azure_route(
    base_url: str | None,
    gateway_key: str | None,
    api_version: str | None,
) -> bool:
    """Whether to address the endpoint as Azure deployments.

    An explicit ``api_version`` always selects it.  Otherwise it is
    implied by a gateway key (a gateway fronting Azure publishes OpenAI
    models as deployments) or by an Azure OpenAI hostname — unless the
    URL names the OpenAI-compatible route, which wins.
    """
    if api_version:
        return True
    parts = urlsplit(base_url or "")
    if _GATEWAY_OPENAI_PATH in parts.path:
        return False
    return bool(gateway_key) or (parts.hostname or "").endswith(
        _AZURE_HOST_SUFFIX
    )


def _deployment_url(base_url: str, model: str) -> str:
    """Build the Azure deployments URL for a single deployment."""
    url = base_url.rstrip("/")
    if _DEPLOYMENTS_PATH.rstrip("/") in url:
        return url
    return f"{url}{_DEPLOYMENTS_PATH}{model}"


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Decode tool-call arguments from a string or an already-parsed object."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Malformed tool-call arguments: %r", raw)
        return {}
    if isinstance(parsed, dict):
        return parsed
    logger.warning("Tool-call arguments were JSON %s, not an object", type(parsed).__name__)
    return {}


def _json_string(value: Any) -> str:
    """JSON object the OpenAI chat template expects as a string."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _message_content_for_history(content: Any) -> str | None:
    """Coerce message content to a string.

    OpenAI-compatible servers (vLLM / AIM / DeepSeek) concatenate content
    into the chat template. A dict there raises
    ``can only concatenate str (not "dict") to str`` as HTTP 400.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    continue
                parts.append(text if isinstance(text, str) else json.dumps(text))
        return "".join(parts) or None
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content)
    return str(content)


def _history_message(
    message: dict[str, Any],
    *,
    quote_tool_arguments: bool = False,
) -> dict[str, Any]:
    """Copy one history entry with string content and tool arguments.

    ``quote_tool_arguments`` encodes the arguments JSON as a JSON string,
    so an endpoint that decodes them before rendering its chat template
    (see :data:`_TEMPLATE_CONCAT_RE`) is left with a string to render.
    """
    out = dict(message)
    if "content" in out:
        out["content"] = _message_content_for_history(out["content"])
    tool_calls = out.get("tool_calls")
    if tool_calls:
        coerced: list[dict[str, Any]] = []
        for tc in tool_calls:
            entry = dict(tc)
            function = dict(entry.get("function") or {})
            if "arguments" in function:
                arguments = _json_string(function["arguments"])
                function["arguments"] = (
                    json.dumps(arguments) if quote_tool_arguments else arguments
                )
            entry["function"] = function
            coerced.append(entry)
        out["tool_calls"] = coerced
    return out


def _message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert an SDK message object to a plain dict for the history."""
    d: dict[str, Any] = {
        "role": message.role,
        "content": _message_content_for_history(message.content),
    }
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": _json_string(tc.function.arguments),
                },
            }
            for tc in message.tool_calls
        ]
    return d


def _get_username() -> str:
    """Get the current username, safe for non-interactive environments."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")
