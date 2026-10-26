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

    # ------------------------------------------------------------------
    # LLMBackend interface
    # ------------------------------------------------------------------

    def complete(
        self,
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ChatResult:
        self.repair_dangling_tool_use()
        openai_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages,
            self._token_param: self._token_cap(max_tokens),
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._request(kwargs)
        except openai.BadRequestError as exc:
            swapped = self._swap_token_param(exc)
            if swapped is None:
                raise EndpointError(
                    rejected_request_report(
                        exc,
                        self._endpoint_label(),
                        hints=self._route_hints(),
                    )
                ) from exc
            old, new = swapped
            kwargs[new] = self._token_cap(kwargs.pop(old))
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
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Malformed arguments for tool %s: %r",
                    tc.function.name,
                    tc.function.arguments,
                )
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args)
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
        if not (message.content or "").strip() and not tool_calls:
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

        self._messages.append(_message_to_dict(message))

        return ChatResult(
            text=message.content,
            tool_calls=tool_calls,
            thinking=_reasoning_text(message),
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
                    "content": r.content,
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


def _message_to_dict(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert an SDK message object to a plain dict for the history."""
    d: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    reasoning = _reasoning_text(message)
    if reasoning:
        # Chat templates with preserved thinking (Qwen3.x) read it back from here.
        d["reasoning_content"] = reasoning
    return d


def _reasoning_text(message: ChatCompletionMessage) -> str | None:
    """Visible reasoning from an OpenAI-compatible server, if it sent any.

    vLLM and SGLang return it outside the OpenAI schema, as ``reasoning``
    or ``reasoning_content``; api.openai.com never does.
    """
    extra = getattr(message, "model_extra", None) or {}
    text = extra.get("reasoning") or extra.get("reasoning_content")
    if not isinstance(text, str):
        return None
    return text.strip() or None


def _get_username() -> str:
    """Get the current username, safe for non-interactive environments."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")
