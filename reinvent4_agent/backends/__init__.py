# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLM backend abstraction for the Drug Discovery Agent.

Provides a common interface over Anthropic and OpenAI-compatible APIs
so the agent loop is backend-agnostic.  Each backend owns its own
message history format, tool schema conversion, and thinking support.
"""

from __future__ import annotations

import os
import socket
import ssl
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: Per-probe timeout when diagnosing an unreachable endpoint.  The probes
#: only run on the error path, but the user is waiting on them.
_PROBE_TIMEOUT = 3.0


class EndpointError(ConnectionError):
    """The LLM endpoint is unreachable or misconfigured.

    Subclasses :class:`ConnectionError` so the CLI prints the (already
    formatted) message instead of a bare traceback.
    """


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call, fed back to the LLM."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ChatResult:
    """Unified response from any LLM backend."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None
    needs_tool_response: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    #: Part of ``output_tokens`` spent on hidden reasoning, when the
    #: endpoint reports it.  Billed and capped as output, but invisible.
    reasoning_tokens: int = 0


class LLMBackend(ABC):
    """Abstract base for LLM API backends.

    Each backend maintains its own conversation history in its native
    format and handles schema conversion for tools.
    """

    @abstractmethod
    def complete(
        self,
        tools: list[dict[str, Any]],
        max_tokens: int,
        *,
        force_tool_use: bool = False,
    ) -> ChatResult:
        """Send the current conversation to the LLM and return a result.

        ``tools`` are raw MCP tool definitions (name, description,
        input_schema).  The backend converts them to its native format.

        ``force_tool_use`` asks the model to emit a tool call rather than
        a plain-text answer (``tool_choice=required``, or Anthropic
        ``any``).  The agent leaves it off so a turn can answer in text.
        """

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        """Append a user message to the conversation history."""

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None:
        """Append tool results so the LLM can continue reasoning."""

    @abstractmethod
    def repair_dangling_tool_use(self) -> None:
        """Heal a history that ends with unanswered tool calls.

        If the last assistant turn requested tool calls but no matching
        tool results were ever appended (e.g. the previous turn crashed or
        was cancelled mid-loop), synthesize error results so the next API
        request is accepted instead of rejected with HTTP 400.
        """

    @abstractmethod
    def reset(self, system_prompt: str) -> None:
        """Clear history and reinitialise with the system prompt."""


def connection_failure_report(
    exc: BaseException,
    endpoint: str,
    *,
    verify_ssl: bool = True,
    hints: Sequence[str] = (),
) -> str:
    """Explain why ``endpoint`` could not be reached.

    SDK connection errors say little more than "Connection error", so
    unwrap the underlying exception chain and probe the endpoint (DNS,
    TCP, TLS) to show which layer actually failed.  Everything here runs
    on a failure path, so no probe may raise.
    """
    lines = [f"Cannot reach LLM API at {endpoint}", "", "  Cause:"]
    lines += [f"    {c}" for c in _cause_chain(exc)]

    parts = urlsplit(endpoint)
    host = parts.hostname
    if host:
        port = parts.port or (80 if parts.scheme == "http" else 443)
        lines += ["", f"  Endpoint probe ({host}:{port}):"]
        lines += [
            f"    {line}"
            for line in _probe(host, port, parts.scheme, verify_ssl)
        ]

    proxies = _proxy_env()
    lines += ["", "  Environment:"]
    lines.append(
        f"    TLS verification: {'on' if verify_ssl else 'off'}"
    )
    lines.append(
        "    Proxy: " + (", ".join(proxies) if proxies else "none set")
    )

    all_hints = [
        "Confirm the base URL path matches the endpoint "
        "(REINVENT4_AGENT_BASE_URL).",
        "If the endpoint listens on a non-standard port, the base URL "
        "must carry it (e.g. https://host:8443/...).",
        *hints,
    ]
    lines += ["", "  Check:"]
    lines += [f"    - {h}" for h in all_hints]
    return "\n".join(lines)


def rejected_request_report(
    exc: Any,
    endpoint: str,
    *,
    hints: Sequence[str] = (),
) -> str:
    """Describe an HTTP error, including the URL that produced it.

    The SDK only shows the status and response body, which is ambiguous
    when the endpoint routes on the URL path — the request line is what
    tells a routing mistake apart from a bad payload.
    """
    status = getattr(exc, "status_code", None) or "error"
    lines = [f"Endpoint rejected the request (HTTP {status})", ""]

    request = getattr(getattr(exc, "response", None), "request", None)
    if request is not None:
        lines.append(f"  Request:  {request.method} {request.url}")
    lines.append(f"  Endpoint: {endpoint}")
    lines += ["", f"  Response: {getattr(exc, 'message', exc)}"]

    if hints:
        lines += ["", "  Check:"]
        lines += [f"    - {h}" for h in hints]
    return "\n".join(lines)


def empty_response_report(
    endpoint: str,
    *,
    model: str,
    stop_reason: str | None,
    token_cap: str,
    output_tokens: int,
    reasoning_tokens: int,
    hints: Sequence[str] = (),
) -> str:
    """Explain a turn that produced neither text nor a tool call.

    The agent loop treats a text-less, tool-less turn as a final answer,
    so without this the session goes quiet with no indication that the
    model was cut off rather than finished.
    """
    lines = [
        "LLM returned an empty response (no text, no tool calls)",
        "",
        f"  Endpoint:    {endpoint}",
        f"  Model:       {model}",
        f"  Stop reason: {stop_reason or 'not reported'}",
        f"  Token cap:   {token_cap}",
    ]
    spent = f"  Spent:       {output_tokens} output token(s)"
    if reasoning_tokens:
        spent += f", {reasoning_tokens} of them on hidden reasoning"
    lines.append(spent)

    if hints:
        lines += ["", "  Check:"]
        lines += [f"    - {h}" for h in hints]
    return "\n".join(lines)


def _cause_chain(exc: BaseException, limit: int = 5) -> list[str]:
    """Flatten an exception's ``__cause__``/``__context__` chain."""
    out: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(out) < limit and id(cur) not in seen:
        seen.add(id(cur))
        detail = str(cur).strip()
        name = type(cur).__name__
        out.append(f"{name}: {detail}" if detail else name)
        cur = cur.__cause__ or cur.__context__
    return out


def _probe(
    host: str,
    port: int,
    scheme: str,
    verify_ssl: bool,
) -> list[str]:
    """Report which of DNS / TCP / TLS fails for ``host:port``."""
    try:
        infos = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    except OSError as exc:
        return [
            f"DNS: cannot resolve {host} ({exc})",
            "TCP: not attempted",
        ]

    addrs = sorted({info[4][0] for info in infos})
    out = [f"DNS: resolves to {', '.join(addrs)}"]

    sockaddr = infos[0][4]
    try:
        sock = socket.create_connection(sockaddr[:2], _PROBE_TIMEOUT)
    except OSError as exc:
        out.append(f"TCP: cannot connect to {sockaddr[0]}:{port} ({exc})")
        return out

    out.append(f"TCP: connected to {sockaddr[0]}:{port}")
    with sock:
        if scheme != "https":
            return out
        context = (
            ssl.create_default_context()
            if verify_ssl
            else ssl._create_unverified_context()  # noqa: SLF001
        )
        try:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                out.append(f"TLS: handshake OK ({tls.version()})")
        except ssl.SSLError as exc:
            out.append(f"TLS: handshake failed ({exc})")
        except OSError as exc:
            out.append(f"TLS: handshake failed ({exc})")
    return out


def _proxy_env() -> list[str]:
    """Proxy variables that httpx honours, which can silently redirect us."""
    names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    )
    found = []
    for name in names:
        value = os.environ.get(name) or os.environ.get(name.lower())
        if value:
            found.append(f"{name}={value}")
    return found


def create_backend(
    backend_type: str,
    *,
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
    gateway_key: str | None = None,
    api_version: str | None = None,
    verify_ssl: bool = True,
    thinking_budget: int | None = None,
    reasoning_effort: str | None = None,
    system_prompt: str,
) -> LLMBackend:
    """Factory: instantiate the correct backend by name.

    Parameters
    ----------
    backend_type:
        ``"anthropic"`` or ``"openai"``.
    model:
        Model name as the endpoint knows it (e.g. ``claude-sonnet-4.6``).
    base_url:
        Custom API endpoint
        (e.g. ``https://llm-gateway.example.com/Anthropic``).
    api_key:
        API key (or ``"dummy"`` when gateway auth is used).
    gateway_key:
        Value for the ``Ocp-Apim-Subscription-Key`` header.
    api_version:
        Azure ``api-version`` (OpenAI only); selects the Azure-style
        deployments endpoint.
    verify_ssl:
        Whether to verify TLS certificates.
    thinking_budget:
        Token budget for extended thinking (Anthropic only).
    reasoning_effort:
        Reasoning effort for o-series models (OpenAI only).
    system_prompt:
        System prompt to initialise the conversation.
    """
    if backend_type == "anthropic":
        from reinvent4_agent.backends.anthropic_backend import (
            AnthropicBackend,
        )

        return AnthropicBackend(
            model=model,
            base_url=base_url,
            api_key=api_key,
            gateway_key=gateway_key,
            verify_ssl=verify_ssl,
            thinking_budget=thinking_budget,
            system_prompt=system_prompt,
        )

    if backend_type == "openai":
        from reinvent4_agent.backends.openai_backend import OpenAIBackend

        return OpenAIBackend(
            model=model,
            base_url=base_url,
            api_key=api_key,
            gateway_key=gateway_key,
            api_version=api_version,
            verify_ssl=verify_ssl,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
        )

    raise ValueError(
        f"Unknown backend '{backend_type}'. Use 'anthropic' or 'openai'."
    )
