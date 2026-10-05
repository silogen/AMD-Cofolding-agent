"""Anthropic Messages API backend with extended thinking support.

Uses the ``anthropic`` SDK to talk to the Anthropic Messages API
(directly or via an AMD gateway).  Supports ``thinking`` blocks for
reasoning-capable models like claude-sonnet-4.6.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from reinvent4_agent.backends import (
    ChatResult,
    EndpointError,
    LLMBackend,
    ToolCall,
    ToolResult,
    connection_failure_report,
)

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """Anthropic Messages API backend.

    Parameters
    ----------
    model:
        Model name (e.g. ``claude-sonnet-4.6``, ``claude-sonnet-4``).
    base_url:
        Custom API endpoint.  ``None`` → default ``api.anthropic.com``.
    api_key:
        API key.  Use ``"dummy"`` when gateway auth handles authn.
    gateway_key:
        ``Ocp-Apim-Subscription-Key`` header value for AMD gateway.
    verify_ssl:
        Whether to verify TLS certificates.
    thinking_budget:
        Token budget for extended thinking.  ``None`` → thinking disabled.
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
        verify_ssl: bool = True,
        thinking_budget: int | None = None,
        system_prompt: str,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.thinking_budget = thinking_budget
        self._verify_ssl = verify_ssl
        self._system_prompt = system_prompt

        default_headers: dict[str, str] = {}
        if gateway_key:
            default_headers["Ocp-Apim-Subscription-Key"] = gateway_key
            default_headers["user"] = _get_username()
            default_headers["anthropic-version"] = "vertex-2023-10-16"

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        if default_headers:
            kwargs["default_headers"] = default_headers
        if not verify_ssl:
            # Must come from the SDK: it type-checks http_client against the
            # httpx distribution it was built against, which newer releases
            # ship as `httpx2`, so a plain `httpx.Client` is rejected.
            kwargs["http_client"] = anthropic.DefaultHttpxClient(verify=False)

        self.client = anthropic.Anthropic(**kwargs)

        self._messages: list[dict[str, Any]] = []

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
        anthropic_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._system_prompt,
            "messages": self._messages,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            # Extended thinking rejects tool_choice any/tool; leave it unset
            # (auto) so a first-round force_tool_use still runs.
            if force_tool_use and self.thinking_budget is None:
                kwargs["tool_choice"] = {"type": "any"}
        if self.thinking_budget is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIConnectionError as exc:
            raise EndpointError(
                connection_failure_report(
                    exc,
                    self.base_url or "https://api.anthropic.com",
                    verify_ssl=self._verify_ssl,
                    hints=[
                        "On the AMD gateway the Anthropic route is "
                        "/Anthropic (POST /Anthropic/v1/messages).",
                        "If the endpoint uses a private CA, retry with "
                        "--no-verify-ssl.",
                    ],
                )
            ) from exc
        except anthropic.AuthenticationError as exc:
            endpoint = self.base_url or "https://api.anthropic.com"
            raise EndpointError(
                f"Authentication failed at {endpoint} — {exc}\n"
                "  Check ANTHROPIC_API_KEY or LLM_GATEWAY_KEY."
            ) from exc

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        needs_tool = response.stop_reason == "tool_use" and len(tool_calls) > 0

        self._messages.append(
            {"role": "assistant", "content": response.content}
        )

        usage = getattr(response, "usage", None)
        return ChatResult(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            thinking="\n\n".join(thinking_parts) if thinking_parts else None,
            needs_tool_response=needs_tool,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in results
        ]
        self._messages.append({"role": "user", "content": content})

    def repair_dangling_tool_use(self) -> None:
        if not self._messages or self._messages[-1].get("role") != "assistant":
            return
        last = self._messages[-1]["content"]
        ids = [
            b.id for b in last if getattr(b, "type", None) == "tool_use"
        ]
        if ids:
            self._messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": i,
                        "content": "Tool call aborted.",
                        "is_error": True,
                    }
                    for i in ids
                ],
            })

    def reset(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt
        self._messages.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(
        mcp_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """MCP tool defs → Anthropic tool format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in mcp_tools
        ]


def _get_username() -> str:
    """Get the current username, safe for non-interactive environments."""
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")
