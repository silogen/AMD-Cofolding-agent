# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent loop: user input -> LLM backend -> MCP tool calls -> display.

Implements the agentic loop where the LLM reasons about the user request,
decides which drug design tools to call, processes results, and continues
until a final text response is ready.  Backend-agnostic: works with both
Anthropic and OpenAI-compatible APIs via the LLMBackend abstraction.

Every interaction is logged at DEBUG level, and optionally written to a
structured JSONL transcript file for post-hoc analysis.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reinvent4_agent.backends import (
    ChatResult,
    LLMBackend,
    ToolResult,
    TransientEndpointError,
)
from reinvent4_agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 20

#: Attempts per LLM call while the endpoint is merely unreachable.  A locally
#: served model that is restarting is back within seconds, and without this
#: the whole turn — tool results and all — is lost with it.
MAX_LLM_ATTEMPTS = 3

#: First wait between attempts, doubled each time, plus a little jitter so
#: several agents do not come back in lockstep.
LLM_RETRY_DELAY = 5.0

#: Ceiling on the waiting, so a genuinely dead endpoint is reported promptly
#: instead of leaving the user watching a spinner.
LLM_RETRY_BUDGET = 30.0

#: A turn ends once the same tool fails this often with the same arguments:
#: the model is looping, and each retry costs a full LLM round.
MAX_IDENTICAL_FAILURES = 3

#: Longest tool result the model sees.  Some tools return tens of thousands
#: of characters (every scoring component, a whole results file), which fills
#: the context and slows every later round.  The event handler still gets the
#: full text.
MAX_TOOL_RESULT_CHARS = int(os.environ.get("REINVENT4_AGENT_MAX_TOOL_RESULT_CHARS") or 12000)


@runtime_checkable
class AgentEventHandler(Protocol):
    """Protocol for receiving agent lifecycle events.

    Implement to customise how the agent displays thinking, tool calls,
    tool results, and errors (e.g. Rich terminal, WebSocket, silent).
    """

    def on_thinking(self, text: str) -> None: ...
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: ...
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: ...
    def on_error(self, message: str) -> None: ...
    def on_notice(self, message: str) -> None: ...
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: ...
    def on_monitor_response(self, text: str) -> None: ...


class _NullEventHandler:
    """No-op event handler used when no handler is provided."""

    def on_thinking(self, text: str) -> None: pass
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: pass
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: pass
    def on_error(self, message: str) -> None: pass
    def on_notice(self, message: str) -> None: pass
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: pass
    def on_monitor_response(self, text: str) -> None: pass


class Agent:
    """Conversational agent that connects an LLM to drug design tools via MCP.

    Maintains conversation history (inside the backend) across turns and
    handles the tool-use loop: the LLM proposes tool calls, the agent
    executes them against the MCP server, feeds results back, and repeats
    until the LLM produces a final text response.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: LLMBackend,
        *,
        max_tokens: int = 4096,
        log_dir: str | None = None,
        event_handler: AgentEventHandler | None = None,
        name: str = "agent",
        system_prompt: str | None = None,
        allowed_tools: set[str] | None = None,
    ) -> None:
        self.mcp = mcp_client
        self.backend = backend
        self.max_tokens = max_tokens
        self.name = name
        self._system_prompt = system_prompt
        self._allowed_tools = allowed_tools
        self._transcript_path = _init_transcript(log_dir, name)
        self._turn = 0
        self._events: AgentEventHandler = event_handler or _NullEventHandler()
        self._chat_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_running = False
        self._seen_alerts: set[tuple[str, str]] = set()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        #: The LLM call in flight.  A cancelled turn cannot stop the thread it
        #: runs in, and the backend appends the reply to the history when it
        #: arrives, so the next turn waits for it to keep the history in order.
        self._inflight: asyncio.Future[ChatResult] | None = None
        self._failures: dict[tuple[str, str], int] = {}

    @property
    def total_tokens(self) -> int:
        """Cumulative input + output tokens consumed by this agent."""
        return self.total_input_tokens + self.total_output_tokens

    async def _scoped_tools(self) -> list[dict[str, Any]]:
        """Return the MCP tools this agent is permitted to use.

        When ``allowed_tools`` is set (e.g. the read-only Reviewer), the
        shared client's tool list is filtered down to that subset so the
        agent physically cannot call mutating tools.
        """
        tools = await self.mcp.list_tools()
        if self._allowed_tools is None:
            return tools
        return [t for t in tools if t["name"] in self._allowed_tools]

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's text response.

        This may involve multiple rounds of tool calls before the LLM
        produces its final answer.
        """
        if self._inflight is not None and not self._inflight.done():
            logger.info("Waiting for the reply of the stopped turn before starting the next one")
            with contextlib.suppress(Exception):
                await self._inflight
        # A stopped turn can leave tool calls without results at the end of
        # the history; they must be answered before the next user message.
        self.backend.repair_dangling_tool_use()

        self._turn += 1
        self._failures.clear()
        logger.info("Turn %d — user: %s", self._turn, user_message[:200])
        self._write_event("user_message", {"content": user_message})

        self.backend.add_user_message(user_message)
        tools = await self._scoped_tools()

        for round_idx in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            result = await self._complete(tools)
            elapsed = time.monotonic() - t0

            self.total_input_tokens += result.input_tokens
            self.total_output_tokens += result.output_tokens

            logger.info(
                "Turn %d round %d — LLM responded in %.2fs "
                "(text=%d chars, tool_calls=%d, thinking=%s, "
                "out_tokens=%d of which reasoning=%d)",
                self._turn,
                round_idx,
                elapsed,
                len(result.text or ""),
                len(result.tool_calls),
                "yes" if result.thinking else "no",
                result.output_tokens,
                result.reasoning_tokens,
            )

            if result.thinking:
                logger.debug("Thinking: %s", result.thinking[:500])
                self._write_event("thinking", {
                    "content": result.thinking,
                    "round": round_idx,
                })
                self._events.on_thinking(result.thinking)

            if not result.needs_tool_response:
                logger.info(
                    "Turn %d — final response: %s",
                    self._turn,
                    (result.text or "")[:200],
                )
                self._write_event("assistant_message", {
                    "content": result.text,
                    "round": round_idx,
                    "elapsed_s": round(elapsed, 3),
                    "total_input_tokens": self.total_input_tokens,
                    "total_output_tokens": self.total_output_tokens,
                })
                return result.text or ""

            tool_results: list[ToolResult] = []
            try:
                tool_results = await self._handle_tool_calls(result, round_idx)
            finally:
                # Even if a tool raised/cancelled, every tool_use must get a
                # matching tool_result, otherwise the conversation becomes
                # permanently invalid and the API rejects every later request.
                seen = {r.tool_call_id for r in tool_results}
                for tc in result.tool_calls:
                    if tc.id not in seen:
                        tool_results.append(ToolResult(
                            tool_call_id=tc.id,
                            content="Tool call aborted.",
                            is_error=True,
                        ))
                self.backend.add_tool_results(tool_results)

            looping = [key for key, count in self._failures.items() if count >= MAX_IDENTICAL_FAILURES]
            if looping:
                name = looping[0][0]
                logger.warning("Turn %d — %s failed %d times with the same arguments", self._turn, name, MAX_IDENTICAL_FAILURES)
                self._write_event("repeated_failure", {"tool": name, "count": MAX_IDENTICAL_FAILURES})
                return (
                    f"I stopped because `{name}` failed {MAX_IDENTICAL_FAILURES} times with the same "
                    "arguments. The last error is in the tool results above. Tell me how to adjust "
                    "the request, or check the service it depends on."
                )

        msg = (
            "I've reached the maximum number of tool-call rounds. "
            "Please try breaking your request into smaller steps."
        )
        logger.warning("Turn %d — max tool rounds reached", self._turn)
        self._write_event("max_rounds", {"turn": self._turn})
        return msg

    async def _complete(self, tools: list[dict[str, Any]]) -> ChatResult:
        """One LLM call, retried while the endpoint is only unreachable.

        The call runs off the event loop: a reasoning model can take minutes,
        and the web gateway must keep answering WebSocket pings meanwhile.
        Only a transient failure is retried — a rejected request or a tool
        error would fail identically — and the waiting is interruptible, so
        stopping the turn still stops it at once.
        """
        budget = LLM_RETRY_BUDGET
        attempt = 0
        while True:
            attempt += 1
            self._inflight = asyncio.ensure_future(
                asyncio.to_thread(self.backend.complete, tools, self.max_tokens)
            )
            try:
                return await asyncio.shield(self._inflight)
            except (TransientEndpointError, TimeoutError) as exc:
                delay = min(
                    LLM_RETRY_DELAY * 2 ** (attempt - 1) * (1 + random.uniform(0, 0.25)),
                    budget,
                )
                if attempt >= MAX_LLM_ATTEMPTS or delay <= 0:
                    logger.error(
                        "Turn %d — the endpoint stayed unreachable over %d attempt(s)",
                        self._turn,
                        attempt,
                    )
                    raise TransientEndpointError(
                        f"The model endpoint did not answer after {attempt} "
                        "attempts; it may be restarting, so try again in a "
                        f"moment.\n{exc}"
                    ) from exc
                budget -= delay
                logger.warning(
                    "Turn %d — the endpoint did not answer (attempt %d of %d); "
                    "retrying in %.1fs: %s",
                    self._turn,
                    attempt,
                    MAX_LLM_ATTEMPTS,
                    delay,
                    str(exc).partition("\n")[0],
                )
                self._write_event("llm_retry", {"attempt": attempt, "delay_s": round(delay, 2)})
                self._events.on_notice(
                    "The model endpoint is not answering, retrying "
                    f"({attempt + 1} of {MAX_LLM_ATTEMPTS})..."
                )
                await asyncio.sleep(delay)

    async def _handle_tool_calls(
        self, result: ChatResult, round_idx: int
    ) -> list[ToolResult]:
        """Execute all tool calls and return results for the backend."""
        results: list[ToolResult] = []

        for tc in result.tool_calls:
            self._events.on_tool_call(tc.name, tc.arguments)
            logger.info(
                "Turn %d round %d — calling tool %s(%s)",
                self._turn,
                round_idx,
                tc.name,
                _summarize_dict(tc.arguments),
            )
            self._write_event("tool_call", {
                "name": tc.name,
                "arguments": tc.arguments,
                "round": round_idx,
            })

            t0 = time.monotonic()
            try:
                result_text = await self.mcp.call_tool(tc.name, tc.arguments)
                elapsed = time.monotonic() - t0

                logger.info(
                    "Turn %d round %d — tool %s returned %d chars in %.2fs",
                    self._turn,
                    round_idx,
                    tc.name,
                    len(result_text),
                    elapsed,
                )
                self._write_event("tool_result", {
                    "name": tc.name,
                    "content_length": len(result_text),
                    "elapsed_s": round(elapsed, 3),
                    "is_error": False,
                    "round": round_idx,
                })

                self._events.on_tool_result(tc.name, result_text, is_error=False)

                results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=_cap_for_model(result_text),
                ))
            except Exception as exc:
                elapsed = time.monotonic() - t0
                error_msg = f"Tool call failed: {exc}"
                logger.error(
                    "Turn %d round %d — tool %s failed after %.2fs: %s",
                    self._turn,
                    round_idx,
                    tc.name,
                    elapsed,
                    exc,
                    exc_info=True,
                )
                self._write_event("tool_result", {
                    "name": tc.name,
                    "error": str(exc),
                    "elapsed_s": round(elapsed, 3),
                    "is_error": True,
                    "round": round_idx,
                })
                self._events.on_tool_result(tc.name, error_msg, is_error=True)
                key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
                self._failures[key] = self._failures.get(key, 0) + 1
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    content=error_msg,
                    is_error=True,
                ))

        return results

    def reset_history(self) -> None:
        """Clear conversation history to start a fresh session."""
        logger.info("Conversation history reset")
        self._write_event("reset", {})
        self._turn = 0
        if self._system_prompt is not None:
            prompt = self._system_prompt
        else:
            from reinvent4_agent.system_prompt import SYSTEM_PROMPT

            prompt = SYSTEM_PROMPT
        self.backend.reset(prompt)

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def start_monitoring(self, poll_interval: float = 120.0) -> None:
        """Start background job monitoring."""
        if self._monitor_task is not None:
            return
        self._monitor_running = True
        self._seen_alerts.clear()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(poll_interval)
        )
        logger.info("Monitor started (poll every %.0fs)", poll_interval)

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitor_running = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Monitor stopped")

    async def _monitor_loop(self, poll_interval: float) -> None:
        """Periodically call get_job_alerts and act on results."""
        while self._monitor_running:
            try:
                await asyncio.sleep(poll_interval)
                if not self._monitor_running:
                    break
                await self._check_alerts()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Monitor loop error", exc_info=True)
                await asyncio.sleep(30)

    async def _check_alerts(self) -> None:
        """Fetch alerts and dispatch actions based on severity."""
        async with self._chat_lock:
            alerts_json = await self.mcp.call_tool("get_job_alerts", {})

        try:
            alerts = json.loads(alerts_json)
        except (json.JSONDecodeError, TypeError):
            return

        if not alerts:
            return

        for alert in alerts:
            job_id = alert.get("job_id", "")
            alert_type = alert.get("type", "")
            key = (job_id, alert_type)
            if key in self._seen_alerts:
                continue
            self._seen_alerts.add(key)

            severity = alert.get("severity", "low")
            logger.info(
                "Monitor alert [%s] job=%s type=%s: %s",
                severity, job_id, alert_type,
                alert.get("message", "")[:200],
            )

            self._events.on_monitor_alert(alert)

            prompt = (
                f"[MONITOR] Alert ({severity}): "
                f"{alert.get('message', '')} "
                f"[job_id={job_id}, type={alert_type}]"
            )
            try:
                async with self._chat_lock:
                    response = await self.chat(prompt)
                self._events.on_monitor_response(response)
                self._log_monitor_action(alert, response)
            except Exception:
                logger.error(
                    "Monitor autonomous action failed",
                    exc_info=True,
                )

    def _log_monitor_action(
        self, alert: dict[str, Any], response: str | None,
    ) -> None:
        """Write a monitor action to the job's monitor_actions.jsonl."""
        self._write_event("monitor_alert", {
            "alert": alert,
            "response": response[:500] if response else None,
        })
        working_dir = (alert.get("details") or {}).get("working_dir")
        if not working_dir:
            return
        record = {
            "ts": datetime.now().isoformat(),
            "alert_type": alert.get("type"),
            "severity": alert.get("severity"),
            "message": alert.get("message"),
            "action": (
                "Autonomous LLM response" if response
                else "Notified user (medium severity)"
            ),
            "agent_response": response[:1000] if response else None,
        }
        try:
            p = Path(working_dir) / "monitor_actions.jsonl"
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.debug("Failed to write monitor action log", exc_info=True)

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _write_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Append a structured event to the JSONL transcript file."""
        if self._transcript_path is None:
            return
        record = {
            "ts": datetime.now().isoformat(),
            "turn": self._turn,
            "event": event_type,
            **data,
        }
        try:
            with open(self._transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.debug("Failed to write transcript event", exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_transcript(log_dir: str | None, name: str = "agent") -> str | None:
    """Create the transcript JSONL file path if log_dir is set.

    ``name`` is included in the filename so that the Driver and Reviewer
    agents in a co-scientist run write to separate transcripts instead of
    interleaving into one file.
    """
    if not log_dir:
        return None
    d = Path(log_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(d / f"session_{ts}_{name}.jsonl")
    logger.info("Transcript file: %s", path)
    return path


def _cap_for_model(text: str) -> str:
    """Shorten a tool result to MAX_TOOL_RESULT_CHARS, saying so in the text."""
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return (
        text[:MAX_TOOL_RESULT_CHARS]
        + f"\n\n[Result cut at {MAX_TOOL_RESULT_CHARS} of {len(text)} characters. "
        "If you need the rest, ask the tool for less, for example fewer rows or one item.]"
    )


def _summarize_dict(d: dict[str, Any], max_len: int = 120) -> str:
    """Short repr of a dict for log messages."""
    s = json.dumps(d, default=str)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


