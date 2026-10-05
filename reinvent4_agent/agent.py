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
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reinvent4_agent.backends import ChatResult, LLMBackend, ToolResult
from reinvent4_agent.mcp_client import MCPClient

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 20


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
    def on_monitor_alert(self, alert: dict[str, Any]) -> None: ...
    def on_monitor_response(self, text: str) -> None: ...


class _NullEventHandler:
    """No-op event handler used when no handler is provided."""

    def on_thinking(self, text: str) -> None: pass
    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: pass
    def on_tool_result(self, name: str, result: str, is_error: bool) -> None: pass
    def on_error(self, message: str) -> None: pass
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

    def _tool_allowed(self, name: str) -> bool:
        """Whether this agent may execute the canonical tool ``name``."""
        return self._allowed_tools is None or name in self._allowed_tools

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's text response.

        This may involve multiple rounds of tool calls before the LLM
        produces its final answer.
        """
        self._turn += 1
        logger.info("Turn %d — user: %s", self._turn, user_message[:200])
        self._write_event("user_message", {"content": user_message})

        self.backend.add_user_message(user_message)
        tools = await self._scoped_tools()

        for round_idx in range(MAX_TOOL_ROUNDS):
            t0 = time.monotonic()
            # Leave tool_choice on auto. Forcing a call on round 0 of every
            # user turn sends greetings and clarifying questions through an
            # arbitrary MCP tool, including mutating ones.
            result = self.backend.complete(tools, self.max_tokens)
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

        msg = (
            "I've reached the maximum number of tool-call rounds. "
            "Please try breaking your request into smaller steps."
        )
        logger.warning("Turn %d — max tool rounds reached", self._turn)
        self._write_event("max_rounds", {"turn": self._turn})
        return msg

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
                resolved = self.mcp.resolve_tool_name(tc.name)
                # Resolve against the full catalog, then refuse anything
                # outside this agent's allow-list. An alias can name a
                # mutating tool the model was never offered.
                if resolved is not None and not self._tool_allowed(resolved):
                    denied = resolved
                    resolved = None
                else:
                    denied = None
                if resolved is None:
                    known = ", ".join(
                        t["name"] for t in await self._scoped_tools()
                    )
                    if denied is not None:
                        result_text = (
                            f"Tool '{denied}' is not available to this "
                            f"agent. Available tools: {known}."
                        )
                    else:
                        result_text = (
                            f"Unknown tool '{tc.name}'. Available tools: "
                            f"{known}. Scoring components are listed by "
                            "list_scoring_components."
                        )
                    elapsed = time.monotonic() - t0
                    logger.warning(
                        "Turn %d round %d — unknown tool %s (%.2fs)",
                        self._turn,
                        round_idx,
                        tc.name,
                        elapsed,
                    )
                    self._write_event("tool_result", {
                        "name": tc.name,
                        "content_length": len(result_text),
                        "elapsed_s": round(elapsed, 3),
                        "is_error": True,
                        "round": round_idx,
                    })
                    self._events.on_tool_result(
                        tc.name, result_text, is_error=True
                    )
                    results.append(ToolResult(
                        tool_call_id=tc.id,
                        content=result_text,
                        is_error=True,
                    ))
                    continue
                if resolved != tc.name:
                    logger.info(
                        "Turn %d round %d — aliasing tool %s -> %s",
                        self._turn,
                        round_idx,
                        tc.name,
                        resolved,
                    )
                result_text = await self.mcp.call_tool(resolved, tc.arguments)
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
                    content=result_text,
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


def _summarize_dict(d: dict[str, Any], max_len: int = 120) -> str:
    """Short repr of a dict for log messages."""
    s = json.dumps(d, default=str)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


