# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Event handler that forwards agent events to a browser.

``reinvent4_agent.agent.AgentEventHandler`` is a synchronous protocol: the
agent calls the handler from inside its own event loop and expects the call to
return at once.  So the handler never awaits.  It puts each event on an
``asyncio.Queue`` and the WebSocket task drains that queue and sends it.  The
queue is unbounded on purpose: dropping an event would silently lose a tool
result, which is exactly the "hard to follow what was done when" problem the
UI exists to avoid.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Callable
from typing import Any

from reinvent4_agent.web.artifacts import extract_panels

logger = logging.getLogger(__name__)

#: Tool results can be very large (a full REINVENT results file).  The chat
#: keeps a readable excerpt; the panels carry the structured data.
MAX_RESULT_CHARS = 8000


class WebSocketEventHandler:
    """Implements ``AgentEventHandler`` by queueing JSON-serialisable events.

    One handler belongs to one chat session.  ``drain()`` yields events in the
    order the agent produced them.  With a *sink*, events go to the sink
    instead of the queue; a saved session uses that to record them.
    """

    def __init__(
        self,
        agent_label: str = "agent",
        sink: Callable[[dict[str, Any]], None] | None = None,
        id_prefix: str = "call",
    ) -> None:
        self.agent_label = agent_label
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sink = sink
        self._id_prefix = id_prefix
        self._ids = itertools.count(1)
        #: Id of the open call per tool name.  The agent runs a round's calls
        #: one after another, so at most one call per name is open.
        self._pending: dict[str, str] = {}
        self._started: dict[str, float] = {}

    # -- AgentEventHandler ------------------------------------------------

    def on_thinking(self, text: str) -> None:
        self._emit({"type": "thinking", "content": text})

    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        call_id = f"{self._id_prefix}-{next(self._ids)}"
        self._pending[name] = call_id
        self._started[call_id] = time.monotonic()
        self._emit(
            {
                "type": "tool_call",
                "id": call_id,
                "name": name,
                "arguments": _safe_json(arguments),
            }
        )

    def on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        call_id = self._pending.pop(name, None)
        started = self._started.pop(call_id, None) if call_id else None
        panels = [] if is_error else extract_panels(name, result)
        excerpt = result if len(result) <= MAX_RESULT_CHARS else result[:MAX_RESULT_CHARS]
        self._emit(
            {
                "type": "tool_result",
                "id": call_id,
                "name": name,
                "result": excerpt,
                "truncated": len(result) > MAX_RESULT_CHARS,
                "isError": is_error,
                "panels": panels,
                "elapsedMs": round((time.monotonic() - started) * 1000) if started is not None else None,
            }
        )

    def on_error(self, message: str) -> None:
        self._emit({"type": "error", "message": message})

    def on_monitor_alert(self, alert: dict[str, Any]) -> None:
        self._emit({"type": "monitor_alert", "alert": _safe_json(alert)})

    def on_monitor_response(self, text: str) -> None:
        self._emit({"type": "monitor_response", "content": text})

    # -- used by the server ------------------------------------------------

    def emit(self, event: dict[str, Any]) -> None:
        """Queue an event the server itself raises (status, final answer)."""
        self._emit(event)

    async def drain(self) -> dict[str, Any]:
        """Wait for and return the next queued event."""
        return await self.queue.get()

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("agent", self.agent_label)
        if self._sink is not None:
            self._sink(event)
            return
        try:
            self.queue.put_nowait(event)
        except Exception:  # pragma: no cover - unbounded queue does not reject
            logger.exception("Failed to queue agent event %s", event.get("type"))


def _safe_json(value: Any) -> Any:
    """Make a value safe to serialise, replacing anything exotic with text."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return repr(value)
