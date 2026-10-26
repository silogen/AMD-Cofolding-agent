# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conversations that outlive the browser tab.

A session is one conversation with the agent.  It belongs to the gateway, not
to a WebSocket: a turn keeps running when the tab closes, and a tab that comes
back (or a new one) attaches to the session and receives every event it
missed.  Each session is a directory::

    <root>/<session id>/
        meta.json      title, timestamps, turn and token counts
        events.jsonl   every event sent to the browser, in order, numbered
        history.json   the LLM conversation, written after each turn

The events are what the UI replays; the history is what the agent needs to
carry on.  They are kept apart because they differ: tool results in the
history are the full text the model saw, while events carry an excerpt and
the panels derived from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reinvent4_agent.web.artifacts import extract_panels
from reinvent4_agent.web.events import WebSocketEventHandler

logger = logging.getLogger(__name__)

#: Session ids are generated here and also arrive from the browser, where they
#: end up in a file path, so they are held to this shape.
_ID_PATTERN = re.compile(r"^s-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

#: Characters of the first user message used as the default title.
_TITLE_CHARS = 80

#: Idle sessions kept in memory.  Older ones are reloaded from disk on demand.
MAX_LIVE_SESSIONS = 20


def valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and _ID_PATTERN.match(session_id) is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"s-{stamp}-{secrets.token_hex(3)}"


def refresh_panels(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive the panels of recorded tool results that have none.

    Panels are derived when a result arrives, so a conversation recorded
    before the UI could show a kind of result (ADMET predictions, say) would
    never show it.  Results with panels are left as recorded.
    """
    arguments: dict[str, Any] = {}
    for event in events:
        kind = event.get("type")
        if kind == "tool_call":
            arguments[event.get("id")] = event.get("arguments")
        elif kind == "tool_result" and not event.get("panels") and not event.get("isError"):
            result = event.get("result")
            if isinstance(result, str):
                called = arguments.get(event.get("id"))
                event["panels"] = extract_panels(event.get("name", ""), result, called if isinstance(called, dict) else None)
    return events


class SessionStore:
    """Reads and writes session directories below *root*."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, session_id: str) -> Path:
        if not valid_session_id(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        return self.root / session_id

    def exists(self, session_id: str) -> bool:
        return valid_session_id(session_id) and (self._dir(session_id) / "meta.json").is_file()

    def create(self, meta: dict[str, Any]) -> None:
        directory = self._dir(meta["id"])
        directory.mkdir(parents=True, exist_ok=False)
        self.save_meta(meta)

    def list_meta(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        metas = []
        for directory in self.root.iterdir():
            if not valid_session_id(directory.name):
                continue
            meta = self._read_json(directory / "meta.json")
            if isinstance(meta, dict):
                metas.append(meta)
        return sorted(metas, key=lambda m: m.get("updated", ""), reverse=True)

    def load_meta(self, session_id: str) -> dict[str, Any] | None:
        meta = self._read_json(self._dir(session_id) / "meta.json")
        return meta if isinstance(meta, dict) else None

    def save_meta(self, meta: dict[str, Any]) -> None:
        self._write_json(self._dir(meta["id"]) / "meta.json", meta)

    def load_history(self, session_id: str) -> list[dict[str, Any]]:
        history = self._read_json(self._dir(session_id) / "history.json")
        return history if isinstance(history, list) else []

    def save_history(self, session_id: str, history: list[dict[str, Any]]) -> None:
        self._write_json(self._dir(session_id) / "history.json", history)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        path = self._dir(session_id) / "events.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.exception("Could not record an event of session %s", session_id)

    def load_events(self, session_id: str) -> list[dict[str, Any]]:
        path = self._dir(session_id) / "events.jsonl"
        events: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    # A line cut short by a crash is skipped, not fatal.
                    with contextlib.suppress(ValueError):
                        events.append(json.loads(line))
        except FileNotFoundError:
            pass
        return refresh_panels(events)

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self._dir(session_id), ignore_errors=True)

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        # Write then rename, so a crash never leaves half a file behind.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
        tmp.replace(path)


@dataclass
class LiveSession:
    """A session loaded in memory: its agent, its event log, its listeners."""

    meta: dict[str, Any]
    store: SessionStore
    events: list[dict[str, Any]] = field(default_factory=list)
    agent: Any = None
    handler: WebSocketEventHandler | None = None
    running: asyncio.Task[None] | None = None
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def busy(self) -> bool:
        return self.running is not None and not self.running.done()

    @property
    def last_seq(self) -> int:
        return self.events[-1]["seq"] if self.events else 0

    def publish(self, event: dict[str, Any]) -> None:
        """Number, record and fan out one event."""
        event = {**event, "seq": self.last_seq + 1, "ts": _now()}
        self.events.append(event)
        self.store.append_event(self.id, event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)

    def events_after(self, seq: int) -> list[dict[str, Any]]:
        return [event for event in self.events if event["seq"] > seq]

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    def start_turn(self, content: str, effort: str | None = None) -> bool:
        """Start a turn.  False when one is already running.

        *effort* sets the reasoning effort of this turn and the ones after
        it, on backends that have one.
        """
        if self.busy:
            return False
        backend = getattr(self.agent, "backend", None)
        if effort and hasattr(backend, "reasoning_effort"):
            backend.reasoning_effort = effort
        if not self.meta.get("title"):
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            self.meta["title"] = first_line[:_TITLE_CHARS]
        self.publish({"type": "user_message", "content": content, **({"effort": effort} if effort else {})})
        self.publish({"type": "status", "state": "busy"})
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        self.meta["updated"] = _now()
        self.store.save_meta(self.meta)
        self.running = asyncio.create_task(self._run_turn(content))
        return True

    def cancel(self) -> bool:
        """Stop the running turn.  False when there is none."""
        if not self.busy:
            return False
        assert self.running is not None
        self.running.cancel()
        return True

    async def _run_turn(self, content: str) -> None:
        try:
            answer = await self.agent.chat(content)
            self.publish(
                {
                    "type": "assistant_message",
                    "content": answer,
                    "inputTokens": getattr(self.agent, "total_input_tokens", 0),
                    "outputTokens": getattr(self.agent, "total_output_tokens", 0),
                }
            )
        except asyncio.CancelledError:
            self.publish({"type": "cancelled"})
        except Exception as exc:
            logger.exception("Agent turn failed in session %s", self.id)
            self.publish({"type": "error", "message": str(exc)})
        finally:
            self.publish({"type": "status", "state": "idle"})
            self.save()

    def save(self) -> None:
        """Write the metadata and the LLM history."""
        self.meta["updated"] = _now()
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        if self.agent is not None:
            self.meta["inputTokens"] = getattr(self.agent, "total_input_tokens", 0)
            self.meta["outputTokens"] = getattr(self.agent, "total_output_tokens", 0)
            try:
                self.store.save_history(self.id, self.agent.backend.export_history())
            except Exception:
                logger.exception("Could not save the history of session %s", self.id)
        self.store.save_meta(self.meta)

    def summary(self) -> dict[str, Any]:
        return {**self.meta, "busy": self.busy}


#: Builds an agent whose events go to the given sink.
AgentFactory = Callable[[WebSocketEventHandler], Any]


class SessionManager:
    """Creates, loads and forgets live sessions."""

    def __init__(self, store: SessionStore, agent_factory: AgentFactory) -> None:
        self.store = store
        self._agent_factory = agent_factory
        self._live: dict[str, LiveSession] = {}

    def _attach_agent(self, live: LiveSession) -> None:
        # A fresh id prefix per load keeps call ids unique across restores.
        handler = WebSocketEventHandler(sink=live.publish, id_prefix=f"call-{secrets.token_hex(3)}")
        live.handler = handler
        live.agent = self._agent_factory(handler)

    def create(self) -> LiveSession:
        stamp = _now()
        meta = {"id": _new_id(), "title": "", "created": stamp, "updated": stamp, "turns": 0}
        self.store.create(meta)
        live = LiveSession(meta=meta, store=self.store)
        self._attach_agent(live)
        self._remember(live)
        return live

    def get(self, session_id: str) -> LiveSession | None:
        """The live session, loading it from disk if needed."""
        if session_id in self._live:
            return self._live[session_id]
        if not self.store.exists(session_id):
            return None
        meta = self.store.load_meta(session_id)
        if meta is None:
            return None
        live = LiveSession(meta=meta, store=self.store, events=self.store.load_events(session_id))
        self._attach_agent(live)
        history = self.store.load_history(session_id)
        if history:
            live.agent.backend.import_history(history)
        live.agent.total_input_tokens = meta.get("inputTokens", 0)
        live.agent.total_output_tokens = meta.get("outputTokens", 0)
        # A turn that was running when the gateway stopped never finished.
        statuses = [event for event in live.events if event.get("type") == "status"]
        if statuses and statuses[-1].get("state") == "busy":
            live.publish({"type": "error", "message": "The gateway restarted while this turn was running."})
            live.publish({"type": "status", "state": "idle"})
        self._remember(live)
        return live

    def list(self) -> list[dict[str, Any]]:
        """Sessions with at least one message, newest first.

        Every new browser tab gets a session before anyone types, so the
        empty ones are left out rather than filling the list.
        """
        summaries = {meta["id"]: {**meta, "busy": False} for meta in self.store.list_meta()}
        for live in self._live.values():
            summaries[live.id] = live.summary()
        shown = [s for s in summaries.values() if s.get("turns") or s.get("busy")]
        return sorted(shown, key=lambda m: m.get("updated", ""), reverse=True)

    def rename(self, session_id: str, title: str) -> dict[str, Any] | None:
        live = self.get(session_id)
        if live is None:
            return None
        live.meta["title"] = title.strip()[:200]
        live.store.save_meta(live.meta)
        return live.summary()

    def delete(self, session_id: str) -> bool:
        """Delete a session.  Refuses one with a turn running."""
        live = self._live.get(session_id)
        if live is not None:
            if live.busy:
                return False
            del self._live[session_id]
        self.store.delete(session_id)
        return True

    async def close(self) -> None:
        """Cancel running turns; used at shutdown."""
        for live in list(self._live.values()):
            if live.busy and live.running is not None:
                live.running.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await live.running

    def _remember(self, live: LiveSession) -> None:
        self._live[live.id] = live
        idle = [s for s in self._live.values() if not s.busy and not s.subscribers and s is not live]
        while len(self._live) > MAX_LIVE_SESSIONS and idle:
            del self._live[idle.pop(0).id]
