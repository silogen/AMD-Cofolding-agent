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
from datetime import datetime, timedelta, timezone
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

#: "## Score: c1ccccc1" at the top of an answer, the agent's own name for a turn.
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def first_heading(markdown: str) -> str | None:
    """The first markdown heading of an answer, without its emphasis marks."""
    match = _HEADING_PATTERN.search(markdown or "")
    if match is None:
        return None
    text = re.sub(r"[*_`]", "", match.group(1)).strip()
    return text or None

#: Idle sessions kept in memory.  Older ones are reloaded from disk on demand.
MAX_LIVE_SESSIONS = 20

#: Longest autonomy budget a browser may ask for, in minutes.
MAX_AUTONOMY_MINUTES = 480

#: Automatic turns in a row before the loop gives up, whatever the budget
#: says.  A campaign that needs more than this needs the user.
MAX_AUTO_TURNS = 50

#: Pause between automatic turns.  A REINVENT campaign runs for hours, so a
#: continuation that finds nothing new must wait rather than spin.
AUTO_TURN_DELAY_S = 60.0

#: What a continuation turn asks of the agent.  It is not a user message: it
#: is never recorded as one, and the turn it opens is marked automatic.
AUTO_INSTRUCTION = (
    "[AUTO] You are continuing on your own; nobody is waiting for an answer. "
    "Check the jobs you started (get_job_alerts, get_optimization_status), read "
    "and analyse whatever results are ready (get_top_candidates, read_results), "
    "then decide the next step towards the goal and take it. Keep it short. "
    "If a run is still going and nothing has changed, say so in one line and "
    "leave it at that -- you will be called again. End with AUTONOMY DONE only "
    "when the campaign itself is over: the goal is reached, or the next move is "
    "the user's. A job still running is never done."
)

#: How an answer says the autonomy loop has nothing left to do.
_DONE_PATTERN = re.compile(r"\bAUTONOMY\s+DONE\b", re.IGNORECASE)

#: The same phrase with the markdown and punctuation a model wraps it in: it
#: is addressed to the gateway, so it is cut out of what the browser shows.
_DONE_MARKUP = re.compile(r"\s*[*_#]*\bAUTONOMY\s+DONE\b[*_#.!]*", re.IGNORECASE)

#: Events that speak to whoever is watching now rather than to the
#: conversation, so they are neither numbered nor recorded.
_LIVE_ONLY = frozenset({"notice", "auto"})


def valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and _ID_PATTERN.match(session_id) is not None


def spoken_answer(answer: str | None) -> str:
    """The answer as the conversation keeps it, without the stop phrase."""
    return _DONE_MARKUP.sub("", answer or "").strip()


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


def backfill_title(meta: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    """Rename a still-automatic title after the first answer's heading.

    Sessions recorded before titles came from the answer are named after the
    first line of the question, which is often a raw SMILES and tells the
    user nothing.  Returns whether the title changed, so the caller can save
    the metadata; clearing ``auto_title`` keeps it a one-off, and a title the
    user chose is never touched.
    """
    if "auto_title" in meta:
        if not meta["auto_title"]:
            return False
    elif meta.get("title", "") != _first_user_line(events):
        # Recorded before auto_title existed: only a title that still matches
        # the question verbatim can be assumed to be automatic.
        return False

    first_answer = next(
        (e for e in events if e.get("type") == "assistant_message"),
        None,
    )
    heading = first_heading(str((first_answer or {}).get("content") or ""))
    if not heading:
        return False
    meta["title"] = heading[:_TITLE_CHARS]
    meta["auto_title"] = False
    return True


def _first_user_line(events: list[dict[str, Any]]) -> str:
    """The title ``start_turn`` would have derived from the first question."""
    first = next((e for e in events if e.get("type") == "user_message"), None)
    content = str((first or {}).get("content") or "").strip()
    return content.splitlines()[0][:_TITLE_CHARS] if content else ""


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
    #: End of the autonomy budget; None when the agent only answers requests.
    auto_until: datetime | None = None
    #: The continuation turn waiting out its pause, if one is queued.
    auto_task: asyncio.Task[None] | None = None
    #: Automatic turns taken since the last user message.
    auto_turns: int = 0
    #: Pause before a continuation; read per session so tests can shorten it.
    auto_delay: float = field(default_factory=lambda: AUTO_TURN_DELAY_S)

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
        """Number, record and fan out one event.

        A notice, and the state of the autonomy budget, speak to whoever is
        watching now rather than to the conversation, so they are neither
        numbered nor recorded: replayed on a reconnect they would announce a
        retry that finished long ago, or a budget that has run out.  A
        browser that comes back reads the budget from ``summary()``.
        """
        if event.get("type") in _LIVE_ONLY:
            for queue in list(self.subscribers):
                queue.put_nowait(event)
            return
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
        # The user leads: whatever the agent was about to do on its own is dropped.
        self.cancel_continuation()
        self.auto_turns = 0
        backend = getattr(self.agent, "backend", None)
        if effort and hasattr(backend, "reasoning_effort"):
            backend.reasoning_effort = effort
        if not self.meta.get("title"):
            first_line = content.strip().splitlines()[0] if content.strip() else ""
            self.meta["title"] = first_line[:_TITLE_CHARS]
            self.meta["auto_title"] = True
        self.publish({"type": "user_message", "content": content, **({"effort": effort} if effort else {})})
        self._begin(content)
        return True

    def cancel(self) -> bool:
        """Stop the running turn and any continuation.  False when none runs.

        Stop means stop: the budget goes with the turn, or the next
        continuation would start the agent up again a minute later.
        """
        self.end_autonomy()
        if not self.busy:
            return False
        assert self.running is not None
        self.running.cancel()
        return True

    # -- working on its own ------------------------------------------------

    @property
    def auto_on(self) -> bool:
        """Whether the autonomy budget still has time left."""
        return self.auto_until is not None and self.auto_until > datetime.now(timezone.utc)

    def set_autonomy(self, minutes: float) -> datetime | None:
        """Start or extend the autonomy budget; 0 minutes stops it.

        The deadline is absolute, so a browser that reconnects reads the time
        left from ``summary()`` instead of counting from its own clock.
        """
        if minutes > 0:
            self.auto_until = datetime.now(timezone.utc) + timedelta(minutes=min(float(minutes), float(MAX_AUTONOMY_MINUTES)))
            self.auto_turns = 0
            # Turned on between two turns: pick the campaign up at once rather
            # than waiting for a message the user does not intend to send.
            if not self.busy and self.meta.get("turns"):
                self._schedule_continuation(0.0)
        else:
            self.auto_until = None
            self.cancel_continuation()
        self.publish({"type": "auto", "until": self._auto_iso()})
        return self.auto_until

    def cancel_continuation(self) -> None:
        """Drop a continuation that has not started yet."""
        if self.auto_task is not None:
            self.auto_task.cancel()
            self.auto_task = None

    def end_autonomy(self) -> None:
        """Give the budget back once the loop stops taking turns.

        The browser counts the budget down from the deadline, so a loop that
        has stopped -- the agent said it was done, the turn failed, the user
        hit stop, the turn cap ran out -- must say so, or the chip keeps
        promising work that nobody is doing.
        """
        running = self.auto_until is not None
        self.auto_until = None
        self.cancel_continuation()
        if running:
            self.publish({"type": "auto", "until": None})

    def _auto_iso(self) -> str | None:
        return self.auto_until.isoformat(timespec="seconds") if self.auto_on else None

    def _schedule_continuation(self, delay: float) -> None:
        """Queue the next automatic turn.  The task checks again before it starts."""
        if not self.auto_on:
            return
        if self.auto_turns >= MAX_AUTO_TURNS:
            self.end_autonomy()
            return
        self.cancel_continuation()
        self.auto_task = asyncio.create_task(self._continue_after(delay))

    async def _continue_after(self, delay: float) -> None:
        """Wait out *delay*, then take another turn unless anything changed."""
        if delay > 0:
            await asyncio.sleep(delay)
        self.auto_task = None
        if self.busy or not self.auto_on:
            return
        if self.auto_turns >= MAX_AUTO_TURNS:
            self.end_autonomy()
            return
        self.auto_turns += 1
        self._begin(AUTO_INSTRUCTION, auto=True)

    def _begin(self, content: str, auto: bool = False) -> None:
        """Record the start of a turn and run it."""
        self.publish({"type": "status", "state": "busy", **({"auto": True} if auto else {})})
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        self.meta["updated"] = _now()
        self.store.save_meta(self.meta)
        self.running = asyncio.create_task(self._run_turn(content))

    async def _run_turn(self, content: str) -> None:
        unfinished = False
        try:
            answer = await self.agent.chat(content)
            spoken = spoken_answer(answer)
            self._title_from_answer(spoken)
            self.publish(
                {
                    "type": "assistant_message",
                    "content": spoken,
                    "inputTokens": getattr(self.agent, "total_input_tokens", 0),
                    "outputTokens": getattr(self.agent, "total_output_tokens", 0),
                }
            )
            unfinished = _DONE_PATTERN.search(answer or "") is None
        except asyncio.CancelledError:
            self.publish({"type": "cancelled"})
        except Exception as exc:
            logger.exception("Agent turn failed in session %s", self.id)
            self.publish({"type": "error", "message": str(exc)})
        finally:
            self.publish({"type": "status", "state": "idle"})
            self.save()
            # Never after a cancel or an error: both need the user.
            if unfinished:
                self._schedule_continuation(self.auto_delay)
            else:
                self.end_autonomy()

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

    def _title_from_answer(self, answer: str) -> None:
        """Name a session after the first answer's heading, which beats a raw SMILES."""
        if not self.meta.get("auto_title"):
            return
        heading = first_heading(answer)
        if heading:
            self.meta["title"] = heading[:_TITLE_CHARS]
            self.meta["auto_title"] = False

    def summary(self) -> dict[str, Any]:
        return {**self.meta, "busy": self.busy, "auto_until": self._auto_iso()}


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
        if backfill_title(live.meta, live.events):
            self.store.save_meta(live.meta)
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

    def busy_ids(self) -> list[str]:
        """Sessions with a turn running, so a caller can refuse to disturb them."""
        return [live.id for live in self._live.values() if live.busy]

    def rename(self, session_id: str, title: str) -> dict[str, Any] | None:
        live = self.get(session_id)
        if live is None:
            return None
        live.meta["title"] = title.strip()[:200]
        live.meta["auto_title"] = False
        live.store.save_meta(live.meta)
        return live.summary()

    def delete(self, session_id: str) -> bool:
        """Delete a session.  Refuses one with a turn running."""
        live = self._live.get(session_id)
        if live is not None:
            if live.busy:
                return False
            live.cancel_continuation()
            del self._live[session_id]
        self.store.delete(session_id)
        return True

    async def close(self) -> None:
        """Cancel running turns and queued continuations; used at shutdown."""
        for live in list(self._live.values()):
            live.cancel_continuation()
            if live.busy and live.running is not None:
                live.running.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await live.running

    def _remember(self, live: LiveSession) -> None:
        self._live[live.id] = live
        # A session working on its own is not idle, whoever is watching.
        idle = [s for s in self._live.values() if not s.busy and not s.auto_on and not s.subscribers and s is not live]
        while len(self._live) > MAX_LIVE_SESSIONS and idle:
            del self._live[idle.pop(0).id]
