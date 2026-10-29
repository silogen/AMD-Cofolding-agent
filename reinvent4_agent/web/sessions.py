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
        campaign.json  the agent's campaign log (see campaign.py)

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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from reinvent4_agent.web import campaign as campaign_log
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

#: Pause before carrying on when the agent is not waiting for any run: it is
#: in the middle of something, and the pause only keeps it from spinning.
AUTO_TURN_DELAY_S = 60.0

#: How often the gateway reads the state of the runs of conversations working
#: on their own.  Reading job metadata costs nothing; the model is what costs,
#: so it is woken only when something changed.
WATCH_INTERVAL_S = 30.0

#: Longest an agent goes without a turn while its runs are still going and
#: nothing has happened: long enough not to pay for "still running", short
#: enough that the time a run takes is spent on something.
AUTO_BACKSTOP_S = 15 * 60.0

#: Context, in input tokens, above which an automatic turn starts from a
#: brief and the campaign log instead of re-reading the whole conversation.
COMPACT_ABOVE_TOKENS = 100_000

#: Run states, as job metadata spells them.
ACTIVE_RUN_STATES = frozenset({"running", "queued"})
FINISHED_RUN_STATES = frozenset({"completed", "failed", "stopped"})

#: Alerts that are about a run ending; the run's own state says that first.
_END_ALERTS = frozenset({"job_completed", "job_failed"})

#: Every automatic turn opens with this.  It is not a user message: it is
#: never recorded as one, and the turn it opens is marked automatic.
AUTO_PREFIX = "[AUTO] You are running this campaign on your own; nobody is waiting for an answer."

#: What working on its own means, said on every automatic turn.
AUTO_POLICY = (
    "Decide the next step from the latest results and take it. When an "
    "experiment has finished, analyse it and run the experiment its results "
    "argue for. While a run is going, spend the time on work that does not "
    "compete with it -- score, cofold, profile or counter-screen the current "
    "best, or prepare the next TTP -- and start a second run only to test "
    "something different (other seeds, another chemotype, other weights). Treat "
    "a problem with a run as the [MONITOR] section describes. Record what you "
    "learned and the next step, with the result behind it, in campaign_log. "
    "Keep the reply to a few lines. End with AUTONOMY DONE only when you cannot "
    "name a result that justifies another step, or the next step needs something "
    "only the user can give. Next steps you could carry out yourself are not the "
    "user's call: choose the best one and start it. A run still going is never a "
    "reason to stop."
)

#: Why an automatic turn was taken, as the turn is told.
_WAKE_REASONS = {
    "start": "The user has just left the campaign with you.",
    "continue": "Carry on from your last turn.",
    "confirm": (
        "You were about to hand the campaign back. Check that first: if any next "
        "step in your last answer or in the campaign log can be done with your "
        "tools -- fix a TTP, cofold, score or profile another compound, start or "
        "extend a run -- it is not the user's decision. Pick the one the latest "
        "results support most and start it now. Hand back with AUTONOMY DONE only "
        "if every remaining step needs something only the user can give: a target "
        "they have not named, data you cannot fetch, or a choice between their own "
        "goals."
    ),
}

#: Last answer and user requests a condensed history keeps, in characters.
_BRIEF_ANSWER_CHARS = 1500
_BRIEF_REQUEST_CHARS = 500
_BRIEF_REQUESTS = 6

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


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


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

    def load_campaign(self, session_id: str) -> dict[str, Any]:
        return campaign_log.normalise(self._read_json(self._dir(session_id) / "campaign.json"))

    def save_campaign(self, session_id: str, log: dict[str, Any]) -> None:
        self._write_json(self._dir(session_id) / "campaign.json", log)

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
    #: Asks the gateway what this conversation's runs are doing.  None where
    #: nothing watches runs (the tests, the CLI), and then the agent is simply
    #: never waiting for one.
    observe: Callable[[LiveSession], Awaitable[None]] | None = None
    #: Last state seen of each run this conversation launched.
    runs_seen: dict[str, str] = field(default_factory=dict)
    #: Problems already reported, so a plateau is news once.
    alerts_seen: set[tuple[str, str]] = field(default_factory=set)
    #: What happened since the agent last looked, for its next turn to say.
    pending: list[str] = field(default_factory=list)
    #: When the agent last finished a turn, on the monotonic clock: the start
    #: of the quiet spell a long wait is measured from.
    last_turn: float = field(default_factory=time.monotonic)
    #: The agent has said it is done once since it last did any work; the
    #: first time is checked with a confirm turn, the second time is final.
    done_once: bool = False
    #: The agent's campaign log.
    campaign: dict[str, Any] = field(default_factory=campaign_log.empty)

    def __post_init__(self) -> None:
        self.campaign = self.store.load_campaign(self.id)
        self.install_tools()

    @property
    def id(self) -> str:
        return self.meta["id"]

    def install_tools(self) -> None:
        """Give the agent the tools that belong to this conversation."""
        tools = getattr(self.agent, "local_tools", None)
        if isinstance(tools, dict):
            tools[campaign_log.TOOL_NAME] = (campaign_log.TOOL_DEFINITION, self._campaign_tool)

    def _campaign_tool(self, arguments: dict[str, Any]) -> str:
        self.campaign = campaign_log.apply(self.campaign, arguments)
        self.store.save_campaign(self.id, self.campaign)
        self.publish({"type": "campaign", "campaign": self.campaign})
        return campaign_log.as_tool_result(self.campaign)

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
            starting = not self.auto_on
            self.auto_until = datetime.now(timezone.utc) + timedelta(minutes=min(float(minutes), float(MAX_AUTONOMY_MINUTES)))
            self.auto_turns = 0
            if starting:
                # What happened while nobody was watching is not news: the
                # first look sets what counts as unchanged.
                self.runs_seen.clear()
                self.alerts_seen.clear()
                self.pending.clear()
                self.done_once = False
            # Turned on between two turns: pick the campaign up at once rather
            # than waiting for a message the user does not intend to send.
            if not self.busy and self.meta.get("turns"):
                self._schedule_continuation(0.0, reason="start")
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

    def _schedule_continuation(self, delay: float, reason: str = "continue") -> None:
        """Queue the next automatic turn.  The task checks again before it starts."""
        if not self.auto_on:
            return
        if self.auto_turns >= MAX_AUTO_TURNS:
            self.end_autonomy()
            return
        self.cancel_continuation()
        self.auto_task = asyncio.create_task(self._continue_after(delay, reason))

    async def _continue_after(self, delay: float, reason: str = "continue") -> None:
        """Wait out *delay*, then take another turn unless it should wait for a run.

        A turn that left runs going and nothing new to look at hands over to
        the watcher, which wakes the agent when one of them has news.  The
        user handing over the campaign is a reason on its own.
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if self.observe is not None:
            try:
                await self.observe(self)
            except Exception:
                logger.exception("Could not read the runs of session %s", self.id)
        self.auto_task = None
        if self.busy or not self.auto_on:
            return
        if reason not in ("start", "confirm") and self.running_runs and not self.pending:
            return
        self._wake(reason)

    # -- what the watcher sees -------------------------------------------------

    @property
    def running_runs(self) -> list[str]:
        """Runs this conversation launched that are queued or running."""
        return [job_id for job_id, state in self.runs_seen.items() if state in ACTIVE_RUN_STATES]

    def note_runs(self, states: dict[str, str]) -> None:
        """Take the latest state of this conversation's runs; a run that ended is news."""
        for job_id, state in states.items():
            before = self.runs_seen.get(job_id)
            if before in ACTIVE_RUN_STATES and state in FINISHED_RUN_STATES:
                self.pending.append(f"Run {job_id} {state}.")
            self.runs_seen[job_id] = state

    def note_alerts(self, alerts: list[dict[str, Any]]) -> None:
        """Take the problems reported for running jobs; each is news once."""
        running = set(self.running_runs)
        for alert in alerts:
            job_id = str(alert.get("job_id") or "")
            kind = str(alert.get("type") or "")
            if job_id not in running or kind in _END_ALERTS or (job_id, kind) in self.alerts_seen:
                continue
            self.alerts_seen.add((job_id, kind))
            message = str(alert.get("message") or kind).strip()
            self.pending.append(f"Run {job_id} ({alert.get('severity') or 'medium'}): {message}")

    def wake_if_due(self) -> None:
        """Take a turn if something happened, or if the runs have been quiet too long."""
        if self.busy or not self.auto_on or self.auto_task is not None:
            return
        if self.pending:
            self._wake("events")
        elif self.running_runs and time.monotonic() - self.last_turn >= AUTO_BACKSTOP_S:
            self._wake("quiet")

    def _wake(self, reason: str) -> None:
        if self.auto_turns >= MAX_AUTO_TURNS:
            self.end_autonomy()
            return
        self.auto_turns += 1
        events, self.pending = self.pending, []
        brief = self._condense()
        self._begin(self._wake_prompt(reason, events, brief), auto=True)

    def _wake_prompt(self, reason: str, events: list[str], brief: str) -> str:
        parts = [AUTO_PREFIX]
        if brief:
            parts.append(brief)
        if events:
            parts.append("Since you last looked:\n" + "\n".join(f"- {event}" for event in events))
        elif reason == "quiet":
            minutes = round(AUTO_BACKSTOP_S / 60)
            parts.append(
                f"Runs still going: {', '.join(self.running_runs)}. Nothing has changed "
                f"for {minutes} minutes; use the time."
            )
        else:
            parts.append(_WAKE_REASONS.get(reason, _WAKE_REASONS["continue"]))
        parts.append(
            campaign_log.render(self.campaign)
            or "You have no campaign log yet: set the goal with campaign_log before anything else."
        )
        parts.append(AUTO_POLICY)
        return "\n\n".join(parts)

    def _condense(self) -> str:
        """Start the agent afresh from a brief once its context has grown too long.

        Returns the brief, which the turn then opens with, or "" when the
        history is short enough to keep.  The browser's record of the
        conversation is untouched: only what the model re-reads shrinks.
        """
        tokens = getattr(self.agent, "context_tokens", 0) or 0
        reset = getattr(self.agent, "reset_history", None)
        if tokens <= COMPACT_ABOVE_TOKENS or not callable(reset):
            return ""
        requests = [e.get("content", "") for e in self.events if e.get("type") == "user_message"]
        answers = [e.get("content", "") for e in self.events if e.get("type") == "assistant_message"]
        lines = [
            "Your earlier conversation was condensed to keep turns fast; what matters "
            "of it is below and in the campaign log. Look up anything else with your "
            "tools (list_jobs, get_optimization_status, list_ttps, list_shared_assets).",
            "What the user asked, oldest first:",
            *(f"- {_clip(text, _BRIEF_REQUEST_CHARS)}" for text in requests[-_BRIEF_REQUESTS:]),
        ]
        if self.runs_seen:
            lines.append("Runs of this conversation: " + ", ".join(f"{job} ({state})" for job, state in self.runs_seen.items()))
        if answers:
            lines.append("Your last answer:\n" + _clip(answers[-1], _BRIEF_ANSWER_CHARS))
        reset()
        self.agent.context_tokens = 0
        logger.info("Session %s: condensed a %d-token history before an automatic turn", self.id, tokens)
        return "\n".join(lines)

    def _begin(self, content: str, auto: bool = False) -> None:
        """Record the start of a turn and run it."""
        self.publish({"type": "status", "state": "busy", **({"auto": True} if auto else {})})
        self.meta["turns"] = sum(1 for event in self.events if event.get("type") == "user_message")
        self.meta["updated"] = _now()
        self.store.save_meta(self.meta)
        self.running = asyncio.create_task(self._run_turn(content))

    async def _run_turn(self, content: str) -> None:
        unfinished = done = False
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
            done = _DONE_PATTERN.search(answer or "") is not None
            unfinished = not done
        except asyncio.CancelledError:
            self.publish({"type": "cancelled"})
        except Exception as exc:
            logger.exception("Agent turn failed in session %s", self.id)
            self.publish({"type": "error", "message": str(exc)})
        finally:
            self.last_turn = time.monotonic()
            self.publish({"type": "status", "state": "idle"})
            self.save()
            # Never after a cancel or an error: both need the user.
            if unfinished:
                self.done_once = False
                self._schedule_continuation(self.auto_delay)
            elif done and self.auto_on and not self.done_once:
                self.done_once = True
                self._schedule_continuation(self.auto_delay, reason="confirm")
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

#: ``{session id: {job id: state}}`` for the runs each conversation launched.
RunStates = Callable[[list[str]], Awaitable[dict[str, dict[str, str]]]]

#: The problems the job manager currently reports, one dict per alert.
JobAlerts = Callable[[], Awaitable[list[dict[str, Any]]]]


class SessionManager:
    """Creates, loads and forgets live sessions, and watches the runs of busy ones."""

    def __init__(self, store: SessionStore, agent_factory: AgentFactory) -> None:
        self.store = store
        self._agent_factory = agent_factory
        self._live: dict[str, LiveSession] = {}
        self._run_states: RunStates | None = None
        self._job_alerts: JobAlerts | None = None
        self._watcher: asyncio.Task[None] | None = None
        self.watch_interval = WATCH_INTERVAL_S

    def _attach_agent(self, live: LiveSession) -> None:
        # A fresh id prefix per load keeps call ids unique across restores.
        handler = WebSocketEventHandler(sink=live.publish, id_prefix=f"call-{secrets.token_hex(3)}")
        live.handler = handler
        live.agent = self._agent_factory(handler)
        live.install_tools()
        if self._run_states is not None:
            live.observe = self._observe

    # -- watching runs ---------------------------------------------------------

    def watch(self, run_states: RunStates, job_alerts: JobAlerts) -> None:
        """Follow the runs of conversations working on their own, without the model.

        Every few seconds the state of their runs is read; a run that ended,
        or a new problem with one, wakes the agent with the news.
        """
        self._run_states = run_states
        self._job_alerts = job_alerts
        for live in self._live.values():
            live.observe = self._observe
        if self._watcher is None:
            self._watcher = asyncio.create_task(self._watch_loop())

    async def _observe(self, live: LiveSession) -> None:
        await self._look([live])

    async def _look(self, sessions: list[LiveSession]) -> None:
        """Bring the sessions' view of their runs up to date."""
        if self._run_states is None or not sessions:
            return
        states = await self._run_states([live.id for live in sessions])
        for live in sessions:
            live.note_runs(states.get(live.id, {}))
        if self._job_alerts is not None and any(live.running_runs for live in sessions):
            alerts = await self._job_alerts()
            for live in sessions:
                live.note_alerts(alerts)

    async def _watch_loop(self) -> None:
        while True:
            await asyncio.sleep(self.watch_interval)
            working = [live for live in self._live.values() if live.auto_on and not live.busy]
            try:
                await self._look(working)
            except Exception:
                logger.exception("Could not read the state of the runs")
                continue
            for live in working:
                live.wake_if_due()

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
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
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
