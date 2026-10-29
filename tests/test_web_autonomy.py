# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the autonomy budget of a chat session.

These drive ``LiveSession`` directly, with a fake agent and a store under
``tmp_path``, so they need neither FastAPI nor an LLM.  The event loop is
started per test with ``asyncio.run`` rather than a plugin.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

from reinvent4_agent.web import sessions as web_sessions
from reinvent4_agent.web.sessions import LiveSession, SessionStore


class FakeBackend:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return list(self.history)


class FakeAgent:
    """Answers at once, recording what it was asked.

    *answer* replaces the default reply, which is how a test makes the agent
    say it is done; *hang* makes a turn run until it is cancelled.
    """

    def __init__(self, answer: str = "working on it", hang: bool = False) -> None:
        self.asked: list[str] = []
        self.answer = answer
        self.hang = hang
        self.cancelled = False
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.backend = FakeBackend()

    async def chat(self, content: str) -> str:
        self.asked.append(content)
        if self.hang:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return self.answer


def make_session(tmp_path: Path, agent: FakeAgent) -> LiveSession:
    store = SessionStore(tmp_path / "sessions")
    meta = {"id": "s-20260101-000000-abcdef", "title": "BRD4", "created": "", "updated": "", "turns": 0}
    store.create(meta)
    live = LiveSession(meta=meta, store=store, agent=agent)
    # Real continuations wait a minute; the tests must not.
    live.auto_delay = 0.0
    return live


def published(live: LiveSession, kind: str) -> list[dict[str, Any]]:
    return [event for event in live.events if event.get("type") == kind]


def busy_statuses(live: LiveSession) -> list[dict[str, Any]]:
    return [event for event in published(live, "status") if event["state"] == "busy"]


async def wait_until(check: Any, timeout: float = 5.0) -> None:
    """Run the loop until *check* holds.  Fails rather than hanging."""
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("The session never reached the expected state")
        await asyncio.sleep(0.005)


async def quiesce(live: LiveSession) -> None:
    """Stop the autonomy loop and wait out the turn it is on."""
    live.set_autonomy(0)
    if live.running is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(live.running), timeout=5)


async def idle(live: LiveSession) -> None:
    """Wait until nothing runs and nothing is queued."""
    await wait_until(lambda: not live.busy and live.auto_task is None)


# -- the budget itself -----------------------------------------------------


def test_the_budget_is_set_extended_and_cleared(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        seen: list[dict[str, Any]] = []
        queue = live.subscribe()

        assert live.auto_until is None
        assert live.summary()["auto_until"] is None

        first = live.set_autonomy(30)
        assert live.auto_on
        assert live.summary()["auto_until"] == first.isoformat(timespec="seconds")

        later = live.set_autonomy(120)
        assert later > first

        # Far beyond the cap: the deadline is the cap, not what was asked.
        capped = live.set_autonomy(10_000)
        assert (capped - later).total_seconds() < web_sessions.MAX_AUTONOMY_MINUTES * 60

        assert live.set_autonomy(0) is None
        assert not live.auto_on
        assert live.summary()["auto_until"] is None

        while not queue.empty():
            seen.append(queue.get_nowait())
        assert [event["type"] for event in seen] == ["auto"] * 4
        assert seen[-1]["until"] is None
        # The budget is live state, not part of the conversation.
        assert published(live, "auto") == []

    asyncio.run(run())


def test_a_budget_that_has_run_out_reports_nothing_left(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.set_autonomy(60)
        live.auto_until = live.auto_until.replace(year=2000)
        assert not live.auto_on
        assert live.summary()["auto_until"] is None

    asyncio.run(run())


# -- continuation turns ----------------------------------------------------


def test_a_turn_continues_by_itself_while_the_budget_lasts(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: len(agent.asked) >= 2)
        await quiesce(live)

        assert agent.asked[0] == "Design ligands for BRD4"
        assert agent.asked[1].startswith("[AUTO]")
        # Only the question is recorded as a user message; the instruction is not.
        assert len(published(live, "user_message")) == 1
        assert "auto" not in busy_statuses(live)[0]
        assert busy_statuses(live)[1]["auto"] is True

    asyncio.run(run())


def test_nothing_continues_once_the_budget_has_expired(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 0.05
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)
        # The budget runs out while the continuation waits out its pause.
        live.auto_until = live.auto_until.replace(year=2000)
        await idle(live)

        assert agent.asked == ["Design ligands for BRD4"]

    asyncio.run(run())


def test_without_a_budget_a_turn_stands_alone(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("Design ligands for BRD4")
        await idle(live)
        await asyncio.sleep(0.05)
        assert agent.asked == ["Design ligands for BRD4"]

    asyncio.run(run())


def test_turning_the_budget_on_between_turns_picks_the_campaign_up(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        assert live.start_turn("Launch the campaign")
        await idle(live)
        assert agent.asked == ["Launch the campaign"]

        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await quiesce(live)
        assert agent.asked[1].startswith("[AUTO]")
        assert busy_statuses(live)[1]["auto"] is True

    asyncio.run(run())


def test_the_loop_stops_when_the_agent_says_it_is_done(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="Nothing left to try. AUTONOMY DONE")
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await idle(live)
        await asyncio.sleep(0.05)

        assert agent.asked == ["Design ligands for BRD4"]
        # The budget is still there; the user may send it off again.
        assert live.auto_on

    asyncio.run(run())


def test_consecutive_automatic_turns_are_capped(tmp_path, monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(web_sessions, "MAX_AUTO_TURNS", 3)
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.set_autonomy(120)
        assert live.start_turn("Design ligands for BRD4")
        await idle(live)
        await asyncio.sleep(0.05)

        assert agent.asked[0] == "Design ligands for BRD4"
        assert len(agent.asked) == 4
        assert all(asked.startswith("[AUTO]") for asked in agent.asked[1:])
        assert live.auto_on

    asyncio.run(run())


# -- the hard safety rules -------------------------------------------------


def test_a_cancelled_turn_never_continues(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: agent.asked == ["Design ligands for BRD4"])
        assert live.cancel()
        await idle(live)
        await asyncio.sleep(0.05)

        assert agent.cancelled
        assert agent.asked == ["Design ligands for BRD4"]
        assert published(live, "cancelled")

    asyncio.run(run())


def test_a_failed_turn_never_continues(tmp_path):
    async def run() -> None:
        agent = FakeAgent()

        async def fail(content: str) -> str:
            agent.asked.append(content)
            raise RuntimeError("Boltz is unreachable")

        agent.chat = fail
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await idle(live)
        await asyncio.sleep(0.05)

        assert agent.asked == ["Design ligands for BRD4"]
        assert published(live, "error")[0]["message"] == "Boltz is unreachable"

    asyncio.run(run())


def test_a_user_message_takes_priority_over_a_pending_continuation(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 30.0
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)
        pending = live.auto_task

        assert live.start_turn("Use the other scaffold instead")
        assert live.auto_task is None
        assert live.auto_turns == 0
        await quiesce(live)
        await asyncio.sleep(0)

        assert pending.cancelled()
        assert agent.asked == ["Design ligands for BRD4", "Use the other scaffold instead"]

    asyncio.run(run())


def test_two_turns_never_run_at_once(tmp_path):
    async def run() -> None:
        agent = FakeAgent(hang=True)
        live = make_session(tmp_path, agent)
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.busy and agent.asked != [])
        assert not live.start_turn("and another")
        # A continuation that wakes while a turn runs stands down.
        await live._continue_after(0.0)
        assert agent.asked == ["Design ligands for BRD4"]

        live.cancel()
        await idle(live)

    asyncio.run(run())


def test_stopping_the_budget_drops_the_queued_continuation(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent)
        live.auto_delay = 30.0
        live.set_autonomy(60)
        assert live.start_turn("Design ligands for BRD4")
        await wait_until(lambda: live.auto_task is not None)

        live.set_autonomy(0)
        assert live.auto_task is None
        await asyncio.sleep(0.05)
        assert agent.asked == ["Design ligands for BRD4"]

    asyncio.run(run())
