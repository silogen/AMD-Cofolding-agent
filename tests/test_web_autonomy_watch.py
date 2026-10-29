# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for what wakes an agent working on its own, and what it is told.

The gateway follows the runs itself and wakes the agent only with news: a
run that ended, a new problem with one, the user handing over, or a long
quiet spell.  These drive ``LiveSession`` and ``SessionManager`` with fake
agents and fake run readers, so they need neither FastAPI nor an LLM.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

import pytest

from reinvent4_agent.web import campaign
from reinvent4_agent.web import sessions as web_sessions
from reinvent4_agent.web.sessions import LiveSession, SessionManager, SessionStore

SESSION_ID = "s-20260101-000000-abcdef"


class FakeBackend:
    def __init__(self) -> None:
        self.reasoning_effort: str | None = None

    def export_history(self) -> list[dict]:
        return []


class FakeAgent:
    """Answers at once and records what it was asked, like the real one's surface."""

    def __init__(self, answer: str = "working on it") -> None:
        self.asked: list[str] = []
        self.answer = answer
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.context_tokens = 0
        self.local_tools: dict[str, Any] = {}
        self.resets = 0
        self.backend = FakeBackend()

    async def chat(self, content: str) -> str:
        self.asked.append(content)
        return self.answer

    def reset_history(self) -> None:
        self.resets += 1


def make_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def make_session(tmp_path: Path, agent: FakeAgent, runs: dict[str, str] | None = None) -> LiveSession:
    store = make_store(tmp_path)
    meta = {"id": SESSION_ID, "title": "BRD4", "created": "", "updated": "", "turns": 1}
    store.create(meta)
    live = LiveSession(meta=meta, store=store, agent=agent)
    live.auto_delay = 0.0
    if runs is not None:
        state = dict(runs)

        async def observe(session: LiveSession) -> None:
            session.note_runs(state)

        live.observe = observe
        live.runs_state = state  # type: ignore[attr-defined]
    return live


async def wait_until(check: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("The session never reached the expected state")
        await asyncio.sleep(0.005)


async def settle(live: LiveSession) -> None:
    await wait_until(lambda: not live.busy and live.auto_task is None)
    await asyncio.sleep(0.02)


async def stop(live: LiveSession) -> None:
    live.set_autonomy(0)
    if live.running is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(live.running), timeout=5)


# -- what wakes the agent --------------------------------------------------


def test_handing_over_the_campaign_wakes_the_agent_even_while_a_run_goes(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert agent.asked[0].startswith("[AUTO]")
        assert "The user has just left the campaign with you." in agent.asked[0]

    asyncio.run(run())


def test_with_a_run_going_and_nothing_new_the_agent_is_left_alone(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        # The turn did not finish the campaign, but the agent is waiting on a
        # run: no "still running" turn is paid for.
        assert len(agent.asked) == 1
        assert live.auto_on
        live.wake_if_due()
        await settle(live)
        assert len(agent.asked) == 1
        await stop(live)

    asyncio.run(run())


def test_a_run_that_ends_wakes_the_agent_with_the_news(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        live.note_runs({"abc12345": "completed"})
        live.wake_if_due()
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Since you last looked:\n- Run abc12345 completed." in agent.asked[1]

    asyncio.run(run())


def test_a_long_quiet_spell_wakes_the_agent_to_use_the_time(tmp_path, monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(web_sessions, "AUTO_BACKSTOP_S", 0.0)
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        live.wake_if_due()
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Runs still going: abc12345." in agent.asked[1]
        assert "use the time" in agent.asked[1]

    asyncio.run(run())


def test_a_quiet_spell_is_counted_from_the_last_turn_not_from_the_last_wake(tmp_path, monkeypatch):
    async def run() -> None:
        monkeypatch.setattr(web_sessions, "AUTO_BACKSTOP_S", 3600.0)
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.meta["turns"] = 0
        live.set_autonomy(60)
        assert live.start_turn("Launch the JAK3 campaign")
        await settle(live)

        # The user's own turn just ended: an hour has not gone by.
        live.wake_if_due()
        await settle(live)
        await stop(live)

        assert agent.asked == ["Launch the JAK3 campaign"]

    asyncio.run(run())


def test_with_no_run_to_wait_for_an_unfinished_campaign_carries_on(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "completed"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 2)
        await stop(live)

        assert "Carry on from your last turn." in agent.asked[1]

    asyncio.run(run())


def test_runs_that_ended_before_the_handover_are_not_news(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"old00001": "completed"})
    live.note_runs({"old00001": "completed"})
    assert live.pending == []


def test_a_user_turn_takes_over_from_a_wake(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "running"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)
        assert live.start_turn("Try the morpholine series instead")
        await settle(live)
        await stop(live)

        assert agent.asked[1] == "Try the morpholine series instead"

    asyncio.run(run())


# -- problems with runs ----------------------------------------------------


def alert(job_id: str, kind: str, message: str = "Score plateau since step 180", severity: str = "medium") -> dict:
    return {"job_id": job_id, "type": kind, "message": message, "severity": severity}


def test_a_problem_with_a_running_run_is_news_once(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("abc12345", "score_plateau")])
    live.note_alerts([alert("abc12345", "score_plateau")])

    assert live.pending == ["Run abc12345 (medium): Score plateau since step 180"]


def test_problems_with_other_conversations_runs_are_not_ours(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("zzz99999", "component_failure", "ADMET NaN for 90%", "high")])

    assert live.pending == []


def test_a_run_ending_is_told_once_not_twice(tmp_path):
    live = make_session(tmp_path, FakeAgent())
    live.note_runs({"abc12345": "running"})

    live.note_alerts([alert("abc12345", "job_completed", "Job abc12345 completed", "low")])
    live.note_runs({"abc12345": "completed"})

    assert live.pending == ["Run abc12345 completed."]


def test_handing_over_again_forgets_what_was_seen_before(tmp_path):
    async def run() -> None:
        live = make_session(tmp_path, FakeAgent())
        live.meta["turns"] = 0
        live.note_runs({"abc12345": "running"})
        live.note_alerts([alert("abc12345", "score_plateau")])

        live.set_autonomy(30)

        assert live.pending == []
        assert live.runs_seen == {}
        assert live.alerts_seen == set()
        live.set_autonomy(0)

    asyncio.run(run())


# -- the campaign log ------------------------------------------------------


def test_the_agent_gets_a_campaign_log_tool(tmp_path):
    agent = FakeAgent()
    make_session(tmp_path, agent)
    definition, _ = agent.local_tools[campaign.TOOL_NAME]
    assert definition["name"] == "campaign_log"
    assert set(definition["input_schema"]["properties"]) == {"goal", "finding", "next"}


def test_the_log_is_kept_published_and_survives_a_reload(tmp_path):
    agent = FakeAgent()
    live = make_session(tmp_path, agent)
    _, handler = agent.local_tools[campaign.TOOL_NAME]

    handler({"goal": "BRD4 BD1 binders with logS above -4"})
    handler({"finding": "Run abc12345: best 0.82, solubility flat after step 180", "next": "Add a morpholine seed set"})

    assert live.campaign["goal"] == "BRD4 BD1 binders with logS above -4"
    assert [f["text"] for f in live.campaign["findings"]] == ["Run abc12345: best 0.82, solubility flat after step 180"]
    events = [e for e in live.events if e["type"] == "campaign"]
    assert len(events) == 2
    assert events[-1]["campaign"]["next"] == "Add a morpholine seed set"

    reloaded = LiveSession(meta=live.meta, store=make_store(tmp_path), agent=FakeAgent())
    assert reloaded.campaign == live.campaign


def test_a_call_that_changes_nothing_is_refused(tmp_path):
    agent = FakeAgent()
    make_session(tmp_path, agent)
    _, handler = agent.local_tools[campaign.TOOL_NAME]
    with pytest.raises(ValueError):
        handler({})


def test_every_wake_shows_the_log(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        _, handler = agent.local_tools[campaign.TOOL_NAME]
        handler({"goal": "Soluble JAK3 inhibitors", "next": "Cofold the top 5"})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert "Goal: Soluble JAK3 inhibitors" in agent.asked[0]
        assert "Next: Cofold the top 5" in agent.asked[0]

    asyncio.run(run())


def test_without_a_log_the_wake_asks_for_the_goal_first(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={})
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert "set the goal with campaign_log" in agent.asked[0]

    asyncio.run(run())


# -- keeping turns small ---------------------------------------------------


def test_a_long_history_is_condensed_before_an_automatic_turn(tmp_path):
    async def run() -> None:
        agent = FakeAgent()
        live = make_session(tmp_path, agent, runs={"abc12345": "completed"})
        live.publish({"type": "user_message", "content": "Design soluble JAK3 inhibitors"})
        live.publish({"type": "assistant_message", "content": "Launched run abc12345."})
        agent.context_tokens = web_sessions.COMPACT_ABOVE_TOKENS + 1
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await stop(live)

        assert agent.resets == 1
        assert "condensed" in agent.asked[0]
        assert "- Design soluble JAK3 inhibitors" in agent.asked[0]
        assert "Launched run abc12345." in agent.asked[0]
        assert "abc12345 (completed)" in agent.asked[0]

    asyncio.run(run())


def test_a_short_history_is_kept(tmp_path):
    async def run() -> None:
        agent = FakeAgent(answer="Nothing to add. AUTONOMY DONE")
        live = make_session(tmp_path, agent, runs={})
        agent.context_tokens = 5_000
        live.set_autonomy(60)
        await wait_until(lambda: len(agent.asked) >= 1)
        await settle(live)

        assert agent.resets == 0
        assert "condensed" not in agent.asked[0]

    asyncio.run(run())


# -- the watcher -----------------------------------------------------------


def test_the_watcher_wakes_a_session_whose_run_ended(tmp_path):
    async def run() -> None:
        agents: list[FakeAgent] = []

        def factory(handler: Any) -> FakeAgent:
            agents.append(FakeAgent())
            return agents[-1]

        states = {"abc12345": "running"}
        looked: list[list[str]] = []

        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            looked.append(ids)
            return {sid: dict(states) for sid in ids}

        async def job_alerts() -> list[dict[str, Any]]:
            return []

        manager = SessionManager(make_store(tmp_path), factory)
        manager.watch_interval = 0.01
        live = manager.create()
        live.auto_delay = 0.0
        live.meta["turns"] = 1
        manager.watch(run_states, job_alerts)
        try:
            live.set_autonomy(60)
            await wait_until(lambda: len(agents[0].asked) >= 1)
            await settle(live)
            assert len(agents[0].asked) == 1

            states["abc12345"] = "completed"
            await wait_until(lambda: len(agents[0].asked) >= 2)
            assert "Run abc12345 completed." in agents[0].asked[1]
        finally:
            live.set_autonomy(0)
            await manager.close()

    asyncio.run(run())


def test_the_watcher_reads_nothing_while_nobody_works_on_their_own(tmp_path):
    async def run() -> None:
        looked: list[list[str]] = []

        async def run_states(ids: list[str]) -> dict[str, dict[str, str]]:
            looked.append(ids)
            return {}

        async def job_alerts() -> list[dict[str, Any]]:
            raise AssertionError("no alerts should be read")

        manager = SessionManager(make_store(tmp_path), lambda handler: FakeAgent())
        manager.watch_interval = 0.01
        manager.create()
        manager.watch(run_states, job_alerts)
        await asyncio.sleep(0.05)
        await manager.close()

        assert all(ids == [] for ids in looked)

    asyncio.run(run())
