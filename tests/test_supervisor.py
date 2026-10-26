"""Tests for reinvent4_agent/supervisor.py -- the headless worker loop.

The campaign run is injected into ``Supervisor``, so everything around it --
leases, failures, deadlines, shutdown, and reading the achieved profile out of
a results CSV -- is testable with no LLM, no MCP server and no GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from reinvent4_agent.supervisor import (
    DRIVER_DENIED_TOOLS,
    STEP_SECONDS_HINT,
    CampaignStoreClient,
    LeaseLost,
    StoreUnavailable,
    Supervisor,
    TaskOutcome,
    _deadline_seconds,
    apply_overrides,
    build_goal,
    job_for_ttp,
    objective_profile,
    prepare_ttp_variant,
)

BASE_TTP = {
    "id": "brd4-bd1-lead-opt",
    "name": "BRD4 BD1 lead optimisation",
    "version": 3,
    "pre_scoring": {
        "components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.5},
        ]
    },
    "scoring": {
        "type": "geometric_mean",
        "objectives": [
            {"name": "BoltzAffinity", "type": "external", "service": "boltz",
             "weight": 1.0},
            {"name": "logD", "type": "external", "service": "admet", "weight": 0.7},
        ],
    },
}


def _task(**spec: Any) -> dict[str, Any]:
    return {
        "task_id": "a1b2c3d4",
        "campaign_id": "feed0001",
        "state": "claimed",
        "spec": {"weight_overrides": {}, "fidelity_tier": "exploration", **spec},
    }


class FakeStore:
    """A campaign store that hands out a fixed list of tasks."""

    def __init__(self, tasks: list[dict[str, Any]] | None = None) -> None:
        self.base_url = "http://store"
        self.owner = "supervisor-0"
        self.queue = list(tasks or [])
        self.results: list[tuple[str, TaskOutcome]] = []
        self.failures: list[tuple[str, str, bool]] = []
        self.heartbeats: list[str] = []
        self.presence: list[tuple[str, Any]] = []
        self.left_the_pool = False
        self.claims = 0

    def claim(self, campaign_id=None, lease_seconds=None):
        self.claims += 1
        return self.queue.pop(0) if self.queue else None

    def report_presence(self, state, campaign_id=None, task_id=None,
                        tasks_completed=0, tasks_failed=0, detail=None):
        self.presence.append((state, task_id))
        return {"owner": self.owner, "state": state}

    def drop_presence(self):
        self.left_the_pool = True

    def heartbeat(self, task_id, lease_seconds=None):
        self.heartbeats.append(task_id)

    def post_result(self, task_id, outcome):
        self.results.append((task_id, outcome))
        return {"task_id": task_id}

    def fail(self, task_id, error, requeue=True):
        self.failures.append((task_id, error, requeue))
        return {"task_id": task_id, "state": "queued" if requeue else "failed"}


async def _succeeds(task: dict[str, Any]) -> TaskOutcome:
    return TaskOutcome(
        objective_profile={"BoltzAffinity": 0.8, "QED": 0.7},
        candidates=[{"smiles": "CCO"}],
        job_id="job1",
    )


# ---------------------------------------------------------------------------
# Weight overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:

    def test_overrides_a_scoring_objective(self):
        updated = apply_overrides(BASE_TTP, {"BoltzAffinity": 2.5})
        weights = {
            o["name"]: o["weight"] for o in updated["scoring"]["objectives"]
        }

        assert weights["BoltzAffinity"] == 2.5
        assert weights["logD"] == 0.7

    def test_overrides_a_pre_scoring_component(self):
        """A sweep may well want to lean on drug-likeness, not just affinity."""
        updated = apply_overrides(BASE_TTP, {"QED": 1.5})

        assert updated["pre_scoring"]["components"][0]["weight"] == 1.5

    def test_overrides_a_transform(self):
        updated = apply_overrides(
            BASE_TTP, transform_overrides={"logD": {"type": "sigmoid", "k": 0.4}}
        )
        logd = next(
            o for o in updated["scoring"]["objectives"] if o["name"] == "logD"
        )

        assert logd["transform"] == {"type": "sigmoid", "k": 0.4}

    def test_the_base_ttp_is_not_mutated(self):
        """N supervisors share the TTP store; the base must survive untouched."""
        apply_overrides(BASE_TTP, {"BoltzAffinity": 9.0})

        original = next(
            o for o in BASE_TTP["scoring"]["objectives"]
            if o["name"] == "BoltzAffinity"
        )
        assert original["weight"] == 1.0

    def test_an_unknown_objective_is_an_error(self):
        """A typo must not become a no-op.

        Silently ignoring it would have the task report a profile for a weight
        vector it never ran, and the conductor would draw conclusions from a
        point that was never tried.
        """
        with pytest.raises(ValueError) as exc:
            apply_overrides(BASE_TTP, {"BoltzAfinity": 2.0})

        assert "BoltzAfinity" in str(exc.value)
        assert "BoltzAffinity" in str(exc.value), "should list what is available"

    def test_no_overrides_is_a_faithful_copy(self):
        assert apply_overrides(BASE_TTP) == BASE_TTP

    def test_a_ttp_without_pre_scoring_is_fine(self):
        bare = {"id": "x", "name": "x", "scoring": {"objectives": [
            {"name": "BoltzAffinity", "type": "external", "weight": 1.0},
        ]}}

        assert apply_overrides(bare, {"BoltzAffinity": 2.0})["scoring"][
            "objectives"
        ][0]["weight"] == 2.0


# ---------------------------------------------------------------------------
# Objective profile
# ---------------------------------------------------------------------------


class TestObjectiveProfile:

    def test_the_profile_is_one_molecules_scores(self):
        """Not the best value per objective, which would mix molecules.

        A per-objective maximum describes a compound nobody made, which is
        worse than useless on a front whose job is to show achievable
        trade-offs.
        """
        candidates = [
            {
                "smiles": "best", "total_score": 0.9,
                "components": {
                    "BoltzAffinity": {"transformed": 0.85},
                    "QED": {"transformed": 0.60},
                },
            },
            {
                "smiles": "spikier", "total_score": 0.4,
                "components": {
                    "BoltzAffinity": {"transformed": 0.10},
                    "QED": {"transformed": 0.99},
                },
            },
        ]

        assert objective_profile(candidates) == {
            "BoltzAffinity": 0.85, "QED": 0.60,
        }

    def test_objectives_can_be_restricted_to_the_campaigns(self, ):
        candidates = [{
            "total_score": 0.9,
            "components": {
                "BoltzAffinity": {"transformed": 0.8},
                "MolecularWeight": {"transformed": 0.5},
            },
        }]

        assert objective_profile(candidates, ["BoltzAffinity"]) == {
            "BoltzAffinity": 0.8
        }

    def test_no_candidates_is_an_empty_profile(self):
        assert objective_profile([]) == {}

    def test_candidates_without_a_total_score_are_skipped(self):
        assert objective_profile([{"smiles": "CCO"}]) == {}

    def test_a_nan_component_is_left_out(self):
        """A failed scorer must not become a number on the front."""
        candidates = [{
            "total_score": 0.5,
            "components": {
                "BoltzAffinity": {"transformed": None},
                "QED": {"transformed": 0.7},
            },
        }]

        assert objective_profile(candidates) == {"QED": 0.7}


# ---------------------------------------------------------------------------
# The goal handed to the Driver
# ---------------------------------------------------------------------------


class TestBuildGoal:

    def test_the_goal_names_the_prepared_ttp(self):
        """One source of truth for the weight vector: the TTP, not prose."""
        goal = build_goal(_task(weight_overrides={"QED": 2.0}), "brd4-a1b2c3d4")

        assert "brd4-a1b2c3d4" in goal
        assert "do not modify" in goal
        assert "2.0" not in goal, "restating the weights invites paraphrasing"

    def test_the_step_budget_is_passed_on_as_a_ceiling(self):
        goal = build_goal(_task(max_steps=120), "ttp-1")

        assert "120 RL steps" in goal
        assert "do not raise it" in goal

    def test_a_validation_task_is_told_not_to_use_the_cache(self):
        goal = build_goal(_task(fidelity_tier="validation"), "ttp-1")

        assert "validation-tier" in goal

    def test_an_exploration_task_is_not(self):
        assert "validation-tier" not in build_goal(_task(), "ttp-1")

    def test_the_tasks_intent_reaches_the_agent_verbatim(self):
        """The channel for what the structured fields cannot say.

        Holding a chemotype is a choice of generative prior, and the agent can
        look the reference compound up and pick a model. It ran de novo not
        because it could not do better but because nothing asked it to.
        """
        intent = "Hold the JQ1 methyl-triazolo-benzodiazepine core fixed."

        goal = build_goal(_task(intent=intent), "ttp-1")

        assert intent in goal

    def test_an_intent_invites_the_agent_to_choose_the_prior(self):
        goal = build_goal(_task(intent="stay on the I-BET151 chemotype"), "ttp-1")

        assert "choose the model" in goal
        assert "look the reference" in goal
        assert "why before starting" in goal

    def test_a_task_without_an_intent_says_nothing_about_priors(self):
        """Silence keeps the default de novo behaviour for ordinary sweeps."""
        goal = build_goal(_task(), "ttp-1")

        assert "What this run is for" not in goal
        assert "de novo" not in goal


class TestTheDeNovoFallback:
    """When starting the run ourselves is still the task's own experiment.

    The supervisor starts the run when its agent finishes without doing so,
    using only the TTP and step count -- so from the default de novo prior.
    """

    def test_an_ordinary_task_can_be_started_for_the_agent(self):
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert default_run_substitutes_for({"weight_overrides": {"QED": 2.0}})

    def test_a_task_with_an_intent_cannot(self):
        """A substituted run posts into the same front while measuring else."""
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert not default_run_substitutes_for({"intent": "hold the JQ1 core"})

    def test_an_absent_spec_is_treated_as_ordinary(self):
        from reinvent4_agent.supervisor import default_run_substitutes_for

        assert default_run_substitutes_for({})
        assert default_run_substitutes_for(None)

    def test_the_driver_is_told_how_long_a_step_takes(self):
        """Without a number it cannot tell a slow step from a broken one.

        The first live run had the Driver poll thirteen times in six minutes,
        decide the job had stalled, and stop it inside its first step -- where
        REINVENT has nothing to checkpoint and no candidates exist yet.
        """
        goal = build_goal(_task(), "ttp-1")

        assert STEP_SECONDS_HINT in goal
        assert "is normal" in goal
        assert "Do not try to stop" in goal


class TestDriverToolBoundary:
    """A supervisor's Driver runs the task it was given, and cannot redefine it."""

    def test_stopping_the_run_is_not_the_drivers_to_do(self):
        # The supervisor owns the deadline and stops with a checkpoint. The
        # Driver stopping on impatience loses the work.
        assert "stop_optimization" in DRIVER_DENIED_TOOLS

    def test_it_cannot_edit_or_replace_a_ttp(self):
        # The whole point of the variant TTP is that the weight vector is
        # fixed in advance; a Driver that can write TTPs can quietly run
        # something other than the point the campaign is measuring.
        assert {"create_ttp", "update_ttp", "delete_ttp"} <= DRIVER_DENIED_TOOLS

    def test_it_cannot_steer_the_fleet(self):
        # A worker must not open campaigns, queue work, or cancel its peers.
        assert {
            "start_campaign", "enqueue_task", "set_concurrency",
            "cancel_supervisor", "close_campaign", "finalize_campaign",
        } <= DRIVER_DENIED_TOOLS

    def test_reading_is_still_allowed(self):
        # The denial is about mutation, not curiosity -- it still has to be
        # able to watch its own job.
        for tool in (
            "run_optimization", "get_optimization_status", "get_top_candidates",
            "get_ttp", "get_campaign_state", "get_pareto_front",
        ):
            assert tool not in DRIVER_DENIED_TOOLS

    def test_the_allow_list_is_the_live_tools_minus_the_denied(self):
        """Computed, not hardcoded, so a tool added later is allowed by default."""
        live = {
            "run_optimization", "get_optimization_status", "stop_optimization",
            "update_ttp", "some_future_tool",
        }

        allowed = live - DRIVER_DENIED_TOOLS

        assert allowed == {
            "run_optimization", "get_optimization_status", "some_future_tool",
        }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestSupervisorLoop:
    """The loop drives its own event loop: the repo has no pytest-asyncio."""

    def test_an_empty_queue_is_not_work(self):
        store = FakeStore()
        supervisor = Supervisor(store, _succeeds)

        assert asyncio.run(supervisor.run_once()) is False
        assert store.results == []

    def test_a_successful_task_posts_its_profile(self):
        store = FakeStore([_task()])
        supervisor = Supervisor(store, _succeeds)

        assert asyncio.run(supervisor.run_once()) is True

        task_id, outcome = store.results[0]
        assert task_id == "a1b2c3d4"
        assert outcome.objective_profile == {"BoltzAffinity": 0.8, "QED": 0.7}
        assert supervisor.tasks_completed == 1

    def test_a_failing_task_is_reported_and_requeued(self):
        async def _explodes(task):
            raise RuntimeError("the runner went away")

        store = FakeStore([_task()])
        supervisor = Supervisor(store, _explodes)

        assert asyncio.run(supervisor.run_once()) is True

        assert store.failures == [("a1b2c3d4", "the runner went away", True)]
        assert store.results == []
        assert supervisor.tasks_failed == 1

    def test_a_lost_lease_mid_task_posts_nothing(self):
        """The new holder's result must not be overwritten by ours."""
        async def _loses_lease(task):
            raise LeaseLost("held by supervisor-9")

        store = FakeStore([_task()])
        supervisor = Supervisor(store, _loses_lease)

        assert asyncio.run(supervisor.run_once()) is True

        assert store.results == []
        assert store.failures == [], "a task we no longer own is not ours to fail"

    def test_a_failure_report_that_cannot_be_delivered_is_survivable(self):
        """A store outage must not take the pod down with it."""
        class Unreachable(FakeStore):
            def fail(self, task_id, error, requeue=True):
                raise StoreUnavailable("store is down")

        async def _explodes(task):
            raise RuntimeError("boom")

        store = Unreachable([_task()])

        assert asyncio.run(Supervisor(store, _explodes).run_once()) is True

    def test_the_lease_is_renewed_while_the_task_runs(self):
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(store, _slow, heartbeat_seconds=0.01)
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.08)
            slow_enough.set()
            await running

        asyncio.run(scenario())

        assert store.heartbeats, "the lease was never renewed"
        assert set(store.heartbeats) == {"a1b2c3d4"}

    def test_heartbeating_stops_when_the_task_does(self):
        store = FakeStore([_task()])

        async def scenario():
            supervisor = Supervisor(store, _succeeds, heartbeat_seconds=0.01)
            await supervisor.run_once()
            before = len(store.heartbeats)
            await asyncio.sleep(0.05)
            return before

        before = asyncio.run(scenario())

        assert len(store.heartbeats) == before

    def test_a_store_outage_does_not_end_a_running_task(self):
        """The work in flight is worth more than the ledger entry."""
        class FlakyHeartbeat(FakeStore):
            def heartbeat(self, task_id, lease_seconds=None):
                super().heartbeat(task_id, lease_seconds)
                raise StoreUnavailable("store is down")

        store = FlakyHeartbeat([_task()])

        async def scenario():
            done = asyncio.Event()

            async def _slow(task):
                await done.wait()
                return await _succeeds(task)

            supervisor = Supervisor(store, _slow, heartbeat_seconds=0.01)
            running = asyncio.create_task(supervisor.run_once())
            await asyncio.sleep(0.05)
            done.set()
            await running

        asyncio.run(scenario())

        assert store.results, "the task should still have completed"

    def test_run_forever_drains_the_queue_then_stops(self):
        store = FakeStore([_task(), _task()])

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_once_empty():
                while store.queue:
                    await asyncio.sleep(0.01)
                supervisor.stop()

            await asyncio.gather(supervisor.run_forever(), _stop_once_empty())

        asyncio.run(scenario())

        assert len(store.results) == 2

    def test_shutdown_wakes_the_poll_immediately(self):
        """A drain must not wait out a long empty-queue poll."""
        async def scenario():
            supervisor = Supervisor(FakeStore(), _succeeds, poll_seconds=30)

            async def _stop_soon():
                await asyncio.sleep(0.05)
                supervisor.stop()

            await asyncio.wait_for(
                asyncio.gather(supervisor.run_forever(), _stop_soon()), timeout=5
            )

        asyncio.run(scenario())

    def test_an_unreachable_store_backs_off_instead_of_crashing(self):
        class Dead(FakeStore):
            def claim(self, campaign_id=None, lease_seconds=None):
                self.claims += 1
                raise StoreUnavailable("connection refused")

        store = Dead()

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_after_a_few():
                while store.claims < 3:
                    await asyncio.sleep(0.01)
                supervisor.stop()

            await asyncio.wait_for(
                asyncio.gather(supervisor.run_forever(), _stop_after_a_few()),
                timeout=5,
            )

        asyncio.run(scenario())

        assert store.claims >= 3, "it should keep retrying while the store is down"

    def test_it_reports_itself_idle_before_claiming(self):
        """The pass that finds an empty queue is the one worth reporting.

        A supervisor otherwise touches the ledger only when it claims, so the
        conductor could not tell a free one from one that never started.
        """
        store = FakeStore()

        asyncio.run(Supervisor(store, _succeeds).run_once())

        assert store.presence == [("idle", None)]

    def test_it_reports_itself_busy_once_it_holds_a_task(self):
        store = FakeStore([_task()])

        asyncio.run(Supervisor(store, _succeeds).run_once())

        assert ("busy", "a1b2c3d4") in store.presence

    def test_presence_is_renewed_while_a_task_runs(self):
        """A task outlasts the presence TTL many times over."""
        store = FakeStore([_task()])

        async def scenario():
            slow_enough = asyncio.Event()

            async def _slow(task):
                await slow_enough.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, poll_seconds=0.01, presence_seconds=0.01
            )
            running = asyncio.create_task(supervisor.run_forever())
            await asyncio.sleep(0.1)
            slow_enough.set()
            supervisor.stop()
            await running

        asyncio.run(scenario())

        assert store.presence.count(("busy", "a1b2c3d4")) > 1

    def test_losing_a_lease_does_not_remove_it_from_the_pool(self):
        """Presence is about the supervisor; the lease is about the task.

        They were renewed on the same beat, so a supervisor whose task was
        taken -- by a cancel, or its campaign being finalized -- kept working
        on the job it had already started while vanishing from the pool. Two
        did exactly that: alive, holding a GPU, invisible.
        """
        class LeaseTaken(FakeStore):
            def heartbeat(self, task_id, lease_seconds=None):
                raise LeaseLost("held by someone else")

        store = LeaseTaken([_task()])

        async def scenario():
            still_working = asyncio.Event()

            async def _slow(task):
                await still_working.wait()
                return await _succeeds(task)

            supervisor = Supervisor(
                store, _slow, poll_seconds=0.01,
                heartbeat_seconds=0.01, presence_seconds=0.01,
            )
            running = asyncio.create_task(supervisor.run_forever())
            await asyncio.sleep(0.1)
            # The lease is gone by now; presence must have carried on.
            renewals = store.presence.count(("busy", "a1b2c3d4"))
            still_working.set()
            supervisor.stop()
            await running
            return renewals

        assert asyncio.run(scenario()) > 1

    def test_it_leaves_the_pool_on_shutdown(self):
        """Draining on command should not read as a crash for the next TTL."""
        store = FakeStore()

        async def scenario():
            supervisor = Supervisor(store, _succeeds, poll_seconds=0.01)

            async def _stop_soon():
                await asyncio.sleep(0.03)
                supervisor.stop()

            await asyncio.gather(supervisor.run_forever(), _stop_soon())

        asyncio.run(scenario())

        assert store.left_the_pool

    def test_a_store_that_cannot_take_presence_does_not_lose_the_task(self):
        """Being invisible is better than losing an hour of RL over it."""
        class NoPresence(FakeStore):
            def report_presence(self, *a, **k):
                raise StoreUnavailable("store is down")

        store = NoPresence([_task()])

        assert asyncio.run(Supervisor(store, _succeeds).run_once()) is True
        assert store.results

    def test_claims_can_be_scoped_to_one_campaign(self):
        seen: list[Any] = []

        class Recording(FakeStore):
            def claim(self, campaign_id=None, lease_seconds=None):
                seen.append(campaign_id)
                return None

        asyncio.run(
            Supervisor(Recording(), _succeeds, campaign_id="feed0001").run_once()
        )

        assert seen == ["feed0001"]


# ---------------------------------------------------------------------------
# TTP variants and deadlines
# ---------------------------------------------------------------------------


@pytest.fixture
def ttp_store(tmp_path, monkeypatch):
    """A TTP store in tmp_path, with one simple base TTP in it."""

    from reinvent4_mcp.config import ProcessConfig, reset_config
    from reinvent4_mcp.tools import ttp as ttp_tools

    config = ProcessConfig(ttp={"storage_dir": str(tmp_path / "ttps")})
    monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: config)
    monkeypatch.setattr("reinvent4_mcp.tools.ttp.get_config", lambda: config)
    ttp_tools.reset_store()

    ttp_tools.create_ttp({
        "id": "brd4-bd1-lead-opt",
        "name": "BRD4 BD1 lead optimisation",
        "pre_scoring": {"components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.5},
        ]},
        "scoring": {"objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore",
             "weight": 1.0},
        ]},
    })
    yield ttp_tools
    ttp_tools.reset_store()
    reset_config()


class TestPrepareTtpVariant:

    def test_the_variant_is_named_after_the_task(self, ttp_store):
        """So a result traces back to exactly the TTP that produced it."""
        variant_id = prepare_ttp_variant(
            "brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0}
        )

        assert variant_id == "brd4-bd1-lead-opt-a1b2c3d4"

    def test_the_variant_carries_the_overridden_weights(self, ttp_store):
        variant_id = prepare_ttp_variant(
            "brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0, "QED": 1.5}
        )
        stored = ttp_store.get_ttp(variant_id)

        assert stored["scoring"]["objectives"][0]["weight"] == 2.0
        assert stored["pre_scoring"]["components"][0]["weight"] == 1.5

    def test_the_base_ttp_is_left_alone(self, ttp_store):
        """Every supervisor reads the same base out of a shared store."""
        prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 9.0})
        base = ttp_store.get_ttp("brd4-bd1-lead-opt")

        assert base["scoring"]["objectives"][0]["weight"] == 1.0

    def test_the_variant_starts_at_version_one(self, ttp_store):
        """It is a new TTP, not a revision of the base's lineage."""
        variant_id = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4")

        assert ttp_store.get_ttp(variant_id)["version"] == 1

    def test_two_tasks_do_not_collide(self, ttp_store):
        """N supervisors write into one shared TTP store."""
        first = prepare_ttp_variant("brd4-bd1-lead-opt", "aaaaaaaa", {"SA Score": 1.0})
        second = prepare_ttp_variant("brd4-bd1-lead-opt", "bbbbbbbb", {"SA Score": 2.0})

        assert first != second
        assert ttp_store.get_ttp(second)["scoring"]["objectives"][0][
            "weight"
        ] == 2.0

    def test_a_retried_task_reuses_its_variant(self, ttp_store):
        """A requeued task must still be runnable.

        create_ttp refuses an id that exists, so the first live retry died
        with "TTP ... already exists" before it started -- meaning every
        retry was doomed. The content is a function of the task, so rewriting
        it changes nothing.
        """
        first = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0})
        second = prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"SA Score": 2.0})

        assert first == second
        stored = ttp_store.get_ttp(second)
        assert stored["scoring"]["objectives"][0]["weight"] == 2.0

    def test_a_missing_base_ttp_is_an_error(self, ttp_store):
        with pytest.raises(ValueError, match="nope"):
            prepare_ttp_variant("nope", "a1b2c3d4")

    def test_an_unknown_override_never_reaches_the_store(self, ttp_store):
        """Better to fail the task than to run the wrong weight vector."""
        with pytest.raises(ValueError, match="Affinity"):
            prepare_ttp_variant("brd4-bd1-lead-opt", "a1b2c3d4", {"Affinity": 2.0})

        assert "error" in ttp_store.get_ttp("brd4-bd1-lead-opt-a1b2c3d4")


class TestBaseTtpResolution:
    """Where the base TTP comes from.

    It belongs to the *campaign*, not to each task: a sweep varies weights
    against one fixed TTP. Reading it from the task spec instead left it empty
    and failed every task with "TTP '' not found" the first time a supervisor
    ran against a real ledger.
    """

    class _Store:
        def __init__(self, campaign: dict[str, Any]) -> None:
            self._campaign = campaign
            self.asked_for: list[str] = []

        def campaign(self, campaign_id: str) -> dict[str, Any]:
            self.asked_for.append(campaign_id)
            return self._campaign

    def test_it_comes_from_the_campaign(self):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        store = self._Store({"base_ttp_id": "brd4-bd1-lead-opt"})
        runner = CoscientistTaskRunner(object(), store)

        assert runner._base_ttp_id(_task()) == "brd4-bd1-lead-opt"
        assert store.asked_for == ["feed0001"]

    def test_a_campaign_without_one_is_a_clear_error(self):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        runner = CoscientistTaskRunner(object(), self._Store({}))

        with pytest.raises(RuntimeError, match="no base TTP"):
            runner._base_ttp_id(_task())


class TestJobDiscovery:
    """How the supervisor finds the job its Driver started.

    By observation, not by what the model said. The first live run failed a
    task with "the Driver finished without starting an optimization job"
    while that job was running fine -- the session had ended ``incomplete``,
    so there was no structured handoff to read the id out of.
    """

    @staticmethod
    def _jobs(*jobs: dict[str, Any]):
        manager = MagicMock()
        manager.list_jobs.return_value = list(jobs)
        return patch(
            "reinvent4_mcp.jobs.manager.get_job_manager", return_value=manager
        )

    def test_the_job_is_found_by_the_tasks_own_ttp(self):
        with self._jobs(
            {"job_id": "other", "ttp_id": "brd4-bd1-lead-opt-zzzz", "created_at": "1"},
            {"job_id": "mine", "ttp_id": "brd4-bd1-lead-opt-a1b2", "created_at": "2"},
        ):
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") == "mine"

    def test_no_job_for_the_ttp_is_reported_as_none(self):
        with self._jobs({"job_id": "other", "ttp_id": "something-else"}):
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") is None

    def test_an_empty_job_table_is_none(self):
        with self._jobs():
            assert job_for_ttp("brd4-bd1-lead-opt-a1b2") is None

    def test_a_retry_picks_the_newest_job(self):
        """A retried task can leave an older job of its own behind."""
        with self._jobs(
            {"job_id": "old", "ttp_id": "ttp-1", "created_at": "2026-10-08T10:00:00"},
            {"job_id": "new", "ttp_id": "ttp-1", "created_at": "2026-10-08T12:00:00"},
        ):
            assert job_for_ttp("ttp-1") == "new"

    def test_jobs_without_a_timestamp_do_not_break_the_choice(self):
        with self._jobs(
            {"job_id": "undated", "ttp_id": "ttp-1"},
            {"job_id": "dated", "ttp_id": "ttp-1", "created_at": "2026-10-08T12:00:00"},
        ):
            assert job_for_ttp("ttp-1") == "dated"


class TestSingleAgentMode:
    """REINVENT4_AGENT_MULTI_AGENT=false runs the Driver with no Reviewer.

    The Reviewer produces no molecules and roughly doubles the token spend,
    but turning it off is not purely a saving: ``monitor_campaign`` *is* the
    Reviewer cycle, so the supervisor needs its own way to wait for the job.
    """

    @staticmethod
    def _runner(monitor_interval: float = 0.01):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        args = MagicMock()
        args.monitor_interval = monitor_interval
        return CoscientistTaskRunner(args, MagicMock())

    def test_it_waits_for_the_job_to_stop_running(self):
        states = ["running", "running", "completed"]

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            side_effect=lambda job_id: {"state": states.pop(0)},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "it should have polled until the job finished"

    def test_a_queued_job_is_not_mistaken_for_a_finished_one(self):
        """A job is queued for a moment before it runs.

        Waiting only on "running" exited on the first poll, so the task
        harvested a 0-byte results CSV, failed with "No columns to parse from
        file", and left its job folding molecules as an orphan.
        """
        states = ["queued", "queued", "running", "completed"]

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            side_effect=lambda job_id: {"state": states.pop(0)},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "queued and running both mean the job owes a result"

    def test_the_wait_tracks_the_managers_own_active_states(self):
        """So a new job state cannot silently become 'finished' here."""
        from reinvent4_mcp.jobs.manager import ACTIVE_JOB_STATES

        assert {s.value for s in ACTIVE_JOB_STATES} == {"queued", "running"}

    def test_a_healthy_job_carries_a_null_error_and_is_still_waited_on(self):
        """The real status dict always has an "error" key, set to None.

        Testing for the key rather than its value was true for every job, so
        the wait returned on the first poll and harvested a 0-byte CSV while
        the job ran on. The earlier tests here passed a bare {"state": ...},
        which is why they never caught it -- so this one uses the shape the
        job manager actually produces.
        """
        states = ["running", "running", "completed"]

        def _status(job_id):
            return {
                "job_id": job_id,
                "state": states.pop(0),
                "error": None,
                "exit_code": None,
                "trajectory": {"current_step": 1},
            }

        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status", side_effect=_status
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

        assert states == [], "a null error must not end the wait"

    def test_a_status_with_no_state_is_not_waited_on(self):
        """A job the runner cannot describe at all."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"error": "No job job-1"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))


class TestFailedTaskReleasesItsJob:
    """A task that fails must not leave its RL job running.

    Nothing waits on the job once the task is failing, but it still holds a
    GPU and one of the runner's concurrency slots. Two abandoned jobs blocked
    every later task in a campaign with "Maximum concurrent jobs (2) already
    active".
    """

    @staticmethod
    def _runner():
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        return CoscientistTaskRunner(MagicMock(), MagicMock())

    def _release(self, runner, harvest, stop):
        return patch.object(
            type(runner), "_harvest", staticmethod(harvest)
        ), patch.object(type(runner), "_stop_job", staticmethod(stop))

    def test_a_good_harvest_leaves_the_job_alone(self):
        runner = self._runner()
        outcome = TaskOutcome(objective_profile={"QED": 0.8})
        stopped: list[str] = []

        harvest, stop = self._release(
            runner, lambda *a: outcome, lambda j: stopped.append(j)
        )
        with harvest, stop:
            got = asyncio.run(
                runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
            )

        assert got is outcome
        assert stopped == []

    def test_a_failed_harvest_stops_the_job(self):
        runner = self._runner()
        stopped: list[str] = []

        def _no_results(*args):
            raise RuntimeError("produced no readable results")

        harvest, stop = self._release(runner, _no_results, lambda j: stopped.append(j))
        with harvest, stop:
            with pytest.raises(RuntimeError, match="no readable results"):
                asyncio.run(
                    runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
                )

        assert stopped == ["job-1"]

    def test_a_stop_that_fails_does_not_mask_the_real_error(self):
        """The task's own failure is the one worth reporting."""
        runner = self._runner()

        def _no_results(*args):
            raise RuntimeError("produced no readable results")

        def _unreachable(job_id):
            raise RuntimeError("runner unreachable")

        harvest, stop = self._release(runner, _no_results, _unreachable)
        with harvest, stop:
            with pytest.raises(RuntimeError, match="no readable results"):
                asyncio.run(
                    runner._harvest_or_release(_task(), "job-1", "ok", 1, 0, False)
                )


class TestStartingTheRunDirectly:
    """What happens when the agent does not start the run itself.

    With a Reviewer the Coordinator took several turns, so a Driver that got
    distracted had more chances; a single pass gets one. Nothing about the
    call is the agent's decision -- the TTP and step count both come from the
    task -- so the supervisor starts it rather than asking again and hoping.
    """

    @staticmethod
    def _runner():
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        return CoscientistTaskRunner(MagicMock(), MagicMock())

    def test_it_starts_the_run_from_the_task(self):
        calls: list[dict[str, Any]] = []

        def _run(**kwargs):
            calls.append(kwargs)
            return {"job_id": "job-9", "error": None}

        with patch("reinvent4_mcp.tools.optimization.run_optimization", _run):
            assert self._runner()._start_job("ttp-1", 5) == "job-9"

        assert calls == [{"ttp_id": "ttp-1", "max_steps": 5}]

    def test_no_step_budget_is_left_to_the_default(self):
        calls: list[dict[str, Any]] = []

        def _run(**kwargs):
            calls.append(kwargs)
            return {"job_id": "job-9"}

        with patch("reinvent4_mcp.tools.optimization.run_optimization", _run):
            self._runner()._start_job("ttp-1", None)

        assert calls == [{"ttp_id": "ttp-1"}]

    def test_a_refused_run_is_an_error(self):
        """e.g. the concurrency cap, which names the jobs to stop."""
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"error": "Maximum concurrent jobs (2) reached"},
        ):
            with pytest.raises(RuntimeError, match="Maximum concurrent jobs"):
                self._runner()._start_job("ttp-1", 5)

    def test_a_null_error_is_not_a_failure(self):
        """Same trap as the status dict: the key is always there."""
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization",
            return_value={"job_id": "job-9", "error": None},
        ):
            assert self._runner()._start_job("ttp-1", 5) == "job-9"

    def test_a_response_without_a_job_id_is_an_error(self):
        with patch(
            "reinvent4_mcp.tools.optimization.run_optimization", return_value={}
        ):
            with pytest.raises(RuntimeError, match="no job id"):
                self._runner()._start_job("ttp-1", 5)

    def test_a_failed_job_also_ends_the_wait(self):
        """Any terminal state, not just success -- otherwise it waits forever."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"state": "failed"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

    def test_an_undescribable_job_is_not_waited_on(self):
        """A job the runner cannot report on is not one we can wait for."""
        with patch(
            "reinvent4_mcp.tools.results.get_optimization_status",
            return_value={"error": "No job job-1"},
        ):
            asyncio.run(self._runner()._watch_job("job-1"))

    def test_the_deadline_still_stops_the_job(self):
        """The single-agent watch is bounded by the task deadline, as before."""
        stopped: list[str] = []

        async def _never_finishes(job_id):
            await asyncio.sleep(10)

        runner = self._runner()
        with patch.object(runner, "_watch_job", _never_finishes), patch.object(
            type(runner), "_stop_job", staticmethod(lambda job_id: stopped.append(job_id))
        ):
            timed_out = asyncio.run(
                runner._monitor(None, "goal", "job-1", remaining=0.05)
            )

        assert timed_out is True
        assert stopped == ["job-1"]


class TestHarvestedProfileIsScopedToTheCampaign:
    """The profile reports the campaign's objectives, not every scored column.

    A results CSV carries a column per scorer diagnostic -- std,
    confidence_score, affinity_probability_binary and friends -- and reporting
    those as objectives put a dozen keys on every point of the front where
    three belonged, and bloated the stored results.
    """

    @staticmethod
    def _runner(campaign: dict[str, Any] | Exception):
        from reinvent4_agent.supervisor import CoscientistTaskRunner

        store = MagicMock()
        if isinstance(campaign, Exception):
            store.campaign.side_effect = campaign
        else:
            store.campaign.return_value = campaign
        return CoscientistTaskRunner(MagicMock(), store)

    def test_it_reads_the_objectives_off_the_campaign(self):
        runner = self._runner({"objectives": ["BoltzAffinity", "QED"]})

        assert runner._objectives(_task()) == ["BoltzAffinity", "QED"]

    def test_an_unreadable_campaign_falls_back_to_every_column(self):
        """Reporting nothing would fail the task; reporting extra only clutters."""
        runner = self._runner(RuntimeError("store down"))

        assert runner._objectives(_task()) is None

    def test_the_profile_drops_the_diagnostic_columns(self):
        candidates = [{
            "total_score": 0.9,
            "components": {
                "BoltzAffinity": {"transformed": 0.9},
                "QED": {"transformed": 0.8},
                "confidence_score (BoltzAffinity)": {"transformed": 0.87},
                "std (logD)": {"transformed": 0.18},
            },
        }]

        scoped = objective_profile(candidates, ["BoltzAffinity", "QED"])

        assert set(scoped) == {"BoltzAffinity", "QED"}


class TestDeadlines:

    def test_a_future_deadline_is_the_time_remaining(self):
        when = datetime.now(timezone.utc) + timedelta(seconds=600)
        remaining = _deadline_seconds(_task(deadline=when.isoformat()))

        assert 590 < remaining <= 600

    def test_a_past_deadline_is_zero_not_negative(self):
        """A negative timeout would raise rather than stop the task."""
        when = datetime.now(timezone.utc) - timedelta(seconds=60)

        assert _deadline_seconds(_task(deadline=when.isoformat())) == 0.0

    def test_no_deadline_means_no_limit(self):
        assert _deadline_seconds(_task()) is None

    def test_a_naive_deadline_is_read_as_utc(self):
        """Everything else in the ledger is UTC, so a bare timestamp is too."""
        when = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=300)
        remaining = _deadline_seconds(_task(deadline=when.isoformat()))

        assert 290 < remaining <= 300

    def test_an_unparseable_deadline_is_ignored_not_fatal(self):
        assert _deadline_seconds(_task(deadline="whenever")) is None


# ---------------------------------------------------------------------------
# Store client
# ---------------------------------------------------------------------------


class TestCampaignStoreClient:

    def test_204_means_nothing_to_claim(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request", lambda *a, **k: _response(204, None)
        )

        assert CampaignStoreClient(owner="s0").claim() is None

    def test_a_claim_returns_the_task(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request",
            lambda *a, **k: _response(200, {"task_id": "abc"}),
        )

        assert CampaignStoreClient(owner="s0").claim()["task_id"] == "abc"

    def test_409_is_a_lost_lease_not_an_outage(self, monkeypatch):
        """They are different problems: one means stop, the other means retry."""
        monkeypatch.setattr(
            requests, "request",
            lambda *a, **k: _response(409, {"detail": "held by s9"}),
        )

        with pytest.raises(LeaseLost, match="held by s9"):
            CampaignStoreClient(owner="s0").heartbeat("abc")

    def test_a_connection_error_is_an_outage(self, monkeypatch):
        def _refuse(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "request", _refuse)

        with pytest.raises(StoreUnavailable, match="unreachable"):
            CampaignStoreClient(owner="s0").claim()

    def test_a_server_error_is_an_outage(self, monkeypatch):
        monkeypatch.setattr(
            requests, "request", lambda *a, **k: _response(500, {"error": "boom"})
        )

        with pytest.raises(StoreUnavailable, match="500"):
            CampaignStoreClient(owner="s0").claim()

    def test_the_owner_is_sent_on_every_call(self, monkeypatch):
        sent: list[dict[str, Any]] = []

        def _capture(method, url, json=None, timeout=None):
            sent.append(json or {})
            return _response(200, {"task_id": "abc"})

        monkeypatch.setattr(requests, "request", _capture)
        client = CampaignStoreClient(owner="supervisor-3")
        client.claim()
        client.heartbeat("abc")
        client.post_result("abc", TaskOutcome(objective_profile={"QED": 0.5}))
        client.fail("abc", "boom")

        assert [body.get("owner") for body in sent] == ["supervisor-3"] * 4


class _response:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.ok = status_code < 400
        self.content = b"" if body is None else b"{}"
        self.text = str(body)

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body
