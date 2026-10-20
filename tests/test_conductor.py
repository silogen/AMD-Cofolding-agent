"""Tests for reinvent4_agent/conductor.py -- the campaign loop.

``tick()`` is a pure function of a ledger read, so the whole loop is testable
against a fake store: the windows, the top-up arithmetic, the Pareto-guided
proposals, admission control, and the hard stop.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from unittest.mock import patch

import pytest
import requests

from reinvent4_agent.conductor import (
    CampaignConductor,
    ParetoProposer,
    PoolScaler,
    WeightVector,
    format_status,
    front_weights,
    live_campaign_id,
    main,
    objectives_of,
    queue_pressure,
    tried_vectors,
)

OBJECTIVES = ["BoltzAffinity", "QED"]


def _campaign(**overrides: Any) -> dict[str, Any]:
    campaign = {
        "campaign_id": "feed0001",
        "goal": "Improve BRD4 BD1 affinity without losing drug-likeness",
        "base_ttp_id": "brd4-bd1-lead-opt",
        "objectives": list(OBJECTIVES),
        "concurrency": 2,
        "state": "dispatching",
        "created_at": "2026-10-08T12:00:00+00:00",
        "dispatch_until": None,
        "hard_stop_at": None,
        "finalized_at": None,
    }
    campaign.update(overrides)
    return campaign


def _status(
    phase: str = "dispatching",
    counts: dict[str, int] | None = None,
    front: list[dict[str, Any]] | None = None,
    **campaign_overrides: Any,
) -> dict[str, Any]:
    return {
        "campaign": _campaign(**campaign_overrides),
        "task_counts": counts or {},
        "in_flight": [],
        "n_results": len(front or []),
        "front": front or [],
        "n_ineligible": 0,
        "window": {
            "dispatch_remaining": None,
            "hard_stop_remaining": None,
            "phase": phase,
        },
    }


class FakeStore:
    """A ledger that answers from a canned status."""

    def __init__(self, status: dict[str, Any], tasks: list[dict[str, Any]] | None = None):
        self.base_url = "http://store"
        self.owner = "conductor"
        self._status = status
        self._tasks = tasks or []
        self.enqueued: list[dict[str, Any]] = []
        self.closed = 0
        self.finalized = 0
        self.calls: list[str] = []

    def status(self, campaign_id):
        return self._status

    def campaign(self, campaign_id):
        return self._status["campaign"]

    def tasks(self, campaign_id, state=None):
        self.calls.append("tasks")
        return self._tasks

    def pareto(self, campaign_id):
        return {"front": self._status["front"], "ineligible": []}

    def enqueue_task(self, campaign_id, **spec):
        self.enqueued.append(spec)
        return {"task_id": f"t{len(self.enqueued)}", "spec": spec}

    def close_campaign(self, campaign_id):
        self.closed += 1
        self._status["campaign"]["state"] = "draining"
        return self._status["campaign"]

    def finalize_campaign(self, campaign_id):
        self.finalized += 1
        self.calls.append("finalize")
        self._status["campaign"]["state"] = "finalized"
        # The store abandons whatever was open, as the real one does.
        for task in self._tasks:
            if task["state"] in ("queued", "claimed"):
                task["state"] = "abandoned"
        return self._status["campaign"]


class FakeScaler:
    def __init__(self) -> None:
        self.scaled_to: list[int] = []

    def scale(self, replicas: int) -> bool:
        self.scaled_to.append(replicas)
        return True


def _conductor(store, **kwargs) -> CampaignConductor:
    kwargs.setdefault("stats_url", "http://gateway/unreachable")
    return CampaignConductor(store, "feed0001", **kwargs)


# ---------------------------------------------------------------------------
# Weight vectors
# ---------------------------------------------------------------------------


class TestWeightVector:

    def test_order_does_not_change_identity(self):
        """Two spellings of the same point must not be dispatched twice."""
        a = WeightVector.of({"QED": 1.0, "BoltzAffinity": 2.0})
        b = WeightVector.of({"BoltzAffinity": 2.0, "QED": 1.0})

        assert a == b
        assert len({a, b}) == 1

    def test_near_identical_weights_are_the_same_point(self):
        """Float noise must not make a 'new' vector worth a GPU hour."""
        a = WeightVector.of({"QED": 1.0})
        b = WeightVector.of({"QED": 1.00004})

        assert a == b

    def test_it_round_trips_to_a_dict(self):
        assert WeightVector.of({"QED": 1.5}).as_dict() == {"QED": 1.5}


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


class TestParetoProposer:

    def test_it_seeds_the_extremes_before_refining(self):
        """With no results there is nothing to learn from.

        Bounding the front's extremes early is what makes later refinement
        meaningful; refining around one lucky point first explores a corner.
        """
        proposals = ParetoProposer(OBJECTIVES).propose([], set(), 10)
        weights = [p.as_dict() for p in proposals]

        assert {"BoltzAffinity": 1.0, "QED": 1.0} in weights
        assert {"BoltzAffinity": 2.0, "QED": 1.0} in weights
        assert {"BoltzAffinity": 1.0, "QED": 2.0} in weights
        assert len(weights) == 3

    def test_it_respects_the_count(self):
        assert len(ParetoProposer(OBJECTIVES).propose([], set(), 2)) == 2

    def test_it_never_reproposes_a_tried_vector(self):
        """A campaign must not pay twice for the same point."""
        tried = {WeightVector.of({"BoltzAffinity": 1.0, "QED": 1.0})}

        proposals = ParetoProposer(OBJECTIVES).propose([], tried, 10)

        assert WeightVector.of({"BoltzAffinity": 1.0, "QED": 1.0}) not in proposals

    def test_an_exhausted_sweep_proposes_nothing(self):
        """Better to stop dispatching than to re-run paid-for points."""
        proposer = ParetoProposer(OBJECTIVES)
        everything = set(proposer.propose([], set(), 100))

        assert proposer.propose([], everything, 5) == []

    def test_it_refines_around_the_front(self):
        front = [{
            "task_id": "a",
            "objective_profile": {"BoltzAffinity": 0.9, "QED": 0.4},
            "weights": {"BoltzAffinity": 2.0, "QED": 1.0},
        }]
        seeds = set(ParetoProposer(OBJECTIVES).propose([], set(), 100))

        proposals = ParetoProposer(OBJECTIVES).propose(front, seeds, 4)
        weights = [p.as_dict() for p in proposals]

        assert proposals, "a front should produce neighbours to try"
        # Every proposal moves exactly one objective off the front member.
        for candidate in weights:
            differences = [
                name for name in OBJECTIVES
                if candidate[name] != front[0]["weights"][name]
            ]
            assert len(differences) == 1

    def test_refinement_keeps_weights_positive(self):
        """A zero weight would silently drop an objective from the score."""
        front = [{
            "task_id": "a",
            "objective_profile": {},
            "weights": {"BoltzAffinity": 0.05, "QED": 1.0},
        }]

        for proposal in ParetoProposer(OBJECTIVES).propose(front, set(), 50):
            assert all(weight > 0 for weight in proposal.as_dict().values())

    def test_a_campaign_needs_an_objective(self):
        with pytest.raises(ValueError, match="at least one objective"):
            ParetoProposer([])


class TestLedgerDerivedState:

    def test_tried_vectors_come_from_the_ledger(self):
        """So a restarted conductor does not re-dispatch the whole sweep."""
        tasks = [
            {"task_id": "a", "spec": {"weight_overrides": {"QED": 1.0}}},
            {"task_id": "b", "spec": {"weight_overrides": {"QED": 2.0}}},
        ]

        assert tried_vectors(tasks) == {
            WeightVector.of({"QED": 1.0}), WeightVector.of({"QED": 2.0})
        }

    def test_tasks_without_overrides_are_not_points(self):
        assert tried_vectors([{"task_id": "a", "spec": {}}]) == set()

    def test_front_members_are_paired_with_their_weights(self):
        """The store's front knows scores; refinement needs the weights."""
        front = [{"task_id": "a", "objective_profile": {"QED": 0.9}}]
        tasks = [{"task_id": "a", "spec": {"weight_overrides": {"QED": 2.0}}}]

        paired = front_weights(front, tasks)

        assert paired[0]["weights"] == {"QED": 2.0}
        assert paired[0]["objective_profile"] == {"QED": 0.9}

    def test_a_front_member_with_no_known_weights_is_dropped(self):
        """A hand-queued task with no overrides has no neighbourhood."""
        front = [{"task_id": "ghost", "objective_profile": {"QED": 0.9}}]

        assert front_weights(front, []) == []


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------


def _stats(healthy=2, queued=0, in_flight=0):
    """The shape boltz-gateway's /stats actually serves.

    Spelled out here because the earlier tests asserted against flat
    ``queue_depth``/``healthy_replicas`` keys that the gateway has never
    served. They passed while queue_pressure returned None in production
    against every real reply, so admission control had no reading at all.
    """

    return {
        "queue": {
            "gateway_molecules_waiting": 0,
            "gateway_in_flight_molecules": 0,
            "proxy_queued_requests": queued,
            "proxy_in_flight_requests": in_flight,
            "inflight_limit": healthy,
        },
        "replicas": {"healthy": healthy, "stats_reachable": True},
    }


class TestQueuePressure:

    def test_it_is_batches_in_hand_per_replica(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok(_stats(healthy=2, queued=4, in_flight=2)),
        )

        assert queue_pressure("http://gateway/stats") == 3.0

    def test_a_busy_scorer_keeping_up_is_one_per_replica(self, monkeypatch):
        """Every replica predicting with nothing queued is not saturation."""
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok(_stats(healthy=4, queued=0, in_flight=4)),
        )

        assert queue_pressure("http://gateway/stats") == 1.0

    def test_an_idle_scorer_is_zero(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: _ok(_stats(healthy=4)))

        assert queue_pressure("http://gateway/stats") == 0.0

    def test_in_flight_work_is_counted_not_only_queued(self, monkeypatch):
        """The gateway caps upstream calls at the replica count.

        Work beyond capacity waits on a semaphore inside the gateway rather
        than queueing at the proxy, so queued alone reads ~0 with every
        replica saturated.
        """
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok(_stats(healthy=1, queued=0, in_flight=3)),
        )

        assert queue_pressure("http://gateway/stats") == 3.0

    def test_an_unreachable_gateway_is_unknown_not_idle(self, monkeypatch):
        """Reading a silent gateway as empty would dispatch into a deep queue."""
        def _refuse(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _refuse)

        assert queue_pressure("http://gateway/stats") is None

    def test_no_replicas_is_unknown_not_division_by_zero(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get", lambda *a, **k: _ok(_stats(healthy=0, queued=4))
        )

        assert queue_pressure("http://gateway/stats") is None

    def test_a_gateway_without_a_queue_figure_is_unknown(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok({"replicas": {"healthy": 2}}),
        )

        assert queue_pressure("http://gateway/stats") is None

    def test_the_old_flat_shape_is_not_silently_accepted(self, monkeypatch):
        """It was never served; treating it as valid is what hid the bug."""
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok({"queue_depth": 6, "healthy_replicas": 2}),
        )

        assert queue_pressure("http://gateway/stats") is None


def _ok(body: dict[str, Any]):
    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return body

    return _Response()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestDispatchWindow:

    def test_it_tops_the_pool_up_to_the_concurrency(self):
        store = FakeStore(_status(counts={}))

        report = _conductor(store).tick()

        assert report.dispatched == 2, "concurrency is 2 and nothing is in flight"
        assert len(store.enqueued) == 2

    def test_queued_and_claimed_both_count_against_the_target(self):
        """A queued task is already committed work; only the shortfall is new."""
        store = FakeStore(_status(counts={"claimed": 1, "queued": 1}))

        assert _conductor(store).tick().dispatched == 0
        assert store.enqueued == []

    def test_a_partial_shortfall_dispatches_the_difference(self):
        store = FakeStore(_status(counts={"claimed": 1}))

        assert _conductor(store).tick().dispatched == 1

    def test_finished_tasks_do_not_count_as_in_flight(self):
        store = FakeStore(_status(counts={"done": 5, "failed": 2}))

        assert _conductor(store).tick().dispatched == 2

    def test_it_scales_the_pool_to_the_campaigns_concurrency(self):
        store = FakeStore(_status())
        scaler = FakeScaler()

        _conductor(store, scaler=scaler).tick()

        assert scaler.scaled_to == [2]

    def test_it_does_not_redispatch_what_the_ledger_already_has(self):
        """Idempotence: a conductor restart must not duplicate the sweep."""
        tried = [
            {"task_id": "a", "spec": {"weight_overrides": {
                "BoltzAffinity": 1.0, "QED": 1.0}}},
        ]
        store = FakeStore(_status(counts={"done": 1}), tasks=tried)

        _conductor(store).tick()

        for spec in store.enqueued:
            assert spec["weight_overrides"] != {"BoltzAffinity": 1.0, "QED": 1.0}

    def test_it_stops_dispatching_when_the_sweep_is_exhausted(self):
        proposer = ParetoProposer(OBJECTIVES)
        everything = [
            {"task_id": str(i), "spec": {"weight_overrides": v.as_dict()}}
            for i, v in enumerate(proposer.propose([], set(), 100))
        ]
        store = FakeStore(_status(counts={"done": len(everything)}), tasks=everything)

        assert _conductor(store).tick().dispatched == 0


class TestAdmissionControl:

    def test_a_deep_scorer_queue_holds_dispatch_back(self, monkeypatch):
        """Another supervisor would deepen the bottleneck, not relieve it."""
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok(_stats(healthy=2, queued=20)),
        )
        store = FakeStore(_status())

        report = _conductor(store, max_queue_pressure=2.0).tick()

        assert report.dispatched == 0
        assert "boltz queue" in report.held_back
        assert store.enqueued == []

    def test_a_shallow_queue_lets_work_through(self, monkeypatch):
        monkeypatch.setattr(
            requests, "get",
            lambda *a, **k: _ok(_stats(healthy=2, queued=1)),
        )
        store = FakeStore(_status())

        assert _conductor(store, max_queue_pressure=2.0).tick().dispatched == 2

    def test_an_unknown_queue_does_not_block_the_campaign(self, monkeypatch):
        """A gateway that never answers must not stall dispatch forever."""
        def _refuse(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", _refuse)
        store = FakeStore(_status())

        assert _conductor(store).tick().dispatched == 2


class TestDeadlines:

    def test_tasks_get_the_task_budget_as_a_deadline(self):
        store = FakeStore(_status())

        _conductor(store, task_budget_seconds=600).tick()

        deadline = datetime.fromisoformat(store.enqueued[0]["deadline"])
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        assert 590 < remaining <= 600

    def test_the_campaign_hard_stop_caps_the_task_budget(self):
        """No supervisor should start a run that outlives the campaign."""
        store = FakeStore(_status(hard_stop_at=time.time() + 120))

        _conductor(store, task_budget_seconds=36000).tick()

        deadline = datetime.fromisoformat(store.enqueued[0]["deadline"])
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        assert remaining <= 120

    def test_no_budget_and_no_hard_stop_means_no_deadline(self):
        store = FakeStore(_status())

        _conductor(store).tick()

        assert "deadline" not in store.enqueued[0]

    def test_the_step_budget_is_passed_through(self):
        store = FakeStore(_status())

        _conductor(store, max_steps=120).tick()

        assert store.enqueued[0]["max_steps"] == 120


class TestWindows:

    def test_draining_closes_the_queue_once(self):
        store = FakeStore(_status(phase="draining"))
        conductor = _conductor(store)

        first = conductor.tick()
        second = conductor.tick()

        assert first.closed is True
        assert second.closed is False, "closing twice would be a pointless write"
        assert store.closed == 1

    def test_draining_dispatches_nothing(self):
        store = FakeStore(_status(phase="draining", counts={}))

        assert _conductor(store).tick().dispatched == 0
        assert store.enqueued == []

    def test_the_hard_stop_finalizes_and_sends_the_pool_home(self):
        store = FakeStore(_status(phase="past_hard_stop"))
        scaler = FakeScaler()

        report = _conductor(store, scaler=scaler).tick()

        assert report.finalized is True
        assert store.finalized == 1
        assert scaler.scaled_to == [0]

    def test_a_finalized_campaign_is_left_alone(self):
        store = FakeStore(_status(phase="finalized", state="finalized"))
        scaler = FakeScaler()

        report = _conductor(store, scaler=scaler).tick()

        assert report.finalized is False
        assert store.finalized == 0
        assert scaler.scaled_to == []
        assert store.enqueued == []

    def test_the_hard_stop_stops_the_jobs_nobody_is_waiting_for(self):
        """Abandoning a task does not stop its RL job.

        The job lives in the runner and holds a GPU until something tells it
        to stop. The first live wind-down left two running for this reason.
        """
        tasks = [
            {"task_id": "aaa", "state": "claimed", "spec": {}},
            {"task_id": "bbb", "state": "queued", "spec": {}},
            {"task_id": "ccc", "state": "done", "spec": {}},
        ]
        store = FakeStore(_status(phase="past_hard_stop"), tasks=tasks)
        stopped: list[str] = []

        def _job_for(ttp_id):
            # Only the two open tasks ever started a job.
            return {"brd4-bd1-lead-opt-aaa": "job-a",
                    "brd4-bd1-lead-opt-bbb": "job-b"}.get(ttp_id)

        with patch("reinvent4_agent.supervisor.job_for_ttp", _job_for), patch(
            "reinvent4_mcp.tools.optimization.stop_optimization",
            lambda job_id: stopped.append(job_id) or {"message": "stopped"},
        ):
            _conductor(store).tick()

        assert sorted(stopped) == ["job-a", "job-b"]

    def test_a_finished_task_is_not_chased(self):
        """It already reported; its job is long gone."""
        store = FakeStore(
            _status(phase="past_hard_stop"),
            tasks=[{"task_id": "ccc", "state": "done", "spec": {}}],
        )
        stopped: list[str] = []

        with patch("reinvent4_agent.supervisor.job_for_ttp", lambda t: "job-c"), patch(
            "reinvent4_mcp.tools.optimization.stop_optimization",
            lambda job_id: stopped.append(job_id) or {},
        ):
            _conductor(store).tick()

        assert stopped == []

    def test_the_books_are_closed_before_the_jobs_are_stopped(self):
        """So a supervisor cannot post against a campaign that is over.

        A result worth keeping comes from the task deadline, which fires
        earlier and stops the job *with* a checkpoint.
        """
        store = FakeStore(
            _status(phase="past_hard_stop"),
            tasks=[{"task_id": "aaa", "state": "claimed", "spec": {}}],
        )

        with patch("reinvent4_agent.supervisor.job_for_ttp", lambda t: "job-a"), patch(
            "reinvent4_mcp.tools.optimization.stop_optimization",
            lambda job_id: store.calls.append("stop") or {},
        ):
            _conductor(store).tick()

        assert store.calls == ["tasks", "finalize", "stop"]

    def test_a_task_with_no_job_is_skipped_quietly(self):
        store = FakeStore(
            _status(phase="past_hard_stop"),
            tasks=[{"task_id": "aaa", "state": "queued", "spec": {}}],
        )

        with patch("reinvent4_agent.supervisor.job_for_ttp", lambda t: None):
            report = _conductor(store).tick()

        assert report.finalized is True

    def test_a_failed_job_lookup_does_not_block_the_wind_down(self):
        """Freeing the GPUs is best-effort; closing the campaign is not."""
        store = FakeStore(
            _status(phase="past_hard_stop"),
            tasks=[{"task_id": "aaa", "state": "claimed", "spec": {}}],
        )
        scaler = FakeScaler()

        def _explodes(ttp_id):
            raise RuntimeError("runner unreachable")

        with patch("reinvent4_agent.supervisor.job_for_ttp", _explodes):
            report = _conductor(store, scaler=scaler).tick()

        assert report.finalized is True
        assert store.finalized == 1
        assert scaler.scaled_to == [0]

    def test_run_stops_at_the_hard_stop(self):
        store = FakeStore(_status(phase="past_hard_stop"))

        report = _conductor(store, tick_seconds=0).run(max_ticks=5)

        assert report.finalized is True
        assert store.finalized == 1, "it should not keep finalizing"

    def test_run_gives_up_after_max_ticks(self):
        store = FakeStore(_status(counts={"claimed": 2}))

        report = _conductor(store, tick_seconds=0).run(max_ticks=3)

        assert report.finalized is False


class TestCampaignResolution:
    """Naming the campaign is only informative when several are live."""

    class _Store:
        def __init__(self, campaigns: list[dict[str, Any]]) -> None:
            self._campaigns = campaigns

        def campaigns(self) -> list[dict[str, Any]]:
            return self._campaigns

    def test_the_only_live_campaign_is_used(self):
        store = self._Store([
            _campaign(campaign_id="live", state="dispatching"),
            _campaign(campaign_id="old", state="finalized"),
        ])

        assert live_campaign_id(store) == "live"

    def test_a_draining_campaign_still_counts_as_live(self):
        """It has in-flight work and a hard stop still to enforce."""
        store = self._Store([_campaign(campaign_id="draining", state="draining")])

        assert live_campaign_id(store) == "draining"

    def test_no_live_campaign_says_how_to_make_one(self):
        store = self._Store([_campaign(state="finalized")])

        with pytest.raises(SystemExit, match="--start"):
            live_campaign_id(store)

    def test_it_refuses_to_guess_between_two(self):
        """Conducting the wrong campaign would spend GPU on the wrong goal."""
        store = self._Store([
            _campaign(campaign_id="aaa", goal="affinity first"),
            _campaign(campaign_id="bbb", goal="selectivity first"),
        ])

        with pytest.raises(SystemExit) as exc:
            live_campaign_id(store)

        assert "aaa" in str(exc.value) and "bbb" in str(exc.value)
        assert "affinity first" in str(exc.value), "name them so one can be picked"


class TestObjectiveDefaults:
    """Where a new campaign's objectives come from when not given."""

    @staticmethod
    def _ttp(ttp: dict[str, Any]):
        return patch("reinvent4_mcp.tools.ttp.get_ttp", lambda _id: ttp)

    def test_it_reads_them_off_the_base_ttp(self):
        """Typing them by hand is the easiest way to get a campaign wrong.

        An objective the TTP does not have fails every task referencing it,
        and only surfaces minutes later as a task failure rather than a typo.
        """
        with self._ttp({"scoring": {"objectives": [
            {"name": "BoltzAffinity"}, {"name": "logD"},
        ]}}):
            assert objectives_of("ttp-1") == ["BoltzAffinity", "logD"]

    def test_pre_scoring_components_are_objectives_too(self):
        with self._ttp({
            "pre_scoring": {"components": [{"name": "QED"}]},
            "scoring": {"objectives": [{"name": "BoltzAffinity"}]},
        }):
            assert objectives_of("ttp-1") == ["BoltzAffinity", "QED"]

    def test_a_name_in_both_sections_appears_once(self):
        with self._ttp({
            "pre_scoring": {"components": [{"name": "QED"}]},
            "scoring": {"objectives": [{"name": "QED"}]},
        }):
            assert objectives_of("ttp-1") == ["QED"]

    def test_a_missing_ttp_is_a_clear_exit(self):
        with self._ttp({"error": "TTP 'nope' not found"}):
            with pytest.raises(SystemExit, match="not found"):
                objectives_of("nope")

    def test_a_ttp_with_no_objectives_is_refused(self):
        with self._ttp({"scoring": {"objectives": []}}):
            with pytest.raises(SystemExit, match="no objectives"):
                objectives_of("ttp-1")


class TestEntrypointGuards:
    """Argument combinations that would mean two different things."""

    def test_start_needs_a_goal_and_a_base_ttp(self):
        with pytest.raises(SystemExit):
            main(["--start"])

    def test_start_and_a_campaign_id_are_contradictory(self):
        with pytest.raises(SystemExit):
            main(["--start", "--goal", "g", "--base-ttp", "t", "some-id"])

    def test_opening_flags_without_start_are_refused(self):
        # Quietly ignoring them would look like the campaign was configured.
        with pytest.raises(SystemExit):
            main(["some-id", "--goal", "g"])


class TestStatusView:

    def test_it_renders_the_things_an_operator_asks_about(self):
        status = _status(
            counts={"claimed": 1, "done": 2},
            front=[{"task_id": "a", "objective_profile": {"QED": 0.812}}],
        )
        status["in_flight"] = [{
            "task_id": "b", "owner": "dd-agentic-supervisor-xyz",
            "claimed_at": "2026-10-08T12:00:00+00:00",
            "lease_expires_in": 1800.0,
            "spec": {"weight_overrides": {"QED": 2.0}},
        }]
        status["window"]["dispatch_remaining"] = 7200.0
        status["window"]["hard_stop_remaining"] = 10800.0

        rendered = format_status(status)

        assert "feed0001" in rendered
        assert "claimed=1, done=2" in rendered
        assert "dd-agentic-supervisor-xyz" in rendered
        assert "QED=0.812" in rendered
        assert "2.0h" in rendered

    def test_an_empty_campaign_reads_sensibly(self):
        rendered = format_status(_status())

        assert "none yet" in rendered
        assert "in flight: none" in rendered

    def test_the_pool_is_shown_when_it_is_known(self):
        """Two in flight means something different with an idle one beside it."""
        rendered = format_status(_status(), {
            "n_alive": 3, "idle": 1, "busy": 2,
            "supervisors": [
                {"owner": "sup-a", "state": "idle", "task_id": None,
                 "last_seen_seconds_ago": 4.0},
                {"owner": "sup-b", "state": "busy", "task_id": "t1",
                 "last_seen_seconds_ago": 12.0},
            ],
        })

        assert "1 idle, 2 busy of 3 alive" in rendered
        assert "sup-b  busy on t1" in rendered

    def test_the_pool_is_simply_absent_when_unknown(self):
        """A store that did not answer should not print a pool of zero."""
        rendered = format_status(_status())

        assert "supervisors:" not in rendered

    def test_results_off_the_front_are_called_out(self):
        """A missing objective usually means a failed scorer, not a bad run."""
        status = _status()
        status["n_ineligible"] = 2

        assert "failed scorer" in format_status(status)


# ---------------------------------------------------------------------------
# Pool scaling
# ---------------------------------------------------------------------------


class TestPoolScaler:

    def test_outside_a_cluster_it_is_a_noop(self, tmp_path):
        """A laptop run should exercise the loop, not fail on the one cluster bit."""
        scaler = PoolScaler("dd-agentic-supervisor", "demo", tmp_path)

        assert scaler.in_cluster is False
        assert scaler.scale(4) is False
        assert scaler.current() is None

    def test_the_namespace_comes_from_the_service_account(self, tmp_path):
        (tmp_path / "namespace").write_text("demo-lux\n")

        assert PoolScaler("d", None, tmp_path).namespace == "demo-lux"

    def test_it_patches_only_the_replica_count(self, tmp_path, monkeypatch):
        _service_account(tmp_path)
        calls: list[dict[str, Any]] = []

        def _record(method, url, json=None, headers=None, verify=None, timeout=None):
            calls.append({"method": method, "url": url, "json": json,
                          "headers": headers})
            return _ok({"spec": {"replicas": 0}})

        monkeypatch.setattr(requests, "request", _record)
        scaler = PoolScaler("dd-agentic-supervisor", "demo-lux", tmp_path)

        assert scaler.scale(4) is True

        patch = calls[-1]
        assert patch["method"] == "PATCH"
        assert patch["json"] == {"spec": {"replicas": 4}}
        assert patch["headers"]["Content-Type"] == "application/merge-patch+json"
        assert patch["url"].endswith(
            "/apis/apps/v1/namespaces/demo-lux/deployments/"
            "dd-agentic-supervisor/scale"
        )
        assert patch["headers"]["Authorization"] == "Bearer sa-token"

    def test_scaling_to_the_current_count_writes_nothing(self, tmp_path, monkeypatch):
        _service_account(tmp_path)
        methods: list[str] = []

        def _record(method, url, **kwargs):
            methods.append(method)
            return _ok({"spec": {"replicas": 3}})

        monkeypatch.setattr(requests, "request", _record)

        assert PoolScaler("d", "demo", tmp_path).scale(3) is True
        assert "PATCH" not in methods

    def test_a_refused_scale_is_reported_not_raised(self, tmp_path, monkeypatch):
        """Most likely the ServiceAccount lacks deployments/scale."""
        _service_account(tmp_path)

        def _forbid(method, url, **kwargs):
            if method == "GET":
                return _ok({"spec": {"replicas": 0}})
            raise requests.HTTPError("403 Forbidden")

        monkeypatch.setattr(requests, "request", _forbid)

        assert PoolScaler("d", "demo", tmp_path).scale(4) is False


def _service_account(path) -> None:
    (path / "token").write_text("sa-token")
    (path / "ca.crt").write_text("ca")
    (path / "namespace").write_text("demo-lux")
