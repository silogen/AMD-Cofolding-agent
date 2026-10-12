"""Tests for services/campaign-store/app.py -- the fleet control-plane ledger.

The two things worth testing hard are the ones that cost real GPU time when
they are wrong: two supervisors must never be handed the same task, and a
supervisor that dies holding one must give it back.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_STORE = Path(__file__).resolve().parents[1] / "services" / "campaign-store" / "app.py"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store module bound to a per-test SQLite file.

    The module reads its config at import time, so it is imported per test
    rather than once, which also guarantees no state leaks between tests.
    """

    monkeypatch.setenv("CAMPAIGN_STORE_DB", str(tmp_path / "campaigns.db"))
    monkeypatch.setenv("TASK_MAX_ATTEMPTS", "2")

    spec = importlib.util.spec_from_file_location("campaign_store_under_test", _STORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.fixture
def client(store):
    with TestClient(store.app) as c:
        yield c


def _campaign(client, **overrides):
    body = {
        "goal": "Improve BRD4 BD1 affinity without losing drug-likeness",
        "base_ttp_id": "brd4-bd1-lead-opt",
        "objectives": ["BoltzAffinity", "QED"],
        "concurrency": 2,
    }
    body.update(overrides)
    response = client.post("/campaigns", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _task(client, campaign_id, **overrides):
    body = {"weight_overrides": {"BoltzAffinity": 1.0, "QED": 0.3}}
    body.update(overrides)
    response = client.post(f"/campaigns/{campaign_id}/tasks", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _claim(client, owner, **overrides):
    body = {"owner": owner}
    body.update(overrides)
    return client.post("/tasks/claim", json=body)


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


class TestCampaigns:

    def test_health_reports_the_db_and_open_work(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["active_campaigns"] == 0
        assert body["open_tasks"] == 0

    def test_create_and_read_back(self, client):
        created = _campaign(client)
        assert created["state"] == "dispatching"
        assert created["objectives"] == ["BoltzAffinity", "QED"]

        fetched = client.get(f"/campaigns/{created['campaign_id']}").json()
        assert fetched == created

    def test_unknown_campaign_is_404(self, client):
        assert client.get("/campaigns/nope").status_code == 404

    def test_objectives_are_required(self, client):
        """The front's dimensions are fixed at creation, so they cannot be empty."""
        response = client.post(
            "/campaigns",
            json={"goal": "g", "base_ttp_id": "t", "objectives": []},
        )
        assert response.status_code == 422

    def test_deadlines_derive_from_the_dispatch_and_grace_windows(self, client):
        campaign = _campaign(client, dispatch_seconds=600, grace_seconds=300)
        assert campaign["hard_stop_at"] - campaign["dispatch_until"] == pytest.approx(300)

    def test_no_dispatch_deadline_means_no_hard_stop(self, client):
        campaign = _campaign(client)
        assert campaign["dispatch_until"] is None
        assert campaign["hard_stop_at"] is None

    def test_closing_shuts_the_queue_but_not_the_campaign(self, client):
        campaign = _campaign(client)
        cid = campaign["campaign_id"]
        _task(client, cid)

        assert client.post(f"/campaigns/{cid}/close").json()["state"] == "draining"

        refused = client.post(f"/campaigns/{cid}/tasks", json={})
        assert refused.status_code == 409
        assert "closed" in refused.json()["detail"]

        # Already-queued work is still claimable during the drain window.
        assert _claim(client, "supervisor-0").status_code == 200

    def test_concurrency_can_be_changed(self, client):
        """Held in the ledger so it survives a conductor restart."""
        cid = _campaign(client)["campaign_id"]

        updated = client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": 5}
        ).json()

        assert updated["concurrency"] == 5
        assert client.get(f"/campaigns/{cid}").json()["concurrency"] == 5

    def test_concurrency_can_be_dropped_to_zero(self, client):
        """Pausing a campaign without closing it."""
        cid = _campaign(client)["campaign_id"]

        assert client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": 0}
        ).status_code == 200

    def test_a_negative_concurrency_is_refused(self, client):
        cid = _campaign(client)["campaign_id"]

        assert client.post(
            f"/campaigns/{cid}/concurrency", json={"concurrency": -1}
        ).status_code == 422

    def test_finalize_abandons_outstanding_work(self, client):
        """Finalizing must not leave tasks that nobody will ever run as queued."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        _task(client, cid)
        _claim(client, "supervisor-0")

        assert client.post(f"/campaigns/{cid}/finalize").json()["state"] == "finalized"

        states = {t["state"] for t in client.get(f"/campaigns/{cid}/tasks").json()}
        assert states == {"abandoned"}
        assert _claim(client, "supervisor-1").status_code == 204


# ---------------------------------------------------------------------------
# Claims and leases
# ---------------------------------------------------------------------------


class TestClaims:

    def test_claim_returns_204_when_idle(self, client):
        _campaign(client)
        assert _claim(client, "supervisor-0").status_code == 204

    def test_claim_hands_out_the_oldest_task_first(self, client):
        cid = _campaign(client)["campaign_id"]
        first = _task(client, cid, weight_overrides={"QED": 0.1})
        second = _task(client, cid, weight_overrides={"QED": 0.9})

        assert _claim(client, "s0").json()["task_id"] == first["task_id"]
        assert _claim(client, "s1").json()["task_id"] == second["task_id"]

    def test_claim_carries_the_spec_the_supervisor_needs(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(
            client, cid,
            weight_overrides={"BoltzAffinity": 2.0},
            fidelity_tier="validation",
            max_steps=120,
        )

        spec = _claim(client, "s0").json()["spec"]
        assert spec["weight_overrides"] == {"BoltzAffinity": 2.0}
        assert spec["fidelity_tier"] == "validation"
        assert spec["max_steps"] == 120

    def test_a_task_is_only_handed_out_once(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        assert _claim(client, "s0").status_code == 200
        assert _claim(client, "s1").status_code == 204

    def test_concurrent_claims_never_hand_out_the_same_task(self, store, client):
        """The race that costs real GPU time if the claim is not atomic.

        Without BEGIN IMMEDIATE around the select-then-update, two supervisors
        arriving together both read the task as queued and both run the same
        weight vector on separate GPUs.
        """
        cid = _campaign(client)["campaign_id"]
        for _ in range(8):
            _task(client, cid)

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(
                pool.map(
                    lambda owner: _claim(client, owner),
                    [f"supervisor-{i}" for i in range(8)],
                )
            )

        claimed = [r.json()["task_id"] for r in responses if r.status_code == 200]
        assert len(claimed) == 8, "every task should have been claimed"
        assert len(set(claimed)) == 8, f"a task was handed out twice: {claimed}"

    def test_claim_is_scoped_to_one_campaign_when_asked(self, client):
        other = _campaign(client, goal="other")["campaign_id"]
        mine = _campaign(client, goal="mine")["campaign_id"]
        _task(client, other)
        wanted = _task(client, mine)

        claimed = _claim(client, "s0", campaign_id=mine).json()
        assert claimed["task_id"] == wanted["task_id"]

    def test_heartbeat_extends_the_lease(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0", lease_seconds=5).json()

        extended = client.post(
            f"/tasks/{task['task_id']}/heartbeat",
            json={"owner": "s0", "lease_seconds": 600},
        ).json()

        assert extended["lease_expires_at"] > task["lease_expires_at"]

    def test_heartbeat_from_a_stranger_is_refused(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/heartbeat", json={"owner": "s1"}
        )
        assert response.status_code == 409
        assert "held by s0" in response.json()["detail"]

    def test_an_expired_lease_is_requeued_for_someone_else(self, client):
        """A supervisor that is OOM-killed has to give its task back."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "doomed", lease_seconds=0.01).json()
        time.sleep(0.05)

        reclaimed = _claim(client, "survivor")
        assert reclaimed.status_code == 200
        assert reclaimed.json()["task_id"] == task["task_id"]
        assert reclaimed.json()["lease_owner"] == "survivor"
        assert reclaimed.json()["attempts"] == 2

    def test_a_task_that_keeps_losing_its_lease_is_abandoned(self, client):
        """Stops a poison task cycling through every supervisor forever."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        for owner in ("s0", "s1"):
            assert _claim(client, owner, lease_seconds=0.01).status_code == 200
            time.sleep(0.05)

        assert _claim(client, "s2").status_code == 204
        task = client.get(f"/campaigns/{cid}/tasks").json()[0]
        assert task["state"] == "abandoned"
        assert "lease expired" in task["error"]

    def test_a_task_can_be_cancelled_out_from_under_its_holder(self, client):
        """The conductor overriding a supervisor, not a supervisor reporting.

        So this one is deliberately not owner-checked -- the holder finds out
        on its next heartbeat, which already treats a lost lease as "stop and
        do not post a result".
        """
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        cancelled = client.post(f"/tasks/{task['task_id']}/cancel").json()

        assert cancelled["state"] == "abandoned"
        assert cancelled["lease_owner"] is None
        assert "cancelled" in cancelled["error"]

        refused = client.post(
            f"/tasks/{task['task_id']}/heartbeat", json={"owner": "s0"}
        )
        assert refused.status_code == 409

    def test_a_cancelled_task_is_not_handed_out_again(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(f"/tasks/{task['task_id']}/cancel")

        assert _claim(client, "s1").status_code == 204

    def test_a_queued_task_can_be_cancelled_too(self, client):
        cid = _campaign(client)["campaign_id"]
        task = _task(client, cid)

        assert client.post(
            f"/tasks/{task['task_id']}/cancel"
        ).json()["state"] == "abandoned"

    def test_cancelling_a_finished_task_is_refused(self, client):
        """Its result is already on the front; abandoning it would lose that."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.8}},
        )

        response = client.post(f"/tasks/{task['task_id']}/cancel")

        assert response.status_code == 409
        assert "already finished" in response.json()["detail"]

    def test_cancelling_an_unknown_task_is_404(self, client):
        assert client.post("/tasks/nope/cancel").status_code == 404

    def test_posting_after_losing_the_lease_is_refused(self, client):
        """The new holder's result must not be overwritten by the old one's."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "slow", lease_seconds=0.01).json()
        time.sleep(0.05)
        _claim(client, "fast")

        response = client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "slow", "objective_profile": {"BoltzAffinity": 0.9, "QED": 0.8}},
        )
        assert response.status_code == 409


# ---------------------------------------------------------------------------
# Results and failures
# ---------------------------------------------------------------------------


class TestResults:

    def _done(self, client, owner, profile, **extra):
        task = _claim(client, owner).json()
        body = {"owner": owner, "objective_profile": profile}
        body.update(extra)
        response = client.post(f"/tasks/{task['task_id']}/result", json=body)
        assert response.status_code == 200, response.text
        return task["task_id"], response.json()

    def test_posting_a_result_closes_the_task(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        task_id, result = self._done(
            client, "s0", {"BoltzAffinity": 0.81, "QED": 0.74},
            job_id="b02764da", candidates=[{"smiles": "CCO", "score": 0.8}],
        )

        assert result["job_id"] == "b02764da"
        assert result["candidates"][0]["smiles"] == "CCO"

        task = client.get(f"/campaigns/{cid}/tasks").json()[0]
        assert task["state"] == "done"
        assert task["lease_owner"] is None
        assert client.get(f"/campaigns/{cid}/results").json()[0]["task_id"] == task_id

    def test_a_failure_is_requeued_by_default(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/fail",
            json={"owner": "s0", "error": "runner unreachable"},
        )
        assert response.json()["state"] == "queued"
        assert _claim(client, "s1").status_code == 200

    def test_a_failure_can_be_terminal(self, client):
        """Some failures are the task's fault and retrying just burns a slot."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()

        response = client.post(
            f"/tasks/{task['task_id']}/fail",
            json={"owner": "s0", "error": "weights invalid", "requeue": False},
        )
        assert response.json()["state"] == "failed"
        assert _claim(client, "s1").status_code == 204

    def test_a_failure_stops_being_requeued_at_the_attempt_limit(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)

        for owner in ("s0", "s1"):
            task = _claim(client, owner).json()
            state = client.post(
                f"/tasks/{task['task_id']}/fail",
                json={"owner": owner, "error": "boom"},
            ).json()["state"]

        assert state == "failed"


class TestResultsAreNarrowedForAContextWindow:
    """A result's caller is an LLM, and a scorer emits far more than the axes.

    Boltz reports ten per-molecule diagnostics beside the one objective they
    explain, which made a 25-molecule result ~22k characters. Two of those in
    a turn overflowed the conductor's 32768-token context: the front fix alone
    left the same question failing with "max_tokens ... got -2623".
    """

    # The column set a real Boltz + QED candidate carries, because the ratio
    # these tests assert on is a property of that shape.
    DIAGNOSTIC_CANDIDATE = {
        "smiles": "CCNc1cc(C(=O)NC(Cc2ccccc2)C(O)CNC2CCN3CCCCC3C2)cc(N2CCCCS2(=O)=O)c1",
        "total_score": 5.71,
        "components": {
            "BoltzAffinity": {"transformed": 1.0, "raw": 0.0092},
            "QED": {"transformed": 0.8, "raw": 0.8},
            "ligand_iptm (BoltzAffinity)": {"transformed": 0.9049},
            "confidence_score (BoltzAffinity)": {"transformed": 0.8692},
            "skip_reason (BoltzAffinity)": {"transformed": None},
            "affinity_pred_value (BoltzAffinity)": {"transformed": 0.0092},
            "affinity_pred_value1 (BoltzAffinity)": {"transformed": -0.8188},
            "affinity_pred_value2 (BoltzAffinity)": {"transformed": 0.8372},
            "affinity_probability_binary (BoltzAffinity)": {"transformed": 0.2148},
            "affinity_probability_binary1 (BoltzAffinity)": {"transformed": 0.1983},
            "affinity_probability_binary2 (BoltzAffinity)": {"transformed": 0.2313},
            "std (QED)": {"transformed": 0.1824},
        },
    }

    def _result(self, client, **extra):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        body = {
            "owner": "s0",
            "objective_profile": {
                "BoltzAffinity": 0.9, "QED": 0.8,
                "confidence_score (BoltzAffinity)": 0.87,
            },
            "candidates": [dict(self.DIAGNOSTIC_CANDIDATE)],
        }
        body.update(extra)
        client.post(f"/tasks/{task['task_id']}/result", json=body)
        return cid

    def test_the_redundant_half_of_a_scorers_columns_is_dropped(self, client):
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert "affinity_pred_value1 (BoltzAffinity)" not in candidate["components"]
        assert "std (QED)" not in candidate["components"]
        assert candidate["components"]["BoltzAffinity"]["raw"] == 0.0092
        assert candidate["smiles"] == self.DIAGNOSTIC_CANDIDATE["smiles"]
        assert candidate["total_score"] == 5.71

    def test_the_signals_that_qualify_an_objective_are_kept(self, client):
        """An affinity with no confidence beside it gets narrated, not read.

        Asked for the front's molecules with these dropped, the conductor
        asserted a quality gate it could not see, inventing PoseBusters
        numbers for a campaign that never ran PoseBusters.
        """
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"]["confidence_score (BoltzAffinity)"] == {
            "transformed": 0.8692
        }
        assert candidate["components"]["ligand_iptm (BoltzAffinity)"] == {
            "transformed": 0.9049
        }

    def test_a_qualifier_with_nothing_to_report_is_left_out(self, client):
        """``skip_reason`` is null for every molecule the scorer did score."""
        cid = self._result(client)

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert "skip_reason (BoltzAffinity)" not in candidate["components"]

    def test_a_skipped_molecule_says_so(self, client):
        """The one case where the objective's value means nothing."""
        cid = self._result(client, candidates=[{
            "smiles": "CCO",
            "components": {
                "BoltzAffinity": {"transformed": 0.0},
                "skip_reason (BoltzAffinity)": {"transformed": "fold failed"},
            },
        }])

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"]["skip_reason (BoltzAffinity)"] == {
            "transformed": "fold failed"
        }

    def test_the_objective_profile_is_narrowed_too(self, client):
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results").json()[0]

        assert set(result["objective_profile"]) == {
            "BoltzAffinity", "QED", "confidence_score (BoltzAffinity)"
        }

    def test_a_result_says_where_the_diagnostics_are(self, client):
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results").json()[0]

        assert "read_results" in result["detail_hint"]

    def test_full_returns_the_stored_record_untouched(self, client):
        """The ledger is not lossy; only its default read is narrow."""
        cid = self._result(client)

        result = client.get(f"/campaigns/{cid}/results?full=1").json()[0]

        assert result["objective_profile"]["confidence_score (BoltzAffinity)"] == 0.87
        assert (
            result["candidates"][0]["components"]
            == self.DIAGNOSTIC_CANDIDATE["components"]
        )
        assert "detail_hint" not in result

    def test_narrowing_survives_a_candidate_with_no_components(self, client):
        cid = self._result(client, candidates=[{"smiles": "CCO"}])

        candidate = client.get(f"/campaigns/{cid}/results").json()[0]["candidates"][0]

        assert candidate["components"] == {}
        assert candidate["total_score"] is None

    def test_narrowing_is_worth_doing(self, client):
        """The point of all of this, asserted rather than assumed."""
        cid = self._result(
            client, candidates=[dict(self.DIAGNOSTIC_CANDIDATE)] * 25
        )

        narrow = client.get(f"/campaigns/{cid}/results").text
        full = client.get(f"/campaigns/{cid}/results?full=1").text

        assert len(narrow) < len(full) / 2


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:

    def test_dominated_results_are_off_the_front(self, store):
        results = [
            {"task_id": "a", "objective_profile": {"x": 0.9, "y": 0.2}},
            {"task_id": "b", "objective_profile": {"x": 0.2, "y": 0.9}},
            {"task_id": "c", "objective_profile": {"x": 0.1, "y": 0.1}},
        ]
        front, ineligible = store.pareto_front(results, ["x", "y"])

        assert {r["task_id"] for r in front} == {"a", "b"}
        assert ineligible == []

    def test_a_result_better_everywhere_takes_the_front_alone(self, store):
        results = [
            {"task_id": "best", "objective_profile": {"x": 0.9, "y": 0.9}},
            {"task_id": "worse", "objective_profile": {"x": 0.8, "y": 0.8}},
        ]
        front, _ = store.pareto_front(results, ["x", "y"])
        assert [r["task_id"] for r in front] == ["best"]

    def test_identical_profiles_both_stay(self, store):
        """Neither dominates, so dropping one would be arbitrary."""
        results = [
            {"task_id": "a", "objective_profile": {"x": 0.5, "y": 0.5}},
            {"task_id": "b", "objective_profile": {"x": 0.5, "y": 0.5}},
        ]
        front, _ = store.pareto_front(results, ["x", "y"])
        assert len(front) == 2

    def test_a_missing_objective_is_reported_not_defaulted(self, store):
        """A missing objective means a scorer failed; scoring it 0 would hide that."""
        results = [
            {"task_id": "ok", "objective_profile": {"x": 0.5, "y": 0.5}},
            {"task_id": "partial", "objective_profile": {"x": 0.9}},
        ]
        front, ineligible = store.pareto_front(results, ["x", "y"])

        assert [r["task_id"] for r in front] == ["ok"]
        assert ineligible[0]["task_id"] == "partial"
        assert ineligible[0]["missing_objectives"] == ["y"]

    def test_a_null_objective_is_also_ineligible(self, store):
        """Services return null for a molecule they could not score."""
        results = [{"task_id": "nulled", "objective_profile": {"x": 0.5, "y": None}}]
        front, ineligible = store.pareto_front(results, ["x", "y"])

        assert front == []
        assert ineligible[0]["missing_objectives"] == ["y"]

    def test_the_front_reports_coordinates_not_molecules(self, client):
        """A front with the molecules inline does not fit in a context window.

        Three members with 25 candidates each came to ~42k tokens -- larger
        on its own than the conductor model's 32768 -- so asking about the
        front failed with "max_tokens must be at least 1, got -72257".
        """
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={
                "owner": "s0",
                "objective_profile": {"BoltzAffinity": 0.9, "QED": 0.8},
                "candidates": [{"smiles": "CCO", "score": 0.9}] * 25,
            },
        )

        member = client.get(f"/campaigns/{cid}/pareto").json()["front"][0]

        assert "candidates" not in member
        assert member["n_candidates"] == 25
        assert member["task_id"] == task["task_id"]

    def test_the_front_profile_is_only_the_campaigns_objectives(self, client):
        """A stored profile can carry scorer diagnostics; they are not axes."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={
                "owner": "s0",
                "objective_profile": {
                    "BoltzAffinity": 0.9, "QED": 0.8,
                    "confidence_score (BoltzAffinity)": 0.87,
                    "std (logD)": 0.18,
                },
            },
        )

        member = client.get(f"/campaigns/{cid}/pareto").json()["front"][0]

        assert set(member["objective_profile"]) == {"BoltzAffinity", "QED"}

    def test_the_front_says_where_the_molecules_are(self, client):
        cid = _campaign(client)["campaign_id"]

        body = client.get(f"/campaigns/{cid}/pareto").json()

        assert "get_supervisor_result" in body["molecules_hint"]

    def test_front_endpoint_uses_the_campaigns_objectives(self, client):
        cid = _campaign(client)["campaign_id"]
        profiles = [
            {"BoltzAffinity": 0.9, "QED": 0.2},
            {"BoltzAffinity": 0.2, "QED": 0.9},
            {"BoltzAffinity": 0.1, "QED": 0.1},
        ]
        for i, profile in enumerate(profiles):
            _task(client, cid)
            task = _claim(client, f"s{i}").json()
            client.post(
                f"/tasks/{task['task_id']}/result",
                json={"owner": f"s{i}", "objective_profile": profile},
            )

        body = client.get(f"/campaigns/{cid}/pareto").json()
        assert body["objectives"] == ["BoltzAffinity", "QED"]
        assert body["n_results"] == 3
        assert len(body["front"]) == 2


# ---------------------------------------------------------------------------
# Events and status
# ---------------------------------------------------------------------------


class TestEventsAndStatus:

    def test_events_record_the_task_lifecycle(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        task = _claim(client, "s0").json()
        client.post(
            f"/tasks/{task['task_id']}/result",
            json={"owner": "s0", "objective_profile": {"BoltzAffinity": 0.8, "QED": 0.7}},
        )

        kinds = [e["kind"] for e in client.get(f"/campaigns/{cid}/events").json()["events"]]
        assert kinds == ["campaign_created", "task_enqueued", "task_claimed", "task_done"]

    def test_events_can_be_polled_incrementally(self, client):
        """The conductor's completion feed: poll with the last id it saw."""
        cid = _campaign(client)["campaign_id"]
        first = client.get(f"/campaigns/{cid}/events").json()

        _task(client, cid)
        second = client.get(
            f"/campaigns/{cid}/events", params={"since": first["last_event_id"]}
        ).json()

        assert [e["kind"] for e in second["events"]] == ["task_enqueued"]
        assert second["last_event_id"] > first["last_event_id"]

    def test_polling_with_nothing_new_keeps_the_cursor(self, client):
        cid = _campaign(client)["campaign_id"]
        body = client.get(f"/campaigns/{cid}/events").json()
        again = client.get(
            f"/campaigns/{cid}/events", params={"since": body["last_event_id"]}
        ).json()

        assert again["events"] == []
        assert again["last_event_id"] == body["last_event_id"]

    def test_status_reports_counts_in_flight_work_and_the_front(self, client):
        cid = _campaign(client)["campaign_id"]
        _task(client, cid, weight_overrides={"QED": 0.5})
        _task(client, cid)
        running = _claim(client, "supervisor-0").json()

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["task_counts"] == {"claimed": 1, "queued": 1}
        assert len(status["in_flight"]) == 1
        assert status["in_flight"][0]["owner"] == "supervisor-0"
        assert status["in_flight"][0]["task_id"] == running["task_id"]
        assert status["in_flight"][0]["spec"]["weight_overrides"] == {"QED": 0.5}
        assert status["in_flight"][0]["lease_expires_in"] > 0

    def test_status_sweeps_dead_supervisors_out_of_the_in_flight_list(self, client):
        """In-flight should mean alive, not "claimed by something, once"."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid)
        _claim(client, "doomed", lease_seconds=0.01)
        time.sleep(0.05)

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["in_flight"] == []
        assert status["task_counts"] == {"queued": 1}

    def test_status_window_phase_follows_the_clock_not_the_state(self, client):
        """A conductor that missed its own deadline should still see the truth."""
        cid = _campaign(client, dispatch_seconds=0.01, grace_seconds=600)["campaign_id"]
        time.sleep(0.2)

        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["campaign"]["state"] == "dispatching"
        assert status["window"]["phase"] == "draining"
        assert status["window"]["dispatch_remaining"] < 0

    def test_status_window_phase_without_deadlines(self, client):
        cid = _campaign(client)["campaign_id"]
        status = client.get(f"/campaigns/{cid}/status").json()

        assert status["window"]["phase"] == "dispatching"
        assert status["window"]["dispatch_remaining"] is None


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


class TestDurability:

    def test_the_ledger_survives_a_restart(self, store, tmp_path):
        """The conductor rebuilds its loop from the ledger, so it has to persist."""
        with TestClient(store.app) as first:
            cid = _campaign(first)["campaign_id"]
            _task(first, cid)
            task = _claim(first, "s0").json()
            first.post(
                f"/tasks/{task['task_id']}/result",
                json={
                    "owner": "s0",
                    "objective_profile": {"BoltzAffinity": 0.8, "QED": 0.7},
                },
            )

        with TestClient(store.app) as second:
            assert second.get(f"/campaigns/{cid}").status_code == 200
            results = second.get(f"/campaigns/{cid}/results").json()
            assert results[0]["objective_profile"]["BoltzAffinity"] == 0.8

    def test_the_spec_round_trips_as_json_not_a_repr(self, store, client):
        """Stored specs are read back by other processes, so they must be JSON."""
        cid = _campaign(client)["campaign_id"]
        _task(client, cid, transform_overrides={"QED": {"type": "sigmoid", "k": 0.5}})

        with store._read() as conn:
            raw = conn.execute("SELECT spec FROM tasks").fetchone()["spec"]

        assert json.loads(raw)["transform_overrides"]["QED"]["k"] == 0.5
