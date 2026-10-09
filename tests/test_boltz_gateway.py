"""Tests for the Boltz gateway: result cache, batch coalescing, async polling.

The gateway sits between ``ExternalRESTScorer`` and the Boltz replicas and owns
the things a proxy cannot do, because they need to understand the payload:

- a **per-molecule** cache keyed on canonical SMILES, so a batch of 16 with 10
  repeats costs 6 predictions. Canonicalisation is load-bearing: the RL config
  sets ``randomize_smiles``, so the same molecule arrives spelled differently on
  every step
- **coalescing** concurrent requests that share a target and prediction params
  into one upstream call, which is the strongest lever available because at
  least 124 s of the 386 s batch is fixed per-invocation cost
- **in-flight de-duplication**, so two supervisors asking for the same molecule
  at the same moment cost one prediction
- ``202`` + ``GET /result/{task_id}``, so a queued batch is not a failed one

The upstream Boltz call is stubbed throughout, so no GPU, cluster, or service
is needed. The cache is pointed at a tmp_path.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "boltz_gateway_app", str(_ROOT / "services" / "boltz-gateway" / "app.py")
)
gateway = importlib.util.module_from_spec(_spec)
sys.modules["boltz_gateway_app"] = gateway
_spec.loader.exec_module(gateway)

from fastapi.testclient import TestClient  # noqa: E402  (after the module loads)

SEQUENCE = "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYY"
# Two spellings of ethanol and one of benzene: the first two must share a cache
# entry, which is the whole point of canonicalising.
ETHANOL = "CCO"
ETHANOL_ALT = "OCC"
BENZENE = "c1ccccc1"


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Fresh cache dir and counters per test, with no real upstream."""
    monkeypatch.setattr(gateway, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(gateway, "CACHE_ENABLED", True)
    monkeypatch.setattr(gateway, "COALESCE_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(gateway, "COALESCE_MAX_MOLECULES", 16)
    monkeypatch.setattr(gateway, "counters", gateway.Counters())
    monkeypatch.setattr(gateway, "batcher", gateway.Batcher())
    monkeypatch.setattr(gateway, "replicas", gateway.ReplicaView())
    monkeypatch.setattr(gateway, "tasks", {})
    # Refused immediately, so the background replica refresher neither reaches
    # the network nor delays shutdown on a DNS timeout.
    monkeypatch.setattr(gateway, "PROXY_STATS_URL", "http://127.0.0.1:1/stats;csv")
    gateway._canonical_cache.clear()
    gateway._msa_digest_cache.clear()
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)


class _Upstream:
    """Records the batches it is asked to predict and answers deterministically.

    Structures are written for real: the gateway drops a cache entry whose
    structure file has gone, so fake paths would make every hit look like a
    miss.
    """

    def __init__(self, structure_dir: Path, structures: bool = True):
        self.calls: list[list[str]] = []
        self.structure_dir = structure_dir
        self.structures = structures
        self.fail = False
        # Held open so a test can keep a batch in flight while another request
        # arrives, which is how coalescing and de-duplication are observed.
        self.release = threading.Event()
        self.release.set()

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        smiles = payload["smiles"]
        self.calls.append(list(smiles))
        self.release.wait(timeout=10)
        if self.fail:
            raise RuntimeError("boltz exploded")

        structures: list[Optional[str]] = []
        for one in smiles:
            if not self.structures:
                structures.append(None)
                continue
            path = self.structure_dir / f"{abs(hash(one))}.pdb"
            path.write_text("ATOM")
            structures.append(str(path))
        return {
            # A stable, recognisable score per molecule.
            "scores": [float(len(s)) for s in smiles],
            "metadata": {"affinity_pred_value": [float(len(s)) for s in smiles]},
            "artifacts": {"structures": structures},
        }

    @property
    def molecules(self) -> int:
        return sum(len(call) for call in self.calls)


@pytest.fixture
def upstream(tmp_path, monkeypatch) -> _Upstream:
    structure_dir = tmp_path / "predictions"
    structure_dir.mkdir(parents=True, exist_ok=True)
    stub = _Upstream(structure_dir)
    monkeypatch.setattr(gateway, "_post_upstream_blocking", stub)
    return stub


@pytest.fixture
def client(upstream):
    # A context manager, so every request shares one event loop. Without it
    # TestClient spins up a fresh loop per request and the batcher's futures,
    # locks and semaphores end up bound to different loops.
    with TestClient(gateway.app) as test_client:
        yield test_client


def _params(**overrides: Any) -> dict[str, Any]:
    params = {"protein_sequence": SEQUENCE, "recycling_steps": 3}
    params.update(overrides)
    return params


def _score(client: TestClient, smiles: list[str], **params: Any):
    return client.post("/score", json={"smiles": smiles, "params": _params(**params)})


def _poll(client: TestClient, task_id: str, timeout: float = 15.0):
    """Poll /result until it stops saying 202, the way a real caller would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/result/{task_id}")
        if response.status_code != 202:
            return response
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never finished within {timeout}s")


# ---------------------------------------------------------------------------
# TestCacheKeying
# ---------------------------------------------------------------------------


class TestReplicaParallelism:
    """Batches for one key must occupy every replica, not queue behind each other.

    There is one dispatch loop per coalescing key, and a fleet sweep puts
    every supervisor on the same target with the same predict params -- so
    all of its traffic shares one key. Awaiting each upstream call inside the
    loop served those batches strictly one at a time and left every replica
    but one idle, whatever the semaphore allowed. Measured live as
    upstream_p50 393 s against request_p50 752 s, with 16 molecules in flight
    and 16 waiting on two healthy replicas.
    """

    @staticmethod
    def _concurrency_probe(limit: int):
        """An upstream that reports the most calls it ever saw at once."""

        state = {"now": 0, "peak": 0}
        started = asyncio.Event()

        async def _call(key, params, batch):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            if state["now"] >= limit:
                started.set()
            try:
                # Long enough that a serialising loop cannot overlap by luck.
                await asyncio.sleep(0.2)
            finally:
                state["now"] -= 1

        return _call, state, started

    def _peak_concurrency(self, monkeypatch, batches: int, replicas: int) -> int:
        """Most batches ever upstream at once, given `batches` waiting."""

        batcher = gateway.Batcher()
        run, state, _ = self._concurrency_probe(replicas)
        monkeypatch.setattr(batcher, "_run_batch", run)

        async def scenario():
            # One semaphore for the whole run. Production caches it in
            # `_slots`; handing back a new one per call would bound nothing.
            slots = asyncio.Semaphore(replicas)
            monkeypatch.setattr(batcher, "_semaphore", lambda: slots)

            params = gateway.ScoreParams(protein_sequence="MVL")
            key = gateway.coalesce_key(params)
            async with batcher._lock:
                batcher._params[key] = params
                batcher._pending[key] = [
                    gateway.Waiter(
                        digest=f"d{i}",
                        canonical=f"C{i}",
                        future=asyncio.get_running_loop().create_future(),
                    )
                    for i in range(gateway.COALESCE_MAX_MOLECULES * batches)
                ]
                batcher._first_enqueued[key] = 0.0

            loop = asyncio.create_task(batcher._dispatch_loop(key))
            await asyncio.sleep(0.1)
            loop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop
            # Let the spawned calls finish so nothing is left pending.
            await asyncio.gather(*batcher._running, return_exceptions=True)

        asyncio.run(scenario())
        return state["peak"]

    def test_two_batches_for_one_key_run_at_once(self, monkeypatch):
        peak = self._peak_concurrency(monkeypatch, batches=2, replicas=2)

        assert peak == 2, (
            f"only {peak} batch(es) ran at once; the second replica was idle"
        )

    def test_concurrency_is_still_capped_by_the_replica_count(self, monkeypatch):
        """More waiting batches than replicas must not oversubscribe the GPUs."""
        peak = self._peak_concurrency(monkeypatch, batches=5, replicas=2)

        assert peak <= 2, f"{peak} batches ran at once against 2 replicas"

    def test_one_replica_still_means_one_call(self, monkeypatch):
        """The bound is the replica count, not a fixed number."""
        peak = self._peak_concurrency(monkeypatch, batches=4, replicas=1)

        assert peak == 1, f"{peak} batches ran at once against 1 replica"


class TestCacheKeying:

    def test_the_same_molecule_spelled_differently_shares_a_key(self):
        params = gateway.ScoreParams(protein_sequence=SEQUENCE)

        first = gateway.molecule_key(gateway.canonical_smiles(ETHANOL), params)
        second = gateway.molecule_key(gateway.canonical_smiles(ETHANOL_ALT), params)

        # randomize_smiles means this is the common case, not an edge case.
        assert first == second

    def test_different_molecules_do_not(self):
        params = gateway.ScoreParams(protein_sequence=SEQUENCE)

        assert gateway.molecule_key(ETHANOL, params) != gateway.molecule_key(
            BENZENE, params
        )

    def test_an_unreadable_smiles_has_no_canonical_form(self):
        assert gateway.canonical_smiles("not-a-molecule") is None

    def test_the_target_is_part_of_the_key(self):
        # The same ligand against another protein is a different prediction.
        here = gateway.ScoreParams(protein_sequence=SEQUENCE)
        there = gateway.ScoreParams(protein_sequence="MVLSPADKTNVKAAWGKV")

        assert gateway.molecule_key(ETHANOL, here) != gateway.molecule_key(
            ETHANOL, there
        )

    def test_predict_params_are_part_of_the_key(self):
        cheap = gateway.ScoreParams(protein_sequence=SEQUENCE, recycling_steps=1)
        full = gateway.ScoreParams(protein_sequence=SEQUENCE, recycling_steps=3)

        # Exploration fidelity must not be served to a validation request.
        assert gateway.molecule_key(ETHANOL, cheap) != gateway.molecule_key(
            ETHANOL, full
        )

    def test_policy_is_not_part_of_the_key(self):
        # Asking to bypass the cache must not change what the entry is called,
        # or a validation run could never populate anything reusable.
        explore = gateway.ScoreParams(
            protein_sequence=SEQUENCE, fidelity_tier="exploration"
        )
        validate = gateway.ScoreParams(
            protein_sequence=SEQUENCE, fidelity_tier="validation",
            cache_policy="write_only",
        )

        assert gateway.molecule_key(ETHANOL, explore) == gateway.molecule_key(
            ETHANOL, validate
        )

    def test_per_call_plumbing_is_not_part_of_the_key(self):
        bare = gateway.ScoreParams(protein_sequence=SEQUENCE)
        piped = gateway.ScoreParams(
            protein_sequence=SEQUENCE,
            structure_paths=["/shared/a.pdb"],
            protein_pdb_path="/shared/a.pdb",
        )

        assert gateway.molecule_key(ETHANOL, bare) == gateway.molecule_key(
            ETHANOL, piped
        )

    def test_the_msa_is_keyed_by_content_not_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway, "SHARED_DIR", tmp_path)
        (tmp_path / "a.a3m").write_text(">q\nABC\n")
        (tmp_path / "b.a3m").write_text(">q\nABC\n")
        here = gateway.ScoreParams(protein_sequence=SEQUENCE, msa_path="a.a3m")
        there = gateway.ScoreParams(protein_sequence=SEQUENCE, msa_path="b.a3m")

        # Two paths, one alignment: the whole key has to match, not just the
        # MSA digest -- the path must stay out of the params digest too.
        assert gateway.msa_digest(here) == gateway.msa_digest(there)
        assert gateway.molecule_key(ETHANOL, here) == gateway.molecule_key(
            ETHANOL, there
        )
        assert gateway.coalesce_key(here) == gateway.coalesce_key(there)

    def test_the_msa_path_is_still_forwarded_upstream(self):
        # Excluded from the key, but Boltz needs it to find the alignment.
        params = gateway.ScoreParams(protein_sequence=SEQUENCE, msa_path="brd4.a3m")

        assert gateway.prediction_params(params)["msa_path"] == "brd4.a3m"

    def test_editing_the_msa_invalidates_the_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gateway, "SHARED_DIR", tmp_path)
        path = tmp_path / "a.a3m"
        path.write_text(">q\nABC\n")
        before = gateway.msa_digest(gateway.ScoreParams(msa_path="a.a3m"))

        gateway._msa_digest_cache.clear()
        path.write_text(">q\nDEF\n")
        after = gateway.msa_digest(gateway.ScoreParams(msa_path="a.a3m"))

        assert before != after

    def test_a_server_derived_msa_needs_no_file(self):
        # It is a function of the sequence, which is already in the key.
        digest = gateway.msa_digest(gateway.ScoreParams(use_msa_server=True))

        assert digest == "msa:derived-from-sequence"


# ---------------------------------------------------------------------------
# TestCacheBehaviour
# ---------------------------------------------------------------------------


class TestCacheBehaviour:

    def test_a_repeat_batch_predicts_nothing(self, client, upstream):
        first = _score(client, [ETHANOL, BENZENE])
        second = _score(client, [ETHANOL, BENZENE])

        assert first.json()["scores"] == second.json()["scores"]
        assert upstream.molecules == 2, "the second batch should be all hits"

    def test_a_randomised_spelling_still_hits(self, client, upstream):
        _score(client, [ETHANOL])
        _score(client, [ETHANOL_ALT])

        assert upstream.molecules == 1

    def test_a_duplicate_inside_one_batch_is_predicted_once(self, client, upstream):
        response = _score(client, [ETHANOL, ETHANOL_ALT, ETHANOL])

        assert upstream.calls == [[gateway.canonical_smiles(ETHANOL)]]
        # Every slot still gets the answer, in the caller's order.
        assert response.json()["scores"] == [3.0, 3.0, 3.0]

    def test_a_partly_cached_batch_only_predicts_the_rest(self, client, upstream):
        _score(client, [ETHANOL])
        upstream.calls.clear()

        _score(client, [ETHANOL, BENZENE])

        assert upstream.calls == [[gateway.canonical_smiles(BENZENE)]]

    def test_validation_tier_bypasses_the_cache(self, client, upstream):
        _score(client, [ETHANOL])

        _score(client, [ETHANOL], fidelity_tier="validation")

        # A reported result must be a fresh prediction, never a stale sample.
        assert upstream.molecules == 2

    def test_validation_still_populates_the_cache(self, client, upstream, tmp_path):
        _score(client, [ETHANOL], fidelity_tier="validation")
        upstream.calls.clear()

        _score(client, [ETHANOL])

        assert upstream.calls == []

    def test_cache_policy_none_neither_reads_nor_writes(self, client, upstream):
        _score(client, [ETHANOL], cache_policy="none")
        _score(client, [ETHANOL], cache_policy="none")

        assert upstream.molecules == 2

    def test_the_structure_path_survives_a_cache_hit(self, client, upstream, tmp_path):
        # Downstream scorers re-open these by path, so a hit has to carry it.
        structure = tmp_path / "pose.pdb"
        structure.write_text("ATOM")
        digest = gateway.molecule_key(
            gateway.canonical_smiles(ETHANOL),
            gateway.ScoreParams(**_params()),
        )
        gateway.cache_write(digest, gateway.canonical_smiles(ETHANOL),
                            gateway.MoleculeResult(score=1.0, structure=str(structure)))

        response = _score(client, [ETHANOL])

        assert response.json()["artifacts"]["structures"] == [str(structure)]
        assert upstream.calls == []

    def test_a_hit_pointing_at_a_deleted_structure_is_dropped(
        self, client, upstream, tmp_path
    ):
        # Worse than a miss: the consumer would get a path it cannot open.
        digest = gateway.molecule_key(
            gateway.canonical_smiles(ETHANOL), gateway.ScoreParams(**_params())
        )
        gateway.cache_write(digest, gateway.canonical_smiles(ETHANOL),
                            gateway.MoleculeResult(
                                score=1.0, structure=str(tmp_path / "gone.pdb")))

        _score(client, [ETHANOL])

        assert upstream.molecules == 1

    def test_a_failed_prediction_is_not_cached(self):
        # One bad run must not poison a molecule forever.
        result = gateway.MoleculeResult(score=None)

        assert not result.cacheable

    def test_a_deterministic_skip_is_cached(self):
        # Too many heavy atoms will be true next time too.
        result = gateway.MoleculeResult(
            score=None, metadata={"skip_reason": "60 heavy atoms exceeds Boltz limit"}
        )

        assert result.cacheable

    def test_an_unreadable_smiles_scores_none_without_an_upstream_call(
        self, client, upstream
    ):
        response = _score(client, ["not-a-molecule", ETHANOL])

        assert response.json()["scores"][0] is None
        assert response.json()["metadata"]["skip_reason"][0]
        assert upstream.calls == [[gateway.canonical_smiles(ETHANOL)]]


# ---------------------------------------------------------------------------
# TestCoalescing
# ---------------------------------------------------------------------------


class TestCoalescing:

    def test_concurrent_batches_share_one_upstream_call(self, client, upstream):
        # The lever: one invocation pays the >=124 s fixed cost once.
        upstream.release.clear()
        results: list[Any] = []

        def _fire(smiles):
            results.append(_score(client, smiles))

        threads = [
            threading.Thread(target=_fire, args=([ETHANOL],)),
            threading.Thread(target=_fire, args=([BENZENE],)),
        ]
        for thread in threads:
            thread.start()
        # Both are inside the coalescing window; let the batch form, then let
        # the upstream call return.
        threading.Timer(0.5, upstream.release.set).start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(upstream.calls) == 1
        assert sorted(upstream.calls[0]) == sorted(
            [gateway.canonical_smiles(ETHANOL), gateway.canonical_smiles(BENZENE)]
        )
        assert all(r.status_code == 200 for r in results)

    def test_different_targets_are_not_merged(self, client, upstream):
        # Merging would fold two proteins into one boltz run.
        upstream.release.clear()

        def _fire(sequence):
            client.post("/score", json={
                "smiles": [ETHANOL],
                "params": _params(protein_sequence=sequence),
            })

        threads = [
            threading.Thread(target=_fire, args=(SEQUENCE,)),
            threading.Thread(target=_fire, args=("MVLSPADKTNVKAAWGKV",)),
        ]
        for thread in threads:
            thread.start()
        threading.Timer(0.5, upstream.release.set).start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(upstream.calls) == 2

    def test_different_fidelity_is_not_merged(self, client, upstream):
        upstream.release.clear()

        def _fire(steps):
            client.post("/score", json={
                "smiles": [ETHANOL],
                "params": _params(recycling_steps=steps, cache_policy="none"),
            })

        threads = [
            threading.Thread(target=_fire, args=(1,)),
            threading.Thread(target=_fire, args=(3,)),
        ]
        for thread in threads:
            thread.start()
        threading.Timer(0.5, upstream.release.set).start()
        for thread in threads:
            thread.join(timeout=20)

        assert len(upstream.calls) == 2

    def test_the_same_molecule_in_flight_is_predicted_once(self, client, upstream):
        # Two supervisors, one molecule, one GPU batch -- the cross-supervisor
        # saving that caching alone cannot give, since neither has finished yet.
        upstream.release.clear()

        def _fire():
            client.post("/score", json={
                "smiles": [ETHANOL], "params": _params(cache_policy="none"),
            })

        threads = [threading.Thread(target=_fire) for _ in range(3)]
        for thread in threads:
            thread.start()
        threading.Timer(0.5, upstream.release.set).start()
        for thread in threads:
            thread.join(timeout=20)

        assert upstream.molecules == 1
        assert gateway.batcher.molecules_deduplicated >= 2

    def test_a_full_batch_dispatches_without_waiting(self, client, upstream, monkeypatch):
        monkeypatch.setattr(gateway, "COALESCE_MAX_MOLECULES", 2)
        monkeypatch.setattr(gateway, "COALESCE_WINDOW_SECONDS", 30.0)

        # Would block for 30 s if the cap did not force a dispatch.
        response = _score(client, [ETHANOL, BENZENE])

        assert response.status_code == 200
        assert len(upstream.calls) == 1

    def test_a_batch_over_the_cap_is_split(self, client, upstream, monkeypatch):
        monkeypatch.setattr(gateway, "COALESCE_MAX_MOLECULES", 2)

        response = _score(client, [ETHANOL, BENZENE, "CCC", "CCCC"])

        assert response.status_code == 200
        assert len(upstream.calls) == 2
        assert sorted(len(call) for call in upstream.calls) == [2, 2]


# ---------------------------------------------------------------------------
# TestResponseShape
# ---------------------------------------------------------------------------


class TestResponseShape:

    def test_scores_come_back_in_the_callers_order(self, client, upstream):
        response = _score(client, [BENZENE, ETHANOL])

        # len() of the canonical SMILES, so order is checkable.
        assert response.json()["scores"] == [8.0, 3.0]

    def test_metadata_is_per_molecule_and_aligned(self, client, upstream):
        response = _score(client, [ETHANOL, BENZENE])
        metadata = response.json()["metadata"]

        # ExternalRESTScorer only keeps list-valued metadata of the right
        # length, so anything else is silently dropped downstream.
        assert metadata["affinity_pred_value"] == [3.0, 8.0]
        for values in metadata.values():
            assert len(values) == 2

    def test_structures_are_aligned_with_the_smiles(self, client, upstream):
        response = _score(client, [ETHANOL, BENZENE])

        structures = response.json()["artifacts"]["structures"]
        assert len(structures) == 2
        assert len(set(structures)) == 2, "each molecule needs its own pose"
        assert all(Path(s).is_file() for s in structures)

    def test_an_upstream_failure_is_a_502_not_a_silent_none(self, client, upstream):
        # The scorer fills NaN on a non-2xx, which is correct here: nothing was
        # predicted. What matters is that it is not reported as a real score.
        upstream.fail = True

        response = _score(client, [ETHANOL])

        assert response.status_code == 502

    def test_a_failure_leaves_nothing_cached(self, client, upstream):
        upstream.fail = True
        _score(client, [ETHANOL])
        upstream.fail = False
        upstream.calls.clear()

        _score(client, [ETHANOL])

        # Predicted again rather than served from a cached failure.
        assert upstream.calls == [[gateway.canonical_smiles(ETHANOL)]]

    def test_policy_fields_are_not_forwarded_upstream(self, client, upstream):
        # Boltz ignores unknown params, but sending policy knobs to it would
        # mean they had leaked into the prediction identity.
        client.post("/score", json={
            "smiles": [ETHANOL],
            "params": _params(fidelity_tier="validation", cache_policy="none"),
        })

        assert gateway.prediction_params(
            gateway.ScoreParams(**_params(fidelity_tier="validation"))
        ).keys() == {"protein_sequence", "recycling_steps", "msa_path"}


# ---------------------------------------------------------------------------
# TestAsyncApi
# ---------------------------------------------------------------------------


class TestAsyncApi:

    def test_submit_returns_202_with_a_task_id(self, client, upstream):
        upstream.release.clear()

        response = client.post(
            "/score?mode=async", json={"smiles": [ETHANOL], "params": _params()}
        )

        assert response.status_code == 202
        assert response.json()["task_id"]
        upstream.release.set()

    def test_polling_returns_202_until_it_is_done(self, client, upstream):
        upstream.release.clear()
        task_id = client.post(
            "/score?mode=async", json={"smiles": [ETHANOL], "params": _params()}
        ).json()["task_id"]

        pending = client.get(f"/result/{task_id}")

        # Queued is not failed -- that confusion is what poisons the RL signal.
        assert pending.status_code == 202
        assert pending.json()["status"] == "pending"
        upstream.release.set()

    def test_the_result_arrives_and_matches_the_sync_path(self, client, upstream):
        task_id = client.post(
            "/score?mode=async", json={"smiles": [ETHANOL], "params": _params()}
        ).json()["task_id"]

        response = _poll(client, task_id)

        assert response.status_code == 200
        assert response.json()["scores"] == [3.0]

    def test_a_result_is_served_once(self, client, upstream):
        task_id = client.post(
            "/score?mode=async", json={"smiles": [ETHANOL], "params": _params()}
        ).json()["task_id"]
        assert _poll(client, task_id).status_code == 200

        # Freed after collection, so a long campaign cannot grow the registry
        # without bound.
        assert client.get(f"/result/{task_id}").status_code == 404

    def test_an_unknown_task_is_404(self, client):
        assert client.get("/result/nope").status_code == 404

    def test_an_async_failure_surfaces_as_502(self, client, upstream):
        upstream.fail = True
        task_id = client.post(
            "/score?mode=async", json={"smiles": [ETHANOL], "params": _params()}
        ).json()["task_id"]

        response = _poll(client, task_id)

        assert response.status_code == 502


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------


class TestStats:

    def test_the_hit_rate_is_reported(self, client, upstream):
        # The number every sizing calculation has had to assume so far.
        _score(client, [ETHANOL, BENZENE])
        _score(client, [ETHANOL, BENZENE])

        cache = client.get("/stats").json()["cache"]

        assert cache["hits"] == 2
        assert cache["misses"] == 2
        assert cache["hit_rate"] == pytest.approx(0.5)

    def test_the_hit_rate_is_none_before_anything_is_looked_up(self, client):
        assert client.get("/stats").json()["cache"]["hit_rate"] is None

    def test_molecules_per_call_shows_the_coalescing(self, client, upstream):
        _score(client, [ETHANOL, BENZENE, "CCC"])

        coalescing = client.get("/stats").json()["coalescing"]

        assert coalescing["upstream_calls"] == 1
        assert coalescing["molecules_per_call"] == pytest.approx(3.0)

    def test_latency_percentiles_are_reported(self, client, upstream):
        _score(client, [ETHANOL])

        latency = client.get("/stats").json()["latency_seconds"]

        assert latency["upstream_p50"] is not None
        assert latency["request_p50"] is not None

    def test_duplicates_and_invalid_molecules_are_counted(self, client, upstream):
        _score(client, [ETHANOL, ETHANOL_ALT, "not-a-molecule"])

        molecules = client.get("/stats").json()["molecules"]

        assert molecules["requested"] == 3
        assert molecules["duplicates_within_requests"] == 1
        assert molecules["invalid"] == 1

    def test_queue_and_replica_view_are_exposed(self, client):
        queue = client.get("/stats").json()["queue"]

        # What the conductor throttles N against.
        assert "gateway_molecules_waiting" in queue
        assert "proxy_queued_requests" in queue
        assert "inflight_limit" in queue

    def test_health_reports_the_cache_and_upstream(self, client):
        health = client.get("/health").json()

        assert health["status"] == "ok"
        assert health["cache_enabled"] is True
        assert health["upstream"].endswith("/score")


# ---------------------------------------------------------------------------
# TestReplicaView
# ---------------------------------------------------------------------------


class TestReplicaView:

    CSV = (
        "# pxname,svname,qcur,qmax,scur,smax,slim,stot,status\n"
        "boltz,boltz1,0,0,1,1,1,3,UP\n"
        "boltz,boltz2,0,0,0,0,1,3,UP\n"
        "boltz,boltz3,0,0,0,0,1,0,MAINT (resolution)\n"
        "boltz,BACKEND,2,2,1,2,100,7,UP\n"
    )

    def _view(self, monkeypatch, csv: str) -> gateway.ReplicaView:
        class _Response:
            def read(self):
                return csv.encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            gateway.urllib.request, "urlopen", lambda url, timeout=None: _Response()
        )
        view = gateway.ReplicaView()
        view.refresh()
        return view

    def test_healthy_replicas_are_counted(self, monkeypatch):
        view = self._view(monkeypatch, self.CSV)

        assert view.healthy == 2
        assert view.reachable is True

    def test_the_proxy_queue_is_read_from_the_backend_row(self, monkeypatch):
        # Per-server qcur stays 0 while the backend queues, which is where a
        # request waiting for a free replica actually sits.
        view = self._view(monkeypatch, self.CSV)

        assert view.proxy_queued == 2
        assert view.proxy_in_flight == 1

    def test_in_flight_limit_follows_the_replica_count(self, monkeypatch):
        view = self._view(monkeypatch, self.CSV)

        assert view.inflight_limit == 2

    def test_an_unreachable_proxy_still_allows_one_call(self, monkeypatch):
        def _boom(url, timeout=None):
            raise OSError("refused")

        monkeypatch.setattr(gateway.urllib.request, "urlopen", _boom)
        view = gateway.ReplicaView()
        view.refresh()

        # Never zero: that would wedge the gateway if the scrape broke.
        assert view.reachable is False
        assert view.inflight_limit == 1

    def test_an_explicit_limit_wins(self, monkeypatch):
        monkeypatch.setattr(gateway, "MAX_INFLIGHT", 5)
        view = self._view(monkeypatch, self.CSV)

        assert view.inflight_limit == 5


# ---------------------------------------------------------------------------
# TestContract
# ---------------------------------------------------------------------------


class TestContract:
    """The gateway has to be a drop-in for a Boltz replica."""

    def test_it_speaks_the_same_request_shape_as_boltz(self):
        boltz_spec = importlib.util.spec_from_file_location(
            "boltz2_app_for_gateway", str(_ROOT / "services" / "boltz2" / "app.py")
        )
        boltz = importlib.util.module_from_spec(boltz_spec)
        sys.modules["boltz2_app_for_gateway"] = boltz
        boltz_spec.loader.exec_module(boltz)

        payload = {"smiles": [ETHANOL], "params": _params(msa_path="brd4_bd1.a3m")}

        # The same body validates against both, so the scorer needs no change.
        assert gateway.ScoreRequest(**payload).smiles == [ETHANOL]
        assert boltz.ScoreRequest(**payload).params.msa_path == "brd4_bd1.a3m"

    def test_the_response_has_the_three_keys_the_scorer_reads(self, client, upstream):
        body = _score(client, [ETHANOL]).json()

        assert set(body) == {"scores", "metadata", "artifacts"}
        assert "structures" in body["artifacts"]
