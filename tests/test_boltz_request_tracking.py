# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the requests in progress that the Boltz2 wrapper reports on /health.

Models mocked, no GPU needed: molecules finish when the test opens their gate.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_BOLTZ_DIR = Path(__file__).resolve().parent.parent / "services" / "boltz2"

if "boltz2_app" in sys.modules:
    boltz_app = sys.modules["boltz2_app"]
else:
    _spec = importlib.util.spec_from_file_location("boltz2_app", str(_BOLTZ_DIR / "app.py"))
    boltz_app = importlib.util.module_from_spec(_spec)
    sys.modules["boltz2_app"] = boltz_app
    _spec.loader.exec_module(boltz_app)

import worker_pool  # noqa: E402  (services/boltz2 is put on sys.path by app.py)

SEQUENCE = "MVLSPADKTN"


class GatedBackend:
    """A worker whose molecule ``mol_<i>`` finishes once ``gates[i]`` is set."""

    gates: dict[int, threading.Event] = {}

    def __init__(self, worker_id: str, device: int, cache_dir: str = "/cache") -> None:
        self.worker_id = worker_id
        self.device = device

    def start(self) -> dict:
        return {}

    def call(self, payload: dict, timeout: float) -> dict:
        mol_id = Path(payload["yaml_path"]).stem if "yaml_path" in payload else "mol_0"
        index = int(mol_id.split("_")[1])
        self.gates.setdefault(index, threading.Event()).wait(10)
        if "out_dir" in payload:
            mol_dir = Path(payload["out_dir"]) / "predictions" / mol_id
            mol_dir.mkdir(parents=True, exist_ok=True)
            (mol_dir / f"affinity_{mol_id}.json").write_text(json.dumps({"affinity_pred_value": -5.0}))
        return {"status": "ok"}

    def stop(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _gates():
    GatedBackend.gates = {i: threading.Event() for i in range(8)}
    yield
    for gate in GatedBackend.gates.values():
        gate.set()
    boltz_app._requests.clear()


def _wait_for(client, predicate, attempts=250):
    for _ in range(attempts):
        data = client.get("/health").json()
        if predicate(data):
            return data
        time.sleep(0.02)
    raise AssertionError(f"health never matched: {data}")


@pytest.fixture
def pool_client(tmp_path):
    msa = tmp_path / "t.a3m"
    msa.write_text(f">q\n{SEQUENCE}\n")
    with patch.object(boltz_app, "SHARED_DIR", tmp_path), \
            patch.object(boltz_app, "BOLTZ_WORKERS_PER_GPU", 1), \
            patch.object(boltz_app, "_visible_gpu_devices", return_value=[0, 1]), \
            patch.object(boltz_app.worker_pool, "ProcessBackend", GatedBackend):
        with TestClient(boltz_app.app) as client:
            _wait_for(client, lambda d: d["status"] == "ok")
            yield client, str(msa)
    boltz_app._pool = None


def _post_in_thread(client, smiles, msa):
    box: dict = {}

    def post():
        box["response"] = client.post("/score", json={
            "smiles": smiles, "params": {"protein_sequence": SEQUENCE, "msa_path": msa},
        })

    thread = threading.Thread(target=post)
    thread.start()
    return thread, box


class TestPoolProgress:

    def test_request_progress_counts_queued_and_running(self):
        async def main():
            pool = worker_pool.WorkerPool([0], 1, GatedBackend, job_timeout=60)
            await pool.start()
            for _ in range(100):
                if pool.status == "ok":
                    break
                await asyncio.sleep(0.01)
            futures_a = pool.submit("A", [{}, {}])
            futures_b = pool.submit("B", [{}])
            await asyncio.sleep(0.05)
            during = pool.request_progress()
            for gate in GatedBackend.gates.values():
                gate.set()
            await asyncio.gather(*futures_a, *futures_b)
            return during, pool.request_progress()

        during, after = asyncio.run(main())
        assert during == {"A": {"queued": 1, "running": 1}, "B": {"queued": 1, "running": 0}}
        assert after == {}


class TestMoleculeTimes:

    def test_running_molecules_report_how_long_they_have_run(self):
        async def main():
            pool = worker_pool.WorkerPool([0], 1, GatedBackend, job_timeout=60)
            await pool.start()
            for _ in range(100):
                if pool.status == "ok":
                    break
                await asyncio.sleep(0.01)
            before = pool.molecule_seconds()
            futures = pool.submit("A", [{}, {}])
            await asyncio.sleep(0.15)
            running = pool.running_elapsed()
            GatedBackend.gates[0].set()
            GatedBackend.gates[1].set()
            await asyncio.gather(*futures)
            return before, running, pool.running_elapsed(), pool.molecule_seconds(), pool.health()

        before, running, after, typical, health = asyncio.run(main())
        assert before is None
        assert list(running) == ["A"]
        (seconds,) = running["A"]
        assert 0.1 <= seconds < 5
        assert after == {}
        assert typical is not None and 0 <= typical < 5
        assert health["molecule_s"] == typical

    def test_failed_molecules_do_not_count_towards_the_typical_time(self):
        class Failing(GatedBackend):
            def call(self, payload: dict, timeout: float) -> dict:
                return {"status": "error", "error": "boom"}

        async def main():
            pool = worker_pool.WorkerPool([0], 1, Failing, job_timeout=60)
            await pool.start()
            for _ in range(100):
                if pool.status == "ok":
                    break
                await asyncio.sleep(0.01)
            await asyncio.gather(*pool.submit("A", [{}]))
            return pool.molecule_seconds()

        assert asyncio.run(main()) is None


class TestHealthRequests:

    def test_idle_service_lists_no_requests(self, pool_client):
        client, _ = pool_client
        data = client.get("/health").json()
        assert data["requests"] == []
        assert data["mode"] == "pool"
        assert len(data["workers"]) == 2

    def test_molecules_done_count_up_while_the_request_runs(self, pool_client):
        client, msa = pool_client
        thread, box = _post_in_thread(client, ["CCO", "CCN", "CCC"], msa)

        running = _wait_for(client, lambda d: d["requests"] and d["requests"][0]["running"] == 2)
        (req,) = running["requests"]
        assert req["molecules"] == 3
        assert req["done"] == 0
        assert req["queued"] == 1
        assert req["state"] == "running"
        assert req["elapsed_s"] >= 0
        assert abs(req["started_at"] - time.time()) < 60

        assert len(req["running_s"]) == 2
        assert all(0 <= s < 60 for s in req["running_s"])
        assert running["molecule_s"] is None

        GatedBackend.gates[0].set()
        one_done = _wait_for(client, lambda d: d["requests"][0]["done"] == 1)
        assert one_done["requests"][0]["queued"] == 0
        assert one_done["requests"][0]["running"] == 2
        assert len(one_done["requests"][0]["running_s"]) == 2
        assert one_done["molecule_s"] is not None

        GatedBackend.gates[1].set()
        GatedBackend.gates[2].set()
        thread.join(10)
        assert box["response"].status_code == 200
        assert client.get("/health").json()["requests"] == []

    def test_score_response_contract_is_unchanged(self, pool_client):
        client, msa = pool_client
        for gate in GatedBackend.gates.values():
            gate.set()
        resp = client.post("/score", json={
            "smiles": ["CCO"], "params": {"protein_sequence": SEQUENCE, "msa_path": msa},
        })
        assert resp.status_code == 200
        assert set(resp.json()) == {"scores", "metadata", "artifacts"}

    def test_invalid_smiles_are_not_counted(self, pool_client):
        client, msa = pool_client
        thread, box = _post_in_thread(client, ["CCO", "not_a_smiles"], msa)
        data = _wait_for(client, lambda d: d["requests"] and d["requests"][0]["running"] == 1)
        assert data["requests"][0]["molecules"] == 1
        GatedBackend.gates[0].set()
        thread.join(10)
        assert box["response"].status_code == 200


class TestSubprocessPath:

    def test_waiting_and_running_requests(self, tmp_path):
        msa = tmp_path / "t.a3m"
        msa.write_text(f">q\n{SEQUENCE}\n")
        release = threading.Event()

        def slow_prediction(params, valid_smiles, valid_indices, skip_reasons, n_total, handle=None):
            release.wait(10)
            n = len(valid_smiles)
            return boltz_app.ScoreResponse(scores=[-5.0] * n, metadata={"skip_reason": [""] * n})

        boltz_app._pool = None
        with patch.object(boltz_app, "SHARED_DIR", tmp_path), \
                patch.object(boltz_app, "BOLTZ_WORKERS_PER_GPU", 0), \
                patch.object(boltz_app, "_run_prediction", slow_prediction):
            with TestClient(boltz_app.app) as client:
                first, box_a = _post_in_thread(client, ["CCO", "CCN"], str(msa))
                _wait_for(client, lambda d: len(d.get("requests", [])) == 1 and d["requests"][0]["state"] == "running")
                second, box_b = _post_in_thread(client, ["CCC"], str(msa))
                data = _wait_for(client, lambda d: len(d.get("requests", [])) == 2)
                states = sorted((r["state"], r["molecules"], r["queued"], r["running"]) for r in data["requests"])
                assert states == [("queued", 1, 1, 0), ("running", 2, 0, 2)]
                release.set()
                first.join(10)
                second.join(10)
                assert box_a["response"].status_code == 200
                assert box_b["response"].status_code == 200
                assert client.get("/health").json() == {"status": "ok"}
