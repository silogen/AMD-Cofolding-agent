# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Boltz2 persistent worker pool and the pool-backed /score path.

Covers (models mocked, no GPU needed):
- Scheduler: parallel split of a request, GPU spreading, FIFO across requests,
  many concurrent requests, OOM retry with the GPU to itself, cancellation,
  worker start failure / crash handling, health reporting
- Template validation and the Boltz-2 ``templates:`` YAML
- /score through the pool: response layout, per-molecule skip reasons,
  fallback to the subprocess path, /health
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
import yaml
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

WorkerPool = worker_pool.WorkerPool


# ---------------------------------------------------------------------------
# Fake worker backend
# ---------------------------------------------------------------------------


class FakeBackend:
    """Stands in for a worker process. Behaviour is driven by the payload:

    ``sleep``      seconds the molecule takes
    ``oom_shared`` return "oom" while another molecule runs on the same GPU
    ``oom``        always return "oom"
    ``crash``      raise WorkerCrashed
    """

    lock = threading.Lock()
    running: dict[int, int] = {}
    # (event, device, tag, n): n is the molecules on the device at start for
    # "start", and the most seen at start or end of the run for "end".
    log: list[tuple[str, int, str, int]] = []
    max_concurrent = 0
    start_delay = 0.0
    fail_start: set[str] = set()

    @classmethod
    def reset(cls) -> None:
        cls.running = {}
        cls.log = []
        cls.max_concurrent = 0
        cls.start_delay = 0.0
        cls.fail_start = set()

    def __init__(self, worker_id: str, device: int, cache_dir: str = "/cache") -> None:
        self.worker_id = worker_id
        self.device = device

    def start(self) -> dict:
        time.sleep(self.start_delay)
        if self.worker_id in self.fail_start:
            raise RuntimeError("HIP error: no device")
        return {"device_used_gb": 4.2}

    def call(self, payload: dict, timeout: float) -> dict:
        cls = FakeBackend
        with cls.lock:
            cls.running[self.device] = cls.running.get(self.device, 0) + 1
            concurrent_dev = cls.running[self.device]
            cls.max_concurrent = max(cls.max_concurrent, sum(cls.running.values()))
            cls.log.append(("start", self.device, payload.get("tag", ""), concurrent_dev))
        time.sleep(payload.get("sleep", 0.05))
        with cls.lock:
            seen = max(concurrent_dev, cls.running[self.device])
            cls.running[self.device] -= 1
            cls.log.append(("end", self.device, payload.get("tag", ""), seen))
        if payload.get("crash"):
            raise worker_pool.WorkerCrashed("worker process died (exit code -11)")
        if payload.get("oom"):
            return {"status": "oom", "error": "HIP out of memory"}
        if payload.get("oom_shared") and seen > 1:
            return {"status": "oom", "error": "HIP out of memory"}
        if "write" in payload:
            payload["write"](payload)
        return {"status": "ok", "timings": {"total_s": 0.05}}

    def stop(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeBackend.reset()
    yield
    FakeBackend.reset()


async def _ready_pool(devices=(0,), per_gpu=2, timeout=600.0) -> WorkerPool:
    pool = WorkerPool(list(devices), per_gpu, FakeBackend, job_timeout=timeout)
    await pool.start()
    for _ in range(200):
        if all(w.state != "loading" for w in pool.workers):
            break
        await asyncio.sleep(0.01)
    return pool


# ---------------------------------------------------------------------------
# TestScheduler
# ---------------------------------------------------------------------------


class TestScheduler:

    def test_request_split_runs_in_parallel(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            t0 = time.monotonic()
            futs = pool.submit("r1", [{"sleep": 0.3, "tag": "a"}, {"sleep": 0.3, "tag": "b"}])
            results = await asyncio.gather(*futs)
            return time.monotonic() - t0, results

        elapsed, results = asyncio.run(main())
        assert [r["status"] for r in results] == ["ok", "ok"]
        assert FakeBackend.max_concurrent == 2
        assert elapsed < 0.55

    def test_molecules_spread_over_gpus_first(self):
        async def main():
            pool = await _ready_pool(devices=(0, 1, 2), per_gpu=2)
            futs = pool.submit("r1", [{"sleep": 0.2}, {"sleep": 0.2}, {"sleep": 0.2}])
            return await asyncio.gather(*futs)

        results = asyncio.run(main())
        assert sorted(r["device"] for r in results) == [0, 1, 2]
        assert not any(r["shared_gpu"] for r in results)

    def test_fifo_across_requests(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=1)
            fa = pool.submit("A", [{"tag": "A0"}, {"tag": "A1"}, {"tag": "A2"}])
            fb = pool.submit("B", [{"tag": "B0"}])
            await asyncio.gather(*fa, *fb)

        asyncio.run(main())
        order = [tag for ev, _, tag, _ in FakeBackend.log if ev == "start"]
        assert order == ["A0", "A1", "A2", "B0"]

    def test_many_concurrent_requests_all_succeed(self):
        async def main():
            pool = await _ready_pool(devices=(0, 1), per_gpu=2)
            futs = [pool.submit(f"r{i}", [{"sleep": 0.02}, {"sleep": 0.02}]) for i in range(10)]
            return await asyncio.gather(*(asyncio.gather(*f) for f in futs))

        results = asyncio.run(main())
        assert all(r["status"] == "ok" for req in results for r in req)
        assert FakeBackend.max_concurrent <= 4

    def test_requests_queue_while_workers_load(self):
        FakeBackend.start_delay = 0.2

        async def main():
            pool = WorkerPool([0], 2, FakeBackend, job_timeout=60)
            await pool.start()
            assert pool.status == "loading"
            futs = pool.submit("r1", [{}, {}])
            assert pool.queue_length() == 2
            return await asyncio.gather(*futs)

        results = asyncio.run(main())
        assert [r["status"] for r in results] == ["ok", "ok"]

    def test_oom_next_to_other_molecule_retries_alone(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            futs = pool.submit("r1", [
                {"sleep": 0.2, "tag": "big", "oom_shared": True},
                {"sleep": 0.4, "tag": "small"},
            ])
            return await asyncio.gather(*futs)

        big, small = asyncio.run(main())
        assert big["status"] == "ok"
        assert big["retried"] is True
        assert small["status"] == "ok"
        big_runs = [n for ev, _, tag, n in FakeBackend.log if ev == "end" and tag == "big"]
        assert big_runs == [2, 1]  # first shared, retry alone on the GPU

    def test_retry_blocks_gpu_for_later_jobs(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            f1 = pool.submit("r1", [
                {"sleep": 0.2, "tag": "big", "oom_shared": True},
                {"sleep": 0.4, "tag": "small"},
            ])
            f2 = pool.submit("r2", [{"sleep": 0.2, "tag": "later"}])
            return await asyncio.gather(*f1, *f2)

        results = asyncio.run(main())
        assert all(r["status"] == "ok" for r in results)
        events = [(ev, tag, n) for ev, _, tag, n in FakeBackend.log]
        assert events.count(("end", "big", 1)) == 1
        retry_end = events.index(("end", "big", 1))
        later_start = next(i for i, e in enumerate(events) if e[:2] == ("start", "later"))
        assert later_start > retry_end

    def test_oom_alone_is_reported_not_retried(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            (fut,) = pool.submit("r1", [{"oom": True, "tag": "x"}])
            return await fut

        result = asyncio.run(main())
        assert result["status"] == "oom"
        assert result["retried"] is False
        assert len([e for e in FakeBackend.log if e[0] == "start"]) == 1

    def test_oom_on_retry_is_reported(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            futs = pool.submit("r1", [{"oom": True, "sleep": 0.1}, {"sleep": 0.2}])
            return await asyncio.gather(*futs)

        big, small = asyncio.run(main())
        assert big["status"] == "oom"
        assert big["retried"] is True
        assert small["status"] == "ok"

    def test_cancel_drops_queued_molecules(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=1)
            fa = pool.submit("A", [{"sleep": 0.2, "tag": "A0"}])
            fb = pool.submit("B", [{"tag": "B0"}, {"tag": "B1"}])
            await asyncio.sleep(0.05)
            dropped = pool.cancel("B")
            res_b = await asyncio.gather(*fb, return_exceptions=True)
            res_a = await asyncio.gather(*fa)
            return dropped, res_a, res_b, pool.queue_length()

        dropped, res_a, res_b, qlen = asyncio.run(main())
        assert dropped == 2
        assert res_a[0]["status"] == "ok"
        assert all(isinstance(r, worker_pool.JobCancelled) for r in res_b)
        assert qlen == 0
        assert [t for ev, _, t, _ in FakeBackend.log if ev == "start"] == ["A0"]

    def test_all_workers_fail_to_start(self):
        FakeBackend.fail_start = {"gpu0-w0", "gpu0-w1"}

        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=2)
            with pytest.raises(worker_pool.PoolUnavailable):
                pool.submit("r1", [{}])
            return pool.health()

        health = asyncio.run(main())
        assert health["status"] == "ok"
        assert health["mode"] == "subprocess"
        assert {w["state"] for w in health["workers"]} == {"failed"}
        assert "HIP error" in health["workers"][0]["error"]

    def test_queued_jobs_fail_over_when_last_worker_fails(self):
        FakeBackend.start_delay = 0.1
        FakeBackend.fail_start = {"gpu0-w0"}

        async def main():
            pool = WorkerPool([0], 1, FakeBackend, job_timeout=60)
            await pool.start()
            futs = pool.submit("r1", [{}])
            return await asyncio.gather(*futs, return_exceptions=True)

        (result,) = asyncio.run(main())
        assert isinstance(result, worker_pool.PoolUnavailable)

    def test_crash_reports_error_and_restarts_worker(self):
        async def main():
            pool = await _ready_pool(devices=(0,), per_gpu=1)
            (f1,) = pool.submit("r1", [{"crash": True}])
            r1 = await f1
            (f2,) = pool.submit("r2", [{}])
            r2 = await f2
            return r1, r2, pool.workers[0].restarts

        r1, r2, restarts = asyncio.run(main())
        assert r1["status"] == "error"
        assert "exit code -11" in r1["error"]
        assert r2["status"] == "ok"
        assert restarts == 1

    def test_health_reports_workers_and_queue(self):
        async def main():
            pool = await _ready_pool(devices=(0, 1), per_gpu=2)
            futs = pool.submit("r1", [{"sleep": 0.2}] * 6)
            await asyncio.sleep(0.05)
            health = pool.health()
            await asyncio.gather(*futs)
            return health, pool.health()

        busy, idle = asyncio.run(main())
        assert busy["status"] == "ok"
        assert busy["mode"] == "pool"
        assert busy["queue_length"] == 2
        assert sorted(w["state"] for w in busy["workers"]) == ["busy"] * 4
        assert {w["device"] for w in busy["workers"]} == {0, 1}
        assert idle["queue_length"] == 0
        assert {w["state"] for w in idle["workers"]} == {"ready"}
        assert busy["workers"][0]["memory"] == {"device_used_gb": 4.2}


# ---------------------------------------------------------------------------
# TestTemplates
# ---------------------------------------------------------------------------


class TestTemplates:

    def _specs(self, *items):
        return [boltz_app.TemplateSpec(**i) for i in items]

    def test_relative_path_resolves_under_shared(self, tmp_path):
        (tmp_path / "templates").mkdir()
        (tmp_path / "templates" / "3ert_clean.cif").write_text("data_x\n")
        with patch.object(boltz_app, "SHARED_DIR", tmp_path):
            entries, errors = boltz_app._resolve_templates(
                self._specs({"path": "templates/3ert_clean.cif"})
            )
        assert errors == []
        assert entries == [{"cif": str(tmp_path / "templates" / "3ert_clean.cif")}]

    def test_pdb_with_chain_force_threshold(self, tmp_path):
        pdb = tmp_path / "1abc.pdb"
        pdb.write_text("ATOM\n")
        entries, errors = boltz_app._resolve_templates(
            self._specs({"path": str(pdb), "chain_id": "B", "force": True, "threshold": 2.0})
        )
        assert errors == []
        assert entries == [{
            "pdb": str(pdb), "chain_id": "A", "template_id": "B",
            "force": True, "threshold": 2.0,
        }]

    def test_threshold_without_force_is_not_emitted(self, tmp_path):
        cif = tmp_path / "t.mmcif"
        cif.write_text("data_x\n")
        entries, errors = boltz_app._resolve_templates(
            self._specs({"path": str(cif), "threshold": 3.0})
        )
        assert errors == []
        assert entries == [{"cif": str(cif)}]

    @pytest.mark.parametrize("spec, fragment", [
        ({"path": "missing.cif"}, "not found"),
        ({"path": "model.sdf"}, ".cif/.mmcif or .pdb"),
        ({"path": "PLACEHOLDER", "force": True}, "threshold"),
        ({"path": "PLACEHOLDER", "force": True, "threshold": -1.0}, "positive"),
    ])
    def test_invalid_templates(self, tmp_path, spec, fragment):
        good = tmp_path / "ok.cif"
        good.write_text("data_x\n")
        spec = {**spec, "path": str(good) if spec["path"] == "PLACEHOLDER" else spec["path"]}
        with patch.object(boltz_app, "SHARED_DIR", tmp_path):
            _, errors = boltz_app._resolve_templates(self._specs(spec))
        assert len(errors) == 1
        assert fragment in errors[0]

    def test_duplicate_stems_rejected(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "t.cif").write_text("x")
        (tmp_path / "b" / "t.pdb").write_text("x")
        _, errors = boltz_app._resolve_templates(self._specs(
            {"path": str(tmp_path / "a" / "t.cif")},
            {"path": str(tmp_path / "b" / "t.pdb")},
        ))
        assert len(errors) == 1
        assert "unique" in errors[0]

    def test_yaml_contains_templates_block(self, tmp_path):
        entries = [{"cif": "/shared/t.cif", "chain_id": "A", "template_id": "C",
                    "force": True, "threshold": 1.5}]
        paths = boltz_app._write_batch_yamls(
            tmp_path / "b", ["CCO"], "MVLSPADKTN", "/shared/m.a3m", templates=entries,
        )
        doc = yaml.safe_load(paths[0].read_text())
        assert doc["templates"] == entries
        assert list(doc) == ["version", "sequences", "templates", "properties"]

    def test_yaml_without_templates_unchanged(self):
        doc = boltz_app._build_molecule_yaml("CCO", "MVLSPADKTN", "/m.a3m")
        assert "templates" not in doc
        assert list(doc) == ["version", "sequences", "properties"]

    def test_score_rejects_bad_template_with_422(self):
        client = TestClient(boltz_app.app)
        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "msa_path": "/shared/x.a3m",
                "templates": [{"path": "/definitely/missing.cif"}],
            },
        })
        assert resp.status_code == 422
        assert "templates[0]" in resp.json()["detail"][0]


# ---------------------------------------------------------------------------
# TestPoolScoreEndpoint
# ---------------------------------------------------------------------------


def _write_outputs(payload: dict) -> None:
    """Write what the real worker writes for one molecule."""
    mol_id = Path(payload["yaml_path"]).stem
    mol_dir = Path(payload["out_dir"]) / "predictions" / mol_id
    mol_dir.mkdir(parents=True, exist_ok=True)
    value = -5.0 - int(mol_id.split("_")[1])
    (mol_dir / f"affinity_{mol_id}.json").write_text(json.dumps({
        "affinity_pred_value": value, "affinity_probability_binary": 0.7,
    }))
    (mol_dir / f"confidence_{mol_id}_model_0.json").write_text(json.dumps({
        "confidence_score": 0.8, "ligand_iptm": 0.6,
    }))
    (mol_dir / f"{mol_id}_model_0.pdb").write_text("ATOM\n")


class _WritingBackend(FakeBackend):
    oom_molecules: set[str] = set()

    def call(self, payload: dict, timeout: float) -> dict:
        mol_id = Path(payload["yaml_path"]).stem
        if mol_id in self.oom_molecules:
            payload = {**payload, "oom": True}
        else:
            payload = {**payload, "write": _write_outputs}
        return super().call(payload, timeout)


@pytest.fixture
def pool_client(tmp_path):
    msa = tmp_path / "t.a3m"
    msa.write_text(">q\nMVLSPADKTN\n")
    _WritingBackend.oom_molecules = set()
    with patch.object(boltz_app, "SHARED_DIR", tmp_path), \
            patch.object(boltz_app, "BOLTZ_WORKERS_PER_GPU", 2), \
            patch.object(boltz_app, "_visible_gpu_devices", return_value=[0, 1]), \
            patch.object(boltz_app.worker_pool, "ProcessBackend", _WritingBackend):
        with TestClient(boltz_app.app) as client:
            for _ in range(100):
                if client.get("/health").json()["status"] == "ok":
                    break
                time.sleep(0.02)
            yield client, tmp_path, str(msa)
    boltz_app._pool = None


class TestPoolScoreEndpoint:

    def test_health_lists_workers(self, pool_client):
        client, _, _ = pool_client
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["queue_length"] == 0
        assert [(w["device"], w["state"]) for w in data["workers"]] == [
            (0, "ready"), (0, "ready"), (1, "ready"), (1, "ready"),
        ]

    def test_response_layout_matches_subprocess_path(self, pool_client):
        client, shared, msa = pool_client
        resp = client.post("/score", json={
            "smiles": ["CC(=O)Nc1ccc(O)cc1", "not_a_smiles", "c1ccc2[nH]ccc2c1"],
            "params": {"protein_sequence": "MVLSPADKTN", "msa_path": msa},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {"scores", "metadata", "artifacts"}
        assert data["scores"] == [pytest.approx(-5.0), None, pytest.approx(-6.0)]
        meta = data["metadata"]
        assert list(meta) == [
            "skip_reason", "affinity_pred_value", "affinity_probability_binary",
            "confidence_score", "ligand_iptm",
        ]
        assert meta["skip_reason"][0] == "" and meta["skip_reason"][2] == ""
        assert "RDKit" in meta["skip_reason"][1]
        assert meta["confidence_score"] == [0.8, None, 0.8]

        structs = data["artifacts"]["structures"]
        assert structs[1] is None
        for s, mol in ((structs[0], "mol_0"), (structs[2], "mol_1")):
            p = Path(s)
            assert p.name == f"{mol}_model_0.pdb"
            assert p.parent.name == mol
            assert p.parent.parent.name == "predictions"
            batch = p.parent.parent.parent.name
            assert batch.startswith("boltz_results_batch_")
            out = p.parent.parent.parent.parent
            assert out.name == "out_" + batch[len("boltz_results_batch_"):]
            assert out.parent == shared / "predictions"
        assert not any((shared / "batches").iterdir())

    def test_oom_molecule_gets_skip_reason(self, pool_client):
        client, _, msa = pool_client
        _WritingBackend.oom_molecules = {"mol_1"}
        resp = client.post("/score", json={
            "smiles": ["CCO", "CCN"],
            "params": {"protein_sequence": "MVLSPADKTN", "msa_path": msa},
        })
        data = resp.json()
        assert data["scores"][0] == pytest.approx(-5.0)
        assert data["scores"][1] is None
        assert "GPU out of memory" in data["metadata"]["skip_reason"][1]
        assert data["artifacts"]["structures"][1] is None

    def test_template_reaches_worker_yaml(self, pool_client):
        client, shared, msa = pool_client
        (shared / "tpl.cif").write_text("data_x\n")
        seen: list[dict] = []
        original = _WritingBackend.call

        def spy(self, payload, timeout):
            seen.append(yaml.safe_load(Path(payload["yaml_path"]).read_text()))
            return original(self, payload, timeout)

        with patch.object(_WritingBackend, "call", spy):
            resp = client.post("/score", json={
                "smiles": ["CCO"],
                "params": {
                    "protein_sequence": "MVLSPADKTN", "msa_path": msa,
                    "templates": [{"path": "tpl.cif", "chain_id": "A"}],
                },
            })
        assert resp.status_code == 200
        assert seen[0]["templates"] == [
            {"cif": str(shared / "tpl.cif"), "chain_id": "A", "template_id": "A"},
        ]

    def test_concurrent_requests_all_succeed(self, pool_client):
        client, _, msa = pool_client
        body = {
            "smiles": ["CCO", "CCN"],
            "params": {"protein_sequence": "MVLSPADKTN", "msa_path": msa},
        }
        results: list = [None] * 4

        def post(i):
            results[i] = client.post("/score", json=body)

        threads = [threading.Thread(target=post, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r.status_code == 200 for r in results)
        assert all(r.json()["scores"][0] == pytest.approx(-5.0) for r in results)

    def test_falls_back_to_subprocess_when_workers_failed(self, tmp_path):
        msa = tmp_path / "t.a3m"
        msa.write_text(">q\nMVLSPADKTN\n")
        FakeBackend.fail_start = {"gpu0-w0", "gpu0-w1"}
        fallback = boltz_app.ScoreResponse(
            scores=[-7.0], metadata={"skip_reason": [""]}, artifacts={"structures": ["/x.pdb"]},
        )
        with patch.object(boltz_app, "SHARED_DIR", tmp_path), \
                patch.object(boltz_app, "BOLTZ_WORKERS_PER_GPU", 2), \
                patch.object(boltz_app, "_visible_gpu_devices", return_value=[0]), \
                patch.object(boltz_app.worker_pool, "ProcessBackend", FakeBackend), \
                patch.object(boltz_app, "_run_prediction", return_value=fallback) as run_pred:
            with TestClient(boltz_app.app) as client:
                for _ in range(100):
                    if all(w["state"] == "failed" for w in client.get("/health").json()["workers"]):
                        break
                    time.sleep(0.02)
                resp = client.post("/score", json={
                    "smiles": ["CCO"],
                    "params": {"protein_sequence": "MVLSPADKTN", "msa_path": str(msa)},
                })
        boltz_app._pool = None
        assert resp.status_code == 200
        assert resp.json()["scores"] == [-7.0]
        run_pred.assert_called_once()

    def test_no_pool_keeps_plain_health(self):
        boltz_app._pool = None
        resp = TestClient(boltz_app.app).get("/health")
        assert resp.json() == {"status": "ok"}
