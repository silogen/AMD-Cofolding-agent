# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke-test + speed-benchmark the interaction-profiler REST service.

Starts (or connects to) the FastAPI service, runs a functional smoke test
(``/health``, ``/profile`` on a SMILES, ``/analyze`` on structures), then
benchmarks it over a validation batch:

* **sequential** - one request at a time via ``/profile``; measures
  per-structure latency (mean / median / p95) and correctness.
* **concurrent** - the same molecules fired with a thread pool against
  ``/profile``; measures end-to-end throughput and the concurrency
  speed-up (exercises FastAPI's sync-endpoint thread pool).

Writes a JSON report and prints a summary. Structure paths in the batch
are resolved to absolute paths client-side, so the service needs no
shared volume for the benchmark.

Usage::

    python scripts/service_benchmark.py \
        --batch examples/example_batch.json \
        --concurrency 8 --out /tmp/service_benchmark.json

For a heavier benchmark, point ``--batch`` at your own JSON of many
structures (same schema as ``examples/example_batch.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def _wait_health(client: httpx.Client, base: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = client.get(f"{base}/health", timeout=5.0)
            if r.status_code == 200:
                return r.json()
            last = r.text
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"service did not become healthy in {timeout}s: {last}")


def _load_specs(batch_path: Path, limit: int | None) -> list[dict]:
    data = json.loads(batch_path.read_text())
    items = data.get("molecules", data) if isinstance(data, dict) else data
    base_dir = batch_path.parent
    specs: list[dict] = []
    for it in items:
        structure = it.get("structure") or it.get("structure_path")
        if structure:
            p = Path(structure)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            structure = str(p)
        specs.append(
            {
                "name": it.get("name") or it.get("id"),
                "smiles": it.get("smiles"),
                "structure": structure,
                "ligand_resname": it.get("ligand_resname"),
                "ligand_chain": it.get("ligand_chain"),
                "ligand_keyword": it.get("ligand_keyword"),
                "peptide_ligand_chain": it.get("peptide_ligand_chain"),
                "source": it.get("source"),
            }
        )
    if limit:
        specs = specs[:limit]
    return specs


def _profile_one(client: httpx.Client, base: str, spec: dict, opts: dict) -> dict:
    t0 = time.time()
    try:
        r = client.post(
            f"{base}/profile",
            json={"molecule": spec, "options": opts},
            timeout=300.0,
        )
        dt = time.time() - t0
        if r.status_code != 200:
            return {"name": spec["name"], "seconds": dt, "ok": False,
                    "http": r.status_code, "detail": r.text[:200]}
        payload = r.json()
        results = payload.get("results", [])
        n_int = sum(res.get("metadata", {}).get("n_interactions", 0) for res in results)
        n_trip = sum(res.get("metadata", {}).get("n_triples", 0) for res in results)
        errs = [res.get("error") for res in results if res.get("error")]
        server_secs = sum(res.get("metadata", {}).get("timing_seconds") or 0.0 for res in results)
        return {
            "name": spec["name"], "seconds": dt, "server_seconds": server_secs,
            "ok": not errs, "n_results": len(results),
            "n_interactions": n_int, "n_triples": n_trip,
            "error": errs[0] if errs else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": spec["name"], "seconds": time.time() - t0, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _summary(times: list[float]) -> dict:
    if not times:
        return {}
    s = sorted(times)
    return {
        "n": len(s),
        "total_s": round(sum(s), 3),
        "mean_s": round(statistics.mean(s), 3),
        "median_s": round(statistics.median(s), 3),
        "p95_s": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
        "min_s": round(s[0], 3),
        "max_s": round(s[-1], 3),
    }


def run_smoke(client: httpx.Client, base: str, sample_structure: dict | None) -> dict:
    out: dict = {}
    out["health"] = client.get(f"{base}/health").json()
    schema = client.get(f"{base}/schema").json()
    out["schema_ok"] = schema.get("title") == "BatchResult" or "properties" in schema
    # SMILES-only -> taxonomy
    r = client.post(f"{base}/profile", json={
        "molecule": {"name": "aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        "options": {},
    }, timeout=120.0)
    smi = r.json()
    n_frag = sum(len(res.get("composition", {}).get("fragments", [])) for res in smi.get("results", []))
    out["smiles_only"] = {"http": r.status_code, "n_fragments": n_frag}
    # structure
    if sample_structure:
        r = client.post(f"{base}/analyze", json={
            "batch_id": "smoke", "molecules": [sample_structure], "options": {},
        }, timeout=300.0)
        res = r.json()
        md = (res.get("molecules") or [{}])[0].get("metadata", {})
        out["structure"] = {"http": r.status_code, "name": md.get("name"),
                            "n_interactions": md.get("n_interactions"),
                            "n_triples": md.get("n_triples")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, default=Path("examples/example_batch.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--base-url", default=None, help="Connect to an existing service instead of spawning one.")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--server-workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("examples/service_benchmark.json"))
    args = ap.parse_args()

    specs = _load_specs(args.batch, args.limit)
    opts = {"ph": 7.4, "radius": 8.0, "drop_uncertain": True}

    proc = None
    base = args.base_url
    if not base:
        base = f"http://127.0.0.1:{args.port}"
        env = dict(os.environ)
        env["IP_PORT"] = str(args.port)
        env["IP_HOST"] = "127.0.0.1"
        # App now lives in the deployment wrapper (single app per boltz2/posebusters convention).
        _svc_dir = Path(__file__).resolve().parents[2] / "services" / "interaction-profiler"
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "app:app", "--app-dir", str(_svc_dir),
             "--host", "127.0.0.1", "--port", str(args.port),
             "--workers", str(args.server_workers), "--log-level", "warning"],
            env=env,
        )

    report: dict = {"batch": str(args.batch), "n_specs": len(specs),
                    "concurrency": args.concurrency, "server_workers": args.server_workers}
    limits = httpx.Limits(max_connections=args.concurrency + 4, max_keepalive_connections=args.concurrency + 4)
    try:
        with httpx.Client(limits=limits) as client:
            health = _wait_health(client, base)
            report["health"] = health
            print(f"[health] {health}")

            sample = next((s for s in specs if s.get("structure")), None)
            report["smoke"] = run_smoke(client, base, sample)
            print(f"[smoke] {json.dumps(report['smoke'])}")

            # warm-up (excluded from timings): first structure primes caches/imports
            if sample:
                _profile_one(client, base, sample, opts)

            # --- sequential ---
            seq_rows = []
            t0 = time.time()
            for spec in specs:
                seq_rows.append(_profile_one(client, base, spec, opts))
            seq_wall = time.time() - t0
            report["sequential"] = {
                "wall_s": round(seq_wall, 3),
                "throughput_per_min": round(60.0 * len(specs) / seq_wall, 2) if seq_wall else None,
                "latency": _summary([r["seconds"] for r in seq_rows]),
                "n_ok": sum(1 for r in seq_rows if r["ok"]),
                "n_err": sum(1 for r in seq_rows if not r["ok"]),
                "rows": seq_rows,
            }
            print(f"[sequential] wall={seq_wall:.1f}s "
                  f"ok={report['sequential']['n_ok']} err={report['sequential']['n_err']} "
                  f"tput={report['sequential']['throughput_per_min']}/min")

            # --- concurrent ---
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                con_rows = list(ex.map(lambda s: _profile_one(client, base, s, opts), specs))
            con_wall = time.time() - t0
            report["concurrent"] = {
                "wall_s": round(con_wall, 3),
                "throughput_per_min": round(60.0 * len(specs) / con_wall, 2) if con_wall else None,
                "speedup_vs_sequential": round(seq_wall / con_wall, 2) if con_wall else None,
                "latency": _summary([r["seconds"] for r in con_rows]),
                "n_ok": sum(1 for r in con_rows if r["ok"]),
                "n_err": sum(1 for r in con_rows if not r["ok"]),
            }
            print(f"[concurrent x{args.concurrency}] wall={con_wall:.1f}s "
                  f"speedup={report['concurrent']['speedup_vs_sequential']}x "
                  f"tput={report['concurrent']['throughput_per_min']}/min")
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
