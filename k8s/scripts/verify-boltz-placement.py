#!/usr/bin/env python3
"""Prove the gateway places batches on idle Boltz replicas, and queues the rest.

A Boltz replica runs one prediction at a time (``BOLTZ_MAX_CONCURRENCY=1``), so
replicas are not interchangeable mid-flight: handing a six-minute batch to a
busy replica while another sits idle costs a whole RL step. A plain ClusterIP
Service does exactly that, because kube-proxy round-robins without knowing who
is busy. ``dd-agentic-boltz-proxy`` is HAProxy with ``balance leastconn`` and
``maxconn 1`` per replica, which makes least-connections mean least-busy and
turns a surplus request into a queued one rather than a doubled-up one.

This fires several ``/score`` calls at once and checks three things that are
easy to assume and easy to get wrong:

  1. **Placement.** While an idle replica exists, no replica is given a second
     concurrent batch. Read from the ``X-Boltz-Replica`` response header the
     gateway stamps, and corroborated by the per-server in-flight counter on
     the stats endpoint.
  2. **Parallelism.** M replicas really do M batches at once, so the wall clock
     for M concurrent batches is about one batch, not M. This is the number
     ``M ~ 1.3 x N`` is derived from.
  3. **Queueing, not failure.** With more requests than replicas the surplus
     waits in the gateway queue and still succeeds. A 503 here would mean the
     queue timeout is below the prediction time, which reads downstream as a
     component failure and poisons the RL signal with NaN.

Defaults are deliberately cheap: one molecule per call at reduced fidelity.
This measures *placement*, not latency -- use ab-boltz-kernels.py for the
per-batch wall clock that replica counts are sized on.

    kubectl --namespace demo-lux port-forward \
        deployment/dd-agentic-boltz-proxy 8091:8091 8404:8404 &
    ./k8s/scripts/verify-boltz-placement.py --namespace demo-lux

Exit status: 0 placement proven, 1 disproven, 2 inconclusive (nothing ran).

No third-party deps.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_URL = "http://localhost:8091"
DEFAULT_STATS_URL = "http://localhost:8404"
# The backend stanza in the gateway's haproxy.cfg, which is what the stats rows
# are keyed by.
DEFAULT_BACKEND = "boltz"
# Above the service's own 1800s ceiling and the gateway's queue timeout, so
# this harness is never the thing that gives up first.
DEFAULT_TIMEOUT = 2100.0

# The header the gateway stamps with the replica that served the request.
REPLICA_HEADER = "X-Boltz-Replica"

# Same target and precomputed alignment as the other harnesses, so no call
# pays an MSA-server round trip.
BRD4_BD1_SEQUENCE = (
    "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKK"
    "RLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE"
)
DEFAULT_MSA_PATH = "brd4_bd1.a3m"

# One ligand per call. Distinct molecules so a result cache, when one lands,
# cannot turn these into cache hits and hide the placement being measured.
PANEL = [
    ("JQ1", "CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2c1ccc(Cl)cc1"),
    ("birabresib", "CC(=O)Nc1ccc(cc1)C1=NC(CC(=O)N)c2nnc(C)n2-c2sc(C)c(C)c21"),
    ("celecoxib", "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F"),
    ("olaparib", "O=C(c1ccc(CC2=NNC(=O)c3ccccc32)cc1F)N1CCN(C(=O)C2CC2)CC1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

@dataclass
class Call:
    label: str
    started: float = 0.0
    finished: float = 0.0
    status: Optional[int] = None
    replica: Optional[str] = None
    scores: list[Any] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def elapsed(self) -> float:
        return self.finished - self.started

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200


def post_score(url: str, payload: dict[str, Any], timeout: float, call: Call) -> None:
    """POST /score, recording the replica that answered on *call*."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    call.started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            call.status = response.status
            call.replica = response.headers.get(REPLICA_HEADER)
            call.scores = (json.loads(body) or {}).get("scores") or []
    except urllib.error.HTTPError as exc:
        call.status = exc.code
        call.replica = exc.headers.get(REPLICA_HEADER)
        call.error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
    except (urllib.error.URLError, OSError) as exc:
        call.error = f"request failed: {exc}"
    except ValueError as exc:
        call.error = f"unparseable response: {exc}"
    finally:
        call.finished = time.monotonic()


# ---------------------------------------------------------------------------
# Gateway stats
# ---------------------------------------------------------------------------

@dataclass
class ServerStat:
    name: str
    status: str
    in_flight: int
    queued: int


def read_stats(
    stats_url: str,
    backend: str,
    timeout: float = 5.0,
) -> tuple[list[ServerStat], Optional[ServerStat]]:
    """``(per-replica rows, backend row)`` from HAProxy's CSV stats.

    The backend row matters as much as the servers: a request that cannot be
    placed yet waits in the *backend* pending queue, so its ``qcur`` is where
    queueing shows up -- the per-server ``qcur`` stays 0 throughout.
    """
    try:
        with urllib.request.urlopen(f"{stats_url}/stats;csv", timeout=timeout) as response:
            text = response.read().decode()
    except (urllib.error.URLError, OSError):
        return [], None

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], None
    # The header row is "# pxname,svname,qcur,...": index by name, since the
    # column count varies between HAProxy versions.
    columns = lines[0].lstrip("# ").split(",")
    try:
        px, sv = columns.index("pxname"), columns.index("svname")
        qcur, scur, status = (
            columns.index("qcur"), columns.index("scur"), columns.index("status"),
        )
    except ValueError:
        return [], None

    servers: list[ServerStat] = []
    pending: Optional[ServerStat] = None
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) <= max(px, sv, qcur, scur, status):
            continue
        if fields[px] != backend:
            continue
        row = ServerStat(
            name=fields[sv],
            status=fields[status],
            in_flight=int(fields[scur] or 0),
            queued=int(fields[qcur] or 0),
        )
        if row.name == "BACKEND":
            pending = row
        elif row.name != "FRONTEND":
            servers.append(row)
    return servers, pending


class StatsSampler:
    """Polls the stats endpoint during the run and keeps the high-water marks."""

    def __init__(self, stats_url: str, backend: str, interval: float = 1.0):
        self.stats_url = stats_url
        self.backend = backend
        self.interval = interval
        self.max_in_flight: dict[str, int] = {}
        self.max_queued = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            servers, pending = read_stats(self.stats_url, self.backend)
            for stat in servers:
                previous = self.max_in_flight.get(stat.name, 0)
                self.max_in_flight[stat.name] = max(previous, stat.in_flight)
                self.max_queued = max(self.max_queued, stat.queued)
            if pending is not None:
                # Where an unplaceable request actually waits.
                self.max_queued = max(self.max_queued, pending.queued)
            self.samples += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> StatsSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    @property
    def doubled_up(self) -> list[str]:
        """Replicas seen running more than one batch at once."""
        return [name for name, peak in self.max_in_flight.items() if peak > 1]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def build_payload(args: argparse.Namespace, smiles: list[str]) -> dict[str, Any]:
    return {
        "smiles": smiles,
        "params": {
            "protein_sequence": args.protein_sequence,
            "msa_path": args.msa_path,
            "recycling_steps": args.recycling_steps,
            "diffusion_samples": 1,
            "sampling_steps": args.sampling_steps,
            "sampling_steps_affinity": args.sampling_steps_affinity,
            "diffusion_samples_affinity": args.diffusion_samples_affinity,
            "output_format": "pdb",
        },
    }


def ready_replicas(stats_url: str, backend: str) -> list[str]:
    servers, _pending = read_stats(stats_url, backend)
    return [s.name for s in servers if s.status.startswith("UP")]


def fire_concurrently(args: argparse.Namespace, batch: list[tuple[str, str]]) -> list[Call]:
    calls = [Call(label=name) for name, _ in batch]
    threads = [
        threading.Thread(
            target=post_score,
            args=(f"{args.url}/score", build_payload(args, [smiles]), args.timeout, call),
        )
        for (_, smiles), call in zip(batch, calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return calls


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render(
    calls: list[Call],
    replicas: list[str],
    sampler: StatsSampler,
    wall_clock: float,
    verdict: tuple[Optional[bool], str],
) -> None:
    print("\n" + "=" * 72)
    print(f"boltz placement | {len(calls)} concurrent /score calls | "
          f"{len(replicas)} replica(s) up")
    print("=" * 72)

    print(f"{'molecule':<14} {'replica':<10} {'status':<8} {'start':>7} {'elapsed':>9}")
    first_start = min((c.started for c in calls), default=0.0)
    for call in sorted(calls, key=lambda c: c.started):
        status = str(call.status or "-")
        print(f"{call.label[:14]:<14} {(call.replica or '?'):<10} {status:<8} "
              f"{call.started - first_start:>6.1f}s {call.elapsed:>8.1f}s")
        if call.error:
            print(f"               error: {call.error}")

    served = [c.replica for c in calls if c.ok and c.replica]
    print(f"\nreplicas used        {len(set(served))} of {len(replicas)} up "
          f"({', '.join(sorted(set(served))) or 'none'})")
    if sampler.samples:
        peaks = ", ".join(
            f"{name}={peak}" for name, peak in sorted(sampler.max_in_flight.items())
            if peak
        )
        print(f"peak in-flight       {peaks or 'nothing observed'}")
        print(f"peak queue depth     {sampler.max_queued}")
    longest = max((c.elapsed for c in calls if c.ok), default=0.0)
    if longest:
        print(f"wall clock           {wall_clock:.1f}s for {len(calls)} calls; "
              f"longest single call {longest:.1f}s")
        serial = sum(c.elapsed for c in calls if c.ok)
        print(f"speedup vs serial    {serial / wall_clock:.2f}x "
              f"({serial:.1f}s of work in {wall_clock:.1f}s)")

    proven, reason = verdict
    label = "PROVEN" if proven else ("INCONCLUSIVE" if proven is None else "NOT PROVEN")
    print(f"\n{label}: {reason}")


def decide(
    calls: list[Call],
    replicas: list[str],
    sampler: StatsSampler,
) -> tuple[Optional[bool], str]:
    if not calls:
        return None, "nothing was sent"
    if not replicas:
        return None, (
            "the gateway reports no replica up, so there was nothing to place "
            "work on. Check the stats endpoint and the boltz rollout"
        )

    failed = [c for c in calls if not c.ok]
    if failed:
        queued_out = [c for c in failed if c.status == 503]
        if queued_out:
            return False, (
                f"{len(queued_out)} call(s) got 503: the gateway gave up waiting "
                "for a free replica. Raise timeout queue above the prediction "
                "time -- a queued batch must never read as a failure"
            )
        return False, f"{len(failed)} of {len(calls)} call(s) failed outright"

    if sampler.doubled_up:
        return False, (
            f"{', '.join(sampler.doubled_up)} ran more than one batch at once, "
            "so maxconn 1 is not holding and a replica can be handed work while "
            "it is already predicting"
        )

    served = {c.replica for c in calls if c.replica}
    if None in served or not served:
        return None, (
            f"the gateway did not stamp {REPLICA_HEADER}, so placement cannot be "
            "read from the responses. Check it is the gateway being dialled and "
            "not a replica directly"
        )

    expected = min(len(calls), len(replicas))
    if len(served) < expected:
        return False, (
            f"{len(calls)} concurrent calls used only {len(served)} of "
            f"{len(replicas)} replicas: work is being stacked on a busy replica "
            "while another sits idle, which is what least-busy placement is for"
        )

    if len(calls) > len(replicas) and sampler.samples and not sampler.max_queued:
        return True, (
            f"every call was placed on its own replica ({len(served)} of "
            f"{len(replicas)}), though the surplus was never seen waiting -- "
            "sampling may have missed a short queue"
        )

    queued = (f", and the {len(calls) - len(replicas)} surplus call(s) waited in "
              f"the queue (depth {sampler.max_queued}) instead of doubling up"
              if len(calls) > len(replicas) else "")
    return True, (
        f"{len(calls)} concurrent batches were spread over {len(served)} of "
        f"{len(replicas)} replicas, one at a time each{queued}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="proxy base URL (NOT a replica, and not the caching "
                             "gateway: placement is what is measured)")
    parser.add_argument("--stats-url", default=DEFAULT_STATS_URL,
                        help="proxy stats base URL")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="HAProxy backend name carrying the replicas")
    parser.add_argument("--namespace", default=None,
                        help="reported for context only; no kubectl is required")
    parser.add_argument("--requests", type=int, default=None,
                        help="concurrent /score calls (default: one more than "
                             "the replicas up, so the queue is exercised too)")
    parser.add_argument("--msa-path", default=DEFAULT_MSA_PATH)
    parser.add_argument("--protein-sequence", default=BRD4_BD1_SEQUENCE)
    # Cheap by default: this measures placement, not per-batch latency.
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-steps-affinity", type=int, default=50)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="above the service's own 1800s cap")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="how often to poll the stats endpoint")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    args.url = args.url.rstrip("/")
    args.stats_url = args.stats_url.rstrip("/")

    replicas = ready_replicas(args.stats_url, args.backend)
    if not replicas:
        print(f"no replica reported up at {args.stats_url}/stats;csv -- is the "
              f"port-forward up and is {args.backend!r} the backend name?",
              file=sys.stderr)
        return 2

    wanted = args.requests if args.requests is not None else len(replicas) + 1
    if wanted < 1:
        parser.error("--requests must be at least 1")
    if wanted > len(PANEL):
        parser.error(f"--requests {wanted} exceeds the {len(PANEL)} molecules in "
                     "the panel; distinct molecules keep a result cache from "
                     "answering these instead of a replica")
    batch = PANEL[:wanted]

    print(f"{len(replicas)} replica(s) up: {', '.join(replicas)}")
    print(f"firing {wanted} concurrent /score calls at {args.url} "
          f"(1 molecule each, reduced fidelity)", flush=True)

    start = time.monotonic()
    with StatsSampler(args.stats_url, args.backend, args.sample_interval) as sampler:
        calls = fire_concurrently(args, batch)
    wall_clock = time.monotonic() - start

    proven, reason = decide(calls, replicas, sampler)
    render(calls, replicas, sampler, wall_clock, (proven, reason))

    if args.json:
        print()
        json.dump(
            {
                "proven": proven,
                "reason": reason,
                "namespace": args.namespace,
                "replicas_up": replicas,
                "wall_clock_seconds": wall_clock,
                "peak_queue_depth": sampler.max_queued,
                "peak_in_flight": sampler.max_in_flight,
                "calls": [
                    {"molecule": c.label, "replica": c.replica, "status": c.status,
                     "elapsed_seconds": c.elapsed, "scores": c.scores,
                     "error": c.error}
                    for c in calls
                ],
            },
            sys.stdout,
            indent=2,
        )
        print()

    if proven is None:
        return 2
    return 0 if proven else 1


if __name__ == "__main__":
    sys.exit(main())
