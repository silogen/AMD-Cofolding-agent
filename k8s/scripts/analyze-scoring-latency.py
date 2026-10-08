#!/usr/bin/env python3
"""Harvest scorer latency and cache hit rate from external_scoring.jsonl.

ExternalRESTScorer appends one JSON record per external call, carrying
``elapsed_seconds``, the service name, the status code, and the full request
(so batch size is recoverable from ``request.smiles``). Prior campaigns have
therefore already measured the two numbers the fleet is sized on -- no GPU
benchmark run is needed:

  * ``t_boltz_per_batch`` -- wall clock for one /score call on one replica,
    reported per batch size because it scales with the batch.
  * ``cache_hit_rate``    -- SMILES seen more than once under the same protein
    sequence, i.e. the share of molecule-level demand a result cache serves.

Records with ``error`` set or a non-2xx ``status_code`` are excluded from the
latency distribution: a prediction is killed at PREDICT_TIMEOUT_SECONDS and a
capped timeout would badly skew p95. They are still counted and broken out by
reason.

The log covers every external service, so PoseBusters, ADMET, RetroDMTA, and
the interaction profiler are characterized at the same time -- which is what
confirms or refutes the assumption that they are not bottlenecks.

Read the log straight off the cluster:

    kubectl --namespace demo-lux exec deployment/dd-agentic --container dd-agent \
        -- cat /data/reinvent4_mcp/logs/external_scoring.jsonl \
        | k8s/scripts/analyze-scoring-latency.py -

No third-party deps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_LOG = Path("/data/reinvent4_mcp/logs/external_scoring.jsonl")
LOG_FILENAME = "external_scoring.jsonl"

# The service identifier TTPs use for cofolding is "boltz" (not "boltz2");
# see reinvent4_mcp/ttp/schema.py KNOWN_EXTERNAL_SERVICES.
DEFAULT_BOLTZ_SERVICE = "boltz"

# services/boltz2/app.py kills a prediction at this many seconds and
# ExternalRESTScorer gives up at its own matching timeout, so anything at or
# above it is a cap rather than a measurement.
PREDICT_TIMEOUT_SECONDS = 1800

# Params that vary per call rather than per prediction, excluded from the
# cache-key digest so a batch's structure handoff does not read as a different
# fidelity setting.
PER_CALL_PARAM_KEYS = frozenset({"smiles", "structure_paths", "protein_pdb_path"})

# A non-Boltz service is worth calling out once its p95 reaches this share of
# Boltz's, since the plan assumes the CPU scorers are negligible.
NOT_NEGLIGIBLE_RATIO = 0.10


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

@dataclass
class Call:
    """One ExternalRESTScorer -> service call, as logged."""

    service: str
    elapsed: Optional[float]
    batch_size: int
    ok: bool
    failure: str
    timestamp: Optional[datetime]
    protein_sequence: str
    params_digest: str
    smiles: list[str]


def _params_digest(params: dict[str, Any]) -> str:
    """Stable digest of the params that decide the prediction.

    Mirrors the cache key the gateway will use (protein sequence, MSA, and
    predict params) minus the per-call artifact plumbing.
    """
    stable = {
        key: value
        for key, value in sorted(params.items())
        if key not in PER_CALL_PARAM_KEYS
    }
    return json.dumps(stable, sort_keys=True, default=str)


def _parse_timestamp(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _classify(record: dict[str, Any], elapsed: Optional[float]) -> tuple[bool, str]:
    """Return ``(ok, failure_reason)`` for one record.

    ExternalRESTScorer writes ``error: "HTTP <code>"`` for a non-2xx response
    and ``error: "request_failed"`` for an exception (timeout, connection
    refused, DNS). Both must stay out of the latency distribution.
    """
    status = record.get("status_code")
    error = record.get("error")

    if error:
        if isinstance(error, str) and error.startswith("HTTP "):
            return False, f"http_{error[5:].strip()}"
        if elapsed is not None and elapsed >= PREDICT_TIMEOUT_SECONDS * 0.98:
            return False, "timeout"
        return False, "request_failed"

    if not isinstance(status, int):
        return False, "no_status_code"
    if not 200 <= status < 300:
        return False, f"http_{status}"
    if not isinstance(elapsed, (int, float)):
        return False, "no_elapsed_seconds"
    return True, ""


def parse_record(record: dict[str, Any]) -> Call:
    request = record.get("request") or {}
    params = request.get("params") or {}
    smiles = request.get("smiles") or []
    if not isinstance(smiles, list):
        smiles = []

    raw_elapsed = record.get("elapsed_seconds")
    elapsed = float(raw_elapsed) if isinstance(raw_elapsed, (int, float)) else None
    ok, failure = _classify(record, elapsed)

    sequence = params.get("protein_sequence") or ""
    if isinstance(sequence, str):
        sequence = sequence.strip().upper()
    else:
        sequence = ""

    return Call(
        service=str(record.get("service") or "unknown"),
        elapsed=elapsed,
        batch_size=len(smiles),
        ok=ok,
        failure=failure,
        timestamp=_parse_timestamp(record.get("timestamp")),
        protein_sequence=sequence,
        params_digest=_params_digest(params) if isinstance(params, dict) else "",
        smiles=[s for s in smiles if isinstance(s, str)],
    )


def _open_sources(paths: list[Path]) -> Iterator[tuple[str, Iterator[str]]]:
    for path in paths:
        if str(path) == "-":
            yield "<stdin>", iter(sys.stdin)
            continue
        target = path / LOG_FILENAME if path.is_dir() else path
        with open(target, errors="replace") as handle:
            yield str(target), handle


def load_calls(paths: list[Path], since: Optional[datetime]) -> tuple[list[Call], int, list[str]]:
    """Return ``(calls, unparseable_lines, sources_read)``."""
    calls: list[Call] = []
    unparseable = 0
    sources: list[str] = []

    for name, lines in _open_sources(paths):
        sources.append(name)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                unparseable += 1
                continue
            if not isinstance(record, dict):
                unparseable += 1
                continue
            call = parse_record(record)
            if since and call.timestamp and call.timestamp < since:
                continue
            calls.append(call)

    return calls, unparseable, sources


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile of an unsorted list; ``q`` in [0, 1]."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


@dataclass
class ServiceReport:
    service: str
    n_ok: int = 0
    n_failed: int = 0
    failures: Counter = field(default_factory=Counter)
    batch: dict[str, Optional[float]] = field(default_factory=dict)
    per_molecule: dict[str, Optional[float]] = field(default_factory=dict)
    batch_sizes: Counter = field(default_factory=Counter)
    by_batch_size: dict[int, dict[str, Optional[float]]] = field(default_factory=dict)
    modal_batch_size: Optional[int] = None
    n_sequences: int = 0
    molecule_calls: int = 0
    unique_molecules: int = 0
    unique_molecules_strict: int = 0
    cache_hit_rate: Optional[float] = None
    cache_hit_rate_strict: Optional[float] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None


def build_report(service: str, calls: list[Call]) -> ServiceReport:
    report = ServiceReport(service=service)

    ok_calls = [c for c in calls if c.ok and c.elapsed is not None]
    report.n_ok = len(ok_calls)
    report.n_failed = len(calls) - len(ok_calls)
    for call in calls:
        if not call.ok:
            report.failures[call.failure] += 1

    stamps = [c.timestamp for c in calls if c.timestamp]
    if stamps:
        report.first_seen = min(stamps)
        report.last_seen = max(stamps)

    report.batch = summarize([c.elapsed for c in ok_calls])
    report.per_molecule = summarize(
        [c.elapsed / c.batch_size for c in ok_calls if c.batch_size > 0]
    )

    per_size: dict[int, list[float]] = defaultdict(list)
    for call in ok_calls:
        report.batch_sizes[call.batch_size] += 1
        per_size[call.batch_size].append(call.elapsed)
    report.by_batch_size = {size: summarize(vals) for size, vals in per_size.items()}
    if report.batch_sizes:
        # Ties broken towards the larger batch: the production batch is the
        # big one, and a tie with a stray small batch should not win.
        report.modal_batch_size = max(
            report.batch_sizes, key=lambda size: (report.batch_sizes[size], size)
        )

    # Cache analysis runs over every call carrying SMILES, successful or not:
    # a gateway cache sits in front of the whole demand stream, and the first
    # occurrence of each distinct molecule is the miss that populates it.
    loose: set[tuple[str, str]] = set()
    strict: set[tuple[str, str, str]] = set()
    sequences: set[str] = set()
    molecule_calls = 0
    for call in calls:
        sequences.add(call.protein_sequence)
        for smi in call.smiles:
            molecule_calls += 1
            loose.add((call.protein_sequence, smi))
            strict.add((call.protein_sequence, call.params_digest, smi))

    report.n_sequences = len({s for s in sequences if s})
    report.molecule_calls = molecule_calls
    report.unique_molecules = len(loose)
    report.unique_molecules_strict = len(strict)
    if molecule_calls:
        report.cache_hit_rate = (molecule_calls - len(loose)) / molecule_calls
        report.cache_hit_rate_strict = (molecule_calls - len(strict)) / molecule_calls

    return report


# ---------------------------------------------------------------------------
# Boltz sizing
# ---------------------------------------------------------------------------

def size_fleet(
    t_batch: float,
    cache_hit_rate: float,
    supervisors: int,
    step_seconds: float,
    campaign_seconds: float,
    replicas: int,
    step_counts: list[int],
) -> dict[str, Any]:
    """Derive the replica counts and campaign lengths that scale off t_batch.

    Uses the admission-control multiplier from the plan, ``(1 + hit_rate)``.
    The exact rate at which a cache lets a replica serve demand is
    ``1 / (1 - hit_rate)``, which is larger, so every count here is the
    conservative one.
    """
    multiplier = 1.0 + cache_hit_rate
    exact_multiplier = 1.0 / (1.0 - cache_hit_rate) if cache_hit_rate < 1 else math.inf

    # Demand a replica can absorb, in batches per second, and what one
    # supervisor asks for at one batch per RL step.
    replica_throughput = multiplier / t_batch
    per_supervisor_demand = 1.0 / step_seconds

    m_required = math.ceil(per_supervisor_demand * supervisors / replica_throughput)
    m_required_no_cache = math.ceil(supervisors * t_batch / step_seconds)
    n_supported = math.floor(replicas * replica_throughput / per_supervisor_demand)

    # A dedicated replica advances one step per batch; the cache shortens the
    # effective step.
    effective_step = t_batch / multiplier
    return {
        "t_batch_seconds": t_batch,
        "cache_hit_rate": cache_hit_rate,
        "cache_multiplier_plan": multiplier,
        "cache_multiplier_exact": exact_multiplier,
        "replica_batches_per_hour": replica_throughput * 3600,
        "replicas_per_supervisor": t_batch / step_seconds,
        "target_step_seconds": step_seconds,
        "supervisors": supervisors,
        "m_required": m_required,
        "m_required_no_cache": m_required_no_cache,
        "replicas": replicas,
        "n_supported": n_supported,
        "effective_step_seconds": effective_step,
        "campaign_seconds": campaign_seconds,
        "max_steps_in_campaign": math.floor(campaign_seconds / effective_step),
        "max_steps_in_campaign_dedicated": math.floor(campaign_seconds / t_batch),
        "campaign_hours_for_steps": {
            steps: {
                "dedicated": steps * t_batch / 3600,
                "with_cache": steps * effective_step / 3600,
            }
            for steps in step_counts
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value >= 120:
        return f"{value:.1f}s ({value / 60:.1f} min)"
    return f"{value:.1f}s"


def fmt_stats(stats: dict[str, Optional[float]]) -> str:
    if not stats.get("n"):
        return "no successful calls"
    return (
        f"p50 {fmt_seconds(stats['p50'])}   p95 {fmt_seconds(stats['p95'])}   "
        f"mean {fmt_seconds(stats['mean'])}   "
        f"min {stats['min']:.1f}s   max {stats['max']:.1f}s"
    )


def render_service(report: ServiceReport, out) -> None:
    total = report.n_ok + report.n_failed
    fail_pct = (report.n_failed / total * 100) if total else 0.0

    print(f"\n{report.service}", file=out)
    print(f"  calls            {report.n_ok} ok, {report.n_failed} failed "
          f"({fail_pct:.1f}%)", file=out)
    if report.failures:
        breakdown = ", ".join(
            f"{reason}: {count}" for reason, count in report.failures.most_common()
        )
        print(f"  failures         {breakdown}  (excluded from latency)", file=out)
    if report.first_seen and report.last_seen:
        print(f"  window           {report.first_seen.isoformat(timespec='seconds')}"
              f" .. {report.last_seen.isoformat(timespec='seconds')}", file=out)

    print(f"  per batch        {fmt_stats(report.batch)}", file=out)
    print(f"  per molecule     {fmt_stats(report.per_molecule)}", file=out)

    if report.batch_sizes:
        sizes = ", ".join(
            f"{size}: {count}"
            for size, count in sorted(report.batch_sizes.items(), key=lambda kv: -kv[1])
        )
        print(f"  batch sizes      {sizes}", file=out)
        for size in sorted(report.by_batch_size):
            stats = report.by_batch_size[size]
            marker = "  <- modal" if size == report.modal_batch_size else ""
            print(f"    size {size:<4}     n={stats['n']:<5} "
                  f"p50 {fmt_seconds(stats['p50'])}   "
                  f"p95 {fmt_seconds(stats['p95'])}{marker}", file=out)

    if report.cache_hit_rate is None:
        print("  cache hit rate   n/a (no SMILES logged)", file=out)
    else:
        print(f"  cache hit rate   {report.cache_hit_rate:.3f} on sequence+SMILES, "
              f"{report.cache_hit_rate_strict:.3f} including predict params", file=out)
        print(f"  molecule calls   {report.molecule_calls} total, "
              f"{report.unique_molecules} distinct "
              f"({report.unique_molecules_strict} distinct at full cache key), "
              f"{report.n_sequences} protein sequence(s)", file=out)


def render_sizing(sizing: dict[str, Any], out) -> None:
    print("\nBoltz fleet sizing", file=out)
    print(f"  t_boltz_per_batch        {fmt_seconds(sizing['t_batch_seconds'])}", file=out)
    print(f"  cache_hit_rate           {sizing['cache_hit_rate']:.3f} "
          f"-> multiplier {sizing['cache_multiplier_plan']:.2f} "
          f"(exact serve rate would be {sizing['cache_multiplier_exact']:.2f}, "
          f"so these counts are conservative)", file=out)
    print(f"  per-replica throughput   "
          f"{sizing['replica_batches_per_hour']:.2f} batches/hour", file=out)
    print(f"  replicas per supervisor  {sizing['replicas_per_supervisor']:.2f}  "
          f"(plan assumes M ~ 2N at a "
          f"{sizing['target_step_seconds'] / 60:.0f} min step)", file=out)
    print(f"  M for N={sizing['supervisors']:<3}             "
          f"{sizing['m_required']} boltz replicas "
          f"({sizing['m_required_no_cache']} before caching)", file=out)
    print(f"  N at M={sizing['replicas']:<3}              "
          f"{sizing['n_supported']} supervisors", file=out)
    print(f"  effective step           "
          f"{fmt_seconds(sizing['effective_step_seconds'])} (one batch, with cache)",
          file=out)
    print(f"  campaign window          "
          f"{sizing['campaign_seconds'] / 3600:.1f} h -> "
          f"{sizing['max_steps_in_campaign_dedicated']} steps on a dedicated "
          f"replica, {sizing['max_steps_in_campaign']} with cache", file=out)
    for steps, hours in sorted(sizing["campaign_hours_for_steps"].items()):
        print(f"    {steps:<4} steps        {hours['dedicated']:.1f} h dedicated, "
              f"{hours['with_cache']:.1f} h with cache", file=out)


def render_warnings(warnings: list[str], out) -> None:
    if not warnings:
        return
    print("\nWarnings", file=out)
    for warning in warnings:
        print(f"  ! {warning}", file=out)


def render_fallback(out) -> None:
    print(
        "\nNo usable latency records. Fall back to one measured /score call "
        "against BRD4 BD1 with the precomputed MSA -- one call, no MSA-server "
        "round trip:\n"
        "    k8s/scripts/ab-boltz-kernels.py --baseline-only\n"
        "cache_hit_rate then starts as an assumption and self-corrects from "
        "the gateway's /stats.",
        file=out,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_warnings(
    reports: dict[str, ServiceReport],
    boltz: Optional[ServiceReport],
    sizing: Optional[dict[str, Any]],
    min_campaign_steps: int,
) -> list[str]:
    warnings: list[str] = []

    for report in reports.values():
        total = report.n_ok + report.n_failed
        if total and report.n_failed / total > 0.10:
            warnings.append(
                f"{report.service}: {report.n_failed}/{total} calls failed; "
                "the harvested latency describes only the calls that survived"
            )

    if boltz and boltz.n_ok and boltz.n_ok < 20:
        warnings.append(
            f"{boltz.service}: only {boltz.n_ok} successful calls, so p95 is "
            "weakly determined; treat the sizing as provisional"
        )

    boltz_p95 = (boltz.batch.get("p95") if boltz else None) or 0.0
    if boltz_p95:
        for report in reports.values():
            if boltz and report.service == boltz.service:
                continue
            p95 = report.batch.get("p95")
            if p95 and p95 / boltz_p95 >= NOT_NEGLIGIBLE_RATIO:
                warnings.append(
                    f"{report.service}: p95 {fmt_seconds(p95)} is "
                    f"{p95 / boltz_p95 * 100:.0f}% of Boltz's -- not negligible, "
                    "contrary to the assumption that only Boltz is a bottleneck"
                )

    if sizing and sizing["max_steps_in_campaign"] < min_campaign_steps:
        warnings.append(
            f"a campaign window fits only {sizing['max_steps_in_campaign']} steps, "
            f"below the {min_campaign_steps}-step floor for one task: the "
            "supervisor task unit has to shrink below a full RL campaign"
        )
    elif sizing and sizing["max_steps_in_campaign_dedicated"] < min_campaign_steps:
        warnings.append(
            f"a {min_campaign_steps}-step task fits the window only on the "
            f"measured cache hit rate "
            f"({sizing['max_steps_in_campaign_dedicated']} steps without it), so "
            "the task unit depends on the cache holding up across supervisors"
        )

    if boltz and boltz.n_sequences > 1:
        warnings.append(
            f"{boltz.service}: {boltz.n_sequences} distinct protein sequences in "
            "the log, so the harvested cache hit rate mixes targets; a "
            "fixed-target weight sweep should hit more often than this"
        )

    return warnings


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", type=Path, default=None,
        help=f"jsonl files, directories containing {LOG_FILENAME}, or - for "
             f"stdin (default: {DEFAULT_LOG})",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument("--service", action="append", default=None,
                        help="restrict to this service (repeatable)")
    parser.add_argument("--since", default=None,
                        help="ignore records before this ISO-8601 timestamp")
    parser.add_argument("--boltz-service", default=DEFAULT_BOLTZ_SERVICE,
                        help="service name to size the fleet against")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="read t_boltz_per_batch at this batch size "
                             "(default: the modal batch size)")
    parser.add_argument("--latency-percentile", type=float, default=50.0,
                        help="percentile of elapsed_seconds used as "
                             "t_boltz_per_batch (default: 50)")
    parser.add_argument("--supervisors", type=int, default=3,
                        help="N, for the required replica count")
    parser.add_argument("--replicas", type=int, default=5,
                        help="M, for the supported supervisor count")
    parser.add_argument("--step-minutes", type=float, default=5.0,
                        help="target wall clock per RL step")
    parser.add_argument("--campaign-hours", type=float, default=12.0,
                        help="dispatch window one supervisor task must fit in")
    parser.add_argument("--min-campaign-steps", type=int, default=30,
                        help="shortest campaign that still counts as a task unit")
    args = parser.parse_args(argv)

    paths = args.paths or [DEFAULT_LOG]
    since = None
    if args.since:
        since = _parse_timestamp(args.since)
        if since is None:
            parser.error(f"--since {args.since!r} is not an ISO-8601 timestamp")

    try:
        calls, unparseable, sources = load_calls(paths, since)
    except OSError as exc:
        print(f"could not read the scoring log: {exc}", file=sys.stderr)
        print("Pull it off the cluster first, e.g.\n"
              "    kubectl --namespace demo-lux exec deployment/dd-agentic "
              "--container dd-agent -- cat "
              f"{DEFAULT_LOG} | {Path(sys.argv[0]).name} -", file=sys.stderr)
        return 1

    if args.service:
        wanted = set(args.service)
        calls = [c for c in calls if c.service in wanted]

    by_service: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_service[call.service].append(call)

    reports = {
        service: build_report(service, service_calls)
        for service, service_calls in sorted(by_service.items())
    }
    boltz = reports.get(args.boltz_service)

    sizing: Optional[dict[str, Any]] = None
    t_batch: Optional[float] = None
    batch_size_used: Optional[int] = None
    if boltz and boltz.n_ok:
        batch_size_used = args.batch_size or boltz.modal_batch_size
        stats = boltz.by_batch_size.get(batch_size_used) if batch_size_used else None
        if stats and stats["n"]:
            t_batch = percentile(
                [c.elapsed for c in by_service[args.boltz_service]
                 if c.ok and c.elapsed is not None and c.batch_size == batch_size_used],
                args.latency_percentile / 100.0,
            )
        else:
            # Asked-for batch size never ran; fall back to the whole
            # distribution rather than reporting nothing.
            batch_size_used = None
            t_batch = percentile(
                [c.elapsed for c in by_service[args.boltz_service]
                 if c.ok and c.elapsed is not None],
                args.latency_percentile / 100.0,
            )
        if t_batch:
            sizing = size_fleet(
                t_batch=t_batch,
                cache_hit_rate=boltz.cache_hit_rate or 0.0,
                supervisors=args.supervisors,
                step_seconds=args.step_minutes * 60,
                campaign_seconds=args.campaign_hours * 3600,
                replicas=args.replicas,
                step_counts=[30, 50, 100],
            )
            sizing["batch_size"] = batch_size_used
            sizing["latency_percentile"] = args.latency_percentile

    warnings = collect_warnings(reports, boltz, sizing, args.min_campaign_steps)

    if args.json:
        payload = {
            "sources": sources,
            "records": len(calls),
            "unparseable_lines": unparseable,
            "services": {
                name: {
                    "n_ok": report.n_ok,
                    "n_failed": report.n_failed,
                    "failures": dict(report.failures),
                    "batch_seconds": report.batch,
                    "per_molecule_seconds": report.per_molecule,
                    "batch_sizes": {str(k): v for k, v in report.batch_sizes.items()},
                    "by_batch_size": {
                        str(k): v for k, v in report.by_batch_size.items()
                    },
                    "modal_batch_size": report.modal_batch_size,
                    "molecule_calls": report.molecule_calls,
                    "unique_molecules": report.unique_molecules,
                    "unique_molecules_strict": report.unique_molecules_strict,
                    "cache_hit_rate": report.cache_hit_rate,
                    "cache_hit_rate_strict": report.cache_hit_rate_strict,
                    "n_protein_sequences": report.n_sequences,
                    "first_seen": report.first_seen.isoformat() if report.first_seen else None,
                    "last_seen": report.last_seen.isoformat() if report.last_seen else None,
                }
                for name, report in reports.items()
            },
            "sizing": sizing,
            "warnings": warnings,
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        print()
    else:
        out = sys.stdout
        print(f"{LOG_FILENAME} harvest -- {len(sources)} source(s), "
              f"{len(calls)} record(s)"
              + (f", {unparseable} unparseable line(s)" if unparseable else ""),
              file=out)
        for source in sources:
            print(f"  from {source}", file=out)
        for report in reports.values():
            render_service(report, out)
        if sizing:
            render_sizing(sizing, out)
        elif boltz is None:
            print(f"\nNo {args.boltz_service!r} calls in the log, so no fleet "
                  "sizing. Services seen: "
                  f"{', '.join(reports) or 'none'}", file=out)
        render_warnings(warnings, out)

    if not calls or (boltz is not None and not boltz.n_ok) or (boltz is None):
        if not args.json:
            render_fallback(sys.stdout)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
