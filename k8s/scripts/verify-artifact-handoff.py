#!/usr/bin/env python3
"""Prove the structure handoff across pods: Boltz writes, the others read.

Boltz returns *absolute paths* on ``/shared`` and PoseBusters and the
interaction profiler re-open them by path -- no structure is ever carried in a
payload, so every consumer has to see the producer's filesystem. That was free
while all six containers shared one pod. Now each is its own Deployment, and
the handoff holds only because every ``/shared``-mounting pod is pinned to one
node (``ReadWriteOnce`` is per node, not per pod), or because Kyverno mutated
the claim to ``ReadWriteMany``. Both are assumptions about the storage class,
which is why this is worth a run rather than an argument.

The script drives the production path end to end:

  1. ``POST /score`` to boltz (pod A) for a small BRD4 BD1 batch, and keep
     ``artifacts.structures``.
  2. Audit those paths: absolute, under ``/shared``, index-aligned with the
     batch that produced them.
  3. ``POST /score`` to posebusters (pod B) with exactly the payload
     ``ExternalRESTScorer`` injects -- the full index-aligned
     ``structure_paths`` plus ``protein_pdb_path``.
  4. The same paths to the interaction profiler (pod C).
  5. Ask boltz's own pod to stat the ``_lig.sdf`` PoseBusters extracted beside
     the PDB, which proves the write direction as well as the read.

A consumer that answers ``200`` with nothing but nulls is the failure this
looks for: the path was syntactically fine and the file was not there. That is
reported separately from a consumer that rejected the request shape, because
the first is a storage problem and the second is a config one.

    # in-cluster, no port-forward: pipe the script into the agent pod
    kubectl --namespace demo-lux exec --stdin deployment/dd-agentic \
        --container dd-agent -- python - --in-cluster \
        < k8s/scripts/verify-artifact-handoff.py

    # from a workstation, one port-forward per service
    kubectl --namespace demo-lux port-forward deployment/dd-agentic-boltz-gateway 8091:8091 &
    kubectl --namespace demo-lux port-forward deployment/dd-agentic-posebusters 8082:8082 &
    kubectl --namespace demo-lux port-forward deployment/dd-agentic-interaction-profiler 8083:8083 &
    ./k8s/scripts/verify-artifact-handoff.py --namespace demo-lux

    # Compose publishes the same three ports on localhost, so the defaults
    # already rehearse this against separate containers on one shared volume
    docker compose up -d boltz posebusters interaction-profiler
    ./k8s/scripts/verify-artifact-handoff.py

``--namespace`` adds the kubectl-side checks -- which pod and node answered,
and the stat -- and is what makes a green result *cross-pod* rather than
merely green. Without it the run still exercises the handoff, but cannot tell
you it crossed a pod boundary.

Re-run after anything that moves a pod or changes the storage class. Use
``--structure``/``--smiles`` to replay against structures already on the
volume instead of paying for another prediction.

Exit status: 0 proven, 1 disproven, 2 inconclusive (nothing was produced, or a
service was unreachable).

No third-party deps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SHARED_ROOT = "/shared"

BOLTZ = "boltz"
POSEBUSTERS = "posebusters"
PROFILER = "interaction-profiler"

# Ports and Service names from k8s/manifests/configmap.yaml. Unqualified, so
# they resolve in whichever namespace apply.sh deployed into. Boltz is reached
# through its gateway, which is the path the RL loop takes.
CLUSTER_URLS = {
    BOLTZ: "http://dd-agentic-boltz-gateway:8091",
    POSEBUSTERS: "http://dd-agentic-posebusters:8082",
    PROFILER: "http://dd-agentic-interaction-profiler:8083",
}
LOCAL_URLS = {
    BOLTZ: "http://localhost:8091",
    POSEBUSTERS: "http://localhost:8082",
    PROFILER: "http://localhost:8083",
}
# Deployment -> container, for `kubectl exec` and for naming the pod that
# answered.
DEPLOYMENTS = {
    BOLTZ: ("dd-agentic-boltz", "boltz"),
    POSEBUSTERS: ("dd-agentic-posebusters", "posebusters"),
    PROFILER: ("dd-agentic-interaction-profiler", "interaction-profiler"),
}

# Same target and precomputed alignment as k8s/scripts/ab-boltz-kernels.py, so
# a run here reuses the MSA that script already left on the volume instead of
# querying the MSA server.
BRD4_BD1_SEQUENCE = (
    "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKK"
    "RLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE"
)
DEFAULT_MSA_PATH = "brd4_bd1.a3m"

# Two BET chemotypes, so both should land real contacts in the BD1 pocket: a
# zero interaction count here is a finding, not chemistry. Kept to two
# molecules because the handoff does not get more proven by a bigger batch and
# each one costs ~24 s of GPU.
PANEL: list[tuple[str, str]] = [
    ("JQ1", "CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2c1ccc(Cl)cc1"),
    ("birabresib", "CC(=O)Nc1ccc(cc1)C1=NC(CC(=O)N)c2nnc(C)n2-c2sc(C)c(C)c21"),
]

# What the ConfigMap configures for the two consumers, so the run exercises
# the settings the RL loop uses. "all" also avoids the structurally-zero trap:
# hbond counts need the waters Boltz does not model.
DEFAULT_POSEBUSTERS_MODE = "dock"
DEFAULT_INTERACTION_TYPE = "all"

# 60 heavy atoms, past the wrapper's MAX_HEAVY_ATOMS of 56, so boltz skips it
# and leaves a None hole mid-batch. ExternalRESTScorer forwards those holes
# rather than compacting the list, and the consumers pair structure_paths[i]
# with smiles[i] -- so a batch without a hole would never test the alignment
# that every partial-failure RL batch depends on.
HOLE = ("oversized-alkane", "C" * 60)
HOLE_INDEX = 1

# Status of one consumer, worst last: the verdict is decided on these.
OK = "ok"
PARTIAL = "partial"
BLIND = "blind"
REJECTED = "rejected"
UNREACHABLE = "unreachable"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def get_json(url: str, timeout: float) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode()), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"unparseable response: {exc}"


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[float, dict[str, Any], Optional[int], Optional[str]]:
    """POST and return ``(elapsed, data, status_code, error)``.

    The status code is kept even on failure: a 4xx means the consumer refused
    the request shape, which is a different diagnosis from a file it could not
    open, and the two must not be collapsed.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
            return time.monotonic() - start, json.loads(body), response.status, None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        return time.monotonic() - start, {}, exc.code, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, OSError) as exc:
        return time.monotonic() - start, {}, None, f"request failed: {exc}"
    except ValueError as exc:
        return time.monotonic() - start, {}, None, f"unparseable response: {exc}"


# ---------------------------------------------------------------------------
# kubectl: who answered, and what they can see
# ---------------------------------------------------------------------------

@dataclass
class PodRef:
    name: str
    node: str
    container: str


def kubectl(args: list[str], namespace: str, capture: bool = False) -> subprocess.CompletedProcess:
    argv = ["kubectl", "--namespace", namespace, *args]
    return subprocess.run(argv, check=True, capture_output=capture, text=True)


def pods_for(namespace: str, service: str) -> list[PodRef]:
    """Every running pod backing *service*.

    A list rather than one pod because boltz runs M replicas behind its
    gateway: which of them served a given batch is the gateway's choice, so
    naming one would be a guess.
    """
    deployment, container = DEPLOYMENTS[service]
    try:
        proc = kubectl(
            ["get", "pods",
             "--selector", f"app.kubernetes.io/name={deployment}",
             "--field-selector", "status.phase=Running",
             "--output", "json"],
            namespace, capture=True,
        )
        items = json.loads(proc.stdout).get("items") or []
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return []
    return [
        PodRef(
            name=pod["metadata"]["name"],
            node=pod["spec"].get("nodeName") or "?",
            container=container,
        )
        for pod in items
    ]


def describe_pods(pods: list[PodRef]) -> str:
    if not pods:
        return "(pods unknown: no --namespace)"
    nodes = sorted({pod.node for pod in pods})
    if len(pods) == 1:
        return f"{pods[0].name} on {nodes[0]}"
    return f"{len(pods)} replicas on {', '.join(nodes)}: " + ", ".join(
        sorted(pod.name for pod in pods)
    )


# argv[0] is "-c" under `python -c`, so the paths start at argv[1]. Reports a
# size rather than a bool so an empty file is distinguishable from a missing
# one.
_STAT_SNIPPET = (
    "import json,os,sys;"
    "print(json.dumps({p: (os.path.getsize(p) if os.path.isfile(p) else None)"
    " for p in sys.argv[1:]}))"
)


def stat_in_pod(
    namespace: str,
    pod: PodRef,
    paths: list[str],
) -> Optional[dict[str, Optional[int]]]:
    """Size of each path as seen from inside *pod*, or None if it cannot run."""
    if not paths:
        return {}
    try:
        proc = kubectl(
            ["exec", pod.name, "--container", pod.container,
             "--", "python", "-c", _STAT_SNIPPET, *paths],
            namespace, capture=True,
        )
        return json.loads(proc.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def tail_pod_log(namespace: str, pod: PodRef, lines: int = 20) -> Optional[str]:
    """Last log lines from *pod*, used to name a failure the HTTP body hides."""
    try:
        proc = kubectl(
            ["logs", pod.name, "--container", pod.container, f"--tail={lines}"],
            namespace, capture=True,
        )
        return proc.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# The handoff itself
# ---------------------------------------------------------------------------

def derived_sdf_path(structure_path: str) -> str:
    """Where PoseBusters writes the ligand it pulls out of a Boltz PDB.

    ``_boltz_pdb_to_sdf`` writes beside its input, i.e. into the *producer's*
    output directory, so the consumer's ``/shared`` mount has to be writable
    and not merely readable. Mirrored here so the file can be looked for from
    the producer's own pod.
    """
    path = Path(structure_path)
    return str(path.with_name(path.stem + "_lig.sdf"))


def boltz_payload(args: argparse.Namespace, smiles: list[str]) -> dict[str, Any]:
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
            # PoseBusters pulls HETATM records out of the complex, so the
            # handoff only works in pdb. mmcif would score None everywhere for
            # a reason that has nothing to do with the volume.
            "output_format": "pdb",
        },
    }


def posebusters_payload(
    smiles: list[str],
    structures: list[Optional[str]],
    mode: str,
) -> dict[str, Any]:
    """The payload ``ExternalRESTScorer`` builds for a PoseBusters objective.

    The full index-aligned list, holes included (the service enforces
    ``len(structure_paths) == len(smiles)``), and ``protein_pdb_path`` falling
    back to the first structure that exists -- the Boltz complex doubles as
    the receptor.
    """
    valid = [s for s in structures if s]
    params: dict[str, Any] = {"mode": mode, "structure_paths": list(structures)}
    if valid:
        params["protein_pdb_path"] = valid[0]
    return {"smiles": smiles, "params": params}


def profiler_payload(
    smiles: list[str],
    structures: list[Optional[str]],
    interaction_type: str,
) -> dict[str, Any]:
    return {
        "smiles": smiles,
        "params": {
            "interaction_type": interaction_type,
            "structure_paths": list(structures),
            "radius": 8.0,
            "ph": 7.4,
        },
    }


def audit_paths(
    structures: list[Optional[str]],
    expected_indices: list[int],
    n_molecules: int,
) -> list[str]:
    """Reasons the returned paths could not survive a pod boundary.

    A relative path would resolve against each consumer's working directory,
    and one outside ``/shared`` is on a container filesystem no other pod
    mounts -- the interaction profiler rejects those outright. Either is a
    handoff failure that has nothing to do with the storage class.
    """
    problems: list[str] = []
    if len(structures) != n_molecules:
        problems.append(
            f"boltz returned {len(structures)} structure slots for "
            f"{n_molecules} molecules: the list is no longer index-aligned, so "
            "consumers would pair poses with the wrong SMILES"
        )
    for idx, path in enumerate(structures):
        if path is None:
            if idx in expected_indices:
                problems.append(f"molecule {idx} scored but produced no structure path")
            continue
        if idx not in expected_indices:
            problems.append(
                f"molecule {idx} was skipped by boltz yet carries a structure "
                f"path ({path}): the holes have shifted"
            )
        if not path.startswith("/"):
            problems.append(f"molecule {idx}: {path!r} is relative, so it means "
                            "something different in every pod")
        elif not path.startswith(SHARED_ROOT + "/"):
            problems.append(f"molecule {idx}: {path!r} is outside {SHARED_ROOT}, "
                            "which is the only filesystem the pods share")
    return problems


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

@dataclass
class Producer:
    pods: list[PodRef] = field(default_factory=list)
    elapsed: Optional[float] = None
    structures: list[Optional[str]] = field(default_factory=list)
    scores: list[Optional[float]] = field(default_factory=list)
    skip_reasons: list[str] = field(default_factory=list)
    path_problems: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def valid_structures(self) -> list[str]:
        return [s for s in self.structures if s]


@dataclass
class Consumer:
    label: str
    url: str
    pods: list[PodRef] = field(default_factory=list)
    status: str = UNREACHABLE
    detail: str = ""
    elapsed: Optional[float] = None
    scores: list[Optional[float]] = field(default_factory=list)
    log_tail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == OK


def check_health(urls: dict[str, str], timeout: float) -> list[str]:
    """Fail fast on an unreachable service rather than after a GPU prediction."""
    unreachable: list[str] = []
    for label, base in urls.items():
        health, error = get_json(f"{base}/health", timeout)
        if health is None:
            unreachable.append(f"{label} ({base}/health): {error}")
            continue
        print(f"   {label:<21} {base}  {health.get('status', '?')}", flush=True)
        # The profiler reports whether it can see the volume at all, which is
        # the cheapest possible version of this whole check.
        if health.get("shared_dir_exists") is False:
            unreachable.append(
                f"{label} reports shared_dir {health.get('shared_dir')!r} does "
                "not exist: the pod has no /shared mount, so no path from "
                "boltz can ever resolve there"
            )
    return unreachable


def produce(args: argparse.Namespace, batch: list[tuple[str, str]]) -> Producer:
    """Run the batch through boltz and keep the paths it wrote."""
    producer = Producer()
    smiles = [smi for _, smi in batch]

    print(f"\n-- producer: boltz at {args.boltz_url} --", flush=True)
    if args.namespace:
        producer.pods = pods_for(args.namespace, BOLTZ)
        if producer.pods:
            print(f"   {describe_pods(producer.pods)}", flush=True)
    print(f"   scoring {len(smiles)} molecules against BRD4 BD1 "
          f"(msa_path={args.msa_path})", flush=True)

    elapsed, data, _status, error = post_json(
        f"{args.boltz_url}/score", boltz_payload(args, smiles), args.boltz_timeout,
    )
    producer.elapsed = elapsed
    if error:
        producer.error = error
        print(f"   failed after {elapsed:.1f}s -- {error}", flush=True)
        return producer

    producer.scores = data.get("scores") or []
    metadata = data.get("metadata") or {}
    producer.skip_reasons = [
        r or "" for r in (metadata.get("skip_reason") or [""] * len(smiles))
    ]
    producer.structures = (data.get("artifacts") or {}).get("structures") or []
    expected = [i for i, reason in enumerate(producer.skip_reasons) if not reason]
    producer.path_problems = audit_paths(producer.structures, expected, len(smiles))

    print(f"   {elapsed:.1f}s ({elapsed / 60:.1f} min), "
          f"{len(producer.valid_structures)}/{len(smiles)} structures written",
          flush=True)
    for idx, reason in enumerate(producer.skip_reasons):
        if reason:
            print(f"   molecule {idx} skipped: {reason}", flush=True)
    for path in producer.valid_structures:
        print(f"   {path}", flush=True)
    return producer


def consume(
    label: str,
    url: str,
    payload: dict[str, Any],
    expected_indices: list[int],
    timeout: float,
    namespace: Optional[str],
) -> Consumer:
    """POST the relayed paths to one consumer and classify what came back."""
    consumer = Consumer(label=label, url=url)

    print(f"\n-- consumer: {label} at {url} --", flush=True)
    if namespace:
        consumer.pods = pods_for(namespace, label)
        if consumer.pods:
            print(f"   {describe_pods(consumer.pods)}", flush=True)

    elapsed, data, status, error = post_json(f"{url}/score", payload, timeout)
    consumer.elapsed = elapsed
    if error:
        # A 4xx is the consumer refusing the payload: wrong shape, wrong
        # length, a null where it wants a string. Nothing to do with storage.
        consumer.status = REJECTED if status and 400 <= status < 500 else UNREACHABLE
        consumer.detail = error
        print(f"   {consumer.status} after {elapsed:.1f}s -- {error}", flush=True)
        return consumer

    consumer.scores = data.get("scores") or []
    scored = [i for i in expected_indices
              if i < len(consumer.scores) and consumer.scores[i] is not None]
    holes_kept = all(
        consumer.scores[i] is None
        for i in range(min(len(consumer.scores), len(payload["smiles"])))
        if i not in expected_indices
    )

    if not expected_indices:
        consumer.status = BLIND
        consumer.detail = "no structures were produced, so nothing was handed over"
    elif len(scored) == len(expected_indices) and holes_kept:
        consumer.status = OK
        consumer.detail = f"{len(scored)}/{len(expected_indices)} structures read"
    elif not scored:
        consumer.status = BLIND
        consumer.detail = (
            f"answered {len(consumer.scores)} nulls: every path was accepted "
            "and no file behind it could be opened"
        )
    else:
        consumer.status = PARTIAL
        consumer.detail = (
            f"{len(scored)}/{len(expected_indices)} structures read"
            + ("" if holes_kept else "; a skipped molecule came back scored, so "
                                     "the paths are misaligned")
        )

    print(f"   {elapsed:.1f}s, {consumer.status}: {consumer.detail}", flush=True)
    print(f"   scores {consumer.scores}", flush=True)

    if consumer.status in (BLIND, PARTIAL) and namespace and consumer.pods:
        consumer.log_tail = tail_pod_log(namespace, consumer.pods[0])
    return consumer


def explain_profiler_nulls(
    url: str,
    structures: list[Optional[str]],
    smiles: list[str],
    timeout: float,
) -> list[str]:
    """Ask the profiler's /analyze why a structure yielded nothing.

    ``/score`` collapses every failure to a null, but ``/analyze`` returns the
    reason per molecule -- and "structure not found" is precisely the storage
    failure this script exists to catch, as distinct from a structure that
    parsed and simply had no qualifying contacts.
    """
    molecules = [
        {"name": f"mol_{i}", "smiles": smiles[i], "structure": path}
        for i, path in enumerate(structures) if path
    ]
    if not molecules:
        return []
    _elapsed, data, _status, error = post_json(
        f"{url}/analyze",
        {"batch_id": "handoff", "molecules": molecules},
        timeout,
    )
    if error:
        return [f"/analyze could not be reached either: {error}"]
    reasons: list[str] = []
    for result in data.get("molecules") or []:
        reason = result.get("error")
        if reason:
            reasons.append(reason)
    return reasons


def check_reverse_visibility(
    namespace: str,
    producer: Producer,
) -> tuple[Optional[dict[str, Optional[int]]], list[str]]:
    """Look for PoseBusters' extracted ligand from inside every boltz replica.

    PoseBusters writes ``<structure>_lig.sdf`` beside the PDB it was given, so
    finding that file from a producer pod proves the volume is shared in the
    write direction too, not just that one pod could read another's output.

    Every replica is checked rather than one, because with the gateway placing
    batches the next one may land anywhere: a replica that cannot see what its
    peers wrote would break the handoff intermittently, which is far worse to
    debug than breaking it outright.
    """
    if not producer.pods or not producer.valid_structures:
        return None, []
    expected = [derived_sdf_path(p) for p in producer.valid_structures]

    seen: dict[str, Optional[int]] = {}
    missing: list[str] = []
    for pod in producer.pods:
        sizes = stat_in_pod(namespace, pod, expected)
        if sizes is None:
            return None, []
        for path, size in sizes.items():
            # Keep the smallest answer across replicas, so one replica that
            # cannot see a file is not hidden by one that can.
            if path not in seen or (size or 0) < (seen[path] or 0):
                seen[path] = size
            if not size and path not in missing:
                missing.append(path)
    return seen, missing


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_molecules(
    batch: list[tuple[str, str]],
    producer: Producer,
    consumers: list[Consumer],
) -> None:
    header = f"{'idx':<4} {'molecule':<18} {'affinity':>9}"
    for consumer in consumers:
        header += f" {consumer.label.rsplit('-', 1)[-1]:>12}"
    header += "  structure"
    print("\n" + header)

    def cell(values: list[Optional[float]], idx: int) -> str:
        if idx >= len(values) or values[idx] is None:
            return "--"
        return f"{values[idx]:.3f}"

    for idx, (name, _smi) in enumerate(batch):
        path = producer.structures[idx] if idx < len(producer.structures) else None
        row = f"{idx:<4} {name[:18]:<18} {cell(producer.scores, idx):>9}"
        for consumer in consumers:
            row += f" {cell(consumer.scores, idx):>12}"
        row += f"  {path or '--'}"
        print(row)


def render(
    batch: list[tuple[str, str]],
    producer: Producer,
    consumers: list[Consumer],
    reverse: Optional[dict[str, Optional[int]]],
    findings: list[str],
    verdict: tuple[bool, str],
) -> None:
    print("\n" + "=" * 72)
    print(f"cross-pod structure handoff | BRD4 BD1 | {len(batch)} molecules "
          f"({sum(1 for r in producer.skip_reasons if r)} skipped by boltz)")
    print("=" * 72)

    print(f"{'producer':<10} {BOLTZ:<21} {describe_pods(producer.pods)}")
    print(f"{'':<10} {'':<21} "
          f"{len(producer.valid_structures)}/{len(batch)} structures under {SHARED_ROOT}")
    for consumer in consumers:
        print(f"{'consumer':<10} {consumer.label:<21} {describe_pods(consumer.pods)}")
        print(f"{'':<10} {'':<21} {consumer.status}: {consumer.detail}")

    if reverse is not None:
        found = sum(1 for size in reverse.values() if size)
        replicas = len(producer.pods) or 1
        print(f"{'reverse':<10} {'_lig.sdf from boltz':<21} "
              f"{found}/{len(reverse)} of PoseBusters' extracted ligands are "
              f"visible from all {replicas} boltz replica(s)")

    render_molecules(batch, producer, consumers)

    if findings:
        print("\nFindings")
        for finding in findings:
            print(f"  ! {finding}")

    for consumer in consumers:
        if consumer.log_tail:
            print(f"\nLast log lines from {consumer.label}:")
            for line in consumer.log_tail.splitlines():
                print(f"  {line}")

    proven, reason = verdict
    print(f"\n{'PROVEN' if proven else 'NOT PROVEN'}: {reason}")


def collect_findings(
    producer: Producer,
    consumers: list[Consumer],
    reverse_missing: list[str],
    namespace: Optional[str],
) -> list[str]:
    findings: list[str] = list(producer.path_problems)

    producer_names = {pod.name for pod in producer.pods}
    consumer_pods = {c.label: c.pods for c in consumers if c.pods}
    if producer_names and consumer_pods:
        shared = [label for label, pods in consumer_pods.items()
                  if producer_names & {pod.name for pod in pods}]
        if shared:
            findings.append(
                f"{', '.join(shared)} answered from a pod boltz also runs in, "
                "so this run proves the in-pod handoff again and not the "
                "cross-pod one"
            )
        nodes = {pod.node for pods in consumer_pods.values() for pod in pods}
        nodes |= {pod.node for pod in producer.pods}
        if len(nodes) > 1:
            findings.append(
                f"the pods are spread across {len(nodes)} nodes "
                f"({', '.join(sorted(nodes))}), so the claim must be "
                "ReadWriteMany -- on a ReadWriteOnce claim this configuration "
                "would not have mounted at all"
            )

    for consumer in consumers:
        if consumer.status == REJECTED:
            findings.append(
                f"{consumer.label} rejected the payload ExternalRESTScorer "
                f"sends ({consumer.detail.splitlines()[0]}): a config or schema "
                "problem, not a storage one -- the file was never looked for"
            )
        elif consumer.status == BLIND and producer.valid_structures:
            findings.append(
                f"{consumer.label} accepted paths under {SHARED_ROOT} and could "
                "not open any of them: this is the storage failure the fallback "
                "to object-storage URIs exists for"
            )
        elif consumer.status == PARTIAL:
            findings.append(f"{consumer.label}: {consumer.detail}")

    if reverse_missing:
        findings.append(
            f"{len(reverse_missing)} extracted ligand(s) PoseBusters should "
            "have written next to the Boltz PDB are not visible from the boltz "
            "pod, so the mount is readable but not shared for writes"
        )

    if not namespace:
        findings.append(
            "run with --namespace to confirm which pod and node answered; "
            "without it a green result cannot distinguish cross-pod from in-pod"
        )
    return findings


def decide(
    producer: Producer,
    consumers: list[Consumer],
    reverse_missing: list[str],
    namespace: Optional[str],
) -> tuple[Optional[bool], str]:
    """``(proven, reason)``; ``proven`` is None when the run was inconclusive."""
    if producer.error:
        return None, f"boltz never produced a structure: {producer.error}"
    if not producer.valid_structures:
        return None, (
            "boltz scored nothing, so no path was handed over. Check the batch "
            "and the MSA before reading anything into the consumers"
        )
    if producer.path_problems:
        return False, ("the paths boltz returned cannot cross a pod boundary; "
                       "see Findings")

    unreachable = [c.label for c in consumers if c.status == UNREACHABLE]
    if unreachable:
        return None, f"could not reach {', '.join(unreachable)}"

    failed = [c for c in consumers if not c.ok]
    if failed:
        return False, (
            f"{', '.join(c.label for c in failed)} did not read the structures "
            f"boltz wrote ({', '.join(c.status for c in failed)})"
        )

    if reverse_missing:
        # PoseBusters cannot have scored without extracting the ligand, and it
        # extracts it beside the PDB it was handed. If that file is not there
        # from a producer replica, those pods resolved the same path on
        # different filesystems and the reads were a coincidence.
        return False, (
            f"{len(reverse_missing)} ligand(s) PoseBusters extracted next to "
            "the Boltz PDB are not visible from every boltz replica, so some of "
            "these pods resolve these paths on different filesystems"
        )

    producer_names = {pod.name for pod in producer.pods}
    same_pod = [
        c.label for c in consumers
        if producer_names & {pod.name for pod in c.pods}
    ]
    if same_pod:
        return None, (
            f"{', '.join(same_pod)} shares a pod with boltz, so the handoff was "
            "never crossed"
        )
    if not namespace or not producer.pods:
        return True, (
            f"every consumer read the structures boltz wrote to {SHARED_ROOT}. "
            "Pod identity was not checked, so re-run with --namespace to call "
            "this cross-pod"
        )

    pods = producer.pods + [pod for c in consumers for pod in c.pods]
    nodes = {pod.node for pod in pods}
    scope = (f"across {len(nodes)} nodes" if len(nodes) > 1
             else f"across {len(pods)} pods on node {pods[0].node}")
    replicas = (f"one of its {len(producer.pods)} replicas"
                if len(producer.pods) > 1 else producer.pods[0].name)
    return True, (
        f"boltz wrote structures in {replicas} and every consumer re-opened "
        f"them by path, {scope}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_batch(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The molecules to send, with the deliberate hole unless it is turned off."""
    if args.smiles:
        batch = [(f"molecule_{i}", smi) for i, smi in enumerate(args.smiles)]
    else:
        batch = list(PANEL)[: args.batch_size]
    if args.hole and not args.structure:
        at = min(HOLE_INDEX, len(batch))
        batch = batch[:at] + [HOLE] + batch[at:]
    return batch


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--namespace", default=None,
                        help="resolve which pod and node answered, and stat the "
                             "handed-over files from inside them")
    parser.add_argument("--in-cluster", action="store_true",
                        help="dial the ClusterIP Services instead of localhost "
                             "(for running inside a pod)")
    parser.add_argument("--boltz-url", default=None, help="boltz base URL")
    parser.add_argument("--posebusters-url", default=None)
    parser.add_argument("--profiler-url", default=None,
                        help="interaction-profiler base URL")
    parser.add_argument("--structure", action="append", default=None,
                        help="replay against a structure already on the volume "
                             "instead of predicting one; repeatable, and needs a "
                             "matching --smiles for each")
    parser.add_argument("--smiles", action="append", default=None,
                        help="SMILES to send instead of the built-in panel")
    parser.add_argument("--batch-size", type=int, default=len(PANEL),
                        help="molecules from the built-in panel (plus the hole)")
    parser.add_argument("--no-hole", dest="hole", action="store_false",
                        help="omit the oversized molecule that tests index "
                             "alignment on a partial batch")
    parser.add_argument("--msa-path", default=DEFAULT_MSA_PATH,
                        help="precomputed .a3m on the shared volume")
    parser.add_argument("--protein-sequence", default=BRD4_BD1_SEQUENCE)
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--sampling-steps-affinity", type=int, default=200)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=5)
    parser.add_argument("--posebusters-mode", default=DEFAULT_POSEBUSTERS_MODE,
                        help="PoseBusters config; the ConfigMap sets dock")
    parser.add_argument("--interaction-type", default=DEFAULT_INTERACTION_TYPE,
                        help="interaction family to count; 'all' avoids the "
                             "structurally-zero objectives (hbond needs waters "
                             "Boltz does not model)")
    parser.add_argument("--boltz-timeout", type=float, default=2100.0,
                        help="above the service's own 1800s cap, so a timed-out "
                             "prediction reports itself")
    parser.add_argument("--consumer-timeout", type=float, default=600.0)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true",
                        help="also emit the run as JSON")
    args = parser.parse_args(argv)

    defaults = CLUSTER_URLS if args.in_cluster else LOCAL_URLS
    args.boltz_url = (args.boltz_url or defaults[BOLTZ]).rstrip("/")
    args.posebusters_url = (args.posebusters_url or defaults[POSEBUSTERS]).rstrip("/")
    args.profiler_url = (args.profiler_url or defaults[PROFILER]).rstrip("/")

    if args.structure and not args.smiles:
        parser.error("--structure replays a known pose, so it needs the "
                     "--smiles that produced it")
    if args.structure and len(args.structure) != len(args.smiles):
        parser.error(f"{len(args.structure)} --structure(s) against "
                     f"{len(args.smiles)} --smiles: the two are index-aligned")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    batch = build_batch(args)
    smiles = [smi for _, smi in batch]

    print("checking the services are up", flush=True)
    urls = {
        BOLTZ: args.boltz_url,
        POSEBUSTERS: args.posebusters_url,
        PROFILER: args.profiler_url,
    }
    if args.structure:
        # Nothing is predicted in replay mode, so boltz need not be running.
        urls.pop(BOLTZ)
    unreachable = check_health(urls, args.health_timeout)
    if unreachable:
        for problem in unreachable:
            print(f"  ! {problem}", file=sys.stderr)
        print("\nNothing was measured. Start the port-forwards (or pass "
              "--in-cluster) and retry.", file=sys.stderr)
        return 2

    if args.structure:
        producer = Producer(
            pods=pods_for(args.namespace, BOLTZ) if args.namespace else [],
            structures=list(args.structure),
            scores=[None] * len(batch),
            skip_reasons=[""] * len(batch),
        )
        producer.path_problems = audit_paths(
            producer.structures, list(range(len(batch))), len(batch),
        )
        print(f"\n-- producer: replaying {len(producer.valid_structures)} "
              f"structure(s) already on the volume --", flush=True)
        for path in producer.valid_structures:
            print(f"   {path}", flush=True)
    else:
        producer = produce(args, batch)

    expected = [i for i, path in enumerate(producer.structures) if path]
    consumers: list[Consumer] = []
    if producer.valid_structures:
        consumers.append(consume(
            POSEBUSTERS, args.posebusters_url,
            posebusters_payload(smiles, producer.structures, args.posebusters_mode),
            expected, args.consumer_timeout, args.namespace,
        ))
        profiler = consume(
            PROFILER, args.profiler_url,
            profiler_payload(smiles, producer.structures, args.interaction_type),
            expected, args.consumer_timeout, args.namespace,
        )
        if profiler.status in (BLIND, PARTIAL):
            for reason in explain_profiler_nulls(
                args.profiler_url, producer.structures, smiles, args.consumer_timeout,
            ):
                print(f"   /analyze: {reason}", flush=True)
                profiler.detail += f"; /analyze says: {reason}"
        consumers.append(profiler)

    reverse: Optional[dict[str, Optional[int]]] = None
    reverse_missing: list[str] = []
    if args.namespace and any(c.label == POSEBUSTERS and c.ok for c in consumers):
        reverse, reverse_missing = check_reverse_visibility(args.namespace, producer)

    findings = collect_findings(producer, consumers, reverse_missing, args.namespace)
    proven, reason = decide(producer, consumers, reverse_missing, args.namespace)
    render(batch, producer, consumers, reverse, findings, (bool(proven), reason))

    if args.json:
        print()
        json.dump(
            {
                "proven": proven,
                "reason": reason,
                "molecules": [
                    {"name": name, "smiles": smi,
                     "structure": producer.structures[i]
                     if i < len(producer.structures) else None,
                     "skip_reason": producer.skip_reasons[i]
                     if i < len(producer.skip_reasons) else ""}
                    for i, (name, smi) in enumerate(batch)
                ],
                "producer": {
                    "pods": [
                        {"name": pod.name, "node": pod.node} for pod in producer.pods
                    ],
                    "elapsed_seconds": producer.elapsed,
                    "scores": producer.scores,
                    "structures": producer.structures,
                    "path_problems": producer.path_problems,
                    "error": producer.error,
                },
                "consumers": [
                    {"label": c.label, "url": c.url,
                     "pods": [
                         {"name": pod.name, "node": pod.node} for pod in c.pods
                     ],
                     "status": c.status, "detail": c.detail,
                     "elapsed_seconds": c.elapsed, "scores": c.scores}
                    for c in consumers
                ],
                "reverse_visibility": reverse,
                "findings": findings,
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
