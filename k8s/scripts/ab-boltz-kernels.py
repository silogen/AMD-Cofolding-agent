#!/usr/bin/env python3
"""Time one Boltz /score call with and without the optimized kernels.

``services/boltz2/app.py`` passes ``--no_kernels`` to ``boltz predict`` by
default, which turns Boltz's optimized kernels off. ``BOLTZ_NO_KERNELS=0``
turns them on. The per-batch wall clock that choice produces is what sets how
many Boltz replicas a given number of concurrent agents needs, so it is worth
one measurement per GPU generation.

Both arms score the same 16 drug-like molecules against BRD4 BD1 using the
precomputed ``brd4_bd1.a3m`` on the shared volume, so neither pays an
MSA-server round trip and the only difference is the kernel flag.

    # full A/B: flips BOLTZ_NO_KERNELS on the Deployment between arms
    kubectl --namespace demo-lux port-forward deployment/dd-agentic-boltz 8091:8091 &
    ./k8s/scripts/ab-boltz-kernels.py --namespace demo-lux

    # Compose
    ./k8s/scripts/ab-boltz-kernels.py --restart-command 'docker compose up -d boltz'

    # one arm only, e.g. to measure t_boltz_per_batch when the scoring log
    # has no usable history
    ./k8s/scripts/ab-boltz-kernels.py --baseline-only

With ``--namespace`` the script edits the live Deployment and restores the
original value on exit; re-running ``k8s/apply.sh`` also resets it to the
manifest.

No third-party deps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

ENV_KEY = "BOLTZ_NO_KERNELS"
DEFAULT_URL = "http://localhost:8091/score"
DEFAULT_MSA_PATH = "brd4_bd1.a3m"

# The query (first) record of data/shared/brd4_bd1.a3m -- the BD1 bromodomain
# of BRD4. Sending it explicitly keeps the two arms identical; the service
# would otherwise derive it from the MSA.
BRD4_BD1_SEQUENCE = (
    "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKK"
    "RLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE"
)

# 16 approved or clinical small molecules, 17-37 heavy atoms (mean 29), all
# inside the wrapper's MAX_HEAVY_ATOMS/MAX_TOTAL_ATOMS limits so nothing is
# silently skipped. JQ1 and birabresib are the BET chemotypes; the rest make
# the batch look like a real RL batch, since runtime tracks atom count.
PANEL: list[tuple[str, str]] = [
    ("JQ1", "CC1=C(C)C2=C(S1)N1C(C)=NN=C1[C@H](CC(=O)OC(C)(C)C)N=C2c1ccc(Cl)cc1"),
    ("birabresib", "CC(=O)Nc1ccc(cc1)C1=NC(CC(=O)N)c2nnc(C)n2-c2sc(C)c(C)c21"),
    ("celecoxib", "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F"),
    ("imatinib", "Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)NC(=O)c1ccc(CN2CCN(C)CC2)cc1"),
    ("gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"),
    ("erlotinib", "COCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC"),
    ("sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(c3)C(F)(F)F)cc2)ccn1"),
    ("dasatinib", "Cc1nc(Nc2ncc(s2)C(=O)Nc2c(C)cccc2Cl)cc(n1)N1CCN(CCO)CC1"),
    ("pazopanib", "Cc1ccc(Nc2nccc(N(C)c3ccc4c(C)n(C)nc4c3)n2)cc1S(N)(=O)=O"),
    ("sunitinib", "CCN(CC)CCNC(=O)c1c(C)[nH]c(c1C)/C=C1\\C(=O)Nc2ccc(F)cc21"),
    ("olaparib", "O=C(c1ccc(CC2=NNC(=O)c3ccccc32)cc1F)N1CCN(C(=O)C2CC2)CC1"),
    ("sildenafil", "CCCc1nn(C)c2c1nc([nH]c2=O)-c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1"),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nn[nH]n1"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("naproxen", "COc1ccc2cc(ccc2c1)C(C)C(=O)O"),
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def health_url_for(score_url: str) -> str:
    return score_url[: -len("/score")] + "/health" if score_url.endswith("/score") else score_url


def get_health(url: str, timeout: float) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Return ``(health, error)`` -- the error so a wait can say why it failed."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode()), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError) as exc:
        return None, str(exc)
    except ValueError as exc:
        return None, f"unparseable /health response: {exc}"


def post_score(url: str, payload: dict[str, Any], timeout: float) -> tuple[float, dict[str, Any], Optional[str]]:
    """POST /score and return ``(elapsed_seconds, data, error)``."""
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
        return time.monotonic() - start, json.loads(body), None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        return time.monotonic() - start, {}, f"HTTP {exc.code}: {detail}"
    except (urllib.error.URLError, OSError) as exc:
        return time.monotonic() - start, {}, f"request failed: {exc}"
    except ValueError as exc:
        return time.monotonic() - start, {}, f"unparseable response: {exc}"


# ---------------------------------------------------------------------------
# Flipping the flag
# ---------------------------------------------------------------------------

def kubectl(args: list[str], namespace: str, capture: bool = False) -> subprocess.CompletedProcess:
    argv = ["kubectl", "--namespace", namespace, *args]
    return subprocess.run(argv, check=True, capture_output=capture, text=True)


def read_container_env(namespace: str, deployment: str, container: str) -> Optional[str]:
    """Current ``BOLTZ_NO_KERNELS`` on the container, or None when unset.

    A value sourced from ``valueFrom`` reads as unset, which is the right
    answer for restore purposes: this script only ever sets literals.
    """
    proc = kubectl(["get", f"deployment/{deployment}", "--output", "json"],
                   namespace, capture=True)
    spec = json.loads(proc.stdout)["spec"]["template"]["spec"]
    for entry in spec.get("containers") or []:
        if entry.get("name") != container:
            continue
        for var in entry.get("env") or []:
            if var.get("name") == ENV_KEY:
                return var.get("value")
    return None


def write_container_env(
    namespace: str,
    deployment: str,
    container: str,
    value: Optional[str],
    rollout_timeout: str,
) -> None:
    # A trailing dash removes the variable, which is how an originally unset
    # value is restored.
    assignment = f"{ENV_KEY}={value}" if value is not None else f"{ENV_KEY}-"
    kubectl(["set", "env", f"deployment/{deployment}",
             f"--containers={container}", assignment], namespace)
    kubectl(["rollout", "status", f"deployment/{deployment}",
             f"--timeout={rollout_timeout}"], namespace)


def run_restart_command(command: str, value: str) -> None:
    env = {**os.environ, ENV_KEY: value}
    subprocess.run(shlex.split(command), check=True, env=env)


def wait_for_health(
    url: str,
    expect_no_kernels: bool,
    deadline_seconds: float,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Poll /health until the service reports the arm we asked for.

    Returns ``(health, error)``; the error carries the last reason the wait
    did not succeed, so an unreachable endpoint is distinguishable from one
    that is up but serving the other arm.
    """
    deadline = time.monotonic() + deadline_seconds
    while True:
        health, error = get_health(url, timeout=5.0)
        if health is not None:
            reported = health.get("no_kernels")
            if reported is None or bool(reported) == expect_no_kernels:
                return health, None
            error = f"serving no_kernels={bool(reported)}"
        if time.monotonic() >= deadline:
            return health, error
        time.sleep(3.0)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    label: str
    no_kernels: bool
    elapsed: Optional[float] = None
    n_scored: int = 0
    n_molecules: int = 0
    skipped: list[str] = field(default_factory=list)
    error: Optional[str] = None
    reported_no_kernels: Optional[bool] = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.elapsed is not None and self.n_scored > 0

    @property
    def per_molecule(self) -> Optional[float]:
        if self.elapsed is None or not self.n_molecules:
            return None
        return self.elapsed / self.n_molecules


def build_payload(args: argparse.Namespace, smiles: list[str]) -> dict[str, Any]:
    return {
        "smiles": smiles,
        "params": {
            "protein_sequence": args.protein_sequence,
            "msa_path": args.msa_path,
            "recycling_steps": args.recycling_steps,
            "diffusion_samples": args.diffusion_samples,
            "sampling_steps": args.sampling_steps,
            "sampling_steps_affinity": args.sampling_steps_affinity,
            "diffusion_samples_affinity": args.diffusion_samples_affinity,
        },
    }


def run_arm(
    label: str,
    no_kernels: bool,
    args: argparse.Namespace,
    smiles: list[str],
    can_flip: bool,
) -> Arm:
    arm = Arm(label=label, no_kernels=no_kernels, n_molecules=len(smiles))
    value = "1" if no_kernels else "0"
    health = health_url_for(args.url)

    print(f"\n-- {label} ({ENV_KEY}={value}) --", flush=True)

    if args.namespace:
        print(f"   setting {ENV_KEY}={value} on "
              f"{args.deployment}/{args.container} and waiting for rollout",
              flush=True)
        write_container_env(args.namespace, args.deployment, args.container,
                            value, args.rollout_timeout)
    elif args.restart_command:
        print(f"   restarting with {ENV_KEY}={value}: {args.restart_command}",
              flush=True)
        run_restart_command(args.restart_command, value)

    reported, health_error = wait_for_health(health, no_kernels, args.health_timeout)
    if reported is None:
        arm.error = f"{health} unreachable: {health_error}"
        print(f"   {arm.error}", flush=True)
        return arm

    arm.reported_no_kernels = (
        bool(reported["no_kernels"]) if "no_kernels" in reported else None
    )
    if arm.reported_no_kernels is None:
        print("   /health does not report no_kernels (older image): trusting "
              "the requested arm", flush=True)
    elif arm.reported_no_kernels != no_kernels:
        if can_flip:
            arm.error = (
                f"asked for {ENV_KEY}={value} but the live replica reports "
                f"no_kernels={arm.reported_no_kernels}"
            )
            print(f"   {arm.error}", flush=True)
            return arm
        print(f"   live replica reports no_kernels="
              f"{arm.reported_no_kernels}, measuring that instead", flush=True)
        arm.no_kernels = arm.reported_no_kernels
        arm.label = "kernels" if not arm.reported_no_kernels else "baseline"

    print(f"   scoring {len(smiles)} molecules against BRD4 BD1 "
          f"(msa_path={args.msa_path})", flush=True)
    elapsed, data, error = post_score(args.url, build_payload(args, smiles),
                                      args.timeout)
    arm.elapsed = elapsed
    if error:
        arm.error = error
        print(f"   failed after {elapsed:.1f}s -- {error}", flush=True)
        return arm

    scores = data.get("scores") or []
    arm.n_scored = sum(1 for s in scores if s is not None)
    arm.skipped = [r for r in (data.get("metadata") or {}).get("skip_reason", []) if r]
    print(f"   {elapsed:.1f}s ({elapsed / 60:.1f} min), "
          f"{arm.n_scored}/{len(smiles)} scored", flush=True)
    for reason in arm.skipped:
        print(f"   skipped: {reason}", flush=True)
    return arm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def replicas_for(t_batch: float, supervisors: int, step_seconds: float) -> int:
    """Boltz replicas N agents need at one batch per RL step, ignoring cache."""
    return math.ceil(supervisors * t_batch / step_seconds)


def render(arms: list[Arm], args: argparse.Namespace) -> None:
    print("\n" + "=" * 68)
    print(f"BRD4 BD1 | {arms[0].n_molecules} molecules | "
          f"msa_path={args.msa_path} | recycling_steps={args.recycling_steps} "
          f"| sampling_steps_affinity={args.sampling_steps_affinity} "
          f"| diffusion_samples_affinity={args.diffusion_samples_affinity}")
    print("=" * 68)
    print(f"{'arm':<10} {'no_kernels':<12} {'wall clock':<22} {'scored':<8}")
    for arm in arms:
        wall = (f"{arm.elapsed:.1f}s ({arm.elapsed / 60:.1f} min)"
                if arm.elapsed is not None else "n/a")
        print(f"{arm.label:<10} {str(arm.no_kernels).lower():<12} {wall:<22} "
              f"{arm.n_scored}/{arm.n_molecules}")
        if arm.error:
            print(f"           error: {arm.error}")

    usable = [a for a in arms if a.usable]
    step_seconds = args.step_minutes * 60

    if len(usable) < 2:
        if len(usable) == 1:
            arm = usable[0]
            print(f"\nt_boltz_per_batch = {arm.elapsed:.1f}s "
                  f"({arm.elapsed / 60:.1f} min) at batch_size "
                  f"{arm.n_molecules}, per molecule "
                  f"{arm.per_molecule:.1f}s, with no_kernels="
                  f"{str(arm.no_kernels).lower()}")
            print(f"M for N={args.supervisors} at a "
                  f"{args.step_minutes:g} min step: "
                  f"{replicas_for(arm.elapsed, args.supervisors, step_seconds)} "
                  "boltz replicas before caching")
        # Only blame the kernels when the kernels arm actually answered a
        # /score call; an unreachable service says nothing about them.
        reached = [a for a in arms
                   if not a.usable and not a.no_kernels and a.elapsed is not None]
        if reached:
            print("\nThe optimized kernels did not produce a usable result on "
                  f"this GPU: keep {ENV_KEY}=1.")
        return

    baseline = next(a for a in usable if a.no_kernels)
    kernels = next(a for a in usable if not a.no_kernels)
    speedup = baseline.elapsed / kernels.elapsed

    m_baseline = replicas_for(baseline.elapsed, args.supervisors, step_seconds)
    m_kernels = replicas_for(kernels.elapsed, args.supervisors, step_seconds)

    print(f"\nspeedup        {speedup:.2f}x  (per molecule "
          f"{baseline.per_molecule:.1f}s -> {kernels.per_molecule:.1f}s)")
    print(f"replica count  M for N={args.supervisors} at a "
          f"{args.step_minutes:g} min step: {m_baseline} -> {m_kernels}")

    if kernels.n_scored < baseline.n_scored:
        print(f"\nThe kernels arm scored fewer molecules "
              f"({kernels.n_scored} vs {baseline.n_scored}): treat the speedup "
              f"as unproven and keep {ENV_KEY}=1.")
    elif speedup >= 1.1:
        print(f"\nThe optimized kernels work here. Set {ENV_KEY}=0 on the boltz "
              "container and re-derive M before provisioning.")
    elif speedup <= 0.95:
        print(f"\nThe optimized kernels are slower here: keep {ENV_KEY}=1.")
    else:
        print("\nNo meaningful difference on one call; the flag is not a lever "
              "on this GPU.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="boltz /score endpoint")
    parser.add_argument("--namespace", default=None,
                        help="flip the flag by editing this namespace's Deployment")
    parser.add_argument("--deployment", default="dd-agentic-boltz",
                        help="Deployment carrying the boltz container")
    parser.add_argument("--container", default="boltz")
    parser.add_argument("--restart-command", default=None,
                        help=f"shell command that restarts boltz, run with "
                             f"{ENV_KEY} exported (e.g. 'docker compose up -d boltz')")
    parser.add_argument("--baseline-only", action="store_true",
                        help="measure only the default arm (kernels off)")
    parser.add_argument("--kernels-only", action="store_true",
                        help="measure only the optimized-kernels arm")
    parser.add_argument("--msa-path", default=DEFAULT_MSA_PATH,
                        help="precomputed .a3m on the shared volume")
    parser.add_argument("--protein-sequence", default=BRD4_BD1_SEQUENCE)
    parser.add_argument("--smiles-file", default=None,
                        help="newline-separated SMILES instead of the built-in panel")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="molecules per /score call (production is 16)")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=200)
    parser.add_argument("--sampling-steps-affinity", type=int, default=200)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=2100.0,
                        help="HTTP timeout; above the service's 1800s cap so a "
                             "timed-out prediction reports itself")
    parser.add_argument("--health-timeout", type=float, default=600.0,
                        help="seconds to wait for the service after a restart")
    parser.add_argument("--rollout-timeout", default="10m")
    parser.add_argument("--supervisors", type=int, default=3,
                        help="N, for the implied replica count")
    parser.add_argument("--step-minutes", type=float, default=5.0,
                        help="target wall clock per RL step")
    parser.add_argument("--json", action="store_true",
                        help="also emit the arms as JSON")
    args = parser.parse_args(argv)

    if args.baseline_only and args.kernels_only:
        parser.error("--baseline-only and --kernels-only are mutually exclusive")
    if args.namespace and args.restart_command:
        parser.error("use either --namespace or --restart-command, not both")

    if args.smiles_file:
        smiles = [
            line.strip()
            for line in open(args.smiles_file)
            if line.strip() and not line.startswith("#")
        ]
    else:
        smiles = [smi for _, smi in PANEL]
    if args.batch_size > len(smiles):
        parser.error(
            f"--batch-size {args.batch_size} exceeds the {len(smiles)} available "
            "SMILES; pass more with --smiles-file"
        )
    smiles = smiles[: args.batch_size]

    can_flip = bool(args.namespace or args.restart_command)
    single_arm = args.baseline_only or args.kernels_only
    if not can_flip and not single_arm:
        parser.error(
            "a two-arm A/B needs a way to restart boltz with the other flag: "
            "pass --namespace or --restart-command, or measure one arm with "
            "--baseline-only / --kernels-only"
        )

    if args.baseline_only:
        plan = [("baseline", True)]
    elif args.kernels_only:
        plan = [("kernels", False)]
    else:
        plan = [("baseline", True), ("kernels", False)]

    original: Optional[str] = None
    restore = False
    if args.namespace:
        original = read_container_env(args.namespace, args.deployment, args.container)
        restore = True
        print(f"{args.deployment}/{args.container} currently has "
              f"{ENV_KEY}={original if original is not None else '<unset>'}")

    arms: list[Arm] = []
    try:
        for label, no_kernels in plan:
            arms.append(run_arm(label, no_kernels, args, smiles, can_flip))
    finally:
        if restore:
            print(f"\nrestoring {ENV_KEY}="
                  f"{original if original is not None else '<unset>'}", flush=True)
            try:
                write_container_env(args.namespace, args.deployment,
                                    args.container, original, args.rollout_timeout)
            except subprocess.CalledProcessError as exc:
                print(f"could not restore {ENV_KEY}: {exc}. Re-run k8s/apply.sh.",
                      file=sys.stderr)

    render(arms, args)

    if args.json:
        print()
        json.dump(
            {
                "batch_size": len(smiles),
                "msa_path": args.msa_path,
                "arms": [
                    {
                        "label": arm.label,
                        "no_kernels": arm.no_kernels,
                        "elapsed_seconds": arm.elapsed,
                        "per_molecule_seconds": arm.per_molecule,
                        "n_scored": arm.n_scored,
                        "n_molecules": arm.n_molecules,
                        "skipped": arm.skipped,
                        "error": arm.error,
                    }
                    for arm in arms
                ],
            },
            sys.stdout,
            indent=2,
        )
        print()

    return 0 if any(arm.usable for arm in arms) else 1


if __name__ == "__main__":
    sys.exit(main())
