#!/usr/bin/env python3
"""Switch dd-agentic-aim to another catalog AIMClusterModel.

Invoked by switch-aim-model.sh (same directory). Patches k8s/manifests/.
No third-party deps.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

K8S_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = K8S_DIR / "manifests"
AIM_SERVICE_FILE = MANIFESTS_DIR / "aim-service.yaml"
CONFIGMAP_FILE = MANIFESTS_DIR / "configmap.yaml"
AIM_SERVICE_NAME = "dd-agentic-aim"
DEPLOYMENT_NAME = "dd-agentic"
CONFIGMAP_NAME = "dd-agentic-config"
SOURCE_MODEL_LABEL = "aim.eai.amd.com/source-model"
PROFILE_ROLE_LABEL = "aim.eai.amd.com/profile-role"


def run(argv: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=check, capture_output=capture, text=True)


def kubectl_json(argv: list[str]) -> Any:
    proc = run(["kubectl", *argv, "--output", "json"], capture=True)
    return json.loads(proc.stdout)


def aim_id_of(item: dict[str, Any]) -> str:
    status = item.get("status") or {}
    spec = item.get("spec") or {}
    return str(status.get("aimId") or spec.get("aimId") or "")


def dump_model(item: dict[str, Any]) -> dict[str, str]:
    return {
        "name": item["metadata"]["name"],
        "aimId": aim_id_of(item),
        "ready": str((item.get("status") or {}).get("status") or ""),
    }


def resolve_model(models: list[dict[str, Any]], want_model: str, want_name: str) -> dict[str, str]:
    if want_name:
        hits = [m for m in models if m["metadata"]["name"] == want_name]
        if not hits:
            raise SystemExit(f"AIMClusterModel object {want_name!r} not found")
        chosen = dump_model(hits[0])
        if (
            want_model
            and want_model != chosen["name"]
            and want_model.lower() != chosen["aimId"].lower()
        ):
            raise SystemExit(
                f"object {want_name} has AIMID {chosen['aimId']!r}, not {want_model!r}"
            )
        return chosen

    exact_name = [m for m in models if m["metadata"]["name"] == want_model]
    if len(exact_name) == 1:
        return dump_model(exact_name[0])

    aim_hits = [m for m in models if aim_id_of(m).lower() == want_model.lower()]
    if len(aim_hits) == 1:
        return dump_model(aim_hits[0])
    if len(aim_hits) > 1:
        lines = [f"AIMID {want_model!r} matches multiple AIMClusterModel objects:"]
        for item in aim_hits:
            row = dump_model(item)
            lines.append(f"  {row['name']}  AIMID={row['aimId']}  status={row['ready']}")
        lines.append("pass --name with the object name to disambiguate")
        raise SystemExit("\n".join(lines))

    needle = want_model.lower()
    substr = [
        m
        for m in models
        if needle in m["metadata"]["name"].lower() or needle in aim_id_of(m).lower()
    ]
    lines = [f"no unique AIMClusterModel for {want_model!r}"]
    if substr:
        lines.append("near matches:")
        for item in substr:
            row = dump_model(item)
            lines.append(f"  {row['name']}  AIMID={row['aimId']}")
    raise SystemExit("\n".join(lines))


def profile_row(item: dict[str, Any]) -> dict[str, Any]:
    spec = item.get("spec") or {}
    labels = (item.get("metadata") or {}).get("labels") or {}
    return {
        "name": item["metadata"]["name"],
        "acceleratorCount": spec.get("acceleratorCount"),
        "precision": str(spec.get("precision") or "").lower(),
        "metric": str(spec.get("metric") or "").lower(),
        "role": labels.get(PROFILE_ROLE_LABEL) or "",
        "acceleratorModel": spec.get("acceleratorModel") or "",
        "profileId": spec.get("profileId") or "",
        "aimId": spec.get("aimId") or "",
        "sourceModel": labels.get(SOURCE_MODEL_LABEL) or "",
    }


def owned_profiles(items: list[dict[str, Any]], source_name: str, source_aim_id: str) -> list[dict[str, Any]]:
    owned = []
    for item in items:
        row = profile_row(item)
        if row["sourceModel"] == source_name:
            owned.append(item)
            continue
        if source_aim_id and str(row["aimId"]).lower() == source_aim_id.lower():
            owned.append(item)
    return owned


def match_profiles(
    items: list[dict[str, Any]],
    source_name: str,
    source_aim_id: str,
    count: int,
    precision: str,
    metric: str,
) -> list[dict[str, Any]]:
    owned = owned_profiles(items, source_name, source_aim_id)
    matches = []
    for item in owned:
        row = profile_row(item)
        role_ok = row["role"] in ("", "deployable")
        if (
            row["acceleratorCount"] == count
            and row["precision"] == precision
            and row["metric"] == metric
            and role_ok
        ):
            matches.append(row)
    if matches:
        return matches

    lines = [
        "no AIMClusterProfile matched source-model/AIMID + count + precision + metric",
        f"wanted: source={source_name} aimId={source_aim_id} count={count} "
        f"precision={precision} metric={metric} role=deployable",
    ]
    if not owned:
        lines.append("no AIMClusterProfile carries that source-model label or AIMID")
    else:
        lines.append("profiles for this model:")
        for item in owned:
            row = profile_row(item)
            lines.append(
                "  {name}  count={count} precision={prec} metric={met} role={role} "
                "hw={hw} id={pid} source={src}".format(
                    name=row["name"],
                    count=row["acceleratorCount"],
                    prec=row["precision"],
                    met=row["metric"],
                    role=row["role"],
                    hw=row["acceleratorModel"],
                    pid=row["profileId"],
                    src=row["sourceModel"],
                )
            )
    raise SystemExit("\n".join(lines))


def patch_aim_service(path: Path, model_name: str, count: int, precision: str, metric: str) -> None:
    lines = path.read_text().splitlines(keepends=True)
    out: list[str] = []
    in_aim = False
    prev = ""
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("kind: AIMService"):
            in_aim = True
        elif stripped == "---":
            in_aim = False
        if in_aim:
            if prev.strip() == "model:" and stripped.startswith("    name:"):
                line = f"    name: {model_name}\n"
            elif stripped.startswith("      acceleratorCount:"):
                line = f"      acceleratorCount: {count}\n"
            elif stripped.startswith("      precision:"):
                line = f"      precision: {precision}\n"
            elif stripped.startswith("      metric:"):
                line = f"      metric: {metric}\n"
        out.append(line)
        prev = stripped
    path.write_text("".join(out))


def patch_configmap_file(path: Path, served_id: str) -> None:
    quoted = json.dumps(served_id)
    lines = []
    replaced = False
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("  REINVENT4_AGENT_MODEL:") and not replaced:
            lines.append(f"  REINVENT4_AGENT_MODEL: {quoted}\n")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise SystemExit(f"REINVENT4_AGENT_MODEL not found in {path}")
    path.write_text("".join(lines))


def live_model_name(namespace: str, name: str) -> str | None:
    """Return spec.model.name of the live AIMService, or None when absent."""
    proc = run(
        [
            "kubectl",
            "--namespace",
            namespace,
            "get",
            "aimservice",
            name,
            "--output",
            "jsonpath={.spec.model.name}",
        ],
        capture=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def delete_aimservice(namespace: str, name: str, timeout: str) -> None:
    """Delete the AIMService and wait for its cache-cleanup finalizer to finish.

    spec.model is CEL-immutable, so a model change cannot be applied in place.
    The finalizer only removes non-Ready caches; warm caches survive for reuse.
    """
    run(
        [
            "kubectl",
            "--namespace",
            namespace,
            "delete",
            "aimservice",
            name,
            "--wait=true",
            f"--timeout={timeout}",
        ]
    )
    deadline = time.time() + parse_timeout(timeout)
    while time.time() < deadline:
        if live_model_name(namespace, name) is None:
            return
        time.sleep(3)
    raise SystemExit(f"AIMService/{name} still present after delete")


def parse_timeout(timeout: str) -> float:
    if timeout.endswith("m"):
        return int(timeout[:-1]) * 60
    if timeout.endswith("s"):
        return int(timeout[:-1])
    return int(timeout)


def wait_aimservice(namespace: str, name: str, timeout: str) -> None:
    deadline = time.time() + parse_timeout(timeout)
    while time.time() < deadline:
        obj = kubectl_json(["--namespace", namespace, "get", "aimservice", name])
        status = obj.get("status") or {}
        phase = str(status.get("status") or status.get("phase") or "")
        conditions = status.get("conditions") or []
        ready = any(
            c.get("type") == "Ready" and str(c.get("status")) == "True" for c in conditions
        )
        if ready or phase == "Running":
            print(f"AIMService status={phase or 'Ready'}")
            return
        time.sleep(10)
    run(["kubectl", "--namespace", namespace, "get", "aimservice", name], check=False)
    raise SystemExit("AIMService did not become Ready/Running before timeout")


def wait_endpoints(namespace: str, name: str, timeout: str) -> None:
    deadline = time.time() + parse_timeout(timeout)
    while time.time() < deadline:
        proc = run(
            [
                "kubectl",
                "--namespace",
                namespace,
                "get",
                "endpoints",
                name,
                "--output",
                "jsonpath={.subsets[*].addresses[*].ip}",
            ],
            capture=True,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        if proc.returncode == 0 and raw:
            print(f"endpoints: {raw}")
            return
        time.sleep(5)
    raise SystemExit(f"no endpoints on Service/{name}")


FETCH_MODELS_PY = r"""
import json, sys, urllib.request
aim_id = sys.argv[1]
raw = urllib.request.urlopen("http://dd-agentic-aim/v1/models", timeout=120).read()
payload = json.loads(raw)
ids = [item.get("id") for item in payload.get("data") or [] if item.get("id")]
if not ids:
    sys.stderr.write("/v1/models returned no ids\n")
    sys.exit(1)
if len(ids) == 1:
    print(ids[0])
    raise SystemExit(0)
preferred = [i for i in ids if aim_id and aim_id.lower() in i.lower()]
print((preferred or ids)[0])
"""


def fetch_served_id(namespace: str, aim_id: str) -> str:
    last_err = ""
    for pybin in ("python3", "python"):
        proc = run(
            [
                "kubectl",
                "--namespace",
                namespace,
                "exec",
                f"deployment/{DEPLOYMENT_NAME}",
                "--container",
                "dd-agent",
                "--",
                pybin,
                "-c",
                FETCH_MODELS_PY,
                aim_id,
            ],
            capture=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[-1]
        last_err = (proc.stderr or proc.stdout or "").strip()
    raise SystemExit(f"failed to read /v1/models from deployment/{DEPLOYMENT_NAME}: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Point dd-agentic-aim at another catalog model and restart dd-agentic."
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--model", default="", help="AIMID or AIMClusterModel object name")
    parser.add_argument("--name", default="", help="exact AIMClusterModel object name")
    parser.add_argument("--accelerator-count", required=True, type=int)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--metric", default="latency")
    parser.add_argument("--timeout", default="45m")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="abort instead of deleting AIMService/dd-agentic-aim when the model changes",
    )
    args = parser.parse_args()

    if args.accelerator_count < 1:
        raise SystemExit("--accelerator-count must be a positive integer")
    if not args.model and not args.name:
        raise SystemExit("provide --model and/or --name")
    if not AIM_SERVICE_FILE.is_file() or not CONFIGMAP_FILE.is_file():
        raise SystemExit(f"expected {AIM_SERVICE_FILE} and {CONFIGMAP_FILE}")

    ns_proc = run(["kubectl", "get", "namespace", args.namespace], check=False, capture=True)
    if ns_proc.returncode != 0:
        raise SystemExit(f"namespace {args.namespace} does not exist")

    precision = args.precision.lower()
    metric = args.metric.lower()

    models = kubectl_json(["get", "aimclustermodel"]).get("items") or []
    resolved = resolve_model(models, args.model, args.name)
    print(f"AIMClusterModel {resolved['name']}  AIMID={resolved['aimId']}")

    profiles = kubectl_json(["get", "aimclusterprofile"]).get("items") or []
    matches = match_profiles(
        profiles,
        resolved["name"],
        resolved["aimId"],
        args.accelerator_count,
        precision,
        metric,
    )
    print("matched AIMClusterProfile(s):")
    for row in matches:
        print(
            "  {name}  {hw}  count={count}  {precision}  {metric}  {profileId}".format(
                name=row["name"],
                hw=row["acceleratorModel"] or "?",
                count=row["acceleratorCount"],
                precision=row["precision"],
                metric=row["metric"],
                profileId=row["profileId"],
            )
        )

    patch_aim_service(
        AIM_SERVICE_FILE,
        resolved["name"],
        args.accelerator_count,
        precision,
        metric,
    )
    print(
        f"updated {AIM_SERVICE_FILE} model.name={resolved['name']} "
        f"acceleratorCount={args.accelerator_count} precision={precision} metric={metric}"
    )

    live_model = live_model_name(args.namespace, AIM_SERVICE_NAME)
    model_changed = live_model is not None and live_model != resolved["name"]
    if live_model is None:
        print(f"AIMService/{AIM_SERVICE_NAME} does not exist yet; it will be created")
    elif model_changed:
        print(
            f"AIMService/{AIM_SERVICE_NAME} currently serves {live_model}; "
            "spec.model is immutable, so it must be deleted and recreated"
        )
    else:
        print(f"AIMService/{AIM_SERVICE_NAME} already serves {live_model}; patching selector in place")

    if args.dry_run:
        print("dry-run: not applying, not waiting for /v1/models, not restarting")
        if model_changed:
            print(
                f"dry-run: would delete AIMService/{AIM_SERVICE_NAME} and recreate it "
                "from the file above (kubectl diff would report the immutable-model error)"
            )
        else:
            run(
                [
                    "kubectl",
                    "--namespace",
                    args.namespace,
                    "diff",
                    "--filename",
                    str(AIM_SERVICE_FILE),
                ],
                check=False,
            )
        return 0

    if model_changed:
        if args.no_recreate:
            raise SystemExit(
                "model change requires deleting AIMService/"
                f"{AIM_SERVICE_NAME}; re-run without --no-recreate"
            )
        print(f"deleting AIMService/{AIM_SERVICE_NAME} (warm caches are preserved)")
        delete_aimservice(args.namespace, AIM_SERVICE_NAME, args.timeout)

    print(f"applying AIMService/{AIM_SERVICE_NAME} into namespace {args.namespace}")
    run(
        [
            "kubectl",
            "apply",
            "--namespace",
            args.namespace,
            "--server-side",
            "--force-conflicts",
            "--filename",
            str(AIM_SERVICE_FILE),
        ]
    )
    if not model_changed:
        # Same model, narrowed selector: nudge the sticky binding to re-rank.
        run(
            [
                "kubectl",
                "--namespace",
                args.namespace,
                "annotate",
                "aimservice",
                AIM_SERVICE_NAME,
                f"aim.eai.amd.com/force-rebind={int(time.time())}",
                "--overwrite",
            ],
            capture=True,
        )

    print(f"waiting for AIMService/{AIM_SERVICE_NAME} (timeout {args.timeout})")
    wait_aimservice(args.namespace, AIM_SERVICE_NAME, args.timeout)
    print(f"waiting for Service/{AIM_SERVICE_NAME} endpoints")
    wait_endpoints(args.namespace, AIM_SERVICE_NAME, args.timeout)

    print("reading served model id from http://dd-agentic-aim/v1/models")
    served_id = fetch_served_id(args.namespace, resolved["aimId"])
    print(f"served /v1/models id: {served_id}")

    patch_configmap_file(CONFIGMAP_FILE, served_id)
    patch_body = json.dumps({"data": {"REINVENT4_AGENT_MODEL": served_id}})
    run(
        [
            "kubectl",
            "--namespace",
            args.namespace,
            "patch",
            "configmap",
            CONFIGMAP_NAME,
            "--type",
            "merge",
            "--patch",
            patch_body,
        ]
    )
    print(f"updated ConfigMap/{CONFIGMAP_NAME} and {CONFIGMAP_FILE}")
    print(f"restarting deployment/{DEPLOYMENT_NAME}")
    run(
        [
            "kubectl",
            "--namespace",
            args.namespace,
            "rollout",
            "restart",
            f"deployment/{DEPLOYMENT_NAME}",
        ]
    )
    run(
        [
            "kubectl",
            "--namespace",
            args.namespace,
            "rollout",
            "status",
            f"deployment/{DEPLOYMENT_NAME}",
            "--timeout=20m",
        ]
    )
    print(f"done: AIM {resolved['name']} serving {served_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        err = exc.stderr or exc.stdout or str(exc)
        raise SystemExit(err) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None
