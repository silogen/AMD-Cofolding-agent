"""External service discovery and direct-prediction helpers.

Two roles:

* ``list_external_services`` -- health + capability discovery for every
  configured external service. For the multi-endpoint predictors it also
  reports the selectors the agent must supply (admet: ``endpoint``;
  retrodmta: ``target`` [+ ``endpoint``]).
* ``predict_admet`` / ``predict_retrodmta`` -- direct (out-of-band) property
  prediction for the agent. These wrap the services' ``/predict`` endpoint so
  the agent can answer "what is the permeability of these molecules?" without
  authoring a TTP. The RL path still goes through ExternalRESTScorer + the TTP
  mapper (``/score``); both call the same services, so there is one source of
  truth for the model outputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import requests

from reinvent4_mcp.config import get_config

logger = logging.getLogger(__name__)

_TIMEOUT = 120

# File-extension buckets for shared-asset discovery.
_STRUCTURE_EXTS = {".pdb", ".cif", ".mmcif", ".ent"}
_MSA_EXTS = {".a3m", ".a2m", ".fasta", ".fa", ".sto"}


def _shared_root() -> Path:
    """Return the shared directory root (mount point) for asset discovery.

    Mirrors the interaction-profiler's ``IP_SHARED_DIR`` convention so the
    MCP server and the services agree on where the shared volume lives.
    """
    return Path(os.environ.get("IP_SHARED_DIR", "/shared"))


def _mtime(path: Path) -> float:
    """Modification time, or 0.0 for a file that went away mid-scan."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _a3m_query_length(path: Path) -> Optional[int]:
    """Length of the query (first) sequence of an a3m/fasta MSA, or None.

    Lets the agent pair an MSA with a matching-length target sequence instead
    of, e.g., combining a domain-only MSA with a full-length sequence.
    """
    try:
        seq_lines: list[str] = []
        seen_header = False
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seen_header:
                        break
                    seen_header = True
                    continue
                if seen_header:
                    seq_lines.append(line)
        seq = "".join(seq_lines).replace("-", "")
        return len(seq) or None
    except Exception:
        return None


#: How many paths to name per category. The volume accumulates one cofold
#: output per predicted molecule, so this list grows with every batch: at 1763
#: of them it came to 138k tokens, more than the conductor model's entire
#: 131072-token context, and the call failed with "max_tokens must be at least
#: 1, got -31572". The counts below are always exact; only the naming is cut.
SHARED_ASSET_SAMPLE = 10


def list_shared_assets(
    prefix: Optional[str] = None, limit: int = SHARED_ASSET_SAMPLE
) -> dict[str, Any]:
    """List usable files on the shared volume, categorized and /shared-relative.

    Scans the shared mount for the artifacts the agent needs so it never has
    to guess a path: hand-placed protein/complex ``structures`` (PDB/mmCIF),
    ``msas`` (.a3m/.fasta/...), and Boltz ``cofold_outputs`` (structures under
    a ``predictions/`` directory). Every returned path is expressed relative
    to ``/shared`` -- exactly the form ``profile_interactions`` and TTP
    ``msa_path`` accept. Returns an empty (non-error) result when nothing is
    found.

    A structure is in exactly one of ``structures`` and ``cofold_outputs``,
    by whether a run produced it. Cofold outputs are newest first, since the
    one worth naming is usually the one the last job just wrote.

    Parameters
    ----------
    prefix:
        Only consider paths starting with this, for finding the outputs of
        one job rather than every job: ``predictions/out_035c34d9``.
    limit:
        How many paths to name per category. ``counts`` reports how many
        there really are, so a cut is always visible.
    """
    root = _shared_root()
    result: dict[str, Any] = {
        "shared_dir": str(root),
        "structures": [],
        "msas": [],
        "msa_details": [],
        "cofold_outputs": [],
    }
    if prefix:
        result["prefix"] = prefix
    if not root.exists() or not root.is_dir():
        result["note"] = f"Shared directory {root} does not exist or is empty."
        return result

    structures: list[str] = []
    cofold: list[tuple[float, str]] = []
    msas: list[str] = []
    msa_details: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            continue
        if prefix and not rel.startswith(prefix):
            continue
        ext = path.suffix.lower()

        if ext in _STRUCTURE_EXTS:
            if "predictions" in path.parts:
                cofold.append((_mtime(path), rel))
            else:
                structures.append(rel)
        elif ext in _MSA_EXTS:
            msas.append(rel)
            # Report the query-sequence length so the agent pairs the MSA with
            # a matching-length target sequence (a domain MSA must go with the
            # domain sequence, not the full-length protein).
            msa_details.append({
                "path": rel,
                "query_length": _a3m_query_length(path),
            })

    cofold.sort(key=lambda pair: pair[0], reverse=True)
    cofold_paths = [rel for _, rel in cofold]

    result["structures"] = structures[:limit]
    result["msas"] = msas[:limit]
    result["msa_details"] = msa_details[:limit]
    result["cofold_outputs"] = cofold_paths[:limit]
    result["counts"] = {
        "structures": len(structures),
        "msas": len(msas),
        "cofold_outputs": len(cofold_paths),
    }

    withheld = {
        name: count - limit
        for name, count in result["counts"].items()
        if count > limit
    }
    if withheld:
        result["not_listed"] = withheld
        result["more"] = (
            "Narrow with prefix= to reach a specific job's files, or raise "
            "limit= to name more. Every count above is exact."
        )
    if not any(result["counts"].values()):
        result["note"] = (
            f"No structures, MSAs, or cofold outputs found under {root}"
            + (f" matching {prefix!r}." if prefix else ".")
        )
    return result


def _base_url(score_url: str) -> str:
    """Strip the trailing ``/score`` to get the service root URL."""
    return score_url.rsplit("/", 1)[0]


def _normalize_shared_path(path: Optional[str]) -> Optional[str]:
    """Rewrite a shared-volume path to one the services resolve under ``/shared``.

    The services mount the host ``.../data/shared`` directory at ``/shared``.
    Callers (the agent) naturally supply the *host* path they see when browsing
    (e.g. ``/home/me/dd_agentic/data/shared/predictions/...`` or the
    workspace-relative ``data/shared/predictions/...``), but the container only
    understands paths under ``/shared``.  Normalize by locating the
    ``data/shared/`` (or ``/shared/``) segment and re-rooting the remainder at
    ``/shared``.  Paths already under ``/shared`` and plain relative paths (which
    the service resolves under the shared dir) pass through unchanged.
    """
    if not path:
        return path
    p = str(path).replace("\\", "/")
    # Already a container-shared path.
    if p == "/shared" or p.startswith("/shared/"):
        return p
    # Re-root anything containing a ``data/shared/`` or ``/shared/`` marker.
    for marker in ("/data/shared/", "data/shared/", "/shared/"):
        idx = p.find(marker)
        if idx != -1:
            remainder = p[idx + len(marker):]
            return "/shared/" + remainder.lstrip("/")
    # Absolute host path with no shared marker -> cannot map; leave as-is so the
    # service reports a clear, specific error rather than us guessing.
    return p


def _get_service_url(service: str) -> str:
    """Return the configured ``/score`` URL for a service or raise."""
    url = get_config().enabled_external_services.get(service)
    if not url:
        raise ValueError(
            f"service {service!r} is not enabled; available: "
            f"{list(get_config().enabled_external_services)}"
        )
    return url


def list_external_services() -> dict[str, Any]:
    """Check health and report capabilities of all configured external services.

    For each enabled service, GET /health and record status. For the
    multi-endpoint predictors, additionally report the selectors the caller
    must supply so scoring does not silently fall back to a default:

    * ``admet``     -> ``endpoints`` (choose one via ``endpoint``)
    * ``retrodmta`` -> ``targets`` (choose one via ``target`` [+ ``endpoint``])
    """

    services = get_config().enabled_external_services

    results: dict[str, Any] = {}
    for name, url in services.items():
        base = _base_url(url)
        entry: dict[str, Any] = {"url": url, "health_url": f"{base}/health"}
        try:
            r = requests.get(f"{base}/health", timeout=5)
            health = r.json() if r.ok else {}
            entry["status"] = health.get("status", "unknown") if r.ok else "unhealthy"
        except Exception:
            entry["status"] = "unreachable"
            health = {}

        # Surface the required selector(s) so the agent authors valid calls.
        if name == "admet":
            entry["selector"] = "endpoint"
            entry["endpoints"] = health.get("endpoints", [])
            entry["usage"] = (
                "Direct: predict_admet(smiles, endpoints=[...]). "
                "RL objective: service_params={'endpoint': '<one of endpoints>'}."
            )
        elif name == "retrodmta":
            entry["selector"] = "target (+ optional endpoint)"
            entry["targets"] = health.get("targets", [])
            try:
                tr = requests.get(f"{base}/targets", timeout=5)
                if tr.ok:
                    entry["targets_detail"] = tr.json()
            except Exception:
                pass
            entry["usage"] = (
                "Direct: predict_retrodmta(smiles, target='FXA'). "
                "RL objective: service_params={'target': 'FXA', 'endpoint': 'FXA_pKi'}."
            )
        elif name == "interaction_profiler":
            entry["selector"] = "interaction_type (RL) / structure (interactions)"
            entry["interaction_types"] = [
                "hbond", "hydrophobic", "ionic", "pi_stacking", "pi_cation",
                "cation_pi", "halogen_bond", "metal", "water_bridge", "all",
            ]
            entry["usage"] = (
                "Direct: profile_interactions(smiles=[...]) for medchem fragments, or "
                "profile_interactions(structure_paths=[...]) for typed protein-ligand "
                "interactions. RL objective: service_params={'interaction_type': 'hbond'} "
                "with depends_on a structure producer (e.g. boltz) so a structure is available."
            )

        results[name] = entry

    return results


def predict_admet(
    smiles: list[str],
    endpoints: Optional[list[str]] = None,
    batch_size: Optional[int] = None,
) -> dict[str, Any]:
    """Predict ADMET properties for a list of SMILES (direct, out-of-band).

    Calls the ADMET service ``/predict`` (all endpoints), then filters to the
    requested ``endpoints`` if given. Returns per-molecule mean/std plus the
    applicability-domain flag. Use this to answer property questions directly;
    for RL scoring use a TTP external objective instead.
    """

    base = _base_url(_get_service_url("admet"))
    payload: dict[str, Any] = {"smiles": smiles, "format": "records"}
    if batch_size:
        payload["batch_size"] = batch_size
    r = requests.post(f"{base}/predict", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    records = r.json()

    if endpoints:
        wanted = set(endpoints)
        for rec in records:
            preds = rec.get("predictions", {})
            rec["predictions"] = {k: v for k, v in preds.items() if k in wanted}

    return {"service": "admet", "endpoints": endpoints, "n": len(smiles), "results": records}


def predict_retrodmta(
    smiles: list[str],
    target: str,
    window: str = "latest",
    method: str = "ensemble",
    batch_size: Optional[int] = None,
) -> dict[str, Any]:
    """Predict RetroDMTA project endpoints for a target (direct, out-of-band).

    ``target`` is required (e.g. 'FXA'); call ``list_external_services`` to see
    valid targets. Calls the service ``/predict`` and returns per-molecule
    per-endpoint mean/std for that target's endpoints. For RL scoring use a TTP
    external objective with ``service_params={'target': ...}`` instead.
    """

    base = _base_url(_get_service_url("retrodmta"))
    payload: dict[str, Any] = {
        "target": target,
        "smiles": smiles,
        "window": window,
        "method": method,
        "format": "records",
    }
    if batch_size:
        payload["batch_size"] = batch_size
    r = requests.post(f"{base}/predict", json=payload, timeout=_TIMEOUT)
    if r.status_code == 400:
        # surface the service's own message (e.g. unknown target) verbatim
        raise ValueError(r.json().get("detail", r.text))
    r.raise_for_status()
    records = r.json()

    return {
        "service": "retrodmta", "target": target, "window": window,
        "method": method, "n": len(smiles), "results": records,
    }


def profile_interactions(
    smiles: Optional[list[str]] = None,
    structure_paths: Optional[list[str]] = None,
    ph: float = 7.4,
    radius: float = 8.0,
    all_ligands: bool = False,
) -> dict[str, Any]:
    """Profile molecules with the interaction-profiler (direct, out-of-band).

    Two modes, combinable index-wise:
      * ``smiles`` only        -> medchem taxonomy (named functional-group / ring /
        scaffold fragments) for each molecule; no interactions (no structure).
      * ``structure_paths``    -> typed protein-ligand interactions (H-bond, ionic,
        hydrophobic, pi-stacking, ...) with residue/atom/geometry, plus fragments.
        Paths resolve under the service's shared dir. Pair with ``smiles`` (same
        index) to use the SMILES as a bond-order template and verify the ligand.

    Returns the profiler's rich per-molecule JSON (composition + interactions +
    triples). For RL scoring use a TTP external objective (service=
    'interaction_profiler', service_params={'interaction_type': ...}) instead.
    """

    base = _base_url(_get_service_url("interaction_profiler"))
    smiles = smiles or []
    structs = structure_paths or []
    n = max(len(smiles), len(structs))
    if n == 0:
        raise ValueError("provide smiles and/or structure_paths")

    molecules = []
    for i in range(n):
        molecules.append({
            "name": f"mol_{i}",
            "smiles": smiles[i] if i < len(smiles) else None,
            "structure": (
                _normalize_shared_path(structs[i]) if i < len(structs) else None
            ),
        })
    payload = {
        "batch_id": "profile",
        "molecules": molecules,
        "options": {"ph": ph, "radius": radius, "all_ligands": all_ligands},
    }
    r = requests.post(f"{base}/analyze", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()
