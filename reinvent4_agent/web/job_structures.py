# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The predicted complex of each molecule a REINVENT run cofolded.

A Boltz scoring component (``ExternalRESTScorer``) hands REINVENT one
structure path per molecule as metadata, so the run's results CSV carries it
as ``structure_path (<component>)`` next to the Boltz values
(``ligand_iptm (<component>)``, ``affinity_pred_value (<component>)`` ...).
This reads those columns back into one entry per SMILES, its best-scoring
row, with the path relative to the shared volume so ``/api/artifacts`` can
serve the file to the viewer.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path
from typing import Any

from reinvent4_agent.web.artifacts import _STRUCTURE_METRICS
from reinvent4_agent.web.session_data import shared_relative

logger = logging.getLogger(__name__)

_RESULTS = re.compile(r"^results_(\d+)\.csv$")
_SUFFIXED = re.compile(r"^(.+?)\s*\((.+)\)$")
_SMILES_KEYS = ("smiles",)
_SCORE_KEYS = ("score", "total_score")
_STEP_KEYS = ("step",)

#: Parsed results by job directory, kept while the CSVs are unchanged.
_cache: dict[Path, tuple[tuple[tuple[str, int, int], ...], list[dict[str, Any]]]] = {}


def results_files(job_dir: Path) -> list[Path]:
    """The run's results CSVs, one per stage, in stage order."""
    found = [(int(m.group(1)), p) for p in job_dir.glob("results_*.csv") if (m := _RESULTS.match(p.name))]
    return [p for _, p in sorted(found)]


def job_structures(job_dir: Path, shared_root: Path) -> list[dict[str, Any]] | None:
    """``[{smiles, step, score, path, metrics}]`` best score first, or None if there is no such job."""
    if not job_dir.is_dir():
        return None
    files = results_files(job_dir)
    try:
        signature = tuple((p.name, (s := p.stat()).st_mtime_ns, s.st_size) for p in files)
    except OSError:
        return []
    cached = _cache.get(job_dir)
    if cached and cached[0] == signature:
        return cached[1]
    best: dict[str, dict[str, Any]] = {}
    for path in files:
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                _read(csv.reader(handle), shared_root, best)
        except (OSError, csv.Error) as exc:
            logger.warning("Could not read %s: %s", path, exc)
    entries = sorted(best.values(), key=lambda e: -e["score"] if e["score"] is not None else math.inf)
    _cache[job_dir] = (signature, entries)
    return entries


def _read(rows: Any, shared_root: Path, best: dict[str, dict[str, Any]]) -> None:
    headers = [h.strip() for h in next(rows, [])]
    lower = [h.lower() for h in headers]
    smiles_at = _index(lower, _SMILES_KEYS)
    structure_at = [i for i, h in enumerate(lower) if h == "structure_path" or h.startswith("structure_path (")]
    if smiles_at is None or not structure_at:
        return
    score_at = _index(lower, _SCORE_KEYS)
    step_at = _index(lower, _STEP_KEYS)
    metric_at: list[tuple[int, str]] = []
    for i, header in enumerate(headers):
        match = _SUFFIXED.match(header)
        name = _STRUCTURE_METRICS.get(match.group(1) if match else header)
        if name and name != "failedChecks":
            metric_at.append((i, name))

    width = len(headers)
    for row in rows:
        # A file read while REINVENT appends to it can end in half a line.
        if len(row) != width:
            continue
        smiles = row[smiles_at].strip()
        raw = next((row[i].strip() for i in structure_at if row[i].strip()), "")
        rel = shared_relative(raw, shared_root) if smiles and raw else None
        if not rel or not rel.lower().endswith((".pdb", ".cif", ".mmcif")):
            continue
        score = _number(row[score_at]) if score_at is not None else None
        seen = best.get(smiles)
        if seen and (score is None or (seen["score"] is not None and seen["score"] >= score)):
            continue
        metrics: dict[str, float | int] = {}
        for i, name in metric_at:
            value = _number(row[i])
            if value is not None and name not in metrics:
                metrics[name] = int(value) if name.startswith("posebusters") else value
        step = _number(row[step_at]) if step_at is not None else None
        best[smiles] = {
            "smiles": smiles,
            "step": int(step) if step is not None else None,
            "score": score,
            "path": rel,
            "metrics": metrics,
        }


def _index(headers: list[str], keys: tuple[str, ...]) -> int | None:
    return next((i for i, h in enumerate(headers) if h in keys), None)


def _number(text: str) -> float | None:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None
