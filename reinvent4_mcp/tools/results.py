# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Results tools: get_optimization_status, get_top_candidates, read_results.

Provides monitoring and results extraction for agent-driven
convergence diagnostics during and after RL optimization runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reinvent4_mcp.jobs.results_csv import (
    is_component_column,
    read_results_csv,
    total_score_column,
)

logger = logging.getLogger(__name__)


def get_optimization_status(job_id: str) -> dict[str, Any]:
    """Get current status of an optimization job with convergence diagnostics.

    Returns the job state, step count, score trajectory, per-component
    statistics, and automated convergence diagnostics with actionable
    warnings and recommendations.

    The ``trajectory.diagnostics`` section provides:

    - **convergence_status**: ``'early'``, ``'improving'``,
      ``'plateau'``, ``'diverging'``, ``'stable'``, or ``'completed'``
    - **plateau_detected / plateau_since_step**: whether scores
      have stalled and from which step
    - **diverging**: whether scores are declining
    - **improvement_rate**: slope of score change in the recent window
    - **trend**: early/mid/recent mean scores and overall improvement
    - **component_health**: per-component NaN rates with status
      (``'healthy'``, ``'degraded'``, ``'failing'``)
    - **warnings**: actionable alerts (low validity, plateau, component
      failure, divergence)
    - **recommendations**: suggested next actions for the agent

    Parameters
    ----------
    job_id:
        ID of the optimization job to query.

    Returns
    -------
    dict
        Comprehensive status report including trajectory data and
        convergence diagnostics.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    manager = get_job_manager()
    return manager.get_status(job_id)


def get_top_candidates(
    job_id: Optional[str] = None,
    csv_path: Optional[str] = None,
    top_n: int = 50,
    min_score: Optional[float] = None,
) -> dict[str, Any]:
    """Extract the best molecules from a completed or running optimization.

    Reads the output CSV (either from a tracked job or a direct file
    path), filters by validity and optional minimum score, deduplicates
    by SMILES, and returns the top-N molecules ranked by total score
    with full component breakdowns.

    Parameters
    ----------
    job_id:
        ID of a tracked optimization job.
    csv_path:
        Direct path to a REINVENT4 output CSV (alternative to *job_id*).
    top_n:
        Number of top candidates to return.
    min_score:
        Minimum total score threshold.

    Returns
    -------
    dict
        List of top candidates with SMILES and component scores.
    """

    if job_id and csv_path:
        return {"error": "Provide either job_id or csv_path, not both."}

    if job_id:
        from reinvent4_mcp.jobs.manager import get_job_manager

        manager = get_job_manager()
        return manager.get_top_candidates(job_id, top_n, min_score)

    if csv_path:
        return _parse_csv_candidates(csv_path, top_n, min_score)

    return {"error": "Provide either job_id or csv_path."}


def read_results(csv_path: str) -> dict[str, Any]:
    """Parse a REINVENT4 output CSV and return summary statistics.

    Provides a high-level overview of an optimization run's output
    without requiring a tracked job ID.

    Parameters
    ----------
    csv_path:
        Path to a REINVENT4 summary CSV file.

    Returns
    -------
    dict
        Summary statistics and sample molecules.
    """

    path = Path(csv_path)
    if not path.exists():
        return {"error": f"CSV file not found: {csv_path}"}

    try:
        df = read_results_csv(path)
    except Exception as exc:
        return {"error": f"Failed to read CSV: {exc}"}

    if df.empty:
        return {"n_molecules": 0, "columns": list(df.columns)}

    result: dict[str, Any] = {
        "csv_path": csv_path,
        "n_rows": len(df),
        "columns": list(df.columns),
    }

    smiles_col = "SMILES" if "SMILES" in df.columns else None
    total_col = total_score_column(df)

    if smiles_col:
        n_unique = df[smiles_col].nunique()
        result["n_unique_smiles"] = n_unique

    if "SMILES_state" in df.columns:
        state_counts = df["SMILES_state"].value_counts().to_dict()
        result["smiles_states"] = {
            "valid": int(state_counts.get(1, 0)),
            "invalid": int(state_counts.get(0, 0)),
            "duplicate": int(state_counts.get(2, 0)),
        }
        n_valid = int(state_counts.get(1, 0))
        result["validity_rate"] = round(n_valid / len(df), 3) if len(df) > 0 else 0.0

    if total_col and total_col in df.columns:
        scores = df[total_col].dropna()
        if not scores.empty:
            result["total_score_stats"] = {
                "mean": round(float(scores.mean()), 4),
                "median": round(float(scores.median()), 4),
                "max": round(float(scores.max()), 4),
                "min": round(float(scores.min()), 4),
                "std": round(float(scores.std()), 4),
            }

    score_cols = [
        c for c in df.columns
        if is_component_column(c)
        and "(raw)" not in c
        and df[c].dtype in ("float64", "float32")
    ]
    if score_cols:
        component_summary: dict[str, dict[str, float]] = {}
        for col in score_cols:
            vals = df[col].dropna()
            if not vals.empty:
                component_summary[col] = {
                    "mean": round(float(vals.mean()), 4),
                    "max": round(float(vals.max()), 4),
                }
        result["component_summary"] = component_summary

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN or missing."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return None


def _parse_csv_candidates(
    csv_path: str,
    top_n: int,
    min_score: Optional[float],
) -> dict[str, Any]:
    """Parse a CSV file directly (not from a tracked job).

    Returns ranked candidates with full per-component breakdowns
    (transformed + raw scores) matching the output format of the
    job-manager version.
    """

    path = Path(csv_path)
    if not path.exists():
        return {"error": f"CSV file not found: {csv_path}"}

    try:
        df = read_results_csv(path)
    except Exception as exc:
        return {"error": f"Failed to read CSV: {exc}"}

    if df.empty:
        return {"candidates": [], "n_total": 0}

    smiles_col = "SMILES" if "SMILES" in df.columns else None
    total_col = total_score_column(df)

    if smiles_col is None:
        return {"error": "CSV has no SMILES column"}

    if "SMILES_state" in df.columns:
        df = df[df["SMILES_state"] == 1]

    if total_col and min_score is not None:
        df = df[df[total_col] >= min_score]

    if total_col:
        df = df.sort_values(total_col, ascending=False)

    df = df.drop_duplicates(subset=[smiles_col])
    df = df.head(top_n)

    score_columns = [
        c for c in df.columns
        if is_component_column(c)
        and df[c].dtype in ("float64", "float32", "int64", "int32")
        and "(raw)" not in c
        and c != total_col
    ]

    candidates: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        mol: dict[str, Any] = {"smiles": row[smiles_col]}
        if total_col:
            mol["total_score"] = _safe_float(row.get(total_col))

        components: dict[str, dict[str, Any]] = {}
        for col in score_columns:
            comp_data: dict[str, Any] = {"transformed": _safe_float(row.get(col))}
            raw_col = f"{col} (raw)"
            if raw_col in df.columns:
                comp_data["raw"] = _safe_float(row.get(raw_col))
            components[col] = comp_data

        if components:
            mol["components"] = components
        candidates.append(mol)

    return {
        "candidates": candidates,
        "n_total": len(candidates),
        "csv_path": csv_path,
    }
