# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PoseBusters FastAPI wrapper — validates plausibility of predicted 3D poses via REST."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("posebusters_wrapper")
logger.setLevel(logging.INFO)

app = FastAPI(title="PoseBusters Scoring Wrapper")

MODES_REQUIRING_PROTEIN = frozenset(
    {"dock", "dock_fast", "redock", "redock_fast", "gen", "gen_fast"}
)
MODES_REQUIRING_REFERENCE = frozenset(
    {"redock", "redock_fast", "gen", "gen_fast"}
)
VALID_MODES = frozenset(
    {
        "dock",
        "dock_fast",
        "redock",
        "redock_fast",
        "gen",
        "gen_fast",
        "mol",
        "mol_fast",
    }
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ScoreParams(BaseModel):
    mode: str = "dock"
    protein_pdb_path: Optional[str] = None
    structure_paths: list[Optional[str]]
    reference_paths: Optional[list[Optional[str]]] = None


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Boltz2 ligand extraction
# ---------------------------------------------------------------------------


def _boltz_pdb_to_sdf(pdb_path: Path, resname: str = "LIG") -> Path:
    """Extract ligand HETATM records from a Boltz2 complex PDB and convert to SDF."""
    from rdkit import Chem

    lines = [
        line
        for line in pdb_path.read_text().splitlines()
        if (line.startswith("HETATM") and line[17:20].strip() == resname)
        or line.startswith("CONECT")
    ]
    if not any(line.startswith("HETATM") for line in lines):
        raise ValueError(f"No HETATM records with resname '{resname}' in {pdb_path}")

    lig_pdb_text = "\n".join(lines + ["END"])
    mol = Chem.MolFromPDBBlock(lig_pdb_text, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse ligand from {pdb_path}")

    sdf_path = pdb_path.with_name(pdb_path.stem + "_lig.sdf")
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return sdf_path


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------


def _score_single_molecule(
    buster,
    structure_path: Path,
    mol_cond: Optional[Path],
    mol_true: Optional[Path],
) -> tuple[Optional[float], dict[str, Any]]:
    """Run PoseBusters checks on one molecule. Returns ``(score, per_check_dict)``."""
    try:
        bust_kwargs: dict[str, Any] = {"mol_pred": [structure_path]}
        if mol_cond is not None:
            bust_kwargs["mol_cond"] = mol_cond
        if mol_true is not None:
            bust_kwargs["mol_true"] = [mol_true]

        df = buster.bust(**bust_kwargs)

        if df.empty:
            return None, {}

        row = df.iloc[0]
        bool_cols = df.select_dtypes(include="bool").columns

        n_passed = int(row[bool_cols].sum())
        n_total = int(row[bool_cols].count())
        score = n_passed / n_total if n_total > 0 else None

        checks: dict[str, Any] = {}
        for col in bool_cols:
            val = row[col]
            checks[col] = None if pd.isna(val) else bool(val)
        checks["n_checks_passed"] = n_passed
        checks["n_checks_total"] = n_total

        return score, checks

    except Exception:
        logger.exception("PoseBusters failed for %s", structure_path)
        return None, {}


def score_molecules(
    smiles: list[str],
    params: dict[str, Any],
) -> tuple[list[Optional[float]], dict[str, Any]]:
    """Score all molecules and return ``(scores, metadata)`` with per-check detail."""
    from posebusters import PoseBusters

    mode = params.get("mode", "dock")
    protein_pdb_path = params.get("protein_pdb_path")
    structure_paths = params["structure_paths"]
    reference_paths = params.get("reference_paths")

    buster = PoseBusters(config=mode, top_n=1)
    mol_cond = Path(protein_pdb_path) if protein_pdb_path else None

    if mol_cond and not mol_cond.exists():
        logger.warning("protein_pdb_path does not exist: %s", mol_cond)

    scores: list[Optional[float]] = []
    all_checks: list[dict[str, Any]] = []

    for i, struct_path_str in enumerate(structure_paths):
        if struct_path_str is None:
            logger.warning("Molecule %d: structure_path is null, skipping", i)
            scores.append(None)
            all_checks.append({})
            continue

        struct_path = Path(struct_path_str)
        if not struct_path.exists():
            logger.warning("Molecule %d: structure file not found: %s", i, struct_path)
            scores.append(None)
            all_checks.append({})
            continue

        try:
            struct_path = _boltz_pdb_to_sdf(struct_path)
            logger.info("Molecule %d: extracted ligand to %s", i, struct_path)
        except Exception:
            logger.exception("Molecule %d: ligand extraction failed for %s", i, struct_path)
            scores.append(None)
            all_checks.append({})
            continue

        logger.info("Molecule %d: validating %s (mode=%s)", i, struct_path, mode)

        mol_true = (
            Path(reference_paths[i])
            if reference_paths
            and i < len(reference_paths)
            and reference_paths[i] is not None
            else None
        )
        score, checks = _score_single_molecule(
            buster,
            struct_path,
            mol_cond,
            mol_true,
        )
        scores.append(score)
        all_checks.append(checks)

    # Pivot per-molecule dicts into per-check lists (preserving insertion order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for checks in all_checks:
        for k in checks:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    metadata: dict[str, list] = {}
    for key in all_keys:
        metadata[key] = [c.get(key) for c in all_checks]

    return scores, metadata


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _validated_mode(mode: str) -> str:
    """Return the allow-listed constant equal to ``mode``.

    Handing back the constant rather than the request string keeps
    caller-controlled text out of everything downstream, log records
    included.
    """
    canonical = next((valid for valid in VALID_MODES if valid == mode), None)
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode '{mode}'. "
                f"Must be one of: {sorted(VALID_MODES)}"
            ),
        )
    return canonical


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    params = request.params
    mode = _validated_mode(params.mode)

    if mode in MODES_REQUIRING_PROTEIN and not params.protein_pdb_path:
        raise HTTPException(
            status_code=400,
            detail=f"'protein_pdb_path' is required for mode '{mode}'",
        )

    if mode in MODES_REQUIRING_REFERENCE and not params.reference_paths:
        raise HTTPException(
            status_code=400,
            detail=f"'reference_paths' is required for mode '{mode}'",
        )

    if len(params.structure_paths) != len(request.smiles):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Length mismatch: {len(request.smiles)} SMILES "
                f"vs {len(params.structure_paths)} structure_paths"
            ),
        )

    if params.reference_paths and len(params.reference_paths) != len(request.smiles):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Length mismatch: {len(request.smiles)} SMILES "
                f"vs {len(params.reference_paths)} reference_paths"
            ),
        )

    try:
        t0 = time.monotonic()
        scores, metadata = await asyncio.to_thread(
            score_molecules, request.smiles, {**params.model_dump(), "mode": mode}
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "Scored %d molecules in %.2fs (mode=%s)",
            len(request.smiles),
            elapsed,
            mode,
        )
        return ScoreResponse(scores=scores, metadata=metadata)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during scoring")
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc)},
        )
