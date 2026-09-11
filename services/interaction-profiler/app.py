"""FastAPI application exposing the interaction profiler over HTTP.

Single service app (boltz2/posebusters convention): this is the only FastAPI app; the
implementation lives in the ``interaction_profiler`` package (installed into the image by the
Dockerfile).

DUAL ROLE:
  1. Out-of-band analysis for the agent -- ``/analyze`` and ``/profile`` emit rich per-complex
     descriptors (medchem taxonomy from SMILES; typed protein-ligand interaction geometry from a
     structure). The agent calls these via the ``profile_interactions`` MCP tool to reason about
     what a ligand is / would be doing in the pocket.
  2. REINVENT RL scorer -- ``/score`` implements the ``{smiles, params}`` -> ``{scores, metadata,
     artifacts}`` contract and returns, per input, the COUNT of interactions of a chosen type
     (``params.interaction_type``: hbond | ionic | hydrophobic | pi_stacking | ... | all).
     Counting interactions needs a structure, so for RL this objective is typically chained after
     a structure producer (e.g. Boltz cofold) via ``depends_on`` (the ``structure_paths`` are then
     injected, exactly like PoseBusters). SMILES-only inputs have no structure, so their score is
     None. The service returns the raw count; the TTP transform owns the direction. NOTE: "more
     interactions = better" can bias RL toward larger/greasier molecules -- prefer counting a
     specific interaction type (e.g. hbond) and/or a normalising transform.

* ``GET  /health``  -> liveness + tool versions.
* ``GET  /schema``  -> the published JSON schema of the batch result.
* ``POST /analyze`` -> profile a *list* of molecules (structures and/or SMILES).
* ``POST /profile`` -> convenience single-molecule endpoint.
* ``POST /score``   -> REINVENT scorer contract: one interaction-count scalar per input SMILES.

Structure paths resolve against a shared directory (``IP_SHARED_DIR``, default ``/shared``).

Run:  uvicorn app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from interaction_profiler.io.batch_input import MoleculeInput
from interaction_profiler.schemas import (
    BatchProvenance,
    BatchResult,
    MoleculeResult,
    detect_tool_versions,
)

logger = logging.getLogger("interaction_profiler.service")

# HTTP API version (independent of the profiler package version, which is
# reported per result under provenance.version).
SERVICE_VERSION = "0.1.0"


def _shared_dir() -> Path:
    return Path(os.environ.get("IP_SHARED_DIR", "/shared"))


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class MoleculeSpec(BaseModel):
    """One molecule to profile.

    Provide a ``structure`` (path, resolved under the shared dir if
    relative) and/or a ``smiles``. SMILES-only -> medchem taxonomy only.
    Structure (+ optional SMILES template) -> full interaction profile.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier for this molecule / complex.")
    smiles: str | None = Field(default=None, description="Ligand SMILES (template and/or taxonomy input).")
    structure: str | None = Field(
        default=None,
        description="Structure path (PDB/mmCIF). Relative paths resolve under the shared dir.",
    )
    ligand_resname: str | None = Field(default=None, description="Pin the ligand by residue name.")
    ligand_chain: str | None = Field(default=None, description="Pin the ligand by chain id.")
    ligand_keyword: str | None = Field(default=None, description="Case-insensitive substring match on the ligand residue id.")
    peptide_ligand_chain: str | None = Field(default=None, description="Treat this whole chain as the ligand.")
    source: str | None = Field(default=None, description="Free-form provenance tag, e.g. 'crystal' | 'cofold'.")
    ph: float | None = Field(default=None, ge=0.0, le=14.0, description="Override protonation pH for this molecule.")


class AnalyzeOptions(BaseModel):
    """Run-level options shared by every molecule in the request."""

    model_config = ConfigDict(extra="forbid")

    ph: float = Field(default=7.4, ge=0.0, le=14.0, description="Default protonation pH.")
    radius: float = Field(default=8.0, ge=1.0, le=25.0, description="Pocket crop radius (A).")
    max_interactions: int | None = Field(
        default=None,
        description="Cap on interactions kept per input. None = adaptive (~1.2x heavy atoms). 0 = no cap.",
    )
    drop_uncertain: bool = Field(default=True, description="Keep only confident interactions (drop uncertain after the cap).")
    all_ligands: bool = Field(default=False, description="Profile every qualifying ligand in each structure.")
    use_gpu: bool = Field(default=False, description="Use Torch GPU for distance matrices if available.")
    strict_ligand_match: bool = Field(
        default=False,
        description="For structure+SMILES inputs, fail the molecule when the prepared ligand's "
        "InChIKey skeleton does not match the SMILES template.",
    )


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(default="analyze", description="Label echoed back on the result.")
    molecules: list[MoleculeSpec] = Field(description="Molecules to profile (>=1).", min_length=1)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class ProfileRequest(BaseModel):
    """Single-molecule convenience request (spec fields + options inline)."""

    model_config = ConfigDict(extra="forbid")

    molecule: MoleculeSpec
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class HealthResponse(BaseModel):
    status: str = "ok"
    tool: str = "interaction-profiler"
    service_version: str = SERVICE_VERSION
    shared_dir: str
    shared_dir_exists: bool
    tool_versions: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer contract (REINVENT RL): count interactions of a chosen type
# ---------------------------------------------------------------------------


class ScoreParams(BaseModel):
    """Parameters for the /score scorer contract.

    ``extra="allow"`` so ExternalRESTScorer's forwarded keys are tolerated.
    """

    model_config = ConfigDict(extra="allow")

    interaction_type: str = Field(
        default="all",
        description="Interaction family to count: hbond | hydrophobic | ionic | pi_stacking | "
        "pi_cation | cation_pi | halogen_bond | metal | water_bridge | ... | all (total).",
    )
    structure_paths: list[str] | None = Field(
        default=None,
        description="Per-SMILES structure paths (index-aligned); injected from an upstream "
        "structure producer (e.g. Boltz) via depends_on. Required to count interactions.",
    )
    ph: float = Field(default=7.4, ge=0.0, le=14.0, description="Protonation pH.")
    radius: float = Field(default=8.0, ge=1.0, le=25.0, description="Pocket crop radius (A).")


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams = Field(default_factory=ScoreParams)


class ScoreResponse(BaseModel):
    scores: list[float | None]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core translation: request spec -> MoleculeInput
# ---------------------------------------------------------------------------


def _resolve_structure(structure: str | None) -> tuple[str | None, str | None]:
    """Return ``(resolved_path, error)``.

    Relative paths resolve under the shared dir. The resolved path is
    confined to the shared directory (``..`` traversal and absolute paths
    outside it are rejected) so a caller can only read structures from the
    mounted volume. A provided-but-missing path yields an explanatory error.
    """
    if not structure:
        return None, None
    shared = _shared_dir().resolve()
    p = Path(structure)
    p = p if p.is_absolute() else shared / p
    try:
        rp = p.resolve()
    except Exception:
        return str(p), f"invalid structure path: {structure!r}"
    if rp != shared and not rp.is_relative_to(shared):
        return str(rp), f"structure path escapes the shared directory: {structure!r}"
    if not rp.exists():
        return str(rp), f"structure not found: {rp}"
    if not rp.is_file():
        return str(rp), f"structure path is not a file: {rp}"
    return str(rp), None


def _spec_to_input(spec: MoleculeSpec) -> tuple[MoleculeInput | None, str | None]:
    structure, err = _resolve_structure(spec.structure)
    if err:
        return None, err
    if not structure and not spec.smiles:
        return None, "molecule has neither 'structure' nor 'smiles'"
    # Validate the SMILES up front (parse + InChIKey in one pass) so a typo
    # fails fast with a clear message instead of silently degrading to
    # geometry-perceived bond orders.
    if spec.smiles:
        from interaction_profiler.prep import validate_smiles

        ok, _ = validate_smiles(spec.smiles)
        if not ok:
            return None, f"invalid SMILES: {spec.smiles!r}"
    inp = MoleculeInput(
        name=spec.name,
        smiles=spec.smiles or None,
        structure=structure,
        ligand_resname=spec.ligand_resname or None,
        ligand_chain=spec.ligand_chain or None,
        ligand_keyword=spec.ligand_keyword or None,
        peptide_ligand_chain=spec.peptide_ligand_chain or None,
        source=spec.source or None,
        ph=spec.ph,
    )
    return inp, None


def _error_result(name: str, ph: float, err: str) -> MoleculeResult:
    from interaction_profiler.schemas import MoleculeMetadata

    md = MoleculeMetadata(name=name, ph=ph)
    return MoleculeResult(metadata=md, error=err)


def _profile_specs(specs: list[MoleculeSpec], opts: AnalyzeOptions, batch_id: str) -> BatchResult:
    from interaction_profiler.core import profile_molecule_all
    from interaction_profiler.interactions.geometry import gpu_available

    prov = BatchProvenance(
        interaction_radius=opts.radius,
        max_interactions=opts.max_interactions,
        drop_uncertain=opts.drop_uncertain,
        gpu=bool(opts.use_gpu and gpu_available()),
    )
    result = BatchResult(batch_id=batch_id, n_molecules=0, provenance=prov)

    for spec in specs:
        inp, err = _spec_to_input(spec)
        if err is not None:
            result.molecules.append(_error_result(spec.name, spec.ph or opts.ph, err))
            continue
        result.molecules.extend(
            profile_molecule_all(
                inp,
                ph_default=opts.ph,
                radius=opts.radius,
                use_gpu=opts.use_gpu,
                max_interactions=opts.max_interactions,
                drop_uncertain=opts.drop_uncertain,
                all_ligands=opts.all_ligands,
                strict_ligand_match=opts.strict_ligand_match,
            )
        )
    result.n_molecules = len(result.molecules)
    return result


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    # Default the prepared-receptor cache to a private writable dir so we never
    # write into the (possibly read-only, shared) artifact volume and stay
    # concurrency-safe. Callers can override IP_CACHE_DIR.
    os.environ.setdefault(
        "IP_CACHE_DIR", str(Path(tempfile.gettempdir()) / "interaction_profiler_receptors")
    )

    app = FastAPI(
        title="interaction-profiler",
        version=SERVICE_VERSION,
        description=(
            "Per-complex medchem taxonomy + tripartite (fragment, residue, "
            "interaction) profiling as an HTTP service. Dual role: out-of-band "
            "analysis (POST /analyze, /profile -> self-describing JSON) and a "
            "REINVENT RL scorer (POST /score -> one interaction-count scalar per "
            "input SMILES)."
        ),
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        sd = _shared_dir()
        return HealthResponse(
            shared_dir=str(sd),
            shared_dir_exists=sd.exists(),
            tool_versions=detect_tool_versions(),
        )

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return BatchResult.model_json_schema()

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> JSONResponse:
        t0 = time.time()
        try:
            result = _profile_specs(req.molecules, req.options, req.batch_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("analyze failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        logger.info(
            "analyze: %d molecule(s) in %.3fs", result.n_molecules, time.time() - t0
        )
        return JSONResponse(content=result.model_dump(mode="json", exclude_none=False))

    @app.post("/profile")
    def profile(req: ProfileRequest) -> JSONResponse:
        """Single-molecule convenience: returns the per-ligand result list."""
        try:
            result = _profile_specs([req.molecule], req.options, req.molecule.name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("profile failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        payload = {
            "provenance": result.provenance.model_dump(mode="json"),
            "results": [m.model_dump(mode="json", exclude_none=False) for m in result.molecules],
        }
        return JSONResponse(content=payload)

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        """REINVENT scorer contract: one interaction-count scalar per input SMILES.

        Counts interactions of ``params.interaction_type`` ('all' = total) on each
        molecule's structure. One result per input, index-aligned, None on failure
        or when no structure is available (interaction counts require a structure).
        """
        p = req.params
        itype = (p.interaction_type or "all").lower()
        structs = p.structure_paths or []
        opts = AnalyzeOptions(ph=p.ph, radius=p.radius)

        scores: list[float | None] = []
        counts: list[int | None] = []
        for i, smi in enumerate(req.smiles):
            struct = structs[i] if i < len(structs) else None
            spec = MoleculeSpec(name=f"mol_{i}", smiles=smi or None, structure=struct)
            try:
                # all_ligands defaults False -> exactly one primary result per input
                result = _profile_specs([spec], opts, f"score_{i}")
                mol = result.molecules[0] if result.molecules else None
            except Exception:  # noqa: BLE001
                logger.exception("score failed for input %d", i)
                mol = None

            if mol is None or mol.error or not mol.metadata.has_structure:
                scores.append(None)
                counts.append(None)
                continue

            if itype in ("all", "*", ""):
                n = len(mol.interactions)
            else:
                n = sum(1 for it in mol.interactions if it.interaction_type == itype)
            scores.append(float(n))
            counts.append(n)

        return ScoreResponse(
            scores=scores,
            metadata={"interaction_type": itype, "counts": counts},
        )

    return app


# Module-level app so ``uvicorn app:app`` works.
try:  # pragma: no cover - only when fastapi is installed
    app = create_app()
except Exception:  # pragma: no cover
    app = None


def main() -> None:  # pragma: no cover - console entry point
    import uvicorn

    host = os.environ.get("IP_HOST", "0.0.0.0")
    port = int(os.environ.get("IP_PORT", "8080"))
    workers = int(os.environ.get("IP_WORKERS", "1"))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":  # pragma: no cover
    main()
