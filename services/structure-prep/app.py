"""Structure Preparation FastAPI service.

Endpoints:
  GET  /health           — liveness + tool versions
  GET  /schema           — PreparationProfile JSON schema
  POST /validate         — synchronous Stage 0 inventory + profile_preview (< 5 s)
  POST /jobs             — submit async preparation job; returns job_id
  GET  /jobs/{job_id}    — poll status; returns artefact paths on completion
  POST /jobs/{job_id}/cancel — cooperative cancellation

All structure paths are confined to SP_SHARED_DIR (default /shared).
Path traversal attempts are rejected with HTTP 400 before any file read.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from structure_prep.models import PreparationProfile, PreparationReport, JobRecord
from structure_prep.pipeline import run_pipeline, PrepError
from structure_prep.additives import get_registry

logger = logging.getLogger("structure_prep_service")

SHARED_DIR = Path(os.environ.get("SP_SHARED_DIR", "/shared"))
JOB_ROOT = SHARED_DIR / "structure-prep"
SP_MAX_CONCURRENCY = int(os.environ.get("SP_MAX_CONCURRENCY", "1"))

_prep_sem = asyncio.Semaphore(SP_MAX_CONCURRENCY)
_jobs: dict[str, JobRecord] = {}  # in-memory; consider disk persistence for restarts

app = FastAPI(title="Structure Preparator", version="1.0.0")

# ---- Path confinement ----

def _resolve_safe(path_str: str) -> Path:
    """Resolve path under SHARED_DIR, reject traversal attempts."""
    p = (SHARED_DIR / path_str).resolve()
    try:
        p.relative_to(SHARED_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Path outside shared dir: {path_str}")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path_str}")
    return p

# ---- Request models ----

class ValidateRequest(BaseModel):
    path: str
    profile: PreparationProfile | None = None

class PrepareRequest(BaseModel):
    path: str
    profile: PreparationProfile

# ---- Endpoints ----

@app.get("/health")
async def health():
    versions = {}
    try:
        import gemmi; versions["gemmi"] = gemmi.__version__
    except Exception: pass
    try:
        import openmm; versions["openmm"] = openmm.__version__
    except Exception: pass
    try:
        import pdb2pqr; versions["pdb2pqr"] = pdb2pqr.__version__
    except Exception: pass
    try:
        import rdkit; versions["rdkit"] = rdkit.__version__
    except Exception: pass
    try:
        import openff.toolkit; versions["openff_toolkit"] = openff.toolkit.__version__
    except Exception: pass
    return {
        "status": "ok",
        "service": "structure-prep",
        "version": "1.0.0",
        "shared_dir": str(SHARED_DIR),
        "shared_dir_exists": SHARED_DIR.exists(),
        "tool_versions": versions,
        "max_concurrency": SP_MAX_CONCURRENCY,
    }


@app.get("/schema")
async def schema():
    return PreparationProfile.model_json_schema()


@app.post("/validate")
async def validate(req: ValidateRequest):
    """Run Stage 0 (inventory only) synchronously. Does not modify any file."""
    try:
        return _validate_sync(req)
    except HTTPException:
        raise
    except PrepError as exc:
        # The message already carries a "[StageN]" prefix.
        raise HTTPException(status_code=422, detail={"error": str(exc)}) from exc
    except Exception as exc:  # noqa: BLE001
        # Without this the endpoint returns a bare 500 with no body, which
        # leaves the caller nothing to act on.
        logger.exception("validate failed for %s", req.path)
        raise HTTPException(
            status_code=500,
            detail={"error": f"{type(exc).__name__}: {exc}", "path": req.path},
        ) from exc


def _validate_sync(req: ValidateRequest):
    input_path = _resolve_safe(req.path)
    profile = req.profile or PreparationProfile()
    registry = get_registry(profile.stage3_curation.additive_registry_path)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_job_dir = Path(tmpdir)
        from structure_prep.stages.inventory import run as run_inventory
        structure, inventory = run_inventory(input_path, tmp_job_dir, registry)

    # Build profile_preview: proposed roles (using Stage 2 classification)
    from structure_prep.stages.roles import run as run_roles
    from structure_prep.stages.assembly import run as run_assembly
    _, s1 = run_assembly(structure, profile.stage1_assembly)
    _, role_map = run_roles(structure, profile.stage2_roles, inventory, registry)
    from structure_prep.construct import assess_construct
    construct_assessment = assess_construct(structure)

    profile_preview = {
        "proposed_protein_chains": role_map.protein_chains,
        "proposed_binder_chains": role_map.binder_chains,
        "ambiguous_groups": role_map.ambiguous_groups,
        "detected_ligands": [
            {"comp_id": g.comp_id, "chain": g.chain, "n_instances": g.n_instances,
             "classification": g.classification}
            for g in inventory.hetatm_groups
            if g.classification in ("likely_ligand", "cofactor", "covalent_ligand")
        ],
        "detected_additives": [
            {"comp_id": g.comp_id, "chain": g.chain, "category": g.registry_category}
            for g in inventory.hetatm_groups
            if g.classification in ("additive", "buffer_ion", "phasing_agent")
        ],
        "altloc_candidates": [
            {"chain": a.chain, "resseq": a.resseq, "resname": a.resname,
             "altloc_codes": a.altloc_codes, "near_ligand": a.near_ligand}
            for a in inventory.altloc_positions
        ],
        "missing_residues_count": sum(
            len(v.get("missing_from_atom", []))
            for v in inventory.seqres_vs_atom.values()
        ),
        "space_group": inventory.space_group,
        "construct_assessment": [
            assessment.model_dump(mode="json")
            for assessment in construct_assessment
        ],
        "construct_form": (
            "immature"
            if any(a.form == "immature" for a in construct_assessment)
            else (
                "mature"
                if construct_assessment
                and all(a.form == "mature" for a in construct_assessment)
                else "unknown"
            )
        ),
        "recommended_trim": (
            "native"
            if any(a.non_native_n_terminal for a in construct_assessment)
            else "none"
        ),
        "format_warnings": inventory.format_warnings,
    }

    return {"inventory": inventory.model_dump(), "profile_preview": profile_preview}


@app.post("/jobs", status_code=202)
async def submit_job(req: PrepareRequest, background_tasks: BackgroundTasks):
    """Submit an async preparation job. Returns job_id immediately."""
    try:
        input_path = _resolve_safe(req.path)
        job_id = uuid.uuid4().hex[:12]

        job_dir = JOB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        record = JobRecord(
            job_id=job_id,
            status="queued",
            input_path=str(input_path),
            profile=req.profile,
            job_dir=str(job_dir),
        )
        _jobs[job_id] = record

        background_tasks.add_task(_run_job, job_id, input_path, req.profile, job_dir)

        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("job submission failed for %s", req.path)
        raise HTTPException(
            status_code=500,
            detail={"error": f"{type(exc).__name__}: {exc}", "path": req.path},
        ) from exc


async def _run_job(job_id: str, input_path: Path, profile: PreparationProfile, job_dir: Path):
    """Background task: acquire semaphore and run pipeline."""
    from datetime import datetime
    record = _jobs[job_id]
    record.status = "running"
    record.started_at = datetime.utcnow().isoformat()

    async with _prep_sem:
        try:
            report = await asyncio.to_thread(
                run_pipeline, input_path, profile, job_id, job_dir
            )
            record.status = "completed"
            record.report = report
        except PrepError as e:
            record.status = "failed"
            record.error = str(e)
            logger.error("Job %s failed (PrepError): %s", job_id, e)
        except Exception as e:
            record.status = "failed"
            record.error = f"{type(e).__name__}: {e}"
            logger.exception("Job %s failed (unexpected): %s", job_id, e)
        finally:
            record.completed_at = datetime.utcnow().isoformat()


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll job status."""
    record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": record.status,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "error": record.error,
    }
    if record.status == "completed" and record.report:
        response["artifacts"] = record.report.artifacts.model_dump()
        response["report_summary"] = {
            "protonation_method": record.report.protonation_method,
            "n_h_atoms_added": record.report.n_h_atoms_added,
            "histidines": [h.model_dump() for h in record.report.histidines],
            "removed_additives": record.report.removed_additives,
            "warnings": record.report.warnings,
            "stage_timings_s": record.report.stage_timings_s,
        }
    return response


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cooperative cancellation."""
    record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if record.status in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": record.status, "message": "Job already finished"}
    record.status = "cancelled"
    return {"job_id": job_id, "status": "cancelled"}


# ---- Entrypoint ----

def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(
        "app:app",
        host=os.environ.get("SP_HOST", "0.0.0.0"),
        port=int(os.environ.get("SP_PORT", "8080")),
        workers=1,  # single worker (asyncio semaphore for serialization)
    )

if __name__ == "__main__":
    main()
