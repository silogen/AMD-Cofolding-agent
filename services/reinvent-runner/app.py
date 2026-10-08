"""reinvent-runner — REINVENT4 job lifecycle and scoring behind REST.

Today the agent, the MCP server and REINVENT4 all live in one 11 GiB image, so
every agent replica pins a GPU for RL whether it is running a job or not. This
service is the GPU half of that split: it owns the ``JobManager`` (RL and
transfer-learning subprocesses), the in-process ``Scorer`` used by
``score_molecules``, and the component/model/transform registries. The agent
becomes an HTTP client and needs none of it.

**This is the staged first step.** The service deliberately *imports* the
existing ``reinvent4_mcp`` modules rather than moving code out of them, so the
job lifecycle — the most load-bearing code in the repo — is unchanged and can
be verified against a real RL run before anything is carved up. Splitting the
package so the agent image no longer needs REINVENT4 comes after that.

What stays with the agent: TTP storage and the TTP → scorer-config mapping. The
mapper is pure Python with no REINVENT import, and ``run_optimization`` already
resolves a TTP into a plain ``scorer_config`` dict before the manager ever sees
it. So the runner never needs the TTP store for RL, and ``/score`` takes the
TTP inline when it needs one.

**One worker only.** The ``JobManager`` is a per-process singleton holding the
job table and the concurrency slots in memory. A second uvicorn worker would
keep its own copy and double the effective ``max_concurrent_jobs``, which is
the one thing the slot limit exists to prevent.

Application errors are returned as ``200`` with ``{"error": ...}``, because
that is the manager's own contract and preserving it across the wire keeps the
client a drop-in replacement. Only transport and unexpected failures are 5xx.

Env:
    REINVENT4_MCP_CONFIG   process config, same file the MCP server reads
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("reinvent_runner")

app = FastAPI(
    title="REINVENT4 Runner",
    description=(
        "RL/TL job lifecycle, in-process scoring, and the REINVENT registries, "
        "so the agent does not need REINVENT4 or a GPU."
    ),
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    """Arguments of ``JobManager.create_and_start``.

    ``scorer_config`` arrives already mapped from the TTP by the caller, which
    is why the runner needs no TTP store. ``ttp_id``/``ttp_version`` are
    recorded on the job as labels only.
    """

    model_config = ConfigDict(extra="forbid")

    ttp_id: str
    ttp_version: int
    scorer_config: dict[str, Any]
    model: str = ".reinvent"
    max_steps: int = 300
    batch_size: Optional[int] = None
    sigma: Optional[int] = None
    learning_rate: Optional[float] = None
    resume_from: Optional[str] = None
    termination: Optional[dict[str, Any]] = None
    seed_smiles: Optional[list[str]] = None


class TLJobRequest(BaseModel):
    """Arguments of ``JobManager.create_and_start_tl``."""

    model_config = ConfigDict(extra="forbid")

    model: str = ".reinvent"
    smiles: Optional[list[str]] = None
    smiles_file: Optional[str] = None
    num_epochs: int = 50
    batch_size: int = 128
    save_every_n_epochs: int = 5
    sample_batch_size: int = 2000
    validation_smiles: Optional[list[str]] = None
    validation_smiles_file: Optional[str] = None
    standardize_smiles: bool = True
    randomize_smiles: bool = True
    output_model_file: str = "tl_model.model"


class ScoreRequest(BaseModel):
    """Arguments of ``score_molecules``.

    The TTP is passed **inline**: the caller owns the store, so sending the
    resolved object keeps this service stateless about TTPs.
    """

    model_config = ConfigDict(extra="forbid")

    smiles: list[str]
    components: Optional[list[dict[str, Any]]] = None
    aggregation: str = "geometric_mean"
    ttp: Optional[dict[str, Any]] = Field(
        default=None,
        description="Resolved TTP dict. Pass this rather than a ttp_id: the "
                    "runner has no TTP store.",
    )


# ---------------------------------------------------------------------------
# Lazy accessors
# ---------------------------------------------------------------------------

_local_manager = None


def _manager():
    """This process's own JobManager, deliberately not via ``get_job_manager``.

    That accessor honours ``runner.enabled``, and this service reads the same
    config file as the agent -- so going through it would make the runner
    forward every job to its own address. ``REINVENT_RUNNER_SELF`` guards the
    scoring and registry paths for the same reason; owning the manager here
    means the job endpoints cannot loop even if that env were lost.

    Imported late so module import stays cheap and torch is only pulled in on
    the first request that needs it.
    """
    global _local_manager
    if _local_manager is None:
        from reinvent4_mcp.jobs.manager import JobManager

        _local_manager = JobManager()
    return _local_manager


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus the two numbers a caller needs to place work.

    ``reinvent_importable`` is the one that matters on a fresh image: the whole
    point of this service is that REINVENT4 is here and not in the agent, so a
    false here means the image is wrong rather than the service being down.
    """
    from reinvent4_mcp.config import get_config

    config = get_config()
    reinvent_importable = True
    detail: Optional[str] = None
    try:
        from reinvent4_mcp.utils import ensure_reinvent4_importable

        ensure_reinvent4_importable()
    except Exception as exc:  # noqa: BLE001
        reinvent_importable = False
        detail = str(exc)[:300]

    active = _manager().active_job_ids()
    return {
        "status": "ok",
        "reinvent_importable": reinvent_importable,
        "reinvent_error": detail,
        "active_jobs": len(active),
        "max_concurrent_jobs": config.jobs.max_concurrent_jobs,
        "working_dir": config.jobs.working_dir,
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs")
def create_job(request: JobRequest) -> dict[str, Any]:
    """Start an RL optimization job. Refused when every slot is taken."""
    return _manager().create_and_start(**request.model_dump())


@app.post("/tl-jobs")
def create_tl_job(request: TLJobRequest) -> dict[str, Any]:
    """Start a transfer-learning job. Shares the slots with RL."""
    return _manager().create_and_start_tl(**request.model_dump())


@app.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return _manager().list_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _manager().get_status(job_id)


@app.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict[str, Any]:
    """Stop a job, checkpointing first when configured to."""
    return _manager().stop_job(job_id)


@app.get("/jobs/{job_id}/candidates")
def job_candidates(
    job_id: str,
    top_n: int = 10,
    min_score: float = 0.0,
) -> dict[str, Any]:
    return _manager().get_top_candidates(job_id, top_n, min_score)


@app.get("/alerts")
def alerts() -> list[dict[str, Any]]:
    """Diagnostics across all jobs (failing components, plateaus)."""
    return _manager().get_alerts()


# ---------------------------------------------------------------------------
# Scoring and registries
# ---------------------------------------------------------------------------

@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    """Score SMILES with REINVENT's in-process Scorer.

    No generation: this evaluates molecules against components or a TTP.
    """
    from reinvent4_mcp.tools.scoring import score_molecules

    return score_molecules(
        smiles=request.smiles,
        components=request.components,
        aggregation=request.aggregation,
        ttp=request.ttp,
    )


@app.get("/components")
def components() -> list[dict[str, Any]]:
    from reinvent4_mcp.tools.discovery import list_scoring_components

    return list_scoring_components()


@app.get("/models")
def models() -> list[dict[str, Any]]:
    from reinvent4_mcp.tools.discovery import list_available_models

    return list_available_models()


@app.get("/transforms")
def transforms() -> list[dict[str, Any]]:
    from reinvent4_mcp.tools.discovery import list_transforms

    return list_transforms()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
def unexpected_error(_request, exc: Exception):  # noqa: ANN201
    """Anything not already an error dict is a 500.

    The manager returns ``{"error": ...}`` for outcomes it expects, and those
    pass through as 200 so the client stays a drop-in. Reaching here means
    something genuinely unexpected happened.
    """
    from fastapi.responses import JSONResponse

    logger.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": f"{type(exc).__name__}: {exc}"[:500]},
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # One worker, deliberately: see the module docstring.
    uvicorn.run(app, host="0.0.0.0", port=8086, workers=1)
