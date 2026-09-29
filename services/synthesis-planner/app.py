"""Synthesis Planning Service — FastAPI application.

Exposes AiZynthFinder retrosynthesis and building-block availability over HTTP.

Endpoints
---------
GET  /health                  liveness + asset identity
GET  /schema                  JSON schema for all request/response models
POST /plan                    synchronous retrosynthesis (≤ 5 SMILES)
POST /availability            building-block stock lookup (synchronous)
POST /jobs                    asynchronous batch planning
GET  /jobs/{job_id}           job status + results
POST /jobs/{job_id}/cancel    best-effort job cancellation

Patterns
--------
- ADMET: batch SMILES, per-molecule status, order-preserving, lifespan warm-up.
- Boltz2: asyncio.to_thread + semaphore for long CPU work, job cancellation.
- RetroDMTA: manifest-based asset identity exposed in /health.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8086
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("synthesis_planner.service")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
SERVICE_VERSION = "1.0.0"

PLANNER_MAX_CONCURRENCY = int(os.environ.get("PLANNER_MAX_CONCURRENCY", "2"))
_plan_sem = asyncio.Semaphore(PLANNER_MAX_CONCURRENCY)

# Maximum SMILES for a synchronous /plan call; larger batches return a job_id.
SYNC_BATCH_LIMIT = int(os.environ.get("SYNC_BATCH_LIMIT", "5"))


# ---------------------------------------------------------------------------
# Job registry (in-memory, process-local)
# ---------------------------------------------------------------------------

class _Job:
    """Minimal in-process job record."""
    def __init__(self, job_id: str, smiles: list[str], params: dict) -> None:
        self.job_id   = job_id
        self.smiles   = smiles
        self.params   = params
        self.status   = "queued"   # queued | running | done | failed | cancelled
        self.result: dict | None = None
        self.error: str | None = None
        self._cancel  = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


_jobs: dict[str, _Job] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifespan — load models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AiZynthFinder assets at startup; fail fast if corrupt or missing."""
    from synthesis_planner.manifest import load_and_verify, build_aizynthfinder_config  # noqa: PLC0415
    from synthesis_planner.runner import SynthesisPlanner  # noqa: PLC0415

    logger.info("Loading asset manifest from %s …", MODELS_DIR)
    try:
        manifest = load_and_verify(MODELS_DIR)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Asset verification failed: %s", exc)
        # Allow startup in dev/test mode without real model assets
        manifest = None
        app.state.planner = None
        app.state.manifest = None
        app.state.ready = False
        logger.warning(
            "Service is starting WITHOUT model assets — /plan will return "
            "503 until assets are provided.  Set MODELS_DIR correctly."
        )
        yield
        return

    config_dict = build_aizynthfinder_config(manifest, MODELS_DIR)
    try:
        planner = SynthesisPlanner(config_dict)
    except Exception as exc:
        logger.error("Failed to load AiZynthFinder: %s", exc)
        app.state.planner = None
        app.state.manifest = manifest
        app.state.ready = False
        yield
        return

    app.state.planner = planner
    app.state.manifest = manifest
    app.state.ready = True
    logger.info("Synthesis planner ready (policy=%s, templates=%s).",
                manifest.policy_version(), manifest.templates_version())
    yield


app = FastAPI(
    title="Synthesis Planner",
    description=(
        "AiZynthFinder-based retrosynthesis and building-block availability service."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PlanParams(BaseModel):
    """Search parameters forwarded to AiZynthFinder."""
    max_transforms: int = Field(
        default=6, ge=1, le=12,
        description="Maximum retrosynthesis tree depth (steps).",
    )
    iteration_limit: int = Field(
        default=100, ge=1, le=2000,
        description="Maximum MCTS iterations per molecule.",
    )
    time_limit: float = Field(
        default=120.0, ge=1.0, le=600.0,
        description="Wall-clock time limit per molecule in seconds.",
    )
    return_first: bool = Field(
        default=False,
        description="Stop at the first valid route found.",
    )
    policy: str = Field(
        default="default",
        description="Expansion policy key (must exist in the asset manifest).",
    )
    stock: str = Field(
        default="default",
        description="Stock catalog key for in-stock leaf checks.",
    )


class PlanRequest(BaseModel):
    smiles: list[str] = Field(description="Candidate SMILES to plan routes for.")
    params: PlanParams = Field(default_factory=PlanParams)


class PlanMoleculeResult(BaseModel):
    """Route-planning result for one candidate SMILES."""
    input_smiles: str
    canonical_smiles: Optional[str] = None
    status: str = Field(
        description="success | no_route | timeout | invalid_structure | "
                    "unsupported_chemistry | service_failure",
    )
    routes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Route candidates, sorted by score descending.",
    )
    n_routes: int = 0
    search_stats: dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    warning: Optional[str] = None
    elapsed_seconds: float = 0.0


class PlanResponse(BaseModel):
    results: list[PlanMoleculeResult] = Field(
        description="One entry per input SMILES, order-preserving.",
    )
    model_version: str
    templates_version: str
    stock_version: str
    n_molecules: int
    n_success: int
    n_no_route: int
    n_failed: int


class AvailabilityRequest(BaseModel):
    smiles: list[str] = Field(description="SMILES to check against the stock catalog.")
    stock: str = Field(default="default", description="Stock catalog key.")


class BuildingBlockStatus(BaseModel):
    smiles: str
    canonical_smiles: Optional[str] = None
    availability: str = Field(
        description="available | unavailable | unknown",
    )
    in_stock: Optional[bool] = None
    stock_source: Optional[str] = None
    warning: Optional[str] = None


class AvailabilityResponse(BaseModel):
    results: list[BuildingBlockStatus]
    stock_version: str
    n_molecules: int
    n_available: int
    n_unavailable: int
    n_unknown: int


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str = "queued"
    n_molecules: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    n_molecules: int
    result: Optional[PlanResponse] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_planner() -> Any:
    """Return the loaded SynthesisPlanner or raise 503."""
    planner = getattr(app.state, "planner", None)
    if planner is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Synthesis planner is not ready — model assets may not be "
                "loaded.  Check service logs and ensure MODELS_DIR is correct."
            ),
        )
    return planner


def _get_manifest() -> Any:
    return getattr(app.state, "manifest", None)


def _plan_result_to_model(pr: Any) -> PlanMoleculeResult:
    return PlanMoleculeResult(
        input_smiles=pr.input_smiles,
        canonical_smiles=pr.canonical_smiles,
        status=pr.status.value if hasattr(pr.status, "value") else str(pr.status),
        routes=pr.routes,
        n_routes=pr.n_routes,
        search_stats=pr.search_stats,
        failure_reason=pr.failure_reason,
        warning=pr.warning,
        elapsed_seconds=pr.elapsed_seconds,
    )


def _build_plan_response(results: list[Any], manifest: Any) -> PlanResponse:
    mol_results = [_plan_result_to_model(r) for r in results]
    statuses = [r.status for r in mol_results]
    return PlanResponse(
        results=mol_results,
        model_version=manifest.policy_version() if manifest else "unknown",
        templates_version=manifest.templates_version() if manifest else "unknown",
        stock_version=manifest.stock_version() if manifest else "unknown",
        n_molecules=len(mol_results),
        n_success=sum(1 for s in statuses if s == "success"),
        n_no_route=sum(1 for s in statuses if s in ("no_route", "timeout")),
        n_failed=sum(1 for s in statuses if s in (
            "invalid_structure", "unsupported_chemistry", "service_failure"
        )),
    )


def _run_batch(
    planner: Any,
    smiles_list: list[str],
    params: PlanParams,
    cancel_event: threading.Event | None = None,
) -> list[Any]:
    """Blocking batch planning — runs off the event loop via asyncio.to_thread."""
    results = []
    for smi in smiles_list:
        if cancel_event and cancel_event.is_set():
            break
        result = planner.plan(
            smi,
            max_transforms=params.max_transforms,
            iteration_limit=params.iteration_limit,
            time_limit=params.time_limit,
            return_first=params.return_first,
            policy=params.policy,
            stock=params.stock,
        )
        results.append(result)
    return results


def _run_job(job: _Job, planner: Any) -> None:
    """Background thread function for async job execution."""
    with _jobs_lock:
        if job.status == "cancelled":
            return
        job.status = "running"

    try:
        from synthesis_planner.availability import check_availability  # noqa: PLC0415
        params = PlanParams(**job.params)
        results = _run_batch(planner, job.smiles, params, cancel_event=job._cancel)
        manifest = _get_manifest()
        response = _build_plan_response(results, manifest)

        with _jobs_lock:
            if not job.cancelled:
                job.status = "done"
                job.result = response.model_dump()
            else:
                job.status = "cancelled"
    except Exception as exc:
        logger.exception("Job %s failed", job.job_id)
        with _jobs_lock:
            job.status = "failed"
            job.error = str(exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + readiness + asset identity."""
    manifest = _get_manifest()
    ready = getattr(app.state, "ready", False)
    return {
        "status": "ok" if ready else "degraded",
        "service": "synthesis-planner",
        "version": SERVICE_VERSION,
        "ready": ready,
        "model": manifest.policy_version() if manifest else None,
        "templates": manifest.templates_version() if manifest else None,
        "stock": manifest.stock_version() if manifest else None,
        "stock_keys": manifest.stock_keys() if manifest else [],
        "max_concurrency": PLANNER_MAX_CONCURRENCY,
        "sync_batch_limit": SYNC_BATCH_LIMIT,
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    """JSON schema for all request and response models."""
    return {
        "PlanRequest":          PlanRequest.model_json_schema(),
        "PlanResponse":         PlanResponse.model_json_schema(),
        "AvailabilityRequest":  AvailabilityRequest.model_json_schema(),
        "AvailabilityResponse": AvailabilityResponse.model_json_schema(),
        "JobSubmitResponse":    JobSubmitResponse.model_json_schema(),
        "JobStatusResponse":    JobStatusResponse.model_json_schema(),
    }


@app.post("/plan", response_model=PlanResponse)
async def plan(request: PlanRequest) -> PlanResponse:
    """Synchronous retrosynthesis for small batches (≤ SYNC_BATCH_LIMIT molecules).

    Larger batches should use ``POST /jobs``.  One molecule failure does not
    shift results for other candidates — every input SMILES has its own status.
    """
    planner  = _get_planner()
    manifest = _get_manifest()

    if not request.smiles:
        raise HTTPException(status_code=422, detail="smiles list must not be empty.")

    if len(request.smiles) > SYNC_BATCH_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Synchronous /plan accepts at most {SYNC_BATCH_LIMIT} molecules. "
                f"Submit {len(request.smiles)} molecules via POST /jobs instead."
            ),
        )

    async with _plan_sem:
        results = await asyncio.to_thread(
            _run_batch, planner, request.smiles, request.params,
        )

    return _build_plan_response(results, manifest)


@app.post("/availability", response_model=AvailabilityResponse)
def availability(request: AvailabilityRequest) -> AvailabilityResponse:
    """Synchronous stock lookup (fast — in-memory index).

    Returns ``unknown`` for every molecule when no stock catalog is loaded.
    Does not call external supplier APIs.
    """
    from synthesis_planner.availability import check_availability  # noqa: PLC0415

    planner  = _get_planner()
    manifest = _get_manifest()

    if not request.smiles:
        raise HTTPException(status_code=422, detail="smiles list must not be empty.")

    stock_obj = getattr(planner._finder, "stock", None) if planner._finder else None

    statuses = check_availability(
        request.smiles,
        stock=stock_obj,
        stock_key=request.stock,
    )

    results = [
        BuildingBlockStatus(
            smiles=s.smiles,
            canonical_smiles=s.canonical_smiles,
            availability=s.availability,
            in_stock=s.in_stock,
            stock_source=s.stock_source,
            warning=s.warning,
        )
        for s in statuses
    ]

    return AvailabilityResponse(
        results=results,
        stock_version=manifest.stock_version(request.stock) if manifest else "unknown",
        n_molecules=len(results),
        n_available=sum(1 for r in results if r.availability == "available"),
        n_unavailable=sum(1 for r in results if r.availability == "unavailable"),
        n_unknown=sum(1 for r in results if r.availability == "unknown"),
    )


class ScoreRequest(BaseModel):
    """Request model for the /score endpoint.

    Accepts both direct calls and REINVENT4 ExternalRESTScorer calls.

    ExternalRESTScorer wraps all extra TTP service_params inside a ``params``
    dict:  {"smiles": [...], "params": {"iteration_limit": 25, "score_fn": "solved_steps"}}
    Top-level fields (iteration_limit, score_fn) are used when called directly.
    The ``params`` dict overrides top-level defaults when present.
    """
    smiles: list[str] = Field(..., min_length=1)
    # ExternalRESTScorer wraps service_params here
    params: dict | None = Field(default=None, description="REINVENT4 ExternalRESTScorer params dict (overrides top-level fields)")
    # Direct-call fields (also accepted as params overrides)
    max_transforms: int = Field(default=3, ge=1, le=6)
    iteration_limit: int = Field(default=25, ge=5, le=200,
        description="Lower = faster but misses harder routes. 25 is the RL default.")
    score_fn: str = Field(default="solved_steps",
        description=(
            "How to convert retrosynthesis results to a [0,1] score:\n"
            "  solved_steps — 1.0 if solved in ≤2 steps, 0.7 if ≤4, 0.3 if routes "
            "found (any length), 0.0 if no route.\n"
            "  n_routes     — sigmoid on number of routes found (more = easier).\n"
            "  best_score   — raw AiZynthFinder MCTS top_score (0–1)."
        ))


class ScoreResult(BaseModel):
    smiles: str
    score: float
    n_routes: int
    shortest_steps: int | None
    is_solved: bool
    status: str


class ScoreResponse(BaseModel):
    """Response model for the /score endpoint.

    Follows the ExternalRESTScorer protocol expected by REINVENT4:
      - ``scores``: flat list of floats, one per input SMILES (in order)
      - ``metadata``: optional dict of per-molecule lists (same length as scores)
      - ``elapsed_seconds``: wall-clock time for the batch

    The legacy ``results`` field (list of ScoreResult) is also returned for
    direct callers that want the full per-molecule breakdown.
    """
    scores: list[float]                        # ExternalRESTScorer reads data["scores"]
    metadata: dict[str, list] | None = None    # per-molecule metadata lists
    results: list[ScoreResult] | None = None   # full breakdown for direct callers
    model_version: str | None = None
    elapsed_seconds: float


def _retro_score(result: "PlanMoleculeResult", score_fn: str) -> float:
    """Convert a /plan molecule result to a [0,1] RL score."""
    import math
    stats = result.search_stats or {}
    n_routes: int = result.n_routes or 0
    routes = result.routes or []
    steps_list = [r["n_steps"] for r in routes if r.get("n_steps", 0) > 0]
    steps = min(steps_list) if steps_list else None
    is_solved: bool = bool(stats.get("is_solved", False))

    if score_fn == "n_routes":
        return round(1 / (1 + math.exp(-0.15 * (n_routes - 10))), 4)
    elif score_fn == "best_score":
        return round(float(stats.get("top_score") or 0.0), 4)
    else:  # solved_steps (default)
        if n_routes == 0:
            return 0.0
        if steps is None or steps == 0:
            return 1.0
        if steps <= 2:
            return 1.0
        if steps <= 4:
            return 0.7
        return 0.3


@app.post("/score")
async def score(request: ScoreRequest) -> ScoreResponse:
    """Per-molecule AiZynthFinder synthesis score for REINVENT RL objectives.

    Returns a [0,1] score per SMILES based on retrosynthesis route quality.
    Designed to be called by REINVENT4 ExternalRESTScorer TTPs.

    REINVENT4 ExternalRESTScorer sends:
      {"smiles": [...], "params": {"iteration_limit": 25, "score_fn": "solved_steps"}}

    The response follows the ExternalRESTScorer protocol:
      {"scores": [0.7, 1.0, ...], "metadata": {...per-molecule lists...}}

    Score semantics (default score_fn='solved_steps'):
      1.0 — ≤2-step synthesis from catalogue building blocks
      0.7 — ≤4-step synthesis found
      0.3 — route found, >4 steps
      0.0 — no route found (AiZynthFinder MCTS exhausted without solution)

    Faster than /plan (lower iteration_limit=25) — suitable for RL step latency.
    """
    import time
    t0 = time.monotonic()

    # ExternalRESTScorer wraps service_params in a "params" dict.
    # Unpack and override top-level defaults so TTP iteration_limit/score_fn work.
    extra: dict = request.params or {}
    iteration_limit = int(extra.get("iteration_limit", request.iteration_limit))
    score_fn = str(extra.get("score_fn", request.score_fn))
    max_transforms = int(extra.get("max_transforms", request.max_transforms))

    planner = _get_planner()
    manifest = _get_manifest()

    plan_params = PlanParams(
        max_transforms=max_transforms,
        iteration_limit=iteration_limit,
        return_first=False,
        time_limit=60,
    )

    async with _plan_sem:
        raw_results = await asyncio.to_thread(
            _run_batch, planner, request.smiles, plan_params,
        )

    scored = []
    for r in raw_results:
        stats = r.search_stats or {}
        routes = r.routes or []
        steps_list = [rt["n_steps"] for rt in routes if rt.get("n_steps", 0) > 0]
        shortest = min(steps_list) if steps_list else None
        scored.append(ScoreResult(
            smiles=r.input_smiles or "",
            score=_retro_score(r, score_fn),
            n_routes=r.n_routes or 0,
            shortest_steps=shortest,
            is_solved=bool(stats.get("is_solved", False)),
            status=r.status or "unknown",
        ))

    model_version = manifest.policy_version() if manifest else None
    elapsed = round(time.monotonic() - t0, 3)

    # Build the ExternalRESTScorer-compatible response:
    #   scores  — flat list of floats (one per SMILES, in order)
    #   metadata — per-molecule lists of the same length
    return ScoreResponse(
        scores=[r.score for r in scored],
        metadata={
            "smiles": [r.smiles for r in scored],
            "n_routes": [r.n_routes for r in scored],
            "shortest_steps": [r.shortest_steps for r in scored],
            "is_solved": [r.is_solved for r in scored],
            "status": [r.status for r in scored],
        },
        results=scored,           # full breakdown for direct callers
        model_version=model_version,
        elapsed_seconds=elapsed,
    )


@app.post("/jobs", response_model=JobSubmitResponse)
async def submit_job(request: PlanRequest) -> JobSubmitResponse:
    """Submit an asynchronous batch planning job.

    Returns a ``job_id`` immediately.  Poll ``GET /jobs/{job_id}`` for status.
    """
    planner = _get_planner()

    if not request.smiles:
        raise HTTPException(status_code=422, detail="smiles list must not be empty.")

    job_id = uuid.uuid4().hex[:16]
    job = _Job(job_id, list(request.smiles), request.params.model_dump())

    with _jobs_lock:
        _jobs[job_id] = job

    # Launch in a daemon background thread
    thread = threading.Thread(
        target=_run_job, args=(job, planner), daemon=True, name=f"job-{job_id}"
    )
    thread.start()

    return JobSubmitResponse(job_id=job_id, status="queued", n_molecules=len(request.smiles))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    """Poll the status of an async planning job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    response_result = None
    if job.status == "done" and job.result:
        response_result = PlanResponse(**job.result)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        n_molecules=len(job.smiles),
        result=response_result,
        error=job.error,
    )


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    """Best-effort cancellation of a pending or running job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    if job.status in ("done", "failed", "cancelled"):
        return {"job_id": job_id, "status": job.status, "message": "Job already finished."}

    job.cancel()
    with _jobs_lock:
        if job.status == "queued":
            job.status = "cancelled"

    return {"job_id": job_id, "status": "cancellation_requested"}
