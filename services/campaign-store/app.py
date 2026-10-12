"""Campaign store -- the control-plane ledger for a conductor-led agent fleet.

One conductor hands out work; N supervisors claim it, run a full RL campaign
each, and post back what they achieved. This service is the only thing that
knows the truth about that: which weight vectors are queued, who is holding
which task and until when, what came back, and which results are on the Pareto
front. Everything else is a client, so a conductor that restarts rebuilds its
loop by reading the ledger rather than from memory.

A note on vocabulary, because the codebase already uses the word differently:
here a **campaign** is the whole sweep (one goal, N supervisors, a wall-clock
budget) and a **task** is one supervisor run -- which is what
``reinvent4_agent/coordinator.py`` calls a campaign. The plan locks these
names; the collision is only in the English.

Design notes:

* **SQLite, single writer.** The Deployment is pinned to one replica with
  ``Recreate``, so there is exactly one process writing. SQLite then buys
  durability and atomic claims for none of the operational cost of a database
  server, and the file lives on ``/shared`` beside the job working directories
  it refers to. WAL mode so the conductor's polling of ``/status`` never blocks
  a supervisor posting a result.

* **Claims are atomic.** ``POST /tasks/claim`` selects and marks in one
  ``BEGIN IMMEDIATE`` transaction. Starlette runs these sync endpoints in a
  threadpool, so two supervisors really can arrive together; without the
  write-lock held across the select both would be handed the same task and
  would run the same weight vector on separate GPUs.

* **Leases, not assignments.** A supervisor holds a task for
  ``lease_seconds`` and has to heartbeat to keep it. A pod that is OOM-killed
  mid-campaign therefore releases its task without anyone having to notice the
  crash. Expiry is swept inline on claim and status rather than by a background
  task, so the behaviour is a function of the requests made and a test does not
  have to wait on a timer.

* **The front is computed, not stored.** Pareto membership changes every time a
  result arrives, so caching it would mean invalidating it correctly. Campaigns
  hold tens of tasks, not millions, and the objective count is single digits.

Env:
    CAMPAIGN_STORE_DB       SQLite path (default /shared/campaign-store/campaigns.db)
    CAMPAIGN_STORE_PORT     listen port (default 8087)
    TASK_MAX_ATTEMPTS       claims a task gets before it is abandoned (default 2)
    DEFAULT_LEASE_SECONDS   lease granted when a claim does not ask (default 900)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("campaign_store")

DB_PATH = Path(os.getenv("CAMPAIGN_STORE_DB", "/shared/campaign-store/campaigns.db"))
TASK_MAX_ATTEMPTS = int(os.getenv("TASK_MAX_ATTEMPTS", "2"))
DEFAULT_LEASE_SECONDS = float(os.getenv("DEFAULT_LEASE_SECONDS", "900"))

# Campaign states: `dispatching` accepts new tasks, `draining` is the plan's
# post-T window where the queue is closed but claimed tasks run on.
TaskState = Literal["queued", "claimed", "done", "failed", "abandoned"]

# A task not yet finished one way or another, so it still owes the campaign an
# answer. Spelled as SQL because it appears in three IN clauses.
OPEN_TASK_STATES = "('queued', 'claimed')"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id       TEXT PRIMARY KEY,
    goal              TEXT NOT NULL,
    base_ttp_id       TEXT NOT NULL,
    objectives        TEXT NOT NULL,
    concurrency       INTEGER NOT NULL,
    state             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    dispatch_until    REAL,
    hard_stop_at      REAL,
    finalized_at      TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id           TEXT PRIMARY KEY,
    campaign_id       TEXT NOT NULL REFERENCES campaigns(campaign_id),
    spec              TEXT NOT NULL,
    state             TEXT NOT NULL,
    attempts          INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    lease_owner       TEXT,
    lease_expires_at  REAL,
    claimed_at        TEXT,
    finished_at       TEXT,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS tasks_queue
    ON tasks (campaign_id, state, created_at);

CREATE TABLE IF NOT EXISTS results (
    task_id           TEXT PRIMARY KEY REFERENCES tasks(task_id),
    campaign_id       TEXT NOT NULL,
    job_id            TEXT,
    objective_profile TEXT NOT NULL,
    candidates        TEXT NOT NULL,
    summary           TEXT,
    posted_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id       TEXT,
    task_id           TEXT,
    kind              TEXT NOT NULL,
    payload           TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS events_feed ON events (campaign_id, event_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _connect() -> sqlite3.Connection:
    """A fresh connection; sqlite3 connections are not thread-safe to share.

    ``busy_timeout`` matters because claims take the write lock: a supervisor
    arriving mid-claim should wait its turn rather than get "database is
    locked".
    """

    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
    logger.info("campaign store ready at %s", DB_PATH)


@contextmanager
def _write() -> Iterator[sqlite3.Connection]:
    """A transaction that takes the write lock up front.

    Python's sqlite3 defers BEGIN until the first write, which would leave a
    select-then-update sequence racing another connection. ``BEGIN IMMEDIATE``
    is what makes the claim below safe.
    """

    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _read() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _emit(
    conn: sqlite3.Connection,
    kind: str,
    campaign_id: Optional[str],
    task_id: Optional[str] = None,
    **payload: Any,
) -> None:
    conn.execute(
        "INSERT INTO events (campaign_id, task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (campaign_id, task_id, kind, json.dumps(payload), _now_iso()),
    )


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

def _sweep_expired_leases(conn: sqlite3.Connection, campaign_id: Optional[str] = None) -> int:
    """Requeue or abandon tasks whose holder stopped heartbeating.

    Called on the paths that care (claim, status) instead of from a timer, so
    an expired lease is always observed before it could matter.
    """

    sql = (
        "SELECT task_id, campaign_id, attempts, lease_owner FROM tasks "
        "WHERE state = 'claimed' AND lease_expires_at IS NOT NULL "
        "AND lease_expires_at < ?"
    )
    args: list[Any] = [time.time()]
    if campaign_id:
        sql += " AND campaign_id = ?"
        args.append(campaign_id)

    expired = conn.execute(sql, args).fetchall()

    for row in expired:
        attempts = row["attempts"]
        if attempts < TASK_MAX_ATTEMPTS:
            conn.execute(
                "UPDATE tasks SET state = 'queued', lease_owner = NULL, "
                "lease_expires_at = NULL, claimed_at = NULL WHERE task_id = ?",
                (row["task_id"],),
            )
            kind = "task_requeued"
        else:
            conn.execute(
                "UPDATE tasks SET state = 'abandoned', lease_owner = NULL, "
                "lease_expires_at = NULL, finished_at = ?, error = ? WHERE task_id = ?",
                (_now_iso(), f"lease expired {attempts} times", row["task_id"]),
            )
            kind = "task_abandoned"

        logger.warning(
            "%s: lease held by %s expired (attempt %d)",
            row["task_id"], row["lease_owner"], attempts,
        )
        _emit(
            conn, kind, row["campaign_id"], row["task_id"],
            previous_owner=row["lease_owner"], attempts=attempts,
        )

    return len(expired)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def _dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """True when `a` is at least as good everywhere and better somewhere.

    Objective values are post-transform REINVENT scores, so higher is better on
    every axis and no per-objective direction is needed.
    """

    return all(a[o] >= b[o] for o in objectives) and any(a[o] > b[o] for o in objectives)


def pareto_front(
    results: list[dict[str, Any]], objectives: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split results into the non-dominated front and everything else.

    Results that do not report every campaign objective are returned as
    ineligible rather than silently dropped or defaulted to zero -- a missing
    objective usually means a scorer failed, and treating that as "scored 0"
    would quietly park a broken run at the bottom of the front instead of
    telling someone.
    """

    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []

    for result in results:
        profile = result["objective_profile"]
        missing = [o for o in objectives if not isinstance(profile.get(o), (int, float))]
        if missing:
            ineligible.append({**result, "missing_objectives": missing})
        else:
            eligible.append(result)

    front = [
        candidate for candidate in eligible
        if not any(
            _dominates(other["objective_profile"], candidate["objective_profile"], objectives)
            for other in eligible
            if other is not candidate
        )
    ]
    return front, ineligible


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class CampaignCreate(BaseModel):
    model_config = {"extra": "forbid"}

    goal: str
    base_ttp_id: str
    objectives: list[str] = Field(
        min_length=1,
        description="Objective names the Pareto front is computed over. Fixed at "
                    "creation so the front's dimensions cannot drift mid-campaign",
    )
    concurrency: int = Field(default=1, ge=1, description="Supervisors to hold in flight")
    dispatch_seconds: Optional[float] = Field(
        default=None, gt=0,
        description="Length of the dispatch window T. Omit for no deadline",
    )
    grace_seconds: float = Field(
        default=0.0, ge=0,
        description="Drain window after T, during which claimed tasks finish",
    )


class TaskCreate(BaseModel):
    model_config = {"extra": "forbid"}

    weight_overrides: dict[str, float] = Field(default_factory=dict)
    transform_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fidelity_tier: Literal["exploration", "validation"] = "exploration"
    max_steps: Optional[int] = Field(default=None, gt=0)
    deadline: Optional[str] = None
    token_budget: Optional[int] = Field(default=None, gt=0)
    max_turns: Optional[int] = Field(default=None, gt=0)


class ConcurrencyUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    concurrency: int = Field(ge=0, description="Supervisors to hold in flight")


class ClaimRequest(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str = Field(description="Supervisor identity, normally the pod name")
    campaign_id: Optional[str] = None
    lease_seconds: float = Field(default=DEFAULT_LEASE_SECONDS, gt=0)


class HeartbeatRequest(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    lease_seconds: float = Field(default=DEFAULT_LEASE_SECONDS, gt=0)


class ResultPost(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    objective_profile: dict[str, float] = Field(
        description="Achieved value per objective, the coordinates of this task "
                    "on the Pareto front",
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    job_id: Optional[str] = None
    summary: Optional[str] = None


class FailurePost(BaseModel):
    model_config = {"extra": "forbid"}

    owner: str
    error: str
    requeue: bool = Field(
        default=True,
        description="Whether the task may be retried by another supervisor",
    )


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

def _campaign_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "campaign_id": row["campaign_id"],
        "goal": row["goal"],
        "base_ttp_id": row["base_ttp_id"],
        "objectives": json.loads(row["objectives"]),
        "concurrency": row["concurrency"],
        "state": row["state"],
        "created_at": row["created_at"],
        "dispatch_until": row["dispatch_until"],
        "hard_stop_at": row["hard_stop_at"],
        "finalized_at": row["finalized_at"],
    }


def _task_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "campaign_id": row["campaign_id"],
        "spec": json.loads(row["spec"]),
        "state": row["state"],
        "attempts": row["attempts"],
        "created_at": row["created_at"],
        "lease_owner": row["lease_owner"],
        "lease_expires_at": row["lease_expires_at"],
        "claimed_at": row["claimed_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
    }


def _result_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "task_id": row["task_id"],
        "campaign_id": row["campaign_id"],
        "job_id": row["job_id"],
        "objective_profile": json.loads(row["objective_profile"]),
        "candidates": json.loads(row["candidates"]),
        "summary": row["summary"],
        "posted_at": row["posted_at"],
    }


def _get_campaign(conn: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No campaign {campaign_id}")
    return row


def _owned_task(conn: sqlite3.Connection, task_id: str, owner: str) -> sqlite3.Row:
    """The task, if this owner still holds its lease.

    A supervisor whose lease expired and was handed to someone else must not be
    able to post a result over the top of the new holder's work, so the owner
    check is a precondition rather than advisory.
    """

    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No task {task_id}")
    if row["state"] != "claimed":
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is {row['state']}, not claimed",
        )
    if row["lease_owner"] != owner:
        raise HTTPException(
            status_code=409,
            detail=f"Task {task_id} is held by {row['lease_owner']}, not {owner}",
        )
    return row


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="campaign-store",
    description="Control-plane ledger for the conductor/supervisor fleet",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    with _read() as conn:
        campaigns = conn.execute(
            "SELECT COUNT(*) AS n FROM campaigns WHERE state = 'dispatching'"
        ).fetchone()["n"]
        open_tasks = conn.execute(
            f"SELECT COUNT(*) AS n FROM tasks WHERE state IN {OPEN_TASK_STATES}"
        ).fetchone()["n"]

    return {
        "status": "healthy",
        "db": str(DB_PATH),
        "active_campaigns": campaigns,
        "open_tasks": open_tasks,
    }


# -- campaigns --------------------------------------------------------------

@app.post("/campaigns", status_code=201)
def create_campaign(body: CampaignCreate) -> dict[str, Any]:
    campaign_id = _new_id()
    now = time.time()
    dispatch_until = now + body.dispatch_seconds if body.dispatch_seconds else None
    hard_stop_at = dispatch_until + body.grace_seconds if dispatch_until else None

    with _write() as conn:
        conn.execute(
            "INSERT INTO campaigns (campaign_id, goal, base_ttp_id, objectives, "
            "concurrency, state, created_at, dispatch_until, hard_stop_at) "
            "VALUES (?, ?, ?, ?, ?, 'dispatching', ?, ?, ?)",
            (
                campaign_id, body.goal, body.base_ttp_id,
                json.dumps(body.objectives), body.concurrency,
                _now_iso(), dispatch_until, hard_stop_at,
            ),
        )
        _emit(conn, "campaign_created", campaign_id, goal=body.goal,
              concurrency=body.concurrency)
        row = _get_campaign(conn, campaign_id)

    logger.info("campaign %s created: %s", campaign_id, body.goal)
    return _campaign_row(row)


@app.get("/campaigns")
def list_campaigns() -> list[dict[str, Any]]:
    with _read() as conn:
        rows = conn.execute(
            "SELECT * FROM campaigns ORDER BY created_at DESC"
        ).fetchall()
    return [_campaign_row(r) for r in rows]


@app.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: str) -> dict[str, Any]:
    with _read() as conn:
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/campaigns/{campaign_id}/close")
def close_campaign(campaign_id: str) -> dict[str, Any]:
    """Enter the drain window: no new tasks, claimed ones run on."""

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET state = 'draining' WHERE campaign_id = ? "
            "AND state = 'dispatching'",
            (campaign_id,),
        )
        _emit(conn, "campaign_closed", campaign_id)
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/campaigns/{campaign_id}/concurrency")
def set_concurrency(campaign_id: str, body: ConcurrencyUpdate) -> dict[str, Any]:
    """Change how many supervisors the campaign wants in flight.

    Held here rather than only in the conductor's memory, so the number
    survives a conductor restart and the pool it scales to is always
    reconcilable against the ledger.
    """

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE campaigns SET concurrency = ? WHERE campaign_id = ?",
            (body.concurrency, campaign_id),
        )
        _emit(conn, "concurrency_changed", campaign_id, concurrency=body.concurrency)
        return _campaign_row(_get_campaign(conn, campaign_id))


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict[str, Any]:
    """Take a task back, whoever holds it.

    Deliberately not owner-checked: this is the conductor overriding a
    supervisor, not a supervisor reporting on itself. The holder finds out on
    its next heartbeat, which already treats a lost lease as "stop work and do
    not post a result".
    """

    with _write() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No task {task_id}")
        if row["state"] in ("done", "failed", "abandoned"):
            raise HTTPException(
                status_code=409,
                detail=f"Task {task_id} already finished as {row['state']}",
            )

        conn.execute(
            "UPDATE tasks SET state = 'abandoned', finished_at = ?, "
            "lease_owner = NULL, lease_expires_at = NULL, error = ? "
            "WHERE task_id = ?",
            (_now_iso(), "cancelled by the conductor", task_id),
        )
        _emit(conn, "task_cancelled", row["campaign_id"], task_id,
              previous_owner=row["lease_owner"])
        cancelled = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.warning("task %s cancelled (was held by %s)", task_id, row["lease_owner"])
    return _task_row(cancelled)


@app.post("/campaigns/{campaign_id}/finalize")
def finalize_campaign(campaign_id: str) -> dict[str, Any]:
    """Close the ledger. Still-queued tasks are abandoned, not left dangling."""

    with _write() as conn:
        _get_campaign(conn, campaign_id)
        conn.execute(
            "UPDATE tasks SET state = 'abandoned', finished_at = ?, "
            "error = 'campaign finalized', lease_owner = NULL, "
            "lease_expires_at = NULL "
            f"WHERE campaign_id = ? AND state IN {OPEN_TASK_STATES}",
            (_now_iso(), campaign_id),
        )
        conn.execute(
            "UPDATE campaigns SET state = 'finalized', finalized_at = ? "
            "WHERE campaign_id = ?",
            (_now_iso(), campaign_id),
        )
        _emit(conn, "campaign_finalized", campaign_id)
        return _campaign_row(_get_campaign(conn, campaign_id))


# -- tasks ------------------------------------------------------------------

@app.post("/campaigns/{campaign_id}/tasks", status_code=201)
def enqueue_task(campaign_id: str, body: TaskCreate) -> dict[str, Any]:
    with _write() as conn:
        campaign = _get_campaign(conn, campaign_id)
        if campaign["state"] != "dispatching":
            raise HTTPException(
                status_code=409,
                detail=f"Campaign {campaign_id} is {campaign['state']}; "
                       "the queue is closed to new tasks",
            )

        task_id = _new_id()
        conn.execute(
            "INSERT INTO tasks (task_id, campaign_id, spec, state, created_at) "
            "VALUES (?, ?, ?, 'queued', ?)",
            (task_id, campaign_id, body.model_dump_json(), _now_iso()),
        )
        _emit(conn, "task_enqueued", campaign_id, task_id,
              weight_overrides=body.weight_overrides,
              fidelity_tier=body.fidelity_tier)
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    return _task_row(row)


@app.get("/campaigns/{campaign_id}/tasks")
def list_tasks(
    campaign_id: str, state: Optional[TaskState] = None
) -> list[dict[str, Any]]:
    with _read() as conn:
        _get_campaign(conn, campaign_id)
        sql = "SELECT * FROM tasks WHERE campaign_id = ?"
        args: list[Any] = [campaign_id]
        if state:
            sql += " AND state = ?"
            args.append(state)
        rows = conn.execute(sql + " ORDER BY created_at", args).fetchall()
    return [_task_row(r) for r in rows]


@app.post("/tasks/claim")
def claim_task(body: ClaimRequest) -> Any:
    """Take the oldest queued task and lease it.

    Answers 204 when there is nothing to claim, which is the normal case for an
    idle supervisor and not worth an error.
    """

    with _write() as conn:
        _sweep_expired_leases(conn, body.campaign_id)

        sql = (
            "SELECT t.* FROM tasks t JOIN campaigns c USING (campaign_id) "
            "WHERE t.state = 'queued' AND c.state IN ('dispatching', 'draining')"
        )
        args: list[Any] = []
        if body.campaign_id:
            sql += " AND t.campaign_id = ?"
            args.append(body.campaign_id)

        row = conn.execute(sql + " ORDER BY t.created_at LIMIT 1", args).fetchone()
        if row is None:
            return Response(status_code=204)

        expires = time.time() + body.lease_seconds
        conn.execute(
            "UPDATE tasks SET state = 'claimed', lease_owner = ?, "
            "lease_expires_at = ?, claimed_at = ?, attempts = attempts + 1 "
            "WHERE task_id = ?",
            (body.owner, expires, _now_iso(), row["task_id"]),
        )
        _emit(conn, "task_claimed", row["campaign_id"], row["task_id"],
              owner=body.owner, lease_seconds=body.lease_seconds)

        claimed = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)
        ).fetchone()

    logger.info("task %s claimed by %s", claimed["task_id"], body.owner)
    return _task_row(claimed)


@app.post("/tasks/{task_id}/heartbeat")
def heartbeat(task_id: str, body: HeartbeatRequest) -> dict[str, Any]:
    with _write() as conn:
        _owned_task(conn, task_id, body.owner)
        expires = time.time() + body.lease_seconds
        conn.execute(
            "UPDATE tasks SET lease_expires_at = ? WHERE task_id = ?",
            (expires, task_id),
        )
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    return _task_row(row)


@app.post("/tasks/{task_id}/result")
def post_result(task_id: str, body: ResultPost) -> dict[str, Any]:
    with _write() as conn:
        task = _owned_task(conn, task_id, body.owner)
        conn.execute(
            "INSERT OR REPLACE INTO results (task_id, campaign_id, job_id, "
            "objective_profile, candidates, summary, posted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id, task["campaign_id"], body.job_id,
                json.dumps(body.objective_profile), json.dumps(body.candidates),
                body.summary, _now_iso(),
            ),
        )
        conn.execute(
            "UPDATE tasks SET state = 'done', finished_at = ?, lease_owner = NULL, "
            "lease_expires_at = NULL WHERE task_id = ?",
            (_now_iso(), task_id),
        )
        _emit(conn, "task_done", task["campaign_id"], task_id,
              owner=body.owner, objective_profile=body.objective_profile,
              n_candidates=len(body.candidates))
        row = conn.execute(
            "SELECT * FROM results WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.info("task %s done: %s", task_id, body.objective_profile)
    return _result_row(row)


@app.post("/tasks/{task_id}/fail")
def fail_task(task_id: str, body: FailurePost) -> dict[str, Any]:
    """Report a task as failed, requeuing it unless told not to."""

    with _write() as conn:
        task = _owned_task(conn, task_id, body.owner)
        requeue = body.requeue and task["attempts"] < TASK_MAX_ATTEMPTS

        if requeue:
            conn.execute(
                "UPDATE tasks SET state = 'queued', lease_owner = NULL, "
                "lease_expires_at = NULL, claimed_at = NULL, error = ? "
                "WHERE task_id = ?",
                (body.error, task_id),
            )
        else:
            conn.execute(
                "UPDATE tasks SET state = 'failed', finished_at = ?, "
                "lease_owner = NULL, lease_expires_at = NULL, error = ? "
                "WHERE task_id = ?",
                (_now_iso(), body.error, task_id),
            )

        _emit(conn, "task_requeued" if requeue else "task_failed",
              task["campaign_id"], task_id, owner=body.owner, error=body.error)
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    logger.warning("task %s failed (%s): %s", task_id,
                   "requeued" if requeue else "terminal", body.error)
    return _task_row(row)


# -- results, front, events -------------------------------------------------

# Scorer columns that say whether an objective's value means anything, as
# opposed to re-expressing it. Kept beside the objectives they qualify: an
# affinity from a failed fold is not a good score, and a reader who cannot see
# that is worse off than one who sees no affinity at all.
VALIDITY_METRICS = ("skip_reason", "confidence_score", "ligand_iptm")


def _objective_scores(
    components: dict[str, Any], objectives: list[str]
) -> dict[str, Any]:
    """An objective's own score plus whatever qualifies it."""

    scores = {name: components[name] for name in objectives if name in components}
    for objective in objectives:
        for metric in VALIDITY_METRICS:
            key = f"{metric} ({objective})"
            value = components.get(key)
            if value is None:
                continue
            # A null ``transformed`` is the scorer saying "nothing to report",
            # which for skip_reason is the good case and not worth the tokens.
            if isinstance(value, dict) and value.get("transformed") is None:
                continue
            scores[key] = value
    return scores


def _scored_candidate(
    candidate: dict[str, Any], objectives: list[str]
) -> dict[str, Any]:
    """One molecule with the scores the campaign is actually steering on.

    A harvested candidate carries every column the scorer emitted, which for
    Boltz is ten per-molecule columns beside the one objective they explain.
    That is 906 characters per molecule, so a 25-molecule result is ~22k --
    two of them overflow the conductor's 32768-token context on their own.

    What is dropped is the redundant half: six ``affinity_pred_value*`` and
    ``affinity_probability_binary*`` re-expressions of the same prediction.
    What is kept is the objectives and the validity signals above, because a
    conductor shown an affinity with no confidence beside it will narrate a
    quality gate it cannot see -- one did, inventing PoseBusters numbers for
    a campaign that never ran PoseBusters.
    """

    components = candidate.get("components") or {}
    return {
        "smiles": candidate.get("smiles"),
        "total_score": candidate.get("total_score"),
        "components": _objective_scores(components, objectives),
    }


@app.get("/campaigns/{campaign_id}/results")
def list_results(campaign_id: str, full: bool = False) -> list[dict[str, Any]]:
    """Finished results, narrowed to the campaign's objectives.

    ``full=true`` returns the stored record untouched, including every scorer
    diagnostic. The ledger keeps all of it; this endpoint defaults to the
    narrow view because its caller is an LLM with a bounded context.
    """

    with _read() as conn:
        campaign = _get_campaign(conn, campaign_id)
        objectives = json.loads(campaign["objectives"])
        rows = conn.execute(
            "SELECT * FROM results WHERE campaign_id = ? ORDER BY posted_at",
            (campaign_id,),
        ).fetchall()

    results = [_result_row(r) for r in rows]
    if full:
        return results

    narrowed = []
    for result in results:
        candidates = result.get("candidates") or []
        narrowed.append({
            **result,
            "objective_profile": _objective_scores(
                result["objective_profile"], objectives
            ),
            "candidates": [
                _scored_candidate(c, objectives) for c in candidates
            ],
            "detail_hint": (
                "These are the campaign's objectives and the signals that "
                "qualify them. The scorer's remaining per-molecule columns "
                "are in the job's results CSV via read_results(job_id)."
            ),
        })
    return narrowed


def _front_entry(result: dict[str, Any], objectives: list[str]) -> dict[str, Any]:
    """One front member as coordinates, not molecules.

    The front answers "which trade-offs are worth looking at"; the molecules
    behind a point are a follow-up question, answered by the result endpoint
    for that one task. Returning them inline made this response ~42k tokens
    for a three-member front -- larger on its own than the conductor model's
    32768-token context, so asking about the front failed outright with
    "max_tokens must be at least 1, got -72257".

    The profile is also narrowed to the campaign's declared objectives. A
    stored profile can carry a scorer's per-molecule diagnostics alongside
    them, and those are not axes of the front.
    """

    profile = result["objective_profile"]
    return {
        "task_id": result["task_id"],
        "job_id": result["job_id"],
        "objective_profile": {
            name: profile[name] for name in objectives if name in profile
        },
        "n_candidates": len(result.get("candidates") or []),
        "summary": result.get("summary"),
        "posted_at": result.get("posted_at"),
    }


@app.get("/campaigns/{campaign_id}/pareto")
def get_pareto(campaign_id: str) -> dict[str, Any]:
    with _read() as conn:
        campaign = _get_campaign(conn, campaign_id)
        objectives = json.loads(campaign["objectives"])
        rows = conn.execute(
            "SELECT * FROM results WHERE campaign_id = ?", (campaign_id,)
        ).fetchall()

    front, ineligible = pareto_front([_result_row(r) for r in rows], objectives)
    return {
        "campaign_id": campaign_id,
        "objectives": objectives,
        "n_results": len(rows),
        "front": [_front_entry(r, objectives) for r in front],
        "ineligible": [
            {**_front_entry(r, objectives),
             "missing_objectives": r.get("missing_objectives")}
            for r in ineligible
        ],
        "molecules_hint": (
            "Use get_supervisor_result(campaign_id, task_id) for the molecules "
            "behind a point."
        ),
    }


@app.get("/campaigns/{campaign_id}/events")
def list_events(campaign_id: str, since: int = 0, limit: int = 200) -> dict[str, Any]:
    """The conductor's completion feed: poll with the last event_id it saw."""

    with _read() as conn:
        _get_campaign(conn, campaign_id)
        rows = conn.execute(
            "SELECT * FROM events WHERE campaign_id = ? AND event_id > ? "
            "ORDER BY event_id LIMIT ?",
            (campaign_id, since, limit),
        ).fetchall()

    events = [
        {
            "event_id": r["event_id"],
            "task_id": r["task_id"],
            "kind": r["kind"],
            "payload": json.loads(r["payload"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return {
        "events": events,
        "last_event_id": events[-1]["event_id"] if events else since,
    }


@app.get("/campaigns/{campaign_id}/status")
def campaign_status(campaign_id: str) -> dict[str, Any]:
    """The aggregate the conductor REPL prints as /status.

    Sweeps expired leases first, so in-flight counts reflect supervisors that
    are actually still alive rather than ones that died holding a task.
    """

    with _write() as conn:
        campaign = _get_campaign(conn, campaign_id)
        _sweep_expired_leases(conn, campaign_id)

        counts = {
            row["state"]: row["n"]
            for row in conn.execute(
                "SELECT state, COUNT(*) AS n FROM tasks WHERE campaign_id = ? "
                "GROUP BY state",
                (campaign_id,),
            )
        }
        in_flight = [
            {
                "task_id": r["task_id"],
                "owner": r["lease_owner"],
                "claimed_at": r["claimed_at"],
                "lease_expires_in": round(r["lease_expires_at"] - time.time(), 1),
                "spec": json.loads(r["spec"]),
            }
            for r in conn.execute(
                "SELECT * FROM tasks WHERE campaign_id = ? AND state = 'claimed' "
                "ORDER BY claimed_at",
                (campaign_id,),
            )
        ]
        results = [
            _result_row(r) for r in conn.execute(
                "SELECT * FROM results WHERE campaign_id = ?", (campaign_id,)
            )
        ]

    objectives = json.loads(campaign["objectives"])
    front, ineligible = pareto_front(results, objectives)
    now = time.time()

    return {
        "campaign": _campaign_row(campaign),
        "task_counts": counts,
        "in_flight": in_flight,
        "n_results": len(results),
        "front": [
            {"task_id": r["task_id"], "objective_profile": r["objective_profile"]}
            for r in front
        ],
        "n_ineligible": len(ineligible),
        "window": {
            "dispatch_remaining": (
                round(campaign["dispatch_until"] - now, 1)
                if campaign["dispatch_until"] else None
            ),
            "hard_stop_remaining": (
                round(campaign["hard_stop_at"] - now, 1)
                if campaign["hard_stop_at"] else None
            ),
            "phase": _window_phase(campaign, now),
        },
    }


def _window_phase(campaign: sqlite3.Row, now: float) -> str:
    """Which deadline window the campaign is in, by the clock not the state.

    Reported separately from ``campaign.state`` so a conductor that died and
    missed its own deadline can see that it should be draining even though
    nothing has moved the state yet.
    """

    if campaign["state"] in ("finalized", "cancelled"):
        return campaign["state"]
    if campaign["dispatch_until"] is None:
        return "dispatching"
    if now < campaign["dispatch_until"]:
        return "dispatching"
    if campaign["hard_stop_at"] is not None and now < campaign["hard_stop_at"]:
        return "draining"
    return "past_hard_stop"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CAMPAIGN_STORE_PORT", "8087")),
    )
