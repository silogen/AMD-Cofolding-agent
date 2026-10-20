"""Campaign tools: the conductor's view of a supervisor fleet.

These are what an interactive conductor calls to open a campaign, see what the
fleet is doing, and steer it. Everything reads through ``campaign-store``, so
the answers are the same ones the supervisors are acting on rather than a
second opinion held in the agent's context.

Every function returns a plain dict and turns a failure into
``{"error": ...}`` rather than raising, matching the rest of the tool surface:
a conductor that asks about a campaign that does not exist should be told so
and carry on, not have its turn aborted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _store():
    from reinvent4_agent.campaign_client import CampaignStoreClient

    return CampaignStoreClient()


def _guard(what: str, call) -> Any:
    from reinvent4_agent.campaign_client import LeaseLost, StoreUnavailable

    try:
        return call()
    except (StoreUnavailable, LeaseLost) as exc:
        logger.error("%s: %s", what, exc)
        return {"error": f"{what}: {exc}"}


def start_campaign(
    goal: str,
    base_ttp_id: str,
    objectives: list[str],
    concurrency: int = 1,
    hours: Optional[float] = None,
    grace_hours: float = 0.0,
) -> dict[str, Any]:
    """Open a campaign: a goal, a pool size, and a wall-clock budget.

    Opening a campaign does not dispatch anything or start any supervisors --
    the conductor loop does that, so a campaign can be set up and inspected
    before it begins spending GPU.

    Parameters
    ----------
    goal:
        What the campaign is for, in a sentence. Carried in the ledger so a
        conductor restarting mid-campaign knows what it was doing.
    base_ttp_id:
        The TTP every task starts from. Each task runs a variant of it with
        its own objective weights.
    objectives:
        The objective names the Pareto front is computed over. Fixed at
        creation, so the front's dimensions cannot drift mid-campaign.
    concurrency:
        How many supervisors to hold in flight.
    hours:
        Length of the dispatch window. Omit for no deadline.
    grace_hours:
        Drain window after the dispatch window closes, during which
        in-flight tasks finish but nothing new is enqueued.

    Returns
    -------
    dict
        The campaign, including its id and the two deadlines.
    """

    from reinvent4_agent.conductor import start_campaign as _start

    return _guard(
        "Could not open the campaign",
        lambda: _start(
            _store(), goal, base_ttp_id, objectives,
            concurrency=concurrency, hours=hours, grace_hours=grace_hours,
        ),
    )


def get_campaign_state(campaign_id: str) -> dict[str, Any]:
    """Everything about a campaign in one call.

    Task counts by state, the supervisors currently holding tasks with their
    weight vectors and lease remaining, the Pareto front, and how long is left
    in the dispatch and drain windows.

    ``window.phase`` follows the clock rather than the stored state, so a
    campaign whose conductor died and missed its own deadline still reports
    that it should be draining.
    """

    return _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().status(campaign_id),
    )


def list_campaigns() -> dict[str, Any]:
    """Every campaign in the ledger, newest first."""

    result = _guard("Could not list campaigns", lambda: _store().campaigns())
    if isinstance(result, dict):
        return result
    return {"campaigns": result, "n_campaigns": len(result)}


def enqueue_task(
    campaign_id: str,
    weight_overrides: dict[str, float],
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
    fidelity_tier: str = "exploration",
    max_steps: Optional[int] = None,
    deadline: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict[str, Any]:
    """Queue one weight vector for a supervisor to run.

    For steering by hand -- following a hunch about a trade-off the automatic
    proposer has not reached. The conductor loop enqueues its own tasks and
    will not re-propose a vector that is already in the ledger, so adding one
    here does not get duplicated later.

    An override naming an objective the base TTP does not have will fail the
    task when a supervisor picks it up, rather than running silently with the
    wrong weights.

    Parameters
    ----------
    campaign_id:
        Campaign to add the task to. Must still be in its dispatch window.
    weight_overrides:
        Objective name to weight. Names must match the base TTP's objectives.
    transform_overrides:
        Objective name to a transform config, for reshaping how a raw value
        maps to [0, 1].
    fidelity_tier:
        ``exploration`` reads and writes the scoring cache; ``validation``
        bypasses the read so a reported result is never a cached sample.
    max_steps:
        RL steps for this task. Budget it against the measured step latency.
    deadline:
        ISO timestamp by which the task must have stopped. The supervisor
        turns this into a checkpointing stop.
    intent:
        What this run is for, in prose, handed to the supervisor's agent
        verbatim. The channel for what the fields above cannot express --
        most usefully which chemistry to stay within, since holding a
        chemotype is a choice of generative prior and the agent can look the
        reference compound up and pick a model itself. Say what the run is
        for, not which tool to call.

        A task with an intent will not fall back to a default de novo run if
        the agent fails to start one: it is handed back to be retried, rather
        than quietly measuring a different experiment.
    """

    spec: dict[str, Any] = {
        "weight_overrides": weight_overrides,
        "fidelity_tier": fidelity_tier,
    }
    if transform_overrides:
        spec["transform_overrides"] = transform_overrides
    if max_steps:
        spec["max_steps"] = max_steps
    if deadline:
        spec["deadline"] = deadline
    if intent:
        spec["intent"] = intent

    return _guard(
        f"Could not enqueue a task on campaign {campaign_id}",
        lambda: _store().enqueue_task(campaign_id, **spec),
    )


def list_supervisors(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """The pool: how many supervisors exist, and how many could take work now.

    ``idle`` is the number that would answer a dispatch immediately. That is
    not the same as ``concurrency``, which is only the ceiling this campaign
    asked for, nor the Deployment's replica count, which counts pods that are
    still starting or already shutting down. A supervisor appears here because
    it reported in, so one that crashed drops out after its presence expires.

    Dead presences and expired leases are both swept before answering, so this
    is who is alive rather than who was last seen.

    Parameters
    ----------
    campaign_id:
        Narrow to the supervisors eligible for one campaign, and include that
        campaign's in-flight tasks. Supervisors not pinned to any campaign
        count towards every campaign's pool. Omit for the whole fleet.
    """

    pool = _guard("Could not read the supervisor pool",
                  lambda: _store().supervisors(campaign_id))
    if "error" in pool:
        return pool

    if campaign_id is None:
        return pool

    status = _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().status(campaign_id),
    )
    if "error" in status:
        return status
    return {
        **pool,
        "in_flight": status["in_flight"],
        "n_in_flight": len(status["in_flight"]),
        "concurrency": status["campaign"]["concurrency"],
    }


def get_supervisor_result(campaign_id: str, task_id: str) -> dict[str, Any]:
    """What one finished task achieved: its objective profile and molecules.

    Scores are narrowed to the campaign's declared objectives. A scorer also
    emits per-molecule diagnostics, which are in the job's results CSV via
    ``read_results`` rather than here.
    """

    results = _guard(
        f"Could not read results for campaign {campaign_id}",
        lambda: _store().results(campaign_id),
    )
    if isinstance(results, dict):
        return results

    for result in results:
        if result["task_id"] == task_id:
            return result
    return {"error": f"Campaign {campaign_id} has no result for task {task_id}"}


#: Files worth naming in a job directory, in the order a person reads them.
#: Everything else there (``tb_logs_*``, ``_config.json``, checkpoint shards)
#: is machinery rather than something to open.
JOB_FILES_OF_INTEREST = (
    "results_*.csv",
    "reinvent.log",
    "config.toml",
    "job_metadata.json",
    "stage.chkpt",
    "seed.smi",
)


def _job_files(job_dir: Any) -> list[dict[str, Any]]:
    """The notable files in one job directory, with sizes, newest pattern first."""

    files = []
    seen: set[str] = set()
    for pattern in JOB_FILES_OF_INTEREST:
        for path in sorted(job_dir.glob(pattern)):
            if path.name in seen or not path.is_file():
                continue
            seen.add(path.name)
            files.append({"name": path.name, "bytes": path.stat().st_size})
    return files


def get_campaign_files(campaign_id: str, limit: int = 20) -> dict[str, Any]:
    """Where a campaign's output actually is on disk, task by task.

    The ledger records what each task achieved; this says which directory it
    wrote while achieving it, so a result can be opened rather than only read
    about. Paths are as the container sees them -- the same volume the agent,
    the runner and the scorers share.

    A task with no directory either never got as far as starting a job or ran
    before this volume, and says which.

    Tasks come back newest first and carry no scores, so position here says
    nothing about which run was any good -- ``get_pareto_front`` ranks them,
    and ``get_top_candidates`` ranks the molecules within one job.

    Parameters
    ----------
    campaign_id:
        Campaign to locate.
    limit:
        How many tasks to report, newest first. ``n_tasks`` is the true total.
    """

    from pathlib import Path

    from reinvent4_mcp.config import get_config

    tasks = _guard(
        f"Could not read tasks for campaign {campaign_id}",
        lambda: _store().tasks(campaign_id),
    )
    if isinstance(tasks, dict):
        return tasks

    results = _guard(
        f"Could not read results for campaign {campaign_id}",
        lambda: _store().results(campaign_id),
    )
    job_of = (
        {}
        if isinstance(results, dict)
        else {r["task_id"]: r.get("job_id") for r in results}
    )

    jobs_root = Path(get_config().jobs.working_dir)
    newest_first = sorted(
        tasks, key=lambda t: t.get("created_at") or "", reverse=True
    )

    entries = []
    for task in newest_first[:limit]:
        job_id = job_of.get(task["task_id"])
        entry: dict[str, Any] = {
            "task_id": task["task_id"],
            "state": task["state"],
            "job_id": job_id,
            "created_at": task.get("created_at"),
            "finished_at": task.get("finished_at"),
        }
        if not job_id:
            entry["directory"] = None
            entry["note"] = (
                "no job id in the ledger: the task did not get as far as "
                f"starting one ({task['state']})"
            )
        else:
            job_dir = jobs_root / f"job_{job_id}"
            entry["directory"] = str(job_dir)
            if job_dir.is_dir():
                entry["files"] = _job_files(job_dir)
            else:
                entry["note"] = "the ledger has this job but the directory is gone"
        entries.append(entry)

    return {
        "campaign_id": campaign_id,
        "jobs_root": str(jobs_root),
        "shared_volume": str(
            Path(os.environ.get("IP_SHARED_DIR", "/shared"))
        ),
        "tasks": entries,
        "n_tasks": len(tasks),
        "n_listed": len(entries),
        "hint": (
            "Cofolded structures are on the shared volume, not in the job "
            "directory -- use list_shared_assets(prefix=...) for those. Row "
            "data is in results_*.csv, which read_results(job_id) parses."
        ),
    }


def get_pareto_front(campaign_id: str) -> dict[str, Any]:
    """The non-dominated results: the trade-offs nothing else beats.

    Also returns ``ineligible`` -- results left off the front because they did
    not report every objective. That usually means a scorer failed, and is
    worth looking at rather than ignoring: those runs are not bad trade-offs,
    they are incomplete measurements.
    """

    return _guard(
        f"Could not read the front for campaign {campaign_id}",
        lambda: _store().pareto(campaign_id),
    )


def set_concurrency(campaign_id: str, concurrency: int) -> dict[str, Any]:
    """Change how many supervisors the campaign holds in flight.

    Recorded in the ledger rather than only in the conductor, so the number
    survives a restart. The conductor loop scales the supervisor Deployment to
    match on its next pass; this call on its own does not move pods.
    """

    return _guard(
        f"Could not set concurrency on campaign {campaign_id}",
        lambda: _store().set_concurrency(campaign_id, concurrency),
    )


def cancel_supervisor(task_id: str) -> dict[str, Any]:
    """Take a task back from the supervisor running it.

    The holder finds out on its next heartbeat and abandons the work without
    posting a result. Its RL job is not stopped by this call -- use
    ``stop_optimization`` on the job id if the GPU is what you need back.
    """

    return _guard(
        f"Could not cancel task {task_id}",
        lambda: _store().cancel_task(task_id),
    )


def get_pool_capacity(campaign_id: Optional[str] = None) -> dict[str, Any]:
    """Whether there is room for more work, and which limit is the binding one.

    Two separate things have to be true to dispatch: a supervisor must be free
    to claim, and Boltz must have queue room to score what it starts. Either
    one alone is misleading -- an idle supervisor handed a task when the
    scorer is saturated just moves the wait, and a quiet scorer is no use with
    nothing alive to dispatch to.

    ``queue_pressure`` is queued batches per healthy replica. ``None`` means
    the gateway did not answer, which is not the same as idle.

    Parameters
    ----------
    campaign_id:
        Count only the supervisors eligible for this campaign. Omit for the
        whole fleet.
    """

    from reinvent4_agent.conductor import queue_pressure

    pool = _guard("Could not read the supervisor pool",
                  lambda: _store().supervisors(campaign_id))
    idle = None if "error" in pool else pool["idle"]

    pressure = queue_pressure()
    scorer = {
        "queue_pressure": None if pressure is None else round(pressure, 2),
        "verdict": "unknown" if pressure is None
                   else ("room" if pressure <= 2.0 else "saturated"),
        "detail": "the boltz gateway did not answer; treat capacity as unknown "
                  "rather than free" if pressure is None
                  else f"{pressure:.1f} batches in hand per healthy Boltz "
                       "replica, queued and in flight together",
    }

    if idle is None:
        verdict, limit = "unknown", "supervisor pool unreadable"
    elif idle == 0:
        verdict, limit = "no", "every supervisor is busy"
    elif scorer["verdict"] == "saturated":
        verdict, limit = "no", "boltz is saturated"
    elif scorer["verdict"] == "unknown":
        verdict, limit = "unknown", "boltz queue depth unreadable"
    else:
        verdict, limit = "yes", None

    return {
        "can_dispatch": verdict,
        "binding_limit": limit,
        "supervisors": {
            "idle": idle,
            "busy": None if "error" in pool else pool["busy"],
            "alive": None if "error" in pool else pool["n_alive"],
        },
        "scorer": scorer,
    }


def close_campaign(campaign_id: str) -> dict[str, Any]:
    """Stop dispatching, let in-flight tasks finish.

    The drain window, brought forward by hand. Queued tasks stay claimable, so
    work already decided on still runs.
    """

    return _guard(
        f"Could not close campaign {campaign_id}",
        lambda: _store().close_campaign(campaign_id),
    )


def finalize_campaign(campaign_id: str) -> dict[str, Any]:
    """Close the books: abandon anything outstanding and stop the campaign.

    Tasks still queued or claimed are marked abandoned rather than left to
    look like work someone will do.
    """

    return _guard(
        f"Could not finalize campaign {campaign_id}",
        lambda: _store().finalize_campaign(campaign_id),
    )
