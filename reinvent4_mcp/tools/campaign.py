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

    return _guard(
        f"Could not enqueue a task on campaign {campaign_id}",
        lambda: _store().enqueue_task(campaign_id, **spec),
    )


def list_supervisors(campaign_id: str) -> dict[str, Any]:
    """The supervisors holding tasks right now, and what each is trying.

    Expired leases are swept first, so this is who is actually alive rather
    than who last claimed something.
    """

    status = _guard(
        f"Could not read campaign {campaign_id}",
        lambda: _store().status(campaign_id),
    )
    if "error" in status:
        return status
    return {
        "supervisors": status["in_flight"],
        "n_in_flight": len(status["in_flight"]),
        "concurrency": status["campaign"]["concurrency"],
    }


def get_supervisor_result(campaign_id: str, task_id: str) -> dict[str, Any]:
    """What one finished task achieved: its objective profile and molecules."""

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


def get_pool_capacity() -> dict[str, Any]:
    """Whether the fleet has room for more work, from the scorer's queue.

    Boltz is the binding constraint, not the agent count, so capacity is a
    question about its queue depth. ``queue_pressure`` is queued batches per
    healthy replica; ``None`` means the gateway could not be reached, which is
    not the same as idle.
    """

    from reinvent4_agent.conductor import queue_pressure

    pressure = queue_pressure()
    if pressure is None:
        return {
            "queue_pressure": None,
            "verdict": "unknown",
            "detail": "the boltz gateway did not answer; treat capacity as unknown "
                      "rather than free",
        }

    return {
        "queue_pressure": round(pressure, 2),
        "verdict": "room" if pressure <= 2.0 else "saturated",
        "detail": f"{pressure:.1f} queued batches per healthy Boltz replica",
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
