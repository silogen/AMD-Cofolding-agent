"""Conductor: keep N supervisors busy against one goal until the clock runs out.

The user gives a goal, a concurrency `N` and a wall-clock budget `T`. This runs
the loop that turns that into work: it keeps `N` tasks in flight, reviews the
Pareto front each time one finishes to decide what to try next, and respects
three windows:

* **dispatch** `[0, T)` -- new tasks may be enqueued
* **drain** `[T, T + grace)` -- the queue is closed, in-flight tasks finish
* **hard stop** at `T + grace` -- the ledger is closed and the pool scaled to 0

Nothing here holds campaign state. Every decision is made from a fresh read of
``campaign-store``, so a conductor that is restarted mid-campaign picks up where
the ledger says it is rather than where its memory said it was. That is also
why the loop is written as a single idempotent ``tick()``: running it twice in
a row changes nothing.

**The soft stop is the task deadline, not a signal.** Every task is enqueued
with ``deadline = min(now + task_budget, hard_stop_at)``, and the supervisor
already stops its RL job when that deadline arrives -- which asks REINVENT4 to
checkpoint, so the GPU comes back and the work is resumable. By the time the
hard stop arrives, every in-flight task has therefore already wound itself up;
the conductor does not have to reach across into the runner to stop jobs it
does not own.

Env:
    CAMPAIGN_STORE_URL        the ledger
    BOLTZ_GATEWAY_STATS_URL   queue depth, for admission control
    SUPERVISOR_DEPLOYMENT     the Deployment to scale (default dd-agentic-supervisor)
    POD_NAMESPACE             namespace for the scale call
    CONDUCTOR_TICK_SECONDS    how often the loop reconsiders
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from reinvent4_agent.campaign_client import CampaignStoreClient, StoreUnavailable

logger = logging.getLogger(__name__)

TICK_SECONDS = float(os.getenv("CONDUCTOR_TICK_SECONDS", "30"))
SUPERVISOR_DEPLOYMENT = os.getenv("SUPERVISOR_DEPLOYMENT", "dd-agentic-supervisor")
GATEWAY_STATS_URL = os.getenv(
    "BOLTZ_GATEWAY_STATS_URL", "http://dd-agentic-boltz-gateway:8091/stats"
)

_SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


# ---------------------------------------------------------------------------
# Weight vectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightVector:
    """One point in the weight space, hashable so it can be de-duplicated."""

    weights: tuple[tuple[str, float], ...]

    @classmethod
    def of(cls, weights: dict[str, float]) -> "WeightVector":
        return cls(tuple(sorted((k, round(float(v), 3)) for k, v in weights.items())))

    def as_dict(self) -> dict[str, float]:
        return dict(self.weights)


class ParetoProposer:
    """Decides which weight vector to try next, from the front so far.

    Two phases, because they answer different questions:

    * **Seeding.** With no results, there is nothing to learn from, so it emits
      one vector per objective with that objective emphasised, plus a balanced
      one. That bounds the front's extremes early, which is what makes later
      refinement meaningful -- refining around a single lucky point first would
      just explore one corner.
    * **Refinement.** With a front, it perturbs a front member's weights. The
      front is by definition the set of trade-offs nothing else beats, so its
      neighbourhood is where an improvement can still be found.

    Deterministic: the same ledger produces the same next proposal, so a
    campaign can be replayed and a restarted conductor does not change course.
    """

    def __init__(
        self,
        objectives: list[str],
        emphasis: float = 2.0,
        step: float = 1.5,
    ) -> None:
        if not objectives:
            raise ValueError("a campaign needs at least one objective to sweep")
        self.objectives = list(objectives)
        self.emphasis = emphasis
        self.step = step

    def _seeds(self) -> list[WeightVector]:
        balanced = {name: 1.0 for name in self.objectives}
        seeds = [WeightVector.of(balanced)]
        for name in self.objectives:
            emphasised = dict(balanced)
            emphasised[name] = self.emphasis
            seeds.append(WeightVector.of(emphasised))
        return seeds

    def _refinements(self, front: list[dict[str, Any]]) -> list[WeightVector]:
        """Neighbours of each front member, one objective moved at a time.

        Ordered by the front member's own position so the strongest trade-offs
        are refined first, and so the order does not depend on dict iteration.
        """

        proposals: list[WeightVector] = []
        for member in sorted(front, key=lambda m: -sum(m["weights"].values())):
            for name in self.objectives:
                for factor in (self.step, 1 / self.step):
                    moved = dict(member["weights"])
                    moved[name] = max(0.05, moved.get(name, 1.0) * factor)
                    proposals.append(WeightVector.of(moved))
        return proposals

    def propose(
        self, front: list[dict[str, Any]], tried: set[WeightVector], count: int
    ) -> list[WeightVector]:
        """Up to `count` vectors that have not been tried yet.

        May return fewer, including none: a campaign whose neighbourhood is
        exhausted should stop dispatching rather than re-run points it has
        already paid for.
        """

        candidates = self._seeds()
        if front:
            candidates += self._refinements(front)

        chosen: list[WeightVector] = []
        seen = set(tried)
        for candidate in candidates:
            if len(chosen) >= count:
                break
            if candidate in seen:
                continue
            seen.add(candidate)
            chosen.append(candidate)
        return chosen


def tried_vectors(tasks: list[dict[str, Any]]) -> set[WeightVector]:
    """Every weight vector the campaign has already enqueued.

    Read from the ledger rather than remembered, so a restarted conductor does
    not re-dispatch the whole sweep.
    """

    tried: set[WeightVector] = set()
    for task in tasks:
        overrides = (task.get("spec") or {}).get("weight_overrides") or {}
        if overrides:
            tried.add(WeightVector.of(overrides))
    return tried


def front_weights(
    front: list[dict[str, Any]], tasks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair each front member with the weight vector that produced it.

    The store's front knows objective *scores*; refinement needs the *weights*
    that got there, which only the task spec has.
    """

    weights_by_task = {
        task["task_id"]: (task.get("spec") or {}).get("weight_overrides") or {}
        for task in tasks
    }
    paired = []
    for member in front:
        weights = weights_by_task.get(member["task_id"])
        if weights:
            paired.append({**member, "weights": weights})
    return paired


# ---------------------------------------------------------------------------
# Pool scaling
# ---------------------------------------------------------------------------

class PoolScaler:
    """Scales the supervisor Deployment through the Kubernetes API.

    Talks to the API directly with the pod's ServiceAccount token rather than
    shelling out to ``kubectl`` (not in the image) or pulling in the Kubernetes
    client (a dependency for two HTTP calls).

    Outside a cluster there is no token, and every call is a logged no-op: a
    conductor run on a laptop should exercise the loop, not fail on the one
    thing that needs a cluster.
    """

    def __init__(
        self,
        deployment: str = SUPERVISOR_DEPLOYMENT,
        namespace: Optional[str] = None,
        service_account: Path = _SERVICE_ACCOUNT,
    ) -> None:
        self.deployment = deployment
        self._sa = service_account
        self.namespace = namespace or os.getenv("POD_NAMESPACE") or self._namespace()

    def _namespace(self) -> str:
        try:
            return (self._sa / "namespace").read_text().strip()
        except OSError:
            return "default"

    @property
    def in_cluster(self) -> bool:
        return (self._sa / "token").is_file()

    def _request(self, method: str, body: Optional[dict[str, Any]] = None) -> Any:
        token = (self._sa / "token").read_text().strip()
        url = (
            f"https://kubernetes.default.svc/apis/apps/v1/namespaces/"
            f"{self.namespace}/deployments/{self.deployment}/scale"
        )
        headers = {"Authorization": f"Bearer {token}"}
        if method == "PATCH":
            # Merge patch: the smallest thing that expresses "only replicas".
            headers["Content-Type"] = "application/merge-patch+json"

        response = requests.request(
            method, url, json=body, headers=headers,
            verify=str(self._sa / "ca.crt"), timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def current(self) -> Optional[int]:
        """Replicas the Deployment is set to, or None when not in a cluster."""

        if not self.in_cluster:
            return None
        try:
            return self._request("GET")["spec"].get("replicas", 0)
        except (requests.RequestException, KeyError, OSError) as exc:
            logger.warning("could not read the supervisor replica count: %s", exc)
            return None

    def scale(self, replicas: int) -> bool:
        """Set the replica count. False when it could not be done."""

        if not self.in_cluster:
            logger.info(
                "not in a cluster; would scale %s to %d", self.deployment, replicas
            )
            return False

        if self.current() == replicas:
            return True

        try:
            self._request("PATCH", {"spec": {"replicas": replicas}})
        except (requests.RequestException, OSError) as exc:
            # Most likely the ServiceAccount has no deployments/scale grant.
            logger.error("could not scale %s to %d: %s", self.deployment, replicas, exc)
            return False

        logger.info("scaled %s to %d", self.deployment, replicas)
        return True


# ---------------------------------------------------------------------------
# Admission control
# ---------------------------------------------------------------------------

def queue_pressure(stats_url: str = GATEWAY_STATS_URL) -> Optional[float]:
    """Boltz batches in the scorer's hands per healthy replica, or None.

    The binding constraint is Boltz, not the agents, so this is what says
    whether another supervisor would actually get work done or just deepen a
    queue. Unknown is not zero: a gateway that cannot be reached should not be
    read as idle.

    Counts in-flight batches as well as queued ones, because the gateway caps
    its own upstream calls at the replica count -- so work beyond capacity
    waits for a semaphore slot inside the gateway rather than showing up as
    proxy queueing. Measured on queued alone this reads ~0 even with every
    replica saturated, which is no signal at all.

    One batch per replica is a scorer that is busy and keeping up; the
    threshold that holds dispatch back sits above that.
    """

    try:
        response = requests.get(stats_url, timeout=10)
        response.raise_for_status()
        stats = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("no gateway stats from %s: %s", stats_url, exc)
        return None

    healthy = (stats.get("replicas") or {}).get("healthy")
    queue = stats.get("queue") or {}
    queued = queue.get("proxy_queued_requests")
    in_flight = queue.get("proxy_in_flight_requests")
    if not healthy or queued is None or in_flight is None:
        logger.warning(
            "gateway stats from %s lack replicas.healthy or queue depth: %s",
            stats_url, sorted(stats),
        )
        return None
    return (float(queued) + float(in_flight)) / float(healthy)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class TickReport:
    """What one pass of the loop did, for logs and the status view."""

    phase: str
    in_flight: int
    queued: int
    dispatched: int = 0
    held_back: Optional[str] = None
    closed: bool = False
    finalized: bool = False


class CampaignConductor:
    """Holds N tasks in flight against one campaign until the clock runs out."""

    def __init__(
        self,
        store: CampaignStoreClient,
        campaign_id: str,
        *,
        scaler: Optional[PoolScaler] = None,
        proposer: Optional[ParetoProposer] = None,
        task_budget_seconds: Optional[float] = None,
        max_steps: Optional[int] = None,
        max_queue_pressure: float = 2.0,
        tick_seconds: float = TICK_SECONDS,
        stats_url: str = GATEWAY_STATS_URL,
    ) -> None:
        self.store = store
        self.campaign_id = campaign_id
        self.scaler = scaler
        self.proposer = proposer
        self.task_budget_seconds = task_budget_seconds
        self.max_steps = max_steps
        self.max_queue_pressure = max_queue_pressure
        self.tick_seconds = tick_seconds
        self.stats_url = stats_url

    # -- deadlines ---------------------------------------------------------

    def _task_deadline(self, campaign: dict[str, Any]) -> Optional[str]:
        """When the task being enqueued now must have stopped.

        The earlier of its own budget and the campaign's hard stop, so no
        supervisor starts a run that would still be going when the campaign is
        over. The supervisor turns this into a checkpointing stop, which is
        what makes the hard stop a formality rather than a kill.
        """

        candidates = []
        if self.task_budget_seconds:
            candidates.append(time.time() + self.task_budget_seconds)
        if campaign.get("hard_stop_at"):
            candidates.append(float(campaign["hard_stop_at"]))
        if not candidates:
            return None

        when = datetime.fromtimestamp(min(candidates), tz=timezone.utc)
        return when.isoformat()

    # -- one pass ----------------------------------------------------------

    def tick(self) -> TickReport:
        """Reconcile the campaign once. Idempotent.

        Every decision comes from this read of the ledger, so running it twice
        changes nothing and a restarted conductor needs no recovery path.
        """

        status = self.store.status(self.campaign_id)
        campaign = status["campaign"]
        counts = status["task_counts"]
        phase = status["window"]["phase"]
        in_flight = counts.get("claimed", 0)
        queued = counts.get("queued", 0)

        report = TickReport(phase=phase, in_flight=in_flight, queued=queued)

        if campaign["state"] == "finalized":
            return report

        if phase == "past_hard_stop":
            self._finalize()
            report.finalized = True
            return report

        if phase == "draining":
            if campaign["state"] == "dispatching":
                logger.info(
                    "dispatch window over; closing the queue on campaign %s",
                    self.campaign_id,
                )
                self.store.close_campaign(self.campaign_id)
                report.closed = True
            return report

        # Dispatch window.
        target = campaign.get("concurrency") or 1
        if self.scaler:
            self.scaler.scale(target)

        shortfall = target - (in_flight + queued)
        if shortfall <= 0:
            return report

        pressure = queue_pressure(self.stats_url)
        if pressure is not None and pressure > self.max_queue_pressure:
            # Another task would only deepen a queue that is already the
            # bottleneck; the supervisors are not what is limiting throughput.
            report.held_back = (
                f"boltz queue at {pressure:.1f} batches per replica, "
                f"over the {self.max_queue_pressure:.1f} limit"
            )
            logger.info("holding dispatch: %s", report.held_back)
            return report

        report.dispatched = self._dispatch(campaign, status, shortfall)
        return report

    def _dispatch(
        self, campaign: dict[str, Any], status: dict[str, Any], count: int
    ) -> int:
        """Enqueue up to `count` untried weight vectors."""

        objectives = campaign["objectives"]
        proposer = self.proposer or ParetoProposer(objectives)
        tasks = self.store.tasks(self.campaign_id)

        front = front_weights(self.store.pareto(self.campaign_id)["front"], tasks)
        proposals = proposer.propose(front, tried_vectors(tasks), count)
        if not proposals:
            logger.info(
                "campaign %s has no untried weight vectors left to dispatch",
                self.campaign_id,
            )
            return 0

        deadline = self._task_deadline(campaign)
        dispatched = 0
        for vector in proposals:
            spec: dict[str, Any] = {"weight_overrides": vector.as_dict()}
            if deadline:
                spec["deadline"] = deadline
            if self.max_steps:
                spec["max_steps"] = self.max_steps
            try:
                task = self.store.enqueue_task(self.campaign_id, **spec)
            except StoreUnavailable as exc:
                logger.warning("could not enqueue %s: %s", vector.as_dict(), exc)
                break
            logger.info(
                "dispatched task %s with weights %s", task["task_id"], vector.as_dict()
            )
            dispatched += 1

        return dispatched

    def _finalize(self) -> None:
        """Close the ledger, stop the orphaned jobs, and send the pool home.

        In the normal case every in-flight task has already stopped itself at
        its own deadline, which is earlier than this. What is left here is the
        tasks that overran or whose supervisor died -- and abandoning a task
        does not stop its RL job, because the job lives in the runner and will
        hold a GPU until something tells it to stop. The first live wind-down
        left two jobs running for exactly this reason.

        Order matters: the ledger is closed first, so a supervisor cannot post
        a result against a campaign that is over, and only then are the jobs
        stopped. A result worth keeping comes from the task deadline, which
        fires before this and stops the job *with* a checkpoint.
        """

        logger.info("hard stop reached; finalizing campaign %s", self.campaign_id)

        campaign = self.store.campaign(self.campaign_id)
        # Read the open tasks before the store abandons them.
        orphans = [
            task for task in self.store.tasks(self.campaign_id)
            if task["state"] in ("queued", "claimed")
        ]

        self.store.finalize_campaign(self.campaign_id)
        self._stop_jobs_for(campaign.get("base_ttp_id") or "", orphans)

        if self.scaler:
            self.scaler.scale(0)

    def _stop_jobs_for(
        self, base_ttp_id: str, tasks: list[dict[str, Any]]
    ) -> list[str]:
        """Stop the RL jobs of tasks that will never report. Returns their ids.

        A task's TTP id is derived from the task id, so this works even for a
        task whose supervisor is gone and cannot clean up after itself.
        """

        from reinvent4_agent.supervisor import job_for_ttp, variant_ttp_id
        from reinvent4_mcp.tools.optimization import stop_optimization

        if not base_ttp_id:
            return []

        stopped: list[str] = []
        for task in tasks:
            try:
                job_id = job_for_ttp(variant_ttp_id(base_ttp_id, task["task_id"]))
            except Exception:
                logger.warning(
                    "could not look up the job for task %s", task["task_id"],
                    exc_info=True,
                )
                continue

            if not job_id:
                continue

            result = stop_optimization(job_id)
            stopped.append(job_id)
            logger.info(
                "stopped orphaned job %s from task %s: %s",
                job_id, task["task_id"],
                result.get("message") or result.get("error") or "stopped",
            )

        return stopped

    # -- the loop ----------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> TickReport:
        """Tick until the campaign is finalized. Returns the last report."""

        ticks = 0
        report = TickReport(phase="unknown", in_flight=0, queued=0)

        while max_ticks is None or ticks < max_ticks:
            try:
                report = self.tick()
            except StoreUnavailable as exc:
                # The store may be rolling. A campaign is hours long; losing
                # the ledger for a minute is not losing the campaign.
                logger.warning("ledger unavailable: %s", exc)
            else:
                if report.finalized:
                    return report

            ticks += 1
            time.sleep(self.tick_seconds)

        return report


# ---------------------------------------------------------------------------
# Starting a campaign
# ---------------------------------------------------------------------------

def start_campaign(
    store: CampaignStoreClient,
    goal: str,
    base_ttp_id: str,
    objectives: list[str],
    concurrency: int = 1,
    hours: Optional[float] = None,
    grace_hours: float = 0.0,
) -> dict[str, Any]:
    """Open a campaign and return it. Dispatch is the conductor loop's job."""

    campaign = store.create_campaign(
        goal=goal,
        base_ttp_id=base_ttp_id,
        objectives=objectives,
        concurrency=concurrency,
        dispatch_seconds=hours * 3600 if hours else None,
        grace_seconds=grace_hours * 3600,
    )
    logger.info(
        "campaign %s open: %d supervisors over %s",
        campaign["campaign_id"], concurrency,
        f"{hours}h" if hours else "no deadline",
    )
    return campaign


def live_campaign_id(store: CampaignStoreClient) -> str:
    """The campaign still running, when there is exactly one.

    The loop has to know which campaign it is reconciling, but naming it only
    carries information when several are live. Refusing to guess between two
    is deliberate: conducting the wrong one would dispatch real GPU work
    against the wrong goal.
    """

    live = [
        c for c in store.campaigns()
        if c["state"] in ("dispatching", "draining")
    ]
    if not live:
        raise SystemExit(
            "no live campaign to conduct. Open one with --start, or name a "
            "campaign id explicitly."
        )
    if len(live) > 1:
        listed = "\n  ".join(
            f"{c['campaign_id']}  {c['state']}  {c['goal'][:60]}" for c in live
        )
        raise SystemExit(
            f"{len(live)} campaigns are live, so name the one to conduct:\n  {listed}"
        )
    return live[0]["campaign_id"]


def objectives_of(base_ttp_id: str) -> list[str]:
    """Every objective name in a TTP, in the order it declares them.

    The default for a new campaign, because naming them by hand is the easiest
    way to get a campaign wrong: an objective the TTP does not have fails every
    task that references it, and the mistake only surfaces once a supervisor
    picks one up -- minutes later, and as a task failure rather than a typo.
    """

    from reinvent4_mcp.tools.ttp import get_ttp

    ttp = get_ttp(base_ttp_id)
    if "error" in ttp:
        raise SystemExit(f"base TTP '{base_ttp_id}': {ttp['error']}")

    names = [
        objective["name"]
        for objective in ((ttp.get("scoring") or {}).get("objectives") or [])
    ]
    names += [
        component["name"]
        for component in ((ttp.get("pre_scoring") or {}).get("components") or [])
    ]
    if not names:
        raise SystemExit(f"base TTP '{base_ttp_id}' declares no objectives")

    # A name can appear in both sections; keep the first and the order.
    return list(dict.fromkeys(names))


def format_status(
    status: dict[str, Any], pool: Optional[dict[str, Any]] = None
) -> str:
    """The ``/status`` view, as text for the REPL.

    A table rather than the raw JSON, because the thing an operator actually
    wants to know -- is it working, what is it trying, how long is left -- is
    three numbers buried in a large document.

    ``pool`` is the supervisor listing, which is about the fleet rather than
    the campaign and so is read separately. Shown because "two tasks in
    flight" means something different with one idle supervisor beside it than
    with none.
    """

    campaign = status["campaign"]
    window = status["window"]
    counts = status["task_counts"]

    tally = ", ".join(f"{state}={n}" for state, n in sorted(counts.items()))
    lines = [
        f"campaign {campaign['campaign_id']}  {campaign['state']} "
        f"(window: {window['phase']})",
        f"  goal: {campaign['goal']}",
        f"  objectives: {', '.join(campaign['objectives'])}",
        f"  tasks: {tally or 'none yet'}",
    ]

    remaining = window.get("dispatch_remaining")
    if remaining is not None:
        lines.append(f"  dispatch window: {_duration(remaining)}")
    remaining = window.get("hard_stop_remaining")
    if remaining is not None:
        lines.append(f"  hard stop: {_duration(remaining)}")

    if pool is not None:
        lines.append(
            f"  supervisors: {pool.get('idle', 0)} idle, "
            f"{pool.get('busy', 0)} busy of {pool.get('n_alive', 0)} alive "
            f"(concurrency {campaign['concurrency']})"
        )
        for supervisor in pool.get("supervisors") or []:
            age = supervisor.get("last_seen_seconds_ago")
            lines.append(
                f"    {supervisor['owner']}  {supervisor['state']}"
                + (f" on {supervisor['task_id']}" if supervisor.get("task_id") else "")
                + (f"  (seen {_duration(age)} ago)" if age is not None else "")
            )

    if status["in_flight"]:
        lines.append(f"  in flight ({len(status['in_flight'])}):")
        for task in status["in_flight"]:
            weights = (task["spec"] or {}).get("weight_overrides") or {}
            lines.append(
                f"    {task['task_id']} on {task['owner']} "
                f"(lease {_duration(task['lease_expires_in'])}) {weights}"
            )
    else:
        lines.append("  in flight: none")

    if status["front"]:
        lines.append(f"  pareto front ({len(status['front'])} of "
                     f"{status['n_results']} results):")
        for member in status["front"]:
            scores = ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(member["objective_profile"].items())
            )
            lines.append(f"    {member['task_id']}  {scores}")
    else:
        lines.append(f"  pareto front: none yet ({status['n_results']} results)")

    if status.get("n_ineligible"):
        lines.append(
            f"  {status['n_ineligible']} result(s) off the front for missing "
            "an objective -- check for a failed scorer"
        )

    return "\n".join(lines)


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0:
        return f"{_duration(-seconds)} ago"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: Optional[list[str]] = None) -> None:
    """Conduct a campaign: an existing one, the only live one, or a new one."""

    import argparse

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Drive a supervisor fleet campaign",
        epilog="With no campaign id and no --start, conducts the only live "
               "campaign; it refuses to guess when several are running.",
    )
    parser.add_argument(
        "campaign_id", nargs="?",
        help="campaign to conduct (default: the only live one)",
    )

    opening = parser.add_argument_group("opening a new campaign (--start)")
    opening.add_argument(
        "--start", action="store_true",
        help="open a campaign and conduct it, instead of attaching to one",
    )
    opening.add_argument("--goal", help="what the campaign is for")
    opening.add_argument("--base-ttp", help="TTP every task varies the weights of")
    opening.add_argument(
        "--objectives", nargs="*", default=None,
        help="objective names the Pareto front is over "
             "(default: every objective the base TTP declares)",
    )
    opening.add_argument(
        "--concurrency", type=int, default=1, help="supervisors to hold in flight",
    )
    opening.add_argument(
        "--hours", type=float, default=None,
        help="dispatch window; omit for no deadline",
    )
    opening.add_argument(
        "--grace-hours", type=float, default=0.0,
        help="drain window after the dispatch window closes",
    )

    parser.add_argument(
        "--task-budget-hours", type=float, default=None,
        help="wall-clock budget per task, capped by the campaign's hard stop",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="RL steps per task; budget it against the measured step latency",
    )
    parser.add_argument(
        "--tick-seconds", type=float, default=TICK_SECONDS,
        help="how often the loop reconsiders",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="do not touch the supervisor Deployment's replica count",
    )
    args = parser.parse_args(argv)

    opening_flags = (
        args.goal, args.base_ttp, args.objectives, args.hours,
    )
    if not args.start and (any(f is not None for f in opening_flags) or args.grace_hours):
        parser.error("--goal/--base-ttp/--objectives/--hours/--grace-hours "
                     "only apply with --start")

    store = CampaignStoreClient()

    if args.start:
        if args.campaign_id:
            parser.error(
                "--start opens a new campaign, so do not also name one to attach to"
            )
        if not args.goal or not args.base_ttp:
            parser.error("--start needs --goal and --base-ttp")

        objectives = args.objectives or objectives_of(args.base_ttp)
        campaign_id = start_campaign(
            store, args.goal, args.base_ttp, objectives,
            concurrency=args.concurrency, hours=args.hours,
            grace_hours=args.grace_hours,
        )["campaign_id"]
        print(f"campaign {campaign_id} open over {', '.join(objectives)}")
    else:
        campaign_id = args.campaign_id or live_campaign_id(store)

    conductor = CampaignConductor(
        store,
        campaign_id,
        scaler=None if args.no_scale else PoolScaler(),
        task_budget_seconds=(
            args.task_budget_hours * 3600 if args.task_budget_hours else None
        ),
        max_steps=args.max_steps,
        tick_seconds=args.tick_seconds,
    )
    conductor.run()
    pool = None
    with contextlib.suppress(StoreUnavailable):
        pool = store.supervisors(campaign_id)
    print(format_status(store.status(campaign_id), pool))


if __name__ == "__main__":
    main()
