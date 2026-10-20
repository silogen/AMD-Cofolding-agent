"""Headless supervisor: claim a task, run a campaign, post the result, repeat.

The interactive agent is a REPL reached by ``kubectl exec``; a fleet needs the
same Driver + Reviewer pair running unattended, N pods at a time, each taking
work from ``campaign-store`` and reporting back something a conductor can rank.
This module is that loop. It runs in the existing ``dd-agent`` image -- there is
no supervisor image, because a supervisor *is* the agent with a different
entrypoint.

Two things are deliberately not left to the language model:

* **The weight vector.** A task's whole purpose is to try specific objective
  weights, so the supervisor applies them itself -- it reads the base TTP,
  overrides the named weights and transforms, and persists the result as a
  variant TTP the Driver is told to run by id. Asking the Driver to "use weight
  1.4 on affinity" invites it to round, reinterpret or forget, and the campaign
  would then be comparing points it cannot name. The variant is kept rather
  than cleaned up: it is the record of what the task actually ran.

* **The objective profile.** What the task achieved is read out of the results
  CSV, not summarised by an agent.

The profile is taken from the **single best-scoring molecule**, not the best
value seen per objective. A per-objective maximum would mix molecules and
describe a compound that does not exist, which is worse than useless on a
Pareto front whose whole job is to show achievable trade-offs.

Env:
    CAMPAIGN_STORE_URL            the ledger (default the in-cluster Service)
    SUPERVISOR_ID                 identity in the ledger (default: hostname)
    SUPERVISOR_CAMPAIGN_ID        restrict claims to one campaign (default: any)
    SUPERVISOR_POLL_SECONDS       wait between claims when the queue is empty
    SUPERVISOR_LEASE_SECONDS      lease to ask for (default: the store's own)
    SUPERVISOR_HEARTBEAT_SECONDS  how often to renew it while working

Plus the REINVENT4_AGENT_* variables the CLI already reads, which configure the
backend, model and budgets identically here. One of them changes the shape of a
task rather than just its settings:

    REINVENT4_AGENT_MULTI_AGENT=false runs the Driver alone. That halves the
    token spend, since the Reviewer produces no molecules -- but note it also
    replaces the monitoring loop, because ``monitor_campaign`` *is* the
    Reviewer cycle (assess the run, hand the Driver a verdict). Without a
    Reviewer the supervisor simply watches the job's state until it stops or
    the task deadline fires.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import signal
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from reinvent4_agent.campaign_client import (
    STORE_URL,
    CampaignStoreClient,
    LeaseLost,
    StoreUnavailable,
    TaskOutcome,
    supervisor_id,
)

logger = logging.getLogger(__name__)

POLL_SECONDS = float(os.getenv("SUPERVISOR_POLL_SECONDS", "15"))
HEARTBEAT_SECONDS = float(os.getenv("SUPERVISOR_HEARTBEAT_SECONDS", "60"))

__all__ = [
    "CampaignStoreClient",
    "CoscientistTaskRunner",
    "LeaseLost",
    "StoreUnavailable",
    "Supervisor",
    "TaskOutcome",
    "apply_overrides",
    "build_goal",
    "default_run_substitutes_for",
    "job_for_ttp",
    "main",
    "objective_profile",
    "prepare_ttp_variant",
    "supervisor_id",
    "variant_ttp_id",
]


# ---------------------------------------------------------------------------
# Weight overrides
# ---------------------------------------------------------------------------

def _named_components(ttp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every weighted, named entry in a TTP, by name.

    Weights live in two places -- ``pre_scoring.components`` and
    ``scoring.objectives`` -- and a sweep may well want to lean on a
    drug-likeness descriptor, so both are addressable by the same name.
    """

    found: dict[str, dict[str, Any]] = {}
    pre_scoring = ttp.get("pre_scoring") or {}
    for entry in (pre_scoring.get("components") or []):
        found[entry["name"]] = entry
    for entry in ((ttp.get("scoring") or {}).get("objectives") or []):
        found[entry["name"]] = entry
    return found


def apply_overrides(
    ttp: dict[str, Any],
    weight_overrides: Optional[dict[str, float]] = None,
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Return `ttp` with the named weights and transforms replaced.

    An override naming something the TTP does not have is an error, not a
    no-op. Silently ignoring it would have the task report a profile for a
    weight vector it never ran, and the conductor would then draw conclusions
    from a point that was never actually tried.
    """

    updated = copy.deepcopy(ttp)
    addressable = _named_components(updated)

    requested = set(weight_overrides or {}) | set(transform_overrides or {})
    unknown = sorted(requested - set(addressable))
    if unknown:
        raise ValueError(
            f"TTP '{ttp.get('id')}' has no objective named {unknown}; "
            f"it has {sorted(addressable)}"
        )

    for name, weight in (weight_overrides or {}).items():
        addressable[name]["weight"] = weight
    for name, transform in (transform_overrides or {}).items():
        addressable[name]["transform"] = transform

    return updated


def variant_ttp_id(base_ttp_id: str, task_id: str) -> str:
    """The TTP id a task runs under.

    Derived rather than looked up, which is what lets the conductor find a
    task's TTP -- and through it the task's job -- for a task whose supervisor
    has already gone.
    """

    return f"{base_ttp_id}-{task_id}"


def prepare_ttp_variant(
    base_ttp_id: str,
    task_id: str,
    weight_overrides: Optional[dict[str, float]] = None,
    transform_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    """Persist the task's weight vector as its own TTP and return its id.

    Named after the task so a result can be traced back to exactly the TTP
    that produced it, and so N supervisors writing into the shared TTP store
    cannot collide.
    """

    from reinvent4_mcp.tools.ttp import create_ttp, get_ttp, update_ttp

    base = get_ttp(base_ttp_id)
    if "error" in base:
        raise ValueError(f"Base TTP '{base_ttp_id}': {base['error']}")

    variant = apply_overrides(base, weight_overrides, transform_overrides)
    variant["id"] = variant_ttp_id(base_ttp_id, task_id)
    variant["name"] = f"{base.get('name', base_ttp_id)} (task {task_id})"
    # create() assigns version 1; carrying the base's version across would
    # claim a lineage this variant does not have.
    variant.pop("version", None)

    created = create_ttp(variant)
    if "error" in created:
        # The variant is already there, which means this task is being
        # retried. Its content is a function of the task, so rewriting it
        # changes nothing -- but failing here would make a requeued task
        # impossible to run, and every retry would die before it started.
        rewritten = {k: v for k, v in variant.items() if k not in ("id", "version")}
        created = update_ttp(variant["id"], rewritten)
        if "error" in created:
            raise ValueError(f"Could not persist TTP variant for task {task_id}: "
                             f"{created['error']}")
        logger.info("task %s is a retry; reused TTP %s", task_id, variant["id"])

    logger.info(
        "task %s runs TTP %s with weights %s",
        task_id, created["id"], weight_overrides or "unchanged",
    )
    return created["id"]


# ---------------------------------------------------------------------------
# Objective profile
# ---------------------------------------------------------------------------

def job_for_ttp(ttp_id: str) -> Optional[str]:
    """The RL job running this task's TTP, observed from the job table.

    Deliberately *not* read out of the Driver's handoff. That handoff is free
    text the model may never produce -- a session that ends ``incomplete`` has
    none -- and the first live run failed a task with "the Driver finished
    without starting an optimization job" while its job was running happily.
    A job that exists must not be abandoned because the agent did not format
    its answer.

    The variant TTP id is unique to this task, so the job is identifiable
    without trusting anything the model said.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    mine = [
        job for job in get_job_manager().list_jobs()
        if job.get("ttp_id") == ttp_id
    ]
    if not mine:
        return None

    # Newest, in case a retry of this task left an older one behind.
    return max(mine, key=lambda job: job.get("created_at") or "")["job_id"]


def objective_profile(
    candidates: list[dict[str, Any]], objectives: Optional[list[str]] = None
) -> dict[str, float]:
    """The best molecule's score on each objective.

    Deliberately one molecule's scores rather than the best value seen per
    objective: the latter describes a compound nobody made.
    """

    scored = [c for c in candidates if isinstance(c.get("total_score"), (int, float))]
    best = max(scored, key=lambda c: c["total_score"], default=None)
    if best is None:
        return {}

    profile: dict[str, float] = {}
    for name, component in (best.get("components") or {}).items():
        if objectives is not None and name not in objectives:
            continue
        value = component.get("transformed")
        if isinstance(value, (int, float)):
            profile[name] = float(value)
    return profile


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

TaskRunner = Callable[[dict[str, Any]], Awaitable[TaskOutcome]]


class Supervisor:
    """Claim, run, report, repeat -- until told to stop.

    The campaign run itself is injected, so the loop's behaviour around it
    (leases, deadlines, failures, shutdown) is testable without an LLM, an MCP
    server or a GPU.
    """

    def __init__(
        self,
        store: CampaignStoreClient,
        run_task: TaskRunner,
        *,
        campaign_id: Optional[str] = None,
        poll_seconds: float = POLL_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        lease_seconds: Optional[float] = None,
    ) -> None:
        self.store = store
        self.run_task = run_task
        self.campaign_id = campaign_id
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds
        self._stopping = asyncio.Event()
        self.tasks_completed = 0
        self.tasks_failed = 0

    def stop(self) -> None:
        """Finish the task in hand, then exit without claiming another."""

        if not self._stopping.is_set():
            logger.info("shutdown requested; no further tasks will be claimed")
        self._stopping.set()

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    async def _report(
        self, state: str, task_id: Optional[str] = None, detail: Optional[str] = None
    ) -> None:
        """Put this supervisor in the pool listing, or refresh where it is.

        Never fatal: the conductor not being able to see us is worse than it
        seeing us late, but neither is worth losing an hour of RL over.
        """

        try:
            await asyncio.to_thread(
                self.store.report_presence,
                state, self.campaign_id, task_id,
                self.tasks_completed, self.tasks_failed, detail,
            )
        except (StoreUnavailable, LeaseLost) as exc:
            logger.debug("could not report presence: %s", exc)

    async def _heartbeat_forever(self, task_id: str) -> None:
        """Renew the lease until cancelled.

        A store that has gone away is logged and retried rather than ending
        the task: the campaign is hours long and the work in flight is worth
        more than the ledger entry. If the store stays down the lease lapses
        and the task is requeued, which is the right outcome anyway.

        Presence is renewed on the same beat, since a task can outlast the
        presence TTL many times over and a supervisor that vanished from the
        pool while working would read as capacity that is not there.
        """

        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await asyncio.to_thread(
                    self.store.heartbeat, task_id, self.lease_seconds
                )
            except LeaseLost:
                logger.error("task %s was taken from us; abandoning it", task_id)
                raise
            except StoreUnavailable as exc:
                logger.warning("heartbeat for %s failed: %s", task_id, exc)
            await self._report("busy", task_id, detail="running task")

    async def run_once(self) -> bool:
        """Claim and run one task. False when there was nothing to do."""

        await self._report("idle", detail="waiting for a task")
        task = await asyncio.to_thread(
            self.store.claim, self.campaign_id, self.lease_seconds
        )
        if not task:
            return False

        task_id = task["task_id"]
        logger.info("claimed task %s of campaign %s", task_id, task.get("campaign_id"))
        await self._report("busy", task_id, detail="starting task")

        heartbeat = asyncio.create_task(self._heartbeat_forever(task_id))
        try:
            outcome = await self.run_task(task)
        except asyncio.CancelledError:
            raise
        except LeaseLost as exc:
            # Someone else owns it; saying anything about it would overwrite
            # their result.
            logger.error("task %s: %s", task_id, exc)
            return True
        except Exception as exc:
            logger.exception("task %s failed", task_id)
            self.tasks_failed += 1
            with contextlib.suppress(StoreUnavailable, LeaseLost):
                await asyncio.to_thread(self.store.fail, task_id, str(exc))
            return True
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, LeaseLost):
                await heartbeat

        try:
            await asyncio.to_thread(self.store.post_result, task_id, outcome)
        except LeaseLost as exc:
            logger.error("task %s finished but the lease was lost: %s", task_id, exc)
            return True

        self.tasks_completed += 1
        logger.info("task %s done: %s", task_id, outcome.objective_profile)
        return True

    async def run_forever(self) -> None:
        """Work until stopped, backing off when the queue or the store is down."""

        logger.info(
            "supervisor %s working %s from %s",
            self.store.owner, self.campaign_id or "any campaign", self.store.base_url,
        )
        await self._report("starting", detail="joining the pool")

        while not self.stopping:
            try:
                did_work = await self.run_once()
            except StoreUnavailable as exc:
                # The conductor may still be starting, or the store may be
                # rolling. Neither is fatal to a pod that lives for hours.
                logger.warning("store unavailable: %s", exc)
                did_work = False

            if not did_work and not self.stopping:
                await self._sleep_unless_stopping(self.poll_seconds)

        # Leave on purpose rather than timing out of the listing, so a pool
        # drained on command does not read as three crashed supervisors for
        # the next two minutes.
        try:
            await asyncio.to_thread(self.store.drop_presence)
        except (StoreUnavailable, LeaseLost) as exc:
            logger.debug("could not leave the pool cleanly: %s", exc)

        logger.info(
            "supervisor %s stopped after %d task(s), %d failed",
            self.store.owner, self.tasks_completed, self.tasks_failed,
        )

    async def _sleep_unless_stopping(self, seconds: float) -> None:
        """Wait, but wake immediately on shutdown so a drain is not delayed."""

        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)


# ---------------------------------------------------------------------------
# The real task runner: Driver + Reviewer against one task
# ---------------------------------------------------------------------------

def _deadline_seconds(task: dict[str, Any]) -> Optional[float]:
    """Seconds left on the task's own deadline, if it has one."""

    deadline = (task.get("spec") or {}).get("deadline")
    if not deadline:
        return None
    try:
        when = datetime.fromisoformat(deadline)
    except ValueError:
        logger.warning("task %s has an unparseable deadline %r",
                       task.get("task_id"), deadline)
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


# Tools a supervisor's Driver must not have. It runs the task it was given; it
# does not get to redefine it, abandon it, or steer the fleet.
#
# `stop_optimization` is here because of what happened the first time a
# supervisor ran for real: the Driver polled the job thirteen times in six
# minutes, saw no completed steps -- a Boltz-scored step takes 3-10 minutes, so
# there was nothing wrong -- decided the run was broken and stopped it. That
# killed the job inside its first step, when REINVENT has nothing to
# checkpoint, and produced no candidates. Stopping is the supervisor's job: it
# owns the deadline and does it with a checkpoint.
#
# The TTP writers are here because the whole point of the variant TTP is that
# the weight vector is fixed in advance. A Driver that can edit or replace a
# TTP can quietly run something other than the point the campaign is measuring.
DRIVER_DENIED_TOOLS = frozenset({
    "stop_optimization",
    "create_ttp",
    "update_ttp",
    "delete_ttp",
    "start_campaign",
    "enqueue_task",
    "set_concurrency",
    "cancel_supervisor",
    "close_campaign",
    "finalize_campaign",
})

# What one RL step costs when every molecule is folded, measured on demo-lux.
# Quoted to the Driver so its patience has a basis: without a number it has no
# way to tell a slow step from a broken one.
STEP_SECONDS_HINT = "3 to 10 minutes"


def default_run_substitutes_for(spec: dict[str, Any]) -> bool:
    """Whether starting a plain run would still be this task's experiment.

    The supervisor starts the run itself when its agent finishes without
    doing so. That call takes only the TTP and the step count, so it generates
    from the default de novo prior -- a faithful stand-in while every task
    does, and not once a task's ``intent`` may have called for another prior.
    A substituted run would post into the same Pareto front while measuring
    something else, which is worse than the task failing and being retried.
    """

    return not (spec or {}).get("intent")


def build_goal(task: dict[str, Any], ttp_id: str) -> str:
    """The instruction handed to the Driver.

    Names the prepared TTP rather than restating the weights, so there is one
    source of truth for the weight vector and the agent cannot paraphrase it
    into something else.

    The task's ``intent`` goes in verbatim. Everything else here is structured
    and therefore only says what the schema anticipated: holding a chemotype,
    for instance, is a decision about which prior to generate from, and the
    agent is better placed to make it -- it can look the reference compound up
    and choose a model -- than a seed field would be. Without a channel for
    that the agent ran de novo not because it could not do better but because
    nobody asked it to.
    """

    spec = task.get("spec") or {}
    lines = [
        f"Run a lead-optimisation campaign against TTP '{ttp_id}'.",
        "",
        "The TTP already carries the exact objective weights and transforms "
        "this task is meant to explore -- use it as it is and do not modify "
        "its weights.",
    ]
    if spec.get("intent"):
        lines += [
            "",
            "What this run is for:",
            str(spec["intent"]).strip(),
            "",
            "That is the experiment. If it calls for generating from "
            "something other than the default de novo prior, choose the "
            "model and any seed molecules yourself -- look the reference "
            "compounds up rather than recalling their structures -- and say "
            "which you chose and why before starting.",
        ]
    if spec.get("max_steps"):
        lines.append(
            f"Run {spec['max_steps']} RL steps: the step count is budgeted "
            "against the campaign deadline, so do not raise it."
        )
    if spec.get("fidelity_tier") == "validation":
        lines.append(
            "This is a validation-tier task: results must not come from the "
            "scoring cache, so do not re-use a previous run's numbers."
        )
    lines += [
        "",
        "Start the optimization and monitor it, then report the best molecules "
        "with their per-objective scores.",
        "",
        f"Pacing: one RL step takes {STEP_SECONDS_HINT}, because every molecule "
        "in the batch is folded by Boltz. A run that reports no completed steps "
        "for several minutes is normal and is not evidence of a problem -- "
        "check again later rather than concluding it has stalled. Do not try to "
        "stop, restart or replace the run: this session has a wall-clock "
        "deadline that is enforced for you, and stopping early loses the work.",
    ]
    return "\n".join(lines)


class CoscientistTaskRunner:
    """Runs one task as a Driver + Reviewer session over MCP.

    Holds the MCP connection and the two agents for the life of one task, then
    tears them down: an RL campaign is long enough that a connection kept
    across tasks would mostly be idle, and a fresh pair starts each task with a
    clean transcript and budget.
    """

    def __init__(self, args: Any, store: CampaignStoreClient) -> None:
        self.args = args
        # Needed for the base TTP, which belongs to the campaign rather than
        # to each of its tasks.
        self.store = store

    def _objectives(self, task: dict[str, Any]) -> Optional[list[str]]:
        """The campaign's objectives, or None if they cannot be read.

        None keeps the old behaviour -- report whatever the CSV scored --
        rather than reporting nothing, which would fail the task.
        """

        try:
            return self.store.campaign(task["campaign_id"]).get("objectives")
        except Exception:
            logger.warning(
                "could not read campaign %s objectives; reporting every scored "
                "column", task.get("campaign_id"), exc_info=True,
            )
            return None

    def _base_ttp_id(self, task: dict[str, Any]) -> str:
        campaign = self.store.campaign(task["campaign_id"])
        base_ttp_id = campaign.get("base_ttp_id")
        if not base_ttp_id:
            raise RuntimeError(
                f"campaign {task['campaign_id']} names no base TTP, so there is "
                "nothing to apply this task's weights to"
            )
        return base_ttp_id

    async def __call__(self, task: dict[str, Any]) -> TaskOutcome:
        from reinvent4_agent.agent import Agent
        from reinvent4_agent.backends import create_backend
        from reinvent4_agent.coordinator import REVIEWER_READ_ONLY_TOOLS, Coordinator
        from reinvent4_agent.mcp_client import MCPClient
        from reinvent4_agent.system_prompt import (
            DRIVER_SYSTEM_PROMPT,
            REVIEWER_SYSTEM_PROMPT,
        )

        args = self.args
        spec = task.get("spec") or {}
        ttp_id = prepare_ttp_variant(
            self._base_ttp_id(task),
            task["task_id"],
            spec.get("weight_overrides"),
            spec.get("transform_overrides"),
        )
        goal = build_goal(task, ttp_id)
        budget = _deadline_seconds(task)

        def _backend(system_prompt: str):
            return create_backend(
                args.backend,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                gateway_key=args.gateway_key,
                api_version=args.api_version,
                verify_ssl=not args.no_verify_ssl,
                thinking_budget=args.thinking_budget,
                reasoning_effort=args.reasoning_effort,
                system_prompt=system_prompt,
            )

        async with MCPClient(
            server_command=args.server_command, server_args=args.server_args
        ) as mcp:
            # Everything except the tools that would let it redefine or
            # abandon its own task. Computed from the live tool list rather
            # than hardcoded, so a tool added later is allowed by default and
            # only the denied names have to be maintained.
            driver_tools = {
                tool["name"] for tool in await mcp.list_tools()
            } - DRIVER_DENIED_TOOLS

            driver = Agent(
                mcp, _backend(DRIVER_SYSTEM_PROMPT),
                max_tokens=args.max_tokens, log_dir=args.log_dir,
                name="driver", system_prompt=DRIVER_SYSTEM_PROMPT,
                allowed_tools=driver_tools,
            )

            coordinator = None
            if getattr(args, "multi_agent", True):
                reviewer = Agent(
                    mcp, _backend(REVIEWER_SYSTEM_PROMPT),
                    max_tokens=args.max_tokens, log_dir=args.log_dir,
                    name="reviewer", system_prompt=REVIEWER_SYSTEM_PROMPT,
                    allowed_tools=set(REVIEWER_READ_ONLY_TOOLS),
                )
                coordinator = Coordinator(
                    driver, reviewer,
                    max_turns=spec.get("max_turns") or args.max_turns,
                    token_budget=spec.get("token_budget") or args.token_budget,
                )

            started = time.monotonic()
            if coordinator is not None:
                result = await coordinator.run(goal)
                verdict, turns = result.status, result.turns
            else:
                # One pass. There is no second agent to hand a verdict to, so
                # there is nothing to take turns over.
                await driver.chat(goal)
                verdict, turns = "single-agent", 1

            job_id = job_for_ttp(ttp_id)
            if not job_id and not default_run_substitutes_for(spec):
                raise RuntimeError(
                    f"the agent did not start a run for TTP '{ttp_id}' "
                    f"(verdict: {verdict}), and this task carries an intent "
                    "that may call for a non-default prior -- starting a "
                    "default run here would silently run a different "
                    "experiment"
                )
            if not job_id:
                # The agent finished without starting the run. With a Reviewer
                # the Coordinator took several turns, so there were more
                # chances; a single pass gets one. Nothing about the call is
                # the agent's decision anyway -- the TTP and the step count
                # both come from the task -- so start it here rather than ask
                # again and hope.
                logger.warning(
                    "the agent did not start a run for TTP '%s' (verdict: %s); "
                    "starting it directly", ttp_id, verdict,
                )
                job_id = await asyncio.to_thread(
                    self._start_job, ttp_id, spec.get("max_steps")
                )

            remaining = None if budget is None else max(
                0.0, budget - (time.monotonic() - started)
            )
            timed_out = await self._monitor(coordinator, goal, job_id, remaining)
            tokens = driver.total_tokens if coordinator is None else result.total_tokens

        return await self._harvest_or_release(
            task, job_id, verdict, turns, tokens, timed_out
        )

    async def _harvest_or_release(
        self,
        task: dict[str, Any],
        job_id: str,
        verdict: str,
        turns: int,
        tokens: int,
        timed_out: bool,
    ) -> TaskOutcome:
        """Read the result, and release the job if there is no result to read.

        Once the task is failing nothing waits on its job, but it still holds
        a GPU and one of the runner's concurrency slots -- which is how two
        abandoned jobs came to block every later task in a campaign with
        "Maximum concurrent jobs (2) already active".
        """

        try:
            return self._harvest(task, job_id, verdict, turns, tokens, timed_out)
        except Exception:
            logger.warning(
                "task %s failed; stopping its job %s", task["task_id"], job_id
            )
            # The task's own failure is the useful one to report, so a stop
            # that also fails must not replace it.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._stop_job, job_id)
            raise

    async def _monitor(
        self,
        coordinator: Optional[Any],
        goal: str,
        job_id: str,
        remaining: Optional[float],
    ) -> bool:
        """Watch the job to its end or the deadline. True if the deadline won.

        On the deadline the job is stopped rather than left running: it asks
        REINVENT4 to checkpoint, so the GPU is handed back and the work can be
        resumed instead of thrown away.
        """

        monitor = (
            coordinator.monitor_campaign(
                goal, job_id=job_id, interval=self.args.monitor_interval
            )
            if coordinator is not None
            else self._watch_job(job_id)
        )
        if remaining is None:
            await monitor
            return False

        try:
            await asyncio.wait_for(monitor, timeout=remaining)
        except asyncio.TimeoutError:
            logger.warning("task deadline reached; stopping job %s", job_id)
            await asyncio.to_thread(self._stop_job, job_id)
            return True
        return False

    async def _watch_job(self, job_id: str) -> None:
        """Wait for the job to stop running. The single-agent monitor.

        ``monitor_campaign`` is the *Reviewer* cycle -- it assesses the run and
        hands the Driver a verdict to act on -- so with no Reviewer there is
        nobody to form that verdict and nothing to act on it. This watches the
        state instead; the task deadline above is what bounds the wait.
        """

        from reinvent4_mcp.jobs.manager import ACTIVE_JOB_STATES
        from reinvent4_mcp.tools.results import get_optimization_status

        # A job owes a result while it is queued *or* running. Waiting only on
        # "running" exits on the first poll, because a job is queued for the
        # moment before it starts -- which harvested a 0-byte CSV and failed
        # the task while its job carried on folding molecules.
        unfinished = {state.value for state in ACTIVE_JOB_STATES}

        while True:
            status = await asyncio.to_thread(get_optimization_status, job_id)

            # Drive off `state`, which is the authoritative field. Testing for
            # an "error" *key* does not work: a job's to_dict() always carries
            # one, set to None when nothing is wrong, so that test is true for
            # every job and returned on the first poll -- harvesting a 0-byte
            # CSV while the job was happily folding molecules.
            state = status.get("state")
            if not state:
                logger.warning(
                    "job %s cannot be described (%s); not waiting on it",
                    job_id, status.get("error") or "no state reported",
                )
                return
            if state not in unfinished:
                logger.info("job %s finished as %s", job_id, state)
                return
            await asyncio.sleep(self.args.monitor_interval)

    @staticmethod
    def _start_job(ttp_id: str, max_steps: Optional[int]) -> str:
        """Start the task's run without the agent. Returns the job id."""

        from reinvent4_mcp.tools.optimization import run_optimization

        kwargs: dict[str, Any] = {"ttp_id": ttp_id}
        if max_steps:
            kwargs["max_steps"] = max_steps

        result = run_optimization(**kwargs)
        if "error" in result and result["error"]:
            raise RuntimeError(
                f"could not start a run for TTP '{ttp_id}': {result['error']}"
            )
        job_id = result.get("job_id")
        if not job_id:
            raise RuntimeError(
                f"starting a run for TTP '{ttp_id}' returned no job id: {result}"
            )

        logger.info("started job %s for TTP %s directly", job_id, ttp_id)
        return job_id

    @staticmethod
    def _stop_job(job_id: str) -> None:
        from reinvent4_mcp.tools.optimization import stop_optimization

        result = stop_optimization(job_id)
        logger.info("stopped job %s: %s", job_id, result.get("message") or result)

    def _harvest(
        self,
        task: dict[str, Any],
        job_id: str,
        verdict: str,
        turns: int,
        tokens: int,
        timed_out: bool,
    ) -> TaskOutcome:
        """Read the achieved profile out of the job's own results."""

        from reinvent4_mcp.tools.results import get_top_candidates

        top = get_top_candidates(job_id, top_n=25)
        if "error" in top:
            raise RuntimeError(f"job {job_id} produced no readable results: "
                               f"{top['error']}")

        candidates = top.get("candidates") or []
        # Restricted to the campaign's own objectives. A results CSV carries a
        # column per scorer diagnostic too -- std, confidence_score,
        # affinity_probability_binary and friends -- and reporting those as
        # objectives put a dozen keys on every point of the front where three
        # belonged.
        profile = objective_profile(candidates, self._objectives(task))
        if not profile:
            raise RuntimeError(
                f"job {job_id} returned {len(candidates)} candidate(s) but no "
                "per-objective scores, so the task has no coordinates on the front"
            )

        summary = (
            f"verdict={verdict}, turns={turns}, "
            f"tokens={tokens}, candidates={len(candidates)}"
        )
        if timed_out:
            summary += ", stopped at the task deadline (checkpointed)"

        return TaskOutcome(
            objective_profile=profile,
            candidates=candidates,
            job_id=job_id,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> Any:
    """Reuse the CLI's parser so the agent is configured the same way here."""

    from reinvent4_agent.cli import parse_args

    return parse_args(argv or [])


async def _run(args: Any) -> None:
    store = CampaignStoreClient(
        os.getenv("CAMPAIGN_STORE_URL", STORE_URL), supervisor_id()
    )
    lease = os.getenv("SUPERVISOR_LEASE_SECONDS")
    supervisor = Supervisor(
        store,
        CoscientistTaskRunner(args, store),
        campaign_id=os.getenv("SUPERVISOR_CAMPAIGN_ID") or None,
        lease_seconds=float(lease) if lease else None,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # A pod being deleted should stop claiming and let the task in hand
        # finish or lapse, not die mid-campaign with the ledger none the wiser.
        loop.add_signal_handler(sig, supervisor.stop)

    await supervisor.run_forever()


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args(argv)
    from reinvent4_agent.cli import _resolve_api_key

    args.api_key = _resolve_api_key(args)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
