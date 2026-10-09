# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two-agent co-scientist coordinator (Driver <-> Reviewer).

Owns two :class:`~reinvent4_agent.agent.Agent` instances that share a single
:class:`~reinvent4_agent.mcp_client.MCPClient` but have independent conversation
state:

- **Driver** — the full-toolset medicinal chemist that produces candidates.
- **Reviewer** — a read-only gatekeeper/critic that vets those candidates.

The coordinator runs a turn-based loop: the Driver produces a batch of
candidates and emits a structured handoff, the Reviewer critiques it and emits
its own structured verdict, and the coordinator either finalizes (on approval)
or feeds the Reviewer's requested changes back to the Driver. A global
turn/token budget guarantees the loop always terminates.

Handoffs are intentionally *typed payloads*, not free-form chat: each agent
appends a small fenced ``json`` block that this module parses, so the exchange
stays cheap and machine-checkable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from reinvent4_agent.agent import Agent

logger = logging.getLogger(__name__)

# Job states that mean the optimization campaign is still going.
_ACTIVE_JOB_STATES = frozenset({"running", "queued"})

# Tools the read-only Reviewer is allowed to call. Everything that creates,
# updates, deletes, runs, or stops state is intentionally excluded so the
# Reviewer physically cannot mutate the campaign.
REVIEWER_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_results",
        "get_top_candidates",
        "get_ttp",
        "list_ttps",
        "get_optimization_status",
        "list_jobs",
        "list_scoring_components",
        "list_available_models",
        "list_transforms",
        "list_external_services",
    }
)

_APPROVE = "approve"
_REVISE = "revise"
_REJECT = "reject"
_VALID_VERDICTS = {_APPROVE, _REVISE, _REJECT}

# Matches a fenced ```json ... ``` block; the last one in a message wins.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Structured handoff protocol
# ---------------------------------------------------------------------------


@dataclass
class DriverHandoff:
    """Structured payload the Driver passes to the Reviewer."""

    candidates: list[str] = field(default_factory=list)
    job_id: str | None = None
    csv_path: str | None = None
    rationale: str = ""
    ready_for_review: bool = False
    raw_text: str = ""

    @classmethod
    def parse(cls, text: str) -> DriverHandoff | None:
        """Extract a Driver handoff from an agent message, if present."""
        payload = _extract_json_block(text)
        if payload is None:
            return None
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list):
            candidates = [str(candidates)]
        return cls(
            candidates=[str(c) for c in candidates],
            job_id=_opt_str(payload.get("job_id")),
            csv_path=_opt_str(payload.get("csv_path")),
            rationale=str(payload.get("rationale", "")),
            ready_for_review=bool(payload.get("ready_for_review", False)),
            raw_text=text,
        )


@dataclass
class ReviewerHandoff:
    """Structured payload the Reviewer passes back to the Driver."""

    verdict: str = _REVISE
    ranking: list[str] = field(default_factory=list)
    liabilities: list[str] = field(default_factory=list)
    requested_changes: list[str] = field(default_factory=list)
    rationale: str = ""
    raw_text: str = ""

    @classmethod
    def parse(cls, text: str) -> ReviewerHandoff | None:
        """Extract a Reviewer handoff from an agent message, if present."""
        payload = _extract_json_block(text)
        if payload is None:
            return None
        verdict = str(payload.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            # Be lenient: default an unrecognised verdict to "revise" so the
            # loop keeps making progress rather than crashing on a typo.
            logger.warning("Unrecognised reviewer verdict %r; treating as revise", verdict)
            verdict = _REVISE
        return cls(
            verdict=verdict,
            ranking=_str_list(payload.get("ranking")),
            liabilities=_str_list(payload.get("liabilities")),
            requested_changes=_str_list(payload.get("requested_changes")),
            rationale=str(payload.get("rationale", "")),
            raw_text=text,
        )


@dataclass
class CoordinatorResult:
    """Outcome of a full co-scientist run."""

    status: str  # "approved" | "rejected" | "budget_exhausted" | "incomplete"
    final_message: str
    turns: int
    total_tokens: int
    last_reviewer: ReviewerHandoff | None = None
    last_driver: DriverHandoff | None = None


@dataclass
class CampaignMonitorResult:
    """Outcome of a periodic campaign-monitoring session."""

    status: str  # "no_active_campaign" | "campaign_<state>" | "max_cycles" | "stopped"
    cycles: int
    job_id: str | None
    total_tokens: int = 0
    last_reviewer: ReviewerHandoff | None = None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class Coordinator:
    """Turn-based Driver <-> Reviewer loop with a global budget."""

    def __init__(
        self,
        driver: Agent,
        reviewer: Agent,
        *,
        max_turns: int = 8,
        token_budget: int | None = None,
    ) -> None:
        self.driver = driver
        self.reviewer = reviewer
        self.max_turns = max_turns
        self.token_budget = token_budget
        self._turn = 0

    @property
    def total_tokens(self) -> int:
        """Combined token usage across both agents."""
        return self.driver.total_tokens + self.reviewer.total_tokens

    def _budget_exhausted(self) -> bool:
        if self._turn >= self.max_turns:
            return True
        if self.token_budget is not None and self.total_tokens >= self.token_budget:
            return True
        return False

    async def run(
        self,
        goal: str,
        *,
        event_sink: "CoordinatorEvents | None" = None,
    ) -> CoordinatorResult:
        """Drive a full Driver -> Reviewer -> Driver exchange to a verdict.

        Returns once the Reviewer approves, rejects, the budget is exhausted,
        or the Driver stops producing review-ready handoffs (e.g. it needs the
        user). ``event_sink`` receives UI callbacks (turn banners, messages).
        """
        sink = event_sink or _NullCoordinatorEvents()
        driver_prompt = _initial_driver_prompt(goal)
        last_driver: DriverHandoff | None = None
        last_reviewer: ReviewerHandoff | None = None

        while not self._budget_exhausted():
            # -- Driver turn ------------------------------------------------
            self._turn += 1
            sink.on_turn_start("driver", self._turn)
            driver_text = await self.driver.chat(driver_prompt)
            sink.on_agent_message("driver", driver_text)
            sink.on_budget(
                self._turn, self.max_turns, self.total_tokens, self.token_budget
            )

            handoff = DriverHandoff.parse(driver_text)
            last_driver = handoff or last_driver
            if handoff is None or not handoff.ready_for_review:
                # The Driver is still working or is asking the user something;
                # there is nothing concrete to review yet, so stop cleanly.
                return CoordinatorResult(
                    status="incomplete",
                    final_message=driver_text,
                    turns=self._turn,
                    total_tokens=self.total_tokens,
                    last_driver=last_driver,
                    last_reviewer=last_reviewer,
                )

            if self._budget_exhausted():
                break

            # -- Reviewer turn ----------------------------------------------
            self._turn += 1
            sink.on_turn_start("reviewer", self._turn)
            review_prompt = _build_review_prompt(goal, handoff)
            reviewer_text = await self.reviewer.chat(review_prompt)
            sink.on_agent_message("reviewer", reviewer_text)
            sink.on_budget(
                self._turn, self.max_turns, self.total_tokens, self.token_budget
            )

            verdict = ReviewerHandoff.parse(reviewer_text)
            last_reviewer = verdict or last_reviewer

            if verdict is not None and verdict.verdict == _APPROVE:
                return CoordinatorResult(
                    status="approved",
                    final_message=reviewer_text,
                    turns=self._turn,
                    total_tokens=self.total_tokens,
                    last_driver=last_driver,
                    last_reviewer=verdict,
                )
            if verdict is not None and verdict.verdict == _REJECT:
                return CoordinatorResult(
                    status="rejected",
                    final_message=reviewer_text,
                    turns=self._turn,
                    total_tokens=self.total_tokens,
                    last_driver=last_driver,
                    last_reviewer=verdict,
                )

            # Revise: feed the Reviewer's requested changes back to the Driver.
            driver_prompt = _build_revision_prompt(verdict, reviewer_text)

        return CoordinatorResult(
            status="budget_exhausted",
            final_message=(
                "Reached the co-scientist budget "
                f"({self._turn} turns, {self.total_tokens:,} tokens) "
                "without an approval. Returning the latest state."
            ),
            turns=self._turn,
            total_tokens=self.total_tokens,
            last_driver=last_driver,
            last_reviewer=last_reviewer,
        )

    # ------------------------------------------------------------------
    # Periodic campaign monitoring (Reviewer critiques, Driver acts)
    # ------------------------------------------------------------------

    async def monitor_campaign(
        self,
        goal: str,
        *,
        job_id: str | None = None,
        interval: float = 300.0,
        max_cycles: int | None = None,
        event_sink: "CoordinatorEvents | None" = None,
    ) -> CampaignMonitorResult:
        """Every ``interval`` seconds, have the Reviewer assess the running
        optimization campaign and hand a verdict to the Driver, who acts on it.

        The Reviewer (read-only) inspects the live trajectory/diagnostics and
        returns a verdict — ``approve`` (healthy, continue), ``revise``
        (intervene with concrete changes), or ``reject`` (stop now). The
        Driver, with its full toolset, then acts: stop, adjust the TTP, resume,
        or let the run continue.

        Loops until the campaign is no longer running, the ``reject`` verdict
        has been acted on, ``max_cycles`` is reached, or the task is cancelled.
        """
        sink = event_sink or _NullCoordinatorEvents()

        if job_id is None:
            job_id = await self._running_optimization_job()
        if job_id is None:
            return CampaignMonitorResult(
                status="no_active_campaign",
                cycles=0,
                job_id=None,
                total_tokens=self.total_tokens,
            )

        cycle = 0
        last_reviewer: ReviewerHandoff | None = None

        while True:
            await asyncio.sleep(interval)

            state = await self._job_state(job_id)
            if state not in _ACTIVE_JOB_STATES:
                sink.on_monitor_end(job_id, state or "ended")
                return CampaignMonitorResult(
                    status=f"campaign_{state or 'ended'}",
                    cycles=cycle,
                    job_id=job_id,
                    total_tokens=self.total_tokens,
                    last_reviewer=last_reviewer,
                )

            cycle += 1
            verdict = await self.run_monitor_cycle(
                goal, job_id, cycle, event_sink=sink
            )
            last_reviewer = verdict or last_reviewer

            if max_cycles is not None and cycle >= max_cycles:
                return CampaignMonitorResult(
                    status="max_cycles",
                    cycles=cycle,
                    job_id=job_id,
                    total_tokens=self.total_tokens,
                    last_reviewer=last_reviewer,
                )
            # On a "reject" verdict the Driver is expected to stop the job; the
            # next loop iteration observes the state change and terminates.

    async def run_monitor_cycle(
        self,
        goal: str,
        job_id: str,
        cycle: int,
        *,
        event_sink: "CoordinatorEvents | None" = None,
    ) -> ReviewerHandoff | None:
        """Run one Reviewer-assess -> Driver-act cycle for a campaign.

        Shared by the autonomous ``monitor_campaign`` loop and the interactive
        session's background monitor. Returns the parsed Reviewer verdict (if
        any). Callers are responsible for serialising concurrent access to the
        shared MCP session (e.g. via the Driver's chat lock).
        """
        sink = event_sink or _NullCoordinatorEvents()
        sink.on_monitor_cycle_start(cycle, job_id)

        # -- Reviewer: independent read-only campaign assessment ------------
        sink.on_turn_start("reviewer", cycle)
        reviewer_text = await self.reviewer.chat(
            _build_monitor_review_prompt(goal, job_id)
        )
        sink.on_agent_message("reviewer", reviewer_text)
        verdict = ReviewerHandoff.parse(reviewer_text)

        # -- Driver: act on the Reviewer's verdict --------------------------
        sink.on_turn_start("driver", cycle)
        driver_text = await self.driver.chat(
            _build_monitor_action_prompt(goal, job_id, verdict, reviewer_text)
        )
        sink.on_agent_message("driver", driver_text)
        return verdict

    async def _running_optimization_job(self) -> str | None:
        """Return the job id of the currently running optimization, if any."""
        try:
            raw = await self.driver.mcp.call_tool("list_jobs", {})
            jobs = json.loads(raw)
        except Exception:
            logger.debug("Failed to list jobs for monitoring", exc_info=True)
            return None
        if not isinstance(jobs, list):
            return None
        for job in jobs:
            if not isinstance(job, dict):
                continue
            if (
                job.get("state") in _ACTIVE_JOB_STATES
                and job.get("job_type", "optimization") == "optimization"
            ):
                return job.get("job_id")
        return None

    async def _job_state(self, job_id: str) -> str | None:
        """Return the current state string for a job, or None if unknown."""
        try:
            raw = await self.driver.mcp.call_tool(
                "get_optimization_status", {"job_id": job_id}
            )
            data = json.loads(raw)
        except Exception:
            logger.debug("Failed to poll job %s state", job_id, exc_info=True)
            return None
        if isinstance(data, dict):
            return data.get("state")
        return None


# ---------------------------------------------------------------------------
# Event sink protocol (keeps the coordinator UI-agnostic)
# ---------------------------------------------------------------------------


class CoordinatorEvents:
    """Callbacks the coordinator emits so a UI can render the exchange."""

    def on_turn_start(self, agent_name: str, turn: int) -> None: ...
    def on_agent_message(self, agent_name: str, text: str) -> None: ...
    def on_budget(
        self, turn: int, max_turns: int, total_tokens: int, budget: int | None
    ) -> None: ...

    def on_monitor_cycle_start(self, cycle: int, job_id: str) -> None: ...
    def on_monitor_end(self, job_id: str, state: str) -> None: ...


class _NullCoordinatorEvents(CoordinatorEvents):
    def on_turn_start(self, agent_name: str, turn: int) -> None:
        pass

    def on_agent_message(self, agent_name: str, text: str) -> None:
        pass

    def on_budget(
        self, turn: int, max_turns: int, total_tokens: int, budget: int | None
    ) -> None:
        pass

    def on_monitor_cycle_start(self, cycle: int, job_id: str) -> None:
        pass

    def on_monitor_end(self, job_id: str, state: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _initial_driver_prompt(goal: str) -> str:
    return (
        f"{goal}\n\n"
        "Work toward this goal and, once you have a concrete batch of scored "
        "candidates, hand them off for independent review using the JSON "
        "handoff block described in your instructions."
    )


def _build_review_prompt(goal: str, handoff: DriverHandoff) -> str:
    lines = [
        "[HANDOFF FROM DRIVER] Independently review the following candidates.",
        "",
        f"Original goal: {goal}",
        "",
        f"Driver rationale: {handoff.rationale or '(none provided)'}",
    ]
    if handoff.job_id:
        lines.append(f"Optimization job id: {handoff.job_id}")
    if handoff.csv_path:
        lines.append(f"Results CSV: {handoff.csv_path}")
    if handoff.candidates:
        lines.append("")
        lines.append("Candidate SMILES:")
        lines.extend(f"  - {smi}" for smi in handoff.candidates)
    lines += [
        "",
        "Pull the real scored rows with your read-only tools, apply the Boltz2 "
        "confidence gate and PoseBusters thresholds, red-team medchem "
        "liabilities, then return your verdict as the JSON handoff block.",
    ]
    return "\n".join(lines)


def _build_monitor_review_prompt(goal: str, job_id: str) -> str:
    return "\n".join(
        [
            "[CAMPAIGN MONITOR] Independently assess the health of the running "
            "optimization campaign and return a verdict.",
            "",
            f"Original goal: {goal}",
            f"Active optimization job id: {job_id}",
            "",
            "Use your read-only tools to gather evidence before judging:",
            f"  - get_optimization_status(job_id='{job_id}') for the trajectory, "
            "convergence_status, per-component health, validity, and warnings.",
            "  - get_top_candidates / read_results to sanity-check emerging "
            "molecules against the goal.",
            "",
            "Report a concise status summary (progress, mean/best score, "
            "convergence_status, notable warnings) and decide the verdict, "
            "reusing the JSON handoff block with this campaign-monitoring "
            "meaning:",
            "  - \"approve\" -> campaign is healthy; let it continue.",
            "  - \"revise\"  -> intervene: put concrete adjustments in "
            "requested_changes (e.g. lower a failing component's weight, change "
            "a transform, stop+resume with new sigma/learning_rate).",
            "  - \"reject\"  -> stop the campaign now (diverging, a component is "
            "failing, the run is stalled, or the goal is already met).",
            "Put the status summary in \"rationale\" and any per-component "
            "concerns in \"liabilities\".",
        ]
    )


def _build_monitor_action_prompt(
    goal: str,
    job_id: str,
    verdict: ReviewerHandoff | None,
    reviewer_text: str,
) -> str:
    if verdict is None:
        return (
            "[CAMPAIGN MONITOR — REVIEWER ASSESSMENT] The Reviewer assessed the "
            f"running campaign (job {job_id}) but did not return a parseable "
            "verdict. Read their assessment below, then act as the campaign "
            "owner: continue, adjust, or stop the job as warranted, and explain "
            "what you did.\n\n" + reviewer_text
        )
    lines = [
        "[CAMPAIGN MONITOR — REVIEWER ASSESSMENT] The Reviewer assessed the "
        f"running campaign (job {job_id}). Act on it as the campaign owner.",
        "",
        f"Goal: {goal}",
        f"Reviewer verdict: {verdict.verdict}",
        f"Reviewer summary: {verdict.rationale or '(none)'}",
    ]
    if verdict.liabilities:
        lines.append("")
        lines.append("Concerns flagged:")
        lines.extend(f"  - {x}" for x in verdict.liabilities)
    if verdict.requested_changes:
        lines.append("")
        lines.append("Requested changes:")
        lines.extend(f"  - {x}" for x in verdict.requested_changes)
    lines += [
        "",
        "Act accordingly:",
        "  - approve -> the run is healthy; confirm briefly and let it "
        "continue (no tool calls required).",
        f"  - revise  -> apply the requested changes to job {job_id} (e.g. "
        "stop_optimization + update_ttp + run_optimization with resume_from), "
        "and explain what you changed and why.",
        f"  - reject  -> stop_optimization(job_id='{job_id}') now and explain "
        "why the campaign was halted.",
        "Only act on this campaign; do not launch unrelated jobs.",
    ]
    return "\n".join(lines)


def _build_revision_prompt(
    verdict: ReviewerHandoff | None, reviewer_text: str
) -> str:
    if verdict is None:
        # Couldn't parse a structured verdict; forward the raw critique.
        return (
            "[REVIEWER FEEDBACK] The Reviewer responded but did not return a "
            "parseable verdict. Read their critique below, address it, and "
            "hand off a revised batch.\n\n" + reviewer_text
        )
    lines = [
        "[REVIEWER FEEDBACK] The Reviewer asked for revisions before approval.",
        "",
        f"Verdict: {verdict.verdict}",
        f"Rationale: {verdict.rationale or '(none)'}",
    ]
    if verdict.liabilities:
        lines.append("")
        lines.append("Liabilities flagged:")
        lines.extend(f"  - {x}" for x in verdict.liabilities)
    if verdict.requested_changes:
        lines.append("")
        lines.append("Requested changes (address each explicitly):")
        lines.extend(f"  - {x}" for x in verdict.requested_changes)
    lines += [
        "",
        "Revise your candidates accordingly and hand off an updated batch for "
        "another review using the JSON handoff block.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Return the parsed content of the last ```json block in ``text``."""
    if not text:
        return None
    matches = _JSON_BLOCK_RE.findall(text)
    for candidate in reversed(matches):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in ("null", "none"):
        return None
    return s


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]
