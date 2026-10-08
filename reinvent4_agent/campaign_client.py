"""HTTP client for ``campaign-store``, shared by the supervisor and conductor.

One module so there is a single definition of the ledger's wire format. The
two sides use disjoint halves of it -- supervisors claim and report, the
conductor opens campaigns, enqueues work and reads the front -- but they have
to agree on what a task looks like, and a second copy would drift.

The error types matter as much as the calls. ``LeaseLost`` and
``StoreUnavailable`` are separated because they demand opposite responses: one
means stop touching this task, the other means wait and try again.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

STORE_URL = os.getenv("CAMPAIGN_STORE_URL", "http://dd-agentic-campaign-store:8087")

# Long enough that a request waiting on the store's write lock is not mistaken
# for the store being down, short enough that a hung call does not stall the
# caller for the whole lease.
STORE_TIMEOUT_SECONDS = 60.0


def supervisor_id() -> str:
    """Identity in the ledger. The pod name, when there is one."""

    return os.getenv("SUPERVISOR_ID") or socket.gethostname()


class StoreUnavailable(Exception):
    """The ledger could not be reached or refused the call."""


class LeaseLost(Exception):
    """This supervisor no longer holds the task it is working on.

    Raised rather than swallowed: continuing would mean spending a GPU on work
    the store has already handed to someone else.
    """


@dataclass
class TaskOutcome:
    """What a finished task reports back to the ledger."""

    objective_profile: dict[str, float]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    job_id: Optional[str] = None
    summary: Optional[str] = None


def _detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)[:300]
    return str(body)[:300]


class CampaignStoreClient:
    """The ledger's HTTP surface. Blocking; async callers use a thread."""

    def __init__(self, base_url: str = STORE_URL, owner: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.owner = owner or supervisor_id()

    def _call(self, method: str, path: str, **json_body: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, json=json_body or None, timeout=STORE_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            raise StoreUnavailable(f"{url} is unreachable: {exc}") from exc

        # 409 is the store telling us someone else owns this task now, which
        # is a different problem from the store being broken.
        if response.status_code == 409:
            raise LeaseLost(_detail(response))
        if not response.ok:
            raise StoreUnavailable(
                f"{url} returned {response.status_code}: {_detail(response)}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def _get(self, path: str) -> Any:
        return self._call("GET", path)

    # -- supervisor side ---------------------------------------------------

    def claim(
        self, campaign_id: Optional[str] = None, lease_seconds: Optional[float] = None
    ) -> Optional[dict[str, Any]]:
        """The next task, or None when the queue is empty."""

        body: dict[str, Any] = {"owner": self.owner}
        if campaign_id:
            body["campaign_id"] = campaign_id
        if lease_seconds:
            body["lease_seconds"] = lease_seconds
        return self._call("POST", "/tasks/claim", **body)

    def heartbeat(self, task_id: str, lease_seconds: Optional[float] = None) -> None:
        body: dict[str, Any] = {"owner": self.owner}
        if lease_seconds:
            body["lease_seconds"] = lease_seconds
        self._call("POST", f"/tasks/{task_id}/heartbeat", **body)

    def post_result(self, task_id: str, outcome: TaskOutcome) -> dict[str, Any]:
        return self._call(
            "POST", f"/tasks/{task_id}/result",
            owner=self.owner,
            objective_profile=outcome.objective_profile,
            candidates=outcome.candidates,
            job_id=outcome.job_id,
            summary=outcome.summary,
        )

    def fail(self, task_id: str, error: str, requeue: bool = True) -> dict[str, Any]:
        return self._call(
            "POST", f"/tasks/{task_id}/fail",
            owner=self.owner, error=error, requeue=requeue,
        )

    # -- conductor side ----------------------------------------------------

    def create_campaign(
        self,
        goal: str,
        base_ttp_id: str,
        objectives: list[str],
        concurrency: int = 1,
        dispatch_seconds: Optional[float] = None,
        grace_seconds: float = 0.0,
    ) -> dict[str, Any]:
        return self._call(
            "POST", "/campaigns",
            goal=goal, base_ttp_id=base_ttp_id, objectives=objectives,
            concurrency=concurrency, dispatch_seconds=dispatch_seconds,
            grace_seconds=grace_seconds,
        )

    def enqueue_task(self, campaign_id: str, **spec: Any) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/tasks", **spec)

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}")

    def campaigns(self) -> list[dict[str, Any]]:
        return self._get("/campaigns")

    def status(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/status")

    def pareto(self, campaign_id: str) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/pareto")

    def results(self, campaign_id: str) -> list[dict[str, Any]]:
        return self._get(f"/campaigns/{campaign_id}/results")

    def tasks(
        self, campaign_id: str, state: Optional[str] = None
    ) -> list[dict[str, Any]]:
        query = f"?state={state}" if state else ""
        return self._get(f"/campaigns/{campaign_id}/tasks{query}")

    def events(self, campaign_id: str, since: int = 0) -> dict[str, Any]:
        return self._get(f"/campaigns/{campaign_id}/events?since={since}")

    def set_concurrency(self, campaign_id: str, concurrency: int) -> dict[str, Any]:
        return self._call(
            "POST", f"/campaigns/{campaign_id}/concurrency", concurrency=concurrency
        )

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Take a task back from whoever holds it."""

        return self._call("POST", f"/tasks/{task_id}/cancel")

    def close_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/close")

    def finalize_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._call("POST", f"/campaigns/{campaign_id}/finalize")
