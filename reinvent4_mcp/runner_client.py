"""HTTP client for ``reinvent-runner``, shaped as a drop-in ``JobManager``.

Every caller of ``get_job_manager()`` uses the same seven methods and reads
plain dicts back, so pointing them at a remote runner needs no change in the
tools: ``get_job_manager()`` returns this instead when ``runner.enabled`` is
set. That is what lets the GPU half of the stack move out of the agent process
without touching the job lifecycle.

Errors keep the manager's contract. The manager returns ``{"error": ...}`` for
outcomes it expects (unknown job, no slots free), and the runner passes those
through as ``200``; anything else -- an unreachable runner, a 5xx -- is turned
into the same shape here, so a tool cannot tell transport trouble from an
application refusal by accident. The one thing it must never do is raise
where the local manager would have returned a dict.

The list-returning methods are the exception, and raise ``RunnerUnavailable``
instead. There is no error dict to hand back without breaking every caller
that iterates, and an empty list is a lie a caller acts on: ``job_for_ttp``
reads it as "the agent started no run" and starts a second one on the same
GPU. A transient runner blip must not become a duplicate RL job.

The three registry lookups and ``score_molecules`` are not manager methods, so
they are proxied as functions.
"""

from __future__ import annotations

__all__ = [
    "RemoteJobManager",
    "RunnerUnavailable",
    "runner_enabled",
    "runner_health",
    "remote_list_scoring_components",
    "remote_list_available_models",
    "remote_list_transforms",
    "remote_score_molecules",
]

import logging
import os
from typing import Any, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# Set inside the runner's own container. The runner reads the same config file
# as the agent, so with ``runner.enabled`` set it would otherwise proxy every
# job and every score to itself.
SELF_ENV = "REINVENT_RUNNER_SELF"


def _config():
    from reinvent4_mcp.config import get_config

    return get_config().runner


def runner_enabled() -> bool:
    """Whether jobs and scoring should go to the runner instead of in-process.

    Always False inside the runner itself, whatever the config says: that
    process is the one holding REINVENT4, and a request it forwarded would
    come straight back to it.
    """
    if os.getenv(SELF_ENV, "").strip().lower() not in ("", "0", "false", "no", "off"):
        return False
    return bool(_config().enabled)


class RunnerUnavailable(RuntimeError):
    """The runner could not be reached, or answered with something unusable."""


def _error(message: str) -> dict[str, Any]:
    logger.error("runner: %s", message)
    return {"error": message}


def _as_list(result: Any, what: str) -> list[dict[str, Any]]:
    """A list from the runner, or ``RunnerUnavailable`` carrying its message."""
    if isinstance(result, list):
        return result
    detail = result.get("error") if isinstance(result, dict) else repr(result)
    raise RunnerUnavailable(f"could not list {what} on reinvent-runner: {detail}")


def _request(
    method: str,
    path: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Call the runner and return its decoded body, or an error dict.

    A non-2xx is reported with the runner's own message when it sent one, so a
    refusal like "Maximum concurrent jobs" survives the trip intact.
    """
    config = _config()
    url = f"{config.base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=json_body,
            params=params,
            timeout=timeout or config.timeout_seconds,
        )
    except requests.RequestException as exc:
        return _error(f"reinvent-runner at {url} is unreachable: {exc}")

    try:
        body = response.json()
    except ValueError:
        body = None

    if not response.ok:
        detail = ""
        if isinstance(body, dict):
            detail = str(body.get("error") or body.get("detail") or "")
        return _error(
            f"reinvent-runner returned {response.status_code} for {path}"
            + (f": {detail[:300]}" if detail else "")
        )

    if body is None:
        return _error(f"reinvent-runner sent an unparseable response for {path}")
    return body


def runner_health() -> dict[str, Any]:
    """Health plus whether REINVENT4 is actually importable over there."""
    return _request("GET", "/health", timeout=10.0)


class RemoteJobManager:
    """The subset of ``JobManager`` that callers outside the manager use.

    Deliberately not a subclass: there is no shared implementation, and
    inheriting would expose methods (``_save_metadata``, the job table) that
    only make sense in the process that owns the subprocesses.
    """

    def create_and_start(self, **kwargs: Any) -> dict[str, Any]:
        return _request("POST", "/jobs", json_body=kwargs)

    def create_and_start_tl(self, **kwargs: Any) -> dict[str, Any]:
        return _request("POST", "/tl-jobs", json_body=kwargs)

    def get_status(self, job_id: str) -> dict[str, Any]:
        return _request("GET", f"/jobs/{quote(job_id, safe='')}")

    def stop_job(self, job_id: str) -> dict[str, Any]:
        return _request("POST", f"/jobs/{quote(job_id, safe='')}/stop")

    def get_top_candidates(
        self,
        job_id: str,
        top_n: int = 10,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        return _request(
            "GET",
            f"/jobs/{quote(job_id, safe='')}/candidates",
            params={"top_n": top_n, "min_score": min_score},
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        return _as_list(_request("GET", "/jobs"), "jobs")

    def get_alerts(self) -> list[dict[str, Any]]:
        return _as_list(_request("GET", "/alerts"), "alerts")


# ---------------------------------------------------------------------------
# Registry and scoring proxies
# ---------------------------------------------------------------------------

def _remote_list(path: str) -> list[dict[str, Any]]:
    result = _request("GET", path, timeout=60.0)
    return result if isinstance(result, list) else []


def remote_list_scoring_components() -> list[dict[str, Any]]:
    return _remote_list("/components")


def remote_list_available_models() -> list[dict[str, Any]]:
    return _remote_list("/models")


def remote_list_transforms() -> list[dict[str, Any]]:
    return _remote_list("/transforms")


def remote_score_molecules(
    smiles: list[str],
    components: Optional[list[dict[str, Any]]] = None,
    aggregation: str = "geometric_mean",
    ttp: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score over HTTP.

    The TTP is sent **resolved and inline**: the store stays on this side, so
    the runner never has to read it. Callers holding only a ``ttp_id`` resolve
    it before calling.
    """
    return _request(
        "POST",
        "/score",
        json_body={
            "smiles": smiles,
            "components": components,
            "aggregation": aggregation,
            "ttp": ttp,
        },
    )
