"""ExternalRESTScorer -- REINVENT4 scoring plugin for external REST services.

Calls external FastAPI services (Boltz2, PoseBusters, ADMET, RetroDMTA,
interaction-profiler) via HTTP POST and converts responses into REINVENT4's
ComponentResults format. Follows the same plugin registration pattern as
comp_generic_rest.py and comp_external_process.py.

This module lives under a ``reinvent_plugins/components/`` namespace-package
layout (no ``__init__.py`` in ``reinvent_plugins/`` or ``components/``) so that
placing this directory's parent on ``sys.path`` makes REINVENT4's
``reinvent_plugins.components`` namespace scan discover it NATIVELY -- in every
process, including the RL job subprocess -- without any fragile runtime
``__path__.append()``.
"""

from __future__ import annotations

__all__ = ["ExternalRESTScorer"]

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import requests
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.components.component_results import ComponentResults

logger = logging.getLogger("reinvent")

_CALL_LOG_DIR = Path("/data/reinvent4_mcp/logs")
_CALL_LOG_FILE = _CALL_LOG_DIR / "external_scoring.jsonl"

# How long to wait for a blocking /score call. A queue-fronted gateway can hold
# a batch far longer than one prediction takes, and giving up turns a request
# that would have succeeded into NaN for the whole batch -- which both poisons
# the RL signal and reports a component failure that never happened. Tunable
# per deployment rather than hardcoded.
DEFAULT_TIMEOUT_SECONDS = 1800.0
TIMEOUT_ENV = "EXTERNAL_SCORER_TIMEOUT_SECONDS"

# Submit-then-poll settings for services that answer 202 + {task_id}. Polling
# costs nothing while queued and removes the timeout question entirely: the
# wait is bounded by the deadline, not by one HTTP connection staying up.
POLL_INTERVAL_SECONDS = 5.0
POLL_MAX_INTERVAL_SECONDS = 60.0
POLL_BACKOFF = 1.5


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%r must be positive; using %s", name, raw, default)
        return default
    return value


def _log_call(record: dict[str, Any]) -> None:
    """Append a JSON record to the external scoring log file."""
    try:
        _CALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CALL_LOG_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write call log", exc_info=True)


@add_tag("__parameters")
@dataclass(config=ConfigDict(extra="allow"))
class Parameters:
    """Parameters for the ExternalRESTScorer component.

    All fields are lists (one element per endpoint) per REINVENT4 convention.
    Single-endpoint usage: each list has exactly one element.

    Service-specific params (recycling_steps, diffusion_samples, etc.) from
    the TTP's ``service_params`` are flattened into the endpoint params by the
    mapper and accepted here via ``extra="allow"``.
    """

    service_url: List[str]
    service: List[str]
    protein_sequence: List[Optional[str]] = None
    msa_path: List[Optional[str]] = None
    protein_pdb_path: List[Optional[str]] = None
    depends_on: List[Optional[str]] = None
    component_name: List[Optional[str]] = None
    affinity_mode: List[str] = None
    failure_score: List[Optional[float]] = None
    # Seconds to wait for a result. Falls back to
    # EXTERNAL_SCORER_TIMEOUT_SECONDS, then to DEFAULT_TIMEOUT_SECONDS.
    timeout: List[Optional[float]] = None
    # Submit with ``?mode=async`` and poll ``/result/{task_id}``. Needs a
    # gateway that answers 202; ignored by the plain scorer services.
    async_poll: List[Optional[bool]] = None


@add_tag("__component")
class ExternalRESTScorer:
    """Score molecules by calling an external REST service.

    POSTs ``{smiles, params}`` to the configured ``service_url`` and
    converts the JSON response into ``ComponentResults``.  On any failure
    (network, timeout, HTTP error), returns all-NaN scores so the RL loop
    can continue with adjusted aggregation weights.
    """

    _INTERNAL_PARAMS = frozenset({
        "service_url", "service", "depends_on", "component_name",
        "failure_score", "timeout", "async_poll",
    })
    _artifact_registry: dict[str, dict] = {}

    def __init__(self, params: Parameters):
        self.service_url = params.service_url[0]
        self.service = params.service[0]

        timeout_list = getattr(params, "timeout", None)
        configured = (
            timeout_list[0]
            if isinstance(timeout_list, list) and timeout_list
            else None
        )
        self.timeout = float(configured) if configured else _env_float(
            TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS
        )

        async_list = getattr(params, "async_poll", None)
        self.async_poll = bool(
            async_list[0]
            if isinstance(async_list, list) and async_list
            else False
        )

        depends_on_list = getattr(params, "depends_on", None)
        self.depends_on = (
            depends_on_list[0]
            if isinstance(depends_on_list, list) and depends_on_list
            else None
        )

        failure_score_list = getattr(params, "failure_score", None)
        self.failure_score = (
            failure_score_list[0]
            if isinstance(failure_score_list, list) and failure_score_list
            else None
        )

        component_name_list = getattr(params, "component_name", None)
        self.component_name = (
            component_name_list[0]
            if isinstance(component_name_list, list) and component_name_list
            else None
        )

        self.request_params: dict = {}
        for key, values in vars(params).items():
            if key in self._INTERNAL_PARAMS or key.startswith("_"):
                continue
            if isinstance(values, list) and values:
                val = values[0]
                if val is not None:
                    self.request_params[key] = val

        extra = getattr(params, "__pydantic_extra__", None) or {}
        for key, values in extra.items():
            if key in self._INTERNAL_PARAMS or key.startswith("_"):
                continue
            if isinstance(values, list) and values:
                val = values[0]
            else:
                val = values
            if val is not None:
                self.request_params[key] = val

        self.last_artifacts: dict | None = None

        # REINVENT4 only logs the component class, so a TTP with several
        # external objectives produces a run of identical "Creating scoring
        # component ExternalRESTScorer" lines. Announce which one this is.
        logger.info(
            "ExternalRESTScorer[%s] -> service=%s url=%s timeout=%.0fs%s%s%s",
            self.component_name or "unnamed",
            self.service,
            self.service_url,
            self.timeout,
            " async-poll" if self.async_poll else "",
            f" depends_on={self.depends_on}" if self.depends_on else "",
            self._selector_summary(),
        )

    def _result_url(self, task_id: str) -> str:
        """Where to poll for a task submitted to ``service_url``."""
        base = self.service_url.split("?", 1)[0]
        if base.endswith("/score"):
            base = base[: -len("/score")]
        return f"{base.rstrip('/')}/result/{task_id}"

    def _submit_and_poll(self, payload: dict, record: dict[str, Any]):
        """Submit with ``mode=async``, then poll until the result is ready.

        The point is that the wait is bounded by ``self.timeout`` overall rather
        than by one HTTP connection staying up, and that a ``202`` means queued
        rather than broken. A gateway that does not speak this contract answers
        the submit normally, and that response is used as-is so the
        misconfiguration is visible instead of silent.
        """
        separator = "&" if "?" in self.service_url else "?"
        submit_url = f"{self.service_url}{separator}mode=async"

        deadline = time.monotonic() + self.timeout
        response = requests.post(
            submit_url, json=payload, timeout=min(self.timeout, 120.0)
        )
        record["status_code"] = response.status_code
        if response.status_code != 202:
            return response

        task_id = (response.json() or {}).get("task_id")
        if not task_id:
            raise RuntimeError("service accepted the batch but returned no task_id")
        record["task_id"] = task_id
        logger.info("%s accepted batch as task %s; polling", self.service, task_id)

        interval = POLL_INTERVAL_SECONDS
        polls = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"no result for task {task_id} within {self.timeout:.0f}s"
                )
            time.sleep(min(interval, remaining))
            interval = min(interval * POLL_BACKOFF, POLL_MAX_INTERVAL_SECONDS)
            polls += 1

            poll = requests.get(self._result_url(task_id), timeout=60)
            record["status_code"] = poll.status_code
            record["polls"] = polls
            if poll.status_code == 202:
                continue
            return poll

    def _selector_summary(self) -> str:
        """Render the params that decide what this instance measures.

        Two ADMET objectives differ only by ``endpoint``, so the selector is
        the only thing that makes the log line identifying.
        """
        selectors = [
            f"{key}={self.request_params[key]}"
            for key in ("endpoint", "target", "interaction_type", "affinity_mode", "mode")
            if key in self.request_params
        ]
        return f" [{', '.join(selectors)}]" if selectors else ""

    def __call__(self, smilies: List[str]) -> ComponentResults:
        # Clear own registry entry so stale artifacts are not consumed
        # by downstream components if this call fails.
        if self.component_name:
            type(self)._artifact_registry.pop(self.component_name, None)

        # Inject artifacts from upstream component (pipeline dependency).
        # This handles the RL optimisation path where PipelineAwareScorer
        # is not available; in the score_molecules path the wrapper may
        # have already set these, which is fine -- same values.
        if self.depends_on:
            # Clear any previously injected per-call artifacts so we never reuse stale structures.
            self.request_params.pop("structure_paths", None)
            # Only clear protein_pdb_path if we didn't get an explicit one from the user/config.
            if not hasattr(self, "_protein_pdb_path_explicit"):
                self._protein_pdb_path_explicit = "protein_pdb_path" in self.request_params
            if not self._protein_pdb_path_explicit:
                self.request_params.pop("protein_pdb_path", None)

            upstream_artifacts = type(self)._artifact_registry.get(self.depends_on)
            if upstream_artifacts:
                structures = upstream_artifacts.get("structures") or []
                valid = [s for s in structures if s]
                if valid:
                    # Send the FULL, index-aligned list (None for molecules whose
                    # upstream structure failed) -- NOT a compacted 'valid' list.
                    # Downstream services pair structure_paths[i] with smiles[i]:
                    # PoseBusters enforces len(structure_paths) == len(smiles) and
                    # interaction-profiler indexes structs[i], so dropping the None
                    # holes silently misassigns poses (or 400s) on partial failures.
                    self.request_params["structure_paths"] = list(structures)
                    if "protein_pdb_path" not in self.request_params:
                        self.request_params["protein_pdb_path"] = valid[0]
                    logger.info(
                        "Injected %d/%d structure paths from '%s' into %s",
                        len(valid), len(structures), self.depends_on, self.service,
                    )
                else:
                    logger.warning(
                        "%s depends on '%s' but upstream returned no structure artifacts",
                        self.service, self.depends_on,
                    )
            else:
                logger.warning(
                    "%s depends on '%s' but no upstream artifacts found",
                    self.service, self.depends_on,
                )

        payload = {"smiles": smilies, "params": self.request_params}
        logger.info(
            "Calling %s at %s with %d SMILES, params keys: %s",
            self.service, self.service_url, len(smilies),
            list(self.request_params.keys()),
        )

        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service,
            "url": self.service_url,
            "request": payload,
        }

        t0 = time.monotonic()
        try:
            if self.async_poll:
                response = self._submit_and_poll(payload, record)
            else:
                response = requests.post(
                    self.service_url,
                    json=payload,
                    timeout=self.timeout,
                )
            elapsed = time.monotonic() - t0
            record["status_code"] = response.status_code
            record["elapsed_seconds"] = round(elapsed, 2)

            try:
                data = response.json()
            except Exception:
                data = {"_raw_text": response.text[:2000]}

            record["response"] = data

            if not response.ok:
                logger.error(
                    "%s returned %d: %s",
                    self.service, response.status_code,
                    response.text[:500],
                )
                record["error"] = f"HTTP {response.status_code}"
                _log_call(record)
                response.raise_for_status()

        except Exception:
            elapsed = time.monotonic() - t0
            record.setdefault("elapsed_seconds", round(elapsed, 2))
            record.setdefault("error", "request_failed")
            _log_call(record)

            logger.exception(
                "External service %s at %s failed", self.service, self.service_url
            )
            fill = self.failure_score if self.failure_score is not None else np.nan
            scores = np.full(len(smilies), fill, dtype=float)
            return ComponentResults(scores=[scores])

        _miss = self.failure_score if self.failure_score is not None else np.nan
        raw_scores = data.get("scores", [])
        scores = np.array(
            [s if s is not None else _miss for s in raw_scores],
            dtype=float,
        )

        self.last_artifacts = data.get("artifacts")
        if self.last_artifacts:
            if self.component_name:
                type(self)._artifact_registry[self.component_name] = (
                    self.last_artifacts
                )
            structures = self.last_artifacts.get("structures", [])
            logger.info(
                "%s returned %d scores, %d structure artifacts",
                self.service, len(raw_scores), len(structures),
            )

        _log_call(record)

        # REINVENT's ComponentResults expects every metadata value to be a
        # per-molecule list (it does zip(*metadata.values())). Services also
        # return scalar context (endpoint, target, window, unit, cut_date);
        # keep only list-valued entries of the right length so a scalar (e.g.
        # window=11) can't raise "'int' object is not iterable" and abort the
        # whole RL scoring pipeline.
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            n = len(scores)
            metadata = {
                k: v for k, v in metadata.items()
                if isinstance(v, list) and len(v) == n
            } or None
        else:
            metadata = None
        return ComponentResults(scores=[scores], metadata=metadata)
