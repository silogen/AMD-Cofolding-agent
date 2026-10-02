"""MCP tools for synthesis planning and building-block availability (block 07).

Exposes two MCP tools backed by the synthesis-planner service:

    plan_synthesis                      POST /plan
    check_building_block_availability   POST /availability

Both tools keep route evidence and availability evidence separate — per the
PLAN, do not convert route confidence or availability into a REINVENT RL
scalar until the semantics are validated.

The tools return rich JSON strings so the agent can inspect routes, step
counts, leaf in-stock status, and confidence scores before deciding how to
use the information in a campaign.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any

import httpx
from pydantic import Field

logger = logging.getLogger(__name__)

_SYNTHESIS_URL = os.environ.get("SYNTHESIS_SERVICE_URL", "http://synthesis-planner:8086")
_TIMEOUT = 600.0   # route search can take up to 10 min for large batches


def _client() -> httpx.Client:
    return httpx.Client(base_url=_SYNTHESIS_URL, timeout=_TIMEOUT)


def register_tools(mcp) -> None:
    """Register synthesis MCP tools with the MCP server."""

    @mcp.tool()
    def plan_synthesis(
        smiles: Annotated[
            list[str],
            Field(
                description=(
                    "List of candidate SMILES to plan retrosynthetic routes for. "
                    "Each molecule is processed independently; one failure does not "
                    "affect the results for others.  Provide at most 5 SMILES for a "
                    "synchronous response; use submit_synthesis_job for larger batches."
                ),
            ),
        ],
        max_transforms: Annotated[
            int,
            Field(
                default=6,
                description=(
                    "Maximum retrosynthesis tree depth (number of steps). "
                    "Higher values explore more complex routes but take longer. "
                    "Range: 1–12.  Default: 6."
                ),
            ),
        ] = 6,
        iteration_limit: Annotated[
            int,
            Field(
                default=100,
                description=(
                    "Maximum MCTS iterations per molecule.  Increase for harder "
                    "molecules at the cost of longer search time.  Range: 1–2000."
                ),
            ),
        ] = 100,
        time_limit: Annotated[
            float,
            Field(
                default=120.0,
                description=(
                    "Wall-clock time limit per molecule in seconds.  The search "
                    "is terminated gracefully when the limit is reached; partial "
                    "routes found so far are returned.  Range: 1–600 s."
                ),
            ),
        ] = 120.0,
        return_first: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "Stop as soon as the first valid route is found.  "
                    "Useful for quick feasibility checks."
                ),
            ),
        ] = False,
        policy: Annotated[
            str,
            Field(
                default="default",
                description=(
                    "Expansion policy key from the asset manifest (e.g. 'default', "
                    "'uspto').  Use GET /health on the synthesis service to list "
                    "available policy keys."
                ),
            ),
        ] = "default",
        stock: Annotated[
            str,
            Field(
                default="default",
                description=(
                    "Stock catalog key for in-stock leaf checks "
                    "(e.g. 'default', 'emolecules').  Leaves are checked against "
                    "this catalog; unavailable or unconfigured stock returns "
                    "in_stock=null (not False)."
                ),
            ),
        ] = "default",
    ) -> str:
        """Plan retrosynthetic routes for candidate molecules via AiZynthFinder.

        Returns JSON with ``results`` (one entry per input SMILES, order-preserving),
        each containing:
        - ``status``: ``success`` | ``no_route`` | ``timeout`` |
          ``invalid_structure`` | ``unsupported_chemistry`` | ``service_failure``
        - ``routes``: list of route candidates, sorted by score descending, each with:
            - ``n_steps``: number of reaction steps
            - ``score``: route-level model confidence
            - ``reactions``: reaction SMARTS and template IDs per step
            - ``leaves``: starting materials with ``in_stock`` flags
        - ``n_routes``: number of routes found (0 for non-success statuses)
        - ``search_stats``: raw AiZynthFinder statistics dict
        - ``failure_reason``: human-readable explanation for non-success cases

        Also includes ``model_version``, ``templates_version``, ``stock_version``
        for provenance.

        Notes
        -----
        - ``no_route`` is a valid scientific result distinct from ``service_failure``.
        - Stereochemistry and formal charge are preserved — routes are
          enantiomer-specific.
        - Do not use ``score`` as a REINVENT RL scalar without domain validation.
        """
        payload: dict[str, Any] = {
            "smiles": smiles,
            "params": {
                "max_transforms":  max_transforms,
                "iteration_limit": iteration_limit,
                "time_limit":      time_limit,
                "return_first":    return_first,
                "policy":          policy,
                "stock":           stock,
            },
        }
        try:
            with _client() as client:
                resp = client.post("/plan", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": str(exc), "detail": exc.response.text}, indent=2
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)}, indent=2)

    @mcp.tool()
    def check_building_block_availability(
        smiles: Annotated[
            list[str],
            Field(
                description=(
                    "SMILES of building blocks to check against the configured "
                    "stock catalog.  Typically the ``leaves`` from plan_synthesis "
                    "results.  Each entry gets an independent status."
                ),
            ),
        ],
        stock: Annotated[
            str,
            Field(
                default="default",
                description=(
                    "Stock catalog key to query (e.g. 'default', 'emolecules'). "
                    "Use GET /health on the synthesis service to list available keys."
                ),
            ),
        ] = "default",
    ) -> str:
        """Check whether building blocks are in the configured stock catalog.

        Returns JSON with ``results`` (one per input SMILES, order-preserving),
        each containing:
        - ``availability``: ``available`` | ``unavailable`` | ``unknown``
        - ``in_stock``: ``true`` | ``false`` | ``null`` (null = no stock configured)
        - ``stock_source``: catalog key that matched, or null
        - ``canonical_smiles``: RDKit-canonical SMILES used for the lookup

        Also includes ``stock_version`` and aggregate counts.

        Notes
        -----
        - ``unknown`` means no stock catalog is loaded — NOT the same as unavailable.
        - Call ``plan_synthesis`` first to get route leaves, then pass them here.
        - Availability is independent of route planning — a building block can be
          purchasable even if no route to the target was found.
        """
        payload: dict[str, Any] = {"smiles": smiles, "stock": stock}
        try:
            with _client() as client:
                resp = client.post("/availability", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as exc:
            return json.dumps(
                {"error": str(exc), "detail": exc.response.text}, indent=2
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)}, indent=2)

    @mcp.tool()
    def submit_synthesis_job(
        smiles: Annotated[
            list[str],
            Field(
                description=(
                    "Large batch of candidate SMILES (> 5 molecules) to plan "
                    "routes for asynchronously.  Returns a job_id immediately; "
                    "poll get_synthesis_job_status to retrieve results."
                ),
            ),
        ],
        max_transforms: Annotated[int, Field(default=6)] = 6,
        iteration_limit: Annotated[int, Field(default=100)] = 100,
        time_limit: Annotated[float, Field(default=120.0)] = 120.0,
        return_first: Annotated[bool, Field(default=False)] = False,
        policy: Annotated[str, Field(default="default")] = "default",
        stock: Annotated[str, Field(default="default")] = "default",
    ) -> str:
        """Submit a large synthesis planning batch as an asynchronous job.

        Returns ``{"job_id": "...", "status": "queued", "n_molecules": N}``.
        Use ``get_synthesis_job_status(job_id)`` to poll for completion.
        """
        payload: dict[str, Any] = {
            "smiles": smiles,
            "params": {
                "max_transforms":  max_transforms,
                "iteration_limit": iteration_limit,
                "time_limit":      time_limit,
                "return_first":    return_first,
                "policy":          policy,
                "stock":           stock,
            },
        }
        try:
            with _client() as client:
                resp = client.post("/jobs", json=payload)
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": str(exc), "detail": exc.response.text}, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, indent=2)

    @mcp.tool()
    def get_synthesis_job_status(
        job_id: Annotated[
            str,
            Field(description="Job ID returned by submit_synthesis_job."),
        ],
    ) -> str:
        """Poll the status of an async synthesis planning job.

        Returns ``{"job_id": ..., "status": "queued|running|done|failed|cancelled",
        "result": {...}}`` when done.
        """
        try:
            with _client() as client:
                resp = client.get(f"/jobs/{job_id}")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPStatusError as exc:
            return json.dumps({"error": str(exc), "detail": exc.response.text}, indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, indent=2)
