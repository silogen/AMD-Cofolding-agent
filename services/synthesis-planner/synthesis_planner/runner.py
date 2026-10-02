"""AiZynthFinder wrapper — loaded once at service startup.

``SynthesisPlanner`` is a thread-safe wrapper that:
1. Loads the AiZynthFinder model, templates, and stock once at startup.
2. Exposes a ``plan()`` method that runs MCTS for one target SMILES.
3. Maps all AiZynthFinder-internal failures to a typed ``RouteStatus`` enum
   so the service never leaks internal exceptions to API callers.

Thread safety: AiZynthFinder's ``tree_search()`` is stateful (it modifies the
search tree on the ``finder`` instance).  Therefore ``SynthesisPlanner.plan()``
acquires a threading lock per call.  The service-layer concurrency semaphore
ensures at most ``PLANNER_MAX_CONCURRENCY`` calls run simultaneously; the lock
here is a defensive guard for the single underlying finder instance.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from synthesis_planner.routes import RouteCandidate, serialize_routes, route_to_dict
from synthesis_planner.standardize import standardize_smiles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum — maps every AiZynthFinder outcome to a clear status string
# ---------------------------------------------------------------------------

class RouteStatus(str, Enum):
    success             = "success"              # ≥1 route found
    no_route            = "no_route"             # search exhausted, no solution
    timeout             = "timeout"              # time_limit exceeded
    invalid_structure   = "invalid_structure"    # RDKit failed to parse SMILES
    unsupported_chemistry = "unsupported_chemistry"  # query atoms, radicals, …
    service_failure     = "service_failure"      # unexpected exception


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Route-planning result for one candidate SMILES."""
    input_smiles: str
    canonical_smiles: str | None
    status: RouteStatus
    routes: list[dict[str, Any]] = field(default_factory=list)
    n_routes: int = 0
    search_stats: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    warning: str | None = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# SynthesisPlanner
# ---------------------------------------------------------------------------

class SynthesisPlanner:
    """Thin, thread-safe wrapper around AiZynthFinder.

    Parameters
    ----------
    config_dict:
        AiZynthFinder configuration dictionary (built from the asset manifest
        by ``manifest.build_aizynthfinder_config``).
    policy_key:
        Default expansion policy key.
    stock_key:
        Default stock key.
    """

    def __init__(
        self,
        config_dict: dict[str, Any],
        policy_key: str = "default",
        stock_key: str = "default",
    ) -> None:
        self._policy_key = policy_key
        self._stock_key  = stock_key
        self._lock = threading.Lock()
        self._finder: Any = None  # lazy import

        # In v4.4+, policy selection uses select_all() after config load.
        # The policy_key / stock_key are kept for multi-policy scenarios.
        logger.info(
            "Loading AiZynthFinder (policy=%s, stock=%s) …",
            policy_key, stock_key,
        )
        t0 = time.monotonic()
        self._finder = self._load(config_dict, policy_key, stock_key)
        elapsed = time.monotonic() - t0
        logger.info("AiZynthFinder loaded in %.1fs.", elapsed)

    # ── Public API ──────────────────────────────────────────────────────────

    def plan(
        self,
        smiles: str,
        *,
        max_transforms: int = 6,
        iteration_limit: int = 100,
        time_limit: float = 120.0,
        return_first: bool = False,
        policy: str | None = None,
        stock: str | None = None,
    ) -> PlanResult:
        """Run MCTS retrosynthesis for one target SMILES.

        Parameters
        ----------
        smiles:
            Target molecule SMILES.
        max_transforms:
            Maximum depth of the retrosynthesis tree.
        iteration_limit:
            MCTS iteration cap.
        time_limit:
            Wall-clock time limit in seconds.
        return_first:
            Stop at the first valid route found.
        policy / stock:
            Override the default policy/stock keys.

        Returns
        -------
        PlanResult
            Never raises — all failures are captured in ``status`` and
            ``failure_reason``.
        """
        t0 = time.monotonic()

        # 1. Standardize input
        std = standardize_smiles(smiles)
        if std.error:
            status = (
                RouteStatus.unsupported_chemistry
                if ("query" in std.error or "radical" in std.error)
                else RouteStatus.invalid_structure
            )
            return PlanResult(
                input_smiles=smiles,
                canonical_smiles=None,
                status=status,
                failure_reason=std.error,
                elapsed_seconds=time.monotonic() - t0,
            )

        canonical = std.canonical_smiles

        # 2. Run MCTS (exclusive lock on the finder instance)
        with self._lock:
            return self._run_search(
                canonical=canonical,
                input_smiles=smiles,
                max_transforms=max_transforms,
                iteration_limit=iteration_limit,
                time_limit=time_limit,
                return_first=return_first,
                policy=policy or self._policy_key,
                stock=stock or self._stock_key,
                t0=t0,
                warning=std.warning,
            )

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _load(
        config_dict: dict[str, Any],
        policy_key: str,
        stock_key: str,
    ) -> Any:
        """Instantiate and configure AiZynthFinder from a config dict.

        IMPORTANT: ``Configuration.from_dict()`` pops keys from its input dict.
        Pass a deep-copy to ``AiZynthFinder`` so the original config_dict is
        not mutated — otherwise a second call would see an empty dict.
        """
        import copy  # noqa: PLC0415

        try:
            from aizynthfinder.aizynthfinder import AiZynthFinder          # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "AiZynthFinder is not installed. "
                "Install it with: pip install aizynthfinder"
            ) from exc

        # Deep-copy because Configuration.from_dict() pops keys in-place.
        finder = AiZynthFinder(configdict=copy.deepcopy(config_dict))

        # v4.4+: select_all() activates all loaded policies at once.
        # single-policy setups (the common case) work fine with this.
        try:
            finder.expansion_policy.select_all()
        except Exception as exc:
            raise RuntimeError(
                f"Could not select expansion policy: {exc}. "
                "Ensure the policy model and template files are present and valid."
            ) from exc

        try:
            finder.stock.select([stock_key])
        except (KeyError, Exception):
            # Stock may not be mounted; warn but do not fail startup
            logger.warning(
                "Stock key %r could not be selected — availability checks will "
                "return 'unknown'.  Mount the stock catalog and restart.",
                stock_key,
            )

        return finder

    def _run_search(
        self,
        *,
        canonical: str,
        input_smiles: str,
        max_transforms: int,
        iteration_limit: int,
        time_limit: float,
        return_first: bool,
        policy: str,
        stock: str,
        t0: float,
        warning: str | None,
    ) -> PlanResult:
        """Execute the MCTS search (must be called under self._lock)."""
        finder = self._finder

        # Apply per-request search parameters
        finder.config.search.max_transforms   = max_transforms
        finder.config.search.iteration_limit  = iteration_limit
        finder.config.search.time_limit       = time_limit
        finder.config.search.return_first     = return_first

        # Policy/stock switching per-request (no-op for single-policy setups)
        # select_all() is always safe; stock switching only if key differs.
        try:
            finder.expansion_policy.select_all()
            if stock != self._stock_key:
                finder.stock.select([stock])
        except Exception as exc:
            return PlanResult(
                input_smiles=input_smiles,
                canonical_smiles=canonical,
                status=RouteStatus.service_failure,
                failure_reason=f"Policy/stock selection failed: {exc}",
                elapsed_seconds=time.monotonic() - t0,
                warning=warning,
            )

        try:
            finder.target_smiles = canonical
            finder.tree_search()
            finder.build_routes()
        except Exception as exc:
            logger.exception("AiZynthFinder search failed for %s", canonical)
            return PlanResult(
                input_smiles=input_smiles,
                canonical_smiles=canonical,
                status=RouteStatus.service_failure,
                failure_reason=str(exc),
                elapsed_seconds=time.monotonic() - t0,
                warning=warning,
            )

        # ── Parse stats ─────────────────────────────────────────────────────
        try:
            stats = finder.extract_statistics() or {}
        except Exception:
            stats = {}

        # ── Determine outcome ────────────────────────────────────────────────
        routes = finder.routes
        n_routes = len(list(routes)) if routes else 0

        if n_routes == 0:
            # Distinguish timeout from genuine "no route".
            # v4.4 uses search_time vs time_limit; older versions had is_time_limit_reached.
            is_timeout = stats.get("is_time_limit_reached", False) or (
                stats.get("search_time", 0) >= time_limit * 0.95
            )
            status = RouteStatus.timeout if is_timeout else RouteStatus.no_route
            return PlanResult(
                input_smiles=input_smiles,
                canonical_smiles=canonical,
                status=status,
                n_routes=0,
                search_stats=stats,
                failure_reason=("Time limit reached during search." if is_timeout else None),
                elapsed_seconds=time.monotonic() - t0,
                warning=warning,
            )

        # ── Serialize routes ─────────────────────────────────────────────────
        stock_obj = finder.stock if finder.stock else None
        candidates: list[RouteCandidate] = serialize_routes(
            routes, stock=stock_obj, max_routes=20,
        )

        return PlanResult(
            input_smiles=input_smiles,
            canonical_smiles=canonical,
            status=RouteStatus.success,
            routes=[route_to_dict(r) for r in candidates],
            n_routes=len(candidates),
            search_stats=stats,
            elapsed_seconds=time.monotonic() - t0,
            warning=warning,
        )
