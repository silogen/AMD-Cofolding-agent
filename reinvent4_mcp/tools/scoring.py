# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Scoring tool: score a list of SMILES using REINVENT4 scoring components.

Supports two modes:
1. Direct component configuration (Phase 1)
2. TTP-based scoring (Phase 2/3) -- native components from a TTP are
   extracted and scored.  When external services (Boltz, PoseBusters) are
   healthy, they are included via the ``PipelineAwareScorer``; otherwise
   only native components are scored with a note.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reinvent4_mcp.utils import ensure_reinvent4_importable

logger = logging.getLogger(__name__)

# External scorers append one JSON record per call here (shared /data mount).
# Reading it is the reliable way to recover produced structure paths, since the
# component instances run in a multiprocessing pool and their in-memory
# ``last_artifacts`` never reach this parent process.
_EXTERNAL_SCORING_LOG = Path("/data/reinvent4_mcp/logs/external_scoring.jsonl")


def _ext_log_offset() -> int:
    try:
        return _EXTERNAL_SCORING_LOG.stat().st_size
    except OSError:
        return 0


def _structures_logged_since(offset: int) -> list[str]:
    """Return structure paths recorded by external scorers after ``offset``."""
    out: list[str] = []
    try:
        with open(_EXTERNAL_SCORING_LOG) as fh:
            fh.seek(offset)
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                arts = (rec.get("response") or {}).get("artifacts") or {}
                for s in (arts.get("structures") or []):
                    if s and s not in out:
                        out.append(s)
    except OSError:
        pass
    return out


def _normalize_endpoint_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize component params to REINVENT4's per-endpoint format.

    Each endpoint's ``params`` holds plain values; REINVENT4's
    ``collect_params`` gathers them across endpoints into the lists its
    component parameters expect, so a list here becomes a list of lists.
    Since MCP always creates single-endpoint components, this function
    fixes common agent formatting mistakes:

    - Plain scalar  ``3``         -> ``3``  (unchanged)
    - Wrapped       ``[3]``       -> ``3``
    - Double-nested ``[[True]]``  -> ``True``
    - List param    ``["CCO"]``   -> ``["CCO"]``  (unchanged)
    - Nested list   ``[["CCO"]]`` -> ``["CCO"]``
    """
    result: dict[str, Any] = {}
    for key, value in params.items():
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (int, float, bool)):
            value = value[0]
        result[key] = value
    return result


def _build_scorer_config(
    components: list[dict[str, Any]],
    aggregation: str = "geometric_mean",
) -> dict:
    """Translate the MCP tool's component list into a REINVENT4 Scorer config.

    The REINVENT4 ``Scorer`` expects a dict of the form::

        {
            "type": "geometric_mean",
            "component": [
                {"ComponentType": {"endpoint": [{name, weight, transform, params}]}}
            ]
        }
    """

    scorer_components = []

    for comp in components:
        comp_type = comp["type"]
        endpoint: dict[str, Any] = {
            "name": comp.get("name", comp_type),
            "weight": comp.get("weight", 1.0),
        }

        transform = comp.get("transform")
        if transform:
            endpoint["transform"] = transform

        params = comp.get("params")
        if params:
            endpoint["params"] = _normalize_endpoint_params(params)

        scorer_components.append({
            comp_type: {
                "endpoint": [endpoint],
            }
        })

    return {
        "type": aggregation,
        "component": scorer_components,
    }


def _extract_raw_scores(
    comp_result,
    smiles_list: list[str],
) -> list[list[float]]:
    """Extract raw (untransformed) scores from a TransformResults object.

    ``comp_result.component_result`` is a ``SmilesAssociatedComponentResults``
    whose ``.fetch_scores(smiles, transpose=True)`` returns scores with
    dimensions ``[n_endpoints, n_smiles]`` -- matching ``transformed_scores``.
    """

    try:
        raw = comp_result.component_result.fetch_scores(
            smiles_list, transpose=True
        )
        return [[float(v) for v in endpoint] for endpoint in raw]
    except Exception:
        return []


#: Per-molecule values from external services that a reader needs to judge a
#: score: Boltz confidence and affinity (the UI only shows an affinity when the
#: confidence passes), the structure it came from, PoseBusters check counts,
#: and ADMET spread and applicability domain.
_DETAIL_KEYS = (
    "affinity_pred_value",
    "affinity_probability_binary",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "confidence_score",
    "ligand_iptm",
    "structure_path",
    "n_checks_passed",
    "n_checks_total",
    "std",
    "ad_pass",
)


def _component_details(md: dict[str, list], idx: int) -> dict[str, Any]:
    """The detail values of one molecule from one component's metadata."""
    details: dict[str, Any] = {}
    for key in _DETAIL_KEYS:
        vals = md.get(key)
        if vals is None or idx >= len(vals):
            continue
        value = vals[idx]
        if isinstance(value, float) and np.isnan(value):
            continue
        if value is not None and value != "None":
            details[key] = value
    # PoseBusters reports one boolean per check; name the ones that failed.
    if "n_checks_total" in md:
        failed = [
            key for key, vals in md.items()
            if key not in _DETAIL_KEYS and idx < len(vals) and vals[idx] is False
        ]
        if failed:
            details["failed_checks"] = failed
    return details


def _format_results(
    score_results,
    smiles_list: list[str],
) -> dict[str, Any]:
    """Format REINVENT4 ``ScoreResults`` into the MCP output structure.

    Returns per-molecule component breakdowns (raw + transformed) and
    summary statistics.
    """

    molecules: list[dict[str, Any]] = []

    # Track NaN counts per component across all molecules so the summary can
    # report which objectives silently failed.
    component_failures: dict[str, int] = {}
    component_failure_reasons: dict[str, str] = {}
    n_components_total = 0

    # Per-component, per-SMILES metadata (e.g. skip_reason / msa_error from an
    # external service) so a NaN can carry an explicit, actionable reason
    # instead of being opaque. Keyed by the component-result object identity.
    component_metadata: dict[int, dict[str, list]] = {}
    for comp_result in score_results.completed_components:
        try:
            component_metadata[id(comp_result)] = (
                comp_result.component_result.fetch_metadata(smiles_list)
            )
        except Exception:
            component_metadata[id(comp_result)] = {}

    def _failure_reason(comp_result, idx: int) -> str | None:
        md = component_metadata.get(id(comp_result), {})
        for key in ("skip_reason", "msa_error", "error"):
            vals = md.get(key)
            if vals and idx < len(vals) and vals[idx] not in (None, "None", ""):
                return str(vals[idx])
        return None

    # Advisory notes (e.g. reduced-confidence MSA-free cofold) that should be
    # surfaced even when the component did NOT fail.
    advisory_notes: set[str] = set()
    for md in component_metadata.values():
        for vals in (md.get("msa_note"),):
            for v in vals or []:
                if v and str(v) not in ("None", ""):
                    advisory_notes.add(str(v))

    for i, smiles in enumerate(smiles_list):
        mol_data: dict[str, Any] = {
            "smiles": smiles,
            "total_score": float(score_results.total_scores[i]),
            "components": {},
        }
        failed_components: list[str] = []

        for comp_result in score_results.completed_components:
            raw_scores = _extract_raw_scores(comp_result, smiles_list)

            for j, comp_name in enumerate(comp_result.component_names):
                transformed = float(comp_result.transformed_scores[j][i])

                if j < len(raw_scores) and i < len(raw_scores[j]):
                    raw = raw_scores[j][i]
                else:
                    raw = transformed

                transform_type = None
                if j < len(comp_result.transform_types):
                    transform_type = comp_result.transform_types[j]

                mol_data["components"][comp_name] = {
                    "transformed": transformed,
                    "raw": raw,
                    "transform_type": transform_type,
                }
                details = _component_details(component_metadata.get(id(comp_result), {}), i)
                if details:
                    mol_data["components"][comp_name]["details"] = details

                if i == 0:
                    n_components_total += 1
                if np.isnan(transformed) or (
                    isinstance(raw, float) and np.isnan(raw)
                ):
                    failed_components.append(comp_name)
                    component_failures[comp_name] = (
                        component_failures.get(comp_name, 0) + 1
                    )
                    reason = _failure_reason(comp_result, i)
                    if reason:
                        mol_data.setdefault("failure_reasons", {})[comp_name] = reason
                        component_failure_reasons.setdefault(comp_name, reason)

        # Only surface the key when something actually failed, so existing
        # consumers see no change on the happy path.
        if failed_components:
            mol_data["failed_components"] = failed_components
            if len(failed_components) == n_components_total:
                mol_data["warning"] = (
                    "All scored objectives failed (NaN) for this molecule; its "
                    "total_score is an artifact of empty aggregation, not a real "
                    "score."
                )

        molecules.append(mol_data)

    total_scores = score_results.total_scores
    valid_mask = ~np.isnan(total_scores)
    n_valid = int(valid_mask.sum())

    per_component_stats: dict[str, dict[str, Any]] = {}
    for comp_result in score_results.completed_components:
        raw_scores = _extract_raw_scores(comp_result, smiles_list)
        for j, comp_name in enumerate(comp_result.component_names):
            t_scores = comp_result.transformed_scores[j]
            stats: dict[str, Any] = {
                "mean_transformed": float(np.nanmean(t_scores)),
            }
            if j < len(raw_scores):
                raw_arr = np.array(raw_scores[j], dtype=float)
                stats["mean_raw"] = float(np.nanmean(raw_arr))
                stats["min_raw"] = float(np.nanmin(raw_arr))
                stats["max_raw"] = float(np.nanmax(raw_arr))
            per_component_stats[comp_name] = stats

    summary: dict[str, Any] = {
        "n_molecules": len(molecules),
        "n_valid": n_valid,
        "mean_total_score": float(np.nanmean(total_scores)) if n_valid > 0 else None,
        "best_total_score": float(np.nanmax(total_scores)) if n_valid > 0 else None,
        "per_component": per_component_stats,
    }

    # Surface failed objectives so a NaN component cannot masquerade as a
    # perfect score. Keys are only added when there are failures, keeping the
    # happy-path output unchanged.
    if component_failures:
        summary["component_failures"] = component_failures
        n = len(molecules)
        warnings = []
        for comp, cnt in sorted(component_failures.items()):
            msg = (
                f"{comp}: returned NaN for {cnt}/{n} molecule(s); its "
                "contribution was dropped from aggregation, so totals may be "
                "inflated."
            )
            reason = component_failure_reasons.get(comp)
            if reason:
                msg += f" Reason: {reason}"
            warnings.append(msg)
        summary["warnings"] = warnings
        if component_failure_reasons:
            summary["component_failure_reasons"] = component_failure_reasons

    if advisory_notes:
        summary["notes"] = sorted(advisory_notes)

    return {"molecules": molecules, "summary": summary}


def _annotate_component_sources(
    result: dict[str, Any],
    ttp,
    external_detail: list[dict[str, Any]],
) -> None:
    """Tag each scored component in ``result`` with where its score came from.

    The per-component stats are keyed by objective name alone, which says
    nothing about whether a number came from a local descriptor or from a
    remote service -- and every external objective is the same
    ``ExternalRESTScorer`` class. Attaching the resolved source makes each
    row self-describing, so scores can be interpreted (and failures
    attributed) without cross-referencing the TTP.
    """

    sources: dict[str, dict[str, Any]] = {}

    if ttp.pre_scoring:
        for comp in ttp.pre_scoring.components:
            sources[comp.name] = {
                "type": "native",
                "component": comp.component,
                "stage": "pre_scoring",
            }

    for obj in ttp.scoring.objectives:
        if obj.type == "native":
            sources[obj.name] = {"type": "native", "component": obj.component}

    for detail in external_detail:
        source = {k: v for k, v in detail.items() if k not in ("name", "weight")}
        source["type"] = "external"
        sources[detail["name"]] = source

    for name, stats in result.get("summary", {}).get("per_component", {}).items():
        source = sources.get(name)
        if source:
            stats["source"] = source


def score_molecules(
    smiles: list[str],
    components: Optional[list[dict[str, Any]]] = None,
    aggregation: str = "geometric_mean",
    ttp_id: Optional[str] = None,
    ttp: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score a list of SMILES with REINVENT4 scoring components.

    This is a standalone scoring tool -- it evaluates a given set of SMILES
    without any molecule generation.  Useful for evaluating candidate
    molecules against a property profile before or after optimization.

    Two modes of operation:

    1. **Direct** -- pass ``components`` (list of component dicts).
    2. **TTP-based** -- pass ``ttp_id`` (stored TTP) or ``ttp`` (inline
       TTP dict).  When external services (Boltz, PoseBusters) are
       healthy, all objectives are scored using pipeline-aware execution
       (Boltz structure artifacts flow to PoseBusters).  When services
       are unreachable, only native components are scored with a note.

    Args:
        smiles: SMILES strings to score.
        components: Component configs (direct mode).  Each dict must
            have ``type``.  Optional: ``name``, ``weight``, ``transform``,
            ``params``.
        aggregation: Aggregation method (ignored in TTP mode -- the TTP's
            ``scoring.aggregation`` is used instead).
        ttp_id: ID of a stored TTP to use.
        ttp: Inline TTP dict.

    Returns:
        Dict with ``molecules`` (per-molecule breakdown), ``summary``,
        and ``scoring_mode`` (``"full"`` or ``"native_only"`` in TTP mode).
        In TTP mode, ``mapper_summary["external_components"]`` and each
        ``summary["per_component"][name]["source"]`` name the service, URL
        and selector behind every ``ExternalRESTScorer`` objective.
    """

    if not smiles:
        return {
            "molecules": [],
            "summary": {"n_molecules": 0, "n_valid": 0, "per_component": {}},
        }

    if ttp_id or ttp:
        return _score_with_ttp(smiles, ttp_id=ttp_id, ttp_data=ttp)

    if not components:
        return {
            "error": (
                "No scoring components specified. Provide 'components' "
                "(direct mode) or 'ttp_id'/'ttp' (TTP mode). "
                "Use list_scoring_components to discover available components."
            ),
        }

    return _score_with_components(smiles, components, aggregation)


def _score_with_components(
    smiles: list[str],
    components: list[dict[str, Any]],
    aggregation: str,
) -> dict[str, Any]:
    """Score using a direct component list (Phase 1 path)."""

    ensure_reinvent4_importable()
    from reinvent.scoring.scorer import Scorer

    scorer_config = _build_scorer_config(components, aggregation)

    try:
        scorer = Scorer(scorer_config)
    except (RuntimeError, KeyError) as exc:
        return {
            "error": f"Failed to create scorer: {exc}",
            "hint": "Use list_scoring_components to see available component names.",
        }

    invalid_mask = np.ones(len(smiles), dtype=bool)
    duplicate_mask = np.ones(len(smiles), dtype=bool)

    try:
        results = scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    except Exception as exc:
        logger.exception("Scoring failed")
        return {"error": f"Scoring failed: {exc}"}

    return _format_results(results, smiles)


def _score_with_ttp(
    smiles: list[str],
    ttp_id: Optional[str] = None,
    ttp_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score using a TTP with automatic external service detection.

    When the TTP includes external objectives (Boltz, PoseBusters) and
    those services are healthy, uses the full ``ttp_to_scorer_config``
    with the ``PipelineAwareScorer`` for artifact passing.  Otherwise
    falls back to native-only scoring with a note about skipped services.
    """

    ensure_reinvent4_importable()

    from reinvent4_mcp.tools.ttp import resolve_ttp
    from reinvent4_mcp.ttp.schema import TTP as TTPModel

    resolved = resolve_ttp(ttp_id=ttp_id, ttp_data=ttp_data)
    if isinstance(resolved, dict):
        return resolved  # error dict

    assert isinstance(resolved, TTPModel)

    use_external = False
    external_status: dict[str, Any] = {}

    if resolved.has_external_objectives:
        external_status = _check_required_services(resolved)
        use_external = bool(external_status) and all(
            info["status"] == "ok" for info in external_status.values()
        )

    if use_external:
        return _score_full(smiles, resolved, external_status)
    return _score_native_only(smiles, resolved, external_status)


def _check_required_services(ttp) -> dict[str, Any]:
    """Health-check only the external services that this TTP actually needs.

    Returns a status entry for every required service.  Services not
    present in the process config get status ``"not_configured"``.
    """

    from reinvent4_mcp.tools.external import list_external_services

    all_services = list_external_services()

    required = {
        obj.service
        for obj in ttp.scoring.objectives
        if obj.type == "external" and obj.service
    }

    result: dict[str, Any] = {}
    for name in sorted(required):
        if name in all_services:
            result[name] = all_services[name]
        else:
            result[name] = {
                "url": "not configured",
                "health_url": "not configured",
                "status": "not_configured",
            }
    return result


def _score_full(
    smiles: list[str],
    ttp,
    external_status: dict[str, Any],
) -> dict[str, Any]:
    """Score with all components (native + external) via PipelineAwareScorer."""

    from reinvent4_mcp.scoring_pipeline import PipelineAwareScorer
    from reinvent4_mcp.ttp.mapper import get_mapper_summary, ttp_to_scorer_config

    scorer_config = ttp_to_scorer_config(ttp)
    if not scorer_config["component"]:
        return {"error": "TTP produced no scoring components."}

    try:
        scorer = PipelineAwareScorer(scorer_config, ttp)
    except (RuntimeError, KeyError) as exc:
        logger.warning("Full scorer creation failed, falling back to native-only: %s", exc)
        return _score_native_only(
            smiles, ttp, external_status, fallback_error=str(exc),
        )

    invalid_mask = np.ones(len(smiles), dtype=bool)
    duplicate_mask = np.ones(len(smiles), dtype=bool)

    log_offset = _ext_log_offset()
    try:
        results = scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    except Exception as exc:
        logger.exception("Full TTP scoring failed, falling back to native-only")
        return _score_native_only(
            smiles, ttp, external_status, fallback_error=str(exc),
        )

    result = _format_results(results, smiles)

    summary = get_mapper_summary(ttp, scorer_config)
    result["scoring_mode"] = "full"
    result["external_services"] = external_status
    result["ttp_id"] = ttp.id
    result["ttp_version"] = ttp.version
    result["mapper_summary"] = summary
    _annotate_component_sources(result, ttp, summary.get("external_components", []))

    # Surface any structures produced this run (e.g. Boltz cofolds) so the agent
    # can profile the exact deposited pose via profile_interactions instead of
    # guessing among pre-existing files. The log is authoritative (component
    # instances run in a pool, so their in-memory artifacts don't reach here);
    # fall back to the instance collector if the log is unavailable.
    structures = _structures_logged_since(log_offset)
    if not structures:
        try:
            structures = scorer.collected_structures()
        except Exception:
            structures = []
    if structures:
        result["structures"] = structures
    return result


def _score_native_only(
    smiles: list[str],
    ttp,
    external_status: dict[str, Any],
    fallback_error: Optional[str] = None,
) -> dict[str, Any]:
    """Score with native components only, noting any skipped external objectives."""

    from reinvent.scoring.scorer import Scorer

    from reinvent4_mcp.ttp.mapper import get_mapper_summary, ttp_to_native_scorer_config

    scorer_config = ttp_to_native_scorer_config(ttp)
    if not scorer_config["component"]:
        return {
            "error": (
                "TTP has no native scoring components. "
                "All objectives are external (Boltz/PoseBusters) which "
                "require running external services."
            ),
            "external_services": external_status,
        }

    try:
        scorer = Scorer(scorer_config)
    except (RuntimeError, KeyError) as exc:
        return {
            "error": f"Failed to create scorer from TTP: {exc}",
            "hint": "Check component names in the TTP against list_scoring_components.",
        }

    invalid_mask = np.ones(len(smiles), dtype=bool)
    duplicate_mask = np.ones(len(smiles), dtype=bool)

    try:
        results = scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    except Exception as exc:
        logger.exception("TTP-based scoring failed")
        return {"error": f"Scoring failed: {exc}"}

    result = _format_results(results, smiles)

    summary = get_mapper_summary(ttp)
    result["scoring_mode"] = "native_only"

    if ttp.has_external_objectives:
        note_parts = []
        skipped = summary.get("external_objective_names", [])
        note_parts.append(
            f"External objectives ({', '.join(skipped)}) were skipped."
        )

        if external_status:
            unreachable = [
                name for name, info in external_status.items()
                if info["status"] != "ok"
            ]
            if unreachable:
                note_parts.append(
                    f"Unreachable services: {', '.join(unreachable)}."
                )

        if fallback_error:
            note_parts.append(f"Full scoring failed: {fallback_error}.")

        note_parts.append("Only native components were scored.")
        result["note"] = " ".join(note_parts)

    if external_status:
        result["external_services"] = external_status

    result["ttp_id"] = ttp.id
    result["ttp_version"] = ttp.version
    result["mapper_summary"] = summary
    _annotate_component_sources(result, ttp, summary.get("external_components", []))
    return result
