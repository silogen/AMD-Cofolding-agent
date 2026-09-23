"""TTP-to-REINVENT4 Scorer configuration mapper.

Translates a TTP into the dict structure expected by REINVENT4's
``Scorer`` class.  Pre-scoring components and scoring objectives are
merged into a single composed Scorer -- all components score every
molecule with no filtering or gating.

Public entry points:

* ``ttp_to_scorer_config``  -- full Scorer config dict (used by RL loop)
* ``ttp_to_native_scorer_config`` -- Scorer config with only native
  components (used by ``score_molecules`` when external services are
  unavailable)
* ``extract_native_components`` -- flat list of native component dicts
  compatible with the Phase-1 ``score_molecules`` direct-mode input
* ``describe_external_objectives`` -- resolved identity of each external
  objective (service, URL, selectors), so callers can tell one
  ``ExternalRESTScorer`` from another
* ``get_mapper_summary`` -- metadata about what the mapper produced
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from reinvent4_mcp.ttp.schema import (
    ConstraintConfig,
    PreScoringComponent,
    ScoringObjective,
    TTP,
)

logger = logging.getLogger(__name__)

_CONSTRAINT_TYPE_MAP: dict[str, str] = {
    "custom_alerts": "CustomAlerts",
}

# Every external objective is mapped to the same REINVENT4 component class
# (``ExternalRESTScorer``), so neither the scorer config nor REINVENT4's own
# "Creating scoring component ExternalRESTScorer" log says what the objective
# actually measures.  These blurbs travel with the resolved description so a
# reader (agent or human) can tell the instances apart.
_SERVICE_DESCRIPTIONS: dict[str, str] = {
    "boltz": (
        "Boltz-2 co-folding: predicts the protein-ligand complex structure and "
        "a binding-affinity score; also emits structure artifacts that "
        "downstream objectives can consume via depends_on."
    ),
    "posebusters": (
        "PoseBusters pose validity: physical/geometric sanity checks on a "
        "predicted pose (requires structures from an upstream producer)."
    ),
    "admet": (
        "ADMET predictor: one physicochemical/ADMET endpoint per objective, "
        "selected by service_params['endpoint']."
    ),
    "retrodmta": (
        "RetroDMTA project model: predicts a project endpoint for a target, "
        "selected by service_params['target'] (+ optional 'endpoint')."
    ),
    "interaction_profiler": (
        "Interaction profiler: counts typed protein-ligand interactions "
        "(service_params['interaction_type']) in a structure, so it is "
        "normally chained after a structure producer via depends_on."
    ),
}

# Params that identify plumbing rather than scientific intent; reported as
# dedicated keys in the description instead of inside ``params``.
_PLUMBING_PARAMS = frozenset({
    "service", "service_url", "component_name", "depends_on", "failure_score",
})

# The params that actually decide *what* an objective measures.  Listed first
# in the resolved description so the distinguishing selector is impossible to
# miss among the operational knobs.
_SELECTOR_PARAMS = ("endpoint", "target", "interaction_type", "affinity_mode", "mode")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ttp_to_scorer_config(ttp: TTP) -> dict[str, Any]:
    """Build a full REINVENT4 Scorer configuration from a TTP.

    Merges ``pre_scoring`` components and ``scoring`` objectives into a
    single flat component list.  Scoring objectives with pipeline
    dependencies (``depends_on``) are topologically sorted so that
    producers appear before consumers (e.g. Boltz before PoseBusters).

    Returns a dict of the form::

        {
            "type": "geometric_mean",
            "component": [
                {"ComponentType": {"endpoint": [{name, weight, ...}]}},
                ...
            ]
        }

    Native components map directly to REINVENT4 component classes.
    External objectives map to ``ExternalRESTScorer`` placeholders
    (Phase 3 will register that component).
    Constraints map to REINVENT4 filter components.
    """

    scorer_components: list[dict[str, Any]] = []

    # 1. Pre-scoring components (always native, always first)
    if ttp.pre_scoring:
        for comp in ttp.pre_scoring.components:
            scorer_components.append(
                _map_native_component(comp.component, comp)
            )

    # 2. Scoring objectives -- topologically sorted so pipeline
    #    dependencies (depends_on) are respected
    ordered = _topological_sort_objectives(ttp.scoring.objectives)
    for obj in ordered:
        if obj.type == "native":
            assert obj.component is not None
            scorer_components.append(
                _map_native_component(obj.component, obj)
            )
        elif obj.type == "external":
            scorer_components.append(_map_external_objective(obj, ttp))

    # 3. Constraints (mapped as REINVENT4 filter components)
    if ttp.constraints:
        for constraint in ttp.constraints:
            scorer_components.append(_map_constraint(constraint))

    return {
        "type": ttp.scoring.aggregation,
        "component": scorer_components,
    }


def ttp_to_native_scorer_config(ttp: TTP) -> dict[str, Any]:
    """Build a Scorer configuration containing only native components.

    This is the preferred entry point for TTP-based scoring when
    external REST services are not yet available (Phase 2).  It
    produces a config dict that REINVENT4's ``Scorer`` can consume
    directly, preserving the TTP's aggregation method, weights, and
    transforms without a lossy round-trip through flat component lists.

    Returns the same structure as ``ttp_to_scorer_config`` but with
    external objectives omitted.
    """

    scorer_components: list[dict[str, Any]] = []

    if ttp.pre_scoring:
        for comp in ttp.pre_scoring.components:
            scorer_components.append(
                _map_native_component(comp.component, comp)
            )

    for obj in ttp.scoring.objectives:
        if obj.type == "native":
            assert obj.component is not None
            scorer_components.append(
                _map_native_component(obj.component, obj)
            )

    if ttp.constraints:
        for constraint in ttp.constraints:
            scorer_components.append(_map_constraint(constraint))

    return {
        "type": ttp.scoring.aggregation,
        "component": scorer_components,
    }


def extract_native_components(ttp: TTP) -> list[dict[str, Any]]:
    """Return only the native components from a TTP as a flat list.

    The returned list is compatible with the ``score_molecules`` tool's
    ``components`` parameter (direct mode), filtering out external
    objectives that require REST services.

    .. note::

       Prefer ``ttp_to_native_scorer_config`` for TTP-based scoring --
       it produces a proper Scorer config dict directly.  This function
       exists for backward compatibility with the Phase-1 direct-mode
       interface.
    """

    components: list[dict[str, Any]] = []

    if ttp.pre_scoring:
        for comp in ttp.pre_scoring.components:
            entry: dict[str, Any] = {
                "type": comp.component,
                "name": comp.name,
                "weight": comp.weight,
            }
            if comp.transform:
                entry["transform"] = comp.transform.model_dump()
            components.append(entry)

    for obj in ttp.scoring.objectives:
        if obj.type == "native":
            entry = {
                "type": obj.component,
                "name": obj.name,
                "weight": obj.weight,
            }
            if obj.transform:
                entry["transform"] = obj.transform.model_dump()
            if obj.params:
                from reinvent4_mcp.tools.scoring import _normalize_endpoint_params

                entry["params"] = _normalize_endpoint_params(obj.params)
            components.append(entry)

    return components


def describe_external_objectives(
    ttp: TTP,
    scorer_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Describe every external objective as it will actually be executed.

    All external objectives collapse onto a single REINVENT4 component class,
    ``ExternalRESTScorer``, so a bare component list cannot say which instance
    talks to Boltz and which to ADMET's ``logp`` endpoint.  This returns, per
    objective, the resolved service, URL, selector params and pipeline role --
    the same values the component receives at runtime, including process-config
    defaults and URL corrections applied by the mapper.

    Pass ``scorer_config`` (from ``ttp_to_scorer_config``) when one has already
    been built, to read the resolved params straight off it instead of mapping
    the TTP a second time.

    Bulky or non-informative values (protein sequences, MSA paths) are
    summarised rather than reproduced, so the result is safe to hand to an LLM.
    """

    from_config = (
        _external_params_by_name(scorer_config) if scorer_config else {}
    )
    descriptions: list[dict[str, Any]] = []

    for obj in ttp.scoring.objectives:
        if obj.type != "external":
            continue

        entry: dict[str, Any] = {
            "name": obj.name,
            "component": "ExternalRESTScorer",
            "service": obj.service,
            "weight": obj.weight,
        }

        # Prefer the fully mapped params (config defaults, resolved URL, cached
        # MSA) over the raw TTP fields.  Mapping raises for objectives missing a
        # required selector, which is exactly the case where the raw fields are
        # the only thing we can report.
        params = from_config.get(obj.name)
        if params is None:
            try:
                mapped = _map_external_objective(obj, ttp)
                params = mapped["ExternalRESTScorer"]["endpoint"][0].get("params", {})
            except Exception:
                params = {
                    "service": obj.service,
                    "service_url": obj.service_url,
                    **(obj.service_params or {}),
                }
                entry["incomplete"] = (
                    "Could not resolve full params for this objective (likely a "
                    "missing required selector); showing the TTP values only."
                )

        entry["service_url"] = params.get("service_url") or obj.service_url

        selectors = {
            key: params[key] for key in _SELECTOR_PARAMS if key in params
        }
        if selectors:
            entry["selects"] = selectors

        other = {
            key: _summarize_param(key, value)
            for key, value in params.items()
            if key not in _PLUMBING_PARAMS and key not in selectors
        }
        if other:
            entry["params"] = other

        if obj.depends_on:
            entry["depends_on"] = obj.depends_on
        if obj.transform:
            entry["transform"] = obj.transform.type
        if params.get("failure_score") is not None:
            entry["failure_score"] = params["failure_score"]
        if obj.comment:
            entry["comment"] = obj.comment

        description = _SERVICE_DESCRIPTIONS.get(obj.service or "")
        if description:
            entry["measures"] = description

        descriptions.append(entry)

    return descriptions


def _external_params_by_name(
    scorer_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index the ``ExternalRESTScorer`` params in a scorer config by endpoint name."""

    params_by_name: dict[str, dict[str, Any]] = {}
    for component in scorer_config.get("component", []):
        external = component.get("ExternalRESTScorer")
        if not external:
            continue
        for endpoint in external.get("endpoint", []):
            name = endpoint.get("name")
            if name:
                params_by_name[name] = endpoint.get("params", {})
    return params_by_name


def _summarize_param(key: str, value: Any) -> Any:
    """Shrink params that are bulky but carry no distinguishing value.

    A protein sequence runs to thousands of characters and is identical for
    every objective on the same target, so repeating it per objective would
    crowd out the selectors the reader is actually looking for.
    """

    if not isinstance(value, str):
        return value
    if key.endswith("sequence"):
        return f"<{len(value)} aa, elided>"
    if len(value) > 120:
        return f"<{len(value)} chars, elided>"
    return value


def get_mapper_summary(
    ttp: TTP,
    scorer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metadata about how a TTP maps to Scorer components.

    Useful for diagnostics and agent feedback -- tells the caller how
    many components of each kind are in the TTP, what each external
    objective actually talks to (``external_components``), and what the
    pipeline dependency chain looks like.

    ``scorer_config`` is an optional already-built config from
    ``ttp_to_scorer_config``; passing it avoids mapping the TTP twice.
    """

    n_pre_scoring = (
        len(ttp.pre_scoring.components) if ttp.pre_scoring else 0
    )
    native_objectives = [
        o.name for o in ttp.scoring.objectives if o.type == "native"
    ]
    external_objectives = [
        o.name for o in ttp.scoring.objectives if o.type == "external"
    ]
    n_constraints = len(ttp.constraints) if ttp.constraints else 0

    pipeline_deps: list[dict[str, str]] = []
    for obj in ttp.scoring.objectives:
        if obj.depends_on:
            pipeline_deps.append(
                {"from": obj.depends_on, "to": obj.name}
            )

    return {
        "n_pre_scoring": n_pre_scoring,
        "n_native_objectives": len(native_objectives),
        "n_external_objectives": len(external_objectives),
        "n_constraints": n_constraints,
        "n_total_components": (
            n_pre_scoring
            + len(native_objectives)
            + len(external_objectives)
            + n_constraints
        ),
        "native_objective_names": native_objectives,
        "external_objective_names": external_objectives,
        "external_components": describe_external_objectives(ttp, scorer_config),
        "pipeline_dependencies": pipeline_deps,
        "aggregation": ttp.scoring.aggregation,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _topological_sort_objectives(
    objectives: list[ScoringObjective],
) -> list[ScoringObjective]:
    """Sort scoring objectives so that ``depends_on`` targets come first.

    Uses Kahn's algorithm.  Objectives without dependencies preserve
    their original relative order (stable sort).  Raises ``ValueError``
    on cycles (which the schema validator should already prevent).
    """

    by_name: dict[str, ScoringObjective] = {o.name: o for o in objectives}

    # Build adjacency: edge from dependency -> dependent
    in_degree: dict[str, int] = {o.name: 0 for o in objectives}
    dependents: dict[str, list[str]] = {o.name: [] for o in objectives}

    for obj in objectives:
        if obj.depends_on and obj.depends_on in by_name:
            in_degree[obj.name] += 1
            dependents[obj.depends_on].append(obj.name)

    # Seed queue with zero-in-degree nodes in original order
    queue: list[str] = [
        o.name for o in objectives if in_degree[o.name] == 0
    ]
    result: list[ScoringObjective] = []

    while queue:
        name = queue.pop(0)
        result.append(by_name[name])
        for dep_name in dependents[name]:
            in_degree[dep_name] -= 1
            if in_degree[dep_name] == 0:
                queue.append(dep_name)

    if len(result) != len(objectives):
        missing = {o.name for o in objectives} - {o.name for o in result}
        raise ValueError(
            f"Circular pipeline dependency detected among: "
            f"{', '.join(sorted(missing))}"
        )

    return result


def _map_native_component(
    component_type: str,
    source: PreScoringComponent | ScoringObjective,
) -> dict[str, Any]:
    """Map a native TTP component/objective to a REINVENT4 component config."""

    endpoint: dict[str, Any] = {
        "name": source.name,
        "weight": source.weight,
    }
    if source.transform:
        endpoint["transform"] = source.transform.model_dump()
    if isinstance(source, ScoringObjective) and source.params:
        from reinvent4_mcp.tools.scoring import _normalize_endpoint_params

        endpoint["params"] = _normalize_endpoint_params(source.params)
    return {
        component_type: {"endpoint": [endpoint]},
    }


def _map_external_objective(
    objective: ScoringObjective,
    ttp: TTP,
) -> dict[str, Any]:
    """Build an ``ExternalRESTScorer`` config entry.

    Packs the service URL, service-specific parameters, target info,
    and pipeline dependency into the endpoint's ``params`` dict so the
    ExternalRESTScorer (Phase 3) can unpack them at runtime.

    For Boltz objectives, operational defaults (recycling_steps,
    diffusion_samples, use_msa_server, etc.) are pulled from the
    process config and can be overridden by ``service_params`` in the
    TTP or ``msa_path`` in the TTP target.
    """

    endpoint: dict[str, Any] = {
        "name": objective.name,
        "weight": objective.weight,
    }
    if objective.transform:
        endpoint["transform"] = objective.transform.model_dump()

    # The canonical service URL lives in the process config (keyed by
    # Docker Compose service name).  The agent often supplies the wrong
    # URL when authoring a TTP -- e.g. the source-tree directory name
    # ``boltz2`` instead of the service name ``boltz``, or omitting the
    # ``/score`` path.  Prefer the config URL for known services so a
    # mistaken value in the TTP cannot break scoring at runtime.
    service_url = _resolve_service_url(objective.service, objective.service_url)

    params: dict[str, Any] = {
        "service": objective.service,
        "service_url": service_url,
        "component_name": objective.name,
    }

    # Optional: emit a fixed score (e.g. 0.0) instead of NaN when the service
    # fails, so a broken objective is penalized rather than dropped from
    # aggregation. Off by default (None).
    from reinvent4_mcp.config import get_config

    _failure_score = get_config().scoring.external_failure_score
    if _failure_score is not None:
        params["failure_score"] = _failure_score

    # Inject process config defaults for Boltz before TTP overrides
    if objective.service == "boltz":
        params.update(_get_boltz_defaults())

    # Inject process config defaults for PoseBusters before TTP overrides
    if objective.service == "posebusters":
        params.update(_get_posebusters_defaults())

    # Inject model-stack default before objective-level ADMET overrides.
    if objective.service == "admet":
        params.update(_get_admet_defaults())

    # Inject process config defaults for RetroDMTA (method/window) before TTP overrides.
    if objective.service == "retrodmta":
        params.update(_get_retrodmta_defaults())

    # Inject interaction-profiler defaults (interaction_type/ph/radius) before TTP
    # overrides. Counting interactions needs a structure, so this objective is
    # normally chained after a structure producer (boltz) via depends_on.
    if objective.service == "interaction_profiler":
        params.update(_get_interaction_profiler_defaults())

    if objective.service_params:
        params.update(objective.service_params)

    # Multi-endpoint predictors need an explicit selector. Fail loudly at
    # scorer-build time rather than silently defaulting (admet -> logp) or
    # letting the service 422 into all-NaN (retrodmta missing target).
    if objective.service == "admet" and "endpoint" not in params:
        raise ValueError(
            f"admet objective '{objective.name}' requires "
            "service_params={'endpoint': '<name>'} (e.g. 'logp', "
            "'caco2_permeability'). Call list_external_services for valid endpoints."
        )
    if objective.service == "retrodmta" and "target" not in params:
        raise ValueError(
            f"retrodmta objective '{objective.name}' requires "
            "service_params={'target': '<name>'} (e.g. 'FXA'). "
            "Call list_external_services for valid targets."
        )

    # Inject protein target info for structure prediction (Boltz)
    if ttp.target and objective.service in ("boltz",):
        params["protein_sequence"] = ttp.target.protein_sequence
        if ttp.target.msa_path:
            params["msa_path"] = ttp.target.msa_path
            # A precomputed MSA and the colabfold server are mutually
            # exclusive: if the user supplied an MSA file, use it and do
            # not let boltz fall back to the remote server.
            params["use_msa_server"] = False
        else:
            # No explicit MSA: reuse a cached one for this sequence if
            # present, else auto-enable the MSA server so the cofold never
            # fails for lack of an alignment.
            params.update(_resolve_boltz_msa(ttp.target.protein_sequence))

    # protein_pdb_path is relevant for both structure prediction and
    # pose validation services
    if ttp.target and ttp.target.protein_pdb_path:
        if objective.service in ("boltz", "posebusters"):
            params.setdefault("protein_pdb_path", ttp.target.protein_pdb_path)

    if objective.depends_on:
        params["depends_on"] = objective.depends_on

    endpoint["params"] = params
    return {
        "ExternalRESTScorer": {"endpoint": [endpoint]},
    }


def _resolve_service_url(service: str | None, ttp_url: str | None) -> str | None:
    """Return the canonical URL for a known external service.

    For services configured in the process config (``boltz``,
    ``posebusters``), the config URL is authoritative -- it reflects the
    actual Docker Compose service name and ``/score`` endpoint.  If the
    TTP supplied a different URL it is overridden (with a warning) so a
    typo such as ``http://boltz2:8080/`` cannot break scoring.  Unknown
    services fall back to the TTP-supplied URL unchanged.
    """

    from reinvent4_mcp.config import get_config

    if not service:
        return ttp_url

    canonical = get_config().enabled_external_services.get(service)
    if canonical is None:
        return ttp_url

    if ttp_url and ttp_url != canonical:
        logger.warning(
            "TTP service_url %r for service %r does not match the configured "
            "URL %r; using the configured URL.",
            ttp_url,
            service,
            canonical,
        )
    return canonical


def _get_boltz_defaults() -> dict[str, Any]:
    """Pull Boltz operational defaults from the process config."""

    from reinvent4_mcp.config import get_config

    config = get_config()
    boltz = config.scoring.boltz
    defaults: dict[str, Any] = {
        "recycling_steps": boltz.recycling_steps,
        "diffusion_samples": boltz.diffusion_samples,
        "sampling_steps_affinity": boltz.sampling_steps_affinity,
        "affinity_mode": boltz.affinity_mode,
    }
    if boltz.use_msa_server:
        defaults["use_msa_server"] = True
        defaults["msa_server_url"] = boltz.msa_server_url
    return defaults


def sequence_msa_hash(sequence: str) -> str:
    """Hash a protein sequence for MSA cache lookup.

    Must match the Boltz service's ``_msa_cache_path`` (``sha1`` of the
    upper-cased, stripped sequence) so the mapper reuses the exact ``.a3m``
    the service persisted at ``<msa_cache_dir>/<hash>.a3m``.
    """
    seq = sequence.upper().strip()
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()  # noqa: S324


def _resolve_boltz_msa(sequence: str) -> dict[str, Any]:
    """Decide how Boltz should obtain its MSA for ``sequence``.

    Resolution order (no explicit ``target.msa_path`` -- that is handled by
    the caller):
      1. If a persisted MSA for this sequence exists in the configured cache
         dir, reuse it (``msa_path``).
      2. Else, if ``auto_msa_fallback`` is enabled, turn on the MSA server so
         Boltz generates one this run (``use_msa_server``).
      3. Else, return nothing (caller/service will surface the missing-MSA
         error).
    Returns a dict of params to merge into the Boltz endpoint params.
    """
    from reinvent4_mcp.config import get_config

    boltz = get_config().scoring.boltz

    cached = Path(boltz.msa_cache_dir) / f"{sequence_msa_hash(sequence)}.a3m"
    try:
        exists = cached.is_file()
    except Exception:
        exists = False
    if exists:
        logger.info("Reusing cached MSA for target: %s", cached)
        return {"msa_path": str(cached), "use_msa_server": False}

    if boltz.auto_msa_fallback:
        logger.info(
            "No local/cached MSA found; enabling MSA server to generate one "
            "(the Boltz service persists it under %s for reuse). MSA-free "
            "fallback=%s if the server is unreachable.",
            boltz.msa_cache_dir,
            boltz.msa_free_fallback,
        )
        return {
            "use_msa_server": True,
            "msa_server_url": boltz.msa_server_url,
            "msa_free_fallback": boltz.msa_free_fallback,
        }

    return {}


def _get_posebusters_defaults() -> dict[str, Any]:
    """Pull PoseBusters operational defaults from the process config."""

    from reinvent4_mcp.config import get_config

    config = get_config()
    pb = config.scoring.posebusters
    return {"mode": pb.mode}


def _get_retrodmta_defaults() -> dict[str, Any]:
    """Pull RetroDMTA operational defaults (method/window) from the process config."""

    from reinvent4_mcp.config import get_config

    r = get_config().scoring.retrodmta
    return {"method": r.method, "window": r.window}


def _get_admet_defaults() -> dict[str, Any]:
    """Pull the default ADMET model stack from process configuration."""

    from reinvent4_mcp.config import get_config

    return {"model_set": get_config().scoring.admet.model_set}


def _get_interaction_profiler_defaults() -> dict[str, Any]:
    """Pull interaction-profiler operational defaults from the process config."""

    from reinvent4_mcp.config import get_config

    ip = get_config().scoring.interaction_profiler
    return {"interaction_type": ip.interaction_type, "ph": ip.ph, "radius": ip.radius}


def _map_constraint(constraint: ConstraintConfig) -> dict[str, Any]:
    """Map a TTP constraint to a REINVENT4 filter component config."""

    constraint_data = constraint.model_dump()
    ctype = constraint_data.pop("type")
    component_name = _CONSTRAINT_TYPE_MAP.get(ctype, ctype)

    endpoint: dict[str, Any] = {
        "name": f"constraint_{ctype}",
        "weight": 1.0,
    }

    params = {k: v for k, v in constraint_data.items() if v is not None}
    if params:
        endpoint["params"] = params

    return {
        component_name: {"endpoint": [endpoint]},
    }


# ---------------------------------------------------------------------------
# Generation config mapping — block 08 addition
# ---------------------------------------------------------------------------

def ttp_to_generation_config(ttp: "TTP") -> dict[str, Any]:  # noqa: F821
    """Map a TTP's generation section to a REINVENT4 generation config dict.

    The returned dict is consumed by ``toml_generator.py`` to produce the
    correct ``[reinvent]`` / ``[libinvent]`` / ``[linkinvent]`` TOML block.

    Parameters
    ----------
    ttp:
        A fully-validated TTP object.

    Returns
    -------
    dict
        Empty dict when ``ttp.generation`` is ``None`` (backward compat).
        Otherwise a dict with all generation parameters.

    Raises
    ------
    ValueError
        If mode-specific required inputs are absent (e.g. linkinvent without
        fragments).  These checks are redundant with schema validators but are
        repeated here so ``ttp_to_generation_config`` is safe to call on dicts
        deserialized outside the Pydantic stack.
    """
    from reinvent4_mcp.ttp.schema import GenerationMode  # noqa: PLC0415

    if ttp.generation is None:
        return {}

    g = ttp.generation

    # Defensive mode-specific checks (Pydantic validators already ran, but
    # this function may also be called from non-Pydantic code paths)
    if g.mode == GenerationMode.linkinvent:
        if not g.linkinvent_input or not g.linkinvent_input.fragments:
            raise ValueError(
                "linkinvent mode requires linkinvent_input.fragments "
                "(at least 2 fragment SMILES with attachment points)."
            )
        if len(g.linkinvent_input.fragments) < 2:
            raise ValueError(
                "linkinvent requires at least 2 fragment SMILES."
            )

    if g.mode == GenerationMode.libinvent:
        if not g.libinvent_input or not g.libinvent_input.scaffold:
            raise ValueError(
                "libinvent mode requires libinvent_input.scaffold "
                "(scaffold SMILES with [*:n] attachment points)."
            )

    if g.mode == GenerationMode.mol2mol:
        if not g.mol2mol_input or not g.mol2mol_input.seed_smiles:
            raise ValueError(
                "mol2mol mode requires mol2mol_input.seed_smiles "
                "(one or more valid seed SMILES)."
            )

    return {
        "mode":               g.mode.value,
        "model":              g.model,
        "batch_size":         g.batch_size,
        "n_steps":            g.n_steps,
        "checkpoint_every_n": g.checkpoint_every_n,
        "seed":               g.seed,
        "device":             g.device,
        "linkinvent_input": (
            g.linkinvent_input.model_dump() if g.linkinvent_input else None
        ),
        "libinvent_input": (
            g.libinvent_input.model_dump() if g.libinvent_input else None
        ),
        "mol2mol_input": (
            g.mol2mol_input.model_dump() if g.mol2mol_input else None
        ),
        "target_artifact_ids":              g.target_artifact_ids,
        "prepared_structure_artifact_ids":  g.prepared_structure_artifact_ids,
        "site_artifact_ids":                g.site_artifact_ids,
    }
