# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pipeline-aware scorer for artifact passing between external components.

When a TTP has pipeline dependencies (e.g., PoseBusters ``depends_on`` Boltz
for predicted structure files), the standard REINVENT4 Scorer executes
components in order but has no mechanism to pass runtime artifacts between
them.  ``PipelineAwareScorer`` wraps the standard Scorer and injects upstream
artifacts into dependent components before they are called.

Architecture
~~~~~~~~~~~~

The Scorer stores components as ``ComponentData`` objects whose ``.params``
tuple is ``(names, component_instance, transforms, weights)``.  The pipeline
wiring works by replacing a dependent ``ExternalRESTScorer`` instance in
that tuple with an ``_ArtifactInjectingWrapper`` that:

1. Reads ``last_artifacts`` from the upstream component
2. Injects ``structure_paths`` (and ``protein_pdb_path``) into the dependent
   component's ``request_params``
3. Delegates to the original ``__call__``

Because the Scorer processes components sequentially in config order, and the
TTP mapper topologically sorts objectives (Boltz before PoseBusters), the
upstream component has always completed before the dependent wrapper runs.
"""

from __future__ import annotations

__all__ = ["PipelineAwareScorer"]

import logging
from typing import Any

from reinvent4_mcp.ttp.schema import TTP

logger = logging.getLogger(__name__)


class _ArtifactInjectingWrapper:
    """Transparent wrapper that injects upstream artifacts before delegating.

    Placed between the Scorer's execution loop and a dependent
    ``ExternalRESTScorer`` instance.  Attribute access (other than
    ``__call__``) is forwarded to the wrapped instance so the Scorer's
    internal machinery (caching, mask handling) works unchanged.
    """

    def __init__(self, wrapped: Any, upstream: Any, ttp: TTP):
        self._wrapped = wrapped
        self._upstream = upstream
        self._ttp = ttp

    def __call__(self, smilies):
        artifacts = self._upstream.last_artifacts
        if artifacts:
            structures = artifacts.get("structures")
            valid_structures = [s for s in (structures or []) if s]
            if valid_structures:
                self._wrapped.request_params["structure_paths"] = structures
                logger.info(
                    "Injected %d structure paths from upstream into '%s'",
                    len(valid_structures),
                    getattr(self._wrapped, "service", "unknown"),
                )
            if self._ttp.target and self._ttp.target.protein_pdb_path:
                self._wrapped.request_params["protein_pdb_path"] = (
                    self._ttp.target.protein_pdb_path
                )
            elif valid_structures:
                self._wrapped.request_params["protein_pdb_path"] = (
                    valid_structures[0]
                )
                logger.info(
                    "Using Boltz complex PDB as protein reference for '%s': %s",
                    getattr(self._wrapped, "service", "unknown"),
                    valid_structures[0],
                )
        else:
            logger.warning(
                "Upstream component produced no artifacts; dependent "
                "component '%s' may fail if it requires structure_paths",
                getattr(self._wrapped, "service", "unknown"),
            )

        return self._wrapped(smilies)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class PipelineAwareScorer:
    """Wraps a standard REINVENT4 Scorer with pipeline artifact injection.

    Delegates all scoring to the standard Scorer, but first wires up
    ``_ArtifactInjectingWrapper`` objects for components that have
    ``depends_on`` in their ``request_params``.

    Usage::

        scorer = PipelineAwareScorer(scorer_config, ttp)
        results = scorer.compute_results(smiles, invalid_mask, duplicate_mask)
    """

    def __init__(self, scorer_config: dict[str, Any], ttp: TTP):
        from reinvent4_mcp.utils import ensure_reinvent4_importable

        ensure_reinvent4_importable()
        from reinvent.scoring.scorer import Scorer

        self._scorer = Scorer(scorer_config)
        self._ttp = ttp

        if self._has_pipeline_dependencies():
            self._wire_pipeline()

    def _has_pipeline_dependencies(self) -> bool:
        """Check whether any TTP objective has a ``depends_on`` relationship."""
        return any(o.depends_on for o in self._ttp.scoring.objectives)

    def _wire_pipeline(self) -> None:
        """Replace dependent component instances with artifact-injecting wrappers.

        Walks the Scorer's component list, identifies ``ExternalRESTScorer``
        instances (which expose ``depends_on``), builds a name->instance
        map, and wraps any instance whose ``depends_on`` points at an
        existing upstream component.
        """

        external_instances: dict[str, Any] = {}
        dependent_components: list[tuple[Any, str]] = []

        for comp_data in self._scorer.components.scorers:
            names, instance, _transforms, _weights = comp_data.params

            if not hasattr(instance, "request_params"):
                continue

            for name in names:
                external_instances[name] = instance

            depends_on = getattr(instance, "depends_on", None)
            if depends_on:
                dependent_components.append((comp_data, depends_on))

        for comp_data, depends_on in dependent_components:
            upstream = external_instances.get(depends_on)
            if upstream is None:
                logger.warning(
                    "Component depends on '%s' but no upstream component "
                    "found with that name; skipping pipeline wiring",
                    depends_on,
                )
                continue

            old_names, old_instance, old_transforms, old_weights = comp_data.params
            wrapper = _ArtifactInjectingWrapper(old_instance, upstream, self._ttp)
            comp_data.params = (old_names, wrapper, old_transforms, old_weights)

            logger.info(
                "Wired pipeline: '%s' artifacts -> '%s'",
                depends_on,
                old_names[0] if old_names else "unknown",
            )

    def compute_results(
        self, smilies, invalid_mask, duplicate_mask, **kwargs
    ):
        """Delegate to the standard Scorer (pipeline wiring is already in place)."""
        return self._scorer.compute_results(
            smilies, invalid_mask, duplicate_mask, **kwargs
        )

    def collected_structures(self) -> list[str]:
        """Return structure paths produced by any component this run.

        After scoring, a structure-producing component (e.g. Boltz) holds its
        outputs in ``last_artifacts['structures']``. Surfacing these lets the
        caller (score_molecules) hand the exact deposited path back to the agent
        so it can profile the RIGHT structure instead of guessing.
        """
        structures: list[str] = []
        try:
            for comp_data in self._scorer.components.scorers:
                _names, instance, _transforms, _weights = comp_data.params
                inst = getattr(instance, "_wrapped", instance)
                arts = getattr(inst, "last_artifacts", None)
                if arts:
                    for s in (arts.get("structures") or []):
                        if s:
                            structures.append(s)
        except Exception:
            logger.debug("Could not collect structures", exc_info=True)
        return structures
