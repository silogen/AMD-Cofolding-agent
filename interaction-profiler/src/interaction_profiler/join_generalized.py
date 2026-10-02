"""Generalised triple builder.

Produces :class:`~interaction_profiler.schemas.GeneralizedTriple` rows from a
list of :class:`~interaction_profiler.schemas.EnrichedInteraction` objects.

The grouping key is ``(binder_feature, target_feature, interaction_type,
subtype, direction)`` — the same semantics as the existing ``build_triples()``
function in :mod:`interaction_profiler.join`, but operating on the enriched
generalised feature fields instead of the raw ``fragment`` label.

Backward compatibility:
    - Existing ``build_triples()`` in ``join.py`` is **not modified**.
    - Callers that use the v1 pipeline get ``Triple`` objects as before.
    - Callers that use the v2 ``/profile_complex`` endpoint get
      ``GeneralizedTriple`` objects from this module.
"""
from __future__ import annotations

import logging
from typing import Any

from interaction_profiler.schemas import EnrichedInteraction, GeneralizedTriple

logger = logging.getLogger(__name__)


def _feature_type(ix: EnrichedInteraction, binder_is_small_molecule: bool) -> str:
    """Infer the binder feature type from interaction fields."""
    if binder_is_small_molecule:
        return "fragment" if ix.fragment else "atom"
    moiety = ix.receptor_moiety or ""
    if moiety == "backbone":
        return "backbone"
    if moiety == "sidechain":
        return "sidechain"
    if ix.receptor_reskind == "ion":
        return "ion"
    if ix.receptor_reskind == "water":
        return "water"
    return "residue"


def build_generalized_triples(
    enriched: list[EnrichedInteraction],
    binder_is_small_molecule: bool = True,
) -> list[GeneralizedTriple]:
    """Group enriched interactions into generalised triple rows.

    Args:
        enriched: Output of
            :func:`interaction_profiler.interactions.enricher.enrich_interactions`.
        binder_is_small_molecule: Controls how ``binder_feature_type`` is
            assigned.  ``True`` for ligand–target mode, ``False`` for
            binder–target protein–protein mode.

    Returns:
        Sorted list of :class:`GeneralizedTriple` objects, one per unique
        ``(binder_feature, target_feature, interaction_type, subtype,
        direction)`` group, with counts and distance/angle ranges.
    """
    # Group interactions by the canonical key.
    groups: dict[tuple, list[EnrichedInteraction]] = {}
    for ix in enriched:
        key = (
            ix.binder_feature or "__none__",
            ix.target_feature or "__none__",
            ix.interaction_type,
            ix.subtype,
            ix.direction,
        )
        groups.setdefault(key, []).append(ix)

    triples: list[GeneralizedTriple] = []

    for (bf, tf, itype, subtype, direction), members in groups.items():
        # Distance and angle ranges across all members.
        distances = [m.distance for m in members if m.distance is not None]
        angles    = [m.angle    for m in members if m.angle    is not None]

        n_confident = sum(1 for m in members if m.confidence == "confident")
        n_uncertain = sum(1 for m in members if m.confidence == "uncertain")

        # Use the first member's target annotations for the group (they share
        # the same target residue by construction of the key).
        first = members[0]

        # Determine binder_feature_type from the first member.
        feat_type = _feature_type(first, binder_is_small_molecule)

        triple = GeneralizedTriple(
            binder_entity_id       = first.binder_entity_id,
            target_entity_id       = first.target_entity_id,
            binder_feature         = bf if bf != "__none__" else "",
            target_feature         = tf if tf != "__none__" else "",
            binder_feature_type    = feat_type,
            interaction_type       = itype,
            subtype                = subtype,
            count                  = len(members),
            distance_min           = min(distances) if distances else None,
            distance_max           = max(distances) if distances else None,
            angle_min              = min(angles)    if angles    else None,
            angle_max              = max(angles)    if angles    else None,
            n_confident            = n_confident,
            n_uncertain            = n_uncertain,
            target_ss_state        = first.target_ss_state,
            target_uniprot_feature = first.target_uniprot_feature,
            target_uniprot_description = first.target_uniprot_description,
            target_interpro_domain = first.target_interpro_domain,
            target_interpro_accession = first.target_interpro_accession,
        )
        triples.append(triple)

    # Sort by count descending, then by binder_feature and target_feature.
    triples.sort(key=lambda t: (-t.count, t.binder_feature, t.target_feature))
    logger.debug("build_generalized_triples: %d triples from %d interactions", len(triples), len(enriched))
    return triples
