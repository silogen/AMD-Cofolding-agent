"""Enrichment layer: annotate existing Interaction objects with biological context.

This module is a pure transformation — it never accesses PDB files or runs
geometry calculations. It takes pre-computed maps (secondary structure,
UniProt features, InterPro domains) and attaches them as nullable fields
to new :class:`EnrichedInteraction` objects, leaving the original
:class:`Interaction` objects untouched.

Pipeline position:
    detect_interactions() → rank_and_cap() → enrich_interactions()
                                              → build_generalized_triples()
"""
from __future__ import annotations

import logging
from typing import Any

from interaction_profiler.schemas import (
    EnrichedInteraction,
    Interaction,
)

logger = logging.getLogger(__name__)


def enrich_interactions(
    interactions: list[Interaction],
    ss_map: dict[tuple[str, int], str] | None = None,
    feature_map: dict[int, dict[str, Any]] | None = None,
    interpro_map: dict[int, dict[str, Any]] | None = None,
    binder_entity_id: str | None = None,
    target_entity_id: str | None = None,
    binder_is_small_molecule: bool = True,
) -> list[EnrichedInteraction]:
    """Attach biological context to a list of Interaction objects.

    Args:
        interactions: Raw :class:`Interaction` list from the geometric detector.
        ss_map: ``{(chain_id, resnum): ss_state}`` from
            ``taxonomy.secondary_structure.assign_secondary_structure()``.
            ``None`` means no secondary structure data available.
        feature_map: ``{resnum: {feature_type, description, mapped, ...}}`` from
            ``taxonomy.annotation_mapper.map_uniprot_features_to_residues()``.
            ``None`` means no UniProt annotations available.
        interpro_map: ``{resnum: {domain_name, accession, db, ...}}`` from
            ``taxonomy.annotation_mapper.map_interpro_to_residues()``.
            ``None`` means no InterPro annotations available.
        binder_entity_id: Entity ID for the binder side (e.g. ``"binder_A"``).
        target_entity_id: Entity ID for the target side (e.g. ``"target_B"``).
        binder_is_small_molecule: When ``True`` the ``binder_feature`` is set
            from ``interaction.fragment`` (existing taxonomy).  When ``False``
            the binder is a protein/peptide and ``binder_feature`` is built as
            ``"{resname}:{resnum}.{chain}"`` — not applicable here since the
            detector is geometry-based for small molecules, but the flag allows
            future protein–protein use.

    Returns:
        A new list of :class:`EnrichedInteraction` objects.  One per input
        interaction, preserving order.  Every new field defaults to ``None``
        when the corresponding map lookup finds no hit.
    """
    ss_map = ss_map or {}
    feature_map = feature_map or {}
    interpro_map = interpro_map or {}

    enriched: list[EnrichedInteraction] = []
    n_ss_hits = 0
    n_uniprot_hits = 0
    n_interpro_hits = 0

    for ix in interactions:
        # Build the base dict from the parent Interaction fields.
        base = ix.model_dump()

        # ── Entity provenance ────────────────────────────────────────────────
        base["binder_entity_id"] = binder_entity_id
        base["target_entity_id"] = target_entity_id

        # ── Secondary structure at target residue ───────────────────────────
        target_resnum = ix.receptor_resnum
        target_chain  = ix.receptor_chain or ""
        if target_resnum is not None:
            ss_state = ss_map.get((target_chain, target_resnum))
            base["target_ss_state"] = ss_state
            if ss_state:
                n_ss_hits += 1
        else:
            base["target_ss_state"] = None

        # ── UniProt feature at target residue ───────────────────────────────
        uniprot_entry: dict[str, Any] | None = None
        if target_resnum is not None:
            uniprot_entry = feature_map.get(target_resnum)
        if uniprot_entry and uniprot_entry.get("mapped"):
            base["target_uniprot_feature"]     = uniprot_entry.get("feature_type")
            base["target_uniprot_description"] = uniprot_entry.get("description")
            n_uniprot_hits += 1
        else:
            base["target_uniprot_feature"]     = None
            base["target_uniprot_description"] = None

        # ── InterPro domain at target residue ───────────────────────────────
        interpro_entry: dict[str, Any] | None = None
        if target_resnum is not None:
            interpro_entry = interpro_map.get(target_resnum)
        if interpro_entry:
            base["target_interpro_domain"]    = interpro_entry.get("domain_name")
            base["target_interpro_accession"] = interpro_entry.get("accession")
            n_interpro_hits += 1
        else:
            base["target_interpro_domain"]    = None
            base["target_interpro_accession"] = None

        # ── Generalised feature labels ───────────────────────────────────────
        if binder_is_small_molecule:
            # Re-use the existing fragment label (may be None if no fragment matched).
            base["binder_feature"] = ix.fragment
        else:
            # Protein / peptide binder: construct a residue label.
            # The ligand fields encode the binder residue in this path.
            base["binder_feature"] = (
                f"{ix.ligand_atom_name or 'UNK'}:{ix.ligand_atom_idx}"
                if ix.ligand_atom_idx >= 0
                else None
            )

        if target_resnum is not None:
            base["target_feature"] = (
                f"{ix.receptor_resname}{target_resnum}.{target_chain}"
            )
        else:
            base["target_feature"] = None

        enriched.append(EnrichedInteraction(**base))

    logger.debug(
        "enrich_interactions: %d interactions enriched — SS=%d, UniProt=%d, InterPro=%d",
        len(enriched), n_ss_hits, n_uniprot_hits, n_interpro_hits,
    )
    return enriched
