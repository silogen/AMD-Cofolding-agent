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

from interaction_profiler.schemas import (
    EnrichedInteraction,
    Interaction,
)

logger = logging.getLogger(__name__)


def _lookup_annotation(
    amap: dict | None,
    chain: str,
    resnum: int | None,
) -> dict | None:
    """Look up a per-residue annotation, preferring a chain-aware key.

    Annotation maps may be keyed either chain-aware as ``(chain, resnum)`` or
    by ``resnum`` alone (single-chain callers).  Secondary-structure maps are
    always chain-aware, so this helper keeps feature/InterPro lookups symmetric
    with them: it tries ``(chain, resnum)`` first and falls back to ``resnum``.
    Without this, residue *n* of chain A and residue *n* of chain B collide in a
    resnum-only lookup, silently attaching the wrong annotation in multi-chain
    complexes.
    """
    if amap is None or resnum is None:
        return None
    entry = amap.get((chain, resnum))
    if entry is None:
        entry = amap.get(resnum)
    return entry


def enrich_interactions(
    interactions: list[Interaction],
    ss_map: dict[tuple[str, int], str] | None = None,
    feature_map: dict | None = None,
    interpro_map: dict | None = None,
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
        feature_map: UniProt per-residue annotations from
            ``taxonomy.annotation_mapper.map_uniprot_features_to_residues()``.
            Keyed either chain-aware as ``{(chain, resnum): {...}}`` or by
            ``{resnum: {...}}`` (single-chain); both are looked up correctly.
            ``None`` means no UniProt annotations available.
        interpro_map: InterPro per-residue annotations, keyed the same way as
            ``feature_map``.  ``None`` means no InterPro annotations available.
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
        uniprot_entry = _lookup_annotation(feature_map, target_chain, target_resnum)
        if uniprot_entry and uniprot_entry.get("mapped", True):
            base["target_uniprot_feature"]     = uniprot_entry.get("feature_type")
            base["target_uniprot_description"] = uniprot_entry.get("description")
            n_uniprot_hits += 1
        else:
            base["target_uniprot_feature"]     = None
            base["target_uniprot_description"] = None

        # ── InterPro domain at target residue ───────────────────────────────
        # Same attach rule as UniProt (consistent): present and mapped (default
        # True when the source map carries no explicit 'mapped' flag).
        interpro_entry = _lookup_annotation(interpro_map, target_chain, target_resnum)
        if interpro_entry and interpro_entry.get("mapped", True):
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
            # ligand_atom_idx may be None (not just a negative sentinel), so
            # check the type before the >= 0 comparison.
            lig_idx = ix.ligand_atom_idx
            base["binder_feature"] = (
                f"{ix.ligand_atom_name or 'UNK'}:{lig_idx}"
                if isinstance(lig_idx, int) and lig_idx >= 0
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
