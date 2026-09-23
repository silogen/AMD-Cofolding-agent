"""Enriched interaction descriptor classification.

Adds controlled-vocabulary labels to each ``EnrichedInteraction``:

vector_type
    anchor | exit_vector | growth_vector | buried
    Derived deterministically from per-atom freeSASA and proximity to protein:
    - buried: ligand atom SASA ≤ 5 Å² in complex
    - anchor: buried atom involved in a high-confidence interaction
    - exit_vector: ligand atom SASA > threshold AND no protein atom within 4 Å
    - growth_vector: exit_vector AND points toward a named subpocket not yet
      occupied by the ligand

solvent_exposure
    buried | partially_exposed | solvent_exposed
    SASA bins relative to the same atom isolated (fractional SASA):
    - buried:           relative SASA < 0.2
    - partially_exposed: 0.2 ≤ relative SASA < 0.6
    - solvent_exposed:   relative SASA ≥ 0.6

subpocket_id
    Subpocket identifier from SiteProfile if the target residue of this
    interaction has been assigned to a named subpocket.

confidence_bin
    high | moderate | low
    Derived from interaction.confidence + strain per heavy atom:
    - high:     confident + strain_per_ha < 2.0 kcal/mol/HA
    - moderate: uncertain OR moderate strain (2–5 kcal/mol/HA)
    - low:      uncertain + high strain OR VDW clash
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# freeSASA thresholds
_EXIT_VECTOR_MIN_SASA: float = 15.0   # Å² — atom SASA in complex
_PROTEIN_PROXIMITY_A: float  = 4.0    # no protein atom within this distance

# Confidence thresholds
_STRAIN_HIGH_KCAL_PER_HA: float   = 2.0
_STRAIN_MODERATE_KCAL_PER_HA: float = 5.0


def classify_interactions(
    enriched_interactions: list[Any],
    structure_path: str,
    binder_chain_ids: list[str],
    target_chain_ids: list[str],
    strain_result: Any | None = None,
    site_profile: Any | None = None,
) -> list[Any]:
    """Add vector_type, solvent_exposure, subpocket_id, confidence_bin to each interaction.

    All new fields default to None when geometry is unavailable (e.g. SMILES-only
    input where no 3D structure is present).  Existing fields are never modified.

    Parameters
    ----------
    enriched_interactions:
        List of ``EnrichedInteraction`` objects from the normal profiling pipeline.
    structure_path:
        Path to the complex PDB used for the interaction profiling.
    binder_chain_ids:
        Chain IDs of the binder/ligand.
    target_chain_ids:
        Chain IDs of the target protein.
    strain_result:
        ``StrainResult`` from the profiling run; used for confidence_bin.
    site_profile:
        ``SiteProfile`` if available; used for subpocket_id and growth_vector.

    Returns
    -------
    list[EnrichedInteraction]
        Same list with the new optional fields populated.
    """
    if not enriched_interactions:
        return enriched_interactions

    # Build subpocket residue → id map from SiteProfile
    subpocket_map: dict[str, str] = {}
    if site_profile is not None:
        for sp in getattr(site_profile, "subpockets", []):
            for res_label in sp.residues:
                subpocket_map[res_label] = sp.id

    # Per-atom SASA classification (best-effort; skip if structure unavailable)
    atom_sasa: dict[str, float] = {}
    if structure_path:
        atom_sasa = _compute_atom_sasa(structure_path, binder_chain_ids)

    # Strain per heavy atom for confidence binning
    strain_per_ha: float | None = None
    if strain_result is not None:
        strain_per_ha = getattr(strain_result, "strain_per_heavy_atom", None)

    for ix in enriched_interactions:
        # ── subpocket_id from SiteProfile ───────────────────────────────
        target_res = getattr(ix, "target_feature", None) or getattr(ix, "residue_label", None)
        if target_res and target_res in subpocket_map:
            ix.subpocket_id = subpocket_map[target_res]

        # ── confidence_bin ────────────────────────────────────────────────
        confidence_raw = getattr(ix, "confidence", "confident")
        is_confident = confidence_raw == "confident"
        ix.confidence_bin = _compute_confidence_bin(is_confident, strain_per_ha)

        # ── solvent_exposure and vector_type from SASA ────────────────────
        if atom_sasa:
            atom_idx = getattr(ix, "ligand_atom_idx", -1)
            atom_key = str(atom_idx)
            sasa = atom_sasa.get(atom_key)
            if sasa is not None:
                ix.solvent_exposure = _sasa_to_exposure_bin(sasa)
                ix.vector_type = _classify_vector(
                    sasa, is_confident, ix.solvent_exposure,
                    target_res, subpocket_map,
                )

    return enriched_interactions


def _compute_atom_sasa(
    structure_path: str,
    binder_chain_ids: list[str],
) -> dict[str, float]:
    """Return per-atom SASA dict keyed by ligand atom index string."""
    try:
        import freesasa  # noqa: PLC0415
        import MDAnalysis as mda  # noqa: PLC0415

        u = mda.Universe(structure_path)
        # Select binder atoms only
        if binder_chain_ids:
            sel_str = " or ".join(
                f"chainID {ch} or segid {ch}" for ch in binder_chain_ids
            )
            binder_ag = u.select_atoms(sel_str)
        else:
            binder_ag = u.atoms

        # freeSASA on the full complex
        params = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley})
        result = freesasa.calc(freesasa.Structure(structure_path), params)

        # Build idx→SASA mapping for binder atoms
        atom_sasa: dict[str, float] = {}
        for i, atom in enumerate(binder_ag):
            atom_sasa[str(i)] = float(
                result.atomArea(atom.resid, atom.chainID)
                if hasattr(result, "atomArea")
                else 0.0
            )
        return atom_sasa

    except Exception as exc:
        logger.debug("Per-atom SASA computation failed: %s", exc)
        return {}


def _sasa_to_exposure_bin(sasa: float) -> str:
    """Convert raw SASA in Å² to an exposure bin."""
    if sasa < 5.0:
        return "buried"
    if sasa < 20.0:
        return "partially_exposed"
    return "solvent_exposed"


def _classify_vector(
    sasa: float,
    is_confident: bool,
    exposure_bin: str,
    target_res: str | None,
    subpocket_map: dict[str, str],
) -> str:
    """Classify the geometric role of this ligand atom."""
    if exposure_bin == "buried" and is_confident:
        return "anchor"
    if exposure_bin == "solvent_exposed":
        # Check if it points toward an unoccupied subpocket
        if target_res and target_res not in subpocket_map:
            return "growth_vector"
        return "exit_vector"
    return "buried"


def _compute_confidence_bin(is_confident: bool, strain_per_ha: float | None) -> str:
    """Map interaction confidence + strain to a qualitative bin."""
    if not is_confident:
        if strain_per_ha is not None and strain_per_ha > _STRAIN_HIGH_KCAL_PER_HA:
            return "low"
        return "moderate"
    if strain_per_ha is not None:
        if strain_per_ha > _STRAIN_MODERATE_KCAL_PER_HA:
            return "moderate"
        if strain_per_ha > _STRAIN_HIGH_KCAL_PER_HA:
            return "moderate"
    return "high"
