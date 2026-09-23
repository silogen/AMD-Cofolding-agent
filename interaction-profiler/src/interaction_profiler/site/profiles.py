"""Analysis profile presets for the interaction profiler.

Profiles map a human-readable name to a set of compute flags, making it easy
for the agent to select the right trade-off between speed and richness without
needing to specify every individual flag.

Resolution rule
---------------
``analysis_profile`` sets the baseline flags; any individual compute flag
explicitly supplied by the caller (non-None) overrides the profile value.
When ``analysis_profile=None`` the explicit flags govern — existing call
sites that never pass ``analysis_profile`` behave exactly as before.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Protein-only mode profiles
# ---------------------------------------------------------------------------

PROTEIN_ONLY_PROFILES: dict[str, dict[str, Any]] = {
    "structure": {
        # Secondary structure + chain geometry only. No fpocket, no network.
        # Cost: ~1 s. Use when you only need a structural overview.
        "run_fpocket":           False,
        "assign_ss":             True,
        "fetch_uniprot":         False,
        "fetch_interpro":        False,
        "classify_residue_roles": False,
        "name_subpockets":       False,
    },
    "pocket": {
        # + fpocket pocket geometry (volume, druggability, character).
        # Cost: ~3 s. Default for protein_only mode.
        "run_fpocket":           True,
        "assign_ss":             True,
        "fetch_uniprot":         False,
        "fetch_interpro":        False,
        "classify_residue_roles": True,
        "name_subpockets":       False,
    },
    "annotated": {
        # + UniProt active/binding site + InterPro domain overlay.
        # Cost: ~8 s (includes network calls). Requires uniprot_accession.
        "run_fpocket":           True,
        "assign_ss":             True,
        "fetch_uniprot":         True,
        "fetch_interpro":        True,
        "classify_residue_roles": True,
        "name_subpockets":       False,
    },
    "full": {
        # + Named subpocket assignment from fpocket clusters + annotations.
        # Cost: ~10 s. Pre-campaign target characterization.
        "run_fpocket":           True,
        "assign_ss":             True,
        "fetch_uniprot":         True,
        "fetch_interpro":        True,
        "classify_residue_roles": True,
        "name_subpockets":       True,
    },
}

# ---------------------------------------------------------------------------
# Interaction mode profiles (ligand_only, ligand_target, binder_target)
# ---------------------------------------------------------------------------

INTERACTION_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        # Interactions only. No geometry, no network. Ideal for screening.
        "compute_strain":     False,
        "compute_efficiency": False,
        "compute_interface":  False,
        "compute_vectors":    False,
    },
    "standard": {
        # Current default behavior — strain + efficiency + BSA/Sc.
        "compute_strain":     True,
        "compute_efficiency": True,
        "compute_interface":  True,
        "compute_vectors":    False,
    },
    "enriched": {
        # + exit/growth vector classification and unsatisfied polar detection.
        "compute_strain":     True,
        "compute_efficiency": True,
        "compute_interface":  True,
        "compute_vectors":    True,
    },
    "full": {
        # + UniProt/InterPro annotation overlay (requires uniprot_accession).
        "compute_strain":     True,
        "compute_efficiency": True,
        "compute_interface":  True,
        "compute_vectors":    True,
        "fetch_uniprot":      True,
        "fetch_interpro":     True,
    },
}

# ---------------------------------------------------------------------------
# Ensemble mode profiles (hotspot_ensemble)
# ---------------------------------------------------------------------------

ENSEMBLE_PROFILES: dict[str, dict[str, Any]] = {
    "hotspots": {
        # Current hotspot_ensemble behavior — per-residue contact occupancy.
        "compute_fragment_families": False,
        "compute_compatible_pairs":  False,
    },
    "families": {
        # + Fragment families grouped by dominant binding mode.
        # Computable from existing triples; no extra geometry.
        "compute_fragment_families": True,
        "compute_compatible_pairs":  False,
    },
    "full": {
        # + Compatible-pair detection (requires exit vectors from 'enriched').
        "compute_fragment_families": True,
        "compute_compatible_pairs":  True,
    },
}

# ---------------------------------------------------------------------------
# Profile resolution helper
# ---------------------------------------------------------------------------

def resolve_profile_flags(
    analysis_profile: str | None,
    mode: str,
    **explicit_flags: Any,
) -> dict[str, Any]:
    """Merge preset profile defaults with any explicitly-supplied flag overrides.

    Parameters
    ----------
    analysis_profile:
        Profile name, or None to use explicit flags only (backward-compat).
    mode:
        Current pipeline mode — determines which profile table to look up.
    **explicit_flags:
        Individual compute flags (e.g. ``compute_strain=False``).  A value
        of ``None`` is treated as "not specified" and does not override the
        profile default.

    Returns
    -------
    dict
        Resolved flag dict ready for consumption by pipeline functions.
    """
    if analysis_profile is None:
        # No profile — use explicit flags as-is (existing behavior)
        return {k: v for k, v in explicit_flags.items() if v is not None}

    if mode == "protein_only":
        table = PROTEIN_ONLY_PROFILES
    elif mode in ("hotspot_ensemble",):
        table = ENSEMBLE_PROFILES
    else:
        table = INTERACTION_PROFILES

    if analysis_profile not in table:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "Unknown analysis_profile %r for mode %r; falling back to explicit flags.",
            analysis_profile, mode,
        )
        return {k: v for k, v in explicit_flags.items() if v is not None}

    flags: dict[str, Any] = dict(table[analysis_profile])
    # Explicit non-None flags override the profile
    for key, val in explicit_flags.items():
        if val is not None:
            flags[key] = val
    return flags
