"""Stage 10 — structure alignment via Cα superposition.

Aligns the mobile structure onto a reference using gemmi.calculate_superposition
with SupSelect.CaP. The transform is applied to EVERY entity in every model
(protein, ligand, water, ions) so the coordinate frame is globally consistent.

The 4×4 matrix, per-chain RMSD, and n_aligned residues are recorded in the
report so that the transform can be inverted and inter-fragment distances
computed in a shared reference frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gemmi

logger = logging.getLogger(__name__)

_AA_POLYMER = gemmi.PolymerType.PeptideL  # for Cα-based superposition


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

# The Stage10Profile lives in models.py; imported here only for type hints.


def _polymer_span(chain: gemmi.Chain) -> gemmi.ResidueSpan:
    """Return the polymer residue span for a chain."""
    return chain.get_polymer()


def _align_chains(
    ref_model: gemmi.Model,
    mob_model: gemmi.Model,
    ref_chain_id: str,
    mob_chain_id: str,
) -> tuple[gemmi.SupResult, list[str]]:
    """Superpose one chain pair and return (SupResult, warnings)."""
    warnings: list[str] = []
    ref_chain = ref_model.find_chain(ref_chain_id)
    mob_chain = mob_model.find_chain(mob_chain_id)
    if ref_chain is None:
        raise ValueError(f"Reference chain '{ref_chain_id}' not found.")
    if mob_chain is None:
        raise ValueError(f"Mobile chain '{mob_chain_id}' not found.")
    ref_span = _polymer_span(ref_chain)
    mob_span = _polymer_span(mob_chain)
    if ref_span.length() == 0 or mob_span.length() == 0:
        raise ValueError(
            f"One or both chains have no polymer residues "
            f"(ref={ref_span.length()}, mob={mob_span.length()})."
        )
    try:
        sup = gemmi.calculate_superposition(
            ref_span,
            mob_span,
            _AA_POLYMER,
            gemmi.SupSelect.CaP,
        )
    except Exception as exc:
        raise ValueError(f"Superposition failed: {exc}") from exc
    if sup.count == 0:
        raise ValueError("Superposition returned 0 aligned atoms — sequence mismatch?")
    return sup, warnings


def _transform_to_matrix(t: gemmi.Transform) -> list[list[float]]:
    """Serialise a gemmi Transform as a 4×4 row-major matrix."""
    rows = t.mat.tolist()  # 3×3 rotation, already a list of lists
    vec = t.vec
    return [
        [rows[0][0], rows[0][1], rows[0][2], vec.x],
        [rows[1][0], rows[1][1], rows[1][2], vec.y],
        [rows[2][0], rows[2][1], rows[2][2], vec.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def run(
    structure: gemmi.Structure,
    profile: Any,  # Stage10Profile
    reference_structure: gemmi.Structure,
) -> tuple[gemmi.Structure, dict[str, Any]]:
    """Align *structure* onto *reference_structure* and return the moved copy.

    Parameters
    ----------
    structure:
        Mobile structure (post-stage-9 coordinates).
    profile:
        Stage10Profile with reference_chain, mobile_chain, max_rmsd_A.
    reference_structure:
        Pre-loaded reference gemmi.Structure.

    Returns
    -------
    (structure, report_dict)
        *structure* is transformed in-place (all entities, all models).
        *report_dict* carries the 4×4 matrix, RMSD and n_aligned.
    """
    report: dict[str, Any] = {
        "performed": True,
        "reference_chain": profile.reference_chain,
        "mobile_chain": profile.mobile_chain,
    }

    if len(reference_structure) == 0 or len(structure) == 0:
        report["performed"] = False
        report["reason"] = "One or both structures have no models."
        return structure, report

    ref_model = reference_structure[0]
    mob_model = structure[0]

    try:
        sup, warn = _align_chains(
            ref_model,
            mob_model,
            profile.reference_chain,
            profile.mobile_chain,
        )
    except ValueError as exc:
        report["performed"] = False
        report["reason"] = str(exc)
        return structure, report

    rmsd = round(sup.rmsd, 4)
    report["n_aligned"] = sup.count
    report["rmsd_ca_A"] = rmsd
    report["transform_4x4"] = _transform_to_matrix(sup.transform)

    if profile.max_rmsd_A is not None and rmsd > profile.max_rmsd_A:
        report["performed"] = False
        report["reason"] = (
            f"RMSD {rmsd:.3f} Å exceeds threshold {profile.max_rmsd_A} Å."
        )
        return structure, report

    # Apply to ALL models so the structure is fully consistent.
    for model in structure:
        model.transform_pos_and_adp(sup.transform)

    report["warnings"] = warn
    logger.info(
        "[Stage 10] Alignment: RMSD=%.3f Å over %d Cα atoms",
        rmsd,
        sup.count,
    )
    return structure, report
