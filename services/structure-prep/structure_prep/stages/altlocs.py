"""Stage 4 — Altloc Resolution.

Collapses residues with multiple alternate conformers to a single model.
Always flags binding-site altlocs (within radius of any retained ligand).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import gemmi

from structure_prep.models import Stage4Profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_altlocs(residue: gemmi.Residue) -> dict[str, list[gemmi.Atom]]:
    """Return {altloc_code: [atoms]} for atoms that carry an explicit altloc.

    Atoms with no altloc (' ', '', '\\x00') are excluded from the returned
    groups; they are always kept regardless of which conformer is selected.
    """
    groups: dict[str, list[gemmi.Atom]] = {}
    for atom in residue:
        alt = atom.altloc
        if alt and alt not in (" ", "\x00"):
            groups.setdefault(alt, []).append(atom)
    return groups


def _near_ligand(
    res_pos: list[tuple[float, float, float]],
    lig_positions: list[tuple[float, float, float]],
    radius: float,
) -> tuple[bool, float]:
    """Return (is_within_radius, min_distance_Å).

    Returns (False, inf) when either list is empty.
    """
    if not lig_positions or not res_pos:
        return False, float("inf")
    min_d = float("inf")
    for px, py, pz in res_pos:
        for lx, ly, lz in lig_positions:
            d = ((px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2) ** 0.5
            if d < min_d:
                min_d = d
                if min_d == 0.0:  # early-exit shortcut
                    return True, 0.0
    return min_d < radius, min_d


def _occupancy_sum(atoms: list[gemmi.Atom]) -> float:
    """Return the summed occupancy across a list of atoms."""
    return sum(a.occ for a in atoms)


def _apply_altloc_selection(
    residue: gemmi.Residue,
    selected: str,
) -> None:
    """Remove all non-selected altloc atoms in-place; clear altloc codes on kept atoms.

    Atoms with no altloc code (blank / '\\x00') are always retained.
    Iterates backwards to preserve index validity during deletion.
    """
    i = len(residue) - 1
    while i >= 0:
        atom = residue[i]
        alt = atom.altloc
        if alt and alt not in (" ", "\x00"):
            # Has an explicit altloc code
            if alt != selected:
                del residue[i]  # discard non-selected conformer atom
            else:
                atom.altloc = "\x00"  # clear code — becomes the "canonical" atom
        i -= 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage4Profile,
    ligand_positions: list[tuple[float, float, float]] | None = None,
) -> tuple[gemmi.Structure, dict]:
    """Stage 4: Altloc Resolution.

    Parameters
    ----------
    structure:
        Gemmi Structure after Stage 3 curation.
    profile:
        Stage4Profile controlling the resolution strategy.
    ligand_positions:
        Heavy-atom (x, y, z) coordinates of all retained ligands, used to
        flag binding-site altlocs.  Pass None or [] if no ligands are present.

    Returns
    -------
    (resolved_structure, stage4_report_dict)

    stage4_report_dict keys
    -----------------------
    altloc_policy:          str
    n_positions_resolved:   int
    altlocs_resolved:       list[dict]  — per-position resolution details
    binding_site_altlocs:   list[dict]  — flagged positions near ligands
    """
    lig_pos: list[tuple[float, float, float]] = ligand_positions or []
    policy = profile.altloc_policy

    st = copy.deepcopy(structure)

    n_positions_resolved = 0
    altlocs_resolved: list[dict[str, Any]] = []
    binding_site_altlocs: list[dict[str, Any]] = []
    unresolved: list[str] = []  # collected for deferred error

    for model in st:
        for chain in model:
            for residue in chain:
                altloc_groups = _get_altlocs(residue)
                if not altloc_groups:
                    continue  # no altlocs — nothing to do

                altloc_codes = sorted(altloc_groups.keys())
                occ_per_alt: dict[str, float] = {
                    alt: _occupancy_sum(atoms)
                    for alt, atoms in altloc_groups.items()
                }
                # Highest-occupancy altloc is always computed as a recommendation
                highest_occ_alt = max(occ_per_alt, key=lambda k: occ_per_alt[k])

                # ----------------------------------------------------------
                # Binding-site proximity check (policy-independent)
                # ----------------------------------------------------------
                res_positions: list[tuple[float, float, float]] = [
                    (a.pos.x, a.pos.y, a.pos.z) for a in residue
                ]
                is_near, dist = _near_ligand(
                    res_positions, lig_pos, profile.binding_site_altloc_radius_A
                )
                dist_rounded = round(dist, 3) if dist != float("inf") else None

                if is_near:
                    binding_site_altlocs.append(
                        {
                            "chain": chain.name,
                            "resseq": residue.seqid.num,
                            "resname": residue.name,
                            # Highest-occupancy is the natural recommendation even
                            # when no resolution is applied (retain_all / fail).
                            "selected_altloc": highest_occ_alt,
                            "selected_occupancy": round(occ_per_alt[highest_occ_alt], 3),
                            "discarded_altlocs": [
                                a for a in altloc_codes if a != highest_occ_alt
                            ],
                            "near_ligand": True,
                            "distance_to_ligand_A": dist_rounded,
                        }
                    )

                # ----------------------------------------------------------
                # Policy dispatch
                # ----------------------------------------------------------
                if policy == "retain_all":
                    # No structural change — leave all altloc atoms intact.
                    continue

                if policy == "fail":
                    # Collect all positions first; raise after iterating.
                    unresolved.append(
                        f"{chain.name}:{residue.seqid.num}:{residue.name}"
                    )
                    continue

                if policy == "highest_occupancy":
                    selected = highest_occ_alt

                elif policy == "selected_conformer":
                    lookup_key = f"{chain.name}:{residue.seqid.num}"
                    selected = profile.altloc_selected.get(lookup_key)
                    if selected is None:
                        # Defer error — collect all unresolved positions first
                        unresolved.append(
                            f"{chain.name}:{residue.seqid.num}:{residue.name}"
                        )
                        continue
                    if selected not in altloc_groups:
                        logger.warning(
                            "Stage 4: requested altloc %r not present in residue "
                            "%s:%s:%s (available: %s); falling back to highest "
                            "occupancy.",
                            selected,
                            chain.name,
                            residue.seqid.num,
                            residue.name,
                            altloc_codes,
                        )
                        selected = highest_occ_alt

                else:
                    # Unrecognised policy — fall back to highest occupancy safely
                    logger.warning(
                        "Stage 4: unrecognised altloc_policy=%r; "
                        "falling back to highest_occupancy.",
                        policy,
                    )
                    selected = highest_occ_alt

                # ----------------------------------------------------------
                # Apply selection: remove non-selected altloc atoms
                # ----------------------------------------------------------
                _apply_altloc_selection(residue, selected)

                discarded = [a for a in altloc_codes if a != selected]
                n_positions_resolved += 1
                altlocs_resolved.append(
                    {
                        "chain": chain.name,
                        "resseq": residue.seqid.num,
                        "resname": residue.name,
                        "selected_altloc": selected,
                        "selected_occupancy": round(occ_per_alt.get(selected, 0.0), 3),
                        "discarded_altlocs": discarded,
                        "near_ligand": is_near,
                        "distance_to_ligand_A": dist_rounded,
                    }
                )

    # ------------------------------------------------------------------
    # Deferred error handling
    # ------------------------------------------------------------------
    if unresolved:
        if policy == "fail":
            raise ValueError(
                f"altloc_policy='fail': {len(unresolved)} position(s) with "
                f"alternate conformers detected — "
                + ", ".join(unresolved)
            )
        if policy == "selected_conformer":
            raise ValueError(
                f"altloc_policy='selected_conformer': no altloc_selected entry "
                f"for {len(unresolved)} position(s) — "
                + ", ".join(unresolved)
            )
        # Should not reach here for other policies, but log defensively
        logger.warning(
            "Stage 4: %d unresolved altloc positions (policy=%r).",
            len(unresolved),
            policy,
        )

    logger.info(
        "Stage 4 complete: policy=%r, %d position(s) resolved, "
        "%d binding-site altloc(s) flagged.",
        policy,
        n_positions_resolved,
        len(binding_site_altlocs),
    )

    return st, {
        "altloc_policy": policy,
        "n_positions_resolved": n_positions_resolved,
        "altlocs_resolved": altlocs_resolved,
        "binding_site_altlocs": binding_site_altlocs,
    }


__all__ = ["run"]
