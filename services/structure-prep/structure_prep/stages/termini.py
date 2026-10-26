"""Stage 7 — Termini Capping.

Optionally adds ACE (N-terminal acetyl) and NME (C-terminal N-methyl
amide) caps. Required for OpenMM force-field simulation to prevent
artificial terminal charges. Default is none (no capping), which is
correct for docking and cofolding inputs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gemmi

from structure_prep.models import CapEntry, Stage7Profile

if TYPE_CHECKING:
    from structure_prep.models import RoleMap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard amino-acid residue names (used to identify polymer chain breaks)
# ---------------------------------------------------------------------------
_STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation-state variants
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "ARN", "CYX", "CYM",
    # common non-standard conversions
    "MSE", "SEC", "PYL",
})

# Bond lengths (Å) used for idealised cap geometry
_BOND_C_N: float = 1.33   # peptide bond C–N
_BOND_C_C: float = 1.52   # Cα–C / C–CH3
_BOND_C_O: float = 1.23   # carbonyl C=O
_BOND_N_CH3: float = 1.45  # N–CH3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit_vec(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    """Return a unit vector along (dx, dy, dz), falling back to (1,0,0)."""
    length = (dx * dx + dy * dy + dz * dz) ** 0.5
    if length < 1e-6:
        return (1.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def _perp_vec(ux: float, uy: float, uz: float) -> tuple[float, float, float]:
    """Return a unit vector perpendicular to (ux, uy, uz) via cross product."""
    if abs(uz) < 0.9:
        # cross with z-axis: (ux,uy,uz) × (0,0,1) = (uy, -ux, 0)
        return _unit_vec(uy, -ux, 0.0)
    else:
        # cross with x-axis: (ux,uy,uz) × (1,0,0) = (0, uz, -uy)
        return _unit_vec(0.0, uz, -uy)


def _make_atom(
    name: str,
    element: str,
    x: float,
    y: float,
    z: float,
    occ: float = 1.0,
    b_iso: float = 0.0,
) -> gemmi.Atom:
    a = gemmi.Atom()
    a.name = name
    a.element = gemmi.Element(element)
    a.pos = gemmi.Position(x, y, z)
    a.occ = occ
    a.b_iso = b_iso
    return a


# ---------------------------------------------------------------------------
# Cap residue builders
# ---------------------------------------------------------------------------

def _build_ace(pivot_residue: gemmi.Residue, seqnum: int) -> gemmi.Residue:
    """Build an ACE cap residue placed before *pivot_residue* (N-terminal side).

    ACE atoms: C (carbonyl carbon, 1.33 Å from chain N), CH3 (methyl), O (carbonyl O).
    The carbonyl C is placed along the N→CA outward direction from the first
    residue's backbone N, so the resulting C–N peptide-bond distance is correct.
    """
    ace = gemmi.Residue()
    ace.name = "ACE"
    try:
        ace.seqid = gemmi.SeqId(str(seqnum))
    except Exception:
        ace.seqid = gemmi.SeqId(seqnum, " ")  # type: ignore[arg-type]
    ace.entity_type = gemmi.EntityType.Polymer

    n_atom = next((a for a in pivot_residue if a.name == "N"), None)
    ca_atom = next((a for a in pivot_residue if a.name == "CA"), None)

    if n_atom and ca_atom:
        nx, ny, nz = n_atom.pos.x, n_atom.pos.y, n_atom.pos.z
        cax, cay, caz = ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z
        # Direction: from CA toward N (outward from chain)
        ux, uy, uz = _unit_vec(nx - cax, ny - cay, nz - caz)
        # ACE C is 1.33 Å beyond N in that direction
        c_x = nx + ux * _BOND_C_N
        c_y = ny + uy * _BOND_C_N
        c_z = nz + uz * _BOND_C_N
    else:
        logger.warning(
            "ACE builder: residue %s%d has no N/CA; placing cap at origin.",
            pivot_residue.name, pivot_residue.seqid.num,
        )
        ux, uy, uz = 1.0, 0.0, 0.0
        c_x, c_y, c_z = 0.0, 0.0, 0.0

    # C (carbonyl carbon)
    ace.add_atom(_make_atom("C", "C", c_x, c_y, c_z))

    # CH3 (methyl carbon, 1.52 Å from C along the same outward direction)
    ace.add_atom(_make_atom(
        "CH3", "C",
        c_x + ux * _BOND_C_C,
        c_y + uy * _BOND_C_C,
        c_z + uz * _BOND_C_C,
    ))

    # O (carbonyl oxygen, 1.23 Å from C, perpendicular to the C–CH3 axis)
    px, py, pz = _perp_vec(ux, uy, uz)
    ace.add_atom(_make_atom(
        "O", "O",
        c_x + px * _BOND_C_O,
        c_y + py * _BOND_C_O,
        c_z + pz * _BOND_C_O,
    ))

    return ace


def _build_nme(pivot_residue: gemmi.Residue, seqnum: int) -> gemmi.Residue:
    """Build an NME cap residue placed after *pivot_residue* (C-terminal side).

    NME atoms: N (amide nitrogen, 1.33 Å from chain C), CH3 (methyl).
    """
    nme = gemmi.Residue()
    nme.name = "NME"
    try:
        nme.seqid = gemmi.SeqId(str(seqnum))
    except Exception:
        nme.seqid = gemmi.SeqId(seqnum, " ")  # type: ignore[arg-type]
    nme.entity_type = gemmi.EntityType.Polymer

    c_atom = next((a for a in pivot_residue if a.name == "C"), None)
    ca_atom = next((a for a in pivot_residue if a.name == "CA"), None)

    if c_atom and ca_atom:
        cx, cy, cz = c_atom.pos.x, c_atom.pos.y, c_atom.pos.z
        cax, cay, caz = ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z
        # Direction: from CA toward C (outward from chain)
        ux, uy, uz = _unit_vec(cx - cax, cy - cay, cz - caz)
        # NME N is 1.33 Å beyond C in that direction
        n_x = cx + ux * _BOND_C_N
        n_y = cy + uy * _BOND_C_N
        n_z = cz + uz * _BOND_C_N
    else:
        logger.warning(
            "NME builder: residue %s%d has no C/CA; placing cap at origin.",
            pivot_residue.name, pivot_residue.seqid.num,
        )
        ux, uy, uz = 1.0, 0.0, 0.0
        n_x, n_y, n_z = 0.0, 0.0, 0.0

    # N (amide nitrogen)
    nme.add_atom(_make_atom("N", "N", n_x, n_y, n_z))

    # CH3 (methyl carbon, 1.45 Å from N along the same outward direction)
    nme.add_atom(_make_atom(
        "CH3", "C",
        n_x + ux * _BOND_N_CH3,
        n_y + uy * _BOND_N_CH3,
        n_z + uz * _BOND_N_CH3,
    ))

    return nme


# ---------------------------------------------------------------------------
# Chain indexing helpers
# ---------------------------------------------------------------------------

def _polymer_indexed(chain: gemmi.Chain) -> list[tuple[int, gemmi.Residue]]:
    """Return (chain_index, residue) for every polymer residue, in order."""
    result: list[tuple[int, gemmi.Residue]] = []
    for i, res in enumerate(chain):
        if res.entity_type == gemmi.EntityType.Polymer:
            result.append((i, res))
    return result


def _detect_sequence_gaps(
    poly: list[tuple[int, gemmi.Residue]],
) -> list[int]:
    """Return indices j into *poly* such that a sequence gap exists AFTER poly[j].

    A gap is any jump in residue sequence number > 1 that has no intervening
    residue in the chain.  This correctly identifies un-modelled loop breaks.
    """
    gaps: list[int] = []
    for j in range(len(poly) - 1):
        curr_seqnum = poly[j][1].seqid.num
        next_seqnum = poly[j + 1][1].seqid.num
        if next_seqnum > curr_seqnum + 1:
            gaps.append(j)
    return gaps


# ---------------------------------------------------------------------------
# Cap insertion for a single chain
# ---------------------------------------------------------------------------

def _cap_chain(
    chain: gemmi.Chain,
    cap_n_term: bool,
    cap_c_term: bool,
    cap_internal: bool,
) -> list[CapEntry]:
    """Insert ACE/NME residues into *chain* in-place; return CapEntry records."""
    poly = _polymer_indexed(chain)
    if not poly:
        logger.debug("Stage 7: chain %s has no polymer residues; skipping.", chain.name)
        return []

    gaps: list[int] = _detect_sequence_gaps(poly) if cap_internal else []

    caps: list[CapEntry] = []

    # We track an offset because each insertion shifts subsequent chain indices.
    offset = 0

    # --- N-terminal ACE --------------------------------------------------
    if cap_n_term:
        first_chain_idx, first_res = poly[0]
        seqnum_ace = first_res.seqid.num - 1
        ace = _build_ace(first_res, seqnum_ace)
        chain.add_residue(ace, first_chain_idx + offset)
        offset += 1
        caps.append(CapEntry(
            chain=chain.name,
            terminus_type="n_terminal",
            cap_residue="ACE",
            position=seqnum_ace,
        ))
        logger.debug(
            "Stage 7: added ACE at N-terminus of chain %s (seqnum %d).",
            chain.name, seqnum_ace,
        )

    # --- Internal break caps (NME on C-side, ACE on N-side of each gap) ---
    for gap_idx in gaps:
        # NME after poly[gap_idx] — C-side of the break
        nme_pivot_chain_idx = poly[gap_idx][0] + offset
        nme_pivot_res = poly[gap_idx][1]
        seqnum_nme = nme_pivot_res.seqid.num + 1
        nme = _build_nme(nme_pivot_res, seqnum_nme)
        chain.add_residue(nme, nme_pivot_chain_idx + 1)
        offset += 1
        caps.append(CapEntry(
            chain=chain.name,
            terminus_type="chain_break_c",
            cap_residue="NME",
            position=seqnum_nme,
        ))
        logger.debug(
            "Stage 7: added NME at break C-side in chain %s (after seqnum %d).",
            chain.name, nme_pivot_res.seqid.num,
        )

        # ACE before poly[gap_idx + 1] — N-side of the break
        ace_pivot_chain_idx = poly[gap_idx + 1][0] + offset
        ace_pivot_res = poly[gap_idx + 1][1]
        seqnum_ace_br = ace_pivot_res.seqid.num - 1
        ace = _build_ace(ace_pivot_res, seqnum_ace_br)
        chain.add_residue(ace, ace_pivot_chain_idx)
        offset += 1
        caps.append(CapEntry(
            chain=chain.name,
            terminus_type="chain_break_n",
            cap_residue="ACE",
            position=seqnum_ace_br,
        ))
        logger.debug(
            "Stage 7: added ACE at break N-side in chain %s (before seqnum %d).",
            chain.name, ace_pivot_res.seqid.num,
        )

    # --- C-terminal NME --------------------------------------------------
    if cap_c_term:
        # Use the original (un-offset) last polymer residue's chain index.
        last_chain_idx, last_res = poly[-1]
        seqnum_nme = last_res.seqid.num + 1
        nme = _build_nme(last_res, seqnum_nme)
        # Insert AFTER the last polymer residue (accounting for all offsets so far)
        chain.add_residue(nme, last_chain_idx + offset + 1)
        offset += 1
        caps.append(CapEntry(
            chain=chain.name,
            terminus_type="c_terminal",
            cap_residue="NME",
            position=seqnum_nme,
        ))
        logger.debug(
            "Stage 7: added NME at C-terminus of chain %s (seqnum %d).",
            chain.name, seqnum_nme,
        )

    return caps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage7Profile,
    role_map: "RoleMap",
    missing_residues: list[dict[str, Any]],  # from Stage 5 — used for break detection hints
) -> tuple[gemmi.Structure, dict[str, Any]]:
    """Stage 7 — Termini Capping.

    Parameters
    ----------
    structure:
        gemmi Structure (after protonation from Stage 6).
    profile:
        Stage7Profile: cap_termini, cap_internal_breaks, cap_binder_chains,
        cap_chains.
    role_map:
        Stage 2 RoleMap, used to identify protein and binder chains.
    missing_residues:
        Stage 5 report entries; passed for informational use and future
        cross-referencing with actual chain gaps.

    Returns
    -------
    (capped_structure, stage7_report_dict)
        ``stage7_report_dict`` keys: ``termini_policy`` (str),
        ``termini_capped`` (list[CapEntry as dict]).
    """
    if profile.cap_termini == "none":
        return structure, {"termini_policy": "none", "termini_capped": []}

    # --- Determine which chains to cap -----------------------------------
    all_protein_chains: set[str] = set(role_map.protein_chains)
    all_binder_chains: set[str] = set(role_map.binder_chains)

    if profile.cap_chains is not None:
        chains_to_cap = set(profile.cap_chains)
    else:
        chains_to_cap = set(all_protein_chains)
        if profile.cap_binder_chains:
            chains_to_cap |= all_binder_chains

    logger.info(
        "Stage 7: cap_termini=%s; %d chain(s) selected for capping: %s",
        profile.cap_termini,
        len(chains_to_cap),
        sorted(chains_to_cap),
    )

    all_caps: list[CapEntry] = []

    if len(structure) == 0:
        logger.warning("Stage 7: structure has no models; nothing to cap.")
        return structure, {
            "termini_policy": profile.cap_termini,
            "termini_capped": [],
        }

    model = structure[0]

    for chain in model:
        if chain.name not in chains_to_cap:
            continue

        try:
            chain_caps = _cap_chain(
                chain=chain,
                cap_n_term=True,
                cap_c_term=True,
                cap_internal=profile.cap_internal_breaks,
            )
            all_caps.extend(chain_caps)
        except Exception:
            logger.exception(
                "Stage 7: failed to cap chain %s; skipping.", chain.name
            )

    logger.info(
        "Stage 7: added %d cap residue(s) across %d chain(s).",
        len(all_caps),
        len(chains_to_cap),
    )

    report: dict[str, Any] = {
        "termini_policy": profile.cap_termini,
        "termini_capped": [c.model_dump() for c in all_caps],
    }
    return structure, report


__all__ = ["run"]
