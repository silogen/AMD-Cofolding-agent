"""Chain selection stage — reduce a prepared structure to target chains.

Runs immediately after stage 2 (role assignment), before curation, so that
HETATM/water curation uses the already-reduced structure.

Policies:
  keep_all                  — no-op (default, backwards-compatible)
  keep_target               — retain only protein_chains from the role map
  keep_target_plus_binder   — retain protein + binder chains
  keep_listed               — retain exactly the listed chains
  dedup_identical           — also collapse chains with ≥ identity_threshold
                              sequence identity (useful for homodimers)

Interface partner policy:
  drop           — no extras beyond the selected chains (default)
  keep_chain     — always retain one nominated partner chain
  keep_within_A  — retain all residues of any chain within radius Å of
                   the selected chains' heavy atoms
"""

from __future__ import annotations

import logging
from typing import Any

import gemmi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_amino_acid(residue: gemmi.Residue) -> bool:
    try:
        return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())
    except Exception:
        return False


def _chain_sequence(chain: gemmi.Chain) -> str:
    parts: list[str] = []
    for res in chain:
        if not _is_amino_acid(res):
            continue
        try:
            code = gemmi.find_tabulated_residue(res.name).one_letter_code
        except Exception:
            code = "X"
        parts.append(code if code and len(code) == 1 else "X")
    return "".join(parts)


def _sequence_identity(a: str, b: str) -> float:
    """Simple pairwise identity on the shorter sequence."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def _heavy_atom_positions(
    structure: gemmi.Structure,
    chain_names: set[str],
) -> list[gemmi.Position]:
    positions: list[gemmi.Position] = []
    if len(structure) == 0:
        return positions
    for chain in structure[0]:
        if chain.name not in chain_names:
            continue
        for res in chain:
            for atom in res:
                if atom.element != gemmi.Element("H") and atom.element != gemmi.Element("D"):
                    positions.append(atom.pos)
    return positions


def _distance(pos1: gemmi.Position, pos2: gemmi.Position) -> float:
    return pos1.dist(pos2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Any,  # Stage2bProfile / chain-selection fields from Stage2Profile
    role_map: Any,  # RoleMap
) -> tuple[gemmi.Structure, dict[str, Any]]:
    """Select chains, optionally dedup homodimers, and return the reduced structure."""

    policy = getattr(profile, "chain_policy", "keep_all")
    identity_threshold = getattr(profile, "dedup_identity_threshold", 0.99)
    interface_policy = getattr(profile, "interface_policy", "drop")
    interface_radius_A = getattr(profile, "interface_radius_A", 5.0)
    keep_partner_chain = getattr(profile, "keep_partner_chain", None)
    listed_chains = set(getattr(profile, "keep_chains", []) or [])

    report: dict[str, Any] = {
        "chain_policy": policy,
        "chains_removed": [],
        "deduped_chains": [],
        "interface_chains_retained": [],
    }

    if policy == "keep_all":
        return structure, report

    # ----------------------------------------------------------------
    # Step 1: build the keep-set from role_map
    # ----------------------------------------------------------------
    protein_set = set(role_map.protein_chains)
    binder_set = set(role_map.binder_chains)

    if policy == "keep_target":
        keep = set(protein_set)
    elif policy == "keep_target_plus_binder":
        keep = protein_set | binder_set
    elif policy == "keep_listed":
        keep = listed_chains
    else:
        keep = protein_set  # fallback

    # ----------------------------------------------------------------
    # Step 2: homodimer dedup
    # ----------------------------------------------------------------
    dedup = getattr(profile, "dedup_identical", False)
    if dedup and len(structure) > 0:
        sequences = {
            chain.name: _chain_sequence(chain)
            for chain in structure[0]
            if chain.name in keep
        }
        seen_seqs: list[tuple[str, str]] = []  # (chain_name, seq)
        drop_dupes: set[str] = set()
        for name, seq in sorted(sequences.items()):  # sorted for determinism
            is_dup = any(
                _sequence_identity(seq, prev_seq) >= identity_threshold
                for _, prev_seq in seen_seqs
            )
            if is_dup:
                drop_dupes.add(name)
                report["deduped_chains"].append(name)
            else:
                seen_seqs.append((name, seq))
        keep -= drop_dupes

    # ----------------------------------------------------------------
    # Step 3: interface partner retention
    # ----------------------------------------------------------------
    if interface_policy == "keep_chain" and keep_partner_chain:
        keep.add(keep_partner_chain)
        report["interface_chains_retained"].append(keep_partner_chain)
    elif interface_policy == "keep_within_A" and len(structure) > 0:
        anchor_positions = _heavy_atom_positions(structure, keep)
        for chain in structure[0]:
            if chain.name in keep:
                continue
            for res in chain:
                if not _is_amino_acid(res):
                    continue
                for atom in res:
                    pos = atom.pos
                    if any(
                        _distance(pos, ref) <= interface_radius_A
                        for ref in anchor_positions
                    ):
                        keep.add(chain.name)
                        report["interface_chains_retained"].append(chain.name)
                        break
                else:
                    continue
                break

    # ----------------------------------------------------------------
    # Step 4: remove chains from all models
    # ----------------------------------------------------------------
    if len(structure) > 0:
        all_chains = {chain.name for chain in structure[0]}
        to_remove = all_chains - keep
        for model in structure:
            for name in list(to_remove):
                model.remove_chain(name)
        report["chains_removed"] = sorted(to_remove)
        report["chains_kept"] = sorted(keep & all_chains)

    logger.info(
        "[Chain selection] policy=%s  removed=%s  kept=%s",
        policy,
        report["chains_removed"],
        report.get("chains_kept", []),
    )
    return structure, report
