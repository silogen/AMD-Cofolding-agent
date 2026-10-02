"""Stage 2 — Chain / Entity Role Assignment.

Labels every chain and every HETATM group with a role. Never modifies
atom coordinates. Populates ambiguous_groups for the profile_preview.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import gemmi

from structure_prep.additives import (
    AdditivesRegistry,
    COFACTORS,
    DNA_RESNAMES,
    METAL_ELEMENTS,
    NUCLEIC_RESNAMES,
    RNA_RESNAMES,
    STANDARD_AA,
    WATER_RESNAMES,
    get_registry,
)
from structure_prep.models import (
    InventoryReport,
    RetainedMetal,
    RoleMap,
    Stage2Profile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ANTIBODY_RESIDUES = 80
_MAX_ANTIBODY_RESIDUES = 130
_CHOTHIA_CYS_SEPARATION_MIN = 60
_CHOTHIA_CYS_SEPARATION_MAX = 70
_AA_POLYMER_THRESHOLD = 0.85
_NUC_POLYMER_THRESHOLD = 0.70  # tRNA has ~80% standard bases; modified bases common → 70% threshold


# ---------------------------------------------------------------------------
# Chain-level helpers
# ---------------------------------------------------------------------------

def _residue_counts(residues: list[gemmi.Residue]) -> dict[str, int]:
    """Count polymer residue types in a list of residues."""
    counts: dict[str, int] = {"aa": 0, "dna": 0, "rna": 0, "nuc": 0, "other": 0}
    for res in residues:
        name = res.name.strip().upper()
        if name in STANDARD_AA:
            counts["aa"] += 1
        elif name in DNA_RESNAMES:
            counts["dna"] += 1
            counts["nuc"] += 1
        elif name in RNA_RESNAMES:
            counts["rna"] += 1
            counts["nuc"] += 1
        else:
            counts["other"] += 1
    return counts


def _collect_polymer_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    """Return residues that are part of the polymer entity (ATOM records)."""
    return [
        res for res in chain
        if res.entity_type == gemmi.EntityType.Polymer
    ]


def _cys_positions(polymer_residues: list[gemmi.Residue]) -> list[int]:
    """Return 0-based residue indices of CYS residues."""
    return [
        i for i, res in enumerate(polymer_residues)
        if res.name.strip().upper() in {"CYS", "CYX"}
    ]


def _has_chothia_cys_signature(polymer_residues: list[gemmi.Residue]) -> bool:
    """Simplified Chothia VH/VL check: two CYS separated by ~65 residues
    within a chain of 80–130 residues."""
    n = len(polymer_residues)
    if not (_MIN_ANTIBODY_RESIDUES <= n <= _MAX_ANTIBODY_RESIDUES):
        return False
    cys_idx = _cys_positions(polymer_residues)
    if len(cys_idx) < 2:
        return False
    # Check all pairs of CYS
    for i in range(len(cys_idx)):
        for j in range(i + 1, len(cys_idx)):
            sep = cys_idx[j] - cys_idx[i]
            if _CHOTHIA_CYS_SEPARATION_MIN <= sep <= _CHOTHIA_CYS_SEPARATION_MAX:
                return True
    return False


def _classify_chain(
    chain: gemmi.Chain,
    profile: Stage2Profile,
) -> tuple[str, dict[str, Any] | None]:
    """Classify a polymer chain and return (role, ambiguous_info | None).

    Priority:
    1. User override (target_chains / binder_chains)
    2. Polymer content rules
    """
    chain_id = chain.name

    # --- User overrides ---
    if profile.target_chains and chain_id in profile.target_chains:
        return "protein", None
    if profile.binder_chains and chain_id in profile.binder_chains:
        return "binder", None

    polymer_residues = _collect_polymer_residues(chain)

    if not polymer_residues:
        # No polymer residues — not a polymer chain, handled separately
        return "unknown_non_polymer", None

    counts = _residue_counts(polymer_residues)
    n_total = len(polymer_residues)

    aa_frac = counts["aa"] / n_total
    dna_frac = counts["dna"] / n_total
    rna_frac = counts["rna"] / n_total

    # Nucleic acid
    if dna_frac >= _NUC_POLYMER_THRESHOLD:
        return "dna", None
    if rna_frac >= _NUC_POLYMER_THRESHOLD:
        return "rna", None

    # Protein / peptide
    if aa_frac >= _AA_POLYMER_THRESHOLD:
        if n_total > profile.peptide_max_residues:
            # Long chain — could be antibody
            if profile.antibody_detection and _has_chothia_cys_signature(polymer_residues):
                ambiguous: dict[str, Any] = {
                    "type": "possible_antibody",
                    "chain": chain_id,
                    "n_residues": n_total,
                    "note": "Chothia CYS signature detected; classify as antibody?",
                    "suggested_role": "antibody",
                }
                return "antibody", ambiguous
            return "protein", None
        elif profile.peptide_min_residues <= n_total <= profile.peptide_max_residues:
            ambiguous = {
                "type": "short_peptide",
                "chain": chain_id,
                "n_residues": n_total,
                "note": (
                    f"Chain has {n_total} residues "
                    f"(peptide_min={profile.peptide_min_residues}, "
                    f"peptide_max={profile.peptide_max_residues}); "
                    "could be peptide_binder or small protein."
                ),
                "suggested_role": "peptide_binder",
            }
            return "peptide_binder", ambiguous
        else:
            # Shorter than peptide_min_residues — treat as peptide_binder
            ambiguous = {
                "type": "very_short_peptide",
                "chain": chain_id,
                "n_residues": n_total,
                "note": (
                    f"Chain has only {n_total} residue(s); may be a fragment."
                ),
                "suggested_role": "peptide_binder",
            }
            return "peptide_binder", ambiguous

    # Mixed / unknown polymer
    return "unknown_polymer", None


# ---------------------------------------------------------------------------
# HETATM helpers
# ---------------------------------------------------------------------------

def _hetatm_key(comp_id: str, chain: str, resseq: int) -> str:
    return f"{comp_id}:{chain}:{resseq}"


def _is_metal(comp_id: str, element: str) -> bool:
    cid = comp_id.strip().upper()
    el = element.strip().upper()
    return cid in METAL_ELEMENTS or el in METAL_ELEMENTS


def _distance(pos1: gemmi.Position, pos2: gemmi.Position) -> float:
    """Euclidean distance between two gemmi.Position objects."""
    dx = pos1.x - pos2.x
    dy = pos1.y - pos2.y
    dz = pos1.z - pos2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _nearest_protein_distance(
    metal_pos: gemmi.Position,
    protein_positions: list[tuple[gemmi.Position, str]],
) -> tuple[float, list[str]]:
    """Find nearest protein heavy-atom distance and coordinating residue labels.

    Parameters
    ----------
    metal_pos:
        Position of the metal atom.
    protein_positions:
        List of (Position, residue_label_str) for all protein heavy atoms.

    Returns
    -------
    (min_distance, coordinating_residue_labels)
        min_distance is ``float('inf')`` when protein_positions is empty.
    """
    if not protein_positions:
        return float("inf"), []

    min_dist = float("inf")
    coordinating: list[str] = []

    for pos, label in protein_positions:
        d = _distance(metal_pos, pos)
        if d < min_dist:
            min_dist = d

    return min_dist, coordinating


def _collect_protein_heavy_atom_positions(
    structure: gemmi.Structure,
    protein_chain_ids: set[str],
) -> list[tuple[gemmi.Position, str]]:
    """Collect (Position, label) for all heavy atoms in protein chains."""
    positions: list[tuple[gemmi.Position, str]] = []
    if len(structure) == 0:
        return positions

    model = structure[0]
    for chain in model:
        if chain.name not in protein_chain_ids:
            continue
        for res in chain:
            if res.entity_type != gemmi.EntityType.Polymer:
                continue
            res_label = f"{chain.name}:{res.name}:{res.seqid}"
            for atom in res:
                if atom.element == gemmi.Element("H") or atom.element == gemmi.Element("D"):
                    continue
                positions.append((atom.pos, res_label))
    return positions


def _collect_coordinating_residues(
    metal_pos: gemmi.Position,
    protein_positions: list[tuple[gemmi.Position, str]],
    cutoff: float,
) -> list[str]:
    """Return labels of protein residues within *cutoff* Å of metal_pos."""
    seen: set[str] = set()
    for pos, label in protein_positions:
        if _distance(metal_pos, pos) <= cutoff:
            if label not in seen:
                seen.add(label)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage2Profile,
    inventory: InventoryReport,
    registry: AdditivesRegistry | None = None,
) -> tuple[gemmi.Structure, RoleMap]:
    """Stage 2: Chain / Entity Role Assignment.

    Parameters
    ----------
    structure:
        Gemmi Structure (post Stage 1; not modified here).
    profile:
        Stage2Profile with role-classification parameters.
    inventory:
        Stage 0 InventoryReport providing HETATM group metadata.
    registry:
        Optional pre-loaded AdditivesRegistry.

    Returns
    -------
    (structure_unchanged, role_map)
        The same Structure object and a fully populated RoleMap.
    """
    reg = registry or get_registry()

    chain_roles: dict[str, str] = {}
    hetatm_roles: dict[str, str] = {}
    ambiguous_groups: list[dict[str, Any]] = []

    protein_chains: list[str] = []
    binder_chains_out: list[str] = []
    nucleic_acid_chains: list[str] = []

    # ------------------------------------------------------------------
    # 1. Classify polymer chains
    # ------------------------------------------------------------------
    if len(structure) > 0:
        model = structure[0]
        for chain in model:
            # Check if chain has any polymer residues
            polymer_residues = _collect_polymer_residues(chain)

            if not polymer_residues:
                # Non-polymer chain — roles assigned by HETATM loop below
                continue

            role, ambig = _classify_chain(chain, profile)

            chain_roles[chain.name] = role

            if role == "protein":
                protein_chains.append(chain.name)
            elif role in ("binder", "peptide_binder", "antibody"):
                binder_chains_out.append(chain.name)
            elif role == "dna":
                nucleic_acid_chains.append(chain.name)
            elif role == "rna":
                nucleic_acid_chains.append(chain.name)

            if ambig is not None:
                ambiguous_groups.append(ambig)

    # ------------------------------------------------------------------
    # 2. Collect protein atom positions for metal distance checks
    # ------------------------------------------------------------------
    protein_chain_set = set(protein_chains)
    protein_positions = _collect_protein_heavy_atom_positions(
        structure, protein_chain_set
    )

    # ------------------------------------------------------------------
    # 3. Classify HETATM groups
    # ------------------------------------------------------------------
    structural_metals: list[RetainedMetal] = []
    crystallographic_ions: list[dict[str, Any]] = []
    cutoff = profile.metal_coordination_cutoff_A

    # Build a quick index of HETATM residues from the first model
    # keyed by (comp_id, chain_name) → list[gemmi.Residue]
    hetatm_index: dict[tuple[str, str], list[gemmi.Residue]] = {}
    if len(structure) > 0:
        model = structure[0]
        for chain in model:
            for res in chain:
                if res.entity_type in (
                    gemmi.EntityType.NonPolymer,
                    gemmi.EntityType.Branched,
                    gemmi.EntityType.Water,
                ):
                    key = (res.name.strip().upper(), chain.name)
                    hetatm_index.setdefault(key, []).append(res)

    for grp in inventory.hetatm_groups:
        comp_id = grp.comp_id.strip().upper()
        chain_name = grp.chain

        # ------ water ------
        if comp_id in WATER_RESNAMES or grp.classification == "water":
            # Assign per-residue keys
            residues = hetatm_index.get((comp_id, chain_name), [])
            for res in residues:
                key = _hetatm_key(comp_id, chain_name, res.seqid.num)
                hetatm_roles[key] = "water"
            if not residues:
                # Fallback: group-level key
                hetatm_roles[_hetatm_key(comp_id, chain_name, 0)] = "water"
            continue

        # ------ metals ------
        # Determine element from first atom of first residue in group
        residues = hetatm_index.get((comp_id, chain_name), [])
        element_str = ""
        if residues and list(residues[0]):
            element_str = list(residues[0])[0].element.name.upper()

        if _is_metal(comp_id, element_str) and (
            not residues or all(len(list(r)) == 1 for r in residues)
        ):
            # Process each metal instance individually
            for res in residues:
                atom_list = list(res)
                if not atom_list:
                    continue
                metal_atom = atom_list[0]
                metal_pos = metal_atom.pos

                min_dist, _ = _nearest_protein_distance(metal_pos, protein_positions)
                coord_residues = _collect_coordinating_residues(
                    metal_pos, protein_positions, cutoff
                )

                key = _hetatm_key(comp_id, chain_name, res.seqid.num)

                if min_dist <= cutoff:
                    hetatm_roles[key] = "metal"
                    structural_metals.append(RetainedMetal(
                        comp_id=comp_id,
                        element=element_str,
                        chain=chain_name,
                        resseq=res.seqid.num,
                        classification="structural",
                        nearest_protein_distance_A=min_dist if min_dist != float("inf") else None,
                        coordinating_residues=coord_residues,
                    ))
                else:
                    hetatm_roles[key] = "crystallographic_ion"
                    crystallographic_ions.append({
                        "comp_id": comp_id,
                        "element": element_str,
                        "chain": chain_name,
                        "resseq": res.seqid.num,
                        "nearest_protein_distance_A": (
                            min_dist if min_dist != float("inf") else None
                        ),
                        "classification": "crystallographic",
                    })
            if not residues:
                # No residue data — fall back to group-level
                hetatm_roles[_hetatm_key(comp_id, chain_name, 0)] = "metal"
            continue

        # ------ cofactor ------
        if comp_id in COFACTORS:
            for res in residues:
                key = _hetatm_key(comp_id, chain_name, res.seqid.num)
                hetatm_roles[key] = "cofactor"
            if not residues:
                hetatm_roles[_hetatm_key(comp_id, chain_name, 0)] = "cofactor"
            continue

        # ------ additive registry ------
        if reg.is_additive(comp_id):
            for res in residues:
                key = _hetatm_key(comp_id, chain_name, res.seqid.num)
                hetatm_roles[key] = "additive"
            if not residues:
                hetatm_roles[_hetatm_key(comp_id, chain_name, 0)] = "additive"
            continue

        # ------ unknown / likely ligand ------
        ambig_info: dict[str, Any] = {
            "type": "unknown_hetatm",
            "comp_id": comp_id,
            "chain": chain_name,
            "n_instances": grp.n_instances,
            "classification_from_inventory": grp.classification,
            "note": (
                f"HETATM '{comp_id}' in chain '{chain_name}' could not be "
                "automatically classified; likely a ligand."
            ),
            "suggested_role": "likely_ligand",
        }
        ambiguous_groups.append(ambig_info)

        for res in residues:
            key = _hetatm_key(comp_id, chain_name, res.seqid.num)
            hetatm_roles[key] = "likely_ligand"
        if not residues:
            hetatm_roles[_hetatm_key(comp_id, chain_name, 0)] = "likely_ligand"

    # ------------------------------------------------------------------
    # 4. Assemble RoleMap
    # ------------------------------------------------------------------
    role_map = RoleMap(
        protein_chains=sorted(set(protein_chains)),
        binder_chains=sorted(set(binder_chains_out)),
        nucleic_acid_chains=sorted(set(nucleic_acid_chains)),
        chain_roles=chain_roles,
        hetatm_roles=hetatm_roles,
        structural_metals=structural_metals,
        crystallographic_ions=crystallographic_ions,
        ambiguous_groups=ambiguous_groups,
    )

    logger.info(
        "Stage 2: %d protein chain(s), %d binder chain(s), "
        "%d nucleic acid chain(s), %d HETATM role(s), "
        "%d structural metal(s), %d crystallographic ion(s), "
        "%d ambiguous group(s).",
        len(role_map.protein_chains),
        len(role_map.binder_chains),
        len(role_map.nucleic_acid_chains),
        len(role_map.hetatm_roles),
        len(role_map.structural_metals),
        len(role_map.crystallographic_ions),
        len(role_map.ambiguous_groups),
    )

    return structure, role_map


__all__ = ["run"]
