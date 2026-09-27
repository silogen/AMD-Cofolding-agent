"""Stage 3 — Curation and Retention Policy.

Removes entities per the profile policy.  Records every removal/retention
decision in a curation_log.  Never silently drops anything.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import gemmi

from structure_prep.additives import (
    COFACTORS,
    METAL_ELEMENTS,
    NUCLEIC_RESNAMES,
    STANDARD_AA,
    WATER_RESNAMES,
    get_registry,
)
from structure_prep.models import CurationEntry, HetAtmGroup, RoleMap, Stage3Profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small residue-level helpers
# ---------------------------------------------------------------------------

def _hetatm_key(res: gemmi.Residue, chain_name: str) -> str:
    """Return the hetatm_roles lookup key: 'COMP:CHAIN:RESSEQ'."""
    return f"{res.name}:{chain_name}:{res.seqid.num}"


def _res_is_polymer(res: gemmi.Residue) -> bool:
    """Return True for polymer (protein / nucleic-acid) residues."""
    try:
        return res.entity_type == gemmi.EntityType.Polymer
    except Exception:
        return res.name in STANDARD_AA or res.name in NUCLEIC_RESNAMES


def _res_is_metal_hetatm(res: gemmi.Residue) -> bool:
    """Heuristic: single-atom residue whose element / comp_id is a known metal."""
    if len(res) != 1:
        return False
    atom = res[0]
    try:
        el = atom.element.name.upper()
    except Exception:
        el = ""
    return el in METAL_ELEMENTS or res.name.upper() in METAL_ELEMENTS


def _mean_occ(res: gemmi.Residue) -> float:
    """Return mean occupancy across all atoms in a residue (0.0 if empty)."""
    n = len(res)
    return sum(a.occ for a in res) / n if n else 0.0


def _mean_bfactor(res: gemmi.Residue) -> float:
    """Return mean isotropic B-factor (inf if empty)."""
    n = len(res)
    return sum(a.b_iso for a in res) / n if n else float("inf")


# ---------------------------------------------------------------------------
# Per-residue occupancy lookup — used by 'primary' ligand policy
# ---------------------------------------------------------------------------

def _residue_occ(
    st: gemmi.Structure,
    comp_id: str,
    chain_name: str,
    seqnum: int,
) -> float:
    """Return mean occupancy for a specific residue; 0.0 if not found."""
    for model in st:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.name == comp_id and res.seqid.num == seqnum:
                    return _mean_occ(res)
    return 0.0


# ---------------------------------------------------------------------------
# Ligand keep-set builder
# ---------------------------------------------------------------------------

def _build_ligand_keep_set(
    profile: Stage3Profile,
    role_map: RoleMap,
    inventory_hetatm: list[HetAtmGroup],
    st: gemmi.Structure,
) -> set[str]:
    """Return the set of hetatm_roles keys for ligands to KEEP.

    Considers both 'likely_ligand' and 'covalent_ligand' roles.
    """
    candidates: dict[str, str] = {
        k: v
        for k, v in role_map.hetatm_roles.items()
        if v in ("likely_ligand", "covalent_ligand")
    }
    exclude: set[str] = set(profile.ligand_exclude_comp_ids)
    policy = profile.ligand_policy

    def _comp(k: str) -> str:
        return k.split(":")[0]

    def _chain(k: str) -> str:
        parts = k.split(":")
        return parts[1] if len(parts) > 1 else ""

    if policy == "none":
        return set()

    if policy == "all":
        return {k for k in candidates if _comp(k) not in exclude}

    if policy == "listed":
        listed: set[str] = set(profile.ligand_comp_ids)
        # "listed" overrides auto-classification: search ALL HETATM roles,
        # not just likely_ligand/covalent_ligand. This allows explicitly
        # listing cofactors (e.g. BTN in streptavidin) or unknown compounds
        # as the ligand of interest regardless of how Stage 2 classified them.
        all_hetatm_keys = set(role_map.hetatm_roles.keys())
        return {
            k
            for k in all_hetatm_keys
            if _comp(k) in listed and _comp(k) not in exclude
        }

    if policy == "primary":
        # Optionally narrow to a specific chain first
        filtered = dict(candidates)
        if profile.ligand_chain:
            chain_filtered = {
                k: v for k, v in candidates.items()
                if _chain(k) == profile.ligand_chain
            }
            if chain_filtered:  # fall back to all candidates if filter yields nothing
                filtered = chain_filtered
            else:
                logger.warning(
                    "Stage 3: ligand_chain=%r produced no candidates; "
                    "falling back to all chains.",
                    profile.ligand_chain,
                )

        # Sort by individual residue occupancy (highest first)
        def _occ_key(k: str) -> float:
            parts = k.split(":")
            if len(parts) < 3:
                return 0.0
            try:
                return _residue_occ(st, parts[0], parts[1], int(parts[2]))
            except (ValueError, IndexError):
                return 0.0

        sorted_keys = sorted(filtered.keys(), key=_occ_key, reverse=True)
        if not sorted_keys:
            return set()

        idx = 0
        if profile.ligand_instance_index is not None:
            idx = min(profile.ligand_instance_index, len(sorted_keys) - 1)
        return {sorted_keys[idx]}

    # Unknown policy — retain all as a safe default
    logger.warning(
        "Stage 3: unrecognised ligand_policy=%r; treating as 'all'.", policy
    )
    return set(candidates.keys())


# ---------------------------------------------------------------------------
# Ligand-position extractor (for pocket-water calculation)
# ---------------------------------------------------------------------------

def _extract_ligand_positions(
    st: gemmi.Structure,
    ligand_keep_keys: set[str],
) -> list[tuple[float, float, float]]:
    """Return (x, y, z) for every heavy atom of every retained ligand residue."""
    positions: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                key = _hetatm_key(res, chain.name)
                if key not in ligand_keep_keys:
                    continue
                for atom in res:
                    try:
                        el = atom.element.name.upper()
                    except Exception:
                        el = ""
                    if el != "H":
                        p = atom.pos
                        positions.append((p.x, p.y, p.z))
    return positions


# ---------------------------------------------------------------------------
# Pocket-water set builder
# ---------------------------------------------------------------------------

def _build_pocket_water_keys(
    st: gemmi.Structure,
    ligand_positions: list[tuple[float, float, float]],
    radius: float,
) -> set[str]:
    """Return keys for water residues within *radius* Å of any ligand heavy atom."""
    if not ligand_positions:
        return set()
    r2 = radius * radius
    keys: set[str] = set()
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in WATER_RESNAMES:
                    continue
                for atom in res:
                    px, py, pz = atom.pos.x, atom.pos.y, atom.pos.z
                    for lx, ly, lz in ligand_positions:
                        if (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2 <= r2:
                            keys.add(_hetatm_key(res, chain.name))
                            break
                    else:
                        continue
                    break  # residue matched — move on
    return keys


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage3Profile,
    role_map: RoleMap,
    inventory_hetatm: list,  # list[HetAtmGroup]
    registry=None,
) -> tuple[gemmi.Structure, dict]:
    """Stage 3: Curation and Retention Policy.

    Returns (pruned_structure, stage3_report_dict).

    stage3_report_dict keys
    -----------------------
    curation_log:              list[CurationEntry]
    n_waters_retained:         int
    removed_additives:         list[dict]
    ligand_residues_retained:  list[dict]  {comp_id, chain, resseq, occupancy}
    metals_retained:           list[dict]  {comp_id, chain, resseq, element}
    """
    if registry is None:
        registry = get_registry(profile.additive_registry_path)

    # Work on a deep copy — Stage 3 must not mutate the caller's structure
    st = structure.clone()

    # ------------------------------------------------------------------
    # Pre-compute ligand keep-set
    # ------------------------------------------------------------------
    ligand_keep_keys = _build_ligand_keep_set(
        profile, role_map, inventory_hetatm, st
    )
    logger.info(
        "Stage 3: ligand_policy=%r → %d ligand residue(s) will be kept.",
        profile.ligand_policy,
        len(ligand_keep_keys),
    )

    # ------------------------------------------------------------------
    # Pre-compute pocket-water keys (must happen before any deletion)
    # ------------------------------------------------------------------
    pocket_water_keys: set[str] = set()
    if profile.keep_waters == "pocket":
        lig_positions = _extract_ligand_positions(st, ligand_keep_keys)
        pocket_water_keys = _build_pocket_water_keys(
            st, lig_positions, profile.water_pocket_radius_A
        )
        logger.info(
            "Stage 3: pocket water scan found %d water residues within %.1f Å.",
            len(pocket_water_keys),
            profile.water_pocket_radius_A,
        )

    # ------------------------------------------------------------------
    # Structural-metal lookup set
    # ------------------------------------------------------------------
    structural_metal_keys: set[str] = {
        f"{m.comp_id}:{m.chain}:{m.resseq}"
        for m in role_map.structural_metals
    }

    # ------------------------------------------------------------------
    # Accumulate report fields
    # ------------------------------------------------------------------
    log: list[CurationEntry] = []
    removed_additives: list[dict[str, Any]] = []
    ligand_residues_retained: list[dict[str, Any]] = []
    metals_retained: list[dict[str, Any]] = []
    n_waters_retained = 0

    # ------------------------------------------------------------------
    # Main curation loop
    # ------------------------------------------------------------------
    for model in st:
        for chain in model:
            to_delete: list[int] = []

            for i, res in enumerate(chain):
                # -------------------------------------------------------
                # Skip polymer (protein / nucleic-acid) residues entirely.
                # They are never removed by Stage 3 and logging each residue
                # would be extremely verbose.
                # -------------------------------------------------------
                if _res_is_polymer(res):
                    continue

                key = _hetatm_key(res, chain.name)

                # Resolve role — prefer the Stage 2 role_map, fall back to
                # name/element-based heuristics for residues not captured there.
                role = role_map.hetatm_roles.get(key, "")
                if not role:
                    if res.name in WATER_RESNAMES:
                        role = "water"
                    elif _res_is_metal_hetatm(res):
                        role = "metal"
                    elif res.name in COFACTORS:
                        role = "cofactor"
                    elif registry.is_additive(res.name):
                        role = "additive"
                    else:
                        role = "unknown"

                remove = False
                reason = ""
                rule = ""

                # -------------------------------------------------------
                # WATER
                # -------------------------------------------------------
                if role == "water" or res.name in WATER_RESNAMES:
                    kw = profile.keep_waters
                    if kw == "none":
                        remove = True
                        reason = "keep_waters=none"
                        rule = "keep_waters:none"
                    elif kw == "pocket":
                        if key in pocket_water_keys:
                            reason = (
                                f"pocket water ≤{profile.water_pocket_radius_A:.1f} Å "
                                "of retained ligand"
                            )
                            rule = "keep_waters:pocket:in"
                        else:
                            remove = True
                            reason = "water outside pocket radius"
                            rule = "keep_waters:pocket:out"
                    elif kw == "structural":
                        avg_b = _mean_bfactor(res)
                        if avg_b < profile.water_structural_bfactor_cutoff:
                            reason = (
                                f"structural water (B={avg_b:.1f} < "
                                f"{profile.water_structural_bfactor_cutoff})"
                            )
                            rule = "keep_waters:structural:in"
                        else:
                            remove = True
                            reason = (
                                f"water B={avg_b:.1f} ≥ cutoff "
                                f"{profile.water_structural_bfactor_cutoff}"
                            )
                            rule = "keep_waters:structural:out"
                    else:  # "all"
                        reason = "keep_waters=all"
                        rule = "keep_waters:all"

                # -------------------------------------------------------
                # METAL (includes structural_metal, crystallographic_ion roles)
                # -------------------------------------------------------
                elif role in ("metal", "structural_metal", "crystallographic_ion") or _res_is_metal_hetatm(res):
                    km = profile.keep_metals
                    if km == "none":
                        remove = True
                        reason = "keep_metals=none"
                        rule = "keep_metals:none"
                    elif km == "all":
                        reason = "keep_metals=all"
                        rule = "keep_metals:all"
                    elif km == "structural":
                        if key in structural_metal_keys:
                            reason = "metal in coordination sphere (structural_metals list)"
                            rule = "keep_metals:structural:in"
                        else:
                            remove = True
                            reason = "metal absent from structural_metals list"
                            rule = "keep_metals:structural:out"
                    elif isinstance(km, list):
                        el = ""
                        if len(res) > 0:
                            try:
                                el = res[0].element.name.upper()
                            except Exception:
                                el = ""
                        allowed = [e.upper() for e in km]
                        if el in allowed or res.name.upper() in allowed:
                            reason = f"metal element {el!r} in keep_metals list"
                            rule = "keep_metals:list:in"
                        else:
                            remove = True
                            reason = f"metal element {el!r} not in keep_metals list"
                            rule = "keep_metals:list:out"
                    else:
                        # Unrecognised setting — keep safely
                        reason = f"metal retained (unrecognised keep_metals={km!r})"
                        rule = "keep_metals:unknown"

                # -------------------------------------------------------
                # LIGAND (likely_ligand / covalent_ligand)
                # -------------------------------------------------------
                elif role in ("likely_ligand", "covalent_ligand"):
                    if key in ligand_keep_keys:
                        reason = f"ligand retained (ligand_policy={profile.ligand_policy!r})"
                        rule = f"ligand_policy:{profile.ligand_policy}:in"
                    else:
                        remove = True
                        if res.name in profile.ligand_exclude_comp_ids:
                            reason = "ligand comp_id in ligand_exclude_comp_ids"
                            rule = "ligand_policy:excluded"
                        else:
                            reason = (
                                f"ligand not selected by "
                                f"ligand_policy={profile.ligand_policy!r}"
                            )
                            rule = f"ligand_policy:{profile.ligand_policy}:out"

                # -------------------------------------------------------
                # COFACTOR
                # -------------------------------------------------------
                elif role == "cofactor" or res.name in COFACTORS:
                    # If the agent explicitly listed this comp_id via
                    # ligand_policy="listed" + ligand_comp_ids, treat it as
                    # a ligand of interest regardless of cofactor classification
                    # (e.g. BTN in streptavidin, ATP in a kinase active-site study).
                    if key in ligand_keep_keys:
                        reason = f"cofactor {res.name!r} promoted to ligand (explicit ligand_comp_ids)"
                        rule = "ligand_policy:listed:override_cofactor"
                        ligand_residues_retained.append({
                            "comp_id": res.name,
                            "chain": chain.name,
                            "resseq": res.seqid.num,
                            "occupancy": (sum(a.occ for a in res) / max(len(list(res)), 1)),
                        })
                        log.append(CurationEntry(entity=key, decision="retained",
                                                  reason=reason, rule=rule))
                        continue
                    kc = profile.keep_cofactors
                    if kc is True:
                        reason = "keep_cofactors=True"
                        rule = "keep_cofactors:all"
                    elif kc is False:
                        remove = True
                        reason = "keep_cofactors=False"
                        rule = "keep_cofactors:none"
                    elif isinstance(kc, list):
                        if res.name in kc:
                            reason = f"cofactor {res.name!r} in keep_cofactors list"
                            rule = "keep_cofactors:list:in"
                        else:
                            remove = True
                            reason = f"cofactor {res.name!r} not in keep_cofactors list"
                            rule = "keep_cofactors:list:out"
                    else:
                        reason = f"cofactor retained (unrecognised keep_cofactors={kc!r})"
                        rule = "keep_cofactors:unknown"

                # -------------------------------------------------------
                # GLYCAN
                # -------------------------------------------------------
                elif role == "glycan":
                    if profile.keep_glycans:
                        reason = "keep_glycans=True"
                        rule = "keep_glycans:true"
                    else:
                        remove = True
                        reason = "keep_glycans=False"
                        rule = "keep_glycans:false"

                # -------------------------------------------------------
                # ADDITIVE / BUFFER_ION / PHASING_AGENT
                # -------------------------------------------------------
                elif (
                    role in ("additive", "buffer_ion", "phasing_agent")
                    or registry.is_additive(res.name)
                ):
                    ka = profile.keep_additives
                    if ka is True:
                        reason = "keep_additives=True"
                        rule = "keep_additives:all"
                    elif ka is False:
                        remove = True
                        reason = f"additive removed (role={role!r}, keep_additives=False)"
                        rule = "keep_additives:none"
                    elif isinstance(ka, list):
                        if res.name in ka:
                            reason = f"additive {res.name!r} in keep_additives list"
                            rule = "keep_additives:list:in"
                        else:
                            remove = True
                            reason = f"additive {res.name!r} not in keep_additives list"
                            rule = "keep_additives:list:out"
                    else:
                        reason = f"additive retained (unrecognised keep_additives={ka!r})"
                        rule = "keep_additives:unknown"

                # -------------------------------------------------------
                # UNKNOWN — retain and log for human review
                # -------------------------------------------------------
                else:
                    reason = f"unknown role={role!r}; retained by default"
                    rule = "unknown:retained"

                # -------------------------------------------------------
                # Commit decision
                # -------------------------------------------------------
                if remove:
                    to_delete.append(i)
                    log.append(
                        CurationEntry(
                            entity=key,
                            decision="removed",
                            reason=reason,
                            rule=rule,
                        )
                    )
                    # Track additive-family removals separately for the report
                    if role in ("additive", "buffer_ion", "phasing_agent"):
                        removed_additives.append(
                            {
                                "comp_id": res.name,
                                "chain": chain.name,
                                "resseq": res.seqid.num,
                                "role": role,
                                "reason": reason,
                            }
                        )
                else:
                    log.append(
                        CurationEntry(
                            entity=key,
                            decision="retained",
                            reason=reason,
                            rule=rule,
                        )
                    )
                    # Populate sub-report fields
                    if res.name in WATER_RESNAMES:
                        n_waters_retained += 1
                    elif role in ("likely_ligand", "covalent_ligand"):
                        ligand_residues_retained.append(
                            {
                                "comp_id": res.name,
                                "chain": chain.name,
                                "resseq": res.seqid.num,
                                "occupancy": round(_mean_occ(res), 3),
                            }
                        )
                    elif (
                        role in ("metal", "structural_metal", "crystallographic_ion")
                        or _res_is_metal_hetatm(res)
                    ):
                        el = ""
                        if len(res) > 0:
                            try:
                                el = res[0].element.name.upper()
                            except Exception:
                                el = ""
                        metals_retained.append(
                            {
                                "comp_id": res.name,
                                "chain": chain.name,
                                "resseq": res.seqid.num,
                                "element": el,
                            }
                        )

            # Delete in reverse order to avoid index drift
            for idx in reversed(to_delete):
                del chain[idx]

        # Remove chains that became empty after deletion
        # Remove chains that became empty after residue deletion (gemmi 0.6+)
        empty = [ch.name for ch in model if len(list(ch)) == 0]
        for _ch_name in empty:
            model.remove_chain(_ch_name)

    logger.info(
        "Stage 3 complete: %d entity decisions logged, %d removed, "
        "%d ligands retained, %d metals retained, %d waters retained.",
        len(log),
        sum(1 for e in log if e.decision == "removed"),
        len(ligand_residues_retained),
        len(metals_retained),
        n_waters_retained,
    )

    return st, {
        "curation_log": log,
        "n_waters_retained": n_waters_retained,
        "removed_additives": removed_additives,
        "ligand_residues_retained": ligand_residues_retained,
        "metals_retained": metals_retained,
    }


__all__ = ["run"]
