"""Stage 0 — Triage and Inventory.

Parses the input PDB or CIF file using gemmi WITHOUT modifying it.
Produces a complete InventoryReport before any pipeline stage mutates
anything. Also writes inventory.json atomically to the job directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import gemmi

from structure_prep.additives import (
    AdditivesRegistry,
    METAL_ELEMENTS,
    NUCLEIC_RESNAMES,
    DNA_RESNAMES,
    RNA_RESNAMES,
    STANDARD_AA,
    WATER_RESNAMES,
    classify_hetatm,
    get_registry,
)
from structure_prep.models import (
    AltlocPosition,
    ChainInfo,
    CovalentLink,
    DisulfideBond,
    HetAtmGroup,
    InventoryReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_format(path: Path) -> str:
    """Return 'pdb' or 'cif' based on file suffix."""
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    # .pdb, .ent, .brk, anything else → pdb
    return "pdb"


def _one_letter(resname: str) -> str:
    """Convert 3-letter residue name to 1-letter code (protein only)."""
    _AA3TO1: dict[str, str] = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        # nonstandard common ones
        "MSE": "M", "SEC": "U", "PYL": "O",
        "HID": "H", "HIE": "H", "HIP": "H",
        "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "ARN": "R",
    }
    return _AA3TO1.get(resname.strip().upper(), "X")


def _classify_entity_type(residues: list[gemmi.Residue]) -> str:
    """Infer entity type from residue names in a chain segment."""
    if not residues:
        return "non-polymer"

    names = [r.name.strip().upper() for r in residues]
    n_total = len(names)

    n_water = sum(1 for n in names if n in WATER_RESNAMES)
    if n_water == n_total:
        return "water"

    n_aa = sum(1 for n in names if n in STANDARD_AA)
    n_nuc = sum(1 for n in names if n in NUCLEIC_RESNAMES)

    if n_aa / n_total >= 0.85:
        return "polymer"
    if n_nuc / n_total >= 0.85:
        return "polymer"

    # Mixed or small: check if majority is polymer-like
    if (n_aa + n_nuc) / n_total >= 0.70:
        return "polymer"

    return "non-polymer"


def _build_sequence_from_atom(residues: list[gemmi.Residue]) -> str:
    """Build one-letter sequence from observed ATOM residues."""
    parts: list[str] = []
    for res in residues:
        name = res.name.strip().upper()
        if name in STANDARD_AA:
            parts.append(_one_letter(name))
        elif name in NUCLEIC_RESNAMES:
            # nucleotide: use first letter of standard name
            parts.append(name[0] if name else "N")
    return "".join(parts)


def _build_sequence_from_entity(entity: gemmi.Entity) -> str | None:
    """Extract SEQRES sequence from a gemmi Entity."""
    if entity.entity_type not in (
        gemmi.EntityType.Polymer,
    ):
        return None
    monomer_list = entity.full_sequence  # list of str (3-letter codes or empty)
    if not monomer_list:
        return None
    parts: list[str] = []
    for name in monomer_list:
        n = name.strip().upper()
        if n in STANDARD_AA:
            parts.append(_one_letter(n))
        elif n in NUCLEIC_RESNAMES:
            parts.append(n[0] if n else "N")
        else:
            parts.append("X")
    return "".join(parts)


def _parse_modres_remarks(raw_remarks: str) -> list[dict[str, Any]]:
    """Extract MODRES records from PDB REMARK section."""
    results: list[dict[str, Any]] = []
    # MODRES line format (PDB v3):
    # MODRES  idCode  resName  chainID  seqNum  iCode  stdRes  comment
    pattern = re.compile(
        r"^MODRES\s+"
        r"(\S+)\s+"        # idCode
        r"(\S+)\s+"        # resName
        r"(\S)\s+"         # chainID
        r"(\s*\d+)\s*"     # seqNum
        r"(\S?)\s+"        # iCode
        r"(\S+)\s*"        # stdRes
        r"(.*?)\s*$",      # comment
        re.MULTILINE,
    )
    for m in pattern.finditer(raw_remarks):
        results.append({
            "pdb_id": m.group(1),
            "resname": m.group(2).strip(),
            "chain": m.group(3).strip(),
            "resseq": int(m.group(4).strip()),
            "icode": m.group(5).strip(),
            "std_resname": m.group(6).strip(),
            "comment": m.group(7).strip(),
        })
    return results


def _assembly_info(structure: gemmi.Structure) -> dict[str, Any]:
    """Summarise available biological assemblies."""
    info: dict[str, Any] = {}
    for asm in structure.assemblies:
        generators: list[dict[str, Any]] = []
        for gen in asm.generators:
            generators.append({
                "chains": list(gen.chains),
                "subchains": list(gen.subchains),
                "n_operators": len(gen.operators),
            })
        info[asm.name] = {
            "name": asm.name,
            "n_generators": len(asm.generators),
            "generators": generators,
        }
    return info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    job_dir: Path,
    registry: AdditivesRegistry | None = None,
) -> tuple[gemmi.Structure, InventoryReport]:
    """Stage 0: Triage and Inventory.

    Parameters
    ----------
    input_path:
        Path to the raw PDB or mmCIF file.
    job_dir:
        Working directory for this job. ``inventory.json`` will be written here.
    registry:
        Optional pre-loaded AdditivesRegistry.  If None, the module singleton
        is used (loaded on first call).

    Returns
    -------
    (structure, inventory_report)
        The parsed gemmi Structure (NOT modified) and a complete InventoryReport.
    """
    input_path = Path(input_path)
    job_dir = Path(job_dir)
    reg = registry or get_registry()

    # ------------------------------------------------------------------
    # 1. Format detection & checksum
    # ------------------------------------------------------------------
    fmt = _detect_format(input_path)
    logger.info("Stage 0: parsing %s as %s", input_path, fmt)
    checksum = _sha256_of_file(input_path)

    # ------------------------------------------------------------------
    # 2. Parse
    # ------------------------------------------------------------------
    structure = gemmi.read_structure(str(input_path))
    # gemmi leaves Residue.entity_type as Unknown until entities are set up.
    # The chain and HETATM classification below filters on that field, so
    # without this every chain reports zero polymer residues and no ligands.
    structure.setup_entities()

    # ------------------------------------------------------------------
    # 3. Basic model metadata
    # ------------------------------------------------------------------
    n_models = len(structure)
    model_ids: list[str] = [model.name for model in structure]

    format_warnings: list[str] = []

    # Detect duplicate chain IDs across the first model
    if n_models > 0:
        first_model = structure[0]
        all_chain_ids = [ch.name for ch in first_model]
        seen: set[str] = set()
        for cid in all_chain_ids:
            if cid in seen:
                format_warnings.append(f"Duplicate chain ID '{cid}' in first model.")
            seen.add(cid)

    # ------------------------------------------------------------------
    # 4. Build entity lookup  (entity_name → Entity)
    # ------------------------------------------------------------------
    entity_map: dict[str, gemmi.Entity] = {}
    for ent in structure.entities:
        entity_map[ent.name] = ent

    # ------------------------------------------------------------------
    # 5. Chain info
    # ------------------------------------------------------------------
    chains: list[ChainInfo] = []

    if n_models > 0:
        first_model = structure[0]
        for chain in first_model:
            # Separate ATOM (polymer) from HETATM residues
            atom_residues: list[gemmi.Residue] = []
            for res in chain:
                if res.entity_type in (
                    gemmi.EntityType.Polymer,
                    gemmi.EntityType.Unknown,
                ):
                    atom_residues.append(res)

            all_residues = list(chain)

            # Entity type detection based on residue content
            entity_type_str = _classify_entity_type(all_residues)

            # Finer-grained water check
            all_names = [r.name.strip().upper() for r in all_residues]
            if all_names and all(n in WATER_RESNAMES for n in all_names):
                entity_type_str = "water"
                # Reassign so ChainInfo Literal is satisfied
                chain_entity_type = "water"
            else:
                # map our internal string to ChainInfo Literal
                chain_entity_type = entity_type_str  # "polymer" | "non-polymer" | "water"

            # If gemmi entity knows better, use it
            # Look up entity by matching subchain IDs (gemmi 0.6+)
            ent = None
            for _span in chain.subchains():
                _sc_id = _span.subchain_id()
                for _entity in structure.entities:
                    if _sc_id in _entity.subchains:
                        ent = _entity
                        break
                if ent is not None:
                    break
            if ent is not None:
                et = ent.entity_type
                if et == gemmi.EntityType.Polymer:
                    chain_entity_type = "polymer"
                elif et == gemmi.EntityType.Water:
                    chain_entity_type = "water"
                elif et == gemmi.EntityType.NonPolymer:
                    chain_entity_type = "non-polymer"
                elif et == gemmi.EntityType.Branched:
                    chain_entity_type = "branched"

            # Polymer residue count (ATOM records — entity_type Polymer)
            polymer_residues = [
                r for r in chain
                if r.entity_type == gemmi.EntityType.Polymer
            ]
            n_residues_atom = len(polymer_residues)

            # SEQRES sequence from entity
            seqres_seq: str | None = None
            n_residues_seqres = 0
            has_seqres = False
            if ent is not None:
                seqres_seq = _build_sequence_from_entity(ent)
                if seqres_seq is not None:
                    n_residues_seqres = len(seqres_seq)
                    has_seqres = True

            # ATOM sequence
            atom_seq = _build_sequence_from_atom(polymer_residues)

            chains.append(ChainInfo(
                id=chain.name,
                entity_type=chain_entity_type,  # type: ignore[arg-type]
                n_residues_atom=n_residues_atom,
                n_residues_seqres=n_residues_seqres,
                has_seqres=has_seqres,
                sequence_from_atom=atom_seq,
                sequence_from_seqres=seqres_seq,
            ))

    # ------------------------------------------------------------------
    # 6. HETATM groups
    # ------------------------------------------------------------------
    # Group by (comp_id, chain_name): collect residues for occupancy/Bfactor
    hetatm_dict: dict[tuple[str, str], list[gemmi.Residue]] = {}

    if n_models > 0:
        first_model = structure[0]
        for chain in first_model:
            for res in chain:
                if res.entity_type in (
                    gemmi.EntityType.NonPolymer,
                    gemmi.EntityType.Branched,
                ):
                    key = (res.name.strip().upper(), chain.name)
                    hetatm_dict.setdefault(key, []).append(res)
                elif res.entity_type == gemmi.EntityType.Water:
                    key = (res.name.strip().upper(), chain.name)
                    hetatm_dict.setdefault(key, []).append(res)

    hetatm_groups: list[HetAtmGroup] = []
    for (comp_id, chain_name), residues in hetatm_dict.items():
        # occupancy and B-factor
        occupancies: list[float] = []
        bfactors: list[float] = []
        n_atoms_per_res: list[int] = []

        for res in residues:
            atom_list = list(res)
            n_atoms_per_res.append(len(atom_list))
            for atom in atom_list:
                occupancies.append(atom.occ)
                bfactors.append(atom.b_iso)

        mean_occ: float | None = (
            sum(occupancies) / len(occupancies) if occupancies else None
        )
        mean_bf: float | None = (
            sum(bfactors) / len(bfactors) if bfactors else None
        )

        # Element from first atom of first residue
        element_str = ""
        if residues and list(residues[0]):
            first_atom = list(residues[0])[0]
            element_str = first_atom.element.name.upper()

        # Average n_atoms per residue instance
        avg_n_atoms = (
            round(sum(n_atoms_per_res) / len(n_atoms_per_res))
            if n_atoms_per_res else 1
        )

        raw_cls = classify_hetatm(
            comp_id=comp_id,
            element=element_str,
            n_atoms_in_residue=avg_n_atoms,
            registry=reg,
        )

        # Map raw classification to HetAtmGroup Literal
        cls_map: dict[str, str] = {
            "water": "water",
            "metal": "metal",
            "phasing_agent": "crystallographic_ion",
            "buffer_ion": "crystallographic_ion",
            "cofactor": "cofactor",
            "additive": "additive",
            "unknown": "unknown",
        }
        classification = cls_map.get(raw_cls, "unknown")

        in_registry = reg.is_additive(comp_id)
        registry_cat = reg.get_category(comp_id) if in_registry else None

        hetatm_groups.append(HetAtmGroup(
            comp_id=comp_id,
            chain=chain_name,
            n_instances=len(residues),
            mean_occupancy=mean_occ,
            mean_bfactor=mean_bf,
            classification=classification,  # type: ignore[arg-type]
            in_additive_registry=in_registry,
            registry_category=registry_cat,
        ))

    # ------------------------------------------------------------------
    # 7. Altloc positions
    # ------------------------------------------------------------------
    altloc_positions: list[AltlocPosition] = []

    if n_models > 0:
        first_model = structure[0]
        for chain in first_model:
            for residue in chain:
                altloc_codes: set[str] = set()
                for atom in residue:
                    al = atom.altloc
                    if al not in (" ", "\x00", ""):
                        altloc_codes.add(al)
                if altloc_codes:
                    # Compute per-altloc mean occupancy
                    occ_by_altloc: dict[str, list[float]] = {}
                    for atom in residue:
                        al = atom.altloc
                        if al in altloc_codes:
                            occ_by_altloc.setdefault(al, []).append(atom.occ)

                    mean_occ_by_altloc: dict[str, float] = {
                        al: (sum(v) / len(v) if v else 0.0)
                        for al, v in occ_by_altloc.items()
                    }

                    altloc_positions.append(AltlocPosition(
                        chain=chain.name,
                        resseq=residue.seqid.num,
                        resname=residue.name,
                        altloc_codes=sorted(altloc_codes),
                        occupancies=mean_occ_by_altloc,
                        near_ligand=False,  # populated later in Stage 4
                    ))

    # ------------------------------------------------------------------
    # 8. Disulfide bonds
    # ------------------------------------------------------------------
    disulfide_bonds: list[DisulfideBond] = []

    for conn in structure.connections:
        if conn.type == gemmi.ConnectionType.Disulf:
            p1 = conn.partner1
            p2 = conn.partner2
            disulfide_bonds.append(DisulfideBond(
                chain1=p1.chain_name,
                resseq1=p1.res_id.seqid.num,
                chain2=p2.chain_name,
                resseq2=p2.res_id.seqid.num,
                distance_A=None,  # distances not pre-computed at Stage 0
            ))

    # ------------------------------------------------------------------
    # 9. Covalent links
    # ------------------------------------------------------------------
    covalent_links: list[CovalentLink] = []

    for conn in structure.connections:
        if conn.type == gemmi.ConnectionType.Covale:
            p1 = conn.partner1
            p2 = conn.partner2
            resname1 = ""
            resname2 = ""
            # Try to look up residue names from the structure
            if n_models > 0:
                try:
                    r1 = structure[0].find_residue_group(
                        p1.chain_name, p1.res_id
                    )[0]
                    resname1 = r1.name
                except Exception:
                    pass
                try:
                    r2 = structure[0].find_residue_group(
                        p2.chain_name, p2.res_id
                    )[0]
                    resname2 = r2.name
                except Exception:
                    pass

            involves_ligand = (
                resname1.strip().upper() not in STANDARD_AA
                or resname2.strip().upper() not in STANDARD_AA
            )

            covalent_links.append(CovalentLink(
                resname1=resname1,
                chain1=p1.chain_name,
                resseq1=p1.res_id.seqid.num,
                atom1=p1.atom_name,
                resname2=resname2,
                chain2=p2.chain_name,
                resseq2=p2.res_id.seqid.num,
                atom2=p2.atom_name,
                distance_A=None,
                involves_ligand=involves_ligand,
            ))

    # ------------------------------------------------------------------
    # 10. Assembly info
    # ------------------------------------------------------------------
    asm_info = _assembly_info(structure)

    # ------------------------------------------------------------------
    # 11. SEQRES vs ATOM gap detection
    # ------------------------------------------------------------------
    seqres_vs_atom: dict[str, Any] = {}

    for ch_info in chains:
        if not ch_info.has_seqres or ch_info.entity_type != "polymer":
            continue
        seq_seqres = ch_info.sequence_from_seqres or ""
        seq_atom = ch_info.sequence_from_atom
        n_seqres = len(seq_seqres)
        n_atom = len(seq_atom)
        n_missing = max(0, n_seqres - n_atom)
        seqres_vs_atom[ch_info.id] = {
            "n_seqres": n_seqres,
            "n_atom": n_atom,
            "n_missing_estimated": n_missing,
            "fraction_observed": (n_atom / n_seqres) if n_seqres else None,
        }

    # ------------------------------------------------------------------
    # 12. MODRES / engineered residues (PDB only)
    # ------------------------------------------------------------------
    engineered_residues: list[dict[str, Any]] = []
    if fmt == "pdb":
        try:
            raw = structure.raw_remarks
            engineered_residues = _parse_modres_remarks(raw)
        except Exception:
            logger.debug("Could not parse MODRES remarks.", exc_info=True)

    # ------------------------------------------------------------------
    # 13. Assemble InventoryReport
    # ------------------------------------------------------------------
    report = InventoryReport(
        format=fmt,  # type: ignore[arg-type]
        n_models=n_models,
        model_ids=model_ids,
        chains=chains,
        hetatm_groups=hetatm_groups,
        disulfide_bonds=disulfide_bonds,
        covalent_links=covalent_links,
        altloc_positions=altloc_positions,
        assembly_info=asm_info,
        space_group=(getattr(structure, "spacegroup_hm", "") or None),
        seqres_vs_atom=seqres_vs_atom,
        engineered_residues=engineered_residues,
        format_warnings=format_warnings,
        checksum_sha256=checksum,
    )

    # ------------------------------------------------------------------
    # 14. Write inventory.json atomically
    # ------------------------------------------------------------------
    job_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = job_dir / "inventory.json"
    tmp_path = inventory_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(report.model_dump(mode="json"), fh, indent=2)
        os.replace(tmp_path, inventory_path)
        logger.info("Stage 0: inventory written to %s", inventory_path)
    except Exception:
        logger.exception("Stage 0: failed to write inventory.json")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return structure, report


__all__ = ["run"]
