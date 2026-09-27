"""Output artefact writers.

All writes are atomic (tmp file → os.replace) to prevent partial outputs
from being consumed by downstream tools during pipeline execution.

Public API
----------
write_all_artefacts(structure, profile, report, ligand_residues, role_map, job_dir)
    → ArtifactPaths

write_report(report, path)
    Write PreparationReport as indented JSON.

write_manifest(job_id, profile, report, job_dir)
    Write lightweight manifest.json for quick job lookup.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import gemmi

from .additives import METAL_ELEMENTS, WATER_RESNAMES
from .models import (
    ArtifactPaths,
    PreparationProfile,
    PreparationReport,
    RoleMap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level atomic write helpers
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a sibling tmp file.

    Creates parent directories if needed.  On failure the tmp file is
    silently removed and the original exception is re-raised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_str, str(path))
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_str)
        raise


def _write_structure_pdb(
    structure: gemmi.Structure,
    path: Path,
    strip_h: bool = False,
) -> None:
    """Write *structure* to a PDB file at *path* atomically.

    Parameters
    ----------
    structure:
        gemmi Structure to serialise.
    path:
        Destination path (created atomically).
    strip_h:
        When True the write uses a deep copy with all H/D atoms removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    # Close the fd immediately — gemmi opens the file via its C++ layer.
    os.close(fd)
    try:
        if strip_h:
            st = structure.clone()
            for model in st:
                for chain in model:
                    for res in chain:
                        to_del = [
                            i for i, a in enumerate(res)
                            if a.element == gemmi.Element("H")
                            or a.element == gemmi.Element("D")
                        ]
                        for i in reversed(to_del):
                            del res[i]
            st.write_pdb(tmp_str)
        else:
            structure.write_pdb(tmp_str)
        os.replace(tmp_str, str(path))
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_str)
        raise


def _write_structure_cif(
    structure: gemmi.Structure,
    path: Path,
    strip_h: bool = False,
) -> None:
    """Write *structure* to an mmCIF file at *path* atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        if strip_h:
            st = structure.clone()
            for model in st:
                for chain in model:
                    for res in chain:
                        to_del = [
                            i for i, a in enumerate(res)
                            if a.element == gemmi.Element("H")
                            or a.element == gemmi.Element("D")
                        ]
                        for i in reversed(to_del):
                            del res[i]
            st.make_mmcif_document().write_file(tmp_str)
        else:
            structure.make_mmcif_document().write_file(tmp_str)
        os.replace(tmp_str, str(path))
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_str)
        raise


def _write_structure(
    structure: gemmi.Structure,
    path: Path,
    output_format: str = "pdb",
    strip_h: bool = False,
) -> None:
    """Dispatch to the correct format writer."""
    if output_format == "cif":
        _write_structure_cif(structure, path, strip_h=strip_h)
    else:
        _write_structure_pdb(structure, path, strip_h=strip_h)


# ---------------------------------------------------------------------------
# Structure sub-selection
# ---------------------------------------------------------------------------

def _extract_chains(
    structure: gemmi.Structure,
    chain_names: set[str],
) -> gemmi.Structure:
    """Return a deep copy of *structure* keeping only *chain_names*.

    Deletion is performed in reverse index order to avoid index shifting.
    """
    st = structure.clone()
    for model in st:
        to_remove = [
            i for i, ch in enumerate(model)
            if ch.name not in chain_names
        ]
        for i in reversed(to_remove):
            del model[i]
    return st


# ---------------------------------------------------------------------------
# PDB HETATM line formatter
# ---------------------------------------------------------------------------

def _hetatm_line(
    atom: gemmi.Atom,
    res: gemmi.Residue,
    chain_name: str,
) -> str:
    """Return a single PDB HETATM record line (with trailing newline).

    Columns follow the PDB v3 fixed-width format:
        1-6   record name
        7-11  serial
        13-16 atom name (left-justified, 4 chars)
        17    blank (no altloc stored)
        18-20 residue name
        22    chain ID
        23-26 residue sequence number
        31-38 X (8.3f)
        39-46 Y (8.3f)
        47-54 Z (8.3f)
        55-60 occupancy (6.2f)
        61-66 B-factor (6.2f)
        77-78 element symbol (right-justified)
    """
    # Atom name: keep exactly 4 chars; pad or truncate as needed
    name = atom.name
    if len(name) >= 4:
        name_field = name[:4]
    else:
        name_field = f"{name:<4s}"

    elem = atom.element.name.upper()  # e.g. "C", "FE", "ZN"

    return (
        f"HETATM{atom.serial:5d} {name_field} {res.name:<3s} "
        f"{chain_name:1s}{res.seqid.num:4d}    "
        f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
        f"{atom.occ:6.2f}{atom.b_iso:6.2f}          "
        f"{elem:>2s}  \n"
    )


# ---------------------------------------------------------------------------
# Relative path helper
# ---------------------------------------------------------------------------

def _rel(path: Path, base: Path) -> str:
    """Return *path* relative to *base*; fall back to absolute string."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Main artefact writer
# ---------------------------------------------------------------------------

def write_all_artefacts(
    structure: gemmi.Structure,
    profile: PreparationProfile,
    report: PreparationReport,
    ligand_residues: list[dict[str, Any]],
    role_map: RoleMap,
    job_dir: Path,
) -> ArtifactPaths:
    """Write all prepared structure files and sequence artefacts.

    Written artefacts
    -----------------
    prepared/complex.pdb        — full retained structure
    prepared/receptor.pdb       — protein chains + structural metal chains
    prepared/receptor_noh.pdb   — receptor without H  (if emit_receptor_no_h)
    prepared/binder.pdb         — binder chains only  (if emit_separate_entities)
    prepared/ligand_CID_CH_SEQ.pdb  — one file per retained ligand instance
    prepared/water.pdb          — retained water molecules (if any)
    prepared/ions.pdb           — structural metal ions
    sequences/receptor.fasta    — per-chain FASTA sequences (if emit_fasta)
    sequences/entity_map.json   — JSON entity role summary

    All writes are atomic (tmp-file → os.replace).

    Parameters
    ----------
    structure:
        Final gemmi Structure after all 9 stages.
    profile:
        PreparationProfile (controls emit flags and output format).
    report:
        PreparationReport (used for sequences_by_chain, n_waters_retained, …).
    ligand_residues:
        List of dicts with keys ``comp_id``, ``chain``, ``resseq``.
    role_map:
        Stage 2 RoleMap (protein_chains, binder_chains, structural_metals, …).
    job_dir:
        Job working directory.  Artefacts go into sub-directories.

    Returns
    -------
    ArtifactPaths
        All path fields set to strings relative to ``job_dir.parent``.
    """
    job_dir = Path(job_dir)
    prepared_dir = job_dir / "prepared"
    sequences_dir = job_dir / "sequences"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir.mkdir(parents=True, exist_ok=True)

    fmt = profile.output.output_format  # "pdb" | "cif"
    ext = ".pdb" if fmt == "pdb" else ".cif"
    base = job_dir.parent  # artefact paths are relative to this

    artifact_paths = ArtifactPaths()

    # ------------------------------------------------------------------
    # complex.pdb — full retained structure as-is
    # ------------------------------------------------------------------
    complex_path = prepared_dir / f"complex{ext}"
    _write_structure(structure, complex_path, output_format=fmt, strip_h=False)
    artifact_paths.complex = _rel(complex_path, base)
    logger.debug("Wrote complex: %s", complex_path)

    # ------------------------------------------------------------------
    # receptor.pdb — protein chains + structural metal chains
    #
    # In many PDB files structural metals are HETATM within the protein
    # chain, so they are automatically included.  When they live in a
    # separate chain (e.g. chain "Z") we explicitly add that chain.
    # ------------------------------------------------------------------
    receptor_chain_names: set[str] = set(role_map.protein_chains)
    for metal in role_map.structural_metals:
        receptor_chain_names.add(metal.chain)

    receptor_st = _extract_chains(structure, receptor_chain_names)
    receptor_path = prepared_dir / f"receptor{ext}"
    _write_structure(receptor_st, receptor_path, output_format=fmt, strip_h=False)
    artifact_paths.receptor = _rel(receptor_path, base)
    logger.debug("Wrote receptor: %s", receptor_path)

    # ------------------------------------------------------------------
    # receptor_noh.pdb — receptor without H (for GNINA, DOCK, etc.)
    # ------------------------------------------------------------------
    if profile.output.emit_receptor_no_h:
        receptor_noh_path = prepared_dir / f"receptor_noh{ext}"
        _write_structure(
            receptor_st, receptor_noh_path, output_format=fmt, strip_h=True
        )
        artifact_paths.receptor_noh = _rel(receptor_noh_path, base)
        logger.debug("Wrote receptor (no H): %s", receptor_noh_path)

    # ------------------------------------------------------------------
    # binder.pdb — binder / peptide / antibody chains (optional)
    # ------------------------------------------------------------------
    if profile.output.emit_separate_entities and role_map.binder_chains:
        binder_chain_names: set[str] = set(role_map.binder_chains)
        binder_st = _extract_chains(structure, binder_chain_names)
        binder_path = prepared_dir / f"binder{ext}"
        _write_structure(binder_st, binder_path, output_format=fmt, strip_h=False)
        artifact_paths.binder = _rel(binder_path, base)
        logger.debug("Wrote binder: %s", binder_path)

    # ------------------------------------------------------------------
    # Per-ligand PDB files
    # Filename: ligand_{comp_id}_{chain}_{resseq}.pdb to be unique per
    # instance even when the same comp_id appears in multiple chains.
    # ------------------------------------------------------------------
    if ligand_residues and len(structure) > 0:
        model = structure[0]
        for lig in ligand_residues:
            comp_id: str = str(lig.get("comp_id", "UNK"))
            chain_name: str = str(lig.get("chain", "A"))
            resseq: int = int(lig.get("resseq", 0))

            lines: list[str] = []
            for chain in model:
                if chain.name != chain_name:
                    continue
                for res in chain:
                    if res.seqid.num == resseq and res.name == comp_id:
                        for atom in res:
                            lines.append(_hetatm_line(atom, res, chain_name))

            if not lines:
                logger.warning(
                    "Ligand %s:%s:%d not found in final structure; skipping PDB write.",
                    comp_id, chain_name, resseq,
                )
                continue

            lines.append("END\n")
            lig_filename = f"ligand_{comp_id}_{chain_name}_{resseq}.pdb"
            lig_path = prepared_dir / lig_filename
            _atomic_write_text(lig_path, "".join(lines))

            lig_key = f"{comp_id}:{chain_name}:{resseq}"
            artifact_paths.ligands.setdefault(lig_key, {})
            artifact_paths.ligands[lig_key]["pdb"] = _rel(lig_path, base)
            artifact_paths.ligands[lig_key]["sdf"] = None
            artifact_paths.ligands[lig_key]["mol2"] = None
            logger.debug("Wrote ligand %s → %s", lig_key, lig_path)

    # ------------------------------------------------------------------
    # water.pdb — retained water molecules (only if any were kept)
    # ------------------------------------------------------------------
    if report.n_waters_retained > 0 and len(structure) > 0:
        water_lines: list[str] = []
        model = structure[0]
        for chain in model:
            for res in chain:
                if res.name.strip().upper() in WATER_RESNAMES:
                    for atom in res:
                        water_lines.append(_hetatm_line(atom, res, chain.name))

        if water_lines:
            water_lines.append("END\n")
            water_path = prepared_dir / "water.pdb"
            _atomic_write_text(water_path, "".join(water_lines))
            artifact_paths.water = _rel(water_path, base)
            logger.debug(
                "Wrote water.pdb: %d HETATM records", len(water_lines) - 1
            )

    # ------------------------------------------------------------------
    # ions.pdb — structural metal ions
    # ------------------------------------------------------------------
    if role_map.structural_metals and len(structure) > 0:
        # Build a precise lookup set from the role map
        metal_keys: set[tuple[str, str, int]] = {
            (m.comp_id.strip().upper(), m.chain, m.resseq)
            for m in role_map.structural_metals
        }
        ion_lines: list[str] = []
        model = structure[0]
        for chain in model:
            for res in chain:
                key = (res.name.strip().upper(), chain.name, res.seqid.num)
                if key in metal_keys:
                    for atom in res:
                        ion_lines.append(_hetatm_line(atom, res, chain.name))

        if ion_lines:
            ion_lines.append("END\n")
            ions_path = prepared_dir / "ions.pdb"
            _atomic_write_text(ions_path, "".join(ion_lines))
            artifact_paths.ions = _rel(ions_path, base)
            logger.debug(
                "Wrote ions.pdb: %d HETATM records", len(ion_lines) - 1
            )

    # ------------------------------------------------------------------
    # sequences/receptor.fasta
    # Wraps sequence at 60 characters per line (NCBI convention).
    # ------------------------------------------------------------------
    if profile.output.emit_fasta:
        fasta_lines: list[str] = []
        protein_chain_set = set(role_map.protein_chains)
        for chain_id, seq in report.sequences_by_chain.items():
            if chain_id not in protein_chain_set:
                continue
            if not seq:
                continue
            fasta_lines.append(f">{chain_id}")
            for i in range(0, len(seq), 60):
                fasta_lines.append(seq[i : i + 60])

        fasta_content = "\n".join(fasta_lines) + "\n" if fasta_lines else ""
        fasta_path = sequences_dir / "receptor.fasta"
        _atomic_write_text(fasta_path, fasta_content)
        artifact_paths.sequences_fasta = _rel(fasta_path, base)
        logger.debug(
            "Wrote receptor.fasta: %d chain(s)", len(protein_chain_set)
        )

    # ------------------------------------------------------------------
    # sequences/residue_numbering.json
    # Author numbering is the public API; sequence and UniProt positions are
    # explicit translations rather than assumptions made by downstream tools.
    # ------------------------------------------------------------------
    numbering_path = sequences_dir / "residue_numbering.json"
    numbering_payload = {
        "construct_form": report.construct_form,
        "entries": [
            entry.model_dump(mode="json") for entry in report.residue_numbering
        ],
    }
    _atomic_write_text(
        numbering_path,
        json.dumps(numbering_payload, indent=2),
    )
    artifact_paths.residue_numbering_json = _rel(numbering_path, base)

    # ------------------------------------------------------------------
    # prepared/aligned_complex.pdb — if alignment was applied, write the
    # transformed complex as a distinct artifact so downstream tools have
    # a named path rather than guessing from the job directory.
    # ------------------------------------------------------------------
    if report.alignment.performed:
        aligned_path = prepared_dir / f"aligned_complex{ext}"
        _write_structure(structure, aligned_path, output_format=fmt, strip_h=False)
        artifact_paths.aligned_complex = _rel(aligned_path, base)
        logger.debug("Wrote aligned complex: %s", aligned_path)

    # ------------------------------------------------------------------
    # sequences/entity_map.json
    # Human- and machine-readable summary of entity roles.
    # ------------------------------------------------------------------
    entity_map: dict[str, Any] = {
        "protein_chains": sorted(role_map.protein_chains),
        "binder_chains": sorted(role_map.binder_chains),
        "nucleic_acid_chains": sorted(role_map.nucleic_acid_chains),
        "structural_metals": [
            {
                "comp_id": m.comp_id,
                "element": m.element,
                "chain": m.chain,
                "resseq": m.resseq,
                "classification": m.classification,
            }
            for m in role_map.structural_metals
        ],
        "crystallographic_ions": role_map.crystallographic_ions,
        "ligands": [
            {
                "comp_id": l.get("comp_id"),
                "chain": l.get("chain"),
                "resseq": l.get("resseq"),
            }
            for l in ligand_residues
        ],
        "ambiguous_groups": role_map.ambiguous_groups,
        "construct_form": report.construct_form,
        "construct_assessment": [
            assessment.model_dump(mode="json")
            for assessment in report.construct_assessment
        ],
        "residue_numbering": artifact_paths.residue_numbering_json,
    }
    entity_map_path = sequences_dir / "entity_map.json"
    _atomic_write_text(entity_map_path, json.dumps(entity_map, indent=2))
    artifact_paths.entity_map = _rel(entity_map_path, base)
    logger.debug("Wrote entity_map.json")

    return artifact_paths


# ---------------------------------------------------------------------------
# Report and manifest writers
# ---------------------------------------------------------------------------

def write_report(report: PreparationReport, path: Path) -> None:
    """Atomically write *report* as indented JSON to *path*.

    Uses Pydantic v2's ``model_dump_json`` which serialises all field types
    (datetime, Literal, nested models, etc.) correctly.
    """
    _atomic_write_text(path, report.model_dump_json(indent=2))
    logger.info("Wrote report.json → %s", path)


def write_manifest(
    job_id: str,
    profile: PreparationProfile,
    report: PreparationReport,
    job_dir: Path,
) -> None:
    """Atomically write a lightweight ``manifest.json`` for quick job lookup.

    The manifest is intentionally smaller than report.json and contains only
    the fields needed for job tracking and artefact discovery without having
    to parse the full report.
    """
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "schema_version": "1.0",
        "profile_hash": report.profile_hash,
        "status": "completed",
        "input_path": report.input_path,
        "received_at": report.received_at,
        "artifacts": {
            "complex": report.artifacts.complex,
            "receptor": report.artifacts.receptor,
            "receptor_noh": report.artifacts.receptor_noh,
            "binder": report.artifacts.binder,
            "water": report.artifacts.water,
            "ions": report.artifacts.ions,
            "sequences_fasta": report.artifacts.sequences_fasta,
            "residue_numbering_json": report.artifacts.residue_numbering_json,
            "entity_map": report.artifacts.entity_map,
            "inventory_json": report.artifacts.inventory_json,
            "report_json": report.artifacts.report_json,
        },
        "stage_timings_s": report.stage_timings_s,
        "n_warnings": len(report.warnings),
        "warnings": report.warnings,
        "n_ligands": len(report.ligands),
        "n_waters_retained": report.n_waters_retained,
        "construct_form": report.construct_form,
        "protonation_method": report.protonation_method,
        "protonation_ph": report.protonation_ph,
        "minimization_performed": report.minimization.performed,
    }
    manifest_path = Path(job_dir) / "manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    logger.info("Wrote manifest.json → %s", manifest_path)


__all__ = [
    "write_all_artefacts",
    "write_report",
    "write_manifest",
    # low-level helpers (useful for tests)
    "_atomic_write_text",
    "_write_structure_pdb",
    "_write_structure_cif",
    "_write_structure",
    "_extract_chains",
    "_hetatm_line",
]
