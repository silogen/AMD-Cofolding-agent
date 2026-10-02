"""Stage 8 — Ligand Parameterization.

Validates each retained ligand's chemistry via RDKit and optionally
assigns partial charges using OpenFF NAGL (MIT; AM1-BCC quality
without AmberTools). Emits SDF files for GNINA/PoseBusters.
"""

from __future__ import annotations

import logging
import warnings as _warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gemmi

from structure_prep.models import LigandParamResult, Stage8Profile

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# PDB HETATM format string (80 cols).
# serial(5) name(4) altloc(1) resname(3) chain(1) resseq(4) icode+3sp(4)
# x(8.3f) y(8.3f) z(8.3f) occ(6.2f) bfac(6.2f) 10sp element(2)
_HETATM_FMT = (
    "HETATM{serial:5d} {name:<4s} {resname:<3s} "
    "{chain:1s}{resseq:4d}    "
    "{x:8.3f}{y:8.3f}{z:8.3f}"
    "{occ:6.2f}{bfac:6.2f}          "
    "{element:>2s}"
)


# ---------------------------------------------------------------------------
# 3-D coordinate extraction
# ---------------------------------------------------------------------------

def _extract_ligand_mol(
    structure: gemmi.Structure,
    comp_id: str,
    chain_name: str,
    resseq: int,
) -> Any | None:
    """Build an RDKit molecule from the HETATM coordinates of the target residue.

    Returns an RDKit Mol (with ``sanitize=False``) or None on failure.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
    except ImportError:
        logger.warning("Stage 8: RDKit not available; skipping 3D mol extraction.")
        return None

    lines: list[str] = []
    serial = 1

    if len(structure) == 0:
        return None

    found = False
    for chain in structure[0]:
        if chain.name != chain_name:
            continue
        for res in chain:
            if res.seqid.num != resseq or res.name != comp_id:
                continue
            found = True
            for atom in res:
                chain_id_1 = (chain_name + " ")[0]  # exactly 1 char
                element_sym = atom.element.name if atom.element != gemmi.Element("X") else "?"
                lines.append(_HETATM_FMT.format(
                    serial=serial,
                    name=atom.name,
                    resname=res.name,
                    chain=chain_id_1,
                    resseq=res.seqid.num,
                    x=atom.pos.x,
                    y=atom.pos.y,
                    z=atom.pos.z,
                    occ=atom.occ,
                    bfac=atom.b_iso,
                    element=element_sym,
                ))
                serial += 1

    if not found:
        logger.warning(
            "Stage 8: residue %s chain=%s resseq=%d not found in structure.",
            comp_id, chain_name, resseq,
        )
        return None

    lines.append("END")
    pdb_block = "\n".join(lines)

    try:
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    except Exception:
        logger.debug("Stage 8: MolFromPDBBlock failed for %s.", comp_id, exc_info=True)
        return None

    return mol


# ---------------------------------------------------------------------------
# SMILES from CIF CCD data
# ---------------------------------------------------------------------------

def _smiles_from_cif_ccd(structure: gemmi.Structure, comp_id: str) -> str | None:
    """Try to get a canonical SMILES from CCD data embedded in the mmCIF structure.

    Only works when the structure was loaded from an mmCIF file that includes
    ``_pdbx_chem_comp_descriptor`` or ``_chem_comp_descriptor`` tables.
    Returns None for PDB-format inputs.
    """
    try:
        cif_block = getattr(structure, "cif_block", None)
        if cif_block is None:
            return None

        # Prefer _pdbx_chem_comp_descriptor (modern mmCIF / PDB NextGen)
        for prefix, tags in [
            (
                "_pdbx_chem_comp_descriptor.",
                ["comp_id", "type", "descriptor"],
            ),
            (
                "_chem_comp_descriptor.",
                ["comp_id", "type", "descriptor"],
            ),
        ]:
            try:
                table = cif_block.find(prefix, tags)
            except Exception:
                continue
            for row in table:
                row_comp = str(row[0]).strip().strip('"')
                if row_comp != comp_id:
                    continue
                desc_type = str(row[1]).strip().upper()
                if "SMILES" not in desc_type:
                    continue
                # Prefer non-stereo SMILES over isomeric to maximise RDKit
                # parse success; both should have the same connectivity block.
                smiles = str(row[2]).strip().strip('"')
                if smiles and smiles not in (".", "?"):
                    return smiles
    except Exception:
        logger.debug(
            "Stage 8: CIF CCD SMILES lookup failed for %s.", comp_id, exc_info=True
        )
    return None


# ---------------------------------------------------------------------------
# Charge assignment methods
# ---------------------------------------------------------------------------

def _assign_charges_nagl(
    mol_rdkit: Any,
    comp_id: str,
) -> tuple[list[float], str, str]:
    """Assign AM1-BCC equivalent charges using OpenFF NAGL.

    Returns (charges, method_name, model_version_str).
    Raises RuntimeError on any failure.
    """
    try:
        from openff.toolkit import Molecule  # type: ignore[import]
        from openff.nagl import GNNModel  # type: ignore[import]
        from openff.nagl_models import list_available_nagl_models  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"OpenFF NAGL not available: {exc}") from exc

    try:
        off_mol = Molecule.from_rdkit(mol_rdkit, allow_undefined_stereo=True)
    except Exception as exc:
        raise RuntimeError(f"Molecule.from_rdkit failed for {comp_id}: {exc}") from exc

    models = list_available_nagl_models()
    if not models:
        raise RuntimeError("list_available_nagl_models() returned no models.")

    model_path = str(models[0])
    try:
        model = GNNModel.load(model_path, eval_mode=True)
    except TypeError:
        # Older API without eval_mode keyword
        model = GNNModel.load(model_path)  # type: ignore[call-arg]

    charges_raw = model.compute_property(off_mol, as_numpy=True)
    charges = [float(c) for c in charges_raw]

    # Annotate RDKit atoms for downstream SDF writing
    for i, atom in enumerate(mol_rdkit.GetAtoms()):
        if i < len(charges):
            atom.SetDoubleProp("_GasteigerCharge", charges[i])

    return charges, "nagl", model_path


def _assign_charges_mmff94(mol_rdkit: Any) -> tuple[list[float], str, str]:
    """Assign MMFF94 partial charges via RDKit.

    Returns (charges, method_name, version_str).
    Raises RuntimeError on failure.
    """
    try:
        from rdkit.Chem.rdForceFieldHelpers import (  # type: ignore[import]
            MMFFGetMoleculeProperties,
        )
    except ImportError as exc:
        raise RuntimeError(f"RDKit MMFF94 not available: {exc}") from exc

    mp = MMFFGetMoleculeProperties(mol_rdkit)
    if mp is None:
        raise RuntimeError("MMFFGetMoleculeProperties returned None.")
    charges = [
        mp.GetMMFFPartialCharge(i) for i in range(mol_rdkit.GetNumAtoms())
    ]
    return charges, "mmff94", "rdkit_mmff94"


def _assign_charges_gasteiger(mol_rdkit: Any) -> tuple[list[float], str, str]:
    """Assign Gasteiger charges via RDKit.

    Returns (charges, method_name, version_str).
    Raises RuntimeError on failure.
    """
    try:
        from rdkit.Chem import AllChem  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"RDKit not available: {exc}") from exc

    try:
        AllChem.ComputeGasteigerCharges(mol_rdkit)
    except Exception as exc:
        raise RuntimeError(f"Gasteiger charge computation failed: {exc}") from exc

    charges: list[float] = []
    for atom in mol_rdkit.GetAtoms():
        try:
            q = atom.GetDoubleProp("_GasteigerCharge")
        except KeyError:
            q = 0.0
        if q != q:  # NaN guard
            q = 0.0
        charges.append(q)

    return charges, "gasteiger", "rdkit_gasteiger"


# ---------------------------------------------------------------------------
# SDF / Mol2 emission
# ---------------------------------------------------------------------------

def _write_sdf(
    mol_3d: Any,
    comp_id: str,
    chain_name: str,
    resseq: int,
    job_dir: Path,
) -> str | None:
    """Write ligand SDF to job_dir and return the path string, or None on failure."""
    try:
        from rdkit.Chem import SDWriter  # type: ignore[import]
    except ImportError:
        logger.warning("Stage 8: RDKit SDWriter not available; skipping SDF emit.")
        return None

    sdf_path = job_dir / f"ligand_{comp_id}_{chain_name}_{resseq}.sdf"
    try:
        writer = SDWriter(str(sdf_path))
        writer.write(mol_3d)
        writer.close()
        logger.debug("Stage 8: wrote SDF → %s", sdf_path)
        return str(sdf_path)
    except Exception:
        logger.debug("Stage 8: SDF write failed for %s.", comp_id, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Per-ligand processing
# ---------------------------------------------------------------------------

def _process_ligand(
    structure: gemmi.Structure,
    profile: Stage8Profile,
    comp_id: str,
    chain_name: str,
    resseq: int,
    job_dir: Path,
) -> LigandParamResult:
    """Process a single ligand residue and return a LigandParamResult."""
    param_warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Count heavy atoms and mean B-factor / occupancy from structure
    # ------------------------------------------------------------------
    n_heavy = 0
    mean_occ: float | None = None
    mean_bfac: float | None = None
    covalent_bond_to: str | None = None

    try:
        occ_vals: list[float] = []
        bfac_vals: list[float] = []
        for chain in structure[0]:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.seqid.num != resseq or res.name != comp_id:
                    continue
                for atom in res:
                    if atom.element.name not in ("H", "D", "T"):
                        n_heavy += 1
                    occ_vals.append(atom.occ)
                    bfac_vals.append(atom.b_iso)
        if occ_vals:
            mean_occ = sum(occ_vals) / len(occ_vals)
        if bfac_vals:
            mean_bfac = sum(bfac_vals) / len(bfac_vals)
    except Exception:
        logger.debug("Stage 8: could not compute occ/bfac for %s.", comp_id, exc_info=True)

    # Check for covalent links involving this residue
    try:
        for conn in structure.connections:
            if conn.type == gemmi.ConnectionType.Covale:
                p1 = conn.partner1
                p2 = conn.partner2
                if (p1.chain_name == chain_name and p1.res_id.seqid.num == resseq):
                    covalent_bond_to = f"{p2.chain_name}:{p2.res_id.seqid.num}:{p2.atom_name}"
                    break
                if (p2.chain_name == chain_name and p2.res_id.seqid.num == resseq):
                    covalent_bond_to = f"{p1.chain_name}:{p1.res_id.seqid.num}:{p1.atom_name}"
                    break
    except Exception:
        logger.debug("Stage 8: covalent link check failed for %s.", comp_id, exc_info=True)

    # ------------------------------------------------------------------
    # 2. Extract 3-D RDKit molecule
    # ------------------------------------------------------------------
    mol_3d = _extract_ligand_mol(structure, comp_id, chain_name, resseq)

    # ------------------------------------------------------------------
    # 3. Resolve SMILES (priority: user override → CCD-CIF → geometry)
    # ------------------------------------------------------------------
    smiles: str | None = None
    smiles_source: str = "unavailable"

    # Priority 1: user override
    override = profile.ligand_smiles_override.get(comp_id)
    if override:
        smiles = override
        smiles_source = "user_override"

    # Priority 2: CCD from mmCIF structure block
    if smiles is None:
        try:
            smiles = _smiles_from_cif_ccd(structure, comp_id)
            if smiles:
                smiles_source = "ccd_cif"
        except Exception:
            logger.debug(
                "Stage 8: CIF CCD SMILES lookup failed for %s.", comp_id, exc_info=True
            )

    # Priority 3: geometry-perceived via RDKit
    if smiles is None and mol_3d is not None:
        try:
            from rdkit import Chem  # type: ignore[import]
            from rdkit.Chem import rdmolfiles  # type: ignore[import]

            mol_copy = Chem.RWMol(mol_3d)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                Chem.SanitizeMol(
                    mol_copy,
                    catchErrors=True,
                )
            smiles = rdmolfiles.MolToSmiles(mol_copy)
            if smiles:
                smiles_source = "geometry_perceived"
        except Exception:
            logger.debug(
                "Stage 8: geometry-perceived SMILES failed for %s.", comp_id, exc_info=True
            )

    # ------------------------------------------------------------------
    # 4. RDKit validation
    # ------------------------------------------------------------------
    formal_charge: int | None = None
    inchikey_smiles: str | None = None
    inchikey_3d: str | None = None
    inchikey_match: bool | None = None
    inchikey_exact_match: bool | None = None
    mol_smi: Any = None

    if smiles:
        try:
            from rdkit import Chem  # type: ignore[import]
            from rdkit.Chem.inchi import MolToInchi, MolToInchiKey  # type: ignore[import]

            mol_smi = Chem.MolFromSmiles(smiles)
            if mol_smi is None:
                msg = f"RDKit cannot parse SMILES '{smiles[:80]}'"
                logger.warning("Stage 8: %s for ligand %s.", msg, comp_id)
                param_warnings.append(msg)
                smiles = None
                smiles_source = "unavailable"
            else:
                formal_charge = Chem.GetFormalCharge(mol_smi)

                # InChIKey from SMILES
                inchi_smi = MolToInchi(mol_smi)
                inchikey_smiles = MolToInchiKey(mol_smi) if inchi_smi else None

                # InChIKey from 3-D geometry
                if mol_3d is not None:
                    try:
                        mol_3d_copy = Chem.RWMol(mol_3d)
                        with _warnings.catch_warnings():
                            _warnings.simplefilter("ignore")
                            Chem.SanitizeMol(mol_3d_copy, catchErrors=True)
                        inchi_3d = MolToInchi(mol_3d_copy)
                        inchikey_3d = MolToInchiKey(mol_3d_copy) if inchi_3d else None
                    except Exception:
                        logger.debug(
                            "Stage 8: 3D InChIKey failed for %s.", comp_id, exc_info=True
                        )

                # Compare InChIKey connectivity blocks
                if inchikey_smiles and inchikey_3d:
                    ik_smi_conn = inchikey_smiles.split("-")[0]
                    ik_3d_conn = inchikey_3d.split("-")[0]
                    inchikey_match = ik_smi_conn == ik_3d_conn
                    inchikey_exact_match = inchikey_smiles == inchikey_3d

                    if not inchikey_match:
                        msg = (
                            f"InChIKey connectivity mismatch for {comp_id}: "
                            f"SMILES={inchikey_smiles}, 3D={inchikey_3d}"
                        )
                        if profile.strict_ligand_match:
                            raise ValueError(msg)
                        logger.warning("Stage 8: %s", msg)
                        param_warnings.append(msg)
        except ValueError:
            raise
        except Exception:
            logger.debug(
                "Stage 8: RDKit validation failed for %s.", comp_id, exc_info=True
            )
            param_warnings.append(f"RDKit validation raised an exception for {comp_id}.")

    # ------------------------------------------------------------------
    # 5. Parameterization (charge assignment)
    # ------------------------------------------------------------------
    charges: list[float] | None = None
    param_method: str = "none"
    param_status: str = "skipped"
    charge_method_str: str | None = None
    ff_version_str: str | None = None
    nagl_model_str: str | None = None

    if profile.ligand_ff == "none":
        param_status = "skipped"
    elif mol_smi is None:
        msg = f"No valid SMILES for {comp_id}; parameterization skipped."
        logger.warning("Stage 8: %s", msg)
        param_warnings.append(msg)
        if profile.unsupported_chemistry_policy == "fail":
            raise ValueError(msg)
        param_status = "unsupported"
    else:
        # Sanitize mol_smi for charge methods
        try:
            from rdkit import Chem  # type: ignore[import]

            mol_for_charges = Chem.RWMol(mol_smi)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                Chem.SanitizeMol(mol_for_charges)
        except Exception as exc:
            logger.warning(
                "Stage 8: sanitization failed for %s (%s); will attempt charges anyway.",
                comp_id, exc,
            )
            mol_for_charges = mol_smi

        method_order: list[str]
        if profile.ligand_charge_method == "nagl":
            method_order = ["nagl", "gasteiger"]
        elif profile.ligand_charge_method == "mmff94":
            method_order = ["mmff94", "gasteiger"]
        elif profile.ligand_charge_method == "gasteiger":
            method_order = ["gasteiger"]
        else:  # "formal_only"
            method_order = []

        last_exc: Exception | None = None
        for method in method_order:
            try:
                if method == "nagl":
                    charges, charge_method_str, nagl_model_str = _assign_charges_nagl(
                        mol_for_charges, comp_id
                    )
                    param_method = "nagl"
                    ff_version_str = "openff_nagl"
                elif method == "mmff94":
                    charges, charge_method_str, ff_version_str = _assign_charges_mmff94(
                        mol_for_charges
                    )
                    param_method = "mmff94"
                elif method == "gasteiger":
                    charges, charge_method_str, ff_version_str = _assign_charges_gasteiger(
                        mol_for_charges
                    )
                    param_method = "gasteiger"
                param_status = "ok"
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "Stage 8: %s charge assignment failed for %s (%s); trying fallback.",
                    method, comp_id, exc,
                )

        if param_status != "ok" and method_order:
            msg = (
                f"All charge methods {method_order} failed for {comp_id}: "
                f"{last_exc}"
            )
            logger.warning("Stage 8: %s", msg)
            param_warnings.append(msg)
            if profile.unsupported_chemistry_policy == "fail":
                raise RuntimeError(msg)
            param_status = "failed"
        elif not method_order:
            # formal_only: assign zero partial charges, status ok
            param_status = "ok"
            param_method = "formal_only"
            charge_method_str = "formal_only"

    # Propagate charges to mol_3d for SDF writing
    if charges is not None and mol_3d is not None:
        try:
            atoms_3d = list(mol_3d.GetAtoms())
            for i, q in enumerate(charges):
                if i < len(atoms_3d):
                    atoms_3d[i].SetDoubleProp("_GasteigerCharge", q)
        except Exception:
            logger.debug("Stage 8: charge copy to mol_3d failed for %s.", comp_id, exc_info=True)

    # ------------------------------------------------------------------
    # 6. Emit SDF
    # ------------------------------------------------------------------
    sdf_path_str: str | None = None
    if profile.emit_sdf and mol_3d is not None:
        sdf_path_str = _write_sdf(mol_3d, comp_id, chain_name, resseq, job_dir)

    # ------------------------------------------------------------------
    # 7. Assemble result
    # ------------------------------------------------------------------
    return LigandParamResult(
        comp_id=comp_id,
        chain=chain_name,
        resseq=resseq,
        formal_charge=formal_charge,
        n_heavy_atoms=n_heavy,
        smiles_used=smiles,
        smiles_source=smiles_source,  # type: ignore[arg-type]
        inchikey_smiles=inchikey_smiles,
        inchikey_3d=inchikey_3d,
        inchikey_match=inchikey_match,
        inchikey_exact_match=inchikey_exact_match,
        occupancy=mean_occ,
        b_factor=mean_bfac,
        covalent_bond_to=covalent_bond_to,
        parameterization_method=param_method,
        parameterization_status=param_status,  # type: ignore[arg-type]
        charge_method=charge_method_str,
        ff_version=ff_version_str,
        nagl_model=nagl_model_str,
        parameterization_warnings=param_warnings,
        artifact_sdf=sdf_path_str,
        artifact_mol2=None,
        artifact_topology_xml=None,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage8Profile,
    ligand_residues: list[dict[str, Any]],  # [{comp_id, chain, resseq}, ...]
    job_dir: Path,
) -> tuple[gemmi.Structure, dict[str, Any]]:
    """Stage 8 — Ligand Parameterization.

    Parameters
    ----------
    structure:
        gemmi Structure (after capping from Stage 7).  Returned unchanged.
    profile:
        Stage8Profile controlling FF, charge method, SDF emission, etc.
    ligand_residues:
        List of dicts with keys ``comp_id``, ``chain``, ``resseq`` identifying
        each ligand residue to process.
    job_dir:
        Job working directory for SDF / Mol2 artefacts.

    Returns
    -------
    (structure_unchanged, stage8_report_dict)
        ``stage8_report_dict`` key: ``ligands`` (list of LigandParamResult
        serialised as dicts).
    """
    job_dir = Path(job_dir)
    results: list[LigandParamResult] = []

    logger.info(
        "Stage 8: processing %d ligand residue(s) with ff=%s charge_method=%s.",
        len(ligand_residues),
        profile.ligand_ff,
        profile.ligand_charge_method,
    )

    for lig_info in ligand_residues:
        comp_id: str = str(lig_info.get("comp_id", "UNK"))
        chain_name: str = str(lig_info.get("chain", "A"))
        resseq: int = int(lig_info.get("resseq", 0))

        logger.debug(
            "Stage 8: processing %s chain=%s resseq=%d.",
            comp_id, chain_name, resseq,
        )

        try:
            result = _process_ligand(
                structure=structure,
                profile=profile,
                comp_id=comp_id,
                chain_name=chain_name,
                resseq=resseq,
                job_dir=job_dir,
            )
        except ValueError:
            # strict_ligand_match or fail policy — re-raise
            raise
        except Exception:
            logger.exception(
                "Stage 8: unhandled error processing ligand %s chain=%s resseq=%d.",
                comp_id, chain_name, resseq,
            )
            result = LigandParamResult(
                comp_id=comp_id,
                chain=chain_name,
                resseq=resseq,
                parameterization_status="failed",
                parameterization_warnings=[
                    "Unhandled exception during Stage 8 processing."
                ],
            )

        results.append(result)
        logger.info(
            "Stage 8: %s chain=%s resseq=%d → status=%s smiles_source=%s.",
            comp_id, chain_name, resseq,
            result.parameterization_status,
            result.smiles_source,
        )

    report: dict[str, Any] = {
        "ligands": [r.model_dump() for r in results],
    }
    return structure, report


__all__ = ["run"]
