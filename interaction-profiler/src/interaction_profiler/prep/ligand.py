# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ligand preparation.

The old pipeline handed ProLIF a ligand with zero hydrogens and wrong
bond orders (RDKit ``DetermineBondOrders(charge=0)`` failed on charged
ligands and fell back to an all-single-bond, aromaticity-blind Mol), so
only ``vdw_contact`` ever survived. This module fixes that:

1. Protonate the reference SMILES at the requested pH (dimorphite_dl).
2. Read the ligand coordinates from the structure (bonds unreliable).
3. Transfer bond orders + formal charges from the protonated SMILES
   template onto the coordinate Mol (``AssignBondOrdersFromTemplate``),
   preserving atom order so the fragment/interaction join stays valid.
4. Add explicit hydrogens with 3D coordinates (``AddHs(addCoords=True)``).

Falls back to coordinate-based bond perception when no template SMILES
is available.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LigandPrep:
    mol: Any | None = None
    atom_names: list[str] = field(default_factory=list)
    smiles_input: str | None = None
    smiles_protonated: str | None = None
    smiles_prepared: str | None = None
    n_hydrogens: int = 0
    bond_assignment: str = "none"
    warnings: list[str] = field(default_factory=list)


def protonate_smiles_at_ph(smiles: str, ph: float) -> str | None:
    """Return the dominant protonation state SMILES at ``ph``.

    dimorphite occasionally over-deprotonates weakly acidic aromatic ring
    nitrogens (imidazole/pyrrole/pyrazole/indole/triazole NH, pKa ~10-17);
    :func:`_neutralize_implausible_charges` reverts those while keeping
    genuinely acidic anions (tetrazole, carboxylate, (sulfon)amide-N-acyl).
    """
    try:
        from dimorphite_dl import protonate_smiles

        out = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.0, max_variants=1)
        if out:
            return _neutralize_implausible_charges(out[0]) or out[0]
    except Exception as exc:
        logger.debug("dimorphite protonation failed for %s: %s", smiles, exc)
    return None


def _neutralize_implausible_charges(smiles: str) -> str | None:
    """Revert implausible aromatic-N anions to neutral NH.

    An aromatic ``[n-]`` is kept only when it is genuinely acidic at pH ~7.4
    -- i.e. part of a tetrazole (pKa ~4.9) or an azole flanked by two ring
    nitrogens (e.g. 1,2,3-triazole). Otherwise it is protonated back to a
    neutral N-H, which is the correct dominant state for imidazole, pyrrole,
    pyrazole, indole, etc.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # atoms allowed to stay anionic: tetrazole ring members
    keep: set[int] = set()
    tetrazole = Chem.MolFromSmarts("[nX2r5]1[nX2r5][nX2r5][nX2r5][c,n]1")
    if tetrazole is not None:
        for match in mol.GetSubstructMatches(tetrazole):
            keep.update(match)

    changed = False
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic() and atom.GetSymbol() == "N" and atom.GetFormalCharge() < 0:
            if atom.GetIdx() in keep:
                continue
            # aromatic N with two aromatic-N ring neighbours is azole-acidic; keep
            arom_n_neighbours = sum(
                1 for nb in atom.GetNeighbors()
                if nb.GetIsAromatic() and nb.GetSymbol() == "N"
            )
            if arom_n_neighbours >= 2:
                continue
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
            changed = True

    if not changed:
        return smiles
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


@contextlib.contextmanager
def _quiet_rdkit():
    """Silence RDKit/InChI stderr chatter (parse errors, 'Invalid InChI
    prefix', 'bond type above 3') during best-effort validity/InChIKey calls
    where a failure is an expected, handled outcome."""
    blocker = None
    with contextlib.suppress(Exception):
        from rdkit import rdBase

        blocker = rdBase.BlockLogs()
    try:
        yield
    finally:
        del blocker


def _canonical_smiles(mol: Any) -> str | None:
    try:
        from rdkit import Chem

        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def smiles_advisories(smiles: str | None) -> list[str]:
    """Advisory notes for a *valid* SMILES that is likely not a clean single
    ligand -- multi-component, wildcard/dummy atoms, or no carbon. These are
    warnings (not errors): the input still profiles, but the caller may have
    passed something unintended (a salt, a Markush pattern, an ion)."""
    if not smiles:
        return []
    notes: list[str] = []
    try:
        from rdkit import Chem

        with _quiet_rdkit():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            notes.append(
                f"multi-component SMILES ({len(frags)} fragments); "
                "supply only the ligand (salts/counterions should be stripped)"
            )
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            notes.append("SMILES contains wildcard/dummy atoms (*); not a concrete ligand")
        if mol.GetNumAtoms() and not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            notes.append("SMILES has no carbon atoms (inorganic / ion?)")
    except Exception as exc:  # noqa: BLE001
        logger.debug("smiles_advisories failed for %s: %s", smiles, exc)
    return notes


def inchikey_of_mol(mol: Any) -> str | None:
    """Full InChIKey of an RDKit Mol (Hs stripped first). None on failure."""
    if mol is None:
        return None
    try:
        from rdkit import Chem

        with _quiet_rdkit():
            return Chem.MolToInchiKey(Chem.RemoveHs(mol)) or None
    except Exception as exc:  # noqa: BLE001 - InChI toolkit may be unavailable
        logger.debug("InChIKey (mol) failed: %s", exc)
        return None


def validate_smiles(smiles: str | None) -> tuple[bool, str | None]:
    """Check a SMILES parses, returning ``(ok, inchikey)``.

    Two birds, one stone: RDKit ``MolFromSmiles`` is the validity gate and
    the InChIKey (used elsewhere for ligand matching) is computed in the
    same pass. ``ok`` is False for unparseable SMILES; ``inchikey`` may
    still be None if the InChI toolkit is unavailable even for a valid mol.
    """
    if not smiles:
        return False, None
    try:
        from rdkit import Chem

        with _quiet_rdkit():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None
            try:
                return True, (Chem.MolToInchiKey(mol) or None)
            except Exception:  # noqa: BLE001 - valid mol, InChI backend missing
                return True, None
    except Exception as exc:  # noqa: BLE001
        logger.debug("validate_smiles failed for %s: %s", smiles, exc)
        return False, None


def inchikey_of_smiles(smiles: str | None) -> str | None:
    """Full InChIKey of a SMILES string. None on failure / empty input."""
    if not smiles:
        return None
    try:
        from rdkit import Chem

        with _quiet_rdkit():
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("InChIKey (smiles) failed for %s: %s", smiles, exc)
        return None


def inchikey_skeleton(key: str | None) -> str | None:
    """First block (connectivity skeleton) of an InChIKey, ignoring
    stereo/protonation which live in the second block."""
    if not key:
        return None
    return key.split("-", 1)[0]


def formula_of_smiles(smiles: str | None) -> dict[str, int] | None:
    """Heavy-atom element counts of a SMILES, e.g. ``{'C': 20, 'N': 4}``.

    Used to match a template against candidate ligand residues in a
    structure without needing (unreliable) bond-order perception.
    """
    if not smiles:
        return None
    try:
        from collections import Counter

        from rdkit import Chem

        with _quiet_rdkit():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        counts: Counter[str] = Counter()
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym != "H":
                counts[sym] += 1
        return dict(counts)
    except Exception as exc:  # noqa: BLE001
        logger.debug("formula_of_smiles failed for %s: %s", smiles, exc)
        return None


def _clean_heavy_template(smiles: str | None) -> Any | None:
    """Parse a reference SMILES into a heavy-atom-only template mol.

    Removes explicit hydrogens including those that define double-bond
    stereo (RCSB SMILES often write ``[H]/N=C...``), which would otherwise
    leave an extra atom and break ``AssignBondOrdersFromTemplate``.
    """
    if not smiles:
        return None
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        params = Chem.RemoveHsParameters()
        params.removeDefiningBondStereo = True
        params.removeDegreeZero = True
        m = Chem.RemoveHs(m, params)
    except Exception:
        try:
            m = Chem.RemoveHs(m)
        except Exception:
            return None
    return m


def prepare_ligand_from_smiles(smiles: str, ph: float = 7.4) -> LigandPrep:
    """Prepare a ligand from SMILES only (taxonomy path, no coordinates)."""
    from rdkit import Chem

    prep = LigandPrep(smiles_input=smiles)
    proto = protonate_smiles_at_ph(smiles, ph)
    prep.smiles_protonated = proto
    mol = Chem.MolFromSmiles(proto or smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            prep.warnings.append(f"RDKit could not parse SMILES: {smiles}")
            return prep
    prep.mol = mol
    prep.smiles_prepared = _canonical_smiles(mol)
    prep.bond_assignment = "smiles_only"
    prep.warnings.extend(smiles_advisories(smiles))
    return prep


def _atomgroup_to_pdb_block(ag: Any) -> str | None:
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as fh:
            tmp = Path(fh.name)
        ag.write(str(tmp), file_format="PDB")
        return tmp.read_text()
    except Exception as exc:
        logger.debug("AtomGroup->PDB failed: %s", exc)
        return None
    finally:
        if tmp is not None:
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)


def _heavy_atom_names(ag: Any) -> list[str]:
    names: list[str] = []
    for a in ag.atoms:
        el = (getattr(a, "element", "") or "").upper()
        nm = (getattr(a, "name", "") or "").strip()
        if el == "H" or (not el and nm.upper().startswith("H")):
            continue
        names.append(nm)
    return names


def prepare_ligand_from_structure(
    ag: Any,
    template_smiles: str | None = None,
    ph: float = 7.4,
) -> LigandPrep:
    """Build a chemistry-correct, hydrogenated ligand Mol from structure coords."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    prep = LigandPrep(smiles_input=template_smiles)
    prep.atom_names = _heavy_atom_names(ag)
    prep.warnings.extend(smiles_advisories(template_smiles))

    block = _atomgroup_to_pdb_block(ag)
    if block is None:
        prep.warnings.append("could not extract ligand coordinates")
        return prep

    raw = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False, proximityBonding=True)
    if raw is None:
        prep.warnings.append("RDKit could not read ligand PDB block")
        return prep
    try:
        lig = Chem.RemoveHs(raw, sanitize=False)
    except Exception:
        lig = raw

    # Template path: transfer bond orders + charges from the reference SMILES.
    # Try the pH-protonated form first, then the raw reference. Reference
    # SMILES from RCSB often carry stereo-defining explicit [H] atoms; those
    # must be stripped or the substructure match fails (extra atom).
    if template_smiles:
        proto = protonate_smiles_at_ph(template_smiles, ph)
        prep.smiles_protonated = proto
        templates = [t for t in (_clean_heavy_template(proto),
                                 _clean_heavy_template(template_smiles)) if t is not None]
        last_exc: Exception | None = None
        for template in templates:
            try:
                fixed = AllChem.AssignBondOrdersFromTemplate(template, lig)
                Chem.SanitizeMol(fixed)
                fixed = Chem.AddHs(fixed, addCoords=True)
                prep.mol = fixed
                prep.bond_assignment = "template"
                prep.n_hydrogens = sum(1 for a in fixed.GetAtoms() if a.GetSymbol() == "H")
                prep.smiles_prepared = _canonical_smiles(fixed)
                return prep
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
        if templates and last_exc is not None:
            t_atoms = templates[0].GetNumAtoms()
            if t_atoms != lig.GetNumAtoms():
                prep.warnings.append(
                    f"reference SMILES has {t_atoms} heavy atoms but structure has "
                    f"{lig.GetNumAtoms()} (partial/disordered ligand?); using geometry perception"
                )
            else:
                prep.warnings.append(f"template bond assignment failed: {last_exc}")

    # Fallback: perceive bonds from geometry, trying a range of net charges.
    mol = _perceive_bonds(lig, prep)
    if mol is not None:
        with contextlib.suppress(Exception):
            mol = Chem.AddHs(mol, addCoords=True)
        prep.mol = mol
        prep.n_hydrogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
        prep.smiles_prepared = _canonical_smiles(mol)
    else:
        prep.warnings.append("bond perception failed; returning connectivity-only mol")
        prep.mol = lig
        prep.bond_assignment = "failed"
    return prep


def _perceive_bonds(mol: Any, prep: LigandPrep) -> Any | None:
    from rdkit import Chem

    try:
        from rdkit.Chem import rdDetermineBonds
    except Exception:
        prep.bond_assignment = "sanitise_only"
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    for charge in (0, -1, 1, -2, 2, -3, 3):
        m = Chem.Mol(mol)
        try:
            rdDetermineBonds.DetermineBonds(m, charge=charge)
            Chem.SanitizeMol(m)
            prep.bond_assignment = f"perceived(charge={charge})"
            return m
        except Exception:
            continue
    try:
        Chem.SanitizeMol(mol)
        prep.bond_assignment = "sanitise_only"
        return mol
    except Exception:
        return None


__all__ = [
    "LigandPrep",
    "prepare_ligand_from_smiles",
    "prepare_ligand_from_structure",
    "protonate_smiles_at_ph",
    "inchikey_of_mol",
    "inchikey_of_smiles",
    "inchikey_skeleton",
    "formula_of_smiles",
    "validate_smiles",
    "smiles_advisories",
]
