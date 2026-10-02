"""Conformational strain computation — Lai 2026 MMFF94 method.

strain = MMFF94_energy(bound_pose) − MMFF94_energy(isolated_optimized_pose)

Both energies are evaluated on the **same all-atom system**.  MMFF94 is an
all-atom force field, so the bound and reference energies must contain the
identical set of atoms — otherwise the difference is dominated by the
hydrogen-term bookkeeping offset (tens of kcal/mol that scale with H count)
rather than by the conformational distortion we want to measure.

To that end the molecule is first given explicit hydrogens (added with
coordinates when absent).  The bound energy is then evaluated with the heavy
atoms held exactly as deposited (crystal / Boltz / docking / generated poses)
while only the hydrogens are relaxed — this removes artefacts from poorly
determined crystallographic H positions without moving the heavy-atom pose.
The reference is a deep copy, fully minimised with MMFF94 to convergence
(hydrogens included).  The difference therefore isolates the heavy-atom
conformational strain imposed by the receptor environment.

Public API
----------
compute_strain_from_mol        RDKit Mol already in memory
compute_strain_from_sdf        Load from an SDF file
compute_strain_from_smiles_3d  Parse from a MOL-block or PDB-block string
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy RDKit import (mirrors the pattern used elsewhere in this package)
# ---------------------------------------------------------------------------
_RDKIT_VERSION: str | None = None


def _rdkit_version() -> str | None:
    """Return the installed RDKit version string, or None if unavailable."""
    global _RDKIT_VERSION
    if _RDKIT_VERSION is None:
        try:
            import rdkit  # noqa: PLC0415

            _RDKIT_VERSION = getattr(rdkit, "__version__", "unknown")
        except ImportError:
            _RDKIT_VERSION = "unavailable"
    return _RDKIT_VERSION if _RDKIT_VERSION != "unavailable" else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_explicit_hs(mol: Any) -> tuple[Any, bool]:
    """Return *mol* with explicit hydrogens, adding them (with coordinates) if absent.

    MMFF94 strain requires the bound and reference energies to be computed on the
    identical atom set.  If the molecule has no explicit hydrogens, add them with
    3D coordinates so both energies are all-atom and directly comparable.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` with at least one conformer.

    Returns
    -------
    (mol, added):
        *mol* with explicit Hs; *added* is ``True`` when hydrogens were added.
    """
    from rdkit import Chem  # noqa: PLC0415

    has_h = any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    if has_h:
        return mol, False
    # addCoords places hydrogens using the existing heavy-atom conformer.
    return Chem.AddHs(mol, addCoords=True), True


def _mmff94_bound_energy(
    mol: Any,
    relax_hydrogens: bool = True,
    max_iters: int = 500,
) -> tuple[float, str | None]:
    """Compute the MMFF94 energy of *mol* at its bound pose.

    The heavy atoms are held exactly as deposited; only hydrogens are relaxed
    (when ``relax_hydrogens`` is True) so that poorly-positioned crystallographic
    H atoms do not distort the energy.  Crucially the atom set is left unchanged,
    so this energy is directly comparable to the fully-minimised reference.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` with at least one conformer (ideally with explicit H).
    relax_hydrogens:
        When True, minimise only hydrogen positions with every heavy atom fixed.
    max_iters:
        Maximum minimisation steps for the hydrogen-only relaxation.

    Returns
    -------
    energy:
        MMFF94 energy in kcal/mol.
    error:
        ``None`` on success, error string otherwise.
    """
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if mp is None:
        return 0.0, "MMFF94 properties unavailable for this molecule (unsupported atom types?)"
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    if ff is None:
        return 0.0, "MMFF94 force field could not be constructed"

    if relax_hydrogens:
        n_atoms = mol.GetNumAtoms()
        n_fixed = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                ff.AddFixedPoint(atom.GetIdx())
                n_fixed += 1
        # Only worth minimising if some hydrogens are free to move.
        if 0 < n_fixed < n_atoms:
            ff.Minimize(maxIts=max_iters)

    return ff.CalcEnergy(), None


def _mmff94_minimized_energy(mol: Any, max_iters: int = 2000) -> tuple[float, str | None]:
    """Return the MMFF94 energy of *mol* after full in-place minimisation.

    A deep copy is created internally so the caller's molecule is never mutated.
    All atoms — including hydrogens — are retained and minimised, so this energy
    uses the same atomic system as :func:`_mmff94_bound_energy`.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` with a conformer (ideally with explicit H).
    max_iters:
        Maximum number of minimisation steps (default 2000).

    Returns
    -------
    energy:
        Minimised MMFF94 energy in kcal/mol.
    error:
        ``None`` on success, error string otherwise.
    """
    from rdkit.Chem import AllChem  # noqa: PLC0415

    opt_mol = copy.deepcopy(mol)

    mp_opt = AllChem.MMFFGetMoleculeProperties(opt_mol, mmffVariant="MMFF94")
    if mp_opt is None:
        return 0.0, "MMFF94 properties unavailable on optimised copy"
    ff_opt = AllChem.MMFFGetMoleculeForceField(opt_mol, mp_opt)
    if ff_opt is None:
        return 0.0, "MMFF94 force field could not be constructed for optimised copy"

    converged = ff_opt.Minimize(maxIts=max_iters)
    if converged != 0:
        logger.debug(
            "MMFF94 minimisation did not converge within %d iterations (code %d); "
            "using current geometry.",
            max_iters,
            converged,
        )

    return ff_opt.CalcEnergy(), None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def compute_strain_from_mol(
    mol: Any,
    coordinate_source: str = "crystal",
) -> "StrainResult":
    """Compute MMFF94 conformational strain for a ligand already in memory.

    Implements the Lai 2026 method: the energy difference between the
    molecule in its bound pose and the same molecule freely minimised in
    isolation, normalised by heavy-atom count.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` object with a valid 3D conformer.  The molecule
        should have been prepared (bond orders / formal charges assigned)
        before calling this function.  Explicit hydrogens are optional but
        recommended for realistic bound-energy evaluation.
    coordinate_source:
        Provenance tag for the 3D coordinates: one of
        ``"crystal"``, ``"boltz"``, ``"gnina"``, ``"generated"``.

    Returns
    -------
    StrainResult
        Populated on success; ``error`` field is set on failure.

    Notes
    -----
    * All heavy atoms must have an assigned MMFF94 atom type.  Molecules
      with exotic functionality (e.g. rare metals, unusual valence) may
      fail MMFF parameterisation, in which case ``error`` is set.
    * Explicit hydrogens are added when absent so the bound and reference
      energies share the same all-atom system.  On the bound pose only the
      hydrogens are relaxed (heavy atoms fixed), so poorly determined
      crystallographic H placement does not inflate the strain.
    """
    from interaction_profiler.schemas import StrainResult  # noqa: PLC0415

    rdkit_ver = _rdkit_version()

    if mol is None:
        return StrainResult(
            error="mol is None",
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )

    # Must have at least one conformer
    try:
        if mol.GetNumConformers() == 0:
            return StrainResult(
                error="mol has no 3D conformer",
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
            )
    except Exception as exc:  # noqa: BLE001
        return StrainResult(
            error=f"Could not inspect mol conformers: {exc}",
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )

    try:
        n_heavy = mol.GetNumHeavyAtoms()

        # Ensure the molecule carries explicit hydrogens so that the bound and
        # reference energies are computed on the identical all-atom system.
        # (MMFF94 is all-atom; comparing an H-containing energy to an H-free one
        # yields a meaningless H-count offset rather than true strain.)
        try:
            mol, _added_h = _ensure_explicit_hs(mol)
        except Exception as exc:  # noqa: BLE001
            return StrainResult(
                error=f"Could not add explicit hydrogens: {exc}",
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
                n_heavy_atoms=n_heavy,
            )

        # --- Bound energy (heavy atoms fixed; only H relaxed) ---
        bound_energy, err_bound = _mmff94_bound_energy(mol)
        if err_bound is not None:
            return StrainResult(
                error=err_bound,
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
                n_heavy_atoms=n_heavy,
            )

        # --- Optimised energy (copy, full minimisation, same atom set) ---
        opt_energy, err_opt = _mmff94_minimized_energy(mol)
        if err_opt is not None:
            return StrainResult(
                error=err_opt,
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
                n_heavy_atoms=n_heavy,
                bound_energy_kcal_mol=bound_energy,
            )

        # --- Strain ---
        strain = bound_energy - opt_energy
        strain_per_ha = strain / n_heavy if n_heavy > 0 else None

        return StrainResult(
            bound_energy_kcal_mol=bound_energy,
            optimized_energy_kcal_mol=opt_energy,
            strain_kcal_mol=strain,
            strain_per_heavy_atom=strain_per_ha,
            n_heavy_atoms=n_heavy,
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in compute_strain_from_mol")
        return StrainResult(
            error=str(exc),
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )


def compute_strain_from_sdf(
    sdf_path: str,
    coordinate_source: str = "crystal",
) -> "StrainResult":
    """Load the first molecule from an SDF file and compute its strain.

    Parameters
    ----------
    sdf_path:
        Absolute or relative path to a valid ``.sdf`` file.  Only the
        first record is read.
    coordinate_source:
        Provenance tag forwarded to :func:`compute_strain_from_mol`.

    Returns
    -------
    StrainResult
        See :func:`compute_strain_from_mol` for field descriptions.
    """
    from interaction_profiler.schemas import StrainResult  # noqa: PLC0415

    rdkit_ver = _rdkit_version()
    try:
        from rdkit.Chem import SDMolSupplier  # noqa: PLC0415

        supplier = SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
        mol = next(iter(supplier), None)
        if mol is None:
            return StrainResult(
                error=f"No valid molecule found in SDF: {sdf_path}",
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
            )
        return compute_strain_from_mol(mol, coordinate_source=coordinate_source)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error reading SDF for strain computation: %s", sdf_path)
        return StrainResult(
            error=str(exc),
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )


def compute_strain_from_smiles_3d(
    mol_block: str,
    coordinate_source: str = "crystal",
) -> "StrainResult":
    """Parse a MOL-block or PDB-block string and compute conformational strain.

    The function first tries to parse the string as a MOL/SDF block (via
    :func:`rdkit.Chem.MolFromMolBlock`).  If that fails, it falls back to
    PDB-block parsing (via :func:`rdkit.Chem.MolFromPDBBlock`).

    Parameters
    ----------
    mol_block:
        A text block in MOL/SDF or PDB format containing 3D coordinates.
    coordinate_source:
        Provenance tag forwarded to :func:`compute_strain_from_mol`.

    Returns
    -------
    StrainResult
        See :func:`compute_strain_from_mol` for field descriptions.
    """
    from interaction_profiler.schemas import StrainResult  # noqa: PLC0415

    rdkit_ver = _rdkit_version()
    try:
        from rdkit.Chem import MolFromMolBlock, MolFromPDBBlock  # noqa: PLC0415

        mol = MolFromMolBlock(mol_block, removeHs=False, sanitize=True)
        parse_method = "MolFromMolBlock"

        if mol is None:
            mol = MolFromPDBBlock(mol_block, removeHs=False, sanitize=True)
            parse_method = "MolFromPDBBlock"

        if mol is None:
            return StrainResult(
                error="Could not parse mol_block as MOL or PDB format",
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
            )

        logger.debug("Parsed mol_block via %s for strain computation.", parse_method)
        return compute_strain_from_mol(mol, coordinate_source=coordinate_source)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error parsing mol_block for strain computation")
        return StrainResult(
            error=str(exc),
            coordinate_source=coordinate_source,
            rdkit_version=rdkit_ver,
        )


__all__ = [
    "compute_strain_from_mol",
    "compute_strain_from_sdf",
    "compute_strain_from_smiles_3d",
]
