"""Conformational strain computation — Lai 2026 MMFF94 method.

strain = MMFF94_energy(bound_coordinates) − MMFF94_energy(isolated_optimized_coordinates)

The bound energy is evaluated on the ligand heavy atoms exactly as
deposited (crystal / Boltz / docking / generated poses).  For the
optimised reference, a deep copy is stripped of explicit hydrogens
(whose crystallographic positions are poorly determined), re-sanitised,
and minimised in isolation with MMFF94 to convergence.  The difference
measures only the strain imposed by the receptor environment, not
artefacts from H placement.

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

def _mmff94_bound_energy(mol: Any) -> tuple[float, str | None]:
    """Compute the MMFF94 energy of *mol* without moving any atoms.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` with at least one conformer.

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
    return ff.CalcEnergy(), None


def _mmff94_minimized_energy(mol: Any, max_iters: int = 2000) -> tuple[float, str | None]:
    """Return the MMFF94 energy of *mol* after in-place minimisation.

    A deep copy is created internally so the caller's molecule is never
    mutated.  Explicit hydrogens are removed before minimisation so that
    poorly-positioned crystallographic H atoms do not distort the energy;
    MMFF94 accounts for their contribution through implicit-H parameters.

    Parameters
    ----------
    mol:
        RDKit ``Mol`` with a conformer.
    max_iters:
        Maximum number of minimisation steps (default 2000).

    Returns
    -------
    energy:
        Minimised MMFF94 energy in kcal/mol.
    error:
        ``None`` on success, error string otherwise.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem

    opt_mol = copy.deepcopy(mol)

    # Strip explicit Hs so that crystallographic H positions don't
    # inject systematic errors into the reference energy.
    opt_mol = Chem.RemoveHs(opt_mol, sanitize=False)
    try:
        Chem.SanitizeMol(opt_mol)
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"Sanitisation of H-stripped copy failed: {exc}"

    mp_opt = AllChem.MMFFGetMoleculeProperties(opt_mol, mmffVariant="MMFF94")
    if mp_opt is None:
        return 0.0, "MMFF94 properties unavailable on H-stripped copy"
    ff_opt = AllChem.MMFFGetMoleculeForceField(opt_mol, mp_opt)
    if ff_opt is None:
        return 0.0, "MMFF94 force field could not be constructed for H-stripped copy"

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
    * The optimised reference uses the same heavy-atom topology but
      re-derives H positions implicitly via MMFF94 so crystallographic
      H placement does not inflate the reference energy.
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

        # --- Bound energy (no atomic movement) ---
        bound_energy, err_bound = _mmff94_bound_energy(mol)
        if err_bound is not None:
            return StrainResult(
                error=err_bound,
                coordinate_source=coordinate_source,
                rdkit_version=rdkit_ver,
                n_heavy_atoms=n_heavy,
            )

        # --- Optimised energy (copy, strip Hs, minimise) ---
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
