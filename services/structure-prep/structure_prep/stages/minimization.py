"""Stage 9 — Optional Minimization.

Uses OpenMM to relax geometry within the specified scope. Always
labelled as geometry cleanup — never as affinity optimisation.
Requires Stage 6 (protonated structure with H atoms).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gemmi

from structure_prep.models import MinimizationResult, Stage9Profile

if TYPE_CHECKING:
    import numpy as np
    from structure_prep.models import LigandParamResult, RoleMap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Force-field file mapping
# ---------------------------------------------------------------------------

# Map Stage9Profile.minimize_ff_protein → list of OpenMM XML filenames.
# Note: ff14SB requires an additional water model for explicit solvent, but
# for implicit solvent the protein XML alone is sufficient.
_FF_PROTEIN_FILES: dict[str, list[str]] = {
    "ff14SB": ["amber14/protein.ff14SB.xml"],
    "AMBER14": ["amber14-all.xml"],
    "CHARMM36": ["charmm36.xml"],
}

# Map Stage9Profile.minimize_ff_nucleic → additional OpenMM XML filenames.
# CHARMM36 nucleic parameters are already included in charmm36.xml.
_FF_NUCLEIC_FILES: dict[str, list[str]] = {
    "DNA.OL3": ["amber14/DNA.OL3.xml"],
    "RNA.OL3": ["amber14/RNA.OL3.xml"],
    "CHARMM36": [],  # included in charmm36.xml already
}

# Map Stage9Profile.minimize_solvent → implicit solvent XML filenames.
_SOLVENT_FILES: dict[str, list[str]] = {
    "implicit_GBn2": ["implicit/gbn2.xml"],
    "implicit_OBC2": ["implicit/obc2.xml"],
    "none": [],
}

# Backbone heavy-atom names used for "backbone" restraint selection.
_BACKBONE_ATOMS: frozenset[str] = frozenset({"N", "CA", "C", "O", "CB"})

# Standard residue names that belong to the polymer/protein entity.
_STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "ASH", "GLH", "LYN", "ARN", "CYX", "CYM",
    "MSE", "SEC", "PYL", "ACE", "NME",
})

_WATER_NAMES: frozenset[str] = frozenset({"HOH", "WAT", "H2O", "TIP", "TIP3", "TIP4", "TIP5"})

# Element symbols for hydrogen / deuterium (for "non_minimized_heavy" selection).
_H_ELEMENTS: frozenset[str] = frozenset({"H", "D", "T"})


# ---------------------------------------------------------------------------
# Scope-atom selection
# ---------------------------------------------------------------------------

def _get_scope_atoms(
    topology: Any,
    pos_A: "np.ndarray",  # shape (N_atoms, 3), Angstroms
    profile: Stage9Profile,
    role_map: "RoleMap",
) -> set[int]:
    """Return the set of atom indices that are FREE to move during minimisation.

    Parameters
    ----------
    topology:
        OpenMM Topology object.
    pos_A:
        Atom positions in Angstroms as a NumPy array of shape (N, 3).
    profile:
        Stage9Profile.
    role_map:
        RoleMap from Stage 2.

    Returns
    -------
    set of integer atom indices (free to move).
    """
    import numpy as _np  # local import — only used if minimization is active

    scope: set[int] = set()

    # Fast paths --------------------------------------------------------
    if profile.minimize == "complete":
        return {i for i in range(len(pos_A))}

    if profile.minimize == "selected_chains":
        target = set(profile.minimize_scope_chains)
        for i, atom in enumerate(topology.atoms()):
            if atom.residue.chain.id in target:
                scope.add(i)
        return scope

    # Classify atoms into ligand / protein / water / nucleic
    polymer_chains: set[str] = set(role_map.protein_chains) | set(role_map.binder_chains)
    nucleic_chains: set[str] = set(role_map.nucleic_acid_chains)
    all_polymer_chains: set[str] = polymer_chains | nucleic_chains

    ligand_indices: list[int] = []
    protein_indices: list[int] = []

    for i, atom in enumerate(topology.atoms()):
        resname = atom.residue.name.strip().upper()
        chain_id = atom.residue.chain.id

        is_water = resname in _WATER_NAMES
        if is_water:
            continue

        is_polymer = (resname in _STANDARD_AA or chain_id in all_polymer_chains)
        if is_polymer:
            protein_indices.append(i)
        else:
            # Non-water, non-polymer → treat as ligand / cofactor
            ligand_indices.append(i)

    # Ligand scope (and base for interface) -----------------------------
    if profile.minimize in ("ligand", "interface"):
        scope.update(ligand_indices)

    if profile.minimize == "interface":
        if not ligand_indices:
            logger.warning(
                "Stage 9: minimize='interface' but no ligand atoms found; "
                "falling back to complete protein scope."
            )
            scope.update(protein_indices)
        else:
            cutoff_A = profile.minimize_interface_cutoff_A
            lig_pos = pos_A[ligand_indices]  # (n_lig, 3)

            for prot_idx in protein_indices:
                prot_pos = pos_A[prot_idx]          # (3,)
                dists = _np.linalg.norm(lig_pos - prot_pos, axis=1)  # (n_lig,)
                if _np.any(dists <= cutoff_A):
                    scope.add(prot_idx)

    return scope


# ---------------------------------------------------------------------------
# Restraint builder
# ---------------------------------------------------------------------------

def _add_restraints(
    system: Any,
    topology: Any,
    positions: Any,  # OpenMM Quantity list
    scope_atoms: set[int],
    profile: Stage9Profile,
) -> None:
    """Add a harmonic positional-restraint force for out-of-scope atoms.

    Modifies *system* in-place.  No-op when ``minimize_restraint_k == 0``
    or ``minimize_restraint_selection == "none"``.
    """
    if profile.minimize_restraint_k <= 0 or profile.minimize_restraint_selection == "none":
        return

    try:
        import openmm  # type: ignore[import]
        import openmm.unit as unit  # type: ignore[import]
    except ImportError as exc:
        logger.warning("Stage 9: openmm not available for restraints: %s", exc)
        return

    k_value = profile.minimize_restraint_k * unit.kilocalories_per_mole / unit.angstrom ** 2

    restraint = openmm.CustomExternalForce(
        "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", k_value)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    n_restrained = 0
    for i, atom in enumerate(topology.atoms()):
        if i in scope_atoms:
            continue  # free to move

        atom_name = atom.name.strip().upper()
        elem_sym = atom.element.symbol.strip().upper() if atom.element else ""

        # Apply restraint selection filter
        should_restrain = False
        if profile.minimize_restraint_selection == "non_minimized_heavy":
            should_restrain = elem_sym not in _H_ELEMENTS
        elif profile.minimize_restraint_selection == "backbone":
            should_restrain = atom_name in _BACKBONE_ATOMS

        if should_restrain:
            x0, y0, z0 = positions[i].value_in_unit(unit.nanometers)
            restraint.addParticle(i, [x0, y0, z0])
            n_restrained += 1

    system.addForce(restraint)
    logger.debug(
        "Stage 9: added positional restraints to %d atom(s) "
        "(k=%.2f kcal/mol/Å², selection=%s).",
        n_restrained,
        profile.minimize_restraint_k,
        profile.minimize_restraint_selection,
    )


# ---------------------------------------------------------------------------
# Core minimisation routine
# ---------------------------------------------------------------------------

def _run_minimization(
    structure: gemmi.Structure,
    profile: Stage9Profile,
    ligand_params: list["LigandParamResult"],
    role_map: "RoleMap",
    job_dir: Path,
) -> tuple[gemmi.Structure, MinimizationResult]:
    """Execute the OpenMM minimisation and return the updated structure."""
    import numpy as _np  # noqa: F401 — used inside _get_scope_atoms

    try:
        import openmm  # type: ignore[import]
        import openmm.app as app  # type: ignore[import]
        import openmm.unit as unit  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"Stage 9: openmm is required for minimisation but is not installed: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 1. Write structure to PDB for OpenMM
    # ------------------------------------------------------------------
    tmp_in = str(job_dir / "minimize_in.pdb")
    tmp_out = str(job_dir / "minimize_out.pdb")
    structure.write_pdb(tmp_in)
    logger.debug("Stage 9: wrote input PDB → %s", tmp_in)

    pdb = app.PDBFile(tmp_in)

    # ------------------------------------------------------------------
    # 2. Assemble force-field file list
    # ------------------------------------------------------------------
    ff_files: list[str] = []

    protein_ff = profile.minimize_ff_protein
    ff_files.extend(_FF_PROTEIN_FILES.get(protein_ff, [f"{protein_ff}.xml"]))

    # Add nucleic acid FF only when AMBER flavour AND structure has nucleic chains
    has_nucleic = bool(role_map and (role_map.nucleic_acid_chains or any(
        role for chain_id, role in (role_map.chain_roles or {}).items()
        if role in ("dna", "rna")
    )))
    if protein_ff != "CHARMM36" and has_nucleic:
        nucleic_ff = profile.minimize_ff_nucleic
        ff_files.extend(_FF_NUCLEIC_FILES.get(nucleic_ff, []))

    # Implicit solvent
    solvent_files = _SOLVENT_FILES.get(profile.minimize_solvent, [])
    ff_files.extend(solvent_files)

    logger.info("Stage 9: force-field files: %s", ff_files)

    try:
        forcefield = app.ForceField(*ff_files)
    except Exception as exc:
        raise RuntimeError(
            f"Stage 9: ForceField construction failed with files {ff_files}: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 3. Modeller
    # ------------------------------------------------------------------
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # ------------------------------------------------------------------
    # 4. Create OpenMM System
    # ------------------------------------------------------------------
    nonbonded_method = app.NoCutoff

    create_kwargs: dict[str, Any] = {
        "nonbondedMethod": nonbonded_method,
        "constraints": app.HBonds,
    }

    try:
        system = forcefield.createSystem(modeller.topology, **create_kwargs)
    except Exception as exc_with_constraints:
        logger.warning(
            "Stage 9: createSystem with HBonds constraints failed (%s); "
            "retrying without constraints.",
            exc_with_constraints,
        )
        try:
            system = forcefield.createSystem(
                modeller.topology, nonbondedMethod=nonbonded_method
            )
        except Exception as exc_bare:
            raise RuntimeError(
                f"Stage 9: createSystem failed: {exc_bare}"
            ) from exc_bare

    # ------------------------------------------------------------------
    # 5. Compute scope atoms and add restraints
    # ------------------------------------------------------------------
    positions = modeller.positions

    # Build positions numpy array in Angstroms for distance-based scope
    pos_A_list = [
        p.value_in_unit(unit.angstrom) for p in positions
    ]
    import numpy as _np
    pos_A = _np.array([[v.x, v.y, v.z] for v in pos_A_list])

    scope_atoms = _get_scope_atoms(
        topology=modeller.topology,
        pos_A=pos_A,
        profile=profile,
        role_map=role_map,
    )
    logger.info(
        "Stage 9: %d of %d atoms are in minimisation scope (%s).",
        len(scope_atoms),
        len(pos_A),
        profile.minimize,
    )

    _add_restraints(
        system=system,
        topology=modeller.topology,
        positions=positions,
        scope_atoms=scope_atoms,
        profile=profile,
    )

    # ------------------------------------------------------------------
    # 6. Integrator and Simulation
    # ------------------------------------------------------------------
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(positions)

    # ------------------------------------------------------------------
    # 7. Record initial energy
    # ------------------------------------------------------------------
    try:
        initial_state = simulation.context.getState(getEnergy=True)
        initial_energy_kj = initial_state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
    except Exception as exc:
        logger.warning("Stage 9: could not read initial energy: %s", exc)
        initial_energy_kj = float("nan")

    logger.info(
        "Stage 9: initial potential energy = %.3f kJ/mol.", initial_energy_kj
    )

    # ------------------------------------------------------------------
    # 8. Run minimisation
    # ------------------------------------------------------------------
    simulation.minimizeEnergy(
        tolerance=profile.minimize_convergence_kj_per_mol * unit.kilojoules_per_mole,
        maxIterations=profile.minimize_max_iterations,
    )

    # ------------------------------------------------------------------
    # 9. Collect final state
    # ------------------------------------------------------------------
    final_state = simulation.context.getState(
        getPositions=True, getEnergy=True
    )
    final_energy_kj = final_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    final_positions = final_state.getPositions()

    energy_delta = final_energy_kj - initial_energy_kj
    converged = abs(energy_delta) < profile.minimize_convergence_kj_per_mol

    logger.info(
        "Stage 9: final potential energy = %.3f kJ/mol (delta = %.3f kJ/mol, "
        "converged=%s).",
        final_energy_kj, energy_delta, converged,
    )

    # ------------------------------------------------------------------
    # 10. Compute maximum coordinate displacement
    # ------------------------------------------------------------------
    final_pos_A = _np.array([
        [p.x, p.y, p.z]
        for p in final_positions.value_in_unit(unit.angstrom)
    ])
    displacements = _np.linalg.norm(final_pos_A - pos_A, axis=1)
    max_disp_A = float(_np.max(displacements)) if len(displacements) > 0 else 0.0

    logger.info("Stage 9: max coordinate displacement = %.3f Å.", max_disp_A)

    # ------------------------------------------------------------------
    # 11. Write minimised structure and re-read with gemmi
    # ------------------------------------------------------------------
    try:
        with open(tmp_out, "w") as fh:
            app.PDBFile.writeFile(
                modeller.topology,
                final_positions,
                fh,
                keepIds=True,
            )
        logger.debug("Stage 9: wrote minimised PDB → %s", tmp_out)
        minimized_structure = gemmi.read_structure(tmp_out)
    except Exception as exc:
        raise RuntimeError(
            f"Stage 9: failed to write / re-read minimised structure: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # 12. Build MinimizationResult
    # ------------------------------------------------------------------
    any_lig_ok = any(
        r.parameterization_status == "ok" for r in ligand_params
    )
    result = MinimizationResult(
        performed=True,
        scope=profile.minimize,
        ff_protein=profile.minimize_ff_protein,
        ff_nucleic=profile.minimize_ff_nucleic,
        ff_ligand="openff_nagl" if any_lig_ok else None,
        solvent=profile.minimize_solvent,
        restraint_k_kcal_mol_A2=profile.minimize_restraint_k,
        n_iterations_run=profile.minimize_max_iterations,
        converged=converged,
        initial_energy_kj_mol=float(initial_energy_kj) if initial_energy_kj == initial_energy_kj else None,
        final_energy_kj_mol=float(final_energy_kj),
        energy_delta_kj_mol=float(energy_delta),
        max_coordinate_displacement_A=max_disp_A,
        label="geometry_cleanup — not an affinity prediction",
    )
    return minimized_structure, result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage9Profile,
    ligand_params: list["LigandParamResult"],
    role_map: "RoleMap",
    job_dir: Path,
) -> tuple[gemmi.Structure, MinimizationResult]:
    """Stage 9 — Optional Energy Minimization.

    Parameters
    ----------
    structure:
        gemmi Structure (after Stage 8; must have H atoms from Stage 6).
    profile:
        Stage9Profile controlling scope, force field, restraints, etc.
    ligand_params:
        List of LigandParamResult from Stage 8 (used to set ``ff_ligand``
        field in the report and resolve ligand vs protein atom identity).
    role_map:
        RoleMap from Stage 2.
    job_dir:
        Job working directory; intermediate PDB files are written here.

    Returns
    -------
    (minimized_structure, MinimizationResult)
        If ``profile.minimize == "none"``, the original structure is returned
        unchanged and ``MinimizationResult(performed=False)``.
    """
    job_dir = Path(job_dir)

    if profile.minimize == "none":
        logger.debug("Stage 9: minimize='none'; skipping.")
        return structure, MinimizationResult(performed=False)

    logger.info(
        "Stage 9: starting minimisation: scope=%s ff=%s solvent=%s "
        "max_iter=%d restraint_k=%.2f.",
        profile.minimize,
        profile.minimize_ff_protein,
        profile.minimize_solvent,
        profile.minimize_max_iterations,
        profile.minimize_restraint_k,
    )

    try:
        return _run_minimization(
            structure=structure,
            profile=profile,
            ligand_params=ligand_params,
            role_map=role_map,
            job_dir=job_dir,
        )
    except Exception as exc:
        logger.exception("Stage 9: minimisation failed: %s", exc)
        # Return the original structure with a failed result so the pipeline
        # can continue rather than crash entirely.
        return structure, MinimizationResult(
            performed=False,
            label="geometry_cleanup_FAILED — " + str(exc)[:200],
        )


__all__ = ["run"]
