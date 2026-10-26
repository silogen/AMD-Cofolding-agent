"""Ligand and complex metric modules for the interaction-profiler upgrade.

This package provides three independent, lazily-importable metric
computations.  Each module can be used standalone or wired into the
:class:`~interaction_profiler.schemas.ComplexResult` pipeline.

Modules
-------
strain
    Conformational strain of a small-molecule ligand using the Lai 2026
    MMFF94 method.  Entry point: :func:`~interaction_profiler.metrics.strain.compute_strain_from_mol`.

efficiency
    Ligand efficiency metrics (LE, LLE, BEI) with priority-based affinity
    selection (experimental > Boltz > model > docking).
    Entry point: :func:`~interaction_profiler.metrics.efficiency.compute_efficiency`.

interface
    Buried surface area (FreeSASA Shrake-Rupley) and simplified shape
    complementarity (Sc) for binder–target pairs.
    Entry point: :func:`~interaction_profiler.metrics.interface.compute_interface_geometry`.

Result schemas are defined in :mod:`interaction_profiler.schemas`:
    * :class:`~interaction_profiler.schemas.StrainResult`
    * :class:`~interaction_profiler.schemas.EfficiencyResult`
    * :class:`~interaction_profiler.schemas.InterfaceGeometry`
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Strain
# ---------------------------------------------------------------------------
from interaction_profiler.metrics.strain import (
    compute_strain_from_mol,
    compute_strain_from_sdf,
    compute_strain_from_smiles_3d,
)

# ---------------------------------------------------------------------------
# Ligand efficiency
# ---------------------------------------------------------------------------
from interaction_profiler.metrics.efficiency import (
    compute_efficiency,
)

# ---------------------------------------------------------------------------
# Interface geometry
# ---------------------------------------------------------------------------
from interaction_profiler.metrics.interface import (
    compute_interface_geometry,
)

__all__ = [
    # strain
    "compute_strain_from_mol",
    "compute_strain_from_sdf",
    "compute_strain_from_smiles_3d",
    # efficiency
    "compute_efficiency",
    # interface
    "compute_interface_geometry",
]
