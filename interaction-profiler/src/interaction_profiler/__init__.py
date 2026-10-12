# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""interaction-profiler (v2): batch medchem taxonomy + structure interaction profiling.

Given a batch of molecules (SMILES and/or structures), emit per molecule:
- a hierarchical medchem functional-group taxonomy (AccFG + curated SMARTS),
- when a structure is supplied: clean ligand + receptor preparation, a rich
  set of geometrically-detected interactions (H-bond with angle/direction,
  hydrophobic, pi-stacking, pi-cation / cation-pi, ionic, halogen bond,
  metal coordination, water bridge, CH-pi, sulfur/selenium-arene,
  chalcogen-pi, amide-pi, covalent, vdW), and a grouped
  ``(fragment, residue, interaction)`` triple mapping.

Also usable as an HTTP service -- the FastAPI app lives in the deployment wrapper
``services/interaction-profiler/app.py`` (see ``SERVICE.md``).

One batch in -> one JSON out.
"""

from __future__ import annotations

import importlib.metadata as _md

try:  # single source of truth = the installed distribution metadata (pyproject)
    __version__ = _md.version("interaction-profiler")
except _md.PackageNotFoundError:  # running from source without an install
    __version__ = "0.3.0"


def __getattr__(name: str):  # lazy to avoid importing heavy deps at package import
    if name in {"profile_batch", "profile_molecule", "to_table"}:
        from interaction_profiler import core

        return getattr(core, name)
    raise AttributeError(f"module 'interaction_profiler' has no attribute {name!r}")


__all__ = ["profile_batch", "profile_molecule", "to_table", "__version__"]
