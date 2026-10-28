# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures and helpers.

Provides:
- ``example_pdb`` fixture: path to the bundled Boltz-2 example structure
  (``examples/1OIU_boltz.pdb``), or a clean skip if it is missing.
- ``requires_full_stack``: skips a test when any heavy scientific dep
  (RDKit, MDAnalysis, AccFG, OpenMM, PDBFixer) is not importable, so a
  fresh checkout still passes the unit tests without the full stack.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# Cached at collection time.
HAS_RDKIT = _try_import("rdkit")
HAS_MDA = _try_import("MDAnalysis")
HAS_ACCFG = _try_import("accfg")
HAS_OPENMM = _try_import("openmm")
HAS_PDBFIXER = _try_import("pdbfixer")
HAS_FULL_STACK = HAS_RDKIT and HAS_MDA and HAS_ACCFG and HAS_OPENMM and HAS_PDBFIXER


@pytest.fixture
def example_pdb() -> Path:
    """Path to the bundled Boltz-2 example structure, or skip if absent."""
    p = EXAMPLES_DIR / "1OIU_boltz.pdb"
    if not p.exists():
        pytest.skip(f"example structure not found at {p}")
    return p


@pytest.fixture
def requires_full_stack():
    """Skip the test if any heavy dependency is missing."""
    missing = [
        name
        for name, present in [
            ("rdkit", HAS_RDKIT),
            ("MDAnalysis", HAS_MDA),
            ("accfg", HAS_ACCFG),
            ("openmm", HAS_OPENMM),
            ("pdbfixer", HAS_PDBFIXER),
        ]
        if not present
    ]
    if missing:
        pytest.skip(f"Missing required deps for integration test: {', '.join(missing)}")
