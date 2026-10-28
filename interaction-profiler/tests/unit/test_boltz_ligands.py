# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Finding the ligand of Boltz cofolds, and saying so when a file has none.

``boltz_small_ligand.pdb`` is a Boltz-2 output trimmed to the ligand (LIG,
chain B, 12 heavy atoms) and the residues within 7 A of it;
``apo_chain.cif`` is the start of a protein-only chain from the PDB, the
kind of file the profiler was asked about when it answered "no
small-molecule ligand detected".
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def mda():
    return pytest.importorskip("MDAnalysis")


def _boltz_lines() -> list[str]:
    return (DATA / "boltz_small_ligand.pdb").read_text().splitlines()


def _write(tmp_path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def test_boltz_ligand_in_its_own_chain(mda):
    from interaction_profiler.io.structure import load_structure

    [sel] = load_structure(DATA / "boltz_small_ligand.pdb").detect_ligands()
    assert (sel.resname, sel.chain, sel.resnum, sel.n_heavy, sel.in_polymer) == ("LIG", "B", 1, 12, False)


def test_boltz_ligand_written_as_atom_records(mda, tmp_path):
    from interaction_profiler.io.structure import load_structure

    lines = [f"ATOM  {line[6:]}" if line.startswith("HETATM") else line for line in _boltz_lines()]
    [sel] = load_structure(_write(tmp_path, "atom_records.pdb", lines)).detect_ligands()
    assert (sel.resname, sel.chain) == ("LIG", "B")


def test_boltz_unl_ligand_with_few_atoms(mda, tmp_path):
    from interaction_profiler.io.structure import load_structure

    lines, kept = [], 0
    for line in _boltz_lines():
        if " LIG " in line:
            kept += 1
            if kept > 4:
                continue
            line = line.replace(" LIG ", " UNL ")
        lines.append(line)
    [sel] = load_structure(_write(tmp_path, "fragment.pdb", lines)).detect_ligands()
    assert (sel.resname, sel.chain, sel.n_heavy) == ("UNL", "B", 4)


def test_small_group_next_to_the_protein_is_still_skipped(mda, tmp_path):
    from interaction_profiler.io.structure import load_structure

    lines, kept = [], 0
    for line in _boltz_lines():
        if " LIG " in line:
            kept += 1
            if kept > 4:
                continue
            line = line[:21] + "A" + line[22:]
        lines.append(line)
    assert load_structure(_write(tmp_path, "same_chain.pdb", lines)).detect_ligands() == []


def test_protein_only_structure_says_so(mda):
    from interaction_profiler.core import profile_molecule_all
    from interaction_profiler.io.batch_input import MoleculeInput

    [res] = profile_molecule_all(MoleculeInput(name="mol_3", structure=str(DATA / "apo_chain.cif")))
    assert res.error == "no ligand in this structure (protein chain A only)"
