# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for batch input parsing."""

from __future__ import annotations

import json

from interaction_profiler.io.batch_input import load_batch


def test_load_json_list(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps([
        {"name": "A", "smiles": "CCO"},
        {"name": "B", "structure": "x.pdb", "ligand_resname": "LIG", "ph": 6.5},
    ]))
    batch_id, mols = load_batch(p)
    assert batch_id == "b"
    assert len(mols) == 2
    assert mols[0].name == "A" and mols[0].smiles == "CCO"
    assert mols[1].ligand_resname == "LIG" and mols[1].ph == 6.5
    # relative structure path resolved against batch dir
    assert mols[1].structure.endswith("x.pdb")


def test_load_json_object_with_molecules(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"batch_id": "run1", "molecules": [{"name": "M", "smiles": "C"}]}))
    batch_id, mols = load_batch(p)
    assert batch_id == "run1"
    assert mols[0].name == "M"


def test_load_csv(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("name,smiles,ph\nA,CCO,7.4\nB,c1ccccc1,7.0\n")
    _, mols = load_batch(p)
    assert len(mols) == 2
    assert mols[0].smiles == "CCO"
    assert mols[1].ph == 7.0
