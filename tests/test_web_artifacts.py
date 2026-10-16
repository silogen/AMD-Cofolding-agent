# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the tool-result classifier behind the web UI.

Run from the repository root::

    python -m pytest tests/test_web_artifacts.py
"""

from __future__ import annotations

import json

import pytest
from reinvent4_agent.web import artifacts


@pytest.fixture(autouse=True)
def shared_root(tmp_path, monkeypatch):
    """Point the artifact root at a temporary directory for every test."""
    monkeypatch.setattr(artifacts, "SHARED_ROOT", tmp_path)
    return tmp_path


def kinds(panels):
    return [panel["kind"] for panel in panels]


def only(panels, kind):
    matches = [panel for panel in panels if panel["kind"] == kind]
    assert matches, f"no {kind} panel in {kinds(panels)}"
    return matches[0]


# -- the uniform /score contract -------------------------------------------


def test_score_result_yields_structure_molecule_and_table(shared_root):
    (shared_root / "out").mkdir()
    (shared_root / "out" / "lig1.cif").write_text("data_x\n")

    raw = json.dumps(
        {
            "scores": [0.82, 0.41],
            "smiles": ["CCO", "c1ccccc1"],
            "metadata": {"affinity_pred_value": [-8.1, -6.2]},
            "artifacts": {"structures": [str(shared_root / "out" / "lig1.cif")]},
        }
    )
    panels = artifacts.extract_panels("score_molecules", raw)

    structure = only(panels, "structure")
    assert structure["path"] == "out/lig1.cif"
    assert structure["format"] == "cif"
    assert structure["url"] == "/api/artifacts/out/lig1.cif"

    table = only(panels, "table")
    assert table["columns"] == ["#", "smiles", "score", "affinity_pred_value"]
    assert table["rows"][0] == {
        "#": 1,
        "smiles": "CCO",
        "score": 0.82,
        "affinity_pred_value": -8.1,
    }
    assert table["totalRows"] == 2
    assert table["truncated"] is False

    assert {panel["smiles"] for panel in panels if panel["kind"] == "molecule"} == {
        "CCO",
        "c1ccccc1",
    }


def test_metadata_of_the_wrong_length_is_not_made_a_column():
    raw = json.dumps({"scores": [0.1, 0.2], "metadata": {"model": "boltz2", "notes": ["a"]}})
    table = only(artifacts.extract_panels("score_molecules", raw), "table")
    assert table["columns"] == ["#", "score"]


def test_large_tables_are_truncated_and_say_so():
    raw = json.dumps({"scores": [0.5] * 600, "smiles": ["CCO"] * 600})
    table = only(artifacts.extract_panels("score_molecules", raw), "table")
    assert len(table["rows"]) == artifacts.MAX_TABLE_ROWS
    assert table["totalRows"] == 600
    assert table["truncated"] is True


# -- record lists ----------------------------------------------------------


def test_record_list_becomes_a_table_with_identity_columns_first():
    raw = json.dumps(
        {
            "candidates": [
                {"mw": 45.1, "smiles": "CCN", "qed": 0.7},
                {"mw": 44.1, "smiles": "CCC", "qed": 0.5},
            ]
        }
    )
    table = only(artifacts.extract_panels("get_top_candidates", raw), "table")
    assert table["columns"][:2] == ["#", "smiles"]
    assert set(table["columns"]) == {"#", "smiles", "mw", "qed"}


def test_nested_values_are_dropped_from_cells():
    raw = json.dumps(
        [
            {"smiles": "CCO", "score": 1.0, "detail": {"a": 1}},
            {"smiles": "CCC", "score": 2.0, "detail": {"a": 2}},
        ]
    )
    table = only(artifacts.extract_panels("read_results", raw), "table")
    assert "detail" not in table["columns"]


def test_a_single_record_is_not_a_table():
    raw = json.dumps([{"smiles": "CCO", "score": 1.0}])
    assert "table" not in kinds(artifacts.extract_panels("read_results", raw))


# -- tolerance -------------------------------------------------------------


def test_non_json_result_produces_no_panels():
    assert artifacts.extract_panels("anything", "not json at all") == []


def test_discovery_result_produces_no_panels():
    raw = json.dumps({"components": ["QED", "MolecularWeight"]})
    assert artifacts.extract_panels("list_scoring_components", raw) == []


# -- path safety -----------------------------------------------------------


def test_structures_outside_the_shared_volume_are_ignored():
    raw = json.dumps({"artifacts": {"structures": ["/etc/secret.pdb"]}})
    assert artifacts.extract_panels("score_molecules", raw) == []


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd", "out/../../etc/passwd", ""])
def test_artifact_requests_cannot_escape_the_shared_volume(path):
    assert artifacts.resolve_artifact(path) is None


def test_a_real_file_inside_the_shared_volume_resolves(shared_root):
    (shared_root / "out").mkdir()
    target = shared_root / "out" / "lig1.cif"
    target.write_text("data_x\n")
    assert artifacts.resolve_artifact("out/lig1.cif") == target


def test_a_directory_is_not_served(shared_root):
    (shared_root / "out").mkdir()
    assert artifacts.resolve_artifact("out") is None
