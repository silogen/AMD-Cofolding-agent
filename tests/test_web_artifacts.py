# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the tool-result classifier behind the web UI.

Run from the repository root::

    python -m pytest tests/test_web_artifacts.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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


def test_a_relative_path_is_read_from_the_shared_volume(shared_root):
    raw = json.dumps({"structures": ["predictions/run_1/model_0.cif"]})
    panel = only(artifacts.extract_panels("list_shared_assets", raw), "structure")
    assert panel["path"] == "predictions/run_1/model_0.cif"


def test_a_relative_path_cannot_escape_the_shared_volume():
    raw = json.dumps({"structures": ["../outside/model_0.cif"]})
    assert artifacts.extract_panels("list_shared_assets", raw) == []


# -- structures carry their molecule and confidence ------------------------


def test_scored_structure_carries_smiles_and_gate_metrics(shared_root):
    path = str(shared_root / "predictions" / "out_1" / "model_0.pdb")
    raw = json.dumps(
        {
            "molecules": [
                {
                    "smiles": "CCO",
                    "total_score": 0.7,
                    "components": {
                        "boltz": {
                            "raw": -1.2,
                            "details": {
                                "structure_path": path,
                                "confidence_score": 0.81,
                                "ligand_iptm": 0.66,
                                "affinity_pred_value": -1.2,
                                "affinity_probability_binary": 0.9,
                            },
                        },
                        "posebusters": {
                            "raw": 1.0,
                            "details": {"n_checks_passed": 18, "n_checks_total": 19, "failed_checks": ["minimum_distance"]},
                        },
                    },
                },
                {"smiles": "CCN", "total_score": 0.1, "components": {}},
            ]
        }
    )

    structure = only(artifacts.extract_panels("score_molecules", raw), "structure")

    assert structure["path"] == "predictions/out_1/model_0.pdb"
    assert structure["smiles"] == "CCO"
    assert structure["metrics"] == {
        "confidence": 0.81,
        "ligandIptm": 0.66,
        "affinity": -1.2,
        "binderProbability": 0.9,
        "posebustersPassed": 18,
        "posebustersTotal": 19,
        "failedChecks": ["minimum_distance"],
    }


# -- ADMET and RetroDMTA ---------------------------------------------------


def test_admet_result_yields_a_property_panel_and_the_molecule():
    raw = json.dumps(
        {
            "service": "admet",
            "n": 1,
            "results": [
                {
                    "input_smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                    "standardized_smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                    "ok": True,
                    "predictions": {
                        "logp": {"mean": 0.05, "std": 0.07, "unit": "logP", "n_models": 3, "per_model": {"a": 0.1}},
                        "aqueous_solubility": {"mean": -0.86, "std": 0.15, "unit": "log10(mol/L)"},
                    },
                    "applicability_domain": {"AD Tanimoto": {"pass": True}, "AD Pass": True},
                }
            ],
        }
    )

    panels = artifacts.extract_panels("predict_admet", raw)

    assert kinds(panels) == ["properties", "molecule"]
    props = panels[0]
    assert props["endpoints"] == ["logp", "aqueous_solubility"]
    molecule = props["molecules"][0]
    assert molecule["smiles"] == "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    assert molecule["adPass"] is True
    assert molecule["values"]["logp"] == {"mean": 0.05, "std": 0.07, "unit": "logP"}
    assert panels[1]["smiles"] == "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_flat_retrodmta_records_pair_mean_and_std():
    raw = json.dumps(
        {
            "service": "retrodmta",
            "target": "FXA",
            "results": [
                {"smiles": "CCO", "pIC50_mean": 6.1, "pIC50_std": 0.3},
                {"smiles": "CCN", "pIC50_mean": 5.2, "pIC50_std": 0.4},
            ],
        }
    )

    props = only(artifacts.extract_panels("predict_retrodmta", raw), "properties")

    assert props["target"] == "FXA"
    assert props["endpoints"] == ["pIC50"]
    assert props["molecules"][1]["values"] == {"pIC50": {"mean": 5.2, "std": 0.4}}


# -- interaction profiles --------------------------------------------------


def test_interaction_profile_yields_contacts_linked_to_the_structure(shared_root):
    path = str(shared_root / "predictions" / "out_1" / "model_0.pdb")
    raw = json.dumps(
        {
            "batch_id": "profile",
            "molecules": [
                {
                    "metadata": {"name": "mol_0", "source": path, "ligand_smiles": "CCO"},
                    "interactions": [
                        {
                            "interaction_type": "hbond",
                            "receptor_chain": "A",
                            "receptor_resname": "ASN",
                            "receptor_resnum": 140,
                            "receptor_atom_name": "ND2",
                            "ligand_atom_name": "O1",
                            "ligand_atom_idx": 2,
                            "distance": 2.9,
                            "fragment": "hydroxyl",
                        },
                        {"interaction_type": "hydrophobic", "receptor_chain": "A", "receptor_resname": "LEU", "receptor_resnum": 92},
                    ],
                    "error": None,
                }
            ],
        }
    )

    panels = artifacts.extract_panels("profile_interactions", raw)

    profile = only(panels, "interactions")
    assert "table" not in kinds(panels)
    assert profile["structurePath"] == "predictions/out_1/model_0.pdb"
    assert profile["smiles"] == "CCO"
    assert profile["counts"] == {"hbond": 1, "hydrophobic": 1}
    first = profile["contacts"][0]
    assert first["label"] == "ASN140"
    assert first["chain"] == "A"
    assert first["ligandAtomIndex"] == 2
    assert first["distance"] == 2.9


def test_profile_without_paths_uses_the_structures_it_was_given(shared_root):
    path = str(shared_root / "predictions" / "out_2" / "model_0.pdb")
    raw = json.dumps(
        {
            "molecules": [
                {
                    "metadata": {"name": "mol_0", "source": None, "ligand_smiles": "CCO"},
                    "composition": {"fragments": [{"name": "benzene", "smiles": "c1ccccc1"}]},
                    "interactions": [{"interaction_type": "ch_pi", "receptor_resname": "TRP", "receptor_resnum": 81}],
                }
            ]
        }
    )

    panels = artifacts.extract_panels("profile_interactions", raw, {"structure_paths": [path]})

    assert kinds(panels) == ["interactions"]
    assert only(panels, "interactions")["structurePath"] == "predictions/out_2/model_0.pdb"


def test_scores_reported_as_text_are_read_as_numbers(shared_root):
    path = str(shared_root / "predictions" / "out_3" / "mol_0_model_0.pdb")
    raw = json.dumps(
        {
            "molecules": [
                {
                    "smiles": "CCO",
                    "components": {
                        "Boltz Affinity": {
                            "details": {"confidence_score": "0.98", "ligand_iptm": "0.97", "structure_path": path}
                        },
                        "Pose Quality": {"details": {"n_checks_passed": "18", "n_checks_total": "22"}},
                    },
                }
            ]
        }
    )

    structure = only(artifacts.extract_panels("score_molecules", raw), "structure")

    assert structure["metrics"] == {"confidence": 0.98, "ligandIptm": 0.97, "posebustersPassed": 18, "posebustersTotal": 22}


def test_the_first_heading_of_an_answer_is_read_without_its_marks():
    from reinvent4_agent.web.sessions import first_heading

    assert first_heading("## Score: `c1ccccc1`\n\nbody") == "Score: c1ccccc1"
    assert first_heading("Some text\n\n### **Top candidates** ###\n") == "Top candidates"
    assert first_heading("No heading here, just prose.") is None
    assert first_heading("#\n") is None
    assert first_heading("") is None


def test_results_recorded_without_panels_get_them_on_load():
    from reinvent4_agent.web.sessions import refresh_panels

    raw = json.dumps({"service": "admet", "results": [{"input_smiles": "CCO", "predictions": {"logp": {"mean": 0.1, "std": 0.02}}}]})
    events = [
        {"type": "tool_call", "id": "c1", "name": "predict_admet", "arguments": {"smiles": ["CCO"]}},
        {"type": "tool_result", "id": "c1", "name": "predict_admet", "result": raw, "isError": False, "panels": []},
        {"type": "tool_result", "id": "c2", "name": "predict_admet", "result": raw, "isError": True, "panels": []},
    ]

    refreshed = refresh_panels(events)

    assert "properties" in kinds(refreshed[1]["panels"])
    assert refreshed[2]["panels"] == []


# -- titles backfilled when an old session is opened -----------------------


def conversation(question="c1cc(COc2cncnc2)ncn1", answer="## Score: aminopyrimidine series\n\nbody"):
    events = [{"type": "user_message", "content": question}]
    if answer is not None:
        events.append({"type": "assistant_message", "content": answer})
    return events


def test_an_automatic_smiles_title_is_replaced_by_the_answer_heading():
    from reinvent4_agent.web.sessions import backfill_title

    meta = {"title": "c1cc(COc2cncnc2)ncn1", "auto_title": True}

    assert backfill_title(meta, conversation()) is True
    assert meta == {"title": "Score: aminopyrimidine series", "auto_title": False}


def test_a_title_recorded_before_auto_title_existed_is_matched_to_the_question():
    from reinvent4_agent.web.sessions import backfill_title

    meta = {"title": "c1cc(COc2cncnc2)ncn1"}

    assert backfill_title(meta, conversation()) is True
    assert meta["title"] == "Score: aminopyrimidine series"


@pytest.mark.parametrize(
    "meta",
    [
        {"title": "BRD4 campaign", "auto_title": False},
        {"title": "BRD4 campaign"},
    ],
    ids=["renamed", "renamed before auto_title existed"],
)
def test_a_title_the_user_chose_is_left_alone(meta):
    from reinvent4_agent.web.sessions import backfill_title

    before = dict(meta)

    assert backfill_title(meta, conversation()) is False
    assert meta == before


@pytest.mark.parametrize(
    "answer",
    [None, "No heading, just prose about the series."],
    ids=["no answer", "answer without a heading"],
)
def test_a_session_with_nothing_to_name_it_after_keeps_its_title(answer):
    from reinvent4_agent.web.sessions import backfill_title

    meta = {"title": "c1cc(COc2cncnc2)ncn1", "auto_title": True}

    assert backfill_title(meta, conversation(answer=answer)) is False
    assert meta["title"] == "c1cc(COc2cncnc2)ncn1"


def test_a_backfilled_title_is_not_rewritten_by_a_later_answer():
    from reinvent4_agent.web.sessions import backfill_title

    meta = {"title": "c1cc(COc2cncnc2)ncn1", "auto_title": True}
    backfill_title(meta, conversation())

    assert backfill_title(meta, conversation(answer="## A later heading\n")) is False
    assert meta["title"] == "Score: aminopyrimidine series"


def test_a_notice_reaches_the_watchers_without_joining_the_conversation(tmp_path):
    """Replayed on a reconnect, a retry notice would be stale and confusing."""
    from reinvent4_agent.web.sessions import LiveSession, SessionStore

    store = SessionStore(tmp_path)
    session_id = "s-20260101-000000-abcdef"
    store.create({"id": session_id, "title": ""})
    live = LiveSession(meta={"id": session_id}, store=store)
    queue = live.subscribe()

    live.publish({"type": "notice", "message": "The model endpoint is not answering, retrying (2 of 3)..."})

    assert queue.get_nowait()["type"] == "notice"
    assert live.events == []
    assert live.last_seq == 0
    assert store.load_events(session_id) == []


def test_opening_a_stored_session_saves_the_backfilled_title(tmp_path):
    """The rename has to reach disk, or the sidebar shows the SMILES again."""
    from reinvent4_agent.web.sessions import SessionManager, SessionStore

    store = SessionStore(tmp_path)
    session_id = "s-20260101-000000-abcdef"
    store.create({"id": session_id, "title": "c1cc(COc2cncnc2)ncn1", "auto_title": True, "turns": 1})
    for event in conversation():
        store.append_event(session_id, event)

    manager = SessionManager(store, lambda handler: SimpleNamespace())
    live = manager.get(session_id)

    assert live is not None
    assert live.meta["title"] == "Score: aminopyrimidine series"
    assert store.load_meta(session_id)["title"] == "Score: aminopyrimidine series"
