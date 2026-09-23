"""Contract tests for block 09 structure-conditioned Boltz support."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_SERVICE_DIR = _ROOT / "services" / "boltz2"
sys.path.insert(0, str(_SERVICE_DIR))

_app_spec = importlib.util.spec_from_file_location(
    "boltz2_structure_app",
    _SERVICE_DIR / "app.py",
)
assert _app_spec and _app_spec.loader
boltz_app = importlib.util.module_from_spec(_app_spec)
sys.modules["boltz2_structure_app"] = boltz_app
_app_spec.loader.exec_module(boltz_app)

_helper_spec = importlib.util.spec_from_file_location(
    "structure_conditioning",
    _SERVICE_DIR / "structure_conditioning.py",
)
assert _helper_spec and _helper_spec.loader
structure_conditioning = importlib.util.module_from_spec(_helper_spec)
sys.modules["structure_conditioning"] = structure_conditioning
_helper_spec.loader.exec_module(structure_conditioning)


@pytest.fixture()
def shared_pdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(structure_conditioning, "_SHARED_ROOT", shared)
    pdb = shared / "prepared.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       1.0   1.0   1.0  1.00  0.00           C\n"
        "ATOM      2  CA  GLY B   2       2.0   2.0   2.0  1.00  0.00           C\n",
        encoding="utf-8",
    )
    return pdb


def test_template_validation_and_hash(shared_pdb: Path) -> None:
    expected = hashlib.sha256(shared_pdb.read_bytes()).hexdigest()
    resolved, actual, chains = structure_conditioning.validate_template_input(
        str(shared_pdb),
        expected,
        ["A"],
    )
    assert resolved == shared_pdb.resolve()
    assert actual == expected
    assert chains == ["A", "B"]


def test_template_validation_rejects_hash_mismatch(shared_pdb: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        structure_conditioning.validate_template_input(
            str(shared_pdb),
            "deadbeef" * 8,
            ["A"],
        )


def test_template_validation_rejects_missing_chain(shared_pdb: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        structure_conditioning.validate_template_input(
            str(shared_pdb),
            None,
            ["Z"],
        )


def test_template_validation_rejects_path_escape(
    shared_pdb: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.pdb"
    outside.write_text("ATOM\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the shared"):
        structure_conditioning.validate_template_input(
            str(outside),
            None,
            ["A"],
        )


def test_mmcif_chain_parser_tracks_auth_asym_column(tmp_path: Path) -> None:
    cif = tmp_path / "prepared.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.auth_asym_id\n"
        "ATOM 1 A\n"
        "ATOM 2 B\n"
        "#\n",
        encoding="utf-8",
    )
    assert structure_conditioning._parse_mmcif_chains(cif) == {"A", "B"}


def test_template_id_assignment_and_yaml_shape(shared_pdb: Path) -> None:
    template_ids = structure_conditioning.auto_assign_template_ids(["A", "B"])
    assert template_ids == ["A1", "B1"]
    assert structure_conditioning.infer_template_ids(
        shared_pdb,
        ["A", "B"],
    ) == ["A1", "B1"]
    assert structure_conditioning.infer_template_ids(
        shared_pdb.with_suffix(".cif"),
        ["A", "B"],
    ) == ["A", "B"]

    template = structure_conditioning.build_template_dict(
        shared_pdb,
        ["A"],
        ["A1"],
        force=True,
        threshold=1.5,
    )
    assert template == {
        "pdb": str(shared_pdb),
        "chain_id": ["A"],
        "template_id": ["A1"],
        "force": True,
        "threshold": 1.5,
    }


def test_sequence_only_yaml_is_backward_compatible() -> None:
    result = boltz_app._build_molecule_yaml(
        "CC",
        "MAGK",
        None,
        msa_free=True,
    )
    assert set(result) == {"version", "sequences", "properties"}
    assert result["sequences"][0]["protein"]["msa"] == "empty"
    assert "templates" not in result


def test_structure_template_is_injected_into_yaml(shared_pdb: Path) -> None:
    template = {
        "pdb": str(shared_pdb),
        "chain_id": ["A"],
        "template_id": ["A1"],
    }
    result = boltz_app._build_molecule_yaml(
        "CCO",
        "MAGK",
        None,
        template_dict=template,
    )
    assert result["templates"] == [template]


def test_score_params_preserve_sequence_only_default() -> None:
    params = boltz_app.ScoreParams(
        protein_sequence="MAGK",
        msa_free_fallback=True,
    )
    assert params.structure_template is None


def test_parse_all_samples_ranks_confidence(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    molecule_dir = predictions / "mol_0"
    molecule_dir.mkdir(parents=True)
    for sample_idx, confidence in enumerate((0.4, 0.9, 0.6)):
        (molecule_dir / f"confidence_mol_0_model_{sample_idx}.json").write_text(
            json.dumps(
                {
                    "confidence_score": confidence,
                    "ligand_iptm": confidence - 0.1,
                }
            ),
            encoding="utf-8",
        )
        (molecule_dir / f"mol_0_model_{sample_idx}.pdb").write_text(
            f"ATOM {sample_idx}\n",
            encoding="utf-8",
        )
    (molecule_dir / "affinity_mol_0.json").write_text(
        json.dumps({"affinity_pred_value": 1.2}),
        encoding="utf-8",
    )

    samples = boltz_app._parse_all_samples(
        predictions,
        0,
        3,
        "pred_value",
        "pdb",
    )
    assert [sample.sample_idx for sample in samples] == [1, 2, 0]
    assert samples[0].structure_path
    assert samples[0].affinity_pred_value == 1.2


def test_predict_structure_rejects_invalid_smiles(
    shared_pdb: Path,
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(boltz_app.app)
    response = client.post(
        "/predict_structure",
        json={
            "smiles": "NOT_A_SMILES",
            "params": {
                "protein_sequence": "MAGK",
                "msa_free_fallback": True,
                "template": {
                    "structure_path": str(shared_pdb),
                    "chain_ids": ["A"],
                },
            },
        },
    )
    assert response.status_code == 422
    assert "invalid smiles" in response.json()["detail"].lower()


def test_predict_structure_rejects_template_id_count_mismatch(
    shared_pdb: Path,
) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(boltz_app.app)
    response = client.post(
        "/predict_structure",
        json={
            "smiles": "CC",
            "params": {
                "protein_sequence": "MAGK",
                "msa_free_fallback": True,
                "template": {
                    "structure_path": str(shared_pdb),
                    "chain_ids": ["A"],
                    "template_ids": ["A1", "B1"],
                },
            },
        },
    )
    assert response.status_code == 422
    assert "same length" in response.json()["detail"]


def test_yaml_writer_propagates_template(
    tmp_path: Path,
    shared_pdb: Path,
) -> None:
    template = {
        "pdb": str(shared_pdb),
        "chain_id": ["A"],
        "template_id": ["A1"],
    }
    paths = boltz_app._write_batch_yamls(
        tmp_path / "batch",
        ["CC", "CCO"],
        "MAGK",
        None,
        template_dict=template,
    )
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            assert yaml.safe_load(handle)["templates"] == [template]
