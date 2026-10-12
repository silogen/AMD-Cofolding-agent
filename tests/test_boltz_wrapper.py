# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Boltz2 FastAPI wrapper service.

Covers:
- YAML generation per SMILES (protein+ligand structure, MSA handling)
- Batch YAML writing to disk
- CLI command construction for ``boltz predict``
- Output directory parsing (affinity JSON, confidence JSON, structure files)
- Response assembly from parsed outputs
- FastAPI endpoint validation and error responses
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import importlib.util

import pytest
import yaml
from fastapi.testclient import TestClient

_boltz_spec = importlib.util.spec_from_file_location(
    "boltz2_app",
    str(Path(__file__).resolve().parent.parent / "services" / "boltz2" / "app.py"),
)
boltz_app = importlib.util.module_from_spec(_boltz_spec)
sys.modules["boltz2_app"] = boltz_app
_boltz_spec.loader.exec_module(boltz_app)

ScoreParams = boltz_app.ScoreParams
_assemble_response = boltz_app._assemble_response
_build_molecule_yaml = boltz_app._build_molecule_yaml
_build_predict_cmd = boltz_app._build_predict_cmd
_find_results_dir = boltz_app._find_results_dir
_parse_molecule_outputs = boltz_app._parse_molecule_outputs
_write_batch_yamls = boltz_app._write_batch_yamls

client = TestClient(boltz_app.app)


# ---------------------------------------------------------------------------
# TestYAMLGeneration
# ---------------------------------------------------------------------------


class TestYAMLGeneration:
    """Unit tests for _build_molecule_yaml."""

    def test_basic_structure(self):
        result = _build_molecule_yaml("CCO", "MVLSPADKTN", msa_path=None)

        assert result["version"] == 1
        assert len(result["sequences"]) == 2
        assert result["sequences"][0]["protein"]["id"] == "A"
        assert result["sequences"][0]["protein"]["sequence"] == "MVLSPADKTN"
        assert result["sequences"][1]["ligand"]["id"] == "B"
        assert result["sequences"][1]["ligand"]["smiles"] == "CCO"
        assert result["properties"] == [{"affinity": {"binder": "B"}}]

    def test_msa_path_included_when_provided(self):
        result = _build_molecule_yaml("CCO", "MVLSPADKTN", msa_path="/data/msa/target.a3m")

        assert result["sequences"][0]["protein"]["msa"] == "/data/msa/target.a3m"

    def test_msa_path_absent_when_none(self):
        result = _build_molecule_yaml("CCO", "MVLSPADKTN", msa_path=None)

        assert "msa" not in result["sequences"][0]["protein"]

    def test_smiles_preserved_exactly(self):
        smi = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        result = _build_molecule_yaml(smi, "MVLSPADKTN", msa_path=None)

        assert result["sequences"][1]["ligand"]["smiles"] == smi

    def test_yaml_roundtrip(self):
        result = _build_molecule_yaml("CCO", "MVLSPADKTN", msa_path="/data/msa.a3m")
        dumped = yaml.dump(result, default_flow_style=False)
        loaded = yaml.safe_load(dumped)

        assert loaded["version"] == 1
        assert loaded["sequences"][0]["protein"]["msa"] == "/data/msa.a3m"


# ---------------------------------------------------------------------------
# TestBatchYAMLWriting
# ---------------------------------------------------------------------------


class TestBatchYAMLWriting:
    """File-system level tests for _write_batch_yamls."""

    def test_creates_one_yaml_per_smiles(self, tmp_path):
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        paths = _write_batch_yamls(tmp_path / "batch", smiles, "MVLSPADKTN", None)

        assert len(paths) == 3
        for i, p in enumerate(paths):
            assert p.name == f"mol_{i}.yaml"
            assert p.exists()

    def test_yaml_content_matches_smiles(self, tmp_path):
        smiles = ["CCO", "c1ccccc1"]
        paths = _write_batch_yamls(tmp_path / "batch", smiles, "MVLSPADKTN", "/data/msa.a3m")

        for i, p in enumerate(paths):
            data = yaml.safe_load(p.read_text())
            assert data["sequences"][1]["ligand"]["smiles"] == smiles[i]
            assert data["sequences"][0]["protein"]["msa"] == "/data/msa.a3m"

    def test_creates_batch_directory(self, tmp_path):
        batch_dir = tmp_path / "nested" / "batch"
        _write_batch_yamls(batch_dir, ["CCO"], "MVLSPADKTN", None)

        assert batch_dir.is_dir()


# ---------------------------------------------------------------------------
# TestPredictCommand
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """CLI argument construction for ``boltz predict``."""

    def _default_params(self, **overrides) -> ScoreParams:
        return ScoreParams(protein_sequence="MVLSPADKTN", **overrides)

    def test_basic_command_structure(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"), self._default_params(), False
        )

        assert cmd[0] == "boltz"
        assert cmd[1] == "predict"
        assert "/batch" in cmd
        assert "--out_dir" in cmd
        assert "--override" in cmd
        assert "--recycling_steps" in cmd

    def test_msa_server_flag(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"),
            self._default_params(use_msa_server=True),
            True,
        )
        assert "--use_msa_server" in cmd
        idx = cmd.index("--msa_server_url")
        assert cmd[idx + 1] == "https://api.colabfold.com"

    def test_no_msa_server_flag_when_disabled(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"),
            self._default_params(msa_path="/data/msa.a3m"),
            False,
        )
        assert "--use_msa_server" not in cmd

    def test_affinity_mw_correction_flag(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"),
            self._default_params(affinity_mw_correction=True, msa_path="/m.a3m"),
            False,
        )
        assert "--affinity_mw_correction" in cmd

    def test_custom_sampling_steps(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"),
            self._default_params(
                sampling_steps=100, sampling_steps_affinity=150,
                msa_path="/m.a3m",
            ),
            False,
        )
        idx = cmd.index("--sampling_steps")
        assert cmd[idx + 1] == "100"
        idx = cmd.index("--sampling_steps_affinity")
        assert cmd[idx + 1] == "150"

    def test_output_format(self):
        cmd = _build_predict_cmd(
            Path("/batch"), Path("/out"),
            self._default_params(output_format="mmcif", msa_path="/m.a3m"),
            False,
        )
        idx = cmd.index("--output_format")
        assert cmd[idx + 1] == "mmcif"


# ---------------------------------------------------------------------------
# TestOutputParsing
# ---------------------------------------------------------------------------


class TestOutputParsing:
    """Parsing of boltz predict output files."""

    def _setup_mol_dir(self, predictions_dir: Path, mol_idx: int,
                       affinity: dict | None = None,
                       confidence: dict | None = None,
                       create_structure: bool = True,
                       output_format: str = "pdb") -> None:
        mol_id = f"mol_{mol_idx}"
        mol_dir = predictions_dir / mol_id
        mol_dir.mkdir(parents=True, exist_ok=True)

        if affinity is not None:
            (mol_dir / f"affinity_{mol_id}.json").write_text(json.dumps(affinity))
        if confidence is not None:
            (mol_dir / f"confidence_{mol_id}_model_0.json").write_text(
                json.dumps(confidence)
            )
        if create_structure:
            (mol_dir / f"{mol_id}_model_0.{output_format}").write_text("ATOM ...")

    def test_parse_complete_outputs(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        self._setup_mol_dir(
            pred_dir, 0,
            affinity={
                "affinity_pred_value": -6.2,
                "affinity_probability_binary": 0.92,
                "affinity_pred_value1": -6.0,
                "affinity_pred_value2": -6.4,
                "affinity_probability_binary1": 0.90,
                "affinity_probability_binary2": 0.94,
            },
            confidence={"confidence_score": 0.85, "ligand_iptm": 0.78},
        )

        score, meta, struct_path = _parse_molecule_outputs(
            pred_dir, 0, "pred_value", "pdb"
        )

        assert score == pytest.approx(-6.2)
        assert meta["affinity_pred_value"] == pytest.approx(-6.2)
        assert meta["confidence_score"] == pytest.approx(0.85)
        assert meta["ligand_iptm"] == pytest.approx(0.78)
        assert struct_path is not None
        assert struct_path.endswith(".pdb")

    def test_parse_probability_binary_mode(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        self._setup_mol_dir(
            pred_dir, 0,
            affinity={
                "affinity_pred_value": -6.2,
                "affinity_probability_binary": 0.92,
            },
        )

        score, _, _ = _parse_molecule_outputs(
            pred_dir, 0, "probability_binary", "pdb"
        )

        assert score == pytest.approx(0.92)

    def test_missing_affinity_returns_none_score(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        mol_dir = pred_dir / "mol_0"
        mol_dir.mkdir(parents=True)
        (mol_dir / "mol_0_model_0.pdb").write_text("ATOM ...")

        score, meta, struct_path = _parse_molecule_outputs(
            pred_dir, 0, "pred_value", "pdb"
        )

        assert score is None
        assert struct_path is not None

    def test_missing_mol_dir_returns_none(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True)

        score, meta, struct_path = _parse_molecule_outputs(
            pred_dir, 5, "pred_value", "pdb"
        )

        assert score is None
        assert meta == {}
        assert struct_path is None

    def test_missing_structure_file(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        self._setup_mol_dir(
            pred_dir, 0,
            affinity={"affinity_pred_value": -6.2},
            create_structure=False,
        )

        score, meta, struct_path = _parse_molecule_outputs(
            pred_dir, 0, "pred_value", "pdb"
        )

        assert score == pytest.approx(-6.2)
        assert struct_path is None

    def test_mmcif_output_format(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        self._setup_mol_dir(
            pred_dir, 0,
            affinity={"affinity_pred_value": -5.0},
            output_format="mmcif",
        )

        _, _, struct_path = _parse_molecule_outputs(
            pred_dir, 0, "pred_value", "mmcif"
        )

        assert struct_path is not None
        assert struct_path.endswith(".mmcif")


# ---------------------------------------------------------------------------
# TestFindResultsDir
# ---------------------------------------------------------------------------


class TestFindResultsDir:

    def test_finds_expected_dir(self, tmp_path):
        batch_dir = tmp_path / "batch_abc"
        expected = tmp_path / "boltz_results_batch_abc" / "predictions"
        expected.mkdir(parents=True)

        result = _find_results_dir(tmp_path, batch_dir)
        assert result == expected

    def test_returns_none_when_missing(self, tmp_path):
        result = _find_results_dir(tmp_path, tmp_path / "batch_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# TestResponseAssembly
# ---------------------------------------------------------------------------


class TestResponseAssembly:
    """Combines per-molecule outputs into a ScoreResponse."""

    def _setup_predictions(self, pred_dir: Path, molecules: list[dict]) -> None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        for i, mol in enumerate(molecules):
            mol_id = f"mol_{i}"
            mol_dir = pred_dir / mol_id
            mol_dir.mkdir()
            if "affinity" in mol:
                (mol_dir / f"affinity_{mol_id}.json").write_text(
                    json.dumps(mol["affinity"])
                )
            if "confidence" in mol:
                (mol_dir / f"confidence_{mol_id}_model_0.json").write_text(
                    json.dumps(mol["confidence"])
                )
            if mol.get("has_structure", True):
                (mol_dir / f"{mol_id}_model_0.pdb").write_text("ATOM ...")

    def test_assembles_multi_molecule_response(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._setup_predictions(pred_dir, [
            {
                "affinity": {"affinity_pred_value": -6.2, "affinity_probability_binary": 0.92},
                "confidence": {"confidence_score": 0.85, "ligand_iptm": 0.78},
            },
            {
                "affinity": {"affinity_pred_value": -4.8, "affinity_probability_binary": 0.45},
                "confidence": {"confidence_score": 0.62, "ligand_iptm": 0.55},
            },
        ])

        resp = _assemble_response(pred_dir, 2, "pred_value", "pdb")

        assert resp.scores == [pytest.approx(-6.2), pytest.approx(-4.8)]
        assert len(resp.metadata["affinity_pred_value"]) == 2
        assert len(resp.artifacts["structures"]) == 2
        assert all(s is not None for s in resp.artifacts["structures"])

    def test_handles_partial_failure(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._setup_predictions(pred_dir, [
            {"affinity": {"affinity_pred_value": -6.2}},
        ])
        # mol_1 directory simply doesn't exist

        resp = _assemble_response(pred_dir, 2, "pred_value", "pdb")

        assert resp.scores[0] == pytest.approx(-6.2)
        assert resp.scores[1] is None
        assert resp.artifacts["structures"][1] is None

    def test_metadata_aligned_with_molecules(self, tmp_path):
        pred_dir = tmp_path / "preds"
        self._setup_predictions(pred_dir, [
            {
                "affinity": {"affinity_pred_value": -6.2},
                "confidence": {"confidence_score": 0.85},
            },
            {
                "affinity": {"affinity_pred_value": -4.8},
            },
        ])

        resp = _assemble_response(pred_dir, 2, "pred_value", "pdb")

        assert resp.metadata["affinity_pred_value"] == [pytest.approx(-6.2), pytest.approx(-4.8)]
        assert resp.metadata["confidence_score"][0] == pytest.approx(0.85)
        assert resp.metadata["confidence_score"][1] is None


# ---------------------------------------------------------------------------
# TestMSAResolution
# ---------------------------------------------------------------------------


class TestMSAResolution:
    """MSA source resolution and one-time server caching."""

    def _params(self, **overrides) -> ScoreParams:
        return ScoreParams(protein_sequence="MVLSPADKTN", **overrides)

    def test_explicit_msa_path_wins(self):
        msa_path, use_server = boltz_app._resolve_msa(
            self._params(msa_path="/shared/brd4_bd1.a3m", use_msa_server=True)
        )
        assert msa_path == "/shared/brd4_bd1.a3m"
        assert use_server is False

    def test_no_source_returns_none(self):
        msa_path, use_server = boltz_app._resolve_msa(self._params())
        assert msa_path is None
        assert use_server is False

    def test_server_path_uses_cache(self, tmp_path):
        with patch.object(boltz_app, "MSA_CACHE_DIR", tmp_path):
            with patch.object(
                boltz_app, "_ensure_cached_msa",
                return_value=str(tmp_path / "abc.a3m"),
            ) as mock_cache:
                msa_path, use_server = boltz_app._resolve_msa(
                    self._params(use_msa_server=True)
                )
        assert msa_path == str(tmp_path / "abc.a3m")
        assert use_server is False
        mock_cache.assert_called_once()

    def test_server_failure_falls_back_to_live_server(self):
        with patch.object(
            boltz_app, "_ensure_cached_msa", side_effect=RuntimeError("server down")
        ):
            msa_path, use_server = boltz_app._resolve_msa(
                self._params(use_msa_server=True)
            )
        assert msa_path is None
        assert use_server is True

    def test_cache_hit_skips_server(self, tmp_path):
        seq = "MVLSPADKTN"
        with patch.object(boltz_app, "MSA_CACHE_DIR", tmp_path):
            _, a3m_path = boltz_app._msa_cache_path(seq)
            a3m_path.write_text(">query\nMVLSPADKTN\n")
            with patch.object(boltz_app, "_call_msa_server") as mock_run:
                result = boltz_app._ensure_cached_msa(seq, "http://server")
        assert result == str(a3m_path)
        mock_run.assert_not_called()

    def test_cache_miss_queries_server_once_and_writes_a3m(self, tmp_path):
        seq = "MVLSPADKTN"
        a3m_content = ">101\nMVLSPADKTN\n>UniRef_1\nMVLSPADKTN\n"
        with patch.object(boltz_app, "MSA_CACHE_DIR", tmp_path):
            with patch.object(
                boltz_app, "_call_msa_server", return_value=a3m_content,
            ) as mock_run:
                result = boltz_app._ensure_cached_msa(seq, "http://server")
                # Second call must hit the freshly written cache, not the server.
                result2 = boltz_app._ensure_cached_msa(seq, "http://server")
        assert Path(result).read_text() == a3m_content
        assert result2 == result
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestScoreEndpoint
# ---------------------------------------------------------------------------


class TestScoreEndpoint:
    """FastAPI /score endpoint tests with mocked subprocess."""

    def test_missing_msa_config_returns_400(self):
        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "msa_path": None,
                "use_msa_server": False,
            },
        })
        assert resp.status_code == 400
        assert "msa" in resp.json()["detail"].lower()

    @patch("boltz2_app._run_predict")
    @patch("boltz2_app._write_batch_yamls")
    @patch("boltz2_app._resolve_msa", return_value=("/fake/msa.a3m", False))
    def test_subprocess_failure_returns_500(self, mock_resolve, mock_write, mock_predict):
        mock_write.return_value = []
        mock_predict.return_value = subprocess.CompletedProcess(
            args=["boltz", "predict"],
            returncode=1,
            stdout="",
            stderr="CUDA OOM",
        )

        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "use_msa_server": True,
            },
        })
        assert resp.status_code == 500

    @patch("boltz2_app._run_predict")
    @patch("boltz2_app._find_results_dir")
    @patch("boltz2_app._assemble_response")
    @patch("boltz2_app._write_batch_yamls")
    @patch("boltz2_app._resolve_msa", return_value=("/fake/msa.a3m", False))
    def test_successful_scoring(
        self, mock_resolve, mock_write, mock_assemble, mock_find, mock_predict
    ):
        mock_write.return_value = []
        mock_predict.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        mock_find.return_value = Path("/fake/predictions")
        mock_assemble.return_value = boltz_app.ScoreResponse(
            scores=[-6.2],
            metadata={"affinity_pred_value": [-6.2]},
            artifacts={"structures": ["/fake/mol_0.pdb"]},
        )

        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "use_msa_server": True,
            },
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["scores"] == [pytest.approx(-6.2)]
        assert "structures" in data["artifacts"]

    @patch("boltz2_app._run_predict")
    @patch("boltz2_app._write_batch_yamls")
    @patch("boltz2_app._resolve_msa", return_value=("/fake/msa.a3m", False))
    def test_timeout_returns_500(self, mock_resolve, mock_write, mock_predict):
        mock_write.return_value = []
        mock_predict.side_effect = subprocess.TimeoutExpired(
            cmd="boltz predict", timeout=1800,
        )

        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "use_msa_server": True,
            },
        })
        assert resp.status_code == 500
        assert "timed out" in resp.json()["detail"]["error"]

    @patch("boltz2_app._run_predict")
    @patch("boltz2_app._find_results_dir")
    @patch("boltz2_app._write_batch_yamls")
    @patch("boltz2_app._resolve_msa", return_value=("/fake/msa.a3m", False))
    def test_missing_results_dir_returns_500(
        self, mock_resolve, mock_write, mock_find, mock_predict
    ):
        mock_write.return_value = []
        mock_predict.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        mock_find.return_value = None

        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "protein_sequence": "MVLSPADKTN",
                "use_msa_server": True,
            },
        })
        assert resp.status_code == 500
