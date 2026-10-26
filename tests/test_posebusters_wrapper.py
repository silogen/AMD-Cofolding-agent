"""Tests for the PoseBusters FastAPI wrapper service.

Covers:
- Boltz2 PDB-to-SDF ligand extraction
- DataFrame scoring: fraction-of-checks-passed computation
- Per-check metadata: boolean arrays aligned with SMILES
- Mode selection and validation
- Failure handling (empty DataFrame, exceptions, missing files)
- FastAPI endpoint validation (400s) and success path
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import importlib.util

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

_pb_spec = importlib.util.spec_from_file_location(
    "posebusters_app",
    str(Path(__file__).resolve().parent.parent / "services" / "posebusters" / "app.py"),
)
pb_app = importlib.util.module_from_spec(_pb_spec)
sys.modules["posebusters_app"] = pb_app
_pb_spec.loader.exec_module(pb_app)

MODES_REQUIRING_PROTEIN = pb_app.MODES_REQUIRING_PROTEIN
MODES_REQUIRING_REFERENCE = pb_app.MODES_REQUIRING_REFERENCE
VALID_MODES = pb_app.VALID_MODES
_boltz_pdb_to_sdf = pb_app._boltz_pdb_to_sdf
_score_single_molecule = pb_app._score_single_molecule
score_molecules = pb_app.score_molecules

client = TestClient(pb_app.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bust_df(checks: dict[str, bool]) -> pd.DataFrame:
    """Create a single-row DataFrame mimicking PoseBusters output."""
    return pd.DataFrame([checks])


def _make_mock_buster(bust_return: pd.DataFrame) -> MagicMock:
    buster = MagicMock()
    buster.bust.return_value = bust_return
    return buster


# ---------------------------------------------------------------------------
# TestBoltzPdbToSdf
# ---------------------------------------------------------------------------

SAMPLE_COMPLEX_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
HETATM    3  C1  LIG B   1       5.000   6.000   7.000  1.00  0.00           C
HETATM    4  O1  LIG B   1       5.500   6.500   7.500  1.00  0.00           O
CONECT    3    4
CONECT    4    3
END
"""

SAMPLE_PDB_NO_LIGAND = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
END
"""

SAMPLE_PDB_NO_CONECT = """\
HETATM    1  C1  LIG B   1       5.000   6.000   7.000  1.00  0.00           C
HETATM    2  O1  LIG B   1       5.500   6.500   7.500  1.00  0.00           O
END
"""


@pytest.mark.skipif(not _HAS_RDKIT, reason="RDKit is not installed")
class TestBoltzPdbToSdf:
    """Unit tests for _boltz_pdb_to_sdf ligand extraction.

    These exercise the real RDKit-backed PDB-to-SDF conversion and are skipped
    in environments where RDKit is not installed.
    """

    def test_extracts_ligand_to_sdf(self, tmp_path):
        pdb_file = tmp_path / "complex.pdb"
        pdb_file.write_text(SAMPLE_COMPLEX_PDB)

        sdf_path = _boltz_pdb_to_sdf(pdb_file)

        assert sdf_path.exists()
        assert sdf_path.suffix == ".sdf"
        assert sdf_path.name == "complex_lig.sdf"
        content = sdf_path.read_text()
        assert "$$$$" in content  # SDF terminator

    def test_raises_on_no_matching_hetatm(self, tmp_path):
        pdb_file = tmp_path / "protein_only.pdb"
        pdb_file.write_text(SAMPLE_PDB_NO_LIGAND)

        with pytest.raises(ValueError, match="No HETATM records"):
            _boltz_pdb_to_sdf(pdb_file)

    def test_works_without_conect_records(self, tmp_path):
        """RDKit infers bonds from 3D coordinates when CONECT records are absent."""
        pdb_file = tmp_path / "no_conect.pdb"
        pdb_file.write_text(SAMPLE_PDB_NO_CONECT)

        sdf_path = _boltz_pdb_to_sdf(pdb_file)

        assert sdf_path.exists()
        assert sdf_path.read_text().strip().endswith("$$$$")

    def test_custom_resname(self, tmp_path):
        custom_pdb = """\
HETATM    1  C1  ATP B   1       5.000   6.000   7.000  1.00  0.00           C
END
"""
        pdb_file = tmp_path / "custom.pdb"
        pdb_file.write_text(custom_pdb)

        with pytest.raises(ValueError, match="No HETATM records"):
            _boltz_pdb_to_sdf(pdb_file, resname="LIG")

        sdf_path = _boltz_pdb_to_sdf(pdb_file, resname="ATP")
        assert sdf_path.exists()


# ---------------------------------------------------------------------------
# TestScoreSingleMolecule
# ---------------------------------------------------------------------------


class TestScoreSingleMolecule:
    """Unit tests for _score_single_molecule with mock PoseBusters."""

    def test_all_checks_pass(self):
        df = _make_bust_df({
            "bond_lengths": True,
            "bond_angles": True,
            "internal_steric_clash": True,
            "sanitization": True,
        })
        buster = _make_mock_buster(df)

        score, checks = _score_single_molecule(
            buster, Path("/fake/mol.pdb"), None, None,
        )

        assert score == pytest.approx(1.0)
        assert checks["bond_lengths"] is True
        assert checks["n_checks_passed"] == 4
        assert checks["n_checks_total"] == 4

    def test_partial_checks_pass(self):
        df = _make_bust_df({
            "bond_lengths": True,
            "bond_angles": True,
            "internal_steric_clash": False,
            "sanitization": True,
        })
        buster = _make_mock_buster(df)

        score, checks = _score_single_molecule(
            buster, Path("/fake/mol.pdb"), None, None,
        )

        assert score == pytest.approx(3 / 4)
        assert checks["internal_steric_clash"] is False
        assert checks["n_checks_passed"] == 3
        assert checks["n_checks_total"] == 4

    def test_empty_dataframe_returns_none(self):
        buster = _make_mock_buster(pd.DataFrame())

        score, checks = _score_single_molecule(
            buster, Path("/fake/mol.pdb"), None, None,
        )

        assert score is None
        assert "pose_status" in checks  # now always populated

    def test_exception_returns_none(self):
        buster = MagicMock()
        buster.bust.side_effect = RuntimeError("RDKit parse failure")

        score, checks = _score_single_molecule(
            buster, Path("/fake/corrupt.pdb"), None, None,
        )

        assert score is None
        assert "pose_status" in checks  # service_error or declined

    def test_passes_mol_cond_and_mol_true(self):
        df = _make_bust_df({"bond_lengths": True})
        buster = _make_mock_buster(df)

        _score_single_molecule(
            buster,
            Path("/fake/mol.pdb"),
            mol_cond=Path("/fake/protein.pdb"),
            mol_true=Path("/fake/ref.pdb"),
        )

        call_kwargs = buster.bust.call_args.kwargs
        assert call_kwargs["mol_cond"] == Path("/fake/protein.pdb")
        assert call_kwargs["mol_true"] == [Path("/fake/ref.pdb")]

    def test_no_bool_columns_returns_none(self):
        """DataFrame with only non-bool columns → no checks to evaluate."""
        df = pd.DataFrame([{"text_col": "value", "num_col": 42}])
        buster = _make_mock_buster(df)

        score, checks = _score_single_molecule(
            buster, Path("/fake/mol.pdb"), None, None,
        )

        assert score is None
        assert checks["n_checks_passed"] == 0
        assert checks["n_checks_total"] == 0


# ---------------------------------------------------------------------------
# TestScoreMolecules
# ---------------------------------------------------------------------------


class TestScoreMolecules:
    """Multi-molecule scoring with metadata pivoting."""

    @pytest.fixture(autouse=True)
    def _mock_pb_lib(self):
        """Inject a mock ``posebusters`` package into sys.modules."""
        self.mock_pb_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.PoseBusters = self.mock_pb_cls
        with patch.dict(sys.modules, {"posebusters": mock_module}):
            yield

    @pytest.fixture(autouse=True)
    def _mock_extraction(self):
        """Bypass ligand extraction — return the input path unchanged."""
        with patch(
            "posebusters_app._boltz_pdb_to_sdf", side_effect=lambda p, **kw: p
        ) as mock:
            self.mock_extract = mock
            yield

    def test_scores_multiple_molecules(self, tmp_path):
        df1 = _make_bust_df({"bond_lengths": True, "sanitization": True})
        df2 = _make_bust_df({"bond_lengths": True, "sanitization": False})
        protein = tmp_path / "protein.pdb"
        mol_0 = tmp_path / "mol_0.pdb"
        mol_1 = tmp_path / "mol_1.pdb"
        for path in (protein, mol_0, mol_1):
            path.touch()

        mock_buster = MagicMock()
        mock_buster.bust.side_effect = [df1, df2]
        self.mock_pb_cls.return_value = mock_buster

        scores, metadata = score_molecules(
            smiles=["CCO", "c1ccccc1"],
            params={
                "mode": "dock",
                "protein_pdb_path": str(protein),
                "structure_paths": [str(mol_0), str(mol_1)],
            },
        )

        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.5)

    def test_metadata_aligned_with_smiles(self, tmp_path):
        df1 = _make_bust_df({"bond_lengths": True, "clash": False})
        df2 = _make_bust_df({"bond_lengths": False, "clash": True})
        mol_a = tmp_path / "a.pdb"
        mol_b = tmp_path / "b.pdb"
        mol_a.touch()
        mol_b.touch()

        mock_buster = MagicMock()
        mock_buster.bust.side_effect = [df1, df2]
        self.mock_pb_cls.return_value = mock_buster

        _, metadata = score_molecules(
            smiles=["A", "B"],
            params={
                "mode": "mol",
                "structure_paths": [str(mol_a), str(mol_b)],
            },
        )

        assert metadata["bond_lengths"] == [True, False]
        assert metadata["clash"] == [False, True]
        assert metadata["n_checks_passed"] == [1, 1]
        assert metadata["n_checks_total"] == [2, 2]

    def test_partial_failure(self, tmp_path):
        df_ok = _make_bust_df({"bond_lengths": True})
        mol_a = tmp_path / "a.pdb"
        mol_b = tmp_path / "b.pdb"
        mol_a.touch()
        mol_b.touch()

        mock_buster = MagicMock()
        mock_buster.bust.side_effect = [df_ok, pd.DataFrame()]
        self.mock_pb_cls.return_value = mock_buster

        scores, metadata = score_molecules(
            smiles=["A", "B"],
            params={
                "mode": "mol",
                "structure_paths": [str(mol_a), str(mol_b)],
            },
        )

        assert scores[0] == pytest.approx(1.0)
        assert scores[1] is None

    def test_reference_paths_forwarded(self, tmp_path):
        df = _make_bust_df({"bond_lengths": True})
        protein = tmp_path / "protein.pdb"
        molecule = tmp_path / "mol.pdb"
        reference = tmp_path / "ref.pdb"
        for path in (protein, molecule, reference):
            path.touch()
        mock_buster = MagicMock()
        mock_buster.bust.return_value = df
        self.mock_pb_cls.return_value = mock_buster

        score_molecules(
            smiles=["CCO"],
            params={
                "mode": "redock",
                "protein_pdb_path": str(protein),
                "structure_paths": [str(molecule)],
                "reference_paths": [str(reference)],
            },
        )

        call_kwargs = mock_buster.bust.call_args.kwargs
        assert call_kwargs["mol_true"] == [reference]

    def test_extraction_failure_skips_molecule(self):
        """When _boltz_pdb_to_sdf raises, the molecule is skipped with None score."""
        self.mock_extract.side_effect = ValueError("No HETATM records")

        df = _make_bust_df({"bond_lengths": True})
        mock_buster = MagicMock()
        mock_buster.bust.return_value = df
        self.mock_pb_cls.return_value = mock_buster

        scores, metadata = score_molecules(
            smiles=["CCO"],
            params={
                "mode": "mol",
                "structure_paths": ["/mol.pdb"],
            },
        )

        assert scores == [None]
        mock_buster.bust.assert_not_called()


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestScoreEndpointValidation
# ---------------------------------------------------------------------------


class TestScoreEndpointValidation:
    """Request validation returning 400 errors."""

    def test_invalid_mode(self):
        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "mode": "invalid_mode",
                "structure_paths": ["/mol.pdb"],
            },
        })
        assert resp.status_code == 400
        assert "Invalid mode" in resp.json()["detail"]

    def test_dock_mode_requires_protein_pdb(self):
        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "mode": "dock",
                "structure_paths": ["/mol.pdb"],
                "protein_pdb_path": None,
            },
        })
        assert resp.status_code == 400
        assert "protein_pdb_path" in resp.json()["detail"]

    def test_redock_mode_requires_reference_paths(self):
        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "mode": "redock",
                "protein_pdb_path": "/protein.pdb",
                "structure_paths": ["/mol.pdb"],
            },
        })
        assert resp.status_code == 400
        assert "reference_paths" in resp.json()["detail"]

    def test_smiles_structure_paths_length_mismatch(self):
        resp = client.post("/score", json={
            "smiles": ["CCO", "c1ccccc1"],
            "params": {
                "mode": "mol",
                "structure_paths": ["/mol.pdb"],
            },
        })
        assert resp.status_code == 400
        assert "Length mismatch" in resp.json()["detail"]

    def test_reference_paths_length_mismatch(self):
        resp = client.post("/score", json={
            "smiles": ["CCO", "c1ccccc1"],
            "params": {
                "mode": "redock",
                "protein_pdb_path": "/protein.pdb",
                "structure_paths": ["/m0.pdb", "/m1.pdb"],
                "reference_paths": ["/r0.pdb"],
            },
        })
        assert resp.status_code == 400
        assert "Length mismatch" in resp.json()["detail"]

    def test_mol_mode_does_not_require_protein(self):
        """mol mode skips protein requirement -- should not 400 on missing protein."""
        with patch("posebusters_app.score_molecules", return_value=([], {})):
            resp = client.post("/score", json={
                "smiles": [],
                "params": {
                    "mode": "mol",
                    "structure_paths": [],
                },
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestScoreEndpointSuccess
# ---------------------------------------------------------------------------


class TestScoreEndpointSuccess:
    """Successful /score requests with mocked PoseBusters."""

    @patch("posebusters_app.score_molecules")
    def test_returns_scores_and_metadata(self, mock_score):
        mock_score.return_value = (
            [0.85, 0.60],
            {
                "bond_lengths": [True, True],
                "sanitization": [True, False],
                "n_checks_passed": [2, 1],
                "n_checks_total": [2, 2],
            },
        )

        resp = client.post("/score", json={
            "smiles": ["CCO", "c1ccccc1"],
            "params": {
                "mode": "dock",
                "protein_pdb_path": "/protein.pdb",
                "structure_paths": ["/m0.pdb", "/m1.pdb"],
            },
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["scores"] == [pytest.approx(0.85), pytest.approx(0.60)]
        assert data["metadata"]["bond_lengths"] == [True, True]
        assert data["metadata"]["n_checks_passed"] == [2, 1]

    @patch("posebusters_app.score_molecules")
    def test_internal_error_returns_500(self, mock_score):
        mock_score.side_effect = RuntimeError("Unexpected crash")

        resp = client.post("/score", json={
            "smiles": ["CCO"],
            "params": {
                "mode": "dock",
                "protein_pdb_path": "/protein.pdb",
                "structure_paths": ["/mol.pdb"],
            },
        })

        assert resp.status_code == 500

    @patch("posebusters_app.score_molecules")
    def test_null_scores_for_failed_molecules(self, mock_score):
        mock_score.return_value = (
            [0.85, None],
            {
                "bond_lengths": [True, None],
                "n_checks_passed": [1, None],
                "n_checks_total": [1, None],
            },
        )

        resp = client.post("/score", json={
            "smiles": ["CCO", "BAD"],
            "params": {
                "mode": "dock",
                "protein_pdb_path": "/protein.pdb",
                "structure_paths": ["/m0.pdb", "/m1.pdb"],
            },
        })

        data = resp.json()
        assert data["scores"][0] == pytest.approx(0.85)
        assert data["scores"][1] is None


# ---------------------------------------------------------------------------
# TestModeConstants
# ---------------------------------------------------------------------------


class TestModeConstants:
    """Verify mode constant sets are consistent."""

    def test_all_protein_modes_are_valid(self):
        assert MODES_REQUIRING_PROTEIN.issubset(VALID_MODES)

    def test_all_reference_modes_are_valid(self):
        assert MODES_REQUIRING_REFERENCE.issubset(VALID_MODES)

    def test_reference_modes_subset_of_protein_modes(self):
        assert MODES_REQUIRING_REFERENCE.issubset(MODES_REQUIRING_PROTEIN)
