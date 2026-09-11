"""Tests for score_molecules tool -- TTP integration (Phase 2).

Tests cover:
- Routing logic (direct vs TTP mode selection)
- TTP resolution (stored vs inline, error cases)
- Response format (TTP metadata, mapper summary, external notes)
- Scorer construction from TTP-derived config
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from reinvent4_mcp.ttp.schema import TTP
from reinvent4_mcp.ttp.store import TTPStore


# ---------------------------------------------------------------------------
# Test TTP fixtures
# ---------------------------------------------------------------------------

MINIMAL_TTP = {
    "id": "test-ttp",
    "name": "Test TTP",
    "scoring": {
        "objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore"},
        ],
    },
}

NATIVE_WITH_PRESCORING_TTP = {
    "id": "native-prescoring",
    "name": "Native + Pre-scoring",
    "pre_scoring": {
        "components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.5},
        ],
    },
    "scoring": {
        "objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore", "weight": 0.5},
        ],
        "aggregation": "geometric_mean",
    },
}

MIXED_TTP = {
    "id": "mixed-ttp",
    "name": "Mixed TTP",
    "target": {
        "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    },
    "pre_scoring": {
        "components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.25},
        ],
    },
    "scoring": {
        "objectives": [
            {
                "name": "Binding Affinity",
                "type": "external",
                "service": "boltz",
                "service_url": "http://boltz:8080/score",
                "weight": 0.5,
            },
            {
                "name": "SA Score",
                "type": "native",
                "component": "SAScore",
                "weight": 0.25,
            },
        ],
    },
}

ALL_EXTERNAL_TTP = {
    "id": "all-external",
    "name": "All External",
    "target": {
        "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
    },
    "scoring": {
        "objectives": [
            {
                "name": "Affinity",
                "type": "external",
                "service": "boltz",
                "service_url": "http://boltz:8080/score",
            },
        ],
    },
}

TTP_WITH_CONSTRAINTS = {
    "id": "with-constraints",
    "name": "TTP With Constraints",
    "scoring": {
        "objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore"},
        ],
    },
    "constraints": [
        {
            "type": "custom_alerts",
            "smarts": ["[#8][#8]", "C#C"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Mock scorer infrastructure
# ---------------------------------------------------------------------------


class MockComponentResult:
    """Minimal mock of REINVENT4's TransformResults for testing."""

    def __init__(
        self,
        names: list[str],
        transformed: list[list[float]],
        raw: list[list[float]],
    ):
        self.component_names = names
        self.transformed_scores = [np.array(t) for t in transformed]
        self.transform_types = ["none"] * len(names)
        self.component_result = self._Fetchable(raw)

    class _Fetchable:
        def __init__(self, raw: list[list[float]]):
            self._raw = raw

        def fetch_scores(self, smiles: list[str], transpose: bool = True):
            return [np.array(r) for r in self._raw]


class MockScoreResults:
    """Minimal mock of REINVENT4's ScoreResults."""

    def __init__(
        self,
        total_scores: list[float],
        component_results: list[MockComponentResult],
    ):
        self.total_scores = np.array(total_scores, dtype=float)
        self.completed_components = component_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_reinvent4(monkeypatch):
    """Mock REINVENT4 so scoring tests run without it installed.

    Patches sys.modules so ``from reinvent.scoring.scorer import Scorer``
    resolves to a mock, and makes ``ensure_reinvent4_importable`` a no-op.

    Returns the mock Scorer class -- configure its return value per test.
    """
    mock_scorer_cls = MagicMock()
    mock_scorer_module = MagicMock()
    mock_scorer_module.Scorer = mock_scorer_cls

    monkeypatch.setitem(sys.modules, "reinvent", MagicMock())
    monkeypatch.setitem(sys.modules, "reinvent.scoring", MagicMock())
    monkeypatch.setitem(sys.modules, "reinvent.scoring.scorer", mock_scorer_module)
    monkeypatch.setattr(
        "reinvent4_mcp.tools.scoring.ensure_reinvent4_importable", lambda: None
    )
    return mock_scorer_cls


@pytest.fixture
def stored_ttp(monkeypatch, tmp_path):
    """Create a TTP store with MINIMAL_TTP and wire it into the tools layer."""
    from reinvent4_mcp.tools.ttp import reset_store

    reset_store()
    store = TTPStore(tmp_path / "ttps")
    ttp = TTP.model_validate(MINIMAL_TTP)
    store.create(ttp)
    monkeypatch.setattr("reinvent4_mcp.tools.ttp._get_store", lambda: store)
    return store


def _configure_mock_scorer(
    mock_scorer_cls: MagicMock,
    total_scores: list[float],
    component_results: list[MockComponentResult],
) -> None:
    """Configure the mock Scorer to return predetermined results."""
    mock_scorer_cls.return_value.compute_results.return_value = MockScoreResults(
        total_scores, component_results
    )


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestScoreMoleculesRouting:
    """Verify score_molecules selects the correct code path."""

    def test_empty_smiles_returns_empty(self):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(smiles=[])
        assert result["molecules"] == []
        assert result["summary"]["n_molecules"] == 0

    def test_no_mode_returns_error(self):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(smiles=["CCO"])
        assert "error" in result
        assert "No scoring components" in result["error"]

    def test_both_ttp_id_and_inline_returns_error(self, mock_reinvent4, stored_ttp):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(
            smiles=["CCO"], ttp_id="test-ttp", ttp=MINIMAL_TTP
        )
        assert "error" in result
        assert "not both" in result["error"]


# ---------------------------------------------------------------------------
# TTP resolution tests
# ---------------------------------------------------------------------------


class TestTTPResolution:
    """Test TTP resolution paths within score_molecules."""

    def test_nonexistent_ttp_id_returns_error(self, mock_reinvent4, stored_ttp):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(smiles=["CCO"], ttp_id="nonexistent")
        assert "error" in result
        assert "not found" in result["error"]

    def test_invalid_inline_ttp_returns_error(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(
            smiles=["CCO"], ttp={"id": "INVALID ID!", "name": "Bad"}
        )
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_all_external_ttp_returns_error(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        result = score_molecules(smiles=["CCO"], ttp=ALL_EXTERNAL_TTP)
        assert "error" in result
        assert "external" in result["error"].lower()


# ---------------------------------------------------------------------------
# TTP-based scoring: inline
# ---------------------------------------------------------------------------


class TestScoreWithInlineTTP:
    """Test score_molecules with an inline TTP definition."""

    def test_minimal_ttp(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.75, 0.60],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.75, 0.60]],
                    raw=[[2.5, 3.2]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO", "c1ccccc1"], ttp=MINIMAL_TTP)

        assert "error" not in result
        assert len(result["molecules"]) == 2
        assert result["molecules"][0]["total_score"] == pytest.approx(0.75)
        assert result["molecules"][1]["total_score"] == pytest.approx(0.60)
        assert "SA Score" in result["molecules"][0]["components"]

    def test_ttp_metadata_in_response(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.5]],
                    raw=[[3.0]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        assert result["ttp_id"] == "test-ttp"
        assert result["ttp_version"] == 1

    def test_mapper_summary_in_response(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.5]],
                    raw=[[3.0]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        summary = result["mapper_summary"]
        assert summary["n_native_objectives"] == 1
        assert summary["aggregation"] == "geometric_mean"

    def test_external_objectives_noted(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["QED", "SA Score"],
                    transformed=[[0.5], [0.5]],
                    raw=[[0.5], [3.0]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp=MIXED_TTP)

        assert "note" in result
        assert "Binding Affinity" in result["note"]
        assert "skipped" in result["note"].lower()

    def test_no_external_no_note(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.5]],
                    raw=[[3.0]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        assert "note" not in result

    def test_prescoring_components_included(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.65],
            component_results=[
                MockComponentResult(
                    names=["QED", "SA Score"],
                    transformed=[[0.8], [0.5]],
                    raw=[[0.8], [3.0]],
                ),
            ],
        )

        result = score_molecules(
            smiles=["CCO"], ttp=NATIVE_WITH_PRESCORING_TTP
        )

        assert result["mapper_summary"]["n_pre_scoring"] == 1
        assert result["mapper_summary"]["n_native_objectives"] == 1

    def test_scorer_config_passed_correctly(self, mock_reinvent4):
        """Verify the TTP-derived scorer config structure."""
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.5]],
                    raw=[[3.0]],
                ),
            ],
        )

        score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        config = mock_reinvent4.call_args[0][0]
        assert config["type"] == "geometric_mean"
        assert len(config["component"]) == 1
        assert "SAScore" in config["component"][0]

    def test_constraints_included_in_scorer_config(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score", "constraint_custom_alerts"],
                    transformed=[[0.5], [1.0]],
                    raw=[[3.0], [1.0]],
                ),
            ],
        )

        score_molecules(smiles=["CCO"], ttp=TTP_WITH_CONSTRAINTS)

        config = mock_reinvent4.call_args[0][0]
        comp_types = [list(c.keys())[0] for c in config["component"]]
        assert "SAScore" in comp_types
        assert "CustomAlerts" in comp_types


# ---------------------------------------------------------------------------
# TTP-based scoring: stored by ID
# ---------------------------------------------------------------------------


class TestScoreWithStoredTTP:
    """Test score_molecules with a stored TTP referenced by ID."""

    def test_score_with_ttp_id(self, mock_reinvent4, stored_ttp):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.75],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.75]],
                    raw=[[2.5]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp_id="test-ttp")

        assert "error" not in result
        assert len(result["molecules"]) == 1
        assert result["ttp_id"] == "test-ttp"
        assert result["ttp_version"] == 1

    def test_updated_ttp_version_reflected(self, mock_reinvent4, stored_ttp):
        from reinvent4_mcp.tools.scoring import score_molecules

        stored_ttp.update("test-ttp", {"name": "Updated Name"})

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.5]],
                    raw=[[3.0]],
                ),
            ],
        )

        result = score_molecules(smiles=["CCO"], ttp_id="test-ttp")

        assert result["ttp_version"] == 2


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Verify summary stats are correctly computed from TTP scoring."""

    def test_summary_fields(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.8, 0.6, 0.4],
            component_results=[
                MockComponentResult(
                    names=["SA Score"],
                    transformed=[[0.8, 0.6, 0.4]],
                    raw=[[2.0, 3.0, 4.0]],
                ),
            ],
        )

        result = score_molecules(
            smiles=["CCO", "c1ccccc1", "CC(=O)O"], ttp=MINIMAL_TTP
        )

        summary = result["summary"]
        assert summary["n_molecules"] == 3
        assert summary["n_valid"] == 3
        assert summary["mean_total_score"] == pytest.approx(0.6)
        assert summary["best_total_score"] == pytest.approx(0.8)
        assert "SA Score" in summary["per_component"]


# ---------------------------------------------------------------------------
# Error handling during scoring
# ---------------------------------------------------------------------------


class TestScoringErrors:
    """Test error handling when Scorer creation or computation fails."""

    def test_scorer_creation_failure(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        mock_reinvent4.side_effect = RuntimeError("Unknown component: FakeComp")

        result = score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        assert "error" in result
        assert "Failed to create scorer" in result["error"]

    def test_compute_results_failure(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        mock_reinvent4.return_value.compute_results.side_effect = RuntimeError(
            "RDKit failure"
        )

        result = score_molecules(smiles=["CCO"], ttp=MINIMAL_TTP)

        assert "error" in result
        assert "Scoring failed" in result["error"]


# ---------------------------------------------------------------------------
# Component provenance
# ---------------------------------------------------------------------------


class TestComponentSources:
    """Per-component stats should say where each score came from, since an
    objective name alone does not reveal which external service produced it."""

    def test_native_and_prescoring_sources(self, mock_reinvent4):
        from reinvent4_mcp.tools.scoring import score_molecules

        _configure_mock_scorer(
            mock_reinvent4,
            total_scores=[0.5],
            component_results=[
                MockComponentResult(
                    names=["QED", "SA Score"],
                    transformed=[[0.5], [0.5]],
                    raw=[[0.5], [3.0]],
                ),
            ],
        )

        result = score_molecules(
            smiles=["CCO"], ttp=NATIVE_WITH_PRESCORING_TTP
        )

        per_component = result["summary"]["per_component"]
        assert per_component["QED"]["source"] == {
            "type": "native",
            "component": "QED",
            "stage": "pre_scoring",
        }
        assert per_component["SA Score"]["source"] == {
            "type": "native",
            "component": "SAScore",
        }

    def test_external_source_identifies_service(self):
        from reinvent4_mcp.tools.scoring import _annotate_component_sources
        from reinvent4_mcp.ttp.mapper import describe_external_objectives

        ttp = TTP.model_validate(MIXED_TTP)
        result = {
            "summary": {
                "per_component": {
                    "Binding Affinity": {"mean_transformed": 0.4},
                    "SA Score": {"mean_transformed": 0.6},
                },
            },
        }

        _annotate_component_sources(
            result, ttp, describe_external_objectives(ttp)
        )

        affinity = result["summary"]["per_component"]["Binding Affinity"]["source"]
        assert affinity["type"] == "external"
        assert affinity["service"] == "boltz"
        assert affinity["component"] == "ExternalRESTScorer"
        assert "Boltz" in affinity["measures"]
        assert result["summary"]["per_component"]["SA Score"]["source"] == {
            "type": "native",
            "component": "SAScore",
        }

    def test_unknown_component_left_untouched(self):
        from reinvent4_mcp.tools.scoring import _annotate_component_sources

        ttp = TTP.model_validate(MINIMAL_TTP)
        result = {
            "summary": {
                "per_component": {"constraint_custom_alerts": {"mean_transformed": 1.0}},
            },
        }

        _annotate_component_sources(result, ttp, [])

        stats = result["summary"]["per_component"]["constraint_custom_alerts"]
        assert "source" not in stats
