# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ExternalRESTScorer, pipeline artifact injection, and NaN propagation.

Covers:
- ExternalRESTScorer: HTTP POST construction, null→NaN, timeout/error handling,
  last_artifacts storage
- list_external_services: health check success/failure/unreachable
- _ArtifactInjectingWrapper: upstream artifact injection, missing artifacts
- PipelineAwareScorer: wiring, dependency detection, compute_results delegation
- NaN propagation through scoring pipeline
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# The REINVENT4 plugin namespace + leaf stubs are set up in conftest.py so this
# imports the SAME ExternalRESTScorer that runs in production (the module under
# reinvent4_mcp/_plugins/reinvent_plugins/components/), not a divergent copy.
from reinvent_plugins.components.comp_external_rest import (
    ExternalRESTScorer,
    Parameters,
)
from reinvent4_mcp.scoring_pipeline import _ArtifactInjectingWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(**kwargs) -> Parameters:
    defaults: dict[str, Any] = {
        "service_url": ["http://boltz:8080/score"],
        "service": ["boltz"],
    }
    defaults.update(kwargs)
    return Parameters(**defaults)


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# TestExternalRESTScorer
# ---------------------------------------------------------------------------


class TestExternalRESTScorer:
    """HTTP call construction, null→NaN conversion, failure semantics."""

    def test_init_extracts_url_and_service(self):
        scorer = ExternalRESTScorer(_make_params())
        assert scorer.service_url == "http://boltz:8080/score"
        assert scorer.service == "boltz"
        assert scorer.timeout == 1800

    def test_init_builds_request_params(self):
        scorer = ExternalRESTScorer(
            _make_params(
                protein_sequence=["MVLSPADKTN"],
                affinity_mode=["pred_value"],
            )
        )
        assert scorer.request_params["protein_sequence"] == "MVLSPADKTN"
        assert scorer.request_params["affinity_mode"] == "pred_value"

    def test_init_skips_none_values(self):
        scorer = ExternalRESTScorer(
            _make_params(msa_path=[None], protein_pdb_path=[None])
        )
        assert "msa_path" not in scorer.request_params
        assert "protein_pdb_path" not in scorer.request_params

    def test_last_artifacts_initially_none(self):
        scorer = ExternalRESTScorer(_make_params())
        assert scorer.last_artifacts is None

    def test_init_logs_identity(self, caplog):
        """REINVENT4 only logs the class name, so several external objectives
        produce identical lines unless the component identifies itself."""
        with caplog.at_level("INFO", logger="reinvent"):
            ExternalRESTScorer(
                _make_params(
                    service=["admet"],
                    service_url=["http://admet-predictor:8084/score"],
                    component_name=["Permeability"],
                    endpoint=["caco2_permeability"],
                )
            )

        (line,) = [r.getMessage() for r in caplog.records]
        assert "ExternalRESTScorer[Permeability]" in line
        assert "service=admet" in line
        assert "endpoint=caco2_permeability" in line

    def test_init_log_reports_pipeline_dependency(self, caplog):
        with caplog.at_level("INFO", logger="reinvent"):
            ExternalRESTScorer(
                _make_params(
                    service=["posebusters"],
                    component_name=["Pose Quality"],
                    depends_on=["Binding Affinity"],
                )
            )

        (line,) = [r.getMessage() for r in caplog.records]
        assert "depends_on=Binding Affinity" in line

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_successful_scoring(self, mock_post):
        mock_post.return_value = _mock_response({
            "scores": [-6.2, -4.8],
            "metadata": {"confidence_score": [0.85, 0.62]},
            "artifacts": {"structures": ["/p/m0.pdb", "/p/m1.pdb"]},
        })

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO", "c1ccccc1"])

        np.testing.assert_array_almost_equal(result.scores[0], [-6.2, -4.8])
        # The structures also travel as metadata, so each stays with its molecule.
        assert result.metadata == {
            "confidence_score": [0.85, 0.62],
            "structure_path": ["/p/m0.pdb", "/p/m1.pdb"],
        }
        assert scorer.last_artifacts == {"structures": ["/p/m0.pdb", "/p/m1.pdb"]}

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_null_to_nan_conversion(self, mock_post):
        mock_post.return_value = _mock_response({
            "scores": [-6.2, None, -4.8],
        })

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO", "BAD", "c1ccccc1"])

        assert result.scores[0][0] == pytest.approx(-6.2)
        assert np.isnan(result.scores[0][1])
        assert result.scores[0][2] == pytest.approx(-4.8)

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_all_nan_on_connection_error(self, mock_post):
        mock_post.side_effect = ConnectionError("Service unreachable")

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO", "c1ccccc1"])

        assert len(result.scores[0]) == 2
        assert all(np.isnan(s) for s in result.scores[0])

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_all_nan_on_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.Timeout("Timeout")

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO"])

        assert len(result.scores[0]) == 1
        assert np.isnan(result.scores[0][0])

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_all_nan_on_http_error(self, mock_post):
        # A non-ok response triggers raise_for_status(), which the scorer
        # catches and converts to all-NaN.
        mock_post.return_value.ok = False
        mock_post.return_value.raise_for_status.side_effect = Exception("500")

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO", "c1ccccc1"])

        assert len(result.scores[0]) == 2
        assert all(np.isnan(s) for s in result.scores[0])

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_http_call_payload(self, mock_post):
        mock_post.return_value = _mock_response({"scores": [0.5]})

        scorer = ExternalRESTScorer(
            _make_params(protein_sequence=["MVLSPADKTN"])
        )
        scorer(["CCO"])

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["smiles"] == ["CCO"]
        assert "protein_sequence" in payload["params"]
        assert call_kwargs.kwargs["timeout"] == 1800

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_last_artifacts_none_when_response_has_no_artifacts(self, mock_post):
        mock_post.return_value = _mock_response({"scores": [0.5]})

        scorer = ExternalRESTScorer(_make_params())
        scorer(["CCO"])

        assert scorer.last_artifacts is None

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_metadata_none_when_absent(self, mock_post):
        mock_post.return_value = _mock_response({"scores": [0.5]})

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO"])

        assert result.metadata is None

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_depends_on_injects_index_aligned_structures(self, mock_post):
        """Injected structure_paths stay index-aligned with the SMILES batch.

        When an upstream producer (e.g. Boltz) fails for some molecules, its
        ``structures`` list carries ``None`` holes. The scorer must forward the
        FULL list (holes preserved), not a compacted one, because downstream
        services pair ``structure_paths[i]`` with ``smiles[i]`` (PoseBusters even
        enforces equal lengths). A compacted list would misassign poses / 400.
        """
        mock_post.return_value = _mock_response({"scores": [1.0, None, 2.0]})

        # mol 0 and mol 2 got a structure; mol 1's upstream cofold failed (None).
        upstream = ["/shared/m0.pdb", None, "/shared/m2.pdb"]
        ExternalRESTScorer._artifact_registry["boltz"] = {"structures": upstream}
        try:
            scorer = ExternalRESTScorer(
                _make_params(service=["interaction_profiler"], depends_on=["boltz"])
            )
            scorer(["CCO", "BAD", "c1ccccc1"])
        finally:
            ExternalRESTScorer._artifact_registry.pop("boltz", None)

        params = mock_post.call_args.kwargs["json"]["params"]
        assert params["structure_paths"] == ["/shared/m0.pdb", None, "/shared/m2.pdb"]
        # protein_pdb_path falls back to the first *valid* structure.
        assert params["protein_pdb_path"] == "/shared/m0.pdb"


# ---------------------------------------------------------------------------
# TestListExternalServices
# ---------------------------------------------------------------------------


class TestListExternalServices:
    """Health-check logic for configured external services."""

    @patch("reinvent4_mcp.tools.external.requests.get")
    @patch("reinvent4_mcp.tools.external.get_config")
    def test_healthy_service(self, mock_config, mock_get):
        mock_config.return_value.enabled_external_services = {
            "boltz": "http://boltz:8080/score",
        }
        mock_get.return_value = _mock_response({"status": "ok"})

        from reinvent4_mcp.tools.external import list_external_services
        result = list_external_services()

        assert result["boltz"]["status"] == "ok"
        assert result["boltz"]["url"] == "http://boltz:8080/score"
        assert result["boltz"]["health_url"] == "http://boltz:8080/health"

    @patch("reinvent4_mcp.tools.external.requests.get")
    @patch("reinvent4_mcp.tools.external.get_config")
    def test_unhealthy_service(self, mock_config, mock_get):
        mock_config.return_value.enabled_external_services = {
            "boltz": "http://boltz:8080/score",
        }
        resp = MagicMock()
        resp.ok = False
        mock_get.return_value = resp

        from reinvent4_mcp.tools.external import list_external_services
        result = list_external_services()

        assert result["boltz"]["status"] == "unhealthy"

    @patch("reinvent4_mcp.tools.external.requests.get")
    @patch("reinvent4_mcp.tools.external.get_config")
    def test_unreachable_service(self, mock_config, mock_get):
        mock_config.return_value.enabled_external_services = {
            "boltz": "http://boltz:8080/score",
        }
        mock_get.side_effect = ConnectionError("unreachable")

        from reinvent4_mcp.tools.external import list_external_services
        result = list_external_services()

        assert result["boltz"]["status"] == "unreachable"

    @patch("reinvent4_mcp.tools.external.requests.get")
    @patch("reinvent4_mcp.tools.external.get_config")
    def test_multiple_services(self, mock_config, mock_get):
        mock_config.return_value.enabled_external_services = {
            "boltz": "http://boltz:8080/score",
            "posebusters": "http://pb:8080/score",
        }

        def _side_effect(url, **kwargs):
            if "boltz" in url:
                return _mock_response({"status": "ok"})
            raise ConnectionError("unreachable")

        mock_get.side_effect = _side_effect

        from reinvent4_mcp.tools.external import list_external_services
        result = list_external_services()

        assert result["boltz"]["status"] == "ok"
        assert result["posebusters"]["status"] == "unreachable"

    @patch("reinvent4_mcp.tools.external.requests.get")
    @patch("reinvent4_mcp.tools.external.get_config")
    def test_no_services_configured(self, mock_config, mock_get):
        mock_config.return_value.enabled_external_services = {}

        from reinvent4_mcp.tools.external import list_external_services
        result = list_external_services()

        assert result == {}
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# TestArtifactInjectingWrapper
# ---------------------------------------------------------------------------


class TestArtifactInjectingWrapper:
    """Transparent wrapper that injects upstream artifacts into dependents."""

    def test_injects_structure_paths(self):
        wrapped = MagicMock()
        wrapped.request_params = {}
        wrapped.service = "posebusters"

        upstream = MagicMock()
        upstream.last_artifacts = {
            "structures": ["/shared/mol_0.pdb", "/shared/mol_1.pdb"],
        }

        ttp = MagicMock()
        ttp.target.protein_pdb_path = "/shared/protein.pdb"

        wrapper = _ArtifactInjectingWrapper(wrapped, upstream, ttp)
        wrapper(["CCO", "c1ccccc1"])

        assert wrapped.request_params["structure_paths"] == [
            "/shared/mol_0.pdb",
            "/shared/mol_1.pdb",
        ]
        assert wrapped.request_params["protein_pdb_path"] == "/shared/protein.pdb"
        wrapped.assert_called_once_with(["CCO", "c1ccccc1"])

    def test_injects_protein_pdb_from_ttp_target(self):
        wrapped = MagicMock()
        wrapped.request_params = {}

        upstream = MagicMock()
        upstream.last_artifacts = {"structures": ["/shared/mol_0.pdb"]}

        ttp = MagicMock()
        ttp.target.protein_pdb_path = "/data/protein.pdb"

        wrapper = _ArtifactInjectingWrapper(wrapped, upstream, ttp)
        wrapper(["CCO"])

        assert wrapped.request_params["protein_pdb_path"] == "/data/protein.pdb"

    def test_delegates_when_no_artifacts(self):
        wrapped = MagicMock()
        wrapped.request_params = {}
        wrapped.service = "posebusters"

        upstream = MagicMock()
        upstream.last_artifacts = None

        ttp = MagicMock()

        wrapper = _ArtifactInjectingWrapper(wrapped, upstream, ttp)
        wrapper(["CCO"])

        wrapped.assert_called_once_with(["CCO"])
        assert "structure_paths" not in wrapped.request_params

    def test_delegates_when_artifacts_have_no_structures(self):
        wrapped = MagicMock()
        wrapped.request_params = {}

        upstream = MagicMock()
        upstream.last_artifacts = {"other_key": "value"}

        ttp = MagicMock()

        wrapper = _ArtifactInjectingWrapper(wrapped, upstream, ttp)
        wrapper(["CCO"])

        wrapped.assert_called_once_with(["CCO"])
        assert "structure_paths" not in wrapped.request_params

    def test_getattr_forwards_to_wrapped(self):
        wrapped = MagicMock()
        wrapped.service = "posebusters"
        wrapped.timeout = 600

        wrapper = _ArtifactInjectingWrapper(
            wrapped, MagicMock(), MagicMock()
        )

        assert wrapper.service == "posebusters"
        assert wrapper.timeout == 600


# ---------------------------------------------------------------------------
# TestPipelineAwareScorer
# ---------------------------------------------------------------------------


class TestPipelineAwareScorer:
    """PipelineAwareScorer wiring and delegation."""

    def _make_component_data(self, names, instance):
        """Build a mock ComponentData matching REINVENT4's structure."""
        cd = MagicMock()
        cd.params = (names, instance, MagicMock(), MagicMock())
        return cd

    @patch("reinvent4_mcp.utils.ensure_reinvent4_importable")
    def test_no_dependencies_skips_wiring(self, mock_ensure, monkeypatch):
        mock_scorer_cls = MagicMock()
        monkeypatch.setitem(sys.modules, "reinvent", MagicMock())
        monkeypatch.setitem(sys.modules, "reinvent.scoring", MagicMock())
        monkeypatch.setitem(
            sys.modules, "reinvent.scoring.scorer",
            MagicMock(Scorer=mock_scorer_cls),
        )

        ttp = MagicMock()
        ttp.scoring.objectives = [
            MagicMock(depends_on=None),
        ]

        from reinvent4_mcp.scoring_pipeline import PipelineAwareScorer
        scorer = PipelineAwareScorer({"type": "geometric_mean", "component": []}, ttp)

        assert scorer._scorer is mock_scorer_cls.return_value

    @patch("reinvent4_mcp.utils.ensure_reinvent4_importable")
    def test_wires_dependent_components(self, mock_ensure, monkeypatch):
        mock_scorer_cls = MagicMock()
        monkeypatch.setitem(sys.modules, "reinvent", MagicMock())
        monkeypatch.setitem(sys.modules, "reinvent.scoring", MagicMock())
        monkeypatch.setitem(
            sys.modules, "reinvent.scoring.scorer",
            MagicMock(Scorer=mock_scorer_cls),
        )

        boltz_instance = MagicMock()
        boltz_instance.request_params = {"service": "boltz"}
        boltz_instance.depends_on = None

        pb_instance = MagicMock()
        pb_instance.request_params = {"service": "posebusters", "depends_on": "Binding Affinity"}
        pb_instance.depends_on = "Binding Affinity"

        boltz_cd = self._make_component_data(["Binding Affinity"], boltz_instance)
        pb_cd = self._make_component_data(["Pose Quality"], pb_instance)

        mock_scorer_cls.return_value.components.scorers = [boltz_cd, pb_cd]

        ttp = MagicMock()
        ttp.scoring.objectives = [
            MagicMock(depends_on=None),
            MagicMock(depends_on="Binding Affinity"),
        ]

        from reinvent4_mcp.scoring_pipeline import PipelineAwareScorer
        PipelineAwareScorer({"type": "geometric_mean", "component": []}, ttp)

        _, new_instance, _, _ = pb_cd.params
        assert isinstance(new_instance, _ArtifactInjectingWrapper)

    @patch("reinvent4_mcp.utils.ensure_reinvent4_importable")
    def test_compute_results_delegates(self, mock_ensure, monkeypatch):
        mock_scorer_cls = MagicMock()
        mock_scorer_cls.return_value.components.scorers = []
        monkeypatch.setitem(sys.modules, "reinvent", MagicMock())
        monkeypatch.setitem(sys.modules, "reinvent.scoring", MagicMock())
        monkeypatch.setitem(
            sys.modules, "reinvent.scoring.scorer",
            MagicMock(Scorer=mock_scorer_cls),
        )

        ttp = MagicMock()
        ttp.scoring.objectives = []

        from reinvent4_mcp.scoring_pipeline import PipelineAwareScorer
        scorer = PipelineAwareScorer({"type": "geometric_mean", "component": []}, ttp)

        smiles = ["CCO"]
        invalid_mask = np.ones(1, dtype=bool)
        dup_mask = np.ones(1, dtype=bool)

        scorer.compute_results(smiles, invalid_mask, dup_mask)
        mock_scorer_cls.return_value.compute_results.assert_called_once_with(
            smiles, invalid_mask, dup_mask,
        )

    @patch("reinvent4_mcp.utils.ensure_reinvent4_importable")
    def test_missing_upstream_logs_warning(self, mock_ensure, monkeypatch):
        mock_scorer_cls = MagicMock()
        monkeypatch.setitem(sys.modules, "reinvent", MagicMock())
        monkeypatch.setitem(sys.modules, "reinvent.scoring", MagicMock())
        monkeypatch.setitem(
            sys.modules, "reinvent.scoring.scorer",
            MagicMock(Scorer=mock_scorer_cls),
        )

        orphan = MagicMock()
        orphan.request_params = {"depends_on": "NonExistent"}
        orphan.depends_on = "NonExistent"

        cd = self._make_component_data(["Orphan"], orphan)
        mock_scorer_cls.return_value.components.scorers = [cd]

        ttp = MagicMock()
        ttp.scoring.objectives = [MagicMock(depends_on="NonExistent")]

        from reinvent4_mcp.scoring_pipeline import PipelineAwareScorer

        with patch("reinvent4_mcp.scoring_pipeline.logger") as mock_logger:
            PipelineAwareScorer(
                {"type": "geometric_mean", "component": []}, ttp,
            )
            mock_logger.warning.assert_called()
            assert "NonExistent" in mock_logger.warning.call_args[0][1]


# ---------------------------------------------------------------------------
# TestNaNPropagation
# ---------------------------------------------------------------------------


class TestNaNPropagation:
    """NaN from external services propagates without crashing the scorer."""

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_service_failure_produces_all_nan(self, mock_post):
        mock_post.side_effect = ConnectionError("down")

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["CCO", "c1ccccc1", "CC(=O)O"])

        assert len(result.scores[0]) == 3
        assert all(np.isnan(s) for s in result.scores[0])

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_partial_null_produces_partial_nan(self, mock_post):
        mock_post.return_value = _mock_response({
            "scores": [0.5, None, 0.7, None],
        })

        scorer = ExternalRESTScorer(_make_params())
        result = scorer(["A", "B", "C", "D"])

        scores = result.scores[0]
        assert scores[0] == pytest.approx(0.5)
        assert np.isnan(scores[1])
        assert scores[2] == pytest.approx(0.7)
        assert np.isnan(scores[3])

    def test_format_results_handles_nan_total_scores(self):
        from reinvent4_mcp.tools.scoring import _format_results

        comp_result = MagicMock()
        comp_result.component_names = ["Affinity"]
        comp_result.transformed_scores = [np.array([0.8, np.nan, 0.6])]
        comp_result.transform_types = ["none"]

        raw_mock = MagicMock()
        raw_mock.fetch_scores.return_value = [np.array([-6.0, np.nan, -4.0])]
        comp_result.component_result = raw_mock

        score_results = MagicMock()
        score_results.total_scores = np.array([0.8, np.nan, 0.6])
        score_results.completed_components = [comp_result]

        result = _format_results(score_results, ["A", "B", "C"])

        assert result["summary"]["n_molecules"] == 3
        assert result["summary"]["n_valid"] == 2
        assert result["molecules"][0]["total_score"] == pytest.approx(0.8)
        assert np.isnan(result["molecules"][1]["total_score"])

    def test_wrapper_delegates_even_with_empty_upstream_artifacts(self):
        """Downstream still runs (and may return NaN) when upstream has no artifacts."""
        wrapped = MagicMock()
        wrapped.request_params = {}
        wrapped.service = "posebusters"
        wrapped.return_value = MagicMock(
            scores=[np.full(2, np.nan)],
        )

        upstream = MagicMock()
        upstream.last_artifacts = None

        ttp = MagicMock()

        wrapper = _ArtifactInjectingWrapper(wrapped, upstream, ttp)
        result = wrapper(["CCO", "c1ccccc1"])

        wrapped.assert_called_once()
        assert all(np.isnan(s) for s in result.scores[0])


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


def test_single_external_rest_scorer_copy():
    """Guard against re-introducing a divergent second ExternalRESTScorer.

    Exactly one ``comp_external_rest.py`` must exist -- the runtime plugin under
    the ``_plugins`` namespace dir. A duplicate elsewhere is what let the tests
    validate a copy that was not the code that ran.
    """
    from pathlib import Path

    pkg_root = Path(__file__).resolve().parents[1] / "reinvent4_mcp"
    copies = sorted(pkg_root.rglob("comp_external_rest.py"))
    assert len(copies) == 1, f"expected exactly one copy, found: {copies}"
    assert "_plugins" in copies[0].parts or "_plugins" in str(copies[0])
