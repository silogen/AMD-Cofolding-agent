"""Tests for Phase 4: background job manager, TOML generation, and optimization tools.

Covers:
- TOML config generation from scorer config + process config
- Job manager lifecycle (create, start, poll, stop)
- Score trajectory parsing from output CSV
- Top candidate extraction
- MCP tool routing (run_optimization, stop_optimization, status, results)
"""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from reinvent4_mcp.config import ProcessConfig, reset_config
from reinvent4_mcp.jobs.manager import (
    JobInfo,
    JobManager,
    JobState,
    _component_health,
    _compute_diagnostics,
    _generate_recommendations,
    _generate_warnings,
    _per_component_stats,
    _per_component_trajectory,
    _safe_float,
    _step_trajectory,
    reset_job_manager,
)
from reinvent4_mcp.jobs.toml_generator import (
    _toml_bool,
    _toml_value,
    generate_reinvent_toml,
    generate_tl_toml,
    resolve_model_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SCORER_CONFIG = {
    "type": "geometric_mean",
    "component": [
        {"QED": {"endpoint": [{"name": "QED", "weight": 0.6}]}},
    ],
}

FULL_SCORER_CONFIG = {
    "type": "geometric_mean",
    "component": [
        {"QED": {"endpoint": [{"name": "QED", "weight": 0.25}]}},
        {
            "MolecularWeight": {
                "endpoint": [
                    {
                        "name": "MW",
                        "weight": 0.2,
                        "transform": {
                            "type": "double_sigmoid",
                            "high": 500,
                            "low": 200,
                            "coef_div": 500,
                            "coef_si": 20,
                            "coef_se": 20,
                        },
                    }
                ]
            }
        },
        {
            "ExternalRESTScorer": {
                "endpoint": [
                    {
                        "name": "Binding Affinity",
                        "weight": 0.5,
                        "params": {
                            "service": "boltz",
                            "service_url": "http://boltz:8080/score",
                            "protein_sequence": "MVLSPADKTN",
                            "affinity_mode": "pred_value",
                        },
                        "transform": {
                            "type": "reverse_sigmoid",
                            "high": -5.0,
                            "low": -8.0,
                            "k": 0.5,
                        },
                    }
                ]
            }
        },
        {
            "CustomAlerts": {
                "endpoint": [
                    {
                        "name": "constraint_custom_alerts",
                        "weight": 1.0,
                        "params": {"smarts": ["[*;r8]", "[#8][#8]"]},
                    }
                ]
            }
        },
    ],
}

MINIMAL_TTP = {
    "id": "test-opt-ttp",
    "name": "Test Optimization TTP",
    "scoring": {
        "objectives": [
            {"name": "SA Score", "type": "native", "component": "SAScore"},
        ],
    },
}


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset singletons between tests."""
    reset_config()
    reset_job_manager()
    yield
    reset_config()
    reset_job_manager()


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    """Create a temporary jobs directory."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    return jobs_dir


@pytest.fixture
def config(tmp_jobs_dir, tmp_path):
    """Create a ProcessConfig with temp directories."""
    return ProcessConfig(
        jobs={"working_dir": str(tmp_jobs_dir)},
        ttp={"storage_dir": str(tmp_path / "ttps")},
        logging={"level": "debug", "tb_logdir": str(tmp_path / "tb"), "log_dir": str(tmp_path / "logs")},
    )


@pytest.fixture
def manager(config):
    return JobManager(config)


# ---------------------------------------------------------------------------
# TOML Value Serialization
# ---------------------------------------------------------------------------


class TestTomlValueSerialization:

    def test_bool_true(self):
        assert _toml_bool(True) == "true"

    def test_bool_false(self):
        assert _toml_bool(False) == "false"

    def test_int(self):
        assert _toml_value(42) == "42"

    def test_float(self):
        assert _toml_value(3.14) == "3.14"

    def test_string(self):
        assert _toml_value("hello") == '"hello"'

    def test_string_with_quotes(self):
        assert _toml_value('say "hi"') == '"say \\"hi\\""'

    def test_list_of_strings(self):
        result = _toml_value(["a", "b"])
        assert result == '["a", "b"]'

    def test_list_of_ints(self):
        assert _toml_value([1, 2, 3]) == "[1, 2, 3]"

    def test_nested_list(self):
        result = _toml_value([["CCO"]])
        assert result == '[["CCO"]]'

    def test_bool_value(self):
        assert _toml_value(True) == "true"
        assert _toml_value(False) == "false"


# ---------------------------------------------------------------------------
# TOML Generation
# ---------------------------------------------------------------------------


class TestTOMLGeneration:

    def test_minimal_config(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/path/to/prior.model",
            max_steps=100,
        )

        assert 'run_type = "staged_learning"' in toml
        assert 'prior_file = "/path/to/prior.model"' in toml
        assert 'agent_file = "/path/to/prior.model"' in toml
        assert "max_steps = 100" in toml
        assert 'type = "geometric_mean"' in toml
        assert "[[stage.scoring.component]]" in toml
        assert "[stage.scoring.component.QED]" in toml
        assert 'name = "QED"' in toml
        assert "weight = 0.6" in toml

    def test_agent_file_defaults_to_prior(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
        )
        assert 'agent_file = "/model/prior.model"' in toml

    def test_checkpoint_resume(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            agent_file="/checkpoint/stage.chkpt",
            use_checkpoint=True,
        )
        assert "use_checkpoint = true" in toml
        assert 'agent_file = "/checkpoint/stage.chkpt"' in toml

    def test_full_config_with_external(self):
        toml = generate_reinvent_toml(
            scorer_config=FULL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            max_steps=200,
            batch_size=32,
        )

        assert "batch_size = 32" in toml
        assert "[stage.scoring.component.ExternalRESTScorer]" in toml
        assert 'params.service = "boltz"' in toml
        assert 'params.service_url = "http://boltz:8080/score"' in toml
        assert 'params.affinity_mode = "pred_value"' in toml

    def test_transform_in_toml(self):
        toml = generate_reinvent_toml(
            scorer_config=FULL_SCORER_CONFIG,
            prior_file="/model/prior.model",
        )

        assert 'transform.type = "double_sigmoid"' in toml
        assert "transform.high = 500" in toml
        assert "transform.low = 200" in toml
        assert 'transform.type = "reverse_sigmoid"' in toml

    def test_constraints_in_toml(self):
        toml = generate_reinvent_toml(
            scorer_config=FULL_SCORER_CONFIG,
            prior_file="/model/prior.model",
        )

        assert "[stage.scoring.component.CustomAlerts]" in toml
        assert 'params.smarts = ["[*;r8]", "[#8][#8]"]' in toml

    def test_diversity_filter(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            diversity_filter_type="ScaffoldSimilarity",
            diversity_filter_bucket_size=10,
            diversity_filter_minscore=0.7,
        )

        assert '[diversity_filter]' in toml
        assert 'type = "ScaffoldSimilarity"' in toml
        assert "bucket_size = 10" in toml
        assert "minscore = 0.7" in toml

    def test_learning_strategy(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            sigma=256,
            learning_rate=0.001,
        )

        assert "[learning_strategy]" in toml
        assert "sigma = 256" in toml
        assert "rate = 0.001" in toml

    def test_device(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            device="cpu",
        )
        assert 'device = "cpu"' in toml


# ---------------------------------------------------------------------------
# Model Resolution
# ---------------------------------------------------------------------------


class TestModelResolution:

    def test_file_path_unchanged(self):
        assert resolve_model_path("/path/to/model.prior") == "/path/to/model.prior"

    def test_relative_path_unchanged(self):
        assert resolve_model_path("models/reinvent.prior") == "models/reinvent.prior"

    def test_registry_key_fallback(self):
        """When REINVENT4 is not available, registry key is returned as-is."""
        with patch("reinvent4_mcp.utils.ensure_reinvent4_importable", side_effect=RuntimeError):
            result = resolve_model_path(".reinvent")
        assert result == ".reinvent"


# ---------------------------------------------------------------------------
# Safe Float Helper
# ---------------------------------------------------------------------------


class TestSafeFloat:

    def test_normal_float(self):
        assert _safe_float(0.5) == 0.5

    def test_nan(self):
        assert _safe_float(float("nan")) is None

    def test_none(self):
        assert _safe_float(None) is None

    def test_string_number(self):
        assert _safe_float("0.75") == 0.75

    def test_bad_string(self):
        assert _safe_float("not_a_number") is None


# ---------------------------------------------------------------------------
# Per-Component Stats
# ---------------------------------------------------------------------------


class TestPerComponentStats:

    def test_basic_stats(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B", "C"],
            "total_score": [0.5, 0.7, 0.9],
            "QED": [0.6, 0.8, 0.7],
            "QED (raw)": [0.6, 0.8, 0.7],
        })
        stats = _per_component_stats(df)
        assert "QED" in stats
        assert stats["QED"]["mean"] == pytest.approx(0.7, abs=0.01)
        assert stats["QED"]["max"] == pytest.approx(0.8, abs=0.01)
        assert "mean_raw" in stats["QED"]

    def test_skips_non_float_columns(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B"],
            "total_score": [0.5, 0.7],
            "text_col": ["x", "y"],
        })
        stats = _per_component_stats(df)
        assert "text_col" not in stats


# ---------------------------------------------------------------------------
# Step Trajectory
# ---------------------------------------------------------------------------


class TestStepTrajectory:

    def test_basic_trajectory(self):
        df = pd.DataFrame({
            "Step": [0, 0, 1, 1, 2, 2],
            "total_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        })
        traj = _step_trajectory(df, "total_score", "Step")

        assert traj is not None
        assert len(traj) == 3
        assert traj[0]["step"] == 0
        assert traj[0]["mean_score"] == pytest.approx(0.15, abs=0.01)
        assert traj[2]["step"] == 2
        assert traj[2]["mean_score"] == pytest.approx(0.55, abs=0.01)

    def test_none_when_no_columns(self):
        df = pd.DataFrame({"SMILES": ["A"]})
        assert _step_trajectory(df, None, None) is None

    def test_sampling_large_trajectory(self):
        n_steps = 200
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": [i / n_steps for i in range(n_steps)],
        })
        traj = _step_trajectory(df, "total_score", "Step", max_points=20)
        assert traj is not None
        assert len(traj) == 20


# ---------------------------------------------------------------------------
# Per-Component Trajectory
# ---------------------------------------------------------------------------


class TestPerComponentTrajectory:

    def test_basic_per_component_trajectory(self):
        df = pd.DataFrame({
            "Step": [0, 0, 1, 1, 2, 2],
            "total_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "QED": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
            "MW": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        })
        result = _per_component_trajectory(df, "Step")

        assert result is not None
        assert "QED" in result
        assert "MW" in result
        assert len(result["QED"]) == 3
        assert result["QED"][0]["step"] == 0
        assert result["QED"][2]["step"] == 2

    def test_skips_non_score_columns(self):
        df = pd.DataFrame({
            "Step": [0, 1],
            "SMILES": ["CCO", "CCC"],
            "total_score": [0.5, 0.6],
            "QED": [0.7, 0.8],
        })
        result = _per_component_trajectory(df, "Step")

        assert result is not None
        assert "QED" in result
        assert "SMILES" not in result
        assert "total_score" not in result

    def test_none_when_no_step_col(self):
        df = pd.DataFrame({"QED": [0.5, 0.6]})
        assert _per_component_trajectory(df, None) is None

    def test_sampling_large_trajectory(self):
        n_steps = 100
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "QED": [i / n_steps for i in range(n_steps)],
        })
        result = _per_component_trajectory(df, "Step", max_points=10)

        assert result is not None
        assert len(result["QED"]) == 10


# ---------------------------------------------------------------------------
# Component Health
# ---------------------------------------------------------------------------


class TestComponentHealth:

    def test_healthy_components(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B", "C", "D", "E"],
            "total_score": [0.5, 0.6, 0.7, 0.8, 0.9],
            "QED": [0.6, 0.7, 0.8, 0.85, 0.9],
        })
        health = _component_health(df)

        assert "QED" in health
        assert health["QED"]["status"] == "healthy"
        assert health["QED"]["nan_rate"] == 0.0

    def test_failing_component_high_nan(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B", "C", "D", "E"],
            "total_score": [0.5, 0.6, 0.7, 0.8, 0.9],
            "Binding Affinity": [0.5, float("nan"), float("nan"), float("nan"), float("nan")],
        })
        health = _component_health(df)

        assert "Binding Affinity" in health
        assert health["Binding Affinity"]["status"] == "failing"
        assert health["Binding Affinity"]["nan_rate"] == 0.8

    def test_degraded_component(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B", "C", "D", "E"],
            "total_score": [0.5, 0.6, 0.7, 0.8, 0.9],
            "SA": [0.5, float("nan"), 0.7, float("nan"), 0.9],
        })
        health = _component_health(df)

        assert health["SA"]["status"] == "degraded"
        assert health["SA"]["nan_rate"] == pytest.approx(0.4, abs=0.01)

    def test_skips_raw_columns(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B"],
            "QED": [0.5, 0.6],
            "QED (raw)": [0.5, 0.6],
        })
        health = _component_health(df)

        assert "QED" in health
        assert "QED (raw)" not in health

    def test_high_zero_rate_degraded(self):
        df = pd.DataFrame({
            "SMILES": ["A", "B", "C", "D", "E"],
            "total_score": [0.5, 0.0, 0.0, 0.0, 0.0],
            "PoseQuality": [0.5, 0.0, 0.0, 0.0, 0.0],
        })
        health = _component_health(df)

        assert health["PoseQuality"]["status"] == "degraded"


# ---------------------------------------------------------------------------
# Convergence Diagnostics
# ---------------------------------------------------------------------------


class TestConvergenceDiagnostics:

    def _make_job(self, tmp_path, max_steps=300):
        return JobInfo(
            job_id="test",
            ttp_id="test-ttp",
            ttp_version=1,
            model=".reinvent",
            working_dir=str(tmp_path),
            toml_path=str(tmp_path / "config.toml"),
            max_steps=max_steps,
        )

    def test_early_stage_few_steps(self, tmp_path):
        df = pd.DataFrame({
            "Step": [0, 1],
            "total_score": [0.1, 0.15],
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag is not None
        assert diag["convergence_status"] == "early"

    def test_improving_trajectory(self, tmp_path):
        n_steps = 50
        scores = [0.1 + (i * 0.01) for i in range(n_steps)]
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": scores,
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag is not None
        assert diag["convergence_status"] == "improving"
        assert diag["plateau_detected"] is False
        assert diag["diverging"] is False
        assert diag["improvement_rate"] > 0

    def test_plateau_detection(self, tmp_path):
        n_steps = 50
        scores = ([0.1 + (i * 0.01) for i in range(30)]
                  + [0.4] * 20)
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": scores,
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag is not None
        assert diag["plateau_detected"] is True
        assert diag["convergence_status"] == "plateau"

    def test_diverging_trajectory(self, tmp_path):
        n_steps = 40
        scores = ([0.1 + (i * 0.02) for i in range(20)]
                  + [0.5 - (i * 0.02) for i in range(20)])
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": scores,
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag is not None
        assert diag["diverging"] is True
        assert diag["convergence_status"] == "diverging"

    def test_completed_job_status(self, tmp_path):
        n_steps = 20
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": [0.5] * n_steps,
        })
        job = self._make_job(tmp_path)
        job.state = JobState.COMPLETED
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag is not None
        assert diag["convergence_status"] == "completed"

    def test_trend_computed(self, tmp_path):
        n_steps = 30
        scores = [0.1 + (i * 0.02) for i in range(n_steps)]
        df = pd.DataFrame({
            "Step": list(range(n_steps)),
            "total_score": scores,
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert "trend" in diag
        trend = diag["trend"]
        assert trend["early_mean"] < trend["recent_mean"]
        assert trend["overall_improvement"] > 0

    def test_progress_percentage(self, tmp_path):
        df = pd.DataFrame({
            "Step": list(range(50)),
            "total_score": [0.3] * 50,
        })
        job = self._make_job(tmp_path, max_steps=100)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert diag["progress_pct"] == 50.0

    def test_component_health_included(self, tmp_path):
        df = pd.DataFrame({
            "Step": list(range(10)),
            "total_score": [0.3] * 10,
            "QED": [0.5] * 10,
            "Boltz": [float("nan")] * 7 + [0.3] * 3,
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert "component_health" in diag
        assert diag["component_health"]["QED"]["status"] == "healthy"
        assert diag["component_health"]["Boltz"]["status"] == "failing"

    def test_recent_validity_rate(self, tmp_path):
        n_steps = 20
        df = pd.DataFrame({
            "Step": [i // 2 for i in range(n_steps * 2)],
            "SMILES_state": ([1, 1] * 15) + ([1, 0] * 5),
            "total_score": [0.3] * (n_steps * 2),
        })
        job = self._make_job(tmp_path)
        diag = _compute_diagnostics(df, "total_score", "Step", {}, job)

        assert "recent_validity_rate" in diag

    def test_none_when_no_total_col(self, tmp_path):
        df = pd.DataFrame({"Step": [0, 1], "QED": [0.5, 0.6]})
        job = self._make_job(tmp_path)
        assert _compute_diagnostics(df, None, "Step", {}, job) is None

    def test_none_when_empty_df(self, tmp_path):
        df = pd.DataFrame(columns=["Step", "total_score"])
        job = self._make_job(tmp_path)
        assert _compute_diagnostics(df, "total_score", "Step", {}, job) is None


# ---------------------------------------------------------------------------
# Warnings and Recommendations
# ---------------------------------------------------------------------------


class TestWarningsAndRecommendations:

    def test_plateau_warning(self):
        diagnostics = {
            "plateau_detected": True,
            "plateau_since_step": 100,
            "diverging": False,
            "convergence_status": "plateau",
        }
        warnings = _generate_warnings(
            diagnostics, {}, pd.DataFrame(), "total_score", None,
        )
        assert any(w["type"] == "score_plateau" for w in warnings)

    def test_diverging_warning(self):
        diagnostics = {
            "plateau_detected": False,
            "diverging": True,
            "convergence_status": "diverging",
        }
        warnings = _generate_warnings(
            diagnostics, {}, pd.DataFrame(), "total_score", None,
        )
        assert any(w["type"] == "diverging" for w in warnings)

    def test_component_failure_warning(self):
        health = {
            "Boltz": {"status": "failing", "nan_rate": 0.8},
        }
        diagnostics = {"plateau_detected": False, "diverging": False}
        warnings = _generate_warnings(
            diagnostics, health, pd.DataFrame(), "total_score", None,
        )
        assert any(w["type"] == "component_failure" for w in warnings)
        assert any("Boltz" in w.get("component", "") for w in warnings)

    def test_low_validity_warning(self):
        diagnostics = {
            "recent_validity_rate": 0.2,
            "plateau_detected": False,
            "diverging": False,
        }
        warnings = _generate_warnings(
            diagnostics, {}, pd.DataFrame(), "total_score", "SMILES_state",
        )
        assert any(w["type"] == "low_validity" for w in warnings)
        assert any(w["severity"] == "high" for w in warnings)

    def test_plateau_recommendation(self):
        diagnostics = {"convergence_status": "plateau"}
        recs = _generate_recommendations(diagnostics, {})
        assert any("plateau" in r.lower() for r in recs)

    def test_diverging_recommendation(self):
        diagnostics = {"convergence_status": "diverging"}
        recs = _generate_recommendations(diagnostics, {})
        assert any("declining" in r.lower() for r in recs)

    def test_failing_component_recommendation(self):
        health = {"Boltz": {"status": "failing", "nan_rate": 0.9}}
        diagnostics = {"convergence_status": "stable"}
        recs = _generate_recommendations(diagnostics, health)
        assert any("Boltz" in r for r in recs)

    def test_improving_near_end_recommendation(self):
        diagnostics = {"convergence_status": "improving", "progress_pct": 85.0}
        recs = _generate_recommendations(diagnostics, {})
        assert any("finish" in r.lower() for r in recs)

    def test_early_recommendation(self):
        diagnostics = {"convergence_status": "early"}
        recs = _generate_recommendations(diagnostics, {})
        assert any("monitoring" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# Integration: get_status with diagnostics
# ---------------------------------------------------------------------------


class TestGetStatusWithDiagnostics:

    def test_status_includes_diagnostics(self, manager):
        """Full get_status includes diagnostics when enough data exists."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        n_steps = 30
        csv_path = Path(result["working_dir"]) / "results_1.csv"
        df = pd.DataFrame({
            "Step": [i // 2 for i in range(n_steps * 2)],
            "SMILES": [f"C{'C' * (i % 5)}" for i in range(n_steps * 2)],
            "SMILES_state": [1] * (n_steps * 2),
            "total_score": [0.1 + (i * 0.005) for i in range(n_steps * 2)],
            "QED": [0.5 + (i * 0.003) for i in range(n_steps * 2)],
        })
        df.to_csv(csv_path, index=False)

        status = manager.get_status(job_id)
        assert "trajectory" in status
        traj = status["trajectory"]
        assert "diagnostics" in traj

        diag = traj["diagnostics"]
        assert "convergence_status" in diag
        assert "component_health" in diag
        assert "trend" in diag
        assert "progress_pct" in diag

    def test_status_includes_per_component_trajectory(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        csv_path = Path(result["working_dir"]) / "results_1.csv"
        df = pd.DataFrame({
            "Step": [0, 0, 1, 1, 2, 2],
            "SMILES": ["CCO", "CCC", "CCN", "CCCO", "CCCC", "CCCN"],
            "total_score": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "QED": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        })
        df.to_csv(csv_path, index=False)

        status = manager.get_status(job_id)
        traj = status["trajectory"]
        assert "per_component_trajectory" in traj
        assert "QED" in traj["per_component_trajectory"]
        assert len(traj["per_component_trajectory"]["QED"]) == 3

    def test_status_early_run_no_crash(self, manager):
        """Status with very few steps returns diagnostics without crashing."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        csv_path = Path(result["working_dir"]) / "results_1.csv"
        df = pd.DataFrame({
            "Step": [0, 0],
            "SMILES": ["CCO", "CCC"],
            "total_score": [0.1, 0.2],
        })
        df.to_csv(csv_path, index=False)

        status = manager.get_status(job_id)
        traj = status["trajectory"]
        assert traj["current_step"] == 1

    def test_status_with_failing_component(self, manager):
        """Diagnostics detect failing components."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        n = 40
        csv_path = Path(result["working_dir"]) / "results_1.csv"
        boltz_scores = [float("nan")] * 35 + [0.3] * 5
        df = pd.DataFrame({
            "Step": list(range(n)),
            "SMILES": [f"C{'C' * (i % 5)}" for i in range(n)],
            "total_score": [0.2] * n,
            "QED": [0.6] * n,
            "Boltz": boltz_scores,
        })
        df.to_csv(csv_path, index=False)

        status = manager.get_status(job_id)
        diag = status["trajectory"]["diagnostics"]
        assert diag["component_health"]["Boltz"]["status"] == "failing"
        assert any(
            w["type"] == "component_failure" and w["component"] == "Boltz"
            for w in diag.get("warnings", [])
        )


# ---------------------------------------------------------------------------
# JobInfo
# ---------------------------------------------------------------------------


class TestJobInfo:

    def test_csv_path(self, tmp_path):
        job = JobInfo(
            job_id="abc",
            ttp_id="test",
            ttp_version=1,
            model=".reinvent",
            working_dir=str(tmp_path),
            toml_path=str(tmp_path / "config.toml"),
        )
        assert job.csv_path == tmp_path / "results_1.csv"

    def test_checkpoint_path(self, tmp_path):
        job = JobInfo(
            job_id="abc",
            ttp_id="test",
            ttp_version=1,
            model=".reinvent",
            working_dir=str(tmp_path),
            toml_path=str(tmp_path / "config.toml"),
        )
        assert job.checkpoint_path == tmp_path / "stage.chkpt"

    def test_to_dict(self, tmp_path):
        job = JobInfo(
            job_id="abc",
            ttp_id="test",
            ttp_version=1,
            model=".reinvent",
            working_dir=str(tmp_path),
            toml_path=str(tmp_path / "config.toml"),
        )
        d = job.to_dict()
        assert d["job_id"] == "abc"
        assert d["state"] == "queued"
        assert d["ttp_id"] == "test"


# ---------------------------------------------------------------------------
# JobManager -- Creation and Status
# ---------------------------------------------------------------------------


class TestJobManagerLifecycle:

    def test_create_and_start_job(self, manager, tmp_path):
        """Creating a job writes TOML and metadata files."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
                max_steps=50,
            )

        assert "error" not in result
        assert "job_id" in result

        job_dir = Path(result["working_dir"])
        assert job_dir.exists()
        assert (job_dir / "config.toml").exists()
        assert (job_dir / "job_metadata.json").exists()

        toml_content = (job_dir / "config.toml").read_text()
        assert 'run_type = "staged_learning"' in toml_content
        assert "max_steps = 50" in toml_content

    def test_max_concurrent_jobs_enforced(self, manager, tmp_path):
        """Cannot exceed max_concurrent_jobs."""
        with patch.object(manager, "_start_process"):
            for i in range(manager._config.jobs.max_concurrent_jobs):
                result = manager.create_and_start(
                    ttp_id=f"ttp-{i}",
                    ttp_version=1,
                    scorer_config=MINIMAL_SCORER_CONFIG,
                    model="/model/prior.model",
                )
                assert "error" not in result
                # _start_process is mocked, so manually set state to RUNNING
                manager._jobs[result["job_id"]].state = JobState.RUNNING

            result = manager.create_and_start(
                ttp_id="ttp-excess",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
            assert "error" in result
            assert "Maximum concurrent jobs" in result["error"]

    def test_several_jobs_run_at_once(self, manager, tmp_path):
        """The point of the slot limit: concurrent jobs, up to the cap.

        Starting a job used to stop every running job first, so one RL run was
        the hard ceiling however high ``max_concurrent_jobs`` was set. An RL
        step is mostly spent waiting on the scoring HTTP call, so the GPU has
        room for several.
        """
        with patch.object(manager, "_start_process"):
            first = manager.create_and_start(
                ttp_id="ttp-a", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )
            manager._jobs[first["job_id"]].state = JobState.RUNNING
            second = manager.create_and_start(
                ttp_id="ttp-b", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        assert "error" not in second
        # The first is untouched: no auto-stop, no checkpoint, still running.
        assert manager._jobs[first["job_id"]].state == JobState.RUNNING
        assert "stopped_previous_jobs" not in second

    def test_a_queued_job_holds_a_slot(self, manager):
        """A job is registered before its subprocess reports RUNNING.

        Counting only RUNNING would leave a window where two callers both pass
        the capacity check.
        """
        with patch.object(manager, "_start_process"):
            for i in range(manager._config.jobs.max_concurrent_jobs):
                created = manager.create_and_start(
                    ttp_id=f"ttp-{i}", ttp_version=1,
                    scorer_config=MINIMAL_SCORER_CONFIG,
                    model="/model/prior.model",
                )
                # Left QUEUED on purpose -- _start_process is mocked.
                assert manager._jobs[created["job_id"]].state == JobState.QUEUED

            refused = manager.create_and_start(
                ttp_id="ttp-excess", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        assert "Maximum concurrent jobs" in refused["error"]

    def test_the_refusal_says_what_to_stop(self, manager):
        with patch.object(manager, "_start_process"):
            held = [
                manager.create_and_start(
                    ttp_id=f"ttp-{i}", ttp_version=1,
                    scorer_config=MINIMAL_SCORER_CONFIG,
                    model="/model/prior.model",
                )["job_id"]
                for i in range(manager._config.jobs.max_concurrent_jobs)
            ]
            refused = manager.create_and_start(
                ttp_id="ttp-excess", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        # An agent has to know which jobs to stop to make room.
        assert sorted(refused["active_jobs"]) == sorted(held)
        assert "jobs.max_concurrent_jobs" in refused["error"]

    @pytest.mark.parametrize(
        "freeing_state",
        [JobState.COMPLETED, JobState.FAILED, JobState.STOPPED],
    )
    def test_a_finished_job_frees_its_slot(self, manager, freeing_state):
        with patch.object(manager, "_start_process"):
            jobs = [
                manager.create_and_start(
                    ttp_id=f"ttp-{i}", ttp_version=1,
                    scorer_config=MINIMAL_SCORER_CONFIG,
                    model="/model/prior.model",
                )["job_id"]
                for i in range(manager._config.jobs.max_concurrent_jobs)
            ]
            manager._jobs[jobs[0]].state = freeing_state

            allowed = manager.create_and_start(
                ttp_id="ttp-next", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        assert "error" not in allowed

    def test_a_refused_job_leaves_nothing_behind(self, manager):
        with patch.object(manager, "_start_process"):
            for i in range(manager._config.jobs.max_concurrent_jobs):
                manager.create_and_start(
                    ttp_id=f"ttp-{i}", ttp_version=1,
                    scorer_config=MINIMAL_SCORER_CONFIG,
                    model="/model/prior.model",
                )
            before = set(manager._working_root.glob("job_*"))

            manager.create_and_start(
                ttp_id="ttp-excess", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        # No orphaned working dir for a job that never started.
        assert set(manager._working_root.glob("job_*")) == before

    def test_rl_and_tl_share_the_slots(self, manager):
        # They share the GPU, so they have to share the cap.
        with patch.object(manager, "_start_process"):
            assert manager._config.jobs.max_concurrent_jobs == 2
            manager.create_and_start(
                ttp_id="ttp-a", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )
            manager.create_and_start_tl(model="/model/prior.model", smiles=["CCO"])

            refused = manager.create_and_start(
                ttp_id="ttp-c", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

        assert "Maximum concurrent jobs" in refused["error"]

    def test_a_restart_frees_slots_that_cannot_be_running(self, manager, config):
        """Otherwise an interrupted job holds a slot forever.

        Nothing survives the process that launched it, so both QUEUED and
        RUNNING are reset on restore -- including QUEUED, which was previously
        left alone because nothing counted it.
        """
        from reinvent4_mcp.jobs.manager import JobManager

        with patch.object(manager, "_start_process"):
            queued = manager.create_and_start(
                ttp_id="ttp-a", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )["job_id"]
            running = manager.create_and_start(
                ttp_id="ttp-b", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )["job_id"]
            manager._jobs[running].state = JobState.RUNNING
            manager._save_metadata(manager._jobs[running])

        revived = JobManager(config)

        assert revived._jobs[queued].state == JobState.FAILED
        assert revived._jobs[running].state == JobState.FAILED
        with patch.object(revived, "_start_process"):
            assert "error" not in revived.create_and_start(
                ttp_id="ttp-c", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            )

    def test_concurrent_callers_cannot_both_take_the_last_slot(self, manager):
        """The count and the registration share one lock acquisition.

        Checking first and registering later would let N supervisors driving
        one runner all pass a check that only one of them should have.
        """
        import threading

        results: list[dict] = []
        barrier = threading.Barrier(4)

        def _create(index: int) -> None:
            barrier.wait()
            results.append(manager.create_and_start(
                ttp_id=f"ttp-{index}", ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG, model="/model/prior.model",
            ))

        with patch.object(manager, "_start_process"):
            threads = [threading.Thread(target=_create, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)

        started = [r for r in results if "error" not in r]
        refused = [r for r in results if "error" in r]

        assert len(started) == manager._config.jobs.max_concurrent_jobs
        assert len(refused) == 4 - manager._config.jobs.max_concurrent_jobs
        assert len(manager._active_job_ids()) == len(started)

    def test_get_status_unknown_job(self, manager):
        result = manager.get_status("nonexistent")
        assert "error" in result

    def test_get_status_with_csv(self, manager, tmp_path):
        """Status includes trajectory when CSV exists."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        csv_path = Path(result["working_dir"]) / "results_1.csv"
        df = pd.DataFrame({
            "Step": [0, 0, 1, 1],
            "SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "CCC"],
            "SMILES_state": [1, 1, 1, 0],
            "total_score": [0.3, 0.4, 0.5, 0.0],
            "QED": [0.6, 0.7, 0.8, 0.0],
        })
        df.to_csv(csv_path, index=False)

        status = manager.get_status(job_id)
        assert "trajectory" in status
        traj = status["trajectory"]
        assert traj["current_step"] == 2
        assert traj["n_molecules_total"] == 4
        assert traj["n_valid"] == 3
        assert traj["validity_rate"] == 0.75

    def test_list_jobs(self, manager):
        with patch.object(manager, "_start_process"):
            manager.create_and_start(
                ttp_id="ttp-1",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        jobs = manager.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["ttp_id"] == "ttp-1"


# ---------------------------------------------------------------------------
# JobManager -- Stop
# ---------------------------------------------------------------------------


class TestJobManagerStop:

    def test_stop_non_running_job(self, manager):
        """Cannot stop a job that isn't running."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]

        job = manager._jobs[job_id]
        job.state = JobState.COMPLETED

        stop_result = manager.stop_job(job_id)
        assert "error" in stop_result

    @staticmethod
    def _running_job(manager, exits_after: int = 0):
        """A RUNNING job whose process stops after `exits_after` signals."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )

        job = manager._jobs[result["job_id"]]
        job.state = JobState.RUNNING
        job.pid = 4242

        process = MagicMock()
        process.poll.return_value = None
        process.returncode = 0
        remaining = {"waits": exits_after}

        def wait(timeout=None):
            if remaining["waits"] <= 0:
                return 0
            remaining["waits"] -= 1
            raise subprocess.TimeoutExpired(cmd="reinvent", timeout=timeout)

        process.wait.side_effect = wait
        job._process = process
        return job

    def test_stop_asks_for_a_checkpoint_with_sigusr1(self, manager):
        """SIGUSR1 is the only signal REINVENT treats as a controlled stop."""
        job = self._running_job(manager)

        with patch("reinvent4_mcp.jobs.manager.os.kill") as kill:
            manager.stop_job(job.job_id)

        assert kill.call_args_list == [call(4242, signal.SIGUSR1)]
        assert job.state == JobState.STOPPED

    def test_stop_waits_a_whole_rl_step_for_the_checkpoint(self, manager):
        """The grace period is configurable and spans more than one step."""
        job = self._running_job(manager)
        manager._config.jobs.stop_grace_seconds = 900.0

        with patch("reinvent4_mcp.jobs.manager.os.kill"):
            manager.stop_job(job.job_id)

        assert job._process.wait.call_args_list == [call(timeout=900.0)]

    def test_stop_escalates_when_the_job_ignores_the_request(self, manager):
        """A job that never checkpoints still gets terminated, then killed."""
        job = self._running_job(manager, exits_after=2)
        manager._config.jobs.stop_grace_seconds = 1.0
        job.csv_path.write_text("Agent,Score\n26.4,0.92\n")

        with patch("reinvent4_mcp.jobs.manager.os.kill") as kill:
            result = manager.stop_job(job.job_id)

        assert [c.args[1] for c in kill.call_args_list] == [
            signal.SIGUSR1, signal.SIGTERM, signal.SIGKILL,
        ]
        assert "no checkpoint" in result["message"]
        assert "stop_grace_seconds" in result["message"]

    def test_stop_before_the_first_step_is_not_reported_as_a_failure(self, manager):
        """REINVENT has no agent to save until a step has updated the weights.

        The empty results CSV is the signal: blaming the grace period here
        would send an operator chasing a timeout that never happened.
        """
        job = self._running_job(manager)
        job.csv_path.write_text("")

        with patch("reinvent4_mcp.jobs.manager.os.kill"):
            result = manager.stop_job(job.job_id)

        assert "first step" in result["message"]
        assert "stop_grace_seconds" not in result["message"]

    def test_stop_skips_the_checkpoint_when_disabled(self, manager):
        """checkpoint_on_stop=false means terminate now, don't wait for a step."""
        job = self._running_job(manager)
        manager._config.jobs.checkpoint_on_stop = False
        manager._config.jobs.stop_grace_seconds = 900.0

        with patch("reinvent4_mcp.jobs.manager.os.kill") as kill:
            result = manager.stop_job(job.job_id)

        assert [c.args[1] for c in kill.call_args_list] == [signal.SIGTERM]
        assert job._process.wait.call_args_list == [call(timeout=10.0)]
        assert "checkpoint_on_stop is off" in result["message"]

    def test_stop_reports_the_checkpoint_it_found(self, manager):
        job = self._running_job(manager)
        job.checkpoint_path.write_text("checkpoint")

        with patch("reinvent4_mcp.jobs.manager.os.kill"):
            result = manager.stop_job(job.job_id)

        assert result["checkpoint_path"] == str(job.checkpoint_path)
        assert "Checkpoint saved at" in result["message"]

    def test_stop_tolerates_an_already_dead_process(self, manager):
        job = self._running_job(manager)

        with patch("reinvent4_mcp.jobs.manager.os.kill", side_effect=ProcessLookupError):
            result = manager.stop_job(job.job_id)

        assert job.state == JobState.STOPPED
        assert result["state"] == "stopped"
        assert not result["error"]


# ---------------------------------------------------------------------------
# JobManager -- Top Candidates
# ---------------------------------------------------------------------------


class TestJobManagerTopCandidates:

    def _completed_job(self, manager, df: pd.DataFrame):
        """A completed job whose results CSV is `df`."""
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job = manager._jobs[result["job_id"]]
        job.state = JobState.COMPLETED
        df.to_csv(job.csv_path, index=False)
        return result["job_id"]

    def test_a_ragged_csv_still_yields_its_good_rows(self, manager):
        """REINVENT appends rows under one header written from the first batch.

        A scorer reporting more metadata keys on a later batch makes those
        rows wider than the header, and pandas then refuses the whole file --
        observed live as "Expected 24 fields in line 18, saw 26", which cost
        the run its trajectory and would have failed the task at harvest.

        The surplus fields are inserted rather than appended (`step` stays
        last), so a wide row cannot be realigned to the header: truncating it
        would read None as the step and shift every metadata value. The rows
        are dropped instead -- partial but accurate.
        """
        job_id = self._completed_job(manager, pd.DataFrame({
            "SMILES": ["a", "b"],
            "SMILES_state": [1, 1],
            "Score": [0.4, 0.9],
            "step": [1, 1],
        }))
        job = manager._jobs[job_id]
        # A second step that reports one extra metadata column.
        with open(job.csv_path, "a") as handle:
            handle.write("c,1,0.7,extra_metadata,2\n")

        candidates = manager.get_top_candidates(job_id, top_n=5)["candidates"]

        assert [c["smiles"] for c in candidates] == ["b", "a"]
        assert "error" not in manager.get_top_candidates(job_id, top_n=5)

    def test_reinvents_own_score_column_is_recognised(self, manager):
        """REINVENT writes the aggregate as `Score`, not `total_score`.

        Resolving only `total_score` found nothing in a real RL results file,
        so the rows were never sorted and "top" candidates came back as the
        first N in the CSV -- the earliest and therefore worst molecules --
        with no `total_score` on them at all.
        """
        job_id = self._completed_job(manager, pd.DataFrame({
            "SMILES": ["worst", "best", "middling"],
            "SMILES_state": [1, 1, 1],
            "Agent": [26.4, 31.1, 28.0],
            "Prior": [26.5, 31.1, 28.1],
            "Target": [92.4, 75.9, 80.0],
            "Score": [0.11, 0.93, 0.50],
            "QED": [0.2, 0.88, 0.5],
        }))

        candidates = manager.get_top_candidates(job_id, top_n=3)["candidates"]

        assert [c["smiles"] for c in candidates] == ["best", "middling", "worst"]
        assert candidates[0]["total_score"] == 0.93

    def test_the_bookkeeping_columns_are_not_objectives(self, manager):
        """Agent, Prior, Target and Score are REINVENT's own, not components.

        Reported as components they show up as objectives beside the real
        ones, and the supervisor would put them on a Pareto front.
        """
        job_id = self._completed_job(manager, pd.DataFrame({
            "SMILES": ["CCO"],
            "SMILES_state": [1],
            "Agent": [26.4],
            "Prior": [26.5],
            "Target": [92.4],
            "Score": [0.93],
            "Scaffold": [0.0],
            "QED": [0.88],
        }))

        components = manager.get_top_candidates(job_id, top_n=1)[
            "candidates"
        ][0]["components"]

        assert set(components) == {"QED"}

    def test_the_older_total_score_column_still_works(self, manager):
        """Transfer-learning output and older CSVs use that name."""
        job_id = self._completed_job(manager, pd.DataFrame({
            "SMILES": ["low", "high"],
            "SMILES_state": [1, 1],
            "total_score": [0.2, 0.8],
        }))

        candidates = manager.get_top_candidates(job_id, top_n=2)["candidates"]

        assert [c["smiles"] for c in candidates] == ["high", "low"]

    def test_top_candidates(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]
        job = manager._jobs[job_id]
        job.state = JobState.COMPLETED

        csv_path = job.csv_path
        df = pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "CC(=O)O", "INVALID"],
            "SMILES_state": [1, 1, 1, 1, 0],
            "total_score": [0.9, 0.7, 0.5, 0.5, 0.0],
            "QED": [0.8, 0.6, 0.4, 0.4, 0.0],
        })
        df.to_csv(csv_path, index=False)

        candidates = manager.get_top_candidates(job_id, top_n=2)
        assert "error" not in candidates
        assert candidates["n_total"] == 2
        assert candidates["candidates"][0]["smiles"] == "CCO"
        assert candidates["candidates"][0]["total_score"] == pytest.approx(0.9)

    def test_top_candidates_with_min_score(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]
        job = manager._jobs[job_id]
        job.state = JobState.COMPLETED

        df = pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"],
            "SMILES_state": [1, 1, 1],
            "total_score": [0.9, 0.5, 0.3],
        })
        df.to_csv(job.csv_path, index=False)

        candidates = manager.get_top_candidates(job_id, top_n=10, min_score=0.6)
        assert candidates["n_total"] == 1
        assert candidates["candidates"][0]["smiles"] == "CCO"

    def test_no_csv_returns_error(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )
        job_id = result["job_id"]
        job = manager._jobs[job_id]
        job.state = JobState.COMPLETED

        candidates = manager.get_top_candidates(job_id)
        assert "error" in candidates


# ---------------------------------------------------------------------------
# JobManager -- Checkpoint Resume
# ---------------------------------------------------------------------------


class TestJobManagerResume:

    def test_resume_from_checkpoint(self, manager, tmp_path):
        """Resuming from a checkpoint sets use_checkpoint and agent_file."""
        chkpt_path = tmp_path / "stage.chkpt"
        chkpt_path.write_text("fake checkpoint")

        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
                resume_from=str(chkpt_path),
            )

        assert "error" not in result
        toml_content = Path(result["working_dir"]).joinpath("config.toml").read_text()
        assert "use_checkpoint = true" in toml_content
        assert str(chkpt_path) in toml_content

    def test_resume_from_nonexistent_checkpoint(self, manager, tmp_path):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
                resume_from="/nonexistent/checkpoint.chkpt",
            )
        assert "error" in result
        assert "Checkpoint not found" in result["error"]


# ---------------------------------------------------------------------------
# JobManager -- Metadata Persistence
# ---------------------------------------------------------------------------


class TestJobMetadataPersistence:

    def test_metadata_written_on_create(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )

        metadata_path = Path(result["working_dir"]) / "job_metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        assert data["job_id"] == result["job_id"]
        assert data["ttp_id"] == "test-ttp"

    def test_restore_jobs_on_init(self, config, tmp_path):
        """JobManager restores jobs from existing metadata on disk."""
        job_dir = Path(config.jobs.working_dir) / "job_abc123"
        job_dir.mkdir(parents=True)
        metadata = {
            "job_id": "abc123",
            "ttp_id": "test",
            "ttp_version": 1,
            "model": ".reinvent",
            "state": "completed",
            "working_dir": str(job_dir),
        }
        (job_dir / "job_metadata.json").write_text(json.dumps(metadata))

        manager = JobManager(config)
        jobs = manager.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "abc123"
        assert jobs[0]["state"] == "completed"

    def test_running_jobs_marked_failed_on_restore(self, config):
        """Jobs that were running when the server restarted are marked failed."""
        job_dir = Path(config.jobs.working_dir) / "job_running1"
        job_dir.mkdir(parents=True)
        metadata = {
            "job_id": "running1",
            "ttp_id": "test",
            "ttp_version": 1,
            "model": ".reinvent",
            "state": "running",
            "working_dir": str(job_dir),
        }
        (job_dir / "job_metadata.json").write_text(json.dumps(metadata))

        manager = JobManager(config)
        jobs = manager.list_jobs()
        assert jobs[0]["state"] == "failed"
        assert "restarted" in jobs[0]["error"]


# ---------------------------------------------------------------------------
# Tools: run_optimization
# ---------------------------------------------------------------------------


class TestRunOptimizationTool:

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_run_with_stored_ttp(self, mock_resolve, mock_summary, mock_scorer, mock_mgr):
        from reinvent4_mcp.tools.optimization import run_optimization
        from reinvent4_mcp.ttp.schema import TTP

        ttp = TTP.model_validate(MINIMAL_TTP)
        mock_resolve.return_value = ttp
        mock_scorer.return_value = {"type": "geometric_mean", "component": [{"QED": {}}]}
        mock_summary.return_value = {"n_total_components": 1}
        mock_mgr.return_value.create_and_start.return_value = {
            "job_id": "abc",
            "state": "running",
        }

        result = run_optimization(ttp_id="test-opt-ttp")

        assert result["job_id"] == "abc"
        assert result["ttp_id"] == "test-opt-ttp"
        mock_mgr.return_value.create_and_start.assert_called_once()

    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_run_with_bad_ttp(self, mock_resolve):
        from reinvent4_mcp.tools.optimization import run_optimization

        mock_resolve.return_value = {"error": "TTP not found"}

        result = run_optimization(ttp_id="nonexistent")
        assert "error" in result

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_empty_scorer_config(self, mock_resolve, mock_summary, mock_scorer, mock_mgr):
        from reinvent4_mcp.tools.optimization import run_optimization
        from reinvent4_mcp.ttp.schema import TTP

        ttp = TTP.model_validate(MINIMAL_TTP)
        mock_resolve.return_value = ttp
        mock_scorer.return_value = {"type": "geometric_mean", "component": []}

        result = run_optimization(ttp_id="test-opt-ttp")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tools: stop_optimization
# ---------------------------------------------------------------------------


class TestStopOptimizationTool:

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_stop_delegates_to_manager(self, mock_mgr):
        from reinvent4_mcp.tools.optimization import stop_optimization

        mock_mgr.return_value.stop_job.return_value = {
            "job_id": "abc",
            "state": "stopped",
            "message": "Job stopped",
        }

        result = stop_optimization("abc")
        assert result["state"] == "stopped"
        mock_mgr.return_value.stop_job.assert_called_once_with("abc")


# ---------------------------------------------------------------------------
# Tools: get_optimization_status
# ---------------------------------------------------------------------------


class TestGetOptimizationStatusTool:

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_status_delegates_to_manager(self, mock_mgr):
        from reinvent4_mcp.tools.results import get_optimization_status

        mock_mgr.return_value.get_status.return_value = {
            "job_id": "abc",
            "state": "running",
            "trajectory": {"current_step": 42},
        }

        result = get_optimization_status("abc")
        assert result["trajectory"]["current_step"] == 42


# ---------------------------------------------------------------------------
# Tools: get_top_candidates
# ---------------------------------------------------------------------------


class TestGetTopCandidatesTool:

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_from_job_id(self, mock_mgr):
        from reinvent4_mcp.tools.results import get_top_candidates

        mock_mgr.return_value.get_top_candidates.return_value = {
            "candidates": [{"smiles": "CCO", "total_score": 0.9}],
            "n_total": 1,
        }

        result = get_top_candidates(job_id="abc")
        assert result["n_total"] == 1

    def test_from_csv_path(self, tmp_path):
        from reinvent4_mcp.tools.results import get_top_candidates

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1"],
            "total_score": [0.9, 0.5],
        })
        df.to_csv(csv_path, index=False)

        result = get_top_candidates(csv_path=str(csv_path))
        assert result["n_total"] == 2

    def test_both_job_and_csv_is_error(self):
        from reinvent4_mcp.tools.results import get_top_candidates

        result = get_top_candidates(job_id="abc", csv_path="/path.csv")
        assert "error" in result


# ---------------------------------------------------------------------------
# Tools: read_results
# ---------------------------------------------------------------------------


class TestReadResultsTool:

    def test_read_results(self, tmp_path):
        from reinvent4_mcp.tools.results import read_results

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1", "BAD"],
            "SMILES_state": [1, 1, 0],
            "total_score": [0.8, 0.6, 0.0],
            "QED": [0.9, 0.7, 0.0],
        })
        df.to_csv(csv_path, index=False)

        result = read_results(str(csv_path))

        assert result["n_rows"] == 3
        assert result["n_unique_smiles"] == 3
        assert result["smiles_states"]["valid"] == 2
        assert result["validity_rate"] == pytest.approx(0.667, abs=0.01)
        assert result["total_score_stats"]["mean"] == pytest.approx(0.4667, abs=0.01)

    def test_missing_csv(self):
        from reinvent4_mcp.tools.results import read_results

        result = read_results("/nonexistent/path.csv")
        assert "error" in result


# ---------------------------------------------------------------------------
# Transfer Learning: TOML Generation
# ---------------------------------------------------------------------------


class TestTLTOMLGeneration:

    def test_minimal_tl_config(self):

        toml = generate_tl_toml(
            input_model_file="/model/prior.model",
            smiles_file="/data/train.smi",
        )

        assert 'run_type = "transfer_learning"' in toml
        assert 'input_model_file = "/model/prior.model"' in toml
        assert 'smiles_file = "/data/train.smi"' in toml
        assert 'output_model_file = "tl_model.model"' in toml
        assert "num_epochs = 50" in toml
        assert "batch_size = 128" in toml
        assert "standardize_smiles = true" in toml

    def test_custom_parameters(self):

        toml = generate_tl_toml(
            input_model_file="/model/checkpoint.chkpt",
            smiles_file="/data/train.smi",
            output_model_file="custom_output.model",
            num_epochs=100,
            batch_size=256,
            save_every_n_epochs=10,
            sample_batch_size=5000,
            device="cpu",
        )

        assert "num_epochs = 100" in toml
        assert "batch_size = 256" in toml
        assert "save_every_n_epochs = 10" in toml
        assert "sample_batch_size = 5000" in toml
        assert 'device = "cpu"' in toml
        assert 'output_model_file = "custom_output.model"' in toml

    def test_with_validation_file(self):

        toml = generate_tl_toml(
            input_model_file="/model/prior.model",
            smiles_file="/data/train.smi",
            validation_smiles_file="/data/val.smi",
        )

        assert 'validation_smiles_file = "/data/val.smi"' in toml

    def test_no_validation_file(self):

        toml = generate_tl_toml(
            input_model_file="/model/prior.model",
            smiles_file="/data/train.smi",
        )

        assert "validation_smiles_file" not in toml

    def test_no_staged_learning_sections(self):

        toml = generate_tl_toml(
            input_model_file="/model/prior.model",
            smiles_file="/data/train.smi",
        )

        assert "[[stage]]" not in toml
        assert "[learning_strategy]" not in toml
        assert "[diversity_filter]" not in toml
        assert "scoring" not in toml


# ---------------------------------------------------------------------------
# Transfer Learning: JobManager
# ---------------------------------------------------------------------------


class TestJobManagerTL:

    def test_create_tl_job(self, manager, tmp_path):
        """Creating a TL job writes SMILES file and TOML config."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

        with patch.object(manager, "_start_process"):
            result = manager.create_and_start_tl(
                model="/model/prior.model",
                smiles=smiles,
                num_epochs=10,
            )

        assert "error" not in result
        assert "job_id" in result
        assert result["job_type"] == "transfer_learning"
        assert result["num_epochs"] == 10
        assert result["n_training_smiles"] == 3

        job_dir = Path(result["working_dir"])
        assert (job_dir / "config.toml").exists()
        assert (job_dir / "train.smi").exists()

        smi_content = (job_dir / "train.smi").read_text()
        assert "CCO" in smi_content
        assert "c1ccccc1" in smi_content

        toml_content = (job_dir / "config.toml").read_text()
        assert 'run_type = "transfer_learning"' in toml_content
        assert "num_epochs = 10" in toml_content

    def test_create_tl_from_file(self, manager, tmp_path):
        smi_path = tmp_path / "my_smiles.smi"
        smi_path.write_text("CCO\nc1ccccc1\n")

        with patch.object(manager, "_start_process"):
            result = manager.create_and_start_tl(
                model="/model/prior.model",
                smiles_file=str(smi_path),
            )

        assert "error" not in result
        toml_content = Path(result["working_dir"]).joinpath("config.toml").read_text()
        assert str(smi_path) in toml_content

    def test_tl_requires_smiles_input(self, manager):
        result = manager.create_and_start_tl(model="/model/prior.model")
        assert "error" in result
        assert "smiles" in result["error"].lower()

    def test_tl_rejects_both_smiles_inputs(self, manager, tmp_path):
        smi_path = tmp_path / "smiles.smi"
        smi_path.write_text("CCO\n")

        result = manager.create_and_start_tl(
            model="/model/prior.model",
            smiles=["CCO"],
            smiles_file=str(smi_path),
        )
        assert "error" in result

    def test_tl_with_validation_smiles(self, manager, tmp_path):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start_tl(
                model="/model/prior.model",
                smiles=["CCO", "CCC"],
                validation_smiles=["CCCO"],
            )

        assert "error" not in result
        job_dir = Path(result["working_dir"])
        assert (job_dir / "validation.smi").exists()
        toml_content = (job_dir / "config.toml").read_text()
        assert "validation_smiles_file" in toml_content

    def test_tl_max_concurrent_enforced(self, manager):
        with patch.object(manager, "_start_process"):
            for i in range(manager._config.jobs.max_concurrent_jobs):
                result = manager.create_and_start_tl(
                    model="/model/prior.model",
                    smiles=["CCO"],
                )
                assert "error" not in result
                manager._jobs[result["job_id"]].state = JobState.RUNNING

            result = manager.create_and_start_tl(
                model="/model/prior.model",
                smiles=["CCO"],
            )
            assert "error" in result

    def test_tl_job_to_dict(self, manager, tmp_path):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start_tl(
                model="/model/prior.model",
                smiles=["CCO"],
                output_model_file="my_model.model",
            )

        job = manager._jobs[result["job_id"]]
        d = job.to_dict()
        assert d["job_type"] == "transfer_learning"
        assert d["num_epochs"] == 50
        assert d["output_model_file"] == "my_model.model"
        assert "max_steps" not in d
        assert "csv_path" not in d

    def test_tl_job_metadata_persistence(self, config, tmp_path):
        """TL job metadata is persisted and restored correctly."""
        job_dir = Path(config.jobs.working_dir) / "job_tl01"
        job_dir.mkdir(parents=True)
        metadata = {
            "job_id": "tl01",
            "ttp_id": "",
            "ttp_version": 0,
            "model": ".reinvent",
            "job_type": "transfer_learning",
            "state": "completed",
            "working_dir": str(job_dir),
            "num_epochs": 30,
            "output_model_file": "tl_model.model",
        }
        (job_dir / "job_metadata.json").write_text(json.dumps(metadata))

        new_manager = JobManager(config)
        jobs = new_manager.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "tl01"
        assert jobs[0]["job_type"] == "transfer_learning"
        assert jobs[0]["num_epochs"] == 30


# ---------------------------------------------------------------------------
# Tools: run_transfer_learning
# ---------------------------------------------------------------------------


class TestRunTransferLearningTool:

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_run_with_smiles(self, mock_mgr):
        from reinvent4_mcp.tools.optimization import run_transfer_learning

        mock_mgr.return_value.create_and_start_tl.return_value = {
            "job_id": "tl01",
            "job_type": "transfer_learning",
            "state": "running",
            "output_model_file": "/jobs/tl01/tl_model.model",
        }

        result = run_transfer_learning(
            smiles=["CCO", "CCC"],
            num_epochs=20,
        )

        assert result["job_id"] == "tl01"
        assert result["job_type"] == "transfer_learning"
        mock_mgr.return_value.create_and_start_tl.assert_called_once()

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_run_with_smiles_file(self, mock_mgr):
        from reinvent4_mcp.tools.optimization import run_transfer_learning

        mock_mgr.return_value.create_and_start_tl.return_value = {
            "job_id": "tl02",
            "job_type": "transfer_learning",
            "state": "running",
        }

        result = run_transfer_learning(
            smiles_file="/data/train.smi",
        )

        assert result["job_id"] == "tl02"
        call_kwargs = mock_mgr.return_value.create_and_start_tl.call_args.kwargs
        assert call_kwargs["smiles_file"] == "/data/train.smi"
        assert call_kwargs["smiles"] is None

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    def test_no_input_returns_error(self, mock_mgr):
        from reinvent4_mcp.tools.optimization import run_transfer_learning

        mock_mgr.return_value.create_and_start_tl.return_value = {
            "error": "Provide either 'smiles' (list) or 'smiles_file' (path)."
        }

        result = run_transfer_learning()
        assert "error" in result


# ---------------------------------------------------------------------------
# get_top_candidates: CSV with component breakdowns
# ---------------------------------------------------------------------------


class TestGetTopCandidatesComponentBreakdown:

    def test_csv_with_components(self, tmp_path):
        from reinvent4_mcp.tools.results import get_top_candidates

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"],
            "SMILES_state": [1, 1, 1],
            "total_score": [0.9, 0.7, 0.5],
            "QED": [0.8, 0.6, 0.4],
            "QED (raw)": [0.8, 0.6, 0.4],
            "MW": [0.95, 0.85, 0.7],
            "MW (raw)": [46.07, 78.11, 60.05],
        })
        df.to_csv(csv_path, index=False)

        result = get_top_candidates(csv_path=str(csv_path), top_n=2)

        assert result["n_total"] == 2
        mol = result["candidates"][0]
        assert mol["smiles"] == "CCO"
        assert "components" in mol
        assert "QED" in mol["components"]
        assert mol["components"]["QED"]["transformed"] == 0.8
        assert mol["components"]["QED"]["raw"] == 0.8
        assert "MW" in mol["components"]
        assert mol["components"]["MW"]["raw"] == pytest.approx(46.07, abs=0.01)

    def test_csv_without_raw_columns(self, tmp_path):
        from reinvent4_mcp.tools.results import get_top_candidates

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame({
            "SMILES": ["CCO", "CCC"],
            "total_score": [0.9, 0.7],
            "QED": [0.8, 0.6],
        })
        df.to_csv(csv_path, index=False)

        result = get_top_candidates(csv_path=str(csv_path))

        mol = result["candidates"][0]
        assert "components" in mol
        assert "QED" in mol["components"]
        assert mol["components"]["QED"]["transformed"] == 0.8
        assert "raw" not in mol["components"]["QED"]


# ---------------------------------------------------------------------------
# Mol2Mol Seed SMILES Support
# ---------------------------------------------------------------------------


class TestSmilesFileInTOML:
    """TOML generation: smiles_file emitted when provided, absent when not."""

    def test_smiles_file_emitted_when_provided(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/m2m.model",
            smiles_file="/data/jobs/job_abc/seed.smi",
        )
        assert 'smiles_file = "/data/jobs/job_abc/seed.smi"' in toml
        assert "[parameters]" in toml

    def test_smiles_file_absent_when_not_provided(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
        )
        assert "smiles_file" not in toml

    def test_smiles_file_absent_when_none(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/prior.model",
            smiles_file=None,
        )
        assert "smiles_file" not in toml

    def test_smiles_file_in_parameters_section(self):
        toml = generate_reinvent_toml(
            scorer_config=MINIMAL_SCORER_CONFIG,
            prior_file="/model/m2m.model",
            smiles_file="/data/seed.smi",
        )
        lines = toml.splitlines()
        in_parameters = False
        found = False
        for line in lines:
            if line.strip() == "[parameters]":
                in_parameters = True
            elif line.strip().startswith("[") and in_parameters:
                break
            elif in_parameters and line.strip().startswith("smiles_file"):
                found = True
                break
        assert found, "smiles_file should appear inside [parameters] section"


class TestSeedSmilesInManager:
    """Job manager: seed SMILES written to seed.smi, path forwarded to TOML."""

    def test_seed_smiles_written_to_file(self, manager):
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/m2m.model",
                seed_smiles=smiles,
            )

        assert "error" not in result
        job_dir = Path(result["working_dir"])
        seed_path = job_dir / "seed.smi"
        assert seed_path.exists()

        content = seed_path.read_text()
        for smi in smiles:
            assert smi in content

    def test_seed_smiles_path_in_toml(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/m2m.model",
                seed_smiles=["CCO", "CCC"],
            )

        toml_content = Path(result["working_dir"]).joinpath("config.toml").read_text()
        assert "smiles_file" in toml_content
        assert "seed.smi" in toml_content

    def test_no_seed_smiles_no_file(self, manager):
        with patch.object(manager, "_start_process"):
            result = manager.create_and_start(
                ttp_id="test-ttp",
                ttp_version=1,
                scorer_config=MINIMAL_SCORER_CONFIG,
                model="/model/prior.model",
            )

        job_dir = Path(result["working_dir"])
        assert not (job_dir / "seed.smi").exists()

        toml_content = (job_dir / "config.toml").read_text()
        assert "smiles_file" not in toml_content


class TestMol2MolValidation:
    """Tool level: seed_smiles forwarded; validation that Mol2Mol requires it."""

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_mol2mol_without_seed_smiles_errors(
        self, mock_resolve, mock_summary, mock_scorer, mock_mgr,
    ):
        from reinvent4_mcp.tools.optimization import run_optimization

        result = run_optimization(
            ttp_id="test-ttp",
            model=".mol2mol_medium_similarity",
        )
        assert "error" in result
        assert "seed_smiles" in result["error"]
        mock_resolve.assert_not_called()

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_m2m_key_detected(
        self, mock_resolve, mock_summary, mock_scorer, mock_mgr,
    ):
        from reinvent4_mcp.tools.optimization import run_optimization

        result = run_optimization(ttp_id="test-ttp", model=".m2m_medium")
        assert "error" in result
        assert "seed_smiles" in result["error"]

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_mol2mol_with_seed_smiles_succeeds(
        self, mock_resolve, mock_summary, mock_scorer, mock_mgr,
    ):
        from reinvent4_mcp.tools.optimization import run_optimization
        from reinvent4_mcp.ttp.schema import TTP

        ttp = TTP.model_validate(MINIMAL_TTP)
        mock_resolve.return_value = ttp
        mock_scorer.return_value = {"type": "geometric_mean", "component": [{"QED": {}}]}
        mock_summary.return_value = {"n_total_components": 1}
        mock_mgr.return_value.create_and_start.return_value = {
            "job_id": "m2m01",
            "state": "running",
        }

        result = run_optimization(
            ttp_id="test-opt-ttp",
            model=".mol2mol_medium_similarity",
            seed_smiles=["CCO", "c1ccccc1"],
        )

        assert "error" not in result
        assert result["job_id"] == "m2m01"
        call_kwargs = mock_mgr.return_value.create_and_start.call_args.kwargs
        assert call_kwargs["seed_smiles"] == ["CCO", "c1ccccc1"]

    @patch("reinvent4_mcp.jobs.manager.get_job_manager")
    @patch("reinvent4_mcp.ttp.mapper.ttp_to_scorer_config")
    @patch("reinvent4_mcp.ttp.mapper.get_mapper_summary")
    @patch("reinvent4_mcp.tools.ttp.resolve_ttp")
    def test_reinvent_with_seed_smiles_warns_and_ignores(
        self, mock_resolve, mock_summary, mock_scorer, mock_mgr,
    ):
        from reinvent4_mcp.tools.optimization import run_optimization
        from reinvent4_mcp.ttp.schema import TTP

        ttp = TTP.model_validate(MINIMAL_TTP)
        mock_resolve.return_value = ttp
        mock_scorer.return_value = {"type": "geometric_mean", "component": [{"QED": {}}]}
        mock_summary.return_value = {"n_total_components": 1}
        mock_mgr.return_value.create_and_start.return_value = {
            "job_id": "abc",
            "state": "running",
        }

        result = run_optimization(
            ttp_id="test-opt-ttp",
            model=".reinvent",
            seed_smiles=["CCO"],
        )

        assert "error" not in result
        call_kwargs = mock_mgr.return_value.create_and_start.call_args.kwargs
        assert call_kwargs["seed_smiles"] is None
