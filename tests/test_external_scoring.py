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

import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# The REINVENT4 plugin namespace + leaf stubs are set up in conftest.py so this
# imports the SAME ExternalRESTScorer that runs in production (the module under
# reinvent4_mcp/_plugins/reinvent_plugins/components/), not a divergent copy.
from reinvent_plugins.components import comp_external_rest
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
        assert scorer.async_poll is False

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
        assert result.metadata == {"confidence_score": [0.85, 0.62]}
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
    def test_the_metadata_schema_is_fixed_by_the_first_batch(self, mock_post):
        """REINVENT writes the results CSV header once and appends rows.

        The metadata keys are unioned across whichever molecules are in a
        batch, so one carrying a `skip_reason` brings keys the others do not.
        A later batch with extra keys makes rows wider than the header and the
        whole file stops being parseable -- seen live as "Expected 24 fields
        in line 18, saw 26", which cost the run its trajectory.
        """
        scorer = ExternalRESTScorer(_make_params())

        mock_post.return_value = _mock_response(
            {"scores": [0.5], "metadata": {"confidence": [0.9]}}
        )
        first = scorer(["CCO"])

        mock_post.return_value = _mock_response(
            {"scores": [0.6], "metadata": {"confidence": [0.8], "skip_reason": ["x"]}}
        )
        second = scorer(["CCC"])

        assert list(first.metadata) == ["confidence"]
        assert list(second.metadata) == ["confidence"], (
            "a later batch must not widen the CSV"
        )

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_a_missing_key_is_padded_rather_than_dropped(self, mock_post):
        """A narrower batch would otherwise make rows *narrower* than the header."""
        scorer = ExternalRESTScorer(_make_params())

        mock_post.return_value = _mock_response(
            {"scores": [0.5], "metadata": {"confidence": [0.9], "note": ["hi"]}}
        )
        scorer(["CCO"])

        mock_post.return_value = _mock_response(
            {"scores": [0.6], "metadata": {"confidence": [0.8]}}
        )
        second = scorer(["CCC"])

        assert list(second.metadata) == ["confidence", "note"]
        assert second.metadata["note"] == [None]

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_a_batch_with_no_metadata_still_fills_the_columns(self, mock_post):
        scorer = ExternalRESTScorer(_make_params())

        mock_post.return_value = _mock_response(
            {"scores": [0.5], "metadata": {"confidence": [0.9]}}
        )
        scorer(["CCO"])

        mock_post.return_value = _mock_response({"scores": [0.6, 0.7]})
        second = scorer(["CCC", "CCCC"])

        assert second.metadata == {"confidence": [None, None]}

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_no_metadata_at_all_stays_none(self, mock_post):
        """Nothing to stabilise, so no empty columns are invented."""
        scorer = ExternalRESTScorer(_make_params())
        mock_post.return_value = _mock_response({"scores": [0.5]})

        assert scorer(["CCO"]).metadata is None

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_stop_request_is_not_swallowed(self, mock_post):
        """A stop request has to reach REINVENT's checkpoint-writing handler.

        REINVENT asks a running stage to stop by raising StageInterrupted*
        from a signal handler, so it surfaces from inside this component's
        socket read. Those are plain Exceptions: if the failure handler here
        catches one, the stop becomes a batch of NaN scores, the RL loop runs
        on to the next step, and the job is eventually killed with no
        checkpoint to resume from.
        """
        class StageInterruptedUncontrolled(Exception):
            pass

        mock_post.side_effect = StageInterruptedUncontrolled("Signal 15")
        scorer = ExternalRESTScorer(_make_params())

        with patch.object(
            comp_external_rest,
            "STOP_REQUEST_EXCEPTIONS",
            (StageInterruptedUncontrolled,),
        ):
            with pytest.raises(StageInterruptedUncontrolled):
                scorer(["CCO", "c1ccccc1"])

    def test_stop_request_exceptions_match_reinvents(self):
        """The names resolved at import time are the ones REINVENT raises."""
        source = (
            Path(__file__).resolve().parents[1]
            / "REINVENT4" / "reinvent" / "runmodes" / "handler.py"
        )
        if not source.exists():
            pytest.skip("REINVENT4 source not checked out")

        raised = set(re.findall(r"raise (StageInterrupted\w+)", source.read_text()))
        assert raised, "REINVENT's handler no longer raises StageInterrupted*"

        declared = set(
            re.findall(
                r"(StageInterrupted\w+)",
                inspect.getsource(comp_external_rest).split("_CALL_LOG_DIR")[0],
            )
        )
        assert raised <= declared

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
# TestTimeoutConfiguration
# ---------------------------------------------------------------------------


class TestTimeoutConfiguration:
    """A deep queue must stop being indistinguishable from a failure.

    The timeout used to be hardcoded at 1800 s, the same ceiling the Boltz
    service itself enforces, so a batch that queued behind another one became
    NaN for every molecule and tripped a component-failure diagnostic that was
    not true.
    """

    def test_the_default_is_unchanged(self):
        # Nothing silently changes for existing configs.
        assert ExternalRESTScorer(_make_params()).timeout == 1800

    def test_an_explicit_timeout_wins(self):
        scorer = ExternalRESTScorer(_make_params(timeout=[7200.0]))

        assert scorer.timeout == 7200.0

    def test_the_environment_sets_it_when_the_config_does_not(self, monkeypatch):
        monkeypatch.setenv("EXTERNAL_SCORER_TIMEOUT_SECONDS", "5400")

        assert ExternalRESTScorer(_make_params()).timeout == 5400.0

    def test_the_config_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("EXTERNAL_SCORER_TIMEOUT_SECONDS", "5400")

        assert ExternalRESTScorer(_make_params(timeout=[600.0])).timeout == 600.0

    @pytest.mark.parametrize("bad", ["not-a-number", "0", "-1", ""])
    def test_a_nonsense_environment_value_falls_back(self, monkeypatch, bad):
        # A typo must not disable the timeout or make it instant.
        monkeypatch.setenv("EXTERNAL_SCORER_TIMEOUT_SECONDS", bad)

        assert ExternalRESTScorer(_make_params()).timeout == 1800

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_the_configured_timeout_is_the_one_used(self, mock_post):
        mock_post.return_value = _mock_response({"scores": [0.5]})

        ExternalRESTScorer(_make_params(timeout=[3600.0]))(["CCO"])

        assert mock_post.call_args.kwargs["timeout"] == 3600.0

    def test_timeout_is_not_forwarded_to_the_service(self):
        # It is ours, not a prediction parameter.
        scorer = ExternalRESTScorer(_make_params(timeout=[600.0], async_poll=[True]))

        assert "timeout" not in scorer.request_params
        assert "async_poll" not in scorer.request_params


# ---------------------------------------------------------------------------
# TestSubmitThenPoll
# ---------------------------------------------------------------------------


class TestSubmitThenPoll:
    """202 + poll, so waiting is bounded by the deadline, not by one socket."""

    def _scorer(self, **kwargs) -> ExternalRESTScorer:
        params = _make_params(
            service_url=["http://gateway:8091/score"], async_poll=[True], **kwargs
        )
        scorer = ExternalRESTScorer(params)
        return scorer

    def test_result_url_derived_from_the_score_url(self):
        scorer = self._scorer()

        assert scorer._result_url("abc") == "http://gateway:8091/result/abc"

    def test_result_url_ignores_an_existing_query(self):
        scorer = ExternalRESTScorer(
            _make_params(service_url=["http://gateway:8091/score?x=1"],
                         async_poll=[True])
        )

        assert scorer._result_url("abc") == "http://gateway:8091/result/abc"

    @patch("reinvent_plugins.components.comp_external_rest.time.sleep")
    @patch("reinvent_plugins.components.comp_external_rest.requests.get")
    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_it_submits_async_then_polls_until_ready(self, mock_post, mock_get, _sleep):
        mock_post.return_value = _mock_response({"task_id": "t1"}, status_code=202)
        mock_get.side_effect = [
            _mock_response({"status": "pending"}, status_code=202),
            _mock_response({"status": "pending"}, status_code=202),
            _mock_response({"scores": [1.5]}),
        ]

        result = self._scorer()(["CCO"])

        # mode=async on the submit, then the result URL until it is ready.
        assert mock_post.call_args.args[0] == "http://gateway:8091/score?mode=async"
        assert mock_get.call_args.args[0] == "http://gateway:8091/result/t1"
        assert mock_get.call_count == 3
        assert result.scores[0][0] == pytest.approx(1.5)

    @patch("reinvent_plugins.components.comp_external_rest.time.sleep")
    @patch("reinvent_plugins.components.comp_external_rest.requests.get")
    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_polling_backs_off(self, mock_post, mock_get, mock_sleep):
        mock_post.return_value = _mock_response({"task_id": "t1"}, status_code=202)
        mock_get.side_effect = [
            _mock_response({}, status_code=202),
            _mock_response({}, status_code=202),
            _mock_response({"scores": [1.0]}),
        ]

        self._scorer()(["CCO"])

        waits = [call.args[0] for call in mock_sleep.call_args_list]
        # Cheap while queued: the interval grows instead of hammering.
        assert waits[1] > waits[0]

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_a_service_that_does_not_do_async_is_used_as_is(self, mock_post):
        # Answering the submit normally rather than 202 means the flag is
        # misconfigured; the response is still honoured so it is visible.
        mock_post.return_value = _mock_response({"scores": [0.25]})

        result = self._scorer()(["CCO"])

        assert result.scores[0][0] == pytest.approx(0.25)

    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_202_without_a_task_id_is_a_failure(self, mock_post):
        mock_post.return_value = _mock_response({}, status_code=202)

        result = self._scorer(failure_score=[0.0])(["CCO"])

        assert result.scores[0][0] == pytest.approx(0.0)

    @patch("reinvent_plugins.components.comp_external_rest.time.sleep")
    @patch("reinvent_plugins.components.comp_external_rest.requests.get")
    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_the_deadline_still_applies(self, mock_post, mock_get, _sleep):
        # Polling forever would hang an RL step; the overall timeout bounds it.
        mock_post.return_value = _mock_response({"task_id": "t1"}, status_code=202)
        mock_get.return_value = _mock_response({}, status_code=202)

        result = self._scorer(timeout=[0.01], failure_score=[0.0])(["CCO"])

        assert result.scores[0][0] == pytest.approx(0.0)

    @patch("reinvent_plugins.components.comp_external_rest.time.sleep")
    @patch("reinvent_plugins.components.comp_external_rest.requests.get")
    @patch("reinvent_plugins.components.comp_external_rest.requests.post")
    def test_the_poll_count_is_logged(self, mock_post, mock_get, _sleep):
        mock_post.return_value = _mock_response({"task_id": "t1"}, status_code=202)
        mock_get.side_effect = [
            _mock_response({}, status_code=202),
            _mock_response({"scores": [1.0]}),
        ]
        records: list[dict] = []
        with patch(
            "reinvent_plugins.components.comp_external_rest._log_call",
            side_effect=records.append,
        ):
            self._scorer()(["CCO"])

        # So a slow step can be explained after the fact.
        assert records[0]["task_id"] == "t1"
        assert records[0]["polls"] == 2


# ---------------------------------------------------------------------------
# TestListExternalServices
# ---------------------------------------------------------------------------


class TestListSharedAssets:
    """What the volume holds, in a form that fits a context window.

    One cofold output lands per predicted molecule, so this listing grows with
    every batch. At 1763 of them it came to 138k tokens -- more than the
    conductor model's entire 131072-token context -- and the call failed
    outright with "max_tokens must be at least 1, got -31572".
    """

    def _volume(self, tmp_path, n_predictions=0, n_structures=0):
        for i in range(n_structures):
            (tmp_path / f"ref_{i}.pdb").write_text("ATOM\n")
        for i in range(n_predictions):
            out = tmp_path / f"predictions/out_{i:04d}/predictions/mol_0"
            out.mkdir(parents=True)
            (out / "mol_0_model_0.pdb").write_text("ATOM\n")
        return tmp_path

    def _assets(self, tmp_path, **kwargs):
        from reinvent4_mcp.tools.external import list_shared_assets

        with patch.dict(os.environ, {"IP_SHARED_DIR": str(tmp_path)}):
            return list_shared_assets(**kwargs)

    def test_a_prediction_is_not_listed_as_a_structure_as_well(self, tmp_path):
        """``structures`` and ``cofold_outputs`` came back byte-identical.

        Every Boltz output was appended to both, so the one useful listing was
        paid for twice.
        """
        self._volume(tmp_path, n_predictions=3, n_structures=1)

        assets = self._assets(tmp_path)

        assert assets["counts"] == {
            "structures": 1, "msas": 0, "cofold_outputs": 3
        }
        assert assets["structures"] == ["ref_0.pdb"]
        assert all("predictions" in p for p in assets["cofold_outputs"])

    def test_the_listing_is_bounded_however_many_files_there_are(self, tmp_path):
        self._volume(tmp_path, n_predictions=50)

        assets = self._assets(tmp_path, limit=10)

        assert len(assets["cofold_outputs"]) == 10

    def test_the_true_total_is_reported_so_a_cut_is_visible(self, tmp_path):
        """A sample that hides its own incompleteness is worse than a count."""
        self._volume(tmp_path, n_predictions=50)

        assets = self._assets(tmp_path, limit=10)

        assert assets["counts"]["cofold_outputs"] == 50
        assert assets["not_listed"] == {"cofold_outputs": 40}
        assert "prefix" in assets["more"]

    def test_nothing_is_withheld_when_everything_fits(self, tmp_path):
        self._volume(tmp_path, n_predictions=3)

        assets = self._assets(tmp_path, limit=10)

        assert "not_listed" not in assets
        assert "more" not in assets

    def test_cofold_outputs_are_newest_first(self, tmp_path):
        """The output worth naming is the one the last job just wrote."""
        self._volume(tmp_path, n_predictions=3)
        paths = sorted(tmp_path.rglob("*.pdb"))
        for i, path in enumerate(paths):
            os.utime(path, (0, 1000 + i * 100))

        assets = self._assets(tmp_path, limit=10)

        assert assets["cofold_outputs"][0] == str(
            paths[-1].relative_to(tmp_path)
        )

    def test_prefix_reaches_one_jobs_outputs(self, tmp_path):
        self._volume(tmp_path, n_predictions=5)

        assets = self._assets(tmp_path, prefix="predictions/out_0003")

        assert assets["counts"]["cofold_outputs"] == 1
        assert assets["cofold_outputs"][0].startswith("predictions/out_0003")
        assert assets["prefix"] == "predictions/out_0003"

    def test_an_msa_still_reports_its_query_length(self, tmp_path):
        """What pairs an MSA with the right target sequence."""
        (tmp_path / "brd4_bd1.a3m").write_text(">query\nMSAES\n>hit\nMSAES\n")

        assets = self._assets(tmp_path)

        assert assets["msas"] == ["brd4_bd1.a3m"]
        assert assets["msa_details"][0]["query_length"] == 5

    def test_an_empty_volume_is_not_an_error(self, tmp_path):
        assets = self._assets(tmp_path)

        assert assets["counts"] == {
            "structures": 0, "msas": 0, "cofold_outputs": 0
        }
        assert "No structures" in assets["note"]

    def test_a_prefix_that_matches_nothing_says_so(self, tmp_path):
        self._volume(tmp_path, n_predictions=2)

        assets = self._assets(tmp_path, prefix="predictions/out_9999")

        assert assets["cofold_outputs"] == []
        assert "out_9999" in assets["note"]

    def test_a_missing_volume_is_reported_not_raised(self, tmp_path):
        assets = self._assets(tmp_path / "absent")

        assert "does not exist" in assets["note"]


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
