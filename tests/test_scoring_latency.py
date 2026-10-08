"""Tests for the external_scoring.jsonl latency/cache harvest script.

Covers:
- Record parsing (batch size, status/error classification, cache-key digest)
- Exclusion of failures and capped timeouts from the latency distribution
- Latency statistics, including the per-batch-size breakdown
- Cache hit rate on the loose (sequence+SMILES) and full cache keys
- Fleet sizing derived from t_boltz_per_batch
- CLI behaviour, including the no-usable-records fallback
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_script = (
    Path(__file__).resolve().parent.parent
    / "k8s" / "scripts" / "analyze-scoring-latency.py"
)
_spec = importlib.util.spec_from_file_location("analyze_scoring_latency", str(_script))
harvest = importlib.util.module_from_spec(_spec)
sys.modules["analyze_scoring_latency"] = harvest
_spec.loader.exec_module(harvest)

SEQ = "NPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIK"
PREDICT_TIMEOUT = harvest.PREDICT_TIMEOUT_SECONDS


def record(
    service: str = "boltz",
    smiles: list[str] | None = None,
    elapsed: float | None = 600.0,
    status_code: int | None = 200,
    error: str | None = None,
    sequence: str = SEQ,
    timestamp: str = "2026-09-01T12:00:00+00:00",
    **params,
) -> dict:
    """Build one record in the shape ExternalRESTScorer._log_call writes."""
    payload_params = {"protein_sequence": sequence, **params}
    rec: dict = {
        "timestamp": timestamp,
        "service": service,
        "url": "http://127.0.0.1:8091/score",
        "request": {
            "smiles": smiles if smiles is not None else ["CCO"] * 16,
            "params": payload_params,
        },
    }
    if status_code is not None:
        rec["status_code"] = status_code
    if elapsed is not None:
        rec["elapsed_seconds"] = elapsed
    if error is not None:
        rec["error"] = error
    return rec


def write_log(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


# ---------------------------------------------------------------------------
# TestRecordParsing
# ---------------------------------------------------------------------------


class TestRecordParsing:
    """parse_record maps a logged call onto the fields the harvest needs."""

    def test_batch_size_recovered_from_request_smiles(self):
        call = harvest.parse_record(record(smiles=["CCO"] * 16))

        assert call.batch_size == 16
        assert call.ok

    def test_2xx_without_error_is_ok(self):
        call = harvest.parse_record(record(status_code=204, elapsed=5.0))

        assert call.ok
        assert call.failure == ""

    def test_http_error_classified_by_code(self):
        call = harvest.parse_record(
            record(status_code=500, elapsed=12.0, error="HTTP 500")
        )

        assert not call.ok
        assert call.failure == "http_500"

    def test_capped_request_failure_classified_as_timeout(self):
        call = harvest.parse_record(
            record(status_code=None, elapsed=PREDICT_TIMEOUT, error="request_failed")
        )

        assert not call.ok
        assert call.failure == "timeout"

    def test_fast_request_failure_is_not_a_timeout(self):
        call = harvest.parse_record(
            record(status_code=None, elapsed=0.3, error="request_failed")
        )

        assert not call.ok
        assert call.failure == "request_failed"

    def test_non_2xx_without_error_key_still_excluded(self):
        call = harvest.parse_record(record(status_code=404, elapsed=1.0))

        assert not call.ok
        assert call.failure == "http_404"

    def test_missing_elapsed_is_not_usable(self):
        call = harvest.parse_record(record(elapsed=None))

        assert not call.ok
        assert call.failure == "no_elapsed_seconds"

    def test_protein_sequence_normalized(self):
        call = harvest.parse_record(record(sequence="  npppp  "))

        assert call.protein_sequence == "NPPPP"

    def test_missing_protein_sequence_tolerated(self):
        raw = record(service="posebusters")
        raw["request"]["params"] = {"mode": "mol"}
        call = harvest.parse_record(raw)

        assert call.protein_sequence == ""
        assert call.ok

    def test_predict_params_change_the_cache_digest(self):
        cheap = harvest.parse_record(record(recycling_steps=1))
        full = harvest.parse_record(record(recycling_steps=3))

        assert cheap.params_digest != full.params_digest

    def test_per_call_artifacts_excluded_from_cache_digest(self):
        bare = harvest.parse_record(record(recycling_steps=3))
        with_structs = harvest.parse_record(
            record(recycling_steps=3, structure_paths=["/shared/a.pdb"])
        )

        assert bare.params_digest == with_structs.params_digest


# ---------------------------------------------------------------------------
# TestFailureExclusion
# ---------------------------------------------------------------------------


class TestFailureExclusion:
    """Capped timeouts must not skew the latency distribution."""

    def test_timeouts_excluded_from_percentiles(self):
        calls = [harvest.parse_record(record(elapsed=600.0)) for _ in range(9)]
        calls += [
            harvest.parse_record(
                record(status_code=None, elapsed=PREDICT_TIMEOUT, error="request_failed")
            )
            for _ in range(9)
        ]
        report = harvest.build_report("boltz", calls)

        assert report.n_ok == 9
        assert report.n_failed == 9
        assert report.batch["p95"] == 600.0

    def test_failures_counted_by_reason(self):
        calls = [
            harvest.parse_record(record(status_code=500, elapsed=1.0, error="HTTP 500")),
            harvest.parse_record(record(status_code=500, elapsed=1.0, error="HTTP 500")),
            harvest.parse_record(
                record(status_code=None, elapsed=PREDICT_TIMEOUT, error="request_failed")
            ),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.failures["http_500"] == 2
        assert report.failures["timeout"] == 1


# ---------------------------------------------------------------------------
# TestLatencyStats
# ---------------------------------------------------------------------------


class TestLatencyStats:
    """Percentiles and the per-batch-size breakdown."""

    def test_percentile_interpolates(self):
        assert harvest.percentile([0.0, 10.0], 0.5) == 5.0
        assert harvest.percentile([1.0, 2.0, 3.0], 0.0) == 1.0
        assert harvest.percentile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_percentile_of_empty_is_none(self):
        assert harvest.percentile([], 0.5) is None

    def test_per_molecule_latency_divides_by_batch_size(self):
        calls = [harvest.parse_record(record(smiles=["CCO"] * 16, elapsed=640.0))]
        report = harvest.build_report("boltz", calls)

        assert report.per_molecule["p50"] == 40.0

    def test_latency_split_by_batch_size(self):
        calls = [harvest.parse_record(record(smiles=["CCO"] * 16, elapsed=600.0))]
        calls += [harvest.parse_record(record(smiles=["CCO"] * 4, elapsed=150.0))]
        report = harvest.build_report("boltz", calls)

        assert report.by_batch_size[16]["p50"] == 600.0
        assert report.by_batch_size[4]["p50"] == 150.0

    def test_modal_batch_size_is_the_most_common(self):
        calls = [harvest.parse_record(record(smiles=["CCO"] * 16)) for _ in range(5)]
        calls += [harvest.parse_record(record(smiles=["CCO"] * 4)) for _ in range(2)]
        report = harvest.build_report("boltz", calls)

        assert report.modal_batch_size == 16

    def test_modal_batch_size_tie_breaks_to_the_larger_batch(self):
        calls = [harvest.parse_record(record(smiles=["CCO"] * 16))]
        calls += [harvest.parse_record(record(smiles=["CCO"] * 4))]
        report = harvest.build_report("boltz", calls)

        assert report.modal_batch_size == 16

    def test_observation_window_reported(self):
        calls = [
            harvest.parse_record(record(timestamp="2026-09-01T00:00:00+00:00")),
            harvest.parse_record(record(timestamp="2026-09-02T00:00:00+00:00")),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.first_seen.day == 1
        assert report.last_seen.day == 2


# ---------------------------------------------------------------------------
# TestCacheHitRate
# ---------------------------------------------------------------------------


class TestCacheHitRate:
    """SMILES seen more than once under one protein sequence are cache hits."""

    def test_no_repeats_means_no_hits(self):
        calls = [
            harvest.parse_record(record(smiles=["CCO", "CCC"])),
            harvest.parse_record(record(smiles=["CCCC", "CCCCC"])),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.molecule_calls == 4
        assert report.unique_molecules == 4
        assert report.cache_hit_rate == 0.0

    def test_repeated_smiles_counted_as_hits(self):
        calls = [harvest.parse_record(record(smiles=["CCO", "CCC"])) for _ in range(3)]
        report = harvest.build_report("boltz", calls)

        assert report.molecule_calls == 6
        assert report.unique_molecules == 2
        assert report.cache_hit_rate == pytest.approx(4 / 6)

    def test_same_smiles_under_another_target_is_a_miss(self):
        calls = [
            harvest.parse_record(record(smiles=["CCO"], sequence="AAAA")),
            harvest.parse_record(record(smiles=["CCO"], sequence="CCCC")),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.cache_hit_rate == 0.0
        assert report.n_sequences == 2

    def test_full_cache_key_separates_fidelity_tiers(self):
        calls = [
            harvest.parse_record(record(smiles=["CCO"], recycling_steps=1)),
            harvest.parse_record(record(smiles=["CCO"], recycling_steps=3)),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.cache_hit_rate == pytest.approx(0.5)
        assert report.cache_hit_rate_strict == 0.0

    def test_failed_calls_still_count_as_demand(self):
        calls = [
            harvest.parse_record(record(smiles=["CCO"])),
            harvest.parse_record(
                record(smiles=["CCO"], status_code=500, error="HTTP 500")
            ),
        ]
        report = harvest.build_report("boltz", calls)

        assert report.molecule_calls == 2
        assert report.cache_hit_rate == pytest.approx(0.5)

    def test_no_smiles_means_no_rate(self):
        raw = record()
        raw["request"]["smiles"] = []
        report = harvest.build_report("boltz", [harvest.parse_record(raw)])

        assert report.cache_hit_rate is None


# ---------------------------------------------------------------------------
# TestFleetSizing
# ---------------------------------------------------------------------------


class TestFleetSizing:
    """Replica counts and campaign lengths derived from t_boltz_per_batch."""

    def _size(self, **overrides):
        kwargs = dict(
            t_batch=600.0,
            cache_hit_rate=0.0,
            supervisors=3,
            step_seconds=300.0,
            campaign_seconds=12 * 3600,
            replicas=5,
            step_counts=[30, 50, 100],
        )
        kwargs.update(overrides)
        return harvest.size_fleet(**kwargs)

    def test_two_replicas_per_supervisor_without_caching(self):
        sizing = self._size()

        # 10 min per batch against a 5 min step is the plan's M ~ 2N.
        assert sizing["replicas_per_supervisor"] == 2.0
        assert sizing["m_required"] == 6
        assert sizing["m_required_no_cache"] == 6

    def test_caching_reduces_required_replicas(self):
        sizing = self._size(cache_hit_rate=0.5)

        assert sizing["m_required"] == 4
        assert sizing["m_required_no_cache"] == 6

    def test_supported_supervisors_from_replica_count(self):
        sizing = self._size(replicas=12)

        assert sizing["n_supported"] == 6

    def test_cache_multiplier_is_the_conservative_one(self):
        sizing = self._size(cache_hit_rate=0.5)

        assert sizing["cache_multiplier_plan"] == 1.5
        assert sizing["cache_multiplier_exact"] == 2.0
        assert sizing["cache_multiplier_exact"] > sizing["cache_multiplier_plan"]

    def test_hundred_step_campaign_hours_match_the_plan(self):
        sizing = self._size()

        # 100 steps at 10 min per batch on a dedicated replica.
        assert sizing["campaign_hours_for_steps"][100]["dedicated"] == pytest.approx(16.7, abs=0.1)
        assert sizing["max_steps_in_campaign_dedicated"] == 72

    def test_slower_batch_shrinks_the_campaign(self):
        sizing = self._size(t_batch=1800.0)

        assert sizing["max_steps_in_campaign_dedicated"] == 24


# ---------------------------------------------------------------------------
# TestWarnings
# ---------------------------------------------------------------------------


class TestWarnings:
    """The harvest flags what would invalidate the plan's assumptions."""

    def test_task_unit_must_shrink_when_window_is_too_small(self):
        reports = {"boltz": harvest.build_report("boltz", [])}
        sizing = harvest.size_fleet(
            t_batch=3600.0, cache_hit_rate=0.0, supervisors=3, step_seconds=300.0,
            campaign_seconds=12 * 3600, replicas=5, step_counts=[30],
        )
        warnings = harvest.collect_warnings(reports, None, sizing, 30)

        assert any("task unit has to shrink" in w for w in warnings)

    def test_slow_cpu_scorer_called_out(self):
        boltz = harvest.build_report(
            "boltz", [harvest.parse_record(record(elapsed=600.0))]
        )
        profiler = harvest.build_report(
            "interaction_profiler",
            [harvest.parse_record(record(service="interaction_profiler", elapsed=300.0))],
        )
        warnings = harvest.collect_warnings(
            {"boltz": boltz, "interaction_profiler": profiler}, boltz, None, 30
        )

        assert any("not negligible" in w for w in warnings)

    def test_fast_cpu_scorer_not_called_out(self):
        boltz = harvest.build_report(
            "boltz", [harvest.parse_record(record(elapsed=600.0))]
        )
        pose = harvest.build_report(
            "posebusters",
            [harvest.parse_record(record(service="posebusters", elapsed=8.0))],
        )
        warnings = harvest.collect_warnings(
            {"boltz": boltz, "posebusters": pose}, boltz, None, 30
        )

        assert not any("not negligible" in w for w in warnings)

    def test_mixed_targets_flagged(self):
        boltz = harvest.build_report("boltz", [
            harvest.parse_record(record(sequence="AAAA")),
            harvest.parse_record(record(sequence="CCCC")),
        ])
        warnings = harvest.collect_warnings({"boltz": boltz}, boltz, None, 30)

        assert any("distinct protein sequences" in w for w in warnings)

    def test_high_failure_rate_flagged(self):
        calls = [harvest.parse_record(record(elapsed=600.0))]
        calls += [
            harvest.parse_record(record(status_code=500, error="HTTP 500"))
            for _ in range(2)
        ]
        boltz = harvest.build_report("boltz", calls)
        warnings = harvest.collect_warnings({"boltz": boltz}, boltz, None, 30)

        assert any("calls failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """End-to-end runs over a log file."""

    def _log(self, tmp_path: Path, n: int = 30) -> Path:
        records = [
            record(smiles=[f"CCO{'C' * (i % 5)}"] * 16, elapsed=600.0 + i)
            for i in range(n)
        ]
        records.append(
            record(status_code=None, elapsed=PREDICT_TIMEOUT, error="request_failed")
        )
        return write_log(tmp_path / "external_scoring.jsonl", records)

    def test_json_report(self, tmp_path, capsys):
        log = self._log(tmp_path)

        assert harvest.main([str(log), "--json"]) == 0

        payload = json.loads(capsys.readouterr().out)
        boltz = payload["services"]["boltz"]
        assert boltz["n_ok"] == 30
        assert boltz["n_failed"] == 1
        assert boltz["modal_batch_size"] == 16
        assert payload["sizing"]["batch_size"] == 16
        assert payload["sizing"]["t_batch_seconds"] == pytest.approx(614.5)

    def test_text_report_names_the_harvested_numbers(self, tmp_path, capsys):
        log = self._log(tmp_path)

        assert harvest.main([str(log)]) == 0

        out = capsys.readouterr().out
        assert "t_boltz_per_batch" in out
        assert "cache_hit_rate" in out
        assert "boltz replicas" in out

    def test_directory_argument_finds_the_log(self, tmp_path, capsys):
        self._log(tmp_path)

        assert harvest.main([str(tmp_path), "--json"]) == 0

    def test_unparseable_lines_counted_not_fatal(self, tmp_path, capsys):
        log = self._log(tmp_path)
        with open(log, "a") as handle:
            handle.write("not json at all\n")

        assert harvest.main([str(log), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["unparseable_lines"] == 1

    def test_no_usable_records_returns_fallback(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [])

        assert harvest.main([str(log)]) == 2
        assert "Fall back to one measured /score call" in capsys.readouterr().out

    def test_only_failed_boltz_calls_returns_fallback(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [
            record(status_code=None, elapsed=PREDICT_TIMEOUT, error="request_failed"),
        ])

        assert harvest.main([str(log)]) == 2

    def test_service_filter(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [
            record(service="boltz", elapsed=600.0),
            record(service="posebusters", elapsed=8.0),
        ])

        assert harvest.main([str(log), "--json", "--service", "boltz"]) == 0

        payload = json.loads(capsys.readouterr().out)
        assert list(payload["services"]) == ["boltz"]

    def test_since_filter_drops_older_records(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [
            record(timestamp="2026-08-01T00:00:00+00:00"),
            record(timestamp="2026-09-01T00:00:00+00:00"),
        ])

        assert harvest.main(
            [str(log), "--json", "--since", "2026-08-15T00:00:00+00:00"]
        ) == 0
        assert json.loads(capsys.readouterr().out)["records"] == 1

    def test_missing_file_is_an_error_not_a_crash(self, tmp_path, capsys):
        assert harvest.main([str(tmp_path / "absent.jsonl")]) == 1
        assert "could not read the scoring log" in capsys.readouterr().err

    def test_batch_size_override(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [
            *[record(smiles=["CCO"] * 16, elapsed=600.0) for _ in range(5)],
            record(smiles=["CCO"] * 4, elapsed=150.0),
        ])

        assert harvest.main([str(log), "--json", "--batch-size", "4"]) == 0

        sizing = json.loads(capsys.readouterr().out)["sizing"]
        assert sizing["batch_size"] == 4
        assert sizing["t_batch_seconds"] == 150.0

    def test_unseen_batch_size_falls_back_to_all_calls(self, tmp_path, capsys):
        log = write_log(tmp_path / "external_scoring.jsonl", [
            record(smiles=["CCO"] * 16, elapsed=600.0),
        ])

        assert harvest.main([str(log), "--json", "--batch-size", "64"]) == 0

        sizing = json.loads(capsys.readouterr().out)["sizing"]
        assert sizing["batch_size"] is None
        assert sizing["t_batch_seconds"] == 600.0
