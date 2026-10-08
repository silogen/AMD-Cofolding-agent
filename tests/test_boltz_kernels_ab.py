"""Tests for the Boltz optimized-kernels A/B harness.

Covers:
- The 16-molecule BRD4 BD1 panel, checked against the wrapper's own validator
  so neither arm can silently score a smaller batch
- Request construction (precomputed MSA, no MSA-server round trip)
- Replica arithmetic and /health URL derivation
- Reading and restoring BOLTZ_NO_KERNELS on the Deployment
- CLI guards that stop a meaningless run

No cluster, service, or GPU is needed: kubectl is stubbed and no arm is run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_ab_spec = importlib.util.spec_from_file_location(
    "ab_boltz_kernels", str(_ROOT / "k8s" / "scripts" / "ab-boltz-kernels.py")
)
ab = importlib.util.module_from_spec(_ab_spec)
sys.modules["ab_boltz_kernels"] = ab
_ab_spec.loader.exec_module(ab)

_boltz_spec = importlib.util.spec_from_file_location(
    "boltz2_app_for_ab", str(_ROOT / "services" / "boltz2" / "app.py")
)
boltz_app = importlib.util.module_from_spec(_boltz_spec)
sys.modules["boltz2_app_for_ab"] = boltz_app
_boltz_spec.loader.exec_module(boltz_app)


# ---------------------------------------------------------------------------
# TestPanel
# ---------------------------------------------------------------------------


class TestPanel:
    """The A/B batch must survive the wrapper's validation intact."""

    def test_sixteen_molecules(self):
        assert len(ab.PANEL) == 16

    def test_names_and_smiles_unique(self):
        assert len({name for name, _ in ab.PANEL}) == 16
        assert len({smi for _, smi in ab.PANEL}) == 16

    def test_every_molecule_passes_wrapper_validation(self):
        for name, smi in ab.PANEL:
            ok, reason = boltz_app._validate_smiles(smi)
            assert ok, f"{name}: {reason}"

    def test_nothing_is_skipped_by_the_partitioner(self):
        smiles = [smi for _, smi in ab.PANEL]
        valid_indices, reasons = boltz_app._partition_smiles(smiles)

        assert len(valid_indices) == 16
        assert not [r for r in reasons if r]

    def test_sequence_is_foldable_by_the_wrapper(self):
        request = boltz_app.ScoreRequest(
            smiles=["CCO"],
            params=boltz_app.ScoreParams(
                protein_sequence=ab.BRD4_BD1_SEQUENCE,
                msa_path=ab.DEFAULT_MSA_PATH,
            ),
        )

        assert boltz_app._validate_protein(request) == []

    def test_sequence_matches_the_precomputed_msa(self):
        a3m = _ROOT / "data" / "shared" / ab.DEFAULT_MSA_PATH
        if not a3m.is_file():
            pytest.skip(f"{a3m} is gitignored and absent here")

        query = boltz_app._read_a3m_query_sequence(str(a3m))

        # A mismatch would make the service silently fold the MSA's sequence
        # instead, so both arms would measure something else.
        assert query == ab.BRD4_BD1_SEQUENCE


# ---------------------------------------------------------------------------
# TestPayload
# ---------------------------------------------------------------------------


class TestPayload:
    """Both arms must issue the same production-shaped request."""

    def _args(self, **overrides):
        defaults = dict(
            protein_sequence=ab.BRD4_BD1_SEQUENCE,
            msa_path=ab.DEFAULT_MSA_PATH,
            recycling_steps=3,
            diffusion_samples=1,
            sampling_steps=200,
            sampling_steps_affinity=200,
            diffusion_samples_affinity=5,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_uses_the_precomputed_msa_not_the_server(self):
        payload = ab.build_payload(self._args(), ["CCO"])

        assert payload["params"]["msa_path"] == ab.DEFAULT_MSA_PATH
        assert "use_msa_server" not in payload["params"]

    def test_carries_production_affinity_settings(self):
        payload = ab.build_payload(self._args(), ["CCO"])

        assert payload["params"]["recycling_steps"] == 3
        assert payload["params"]["sampling_steps_affinity"] == 200
        assert payload["params"]["diffusion_samples_affinity"] == 5

    def test_smiles_passed_through_in_order(self):
        smiles = [smi for _, smi in ab.PANEL]
        payload = ab.build_payload(self._args(), smiles)

        assert payload["smiles"] == smiles

    def test_request_is_accepted_by_the_score_model(self):
        smiles = [smi for _, smi in ab.PANEL]
        payload = ab.build_payload(self._args(), smiles)

        request = boltz_app.ScoreRequest(**payload)

        assert len(request.smiles) == 16
        assert request.params.msa_path == ab.DEFAULT_MSA_PATH
        assert request.params.use_msa_server is False


# ---------------------------------------------------------------------------
# TestUrlsAndMath
# ---------------------------------------------------------------------------


class TestUrlsAndMath:

    def test_health_url_derived_from_score_url(self):
        assert ab.health_url_for("http://boltz:8091/score") == "http://boltz:8091/health"

    def test_health_url_left_alone_when_not_a_score_url(self):
        assert ab.health_url_for("http://boltz:8091/x") == "http://boltz:8091/x"

    def test_two_replicas_per_supervisor_at_twice_the_step(self):
        # A 10 min batch against a 5 min step is the plan's M ~ 2N.
        assert ab.replicas_for(600.0, supervisors=3, step_seconds=300.0) == 6

    def test_faster_batch_needs_fewer_replicas(self):
        assert ab.replicas_for(240.0, supervisors=3, step_seconds=300.0) == 3

    def test_replica_count_rounds_up(self):
        assert ab.replicas_for(310.0, supervisors=1, step_seconds=300.0) == 2


# ---------------------------------------------------------------------------
# TestDeploymentEnv
# ---------------------------------------------------------------------------


def _deployment(containers: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        stdout=json.dumps({"spec": {"template": {"spec": {"containers": containers}}}})
    )


class TestDeploymentEnv:
    """Reading the current flag so the original value can be restored."""

    def test_reads_the_value_from_the_right_container(self, monkeypatch):
        monkeypatch.setattr(ab, "kubectl", lambda *a, **k: _deployment([
            {"name": "dd-agent", "env": [{"name": "BOLTZ_NO_KERNELS", "value": "0"}]},
            {"name": "boltz", "env": [{"name": "BOLTZ_NO_KERNELS", "value": "1"}]},
        ]))

        assert ab.read_container_env("demo-lux", "dd-agentic", "boltz") == "1"

    def test_unset_reads_as_none(self, monkeypatch):
        monkeypatch.setattr(ab, "kubectl", lambda *a, **k: _deployment([
            {"name": "boltz", "env": [{"name": "OTHER", "value": "x"}]},
        ]))

        assert ab.read_container_env("demo-lux", "dd-agentic", "boltz") is None

    def test_container_without_env_reads_as_none(self, monkeypatch):
        monkeypatch.setattr(ab, "kubectl", lambda *a, **k: _deployment([
            {"name": "boltz"},
        ]))

        assert ab.read_container_env("demo-lux", "dd-agentic", "boltz") is None

    def test_missing_container_reads_as_none(self, monkeypatch):
        monkeypatch.setattr(ab, "kubectl", lambda *a, **k: _deployment([
            {"name": "posebusters", "env": [{"name": "BOLTZ_NO_KERNELS", "value": "0"}]},
        ]))

        assert ab.read_container_env("demo-lux", "dd-agentic", "boltz") is None

    def test_setting_a_value_then_waiting_for_rollout(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(ab, "kubectl",
                            lambda args, namespace, capture=False: calls.append(args))

        ab.write_container_env("demo-lux", "dd-agentic", "boltz", "0", "10m")

        assert calls[0] == ["set", "env", "deployment/dd-agentic",
                           "--containers=boltz", "BOLTZ_NO_KERNELS=0"]
        assert calls[1][:2] == ["rollout", "status"]

    def test_restoring_an_unset_value_removes_the_variable(self, monkeypatch):
        calls: list[list[str]] = []
        monkeypatch.setattr(ab, "kubectl",
                            lambda args, namespace, capture=False: calls.append(args))

        ab.write_container_env("demo-lux", "dd-agentic", "boltz", None, "10m")

        assert calls[0][-1] == "BOLTZ_NO_KERNELS-"


# ---------------------------------------------------------------------------
# TestCLIGuards
# ---------------------------------------------------------------------------


class TestCLIGuards:
    """Refuse runs that could not produce a usable comparison."""

    def test_two_arms_need_a_way_to_restart_boltz(self):
        with pytest.raises(SystemExit):
            ab.main(["--url", "http://localhost:8091/score"])

    def test_both_single_arm_flags_rejected(self):
        with pytest.raises(SystemExit):
            ab.main(["--baseline-only", "--kernels-only"])

    def test_two_flip_mechanisms_rejected(self):
        with pytest.raises(SystemExit):
            ab.main(["--namespace", "demo-lux",
                     "--restart-command", "docker compose up -d boltz"])

    def test_batch_size_beyond_the_panel_rejected(self):
        with pytest.raises(SystemExit):
            ab.main(["--baseline-only", "--batch-size", "64"])
