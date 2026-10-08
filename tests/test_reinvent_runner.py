"""Tests for reinvent-runner and the client that makes it a drop-in.

The runner owns REINVENT4: the RL/TL job lifecycle, the in-process Scorer, and
the component/model/transform registries. The agent becomes an HTTP client, so
the thing that has to hold is that nothing above the ``get_job_manager()`` seam
can tell the difference:

- the same seven methods, returning the same plain dicts
- an application refusal (no free slots, unknown job) survives the trip as an
  ``{"error": ...}`` dict rather than becoming an exception
- an unreachable runner produces that same shape, because every caller treats
  a dict with ``error`` as the failure case and would crash on a raise
- the TTP store stays on the agent side: a ``ttp_id`` is resolved before the
  request, so the runner never needs to read it

REINVENT4 is not installed here, so the runner's own module is exercised with
the manager and tool functions patched out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from reinvent4_mcp import runner_client
from reinvent4_mcp.config import ProcessConfig

_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "reinvent_runner_app", str(_ROOT / "services" / "reinvent-runner" / "app.py")
)
runner_app = importlib.util.module_from_spec(_spec)
sys.modules["reinvent_runner_app"] = runner_app
_spec.loader.exec_module(runner_app)

from fastapi.testclient import TestClient  # noqa: E402

JOB = {"job_id": "abc123", "state": "queued", "working_dir": "/data/jobs/job_abc123"}
REFUSAL = {
    "error": "Maximum concurrent jobs (2) already active: a, b. Stop one first.",
    "active_jobs": ["a", "b"],
}


@pytest.fixture
def config(monkeypatch) -> ProcessConfig:
    """A config with the runner enabled, as the agent side would have it."""
    cfg = ProcessConfig()
    cfg.runner.enabled = True
    cfg.runner.base_url = "http://runner:8086"
    cfg.runner.timeout_seconds = 30.0
    monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: cfg)
    return cfg


@pytest.fixture
def manager() -> MagicMock:
    """Stands in for the JobManager inside the runner process."""
    stub = MagicMock()
    stub.active_job_ids.return_value = []
    with patch.object(runner_app, "_manager", return_value=stub):
        yield stub


@pytest.fixture
def client(manager) -> TestClient:
    return TestClient(runner_app.app)


# ---------------------------------------------------------------------------
# TestRunnerEndpoints
# ---------------------------------------------------------------------------


class TestRunnerEndpoints:
    """Each endpoint is a thin pass-through to the code that already exists."""

    def test_create_job_passes_the_scorer_config_through(self, client, manager):
        manager.create_and_start.return_value = JOB
        body = {
            "ttp_id": "ttp-1", "ttp_version": 2,
            "scorer_config": {"type": "geometric_mean", "component": []},
            "model": ".reinvent", "max_steps": 50,
        }

        response = client.post("/jobs", json=body)

        assert response.status_code == 200
        assert response.json() == JOB
        # Already mapped from the TTP by the caller: the runner needs no store.
        assert manager.create_and_start.call_args.kwargs["scorer_config"] == body[
            "scorer_config"
        ]
        assert manager.create_and_start.call_args.kwargs["max_steps"] == 50

    def test_a_refusal_comes_back_as_200_with_an_error(self, client, manager):
        # The manager's contract is to *return* this, and preserving it across
        # the wire is what keeps the client a drop-in.
        manager.create_and_start.return_value = REFUSAL

        response = client.post("/jobs", json={
            "ttp_id": "t", "ttp_version": 1, "scorer_config": {},
        })

        assert response.status_code == 200
        assert response.json()["active_jobs"] == ["a", "b"]

    def test_unknown_fields_are_rejected(self, client, manager):
        # extra="forbid", so a stale client cannot silently drop a parameter.
        response = client.post("/jobs", json={
            "ttp_id": "t", "ttp_version": 1, "scorer_config": {},
            "no_such_option": True,
        })

        assert response.status_code == 422

    def test_tl_job(self, client, manager):
        manager.create_and_start_tl.return_value = {"job_id": "tl1"}

        response = client.post("/tl-jobs", json={
            "model": ".reinvent", "smiles": ["CCO"], "num_epochs": 5,
        })

        assert response.json() == {"job_id": "tl1"}
        assert manager.create_and_start_tl.call_args.kwargs["num_epochs"] == 5

    def test_status_stop_and_candidates(self, client, manager):
        manager.get_status.return_value = {"job_id": "abc123", "state": "running"}
        manager.stop_job.return_value = {"job_id": "abc123", "state": "stopped"}
        manager.get_top_candidates.return_value = {"candidates": []}

        assert client.get("/jobs/abc123").json()["state"] == "running"
        assert client.post("/jobs/abc123/stop").json()["state"] == "stopped"
        assert client.get(
            "/jobs/abc123/candidates", params={"top_n": 3, "min_score": 0.5}
        ).json() == {"candidates": []}
        assert manager.get_top_candidates.call_args.args == ("abc123", 3, 0.5)

    def test_lists(self, client, manager):
        manager.list_jobs.return_value = [JOB]
        manager.get_alerts.return_value = [{"job_id": "abc123", "kind": "plateau"}]

        assert client.get("/jobs").json() == [JOB]
        assert client.get("/alerts").json()[0]["kind"] == "plateau"

    def test_health_reports_whether_reinvent_is_importable(self, client, manager):
        # The point of the service is that REINVENT4 is here. A false means the
        # image is wrong, which is a different problem from being down.
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert "reinvent_importable" in body
        assert "max_concurrent_jobs" in body

    def test_score_takes_the_ttp_inline(self, client):
        with patch("reinvent4_mcp.tools.scoring.score_molecules") as scorer:
            scorer.return_value = {"molecules": []}
            response = client.post("/score", json={
                "smiles": ["CCO"], "ttp": {"id": "ttp-1", "name": "x"},
            })

        assert response.status_code == 200
        # No ttp_id parameter at all: the store is not reachable from here.
        assert scorer.call_args.kwargs["ttp"] == {"id": "ttp-1", "name": "x"}
        assert "ttp_id" not in scorer.call_args.kwargs

    def test_registries_are_exposed(self, client):
        with patch(
            "reinvent4_mcp.tools.discovery.list_scoring_components",
            return_value=[{"name": "QED"}],
        ):
            assert client.get("/components").json() == [{"name": "QED"}]

    def test_an_unexpected_failure_is_a_500(self, manager):
        manager.get_status.side_effect = RuntimeError("torch exploded")
        # raise_server_exceptions=False, or the handler's response is replaced
        # by the exception being re-raised into the test.
        strict = TestClient(runner_app.app, raise_server_exceptions=False)

        response = strict.get("/jobs/abc123")

        assert response.status_code == 500
        assert "torch exploded" in response.json()["error"]


# ---------------------------------------------------------------------------
# TestRemoteJobManager
# ---------------------------------------------------------------------------


def _response(body: Any, status: int = 200) -> MagicMock:
    stub = MagicMock()
    stub.status_code = status
    stub.ok = status < 400
    stub.json.return_value = body
    return stub


class TestRemoteJobManager:
    """A drop-in for the local manager, including how it fails."""

    def test_it_offers_exactly_what_callers_use(self):
        from reinvent4_mcp.jobs.manager import JobManager

        used = {
            "create_and_start", "create_and_start_tl", "get_status", "stop_job",
            "get_top_candidates", "list_jobs", "get_alerts",
        }

        for name in used:
            assert hasattr(runner_client.RemoteJobManager, name), name
            # Present on the local manager too, or the seam is not a seam.
            assert hasattr(JobManager, name), name

    @patch.object(runner_client.requests, "request")
    def test_create_and_start_posts_to_jobs(self, request, config):
        request.return_value = _response(JOB)

        result = runner_client.RemoteJobManager().create_and_start(
            ttp_id="t", ttp_version=1, scorer_config={}, max_steps=10,
        )

        assert result == JOB
        assert request.call_args.args == ("POST", "http://runner:8086/jobs")
        assert request.call_args.kwargs["json"]["max_steps"] == 10
        assert request.call_args.kwargs["timeout"] == 30.0

    @patch.object(runner_client.requests, "request")
    def test_a_refusal_is_returned_not_raised(self, request, config):
        request.return_value = _response(REFUSAL)

        result = runner_client.RemoteJobManager().create_and_start(
            ttp_id="t", ttp_version=1, scorer_config={},
        )

        assert result["active_jobs"] == ["a", "b"]
        assert "Maximum concurrent jobs" in result["error"]

    @patch.object(runner_client.requests, "request")
    def test_an_unreachable_runner_is_an_error_dict(self, request, config):
        # Every caller treats a dict with "error" as the failure case, so a
        # raise here would crash the tool instead of reporting.
        request.side_effect = requests.ConnectionError("no route to host")

        result = runner_client.RemoteJobManager().get_status("abc123")

        assert "unreachable" in result["error"]
        assert "http://runner:8086" in result["error"]

    @patch.object(runner_client.requests, "request")
    def test_a_5xx_keeps_the_runners_message(self, request, config):
        request.return_value = _response({"error": "torch exploded"}, status=500)

        result = runner_client.RemoteJobManager().get_status("abc123")

        assert "500" in result["error"]
        assert "torch exploded" in result["error"]

    @patch.object(runner_client.requests, "request")
    def test_an_unparseable_body_is_an_error_dict(self, request, config):
        stub = MagicMock()
        stub.status_code, stub.ok = 200, True
        stub.json.side_effect = ValueError("not json")
        request.return_value = stub

        result = runner_client.RemoteJobManager().get_status("abc123")

        assert "unparseable" in result["error"]

    @patch.object(runner_client.requests, "request")
    def test_list_methods_degrade_to_an_empty_list(self, request, config):
        # These return lists locally, so an error dict would break every
        # caller that iterates over them.
        request.side_effect = requests.ConnectionError("down")

        assert runner_client.RemoteJobManager().list_jobs() == []
        assert runner_client.RemoteJobManager().get_alerts() == []

    @patch.object(runner_client.requests, "request")
    def test_a_job_id_is_url_encoded(self, request, config):
        request.return_value = _response({})

        runner_client.RemoteJobManager().get_status("weird/../id")

        assert "weird%2F..%2Fid" in request.call_args.args[1]

    @patch.object(runner_client.requests, "request")
    def test_candidates_pass_their_filters_as_query_params(self, request, config):
        request.return_value = _response({"candidates": []})

        runner_client.RemoteJobManager().get_top_candidates("abc", 5, 0.25)

        assert request.call_args.kwargs["params"] == {"top_n": 5, "min_score": 0.25}


# ---------------------------------------------------------------------------
# TestSeam
# ---------------------------------------------------------------------------


class TestSeam:
    """``get_job_manager()`` decides where the work happens; nothing else does."""

    def test_disabled_by_default(self):
        # An unconfigured deployment keeps running REINVENT4 in-process.
        assert ProcessConfig().runner.enabled is False

    def test_the_local_manager_is_used_when_disabled(self, monkeypatch, tmp_path):
        cfg = ProcessConfig()
        cfg.jobs.working_dir = str(tmp_path / "jobs")
        monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: cfg)
        monkeypatch.setattr("reinvent4_mcp.jobs.manager.get_config", lambda: cfg)
        from reinvent4_mcp.jobs import manager as manager_module
        from reinvent4_mcp.jobs.manager import JobManager, get_job_manager

        manager_module.reset_job_manager()

        assert isinstance(get_job_manager(), JobManager)

    def test_the_remote_manager_is_used_when_enabled(self, config):
        from reinvent4_mcp.jobs.manager import get_job_manager

        assert isinstance(get_job_manager(), runner_client.RemoteJobManager)

    def test_flipping_the_config_needs_no_restart(self, monkeypatch, tmp_path):
        cfg = ProcessConfig()
        cfg.jobs.working_dir = str(tmp_path / "jobs")
        monkeypatch.setattr("reinvent4_mcp.config.get_config", lambda: cfg)
        monkeypatch.setattr("reinvent4_mcp.jobs.manager.get_config", lambda: cfg)
        from reinvent4_mcp.jobs import manager as manager_module
        from reinvent4_mcp.jobs.manager import JobManager, get_job_manager

        manager_module.reset_job_manager()
        assert isinstance(get_job_manager(), JobManager)
        cfg.runner.enabled = True

        assert isinstance(get_job_manager(), runner_client.RemoteJobManager)

    @patch.object(runner_client.requests, "request")
    def test_discovery_answers_from_the_runner_when_enabled(self, request, config):
        request.return_value = _response([{"name": "QED"}])
        from reinvent4_mcp.tools.discovery import list_scoring_components

        # Would need REINVENT4 installed locally otherwise, which is the whole
        # point of moving it.
        assert list_scoring_components() == [{"name": "QED"}]
        assert request.call_args.args[1].endswith("/components")

    @patch.object(runner_client.requests, "request")
    def test_scoring_resolves_a_ttp_id_before_sending_it(self, request, config):
        request.return_value = _response({"molecules": []})
        resolved = MagicMock()
        resolved.model_dump.return_value = {"id": "ttp-1", "name": "x"}
        from reinvent4_mcp.tools.scoring import score_molecules

        with patch("reinvent4_mcp.tools.ttp.resolve_ttp", return_value=resolved):
            score_molecules(smiles=["CCO"], ttp_id="ttp-1")

        # Sent inline, because the runner has no TTP store.
        body = request.call_args.kwargs["json"]
        assert body["ttp"] == {"id": "ttp-1", "name": "x"}

    @patch.object(runner_client.requests, "request")
    def test_an_unresolvable_ttp_id_never_reaches_the_runner(self, request, config):
        from reinvent4_mcp.tools.scoring import score_molecules

        with patch(
            "reinvent4_mcp.tools.ttp.resolve_ttp",
            return_value={"error": "TTP 'nope' not found"},
        ):
            result = score_molecules(smiles=["CCO"], ttp_id="nope")

        assert result == {"error": "TTP 'nope' not found"}
        request.assert_not_called()

    def test_an_empty_batch_short_circuits(self, config):
        from reinvent4_mcp.tools.scoring import score_molecules

        # No HTTP call for nothing to score.
        assert score_molecules(smiles=[])["summary"]["n_molecules"] == 0

    def test_the_runner_never_proxies_to_itself(self, config, monkeypatch):
        """The runner reads the same config file as the agent.

        Without this guard, ``runner.enabled`` would make the runner forward
        every job and every score to its own address.
        """
        assert runner_client.runner_enabled() is True

        monkeypatch.setenv(runner_client.SELF_ENV, "1")

        assert runner_client.runner_enabled() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_an_unset_or_falsy_self_flag_leaves_the_config_in_charge(
        self, config, monkeypatch, value
    ):
        monkeypatch.setenv(runner_client.SELF_ENV, value)

        assert runner_client.runner_enabled() is True

    def test_the_runner_image_sets_the_self_flag(self):
        dockerfile = (
            _ROOT / "services" / "reinvent-runner" / "Dockerfile"
        ).read_text()

        assert f"ENV {runner_client.SELF_ENV}=1" in dockerfile

    def test_the_runner_manifest_sets_it_too(self):
        # Belt and braces: the env is in the image, but an operator reading the
        # Deployment should see it, and overriding the image command must not
        # lose it.
        import yaml

        manifest = (
            _ROOT / "k8s" / "manifests" / "runner" / "reinvent-runner.yaml"
        ).read_text()
        deployment = next(
            d for d in yaml.safe_load_all(manifest) if d and d["kind"] == "Deployment"
        )
        env = {
            var["name"]: var.get("value")
            for var in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        }

        assert env[runner_client.SELF_ENV] == "1"

    def test_the_runner_owns_its_manager_rather_than_using_the_seam(self, config):
        """Job endpoints cannot loop even without the env guard.

        ``get_job_manager()`` honours ``runner.enabled``; the runner builds a
        ``JobManager`` itself, so with the config enabled it still gets the
        local one.
        """
        assert runner_client.runner_enabled() is True
        runner_app._local_manager = None

        with patch("reinvent4_mcp.jobs.manager.JobManager") as local:
            local.return_value = "a local manager"
            assert runner_app._manager() == "a local manager"

        runner_app._local_manager = None


# ---------------------------------------------------------------------------
# TestImageSplit
# ---------------------------------------------------------------------------


class TestImageSplit:
    """REINVENT4 and the priors belong to the runner image, not the agent's.

    This is the ~11 GiB that every agent replica used to pull, plus the GPU it
    pinned whether or not it was running a job. The checks here are on the build
    and deployment definitions, since the images themselves are built elsewhere.
    """

    @staticmethod
    def _directives(path: Path) -> str:
        """The Dockerfile's instructions, without the prose.

        Comments explain what each image deliberately does *not* carry, so
        matching on the raw text would find the explanation instead of the
        instruction.
        """
        return "\n".join(
            line for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    @pytest.fixture(scope="class")
    def agent_dockerfile(self) -> str:
        return self._directives(_ROOT / "services" / "dd-agent" / "Dockerfile")

    @pytest.fixture(scope="class")
    def runner_dockerfile(self) -> str:
        return self._directives(_ROOT / "services" / "reinvent-runner" / "Dockerfile")

    @pytest.fixture(scope="class")
    def compose(self) -> dict[str, Any]:
        import yaml

        return yaml.safe_load((_ROOT / "docker-compose.yml").read_text())["services"]

    def test_the_agent_image_carries_no_reinvent4(self, agent_dockerfile):
        assert "COPY REINVENT4/" not in agent_dockerfile
        assert "zenodo.org" not in agent_dockerfile, "priors belong to the runner"
        assert "REINVENT4_ROOT" not in agent_dockerfile

    def test_the_agent_image_is_not_built_on_the_gpu_base(self, agent_dockerfile):
        # rocm/pytorch is most of the 11 GiB.
        assert "rocm/pytorch" not in agent_dockerfile
        assert agent_dockerfile.count("FROM ") == 1
        assert "FROM python:" in agent_dockerfile

    def test_the_agent_build_fails_if_reinvent4_creeps_back(self, agent_dockerfile):
        # A transitive dependency pulling torch back in should break the build,
        # not quietly restore an 11 GiB image.
        assert "find_spec('reinvent') is None" in agent_dockerfile
        assert "find_spec('torch') is None" in agent_dockerfile

    def test_the_runner_image_owns_reinvent4_and_the_priors(self, runner_dockerfile):
        assert "COPY REINVENT4/" in runner_dockerfile
        assert "zenodo.org" in runner_dockerfile
        assert "rocm/pytorch" in runner_dockerfile
        # And still verifies the custom scorer is registered, which is what the
        # RL subprocess needs.
        assert "externalrestscorer" in runner_dockerfile

    def test_the_runner_image_needs_no_agent_package(self, runner_dockerfile):
        # Nothing in the runner talks to an LLM.
        assert "reinvent4_agent" not in runner_dockerfile

    def test_the_runner_is_no_longer_built_from_the_agent(self, runner_dockerfile):
        # It was, deliberately, while the HTTP path was being proven. Now that
        # the agent is slim it cannot be the parent any more.
        assert "FROM dd-agent" not in runner_dockerfile
        assert "BASE_IMAGE" not in runner_dockerfile

    def test_the_build_script_gives_the_runner_the_repo_root(self):
        script = (_ROOT / "k8s" / "build-images.sh").read_text()

        # It copies REINVENT4/ and reinvent4_mcp/, so the context cannot be the
        # service directory.
        assert 'services/reinvent-runner/Dockerfile \\\n    --tag "reinvent-runner:${TAG}" .' in script

    def test_the_agent_deployment_asks_for_no_gpu(self):
        import yaml

        manifest = (_ROOT / "k8s" / "manifests" / "deployment.yaml").read_text()
        deployment = next(
            d for d in yaml.safe_load_all(manifest) if d and d["kind"] == "Deployment"
        )
        container = deployment["spec"]["template"]["spec"]["containers"][0]

        # The whole point: the agent is freely schedulable and the GPU it used
        # to hold is available to Boltz or the runner.
        assert "amd.com/gpu" not in container["resources"]["requests"]
        assert "amd.com/gpu" not in container["resources"]["limits"]

    def test_the_cluster_config_enables_the_runner(self):
        import yaml

        configmap = next(
            d for d in yaml.safe_load_all(
                (_ROOT / "k8s" / "manifests" / "configmap.yaml").read_text()
            )
            if d and d["kind"] == "ConfigMap"
        )
        config = yaml.safe_load(configmap["data"]["reinvent4_mcp_config.yaml"])

        # Not optional with a slim agent image: there is no local REINVENT4 to
        # fall back to.
        assert config["runner"]["enabled"] is True
        assert "dd-agentic-reinvent-runner" in config["runner"]["base_url"]

    def test_the_compose_config_enables_the_runner_too(self):
        import yaml

        config = yaml.safe_load((_ROOT / "reinvent4_mcp_config.yaml").read_text())

        assert config["runner"]["enabled"] is True
        assert "reinvent-runner" in config["runner"]["base_url"]

    def test_compose_moves_the_gpu_to_the_runner(self, compose):
        agent = compose["reinvent4-mcp"]
        runner = compose["reinvent-runner"]

        assert "devices" not in agent, "the agent no longer needs ROCm devices"
        assert "shm_size" not in agent, "the big shm was for REINVENT dataloaders"
        assert runner["devices"] == ["/dev/kfd", "/dev/dri"]
        assert runner["shm_size"] == "128g"

    def test_compose_agent_waits_for_the_runner(self, compose):
        # Its jobs go there, so starting before it is ready would fail them.
        assert "reinvent-runner" in compose["reinvent4-mcp"]["depends_on"]

    def test_compose_runner_has_no_ttp_mount(self, compose):
        targets = [v.split(":")[-1] for v in compose["reinvent-runner"]["volumes"]]

        # TTPs are resolved agent-side; only the mapped scorer_config crosses.
        assert not any("ttps" in t for t in targets)
        assert "/data/reinvent4_mcp/jobs" in targets

    def test_the_runner_image_is_in_the_load_list(self):
        # Or the node would never get it.
        assert "reinvent-runner" in (_ROOT / "k8s" / "load-images.sh").read_text()
