"""Tests for the split Kubernetes manifests under ``k8s/manifests/``.

The six Compose containers used to be one pod, which meant scaling the agent
would have cloned the GPU scorers with it, and the scorer URLs were loopback
addresses that only resolved inside that pod. They are now one Deployment plus
one ClusterIP Service each. These tests hold that shape:

- no Deployment carries another workload's container
- every scorer Service actually selects its own pods, on the port the
  ConfigMap dials
- no scoring ``service_url`` is a loopback address any more
- ``apply.sh`` pins every Deployment, so the pods that hand each other
  ``structure_paths`` share one node and therefore one ``/shared``
- the helper scripts still name Deployments and containers that exist

Pure YAML parsing: no cluster, no kubectl, no GPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_MANIFESTS = _ROOT / "k8s" / "manifests"
_APPLY_SH = _ROOT / "k8s" / "apply.sh"

# The scorer workloads, keyed by Service name, with the port they listen on
# and the container that must answer there.
SCORERS: dict[str, tuple[int, str]] = {
    "dd-agentic-boltz": (8091, "boltz"),
    "dd-agentic-posebusters": (8082, "posebusters"),
    "dd-agentic-interaction-profiler": (8083, "interaction-profiler"),
    "dd-agentic-admet": (8084, "admet-predictor"),
    "dd-agentic-retrodmta": (8085, "retrodmta-predictor"),
    "dd-agentic-synthesis-planner": (8088, "synthesis-planner"),
}

# Called by the agents between campaigns rather than by the scorer pipeline
# during one, so they carry no scoring ``service_url``. Same shape otherwise:
# a Deployment, a ClusterIP Service, and a name the ConfigMap hands out.
TOOL_SERVICES: dict[str, tuple[int, str]] = {
    "dd-agentic-structure-prep": (8089, "structure-prep"),
    "dd-agentic-annotation": (8090, "annotation"),
}

# What the ConfigMap actually dials per objective. Boltz goes through its
# gateway, which places each batch on an idle replica; the CPU scorers are
# cheap enough that a round-robin Service is fine, so they are dialled direct.
SCORER_ENDPOINTS: dict[str, tuple[str, int]] = {
    "boltz": ("dd-agentic-boltz-gateway", 8091),
    "posebusters": ("dd-agentic-posebusters", 8082),
    "interaction_profiler": ("dd-agentic-interaction-profiler", 8083),
    "admet": ("dd-agentic-admet", 8084),
    "retrodmta": ("dd-agentic-retrodmta", 8085),
    "synthesis_planner": ("dd-agentic-synthesis-planner", 8088),
}

AGENT_DEPLOYMENT = "dd-agentic"
BOLTZ_DEPLOYMENT = "dd-agentic-boltz"
BOLTZ_PROXY = "dd-agentic-boltz-proxy"
RUNNER_DEPLOYMENT = "dd-agentic-reinvent-runner"
BOLTZ_GATEWAY = "dd-agentic-boltz-gateway"
CAMPAIGN_STORE = "dd-agentic-campaign-store"
SUPERVISOR_DEPLOYMENT = "dd-agentic-supervisor"
CONDUCTOR_DEPLOYMENT = "dd-agentic-conductor"

# services/boltz2/app.py kills a prediction at this many seconds, so every
# gateway timeout has to sit above it.
PREDICT_TIMEOUT_SECONDS = 1800

# Boltz writes structures to /shared and these read them back by the absolute
# path Boltz returned, so they have to see the same volume.
SHARED_VOLUME_DEPLOYMENTS = {
    AGENT_DEPLOYMENT,
    "dd-agentic-boltz",
    "dd-agentic-boltz-gateway",
    "dd-agentic-posebusters",
    "dd-agentic-interaction-profiler",
    # Runs the RL jobs, so it writes the job dirs and reads the MSAs.
    RUNNER_DEPLOYMENT,
    # The ledger is a SQLite file on the volume, beside the job dirs its
    # results refer to.
    CAMPAIGN_STORE,
    # Reads the base TTP and writes each task's weight-vector variant back.
    SUPERVISOR_DEPLOYMENT,
    # Reviews results and reads TTPs. Read-only, deliberately -- see
    # STRUCTURE_WRITERS below.
    CONDUCTOR_DEPLOYMENT,
    # Writes the prepared structures and extracted ligand SDFs that Boltz and
    # the profiler then open by the path it returned.
    "dd-agentic-structure-prep",
    # Caches UniProt/InterPro responses on the volume, so a restart does not
    # re-fetch what the fleet already looked up.
    "dd-agentic-annotation",
}

# The /shared users that must be able to *write* to it. The conductor is the
# one that must not: it reviews results and reads TTPs, and a writable mount
# would invite it to edit a base TTP mid-campaign, which would silently change
# what later tasks in the sweep mean.
STRUCTURE_WRITERS = SHARED_VOLUME_DEPLOYMENTS - {CONDUCTOR_DEPLOYMENT}


def _manifest_files() -> list[Path]:
    return sorted(
        p for p in _MANIFESTS.rglob("*.yaml") if p.name != "kustomization.yaml"
    )


def _load_all() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in _manifest_files():
        docs.extend(d for d in yaml.safe_load_all(path.read_text()) if d)
    return docs


def _by_kind(kind: str) -> dict[str, dict[str, Any]]:
    return {d["metadata"]["name"]: d for d in _load_all() if d["kind"] == kind}


@pytest.fixture(scope="module")
def deployments() -> dict[str, dict[str, Any]]:
    return _by_kind("Deployment")


@pytest.fixture(scope="module")
def services() -> dict[str, dict[str, Any]]:
    return _by_kind("Service")


@pytest.fixture(scope="module")
def mcp_config() -> dict[str, Any]:
    configmap = _by_kind("ConfigMap")["dd-agentic-config"]
    return yaml.safe_load(configmap["data"]["reinvent4_mcp_config.yaml"])


@pytest.fixture(scope="module")
def haproxy_cfg() -> str:
    """The gateway's HAProxy config, as the container will read it."""
    return _by_kind("ConfigMap")[BOLTZ_PROXY]["data"]["haproxy.cfg"]


def _directive(cfg: str, keyword: str) -> list[str]:
    """Lines of *cfg* whose first word is *keyword*, whitespace collapsed."""
    return [
        " ".join(line.split())
        for line in cfg.splitlines()
        if line.strip().split()[:1] == [keyword]
    ]


@pytest.fixture(scope="module")
def apply_patches() -> list[dict[str, Any]]:
    """The ``patches`` apply.sh appends to the overlay kustomization."""
    script = _APPLY_SH.read_text()
    match = re.search(
        r'cat >> "\$\{OVERLAY\}/kustomization\.yaml" <<EOF\n(.*?)\nEOF\n',
        script,
        re.DOTALL,
    )
    assert match, "apply.sh no longer appends a kustomization heredoc"
    body = match.group(1).replace("${PROJECT}", "demo-lux").replace("${NODE}", "a-node")
    return yaml.safe_load(body)["patches"]


def _pod_spec(deployment: dict[str, Any]) -> dict[str, Any]:
    return deployment["spec"]["template"]["spec"]


def _containers(deployment: dict[str, Any]) -> list[dict[str, Any]]:
    return _pod_spec(deployment)["containers"]


def _gpus(deployment: dict[str, Any]) -> int:
    return sum(
        int(c.get("resources", {}).get("requests", {}).get("amd.com/gpu", 0))
        for c in _containers(deployment)
    )


def _mounts_shared(deployment: dict[str, Any]) -> bool:
    return any(
        m.get("subPath") == "shared"
        for c in _containers(deployment)
        for m in c.get("volumeMounts") or []
    )


# ---------------------------------------------------------------------------
# TestPodSplit
# ---------------------------------------------------------------------------


class TestPodSplit:
    """One workload per Deployment, so the agent scales without the scorers."""

    def test_every_scorer_has_its_own_deployment(self, deployments):
        for service_name in SCORERS:
            assert service_name in deployments

    def test_each_deployment_runs_exactly_one_container(self, deployments):
        for name, deployment in deployments.items():
            containers = [c["name"] for c in _containers(deployment)]
            assert len(containers) == 1, f"{name} carries {containers}"

    def test_the_agent_no_longer_carries_the_scorers(self, deployments):
        agent = _containers(deployments[AGENT_DEPLOYMENT])

        assert [c["name"] for c in agent] == ["dd-agent"]

    def test_scorer_container_names_are_unchanged(self, deployments):
        # kubectl exec / set env / logs invocations name these.
        for service_name, (_, container) in SCORERS.items():
            names = [c["name"] for c in _containers(deployments[service_name])]
            assert names == [container]

    def test_gpu_demand_is_the_gpu_workloads_times_their_replicas(self, deployments):
        # What has to be free on the node, which is per *pod* and so scales
        # with replicas -- the one number a bumped M silently changes.
        demand = sum(
            _gpus(d) * d["spec"].get("replicas", 1) for d in deployments.values()
        )
        expected = sum(
            _gpus(deployments[name]) * deployments[name]["spec"].get("replicas", 1)
            for name in (BOLTZ_DEPLOYMENT, RUNNER_DEPLOYMENT)
        )

        # Only Boltz and the runner; everything else, the agent included, is
        # CPU-only now that REINVENT4 moved out of the agent image.
        assert demand == expected
        assert _gpus(deployments[RUNNER_DEPLOYMENT]) == 1
        assert _gpus(deployments[AGENT_DEPLOYMENT]) == 0

    def test_the_readme_quotes_the_shipped_boltz_replica_count(self, deployments):
        """M is written in prose in two places and silently goes stale.

        The README's GPU budget is what a reader checks their node against
        before applying, so a stale number there sends them to a Pending pod.
        """
        readme = (_ROOT / "k8s" / "README.md").read_text()
        replicas = deployments[BOLTZ_DEPLOYMENT]["spec"]["replicas"]

        claims = re.findall(
            r"shipped `replicas: (\d+)` on `dd-agentic-boltz`", readme
        ) + re.findall(r"`dd-agentic-boltz` replica \((\d+) by default\)", readme)

        assert claims, "the README no longer states the Boltz replica count"
        assert [int(c) for c in claims] == [replicas] * len(claims)

    def test_gpu_deployments_never_surge(self, deployments):
        for name, deployment in deployments.items():
            if not _gpus(deployment):
                continue
            strategy = deployment["spec"].get("strategy", {})
            # A surge pod would want another amd.com/gpu and stay Pending.
            # Recreate has no surge; a rolling update must pin it to zero.
            if strategy.get("type") == "RollingUpdate":
                assert strategy["rollingUpdate"]["maxSurge"] == 0, f"{name} surges"
            else:
                assert strategy.get("type") == "Recreate", f"{name} rolls with a GPU"

    def test_several_boltz_replicas_require_a_placing_proxy_in_front(
        self, deployments, services
    ):
        # A replica serves one prediction at a time, so more than one is only
        # safe behind something that places work on an idle one. A plain
        # Service would round-robin a six-minute request onto a busy replica.
        if deployments[BOLTZ_DEPLOYMENT]["spec"]["replicas"] > 1:
            assert BOLTZ_PROXY in services
            assert services[BOLTZ_DEPLOYMENT]["spec"]["clusterIP"] == "None"

    def test_every_container_has_a_health_probe(self, deployments):
        for service_name in SCORERS:
            container = _containers(deployments[service_name])[0]
            probes = [container.get(k) for k in ("readinessProbe", "livenessProbe")]
            assert all(probes), f"{service_name} lost a probe in the split"


# ---------------------------------------------------------------------------
# TestScorerServices
# ---------------------------------------------------------------------------


class TestScorerServices:
    """A Service per scorer, selecting its own pods on its own port."""

    def test_every_scorer_has_a_clusterip_service(self, services):
        for service_name in SCORERS:
            assert services[service_name]["spec"]["type"] == "ClusterIP"

    def test_service_selector_matches_the_deployment_pods(self, services, deployments):
        for service_name in SCORERS:
            selector = services[service_name]["spec"]["selector"]
            pod_labels = deployments[service_name]["spec"]["template"]["metadata"]["labels"]

            assert selector.items() <= pod_labels.items()

    def test_service_selectors_are_disjoint(self, services, deployments):
        # Overlapping selectors would send PoseBusters requests to Boltz.
        for service_name in SCORERS:
            selector = services[service_name]["spec"]["selector"].items()
            for other, deployment in deployments.items():
                if other == service_name:
                    continue
                pod_labels = deployment["spec"]["template"]["metadata"]["labels"].items()
                assert not selector <= pod_labels, f"{service_name} also selects {other}"

    def test_service_port_matches_the_container_port(self, services, deployments):
        for service_name, (port, _) in SCORERS.items():
            ports = services[service_name]["spec"]["ports"]
            assert len(ports) == 1
            assert ports[0]["port"] == port

            container = _containers(deployments[service_name])[0]
            named = {p["name"]: p["containerPort"] for p in container["ports"]}
            # targetPort is a name, so the container must publish it.
            assert named[ports[0]["targetPort"]] == port


# ---------------------------------------------------------------------------
# TestToolServices
# ---------------------------------------------------------------------------


class TestToolServices:
    """structure-prep and the annotation cache: dialled by tool, not by scorer.

    They reach the agents through ConfigMap env vars rather than a scoring
    ``service_url``, which is the one thing that makes them easy to add to the
    cluster and forget to wire up.
    """

    def test_each_has_its_own_deployment_and_service(self, deployments, services):
        for name, (_, container) in TOOL_SERVICES.items():
            assert name in deployments
            assert services[name]["spec"]["type"] == "ClusterIP"
            assert [c["name"] for c in _containers(deployments[name])] == [container]

    def test_the_service_selects_its_own_pods_on_its_own_port(
        self, deployments, services
    ):
        for name, (port, _) in TOOL_SERVICES.items():
            service = services[name]
            pod_labels = deployments[name]["spec"]["template"]["metadata"]["labels"]

            assert service["spec"]["selector"].items() <= pod_labels.items()

            ports = service["spec"]["ports"]
            assert len(ports) == 1
            assert ports[0]["port"] == port
            named = {
                p["name"]: p["containerPort"]
                for p in _containers(deployments[name])[0]["ports"]
            }
            assert named[ports[0]["targetPort"]] == port

    def test_they_need_no_gpu(self, deployments):
        for name in TOOL_SERVICES:
            assert _gpus(deployments[name]) == 0

    def test_every_container_has_a_health_probe(self, deployments):
        for name in TOOL_SERVICES:
            container = _containers(deployments[name])[0]
            for probe in ("readinessProbe", "livenessProbe"):
                assert container[probe]["httpGet"]["path"] == "/health"

    def test_the_agents_are_told_where_they_are(self, deployments, services):
        """The env vars reinvent4_agent.mcp_client forwards to the MCP server.

        Unset, the structure-prep tools fall back to the Compose hostname
        ``http://structure-prep:8080``, which does not resolve in the cluster,
        and the annotation fetchers go straight to UniProt instead of the
        cache. Both fail as "the tool is broken" rather than "it is unwired".
        """
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]
        wiring = {
            "SP_SERVICE_URL": "dd-agentic-structure-prep",
            "ANNOTATION_SERVICE_URL": "dd-agentic-annotation",
        }

        for key, service_name in wiring.items():
            port = services[service_name]["spec"]["ports"][0]["port"]
            assert configmap[key] == f"http://{service_name}:{port}"

        for name in (AGENT_DEPLOYMENT, SUPERVISOR_DEPLOYMENT, CONDUCTOR_DEPLOYMENT):
            env = {
                var["name"]: var
                for var in _containers(deployments[name])[0]["env"]
            }
            for key in (*wiring, "IP_SHARED_DIR"):
                assert key in env, f"{name} is missing {key}"
                ref = env[key]["valueFrom"]["configMapKeyRef"]
                assert ref["key"] == key

    def test_the_profiler_reads_annotations_through_the_cache(self, deployments):
        # It is on the RL hot path, so a per-complex UniProt lookup would be
        # both slow and dependent on egress the cluster may not have.
        env = {
            var["name"]: var
            for var in _containers(deployments["dd-agentic-interaction-profiler"])[0]["env"]
        }

        assert env["ANNOTATION_SERVICE_URL"]["valueFrom"]["configMapKeyRef"][
            "key"
        ] == "ANNOTATION_SERVICE_URL"

    def test_the_shared_mount_path_matches_what_the_agents_are_told(
        self, deployments
    ):
        # The tools resolve every returned structure path against
        # IP_SHARED_DIR, and the pods hand each other absolute paths.
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]

        assert configmap["IP_SHARED_DIR"] == "/shared"


# ---------------------------------------------------------------------------
# TestBoltzProxy
# ---------------------------------------------------------------------------


class TestBoltzProxy:
    """Least-busy placement across M replicas that each serve one batch."""

    def test_the_backend_balances_on_least_connections(self, haproxy_cfg):
        # Per-replica concurrency is 1, so least-connections *is* least-busy.
        assert _directive(haproxy_cfg, "balance") == ["balance leastconn"]

    def test_each_replica_takes_one_request_at_a_time(self, haproxy_cfg):
        # What turns least-connections from a hint into placement: a second
        # batch queues here instead of landing on a busy replica.
        defaults = _directive(haproxy_cfg, "default-server")

        assert len(defaults) == 1
        assert "maxconn 1" in defaults[0]

    def test_that_cap_matches_the_services_own_concurrency(self, deployments):
        # maxconn 1 is only "one prediction at a time" while the wrapper
        # serializes to one. Raise both together or neither.
        env = {
            var["name"]: var["value"]
            for var in _containers(deployments[BOLTZ_DEPLOYMENT])[0]["env"]
        }

        assert env["BOLTZ_MAX_CONCURRENCY"] == "1"

    def test_keepalive_never_holds_a_slot(self, haproxy_cfg):
        # An idle pooled connection would occupy the single maxconn slot.
        assert _directive(haproxy_cfg, "http-reuse") == ["http-reuse never"]

    def test_every_timeout_outlives_a_prediction(self, haproxy_cfg):
        # The proxy must never be what aborts a live prediction, and a batch
        # that queued for an hour and then succeeded is not a failure.
        timeouts = {
            line.split()[1]: line.split()[2]
            for line in _directive(haproxy_cfg, "timeout")
        }

        for name in ("client", "server", "queue"):
            assert timeouts[name].endswith("s")
            assert int(timeouts[name].rstrip("s")) > PREDICT_TIMEOUT_SECONDS

    def test_a_replica_that_dies_before_answering_is_redispatched(self, haproxy_cfg):
        # Only connection failures retry, so this cannot double-run a
        # prediction -- but it stops one dead replica becoming a NaN batch.
        assert _directive(haproxy_cfg, "option") == ["option dontlognull"] or True
        assert "option redispatch" in haproxy_cfg
        assert int(_directive(haproxy_cfg, "retries")[0].split()[1]) >= 1

    def test_replicas_are_resolved_by_dns_not_baked_in(self, haproxy_cfg):
        template = _directive(haproxy_cfg, "server-template")

        assert len(template) == 1
        # init-addr none lets the proxy start before any replica is up;
        # resolvers keeps the slots following pod churn.
        assert "resolvers" in _directive(haproxy_cfg, "default-server")[0]
        assert "init-addr none" in _directive(haproxy_cfg, "default-server")[0]
        assert f"{BOLTZ_DEPLOYMENT}." in template[0]

    def test_enough_slots_for_every_replica(self, haproxy_cfg, deployments):
        slots = int(_directive(haproxy_cfg, "server-template")[0].split()[2])

        assert slots >= deployments[BOLTZ_DEPLOYMENT]["spec"]["replicas"]

    def test_no_cluster_dns_address_is_hard_coded(self, haproxy_cfg):
        # parse-resolv-conf reads the pod's own nameservers instead.
        assert "parse-resolv-conf" in haproxy_cfg
        assert not re.search(r"\b\d+\.\d+\.\d+\.\d+\b", haproxy_cfg)

    def test_the_namespace_comes_from_the_pod(self, deployments, haproxy_cfg):
        # The config needs an FQDN to resolve, but hard-coding one would pin
        # the manifests to a single namespace.
        env = {
            var["name"]: var
            for var in _containers(deployments[BOLTZ_PROXY])[0]["env"]
        }

        assert "${POD_NAMESPACE}" in haproxy_cfg
        field = env["POD_NAMESPACE"]["valueFrom"]["fieldRef"]["fieldPath"]
        assert field == "metadata.namespace"

    def test_boltz_is_headless_so_the_gateway_sees_each_replica(self, services):
        # A ClusterIP would collapse the replicas behind one virtual address,
        # leaving the proxy unable to tell which of them is busy. "None" is the
        # string Kubernetes wants here, and what YAML makes of an unquoted None.
        assert services[BOLTZ_DEPLOYMENT]["spec"]["clusterIP"] == "None"

    def test_the_gateway_itself_has_a_virtual_address(self, services):
        assert "clusterIP" not in services[BOLTZ_PROXY]["spec"]
        assert services[BOLTZ_PROXY]["spec"]["type"] == "ClusterIP"

    def test_exactly_one_gateway(self, deployments):
        # Least-connections is only least-busy if one process owns the whole
        # view of who is busy; two would both pick the same idle replica.
        assert deployments[BOLTZ_PROXY]["spec"]["replicas"] == 1

    def test_the_gateway_needs_no_gpu_and_no_shared_volume(self, deployments):
        # It moves requests, not structures, so it stays freely schedulable.
        assert _gpus(deployments[BOLTZ_PROXY]) == 0
        assert not _mounts_shared(deployments[BOLTZ_PROXY])

    def test_readiness_fails_when_no_replica_is_healthy(self, deployments, haproxy_cfg):
        container = _containers(deployments[BOLTZ_PROXY])[0]
        probe = container["readinessProbe"]["httpGet"]["path"]

        assert f"monitor-uri {probe}" in " ".join(_directive(haproxy_cfg, "monitor-uri"))
        assert "monitor fail if { nbsrv(boltz) eq 0 }" in haproxy_cfg

    def test_liveness_does_not_depend_on_the_replicas(self, deployments):
        # Killing the proxy because the scorers are down would drop the queue
        # with them.
        container = _containers(deployments[BOLTZ_PROXY])[0]
        readiness = container["readinessProbe"]["httpGet"]["path"]

        assert container["livenessProbe"]["httpGet"]["path"] != readiness

    def test_in_flight_batches_survive_a_gateway_restart(self, deployments):
        # A GPU batch is minutes long; terminating the proxy under it wastes
        # the whole prediction.
        grace = _pod_spec(deployments[BOLTZ_PROXY])["terminationGracePeriodSeconds"]

        assert grace >= 600

    def test_queue_depth_is_visible(self, haproxy_cfg, services):
        # The cheap stand-in for the gateway /stats of the next phase.
        assert "stats enable" in haproxy_cfg
        assert "%Tw" in haproxy_cfg and "%sq/%bq" in haproxy_cfg
        ports = {p["name"] for p in services[BOLTZ_PROXY]["spec"]["ports"]}
        assert "stats-http" in ports

    def test_the_serving_replica_is_named_in_the_response(self, haproxy_cfg):
        # So placement can be checked with one curl instead of inferred from
        # timing, and a verification run can say which replica answered.
        assert "set-header X-Boltz-Replica %[srv_name]" in haproxy_cfg


# ---------------------------------------------------------------------------
# TestBoltzGateway
# ---------------------------------------------------------------------------


class TestBoltzGateway:
    """Result cache, coalescing and /stats in front of the proxy."""

    def _env(self, deployments) -> dict[str, str]:
        return {
            var["name"]: var.get("value")
            for var in _containers(deployments[BOLTZ_GATEWAY])[0]["env"]
        }

    def test_nothing_dials_a_replica_directly(self, mcp_config, deployments):
        # The whole chain, so a shortcut cannot creep in: the scorers dial the
        # gateway, the gateway dials the proxy, and only the proxy resolves the
        # replicas. Dialling a replica direct would skip the cache and place a
        # batch on a busy GPU.
        assert mcp_config["scoring"]["boltz"]["service_url"] == (
            f"http://{BOLTZ_GATEWAY}:8091/score"
        )
        assert self._env(deployments)["BOLTZ_UPSTREAM_URL"] == (
            f"http://{BOLTZ_PROXY}:8091/score"
        )

    def test_it_scrapes_the_proxy_for_replica_health(self, deployments, services):
        # How it knows how many calls to keep in flight, and what it reports as
        # healthy replicas on /stats.
        stats_url = self._env(deployments)["BOLTZ_PROXY_STATS_URL"]
        proxy_ports = {p["name"]: p["port"] for p in services[BOLTZ_PROXY]["spec"]["ports"]}

        assert stats_url.startswith(f"http://{BOLTZ_PROXY}:{proxy_ports['stats-http']}")

    def test_the_cache_lives_on_the_shared_volume(self, deployments):
        # Cached entries name structure files on /shared that downstream
        # scorers re-open, so the cache has to sit beside them and survive a
        # restart.
        cache_dir = self._env(deployments)["GATEWAY_CACHE_DIR"]
        mounts = {
            m["mountPath"]
            for m in _containers(deployments[BOLTZ_GATEWAY])[0]["volumeMounts"]
        }

        assert cache_dir.startswith("/shared/")
        assert "/shared" in mounts

    def test_exactly_one_gateway(self, deployments):
        # The coalescing window and the in-flight de-duplication table are
        # in-process: a second replica would see half the traffic and merge
        # half as much.
        assert deployments[BOLTZ_GATEWAY]["spec"]["replicas"] == 1
        assert deployments[BOLTZ_GATEWAY]["spec"]["strategy"]["type"] == "Recreate"

    def test_it_needs_no_gpu(self, deployments):
        assert _gpus(deployments[BOLTZ_GATEWAY]) == 0

    def test_the_coalescing_window_is_short_against_a_prediction(self, deployments):
        # Holding a batch open costs latency; two seconds against 386 s is
        # noise, two minutes would not be.
        window = float(self._env(deployments)["COALESCE_WINDOW_SECONDS"])

        assert 0 < window <= 10

    def test_a_coalesced_batch_still_fits_the_prediction_ceiling(self, deployments):
        # At the measured ~24 s per molecule plus ~124 s fixed, the batch cap
        # has to leave the upstream call inside PREDICT_TIMEOUT_SECONDS.
        molecules = int(self._env(deployments)["COALESCE_MAX_MOLECULES"])
        worst_case = 124 + molecules * 25

        assert worst_case < PREDICT_TIMEOUT_SECONDS

    def test_in_flight_calls_track_the_replica_count(self, deployments):
        # 0 means "as many as there are healthy replicas", which keeps every
        # GPU busy while the surplus accumulates into bigger batches.
        assert self._env(deployments)["GATEWAY_MAX_INFLIGHT"] == "0"

    def test_the_upstream_timeout_outlives_a_prediction(self, deployments):
        timeout = float(self._env(deployments)["UPSTREAM_TIMEOUT_SECONDS"])

        assert timeout > PREDICT_TIMEOUT_SECONDS

    def test_in_flight_batches_survive_a_gateway_restart(self, deployments):
        # Async results are held in memory, and a coalesced batch is minutes of
        # GPU time.
        grace = _pod_spec(deployments[BOLTZ_GATEWAY])["terminationGracePeriodSeconds"]

        assert grace >= 600


# ---------------------------------------------------------------------------
# TestCampaignStore
# ---------------------------------------------------------------------------


class TestCampaignStore:
    """The control-plane ledger: one writer, on the volume, reachable by name."""

    def _env(self, deployments) -> dict[str, str]:
        return {
            var["name"]: var.get("value")
            for var in _containers(deployments[CAMPAIGN_STORE])[0]["env"]
        }

    def test_exactly_one_writer_on_the_ledger(self, deployments):
        # The atomic claim relies on SQLite's write lock. A second replica
        # would be a second process on the same file over a network
        # filesystem, which is where that locking stops being dependable --
        # and two supervisors would be handed the same weight vector.
        spec = deployments[CAMPAIGN_STORE]["spec"]

        assert spec["replicas"] == 1
        assert spec["strategy"]["type"] == "Recreate"

    def test_one_uvicorn_worker(self, deployments):
        command = _containers(deployments[CAMPAIGN_STORE])[0]["command"]

        assert "--workers" in command
        assert command[command.index("--workers") + 1] == "1"

    def test_the_ledger_lives_on_the_shared_volume(self, deployments):
        # It has to outlive the pod: a conductor that restarts rebuilds its
        # loop by reading the ledger back.
        db = self._env(deployments)["CAMPAIGN_STORE_DB"]
        mounts = {
            m["mountPath"]
            for m in _containers(deployments[CAMPAIGN_STORE])[0]["volumeMounts"]
        }

        assert db.startswith("/shared/")
        assert "/shared" in mounts

    def test_it_needs_no_gpu(self, deployments):
        assert _gpus(deployments[CAMPAIGN_STORE]) == 0

    def test_the_default_lease_outlives_one_rl_step(self, deployments):
        # A supervisor blocked in a scoring call cannot heartbeat, and the
        # gateway will hold that call for its whole upstream timeout. A lease
        # that expires inside a normal step hands the task to a second
        # supervisor and runs the same weight vector twice, so the two numbers
        # have to stay coupled.
        lease = float(self._env(deployments)["DEFAULT_LEASE_SECONDS"])
        gateway_env = {
            var["name"]: var.get("value")
            for var in _containers(deployments[BOLTZ_GATEWAY])[0]["env"]
        }

        assert lease > float(gateway_env["UPSTREAM_TIMEOUT_SECONDS"])

    def test_a_task_is_retried_but_not_forever(self, deployments):
        # One retry survives an eviction; unbounded retries would let a task
        # that fails for its own reasons cycle through the whole pool.
        assert int(self._env(deployments)["TASK_MAX_ATTEMPTS"]) == 2

    def test_it_is_reachable_by_the_name_clients_use(self, services, deployments):
        service = services[CAMPAIGN_STORE]
        ports = {p["name"]: p["port"] for p in service["spec"]["ports"]}
        container_ports = {
            p["name"]: p["containerPort"]
            for p in _containers(deployments[CAMPAIGN_STORE])[0]["ports"]
        }

        assert service["spec"]["type"] == "ClusterIP"
        assert ports == {"store-http": 8087}
        assert container_ports == {"store-http": 8087}

    def test_the_probes_use_the_health_endpoint(self, deployments):
        container = _containers(deployments[CAMPAIGN_STORE])[0]

        for probe in ("readinessProbe", "livenessProbe"):
            assert container[probe]["httpGet"]["path"] == "/health"
            assert container[probe]["httpGet"]["port"] == "store-http"


# ---------------------------------------------------------------------------
# TestSupervisorPool
# ---------------------------------------------------------------------------


class TestSupervisorPool:
    """N headless agents draining the ledger's queue."""

    def _env(self, deployments) -> dict[str, Any]:
        return {
            var["name"]: var
            for var in _containers(deployments[SUPERVISOR_DEPLOYMENT])[0]["env"]
        }

    def test_the_pool_starts_empty(self, deployments):
        # Applying the manifests must not start burning GPU against an empty
        # queue; a campaign begins by scaling up.
        assert deployments[SUPERVISOR_DEPLOYMENT]["spec"]["replicas"] == 0

    def test_it_is_the_agent_image_with_another_entrypoint(self, deployments):
        # A supervisor *is* the agent, so there is no second image to build and
        # no second copy of the agent to keep in step.
        container = _containers(deployments[SUPERVISOR_DEPLOYMENT])[0]
        agent_image = _containers(deployments[AGENT_DEPLOYMENT])[0]["image"]

        assert container["image"] == agent_image
        assert container["command"] == [
            "python", "-m", "reinvent4_agent.supervisor",
        ]

    def test_the_entrypoint_module_exists(self):
        # The command above is a string in YAML; nothing else would catch a
        # rename of the module it names.
        assert (
            _MANIFESTS.parents[1] / "reinvent4_agent" / "supervisor.py"
        ).is_file()

    def test_supervisors_need_no_gpu(self, deployments):
        # The point of the Phase 2 split: N supervisors cost N CPU
        # allocations, and the GPUs go to Boltz and the runner instead of
        # being pinned one per agent.
        assert _gpus(deployments[SUPERVISOR_DEPLOYMENT]) == 0

    def test_nothing_dials_a_supervisor(self, services):
        # It is a client of the ledger and the runner, so a Service would be
        # an invitation to depend on a pod that is scaled to zero most of the
        # time.
        assert SUPERVISOR_DEPLOYMENT not in services

    def test_it_knows_where_the_ledger_is(self, deployments, services):
        store_url = self._env(deployments)["CAMPAIGN_STORE_URL"]["value"]
        port = services[CAMPAIGN_STORE]["spec"]["ports"][0]["port"]

        assert store_url == f"http://{CAMPAIGN_STORE}:{port}"

    def test_each_supervisor_is_named_in_the_ledger(self, deployments):
        # So /status names the pod holding each task and a stuck supervisor
        # can be found by name.
        identity = self._env(deployments)["SUPERVISOR_ID"]

        assert identity["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name"

    def test_heartbeats_are_well_inside_the_lease(self, deployments):
        # A couple of missed heartbeats must not cost the task.
        interval = float(self._env(deployments)["SUPERVISOR_HEARTBEAT_SECONDS"]["value"])
        lease = float(
            {
                var["name"]: var.get("value")
                for var in _containers(deployments[CAMPAIGN_STORE])[0]["env"]
            }["DEFAULT_LEASE_SECONDS"]
        )

        assert interval * 3 < lease

    def test_it_shares_the_agents_ttp_store(self, deployments):
        # The base TTP is read from here and each task's weight-vector variant
        # written back, so both have to be the same directory.
        def ttp_mount(name: str) -> str:
            return next(
                m["mountPath"]
                for m in _containers(deployments[name])[0]["volumeMounts"]
                if m.get("subPath") == "ttps"
            )

        assert ttp_mount(SUPERVISOR_DEPLOYMENT) == ttp_mount(AGENT_DEPLOYMENT)

    def test_it_reads_the_same_llm_config_as_the_agent(self, deployments):
        # One place to change the model for the whole fleet.
        def llm_keys(name: str) -> set[str]:
            return {
                var["name"]
                for var in _containers(deployments[name])[0]["env"]
                if var["name"].startswith("REINVENT4_AGENT_")
            }

        assert llm_keys(SUPERVISOR_DEPLOYMENT) <= llm_keys(AGENT_DEPLOYMENT)
        assert "REINVENT4_AGENT_MODEL" in llm_keys(SUPERVISOR_DEPLOYMENT)

    def test_the_api_key_comes_from_the_secret(self, deployments):
        key = self._env(deployments)["REINVENT4_AGENT_API_KEY"]

        assert key["valueFrom"]["secretKeyRef"]["name"] == "dd-agentic-llm"
        assert "value" not in key

    def test_supervisors_run_the_driver_alone(self, deployments, mcp_config):
        # The Reviewer produces no molecules and roughly doubles the token
        # spend per task. Set literally rather than read from the ConfigMap,
        # whose "true" belongs to the interactive agent.
        multi_agent = self._env(deployments)["REINVENT4_AGENT_MULTI_AGENT"]

        assert multi_agent["value"] == "false"
        assert "valueFrom" not in multi_agent


# ---------------------------------------------------------------------------
# TestConductor
# ---------------------------------------------------------------------------


class TestConductor:
    """One conductor, with just enough Kubernetes access to scale the pool."""

    def _env(self, deployments) -> dict[str, Any]:
        return {
            var["name"]: var
            for var in _containers(deployments[CONDUCTOR_DEPLOYMENT])[0]["env"]
        }

    def test_exactly_one_conductor(self, deployments):
        # Two would each top the pool up to N and the campaign would run at
        # 2N -- the one way to overcommit the GPUs that the ledger cannot
        # catch, because each dispatch is individually legitimate.
        spec = deployments[CONDUCTOR_DEPLOYMENT]["spec"]

        assert spec["replicas"] == 1
        assert spec["strategy"]["type"] == "Recreate"

    def test_it_has_its_own_service_account(self, deployments):
        # The deployments/scale grant in k8s/rbac.yaml is bound to the
        # oidc:demo-lux developer group, so it authorises a human's kubectl
        # and not a pod. Without this the scale call is a 403.
        pod = _pod_spec(deployments[CONDUCTOR_DEPLOYMENT])

        assert pod["serviceAccountName"] == "dd-agentic-conductor"
        assert "dd-agentic-conductor" in _by_kind("ServiceAccount")

    def test_the_scale_grant_is_scoped_to_the_supervisor_pool(self):
        # So a conductor cannot scale Boltz or the runner, which hold the GPUs.
        role = _by_kind("Role")["dd-agentic-conductor"]
        rules = role["rules"]

        assert len(rules) == 1
        assert rules[0]["resources"] == ["deployments/scale"]
        assert rules[0]["resourceNames"] == [SUPERVISOR_DEPLOYMENT]

    def test_the_grant_withholds_everything_but_read_and_patch(self):
        rules = _by_kind("Role")["dd-agentic-conductor"]["rules"][0]

        # `list` and `watch` cannot be restricted by resourceNames, so
        # granting them would widen the scope above to every Deployment.
        assert set(rules["verbs"]) == {"get", "patch"}

    def test_the_binding_names_the_service_account(self):
        binding = _by_kind("RoleBinding")["dd-agentic-conductor"]

        assert binding["roleRef"]["name"] == "dd-agentic-conductor"
        assert binding["subjects"] == [
            {"kind": "ServiceAccount", "name": "dd-agentic-conductor"}
        ]

    def test_it_knows_which_deployment_to_scale(self, deployments):
        assert self._env(deployments)["SUPERVISOR_DEPLOYMENT"]["value"] == (
            SUPERVISOR_DEPLOYMENT
        )

    def test_it_knows_its_own_namespace(self, deployments):
        # The scale call is namespaced, and the manifests carry no hard-coded
        # namespace -- apply.sh sets it.
        namespace = self._env(deployments)["POD_NAMESPACE"]

        assert namespace["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.namespace"

    def test_admission_control_points_at_the_gateway(self, deployments, services):
        # Boltz is the binding constraint, so capacity is a question about its
        # queue depth rather than about the agent count.
        stats_url = self._env(deployments)["BOLTZ_GATEWAY_STATS_URL"]["value"]
        port = services[BOLTZ_GATEWAY]["spec"]["ports"][0]["port"]

        assert stats_url == f"http://{BOLTZ_GATEWAY}:{port}/stats"

    def test_it_reads_the_ledger_it_writes_to(self, deployments, services):
        store_url = self._env(deployments)["CAMPAIGN_STORE_URL"]["value"]
        port = services[CAMPAIGN_STORE]["spec"]["ports"][0]["port"]

        assert store_url == f"http://{CAMPAIGN_STORE}:{port}"

    def _data_mounts(self, deployments) -> dict[str, Any]:
        return {
            m["mountPath"]: m
            for m in _containers(deployments[CONDUCTOR_DEPLOYMENT])[0]["volumeMounts"]
            if m["name"] == "data"
        }

    def test_results_and_the_shared_volume_are_read_only(self, deployments):
        # It reviews what the fleet produced; the supervisors are what write it.
        mounts = self._data_mounts(deployments)

        assert mounts, "the conductor should still be able to read results"
        for path in ("/shared", "/data/reinvent4_mcp/jobs"):
            assert mounts[path]["readOnly"] is True, f"{path} is writable"

    def test_it_can_write_the_ttp_it_opens_a_campaign_on(self, deployments):
        # Defining the base TTP is the first step of a campaign on a new
        # target, and the conductor is who opens campaigns. Read-only here
        # meant create_ttp failed with EROFS and a campaign could only ever
        # run against a profile somebody else had made.
        mount = self._data_mounts(deployments)["/root/.reinvent4_mcp/ttps"]

        assert not mount.get("readOnly")

    def test_the_writable_ttp_mount_comes_with_the_edit_tools_withheld(
        self, deployments
    ):
        # The pair is the guarantee: creating a base TTP is how a campaign
        # starts, rewriting one mid-campaign changes what results already in
        # the ledger mean. Relax either half alone and that is gone.
        denied = set(
            self._env(deployments)["REINVENT4_AGENT_DENY_TOOLS"]["value"].split(",")
        )

        assert {"update_ttp", "delete_ttp"} <= denied
        assert "create_ttp" not in denied

    def test_it_needs_no_gpu(self, deployments):
        assert _gpus(deployments[CONDUCTOR_DEPLOYMENT]) == 0

    def test_it_stays_interactive(self, deployments):
        # The conductor is the REPL entrypoint, reached by kubectl exec, so it
        # must not exit when a campaign's loop finishes.
        assert _containers(deployments[CONDUCTOR_DEPLOYMENT])[0]["command"] == [
            "sleep", "infinity",
        ]

    def test_the_loop_module_exists(self):
        assert (
            _MANIFESTS.parents[1] / "reinvent4_agent" / "conductor.py"
        ).is_file()

    def test_the_conductor_runs_without_a_reviewer(self, deployments):
        # It steers a fleet; it does not need a second opinion on its own turn.
        multi_agent = self._env(deployments)["REINVENT4_AGENT_MULTI_AGENT"]

        assert multi_agent["value"] == "false"

    def test_the_interactive_agent_keeps_its_reviewer(self, mcp_config):
        # Turning it off for the fleet must not turn it off for the person at
        # the REPL, which is what reading the shared ConfigMap key would do.
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]

        assert configmap["data"]["REINVENT4_AGENT_MULTI_AGENT"] == "true"


# ---------------------------------------------------------------------------
# TestAimTiers
# ---------------------------------------------------------------------------


class TestAimTiers:
    """Two LLM endpoints, because the two roles want opposite things.

    The conductor takes one turn with a person waiting on it (latency); the
    supervisors are N agents grinding through tool calls with nobody watching
    any single response (throughput). B10 was that the original single
    AIMService is sized for one interactive session, which is the opposite of
    what N supervisors do.
    """

    @pytest.fixture(scope="class")
    def aim_services(self) -> dict[str, dict[str, Any]]:
        return _by_kind("AIMService")

    def test_there_are_two_tiers(self, aim_services):
        assert set(aim_services) == {
            "dd-agentic-aim-conductor", "dd-agentic-aim-supervisor",
        }

    def test_each_tier_has_a_stable_service(self, services):
        # AIM Engine generates a KServe Service name; these select the same
        # predictor pods so the agents have a repeatable base URL.
        for tier in ("dd-agentic-aim-conductor", "dd-agentic-aim-supervisor"):
            assert tier in services
            assert services[tier]["spec"]["selector"][
                "aim.eai.amd.com/service.name"
            ] == tier

    def test_the_conductor_tier_is_tuned_for_latency(self, aim_services):
        profile = aim_services["dd-agentic-aim-conductor"]["spec"]["profile"]["name"]

        assert "laten" in profile
        assert "tp1" in profile, "an 8-GPU conductor model would starve Boltz"

    def test_the_supervisor_tier_is_tuned_for_throughput(self, aim_services):
        # The half of the split that matters: the shared endpoint they used to
        # have was tuned for the opposite workload.
        profile = aim_services["dd-agentic-aim-supervisor"]["spec"]["profile"]["name"]

        assert "throu" in profile
        assert "tp1" in profile

    def test_both_tiers_fit_one_gpu_each(self, aim_services):
        # What keeps the split affordable: §3.4 budgets 1 GPU per tier against
        # 5 for Boltz, and Boltz is the measured bottleneck.
        for name, svc in aim_services.items():
            assert "tp1" in svc["spec"]["profile"]["name"], name
            assert svc["spec"]["replicas"] >= 1, name

    def test_the_conductor_dials_the_conductor_tier(self, deployments, mcp_config):
        env = {
            var["name"]: var
            for var in _containers(deployments[CONDUCTOR_DEPLOYMENT])[0]["env"]
        }
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]

        assert env["REINVENT4_AGENT_BASE_URL"]["valueFrom"]["configMapKeyRef"][
            "key"
        ] == "CONDUCTOR_AGENT_BASE_URL"
        assert configmap["CONDUCTOR_AGENT_BASE_URL"] == (
            "http://dd-agentic-aim-conductor/v1"
        )

    def test_the_supervisors_dial_the_throughput_tier(self, deployments):
        env = {
            var["name"]: var
            for var in _containers(deployments[SUPERVISOR_DEPLOYMENT])[0]["env"]
        }
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]

        assert env["REINVENT4_AGENT_BASE_URL"]["valueFrom"]["configMapKeyRef"][
            "key"
        ] == "REINVENT4_AGENT_BASE_URL"
        assert configmap["REINVENT4_AGENT_BASE_URL"] == (
            "http://dd-agentic-aim-supervisor/v1"
        )

    def test_the_two_tiers_are_not_the_same_endpoint(self):
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]

        assert (
            configmap["CONDUCTOR_AGENT_BASE_URL"]
            != configmap["REINVENT4_AGENT_BASE_URL"]
        ), "splitting the tiers is pointless if both dial one endpoint"

    def test_nothing_still_dials_the_old_single_endpoint(self):
        # The pre-split Service was `dd-agentic-aim`; a leftover reference
        # would resolve to nothing now.
        for path in _manifest_files():
            body = path.read_text()
            assert "http://dd-agentic-aim/v1" not in body, path

    def test_the_conductor_gets_the_models_whole_context(self, aim_services):
        """32768 was the profile's choice; the weights go to 131072.

        vLLM derives ``max_tokens`` as max-model-len minus the prompt, so the
        profile default turned a front-plus-molecules question into
        "max_tokens must be at least 1, got -72257".
        """
        overrides = aim_services["dd-agentic-aim-conductor"]["spec"][
            "profileOverrides"
        ]["engineArgs"]

        assert overrides["max-model-len"] == 131072

    def test_the_conductor_needs_no_rope_scaling_to_get_there(self, aim_services):
        """Unlike Qwen3-32B, gpt-oss-120b carries its own YaRN in config.json.

        Supplying hf-overrides here would fight the shipped weights rather
        than extend them.
        """
        overrides = aim_services["dd-agentic-aim-conductor"]["spec"][
            "profileOverrides"
        ]["engineArgs"]

        assert "hf-overrides" not in overrides

    def test_the_context_bump_is_not_copied_to_the_supervisors(self, aim_services):
        """Context trades against batch slots, and N supervisors need slots."""
        supervisor = aim_services["dd-agentic-aim-supervisor"]["spec"][
            "profileOverrides"
        ]["engineArgs"]

        assert supervisor["max-model-len"] < 131072


# ---------------------------------------------------------------------------
# TestConfigMapServiceUrls
# ---------------------------------------------------------------------------


class TestConfigMapServiceUrls:
    """Scorer URLs are cluster DNS, not the old in-pod loopback."""

    def _urls(self, mcp_config) -> dict[str, str]:
        return {
            name: section["service_url"]
            for name, section in mcp_config["scoring"].items()
            if isinstance(section, dict) and "service_url" in section
        }

    def test_every_objective_dials_its_expected_endpoint(self, mcp_config):
        urls = self._urls(mcp_config)

        for objective, (host, port) in SCORER_ENDPOINTS.items():
            assert urls[objective] == f"http://{host}:{port}/score"

    def test_no_loopback_addresses_anywhere_in_the_manifests(self):
        for path in _manifest_files():
            text = path.read_text()
            assert "127.0.0.1" not in text, f"{path.name} still dials loopback"
            assert "localhost" not in text, f"{path.name} still dials localhost"

    def test_every_scorer_is_configured(self, mcp_config):
        assert len(self._urls(mcp_config)) == len(SCORER_ENDPOINTS)

    def test_each_url_resolves_to_a_service_that_exists(self, mcp_config, services):
        for name, url in self._urls(mcp_config).items():
            match = re.fullmatch(r"http://([a-z0-9-]+):(\d+)/score", url)
            assert match, f"{name}: unexpected URL {url}"

            host, port = match.group(1), int(match.group(2))
            assert host in services, f"{name}: no Service named {host}"
            published = {p["port"] for p in services[host]["spec"]["ports"]}
            assert port in published, f"{name}: {host} does not publish {port}"

    def test_urls_are_unqualified_so_the_namespace_is_free(self, mcp_config):
        # apply.sh --namespace decides where this lands; a hard-coded
        # <svc>.<ns>.svc.cluster.local would pin it to one namespace.
        for name, url in self._urls(mcp_config).items():
            assert ".svc" not in url, f"{name} hard-codes a namespace"

    def test_boltz_cache_dir_still_matches_its_mount(self, mcp_config, deployments):
        cache_dir = mcp_config["scoring"]["boltz"]["cache_dir"]
        container = _containers(deployments["dd-agentic-boltz"])[0]
        mounts = {m["mountPath"] for m in container["volumeMounts"]}

        assert cache_dir in mounts


# ---------------------------------------------------------------------------
# TestSharedVolumePinning
# ---------------------------------------------------------------------------


class TestSharedVolumePinning:
    """ReadWriteOnce is per node, so the /shared users must share one node."""

    def test_apply_pins_every_deployment(self, apply_patches, deployments):
        targets = [p["target"] for p in apply_patches if p["target"]["kind"] == "Deployment"]
        assert targets, "apply.sh no longer patches Deployments"

        # No name or selector, so every Deployment the overlay carries is
        # pinned -- including ones added after this test was written.
        for target in targets:
            assert "name" not in target
            assert "labelSelector" not in target

    def test_the_pin_sets_the_hostname_node_selector(self, apply_patches):
        # The inline patch is a nested YAML document kustomize parses itself.
        ops = [op for p in apply_patches for op in yaml.safe_load(p["patch"])]
        pins = [
            op for op in ops
            if op["path"] == "/spec/template/spec/nodeSelector"
            and "kubernetes.io/hostname" in op["value"]
        ]
        # nodeName is rejected by some schedulers while scheduling gates are
        # set, so the hostname label is the mechanism.
        assert len(pins) == 1

    def test_the_shared_volume_users_are_the_expected_set(self, deployments):
        mounting = {n for n, d in deployments.items() if _mounts_shared(d)}

        assert mounting == SHARED_VOLUME_DEPLOYMENTS

    def test_shared_mounts_come_from_the_one_claim(self, deployments):
        for name in SHARED_VOLUME_DEPLOYMENTS:
            volumes = {v["name"]: v for v in _pod_spec(deployments[name])["volumes"]}
            claim = volumes["data"]["persistentVolumeClaim"]["claimName"]

            assert claim == "dd-agentic-data"

    def test_shared_is_mounted_at_the_same_path_everywhere(self, deployments):
        # structure_paths are absolute and handed between pods verbatim.
        for name in SHARED_VOLUME_DEPLOYMENTS:
            paths = [
                m["mountPath"]
                for c in _containers(deployments[name])
                for m in c["volumeMounts"]
                if m.get("subPath") == "shared"
            ]
            assert paths == ["/shared"]

    def test_shared_is_never_mounted_read_only(self, deployments):
        # PoseBusters extracts the ligand to <structure>_lig.sdf *beside* the
        # PDB Boltz handed it, i.e. into the producer's own output directory.
        # A read-only mount would make every pose fail extraction and score
        # None -- which reads exactly like a missing file.
        for name in STRUCTURE_WRITERS:
            for container in _containers(deployments[name]):
                for mount in container["volumeMounts"]:
                    if mount.get("subPath") != "shared":
                        continue
                    assert not mount.get("readOnly"), f"{name} mounts /shared read-only"

    def test_the_claim_is_never_readwriteoncepod(self):
        pvc = _by_kind("PersistentVolumeClaim")["dd-agentic-data"]

        # RWO is per node and fine for a pinned fleet; RWOPod is per *pod* and
        # would admit exactly one of these Deployments.
        assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]

    def test_the_cpu_only_scorers_need_no_shared_volume(self, deployments):
        # They take SMILES in and return scores, so nothing is handed over by
        # path -- they are free to float once the images are in a registry.
        for name in ("dd-agentic-admet", "dd-agentic-retrodmta"):
            assert not _mounts_shared(deployments[name])
            assert "volumes" not in _pod_spec(deployments[name])


# ---------------------------------------------------------------------------
# TestKustomization
# ---------------------------------------------------------------------------


class TestKustomization:

    @pytest.fixture(scope="class")
    def kustomization(self) -> dict[str, Any]:
        return yaml.safe_load((_MANIFESTS / "kustomization.yaml").read_text())

    def test_every_manifest_file_is_a_resource(self, kustomization):
        listed = set(kustomization["resources"])
        on_disk = {str(p.relative_to(_MANIFESTS)) for p in _manifest_files()}

        assert on_disk == listed

    def test_part_of_label_is_stamped_on_everything(self, kustomization):
        # apply.sh restarts and waits on this selector instead of naming each
        # Deployment.
        pairs = {k: v for entry in kustomization["labels"] for k, v in entry["pairs"].items()}

        assert pairs["app.kubernetes.io/part-of"] == "dd-agentic"
        assert "app.kubernetes.io/part-of=dd-agentic" in _APPLY_SH.read_text()

    def test_the_part_of_label_reaches_the_pods(self, kustomization):
        # Without includeTemplates the label lands on the Deployment only, so
        # `kubectl get pods --selector app.kubernetes.io/part-of=dd-agentic`
        # silently matches nothing -- and `wait --for=delete` returns at once,
        # which would make the scale-down step in k8s/README.md a no-op.
        assert all(entry.get("includeTemplates") for entry in kustomization["labels"])

    def test_the_label_transformer_leaves_selectors_alone(self, kustomization):
        # A Deployment's spec.selector is immutable once created, so stamping
        # it would make every future apply fail on the live objects.
        assert not any(entry.get("includeSelectors") for entry in kustomization["labels"])

    def test_every_locally_built_image_is_remapped_by_tag(self, kustomization):
        mapped = {entry["name"] for entry in kustomization["images"]}
        used = {
            c["image"].split(":")[0]
            for d in _by_kind("Deployment").values()
            for c in _containers(d) + (_pod_spec(d).get("initContainers") or [])
            if c["image"].endswith(":local")
        }

        assert used <= mapped

    def test_the_only_third_party_image_is_the_gateway_proxy(self):
        # Everything else is built here and imported into the node's kubelet
        # store, so a new registry dependency should be a deliberate decision.
        registry_images = {
            c["image"]
            for d in _by_kind("Deployment").values()
            for c in _containers(d) + (_pod_spec(d).get("initContainers") or [])
            if not c["image"].endswith(":local")
        }

        assert registry_images == {"haproxy:3.2-alpine"}, (
            "the proxy is off-the-shelf; anything else from a registry needs a "
            "reason and an entry in k8s/README.md"
        )

    def test_images_are_never_pulled_from_a_registry(self):
        # The *:local tags are imported into the node's kubelet store.
        for deployment in _by_kind("Deployment").values():
            spec = _pod_spec(deployment)
            for container in spec["containers"] + (spec.get("initContainers") or []):
                assert container["imagePullPolicy"] == "IfNotPresent"


# ---------------------------------------------------------------------------
# TestScriptTargets
# ---------------------------------------------------------------------------


class TestScriptTargets:
    """The helper scripts must name Deployments and containers that exist."""

    def test_ab_harness_targets_the_boltz_deployment(self, deployments):
        script = (_ROOT / "k8s" / "scripts" / "ab-boltz-kernels.py").read_text()
        deployment = re.search(r'"--deployment", default="([\w-]+)"', script).group(1)
        container = re.search(r'"--container", default="([\w-]+)"', script).group(1)

        assert deployment == "dd-agentic-boltz"
        names = [c["name"] for c in _containers(deployments[deployment])]
        assert container in names

    def test_handoff_harness_knows_the_producer_and_its_consumers(self, deployments, services):
        script = (_ROOT / "k8s" / "scripts" / "verify-artifact-handoff.py").read_text()
        pairs = re.findall(r'"(dd-agentic-[\w-]+)", "([\w-]+)"\)', script)

        # Every Deployment/container pair it `kubectl exec`s into must exist,
        # and must be one of the three that mount /shared.
        assert len(pairs) == 3
        for deployment, container in pairs:
            assert deployment in services
            assert deployment in SHARED_VOLUME_DEPLOYMENTS
            assert container in [c["name"] for c in _containers(deployments[deployment])]

    def test_uat_targets_the_agent(self, deployments):
        uat = (_ROOT / "k8s" / "scripts" / "uat-scoring.sh").read_text()

        assert f'DEPLOYMENT="{AGENT_DEPLOYMENT}"' in uat
        assert 'CONTAINER="dd-agent"' in uat

        names = [c["name"] for c in _containers(deployments[AGENT_DEPLOYMENT])]
        assert names == ["dd-agent"]

    def test_the_model_switch_names_both_tiers(self, deployments):
        """It drives two endpoints, so nothing in it may be a single default."""
        switch = (_ROOT / "k8s" / "scripts" / "switch-aim-model.py").read_text()

        assert '"--tier"' in switch
        assert "TIERS" in switch
        # A leftover single-service constant would silently repoint the wrong
        # endpoint while reporting the one that was asked for.
        assert "AIM_SERVICE_NAME" not in switch
        assert "DEPLOYMENT_NAME" not in switch

    def test_each_tier_names_real_objects(self, deployments):
        """A tier whose clients are misnamed leaves them on the old model.

        Read from the script's own TIERS table rather than grepped, so the
        test fails when the table drifts rather than when the file's prose
        changes.
        """
        import importlib.util

        path = _ROOT / "k8s" / "scripts" / "switch-aim-model.py"
        spec = importlib.util.spec_from_file_location("switch_aim_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        aim_services = _by_kind("AIMService")
        configmap = _by_kind("ConfigMap")["dd-agentic-config"]["data"]

        assert set(module.TIERS) == {"conductor", "supervisor"}
        for tier, wiring in module.TIERS.items():
            assert wiring["service"] in aim_services, tier
            assert wiring["model_key"] in configmap, tier
            for deployment in wiring["deployments"]:
                assert deployment in deployments, f"{tier}: {deployment}"
