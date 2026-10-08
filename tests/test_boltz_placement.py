"""Tests for the Boltz placement harness and the gateway it measures.

A replica serves one prediction at a time, so the gateway has to place each
batch on an idle replica and queue the surplus rather than doubling up.
``k8s/scripts/verify-boltz-placement.py`` proves that against a live cluster;
these tests hold the parts that can be checked without one:

- reading HAProxy's CSV stats, including the backend row where an unplaceable
  request actually waits (the per-server ``qcur`` stays 0 the whole time, which
  is easy to mistake for "nothing queued")
- the verdict: doubling up, a queue timeout, and a stacked replica are each
  reported as their own failure
- that the request the harness sends is one Boltz accepts, and that its
  defaults match the gateway manifest

No cluster, service, or GPU: the stats endpoint and the HTTP calls are stubbed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "verify_boltz_placement", str(_ROOT / "k8s" / "scripts" / "verify-boltz-placement.py")
)
placement = importlib.util.module_from_spec(_spec)
sys.modules["verify_boltz_placement"] = placement
_spec.loader.exec_module(placement)

_boltz_spec = importlib.util.spec_from_file_location(
    "boltz2_app_for_placement", str(_ROOT / "services" / "boltz2" / "app.py")
)
boltz_app = importlib.util.module_from_spec(_boltz_spec)
sys.modules["boltz2_app_for_placement"] = boltz_app
_boltz_spec.loader.exec_module(boltz_app)

_PROXY_MANIFEST = _ROOT / "k8s" / "manifests" / "scorers" / "boltz-proxy.yaml"

# Trimmed to the columns the harness reads, in HAProxy's own order. Two
# replicas up, ten slots still awaiting DNS, and one request waiting in the
# backend queue.
STATS_CSV = """\
# pxname,svname,qcur,qmax,scur,smax,slim,stot,status
score,FRONTEND,,,1,2,1024,7,OPEN
boltz,boltz1,0,0,1,1,1,3,UP
boltz,boltz2,0,0,1,1,1,3,UP
boltz,boltz3,0,0,0,0,1,0,MAINT (resolution)
boltz,BACKEND,1,1,2,2,100,7,UP
other,somesrv,9,9,9,9,1,9,UP
"""


def _stats_response(csv: str):
    class _Response:
        def read(self):
            return csv.encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Response()


@pytest.fixture
def stub_stats(monkeypatch):
    """Serve a canned CSV from the stats endpoint."""

    def _install(csv: str = STATS_CSV):
        monkeypatch.setattr(
            placement.urllib.request, "urlopen",
            lambda url, timeout=None: _stats_response(csv),
        )

    return _install


def _call(label: str, replica: Optional[str], status: Optional[int] = 200,
          elapsed: float = 200.0, error: Optional[str] = None) -> placement.Call:
    call = placement.Call(label=label)
    call.started = 0.0
    call.finished = elapsed
    call.status = status
    call.replica = replica
    call.error = error
    return call


class _Sampler:
    """Stands in for StatsSampler with fixed high-water marks."""

    def __init__(self, in_flight: dict[str, int], queued: int = 0, samples: int = 10):
        self.max_in_flight = in_flight
        self.max_queued = queued
        self.samples = samples

    @property
    def doubled_up(self) -> list[str]:
        return [name for name, peak in self.max_in_flight.items() if peak > 1]


# ---------------------------------------------------------------------------
# TestStatsParsing
# ---------------------------------------------------------------------------


class TestStatsParsing:
    """The CSV is indexed by column name, and the backend row is not dropped."""

    def test_only_this_backends_replicas_are_returned(self, stub_stats):
        stub_stats()

        servers, _pending = placement.read_stats("http://x:8404", "boltz")

        assert [s.name for s in servers] == ["boltz1", "boltz2", "boltz3"]

    def test_the_backend_row_is_kept_separately(self, stub_stats):
        # Where a request waits when no replica is free. Reading only the
        # server rows reports a queue depth of 0 while a batch is queued.
        stub_stats()

        _servers, pending = placement.read_stats("http://x:8404", "boltz")

        assert pending is not None
        assert pending.queued == 1

    def test_per_server_queue_stays_zero_while_the_backend_queues(self, stub_stats):
        stub_stats()

        servers, pending = placement.read_stats("http://x:8404", "boltz")

        assert [s.queued for s in servers] == [0, 0, 0]
        assert pending.queued == 1

    def test_in_flight_is_read_per_replica(self, stub_stats):
        stub_stats()

        servers, _pending = placement.read_stats("http://x:8404", "boltz")

        assert {s.name: s.in_flight for s in servers} == {
            "boltz1": 1, "boltz2": 1, "boltz3": 0,
        }

    def test_only_replicas_that_are_up_count_as_ready(self, stub_stats):
        # Unresolved slots sit in MAINT until DNS fills them.
        stub_stats()

        assert placement.ready_replicas("http://x:8404", "boltz") == ["boltz1", "boltz2"]

    def test_an_unreachable_endpoint_reads_as_nothing(self, monkeypatch):
        def _boom(url, timeout=None):
            raise OSError("connection refused")

        monkeypatch.setattr(placement.urllib.request, "urlopen", _boom)

        assert placement.read_stats("http://x:8404", "boltz") == ([], None)
        assert placement.ready_replicas("http://x:8404", "boltz") == []

    def test_a_csv_without_the_expected_columns_reads_as_nothing(self, stub_stats):
        stub_stats("# something,else\nfoo,bar\n")

        assert placement.read_stats("http://x:8404", "boltz") == ([], None)


# ---------------------------------------------------------------------------
# TestVerdict
# ---------------------------------------------------------------------------


class TestVerdict:
    """Each way placement can fail is reported as itself."""

    def test_one_batch_per_replica_is_proven(self):
        calls = [_call("a", "boltz1"), _call("b", "boltz2")]

        proven, reason = placement.decide(
            calls, ["boltz1", "boltz2"], _Sampler({"boltz1": 1, "boltz2": 1}),
        )

        assert proven is True
        assert "spread over 2 of 2 replicas" in reason

    def test_the_surplus_is_reported_as_queued(self):
        # Three calls, two replicas: the third waited instead of doubling up.
        calls = [_call("a", "boltz1"), _call("b", "boltz2"), _call("c", "boltz2")]

        proven, reason = placement.decide(
            calls, ["boltz1", "boltz2"], _Sampler({"boltz1": 1, "boltz2": 1}, queued=1),
        )

        assert proven is True
        assert "waited in the queue (depth 1)" in reason

    def test_a_surplus_that_never_queued_is_hedged(self):
        calls = [_call("a", "boltz1"), _call("b", "boltz2"), _call("c", "boltz2")]

        proven, reason = placement.decide(
            calls, ["boltz1", "boltz2"], _Sampler({"boltz1": 1, "boltz2": 1}, queued=0),
        )

        assert proven is True
        assert "never seen waiting" in reason

    def test_two_batches_on_one_replica_disproves_it(self):
        proven, reason = placement.decide(
            [_call("a", "boltz1"), _call("b", "boltz1")],
            ["boltz1", "boltz2"],
            _Sampler({"boltz1": 2, "boltz2": 0}),
        )

        assert proven is False
        assert "maxconn 1 is not holding" in reason

    def test_an_idle_replica_left_unused_disproves_it(self):
        # Both served, neither doubled up, but one replica never got work
        # while the other took both -- the round-robin failure mode.
        calls = [_call("a", "boltz1"), _call("b", "boltz1")]

        proven, reason = placement.decide(
            calls, ["boltz1", "boltz2"], _Sampler({"boltz1": 1, "boltz2": 0}),
        )

        assert proven is False
        assert "only 1 of 2 replicas" in reason

    def test_a_queue_timeout_names_itself(self):
        # 503 means the gateway gave up waiting, which downstream becomes a
        # whole batch of NaN rather than a slow success.
        calls = [_call("a", "boltz1"), _call("b", None, status=503, error="HTTP 503")]

        proven, reason = placement.decide(
            calls, ["boltz1"], _Sampler({"boltz1": 1}),
        )

        assert proven is False
        assert "timeout queue" in reason

    def test_other_failures_are_not_blamed_on_the_queue(self):
        calls = [_call("a", None, status=500, error="HTTP 500: boom")]

        proven, reason = placement.decide(calls, ["boltz1"], _Sampler({"boltz1": 1}))

        assert proven is False
        assert "failed outright" in reason

    def test_a_missing_replica_header_is_inconclusive(self):
        # Without it the responses say nothing about placement, which usually
        # means a replica was dialled directly instead of the gateway.
        proven, reason = placement.decide(
            [_call("a", None)], ["boltz1"], _Sampler({"boltz1": 1}),
        )

        assert proven is None
        assert placement.REPLICA_HEADER in reason

    def test_no_replica_up_is_inconclusive(self):
        proven, reason = placement.decide([_call("a", "boltz1")], [], _Sampler({}))

        assert proven is None
        assert "no replica up" in reason

    def test_nothing_sent_is_inconclusive(self):
        proven, _reason = placement.decide([], ["boltz1"], _Sampler({}))

        assert proven is None


# ---------------------------------------------------------------------------
# TestRequest
# ---------------------------------------------------------------------------


class TestRequest:

    def _args(self, **overrides: Any) -> SimpleNamespace:
        defaults = dict(
            protein_sequence=placement.BRD4_BD1_SEQUENCE,
            msa_path=placement.DEFAULT_MSA_PATH,
            recycling_steps=1,
            sampling_steps=50,
            sampling_steps_affinity=50,
            diffusion_samples_affinity=1,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_boltz_accepts_the_payload(self):
        payload = placement.build_payload(self._args(), ["CCO"])

        request = boltz_app.ScoreRequest(**payload)

        assert request.params.msa_path == placement.DEFAULT_MSA_PATH
        assert request.params.use_msa_server is False
        assert boltz_app._validate_protein(request) == []

    def test_the_panel_molecules_all_fold(self):
        for name, smiles in placement.PANEL:
            ok, reason = boltz_app._validate_smiles(smiles)
            assert ok, f"{name}: {reason}"

    def test_the_molecules_are_distinct(self):
        # A result cache must not be able to answer these instead of a
        # replica, or there would be no placement to observe.
        assert len({smiles for _name, smiles in placement.PANEL}) == len(placement.PANEL)

    def test_defaults_are_cheaper_than_production(self):
        # This measures placement; ab-boltz-kernels.py owns per-batch latency.
        payload = placement.build_payload(self._args(), ["CCO"])
        params = payload["params"]

        assert params["recycling_steps"] < 3
        assert params["diffusion_samples_affinity"] < 5


# ---------------------------------------------------------------------------
# TestAgainstTheManifest
# ---------------------------------------------------------------------------


class TestAgainstTheManifest:
    """The harness has to describe the proxy that is actually deployed."""

    @pytest.fixture(scope="class")
    def proxy(self) -> dict[str, Any]:
        docs = [d for d in yaml.safe_load_all(_PROXY_MANIFEST.read_text()) if d]
        return {d["kind"]: d for d in docs}

    def test_the_backend_name_matches_the_config(self, proxy):
        # The harness filters stats rows by this name, so a renamed backend
        # would silently report no replicas rather than a failure.
        cfg = proxy["ConfigMap"]["data"]["haproxy.cfg"]

        assert f"backend {placement.DEFAULT_BACKEND}" in cfg

    def test_the_stats_port_matches_the_service(self, proxy):
        ports = {
            port["name"]: port["port"] for port in proxy["Service"]["spec"]["ports"]
        }

        assert placement.DEFAULT_STATS_URL.endswith(f":{ports['stats-http']}")
        assert placement.DEFAULT_URL.endswith(f":{ports['boltz-http']}")

    def test_the_replica_header_is_the_one_the_proxy_sets(self, proxy):
        cfg = proxy["ConfigMap"]["data"]["haproxy.cfg"]

        assert f"set-header {placement.REPLICA_HEADER} " in cfg

    def test_the_timeout_outlasts_the_proxys_queue(self, proxy):
        cfg = proxy["ConfigMap"]["data"]["haproxy.cfg"]
        queue = next(
            int(line.split()[2].rstrip("s"))
            for line in cfg.splitlines()
            if line.split()[:2] == ["timeout", "queue"]
        )

        # Giving up before the proxy does would report a queued batch as a
        # client-side failure, which is the confusion this whole phase removes.
        assert placement.DEFAULT_TIMEOUT >= queue


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:

    def test_no_replicas_exits_inconclusive(self, monkeypatch, capsys):
        monkeypatch.setattr(placement, "ready_replicas", lambda url, backend: [])

        assert placement.main([]) == 2
        assert "no replica reported up" in capsys.readouterr().err

    def test_more_requests_than_molecules_is_refused(self, monkeypatch):
        monkeypatch.setattr(placement, "ready_replicas",
                            lambda url, backend: ["boltz1", "boltz2"])

        with pytest.raises(SystemExit):
            placement.main(["--requests", str(len(placement.PANEL) + 1)])

    def test_zero_requests_is_refused(self, monkeypatch):
        monkeypatch.setattr(placement, "ready_replicas",
                            lambda url, backend: ["boltz1"])

        with pytest.raises(SystemExit):
            placement.main(["--requests", "0"])

    def test_default_is_one_more_request_than_replicas(self, monkeypatch, capsys):
        # So the queue is exercised rather than only the placement.
        monkeypatch.setattr(placement, "ready_replicas",
                            lambda url, backend: ["boltz1", "boltz2"])
        monkeypatch.setattr(placement, "read_stats", lambda *a, **k: ([], None))
        fired: list[int] = []

        def _fire(args, batch):
            fired.append(len(batch))
            return [_call(name, "boltz1") for name, _ in batch]

        monkeypatch.setattr(placement, "fire_concurrently", _fire)
        placement.main([])

        assert fired == [3]
        capsys.readouterr()
