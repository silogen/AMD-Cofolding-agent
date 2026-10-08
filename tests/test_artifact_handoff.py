"""Tests for the cross-pod structure handoff and its verification harness.

Boltz hands PoseBusters and the interaction profiler an absolute path on
``/shared`` -- never the structure itself -- so the consumers only work if they
can open the producer's files. ``k8s/scripts/verify-artifact-handoff.py`` proves
that against a live cluster; these tests hold the parts that can be checked
without one:

- the paths the harness refuses to call portable (relative, or off ``/shared``)
- that the payloads it relays are the ones ``ExternalRESTScorer`` really sends,
  and that both consumer services accept them -- including the ``None`` holes a
  partial batch leaves behind
- that a consumer answering 200-with-nulls is reported as a storage failure,
  and a consumer answering 4xx as a config one
- that a green run on one pod is not mistaken for a cross-pod proof

No cluster, service, or GPU: kubectl and the HTTP calls are stubbed.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

import pydantic
import pytest
import yaml

_ROOT = Path(__file__).resolve().parent.parent

_handoff_spec = importlib.util.spec_from_file_location(
    "verify_artifact_handoff",
    str(_ROOT / "k8s" / "scripts" / "verify-artifact-handoff.py"),
)
handoff = importlib.util.module_from_spec(_handoff_spec)
sys.modules["verify_artifact_handoff"] = handoff
_handoff_spec.loader.exec_module(handoff)

_ab_spec = importlib.util.spec_from_file_location(
    "ab_boltz_kernels_for_handoff", str(_ROOT / "k8s" / "scripts" / "ab-boltz-kernels.py")
)
ab = importlib.util.module_from_spec(_ab_spec)
sys.modules["ab_boltz_kernels_for_handoff"] = ab
_ab_spec.loader.exec_module(ab)

_boltz_spec = importlib.util.spec_from_file_location(
    "boltz2_app_for_handoff", str(_ROOT / "services" / "boltz2" / "app.py")
)
boltz_app = importlib.util.module_from_spec(_boltz_spec)
sys.modules["boltz2_app_for_handoff"] = boltz_app
_boltz_spec.loader.exec_module(boltz_app)

_pb_spec = importlib.util.spec_from_file_location(
    "posebusters_app_for_handoff", str(_ROOT / "services" / "posebusters" / "app.py")
)
pb_app = importlib.util.module_from_spec(_pb_spec)
sys.modules["posebusters_app_for_handoff"] = pb_app
_pb_spec.loader.exec_module(pb_app)

_PROFILER_APP = _ROOT / "services" / "interaction-profiler" / "app.py"

# One ligand, one protein residue: enough for RDKit to read a ligand out of a
# Boltz-shaped complex, which is what PoseBusters does with a handed-over path.
SAMPLE_COMPLEX_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
HETATM    3  C1  LIG B   1       5.000   6.000   7.000  1.00  0.00           C
HETATM    4  O1  LIG B   1       5.500   6.500   7.500  1.00  0.00           O
CONECT    3    4
CONECT    4    3
END
"""

STRUCTURES = [
    "/shared/predictions/out_abc/boltz_results_batch_abc/predictions/mol_0/mol_0_model_0.pdb",
    None,
    "/shared/predictions/out_abc/boltz_results_batch_abc/predictions/mol_1/mol_1_model_0.pdb",
]
SMILES = ["CCO", "C" * 60, "c1ccccc1"]
EXPECTED = [0, 2]


def _pod(name: str, node: str = "node-007", container: str = "x") -> handoff.PodRef:
    return handoff.PodRef(name=name, node=node, container=container)


def _producer(**overrides: Any) -> handoff.Producer:
    defaults = dict(
        pods=[_pod("dd-agentic-boltz-1", container="boltz"),
              _pod("dd-agentic-boltz-2", container="boltz")],
        elapsed=386.0,
        structures=list(STRUCTURES),
        scores=[6.2, None, 5.1],
        skip_reasons=["", "60 heavy atoms exceeds Boltz limit (56)", ""],
    )
    defaults.update(overrides)
    return handoff.Producer(**defaults)


def _consumer(
    label: str,
    status: str,
    pod: Optional[handoff.PodRef] = None,
    pods: Optional[list[handoff.PodRef]] = None,
) -> handoff.Consumer:
    """One consumer. Pass ``pods=[]`` for the no-namespace case."""
    if pods is None:
        pods = [pod or _pod(f"dd-agentic-{label}-1")]
    return handoff.Consumer(
        label=label,
        url=f"http://{label}:8080",
        pods=pods,
        status=status,
    )


def _stub_post(module, scores: list, status: Optional[int] = 200, error: Optional[str] = None):
    """Replace ``post_json`` with a canned ``/score`` response."""
    def _post(url, payload, timeout):
        return 1.0, {"scores": scores}, status, error

    return patch.object(module, "post_json", _post)


# ---------------------------------------------------------------------------
# TestPathAudit
# ---------------------------------------------------------------------------


class TestPathAudit:
    """Only an absolute path under /shared can mean the same thing in two pods."""

    def test_aligned_shared_paths_are_clean(self):
        assert handoff.audit_paths(STRUCTURES, EXPECTED, 3) == []

    def test_relative_path_is_rejected(self):
        # Would resolve against each consumer's working directory.
        problems = handoff.audit_paths(["predictions/mol_0.pdb"], [0], 1)

        assert len(problems) == 1
        assert "relative" in problems[0]

    def test_path_outside_shared_is_rejected(self):
        # /tmp is the container filesystem; no other pod mounts it.
        problems = handoff.audit_paths(["/tmp/mol_0.pdb"], [0], 1)

        assert len(problems) == 1
        assert "outside /shared" in problems[0]

    def test_shared_prefix_must_be_a_directory_boundary(self):
        problems = handoff.audit_paths(["/shared-scratch/mol_0.pdb"], [0], 1)

        assert "outside /shared" in problems[0]

    def test_short_list_breaks_alignment(self):
        # PoseBusters 400s on a length mismatch; the profiler would silently
        # pair poses with the wrong SMILES.
        problems = handoff.audit_paths(["/shared/mol_0.pdb"], [0], 3)

        assert any("index-aligned" in p for p in problems)

    def test_missing_structure_for_a_scored_molecule_is_a_problem(self):
        problems = handoff.audit_paths([None, "/shared/mol_1.pdb"], [0, 1], 2)

        assert any("produced no structure path" in p for p in problems)

    def test_structure_for_a_skipped_molecule_means_shifted_holes(self):
        # The classic off-by-one: boltz skipped molecule 1 but a path landed
        # there, so every pose after it belongs to another SMILES.
        problems = handoff.audit_paths(
            ["/shared/mol_0.pdb", "/shared/mol_1.pdb"], [0], 2,
        )

        assert any("holes have shifted" in p for p in problems)


# ---------------------------------------------------------------------------
# TestRelayedPayloads
# ---------------------------------------------------------------------------


class TestRelayedPayloads:
    """The harness must relay exactly what the RL loop relays."""

    def _injected_params(self, service: str, structures: list) -> dict[str, Any]:
        """Params ``ExternalRESTScorer`` POSTs for a component depending on boltz."""
        from reinvent_plugins.components.comp_external_rest import (
            ExternalRESTScorer,
            Parameters,
        )

        params = Parameters(
            service_url=["http://consumer:8080/score"],
            service=[service],
            depends_on=["cofold"],
            component_name=[service],
        )
        ExternalRESTScorer._artifact_registry["cofold"] = {"structures": structures}
        try:
            scorer = ExternalRESTScorer(params)
            with patch(
                "reinvent_plugins.components.comp_external_rest.requests.post"
            ) as mock_post:
                mock_post.return_value = SimpleNamespace(
                    status_code=200, ok=True, text="",
                    json=lambda: {"scores": [1.0, None, 1.0]},
                )
                scorer(SMILES)
            return mock_post.call_args.kwargs["json"]["params"]
        finally:
            ExternalRESTScorer._artifact_registry.pop("cofold", None)

    def test_posebusters_payload_matches_the_scorer(self):
        sent = self._injected_params("posebusters", STRUCTURES)
        ours = handoff.posebusters_payload(SMILES, STRUCTURES, "dock")["params"]

        assert ours["structure_paths"] == sent["structure_paths"]
        assert ours["protein_pdb_path"] == sent["protein_pdb_path"] == STRUCTURES[0]

    def test_profiler_payload_matches_the_scorer(self):
        sent = self._injected_params("interaction_profiler", STRUCTURES)
        ours = handoff.profiler_payload(SMILES, STRUCTURES, "all")["params"]

        assert ours["structure_paths"] == sent["structure_paths"]

    def test_holes_are_forwarded_not_compacted(self):
        params = handoff.posebusters_payload(SMILES, STRUCTURES, "dock")["params"]

        # A compacted list would make PoseBusters 400 on the length check and
        # the profiler score molecule 2's pose against molecule 1's SMILES.
        assert params["structure_paths"] == STRUCTURES
        assert len(params["structure_paths"]) == len(SMILES)

    def test_posebusters_accepts_the_payload(self):
        payload = handoff.posebusters_payload(SMILES, STRUCTURES, "dock")

        request = pb_app.ScoreRequest(**payload)

        assert request.params.structure_paths == STRUCTURES
        assert request.params.mode == "dock"

    def test_posebusters_mode_is_one_it_knows(self):
        payload = handoff.posebusters_payload(SMILES, STRUCTURES, "dock")

        assert payload["params"]["mode"] in pb_app.VALID_MODES
        # dock needs a receptor, which is why protein_pdb_path is sent.
        assert payload["params"]["mode"] in pb_app.MODES_REQUIRING_PROTEIN
        assert "protein_pdb_path" in payload["params"]

    def test_profiler_accepts_the_holes(self):
        # The profiler package is not installed here, so its model is rebuilt
        # from the annotation the service actually declares. Until that
        # annotation admitted None, an index-aligned list from a partial batch
        # was a 422 and the whole component came back NaN.
        annotation = _profiler_annotation("ScoreParams", "structure_paths")
        probe = pydantic.create_model(
            "ProfilerScoreParamsProbe",
            structure_paths=(eval(annotation, {"list": list, "str": str}), None),  # noqa: S307
        )

        assert probe(structure_paths=STRUCTURES).structure_paths == STRUCTURES

    def test_boltz_payload_asks_for_pdb(self):
        # PoseBusters pulls HETATM records out of the complex, so mmcif would
        # score None everywhere for a reason unrelated to the volume.
        payload = handoff.boltz_payload(_args(), SMILES)

        assert payload["params"]["output_format"] == "pdb"

    def test_boltz_payload_uses_the_precomputed_msa(self):
        payload = handoff.boltz_payload(_args(), SMILES)

        assert payload["params"]["msa_path"] == handoff.DEFAULT_MSA_PATH
        assert "use_msa_server" not in payload["params"]

    def test_boltz_accepts_the_payload(self):
        payload = handoff.boltz_payload(_args(), SMILES)

        request = boltz_app.ScoreRequest(**payload)

        assert request.params.output_format == "pdb"
        assert boltz_app._validate_protein(request) == []


def _args(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        protein_sequence=handoff.BRD4_BD1_SEQUENCE,
        msa_path=handoff.DEFAULT_MSA_PATH,
        recycling_steps=3,
        sampling_steps=200,
        sampling_steps_affinity=200,
        diffusion_samples_affinity=5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _profiler_annotation(class_name: str, field: str) -> str:
    """Source of one annotation in the interaction profiler's service module."""
    tree = ast.parse(_PROFILER_APP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == field
                ):
                    return ast.unparse(stmt.annotation)
    raise AssertionError(f"{class_name}.{field} is gone from {_PROFILER_APP.name}")


# ---------------------------------------------------------------------------
# TestBatch
# ---------------------------------------------------------------------------


class TestBatch:
    """The batch has to contain a hole, and the rest has to survive boltz."""

    def test_panel_molecules_pass_the_wrapper_validator(self):
        for name, smi in handoff.PANEL:
            ok, reason = boltz_app._validate_smiles(smi)
            assert ok, f"{name}: {reason}"

    def test_the_hole_is_skipped_by_boltz(self):
        ok, reason = boltz_app._validate_smiles(handoff.HOLE[1])

        assert not ok
        # Skipped for an explicit size reason, not an unparseable SMILES: the
        # consumers still get a real SMILES at that index.
        assert "heavy atoms" in reason

    def test_the_hole_is_still_a_valid_molecule(self):
        from rdkit import Chem

        # The profiler validates the SMILES it is given, so a junk string
        # there would fail for its own reason and prove nothing about paths.
        assert Chem.MolFromSmiles(handoff.HOLE[1]) is not None

    def test_batch_carries_the_hole_in_the_middle(self):
        batch = handoff.build_batch(SimpleNamespace(
            smiles=None, structure=None, batch_size=2, hole=True,
        ))

        assert [name for name, _ in batch] == ["JQ1", handoff.HOLE[0], "birabresib"]

    def test_hole_can_be_turned_off(self):
        batch = handoff.build_batch(SimpleNamespace(
            smiles=None, structure=None, batch_size=2, hole=False,
        ))

        assert handoff.HOLE not in batch

    def test_replay_mode_drops_the_hole(self):
        # Replaying known structures means every index has one by definition.
        batch = handoff.build_batch(SimpleNamespace(
            smiles=["CCO"], structure=["/shared/mol_0.pdb"], batch_size=2, hole=True,
        ))

        assert len(batch) == 1

    def test_partitioner_leaves_exactly_one_hole(self):
        batch = handoff.build_batch(SimpleNamespace(
            smiles=None, structure=None, batch_size=2, hole=True,
        ))
        valid, reasons = boltz_app._partition_smiles([smi for _, smi in batch])

        assert valid == [0, 2]
        assert [bool(r) for r in reasons] == [False, True, False]

    def test_target_matches_the_ab_harness(self):
        # Both scripts fold the same target with the same precomputed MSA, so
        # a run here reuses the alignment the other one cached.
        assert handoff.BRD4_BD1_SEQUENCE == ab.BRD4_BD1_SEQUENCE
        assert handoff.DEFAULT_MSA_PATH == ab.DEFAULT_MSA_PATH


# ---------------------------------------------------------------------------
# TestDerivedSdfPath
# ---------------------------------------------------------------------------


class TestDerivedSdfPath:
    """PoseBusters writes into the producer's directory, so writes matter too."""

    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None, reason="rdkit not installed"
    )
    def test_matches_where_posebusters_actually_writes(self, tmp_path):
        pdb = tmp_path / "mol_0_model_0.pdb"
        pdb.write_text(SAMPLE_COMPLEX_PDB)

        written = pb_app._boltz_pdb_to_sdf(pdb)

        assert str(written) == handoff.derived_sdf_path(str(pdb))
        assert written.is_file()

    def test_derived_path_stays_in_the_producers_directory(self):
        derived = handoff.derived_sdf_path(STRUCTURES[0])

        assert derived.startswith(str(Path(STRUCTURES[0]).parent) + "/")
        assert derived.endswith("_lig.sdf")


# ---------------------------------------------------------------------------
# TestConsumerClassification
# ---------------------------------------------------------------------------


class TestConsumerClassification:
    """A 200 full of nulls is storage; a 4xx is config. Never the same bucket."""

    def _consume(self, scores: list, status: int = 200, error: Optional[str] = None):
        payload = handoff.posebusters_payload(SMILES, STRUCTURES, "dock")
        with _stub_post(handoff, scores, status, error):
            return handoff.consume(
                handoff.POSEBUSTERS, "http://pb:8082", payload, EXPECTED,
                timeout=1.0, namespace=None,
            )

    def test_real_values_at_every_structure_is_ok(self):
        consumer = self._consume([0.94, None, 0.88])

        assert consumer.status == handoff.OK
        assert consumer.ok

    def test_all_nulls_is_blind(self):
        consumer = self._consume([None, None, None])

        assert consumer.status == handoff.BLIND
        assert "no file behind it could be opened" in consumer.detail

    def test_some_nulls_is_partial(self):
        consumer = self._consume([0.94, None, None])

        assert consumer.status == handoff.PARTIAL
        assert "1/2" in consumer.detail

    def test_a_scored_hole_is_reported_as_misaligned(self):
        # Every structure read, but the molecule boltz skipped came back with a
        # value -- so the consumer paired poses with the wrong SMILES.
        consumer = self._consume([0.94, 0.5, 0.88])

        assert consumer.status == handoff.PARTIAL
        assert "misaligned" in consumer.detail

    def test_client_error_is_rejected_not_blind(self):
        consumer = self._consume([], status=422, error="HTTP 422: null is not a string")

        assert consumer.status == handoff.REJECTED

    def test_length_mismatch_is_rejected(self):
        consumer = self._consume([], status=400, error="HTTP 400: Length mismatch")

        assert consumer.status == handoff.REJECTED

    def test_transport_failure_is_unreachable(self):
        consumer = self._consume([], status=None, error="request failed: refused")

        assert consumer.status == handoff.UNREACHABLE

    def test_server_error_is_unreachable(self):
        # A 500 says nothing about the handoff; it is not evidence either way.
        consumer = self._consume([], status=500, error="HTTP 500: boom")

        assert consumer.status == handoff.UNREACHABLE


# ---------------------------------------------------------------------------
# TestVerdict
# ---------------------------------------------------------------------------


class TestVerdict:
    """Proven, disproven, and the cases that are neither."""

    def test_cross_pod_success_is_proven(self):
        proven, reason = handoff.decide(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.OK),
             _consumer(handoff.PROFILER, handoff.OK)],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is True
        assert "re-opened them by path" in reason

    def test_spread_across_nodes_is_called_out(self):
        proven, reason = handoff.decide(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.OK,
                       pod=_pod("pb-1", node="node-017"))],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is True
        assert "across 2 nodes" in reason

    def test_blind_consumer_disproves_it(self):
        proven, reason = handoff.decide(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.BLIND)],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is False
        assert handoff.BLIND in reason

    def test_unportable_paths_disprove_it_before_the_consumers(self):
        proven, reason = handoff.decide(
            _producer(path_problems=["molecule 0: 'x.pdb' is relative"]),
            [_consumer(handoff.POSEBUSTERS, handoff.OK)],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is False
        assert "cannot cross a pod boundary" in reason

    def test_invisible_extracted_ligand_disproves_it(self):
        # Both consumers read fine, but PoseBusters cannot have scored without
        # writing that SDF -- so if the producer's pod cannot see it, the two
        # resolved the same path on different filesystems.
        proven, reason = handoff.decide(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.OK)],
            reverse_missing=[handoff.derived_sdf_path(STRUCTURES[0])],
            namespace="demo-lux",
        )

        assert proven is False
        assert "different filesystems" in reason

    def test_nothing_produced_is_inconclusive(self):
        proven, reason = handoff.decide(
            _producer(structures=[None, None, None], scores=[None] * 3),
            [],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is None
        assert "scored nothing" in reason

    def test_producer_failure_is_inconclusive(self):
        proven, reason = handoff.decide(
            _producer(structures=[], error="request failed: connection refused"),
            [],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is None

    def test_unreachable_consumer_is_inconclusive(self):
        proven, _reason = handoff.decide(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.UNREACHABLE)],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is None

    def test_same_pod_is_not_a_cross_pod_proof(self):
        # The in-pod handoff was already proven; a re-run of it must not pass
        # as the cross-pod one.
        proven, reason = handoff.decide(
            _producer(pods=[_pod("dd-agentic-1")]),
            [_consumer(handoff.POSEBUSTERS, handoff.OK, pod=_pod("dd-agentic-1"))],
            reverse_missing=[],
            namespace="demo-lux",
        )

        assert proven is None
        assert "never crossed" in reason

    def test_without_a_namespace_the_proof_is_hedged(self):
        proven, reason = handoff.decide(
            _producer(pods=[]),
            [_consumer(handoff.POSEBUSTERS, handoff.OK, pods=[])],
            reverse_missing=[],
            namespace=None,
        )

        assert proven is True
        assert "--namespace" in reason


# ---------------------------------------------------------------------------
# TestFindings
# ---------------------------------------------------------------------------


class TestFindings:
    """What the run tells the operator to do next."""

    def test_blind_consumer_points_at_the_storage_fallback(self):
        findings = handoff.collect_findings(
            _producer(), [_consumer(handoff.POSEBUSTERS, handoff.BLIND)], [], "demo-lux",
        )

        assert any("storage failure" in f for f in findings)

    def test_rejected_consumer_is_named_a_config_problem(self):
        findings = handoff.collect_findings(
            _producer(),
            [handoff.Consumer(label=handoff.PROFILER, url="x",
                              status=handoff.REJECTED,
                              detail="HTTP 422: null is not a string")],
            [], "demo-lux",
        )

        assert any("not a storage one" in f for f in findings)

    def test_missing_extracted_ligand_is_a_write_finding(self):
        findings = handoff.collect_findings(
            _producer(), [_consumer(handoff.POSEBUSTERS, handoff.OK)],
            [handoff.derived_sdf_path(STRUCTURES[0])], "demo-lux",
        )

        assert any("not shared for writes" in f for f in findings)

    def test_same_pod_is_a_finding(self):
        findings = handoff.collect_findings(
            _producer(pods=[_pod("dd-agentic-1")]),
            [_consumer(handoff.POSEBUSTERS, handoff.OK, pod=_pod("dd-agentic-1"))],
            [], "demo-lux",
        )

        assert any("a pod boltz also runs in" in f for f in findings)

    def test_multi_node_implies_readwritemany(self):
        findings = handoff.collect_findings(
            _producer(),
            [_consumer(handoff.POSEBUSTERS, handoff.OK, pod=_pod("pb-1", node="node-017"))],
            [], "demo-lux",
        )

        assert any("ReadWriteMany" in f for f in findings)

    def test_no_namespace_is_a_finding(self):
        findings = handoff.collect_findings(
            _producer(pods=[]), [_consumer(handoff.POSEBUSTERS, handoff.OK, pods=[])],
            [], None,
        )

        assert any("--namespace" in f for f in findings)


# ---------------------------------------------------------------------------
# TestKubectlProbes
# ---------------------------------------------------------------------------


class TestKubectlProbes:
    """Resolving who answered, and what they can see, without a cluster."""

    def test_every_replica_is_resolved_from_the_name_label(self, monkeypatch):
        captured: list[list[str]] = []

        def _kubectl(args, namespace, capture=False):
            captured.append(args)
            return SimpleNamespace(stdout=json.dumps({"items": [
                {"metadata": {"name": "dd-agentic-boltz-abc"},
                 "spec": {"nodeName": "node-007"}},
                {"metadata": {"name": "dd-agentic-boltz-def"},
                 "spec": {"nodeName": "node-007"}},
            ]}))

        monkeypatch.setattr(handoff, "kubectl", _kubectl)
        pods = handoff.pods_for("demo-lux", handoff.BOLTZ)

        # All of them: the gateway decides which replica serves a batch, so
        # naming one would be a guess.
        assert [p.name for p in pods] == [
            "dd-agentic-boltz-abc", "dd-agentic-boltz-def",
        ]
        assert {p.container for p in pods} == {"boltz"}
        assert "app.kubernetes.io/name=dd-agentic-boltz" in captured[0]
        # A Terminating pod from a previous rollout must not be counted.
        assert "status.phase=Running" in captured[0]

    def test_a_single_replica_reads_as_one_pod(self, monkeypatch):
        monkeypatch.setattr(handoff, "kubectl", lambda *a, **k: SimpleNamespace(
            stdout=json.dumps({"items": [
                {"metadata": {"name": "dd-agentic-posebusters-1"},
                 "spec": {"nodeName": "node-007"}},
            ]})))

        pods = handoff.pods_for("demo-lux", handoff.POSEBUSTERS)

        assert len(pods) == 1
        assert handoff.describe_pods(pods) == "dd-agentic-posebusters-1 on node-007"

    def test_several_replicas_are_described_as_a_set(self):
        described = handoff.describe_pods([
            _pod("boltz-a", node="node-007"), _pod("boltz-b", node="node-017"),
        ])

        assert described.startswith("2 replicas on node-007, node-017:")

    def test_no_running_pod_reads_as_unknown(self, monkeypatch):
        monkeypatch.setattr(handoff, "kubectl",
                            lambda *a, **k: SimpleNamespace(stdout='{"items": []}'))

        assert handoff.pods_for("demo-lux", handoff.BOLTZ) == []

    def test_missing_kubectl_reads_as_unknown(self, monkeypatch):
        def _kubectl(*args, **kwargs):
            raise FileNotFoundError("kubectl")

        monkeypatch.setattr(handoff, "kubectl", _kubectl)

        assert handoff.pods_for("demo-lux", handoff.BOLTZ) == []

    def test_stat_reports_sizes_from_inside_the_pod(self, monkeypatch):
        captured: list[list[str]] = []

        def _kubectl(args, namespace, capture=False):
            captured.append(args)
            return SimpleNamespace(stdout='{"/shared/a_lig.sdf": 412}')

        monkeypatch.setattr(handoff, "kubectl", _kubectl)
        sizes = handoff.stat_in_pod(
            "demo-lux", _pod("boltz-1", container="boltz"), ["/shared/a_lig.sdf"],
        )

        assert sizes == {"/shared/a_lig.sdf": 412}
        assert captured[0][:2] == ["exec", "boltz-1"]
        assert captured[0][-1] == "/shared/a_lig.sdf"

    def test_reverse_check_reports_the_missing_ligands(self, monkeypatch):
        producer = _producer()
        expected = [handoff.derived_sdf_path(p) for p in producer.valid_structures]
        monkeypatch.setattr(handoff, "stat_in_pod",
                            lambda ns, pod, paths: {expected[0]: 412, expected[1]: None})

        sizes, missing = handoff.check_reverse_visibility("demo-lux", producer)

        assert sizes == {expected[0]: 412, expected[1]: None}
        assert missing == [expected[1]]

    def test_reverse_check_skipped_when_the_exec_fails(self, monkeypatch):
        monkeypatch.setattr(handoff, "stat_in_pod", lambda ns, pod, paths: None)

        sizes, missing = handoff.check_reverse_visibility("demo-lux", _producer())

        assert sizes is None
        assert missing == []

    def test_an_empty_file_counts_as_missing(self, monkeypatch):
        # A zero-byte SDF means the write reached a different filesystem or
        # failed halfway, which is not a successful handoff.
        producer = _producer()
        expected = [handoff.derived_sdf_path(p) for p in producer.valid_structures]
        monkeypatch.setattr(handoff, "stat_in_pod",
                            lambda ns, pod, paths: {p: 0 for p in expected})

        _sizes, missing = handoff.check_reverse_visibility("demo-lux", producer)

        assert missing == expected


# ---------------------------------------------------------------------------
# TestHealthGate
# ---------------------------------------------------------------------------


class TestHealthGate:
    """Fail before the GPU call, not after it."""

    def test_unreachable_service_is_listed(self, monkeypatch):
        monkeypatch.setattr(handoff, "get_json",
                            lambda url, timeout: (None, "connection refused"))

        problems = handoff.check_health({handoff.BOLTZ: "http://x:8091"}, 1.0)

        assert len(problems) == 1
        assert "connection refused" in problems[0]

    def test_profiler_without_a_shared_mount_is_listed(self, monkeypatch):
        # The profiler publishes this on /health, which is the cheapest
        # possible version of this whole check.
        monkeypatch.setattr(handoff, "get_json", lambda url, timeout: (
            {"status": "ok", "shared_dir": "/shared", "shared_dir_exists": False}, None,
        ))

        problems = handoff.check_health({handoff.PROFILER: "http://x:8083"}, 1.0)

        assert any("no /shared mount" in p for p in problems)

    def test_healthy_services_pass(self, monkeypatch):
        monkeypatch.setattr(handoff, "get_json",
                            lambda url, timeout: ({"status": "ok"}, None))

        assert handoff.check_health({handoff.BOLTZ: "http://x:8091"}, 1.0) == []


# ---------------------------------------------------------------------------
# TestEndpointsAndCli
# ---------------------------------------------------------------------------


class TestEndpointsAndCli:

    @pytest.fixture(scope="class")
    def mcp_config(self) -> dict[str, Any]:
        configmap = next(
            doc
            for doc in yaml.safe_load_all(
                (_ROOT / "k8s" / "manifests" / "configmap.yaml").read_text()
            )
            if doc and doc["kind"] == "ConfigMap"
        )
        return yaml.safe_load(configmap["data"]["reinvent4_mcp_config.yaml"])

    def test_cluster_urls_are_the_ones_the_rl_loop_uses(self, mcp_config):
        # Dialing anything else would prove the handoff for a path the
        # scorers never take.
        scoring = mcp_config["scoring"]
        assert scoring["boltz"]["service_url"] == f"{handoff.CLUSTER_URLS[handoff.BOLTZ]}/score"
        assert (scoring["posebusters"]["service_url"]
                == f"{handoff.CLUSTER_URLS[handoff.POSEBUSTERS]}/score")
        assert (scoring["interaction_profiler"]["service_url"]
                == f"{handoff.CLUSTER_URLS[handoff.PROFILER]}/score")

    def test_local_ports_match_the_cluster_ports(self):
        # The documented port-forwards map <port>:<port>, so a mismatch would
        # silently dial the wrong service.
        for service, url in handoff.CLUSTER_URLS.items():
            assert handoff.LOCAL_URLS[service].rsplit(":", 1)[1] == url.rsplit(":", 1)[1]

    def test_the_defaults_also_hit_the_compose_stack(self):
        # Compose publishes the same ports on localhost and mounts one host
        # directory into all three containers, so the defaults rehearse the
        # handoff across containers before anything reaches a cluster.
        compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())["services"]
        names = {handoff.BOLTZ: "boltz", handoff.POSEBUSTERS: "posebusters",
                 handoff.PROFILER: "interaction-profiler"}

        for service, compose_name in names.items():
            published = compose[compose_name]["ports"][0].rsplit(":", 2)[-2]
            assert handoff.LOCAL_URLS[service].rsplit(":", 1)[1] == published
            mounts = [v.split(":")[-1] for v in compose[compose_name]["volumes"]]
            assert "/shared" in mounts

    def test_consumer_defaults_match_the_configmap(self, mcp_config):
        scoring = mcp_config["scoring"]

        assert handoff.DEFAULT_POSEBUSTERS_MODE == scoring["posebusters"]["mode"]
        assert (handoff.DEFAULT_INTERACTION_TYPE
                == scoring["interaction_profiler"]["interaction_type"])

    def test_replay_needs_the_smiles_that_produced_the_pose(self):
        with pytest.raises(SystemExit):
            handoff.main(["--structure", "/shared/mol_0.pdb"])

    def test_replay_lists_must_be_the_same_length(self):
        with pytest.raises(SystemExit):
            handoff.main(["--structure", "/shared/mol_0.pdb",
                          "--smiles", "CCO", "--smiles", "c1ccccc1"])

    def test_empty_batch_rejected(self):
        with pytest.raises(SystemExit):
            handoff.main(["--batch-size", "0"])

    def test_unreachable_services_exit_inconclusive(self, monkeypatch):
        # Exit 2, not 1: nothing was measured, so nothing was disproven.
        monkeypatch.setattr(handoff, "get_json",
                            lambda url, timeout: (None, "connection refused"))

        assert handoff.main([]) == 2


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class _FakeCluster:
    """Three services and a kubectl, enough to drive ``main`` to a verdict."""

    def __init__(
        self,
        pb_scores: Optional[list] = None,
        ip_scores: Optional[list] = None,
        sdf_size: Optional[int] = 412,
        nodes: Optional[dict[str, str]] = None,
    ):
        self.pb_scores = [0.94, None, 0.88] if pb_scores is None else pb_scores
        self.ip_scores = [34.0, None, 21.0] if ip_scores is None else ip_scores
        self.sdf_size = sdf_size
        self.nodes = nodes or {}
        self.posted: list[tuple[str, dict]] = []

    # -- HTTP --

    def get_json(self, url, timeout):
        return {"status": "ok", "shared_dir": "/shared", "shared_dir_exists": True}, None

    def post_json(self, url, payload, timeout):
        self.posted.append((url, payload))
        if ":8091/score" in url:
            return 386.0, {
                "scores": [6.2, None, 5.1],
                "metadata": {"skip_reason": [
                    "", "60 heavy atoms exceeds Boltz limit (56)", "",
                ]},
                "artifacts": {"structures": list(STRUCTURES)},
            }, 200, None
        if ":8082/score" in url:
            return 8.2, {"scores": self.pb_scores}, 200, None
        if ":8083/score" in url:
            return 12.0, {"scores": self.ip_scores}, 200, None
        if ":8083/analyze" in url:
            return 1.0, {"molecules": [
                {"error": f"structure not found: {path}"}
                for path in STRUCTURES if path
            ]}, 200, None
        raise AssertionError(f"unexpected POST to {url}")

    # -- kubectl --

    def kubectl(self, args, namespace, capture=False):
        if args[0] == "get":
            deployment = next(
                a.split("=", 1)[1] for a in args if a.startswith("app.kubernetes.io/name=")
            )
            # Boltz runs behind its gateway, so it answers with M replicas.
            count = 2 if deployment == "dd-agentic-boltz" else 1
            return SimpleNamespace(stdout=json.dumps({"items": [
                {
                    "metadata": {"name": f"{deployment}-{suffix}"},
                    "spec": {"nodeName": self.nodes.get(deployment, "node-007")},
                }
                for suffix in ("abc", "def")[:count]
            ]}))
        if args[0] == "exec":
            paths = args[args.index("--") + 4:]
            return SimpleNamespace(
                stdout=json.dumps({p: self.sdf_size for p in paths})
            )
        if args[0] == "logs":
            return SimpleNamespace(stdout="structure file not found: /shared/mol_0.pdb")
        raise AssertionError(f"unexpected kubectl {args}")

    def install(self, monkeypatch):
        monkeypatch.setattr(handoff, "get_json", self.get_json)
        monkeypatch.setattr(handoff, "post_json", self.post_json)
        monkeypatch.setattr(handoff, "kubectl", self.kubectl)
        return self


class TestEndToEnd:
    """``main`` all the way to a verdict, with the cluster faked out."""

    def test_happy_path_is_proven(self, monkeypatch, capsys):
        _FakeCluster().install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        out = capsys.readouterr().out
        assert status == 0
        assert "PROVEN:" in out and "NOT PROVEN" not in out
        # The structures boltz returned are printed, so the proof is readable
        # rather than just green.
        assert STRUCTURES[0] in out

    def test_the_verdict_does_not_guess_which_replica_served(self, monkeypatch, capsys):
        # Two replicas behind the gateway: the script must not name one of them
        # as the writer, because the gateway chose and it was not told which.
        _FakeCluster().install(monkeypatch)

        handoff.main(["--namespace", "demo-lux"])

        out = capsys.readouterr().out
        assert "one of its 2 replicas" in out
        assert "2 replicas on node-007" in out

    def test_the_extracted_ligand_is_checked_from_every_replica(self, monkeypatch, capsys):
        # A replica that cannot see what its peers wrote would break the
        # handoff only for the batches the gateway happens to place there.
        _FakeCluster().install(monkeypatch)
        statted: list[str] = []
        real_stat = handoff.stat_in_pod
        monkeypatch.setattr(handoff, "stat_in_pod", lambda ns, pod, paths: (
            statted.append(pod.name) or real_stat(ns, pod, paths)
        ))

        handoff.main(["--namespace", "demo-lux"])

        assert sorted(statted) == ["dd-agentic-boltz-abc", "dd-agentic-boltz-def"]
        assert "visible from all 2 boltz replica(s)" in capsys.readouterr().out

    def test_every_service_is_dialled_once(self, monkeypatch):
        cluster = _FakeCluster().install(monkeypatch)

        handoff.main(["--namespace", "demo-lux"])

        urls = [url for url, _payload in cluster.posted]
        assert urls == [
            f"{handoff.LOCAL_URLS[handoff.BOLTZ]}/score",
            f"{handoff.LOCAL_URLS[handoff.POSEBUSTERS]}/score",
            f"{handoff.LOCAL_URLS[handoff.PROFILER]}/score",
        ]

    def test_consumers_get_the_paths_boltz_wrote(self, monkeypatch):
        cluster = _FakeCluster().install(monkeypatch)

        handoff.main(["--namespace", "demo-lux"])

        for _url, payload in cluster.posted[1:]:
            assert payload["params"]["structure_paths"] == STRUCTURES

    def test_in_cluster_dials_the_services(self, monkeypatch):
        cluster = _FakeCluster().install(monkeypatch)

        handoff.main(["--in-cluster"])

        assert cluster.posted[0][0] == f"{handoff.CLUSTER_URLS[handoff.BOLTZ]}/score"

    def test_blind_consumer_fails_the_run(self, monkeypatch, capsys):
        _FakeCluster(pb_scores=[None, None, None]).install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        out = capsys.readouterr().out
        assert status == 1
        assert "NOT PROVEN" in out
        assert "storage failure" in out
        # The pod log is pulled in to name the failure the nulls hide.
        assert "structure file not found" in out

    def test_profiler_nulls_are_explained_by_analyze(self, monkeypatch, capsys):
        _FakeCluster(ip_scores=[None, None, None]).install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        assert status == 1
        assert "/analyze: structure not found" in capsys.readouterr().out

    def test_invisible_extracted_ligand_fails_the_run(self, monkeypatch, capsys):
        # Both consumers scored, so the reads looked fine -- but PoseBusters'
        # own write is not visible from the producer's pod.
        _FakeCluster(sdf_size=None).install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        out = capsys.readouterr().out
        assert status == 1
        assert "not shared for writes" in out

    def test_shared_pod_is_not_counted_as_cross_pod(self, monkeypatch, capsys):
        cluster = _FakeCluster()
        # Every Deployment resolves to one pod name: the pre-split topology.
        cluster.kubectl = lambda args, namespace, capture=False: SimpleNamespace(
            stdout=json.dumps({"items": [{
                "metadata": {"name": "dd-agentic-abc"},
                "spec": {"nodeName": "node-007"},
            }]})
        ) if args[0] == "get" else SimpleNamespace(stdout="{}")
        cluster.install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        assert status == 2
        assert "never crossed" in capsys.readouterr().out

    def test_multi_node_run_reports_readwritemany(self, monkeypatch, capsys):
        _FakeCluster(nodes={"dd-agentic-posebusters": "node-017"}).install(monkeypatch)

        status = handoff.main(["--namespace", "demo-lux"])

        assert status == 0
        assert "ReadWriteMany" in capsys.readouterr().out

    def test_replay_skips_the_prediction(self, monkeypatch, capsys):
        cluster = _FakeCluster(
            pb_scores=[0.94], ip_scores=[34.0],
        ).install(monkeypatch)

        status = handoff.main([
            "--namespace", "demo-lux",
            "--structure", STRUCTURES[0], "--smiles", "CCO",
        ])

        assert status == 0
        assert all(":8091" not in url for url, _ in cluster.posted)
        assert "replaying 1 structure" in capsys.readouterr().out

    def test_json_output_carries_the_verdict(self, monkeypatch, capsys):
        _FakeCluster().install(monkeypatch)

        handoff.main(["--namespace", "demo-lux", "--json"])

        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{\n  \"proven\""):])
        assert payload["proven"] is True
        assert payload["producer"]["structures"] == STRUCTURES
        assert [c["status"] for c in payload["consumers"]] == [handoff.OK, handoff.OK]
