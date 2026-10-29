# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for TTP schema, store, and mapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from reinvent4_mcp.config import get_config
from reinvent4_mcp.ttp.schema import (
    TTP,
    ConstraintConfig,
    PreScoringComponent,
    PreScoringSection,
    ScoringObjective,
    ScoringSection,
    TargetConfig,
    TransformConfig,
)
from reinvent4_mcp.ttp.store import TTPStore
from reinvent4_mcp.ttp.mapper import (
    describe_external_objectives,
    extract_native_components,
    get_mapper_summary,
    target_sequence,
    ttp_to_native_scorer_config,
    ttp_to_scorer_config,
)
from reinvent4_mcp.ttp.structure import cif_chains, pdb_chains, structure_sequence


# ---------------------------------------------------------------------------
# Fixtures
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

FULL_TTP = {
    "id": "jak2-oral-lead",
    "name": "JAK2 Oral Lead Optimization",
    "version": 1,
    "target": {
        "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        "pdb_id": "3KRR",
        "msa_path": "/data/msa/jak2.a3m",
        "protein_pdb_path": "/data/structures/3KRR_protein.pdb",
    },
    "pre_scoring": {
        "comment": "Bioavailability descriptors",
        "components": [
            {"name": "QED", "type": "native", "component": "QED", "weight": 0.25},
            {
                "name": "MW",
                "type": "native",
                "component": "MolecularWeight",
                "weight": 0.2,
                "transform": {
                    "type": "double_sigmoid",
                    "high": 500,
                    "low": 200,
                    "coef_div": 500,
                    "coef_si": 20,
                    "coef_se": 20,
                },
            },
            {
                "name": "HBD",
                "type": "native",
                "component": "HBondDonors",
                "weight": 0.1,
                "transform": {"type": "reverse_sigmoid", "high": 5, "low": 3, "k": 0.5},
            },
            {
                "name": "HBA",
                "type": "native",
                "component": "HBondAcceptors",
                "weight": 0.1,
                "transform": {"type": "reverse_sigmoid", "high": 10, "low": 7, "k": 0.5},
            },
            {
                "name": "LogP",
                "type": "native",
                "component": "SlogP",
                "weight": 0.2,
                "transform": {
                    "type": "double_sigmoid",
                    "high": 5,
                    "low": 0,
                    "coef_div": 5,
                    "coef_si": 10,
                    "coef_se": 10,
                },
            },
            {
                "name": "RotBonds",
                "type": "native",
                "component": "NumRotBond",
                "weight": 0.15,
                "transform": {"type": "reverse_sigmoid", "high": 10, "low": 7, "k": 0.5},
            },
        ],
    },
    "scoring": {
        "objectives": [
            {
                "name": "Binding Affinity",
                "type": "external",
                "service": "boltz",
                "service_url": "http://boltz:8080/score",
                "service_params": {
                    "affinity_mode": "pred_value",
                    "recycling_steps": 3,
                },
                "weight": 0.5,
                "transform": {"type": "reverse_sigmoid", "high": -5.0, "low": -8.0, "k": 0.5},
            },
            {
                "name": "Binder Probability",
                "type": "external",
                "service": "boltz",
                "service_url": "http://boltz:8080/score",
                "service_params": {"affinity_mode": "probability_binary"},
                "weight": 0.15,
                "transform": None,
                "comment": "Already 0-1 range",
            },
            {
                "name": "Pose Quality",
                "type": "external",
                "service": "posebusters",
                "service_url": "http://posebusters:8080/score",
                "service_params": {"mode": "dock"},
                "depends_on": "Binding Affinity",
                "weight": 0.2,
                "transform": {"type": "right_step", "high": 0.7},
            },
            {
                "name": "SA Score",
                "type": "native",
                "component": "SAScore",
                "weight": 0.15,
                "transform": {"type": "reverse_sigmoid", "high": 3.0, "low": 5.0},
            },
        ],
        "aggregation": "geometric_mean",
    },
    "constraints": [
        {
            "type": "custom_alerts",
            "smarts": ["[*;r{8-17}]", "[#8][#8]", "C#C"],
        },
    ],
}


@pytest.fixture
def tmp_store(tmp_path: Path) -> TTPStore:
    return TTPStore(tmp_path / "ttps")


# ---------------------------------------------------------------------------
# TransformConfig tests
# ---------------------------------------------------------------------------


class TestTransformConfig:
    def test_valid_types(self):
        for t in [
            "sigmoid",
            "reverse_sigmoid",
            "double_sigmoid",
            "right_step",
            "left_step",
            "step",
            "value_mapping",
            "exponential_decay",
        ]:
            tc = TransformConfig(type=t)
            assert tc.type == t

    def test_extra_params_allowed(self):
        tc = TransformConfig(type="sigmoid", low=0.0, high=1.0, k=0.5)
        assert tc.type == "sigmoid"
        assert tc.low == 0.0  # type: ignore[attr-defined]

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError, match="Unknown transform type"):
            TransformConfig(type="unknown_transform")


# ---------------------------------------------------------------------------
# TargetConfig tests
# ---------------------------------------------------------------------------


class TestTargetConfig:
    def test_valid_sequence(self):
        t = TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY")
        assert t.protein_sequence == "ACDEFGHIKLMNPQRSTVWY"

    def test_short_sequence_rejected(self):
        with pytest.raises(ValidationError, match="at least 10"):
            TargetConfig(protein_sequence="SHORT")

    def test_invalid_characters_rejected(self):
        with pytest.raises(ValidationError, match="Invalid amino-acid"):
            TargetConfig(protein_sequence="ACDEFGHIKLX123")

    def test_sequence_uppercased(self):
        t = TargetConfig(protein_sequence="acdefghiklmnpqrstvwy")
        assert t.protein_sequence == "ACDEFGHIKLMNPQRSTVWY"

    def test_templates_default_empty(self):
        assert TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY").templates == []

    def test_template_parsed(self):
        t = TargetConfig(
            protein_sequence="ACDEFGHIKLMNPQRSTVWY",
            templates=[{"path": " uploads/s-1/1M17_A.cif ", "chain_id": " A ", "force": True, "threshold": 2.0}],
        )
        tpl = t.templates[0]
        assert (tpl.path, tpl.chain_id, tpl.force, tpl.threshold) == ("uploads/s-1/1M17_A.cif", "A", True, 2.0)

    def test_template_blank_chain_is_none(self):
        t = TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", templates=[{"path": "a.pdb", "chain_id": ""}])
        assert t.templates[0].chain_id is None

    def test_template_needs_structure_file(self):
        with pytest.raises(ValidationError, match="must be a"):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", templates=[{"path": "msa/x.a3m"}])

    def test_forced_template_needs_threshold(self):
        with pytest.raises(ValidationError, match="threshold"):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", templates=[{"path": "a.cif", "force": True}])

    def test_template_threshold_positive(self):
        with pytest.raises(ValidationError, match="positive"):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", templates=[{"path": "a.cif", "threshold": 0}])

    def test_template_stems_unique(self):
        with pytest.raises(ValidationError, match="unique"):
            TargetConfig(
                protein_sequence="ACDEFGHIKLMNPQRSTVWY",
                templates=[{"path": "x/1abc.cif"}, {"path": "y/1abc.pdb"}],
            )

    def test_template_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", templates=[{"path": "a.cif", "chain": "A"}])

    def test_structure_instead_of_sequence(self):
        t = TargetConfig(structure_path=" uploads/s-1/1M17.cif ", structure_chain=" A ")
        assert (t.protein_sequence, t.structure_path, t.structure_chain) == (None, "uploads/s-1/1M17.cif", "A")
        assert TargetConfig(structure_path="x.pdb", structure_chain="").structure_chain is None
        assert TargetConfig(protein_sequence="", structure_path="x.pdb").protein_sequence is None

    def test_sequence_or_structure_required(self):
        with pytest.raises(ValidationError, match="protein_sequence or structure_path"):
            TargetConfig()
        with pytest.raises(ValidationError, match="protein_sequence or structure_path"):
            TargetConfig(pdb_id="1M17", msa_path="m.a3m")

    def test_sequence_and_structure_exclusive(self):
        with pytest.raises(ValidationError, match="not both"):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", structure_path="a.cif")

    def test_structure_chain_needs_structure(self):
        with pytest.raises(ValidationError, match="structure_chain needs structure_path"):
            TargetConfig(protein_sequence="ACDEFGHIKLMNPQRSTVWY", structure_chain="A")

    def test_structure_path_must_be_structure_file(self):
        with pytest.raises(ValidationError, match="must be a"):
            TargetConfig(structure_path="msas/x.a3m")


# ---------------------------------------------------------------------------
# PreScoringComponent tests
# ---------------------------------------------------------------------------


class TestPreScoringComponent:
    def test_valid_component(self):
        c = PreScoringComponent(name="QED", component="QED")
        assert c.component == "QED"
        assert c.weight == 1.0

    def test_invalid_component_rejected(self):
        with pytest.raises(ValidationError, match="Unknown pre-scoring component"):
            PreScoringComponent(name="Bad", component="NotAComponent")

    def test_zero_weight_rejected(self):
        with pytest.raises(ValidationError):
            PreScoringComponent(name="QED", component="QED", weight=0.0)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            PreScoringComponent(name="QED", component="QED", weight=-1.0)


# ---------------------------------------------------------------------------
# PreScoringSection tests
# ---------------------------------------------------------------------------


class TestPreScoringSection:
    def test_duplicate_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            PreScoringSection(
                components=[
                    PreScoringComponent(name="QED", component="QED"),
                    PreScoringComponent(name="QED", component="SlogP"),
                ]
            )


# ---------------------------------------------------------------------------
# ScoringObjective tests
# ---------------------------------------------------------------------------


class TestScoringObjective:
    def test_native_requires_component(self):
        with pytest.raises(ValidationError, match="requires a 'component'"):
            ScoringObjective(name="Test", type="native")

    def test_external_requires_service(self):
        with pytest.raises(ValidationError, match="requires a 'service'"):
            ScoringObjective(name="Test", type="external")

    def test_external_requires_service_url(self):
        with pytest.raises(ValidationError, match="requires a 'service_url'"):
            ScoringObjective(name="Test", type="external", service="boltz")

    def test_valid_external(self):
        obj = ScoringObjective(
            name="Affinity",
            type="external",
            service="boltz",
            service_url="http://boltz:8080/score",
        )
        assert obj.service == "boltz"

    def test_valid_native(self):
        obj = ScoringObjective(
            name="SA", type="native", component="SAScore"
        )
        assert obj.component == "SAScore"


# ---------------------------------------------------------------------------
# ScoringSection tests
# ---------------------------------------------------------------------------


class TestScoringSection:
    def test_depends_on_nonexistent_rejected(self):
        with pytest.raises(ValidationError, match="depends_on"):
            ScoringSection(
                objectives=[
                    ScoringObjective(
                        name="A",
                        type="external",
                        service="posebusters",
                        service_url="http://pb:8080/score",
                        depends_on="NonExistent",
                    ),
                ]
            )

    def test_self_dependency_rejected(self):
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            ScoringSection(
                objectives=[
                    ScoringObjective(
                        name="Self",
                        type="external",
                        service="boltz",
                        service_url="http://boltz:8080/score",
                        depends_on="Self",
                    ),
                ]
            )

    def test_duplicate_objective_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            ScoringSection(
                objectives=[
                    ScoringObjective(name="A", type="native", component="SAScore"),
                    ScoringObjective(name="A", type="native", component="SAScore"),
                ]
            )


# ---------------------------------------------------------------------------
# ConstraintConfig tests
# ---------------------------------------------------------------------------


class TestConstraintConfig:
    def test_custom_alerts_requires_smarts(self):
        with pytest.raises(ValidationError, match="non-empty 'smarts'"):
            ConstraintConfig(type="custom_alerts")

    def test_valid_custom_alerts(self):
        c = ConstraintConfig(type="custom_alerts", smarts=["[#8][#8]"])
        assert c.smarts == ["[#8][#8]"]


# ---------------------------------------------------------------------------
# TTP tests
# ---------------------------------------------------------------------------


class TestTTP:
    def test_minimal_ttp(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        assert ttp.id == "test-ttp"
        assert ttp.version == 1
        assert len(ttp.scoring.objectives) == 1

    def test_full_ttp(self):
        ttp = TTP.model_validate(FULL_TTP)
        assert ttp.id == "jak2-oral-lead"
        assert ttp.target is not None
        assert ttp.pre_scoring is not None
        assert len(ttp.pre_scoring.components) == 6
        assert len(ttp.scoring.objectives) == 4
        assert ttp.constraints is not None
        assert len(ttp.constraints) == 1

    def test_invalid_id_rejected(self):
        data = {**MINIMAL_TTP, "id": "INVALID ID!"}
        with pytest.raises(ValidationError, match="Invalid TTP id"):
            TTP.model_validate(data)

    def test_boltz_without_target_rejected(self):
        data = {
            "id": "no-target",
            "name": "No Target",
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
        with pytest.raises(ValidationError, match="target"):
            TTP.model_validate(data)

    def test_name_collision_across_sections(self):
        data = {
            "id": "collision-test",
            "name": "Collision",
            "pre_scoring": {
                "components": [
                    {"name": "QED", "type": "native", "component": "QED"},
                ],
            },
            "scoring": {
                "objectives": [
                    {"name": "QED", "type": "native", "component": "SAScore"},
                ],
            },
        }
        with pytest.raises(ValidationError, match="Duplicates"):
            TTP.model_validate(data)

    def test_has_external_objectives(self):
        ttp = TTP.model_validate(FULL_TTP)
        assert ttp.has_external_objectives is True

    def test_native_objective_names(self):
        ttp = TTP.model_validate(FULL_TTP)
        native = ttp.native_objective_names
        assert "QED" in native
        assert "SA Score" in native
        assert "Binding Affinity" not in native

    def test_external_objective_names(self):
        ttp = TTP.model_validate(FULL_TTP)
        external = ttp.external_objective_names
        assert "Binding Affinity" in external
        assert "Pose Quality" in external
        assert "SA Score" not in external

    def test_roundtrip_json(self):
        ttp = TTP.model_validate(FULL_TTP)
        dumped = ttp.model_dump(mode="json")
        restored = TTP.model_validate(dumped)
        assert restored.id == ttp.id
        assert restored.version == ttp.version
        assert len(restored.scoring.objectives) == len(ttp.scoring.objectives)


# ---------------------------------------------------------------------------
# TTPStore tests
# ---------------------------------------------------------------------------


class TestTTPStore:
    def test_create_and_get(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        created = tmp_store.create(ttp)
        assert created.version == 1

        retrieved = tmp_store.get("test-ttp")
        assert retrieved.id == "test-ttp"
        assert retrieved.version == 1

    def test_create_duplicate_rejected(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        tmp_store.create(ttp)
        with pytest.raises(FileExistsError):
            tmp_store.create(ttp)

    def test_get_nonexistent_raises(self, tmp_store: TTPStore):
        with pytest.raises(FileNotFoundError):
            tmp_store.get("nonexistent")

    def test_update_increments_version(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        tmp_store.create(ttp)

        updated = tmp_store.update("test-ttp", {"name": "Updated Name"})
        assert updated.version == 2
        assert updated.name == "Updated Name"

    def test_update_preserves_id(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        tmp_store.create(ttp)

        updated = tmp_store.update("test-ttp", {"id": "hacked-id"})
        assert updated.id == "test-ttp"

    def test_version_history(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        tmp_store.create(ttp)
        tmp_store.update("test-ttp", {"name": "V2"})
        tmp_store.update("test-ttp", {"name": "V3"})

        v1 = tmp_store.get("test-ttp", version=1)
        assert v1.name == "Test TTP"

        v2 = tmp_store.get("test-ttp", version=2)
        assert v2.name == "V2"

        latest = tmp_store.get("test-ttp")
        assert latest.version == 3
        assert latest.name == "V3"

    def test_list_all(self, tmp_store: TTPStore):
        tmp_store.create(TTP.model_validate(MINIMAL_TTP))
        tmp_store.create(
            TTP.model_validate({**MINIMAL_TTP, "id": "second-ttp", "name": "Second"})
        )

        listing = tmp_store.list_all()
        assert len(listing) == 2
        ids = {entry["id"] for entry in listing}
        assert ids == {"test-ttp", "second-ttp"}

    def test_delete(self, tmp_store: TTPStore):
        ttp = TTP.model_validate(MINIMAL_TTP)
        tmp_store.create(ttp)
        tmp_store.update("test-ttp", {"name": "V2"})

        tmp_store.delete("test-ttp")
        assert not tmp_store.exists("test-ttp")
        with pytest.raises(FileNotFoundError):
            tmp_store.get("test-ttp")

    def test_delete_nonexistent_raises(self, tmp_store: TTPStore):
        with pytest.raises(FileNotFoundError):
            tmp_store.delete("nonexistent")

    def test_exists(self, tmp_store: TTPStore):
        assert not tmp_store.exists("test-ttp")
        tmp_store.create(TTP.model_validate(MINIMAL_TTP))
        assert tmp_store.exists("test-ttp")


# ---------------------------------------------------------------------------
# Mapper tests
# ---------------------------------------------------------------------------


class TestMapper:
    """Tests for extract_native_components (flat list output)."""

    def test_extract_native_components_full_ttp(self):
        ttp = TTP.model_validate(FULL_TTP)
        native = extract_native_components(ttp)
        names = [c["name"] for c in native]
        assert "QED" in names
        assert "SA Score" in names
        assert "Binding Affinity" not in names
        assert len(native) == 7  # 6 pre-scoring + 1 native scoring

    def test_extract_native_components_minimal(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        native = extract_native_components(ttp)
        assert len(native) == 1
        assert native[0]["type"] == "SAScore"

    def test_transform_preserved(self):
        ttp = TTP.model_validate(FULL_TTP)
        native = extract_native_components(ttp)
        mw = next(c for c in native if c["name"] == "MW")
        assert mw["transform"]["type"] == "double_sigmoid"
        assert mw["transform"]["high"] == 500


class TestTTPToScorerConfig:
    """Tests for ttp_to_scorer_config (full Scorer config with externals)."""

    def test_structure(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        assert config["type"] == "geometric_mean"
        assert "component" in config
        # 6 pre-scoring + 4 scoring + 1 constraint = 11
        assert len(config["component"]) == 11

    def test_native_component_format(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        config = ttp_to_scorer_config(ttp)
        comp = config["component"][0]
        assert "SAScore" in comp
        assert "endpoint" in comp["SAScore"]
        endpoint = comp["SAScore"]["endpoint"][0]
        assert endpoint["name"] == "SA Score"
        assert endpoint["weight"] == 1.0

    def test_external_component_format(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        assert len(external_comps) == 3  # 2 boltz + 1 posebusters

        boltz_comp = external_comps[0]
        endpoint = boltz_comp["ExternalRESTScorer"]["endpoint"][0]
        assert endpoint["params"]["service"] == "boltz"
        assert "protein_sequence" in endpoint["params"]

    def test_constraint_format(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        constraint_comps = [
            c for c in config["component"] if "CustomAlerts" in c
        ]
        assert len(constraint_comps) == 1
        endpoint = constraint_comps[0]["CustomAlerts"]["endpoint"][0]
        assert "smarts" in endpoint["params"]

    def test_pre_scoring_comes_first(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        components = config["component"]
        pre_scoring_types = {"QED", "MolecularWeight", "HBondDonors",
                             "HBondAcceptors", "SlogP", "NumRotBond"}
        for i in range(6):
            comp_type = list(components[i].keys())[0]
            assert comp_type in pre_scoring_types

    def test_constraints_come_last(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        last = config["component"][-1]
        assert "CustomAlerts" in last

    def test_no_pre_scoring(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        config = ttp_to_scorer_config(ttp)
        assert len(config["component"]) == 1
        assert "SAScore" in config["component"][0]

    def test_aggregation_from_ttp(self):
        data = {
            **MINIMAL_TTP,
            "scoring": {
                "objectives": [
                    {"name": "SA Score", "type": "native", "component": "SAScore"},
                ],
                "aggregation": "arithmetic_mean",
            },
        }
        ttp = TTP.model_validate(data)
        config = ttp_to_scorer_config(ttp)
        assert config["type"] == "arithmetic_mean"

    def test_boltz_target_info_injected(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        boltz_params = external_comps[0]["ExternalRESTScorer"]["endpoint"][0]["params"]
        assert boltz_params["protein_sequence"] == ttp.target.protein_sequence
        assert boltz_params["msa_path"] == "/data/msa/jak2.a3m"
        assert boltz_params["protein_pdb_path"] == "/data/structures/3KRR_protein.pdb"

    def test_msa_path_disables_msa_server(self):
        # A precomputed msa_path must force use_msa_server=False so boltz
        # does not override the file with the remote colabfold server, even
        # when the process config defaults use_msa_server to True.
        from reinvent4_mcp.ttp import mapper

        def _server_on_defaults():
            return {"recycling_steps": 3, "use_msa_server": True}

        original = mapper._get_boltz_defaults
        mapper._get_boltz_defaults = _server_on_defaults
        try:
            ttp = TTP.model_validate(FULL_TTP)
            config = ttp_to_scorer_config(ttp)
        finally:
            mapper._get_boltz_defaults = original

        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        boltz_params = external_comps[0]["ExternalRESTScorer"]["endpoint"][0]["params"]
        assert boltz_params["msa_path"] == "/data/msa/jak2.a3m"
        assert boltz_params["use_msa_server"] is False

    def test_no_templates_no_param(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        for comp in config["component"]:
            if "ExternalRESTScorer" in comp:
                assert "templates" not in comp["ExternalRESTScorer"]["endpoint"][0]["params"]

    def test_templates_reach_boltz_only(self):
        data = {
            **FULL_TTP,
            "target": {
                "protein_sequence": FULL_TTP["target"]["protein_sequence"],
                "templates": [
                    {"path": "uploads/s-1/3KRR_A.cif", "chain_id": "A"},
                    {"path": "/shared/structures/2xyz.pdb", "force": True, "threshold": 2.5},
                ],
            },
        }
        ttp = TTP.model_validate(data)
        config = ttp_to_scorer_config(ttp)
        external = [c["ExternalRESTScorer"]["endpoint"][0] for c in config["component"] if "ExternalRESTScorer" in c]
        boltz = [e["params"] for e in external if e["params"]["service"] == "boltz"]
        assert boltz and all(
            p["templates"]
            == [
                {"path": "uploads/s-1/3KRR_A.cif", "chain_id": "A", "force": False},
                {"path": "/shared/structures/2xyz.pdb", "force": True, "threshold": 2.5},
            ]
            for p in boltz
        )
        assert "protein_pdb_path" not in boltz[0]
        posebusters = next(e["params"] for e in external if e["params"]["service"] == "posebusters")
        assert "templates" not in posebusters

    def test_templates_in_job_toml(self):
        from reinvent4_mcp.jobs.toml_generator import _toml_value

        ttp = TTP.model_validate(
            {**FULL_TTP, "target": {"protein_sequence": FULL_TTP["target"]["protein_sequence"], "templates": [{"path": "t/a.cif"}]}}
        )
        config = ttp_to_scorer_config(ttp)
        boltz = next(c for c in config["component"] if "ExternalRESTScorer" in c)
        templates = boltz["ExternalRESTScorer"]["endpoint"][0]["params"]["templates"]
        assert _toml_value(templates) == '[{path = "t/a.cif", force = false}]'

    def test_posebusters_depends_on_preserved(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        pb_comp = next(
            c for c in external_comps
            if c["ExternalRESTScorer"]["endpoint"][0]["name"] == "Pose Quality"
        )
        assert pb_comp["ExternalRESTScorer"]["endpoint"][0]["params"]["depends_on"] == "Binding Affinity"

    def test_service_params_forwarded(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        boltz_params = external_comps[0]["ExternalRESTScorer"]["endpoint"][0]["params"]
        assert boltz_params["affinity_mode"] == "pred_value"
        assert boltz_params["recycling_steps"] == 3

    def test_external_transform_preserved(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        affinity_endpoint = external_comps[0]["ExternalRESTScorer"]["endpoint"][0]
        assert affinity_endpoint["transform"]["type"] == "reverse_sigmoid"
        assert affinity_endpoint["transform"]["high"] == -5.0

    def test_no_transform_omitted(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        binder_endpoint = next(
            c["ExternalRESTScorer"]["endpoint"][0]
            for c in external_comps
            if c["ExternalRESTScorer"]["endpoint"][0]["name"] == "Binder Probability"
        )
        assert "transform" not in binder_endpoint


class TestTTPToNativeScorerConfig:
    """Tests for ttp_to_native_scorer_config (native-only Scorer config)."""

    def test_excludes_external_objectives(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        comp_types = [list(c.keys())[0] for c in config["component"]]
        assert "ExternalRESTScorer" not in comp_types

    def test_includes_pre_scoring_and_native_objectives(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        comp_types = [list(c.keys())[0] for c in config["component"]]
        assert "QED" in comp_types
        assert "MolecularWeight" in comp_types
        assert "SAScore" in comp_types
        # 6 pre-scoring + 1 native scoring + 1 constraint
        assert len(config["component"]) == 8

    def test_preserves_aggregation(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        assert config["type"] == "geometric_mean"

    def test_minimal_ttp(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        assert len(config["component"]) == 1
        assert "SAScore" in config["component"][0]

    def test_all_external_returns_empty_components(self):
        data = {
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
        ttp = TTP.model_validate(data)
        config = ttp_to_native_scorer_config(ttp)
        assert config["component"] == []

    def test_weights_and_transforms_preserved(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        mw_comp = next(
            c for c in config["component"] if "MolecularWeight" in c
        )
        endpoint = mw_comp["MolecularWeight"]["endpoint"][0]
        assert endpoint["weight"] == 0.2
        assert endpoint["transform"]["type"] == "double_sigmoid"
        assert endpoint["transform"]["high"] == 500

    def test_includes_constraints(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        comp_types = [list(c.keys())[0] for c in config["component"]]
        assert "CustomAlerts" in comp_types

    def test_no_constraints_when_absent(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        config = ttp_to_native_scorer_config(ttp)
        comp_types = [list(c.keys())[0] for c in config["component"]]
        assert "CustomAlerts" not in comp_types


class TestTopologicalSort:
    """Tests for pipeline dependency ordering in ttp_to_scorer_config."""

    def test_boltz_before_posebusters(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        names = [
            c["ExternalRESTScorer"]["endpoint"][0]["name"]
            for c in external_comps
        ]
        affinity_idx = names.index("Binding Affinity")
        pose_idx = names.index("Pose Quality")
        assert affinity_idx < pose_idx

    def test_reversed_input_still_ordered(self):
        """Even if PoseBusters is listed before Boltz in the TTP, the
        mapper should reorder so Boltz (the dependency) comes first."""
        data = {
            "id": "reversed-deps",
            "name": "Reversed Dependencies",
            "target": {
                "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            },
            "scoring": {
                "objectives": [
                    {
                        "name": "Pose Quality",
                        "type": "external",
                        "service": "posebusters",
                        "service_url": "http://pb:8080/score",
                        "depends_on": "Binding Affinity",
                        "weight": 0.3,
                    },
                    {
                        "name": "Binding Affinity",
                        "type": "external",
                        "service": "boltz",
                        "service_url": "http://boltz:8080/score",
                        "weight": 0.7,
                    },
                ],
            },
        }
        ttp = TTP.model_validate(data)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        names = [
            c["ExternalRESTScorer"]["endpoint"][0]["name"]
            for c in external_comps
        ]
        assert names.index("Binding Affinity") < names.index("Pose Quality")

    def test_no_dependencies_preserves_order(self):
        """Without depends_on, objectives keep their original order."""
        data = {
            "id": "no-deps",
            "name": "No Dependencies",
            "target": {
                "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            },
            "scoring": {
                "objectives": [
                    {
                        "name": "B",
                        "type": "external",
                        "service": "boltz",
                        "service_url": "http://boltz:8080/score",
                    },
                    {
                        "name": "A",
                        "type": "external",
                        "service": "boltz",
                        "service_url": "http://boltz:8080/score",
                    },
                ],
            },
        }
        ttp = TTP.model_validate(data)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        names = [
            c["ExternalRESTScorer"]["endpoint"][0]["name"]
            for c in external_comps
        ]
        assert names == ["B", "A"]

    def test_chain_dependency_a_b_c(self):
        """A -> B -> C chain: A must come first, then B, then C."""
        data = {
            "id": "chain-deps",
            "name": "Chain Dependencies",
            "target": {
                "protein_sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            },
            "scoring": {
                "objectives": [
                    {
                        "name": "C",
                        "type": "external",
                        "service": "posebusters",
                        "service_url": "http://pb:8080/score",
                        "depends_on": "B",
                    },
                    {
                        "name": "A",
                        "type": "external",
                        "service": "boltz",
                        "service_url": "http://boltz:8080/score",
                    },
                    {
                        "name": "B",
                        "type": "external",
                        "service": "boltz",
                        "service_url": "http://boltz:8080/score",
                        "depends_on": "A",
                    },
                ],
            },
        }
        ttp = TTP.model_validate(data)
        config = ttp_to_scorer_config(ttp)
        external_comps = [
            c for c in config["component"] if "ExternalRESTScorer" in c
        ]
        names = [
            c["ExternalRESTScorer"]["endpoint"][0]["name"]
            for c in external_comps
        ]
        assert names.index("A") < names.index("B")
        assert names.index("B") < names.index("C")


class TestGetMapperSummary:
    """Tests for get_mapper_summary metadata."""

    def test_full_ttp_summary(self):
        ttp = TTP.model_validate(FULL_TTP)
        summary = get_mapper_summary(ttp)
        assert summary["n_pre_scoring"] == 6
        assert summary["n_native_objectives"] == 1
        assert summary["n_external_objectives"] == 3
        assert summary["n_constraints"] == 1
        assert summary["n_total_components"] == 11
        assert summary["aggregation"] == "geometric_mean"

    def test_full_ttp_names(self):
        ttp = TTP.model_validate(FULL_TTP)
        summary = get_mapper_summary(ttp)
        assert "SA Score" in summary["native_objective_names"]
        assert "Binding Affinity" in summary["external_objective_names"]
        assert "Pose Quality" in summary["external_objective_names"]

    def test_pipeline_dependencies_listed(self):
        ttp = TTP.model_validate(FULL_TTP)
        summary = get_mapper_summary(ttp)
        assert len(summary["pipeline_dependencies"]) == 1
        dep = summary["pipeline_dependencies"][0]
        assert dep["from"] == "Binding Affinity"
        assert dep["to"] == "Pose Quality"

    def test_minimal_ttp_summary(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        summary = get_mapper_summary(ttp)
        assert summary["n_pre_scoring"] == 0
        assert summary["n_native_objectives"] == 1
        assert summary["n_external_objectives"] == 0
        assert summary["n_constraints"] == 0
        assert summary["pipeline_dependencies"] == []

    def test_no_pre_scoring_summary(self):
        data = {
            "id": "no-pre",
            "name": "No Pre-scoring",
            "scoring": {
                "objectives": [
                    {"name": "SA", "type": "native", "component": "SAScore"},
                    {"name": "QED", "type": "native", "component": "QED"},
                ],
            },
        }
        ttp = TTP.model_validate(data)
        summary = get_mapper_summary(ttp)
        assert summary["n_pre_scoring"] == 0
        assert summary["n_native_objectives"] == 2
        assert summary["n_total_components"] == 2

    def test_external_components_in_summary(self):
        ttp = TTP.model_validate(FULL_TTP)
        summary = get_mapper_summary(ttp)
        assert [c["name"] for c in summary["external_components"]] == [
            "Binding Affinity",
            "Binder Probability",
            "Pose Quality",
        ]

    def test_no_external_components_when_native_only(self):
        ttp = TTP.model_validate(MINIMAL_TTP)
        assert get_mapper_summary(ttp)["external_components"] == []


class TestDescribeExternalObjectives:
    """Every external objective maps to the same ExternalRESTScorer class, so
    the description is the only thing that tells the instances apart."""

    def test_service_and_url_reported(self):
        """The URL is where the service actually is, not what the TTP guessed.

        A TTP carries whatever host its author wrote down, which is right only
        on the deployment it came from; the configured URL is the one that
        answers here, so that is the one reported and used.
        """

        ttp = TTP.model_validate(FULL_TTP)
        by_name = {c["name"]: c for c in describe_external_objectives(ttp)}

        affinity = by_name["Binding Affinity"]
        assert affinity["component"] == "ExternalRESTScorer"
        assert affinity["service"] == "boltz"
        assert affinity["service_url"] == get_config().enabled_external_services["boltz"]

        assert by_name["Pose Quality"]["service"] == "posebusters"

    def test_same_service_distinguished_by_selector(self):
        """Two Boltz objectives differ only by affinity_mode."""
        ttp = TTP.model_validate(FULL_TTP)
        by_name = {c["name"]: c for c in describe_external_objectives(ttp)}

        assert by_name["Binding Affinity"]["selects"]["affinity_mode"] == "pred_value"
        assert (
            by_name["Binder Probability"]["selects"]["affinity_mode"]
            == "probability_binary"
        )

    def test_measures_blurb_present(self):
        ttp = TTP.model_validate(FULL_TTP)
        by_name = {c["name"]: c for c in describe_external_objectives(ttp)}
        assert "Boltz" in by_name["Binding Affinity"]["measures"]
        assert "PoseBusters" in by_name["Pose Quality"]["measures"]

    def test_pipeline_role_reported(self):
        ttp = TTP.model_validate(FULL_TTP)
        by_name = {c["name"]: c for c in describe_external_objectives(ttp)}
        assert by_name["Pose Quality"]["depends_on"] == "Binding Affinity"
        assert "depends_on" not in by_name["Binding Affinity"]

    def test_protein_sequence_elided(self):
        """The sequence is identical per objective and would swamp the output."""
        ttp = TTP.model_validate(FULL_TTP)
        seq = FULL_TTP["target"]["protein_sequence"]
        for comp in describe_external_objectives(ttp):
            assert comp.get("params", {}).get("protein_sequence") != seq
        affinity = next(
            c for c in describe_external_objectives(ttp)
            if c["name"] == "Binding Affinity"
        )
        assert affinity["params"]["protein_sequence"] == f"<{len(seq)} aa, elided>"

    def test_plumbing_params_not_duplicated(self):
        ttp = TTP.model_validate(FULL_TTP)
        for comp in describe_external_objectives(ttp):
            params = comp.get("params", {})
            for key in ("service", "service_url", "component_name", "depends_on"):
                assert key not in params

    def test_reuses_prebuilt_scorer_config(self):
        ttp = TTP.model_validate(FULL_TTP)
        config = ttp_to_scorer_config(ttp)
        assert describe_external_objectives(
            ttp, config
        ) == describe_external_objectives(ttp)

    def test_missing_selector_marked_incomplete(self):
        """An admet objective without 'endpoint' cannot be mapped; say so
        instead of dropping the objective from the description."""
        data = {
            "id": "admet-no-endpoint",
            "name": "ADMET missing selector",
            "scoring": {
                "objectives": [
                    {
                        "name": "Permeability",
                        "type": "external",
                        "service": "admet",
                        "service_url": "http://admet-predictor:8084/score",
                    },
                ],
            },
        }
        ttp = TTP.model_validate(data)
        (comp,) = describe_external_objectives(ttp)
        assert comp["service"] == "admet"
        assert "incomplete" in comp


# ---------------------------------------------------------------------------
# Target sequence from a structure file
# ---------------------------------------------------------------------------

CHAIN_A = ["MSE", "ALA", "LYS", "GLU", "THR", "ALA", "ALA", "ALA", "LYS", "PHE", "GLU", "ARG", "GLN", "HIS"]
SEQ_A = "MAKETAAAKFERQH"
CHAIN_C = ["GLY", "SER", "HIS", "MET", "LEU", "GLU", "ASP", "PRO", "VAL", "ILE", "TRP", "TYR"]
SEQ_C = "GSHMLEDPVIWY"
DNA = ["DA", "DC", "DG", "DT", "DA", "DC", "DG", "DT", "DA", "DC", "DG", "DT"]


def _pdb_atom(serial: int, name: str, resname: str, chain: str, resseq: int, icode: str = " ", alt: str = " ", record: str = "ATOM") -> str:
    return (
        f"{record:<6}{serial:>5} {name:<4}{alt}{resname:>3} {chain}{resseq:>4}{icode}   "
        f"{1.0:>8.3f}{2.0:>8.3f}{3.0:>8.3f}{1.0:>6.2f}{20.0:>6.2f}           {name[0]:>1}"
    )


def _seqres(chain: str, residues: list[str]) -> list[str]:
    lines = []
    for n, start in enumerate(range(0, len(residues), 13), 1):
        lines.append(f"SEQRES {n:>3} {chain} {len(residues):>4}  {' '.join(residues[start:start + 13])}")
    return lines


def _pdb(chains: dict[str, list[str]], seqres: dict[str, list[str]] | None = None) -> str:
    lines = ["HEADER    TEST"]
    for chain, residues in (seqres or {}).items():
        lines += _seqres(chain, residues)
    serial = 1
    for chain, residues in chains.items():
        for i, resname in enumerate(residues, 1):
            record = "HETATM" if resname == "MSE" else "ATOM"
            for atom in ("N", "CA"):
                lines.append(_pdb_atom(serial, atom, resname, chain, i, record=record))
                serial += 1
    return "\n".join([*lines, "END", ""])


CIF_ENTITIES = """data_TEST
#
_entry.id TEST
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_strand_id
1 'polypeptide(L)'
;(MSE)AKETAAAKFERQH
;
;MAKETAAAKF
ERQH
;
A,B
2 polydeoxyribonucleotide '(DA)(DC)(DG)(DT)(DA)(DC)(DG)(DT)(DA)(DC)' ACGTACGTAC D
3 'polypeptide(L)' GSHMLEDPVIWYX GSHMLEDPVIWYX C
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 1
C 3
D 2
#
"""


def _cif_atoms(chains: dict[str, list[str]], extra: list[str] = ()) -> str:
    rows = []
    serial = 1
    for chain, residues in chains.items():
        for i, resname in enumerate(residues, 1):
            group = "HETATM" if resname in ("MSE", "HOH", "ATP") else "ATOM"
            for atom in ("N", "CA"):
                rows.append(f"{group} {serial} {atom[0]} {atom} . {resname} {chain} 1 {i} ? 1.0 2.0 3.0 1.00 20.0 {i} {resname} {chain} 1")
                serial += 1
    return "\n".join(
        [
            "data_ATOMS",
            "loop_",
            *(f"_atom_site.{f}" for f in (
                "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
                "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x",
                "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_comp_id",
                "auth_asym_id", "pdbx_PDB_model_num",
            )),
            *rows,
            *extra,
            "#",
            "",
        ]
    )


class TestStructureChains:
    def test_pdb_seqres_is_the_construct(self):
        text = _pdb({"A": CHAIN_A[3:10], "B": DNA, "C": CHAIN_C}, {"A": CHAIN_A, "B": DNA, "C": CHAIN_C})
        chains = pdb_chains(text)
        assert [(c.id, c.sequence, c.source) for c in chains] == [("A", SEQ_A, "construct"), ("C", SEQ_C, "construct")]

    def test_pdb_without_seqres_reads_the_first_model(self):
        lines = _pdb({"A": CHAIN_A}).splitlines()[:-1]
        # An alternate location, an insertion code, a ligand and water, then a second model.
        lines += [
            _pdb_atom(900, "CB", "HIS", "A", 14, alt="B"),
            _pdb_atom(901, "CA", "GLY", "A", 14, icode="A"),
            _pdb_atom(902, "PG", "ATP", "A", 101, record="HETATM"),
            _pdb_atom(903, "O", "HOH", "A", 201, record="HETATM"),
            "ENDMDL",
            "MODEL        2",
            _pdb_atom(904, "CA", "TRP", "A", 1),
        ]
        (chain,) = pdb_chains("\n".join(lines))
        assert (chain.id, chain.sequence, chain.source) == ("A", SEQ_A + "G", "atoms")

    def test_cif_entity_poly_by_author_chain(self):
        chains = cif_chains(CIF_ENTITIES)
        assert [(c.id, c.sequence, c.source) for c in chains] == [
            ("A", SEQ_A, "construct"),
            ("B", SEQ_A, "construct"),
            ("C", SEQ_C, "construct"),
        ]

    def test_cif_single_entity_without_loop(self):
        text = (
            "data_ONE\n_entity_poly.entity_id 1\n_entity_poly.type 'polypeptide(L)'\n"
            "_entity_poly.pdbx_seq_one_letter_code_can\n;MAKETAAAKF\nERQH\n;\n_entity_poly.pdbx_strand_id A\n"
        )
        assert [(c.id, c.sequence) for c in cif_chains(text)] == [("A", SEQ_A)]

    def test_cif_entity_poly_seq_keeps_one_monomer_per_position(self):
        rows = [f"1 {i} {name} n" for i, name in enumerate(CHAIN_A, 1)] + ["1 3 ARG n"]
        text = "\n".join(
            [
                "data_SEQ",
                "loop_",
                "_entity_poly_seq.entity_id",
                "_entity_poly_seq.num",
                "_entity_poly_seq.mon_id",
                "_entity_poly_seq.hetero",
                *rows,
                "#",
                "loop_",
                "_struct_asym.id",
                "_struct_asym.entity_id",
                "A 1",
                "#",
            ]
        )
        assert [(c.id, c.sequence) for c in cif_chains(text)] == [("A", SEQ_A)]

    def test_cif_without_construct_reads_atoms(self):
        extra = [
            "HETATM 990 P PG . ATP E 2 . ? 1.0 2.0 3.0 1.00 20.0 301 ATP A 1",
            "HETATM 991 O O . HOH F 3 . ? 1.0 2.0 3.0 1.00 20.0 401 HOH A 1",
            "ATOM 992 C CA . TRP A 1 1 ? 1.0 2.0 3.0 1.00 20.0 1 TRP A 2",
        ]
        chains = cif_chains(_cif_atoms({"A": CHAIN_A, "C": CHAIN_C}, extra))
        assert [(c.id, c.sequence, c.source) for c in chains] == [("A", SEQ_A, "atoms"), ("C", SEQ_C, "atoms")]


class TestTargetFromStructure:
    @pytest.fixture
    def shared(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setenv("IP_SHARED_DIR", str(tmp_path))
        (tmp_path / "uploads").mkdir()
        (tmp_path / "uploads" / "multi.cif").write_text(CIF_ENTITIES)
        (tmp_path / "uploads" / "dimer.pdb").write_text(_pdb({"A": CHAIN_A, "B": CHAIN_A}))
        return tmp_path

    def test_relative_to_the_shared_volume(self, shared: Path):
        assert structure_sequence("uploads/multi.cif", "C") == SEQ_C
        assert structure_sequence(str(shared / "uploads" / "multi.cif"), "B") == SEQ_A

    def test_one_protein_needs_no_chain(self, shared: Path):
        assert structure_sequence("uploads/dimer.pdb") == SEQ_A

    def test_explains_what_is_wrong(self, shared: Path):
        with pytest.raises(ValueError, match="several proteins.*A \\(14 aa\\).*C \\(12 aa\\)"):
            structure_sequence("uploads/multi.cif")
        with pytest.raises(ValueError, match="no protein chain 'D'"):
            structure_sequence("uploads/multi.cif", "D")
        with pytest.raises(ValueError, match="not found"):
            structure_sequence("uploads/missing.cif", "A")

    def test_mapper_folds_the_structure_chain(self, shared: Path):
        data = {
            **FULL_TTP,
            "target": {"structure_path": "uploads/multi.cif", "structure_chain": "C", "pdb_id": "1ABC", "msa_path": "/data/msa/x.a3m"},
        }
        ttp = TTP.model_validate(data)
        assert target_sequence(ttp.target) == SEQ_C
        config = ttp_to_scorer_config(ttp)
        boltz = [
            c["ExternalRESTScorer"]["endpoint"][0]["params"]
            for c in config["component"]
            if "ExternalRESTScorer" in c and c["ExternalRESTScorer"]["endpoint"][0]["params"]["service"] == "boltz"
        ]
        assert boltz and all(p["protein_sequence"] == SEQ_C and p["msa_path"] == "/data/msa/x.a3m" for p in boltz)
        assert all("structure_path" not in p for p in boltz)

    def test_mapper_reports_an_unreadable_structure(self, shared: Path):
        ttp = TTP.model_validate({**FULL_TTP, "target": {"structure_path": "uploads/multi.cif"}})
        with pytest.raises(ValueError, match="structure_chain"):
            ttp_to_scorer_config(ttp)
        native_only = TTP.model_validate({**MINIMAL_TTP, "target": {"structure_path": "uploads/gone.cif"}})
        assert ttp_to_scorer_config(native_only)["component"]

    def test_round_trips_through_the_store(self, shared: Path, tmp_store: TTPStore):
        created = tmp_store.create(TTP.model_validate({**MINIMAL_TTP, "target": {"structure_path": "uploads/dimer.pdb"}}))
        loaded = tmp_store.get(created.id)
        assert (loaded.target.structure_path, loaded.target.protein_sequence) == ("uploads/dimer.pdb", None)
        assert target_sequence(loaded.target) == SEQ_A
