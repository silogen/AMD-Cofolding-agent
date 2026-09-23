"""Tests for block 08 generation config mapping and TOML generation.

Covers:
- mapper.ttp_to_generation_config: per-mode output, error on missing inputs
- toml_generator: [reinvent] / [libinvent] / [linkinvent] blocks
- generate_generation_toml dispatcher
- Backward compat: generation=None produces unchanged TOML
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from reinvent4_mcp.ttp.schema import (
    GenerationSection, GenerationMode,
    LinkInventInput, LibInventInput, Mol2MolInput,
    ScoringSection, TTP,
)
from reinvent4_mcp.ttp.mapper import ttp_to_generation_config
from reinvent4_mcp.jobs.toml_generator import (
    generate_reinvent_toml,
    generate_libinvent_toml,
    generate_linkinvent_toml,
    generate_generation_toml,
)


def _minimal_ttp(**generation_kwargs):
    gen = GenerationSection(**generation_kwargs) if generation_kwargs else None
    return TTP(
        id="test-ttp",
        name="Test",
        scoring=ScoringSection(objectives=[]),
        generation=gen,
    )


def _li_input(f1="[*:1]CC", f2="[*:2]OC"):
    return LinkInventInput(fragments=[f1, f2])


def _lb_input(scaffold="c1ccc([*:1])cc1"):
    return LibInventInput(scaffold=scaffold)


def _m2m_input(*seeds):
    return Mol2MolInput(seed_smiles=list(seeds or ("CCO",)))


# ===========================================================================
# Category 1: ttp_to_generation_config
# ===========================================================================

class TestTTPToGenerationConfig:
    def test_none_generation_returns_empty(self):
        ttp = _minimal_ttp()
        assert ttp_to_generation_config(ttp) == {}

    def test_reinvent_config_keys(self):
        ttp = _minimal_ttp(mode="reinvent")
        cfg = ttp_to_generation_config(ttp)
        assert cfg["mode"] == "reinvent"
        assert "batch_size" in cfg
        assert "n_steps" in cfg
        assert "model" in cfg

    def test_libinvent_config_has_scaffold(self):
        ttp = _minimal_ttp(mode="libinvent", libinvent_input=_lb_input())
        cfg = ttp_to_generation_config(ttp)
        assert cfg["mode"] == "libinvent"
        assert cfg["libinvent_input"]["scaffold"] == "c1ccc([*:1])cc1"
        assert cfg["linkinvent_input"] is None

    def test_linkinvent_config_has_fragments(self):
        ttp = _minimal_ttp(mode="linkinvent", linkinvent_input=_li_input())
        cfg = ttp_to_generation_config(ttp)
        assert cfg["mode"] == "linkinvent"
        assert cfg["linkinvent_input"]["fragments"] == ["[*:1]CC", "[*:2]OC"]
        assert cfg["libinvent_input"] is None

    def test_mol2mol_config_has_seed_smiles(self):
        ttp = _minimal_ttp(
            mode="mol2mol",
            model=".m2m_medium",
            mol2mol_input=_m2m_input("CCO", "CCN"),
        )
        cfg = ttp_to_generation_config(ttp)
        assert cfg["mode"] == "mol2mol"
        assert cfg["model"] == ".m2m_medium"
        assert cfg["mol2mol_input"]["seed_smiles"] == ["CCO", "CCN"]

    def test_linkinvent_raises_without_fragments(self):
        """mapper should also catch this even though Pydantic already did."""
        # Manually create a GenerationSection that bypasses Pydantic validation
        # by patching the linkinvent_input after construction isn't possible with
        # extra=forbid; instead test via the mapper's defensive check.
        ttp = _minimal_ttp(mode="linkinvent", linkinvent_input=_li_input())
        # Simulate a corrupt/None input at mapper level
        ttp.generation.linkinvent_input = None  # bypassing pydantic
        with pytest.raises(ValueError, match="linkinvent"):
            ttp_to_generation_config(ttp)

    def test_libinvent_raises_without_scaffold(self):
        ttp = _minimal_ttp(mode="libinvent", libinvent_input=_lb_input())
        ttp.generation.libinvent_input = None   # bypass pydantic
        with pytest.raises(ValueError, match="libinvent"):
            ttp_to_generation_config(ttp)

    def test_mol2mol_raises_without_seed_smiles(self):
        ttp = _minimal_ttp(
            mode="mol2mol",
            model=".m2m_medium",
            mol2mol_input=_m2m_input("CCO"),
        )
        ttp.generation.mol2mol_input = None
        with pytest.raises(ValueError, match="mol2mol"):
            ttp_to_generation_config(ttp)

    def test_config_includes_artifact_refs(self):
        ttp = _minimal_ttp(
            mode="reinvent",
            target_artifact_ids=["target-001"],
            site_artifact_ids=["site-a"],
        )
        cfg = ttp_to_generation_config(ttp)
        assert cfg["target_artifact_ids"] == ["target-001"]
        assert cfg["site_artifact_ids"] == ["site-a"]

    def test_seed_forwarded(self):
        ttp = _minimal_ttp(mode="reinvent", seed=42)
        cfg = ttp_to_generation_config(ttp)
        assert cfg["seed"] == 42

    def test_device_forwarded(self):
        ttp = _minimal_ttp(mode="reinvent", device="cpu")
        cfg = ttp_to_generation_config(ttp)
        assert cfg["device"] == "cpu"


# ===========================================================================
# Category 2: TOML section headers
# ===========================================================================

_DUMMY_SCORER = {"type": "arithmetic_mean", "component": []}


class TestTomlHeaders:
    def test_reinvent_toml_header(self):
        toml = generate_reinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/reinvent.prior",
        )
        assert 'run_type = "staged_learning"' in toml

    def test_libinvent_toml_header(self):
        toml = generate_libinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/libinvent.prior",
            scaffold="c1ccc([*:1])cc1",
        )
        assert 'run_type = "staged_learning"' in toml

    def test_libinvent_toml_scaffold_block(self):
        toml = generate_libinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/libinvent.prior",
            scaffold="c1ccc([*:1])cc1",
        )
        assert 'smiles_file = "scaffolds.smi"' in toml
        assert "# Scaffold: c1ccc([*:1])cc1" in toml

    def test_linkinvent_toml_header(self):
        toml = generate_linkinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/linkinvent.prior",
            fragments=["[*:1]CC", "[*:2]OC"],
        )
        assert 'run_type = "staged_learning"' in toml

    def test_linkinvent_toml_fragment_blocks(self):
        toml = generate_linkinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/linkinvent.prior",
            fragments=["[*:1]CC", "[*:2]OC"],
        )
        assert 'smiles_file = "warheads.smi"' in toml
        assert "# Fragments: [*:1]CC | [*:2]OC" in toml

    def test_linkinvent_toml_has_two_fragment_blocks(self):
        toml = generate_linkinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/linkinvent.prior",
            fragments=["[*:1]CC", "[*:2]OC"],
        )
        # REINVENT4 v4 reads one pipe-delimited pair from warheads.smi.
        assert toml.count('smiles_file = "warheads.smi"') == 1
        assert toml.count("# Fragments:") == 1

    def test_linkinvent_requires_two_fragments(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_linkinvent_toml(
                scorer_config=_DUMMY_SCORER,
                prior_file="/models/linkinvent.prior",
                fragments=["[*:1]CC"],
            )

    def test_libinvent_with_decoration_positions(self):
        toml = generate_libinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/libinvent.prior",
            scaffold="c1ccc([*:1])cc1[*:2]",
            decoration_positions=[1, 2],
        )
        assert "# Decoration positions: 1, 2" in toml


# ===========================================================================
# Category 3: generate_generation_toml dispatcher
# ===========================================================================

class TestGenerateGenerationToml:
    def test_reinvent_mode_dispatches_to_staged_learning(self):
        cfg = {"mode": "reinvent", "batch_size": 64, "n_steps": 50, "device": "cpu"}
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/prior.prior",
        )
        assert 'run_type = "staged_learning"' in toml

    def test_libinvent_mode_dispatches_correctly(self):
        cfg = {
            "mode": "libinvent", "batch_size": 32, "n_steps": 20, "device": "cpu",
            "libinvent_input": {"scaffold": "c1ccc([*:1])cc1", "decoration_positions": None},
        }
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/prior.prior",
        )
        assert 'run_type = "staged_learning"' in toml
        assert 'smiles_file = "scaffolds.smi"' in toml

    def test_linkinvent_mode_dispatches_correctly(self):
        cfg = {
            "mode": "linkinvent", "batch_size": 32, "n_steps": 20, "device": "cpu",
            "linkinvent_input": {"fragments": ["[*:1]CC", "[*:2]OC"]},
        }
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/prior.prior",
        )
        assert 'run_type = "staged_learning"' in toml
        assert 'smiles_file = "warheads.smi"' in toml

    def test_mol2mol_mode_dispatches_with_seed_file(self):
        cfg = {
            "mode": "mol2mol",
            "batch_size": 32,
            "n_steps": 20,
            "device": "cpu",
            "mol2mol_input": {"seed_smiles": ["CCO", "CCN"]},
        }
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/mol2mol.prior",
        )
        assert 'smiles_file = "seed.smi"' in toml
        assert "max_steps = 20" in toml

    def test_unknown_mode_raises(self):
        cfg = {"mode": "unknown_mode", "batch_size": 64, "n_steps": 10}
        with pytest.raises(ValueError, match="Unknown generation mode"):
            generate_generation_toml(
                scorer_config=_DUMMY_SCORER,
                generation_config=cfg,
                prior_file="/models/prior.prior",
            )

    def test_libinvent_without_scaffold_raises(self):
        cfg = {"mode": "libinvent", "batch_size": 64, "n_steps": 10,
               "libinvent_input": None}
        with pytest.raises(ValueError, match="scaffold"):
            generate_generation_toml(
                scorer_config=_DUMMY_SCORER,
                generation_config=cfg,
                prior_file="/models/prior.prior",
            )

    def test_batch_size_forwarded(self):
        cfg = {"mode": "reinvent", "batch_size": 256, "n_steps": 50, "device": "cpu"}
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/prior.prior",
        )
        assert "batch_size = 256" in toml

    def test_n_steps_forwarded_as_max_steps(self):
        cfg = {"mode": "reinvent", "batch_size": 64, "n_steps": 77, "device": "cpu"}
        toml = generate_generation_toml(
            scorer_config=_DUMMY_SCORER,
            generation_config=cfg,
            prior_file="/models/prior.prior",
        )
        assert "max_steps = 77" in toml


# ===========================================================================
# Category 4: Backward compat — existing TTP scorer mapping unchanged
# ===========================================================================

class TestScorerBackwardCompat:
    def test_existing_ttp_mapper_unchanged(self):
        """ttp_to_scorer_config must still work for TTPs without generation."""
        from reinvent4_mcp.ttp.mapper import ttp_to_scorer_config  # noqa: PLC0415
        ttp = _minimal_ttp()   # no generation section
        cfg = ttp_to_scorer_config(ttp)
        assert "type" in cfg
        assert "component" in cfg

    def test_generate_reinvent_toml_unchanged(self):
        """generate_reinvent_toml must still accept the same args as before."""
        toml = generate_reinvent_toml(
            scorer_config=_DUMMY_SCORER,
            prior_file="/models/reinvent.prior",
            batch_size=128,
            max_steps=100,
        )
        assert 'run_type = "staged_learning"' in toml
        assert "batch_size = 128" in toml
        assert "max_steps = 100" in toml
