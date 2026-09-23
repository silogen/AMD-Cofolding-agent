"""Tests for block 08 campaign contract schema (schema.py additions).

Covers:
- GenerationMode enum values
- LinkInventInput @field_validator: parse, attachment point count, map numbers, cross-fragment uniqueness
- LibInventInput @field_validator: parse, at least 1 attachment, map numbers, duplicate check
- GenerationSection @model_validator: mode-specific required inputs
- TTP.generation field backward compat (None by default)
- SMILES helper functions exposed as module-level
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from reinvent4_mcp.ttp.schema import (
    GenerationMode,
    GenerationSection,
    LibInventInput,
    LinkInventInput,
    Mol2MolInput,
    ScoringSection,
    TTP,
    _validate_libinvent_scaffold,
    _validate_linkinvent_fragment,
)


# ===========================================================================
# Category 1: _validate_linkinvent_fragment helper
# ===========================================================================

class TestLinkInventFragmentValidator:
    def test_valid_fragment_no_errors(self):
        errs = _validate_linkinvent_fragment("[*:1]CC", 0)
        assert errs == []

    def test_invalid_smiles_returns_error(self):
        errs = _validate_linkinvent_fragment("NOT_SMILES!!!", 0)
        assert len(errs) == 1
        assert "parse" in errs[0].lower() or "cannot" in errs[0].lower()

    def test_fragment_with_no_attachment_point(self):
        errs = _validate_linkinvent_fragment("CC", 0)   # no [*:n]
        assert any("attachment" in e.lower() or "attachment point" in e.lower()
                   for e in errs)

    def test_fragment_with_bare_wildcard_no_map(self):
        errs = _validate_linkinvent_fragment("[*]CC", 0)   # [*] without map
        assert any("map number" in e.lower() or "map" in e.lower() for e in errs)

    def test_fragment_with_two_attachments_fails(self):
        errs = _validate_linkinvent_fragment("[*:1]CC[*:2]", 0)   # 2 attachments
        assert any("exactly 1" in e or "1 attachment" in e for e in errs)

    def test_valid_aromatic_fragment(self):
        errs = _validate_linkinvent_fragment("[*:1]c1ccccc1", 0)
        assert errs == []

    def test_frag_idx_in_error_message(self):
        errs = _validate_linkinvent_fragment("INVALID", 3)
        assert "3" in errs[0]


# ===========================================================================
# Category 2: _validate_libinvent_scaffold helper
# ===========================================================================

class TestLibInventScaffoldValidator:
    def test_valid_monosubstituted_scaffold(self):
        errs = _validate_libinvent_scaffold("c1ccc([*:1])cc1")
        assert errs == []

    def test_valid_disubstituted_scaffold(self):
        errs = _validate_libinvent_scaffold("c1ccc([*:1])cc1[*:2]")
        assert errs == []

    def test_invalid_smiles_returns_error(self):
        errs = _validate_libinvent_scaffold("NOT_SMILES!!!")
        assert len(errs) >= 1
        assert "parse" in errs[0].lower() or "cannot" in errs[0].lower()

    def test_scaffold_with_no_attachment_point(self):
        errs = _validate_libinvent_scaffold("c1ccccc1")  # no [*:n]
        assert len(errs) == 1
        assert "r-group" in errs[0].lower() or "attachment" in errs[0].lower()

    def test_scaffold_bare_wildcard_no_map(self):
        errs = _validate_libinvent_scaffold("c1ccc([*])cc1")  # [*] no map
        assert any("map number" in e.lower() for e in errs)

    def test_scaffold_duplicate_map_numbers(self):
        errs = _validate_libinvent_scaffold("[*:1]c1ccc([*:1])cc1")  # both :1
        assert any("duplicate" in e.lower() for e in errs)

    def test_aspirin_scaffold_valid(self):
        errs = _validate_libinvent_scaffold("CC(=O)Oc1ccccc1[*:1]")
        assert errs == []


# ===========================================================================
# Category 3: LinkInventInput Pydantic model
# ===========================================================================

class TestLinkInventInput:
    def test_valid_two_fragments(self):
        li = LinkInventInput(fragments=["[*:1]CC", "[*:2]OC"])
        assert len(li.fragments) == 2

    def test_single_fragment_raises(self):
        with pytest.raises(Exception, match="at least 2"):
            LinkInventInput(fragments=["[*:1]CC"])

    def test_empty_fragments_raises(self):
        with pytest.raises(Exception):
            LinkInventInput(fragments=[])

    def test_invalid_smiles_in_fragments_raises(self):
        with pytest.raises(Exception):
            LinkInventInput(fragments=["[*:1]CC", "NOT_SMILES!!!"])

    def test_fragments_with_no_attachment_raises(self):
        with pytest.raises(Exception, match="attachment"):
            LinkInventInput(fragments=["CC", "[*:2]OC"])

    def test_fragments_sharing_map_number_raises(self):
        with pytest.raises(Exception, match="unique"):
            LinkInventInput(fragments=["[*:1]CC", "[*:1]OC"])  # both use :1

    def test_optional_fields_default_none(self):
        li = LinkInventInput(fragments=["[*:1]CC", "[*:2]OC"])
        assert li.allowed_atom_types is None
        assert li.max_linker_length is None
        assert li.min_linker_length is None


# ===========================================================================
# Category 4: LibInventInput Pydantic model
# ===========================================================================

class TestLibInventInput:
    def test_valid_scaffold(self):
        lb = LibInventInput(scaffold="c1ccc([*:1])cc1")
        assert lb.scaffold == "c1ccc([*:1])cc1"

    def test_invalid_smiles_raises(self):
        with pytest.raises(Exception):
            LibInventInput(scaffold="NOT_SMILES")

    def test_no_attachment_raises(self):
        with pytest.raises(Exception, match="attachment"):
            LibInventInput(scaffold="c1ccccc1")

    def test_bare_wildcard_raises(self):
        with pytest.raises(Exception, match="map"):
            LibInventInput(scaffold="c1ccc([*])cc1")

    def test_duplicate_map_raises(self):
        with pytest.raises(Exception, match="duplicate"):
            LibInventInput(scaffold="[*:1]c1ccc([*:1])cc1")

    def test_decoration_positions_optional(self):
        lb = LibInventInput(scaffold="c1ccc([*:1])cc1")
        assert lb.decoration_positions is None


# ===========================================================================
# Category 5: GenerationSection
# ===========================================================================

class TestGenerationSection:
    def test_defaults_to_reinvent_mode(self):
        g = GenerationSection()
        assert g.mode == GenerationMode.reinvent

    def test_default_values(self):
        g = GenerationSection()
        assert g.batch_size == 128
        assert g.n_steps == 100
        assert g.checkpoint_every_n == 50
        assert g.model == ".reinvent"
        assert g.seed is None
        assert g.device == "cuda"

    def test_linkinvent_requires_input(self):
        with pytest.raises(Exception, match="linkinvent_input"):
            GenerationSection(mode="linkinvent")

    def test_libinvent_requires_input(self):
        with pytest.raises(Exception, match="libinvent_input"):
            GenerationSection(mode="libinvent")

    def test_valid_linkinvent_section(self):
        g = GenerationSection(
            mode="linkinvent",
            linkinvent_input=LinkInventInput(fragments=["[*:1]CC", "[*:2]OC"]),
        )
        assert g.mode == GenerationMode.linkinvent
        assert len(g.linkinvent_input.fragments) == 2

    def test_valid_libinvent_section(self):
        g = GenerationSection(
            mode="libinvent",
            libinvent_input=LibInventInput(scaffold="c1ccc([*:1])cc1"),
        )
        assert g.mode == GenerationMode.libinvent

    def test_artifact_ids_default_empty(self):
        g = GenerationSection()
        assert g.target_artifact_ids == []
        assert g.prepared_structure_artifact_ids == []
        assert g.site_artifact_ids == []


# ===========================================================================
# Category 6: TTP.generation field backward compat
# ===========================================================================

class TestTTPGenerationField:
    def _minimal_ttp(self, **extra):
        return TTP(
            id="test-ttp",
            name="Test",
            scoring=ScoringSection(objectives=[]),
            **extra,
        )

    def test_generation_defaults_to_none(self):
        ttp = self._minimal_ttp()
        assert ttp.generation is None

    def test_existing_ttp_without_generation_still_loads(self):
        """Backward compat: a TTP with no generation key must still validate."""
        ttp = self._minimal_ttp()
        ttp_dict = ttp.model_dump()
        # Simulate a pre-block-08 TTP without 'generation' key
        ttp_dict.pop("generation", None)
        reloaded = TTP.model_validate(ttp_dict)
        assert reloaded.generation is None

    def test_ttp_with_reinvent_generation(self):
        ttp = self._minimal_ttp(
            generation=GenerationSection(mode="reinvent")
        )
        assert ttp.generation.mode == GenerationMode.reinvent

    def test_ttp_with_linkinvent_generation(self):
        ttp = self._minimal_ttp(
            generation=GenerationSection(
                mode="linkinvent",
                model=".linkinvent",
                linkinvent_input=LinkInventInput(
                    fragments=["[*:1]c1ccccc1", "[*:2]OCC"]
                ),
            )
        )
        assert ttp.generation.mode == GenerationMode.linkinvent
        assert len(ttp.generation.linkinvent_input.fragments) == 2

    def test_ttp_with_libinvent_generation(self):
        ttp = self._minimal_ttp(
            generation=GenerationSection(
                mode="libinvent",
                model=".libinvent",
                libinvent_input=LibInventInput(scaffold="c1ccc([*:1])cc1"),
            )
        )
        assert ttp.generation.mode == GenerationMode.libinvent

    def test_ttp_with_mol2mol_generation(self):
        ttp = self._minimal_ttp(
            generation=GenerationSection(
                mode="mol2mol",
                model=".m2m_medium",
                mol2mol_input=Mol2MolInput(seed_smiles=["CCO", "CCN"]),
            )
        )
        assert ttp.generation.mode == GenerationMode.mol2mol
        assert ttp.generation.mol2mol_input.seed_smiles == ["CCO", "CCN"]

    def test_mol2mol_requires_seed_smiles(self):
        with pytest.raises(Exception, match="mol2mol_input"):
            GenerationSection(mode="mol2mol", model=".m2m_medium")

    def test_mol2mol_rejects_invalid_seed(self):
        with pytest.raises(Exception, match="Invalid Mol2Mol seed"):
            Mol2MolInput(seed_smiles=["NOT_A_SMILES"])

    def test_generation_roundtrips_json(self):
        import json
        ttp = self._minimal_ttp(
            generation=GenerationSection(
                mode="linkinvent",
                linkinvent_input=LinkInventInput(fragments=["[*:1]CC", "[*:2]OC"]),
            )
        )
        dumped = json.dumps(ttp.model_dump())
        reloaded = TTP.model_validate(json.loads(dumped))
        assert reloaded.generation.mode == GenerationMode.linkinvent
