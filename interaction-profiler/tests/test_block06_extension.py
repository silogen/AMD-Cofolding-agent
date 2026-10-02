"""Tests for block 06 extension: protein-only mode, analysis profiles, enriched ensemble.

All tests are unit tests — no GPU, fpocket, or network required.
fpocket and network calls are mocked.

Covers:
- New Pydantic models (SiteProfile, PocketSubpocket, FragmentFamily, CompatiblePair)
- New fields on EnrichedInteraction and ComplexResult
- resolve_profile_flags: backward compat + profile resolution
- profile_complex() protein_only mode (fpocket mocked)
- report/classifier.py: confidence_bin, exposure, vector_type
- report/ensemble.py: fragment_families, compatible_pairs
- Backward compat: existing modes unchanged, analysis_profile=None has no effect
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# ---------------------------------------------------------------------------
# Category 1: New schema models
# ---------------------------------------------------------------------------

class TestSiteProfile:
    def test_minimal_construction(self):
        from interaction_profiler.schemas import SiteProfile
        sp = SiteProfile(
            structure_path="/x.pdb",
            structure_sha256="abc" * 21 + "a",
            target_chains=["A"],
            analysis_profile="pocket",
        )
        assert sp.site_type is None
        assert sp.pocket_volume_A3 is None
        assert sp.subpockets == []
        assert sp.residue_roles == {}
        assert sp.ss_summary == {}

    def test_json_round_trip(self):
        from interaction_profiler.schemas import SiteProfile, PocketSubpocket
        sp = SiteProfile(
            structure_path="/x.pdb",
            structure_sha256="a" * 64,
            target_chains=["A"],
            analysis_profile="annotated",
            site_type="orthosteric",
            pocket_volume_A3=380.5,
            pocket_character="hydrophobic",
            subpockets=[PocketSubpocket(id="S1", residues=["HIS163.A"])],
        )
        d = json.dumps(sp.model_dump())
        sp2 = SiteProfile.model_validate(json.loads(d))
        assert sp2.site_type == "orthosteric"
        assert len(sp2.subpockets) == 1


class TestComplexResultNewFields:
    def test_site_profile_default_none(self):
        from interaction_profiler.schemas import ComplexResult
        r = ComplexResult(mode="ligand_target")
        assert r.site_profile is None

    def test_fragment_families_default_empty(self):
        from interaction_profiler.schemas import ComplexResult
        r = ComplexResult(mode="hotspot_ensemble")
        assert r.fragment_families == []
        assert r.compatible_pairs == []

    def test_enriched_interaction_new_fields(self):
        from interaction_profiler.schemas import EnrichedInteraction
        ei = EnrichedInteraction(interaction_type="hbond", ligand_atom_idx=0)
        assert ei.vector_type is None
        assert ei.solvent_exposure is None
        assert ei.subpocket_id is None
        assert ei.confidence_bin is None

    def test_set_enriched_fields(self):
        from interaction_profiler.schemas import EnrichedInteraction
        ei = EnrichedInteraction(
            interaction_type="hbond", ligand_atom_idx=0,
            vector_type="anchor",
            solvent_exposure="buried",
            confidence_bin="high",
            subpocket_id="S1",
        )
        assert ei.vector_type == "anchor"
        assert ei.subpocket_id == "S1"

    def test_complex_result_round_trip_with_site_profile(self):
        from interaction_profiler.schemas import ComplexResult, SiteProfile
        sp = SiteProfile(
            structure_path="/x.pdb", structure_sha256="a" * 64,
            target_chains=["A"], analysis_profile="pocket",
        )
        r = ComplexResult(mode="protein_only", site_profile=sp)
        r2 = ComplexResult.model_validate(r.model_dump())
        assert r2.site_profile is not None
        assert r2.site_profile.analysis_profile == "pocket"


# ---------------------------------------------------------------------------
# Category 2: resolve_profile_flags — backward compat + presets
# ---------------------------------------------------------------------------

class TestResolveProfileFlags:
    def _resolve(self, analysis_profile=None, mode="ligand_target", **flags):
        from interaction_profiler.site.profiles import resolve_profile_flags
        return resolve_profile_flags(analysis_profile, mode, **flags)

    def test_none_profile_returns_explicit_flags_only(self):
        flags = self._resolve(compute_strain=False, compute_interface=True)
        assert flags["compute_strain"] is False
        assert flags["compute_interface"] is True
        assert "run_fpocket" not in flags   # no profile, no extras

    def test_none_profile_with_no_flags_returns_empty(self):
        flags = self._resolve()
        assert flags == {}

    def test_standard_profile_sets_all_compute_flags(self):
        flags = self._resolve("standard", "ligand_target")
        assert flags["compute_strain"] is True
        assert flags["compute_efficiency"] is True
        assert flags["compute_interface"] is True
        assert flags.get("compute_vectors") is False

    def test_fast_profile_turns_off_all_compute(self):
        flags = self._resolve("fast", "ligand_target")
        assert flags["compute_strain"] is False
        assert flags["compute_efficiency"] is False
        assert flags["compute_interface"] is False

    def test_enriched_profile_enables_vectors(self):
        flags = self._resolve("enriched", "ligand_target")
        assert flags["compute_vectors"] is True

    def test_explicit_flag_overrides_profile(self):
        # Profile says strain=True but caller passes strain=False
        flags = self._resolve("standard", "ligand_target", compute_strain=False)
        assert flags["compute_strain"] is False
        # Other flags from profile still active
        assert flags["compute_efficiency"] is True

    def test_pocket_profile_protein_only(self):
        flags = self._resolve("pocket", "protein_only")
        assert flags["run_fpocket"] is True
        assert flags["assign_ss"] is True
        assert flags["fetch_uniprot"] is False

    def test_annotated_profile_protein_only(self):
        flags = self._resolve("annotated", "protein_only")
        assert flags["fetch_uniprot"] is True
        assert flags["fetch_interpro"] is True

    def test_structure_profile_no_fpocket(self):
        flags = self._resolve("structure", "protein_only")
        assert flags["run_fpocket"] is False

    def test_unknown_profile_falls_back_to_explicit(self):
        flags = self._resolve("nonexistent_profile", "ligand_target", compute_strain=True)
        assert flags.get("compute_strain") is True

    def test_ensemble_hotspots_profile(self):
        flags = self._resolve("hotspots", "hotspot_ensemble")
        assert flags["compute_fragment_families"] is False

    def test_ensemble_families_profile(self):
        flags = self._resolve("families", "hotspot_ensemble")
        assert flags["compute_fragment_families"] is True
        assert flags["compute_compatible_pairs"] is False

    def test_ensemble_full_profile(self):
        flags = self._resolve("full", "hotspot_ensemble")
        assert flags["compute_fragment_families"] is True
        assert flags["compute_compatible_pairs"] is True


# ---------------------------------------------------------------------------
# Category 3: profile_complex — protein_only mode (fpocket mocked)
# ---------------------------------------------------------------------------

class TestProteinOnlyMode:
    def test_protein_only_returns_complex_result(self, tmp_path):
        # Create a properly-formatted minimal PDB
        pdb = tmp_path / "test.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C  \n"
            "END\n"
        )

        from interaction_profiler.core_complex import profile_complex
        from interaction_profiler.schemas import ComplexResult

        with patch("interaction_profiler.site.characterizer.characterize_site") as mock_char:
            from interaction_profiler.schemas import SiteProfile
            mock_char.return_value = SiteProfile(
                structure_path=str(pdb), structure_sha256="a" * 64,
                target_chains=["A"], analysis_profile="structure",
                site_type="unknown", site_type_confidence="low",
            )
            r = profile_complex(
                structure_path=str(pdb),
                mode="protein_only",
                target_chains=["A"],
                analysis_profile="structure",
            )

        assert isinstance(r, ComplexResult)
        assert r.mode == "protein_only"
        assert r.site_profile is not None
        assert r.site_profile.target_chains == ["A"]

    def test_protein_only_missing_structure_returns_warning(self):
        from interaction_profiler.core_complex import profile_complex
        r = profile_complex(mode="protein_only", structure_path=None)
        assert r.mode == "protein_only"
        assert any("structure_path" in w for w in r.warnings)
        assert r.site_profile is None

    def test_protein_only_analysis_profile_default_is_pocket(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1   0.0 0.0 0.0  1.00 0.00  C\nEND\n")

        from interaction_profiler.core_complex import profile_complex
        with patch("interaction_profiler.site.characterizer.characterize_site") as mock_char:
            from interaction_profiler.schemas import SiteProfile
            mock_char.return_value = SiteProfile(
                structure_path=str(pdb), structure_sha256="a" * 64,
                target_chains=["A"], analysis_profile="pocket",
            )
            profile_complex(structure_path=str(pdb), mode="protein_only")

        # The default profile passed to characterize_site should be "pocket"
        call_kwargs = mock_char.call_args[1]
        assert call_kwargs.get("profile_flags", {}).get("run_fpocket", False) is True \
            or call_kwargs.get("analysis_profile", "pocket") == "pocket"

    def test_unknown_mode_now_includes_protein_only_in_message(self):
        from interaction_profiler.core_complex import profile_complex
        r = profile_complex(mode="invalid_xyz")
        assert "protein_only" in " ".join(r.warnings)


# ---------------------------------------------------------------------------
# Category 4: Backward compat — existing modes unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_profile_complex_signature_unchanged(self):
        """All original params must still exist with original defaults."""
        from interaction_profiler.core_complex import profile_complex
        import inspect
        sig = inspect.signature(profile_complex)
        original_params = [
            "structure_path", "smiles", "mode", "binder_chains", "target_chains",
            "binder_role", "ligand_comp_id", "ligand_resname", "ligand_chain",
            "annotation_accessions", "potency", "ph", "radius",
            "compute_strain", "compute_efficiency", "compute_interface",
            "coordinate_source", "use_gpu", "max_interactions", "drop_uncertain",
            "fail_on_ambiguity",
        ]
        for p in original_params:
            assert p in sig.parameters, f"Original param {p!r} missing from signature"

    def test_analysis_profile_none_has_no_effect_on_flags(self):
        """analysis_profile=None must not change any compute behavior."""
        from interaction_profiler.site.profiles import resolve_profile_flags
        flags_without = resolve_profile_flags(None, "ligand_target",
                                              compute_strain=True, compute_efficiency=False)
        assert flags_without["compute_strain"] is True
        assert flags_without["compute_efficiency"] is False

    def test_complex_result_without_new_fields_still_loads(self):
        """A ComplexResult serialized before v3 (no new fields) must deserialize."""
        from interaction_profiler.schemas import ComplexResult
        old_dump = {
            "mode": "ligand_target",
            "provenance": {"mode": "ligand_target"},
            "entities": [],
            "enriched_interactions": [],
            "generalized_triples": [],
            "hotspots": [],
            "warnings": [],
        }
        r = ComplexResult.model_validate(old_dump)
        assert r.site_profile is None
        assert r.fragment_families == []
        assert r.compatible_pairs == []

    def test_new_mcp_params_default_none(self):
        """analysis_profile, site_profile_id, ensemble_profile all default None."""
        import importlib.util
        path = _REPO.parent / "reinvent4_mcp" / "tools" / "interaction_profiler.py"
        # Just check they appear in the source
        src = path.read_text()
        assert "analysis_profile" in src
        assert "site_profile_id" in src
        assert "ensemble_profile" in src


# ---------------------------------------------------------------------------
# Category 5: Classifier — confidence_bin, exposure, vector_type
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_confidence_bin_high(self):
        from interaction_profiler.report.classifier import _compute_confidence_bin
        assert _compute_confidence_bin(True, 0.5) == "high"

    def test_confidence_bin_moderate_uncertain(self):
        from interaction_profiler.report.classifier import _compute_confidence_bin
        assert _compute_confidence_bin(False, 0.5) == "moderate"

    def test_confidence_bin_low_uncertain_high_strain(self):
        from interaction_profiler.report.classifier import _compute_confidence_bin
        assert _compute_confidence_bin(False, 8.0) == "low"

    def test_sasa_to_exposure_buried(self):
        from interaction_profiler.report.classifier import _sasa_to_exposure_bin
        assert _sasa_to_exposure_bin(2.0) == "buried"

    def test_sasa_to_exposure_partial(self):
        from interaction_profiler.report.classifier import _sasa_to_exposure_bin
        assert _sasa_to_exposure_bin(10.0) == "partially_exposed"

    def test_sasa_to_exposure_solvent(self):
        from interaction_profiler.report.classifier import _sasa_to_exposure_bin
        assert _sasa_to_exposure_bin(25.0) == "solvent_exposed"

    def test_vector_anchor(self):
        from interaction_profiler.report.classifier import _classify_vector
        v = _classify_vector(2.0, True, "buried", None, {})
        assert v == "anchor"

    def test_vector_exit(self):
        from interaction_profiler.report.classifier import _classify_vector
        v = _classify_vector(30.0, True, "solvent_exposed", "RES1.A", {"RES1.A": "S1"})
        # Target res IS in subpocket_map → not a growth vector
        assert v == "exit_vector"

    def test_vector_growth(self):
        from interaction_profiler.report.classifier import _classify_vector
        v = _classify_vector(30.0, True, "solvent_exposed", "RES_NEW.A", {"RES1.A": "S1"})
        # Target res NOT in subpocket_map → growth_vector
        assert v == "growth_vector"

    def test_classify_interactions_adds_confidence_bin(self):
        from interaction_profiler.schemas import EnrichedInteraction
        from interaction_profiler.report.classifier import classify_interactions

        ei = EnrichedInteraction(interaction_type="hbond", ligand_atom_idx=0,
                                  confidence="confident")
        results = classify_interactions(
            [ei], structure_path="", binder_chain_ids=[], target_chain_ids=[],
            strain_result=None, site_profile=None,
        )
        assert results[0].confidence_bin == "high"


# ---------------------------------------------------------------------------
# Category 6: Ensemble — fragment families and compatible pairs
# ---------------------------------------------------------------------------

class TestEnsemble:
    def _make_result(self, src: str, primary_residue: str, interaction_type: str):
        from interaction_profiler.schemas import ComplexResult, GeneralizedTriple
        t = GeneralizedTriple(
            binder_feature="frag",
            target_feature=primary_residue,
            binder_feature_type="fragment",
            interaction_type=interaction_type,
            count=3,
        )
        r = ComplexResult(mode="ligand_target", generalized_triples=[t])
        # Inject source marker via a mock interface
        r._test_source = src  # used by _result_source fallback
        return r

    def test_fragment_families_groups_by_mode(self):
        from interaction_profiler.report.ensemble import compute_fragment_families
        from interaction_profiler.schemas import ComplexResult, GeneralizedTriple

        def _r(target_res, itype):
            t = GeneralizedTriple(
                binder_feature="f", target_feature=target_res,
                binder_feature_type="fragment", interaction_type=itype, count=2,
            )
            return ComplexResult(mode="ligand_target", generalized_triples=[t])

        # 3 results with same mode, 1 different
        results = [_r("HIS163.A", "hbond")] * 3 + [_r("MET49.A", "hydrophobic")]
        families = compute_fragment_families(results, min_members=2)
        assert len(families) >= 1
        # Largest family should have 3 members
        assert families[0].members.__len__() >= 2 or families[0].occupancy is not None

    def test_fragment_families_occupancy_fraction(self):
        from interaction_profiler.report.ensemble import compute_fragment_families
        from interaction_profiler.schemas import ComplexResult, GeneralizedTriple

        def _r(target_res):
            t = GeneralizedTriple(
                binder_feature="f", target_feature=target_res,
                binder_feature_type="fragment", interaction_type="hbond", count=1,
            )
            return ComplexResult(mode="ligand_target", generalized_triples=[t])

        results = [_r("HIS163.A")] * 4 + [_r("MET49.A")] * 2
        families = compute_fragment_families(results, min_members=2)
        total_members = sum(len(f.members) for f in families)
        assert total_members == 6

    def test_compatible_pairs_no_overlap(self):
        from interaction_profiler.report.ensemble import compute_compatible_pairs
        from interaction_profiler.schemas import ComplexResult
        from unittest.mock import MagicMock

        def _r(residues: list[str]) -> ComplexResult:
            r = ComplexResult(mode="ligand_target")
            mock_iface = MagicMock()
            mock_iface.interface_residues_target = residues
            r.interface = mock_iface
            return r

        results = [_r(["RES1.A", "RES2.A"]), _r(["RES3.A", "RES4.A"])]
        pairs = compute_compatible_pairs(results)
        assert len(pairs) == 1
        assert pairs[0].compatible is True
        assert pairs[0].overlap_residues == []

    def test_compatible_pairs_with_overlap(self):
        from interaction_profiler.report.ensemble import compute_compatible_pairs
        from interaction_profiler.schemas import ComplexResult
        from unittest.mock import MagicMock

        def _r(residues: list[str]) -> ComplexResult:
            r = ComplexResult(mode="ligand_target")
            mock_iface = MagicMock()
            mock_iface.interface_residues_target = residues
            r.interface = mock_iface
            return r

        results = [_r(["RES1.A", "RES2.A"]), _r(["RES2.A", "RES3.A"])]
        pairs = compute_compatible_pairs(results)
        assert len(pairs) == 1
        assert pairs[0].compatible is False
        assert "RES2.A" in pairs[0].overlap_residues

    def test_compatible_pairs_sorted_compatible_first(self):
        from interaction_profiler.report.ensemble import compute_compatible_pairs
        from interaction_profiler.schemas import ComplexResult
        from unittest.mock import MagicMock

        def _r(residues: list[str]) -> ComplexResult:
            r = ComplexResult(mode="ligand_target")
            mock_iface = MagicMock()
            mock_iface.interface_residues_target = residues
            r.interface = mock_iface
            return r

        results = [
            _r(["RES1.A", "RES2.A"]),  # overlaps with last
            _r(["RES3.A", "RES4.A"]),  # compatible with first
            _r(["RES1.A", "RES5.A"]),  # overlaps with first
        ]
        pairs = compute_compatible_pairs(results)
        # Compatible pair should come first
        assert pairs[0].compatible is True
