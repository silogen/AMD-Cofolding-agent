"""Tests for Tranche 3 — residue numbering and construct detection.

All tests are unit tests: no running service, no network, no Docker.
PDB fixtures are minimal synthetic strings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make structure-prep importable from tests
_PREP_DIR = Path(__file__).parent.parent / "services" / "structure-prep"
sys.path.insert(0, str(_PREP_DIR))

# Make reinvent4_mcp importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Minimal PDB fixture helpers
# ---------------------------------------------------------------------------

def _pdb_chain(chain_id: str, residues: list[tuple[int, str, str]]) -> str:
    """Build a minimal PDB with one ATOM per residue (CA only)."""
    lines: list[str] = []
    atom_n = 1
    for resseq, resname, _ in residues:
        lines.append(
            f"ATOM  {atom_n:>5}  CA  {resname:<3} {chain_id}{resseq:>4}    "
            f"   1.000   1.000   1.000  1.00 10.00           C  "
        )
        atom_n += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write_pdb(tmp_path: Path, chain_id: str, residues: list[tuple[int, str, str]]) -> Path:
    p = tmp_path / "test.pdb"
    p.write_text(_pdb_chain(chain_id, residues))
    return p


# Mpro-like: chain A starting at author -2 (immature form)
MPRO_IMMATURE_RESIDUES = [
    (-2, "MET", "M"), (-1, "ALA", "A"), (0, "SER", "S"),
    (1, "GLY", "G"), (2, "PHE", "F"),
    (41, "HIS", "H"),   # catalytic His
    (145, "CYS", "C"),  # catalytic Cys
    (163, "HIS", "H"),
    (166, "GLU", "E"),
]

# Mature form: starts at 1
MPRO_MATURE_RESIDUES = [
    (1, "SER", "S"), (2, "GLY", "G"), (3, "PHE", "F"),
    (41, "HIS", "H"),
    (145, "CYS", "C"),
    (163, "HIS", "H"),
    (166, "GLU", "E"),
]


# ---------------------------------------------------------------------------
# build_residue_numbering
# ---------------------------------------------------------------------------

class TestBuildResidueNumbering:

    def test_mature_form_sequence_positions_match_author(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES[:3])
        st = gemmi.read_structure(str(pdb))
        entries, seqs = build_residue_numbering(st)
        # For a structure starting at 1, sequence position == author resseq
        assert entries[0].sequence_position == 1
        assert entries[0].author_resseq == 1
        assert entries[0].resname == "SER"

    def test_immature_form_offset(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_IMMATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        entries, _ = build_residue_numbering(st)
        # sequence_position 1 corresponds to author -2
        first = entries[0]
        assert first.author_resseq == -2
        assert first.sequence_position == 1
        # His41 is at sequence_position 7 (positions -2,-1,0,1,2,41,145... -> index 6+1=7)
        his41 = next(e for e in entries if e.author_resseq == 41)
        assert his41.sequence_position == 6  # 1-based: residues at -2,-1,0,1,2 = 5, then 41 is 6

    def test_cys145_resolves_correctly_in_immature(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering, author_to_sequence_position
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_IMMATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        entries, _ = build_residue_numbering(st)
        seq_pos = author_to_sequence_position(entries, "A", 145, expected_resname="CYS")
        assert isinstance(seq_pos, int)
        assert seq_pos > 1  # must not equal 145 in the immature form

    def test_wrong_resname_raises(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering, author_to_sequence_position
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        entries, _ = build_residue_numbering(st)
        with pytest.raises(ValueError, match="Residue identity mismatch"):
            author_to_sequence_position(entries, "A", 145, expected_resname="SER")  # wrong!

    def test_absent_residue_raises(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering, author_to_sequence_position
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES[:2])
        st = gemmi.read_structure(str(pdb))
        entries, _ = build_residue_numbering(st)
        with pytest.raises(ValueError, match="absent from the numbering map"):
            author_to_sequence_position(entries, "A", 999)

    def test_uniprot_offsets_applied(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        # Mpro in P0DTD1: first residue 1 in chain -> 3264 in UniProt polyprotein
        entries, _ = build_residue_numbering(st, uniprot_offsets={"A": 3263})
        first = next(e for e in entries if e.author_resseq == 1)
        assert first.uniprot_position == 3264

    def test_sequences_returned(self, tmp_path):
        from structure_prep.numbering import build_residue_numbering
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES[:3])
        st = gemmi.read_structure(str(pdb))
        _, seqs = build_residue_numbering(st)
        assert "A" in seqs
        assert len(seqs["A"]) == 3


# ---------------------------------------------------------------------------
# assess_construct / apply_construct_policy
# ---------------------------------------------------------------------------

class TestConstructAssessment:

    def test_mature_form_detected(self, tmp_path):
        from structure_prep.construct import assess_construct
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES[:5])
        st = gemmi.read_structure(str(pdb))
        assessments = assess_construct(st)
        assert len(assessments) == 1
        a = assessments[0]
        assert a.form == "mature"
        assert not a.non_native_n_terminal
        assert a.author_start == 1

    def test_immature_form_detected(self, tmp_path):
        from structure_prep.construct import assess_construct
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_IMMATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        assessments = assess_construct(st)
        assert len(assessments) == 1
        a = assessments[0]
        assert a.form == "immature"
        assert a.non_native_n_terminal
        assert a.author_start == -2

    def test_trim_none_leaves_structure_unchanged(self, tmp_path):
        from structure_prep.construct import apply_construct_policy
        from structure_prep.models import ConstructProfile
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_IMMATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        original_count = sum(1 for c in st[0] for r in c)
        profile = ConstructProfile(trim_policy="none")
        st2, _ = apply_construct_policy(st, profile)
        after_count = sum(1 for c in st2[0] for r in c)
        assert after_count == original_count

    def test_trim_native_removes_nonpositive(self, tmp_path):
        from structure_prep.construct import apply_construct_policy
        from structure_prep.models import ConstructProfile
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_IMMATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        profile = ConstructProfile(trim_policy="native")
        st2, assessments = apply_construct_policy(st, profile)
        # Residues at -2, -1, 0 should be gone; 1, 2, 41, 145, 163, 166 remain
        remaining = [r.seqid.num for c in st2[0] for r in c]
        assert all(n > 0 for n in remaining)
        assert 1 in remaining
        assert -2 not in remaining
        # trimmed_residues recorded
        trimmed = [a.trimmed_residues for a in assessments if a.trimmed_residues]
        assert trimmed  # at least one chain recorded trims

    def test_trim_native_does_not_touch_mature(self, tmp_path):
        from structure_prep.construct import apply_construct_policy
        from structure_prep.models import ConstructProfile
        import gemmi
        pdb = _write_pdb(tmp_path, "A", MPRO_MATURE_RESIDUES)
        st = gemmi.read_structure(str(pdb))
        original_count = sum(1 for c in st[0] for r in c)
        profile = ConstructProfile(trim_policy="native")
        st2, assessments = apply_construct_policy(st, profile)
        after_count = sum(1 for c in st2[0] for r in c)
        assert after_count == original_count
        assert not any(a.trimmed_residues for a in assessments)


# ---------------------------------------------------------------------------
# MCP resolve_author_residue helper
# ---------------------------------------------------------------------------

class TestMCPResolveAuthorResidue:

    def _make_entries(self, residues: list[tuple[int, str, str]]) -> list[dict]:
        """Build the same shape as residue_numbering.json entries."""
        return [
            {
                "chain": "A",
                "sequence_position": pos,
                "author_resseq": resseq,
                "author_icode": "",
                "resname": resname,
                "one_letter": letter,
                "uniprot_position": None,
            }
            for pos, (resseq, resname, letter)
            in enumerate(residues, start=1)
        ]

    def test_exact_match(self):
        from reinvent4_mcp.tools._numbering import resolve_author_resseq
        entries = self._make_entries(MPRO_MATURE_RESIDUES)
        result = resolve_author_resseq(entries, "A", 145, expected_resname="CYS")
        assert result["resname"] == "CYS"
        assert result["sequence_position"] == entries[4]["sequence_position"]  # 5th residue

    def test_identity_mismatch_raises(self):
        from reinvent4_mcp.tools._numbering import resolve_author_resseq
        entries = self._make_entries(MPRO_MATURE_RESIDUES)
        with pytest.raises(ValueError, match="identity mismatch"):
            resolve_author_resseq(entries, "A", 145, expected_resname="SER")

    def test_missing_residue_raises(self):
        from reinvent4_mcp.tools._numbering import resolve_author_resseq
        entries = self._make_entries(MPRO_MATURE_RESIDUES[:2])
        with pytest.raises(ValueError, match="absent"):
            resolve_author_resseq(entries, "A", 999)

    def test_immature_his41_not_at_position_41(self):
        from reinvent4_mcp.tools._numbering import resolve_author_resseq
        entries = self._make_entries(MPRO_IMMATURE_RESIDUES)
        result = resolve_author_resseq(entries, "A", 41, expected_resname="HIS")
        # In the immature form starting at -2, His41 is sequence position 7 (not 41)
        assert result["sequence_position"] != 41
        assert result["resname"] == "HIS"
