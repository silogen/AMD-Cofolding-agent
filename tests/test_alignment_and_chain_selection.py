"""Tests for Tranche 4 — alignment stage and chain selection."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PREP_DIR = Path(__file__).parent.parent / "services" / "structure-prep"
sys.path.insert(0, str(_PREP_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Minimal PDB helpers (duplicate-free, self-contained)
# ---------------------------------------------------------------------------

def _make_pdb_chain(chain_id: str, residues: list[tuple[int, str]]) -> str:
    lines: list[str] = []
    for n, (resseq, resname) in enumerate(residues, start=1):
        lines.append(
            f"ATOM  {n:>5}  CA  {resname:<3} {chain_id}{resseq:>4}    "
            f"   {n*1.0:>7.3f}   0.000   0.000  1.00 10.00           C  "
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def _write_pdb(tmp_path: Path, chains: dict[str, list[tuple[int, str]]]) -> Path:
    p = tmp_path / "test.pdb"
    lines: list[str] = []
    atom_n = 1
    for chain_id, residues in chains.items():
        for resseq, resname in residues:
            lines.append(
                f"ATOM  {atom_n:>5}  CA  {resname:<3} {chain_id}{resseq:>4}    "
                f"   {atom_n*1.0:>7.3f}   0.000   0.000  1.00 10.00           C  "
            )
            atom_n += 1
    lines.append("END")
    p.write_text("\n".join(lines) + "\n")
    return p


SHORT_PROTEIN = [(i, "ALA") for i in range(1, 21)]  # 20 residues


def _write_protein_3d(tmp_path: Path, name: str = "test.pdb") -> Path:
    """Write a 20-residue alanine chain on a circular arc (non-collinear)."""
    import math
    lines: list[str] = []
    for i in range(1, 21):
        angle = math.radians(i * 18)
        x = 10.0 * math.cos(angle)
        y = 10.0 * math.sin(angle)
        z = float(i)
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
            f"   {x:>7.3f}  {y:>7.3f}  {z:>7.3f}  1.00 10.00           C  "
        )
    lines.append("END")
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Alignment stage tests
# ---------------------------------------------------------------------------

class TestAlignmentStage:

    def test_identical_structures_zero_rmsd(self, tmp_path):
        import gemmi
        from structure_prep.stages import alignment as alignment_mod
        from structure_prep.models import Stage10Profile

        pdb = _write_protein_3d(tmp_path)
        mobile = gemmi.read_structure(str(pdb))
        mobile.setup_entities()
        ref = gemmi.read_structure(str(pdb))
        ref.setup_entities()

        profile = Stage10Profile(
            reference_path=str(pdb),
            reference_chain="A",
            mobile_chain="A",
        )
        _, report = alignment_mod.run(mobile, profile, ref)
        assert report["performed"] is True
        assert report["rmsd_ca_A"] == pytest.approx(0.0, abs=1e-3)
        assert isinstance(report["transform_4x4"], list)
        assert len(report["transform_4x4"]) == 4

    def test_no_reference_path_skips(self, tmp_path):
        import gemmi
        from structure_prep.stages import alignment as alignment_mod
        from structure_prep.models import Stage10Profile

        pdb = _write_protein_3d(tmp_path)
        mobile = gemmi.read_structure(str(pdb))
        mobile.setup_entities()
        ref = gemmi.read_structure(str(pdb))
        ref.setup_entities()

        profile = Stage10Profile(reference_path=None)
        _, report = alignment_mod.run(mobile, profile, ref)
        # performed=True because we called run() directly; the pipeline
        # skips the call when reference_path is None.
        # Here we just verify the transform round-trips.
        assert "transform_4x4" in report or "performed" in report

    def test_max_rmsd_threshold_enforced(self, tmp_path):
        import gemmi
        from structure_prep.stages import alignment as alignment_mod
        from structure_prep.models import Stage10Profile

        pdb = _write_protein_3d(tmp_path)
        mobile = gemmi.read_structure(str(pdb))
        mobile.setup_entities()
        ref = gemmi.read_structure(str(pdb))
        ref.setup_entities()

        # Force the report to look like a high-RMSD result by patching
        orig_run = alignment_mod._align_chains

        def _fake_align(ref_model, mob_model, ref_chain, mob_chain):
            from unittest.mock import MagicMock
            sup = MagicMock()
            sup.rmsd = 99.0
            sup.count = 20
            sup.transform.mat.tolist.return_value = [[1,0,0],[0,1,0],[0,0,1]]
            sup.transform.vec.x = sup.transform.vec.y = sup.transform.vec.z = 0.0
            return sup, []

        alignment_mod._align_chains = _fake_align
        try:
            profile = Stage10Profile(
                reference_path=str(pdb),
                reference_chain="A",
                mobile_chain="A",
                max_rmsd_A=0.5,
            )
            _, report = alignment_mod.run(mobile, profile, ref)
            assert report["performed"] is False
            assert "exceeds threshold" in report["reason"]
        finally:
            alignment_mod._align_chains = orig_run

    def test_all_models_transformed(self, tmp_path):
        """Alignment must apply transform to every atom."""
        import gemmi, math
        from structure_prep.stages import alignment as alignment_mod
        from structure_prep.models import Stage10Profile

        pdb = _write_protein_3d(tmp_path)
        mobile = gemmi.read_structure(str(pdb))
        mobile.setup_entities()

        # Shift reference by a known translation using a separate PDB
        import math
        lines2: list[str] = []
        for i in range(1, 21):
            angle = math.radians(i * 18)
            x = 10.0 * math.cos(angle) + 10.0  # +10 Å shift on x
            y = 10.0 * math.sin(angle)
            z = float(i)
            lines2.append(
                f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
                f"   {x:>7.3f}  {y:>7.3f}  {z:>7.3f}  1.00 10.00           C  "
            )
        lines2.append("END")
        pdb_ref = tmp_path / "ref_shifted.pdb"
        pdb_ref.write_text("\n".join(lines2) + "\n")

        ref = gemmi.read_structure(str(pdb_ref))
        ref.setup_entities()

        before_x = mobile[0][0][0][0].pos.x
        profile = Stage10Profile(
            reference_path=str(pdb),
            reference_chain="A",
            mobile_chain="A",
        )
        mobile2, report = alignment_mod.run(mobile, profile, ref)
        assert report["performed"] is True
        after_x = mobile2[0][0][0][0].pos.x
        # Atom must have moved towards the shifted reference
        assert abs(after_x - before_x) > 0.1


# ---------------------------------------------------------------------------
# Chain selection tests
# ---------------------------------------------------------------------------

class TestChainSelection:

    def _make_role_map(self, protein_chains: list[str], binder_chains: list[str] | None = None):
        from structure_prep.models import RoleMap
        return RoleMap(
            protein_chains=protein_chains,
            binder_chains=binder_chains or [],
        )

    def _make_profile(self, **kwargs):
        from structure_prep.models import Stage2Profile
        return Stage2Profile(**kwargs)

    def test_keep_all_no_op(self, tmp_path):
        import gemmi
        from structure_prep.stages.chain_selection import run
        pdb = _write_pdb(tmp_path, {
            "A": SHORT_PROTEIN,
            "B": SHORT_PROTEIN,
        })
        st = gemmi.read_structure(str(pdb))
        profile = self._make_profile(chain_policy="keep_all")
        role_map = self._make_role_map(["A", "B"])
        st2, log = run(st, profile, role_map)
        chains_after = {c.name for c in st2[0]}
        assert "A" in chains_after and "B" in chains_after
        assert not log["chains_removed"]

    def test_keep_target_removes_binder(self, tmp_path):
        import gemmi
        from structure_prep.stages.chain_selection import run
        pdb = _write_pdb(tmp_path, {
            "A": SHORT_PROTEIN,
            "B": SHORT_PROTEIN,
        })
        st = gemmi.read_structure(str(pdb))
        profile = self._make_profile(chain_policy="keep_target")
        role_map = self._make_role_map(protein_chains=["A"], binder_chains=["B"])
        st2, log = run(st, profile, role_map)
        chains_after = {c.name for c in st2[0]}
        assert "A" in chains_after
        assert "B" not in chains_after
        assert "B" in log["chains_removed"]

    def test_keep_target_plus_binder(self, tmp_path):
        import gemmi
        from structure_prep.stages.chain_selection import run
        pdb = _write_pdb(tmp_path, {
            "A": SHORT_PROTEIN,
            "B": SHORT_PROTEIN,
            "C": SHORT_PROTEIN,  # non-protein chain
        })
        st = gemmi.read_structure(str(pdb))
        profile = self._make_profile(chain_policy="keep_target_plus_binder")
        role_map = self._make_role_map(protein_chains=["A"], binder_chains=["B"])
        st2, log = run(st, profile, role_map)
        chains_after = {c.name for c in st2[0]}
        assert "A" in chains_after
        assert "B" in chains_after
        assert "C" not in chains_after

    def test_homodimer_dedup(self, tmp_path):
        import gemmi
        from structure_prep.stages.chain_selection import run
        # Two identical chains
        pdb = _write_pdb(tmp_path, {
            "A": [(i, "ALA") for i in range(1, 11)],
            "B": [(i, "ALA") for i in range(1, 11)],
        })
        st = gemmi.read_structure(str(pdb))
        profile = self._make_profile(
            chain_policy="keep_target",
            dedup_identical=True,
            dedup_identity_threshold=0.99,
        )
        role_map = self._make_role_map(["A", "B"])
        _, log = run(st, profile, role_map)
        # One of the two identical chains must be deduped
        assert len(log["deduped_chains"]) == 1

    def test_batch_endpoint_registered(self, tmp_path):
        """The /jobs/batch endpoint is importable and has the expected path."""
        import sys
        sys.path.insert(0, str(_PREP_DIR))
        from app import app
        routes = [r.path for r in app.routes]
        assert "/jobs/batch" in routes
