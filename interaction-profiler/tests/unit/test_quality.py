"""Unit tests for the quality improvements (cap, confidence, vocab, blacklist)."""

from __future__ import annotations

from interaction_profiler.core import _effective_cap
from interaction_profiler.interactions import rules as R
from interaction_profiler.io.structure import COFACTORS
from interaction_profiler.schemas import Interaction


def test_effective_cap_adaptive_and_explicit():
    # adaptive: ~1.2x heavy atoms, clamped to [20, 60]
    assert _effective_cap(None, 5) == 20      # floor
    assert _effective_cap(None, 100) == 60    # ceiling
    assert _effective_cap(None, 40) == 48     # 1.2 * 40
    # explicit overrides; 0 = no cap
    assert _effective_cap(30, 100) == 30
    assert _effective_cap(0, 100) == 0


def test_confidence_bands_hbond_confident_at_130deg():
    # a 3.3 A / 130 deg H-bond should be confident, not uncertain
    verdict = R.combine(R.HBOND_DIST.classify_max(3.3), R.HBOND_ANGLE.classify_min(130.0))
    assert verdict == "confident"
    # borderline geometry -> uncertain, still detected
    verdict2 = R.combine(R.HBOND_DIST.classify_max(3.9), R.HBOND_ANGLE.classify_min(100.0))
    assert verdict2 == "uncertain"
    # out of range -> dropped
    assert R.combine(R.HBOND_DIST.classify_max(4.5), R.HBOND_ANGLE.classify_min(130.0)) is None


def test_hydrophobic_band_widened():
    assert R.HYDROPHOBIC_DIST.classify_max(4.4) == "core"
    assert R.HYDROPHOBIC_DIST.classify_max(4.8) == "ext"
    assert R.HYDROPHOBIC_DIST.classify_max(5.5) is None


def test_cofactor_blacklist():
    for rn in ("GDP", "ATP", "NAD", "FAD", "HEM", "CLR", "IHP"):
        assert rn in COFACTORS


def test_ligand_role_field():
    it = Interaction(interaction_type="pi_cation", ligand_role="pi", ligand_atom_idx=0,
                     receptor_chain="A", receptor_resname="PHE", receptor_resnum=1)
    assert it.ligand_role == "pi"
    assert it.subtype is None


def test_max_water_bridges_constant():
    assert R.MAX_WATER_BRIDGES == 3


def test_confident_only_is_default():
    from interaction_profiler.schemas import BatchProvenance

    assert BatchProvenance().drop_uncertain is True


def test_neutralize_implausible_charges():
    from interaction_profiler.prep.ligand import protonate_smiles_at_ph

    # benzimidazole NH should not become [n-]
    assert "[n-]" not in (protonate_smiles_at_ph("c1ccc2[nH]cnc2c1", 7.4) or "")
    # tetrazole is genuinely acidic -> keep anionic
    assert "[n-]" in (protonate_smiles_at_ph("c1nnn[nH]1", 7.4) or "")


def test_struct_conn_covalent(tmp_path):
    from interaction_profiler.io.structure import parse_covalent_links

    cif = tmp_path / "x.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_struct_conn.conn_type_id\n"
        "_struct_conn.ptnr1_label_comp_id\n"
        "_struct_conn.ptnr1_auth_asym_id\n"
        "_struct_conn.ptnr1_auth_seq_id\n"
        "_struct_conn.ptnr2_label_comp_id\n"
        "_struct_conn.ptnr2_auth_asym_id\n"
        "_struct_conn.ptnr2_auth_seq_id\n"
        "covale LIG A 501 CYS A 145\n"
        "disulf CYS A 10 CYS A 20\n"
    )
    links = parse_covalent_links(cif, "LIG")
    assert ("CYS", 145, "A") in links
    assert ("CYS", 10, "A") not in links  # disulfide, not to ligand


def test_triple_rows_flatten():
    from interaction_profiler.schemas import (
        BatchResult,
        MoleculeMetadata,
        MoleculeResult,
        Triple,
    )

    mol = MoleculeResult(
        metadata=MoleculeMetadata(name="X", receptor="protein"),
        triples=[Triple(fragment="urea", residue_label="ILE50.A", residue_name="ILE",
                        residue_number=50, chain="A", interaction_type="hbond",
                        subtype="hbond_acceptor", count=2, distance_min=3.1, distance_max=3.4,
                        n_confident=2)],
    )
    rows = BatchResult(batch_id="b", n_molecules=1, molecules=[mol]).triple_rows()
    assert len(rows) == 1 and rows[0]["fragment"] == "urea" and rows[0]["residue"] == "ILE50.A"
    assert rows[0]["count"] == 2


def test_render_report_produces_text():
    from interaction_profiler.cli import render_report
    from interaction_profiler.schemas import (
        BatchResult,
        Composition,
        Interaction,
        MoleculeMetadata,
        MoleculeResult,
        Triple,
    )

    mol = MoleculeResult(
        metadata=MoleculeMetadata(name="X", receptor="protein", chains=["A"],
                                  chain_types={"A": "protein"}, has_structure=True,
                                  n_interactions=1, n_triples=1, n_confident=1),
        composition=Composition(),
        interactions=[Interaction(interaction_type="hbond", ligand_atom_idx=1,
                                  receptor_chain="A", receptor_resname="ASP", receptor_resnum=25)],
        triples=[Triple(fragment="hydroxyl", residue_label="ASP25.A", residue_name="ASP",
                        residue_number=25, chain="A", interaction_type="hbond",
                        distance_min=3.0, distance_max=3.0, n_confident=1)],
    )
    text = render_report(BatchResult(batch_id="b", n_molecules=1, molecules=[mol]))
    assert "target=X" in text and "ASP25.A" in text and "hbond" in text
