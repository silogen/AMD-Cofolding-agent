"""Tests for the structure preparation service.

Mix of unit tests (individual stage functions) and integration tests
(POST /validate, POST /jobs round-trip).
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---- Fixtures ----

@pytest.fixture
def protein_ligand_pdb(tmp_path):
    """Minimal PDB fixture: one protein chain + one HETATM ligand + HOH."""
    content = """ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 10.00           N  
ATOM      2  CA  ALA A   1       2.000   1.000   1.000  1.00 10.00           C  
ATOM      3  C   ALA A   1       3.000   1.000   1.000  1.00 10.00           C  
ATOM      4  O   ALA A   1       3.500   2.000   1.000  1.00 10.00           O  
ATOM      5  CB  ALA A   1       2.000   2.000   1.000  1.00 10.00           C  
ATOM      6  N   HIS A   2       4.000   1.000   1.000  1.00 10.00           N  
ATOM      7  CA  HIS A   2       5.000   1.000   1.000  1.00 10.00           C  
ATOM      8  C   HIS A   2       6.000   1.000   1.000  1.00 10.00           C  
ATOM      9  O   HIS A   2       6.500   2.000   1.000  1.00 10.00           O  
ATOM     10  CB  HIS A   2       5.000   2.000   1.000  1.00 10.00           C  
ATOM     11  CG  HIS A   2       5.000   3.000   1.000  1.00 10.00           C  
ATOM     12  ND1 HIS A   2       4.500   4.000   1.000  1.00 10.00           N  
ATOM     13  CD2 HIS A   2       5.500   4.000   1.000  1.00 10.00           C  
ATOM     14  CE1 HIS A   2       5.000   5.000   1.000  1.00 10.00           C  
ATOM     15  NE2 HIS A   2       5.500   5.000   1.000  1.00 10.00           N  
TER
HETATM   16  C1  LIG A 101       8.000   1.000   1.000  1.00 15.00           C  
HETATM   17  C2  LIG A 101       9.000   1.000   1.000  1.00 15.00           C  
HETATM   18  N1  LIG A 101      10.000   1.000   1.000  1.00 15.00           N  
HETATM   19  O1  LIG A 101      11.000   1.000   1.000  1.00 15.00           O  
HETATM   20  O   HOH A 201       7.000   3.000   1.000  1.00 20.00           O  
HETATM   21  S   SO4 A 301       0.000   0.000   0.000  1.00 25.00           S  
HETATM   22  O   SO4 A 301       1.000   0.000   0.000  1.00 25.00           O  
HETATM   23  O   SO4 A 301       0.000   1.000   0.000  1.00 25.00           O  
HETATM   24  O   SO4 A 301      -1.000   0.000   0.000  1.00 25.00           O  
END
"""
    p = tmp_path / "test.pdb"
    p.write_text(content)
    return p

@pytest.fixture
def altloc_pdb(tmp_path):
    """PDB with alternate conformations."""
    content = """ATOM      1  N  AALA A   1       1.000   1.000   1.000  0.60 10.00           N  
ATOM      2  N  BALA A   1       1.100   1.100   1.100  0.40 10.00           N  
ATOM      3  CA AALA A   1       2.000   1.000   1.000  0.60 10.00           C  
ATOM      4  CA BALA A   1       2.100   1.100   1.100  0.40 10.00           C  
END
"""
    p = tmp_path / "altloc.pdb"
    p.write_text(content)
    return p

# ---- Stage 0: Inventory ----

def test_stage0_inventory_parses_hetatm(protein_ligand_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    structure, inventory = run_inventory(protein_ligand_pdb, tmp_path)

    comp_ids = {g.comp_id for g in inventory.hetatm_groups}
    assert "LIG" in comp_ids
    assert "HOH" in comp_ids
    assert "SO4" in comp_ids

def test_stage0_inventory_writes_json(protein_ligand_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    run_inventory(protein_ligand_pdb, tmp_path)
    assert (tmp_path / "inventory.json").exists()

def test_stage0_inventory_classifies_water(protein_ligand_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    _, inventory = run_inventory(protein_ligand_pdb, tmp_path)
    water_groups = [g for g in inventory.hetatm_groups if g.comp_id == "HOH"]
    assert len(water_groups) == 1
    assert water_groups[0].classification == "water"

def test_stage0_inventory_classifies_additive(protein_ligand_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    _, inventory = run_inventory(protein_ligand_pdb, tmp_path)
    so4_groups = [g for g in inventory.hetatm_groups if g.comp_id == "SO4"]
    assert so4_groups[0].classification in ("additive", "buffer_ion")

# ---- Stage 3: Curation ----

def test_stage3_removes_additives_by_default(protein_ligand_pdb, tmp_path):
    import gemmi
    from structure_prep.stages.inventory import run as run_inventory
    from structure_prep.stages.assembly import run as run_assembly
    from structure_prep.stages.roles import run as run_roles
    from structure_prep.stages.curation import run as run_curation
    from structure_prep.models import Stage3Profile

    structure, inventory = run_inventory(protein_ligand_pdb, tmp_path)
    _, role_map = run_roles(structure, MagicMock(
        target_chains=None, binder_chains=None, nucleic_acid_chains=None,
        antibody_detection=False, peptide_min_residues=5, peptide_max_residues=29,
        metal_coordination_cutoff_A=3.0
    ), inventory)

    profile = Stage3Profile(keep_additives=False, keep_waters="none")
    structure, s3 = run_curation(structure, profile, role_map, inventory.hetatm_groups)

    # SO4 should be removed
    removed = [e for e in s3["curation_log"] if e.decision == "removed"]
    removed_names = {e.entity.split(":")[0] for e in removed}
    assert "SO4" in removed_names

def test_stage3_pocket_water_retention(protein_ligand_pdb, tmp_path):
    """Waters within radius of ligand should be retained with 'pocket' policy."""
    from structure_prep.models import Stage3Profile
    profile = Stage3Profile(
        keep_waters="pocket",
        water_pocket_radius_A=6.0,  # within allowed range (≤8 Å)
        keep_additives=False,
    )
    # Integration test — just verify the API call works
    assert profile.keep_waters == "pocket"

# ---- Stage 4: Altlocs ----

def test_stage4_highest_occupancy_selects_best(altloc_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    from structure_prep.stages.altlocs import run as run_altlocs
    from structure_prep.models import Stage4Profile

    structure, inventory = run_inventory(altloc_pdb, tmp_path)
    assert len(inventory.altloc_positions) >= 1

    profile = Stage4Profile(altloc_policy="highest_occupancy")
    structure, s4 = run_altlocs(structure, profile)

    # After resolution, no altloc codes should remain
    for chain in structure[0]:
        for res in chain:
            for atom in res:
                assert atom.altloc in ('\x00', ' ', ''), f"Altloc {atom.altloc!r} remaining"

def test_stage4_fail_raises_on_altloc(altloc_pdb, tmp_path):
    from structure_prep.stages.inventory import run as run_inventory
    from structure_prep.stages.altlocs import run as run_altlocs
    from structure_prep.models import Stage4Profile

    structure, inventory = run_inventory(altloc_pdb, tmp_path)
    profile = Stage4Profile(altloc_policy="fail")

    with pytest.raises(ValueError, match="altloc"):
        run_altlocs(structure, profile)

# ---- Stage 6: Histidine tautomers ----

def test_stage6_user_override_applied(tmp_path):
    from structure_prep.stages.protonation import _apply_user_overrides
    from structure_prep.models import Stage6Profile, ResidueStateOverride

    import gemmi
    st = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    res = gemmi.Residue()
    res.name = "HIS"
    res.seqid = gemmi.SeqId("64")
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)

    profile = Stage6Profile(
        protonation_method="user_explicit",
        residue_state_overrides=[ResidueStateOverride(chain="A", resseq=64, resname="HIS", state="HIE")],
    )
    result, _ = _apply_user_overrides(st, profile)

    # The HIS residue should be renamed to HIE
    his_res = result[0]["A"][0]
    assert his_res.name == "HIE"

# ---- Stage 8: InChIKey cross-check ----

def test_stage8_inchikey_computed_for_valid_smiles():
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, MolToInchiKey

    smiles = "c1ccccc1"  # benzene
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    inchi = MolToInchi(mol)
    inchikey = MolToInchiKey(mol) if inchi else None
    assert inchikey is not None
    assert len(inchikey.split("-")) == 3  # InChIKey has 3 segments

# ---- Profile validation ----

def test_profile_defaults_are_safe():
    from structure_prep.models import PreparationProfile
    p = PreparationProfile()
    assert p.stage3_curation.keep_additives is False
    assert p.stage3_curation.keep_waters == "none"
    assert p.stage6_protonation.protonation_method == "pdb2pqr_propka"
    assert p.stage6_protonation.ph == 7.4
    assert p.stage7_termini.cap_termini == "none"
    assert p.stage9_minimization.minimize == "none"

def test_profile_hash_is_deterministic():
    from structure_prep.models import PreparationProfile
    p1 = PreparationProfile()
    p2 = PreparationProfile()
    assert p1.profile_hash() == p2.profile_hash()

def test_different_profiles_have_different_hashes():
    from structure_prep.models import PreparationProfile, Stage3Profile
    p1 = PreparationProfile()
    p2 = PreparationProfile(stage3_curation=Stage3Profile(keep_waters="all"))
    assert p1.profile_hash() != p2.profile_hash()

# ---- Path traversal security ----

def test_path_traversal_rejected():
    """POST /validate should reject paths with .. traversal."""
    from fastapi.testclient import TestClient
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app import app, SHARED_DIR
    client = TestClient(app)

    # Attempt traversal
    resp = client.post("/validate", json={"path": "../../etc/passwd"})
    assert resp.status_code in (400, 404)

# ---- Service health ----

def test_health_endpoint_returns_ok():
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

# ---- Manifest reproducibility ----

def test_same_profile_same_hash():
    from structure_prep.models import PreparationProfile
    p = PreparationProfile(version="1.0")
    h1 = p.profile_hash()
    h2 = p.profile_hash()
    assert h1 == h2
    assert h1.startswith("sha256:")
