"""Structure preparation Pydantic models.

PreparationProfile    — agent-controlled input (nested by pipeline stage)
PreparationReport     — full audit output (one section per stage)
InventoryReport       — Stage 0 result: complete catalog, no mutation
RoleMap               — Stage 2 result: entity role classification
LigandParamResult     — Stage 8 result: per-ligand parameterization
JobRecord             — job lifecycle tracking (queued/running/completed/failed)
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Profile sub-models (one per stage)
# ---------------------------------------------------------------------------

class Stage1Profile(BaseModel):
    model_config = {"extra": "forbid"}

    assembly_id: str | None = Field(
        default=None,
        description="null=asymmetric unit; '1'=bio assembly 1, '2'=bio assembly 2, ...",
    )
    model_selection: str = Field(
        default="1",
        description="NMR conformer: '1','2',... or 'all'",
    )
    expand_ncs: bool = Field(
        default=False,
        description="Apply NCS operators from REMARK 350",
    )


class Stage2Profile(BaseModel):
    model_config = {"extra": "forbid"}

    target_chains: list[str] | None = Field(default=None, description="null=auto-detect")
    binder_chains: list[str] | None = Field(default=None, description="antibody Fab, nanobody, peptide")
    nucleic_acid_chains: list[str] | None = Field(default=None)
    antibody_detection: bool = Field(default=True, description="Chothia CDR pattern")
    peptide_min_residues: int = Field(default=5, ge=3, le=20)
    peptide_max_residues: int = Field(default=29, ge=10, le=60)
    metal_coordination_cutoff_A: float = Field(default=3.0, ge=2.5, le=4.0)

    # Chain-selection policy (C-items in the PR4 backlog)
    chain_policy: Literal[
        "keep_all",
        "keep_target",
        "keep_target_plus_binder",
        "keep_listed",
    ] = Field(default="keep_all", description="Which chains to retain after role classification.")
    keep_chains: list[str] = Field(
        default_factory=list,
        description="Explicit chain IDs to retain (used when chain_policy='keep_listed').",
    )
    dedup_identical: bool = Field(
        default=False,
        description=(
            "Collapse chains with >= dedup_identity_threshold sequence identity. "
            "The chain with the ligand (or lowest alphabetical ID) is kept."
        ),
    )
    dedup_identity_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    interface_policy: Literal["drop", "keep_chain", "keep_within_A"] = Field(
        default="drop",
        description=(
            "How to handle chains outside the keep-set that contact the selected chains. "
            "'keep_within_A' retains any chain with a heavy atom within interface_radius_A "
            "of the selected chains."
        ),
    )
    keep_partner_chain: str | None = Field(
        default=None,
        description="Always retain this chain (used when interface_policy='keep_chain').",
    )
    interface_radius_A: float = Field(
        default=5.0,
        ge=2.0,
        le=15.0,
        description="Radius for keep_within_A interface policy.",
    )


class ConstructProfile(BaseModel):
    """Construct detection and optional non-native terminus trimming."""

    model_config = {"extra": "forbid"}

    detect_non_native_termini: bool = True
    trim_policy: Literal["none", "native", "user_range"] = "none"
    user_ranges: dict[str, tuple[int, int]] = Field(
        default_factory=dict,
        description="Author residue range to retain per chain, e.g. {'A': (1, 306)}.",
    )
    uniprot_offsets: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-chain additive offset from author numbering to UniProt "
            "numbering; Mpro in P0DTD1 uses 3263."
        ),
    )
    uniprot_accession: str | None = None
    uniprot_sequence: str | None = Field(
        default=None,
        description=(
            "Optional canonical sequence. Exact chain subsequences are mapped "
            "to UniProt positions automatically."
        ),
    )


class Stage3Profile(BaseModel):
    model_config = {"extra": "forbid"}

    ligand_policy: Literal["all", "none", "primary", "listed"] = "all"
    ligand_comp_ids: list[str] = Field(default_factory=list)
    ligand_exclude_comp_ids: list[str] = Field(default_factory=list)
    ligand_chain: str | None = None
    ligand_instance_index: int | None = Field(default=None, ge=0)

    keep_cofactors: bool | list[str] = True
    keep_metals: Literal["structural", "all", "none"] | list[str] = "structural"

    keep_waters: Literal["none", "pocket", "structural", "all"] = "none"
    water_pocket_radius_A: float = Field(default=4.0, ge=2.0, le=8.0)
    water_structural_bfactor_cutoff: float = Field(default=25.0, ge=5.0, le=50.0)

    keep_glycans: bool = True
    glycan_detection_mode: Literal["link_record", "sugar_resname", "both"] = "link_record"

    keep_additives: bool | list[str] = False
    additive_registry_path: str | None = None


class Stage4Profile(BaseModel):
    model_config = {"extra": "forbid"}

    altloc_policy: Literal["highest_occupancy", "selected_conformer", "retain_all", "fail"] = "highest_occupancy"
    altloc_selected: dict[str, str] = Field(default_factory=dict, description="{'CHAIN:RESSEQ': 'A'}")
    binding_site_altloc_radius_A: float = Field(default=5.0, ge=3.0, le=10.0)


class Stage5Profile(BaseModel):
    model_config = {"extra": "forbid"}

    missing_policy: Literal["report_only", "model_atoms", "model_loops", "fail"] = "model_atoms"
    missing_atom_elements: Literal["all"] | list[str] = "all"
    loop_max_length: int = Field(default=5, ge=1, le=20)
    nonstandard_residue_policy: Literal["report_and_skip", "convert_to_standard", "fail"] = "report_and_skip"
    nonstandard_conversions: dict[str, str] = Field(
        default_factory=lambda: {"MSE": "MET", "CSE": "CYS"},
    )


class ResidueStateOverride(BaseModel):
    model_config = {"extra": "forbid"}

    chain: str
    resseq: int
    resname: str
    state: Literal["HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM", "ARN"]
    reason: str = "user_override"


class Stage6Profile(BaseModel):
    model_config = {"extra": "forbid"}

    protonation_method: Literal["pdb2pqr_propka", "pdbfixer_basic", "user_explicit", "none"] = "pdb2pqr_propka"
    ph: float = Field(default=7.4, ge=0.0, le=14.0)
    propka_titration_window: float = Field(default=1.0, ge=0.5, le=3.0)
    his_default_tautomer: Literal["HID", "HIE"] = "HIE"
    metal_coordination_override: bool = True
    metal_coordination_cutoff_A: float = Field(default=2.8, ge=2.0, le=3.5)
    cysteine_disulfide_detection: bool = True
    force_protonate_nucleic_acids: bool = False
    residue_state_overrides: list[ResidueStateOverride] = Field(default_factory=list)


class Stage7Profile(BaseModel):
    model_config = {"extra": "forbid"}

    cap_termini: Literal["none", "ace_nme"] = "none"
    cap_internal_breaks: bool = True
    cap_binder_chains: bool = True
    cap_chains: list[str] | None = None


class Stage8Profile(BaseModel):
    model_config = {"extra": "forbid"}

    ligand_ff: Literal["openff_2.1", "mmff94", "none"] = "openff_2.1"
    ligand_charge_method: Literal["nagl", "mmff94", "gasteiger", "formal_only"] = "nagl"
    ligand_smiles_override: dict[str, str] = Field(default_factory=dict)
    strict_ligand_match: bool = False
    emit_sdf: bool = True
    emit_mol2: bool = False
    emit_xtc_topology: bool = False
    unsupported_chemistry_policy: Literal["warn_and_skip", "fail"] = "warn_and_skip"


class Stage9Profile(BaseModel):
    model_config = {"extra": "forbid"}

    minimize: Literal["none", "ligand", "interface", "selected_chains", "complete"] = "none"
    minimize_ff_protein: Literal["ff14SB", "CHARMM36", "AMBER14"] = "ff14SB"
    minimize_ff_nucleic: Literal["DNA.OL3", "RNA.OL3", "CHARMM36"] = "DNA.OL3"
    minimize_solvent: Literal["implicit_GBn2", "implicit_OBC2", "none"] = "implicit_GBn2"
    minimize_restraint_k: float = Field(default=2.0, ge=0.0, le=100.0)
    minimize_restraint_selection: Literal["non_minimized_heavy", "backbone", "none"] = "non_minimized_heavy"
    minimize_max_iterations: int = Field(default=500, ge=1, le=10000)
    minimize_convergence_kj_per_mol: float = Field(default=1.0, ge=0.01, le=100.0)
    minimize_scope_chains: list[str] = Field(default_factory=list)
    minimize_interface_cutoff_A: float = Field(default=5.0, ge=3.0, le=10.0)


class Stage10Profile(BaseModel):
    """Alignment: superpose the mobile structure onto a reference."""

    model_config = {"extra": "forbid"}

    reference_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the reference PDB/mmCIF. "
            "If None, alignment is skipped."
        ),
    )
    reference_chain: str = Field(
        default="A",
        description="Chain in the reference to superpose onto.",
    )
    mobile_chain: str = Field(
        default="A",
        description="Chain in the mobile structure to move.",
    )
    max_rmsd_A: float | None = Field(
        default=None,
        description="Reject if Cα RMSD exceeds this threshold (Å).",
    )


class OutputProfile(BaseModel):
    model_config = {"extra": "forbid"}

    output_format: Literal["pdb", "cif"] = "pdb"
    emit_separate_entities: bool = True
    emit_fasta: bool = True
    emit_receptor_no_h: bool = True


# ---------------------------------------------------------------------------
# Top-level PreparationProfile
# ---------------------------------------------------------------------------

class PreparationProfile(BaseModel):
    """Complete parameterization of one preparation job.

    The SHA-256 of the canonically serialised profile is the job's artifact
    identity — same input + same profile always produces the same output.
    """
    model_config = {"extra": "forbid"}

    version: Literal["1.0"] = "1.0"

    stage1_assembly: Stage1Profile = Field(default_factory=Stage1Profile)
    stage2_roles: Stage2Profile = Field(default_factory=Stage2Profile)
    construct_profile: ConstructProfile = Field(default_factory=ConstructProfile)
    stage3_curation: Stage3Profile = Field(default_factory=Stage3Profile)
    stage4_altlocs: Stage4Profile = Field(default_factory=Stage4Profile)
    stage5_missing: Stage5Profile = Field(default_factory=Stage5Profile)
    stage6_protonation: Stage6Profile = Field(default_factory=Stage6Profile)
    stage7_termini: Stage7Profile = Field(default_factory=Stage7Profile)
    stage8_ligand: Stage8Profile = Field(default_factory=Stage8Profile)
    stage9_minimization: Stage9Profile = Field(default_factory=Stage9Profile)
    stage10_alignment: Stage10Profile = Field(default_factory=Stage10Profile)
    output: OutputProfile = Field(default_factory=OutputProfile)

    def profile_hash(self) -> str:
        """SHA-256 of canonically serialised profile (stable artifact identity)."""
        canonical = self.model_dump_json()
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inventory report models (Stage 0)
# ---------------------------------------------------------------------------

class ChainInfo(BaseModel):
    id: str
    entity_type: Literal["polymer", "non-polymer", "water", "branched"]
    n_residues_atom: int
    n_residues_seqres: int
    has_seqres: bool
    sequence_from_atom: str = ""
    sequence_from_seqres: str | None = None


class HetAtmGroup(BaseModel):
    comp_id: str
    chain: str
    n_instances: int
    mean_occupancy: float | None = None
    mean_bfactor: float | None = None
    classification: Literal[
        "water", "metal", "structural_metal", "crystallographic_ion",
        "cofactor", "likely_ligand", "additive", "glycan",
        "covalent_ligand", "unknown",
    ] = "unknown"
    in_additive_registry: bool = False
    registry_category: str | None = None


class AltlocPosition(BaseModel):
    chain: str
    resseq: int
    resname: str
    altloc_codes: list[str]
    occupancies: dict[str, float] = Field(default_factory=dict)
    near_ligand: bool = False


class DisulfideBond(BaseModel):
    chain1: str
    resseq1: int
    chain2: str
    resseq2: int
    distance_A: float | None = None


class CovalentLink(BaseModel):
    resname1: str
    chain1: str
    resseq1: int
    atom1: str
    resname2: str
    chain2: str
    resseq2: int
    atom2: str
    distance_A: float | None = None
    involves_ligand: bool = False


class InventoryReport(BaseModel):
    """Stage 0 output — complete catalog of input before any mutation."""

    format: Literal["pdb", "cif"]
    n_models: int
    model_ids: list[str]
    chains: list[ChainInfo]
    hetatm_groups: list[HetAtmGroup]
    disulfide_bonds: list[DisulfideBond] = Field(default_factory=list)
    covalent_links: list[CovalentLink] = Field(default_factory=list)
    altloc_positions: list[AltlocPosition] = Field(default_factory=list)
    assembly_info: dict[str, Any] = Field(default_factory=dict)
    space_group: str | None = None
    seqres_vs_atom: dict[str, Any] = Field(default_factory=dict)
    engineered_residues: list[dict[str, Any]] = Field(default_factory=list)
    format_warnings: list[str] = Field(default_factory=list)
    checksum_sha256: str = ""


# ---------------------------------------------------------------------------
# RoleMap (Stage 2)
# ---------------------------------------------------------------------------

class RetainedMetal(BaseModel):
    comp_id: str
    element: str
    chain: str
    resseq: int
    classification: Literal["structural", "all_policy", "crystallographic"]
    nearest_protein_distance_A: float | None = None
    coordinating_residues: list[str] = Field(default_factory=list)


class RoleMap(BaseModel):
    """Stage 2 output — role label for every entity."""

    protein_chains: list[str] = Field(default_factory=list)
    binder_chains: list[str] = Field(default_factory=list)
    nucleic_acid_chains: list[str] = Field(default_factory=list)
    chain_roles: dict[str, str] = Field(default_factory=dict)
    hetatm_roles: dict[str, str] = Field(default_factory=dict)  # "COMP:CHAIN:RESSEQ" -> role
    structural_metals: list[RetainedMetal] = Field(default_factory=list)
    crystallographic_ions: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous_groups: list[dict[str, Any]] = Field(default_factory=list)


class ResidueNumberingEntry(BaseModel):
    chain: str
    sequence_position: int
    author_resseq: int
    author_icode: str = ""
    label_seq_id: int | None = None
    resname: str
    one_letter: str
    uniprot_position: int | None = None


class ConstructAssessment(BaseModel):
    chain: str
    author_start: int | None = None
    author_end: int | None = None
    n_polymer_residues: int = 0
    non_native_n_terminal: bool = False
    suspected_n_tag: str | None = None
    suspected_c_tag: str | None = None
    form: Literal["mature", "immature", "unknown"] = "unknown"
    trimmed_residues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ligand parameterization result (Stage 8)
# ---------------------------------------------------------------------------

class LigandParamResult(BaseModel):
    comp_id: str
    chain: str
    resseq: int
    formal_charge: int | None = None
    n_heavy_atoms: int = 0
    smiles_used: str | None = None
    smiles_source: Literal["user_override", "ccd_cif", "ccd_rcsb", "geometry_perceived", "unavailable"] = "unavailable"
    inchikey_smiles: str | None = None
    inchikey_3d: str | None = None
    inchikey_match: bool | None = None
    inchikey_exact_match: bool | None = None
    occupancy: float | None = None
    b_factor: float | None = None
    covalent_bond_to: str | None = None
    parameterization_method: str = "none"
    parameterization_status: Literal["ok", "unsupported", "skipped", "failed"] = "skipped"
    charge_method: str | None = None
    ff_version: str | None = None
    nagl_model: str | None = None
    parameterization_warnings: list[str] = Field(default_factory=list)
    artifact_sdf: str | None = None
    artifact_mol2: str | None = None
    artifact_topology_xml: str | None = None


# ---------------------------------------------------------------------------
# PreparationReport (full audit output)
# ---------------------------------------------------------------------------

class CurationEntry(BaseModel):
    entity: str
    decision: Literal["removed", "retained"]
    reason: str
    rule: str


class AltlocResolved(BaseModel):
    chain: str
    resseq: int
    resname: str
    selected_altloc: str
    selected_occupancy: float | None = None
    discarded_altlocs: list[str] = Field(default_factory=list)
    near_ligand: bool = False
    distance_to_ligand_A: float | None = None


class HisState(BaseModel):
    chain: str
    resseq: int
    state: Literal["HID", "HIE", "HIP"]
    propka_pka: float | None = None
    metal_coordination: str | None = None
    reason: str


class CysState(BaseModel):
    chain: str
    resseq: int
    state: Literal["CYS", "CYX", "CYM"]
    disulfide_partner: str | None = None
    reason: str


class CapEntry(BaseModel):
    chain: str
    terminus_type: Literal["n_terminal", "c_terminal", "chain_break_n", "chain_break_c"]
    cap_residue: Literal["ACE", "NME"]
    position: int


class MinimizationResult(BaseModel):
    performed: bool = False
    scope: str | None = None
    ff_protein: str | None = None
    ff_nucleic: str | None = None
    ff_ligand: str | None = None
    solvent: str | None = None
    restraint_k_kcal_mol_A2: float | None = None
    n_iterations_run: int | None = None
    converged: bool | None = None
    initial_energy_kj_mol: float | None = None
    final_energy_kj_mol: float | None = None
    energy_delta_kj_mol: float | None = None
    max_coordinate_displacement_A: float | None = None
    label: str = "geometry_cleanup — not an affinity prediction"


class AlignmentResult(BaseModel):
    """Stage 10 output."""

    performed: bool = False
    reference_path: str | None = None
    reference_chain: str | None = None
    mobile_chain: str | None = None
    n_aligned: int | None = None
    rmsd_ca_A: float | None = None
    transform_4x4: list[list[float]] | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ArtifactPaths(BaseModel):
    complex: str | None = None
    receptor: str | None = None
    receptor_noh: str | None = None
    binder: str | None = None
    water: str | None = None
    ions: str | None = None
    ligands: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    sequences_fasta: str | None = None
    aligned_complex: str | None = None
    residue_numbering_json: str | None = None
    entity_map: str | None = None
    inventory_json: str = ""
    report_json: str = ""
    log: str = ""


class PreparationReport(BaseModel):
    """Full audit report — one section per pipeline stage."""

    schema_version: str = "1.0"
    job_id: str
    profile_hash: str

    # Input metadata
    input_path: str = ""
    input_format: str = ""
    input_sha256: str = ""
    input_source: str = "unknown"
    experimental_context: str = "unknown"
    received_at: str = ""

    # Stage 0
    inventory: InventoryReport | None = None

    # Sequence info
    sequences_by_chain: dict[str, str] = Field(default_factory=dict)
    residue_numbering: list[ResidueNumberingEntry] = Field(default_factory=list)
    construct_assessment: list[ConstructAssessment] = Field(default_factory=list)
    construct_form: str = "unknown"

    # Stage 10
    alignment: AlignmentResult = Field(default_factory=AlignmentResult)
    chain_selection_log: dict[str, Any] = Field(default_factory=dict)
    engineered_residues: list[dict[str, Any]] = Field(default_factory=list)
    seqres_vs_atom: dict[str, Any] = Field(default_factory=dict)

    # Stage 1
    assembly_applied: str | None = None
    chain_rename_map: dict[str, str] = Field(default_factory=dict)
    model_selected: str = "1"

    # Stage 2
    roles: RoleMap | None = None

    # Stage 3
    curation_log: list[CurationEntry] = Field(default_factory=list)
    n_waters_retained: int = 0
    removed_additives: list[dict[str, Any]] = Field(default_factory=list)

    # Stage 4
    altloc_policy: str = "highest_occupancy"
    altlocs_resolved: list[AltlocResolved] = Field(default_factory=list)
    binding_site_altlocs: list[AltlocResolved] = Field(default_factory=list)

    # Stage 5
    missing_policy: str = "model_atoms"
    missing_residues: list[dict[str, Any]] = Field(default_factory=list)
    missing_atoms: list[dict[str, Any]] = Field(default_factory=list)
    n_atoms_modeled: int = 0
    n_residues_modeled: int = 0

    # Stage 6
    protonation_method: str = "pdb2pqr_propka"
    protonation_ph: float = 7.4
    n_h_atoms_added: int = 0
    histidines: list[HisState] = Field(default_factory=list)
    acidic_residues: list[dict[str, Any]] = Field(default_factory=list)
    cysteines: list[CysState] = Field(default_factory=list)
    metals_coordination: list[dict[str, Any]] = Field(default_factory=list)

    # Stage 7
    termini_policy: str = "none"
    termini_capped: list[CapEntry] = Field(default_factory=list)

    # Stage 8
    ligands: list[LigandParamResult] = Field(default_factory=list)

    # Stage 9
    minimization: MinimizationResult = Field(default_factory=MinimizationResult)

    # Output
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
    warnings: list[str] = Field(default_factory=list)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    stage_timings_s: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

class JobRecord(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    input_path: str
    profile: PreparationProfile
    job_dir: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    report: PreparationReport | None = None


__all__ = [
    "PreparationProfile",
    "Stage1Profile", "Stage2Profile", "Stage3Profile", "Stage4Profile",
    "Stage5Profile", "Stage6Profile", "Stage7Profile", "Stage8Profile",
    "Stage9Profile", "OutputProfile", "ResidueStateOverride",
    "PreparationReport", "InventoryReport", "RoleMap", "LigandParamResult",
    "JobRecord", "ArtifactPaths", "MinimizationResult",
    "ChainInfo", "HetAtmGroup", "AltlocPosition", "DisulfideBond",
    "CovalentLink", "RetainedMetal", "CurationEntry", "AltlocResolved",
    "HisState", "CysState", "CapEntry",
]
