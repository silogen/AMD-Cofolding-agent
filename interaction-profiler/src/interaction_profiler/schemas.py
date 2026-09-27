"""Pydantic output models for the interaction profiler (v2).

One batch in -> one :class:`BatchResult` out. Every field is explicitly
named and flat-friendly so a downstream consumer can call
:func:`interaction_profiler.schemas.BatchResult.to_rows` (or the
``to_table`` helper in ``core``) and drop the result straight into a
dataframe -- one row per interaction / triple.
"""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import computed_field  # noqa: E402 — v2.5+ feature, used on ComplexResult


def package_version() -> str:
    """Installed package version (single source of truth = pyproject)."""
    try:
        import importlib.metadata as md

        return md.version("interaction-profiler")
    except Exception:  # noqa: BLE001 - not installed as a dist (editable src run)
        return "0.3.0"


def detect_tool_versions() -> dict[str, str]:
    """Versions of the scientific stack, for provenance. Missing deps are
    simply omitted."""
    versions: dict[str, str] = {}
    for name, mod in (
        ("rdkit", "rdkit"),
        ("MDAnalysis", "MDAnalysis"),
        ("accfg", "accfg"),
        ("openmm", "openmm"),
        ("dimorphite_dl", "dimorphite_dl"),
    ):
        try:
            m = importlib.import_module(mod)
            versions[name] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    return versions

# ---------------------------------------------------------------------------
# Composition / fragmentation
# ---------------------------------------------------------------------------


class Fragment(BaseModel):
    """A named substructure of the ligand.

    ``atom_indices`` are 0-based indices into the prepared ligand RDKit
    Mol (heavy-atom indexing is stable because the join happens before
    Hs are stripped for reporting). ``level`` places the fragment in the
    medchem hierarchy.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool-native fragment label.")
    canonical_name: str = Field(description="Normalised label.")
    level: str = Field(description="Hierarchy level: functional_group | ring_system | scaffold | atom_group.")
    atom_indices: list[int] = Field(description="0-based ligand atom indices in this fragment.")
    smarts: str | None = Field(default=None, description="SMARTS pattern matched, if any.")
    smiles: str | None = Field(default=None, description="SMILES of the isolated fragment, if computed.")


class Composition(BaseModel):
    """Hierarchical medchem taxonomy for one molecule."""

    model_config = ConfigDict(extra="forbid")

    fragments: list[Fragment] = Field(default_factory=list)
    coverage: float = Field(default=0.0, description="Fraction of heavy atoms covered by >=1 fragment.")
    n_heavy_atoms: int = 0


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------


class Interaction(BaseModel):
    """One non-covalent (or covalent / metal / water-bridge) contact.

    All partner metadata is spelled out so a table row is self-describing.
    """

    model_config = ConfigDict(extra="forbid")

    interaction_type: str = Field(description="Canonical family: hbond | hydrophobic | pi_stacking | pi_cation | cation_pi | ionic | halogen_bond | chalcogen_bond | metal | water_bridge | ch_pi | chalcogen_pi | amide_pi | vdw_contact | covalent." )
    subtype: str | None = Field(default=None, description="e.g. hbond_donor/hbond_acceptor, pi_face/pi_edge, ionic_pos/ionic_neg.")

    ligand_atom_idx: int = Field(description="0-based ligand atom index (-1 if not attributable / group centroid).")
    ligand_atom_name: str | None = None
    ligand_element: str | None = None

    receptor_chain: str = ""
    receptor_resname: str = ""
    receptor_resnum: int | None = None
    receptor_reskind: str = Field(default="protein", description="protein | rna | dna | ion | water | other.")
    receptor_atom_name: str | None = None
    receptor_moiety: str | None = Field(default=None, description="sidechain | backbone | ion | water | None.")

    distance: float | None = Field(default=None, description="Angstrom.")
    angle: float | None = Field(default=None, description="Degrees (D-H..A for H-bonds, C-X..A for halogen, ring angle for pi).")
    direction: str | None = Field(default=None, description="ligand_donor | ligand_acceptor | None.")
    ligand_role: str | None = Field(default=None, description="Role of the ligand side: cation | pi | water | None.")

    fragment: str | None = Field(default=None, description="Canonical fragment the ligand atom belongs to, if any.")
    confidence: str = Field(default="confident", description="confident | uncertain.")

    @property
    def residue_label(self) -> str:
        num = "" if self.receptor_resnum is None else str(self.receptor_resnum)
        base = f"{self.receptor_resname}{num}"
        return f"{base}.{self.receptor_chain}" if self.receptor_chain else base


class Triple(BaseModel):
    """Grouped ``fragment x residue x interaction`` row.

    Repeated equivalent contacts (same fragment, residue, moiety,
    interaction subtype, direction) are collapsed into one row with a
    ``count`` and distance / angle ranges -- matching the human summary
    style ``HBondDonor x2  d=3.0-3.3A  theta=141-165``.
    """

    model_config = ConfigDict(extra="forbid")

    fragment: str
    residue_label: str = Field(description="e.g. 'ILE50.A'.")
    residue_name: str
    residue_number: int | None = None
    chain: str = ""
    moiety: str | None = None
    interaction_type: str
    subtype: str | None = None
    direction: str | None = None
    count: int = 1
    distance_min: float | None = None
    distance_max: float | None = None
    angle_min: float | None = None
    angle_max: float | None = None
    n_confident: int = 0
    n_uncertain: int = 0


# ---------------------------------------------------------------------------
# Preparation / provenance
# ---------------------------------------------------------------------------


class StereoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_stereocenters: int = 0
    n_defined: int = 0
    n_undefined: int = 0
    undefined_atom_indices: list[int] = Field(default_factory=list)
    n_double_bond_stereo: int = 0
    n_double_bond_unspecified: int = 0
    matches_input_smiles: bool | None = None


class PreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ligand_prepared: bool = False
    ligand_protonation_ph: float | None = None
    ligand_smiles_input: str | None = None
    ligand_smiles_protonated: str | None = None
    ligand_smiles_prepared: str | None = None
    ligand_n_hydrogens: int = 0
    bond_assignment: str = "none"  # template | perceived | rdkit | failed | smiles_only
    template_inchikey: str | None = Field(default=None, description="InChIKey of the input SMILES template, if any.")
    ligand_inchikey: str | None = Field(default=None, description="InChIKey of the prepared ligand.")
    inchikey_match: bool | None = Field(
        default=None,
        description="Whether the prepared ligand matches the SMILES template by InChIKey "
        "skeleton (first block; ignores stereo/protonation). None when no template.",
    )
    inchikey_exact_match: bool | None = Field(
        default=None,
        description="Whether the full InChIKeys are identical (stereo + charge sensitive). "
        "A True skeleton match with False here means connectivity agrees but "
        "stereochemistry/protonation differs. None when no template.",
    )
    receptor_prepared: bool = False
    receptor_ph: float | None = None
    receptor_atoms_added: int = 0
    receptor_hydrogens_added: bool = False
    prepared_ligand_path: str | None = None
    prepared_receptor_path: str | None = None
    stereo: StereoInfo | None = None
    warnings: list[str] = Field(default_factory=list)


class MoleculeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_mode: str = Field(default="unknown", description="smiles_only | structure_only | structure+smiles.")
    ligand_smiles: str | None = None
    ligand_com: list[float] | None = Field(default=None, description="[x, y, z] center of mass (heavy atoms).")
    receptor: str | None = Field(default=None, description="Summary receptor type: protein | rna | dna | protein+dna | ... | none.")
    chains: list[str] = Field(default_factory=list)
    chain_types: dict[str, str] = Field(default_factory=dict, description="Per-chain type: chain id -> protein | rna | dna | other.")
    ph: float = 7.4
    source: str | None = None
    has_structure: bool = False
    n_interactions: int = 0
    n_interactions_detected: int = 0
    n_dropped: int = 0
    interaction_cap: int | None = None
    n_triples: int = 0
    n_confident: int = 0
    n_uncertain: int = 0
    timing_seconds: float | None = None


class MoleculeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: MoleculeMetadata
    composition: Composition = Field(default_factory=Composition)
    interactions: list[Interaction] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)
    preparation: PreparationReport | None = None
    error: str | None = None


class BatchProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = "interaction-profiler"
    version: str = Field(default_factory=package_version)
    fragment_method: str = "accfg+smarts"
    interaction_method: str = "geometric"
    interaction_radius: float = 8.0
    max_interactions: int | None = Field(default=None, description="Interaction cap per input; None = adaptive (~1.2x heavy atoms).")
    drop_uncertain: bool = Field(default=True, description="If true (default), uncertain interactions are removed AFTER the cap.")
    gpu: bool = False
    tool_versions: dict[str, str] = Field(default_factory=detect_tool_versions)


class BatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str = "batch"
    n_molecules: int = 0
    provenance: BatchProvenance = Field(default_factory=BatchProvenance)
    molecules: list[MoleculeResult] = Field(default_factory=list)

    def triple_rows(self) -> list[dict[str, Any]]:
        """Flatten to one row per grouped triple (table-friendly)."""
        rows: list[dict[str, Any]] = []
        for mol in self.molecules:
            md = mol.metadata
            base = {
                "name": md.name, "ligand_smiles": md.ligand_smiles,
                "receptor": md.receptor, "ph": md.ph, "source": md.source,
            }
            for t in mol.triples:
                rows.append({
                    **base,
                    "fragment": t.fragment,
                    "residue": t.residue_label,
                    "resname": t.residue_name,
                    "resnum": t.residue_number,
                    "chain": t.chain,
                    "moiety": t.moiety,
                    "interaction_type": t.interaction_type,
                    "subtype": t.subtype,
                    "direction": t.direction,
                    "count": t.count,
                    "distance_min": t.distance_min,
                    "distance_max": t.distance_max,
                    "angle_min": t.angle_min,
                    "angle_max": t.angle_max,
                    "n_confident": t.n_confident,
                    "n_uncertain": t.n_uncertain,
                })
        return rows

    def to_rows(self) -> list[dict[str, Any]]:
        """Flatten to one row per interaction (table-friendly)."""
        rows: list[dict[str, Any]] = []
        for mol in self.molecules:
            md = mol.metadata
            base = {
                "name": md.name,
                "ligand_smiles": md.ligand_smiles,
                "receptor": md.receptor,
                "chains": ",".join(md.chains),
                "ph": md.ph,
                "source": md.source,
            }
            if not mol.interactions:
                rows.append({**base, "interaction_type": None})
                continue
            for it in mol.interactions:
                rows.append(
                    {
                        **base,
                        "interaction_type": it.interaction_type,
                        "subtype": it.subtype,
                        "residue": it.residue_label,
                        "resname": it.receptor_resname,
                        "resnum": it.receptor_resnum,
                        "chain": it.receptor_chain,
                        "moiety": it.receptor_moiety,
                        "reskind": it.receptor_reskind,
                        "ligand_atom_idx": it.ligand_atom_idx,
                        "ligand_atom_name": it.ligand_atom_name,
                        "receptor_atom_name": it.receptor_atom_name,
                        "distance": it.distance,
                        "angle": it.angle,
                        "direction": it.direction,
                        "ligand_role": it.ligand_role,
                        "fragment": it.fragment,
                        "confidence": it.confidence,
                    }
                )
        return rows


__all__ = [
    "package_version",
    "detect_tool_versions",
    "Fragment",
    "Composition",
    "Interaction",
    "Triple",
    "StereoInfo",
    "PreparationReport",
    "MoleculeMetadata",
    "MoleculeResult",
    "BatchProvenance",
    "BatchResult",
    # v2 generalized models — all additive, nothing below is changed
    "EntityRole",
    "Entity",
    "EnrichedInteraction",
    "GeneralizedTriple",
    "StrainResult",
    "EfficiencyResult",
    "InterfaceGeometry",
    "HotspotOccupancy",
    "ComplexProvenance",
    "ComplexResult",
]


# ===========================================================================
# v2 — Generalized entity/interaction/metrics models
# ---------------------------------------------------------------------------
# COMPATIBILITY GUARANTEE: nothing above this line is modified.
# All new models are appended here and exported.
# ===========================================================================

from enum import Enum  # noqa: E402  (after existing imports)


class EntityRole(str, Enum):
    """Role of a molecular entity in a complex profiling request."""
    small_molecule = "small_molecule"
    peptide        = "peptide"
    protein        = "protein"
    nanobody       = "nanobody"
    antibody       = "antibody"
    dna            = "dna"
    rna            = "rna"
    cofactor       = "cofactor"
    metal          = "metal"
    water          = "water"


class Entity(BaseModel):
    """One typed participant in a complex (binder or target or context)."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(description="Caller-assigned stable ID, e.g. 'binder_A'.")
    role: EntityRole
    chain_ids: list[str] = Field(description="MDAnalysis chain IDs belonging to this entity.")
    label: str | None = Field(default=None, description="Human-readable description.")
    uniprot_accession: str | None = Field(default=None, description="UniProt accession for annotation fetching.")


class EnrichedInteraction(Interaction):
    """Extends Interaction with biological context annotations.

    All parent fields are unchanged and the parent model still validates
    them with extra='forbid'. New fields are all nullable so existing
    serialised Interaction payloads round-trip cleanly if cast here.
    """

    # entity provenance
    binder_entity_id: str | None = Field(default=None, description="Entity ID of the binder side.")
    target_entity_id: str | None = Field(default=None, description="Entity ID of the target side.")

    # secondary structure at the two sides
    binder_ss_state: str | None = Field(
        default=None,
        description="helix | sheet | loop | turn | disordered | unknown — for the binder atom/residue.",
    )
    target_ss_state: str | None = Field(
        default=None,
        description="helix | sheet | loop | turn | disordered | unknown — for the target residue.",
    )

    # UniProt feature at the target residue
    target_uniprot_feature: str | None = Field(
        default=None,
        description="UniProt feature type at this residue, e.g. 'Active site', 'Binding site', 'Domain'.",
    )
    target_uniprot_description: str | None = Field(
        default=None,
        description="UniProt feature description, e.g. 'Proton acceptor', 'ATP'.",
    )

    # InterPro domain at the target residue
    target_interpro_domain: str | None = Field(
        default=None,
        description="InterPro domain name at this residue, e.g. 'Protein kinase domain'.",
    )
    target_interpro_accession: str | None = Field(
        default=None,
        description="InterPro accession, e.g. 'IPR000719'.",
    )

    # Generalised feature labels (override fragment for non-small-molecule binders)
    binder_feature: str | None = Field(
        default=None,
        description=(
            "Fragment name (small molecule) or 'RESNAME:RESNUM.CHAIN' (protein/peptide binder)."
        ),
    )
    target_feature: str | None = Field(
        default=None,
        description="'RESNAME:RESNUM.CHAIN' for the target residue.",
    )

    # ── Extension block 06.v3 — enriched descriptors (all Optional, backward-compat) ──
    vector_type: str | None = Field(
        default=None,
        description=(
            "Geometric role of the ligand atom in this interaction: "
            "anchor | exit_vector | growth_vector | buried. "
            "Populated when analysis_profile includes vector classification."
        ),
    )
    solvent_exposure: str | None = Field(
        default=None,
        description=(
            "Solvent-exposure bin of the interacting ligand atom: "
            "buried | partially_exposed | solvent_exposed."
        ),
    )
    subpocket_id: str | None = Field(
        default=None,
        description=(
            "Subpocket identifier from SiteProfile (e.g. 'S1', 'hinge_region') "
            "if the target residue has been assigned to a named subpocket."
        ),
    )
    confidence_bin: str | None = Field(
        default=None,
        description=(
            "Interaction confidence bin: high | moderate | low. "
            "Derived from geometric confidence + strain + VDW clashes."
        ),
    )


class GeneralizedTriple(BaseModel):
    """binder_feature × target_feature × interaction_type grouped row.

    For small-molecule binders: binder_feature == fragment name (as in Triple).
    For protein/peptide/antibody binders: binder_feature == residue label.
    This model can therefore represent both classical and generalized contacts
    in a uniform table format.
    """

    model_config = ConfigDict(extra="forbid")

    binder_entity_id: str | None = None
    target_entity_id: str | None = None

    binder_feature: str = Field(description="Fragment name (small mol) or residue label (protein binder).")
    target_feature: str = Field(description="Residue label, e.g. 'ILE50.A'.")
    binder_feature_type: str = Field(
        description="fragment | residue | backbone | sidechain | ion | water.",
    )

    interaction_type: str
    subtype: str | None = None
    count: int = 1

    distance_min: float | None = None
    distance_max: float | None = None
    angle_min: float | None = None
    angle_max: float | None = None
    n_confident: int = 0
    n_uncertain: int = 0

    # Taxonomy annotations from target residue
    target_ss_state: str | None = None
    target_uniprot_feature: str | None = None
    target_uniprot_description: str | None = None
    target_interpro_domain: str | None = None
    target_interpro_accession: str | None = None


# ---------------------------------------------------------------------------
# Metric result models
# ---------------------------------------------------------------------------

class StrainResult(BaseModel):
    """Conformational strain of a small-molecule ligand (Lai 2026 MMFF94 method).

    strain = MMFF94(bound_coordinates) − MMFF94(isolated_optimized_coordinates)
    normalised by heavy-atom count.
    """

    model_config = ConfigDict(extra="forbid")

    bound_energy_kcal_mol: float | None = None
    optimized_energy_kcal_mol: float | None = None
    strain_kcal_mol: float | None = None
    strain_per_heavy_atom: float | None = None
    n_heavy_atoms: int = 0
    coordinate_source: str | None = Field(
        default=None,
        description="crystal | boltz | gnina | generated — where the 3D coordinates came from.",
    )
    method: str = "mmff94_rdkit"
    rdkit_version: str | None = None
    error: str | None = None


class EfficiencyResult(BaseModel):
    """LE / LLE / BEI computed from one potency value and the ligand SMILES.

    Potency priority: experimental > Boltz (gate-checked) > model > docking.
    Every unit conversion is recorded in conversion_record.
    """

    model_config = ConfigDict(extra="forbid")

    le: float | None = Field(default=None, description="Ligand Efficiency kcal/mol per heavy atom.")
    lle: float | None = Field(default=None, description="Lipophilic LE = pIC50 − clogP.")
    bei: float | None = Field(default=None, description="Binding Efficiency Index = pIC50/MW × 1000.")
    # G1: group efficiency for fragment elaboration steps
    ge: float | None = Field(
        default=None,
        description=(
            "Group Efficiency = delta(LE) * parent_n_heavy / delta_n_heavy. "
            "Populated only when parent_le and parent_n_heavy are supplied to "
            "compute_efficiency(). A GE >= 1.0 means the added atoms improved LE."
        ),
    )
    # G2: LELP = LE / clogP (optional, only meaningful when clogP > 0)
    lelp: float | None = Field(default=None, description="LE/clogP = LELP metric.")

    potency_source: str | None = Field(
        default=None,
        description="experimental | boltz | model | docking — which source was used.",
    )
    potency_value_pic50: float | None = Field(
        default=None,
        description="Potency after conversion to pIC50 scale used for LE/LLE/BEI.",
    )
    potency_type_original: str | None = Field(
        default=None,
        description="Original potency type, e.g. IC50, Ki, Kd, pIC50.",
    )
    conversion_record: str | None = Field(
        default=None,
        description="Human-readable log of every conversion step applied.",
    )
    clogp: float | None = None
    mw_da: float | None = None
    n_heavy_atoms: int = 0
    error: str | None = None


class InterfaceGeometry(BaseModel):
    """Buried surface area and shape complementarity for a binder–target pair."""

    model_config = ConfigDict(extra="forbid")

    bsa_A2: float | None = Field(default=None, description="Buried surface area in Angstrom².")
    shape_complementarity: float | None = Field(
        default=None,
        description="Shape complementarity score (Lawrence & Colman Sc, or equivalent).",
    )
    n_interface_residues_binder: int = 0
    n_interface_residues_target: int = 0
    interface_residues_binder: list[str] = Field(
        default_factory=list,
        description="Residue labels of the binder interface.",
    )
    interface_residues_target: list[str] = Field(
        default_factory=list,
        description="Residue labels of the target interface.",
    )
    method_bsa: str = "freesasa_shrake_rupley"
    probe_radius_A: float = 1.4
    method_sc: str | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal quality warnings, e.g. when the binder/target selection "
            "would overflow fixed-column PDB limits (>99,999 atoms, residue "
            "numbers >9,999, or multi-character chain IDs) so per-residue "
            "interface assignments may be unreliable."
        ),
    )
    error: str | None = None


class HotspotOccupancy(BaseModel):
    """Per-hotspot contact occupancy across an ensemble of structures."""

    model_config = ConfigDict(extra="forbid")

    target_residue: str = Field(description="Residue label of the hotspot, e.g. 'ASP189.A'.")
    n_structures_observed: int = 0
    n_structures_total: int = 0
    occupancy: float | None = Field(default=None, description="n_observed / n_total.")
    interaction_types_observed: list[str] = Field(default_factory=list)
    per_structure: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One dict per structure: {source, interaction_type, distance, confidence}.",
    )
    missing_structures: list[str] = Field(
        default_factory=list,
        description="Sources where the hotspot was not found.",
    )


class ComplexProvenance(BaseModel):
    """Provenance block for a /profile_complex result."""

    model_config = ConfigDict(extra="forbid")

    tool: str = "interaction-profiler"
    version: str = Field(default_factory=package_version)
    mode: str = "ligand_target"
    interaction_method: str = "geometric"
    secondary_structure_method: str | None = None
    annotation_source: str | None = None   # "uniprot_api" | "cached_artefact" | None
    interpro_source: str | None = None
    bsa_method: str | None = None
    sc_method: str | None = None
    strain_method: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=detect_tool_versions)
    structure_path: str | None = None  # ensemble member source path (hotspot_ensemble)


class ComplexResult(BaseModel):
    """Rich output from POST /profile_complex.

    Carries the full generalized entity model, enriched interactions,
    generalised triples, and metric bundles. The existing Composition +
    PreparationReport sub-objects are included as a small-molecule
    compatibility projection so callers that understand the v1 models can
    still read them.

    ``extra='ignore'`` is intentional: this is an *output* schema.
    Computed fields (``n_entities``, ``n_interactions``, ``n_triples``)
    are emitted by ``model_dump()`` but are silently re-derived on
    ``model_validate()`` — passing them back in round-trip dicts is safe.
    """

    model_config = ConfigDict(extra="ignore")  # output model: ignore computed fields on re-parse

    mode: str = Field(description="ligand_only | ligand_target | binder_target | hotspot_ensemble.")
    provenance: ComplexProvenance = Field(default_factory=ComplexProvenance)
    entities: list[Entity] = Field(default_factory=list)

    # Core interaction output
    enriched_interactions: list[EnrichedInteraction] = Field(default_factory=list)
    generalized_triples: list[GeneralizedTriple] = Field(default_factory=list)

    # Metric bundles (None when not requested or not applicable)
    strain: StrainResult | None = None
    efficiency: EfficiencyResult | None = None
    interface: InterfaceGeometry | None = None
    hotspots: list[HotspotOccupancy] = Field(default_factory=list)

    # Small-molecule compatibility projection (populated for ligand_only / ligand_target)
    composition: Composition | None = None
    preparation: PreparationReport | None = None

    # ── Extension block 06.v3 — ensemble and site fields (all Optional) ──
    site_profile: "SiteProfile | None" = Field(
        default=None,
        description=(
            "Protein-side pocket characterization (protein_only mode or reused from cache). "
            "Populated when mode='protein_only' or site_profile_id is supplied."
        ),
    )
    fragment_families: "list[FragmentFamily]" = Field(
        default_factory=list,
        description=(
            "Fragment families grouped by dominant binding mode. "
            "Populated by ensemble_profile='families' or 'full'."
        ),
    )
    compatible_pairs: "list[CompatiblePair]" = Field(
        default_factory=list,
        description=(
            "Fragment pairs that bind non-overlapping regions and could be linked. "
            "Populated by ensemble_profile='full'."
        ),
    )

    warnings: list[str] = Field(default_factory=list)
    timing_seconds: float | None = None

    # ── Summary counters (auto-computed, always up-to-date) ──────────────────
    @computed_field  # type: ignore[misc]
    @property
    def n_entities(self) -> int:
        """Number of resolved entities (binder + target + context)."""
        return len(self.entities)

    @computed_field  # type: ignore[misc]
    @property
    def n_interactions(self) -> int:
        """Number of enriched interactions detected."""
        return len(self.enriched_interactions)

    @computed_field  # type: ignore[misc]
    @property
    def n_triples(self) -> int:
        """Number of generalised binder×target×interaction_type triples."""
        return len(self.generalized_triples)

    def generalized_triple_rows(self) -> list[dict[str, Any]]:
        """Flatten to one row per GeneralizedTriple (table-friendly)."""
        return [t.model_dump() for t in self.generalized_triples]


# ===========================================================================
# Extension block 06.v3 — Site characterization and enriched ensemble models
# ---------------------------------------------------------------------------
# All models below are ADDITIVE.  ComplexResult gains optional references to
# SiteProfile / FragmentFamily / CompatiblePair (defaulting to None / []) so
# every existing serialised result continues to deserialise without changes.
# ===========================================================================


class ResidueRoleEnum(str, Enum):
    """Classification of a residue's functional role in the binding site."""
    catalytic      = "catalytic"       # UniProt 'Active site'
    gatekeeper     = "gatekeeper"      # pocket entrance, controls access
    anchor         = "anchor"          # conserved, buried, high-contact
    solvent_exposed = "solvent_exposed" # consistently exposed in pocket
    hinge          = "hinge"           # InterPro hinge / high B-factor
    structural     = "structural"      # scaffold, rarely interacts


class PocketSubpocket(BaseModel):
    """One named sub-region within the overall binding pocket.

    Derived from clustering fpocket alpha-spheres and mapping UniProt/InterPro
    features onto the resulting clusters.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Subpocket identifier, e.g. 'S1', 'S2', 'hinge'.")
    volume_A3: float | None = Field(default=None, description="Approximate volume in Å³.")
    character: str | None = Field(
        default=None,
        description="Physicochemical character: hydrophobic | polar | mixed.",
    )
    residues: list[str] = Field(
        default_factory=list,
        description="Residue labels composing this subpocket, e.g. ['GLY143.A', 'CYS145.A'].",
    )
    uniprot_feature: str | None = Field(
        default=None,
        description="UniProt feature name if this subpocket overlaps an annotated site.",
    )
    druggability_score: float | None = Field(
        default=None,
        description="fpocket druggability score for this subpocket (0–1).",
    )


class SiteProfile(BaseModel):
    """Protein binding-site characterization from protein_only mode.

    Produced by ``profile_complex(mode='protein_only')`` and cached by
    SHA-256 so subsequent ligand_target / ensemble calls can reuse it without
    recomputing pocket geometry.
    """

    model_config = ConfigDict(extra="ignore")

    structure_path: str = Field(description="Absolute path to the characterised structure file.")
    structure_sha256: str = Field(description="SHA-256 hex digest of the structure file.")
    target_chains: list[str] = Field(description="Protein chains that were characterised.")
    analysis_profile: str = Field(
        description="Profile used: structure | pocket | annotated | full.",
    )

    # ── Site type ──────────────────────────────────────────────────────────
    site_type: str | None = Field(
        default=None,
        description="orthosteric | allosteric | ppi | catalytic | unknown.",
    )
    site_type_confidence: str | None = Field(
        default=None,
        description="Classification confidence: high | moderate | low.",
    )

    # ── Pocket geometry (from fpocket) ────────────────────────────────────
    pocket_volume_A3: float | None = Field(
        default=None,
        description="Approximate pocket volume in Å³ (fpocket top-ranked pocket).",
    )
    pocket_druggability: float | None = Field(
        default=None,
        description="fpocket druggability score 0–1 (higher = more druggable).",
    )
    pocket_character: str | None = Field(
        default=None,
        description="Overall pocket physicochemical character: polar | hydrophobic | mixed.",
    )
    polar_fraction: float | None = Field(
        default=None,
        description="Fraction of buried pocket surface that is polar (0–1).",
    )
    subpockets: list[PocketSubpocket] = Field(
        default_factory=list,
        description="Named sub-regions within the pocket.",
    )

    # ── Residue roles ─────────────────────────────────────────────────────
    residue_roles: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Role name → list of residue labels, e.g. {'catalytic': ['HIS163.A']}.",
    )
    active_site_residues: list[str] = Field(
        default_factory=list,
        description="Residues with UniProt 'Active site' annotation.",
    )
    binding_site_residues: list[str] = Field(
        default_factory=list,
        description="Residues with UniProt 'Binding site' annotation.",
    )

    # ── Structure context ─────────────────────────────────────────────────
    ss_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Secondary structure fractions: {'helix': 0.4, 'sheet': 0.3, 'loop': 0.3}.",
    )
    domain_context: dict[str, str] = Field(
        default_factory=dict,
        description="Chain → InterPro domain name for the dominant domain in the pocket.",
    )

    # ── Provenance ────────────────────────────────────────────────────────
    fpocket_version: str | None = Field(default=None, description="fpocket binary version.")
    component_status: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Named status for optional components (available, skipped, or "
            "error:<reason>). Empty scientific fields must not hide missing "
            "dependencies."
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    timing_seconds: float | None = None


class FragmentFamily(BaseModel):
    """A cluster of fragments sharing the same dominant binding mode.

    Derived from clustering ``generalized_triples`` by
    (primary_residue, primary_interaction_type) across an ensemble.
    Requires no additional geometry computation.
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(description="Family identifier, e.g. 'F001'.")
    members: list[str] = Field(
        description="Structure paths or fragment identifiers in this family.",
    )
    primary_residue: str | None = Field(
        default=None,
        description="Dominant interacting residue across family members.",
    )
    primary_interaction: str | None = Field(
        default=None,
        description="Most common interaction type, e.g. 'hbond', 'hydrophobic'.",
    )
    occupancy: float | None = Field(
        default=None,
        description="Fraction of ensemble structures in this family.",
    )
    subpocket_ids: list[str] = Field(
        default_factory=list,
        description="Subpockets occupied by members of this family.",
    )
    binding_mode_label: str | None = Field(
        default=None,
        description=(
            "Human-readable binding-mode label built from primary interaction "
            "and residue, e.g. 'HBD-His163-S1'."
        ),
    )


class CompatiblePair(BaseModel):
    """Two fragments that bind non-overlapping regions and could be linked or merged.

    Determined by comparing ``interface_residues_binder`` sets across two
    ``ComplexResult`` objects from the same ensemble.
    """

    model_config = ConfigDict(extra="forbid")

    fragment_a: str = Field(description="Structure path or fragment ID of the first fragment.")
    fragment_b: str = Field(description="Structure path or fragment ID of the second fragment.")
    overlap_residues: list[str] = Field(
        default_factory=list,
        description="Shared interface residues (empty = non-overlapping = fully compatible).",
    )
    compatible: bool = Field(
        description="True when overlap_residues is empty and geometry allows linking.",
    )
    linker_distance_A: float | None = Field(
        default=None,
        description=(
            "Estimated distance between exit vectors of the two fragments in Å. "
            "None when exit vectors are not classified."
        ),
    )
    notes: str | None = Field(
        default=None,
        description="Human-readable notes, e.g. steric clash risk or preferred linker type.",
    )


# Rebuild ComplexResult so forward references to new models resolve
ComplexResult.model_rebuild()

__all__ = [
    # v1
    "package_version",
    "detect_tool_versions",
    "Fragment",
    "Composition",
    "Interaction",
    "Triple",
    "StereoInfo",
    "PreparationReport",
    "MoleculeMetadata",
    "MoleculeResult",
    "BatchProvenance",
    "BatchResult",
    # v2
    "EntityRole",
    "Entity",
    "EnrichedInteraction",
    "GeneralizedTriple",
    "StrainResult",
    "EfficiencyResult",
    "InterfaceGeometry",
    "HotspotOccupancy",
    "ComplexProvenance",
    "ComplexResult",
    # v3 extension
    "ResidueRoleEnum",
    "PocketSubpocket",
    "SiteProfile",
    "FragmentFamily",
    "CompatiblePair",
]
