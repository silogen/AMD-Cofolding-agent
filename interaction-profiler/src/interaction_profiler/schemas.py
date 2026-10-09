# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
]
