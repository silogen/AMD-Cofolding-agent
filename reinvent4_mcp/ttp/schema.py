# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TTP Pydantic models and validation.

Defines the Therapeutic Target Profile schema used throughout the MCP
server.  A TTP has two scoring stages (pre-scoring and scoring) that
are merged into a single composed Scorer at runtime.

Validation rules
-----------------
* TTP ``id`` must be a lowercase alphanumeric slug.
* Pre-scoring components must reference known fast RDKit descriptors.
* Scoring objectives must be either ``native`` (REINVENT4 built-in) or
  ``external`` (REST service).  External objectives require ``service``
  and ``service_url``.
* ``depends_on`` references are checked against sibling objective names.
* Transform ``type`` is normalised and validated against REINVENT4's
  transform registry.
* Weights must be positive.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TRANSFORM_TYPES = frozenset(
    {
        "sigmoid",
        "reverse_sigmoid",
        "double_sigmoid",
        "right_step",
        "left_step",
        "step",
        "value_mapping",
        "exponential_decay",
    }
)

_TRANSFORM_TYPES_NORMALISED = {t.replace("_", ""): t for t in VALID_TRANSFORM_TYPES}

VALID_PRESCORING_COMPONENTS = frozenset(
    {
        "QED",
        "MolecularWeight",
        "HBondDonors",
        "HBondAcceptors",
        "SlogP",
        "NumRotBond",
    }
)

# First-class external services that have a config entry + canonical URL in the
# process config (see reinvent4_mcp.config.enabled_external_services). This is an
# advisory allow-list, NOT an enforced one: the TTP mapper deliberately supports
# arbitrary custom REST services (it falls back to the TTP-supplied service_url
# for anything not listed here), so an unknown service is permitted rather than
# rejected. Keep this in sync with config.ScoringConfig when adding a service.
KNOWN_EXTERNAL_SERVICES = frozenset(
    {"boltz", "posebusters", "admet", "retrodmta", "interaction_profiler"}
)

_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TransformConfig(BaseModel):
    """Score transform mapping raw component values to [0, 1].

    The ``type`` field selects the transform function.  Additional fields
    are type-specific parameters forwarded to REINVENT4 at runtime.

    Available types::

        sigmoid / reverse_sigmoid  ->  low, high, k
        double_sigmoid             ->  low, high, coef_div, coef_si, coef_se
        right_step / left_step / step  ->  low, high
        value_mapping              ->  mapping (dict)
        exponential_decay          ->  k
    """

    model_config = {"extra": "allow"}

    type: str = Field(description="Transform function name")

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        normalised = v.lower().replace("_", "").replace("-", "")
        if normalised not in _TRANSFORM_TYPES_NORMALISED:
            raise ValueError(
                f"Unknown transform type '{v}'. "
                f"Valid types: {', '.join(sorted(VALID_TRANSFORM_TYPES))}"
            )
        return v


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


class TargetConfig(BaseModel):
    """Protein target information for structure-based scoring.

    ``protein_sequence`` is required for Boltz cofolding.  Other fields
    supply additional structural context.
    """

    model_config = {"extra": "forbid"}

    protein_sequence: str = Field(
        description="Full amino-acid sequence of the target protein",
    )
    pdb_id: Optional[str] = Field(
        default=None,
        description="PDB identifier (e.g. '3KRR')",
    )
    msa_path: Optional[str] = Field(
        default=None,
        description="Path to pre-computed MSA file (.a3m)",
    )
    protein_pdb_path: Optional[str] = Field(
        default=None,
        description="Path to protein structure PDB file",
    )
    # E3: binding-site constraints
    binding_site_residues: Optional[list[list]] = Field(
        default=None,
        description=(
            "Pocket constraint contacts: list of [chain_id, author_resseq] pairs "
            "using PDB author numbering. The framework translates them to Boltz "
            "sequence positions via the residue_numbering.json artifact."
        ),
    )
    binding_site_max_distance_A: float = Field(
        default=6.0,
        ge=4.0,
        le=20.0,
        description="max_distance for the pocket constraint (Angstrom).",
    )
    binding_site_force: bool = Field(
        default=False,
        description="Apply the pocket constraint as a hard potential (E4).",
    )
    contact_pairs: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Per-atom contact constraints. Each entry: "
            "{token1: ['A', resseq], token2: ['B', atom_name], "
            "max_distance: 3.5, force: true}."
        ),
    )
    covalent_bond: Optional[dict] = Field(
        default=None,
        description=(
            "Bond constraint for a covalent warhead (L1). "
            "{atom1: ['A', 145, 'SG'], atom2: ['B', 1, 'C7']}. "
            "Affinity prediction is suppressed automatically (L2)."
        ),
    )

    @field_validator("protein_sequence")
    @classmethod
    def _validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) < 10:
            raise ValueError("Protein sequence must be at least 10 residues")
        invalid = set(v) - _VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(
                f"Invalid amino-acid characters: {', '.join(sorted(invalid))}. "
                f"Sequence must only contain standard amino acids."
            )
        return v


# ---------------------------------------------------------------------------
# Pre-scoring
# ---------------------------------------------------------------------------


class PreScoringComponent(BaseModel):
    """A single pre-scoring component (fast RDKit descriptor).

    Pre-scoring components compute drug-likeness descriptors for every
    molecule and contribute to the final aggregated score.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(description="Display name for this component")
    type: Literal["native"] = Field(
        default="native",
        description="Component type (always 'native' for pre-scoring)",
    )
    component: str = Field(
        description=(
            "REINVENT4 component class name "
            "(e.g. 'QED', 'MolecularWeight', 'HBondDonors')"
        ),
    )
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative weight in score aggregation",
    )
    transform: Optional[TransformConfig] = Field(
        default=None,
        description="Score transform to map raw values to [0, 1]",
    )

    @field_validator("component")
    @classmethod
    def _validate_component(cls, v: str) -> str:
        if v not in VALID_PRESCORING_COMPONENTS:
            raise ValueError(
                f"Unknown pre-scoring component '{v}'. "
                f"Valid: {', '.join(sorted(VALID_PRESCORING_COMPONENTS))}"
            )
        return v


class PreScoringSection(BaseModel):
    """Pre-scoring section of a TTP.

    Fast drug-likeness descriptors (QED, Lipinski properties) computed
    for all molecules.  These contribute to the final aggregated score
    alongside the full scoring objectives.
    """

    model_config = {"extra": "forbid"}

    comment: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    components: list[PreScoringComponent] = Field(
        default_factory=list,
        description="Pre-scoring components to evaluate",
    )

    @field_validator("components")
    @classmethod
    def _validate_unique_names(
        cls, v: list[PreScoringComponent]
    ) -> list[PreScoringComponent]:
        names = [c.name for c in v]
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Duplicate pre-scoring component names: {', '.join(sorted(dupes))}"
            )
        return v


# ---------------------------------------------------------------------------
# Scoring objectives
# ---------------------------------------------------------------------------


class ScoringObjective(BaseModel):
    """A single scoring objective (native REINVENT4 or external REST).

    Native objectives use built-in REINVENT4 components.
    External objectives call REST services (Boltz, PoseBusters).
    """

    model_config = {"extra": "forbid"}

    name: str = Field(description="Display name for this objective")
    type: Literal["native", "external"] = Field(
        description="'native' for REINVENT4 components, 'external' for REST services",
    )

    # --- native fields ---
    component: Optional[str] = Field(
        default=None,
        description="REINVENT4 component class name (required when type='native')",
    )

    # --- external fields ---
    service: Optional[str] = Field(
        default=None,
        description=(
            "External service identifier: 'boltz', 'posebusters', 'admet', "
            "'retrodmta', or 'interaction_profiler'"
        ),
    )
    service_url: Optional[str] = Field(
        default=None,
        description=(
            "REST endpoint URL for the external service. Use the Docker "
            "Compose service name and the '/score' path: "
            "'http://boltz:8080/score' for boltz, "
            "'http://posebusters:8080/score' for posebusters, "
            "'http://admet-predictor:8084/score' for admet, "
            "'http://retrodmta-predictor:8085/score' for retrodmta. "
            "Note the service name is 'boltz' (not 'boltz2'). For known "
            "services this is overridden by the configured URL if it differs."
        ),
    )
    service_params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Service-specific parameters forwarded in the request body. "
            "admet: {'model_set': 'drug'|'agro', 'endpoint': 'logp'} "
            "(one objective per endpoint). "
            "retrodmta: {'target': 'FXA', 'endpoint': 'FXA_pKi', 'window': 'latest', 'method': 'ensemble'}. "
            "interaction_profiler: {'interaction_type': 'hbond'} (counts that interaction "
            "type; needs a structure, so chain after a producer like boltz via depends_on)."
        ),
    )
    depends_on: Optional[str] = Field(
        default=None,
        description=(
            "Name of another objective whose output this objective consumes "
            "(pipeline dependency, e.g. PoseBusters depends on Boltz structures)"
        ),
    )

    # --- common fields ---
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Relative weight in score aggregation",
    )
    transform: Optional[TransformConfig] = Field(
        default=None,
        description="Score transform to map raw values to [0, 1]",
    )
    params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Component-specific parameters forwarded to REINVENT4. "
            "Use plain values -- per-endpoint wrapping is automatic. "
            "Example for TanimotoSimilarity: "
            "{'smiles': ['CCO'], 'radius': 3, 'use_counts': true, "
            "'use_features': false}"
        ),
    )
    comment: Optional[str] = Field(
        default=None,
        description="Human-readable note",
    )

    @model_validator(mode="after")
    def _validate_type_specific_fields(self) -> "ScoringObjective":
        if self.type == "native":
            if not self.component:
                raise ValueError(
                    f"Native objective '{self.name}' requires a 'component' field"
                )
        elif self.type == "external":
            if not self.service:
                raise ValueError(
                    f"External objective '{self.name}' requires a 'service' field"
                )
            if not self.service_url:
                raise ValueError(
                    f"External objective '{self.name}' requires a 'service_url' field"
                )
        return self


class ScoringSection(BaseModel):
    """Scoring section of a TTP.

    Full scoring objectives (affinity, pose quality, SA, etc.) that
    evaluate every molecule.  Both native and external objectives are
    aggregated together with the pre-scoring components.
    """

    model_config = {"extra": "forbid"}

    comment: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    objectives: list[ScoringObjective] = Field(
        description="Scoring objectives to evaluate",
    )
    aggregation: Literal["geometric_mean", "arithmetic_mean"] = Field(
        default="geometric_mean",
        description="Score aggregation method across all components",
    )

    @field_validator("objectives")
    @classmethod
    def _validate_unique_names(
        cls, v: list[ScoringObjective]
    ) -> list[ScoringObjective]:
        names = [o.name for o in v]
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Duplicate scoring objective names: {', '.join(sorted(dupes))}"
            )
        return v

    @model_validator(mode="after")
    def _validate_depends_on(self) -> "ScoringSection":
        objective_names = {o.name for o in self.objectives}
        dep_of: dict[str, str] = {}
        for obj in self.objectives:
            if obj.depends_on and obj.depends_on not in objective_names:
                raise ValueError(
                    f"Objective '{obj.name}' has depends_on='{obj.depends_on}' "
                    f"which is not a defined objective. "
                    f"Available: {', '.join(sorted(objective_names))}"
                )
            if obj.depends_on == obj.name:
                raise ValueError(
                    f"Objective '{obj.name}' cannot depend on itself"
                )
            if obj.depends_on:
                dep_of[obj.name] = obj.depends_on

        # Detect cycles in the depends_on graph (e.g. A->B->A): without a cycle
        # check the pipeline has no valid starting point. Each objective has at
        # most one dependency, so follow each chain and flag any revisit.
        for start in dep_of:
            seen: set[str] = set()
            node: str | None = start
            while node in dep_of:
                if node in seen:
                    chain = " -> ".join([*seen, node])
                    raise ValueError(
                        f"Circular depends_on detected involving objective "
                        f"'{start}' (cycle through: {chain})"
                    )
                seen.add(node)
                node = dep_of[node]
        return self


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class ConstraintConfig(BaseModel):
    """Structural constraint (e.g. toxic SMARTS filter).

    Constraints define structural patterns that molecules must avoid.
    Currently supports ``custom_alerts`` (SMARTS-based substructure
    filters).  Additional constraint-specific fields are accepted via
    ``extra="allow"``.
    """

    model_config = {"extra": "allow"}

    type: str = Field(
        description="Constraint type (e.g. 'custom_alerts')",
    )
    smarts: Optional[list[str]] = Field(
        default=None,
        description="SMARTS patterns for substructure matching",
    )

    @model_validator(mode="after")
    def _validate_smarts_for_alerts(self) -> "ConstraintConfig":
        if self.type == "custom_alerts" and not self.smarts:
            raise ValueError(
                "Constraint type 'custom_alerts' requires a non-empty 'smarts' list"
            )
        return self


# ---------------------------------------------------------------------------
# Top-level TTP
# ---------------------------------------------------------------------------


class TTP(BaseModel):
    """Therapeutic Target Profile -- full multi-property optimisation spec.

    A TTP defines everything needed for a drug design campaign:

    * **target** -- protein sequence, PDB, MSA for structure-based scoring
    * **pre_scoring** -- fast drug-likeness descriptors (QED, Lipinski)
    * **scoring** -- full objectives (Boltz affinity, PoseBusters, SA, …)
    * **constraints** -- structural filters (toxic SMARTS, etc.)

    The ``id`` uniquely identifies the TTP in persistent storage.
    ``version`` is auto-incremented by the TTP store on each update.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(
        description=(
            "Unique identifier -- lowercase alphanumeric with hyphens "
            "or underscores (e.g. 'jak2-oral-lead')"
        ),
    )
    name: str = Field(
        description="Human-readable name for this TTP",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number (auto-incremented on update)",
    )
    target: Optional[TargetConfig] = Field(
        default=None,
        description=(
            "Protein target information. Required when scoring "
            "objectives include Boltz-based services."
        ),
    )
    pre_scoring: Optional[PreScoringSection] = Field(
        default=None,
        description="Fast drug-likeness descriptors (QED, Lipinski)",
    )
    scoring: ScoringSection = Field(
        description="Full scoring objectives (native + external)",
    )
    constraints: Optional[list[ConstraintConfig]] = Field(
        default=None,
        description="Structural constraints (SMARTS filters, etc.)",
    )
    generation: Optional[GenerationSection] = Field(
        default=None,
        description=(
            "Generation campaign configuration (mode, model, batch size, etc.). "
            "None = use existing run_optimization defaults (backward compat)."
        ),
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"Invalid TTP id '{v}'. "
                f"Must be lowercase alphanumeric with hyphens or underscores, "
                f"starting with a letter or digit."
            )
        return v

    @model_validator(mode="after")
    def _validate_target_for_boltz(self) -> "TTP":
        """Require target when Boltz objectives are defined."""
        boltz_objectives = [
            o.name
            for o in self.scoring.objectives
            if o.type == "external" and o.service == "boltz"
        ]
        if boltz_objectives and self.target is None:
            raise ValueError(
                f"Objectives {boltz_objectives} use Boltz scoring which requires "
                f"a 'target' section with at least 'protein_sequence'."
            )
        return self

    @model_validator(mode="after")
    def _validate_boltz_affinity_transform_direction(self) -> "TTP":
        """Warn when pred_value affinity mode is combined with an upward transform.

        affinity_pred_value is log10(IC50/µM) — MORE NEGATIVE = tighter = BETTER.
        An upward sigmoid (low < high, e.g. sigmoid(low=0.3, high=0.8)) rewards
        POSITIVE values (weak binders) instead of negative ones, which is wrong.

        Correct transforms for pred_value mode:
          reverse_sigmoid(low=-2.0, high=0.0)  → score=1.0 at −2 (10 nM), 0.0 at +2 (100 µM)
          sigmoid(low=-2.0, high=0.0)           → score=0.0 at −2, 1.0 at 0  ← also WRONG

        Safe default: use affinity_mode='probability_binary' with sigmoid(low=0.3, high=0.8).
        """
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger("reinvent")

        for obj in self.scoring.objectives:
            sp = obj.service_params or {}
            # service_params may be a dict or Pydantic model
            if hasattr(sp, "get"):
                affinity_mode = sp.get("affinity_mode", "probability_binary")
            else:
                affinity_mode = getattr(sp, "affinity_mode", "probability_binary")
            if affinity_mode != "pred_value":
                continue
            transform = obj.transform
            if transform is None:
                continue
            # transform is a TransformConfig Pydantic model — use attribute access
            t_type = getattr(transform, "type", "") or ""
            t_low = getattr(transform, "low", None)
            t_high = getattr(transform, "high", None)
            # Upward sigmoid (low < high) with pred_value rewards positive (weak) values
            if t_type == "sigmoid" and t_low is not None and t_high is not None:
                if float(t_low) < float(t_high) and float(t_high) >= 0:
                    _log.warning(
                        "⚠️  TTP objective '%s' uses affinity_mode='pred_value' with "
                        "sigmoid(low=%s, high=%s). affinity_pred_value is log10(IC50/µM): "
                        "MORE NEGATIVE = tighter = BETTER. This sigmoid rewards POSITIVE "
                        "(weak-binding) values. Use reverse_sigmoid(low=-2.0, high=0.0) "
                        "to reward tight binders, or switch to affinity_mode='probability_binary' "
                        "with sigmoid(low=0.3, high=0.8).",
                        obj.name, t_low, t_high,
                    )
        return self

    @model_validator(mode="after")
    def _validate_no_name_collisions(self) -> "TTP":
        """Ensure component/objective names are unique across sections."""
        all_names: list[str] = []
        if self.pre_scoring:
            all_names.extend(c.name for c in self.pre_scoring.components)
        all_names.extend(o.name for o in self.scoring.objectives)
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in all_names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"Component/objective names must be unique across pre_scoring "
                f"and scoring sections. Duplicates: {', '.join(sorted(dupes))}"
            )
        return self

    # -- convenience properties -------------------------------------------

    @property
    def has_external_objectives(self) -> bool:
        return any(o.type == "external" for o in self.scoring.objectives)

    @property
    def native_objective_names(self) -> list[str]:
        names: list[str] = []
        if self.pre_scoring:
            names.extend(c.name for c in self.pre_scoring.components)
        names.extend(
            o.name for o in self.scoring.objectives if o.type == "native"
        )
        return names

    @property
    def external_objective_names(self) -> list[str]:
        return [
            o.name
            for o in self.scoring.objectives
            if o.type == "external"
        ]


# ===========================================================================
# Generation contract — block 08 additions
# ---------------------------------------------------------------------------
# All models below are ADDITIVE. The TTP class above gains one new optional
# field (``generation``) which defaults to ``None`` so every existing TTP
# continues to load and validate without changes.
# ===========================================================================

from enum import Enum  # noqa: E402  (after existing imports)


# ---------------------------------------------------------------------------
# SMILES helper validators (module-level so generation.py can import them)
# ---------------------------------------------------------------------------

def _validate_linkinvent_fragment(smiles: str, frag_idx: int) -> list[str]:
    """Return a list of error strings (empty = valid) for one LinkInvent fragment.

    Checks:
    - RDKit can parse the SMILES.
    - Exactly one attachment point atom (atomic num 0) is present.
    - That atom carries a map number (e.g. [*:1], not bare [*]).
    """
    errors: list[str] = []
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        # RDKit unavailable — the structural checks below need it, so skip them
        # rather than report a false "cannot parse" error (matches
        # _validate_seed_smiles, which treats a missing RDKit as "cannot check").
        return errors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [f"Fragment {frag_idx}: cannot parse SMILES {smiles!r}"]

    dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        errors.append(
            f"Fragment {frag_idx}: must have exactly 1 attachment point [*:n], "
            f"found {len(dummy_atoms)}"
        )
    map_nums = [a.GetAtomMapNum() for a in dummy_atoms]
    if any(m == 0 for m in map_nums):
        errors.append(
            f"Fragment {frag_idx}: attachment atom must carry a map number, "
            "e.g. [*:1] not [*]"
        )
    return errors


def _validate_libinvent_scaffold(smiles: str) -> list[str]:
    """Return a list of error strings (empty = valid) for a LibInvent scaffold.

    Checks:
    - RDKit can parse the SMILES.
    - At least one attachment point atom (atomic num 0) is present.
    - All attachment atoms carry map numbers.
    - Map numbers are unique (no duplicates).
    """
    try:
        from rdkit import Chem  # noqa: PLC0415
    except ImportError:
        # RDKit unavailable — skip structural checks rather than report a false
        # parse error (consistent with _validate_seed_smiles / linkinvent).
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [f"Scaffold: cannot parse SMILES {smiles!r}"]

    dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_atoms:
        return [
            "Scaffold: no attachment points [*:n] found — LibInvent needs "
            "at least one R-group position"
        ]

    map_nums = [a.GetAtomMapNum() for a in dummy_atoms]
    errors: list[str] = []
    if any(m == 0 for m in map_nums):
        errors.append(
            "Scaffold: all attachment points must carry map numbers, "
            "e.g. [*:1], [*:2] — not bare [*]"
        )
    if len(set(map_nums)) != len(map_nums):
        errors.append(
            f"Scaffold: duplicate attachment-point map numbers {map_nums}"
        )
    return errors


# ---------------------------------------------------------------------------
# Generation models
# ---------------------------------------------------------------------------

class GenerationMode(str, Enum):
    """Supported REINVENT generation algorithms."""
    reinvent   = "reinvent"
    libinvent  = "libinvent"
    linkinvent = "linkinvent"
    mol2mol    = "mol2mol"


class LinkInventInput(BaseModel):
    """Fragment pair (and optional linker constraints) for LinkInvent.

    Each fragment SMILES must contain exactly one attachment-point atom
    ([*:n]) with a unique map number across the pair.  The SMILES are
    validated by RDKit at model-construction time.
    """

    model_config = {"extra": "forbid"}

    fragments: list[str] = Field(
        description=(
            "Exactly 2+ fragment SMILES, each with one [*:n] attachment point "
            "and distinct map numbers, e.g. ['[*:1]CC', '[*:2]OC']."
        ),
    )
    allowed_atom_types: Optional[list[str]] = Field(
        default=None,
        description="Atom symbols allowed in the generated linker, e.g. ['C','N','O','S'].",
    )
    max_linker_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of heavy atoms in the linker.",
    )
    min_linker_length: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum number of heavy atoms in the linker.",
    )

    @field_validator("fragments")
    @classmethod
    def _validate_fragments(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError(
                "linkinvent requires at least 2 fragment SMILES "
                "(one per side of the linker)."
            )
        errors: list[str] = []
        map_nums_seen: set[int] = set()

        for i, smi in enumerate(v):
            frag_errors = _validate_linkinvent_fragment(smi, i)
            errors.extend(frag_errors)

            # Cross-fragment map-number uniqueness
            try:
                from rdkit import Chem  # noqa: PLC0415
                mol = Chem.MolFromSmiles(smi)
            except ImportError:
                mol = None
            if mol is not None:
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum():
                        mn = atom.GetAtomMapNum()
                        if mn in map_nums_seen:
                            errors.append(
                                f"Fragment {i}: map number :{mn} is already used "
                                "in another fragment — each must be unique"
                            )
                        map_nums_seen.add(mn)

        if errors:
            raise ValueError("\n".join(errors))
        return v


class LibInventInput(BaseModel):
    """Scaffold (and optional R-group constraints) for LibInvent.

    The scaffold SMILES must contain at least one attachment-point atom
    ([*:n]) with a unique map number per position.  Validated by RDKit
    at model-construction time.
    """

    model_config = {"extra": "forbid"}

    scaffold: str = Field(
        description=(
            "Scaffold SMILES with [*:n] attachment points at every R-group "
            "position, e.g. 'c1ccc([*:1])cc1[*:2]'."
        ),
    )
    decoration_positions: Optional[list[int]] = Field(
        default=None,
        description=(
            "Map numbers to decorate (subset of attachment points in scaffold). "
            "None = decorate all positions."
        ),
    )

    @field_validator("scaffold")
    @classmethod
    def _validate_scaffold(cls, v: str) -> str:
        errors = _validate_libinvent_scaffold(v)
        if errors:
            raise ValueError("\n".join(errors))
        return v


class Mol2MolInput(BaseModel):
    """Seed molecules for Mol2Mol lead optimization."""

    model_config = {"extra": "forbid"}

    seed_smiles: list[str] = Field(
        min_length=1,
        max_length=4096,
        description=(
            "One or more valid seed SMILES that Mol2Mol conditions generation on."
        ),
    )

    @field_validator("seed_smiles")
    @classmethod
    def _validate_seed_smiles(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        invalid: list[str] = []
        try:
            from rdkit import Chem  # noqa: PLC0415
        except ImportError:
            Chem = None  # type: ignore[assignment]

        for value in values:
            smiles = (value or "").strip()
            if not smiles or (Chem is not None and Chem.MolFromSmiles(smiles) is None):
                invalid.append(value)
            else:
                cleaned.append(smiles)

        if invalid:
            raise ValueError(f"Invalid Mol2Mol seed SMILES: {invalid}")
        return cleaned


class GenerationSection(BaseModel):
    """Explicit campaign generation contract.

    Attached to a TTP to specify which REINVENT algorithm to use and
    how to configure the generation job.  All fields have safe defaults
    so a minimal ``GenerationSection(mode='reinvent')`` is valid.
    """

    model_config = {"extra": "forbid"}

    mode: GenerationMode = Field(
        default=GenerationMode.reinvent,
        description=(
            "Generation algorithm: 'reinvent' | 'libinvent' | 'linkinvent' | "
            "'mol2mol'."
        ),
    )
    model: str = Field(
        default=".reinvent",
        description=(
            "Prior/model registry key such as '.reinvent', '.libinvent', "
            "'.linkinvent', or '.m2m_medium'; alternatively a host-specific "
            "model path. Resolved via the model manifest."
        ),
    )
    batch_size: int = Field(
        default=128, ge=1, le=4096,
        description="Number of molecules generated per step.",
    )
    n_steps: int = Field(
        default=100, ge=1,
        description="Total number of RL steps.",
    )
    checkpoint_every_n: int = Field(
        default=50, ge=1,
        description="Save a checkpoint every N steps.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility.  None = non-deterministic.",
    )
    device: str = Field(
        default="cuda",
        description="Compute device: 'cuda' or 'cpu'.",
    )

    # Mode-specific inputs — validated against mode before job launch
    linkinvent_input: Optional[LinkInventInput] = Field(
        default=None,
        description="Required for mode='linkinvent'.",
    )
    libinvent_input: Optional[LibInventInput] = Field(
        default=None,
        description="Required for mode='libinvent'.",
    )
    mol2mol_input: Optional[Mol2MolInput] = Field(
        default=None,
        description="Required for mode='mol2mol'.",
    )

    # Design-context artifact references
    target_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Campaign-level target record artifact IDs.",
    )
    prepared_structure_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Prepared structure artifact IDs (from structure preparator).",
    )
    site_artifact_ids: list[str] = Field(
        default_factory=list,
        description="Binding-site or interface artifact IDs.",
    )

    @model_validator(mode="after")
    def _check_mode_inputs(self) -> "GenerationSection":
        if self.mode == GenerationMode.linkinvent:
            if self.linkinvent_input is None:
                raise ValueError(
                    "linkinvent mode requires linkinvent_input "
                    "(fragment SMILES with attachment points)."
                )
        if self.mode == GenerationMode.libinvent:
            if self.libinvent_input is None:
                raise ValueError(
                    "libinvent mode requires libinvent_input "
                    "(scaffold SMILES with [*:n] attachment points)."
                )
        if self.mode == GenerationMode.mol2mol:
            if self.mol2mol_input is None:
                raise ValueError(
                    "mol2mol mode requires mol2mol_input "
                    "(one or more seed SMILES)."
                )
        return self
