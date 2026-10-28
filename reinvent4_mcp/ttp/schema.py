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


TEMPLATE_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")


class TemplateConfig(BaseModel):
    """A structure Boltz-2 conditions the cofold on (its YAML ``templates:``)."""

    model_config = {"extra": "forbid"}

    path: str = Field(
        description=(
            "Template structure (.cif/.mmcif or .pdb/.ent), relative to the "
            "shared volume or absolute. Boltz names templates after the file "
            "stem, so stems must be unique."
        ),
    )
    chain_id: Optional[str] = Field(
        default=None,
        description="Protein chain of the template file to use; None lets Boltz pick",
    )
    force: bool = Field(
        default=False,
        description="Hold the cofold to the template with a potential; needs threshold",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Distance in Angstrom the forced template may deviate by",
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        v = v.strip()
        if not v.lower().endswith(TEMPLATE_SUFFIXES):
            raise ValueError(
                f"Template '{v}' must be a {', '.join(TEMPLATE_SUFFIXES)} file"
            )
        return v

    @field_validator("chain_id")
    @classmethod
    def _validate_chain(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return v.strip() or None

    @model_validator(mode="after")
    def _validate_force(self) -> "TemplateConfig":
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError("Template threshold must be positive (Angstrom)")
        if self.force and self.threshold is None:
            raise ValueError("A forced template needs a threshold (Angstrom)")
        return self


class TargetConfig(BaseModel):
    """Protein target information for structure-based scoring.

    ``protein_sequence`` is required for Boltz cofolding.  Other fields
    supply additional structural context.
    """

    model_config = {"extra": "forbid"}

    protein_sequence: str = Field(
        description=(
            "Amino-acid sequence Boltz folds. Prefer the construct (the chain "
            "of a crystal structure, or the domain) to the full-length "
            "protein: Boltz cost grows steeply with length. Must match the "
            "query sequence of msa_path, else Boltz folds the MSA's query."
        ),
    )
    pdb_id: Optional[str] = Field(
        default=None,
        description="PDB identifier (e.g. '3KRR'); provenance only, no service reads it",
    )
    msa_path: Optional[str] = Field(
        default=None,
        description="Pre-computed MSA (.a3m), relative to the shared volume or absolute; Boltz uses it instead of the MSA server",
    )
    protein_pdb_path: Optional[str] = Field(
        default=None,
        description=(
            "Protein structure PoseBusters checks Boltz poses against, instead "
            "of the Boltz complex itself; it must be in the frame of the "
            "predicted poses and readable by the PoseBusters service. Boltz "
            "does not read it."
        ),
    )
    templates: list[TemplateConfig] = Field(
        default_factory=list,
        description=(
            "Structures Boltz-2 uses as templates for the cofold, e.g. a "
            "cleaned single-chain crystal structure of the target."
        ),
    )

    @field_validator("templates")
    @classmethod
    def _validate_templates(cls, v: list[TemplateConfig]) -> list[TemplateConfig]:
        stems = [re.split(r"[\\/]", t.path)[-1].rsplit(".", 1)[0] for t in v]
        dupes = sorted({s for s in stems if stems.count(s) > 1})
        if dupes:
            raise ValueError(
                f"Template file names must be unique (Boltz names templates by file name): {', '.join(dupes)}"
            )
        return v

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
            "admet: {'endpoint': 'logp'} (one objective per endpoint). "
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
            "{'smiles': ['CCO'], 'radius': 3, 'use_features': true}"
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
