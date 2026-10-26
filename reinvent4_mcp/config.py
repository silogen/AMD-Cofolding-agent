"""Process configuration loading and validation.

Loads the reinvent4_mcp_config.yaml that defines operational parameters
(batch_size, service URLs, storage paths, etc.) separate from the TTP
(scientific objectives).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_VALID_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})


def _expand_path(v: str) -> str:
    """Expand ``~`` and environment variables in a path string."""
    return str(Path(os.path.expandvars(v)).expanduser())


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class DiversityFilterConfig(BaseModel):
    """Diversity filter settings for RL optimization.

    Controls how the RL loop penalises repeated scaffolds to encourage
    exploration of chemical space.
    """

    model_config = {"extra": "forbid"}

    type: str = Field(
        default="IdenticalMurckoScaffold",
        description=(
            "Filter type: IdenticalMurckoScaffold, "
            "IdenticalTopologicalScaffold, ScaffoldSimilarity, NoFilter"
        ),
    )
    bucket_size: int = Field(
        default=25, ge=1, description="Max molecules per scaffold bucket"
    )
    minscore: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum total score for a molecule to enter a bucket",
    )


class ReinventConfig(BaseModel):
    """REINVENT4 RL optimization defaults.

    These values apply to every optimization run unless overridden by
    the TTP or explicit ``run_optimization`` parameters.
    """

    model_config = {"extra": "forbid"}

    device: str = Field(
        default="cuda:0",
        description="Torch device string (cuda:N, cpu, mps)",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Molecules generated per RL step; controls Boltz cost",
    )
    sigma: int = Field(
        default=128, ge=1, description="RL reward sharpness (DAP sigma)"
    )
    learning_rate: float = Field(
        default=0.0001,
        gt=0,
        le=1.0,
        description="Optimizer learning rate",
    )
    unique_sequences: bool = Field(
        default=True,
        description="Deduplicate SMILES within each batch",
    )
    randomize_smiles: bool = Field(
        default=True,
        description="Randomize SMILES representation during generation",
    )
    default_model: str = Field(
        default=".reinvent",
        description="Default prior model registry key (e.g. .reinvent) or file path",
    )
    diversity_filter: DiversityFilterConfig = Field(
        default_factory=DiversityFilterConfig,
    )


class PreScoringComponentToggles(BaseModel):
    """Individual pre-scoring component on/off switches.

    Each toggle maps to a fast, RDKit-based REINVENT4 scoring component
    that runs on every molecule before expensive external scoring.
    """

    model_config = {"extra": "forbid"}

    qed: bool = Field(default=True, description="Quantitative Estimate of Drug-likeness")
    molecular_weight: bool = Field(default=True, description="Molecular weight (Da)")
    hbond_donors: bool = Field(default=True, description="Hydrogen bond donor count")
    hbond_acceptors: bool = Field(default=True, description="Hydrogen bond acceptor count")
    slogp: bool = Field(default=True, description="Wildman-Crippen LogP estimate")
    num_rot_bond: bool = Field(default=True, description="Rotatable bond count")


class PreScoringConfig(BaseModel):
    """Pre-scoring stage configuration.

    Fast property descriptors computed for all molecules in every RL step.
    These contribute to the final aggregated score alongside the full
    scoring objectives (Boltz, PoseBusters, etc.).
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(
        default=True,
        description="Enable the pre-scoring stage",
    )
    components: PreScoringComponentToggles = Field(
        default_factory=PreScoringComponentToggles,
    )


class BoltzConfig(BaseModel):
    """Boltz2 cofolding + affinity prediction service configuration."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable Boltz2 scoring")
    service_url: str = Field(
        default="http://boltz:8080/score",
        description="Boltz2 REST endpoint URL",
    )
    recycling_steps: int = Field(
        default=3,
        ge=1,
        description="Structure prediction recycling iterations",
    )
    diffusion_samples: int = Field(
        default=1,
        ge=1,
        description="Number of diffusion samples per molecule",
    )
    sampling_steps_affinity: int = Field(
        default=200,
        ge=1,
        description="Diffusion sampling steps for affinity prediction",
    )
    affinity_mode: Literal["pred_value", "probability_binary"] = Field(
        default="pred_value",
        description=(
            "Boltz2 affinity output mode.\n"
            "pred_value: log10(IC50/µM) — MORE NEGATIVE = tighter = BETTER. "
            "Use reverse_sigmoid(low=-2.0, high=0.0) in TTP transforms so that "
            "score→1.0 at −2 (10 nM) and score→0.0 at +2 (100 µM). "
            "Do NOT use upward sigmoid(low=0.3, high=0.8) — that rewards weak binders.\n"
            "probability_binary: P(IC50 < 1 µM) ∈ [0,1] — higher = better. "
            "Use sigmoid(low=0.3, high=0.8)."
        ),
    )
    cache_dir: str = Field(
        default="/data/boltz/cache",
        description="Directory for Boltz model weights and MSA cache",
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MSA server instead of pre-computed MSAs",
    )
    msa_server_url: str = Field(
        default="https://api.colabfold.com",
        description="MSA server URL (when use_msa_server is true)",
    )
    auto_msa_fallback: bool = Field(
        default=True,
        description=(
            "When a cofold has no explicit or cached MSA, automatically enable "
            "the MSA server to generate one instead of failing."
        ),
    )
    msa_free_fallback: bool = Field(
        default=True,
        description=(
            "Last-resort fallback: when no local/cached MSA exists AND the MSA "
            "server is unreachable, run Boltz in single-sequence (msa: empty) "
            "mode so a structure is still produced. Confidence/affinity are "
            "reduced; a local .a3m is always preferred."
        ),
    )
    msa_cache_dir: str = Field(
        default="/shared/msa_cache",
        description=(
            "Directory (on the shared volume) where the Boltz service "
            "persists generated MSAs; looked up by protein-sequence hash so "
            "the mapper can reuse a cached alignment instead of regenerating. "
            "Must match the Boltz service's MSA_CACHE_DIR."
        ),
    )

    @field_validator("cache_dir")
    @classmethod
    def _expand_cache_dir(cls, v: str) -> str:
        return _expand_path(v)


class PoseBustersConfig(BaseModel):
    """PoseBusters pose validation service configuration."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(
        default=True, description="Enable PoseBusters validation"
    )
    service_url: str = Field(
        default="http://posebusters:8080/score",
        description="PoseBusters REST endpoint URL",
    )
    mode: Literal["dock", "redock", "mol", "gen"] = Field(
        default="dock",
        description=(
            "Validation mode: dock (predicted ligand + protein), "
            "redock (+ crystal reference), mol (ligand-only), "
            "gen (generated conformers)"
        ),
    )


class AdmetConfig(BaseModel):
    """Selectable pharmaceutical/agrochemical ADMET scoring service."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable ADMET predictor scoring")
    service_url: str = Field(
        default="http://admet-predictor:8084/score",
        description="ADMET predictor REST endpoint URL",
    )
    model_set: Literal["drug", "agro"] = Field(
        default="drug",
        description="Default ADMET model stack when a TTP objective omits it",
    )


class RetroDMTAConfig(BaseModel):
    """RetroDMTA project-endpoint scoring service configuration.

    Per-target, per-temporal-window models. Use one TTP objective per endpoint
    (``service_params: {"target": "FXA", "endpoint": "FXA_pKi"}``).
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable RetroDMTA project-endpoint scoring")
    service_url: str = Field(
        default="http://retrodmta-predictor:8085/score",
        description="RetroDMTA predictor REST endpoint URL",
    )
    method: Literal["ensemble", "xgboost", "chemprop", "chemeleon"] = Field(
        default="ensemble",
        description="Default model/method when a TTP objective omits it",
    )
    window: str = Field(
        default="latest",
        description="Default temporal window (index or 'latest') when a TTP objective omits it",
    )


class InteractionProfilerConfig(BaseModel):
    """Interaction-profiler service configuration.

    Dual role: (1) direct out-of-band analysis for the agent (medchem fragments
    from SMILES; typed protein-ligand interactions from a structure) via the
    ``profile_interactions`` MCP tool, and (2) an optional RL objective that
    counts interactions of a chosen type (needs a structure, typically a Boltz
    cofold passed via ``depends_on``). ``/score`` returns the raw count; the TTP
    transform owns the direction.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable interaction-profiler")
    service_url: str = Field(
        default="http://interaction-profiler:8080/score",
        description="Interaction-profiler REST endpoint URL",
    )
    interaction_type: str = Field(
        default="all",
        description="Default interaction family to count in /score (e.g. hbond, ionic, all)",
    )
    radius: float = Field(default=8.0, description="Default pocket crop radius (A)")
    ph: float = Field(default=7.4, description="Default protonation pH")


class SynthesisPlannerConfig(BaseModel):
    """AiZynthFinder synthesis-planner service configuration.

    The /score endpoint returns a [0,1] synthesizability score per molecule
    based on AiZynthFinder MCTS retrosynthesis results. Use as an RL objective
    to guide generation toward synthetically accessible molecules using actual
    retrosynthesis — not the heuristic SAScore.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable synthesis-planner scoring")
    service_url: str = Field(
        default="http://synthesis-planner:8086/score",
        description="AiZynthFinder /score REST endpoint URL",
    )
    iteration_limit: int = Field(
        default=25,
        description="MCTS iterations per molecule (25 = fast RL mode, ~0.3-0.8s/mol)",
    )
    score_fn: str = Field(
        default="solved_steps",
        description=(
            "Scoring function: solved_steps (1.0=≤2 steps, 0.7=≤4 steps, 0.3=any route, "
            "0.0=no route) | n_routes (sigmoid on route count) | best_score (raw MCTS score)"
        ),
    )


class ScoringConfig(BaseModel):
    """Scoring pipeline configuration.

    Controls the aggregation method, pre-scoring component toggles, and
    external service (Boltz, PoseBusters, ADMET, RetroDMTA, interaction-profiler)
    connection details.
    """

    model_config = {"extra": "forbid"}

    aggregation: Literal["geometric_mean", "arithmetic_mean"] = Field(
        default="geometric_mean",
        description="Score aggregation method across components",
    )
    external_failure_score: Optional[float] = Field(
        default=None,
        description=(
            "Score to emit when an external service call fails or returns a "
            "null score, instead of NaN. Set to 0.0 to penalize failures in "
            "RL (so a broken objective cannot masquerade as a perfect score). "
            "Default None preserves the all-NaN behavior."
        ),
    )
    pre_scoring: PreScoringConfig = Field(
        default_factory=PreScoringConfig,
    )
    boltz: BoltzConfig = Field(default_factory=BoltzConfig)
    posebusters: PoseBustersConfig = Field(default_factory=PoseBustersConfig)
    admet: AdmetConfig = Field(default_factory=AdmetConfig)
    retrodmta: RetroDMTAConfig = Field(default_factory=RetroDMTAConfig)
    interaction_profiler: InteractionProfilerConfig = Field(
        default_factory=InteractionProfilerConfig
    )
    synthesis_planner: SynthesisPlannerConfig = Field(
        default_factory=SynthesisPlannerConfig
    )


class JobsConfig(BaseModel):
    """Background job management settings."""

    model_config = {"extra": "forbid"}

    working_dir: str = Field(
        default="/data/reinvent4_mcp/jobs",
        description="Root directory for job working directories",
    )
    max_concurrent_jobs: int = Field(
        default=2,
        ge=1,
        description="Maximum simultaneous RL/TL jobs",
    )
    checkpoint_on_stop: bool = Field(
        default=True,
        description="Ask a stopping job to save a checkpoint before it exits, "
                    "instead of terminating it straight away",
    )
    stop_grace_seconds: float = Field(
        default=900.0,
        gt=0,
        description="How long to let a stopping job finish its checkpoint. "
                    "REINVENT can only act on the stop between scoring calls, "
                    "so this has to exceed one RL step -- a step that folds "
                    "every molecule takes several minutes",
    )

    @field_validator("working_dir")
    @classmethod
    def _expand_working_dir(cls, v: str) -> str:
        return _expand_path(v)


class TTPStorageConfig(BaseModel):
    """TTP (Therapeutic Target Profile) persistence settings."""

    model_config = {"extra": "forbid"}

    storage_dir: str = Field(
        default="~/.reinvent4_mcp/ttps",
        description="Directory for versioned TTP JSON files",
    )

    @field_validator("storage_dir")
    @classmethod
    def _expand_storage_dir(cls, v: str) -> str:
        return _expand_path(v)


class LoggingConfig(BaseModel):
    """Logging and TensorBoard configuration."""

    model_config = {"extra": "forbid"}

    level: str = Field(
        default="info",
        description="Log level: debug, info, warning, error, critical",
    )
    tb_logdir: str = Field(
        default="/data/reinvent4_mcp/tb_logs",
        description="TensorBoard log directory",
    )
    log_dir: str = Field(
        default="/data/reinvent4_mcp/logs",
        description="Application log file directory",
    )

    @field_validator("level")
    @classmethod
    def _normalise_and_validate_level(cls, v: str) -> str:
        normalised = v.lower()
        if normalised not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level '{v}'. "
                f"Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        return normalised

    @field_validator("tb_logdir", "log_dir")
    @classmethod
    def _expand_log_dirs(cls, v: str) -> str:
        return _expand_path(v)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

# Toggle name -> REINVENT4 component class name
_PRE_SCORING_COMPONENT_MAP: dict[str, str] = {
    "qed": "QED",
    "molecular_weight": "MolecularWeight",
    "hbond_donors": "HBondDonors",
    "hbond_acceptors": "HBondAcceptors",
    "slogp": "SlogP",
    "num_rot_bond": "NumRotBond",
}


class RunnerConfig(BaseModel):
    """Where to send RL/TL jobs and in-process scoring.

    Disabled by default, so an unconfigured deployment keeps running REINVENT4
    inside the MCP process exactly as before. Enabling it moves the GPU work to
    ``reinvent-runner`` and leaves this process needing neither REINVENT4 nor a
    GPU.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(
        default=False,
        description="Send jobs and scoring to reinvent-runner over HTTP "
                    "instead of running them in this process",
    )
    base_url: str = Field(
        default="http://dd-agentic-reinvent-runner:8086",
        description="Base URL of the reinvent-runner service",
    )
    timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="HTTP timeout. Job calls return as soon as the subprocess "
                    "is launched, but in-process scoring can be slow",
    )


class ProcessConfig(BaseModel):
    """Top-level process configuration for the Drug Discovery MCP server.

    Loaded from ``reinvent4_mcp_config.yaml`` at startup.  Provides
    system-wide defaults that can be overridden per-TTP for scientific
    parameters (e.g. batch_size, aggregation method).
    """

    model_config = {"extra": "forbid"}

    reinvent: ReinventConfig = Field(default_factory=ReinventConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    ttp: TTPStorageConfig = Field(default_factory=TTPStorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def ensure_directories(self) -> None:
        """Create storage, job, and log directories if they don't exist."""

        for dir_path in (
            self.jobs.working_dir,
            self.ttp.storage_dir,
            self.logging.log_dir,
            self.logging.tb_logdir,
        ):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @property
    def enabled_pre_scoring_components(self) -> list[str]:
        """Return REINVENT4 component names for all enabled pre-scoring toggles."""

        if not self.scoring.pre_scoring.enabled:
            return []

        toggles = self.scoring.pre_scoring.components
        return [
            reinvent_name
            for toggle_key, reinvent_name in _PRE_SCORING_COMPONENT_MAP.items()
            if getattr(toggles, toggle_key)
        ]

    @property
    def enabled_external_services(self) -> dict[str, str]:
        """Return ``{service_name: url}`` for all enabled external services."""

        services: dict[str, str] = {}
        if self.scoring.boltz.enabled:
            services["boltz"] = self.scoring.boltz.service_url
        if self.scoring.posebusters.enabled:
            services["posebusters"] = self.scoring.posebusters.service_url
        if self.scoring.admet.enabled:
            services["admet"] = self.scoring.admet.service_url
        if self.scoring.retrodmta.enabled:
            services["retrodmta"] = self.scoring.retrodmta.service_url
        if self.scoring.interaction_profiler.enabled:
            services["interaction_profiler"] = self.scoring.interaction_profiler.service_url
        if self.scoring.synthesis_planner.enabled:
            services["synthesis_planner"] = self.scoring.synthesis_planner.service_url
        return services


# ---------------------------------------------------------------------------
# Loading + singleton
# ---------------------------------------------------------------------------

_CONFIG_ENV_VAR = "REINVENT4_MCP_CONFIG"
_DEFAULT_CONFIG_PATHS = [
    Path("reinvent4_mcp_config.yaml"),
    Path("reinvent4_mcp_config.yml"),
    Path.home() / ".reinvent4_mcp" / "config.yaml",
]

_process_config: ProcessConfig | None = None


def load_config(config_path: Optional[str] = None) -> ProcessConfig:
    """Load process configuration from a YAML file.

    Resolution order:

    1. Explicit *config_path* argument
    2. ``REINVENT4_MCP_CONFIG`` environment variable
    3. ``reinvent4_mcp_config.yaml`` in cwd
    4. ``reinvent4_mcp_config.yml`` in cwd
    5. ``~/.reinvent4_mcp/config.yaml``
    6. Fall back to defaults (no file needed)

    Raises:
        FileNotFoundError: if an explicit path or env-var path does not exist.
        pydantic.ValidationError: on schema violations (unknown keys,
            out-of-range values, invalid enum choices).
    """

    path: Optional[Path] = None

    if config_path:
        path = Path(config_path)
    elif _CONFIG_ENV_VAR in os.environ:
        path = Path(os.environ[_CONFIG_ENV_VAR])
    else:
        for candidate in _DEFAULT_CONFIG_PATHS:
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                path = resolved
                break

    if path is not None:
        resolved_path = path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {resolved_path} "
                f"(resolved from '{path}')"
            )
        logger.info("Loading config from %s", resolved_path)
        with open(resolved_path) as fh:
            raw = yaml.safe_load(fh) or {}
        return ProcessConfig(**raw)

    logger.info("No config file found, using defaults")
    return ProcessConfig()


def get_config(config_path: Optional[str] = None) -> ProcessConfig:
    """Return the singleton ``ProcessConfig``, loading it on first call.

    Subsequent calls return the cached instance.  Pass *config_path* only
    on the first call (e.g. from the server entry point).
    """

    global _process_config
    if _process_config is None:
        _process_config = load_config(config_path)
    return _process_config


def reset_config() -> None:
    """Clear the cached config singleton (useful in tests)."""

    global _process_config
    _process_config = None
