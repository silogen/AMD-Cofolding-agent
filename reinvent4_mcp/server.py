"""Drug Discovery MCP Server entry point.

Exposes REINVENT4 drug design capabilities via the Model Context Protocol
using stdio transport. Provides tools for scoring molecules, discovering
available components/models, and (in later phases) running RL optimization.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from reinvent4_mcp import __version__
from reinvent4_mcp.config import get_config
from reinvent4_mcp.utils import ensure_reinvent4_importable

logger = logging.getLogger("reinvent4_mcp")

mcp = FastMCP(
    "DrugDiscovery",
    instructions=(
        "MCP server for de novo drug design powered by REINVENT4. "
        "Score molecules, discover scoring components and generative models, "
        "and run multi-property optimization campaigns."
    ),
)


# ---------------------------------------------------------------------------
# Tool: list_scoring_components
# ---------------------------------------------------------------------------

@mcp.tool()
def list_scoring_components() -> str:
    """List all available native REINVENT4 scoring components.

    Returns component names, types (scorer/filter/penalty), and their
    parameter schemas. Use this to discover what components can be used
    in score_molecules or TTP definitions.

    Common components include: QED, MolecularWeight, HBondDonors,
    HBondAcceptors, SlogP, NumRotBond, TPSA, CustomAlerts, SAScore,
    TanimotoSimilarity, and many more.
    """

    from reinvent4_mcp.tools.discovery import list_scoring_components as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_available_models
# ---------------------------------------------------------------------------

@mcp.tool()
def list_available_models() -> str:
    """List all registered generative model priors and their availability.

    Returns model registry keys (e.g. '.reinvent', '.libinvent'),
    expected file paths, whether the prior file exists on disk, and
    model type descriptions. Use this to check which models are
    available before launching optimization or transfer learning.
    """

    from reinvent4_mcp.tools.discovery import list_available_models as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_transforms
# ---------------------------------------------------------------------------

@mcp.tool()
def list_transforms() -> str:
    """List all available score transform types and their parameters.

    Transforms map raw component scores to the 0-1 range for aggregation.
    Common transforms: sigmoid, reverse_sigmoid, double_sigmoid, step,
    right_step, left_step.
    """

    from reinvent4_mcp.tools.discovery import list_transforms as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: list_external_services
# ---------------------------------------------------------------------------

@mcp.tool()
def list_external_services() -> str:
    """Check health and capabilities of configured external services.

    For each enabled service (Boltz, PoseBusters, ADMET, RetroDMTA) sends a
    GET /health and reports the URL and status ('ok'/'unhealthy'/'unreachable').

    For the multi-endpoint predictors it also reports the selector the caller
    MUST supply so scoring does not silently fall back to a default:
      * admet     -> 'endpoints' (pick one via endpoint / predict_admet)
      * retrodmta -> 'targets'   (pick one via target / predict_retrodmta)

    Call this before authoring a TTP external objective or using
    predict_admet / predict_retrodmta so you use valid endpoint/target names.
    """

    from reinvent4_mcp.tools.external import list_external_services as _impl

    return json.dumps(_impl(), indent=2)


# ---------------------------------------------------------------------------
# Tool: predict_admet
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_admet(
    smiles: Annotated[
        list[str],
        Field(description="SMILES strings to predict ADMET properties for"),
    ],
    endpoints: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Which ADMET endpoints to return (subset of the service's "
                "endpoint list, e.g. ['logp', 'caco2_permeability', "
                "'aqueous_solubility']). Omit to return all endpoints. Call "
                "list_external_services to see the valid endpoint names."
            ),
        ),
    ] = None,
) -> str:
    """Predict ADMET properties for molecules directly (out-of-band, not RL).

    Use this to answer property questions ("what is the permeability/logP/
    solubility of these molecules?"). Returns per-molecule mean/std for each
    requested endpoint plus the applicability-domain pass flag. For RL scoring,
    use a TTP external objective with service='admet' and
    service_params={'endpoint': '<name>'} instead.
    """

    from reinvent4_mcp.tools.external import predict_admet as _impl

    return json.dumps(_impl(smiles, endpoints), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: predict_retrodmta
# ---------------------------------------------------------------------------

@mcp.tool()
def predict_retrodmta(
    smiles: Annotated[
        list[str],
        Field(description="SMILES strings to predict RetroDMTA endpoints for"),
    ],
    target: Annotated[
        str,
        Field(
            description=(
                "Project target to predict for (required), e.g. 'FXA'. Call "
                "list_external_services to see the valid target names."
            ),
        ),
    ],
    window: Annotated[
        str,
        Field(default="latest", description="Temporal window index or 'latest'"),
    ] = "latest",
    method: Annotated[
        str,
        Field(
            default="ensemble",
            description="Model: ensemble | xgboost | chemprop | chemeleon",
        ),
    ] = "ensemble",
) -> str:
    """Predict RetroDMTA project endpoints for a target directly (out-of-band).

    ``target`` is required; without it the service cannot know which model to
    use. Returns per-molecule per-endpoint mean/std for that target. For RL
    scoring, use a TTP external objective with service='retrodmta' and
    service_params={'target': '<name>'} instead.
    """

    from reinvent4_mcp.tools.external import predict_retrodmta as _impl

    return json.dumps(
        _impl(smiles, target, window, method), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: profile_interactions
# ---------------------------------------------------------------------------

@mcp.tool()
def profile_interactions(
    smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Ligand SMILES to profile. SMILES-only returns the medchem "
                "taxonomy (named functional-group / ring / scaffold fragments). "
                "Pair index-wise with structure_paths to also get interactions."
            ),
        ),
    ] = None,
    structure_paths: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Protein-ligand structure files (PDB/mmCIF), resolved under the "
                "service shared dir. Yields typed interactions (H-bond, ionic, "
                "hydrophobic, pi-stacking, ...) with residues/atoms/geometry. "
                "Index-aligned with smiles when both are given."
            ),
        ),
    ] = None,
) -> str:
    """Profile medchem fragments and/or protein-ligand interactions (out-of-band).

    Use this when the user wants to understand what a ligand IS (fragments, from
    SMILES) or what it's DOING in a pocket (typed interactions, from a structure /
    cofold). Not the RL loop: to reward interactions during optimization, add a
    TTP external objective with service='interaction_profiler' and
    service_params={'interaction_type': 'hbond'} (chained after a structure
    producer like Boltz via depends_on).
    """

    from reinvent4_mcp.tools.external import profile_interactions as _impl

    return json.dumps(
        _impl(smiles, structure_paths), indent=2, default=_json_default
    )


# ---------------------------------------------------------------------------
# Tool: score_molecules
# ---------------------------------------------------------------------------

@mcp.tool()
def score_molecules(
    smiles: Annotated[
        list[str],
        Field(description="List of SMILES strings to score"),
    ],
    components: Annotated[
        list[dict[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Scoring components to evaluate (direct mode). Each dict must "
                "have 'type' (e.g. 'QED', 'MolecularWeight'). Optional keys: "
                "'name', 'weight' (float, default 1.0), 'transform' (dict with "
                "'type' and params), 'params' (dict of component-specific "
                "settings, use plain values -- per-endpoint wrapping is "
                "automatic, e.g. {'smiles': ['CCO'], 'radius': 3}). "
                "Omit when using ttp_id or ttp."
            ),
        ),
    ] = None,
    aggregation: Annotated[
        str,
        Field(
            default="geometric_mean",
            description=(
                "Aggregation method: 'geometric_mean' or 'arithmetic_mean'. "
                "Ignored in TTP mode (uses the TTP's aggregation setting)."
            ),
        ),
    ] = "geometric_mean",
    ttp_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "ID of a stored TTP to use for scoring. "
                "When external services (Boltz, PoseBusters) are healthy, "
                "all objectives are scored with pipeline artifact passing. "
                "Otherwise only native components are scored."
            ),
        ),
    ] = None,
    ttp: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description=(
                "Inline TTP definition (dict). Alternative to ttp_id. "
                "Use create_ttp to validate and persist TTPs."
            ),
        ),
    ] = None,
) -> str:
    """Score a batch of molecules using REINVENT4 scoring components.

    Two modes:
    1. Direct: pass 'components' list (e.g. [{"type": "QED", "weight": 0.5}])
    2. TTP-based: pass 'ttp_id' (stored) or 'ttp' (inline TTP dict).
       When external services are healthy, all objectives (including Boltz
       affinity and PoseBusters pose quality) are scored with pipeline
       artifact passing. Otherwise only native components are scored.

    Returns per-molecule component breakdowns (raw + transformed),
    summary statistics, and scoring_mode ('full' or 'native_only').

    In TTP mode every external objective runs through the same
    ExternalRESTScorer class, so read 'mapper_summary.external_components'
    (and each 'summary.per_component[name].source') to see which service,
    URL and selector -- e.g. admet endpoint, boltz affinity_mode -- produced
    a given score.

    Example (direct mode):
    [
      {"type": "QED", "weight": 0.5},
      {"type": "MolecularWeight", "weight": 0.3,
       "transform": {"type": "double_sigmoid", "high": 500, "low": 200,
                      "coef_div": 500, "coef_si": 20, "coef_se": 20}}
    ]
    """

    from reinvent4_mcp.tools.scoring import score_molecules as _impl

    ttp_dict = _ensure_dict(ttp) if ttp is not None else None
    result = _impl(smiles, components, aggregation, ttp_id=ttp_id, ttp=ttp_dict)
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: create_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def create_ttp(
    ttp: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Full TTP definition (JSON object). No extra fields allowed "
                "at any level -- unknown keys cause validation errors.\n"
                "\n"
                "Top-level fields:\n"
                "  id (str, required): lowercase slug, e.g. 'jak2-lead-opt'\n"
                "  name (str, required): human-readable name\n"
                "  scoring (object, required): see below\n"
                "  pre_scoring (object, optional): fast drug-likeness descriptors\n"
                "  target (object, optional): protein info for Boltz scoring\n"
                "  constraints (list, optional): structural filters\n"
                "\n"
                "IMPORTANT: 'name' must be unique across ALL pre_scoring\n"
                "components and scoring objectives combined.\n"
                "\n"
                "pre_scoring structure:\n"
                "  {components: [{name, component, weight?, transform?}, ...]}\n"
                "  'component' must be one of: QED, MolecularWeight, HBondDonors,\n"
                "  HBondAcceptors, SlogP, NumRotBond.\n"
                "\n"
                "scoring structure:\n"
                "  {aggregation?: 'geometric_mean'|'arithmetic_mean',\n"
                "   objectives: [{name, type, component?, weight?, transform?,\n"
                "                 params?, ...}]}\n"
                "  'type' MUST be exactly 'native' or 'external'.\n"
                "  Native objectives require 'component' (REINVENT4 class name).\n"
                "  Native objectives may include 'params' (dict) for component-\n"
                "  specific settings (e.g. reference SMILES for TanimotoSimilarity).\n"
                "  External objectives require 'service' and 'service_url'.\n"
                "\n"
                "constraints structure (list of objects):\n"
                "  [{type: 'custom_alerts', smarts: ['[pattern]', ...]}, ...]\n"
                "\n"
                "transform structure (on any component/objective):\n"
                "  {type: 'sigmoid'|'reverse_sigmoid'|'double_sigmoid'|'step'|\n"
                "   'right_step'|'left_step'|'value_mapping'|'exponential_decay',\n"
                "   ...type-specific params (low, high, k, etc.)}\n"
            ),
        ),
    ],
) -> str:
    """Create and persist a new Therapeutic Target Profile (TTP).

    A TTP defines the full multi-property optimization objective:
    drug-likeness descriptors (pre-scoring), scoring objectives
    (native REINVENT4 components or external REST services), and
    structural constraints.

    Example TTP with QED and SA score objectives:

        {
          "id": "lead-opt-example",
          "name": "Lead Optimization (QED + SA)",
          "scoring": {
            "aggregation": "geometric_mean",
            "objectives": [
              {
                "name": "QED Score",
                "type": "native",
                "component": "QED",
                "weight": 1.0
              },
              {
                "name": "SA Score",
                "type": "native",
                "component": "SAScore",
                "weight": 0.8,
                "transform": {
                  "type": "reverse_sigmoid",
                  "low": 1.0, "high": 4.0, "k": 0.4
                }
              }
            ]
          }
        }

    Note: pre_scoring components and scoring objectives share the same
    name namespace -- every 'name' must be unique across both sections.

    Returns the created TTP with its version number, or an error if
    validation fails or the ID already exists.
    """

    from reinvent4_mcp.tools.ttp import create_ttp as _impl

    return json.dumps(_impl(_ensure_dict(ttp)), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: update_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def update_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to update"),
    ],
    updates: Annotated[
        dict[str, Any],
        Field(
            description=(
                "Partial update to apply. Uses the same schema as "
                "create_ttp -- provide only the fields you want to change. "
                "Nested dicts are merged recursively. Version is "
                "auto-incremented. The same validation rules apply: "
                "no extra fields, type must be 'native' or 'external', "
                "constraints must be a list, etc."
            ),
        ),
    ],
) -> str:
    """Update an existing TTP with partial changes.

    Merges the updates into the current TTP, validates the result,
    and auto-increments the version. Both the updated current file
    and a versioned snapshot are saved.

    Example -- change an objective's weight and add a constraint:

        ttp_id: "jak2-lead-opt"
        updates: {
          "scoring": {
            "objectives": [
              {"name": "SA Score", "type": "native",
               "component": "SAScore", "weight": 0.5}
            ]
          },
          "constraints": [
            {"type": "custom_alerts", "smarts": ["[#7]=[#7]"]}
          ]
        }

    Tip: call get_ttp first to see the current structure.
    """

    from reinvent4_mcp.tools.ttp import update_ttp as _impl

    return json.dumps(_impl(ttp_id, _ensure_dict(updates)), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to retrieve"),
    ],
    version: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Specific version to retrieve. "
                "Omit to get the latest version."
            ),
        ),
    ] = None,
) -> str:
    """Retrieve a stored TTP by ID, optionally at a specific version.

    Returns the full TTP definition including all objectives, weights,
    transforms, target info, and constraints. Use list_ttps to see
    available TTP IDs.
    """

    from reinvent4_mcp.tools.ttp import get_ttp as _impl

    return json.dumps(_impl(ttp_id, version), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_ttps
# ---------------------------------------------------------------------------

@mcp.tool()
def list_ttps() -> str:
    """List all stored TTPs with summary metadata.

    Returns ID, name, and current version for each TTP.
    Use get_ttp to retrieve the full definition.
    """

    from reinvent4_mcp.tools.ttp import list_ttps as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: delete_ttp
# ---------------------------------------------------------------------------

@mcp.tool()
def delete_ttp(
    ttp_id: Annotated[
        str,
        Field(description="ID of the TTP to delete"),
    ],
) -> str:
    """Delete a stored TTP and all its version history.

    This is irreversible. The TTP and all versioned snapshots are removed.
    """

    from reinvent4_mcp.tools.ttp import delete_ttp as _impl

    return json.dumps(_impl(ttp_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: run_optimization
# ---------------------------------------------------------------------------

@mcp.tool()
def run_optimization(
    ttp_id: Annotated[
        str | None,
        Field(
            default=None,
            description="ID of a stored TTP to optimise against.",
        ),
    ] = None,
    ttp: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Inline TTP definition (alternative to ttp_id).",
        ),
    ] = None,
    model: Annotated[
        str,
        Field(
            default=".reinvent",
            description=(
                "Prior model: registry key (e.g. '.reinvent') or file path. "
                "Use list_available_models to see options."
            ),
        ),
    ] = ".reinvent",
    max_steps: Annotated[
        int,
        Field(
            default=300,
            description="Maximum RL steps for this campaign.",
        ),
    ] = 300,
    batch_size: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Molecules generated per RL step. Overrides process config. "
                "Lower values reduce Boltz cost per step."
            ),
        ),
    ] = None,
    sigma: Annotated[
        int | None,
        Field(default=None, description="DAP sigma (reward sharpness). Overrides process config."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        Field(default=None, description="Optimizer learning rate. Overrides process config."),
    ] = None,
    resume_from: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Checkpoint path to resume from. Use the checkpoint_path "
                "returned by stop_optimization or get_optimization_status."
            ),
        ),
    ] = None,
    termination: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Early termination criteria (e.g. convergence threshold).",
        ),
    ] = None,
    seed_smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "List of SMILES strings used as input seeds for Mol2Mol "
                "priors. Required when model is a Mol2Mol variant "
                "(e.g. '.m2m_medium'). Ignored for "
                "Reinvent priors."
            ),
        ),
    ] = None,
) -> str:
    """Launch an RL optimization campaign using a TTP.

    Composes a Scorer from the TTP's full objective set (pre-scoring
    descriptors + native components + external REST services like Boltz
    affinity and PoseBusters pose validation), generates a REINVENT4
    config, and starts a background RL process.

    Monitor progress with get_optimization_status. Stop with
    stop_optimization. Resume from checkpoint by passing the
    checkpoint_path to resume_from.

    Workflow:
    1. run_optimization(ttp_id="jak2-lead", max_steps=300)
    2. get_optimization_status(job_id)  -- poll periodically
    3. stop_optimization(job_id)  -- if intervention needed
    4. update_ttp("jak2-lead", {...})  -- adjust weights/objectives
    5. run_optimization(ttp_id="jak2-lead", resume_from="<checkpoint>")
    """

    from reinvent4_mcp.tools.optimization import run_optimization as _impl

    ttp_dict = _ensure_dict(ttp) if ttp is not None else None
    result = _impl(
        ttp_id=ttp_id,
        ttp=ttp_dict,
        model=model,
        max_steps=max_steps,
        batch_size=batch_size,
        sigma=sigma,
        learning_rate=learning_rate,
        resume_from=resume_from,
        termination=termination,
        seed_smiles=seed_smiles,
    )
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_optimization_status
# ---------------------------------------------------------------------------

@mcp.tool()
def get_optimization_status(
    job_id: Annotated[
        str,
        Field(description="ID of the optimization job to query."),
    ],
) -> str:
    """Monitor a running or completed optimization job with convergence diagnostics.

    Returns the current state, step count, score trajectory,
    per-component statistics, and automated convergence diagnostics
    with actionable warnings and recommendations.

    Key fields for agent decision-making:
    - trajectory.current_step / max_steps: progress
    - trajectory.mean_total_score: overall performance
    - trajectory.latest_mean_score: recent performance (plateau detection)
    - trajectory.per_component: per-component summary stats (mean/max/min)
    - trajectory.validity_rate: model health
    - trajectory.score_trajectory: step-by-step total score trend
    - trajectory.per_component_trajectory: step-by-step trend per component

    Convergence diagnostics (trajectory.diagnostics):
    - convergence_status: 'early' | 'improving' | 'plateau' | 'diverging' | 'stable' | 'completed'
    - plateau_detected: whether the score has stopped improving
    - diverging: whether scores are declining
    - improvement_rate: score change per step (recent window)
    - trend: early_mean, mid_mean, recent_mean, overall_improvement
    - component_health: per-component NaN rates and status ('healthy' | 'degraded' | 'failing')
    - warnings: list of actionable warnings (low_validity, score_plateau, component_failure, etc.)
    - recommendations: list of suggested actions for the agent

    Poll this every few minutes during a run. Use diagnostics.convergence_status
    and diagnostics.warnings to decide when to stop and adjust the TTP.
    """

    from reinvent4_mcp.tools.results import get_optimization_status as _impl

    return json.dumps(_impl(job_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: stop_optimization
# ---------------------------------------------------------------------------

@mcp.tool()
def stop_optimization(
    job_id: Annotated[
        str,
        Field(description="ID of the optimization job to stop."),
    ],
) -> str:
    """Stop a running optimization job and save a checkpoint.

    Sends a graceful interrupt to the REINVENT4 process, which saves
    a checkpoint file. The checkpoint can be used with run_optimization's
    resume_from parameter to continue the run after TTP adjustments.

    Returns the checkpoint path and final status.
    """

    from reinvent4_mcp.tools.optimization import stop_optimization as _impl

    return json.dumps(_impl(job_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: run_transfer_learning
# ---------------------------------------------------------------------------

@mcp.tool()
def run_transfer_learning(
    model: Annotated[
        str,
        Field(
            default=".reinvent",
            description=(
                "Prior model to fine-tune: registry key (e.g. '.reinvent') "
                "or file path. Use list_available_models to see options."
            ),
        ),
    ] = ".reinvent",
    smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Inline list of SMILES strings for training. "
                "Provide either this or smiles_file, not both."
            ),
        ),
    ] = None,
    smiles_file: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Path to a .smi file with training SMILES (one per line). "
                "Provide either this or smiles, not both."
            ),
        ),
    ] = None,
    num_epochs: Annotated[
        int,
        Field(default=50, description="Number of training epochs."),
    ] = 50,
    batch_size: Annotated[
        int,
        Field(default=128, description="Training batch size."),
    ] = 128,
    save_every_n_epochs: Annotated[
        int,
        Field(default=5, description="Save a checkpoint every N epochs."),
    ] = 5,
    sample_batch_size: Annotated[
        int,
        Field(
            default=2000,
            description="Molecules sampled for validation metrics.",
        ),
    ] = 2000,
    validation_smiles: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Inline list of SMILES for validation (optional).",
        ),
    ] = None,
    validation_smiles_file: Annotated[
        str | None,
        Field(
            default=None,
            description="Path to a validation .smi file (optional).",
        ),
    ] = None,
    standardize_smiles: Annotated[
        bool,
        Field(default=True, description="Standardize SMILES before training."),
    ] = True,
    randomize_smiles: Annotated[
        bool,
        Field(default=True, description="Randomize SMILES representations."),
    ] = True,
    output_model_file: Annotated[
        str,
        Field(
            default="tl_model.model",
            description="Base name for the output model file.",
        ),
    ] = "tl_model.model",
) -> str:
    """Fine-tune a generative model on a SMILES dataset via transfer learning.

    Transfer learning adapts a prior model to produce molecules similar
    to the provided SMILES set. The resulting model can then be used as
    the starting agent for run_optimization (RL).

    Workflow:
    1. run_transfer_learning(model=".reinvent", smiles=[...], num_epochs=50)
    2. get_optimization_status(job_id)  -- monitor TL progress
    3. Use the output model in run_optimization(model="<output_model_path>")

    Epoch-level checkpoints are saved as <output_model_file>.<epoch>.chkpt
    in the job working directory. Choose the best epoch based on
    validation NLL (lower is better).
    """

    from reinvent4_mcp.tools.optimization import run_transfer_learning as _impl

    result = _impl(
        model=model,
        smiles=smiles,
        smiles_file=smiles_file,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_every_n_epochs=save_every_n_epochs,
        sample_batch_size=sample_batch_size,
        validation_smiles=validation_smiles,
        validation_smiles_file=validation_smiles_file,
        standardize_smiles=standardize_smiles,
        randomize_smiles=randomize_smiles,
        output_model_file=output_model_file,
    )
    return json.dumps(result, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_top_candidates
# ---------------------------------------------------------------------------

@mcp.tool()
def get_top_candidates(
    job_id: Annotated[
        str | None,
        Field(
            default=None,
            description="ID of an optimization job to extract candidates from.",
        ),
    ] = None,
    csv_path: Annotated[
        str | None,
        Field(
            default=None,
            description="Direct path to a REINVENT4 output CSV (alternative to job_id).",
        ),
    ] = None,
    top_n: Annotated[
        int,
        Field(default=50, description="Number of top candidates to return."),
    ] = 50,
    min_score: Annotated[
        float | None,
        Field(default=None, description="Minimum total score threshold."),
    ] = None,
) -> str:
    """Extract the best molecules from an optimization run.

    Reads the output CSV, filters for valid molecules above the
    minimum score, deduplicates by SMILES, and returns the top-N
    ranked by total score with per-component breakdowns.

    Use after a run completes or during a run to peek at intermediate
    results.
    """

    from reinvent4_mcp.tools.results import get_top_candidates as _impl

    return json.dumps(
        _impl(job_id=job_id, csv_path=csv_path, top_n=top_n, min_score=min_score),
        indent=2,
        default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: read_results
# ---------------------------------------------------------------------------

@mcp.tool()
def read_results(
    csv_path: Annotated[
        str,
        Field(description="Path to a REINVENT4 output CSV file."),
    ],
) -> str:
    """Parse a REINVENT4 output CSV and return summary statistics.

    Provides n_rows, unique SMILES count, validity rate,
    total score statistics, and per-component averages.
    Useful for analysing results from any REINVENT4 run without
    a tracked job ID.
    """

    from reinvent4_mcp.tools.results import read_results as _impl

    return json.dumps(_impl(csv_path), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_job_alerts
# ---------------------------------------------------------------------------

@mcp.tool()
def get_job_alerts() -> str:
    """Check all running jobs for problems that may need intervention.

    Returns alerts for: job failures, scoring service errors,
    score plateaus, divergence, component failures, validity drops,
    and stalled progress. Each alert includes severity and a
    recommended action.

    Call this periodically during long optimization runs.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    return json.dumps(get_job_manager().get_alerts(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_jobs
# ---------------------------------------------------------------------------

@mcp.tool()
def list_jobs() -> str:
    """List all tracked optimization jobs with their current status.

    Returns job IDs, states (queued/running/completed/failed/stopped),
    TTP IDs, and creation times. Use get_optimization_status for
    detailed metrics on a specific job.
    """

    from reinvent4_mcp.jobs.manager import get_job_manager

    manager = get_job_manager()
    return json.dumps(manager.list_jobs(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_shared_assets
# ---------------------------------------------------------------------------

@mcp.tool()
def list_shared_assets() -> str:
    """List usable files on the shared volume (structures, MSAs, cofold outputs).

    Use this to GROUND any path before profiling or cofolding -- never guess or
    invent a file path. Returns categorized, /shared-relative paths:
    - ``structures``: protein/complex PDB/mmCIF files (feed to
      ``profile_interactions`` structure_paths).
    - ``msas``: precomputed alignments (.a3m/.fasta/...) usable as a TTP
      target ``msa_path``.
    - ``cofold_outputs``: structures produced by Boltz (under ``predictions/``).
    Returns an empty (non-error) result when nothing is present.
    """

    from reinvent4_mcp.tools.external import list_shared_assets as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: fetch_target_info
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_target_info(
    query: Annotated[
        str,
        Field(
            description=(
                "A UniProt accession (e.g. 'O60885' for BRD4) or free text "
                "such as a gene name plus organism (e.g. 'BRD4 human')."
            ),
        ),
    ],
) -> str:
    """Look up a protein target on UniProt (canonical sequence, PDBs, function).

    Use this whenever a target is named but no sequence is provided, instead of
    recalling a sequence from memory. Returns the primary accession, protein/
    gene names, organism, the canonical amino-acid sequence (drop straight into
    a TTP ``target.protein_sequence``), a short function summary, and candidate
    PDB IDs. On network failure it returns an error and does NOT fabricate a
    sequence.
    """

    from reinvent4_mcp.tools.uniprot import fetch_target_info as _impl

    return json.dumps(_impl(query), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: fetch_compound
# ---------------------------------------------------------------------------

@mcp.tool()
def fetch_compound(
    name: Annotated[
        str,
        Field(
            description=(
                "Compound name or synonym to resolve to a real SMILES via "
                "PubChem (e.g. 'JQ1', 'aspirin', 'acetazolamide', 'imatinib')."
            ),
        ),
    ],
) -> str:
    """Resolve a molecule NAME to a real SMILES from PubChem.

    Use this whenever a molecule is referred to by name instead of a SMILES.
    NEVER recall a SMILES from memory - hallucinated SMILES are frequently
    invalid (RDKit cannot parse or 3D-embed them) and break cofolding. Returns
    the isomeric ``smiles`` (use it directly for score_molecules /
    profile_interactions), plus CID, formula, and IUPAC name. On failure it
    returns an error and does not fabricate a structure - ask the user to paste
    the SMILES.
    """

    from reinvent4_mcp.tools.pubchem import fetch_compound as _impl

    return json.dumps(_impl(name), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: start_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def start_campaign(
    goal: Annotated[
        str,
        Field(description="What the campaign is for, in a sentence"),
    ],
    base_ttp_id: Annotated[
        str,
        Field(
            description=(
                "TTP every task starts from. Each task runs a variant of it "
                "with its own objective weights."
            ),
        ),
    ],
    objectives: Annotated[
        list[str],
        Field(
            description=(
                "Objective names the Pareto front is computed over. Fixed at "
                "creation, so the front's dimensions cannot drift later. Use "
                "the names as they appear in the base TTP."
            ),
        ),
    ],
    concurrency: Annotated[
        int,
        Field(default=1, ge=1, description="Supervisors to hold in flight"),
    ] = 1,
    hours: Annotated[
        float | None,
        Field(default=None, gt=0, description="Dispatch window; omit for no deadline"),
    ] = None,
    grace_hours: Annotated[
        float,
        Field(
            default=0.0, ge=0,
            description="Drain window after the dispatch window closes, during "
                        "which in-flight tasks finish but nothing new is queued",
        ),
    ] = 0.0,
) -> str:
    """Open a fleet campaign: a goal, a pool size, and a wall-clock budget.

    This only opens the ledger entry. It dispatches no work and starts no
    supervisors, so a campaign can be set up and inspected before it begins
    spending GPU. The conductor loop is what fills it.
    """

    from reinvent4_mcp.tools.campaign import start_campaign as _impl

    return json.dumps(
        _impl(goal, base_ttp_id, objectives, concurrency, hours, grace_hours),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: get_campaign_state
# ---------------------------------------------------------------------------

@mcp.tool()
def get_campaign_state(
    campaign_id: Annotated[str, Field(description="Campaign to inspect")],
) -> str:
    """Everything about a campaign in one call.

    Task counts by state, the supervisors currently holding tasks with their
    weight vectors and lease remaining, the Pareto front, and how long is left
    in the dispatch and drain windows.

    ``window.phase`` follows the clock rather than the stored state, so a
    campaign whose conductor missed its own deadline still reports that it
    should be draining.
    """

    from reinvent4_mcp.tools.campaign import get_campaign_state as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: list_campaigns
# ---------------------------------------------------------------------------

@mcp.tool()
def list_campaigns() -> str:
    """List every campaign in the ledger, newest first."""

    from reinvent4_mcp.tools.campaign import list_campaigns as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: enqueue_task
# ---------------------------------------------------------------------------

@mcp.tool()
def enqueue_task(
    campaign_id: Annotated[
        str,
        Field(description="Campaign to add to; must still be dispatching"),
    ],
    weight_overrides: Annotated[
        dict[str, float],
        Field(
            description=(
                "Objective name to weight. Names must match the base TTP's "
                "objectives -- an unknown name fails the task rather than "
                "running silently with the wrong weights."
            ),
        ),
    ],
    fidelity_tier: Annotated[
        str,
        Field(
            default="exploration",
            description="'exploration' uses the scoring cache; 'validation' "
                        "bypasses the read so a result is never a cached sample",
        ),
    ] = "exploration",
    max_steps: Annotated[
        int | None,
        Field(default=None, gt=0, description="RL steps for this task"),
    ] = None,
) -> str:
    """Queue one weight vector by hand.

    For following a hunch about a trade-off the automatic proposer has not
    reached. The conductor loop will not re-propose a vector already in the
    ledger, so a task added here is not duplicated later.
    """

    from reinvent4_mcp.tools.campaign import enqueue_task as _impl

    return json.dumps(
        _impl(campaign_id, weight_overrides, None, fidelity_tier, max_steps),
        indent=2, default=_json_default,
    )


# ---------------------------------------------------------------------------
# Tool: list_supervisors
# ---------------------------------------------------------------------------

@mcp.tool()
def list_supervisors(
    campaign_id: Annotated[str, Field(description="Campaign to inspect")],
) -> str:
    """The supervisors holding tasks right now, and what each is trying.

    Expired leases are swept first, so this is who is actually alive rather
    than who last claimed something.
    """

    from reinvent4_mcp.tools.campaign import list_supervisors as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_supervisor_result
# ---------------------------------------------------------------------------

@mcp.tool()
def get_supervisor_result(
    campaign_id: Annotated[str, Field(description="Campaign the task belongs to")],
    task_id: Annotated[str, Field(description="Task whose result to fetch")],
) -> str:
    """What one finished task achieved: its objective profile and molecules."""

    from reinvent4_mcp.tools.campaign import get_supervisor_result as _impl

    return json.dumps(_impl(campaign_id, task_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_pareto_front
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pareto_front(
    campaign_id: Annotated[str, Field(description="Campaign to inspect")],
) -> str:
    """The non-dominated results: the trade-offs nothing else beats.

    Also returns ``ineligible`` -- results left off the front for not
    reporting every objective. That usually means a scorer failed, and is
    worth looking at: those runs are not bad trade-offs, they are incomplete
    measurements.
    """

    from reinvent4_mcp.tools.campaign import get_pareto_front as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: set_concurrency
# ---------------------------------------------------------------------------

@mcp.tool()
def set_concurrency(
    campaign_id: Annotated[str, Field(description="Campaign to adjust")],
    concurrency: Annotated[
        int,
        Field(ge=0, description="Supervisors to hold in flight"),
    ],
) -> str:
    """Change how many supervisors the campaign holds in flight.

    Recorded in the ledger, so the number survives a conductor restart. The
    conductor loop scales the supervisor Deployment to match on its next pass;
    this call on its own does not move pods.
    """

    from reinvent4_mcp.tools.campaign import set_concurrency as _impl

    return json.dumps(_impl(campaign_id, concurrency), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: cancel_supervisor
# ---------------------------------------------------------------------------

@mcp.tool()
def cancel_supervisor(
    task_id: Annotated[str, Field(description="Task to take back")],
) -> str:
    """Take a task back from the supervisor running it.

    The holder finds out on its next heartbeat and abandons the work without
    posting a result. This does not stop its RL job -- use stop_optimization
    on the job id if the GPU is what you need back.
    """

    from reinvent4_mcp.tools.campaign import cancel_supervisor as _impl

    return json.dumps(_impl(task_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: get_pool_capacity
# ---------------------------------------------------------------------------

@mcp.tool()
def get_pool_capacity() -> str:
    """Whether the fleet has room for more work, from the scorer's queue.

    Boltz is the binding constraint, not the agent count, so capacity is a
    question about its queue depth. A ``None`` queue_pressure means the
    gateway did not answer, which is not the same as idle.
    """

    from reinvent4_mcp.tools.campaign import get_pool_capacity as _impl

    return json.dumps(_impl(), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: close_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def close_campaign(
    campaign_id: Annotated[str, Field(description="Campaign to stop dispatching")],
) -> str:
    """Stop dispatching, let in-flight tasks finish.

    The drain window brought forward by hand. Queued tasks stay claimable, so
    work already decided on still runs.
    """

    from reinvent4_mcp.tools.campaign import close_campaign as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Tool: finalize_campaign
# ---------------------------------------------------------------------------

@mcp.tool()
def finalize_campaign(
    campaign_id: Annotated[str, Field(description="Campaign to close the books on")],
) -> str:
    """Close the books: abandon anything outstanding and end the campaign.

    Tasks still queued or claimed are marked abandoned rather than left
    looking like work someone will do.
    """

    from reinvent4_mcp.tools.campaign import finalize_campaign as _impl

    return json.dumps(_impl(campaign_id), indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dict(value: Any) -> dict[str, Any]:
    """Coerce a JSON string to a dict if needed.

    LLMs sometimes pass tool arguments as JSON strings instead of
    parsed dicts.  This handles that gracefully instead of failing
    with a Pydantic validation error.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    raise TypeError(f"Expected a JSON object (dict), got {type(value).__name__}")


def _json_default(obj: Any) -> Any:
    """Handle non-standard types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, set):
        return list(obj)
    return repr(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    """Launch the Drug Discovery MCP server with stdio transport."""

    config = get_config()
    configure_logging(config.logging.level)

    logger.info(f"Starting Drug Discovery MCP server v{__version__}")

    try:
        ensure_reinvent4_importable()
        logger.info("REINVENT4 is importable")
    except RuntimeError as exc:
        logger.warning(f"REINVENT4 not found at startup: {exc}")

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
