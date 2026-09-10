"""Generate REINVENT4 TOML configuration from TTP + process config.

Converts the scorer config dict (from ``ttp_to_scorer_config``) and
operational parameters into the TOML format that the ``reinvent`` CLI
expects for ``staged_learning`` runs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def generate_reinvent_toml(
    *,
    scorer_config: dict[str, Any],
    prior_file: str,
    agent_file: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 16,
    sigma: int = 128,
    learning_rate: float = 0.0001,
    unique_sequences: bool = True,
    randomize_smiles: bool = True,
    max_steps: int = 300,
    max_score: float = 1.0,
    use_checkpoint: bool = False,
    summary_csv_prefix: str = "results",
    chkpt_file: str = "stage.chkpt",
    tb_logdir: str = "tb_logs",
    diversity_filter_type: str = "IdenticalMurckoScaffold",
    diversity_filter_bucket_size: int = 25,
    diversity_filter_minscore: float = 0.4,
    termination: Optional[dict[str, Any]] = None,
    smiles_file: Optional[str] = None,
) -> str:
    """Build a complete REINVENT4 TOML config string for staged learning.

    Parameters
    ----------
    scorer_config:
        Dict from ``ttp_to_scorer_config()`` with keys ``type`` (aggregation)
        and ``component`` (list of component dicts).
    prior_file:
        Path to the prior model file.
    agent_file:
        Path to the agent model file (defaults to *prior_file*).
        For checkpoint resume this should be the checkpoint path.
    device:
        Torch device string.
    batch_size:
        Molecules per RL step.
    sigma, learning_rate:
        DAP learning strategy parameters.
    max_steps:
        Maximum RL steps for this stage.
    use_checkpoint:
        Whether to resume from an existing checkpoint.
    chkpt_file:
        Path for saving/loading the stage checkpoint.
    diversity_filter_type, diversity_filter_bucket_size, diversity_filter_minscore:
        Diversity filter configuration.
    termination:
        Optional early-termination criteria.
    smiles_file:
        Path to a SMILES input file.  Required for Mol2Mol priors
        which condition generation on seed molecules.

    Returns
    -------
    str
        Complete TOML configuration string.
    """

    if agent_file is None:
        agent_file = prior_file

    lines: list[str] = []

    _emit_global(lines, device=device, tb_logdir=tb_logdir)
    lines.append("")
    _emit_parameters(
        lines,
        prior_file=prior_file,
        agent_file=agent_file,
        summary_csv_prefix=summary_csv_prefix,
        batch_size=batch_size,
        randomize_smiles=randomize_smiles,
        use_checkpoint=use_checkpoint,
        smiles_file=smiles_file,
    )
    lines.append("")
    _emit_learning_strategy(lines, sigma=sigma, rate=learning_rate)
    lines.append("")
    _emit_diversity_filter(
        lines,
        filter_type=diversity_filter_type,
        bucket_size=diversity_filter_bucket_size,
        minscore=diversity_filter_minscore,
    )
    lines.append("")
    _emit_stage(
        lines,
        scorer_config=scorer_config,
        max_steps=max_steps,
        max_score=max_score,
        chkpt_file=chkpt_file,
        termination=termination,
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Section emitters
# ---------------------------------------------------------------------------


def _emit_global(
    lines: list[str],
    *,
    device: str,
    tb_logdir: str,
) -> None:
    lines.append('run_type = "staged_learning"')
    lines.append(f'device = "{device}"')
    lines.append(f'tb_logdir = "{tb_logdir}"')
    lines.append('json_out_config = "_config.json"')


def _emit_parameters(
    lines: list[str],
    *,
    prior_file: str,
    agent_file: str,
    summary_csv_prefix: str,
    batch_size: int,
    randomize_smiles: bool,
    use_checkpoint: bool,
    smiles_file: Optional[str] = None,
) -> None:
    lines.append("[parameters]")
    lines.append(f'prior_file = "{prior_file}"')
    lines.append(f'agent_file = "{agent_file}"')
    lines.append(f'summary_csv_prefix = "{summary_csv_prefix}"')
    lines.append(f"batch_size = {batch_size}")
    # NOTE: 'unique_sequences' was removed from REINVENT4's RL [parameters]
    # schema (dedup is automatic now); emitting it trips extra="forbid".
    lines.append(f"randomize_smiles = {_toml_bool(randomize_smiles)}")
    lines.append(f"use_checkpoint = {_toml_bool(use_checkpoint)}")
    if smiles_file is not None:
        lines.append(f'smiles_file = "{smiles_file}"')


def _emit_learning_strategy(
    lines: list[str],
    *,
    sigma: int,
    rate: float,
) -> None:
    lines.append("[learning_strategy]")
    lines.append('type = "dap"')
    lines.append(f"sigma = {sigma}")
    lines.append(f"rate = {rate}")


def _emit_diversity_filter(
    lines: list[str],
    *,
    filter_type: str,
    bucket_size: int,
    minscore: float,
) -> None:
    lines.append("[diversity_filter]")
    lines.append(f'type = "{filter_type}"')
    lines.append(f"bucket_size = {bucket_size}")
    lines.append(f"minscore = {minscore}")


def _emit_stage(
    lines: list[str],
    *,
    scorer_config: dict[str, Any],
    max_steps: int,
    max_score: float,
    chkpt_file: str,
    termination: Optional[dict[str, Any]],
) -> None:
    lines.append("[[stage]]")
    lines.append(f"max_score = {max_score}")
    lines.append(f"max_steps = {max_steps}")
    lines.append(f'chkpt_file = "{chkpt_file}"')

    if termination:
        for key, val in termination.items():
            lines.append(f"{key} = {_toml_value(val)}")

    lines.append("")
    lines.append("[stage.scoring]")
    lines.append(f'type = "{scorer_config["type"]}"')

    for comp_dict in scorer_config.get("component", []):
        lines.append("")
        _emit_component(lines, comp_dict)


def _emit_component(lines: list[str], comp_dict: dict[str, Any]) -> None:
    """Emit a single scoring component in REINVENT4 TOML format.

    Input is one element from the scorer config's ``component`` list,
    e.g. ``{"QED": {"endpoint": [{"name": "QED", "weight": 0.6}]}}``.
    """

    for comp_type, comp_body in comp_dict.items():
        endpoints = comp_body.get("endpoint", [])
        lines.append("[[stage.scoring.component]]")
        lines.append(f"[stage.scoring.component.{comp_type}]")

        for endpoint in endpoints:
            lines.append(f"[[stage.scoring.component.{comp_type}.endpoint]]")
            _emit_endpoint_fields(lines, endpoint)


def _emit_endpoint_fields(lines: list[str], endpoint: dict[str, Any]) -> None:
    """Emit key-value pairs for one endpoint, handling nested dicts."""

    for key, val in endpoint.items():
        if key == "transform" and isinstance(val, dict):
            for tkey, tval in val.items():
                if tval is not None:
                    lines.append(f"transform.{tkey} = {_toml_value(tval)}")
        elif key == "params" and isinstance(val, dict):
            for pkey, pval in val.items():
                if pval is not None:
                    lines.append(f"params.{pkey} = {_toml_value(pval)}")
        else:
            if val is not None:
                lines.append(f"{key} = {_toml_value(val)}")


# ---------------------------------------------------------------------------
# TOML value serialization
# ---------------------------------------------------------------------------


def _toml_bool(v: bool) -> str:
    return "true" if v else "false"


def _toml_value(val: Any) -> str:
    """Serialize a Python value to a TOML literal."""

    if isinstance(val, bool):
        return _toml_bool(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val}"
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, list):
        inner = ", ".join(_toml_value(v) for v in val)
        return f"[{inner}]"
    if isinstance(val, dict):
        pairs = ", ".join(f"{k} = {_toml_value(v)}" for k, v in val.items())
        return f"{{{pairs}}}"
    return f'"{val}"'


def generate_tl_toml(
    *,
    input_model_file: str,
    smiles_file: str,
    output_model_file: str = "tl_model.model",
    device: str = "cuda:0",
    num_epochs: int = 50,
    batch_size: int = 128,
    save_every_n_epochs: int = 5,
    sample_batch_size: int = 2000,
    validation_smiles_file: Optional[str] = None,
    standardize_smiles: bool = True,
    randomize_smiles: bool = True,
    randomize_all_smiles: bool = False,
    internal_diversity: bool = True,
    tb_logdir: str = "tb_logs",
) -> str:
    """Build a REINVENT4 TOML config string for transfer learning.

    Parameters
    ----------
    input_model_file:
        Path to the model to fine-tune (prior or checkpoint).
    smiles_file:
        Path to the training SMILES file (.smi).
    output_model_file:
        Base name for the output model file.
    device:
        Torch device string.
    num_epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    save_every_n_epochs:
        Save a checkpoint every N epochs.
    sample_batch_size:
        Number of molecules to sample for validation metrics.
    validation_smiles_file:
        Optional path to a held-out validation SMILES file.
    standardize_smiles, randomize_smiles, randomize_all_smiles:
        SMILES preprocessing options.
    internal_diversity:
        Whether to compute internal diversity metrics.
    tb_logdir:
        TensorBoard log directory.

    Returns
    -------
    str
        Complete TOML configuration string.
    """

    lines: list[str] = []

    lines.append('run_type = "transfer_learning"')
    lines.append(f'device = "{device}"')
    lines.append(f'tb_logdir = "{tb_logdir}"')
    lines.append("")

    lines.append("[parameters]")
    lines.append(f"num_epochs = {num_epochs}")
    lines.append(f"save_every_n_epochs = {save_every_n_epochs}")
    lines.append(f"batch_size = {batch_size}")
    lines.append(f"sample_batch_size = {sample_batch_size}")
    lines.append(f'input_model_file = "{input_model_file}"')
    lines.append(f'output_model_file = "{output_model_file}"')
    lines.append(f'smiles_file = "{smiles_file}"')
    if validation_smiles_file:
        lines.append(f'validation_smiles_file = "{validation_smiles_file}"')
    lines.append(f"standardize_smiles = {_toml_bool(standardize_smiles)}")
    lines.append(f"randomize_smiles = {_toml_bool(randomize_smiles)}")
    lines.append(f"randomize_all_smiles = {_toml_bool(randomize_all_smiles)}")
    lines.append(f"internal_diversity = {_toml_bool(internal_diversity)}")

    return "\n".join(lines) + "\n"


def resolve_model_path(model: str) -> str:
    """Resolve a model registry key (e.g. ``.reinvent``) to a file path.

    If *model* looks like a registry key (starts with ``.``), queries
    REINVENT4's prior registry.  Otherwise returns *model* unchanged
    (assumed to be a file path).

    Raises:
        ValueError: if *model* is a registry key (starts with ``.``) that
            is not present in REINVENT4's prior registry.  This avoids the
            confusing downstream failure where REINVENT4 treats an unknown
            key as a relative file path and resolves it against the job's
            working directory.
    """

    if not model.startswith("."):
        return model

    try:
        from reinvent4_mcp.utils import ensure_reinvent4_importable

        ensure_reinvent4_importable()
        from reinvent.utils.prior_registry import prior_registry
    except Exception:
        # REINVENT4 is not importable here (e.g. in unit tests); we cannot
        # validate the key, so fall back to passing it through unchanged.
        logger.warning(
            "Could not load REINVENT4 prior registry to validate model key "
            "'%s'; using as-is",
            model,
        )
        return model

    if model in prior_registry:
        resolved = str(prior_registry[model])
        logger.info("Resolved model key '%s' to '%s'", model, resolved)
        return resolved

    valid_keys = ", ".join(sorted(prior_registry))
    raise ValueError(
        f"Unknown model registry key '{model}'. "
        f"Valid keys are: {valid_keys}. "
        f"For a custom prior, pass an absolute file path instead. "
        f"Use list_available_models to see options."
    )
