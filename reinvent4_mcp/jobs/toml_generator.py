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

    components = scorer_config.get("component", [])
    # REINVENT4 v4.x requires at least one scoring component.
    # Fall back to QED when no objectives are defined.
    if not components:
        components = [{"QED": {"endpoint": [{"name": "QED score", "weight": 1.0}]}}]

    for comp_dict in components:
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
                    # Mapper plumbing can wrap scalar parameters in a one-item
                    # list (radius=[3], use_counts=[True]). REINVENT4 expects
                    # the bare scalar. Preserve real multi-value parameters
                    # such as CustomAlerts.smarts.
                    if (
                        isinstance(pval, list)
                        and len(pval) == 1
                        and not isinstance(pval[0], (list, dict))
                    ):
                        pval = pval[0]
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



# Transformer prior aliases — not in REINVENT4's built-in prior_registry.
# Maps our dot-keys to the filename under /app/REINVENT4/priors/.
_TRANSFORMER_PRIOR_FILES: dict[str, str] = {
    ".libinvent_transformer":  "libinvent_transformer_pubchem.prior",
    ".linkinvent_transformer": "linkinvent_transformer_pubchem.prior",
}

_PRIOR_KEY_ALIASES: dict[str, str] = {
    ".mol2mol": ".m2m_medium",
    ".mol2mol_medium": ".m2m_medium",
    ".mol2mol_high": ".m2m_high",
    ".mol2mol_similarity": ".m2m_similarity",
    ".mol2mol_scaffold": ".m2m_scaffold",
    ".mol2mol_scaffold_generic": ".m2m_scaffold_generic",
    ".mol2mol_mmp": ".m2m_mmp",
}


def resolve_model_path(model: str) -> str:
    """Resolve a model registry key (e.g. ``.reinvent``) to a file path.

    If *model* looks like a registry key (starts with ``.``), first checks
    our transformer alias table (keys not in REINVENT4's built-in registry),
    then queries REINVENT4's prior_registry.  Returns the model unchanged
    if it does not start with ``.`` (treated as a direct file path).

    Raises:
        ValueError: if *model* is a registry key that cannot be resolved.
    """

    if not model.startswith("."):
        return model

    model = _PRIOR_KEY_ALIASES.get(model, model)

    # ── Transformer priors — resolve directly without prior_registry ──────
    if model in _TRANSFORMER_PRIOR_FILES:
        filename = _TRANSFORMER_PRIOR_FILES[model]
        try:
            from reinvent4_mcp.utils import find_reinvent4_root  # noqa: PLC0415
            import pathlib  # noqa: PLC0415
            priors_dir = pathlib.Path(find_reinvent4_root()) / "priors"
            candidate = priors_dir / filename
            if candidate.is_file():
                logger.info("Resolved transformer key '%s' to '%s'", model, candidate)
                return str(candidate)
            else:
                raise ValueError(
                    f"Transformer prior file '{filename}' not found at {priors_dir}. "
                    "Rebuild the dd-agent Docker image to download transformer priors."
                )
        except Exception as exc:
            raise ValueError(
                f"Could not resolve transformer model key '{model}': {exc}"
            ) from exc

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


# ---------------------------------------------------------------------------
# LibInvent TOML generator — block 08 addition
# ---------------------------------------------------------------------------

def generate_libinvent_toml(
    *,
    scorer_config: dict[str, Any],
    prior_file: str,
    scaffold: str,
    decoration_positions: Optional[list[int]] = None,
    agent_file: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 64,
    sigma: int = 128,
    learning_rate: float = 0.0001,
    randomize_smiles: bool = True,
    use_checkpoint: bool = False,
    max_steps: int = 300,
    max_score: float = 1.0,
    summary_csv_prefix: str = "results",
    chkpt_file: str = "stage.chkpt",
    tb_logdir: str = "tb_logs",
    diversity_filter_type: str = "IdenticalMurckoScaffold",
    diversity_filter_bucket_size: int = 25,
    diversity_filter_minscore: float = 0.4,
    termination: Optional[dict[str, Any]] = None,
) -> str:
    """Build a REINVENT4 TOML config for LibInvent scaffold-decoration RL.

    LibInvent conditions generation on a fixed scaffold with [*:n] attachment
    points and generates decorations (R-groups) at each position.

    Parameters
    ----------
    scaffold:
        Scaffold SMILES with [*:n] attachment points, e.g.
        ``'c1ccc([*:1])cc1[*:2]'``.
    decoration_positions:
        Map numbers to decorate.  None = all positions in the scaffold.
    (All other parameters mirror ``generate_reinvent_toml``.)
    """
    if agent_file is None:
        agent_file = prior_file

    lines: list[str] = []

    # LibInvent RNN (version 1 / version 4 checkpoints) produces garbage SMILES
    # on AMD ROCm / GPU due to precision differences in the LSTM attention
    # mechanism.  Force CPU for RNN-based LibInvent priors; the transformer
    # prior (libinvent_transformer_pubchem.prior) generates TWO R-groups per
    # molecule which also fails, so CPU is the safe default for all LibInvent.
    _is_tf_prior = "transformer" in prior_file.lower()
    _effective_device = device if _is_tf_prior else "cpu"

    # Global block
    lines.append('run_type = "staged_learning"')
    lines.append(f'device = "{_effective_device}"')
    lines.append(f'tb_logdir = "{tb_logdir}"')
    lines.append('json_out_config = "_config.json"')
    lines.append("")

    # Parameters
    lines.append("[parameters]")
    lines.append(f'prior_file = "{prior_file}"')
    lines.append(f'agent_file = "{agent_file}"')
    lines.append(f'summary_csv_prefix = "{summary_csv_prefix}"')
    lines.append(f"batch_size = {batch_size}")
    lines.append(f"randomize_smiles = {_toml_bool(randomize_smiles)}")
    lines.append(f"use_checkpoint = {_toml_bool(use_checkpoint)}")
    lines.append("")

    # Scaffold input — REINVENT4 v4.x LibInvent uses smiles_file (scaffolds.smi),
    # not [[parameters.scaffold_input]]. The smi file contains one scaffold per
    # line; attachment points are [*:n] atoms with unique map numbers.
    # RDKit canonicalization may strip [*:n]; keep bare * for the smi file.
    import re as _re  # noqa: PLC0415
    _scaffold_rnn = _re.sub(r'\[\*:\d+\]', '[*]', scaffold) if '[*:' in scaffold else scaffold
    smiles_filename = "scaffolds.smi"
    lines.append(f'smiles_file = "{smiles_filename}"')
    lines.append("")
    # Store scaffold in comment for auto-extraction by submit_toml
    lines.append(f"# Scaffold: {scaffold}")
    if decoration_positions is not None:
        positions = ", ".join(str(position) for position in decoration_positions)
        lines.append(f"# Decoration positions: {positions}")
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
# LinkInvent TOML generator — block 08 addition
# ---------------------------------------------------------------------------

def generate_linkinvent_toml(
    *,
    scorer_config: dict[str, Any],
    prior_file: str,
    fragments: list[str],
    agent_file: Optional[str] = None,
    device: str = "cuda:0",
    batch_size: int = 64,
    sigma: int = 128,
    learning_rate: float = 0.0001,
    randomize_smiles: bool = True,
    use_checkpoint: bool = False,
    max_steps: int = 300,
    max_score: float = 1.0,
    summary_csv_prefix: str = "results",
    chkpt_file: str = "stage.chkpt",
    tb_logdir: str = "tb_logs",
    diversity_filter_type: str = "IdenticalMurckoScaffold",
    diversity_filter_bucket_size: int = 25,
    diversity_filter_minscore: float = 0.4,
    termination: Optional[dict[str, Any]] = None,
) -> str:
    """Build a REINVENT4 TOML config for LinkInvent linker-generation RL.

    LinkInvent generates linkers between two molecular fragments.  Each
    fragment SMILES must contain exactly one [*:n] attachment point with a
    unique map number.

    Parameters
    ----------
    fragments:
        List of fragment SMILES, each with one [*:n] atom, e.g.
        ``['[*:1]CC', '[*:2]OC']``.
    (All other parameters mirror ``generate_reinvent_toml``.)
    """
    if len(fragments) < 2:
        raise ValueError("LinkInvent requires at least 2 fragment SMILES.")
    if agent_file is None:
        agent_file = prior_file

    lines: list[str] = []

    # Global block
    lines.append('run_type = "staged_learning"')
    lines.append(f'device = "{device}"')
    lines.append(f'tb_logdir = "{tb_logdir}"')
    lines.append('json_out_config = "_config.json"')
    lines.append("")

    # Parameters
    lines.append("[parameters]")
    lines.append(f'prior_file = "{prior_file}"')
    lines.append(f'agent_file = "{agent_file}"')
    lines.append(f'summary_csv_prefix = "{summary_csv_prefix}"')
    lines.append(f"batch_size = {batch_size}")
    lines.append(f"randomize_smiles = {_toml_bool(randomize_smiles)}")
    lines.append(f"use_checkpoint = {_toml_bool(use_checkpoint)}")
    lines.append("")

    # Fragment inputs — REINVENT4 v4.x LinkInvent uses smiles_file (warheads.smi).
    # The file contains one pair per line: SMILES1|SMILES2, each with [*:n] points.
    # We write the file into the job directory alongside the TOML.
    # smiles_file path is resolved relative to working_dir at run time;
    # use a fixed name and write it next to the TOML via _write_linkinvent_smiles_file.
    smiles_filename = "warheads.smi"
    lines.append(f'smiles_file = "{smiles_filename}"')
    lines.append("")
    # Store fragment pair in TOML comment for transparency
    lines.append(f"# Fragments: {' | '.join(fragments)}")
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
# Unified dispatcher — block 08 addition
# ---------------------------------------------------------------------------

def generate_generation_toml(
    *,
    scorer_config: dict[str, Any],
    generation_config: dict[str, Any],
    prior_file: str,
    agent_file: Optional[str] = None,
    sigma: int = 128,
    learning_rate: float = 0.0001,
    max_score: float = 1.0,
    summary_csv_prefix: str = "results",
    chkpt_file: str = "stage.chkpt",
    tb_logdir: str = "tb_logs",
    diversity_filter_type: str = "IdenticalMurckoScaffold",
    diversity_filter_bucket_size: int = 25,
    diversity_filter_minscore: float = 0.4,
    termination: Optional[dict[str, Any]] = None,
) -> str:
    """Dispatch to the correct TOML generator based on generation_config['mode'].

    This is the single entry-point for generation job TOML creation.
    ``generation_config`` is the dict returned by
    ``mapper.ttp_to_generation_config()``.

    Parameters
    ----------
    generation_config:
        Dict with at least ``mode`` (reinvent | libinvent | linkinvent | mol2mol),
        ``batch_size``, ``n_steps``, ``device``, and mode-specific inputs.

    Returns
    -------
    str
        Complete TOML config string ready to write to disk.

    Raises
    ------
    ValueError
        For unknown modes or missing mode-specific inputs.
    """
    mode = generation_config.get("mode", "reinvent")
    device = generation_config.get("device", "cuda:0")
    batch_size = generation_config.get("batch_size", 128)
    n_steps = generation_config.get("n_steps", 100)

    common = dict(
        scorer_config=scorer_config,
        prior_file=prior_file,
        agent_file=agent_file,
        device=device,
        batch_size=batch_size,
        sigma=sigma,
        learning_rate=learning_rate,
        use_checkpoint=False,
        max_steps=n_steps,
        max_score=max_score,
        summary_csv_prefix=summary_csv_prefix,
        chkpt_file=chkpt_file,
        tb_logdir=tb_logdir,
        diversity_filter_type=diversity_filter_type,
        diversity_filter_bucket_size=diversity_filter_bucket_size,
        diversity_filter_minscore=diversity_filter_minscore,
        termination=termination,
    )

    if mode == "reinvent":
        return generate_reinvent_toml(**common)

    if mode == "libinvent":
        lib_input = generation_config.get("libinvent_input") or {}
        scaffold = lib_input.get("scaffold")
        if not scaffold:
            raise ValueError(
                "libinvent mode requires generation_config.libinvent_input.scaffold"
            )
        return generate_libinvent_toml(
            **common,
            scaffold=scaffold,
            decoration_positions=lib_input.get("decoration_positions"),
        )

    if mode == "linkinvent":
        link_input = generation_config.get("linkinvent_input") or {}
        fragments = link_input.get("fragments")
        if not fragments or len(fragments) < 2:
            raise ValueError(
                "linkinvent mode requires generation_config.linkinvent_input.fragments "
                "(at least 2 fragment SMILES)"
            )
        return generate_linkinvent_toml(**common, fragments=fragments)

    if mode == "mol2mol":
        mol2mol_input = generation_config.get("mol2mol_input") or {}
        seed_smiles = mol2mol_input.get("seed_smiles")
        if not seed_smiles:
            raise ValueError(
                "mol2mol mode requires generation_config.mol2mol_input.seed_smiles "
                "(one or more seed SMILES)"
            )
        return generate_reinvent_toml(**common, smiles_file="seed.smi")

    raise ValueError(
        f"Unknown generation mode {mode!r}. "
        "Valid values: 'reinvent', 'libinvent', 'linkinvent', 'mol2mol'."
    )
