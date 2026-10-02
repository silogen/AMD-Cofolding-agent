"""Stage 1 — Assembly and Model Selection.

Applies biological assembly transformations (BIOMT matrices) or selects
NMR conformer(s) from a multi-model structure.
"""

from __future__ import annotations

import logging
from typing import Any

import gemmi

from structure_prep.models import Stage1Profile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_assembly(
    structure: gemmi.Structure,
    assembly_id: str,
) -> gemmi.Assembly | None:
    """Return the first assembly whose name matches *assembly_id* (case-insensitive)."""
    target = assembly_id.strip().upper()
    for asm in structure.assemblies:
        if asm.name.strip().upper() == target:
            return asm
    return None


def _chain_name_map_before_after(
    before: gemmi.Model,
    after: gemmi.Model,
) -> dict[str, list[str]]:
    """Build a mapping of original → [new, …] chain names after assembly expansion.

    gemmi names copies by appending digits (e.g. 'A' → 'A1', 'A2', …).
    This is a best-effort name map for reporting; the canonical mapping is
    determined by the HowToNameCopiedChain strategy used at expansion time.
    """
    before_ids = {ch.name for ch in before}
    mapping: dict[str, list[str]] = {}

    for ch in after:
        name = ch.name
        # Try to find the original chain name by stripping trailing digits
        base = name.rstrip("0123456789")
        if not base:
            base = name
        if base in before_ids:
            mapping.setdefault(base, []).append(name)
        else:
            # Could not associate — record as-is
            mapping.setdefault(name, []).append(name)

    return mapping


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    structure: gemmi.Structure,
    profile: Stage1Profile,
) -> tuple[gemmi.Structure, dict[str, Any]]:
    """Stage 1: Assembly and Model Selection.

    Parameters
    ----------
    structure:
        Parsed gemmi Structure (from Stage 0).
    profile:
        Stage1Profile controlling which assembly and model to use.

    Returns
    -------
    (transformed_structure, stage1_report_dict)
        The (possibly transformed) Structure and a dict with audit keys:
        ``assembly_applied``, ``chain_rename_map``, ``model_selected``,
        ``n_models_retained``.
    """
    report: dict[str, Any] = {
        "assembly_applied": None,
        "chain_rename_map": {},
        "model_selected": profile.model_selection,
        "n_models_retained": len(structure),
    }

    # ------------------------------------------------------------------
    # 1. Biological assembly expansion
    # ------------------------------------------------------------------
    if profile.assembly_id is not None:
        assembly = _find_assembly(structure, profile.assembly_id)
        if assembly is None:
            available = [a.name for a in structure.assemblies]
            logger.warning(
                "Stage 1: requested assembly '%s' not found. "
                "Available assemblies: %s. Skipping expansion.",
                profile.assembly_id,
                available,
            )
            report["assembly_applied"] = None
        else:
            logger.info(
                "Stage 1: applying biological assembly '%s'.", assembly.name
            )

            original_model = structure[0]

            # Build the assembled model
            assembled_model = gemmi.make_assembly(
                assembly,
                original_model,
                gemmi.HowToNameCopiedChain.AddNumber,
            )

            # Capture the chain rename map before mutating structure
            chain_rename_map = _chain_name_map_before_after(
                original_model, assembled_model
            )

            # Replace all models in the structure with the assembled model
            # gemmi.Structure stores models in a list-like container.
            # We clear existing models and insert the assembled one.
            while len(structure) > 0:
                del structure[0]
            structure.add_model(assembled_model, pos=0)

            report["assembly_applied"] = assembly.name
            report["chain_rename_map"] = chain_rename_map
            logger.info(
                "Stage 1: assembly applied; chain rename map has %d entries.",
                len(chain_rename_map),
            )

    # ------------------------------------------------------------------
    # 2. NMR / multi-model selection
    # ------------------------------------------------------------------
    n_models = len(structure)
    model_selection = profile.model_selection.strip().lower()

    if n_models <= 1:
        # Nothing to do
        report["n_models_retained"] = len(structure)
        logger.debug("Stage 1: single-model structure; no model selection needed.")
        return structure, report

    if model_selection == "all":
        logger.info("Stage 1: keeping all %d models.", n_models)
        report["n_models_retained"] = n_models
        return structure, report

    # Parse the 1-based index from the profile string
    try:
        target_index_1based = int(model_selection)
    except ValueError:
        logger.warning(
            "Stage 1: cannot parse model_selection '%s' as integer; defaulting to model 1.",
            model_selection,
        )
        target_index_1based = 1

    # Clamp to valid range
    target_index_1based = max(1, min(target_index_1based, n_models))
    target_index_0based = target_index_1based - 1

    logger.info(
        "Stage 1: selecting model %d (0-based index %d) of %d.",
        target_index_1based,
        target_index_0based,
        n_models,
    )

    # Remove all models except the selected one.
    # Delete from the end to avoid index shifting.
    indices_to_remove = [
        i for i in range(n_models) if i != target_index_0based
    ]
    for i in sorted(indices_to_remove, reverse=True):
        del structure[i]

    report["model_selected"] = str(target_index_1based)
    report["n_models_retained"] = len(structure)

    logger.info(
        "Stage 1: %d model(s) retained.", report["n_models_retained"]
    )
    return structure, report


__all__ = ["run"]
