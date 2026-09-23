"""Entity resolver: maps raw chain/ligand information to typed Entity objects.

Given a :class:`~interaction_profiler.io.structure.LoadedStructure` and the caller-supplied
entity declarations (binder chains, target chains, ligand comp-id, mode), this
module returns a fully-typed list of
:class:`~interaction_profiler.schemas.Entity` objects plus a list of human-readable
warnings when inference is ambiguous.

Supported modes
---------------
- ``ligand_only``    : no structure; empty entity list returned immediately.
- ``ligand_target``  : small-molecule binder vs. protein/nucleic target.
- ``binder_target``  : macromolecular binder (protein/peptide/antibody/nanobody)
                       vs. protein/nucleic target.
- ``hotspot_ensemble``: like ``ligand_target`` but flags that entities are from
                       a single representative structure in a multi-structure ensemble.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from interaction_profiler.schemas import Entity, EntityRole

if TYPE_CHECKING:
    from interaction_profiler.io.structure import LoadedStructure

logger = logging.getLogger(__name__)

# Polymer chain types recognised as nucleic / protein receptors
_POLYMER_TYPES: frozenset[str] = frozenset({"protein", "dna", "rna"})

# Valid explicit binder_role values (subset of EntityRole that makes sense for binders)
_VALID_BINDER_ROLES: frozenset[str] = frozenset({
    "protein", "peptide", "nanobody", "antibody", "dna", "rna",
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _chain_length(struct: "LoadedStructure", chain_id: str) -> int:
    """Return the number of protein residues in *chain_id*.

    Uses MDAnalysis selection ``(chainID {ch} or segid {ch}) and protein`` so
    it works regardless of whether the topology stores chain information as
    ``chainID`` or ``segid``.  Returns 0 on any selection error.
    """
    try:
        sel = struct.u.select_atoms(
            f"(chainID {chain_id} or segid {chain_id}) and protein"
        )
        return int(sel.n_residues)
    except Exception as exc:  # noqa: BLE001
        logger.debug("chain_length failed for chain %r: %s", chain_id, exc)
        return 0


def _classify_protein_by_length(n_residues: int) -> EntityRole:
    """Retained for backward compatibility — no longer called internally.

    Use the explicit ``binder_role`` parameter on :func:`resolve_entities`
    instead of length-based guessing.  This function is deprecated and will
    be removed in a future release.
    """
    return EntityRole.protein


def _resolve_protein_binder_role(
    struct: "LoadedStructure",
    chain_ids: list[str],
    warnings: list[str],
    binder_role: str | None = None,
) -> EntityRole:
    """Return :class:`~interaction_profiler.schemas.EntityRole` for a protein binder.

    Priority order
    --------------
    1. Explicit ``binder_role`` string → validated and used directly.
    2. Multiple chains → ``antibody`` (Fv/Fab heuristic; nanobodies are
       single-chain, Fv/Fab pairs are two-chain).
    3. Single chain, no explicit role → ``protein`` (no length guessing).

    Parameters
    ----------
    struct:
        Loaded structure (kept for API compatibility; not used internally).
    chain_ids:
        Binder chain IDs.
    warnings:
        Mutable list; ambiguity warnings are appended here.
    binder_role:
        Explicit role string from the caller (e.g. ``"nanobody"``,
        ``"peptide"``, ``"antibody"``).  When *None*, the heuristic above
        is applied.
    """
    # 1. Explicit role — validate and return
    if binder_role is not None:
        try:
            role = EntityRole(binder_role)
            logger.debug(
                "Binder chains %s: using explicit binder_role=%r → EntityRole.%s.",
                chain_ids, binder_role, role.value,
            )
            return role
        except ValueError:
            valid = sorted(_VALID_BINDER_ROLES)
            warnings.append(
                f"binder_role={binder_role!r} is not a valid EntityRole value. "
                f"Valid options: {valid}. Falling back to heuristic."
            )

    # 2. Multi-chain → antibody (Fv/Fab: heavy + light chain)
    if len(chain_ids) > 1:
        warnings.append(
            f"Binder spans {len(chain_ids)} chains ({', '.join(chain_ids)}). "
            "Inferred role 'antibody' (multi-chain Fv/Fab heuristic; "
            "nanobody is single-chain). "
            "Pass binder_role='protein' to override for non-antibody multi-chain binders."
        )
        return EntityRole.antibody

    # 3. Single chain, no explicit role → generic protein
    return EntityRole.protein


def _infer_entity_role_from_chain_type(chain_type: str) -> EntityRole:
    """Convert a :meth:`~interaction_profiler.io.structure.LoadedStructure.chain_types`
    string into an :class:`~interaction_profiler.schemas.EntityRole`.

    The protein sub-type (peptide / nanobody / antibody) requires length data and
    is NOT handled here; callers must use :func:`_resolve_protein_binder_role`
    for binder proteins.
    """
    mapping: dict[str, EntityRole] = {
        "protein":  EntityRole.protein,
        "dna":      EntityRole.dna,
        "rna":      EntityRole.rna,
        "ligand":   EntityRole.small_molecule,
        "cofactor": EntityRole.cofactor,
        "ion":      EntityRole.metal,
        "water":    EntityRole.water,
    }
    return mapping.get(chain_type, EntityRole.protein)


def _dominant_chain_type(types: list[str]) -> str:
    """Return the most common chain type from a list, with deterministic tie-breaking.

    Polymer types (protein, dna, rna) are preferred over non-polymer types.
    """
    if not types:
        return "protein"
    priority_order = ["protein", "dna", "rna", "ligand", "cofactor", "ion", "buffer", "water"]
    from collections import Counter
    counts = Counter(types)
    # Sort by (count desc, priority asc)
    def _sort_key(t: str) -> tuple[int, int]:
        prio = priority_order.index(t) if t in priority_order else len(priority_order)
        return (-counts[t], prio)
    return sorted(counts, key=_sort_key)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_entities(
    struct: "LoadedStructure | None",
    binder_chains: list[str] | None,
    target_chains: list[str] | None,
    ligand_comp_id: str | None,
    mode: str,
    binder_uniprot: str | None = None,
    target_uniprot: str | None = None,
    binder_role: str | None = None,
) -> tuple[list[Entity], list[str]]:
    """Resolve chain/ligand information into typed :class:`~interaction_profiler.schemas.Entity` objects.

    Parameters
    ----------
    struct:
        Loaded MDAnalysis structure.  May be *None* when ``mode="ligand_only"``.
    binder_chains:
        Explicit chain IDs for the binder (required for ``binder_target`` mode).
    target_chains:
        Explicit chain IDs for the target receptor.  When *None*, all
        protein/nucleic chains are used as the target (with an ambiguity warning
        if multiple protein chains exist).
    ligand_comp_id:
        Three-letter PDB component identifier for the small-molecule ligand (e.g.
        ``"LIG"``, ``"ATP"``).  *None* triggers auto-detection.
    mode:
        One of ``"ligand_only"``, ``"ligand_target"``, ``"binder_target"``,
        ``"hotspot_ensemble"``.
    binder_uniprot:
        Optional UniProt accession for the binder (stored on the Entity for
        downstream annotation fetching).
    target_uniprot:
        Optional UniProt accession for the target.
    binder_role:
        Explicit role for the binder entity, e.g. ``"nanobody"``, ``"peptide"``,
        ``"antibody"``, ``"protein"``.  When *None*:

        * 2 + binder chains → ``"antibody"`` (Fv/Fab heuristic, since nanobodies
          are single-chain and Fv/Fab pairs are two-chain).
        * 1 binder chain → ``"protein"`` (no length-based guessing).

        Providing this parameter avoids any heuristic and prevents ambiguity
        warnings.

    Returns
    -------
    (entities, warnings)
        *entities* – ordered list, binder first.
        *warnings* – human-readable strings describing ambiguous inferences;
        may be empty.

    Raises
    ------
    Does not raise.  All problems are captured as warnings and the best-effort
    entity list is returned.
    """
    warnings: list[str] = []
    entities: list[Entity] = []

    # ------------------------------------------------------------------
    # ligand_only: no structure needed, no entities
    # ------------------------------------------------------------------
    if mode == "ligand_only":
        logger.debug("mode=ligand_only: returning empty entity list.")
        return [], []

    # ------------------------------------------------------------------
    # All other modes require a structure
    # ------------------------------------------------------------------
    if struct is None:
        warnings.append(
            f"mode={mode!r} requires a loaded structure, but none was provided. "
            "No entities resolved."
        )
        return [], warnings

    # Build chain-type map once (cheap)
    try:
        ct: dict[str, str] = struct.chain_types()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"chain_types() failed: {exc}. Entity resolution may be incomplete.")
        ct = {}

    # ------------------------------------------------------------------
    # ligand_target / hotspot_ensemble
    # ------------------------------------------------------------------
    if mode in ("ligand_target", "hotspot_ensemble"):
        entities, warnings = _resolve_ligand_target(
            struct=struct,
            ct=ct,
            target_chains=target_chains,
            ligand_comp_id=ligand_comp_id,
            binder_uniprot=binder_uniprot,
            target_uniprot=target_uniprot,
            warnings=warnings,
        )

        if mode == "hotspot_ensemble":
            warnings.append(
                "mode='hotspot_ensemble': entities were resolved from a single "
                "representative structure. To compute ensemble-level hotspot occupancy, "
                "call resolve_entities() independently for each structure in the ensemble "
                "and aggregate results."
            )

        return entities, warnings

    # ------------------------------------------------------------------
    # binder_target
    # ------------------------------------------------------------------
    if mode == "binder_target":
        entities, warnings = _resolve_binder_target(
            struct=struct,
            ct=ct,
            binder_chains=binder_chains,
            target_chains=target_chains,
            binder_uniprot=binder_uniprot,
            target_uniprot=target_uniprot,
            binder_role=binder_role,
            warnings=warnings,
        )
        return entities, warnings

    # ------------------------------------------------------------------
    # Unknown mode
    # ------------------------------------------------------------------
    warnings.append(
        f"Unknown mode: {mode!r}. Expected one of: ligand_only, ligand_target, "
        "binder_target, hotspot_ensemble. No entities resolved."
    )
    return [], warnings


# ---------------------------------------------------------------------------
# Mode-specific helpers (private)
# ---------------------------------------------------------------------------

def _resolve_ligand_target(
    *,
    struct: "LoadedStructure",
    ct: dict[str, str],
    target_chains: list[str] | None,
    ligand_comp_id: str | None,
    binder_uniprot: str | None,
    target_uniprot: str | None,
    warnings: list[str],
) -> tuple[list[Entity], list[str]]:
    """Build binder (small molecule) + target (receptor) entities.

    Called for both ``ligand_target`` and ``hotspot_ensemble`` modes.
    """
    entities: list[Entity] = []

    # --- Detect small-molecule binder ---
    try:
        detected = struct.detect_ligands(resname=ligand_comp_id)
    except Exception as exc:  # noqa: BLE001
        detected = []
        warnings.append(f"Ligand detection failed: {exc}.")

    if not detected:
        warnings.append(
            f"No small-molecule ligand detected "
            f"(comp_id={ligand_comp_id!r}). "
            "Binder entity will be absent from the result."
        )
    else:
        sel = detected[0]  # best-ranked candidate
        if len(detected) > 1:
            extras = ", ".join(
                f"{s.resname}{s.resnum}.{s.chain}" for s in detected[1:4]
            )
            warnings.append(
                f"Multiple ligand candidates detected; using "
                f"{sel.resname}{sel.resnum}.{sel.chain} (best-ranked by contacts). "
                f"Other candidates: {extras}{'...' if len(detected) > 4 else ''}."
            )
        entities.append(
            Entity(
                entity_id="binder",
                role=EntityRole.small_molecule,
                chain_ids=[sel.chain],
                label=f"{sel.resname}{sel.resnum}.{sel.chain}",
                uniprot_accession=binder_uniprot,
            )
        )

    # --- Determine target chains ---
    if target_chains:
        tgt_chains = list(target_chains)
    else:
        tgt_chains = [ch for ch, tp in ct.items() if tp in _POLYMER_TYPES]
        protein_chains = [ch for ch, tp in ct.items() if tp == "protein"]
        if len(protein_chains) > 1:
            warnings.append(
                f"Ambiguous target: {len(protein_chains)} protein chains detected "
                f"({', '.join(protein_chains)}) with no 'target_chains' specified. "
                "Using all protein/nucleic chains as target receptor. "
                "Provide 'target_chains' to suppress this warning."
            )

    if tgt_chains:
        tgt_types = [ct.get(ch, "protein") for ch in tgt_chains]
        dominant = _dominant_chain_type(tgt_types)
        target_role = _infer_entity_role_from_chain_type(dominant)
        entities.append(
            Entity(
                entity_id="target",
                role=target_role,
                chain_ids=tgt_chains,
                label="receptor",
                uniprot_accession=target_uniprot,
            )
        )
    else:
        warnings.append(
            "Could not identify any protein/nucleic target chains in the structure. "
            "Target entity will be absent from the result."
        )

    return entities, warnings


def _resolve_binder_target(
    *,
    struct: "LoadedStructure",
    ct: dict[str, str],
    binder_chains: list[str] | None,
    target_chains: list[str] | None,
    binder_uniprot: str | None,
    target_uniprot: str | None,
    binder_role: str | None,
    warnings: list[str],
) -> tuple[list[Entity], list[str]]:
    """Build macromolecular binder + target entities.

    Called for ``binder_target`` mode.
    """
    entities: list[Entity] = []

    # --- Binder ---
    if not binder_chains:
        warnings.append(
            "mode='binder_target' requires 'binder_chains' but none were supplied. "
            "Binder entity will be absent."
        )
    else:
        binder_chain_types = [ct.get(ch, "protein") for ch in binder_chains]
        dominant_binder_type = _dominant_chain_type(binder_chain_types)

        if dominant_binder_type == "protein":
            resolved_binder_role = _resolve_protein_binder_role(
                struct, binder_chains, warnings, binder_role=binder_role,
            )
        else:
            resolved_binder_role = _infer_entity_role_from_chain_type(dominant_binder_type)

        entities.append(
            Entity(
                entity_id="binder_" + "_".join(binder_chains),
                role=resolved_binder_role,
                chain_ids=list(binder_chains),
                label="binder",
                uniprot_accession=binder_uniprot,
            )
        )

    # --- Target ---
    if not target_chains:
        warnings.append(
            "mode='binder_target' requires 'target_chains' but none were supplied. "
            "Target entity will be absent."
        )
    else:
        tgt_types = [ct.get(ch, "protein") for ch in target_chains]
        dominant_tgt = _dominant_chain_type(tgt_types)
        target_role = _infer_entity_role_from_chain_type(dominant_tgt)

        entities.append(
            Entity(
                entity_id="target_" + "_".join(target_chains),
                role=target_role,
                chain_ids=list(target_chains),
                label="target",
                uniprot_accession=target_uniprot,
            )
        )

    return entities, warnings


__all__ = ["resolve_entities"]
