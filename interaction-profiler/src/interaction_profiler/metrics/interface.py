"""Interface geometry metrics: buried surface area (BSA) and shape complementarity (Sc).

BSA (Shrake-Rupley via FreeSASA)
---------------------------------
    BSA = (SASA_binder_alone + SASA_target_alone − SASA_complex) / 2

    Individual SASAs are computed by writing temporary PDB files for the
    binder-only and target-only chains using MDAnalysis, then calling
    FreeSASA on each.

Interface residues
------------------
    A residue is classified as a member of the interface when its SASA
    contribution decreases by more than ``delta_sasa_threshold`` Å² upon
    complex formation (ΔSASA > 0).

Shape complementarity (simplified Sc)
--------------------------------------
    Inspired by Lawrence & Colman (1993).  Interfacial atom pairs within
    ``sc_cutoff_A`` Å are identified via cKDTree.  For each atom the
    outward normal is approximated as the unit vector from the entity's
    centre-of-mass to the atom.  Sc is the negative mean dot-product of
    paired normals (anti-parallel normals → high complementarity → Sc → 1).

    Requires ``scipy``; falls back to ``shape_complementarity=None`` with
    ``method_sc="unavailable"`` if scipy is not installed.

Dependencies
------------
    * freesasa  (SASA calculations)
    * MDAnalysis (chain selection and temporary PDB writing)
    * scipy      (optional; needed for cKDTree-based Sc calculation)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FREESASA_ALGORITHM_NAME = "freesasa_shrake_rupley"
_SC_CUTOFF_A = 4.5         # Å — search radius for interfacial atom pairs
_DELTA_SASA_THRESHOLD = 0.0  # Å² — minimum burial to call a residue interfacial


def _check_freesasa() -> None:
    """Raise ImportError with a helpful message if freesasa is missing."""
    try:
        import freesasa  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "freesasa is required for BSA calculation. "
            "Install with: pip install freesasa"
        ) from exc


def _check_mda() -> None:
    """Raise ImportError with a helpful message if MDAnalysis is missing."""
    try:
        import MDAnalysis  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "MDAnalysis is required for chain selection. "
            "Install with: pip install MDAnalysis"
        ) from exc


def _scipy_available() -> bool:
    """Return True if scipy.spatial.cKDTree is importable."""
    try:
        from scipy.spatial import cKDTree  # noqa: F401, PLC0415
        return True
    except ImportError:
        return False


def _build_chain_selection(chain_ids: list[str]) -> str:
    """Build an MDAnalysis atom-selection string for a list of chain IDs.

    Parameters
    ----------
    chain_ids:
        One or more chain identifiers (e.g. ``["A", "B"]``).

    Returns
    -------
    str
        MDAnalysis selection expression, e.g.
        ``"chainID A or chainID B"``.
    """
    return " or ".join(f"chainID {c}" for c in chain_ids)


def _write_temp_pdb(atom_group: object, suffix: str = ".pdb") -> str:
    """Write an MDAnalysis AtomGroup to a temporary PDB file.

    The caller is responsible for deleting the file.

    .. warning::
        PDB is a fixed-column format: atom serials are capped at 99,999,
        residue numbers at 9,999, and chain IDs at a single character.  For
        very large assemblies these overflow and FreeSASA may read back
        corrupted records.  Call :func:`_detect_pdb_overflow` first to surface
        the risk; MDAnalysis 2.x cannot write mmCIF, so this is the only
        available intermediate format here.

    Parameters
    ----------
    atom_group:
        MDAnalysis ``AtomGroup`` to write.
    suffix:
        File suffix (default ``".pdb"``).

    Returns
    -------
    str
        Absolute path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    atom_group.write(path)  # type: ignore[attr-defined]
    return path


def _detect_pdb_overflow(atom_group: object, label: str) -> list[str]:
    """Return warnings when writing *atom_group* to fixed-column PDB would overflow.

    The SASA inputs are written as temporary PDB files (MDAnalysis 2.x cannot
    write mmCIF).  PDB's fixed columns cannot represent more than 99,999 atoms,
    residue numbers above 9,999, or multi-character chain IDs — exactly the
    cases that arise in large multi-chain assemblies.  When any hold, FreeSASA
    can read back corrupted atom/residue/chain records, so per-residue interface
    assignments (and, in the extreme, BSA) become unreliable.  We report this
    rather than returning silently wrong numbers.
    """
    warns: list[str] = []
    try:
        n_atoms = len(atom_group)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return warns

    if n_atoms > 99_999:
        warns.append(
            f"{label}: {n_atoms} atoms exceeds the PDB 5-column atom-serial limit "
            "(99,999); interface residue assignments may be unreliable."
        )

    try:
        resids = atom_group.residues.resids  # type: ignore[attr-defined]
        if len(resids):
            maxr, minr = int(max(resids)), int(min(resids))
            if maxr > 9_999 or minr < -999:
                warns.append(
                    f"{label}: residue numbers out of PDB 4-column range "
                    f"({minr}..{maxr}); interface residue assignments may be unreliable."
                )
    except Exception:  # noqa: BLE001
        pass

    try:
        chain_ids = set(getattr(atom_group.atoms, "chainIDs", []))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        chain_ids = set()
    multichar = sorted(c for c in chain_ids if isinstance(c, str) and len(c.strip()) > 1)
    if multichar:
        warns.append(
            f"{label}: multi-character chain IDs {multichar} are truncated to one "
            "column in PDB; interface residues for these chains may be lost."
        )
    return warns


def _freesasa_total(path: str, probe_radius: float) -> float:
    """Compute the total SASA (Å²) for a structure file using FreeSASA.

    Parameters
    ----------
    path:
        Path to a PDB file.
    probe_radius:
        Solvent probe radius in Å (default 1.4 Å for water).

    Returns
    -------
    float
        Total SASA in Å².
    """
    import freesasa  # noqa: PLC0415

    params = freesasa.Parameters(
        {
            "algorithm": freesasa.ShrakeRupley,
            "probe-radius": probe_radius,
        }
    )
    result = freesasa.calc(freesasa.Structure(path), params)
    return result.totalArea()


def _freesasa_per_residue(path: str, probe_radius: float) -> dict[str, float]:
    """Compute per-residue SASA values for a structure file.

    Returns a mapping of residue label (``"RESNAME:RESNUM:CHAIN"``) to
    SASA in Å².

    Parameters
    ----------
    path:
        Path to a PDB file.
    probe_radius:
        Solvent probe radius in Å.

    Returns
    -------
    dict[str, float]
        Mapping of residue label → SASA (Å²).
    """
    import freesasa  # noqa: PLC0415

    params = freesasa.Parameters(
        {
            "algorithm": freesasa.ShrakeRupley,
            "probe-radius": probe_radius,
        }
    )
    result = freesasa.calc(freesasa.Structure(path), params)

    residue_areas: dict[str, float] = {}
    # residueAreas() returns {chain_id: {resnum_str: ResidueArea}}
    # ResidueArea has .total attribute (float, Å²)
    for chain_id, res_dict in result.residueAreas().items():
        for resnum_str, area_obj in res_dict.items():
            label = f"{resnum_str.strip()}:{chain_id}"
            residue_areas[label] = float(area_obj.total)
    return residue_areas


def _interface_residues(
    residues_complex: dict[str, float],
    residues_alone: dict[str, float],
    threshold: float = _DELTA_SASA_THRESHOLD,
) -> list[str]:
    """Identify interface residues as those with ΔSASA > threshold.

    Parameters
    ----------
    residues_complex:
        Per-residue SASA in the full complex (Å²).
    residues_alone:
        Per-residue SASA of the chain in isolation (Å²).
    threshold:
        Minimum ΔSASA (Å²) to classify a residue as interfacial.
        Default is 0 (any burial counts).

    Returns
    -------
    list[str]
        Sorted residue labels of interfacial residues.
    """
    interfacial: list[str] = []
    for res, sasa_alone in residues_alone.items():
        sasa_complex = residues_complex.get(res, sasa_alone)
        delta = sasa_alone - sasa_complex  # positive → buried in complex
        if delta > threshold:
            interfacial.append(res)
    return sorted(interfacial)


# ---------------------------------------------------------------------------
# Shape complementarity (simplified Lawrence–Colman Sc)
# ---------------------------------------------------------------------------

def _compute_sc(
    binder_coords: np.ndarray,
    target_coords: np.ndarray,
    cutoff: float = _SC_CUTOFF_A,
) -> float | None:
    """Compute simplified shape complementarity between two sets of atomic coordinates.

    Each atom's outward normal is approximated as the unit vector from
    the entity's centre-of-mass to that atom.  For every binder–target
    atom pair within ``cutoff`` Å, the dot product of their outward
    normals is computed.  Because complementary surfaces have
    anti-parallel normals, Sc is defined as the *negative* mean dot
    product, clipped to [−1, 1].

    Parameters
    ----------
    binder_coords:
        (N_binder, 3) array of binder interface atom positions.
    target_coords:
        (N_target, 3) array of target interface atom positions.
    cutoff:
        Distance cutoff (Å) for defining interfacial atom pairs.

    Returns
    -------
    float | None
        Sc score in [−1, 1], or ``None`` if fewer than 3 interfacial
        pairs are found (insufficient statistics).
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    if len(binder_coords) == 0 or len(target_coords) == 0:
        logger.debug("Shape complementarity: one or both atom sets are empty.")
        return None

    # Outward normals: unit vectors from entity CoM to each atom
    binder_com = binder_coords.mean(axis=0)
    target_com = target_coords.mean(axis=0)

    def _unit_normals(coords: np.ndarray, com: np.ndarray) -> np.ndarray:
        vecs = coords - com
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return vecs / norms

    n_binder = _unit_normals(binder_coords, binder_com)
    n_target = _unit_normals(target_coords, target_com)

    # Find all pairs within cutoff using cKDTree
    tree_target = cKDTree(target_coords)
    pairs = tree_target.query_ball_point(binder_coords, r=cutoff)  # list of lists

    dot_products: list[float] = []
    for i_b, neighbour_ids in enumerate(pairs):
        for i_t in neighbour_ids:
            dot_products.append(float(np.dot(n_binder[i_b], n_target[i_t])))

    if len(dot_products) < 3:
        logger.debug(
            "Shape complementarity: only %d interfacial pairs found (< 3); "
            "returning None.",
            len(dot_products),
        )
        return None

    # Anti-parallel normals (good complementarity) give dot ≈ -1.
    # Negate so that high complementarity → Sc near +1.
    sc = float(-np.mean(dot_products))
    return float(np.clip(sc, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_interface_geometry(
    structure_path: str,
    binder_chain_ids: list[str],
    target_chain_ids: list[str],
    probe_radius: float = 1.4,
) -> "InterfaceGeometry":
    """Compute buried surface area and shape complementarity for a binder–target complex.

    Parameters
    ----------
    structure_path:
        Absolute or relative path to the complex structure file (PDB or mmCIF
        format accepted by MDAnalysis and FreeSASA).
    binder_chain_ids:
        Chain identifiers belonging to the binder entity (e.g. ``["A"]`` for
        an antibody Fv or ``["C"]`` for a peptide binder).
    target_chain_ids:
        Chain identifiers belonging to the target protein (e.g. ``["B"]``).
    probe_radius:
        Solvent probe radius in Å used in the Shrake-Rupley SASA calculation.
        Default is 1.4 Å (water).

    Returns
    -------
    InterfaceGeometry
        On success, all fields are populated.  ``error`` is set on failure.

        * ``bsa_A2``                    — buried surface area in Å².
        * ``shape_complementarity``     — simplified Sc in [−1, 1].
        * ``n_interface_residues_*``    — number of interfacial residues.
        * ``interface_residues_*``      — sorted residue label lists.
        * ``method_bsa``                — always ``"freesasa_shrake_rupley"``.
        * ``probe_radius_A``            — probe radius used.
        * ``method_sc``                 — ``"dot_product_normals_ckdtree"``
                                          or ``"unavailable"`` (no scipy).
        * ``error``                     — ``None`` on success.

    Notes
    -----
    * Both FreeSASA and MDAnalysis must be installed.
    * Shape complementarity requires scipy; it degrades gracefully to
      ``None`` / ``method_sc="unavailable"`` when scipy is absent.
    * Temporary PDB files for isolated chains are written to the system
      temp directory and cleaned up regardless of success or failure.

    Examples
    --------
    >>> result = compute_interface_geometry(
    ...     "/data/complex.pdb",
    ...     binder_chain_ids=["A"],
    ...     target_chain_ids=["B"],
    ... )
    >>> result.bsa_A2
    1452.3
    """
    from interaction_profiler.schemas import InterfaceGeometry  # noqa: PLC0415

    temp_files: list[str] = []

    try:
        _check_freesasa()
        _check_mda()

        import MDAnalysis as mda  # noqa: PLC0415

        # ------------------------------------------------------------------ #
        # 1.  Load complex and select chain groups
        # ------------------------------------------------------------------ #
        universe = mda.Universe(structure_path)

        binder_sel = _build_chain_selection(binder_chain_ids)
        target_sel = _build_chain_selection(target_chain_ids)

        binder_ag = universe.select_atoms(binder_sel)
        target_ag = universe.select_atoms(target_sel)

        if len(binder_ag) == 0:
            return InterfaceGeometry(
                error=f"No atoms found for binder chains {binder_chain_ids!r} "
                f"in {structure_path!r}",
                probe_radius_A=probe_radius,
            )
        if len(target_ag) == 0:
            return InterfaceGeometry(
                error=f"No atoms found for target chains {target_chain_ids!r} "
                f"in {structure_path!r}",
                probe_radius_A=probe_radius,
            )

        # Surface fixed-column PDB overflow risks (big assemblies) rather than
        # letting FreeSASA read back silently corrupted records.
        overflow_warnings = (
            _detect_pdb_overflow(binder_ag, "binder")
            + _detect_pdb_overflow(target_ag, "target")
        )
        for _w in overflow_warnings:
            logger.warning("compute_interface_geometry: %s", _w)

        # ------------------------------------------------------------------ #
        # 2.  Write temporary PDB files for isolated chains
        # ------------------------------------------------------------------ #
        binder_pdb = _write_temp_pdb(binder_ag, suffix="_binder.pdb")
        target_pdb = _write_temp_pdb(target_ag, suffix="_target.pdb")
        temp_files.extend([binder_pdb, target_pdb])

        # ------------------------------------------------------------------ #
        # 3.  SASA calculations
        # ------------------------------------------------------------------ #
        # BSA formula: (SASA_binder_alone + SASA_target_alone - SASA_pair) / 2
        # where SASA_pair is SASA of ONLY the binder+target chains together
        # (NOT the full multi-chain complex — other chains would bury additional
        # surface and yield a negative BSA for multi-chain assemblies).
        binder_target_ag = universe.select_atoms(
            f"({binder_sel}) or ({target_sel})"
        )
        binder_target_pdb = _write_temp_pdb(binder_target_ag, suffix="_pair.pdb")
        temp_files.append(binder_target_pdb)

        sasa_pair        = _freesasa_total(binder_target_pdb, probe_radius)
        sasa_binder_alone = _freesasa_total(binder_pdb, probe_radius)
        sasa_target_alone = _freesasa_total(target_pdb, probe_radius)

        # BSA = (SASA_binder + SASA_target - SASA_pair) / 2  [always ≥ 0]
        bsa = max(0.0, (sasa_binder_alone + sasa_target_alone - sasa_pair) / 2.0)

        # ------------------------------------------------------------------ #
        # 4.  Interface residues (per-residue ΔSASA)
        # ------------------------------------------------------------------ #
        # KEY: compare SASA of each chain in the PAIR (binder+target) vs. in isolation.
        # Using the pair-only SASA (not the full multi-chain complex) ensures correct
        # ΔSASA even when the complex contains many other chains (e.g. 30-chain assemblies).
        full_complex_per_res = _freesasa_per_residue(binder_target_pdb, probe_radius)

        # Filter full-complex per-residue by binder/target chain IDs
        residues_binder_complex = {
            k: v for k, v in full_complex_per_res.items()
            if k.split(":")[-1] in binder_chain_ids
        }
        residues_target_complex = {
            k: v for k, v in full_complex_per_res.items()
            if k.split(":")[-1] in target_chain_ids
        }

        # Isolated SASA (binder and target each in their own temp PDB)
        residues_binder_alone = _freesasa_per_residue(binder_pdb, probe_radius)
        residues_target_alone = _freesasa_per_residue(target_pdb, probe_radius)

        iface_binder = _interface_residues(residues_binder_complex, residues_binder_alone)
        iface_target = _interface_residues(residues_target_complex, residues_target_alone)

        # ------------------------------------------------------------------ #
        # 5.  Shape complementarity
        # ------------------------------------------------------------------ #
        sc_score: float | None = None
        sc_method: str

        if _scipy_available():
            sc_method = "dot_product_normals_ckdtree"
            try:
                # Use only heavy atoms (exclude hydrogens) for the Sc calculation
                binder_heavy = universe.select_atoms(
                    f"({binder_sel}) and not name H*"
                )
                target_heavy = universe.select_atoms(
                    f"({target_sel}) and not name H*"
                )
                binder_coords = binder_heavy.positions  # (N, 3) ndarray
                target_coords = target_heavy.positions  # (M, 3) ndarray
                sc_score = _compute_sc(binder_coords, target_coords, cutoff=_SC_CUTOFF_A)
            except Exception as exc_sc:  # noqa: BLE001
                logger.warning(
                    "Shape complementarity calculation failed: %s. "
                    "Returning sc=None.",
                    exc_sc,
                )
                sc_score = None
        else:
            sc_method = "unavailable"
            logger.debug(
                "scipy not installed; shape complementarity set to None."
            )

        # ------------------------------------------------------------------ #
        # 6.  Assemble result
        # ------------------------------------------------------------------ #
        return InterfaceGeometry(
            bsa_A2=bsa,
            shape_complementarity=sc_score,
            n_interface_residues_binder=len(iface_binder),
            n_interface_residues_target=len(iface_target),
            interface_residues_binder=iface_binder,
            interface_residues_target=iface_target,
            method_bsa=_FREESASA_ALGORITHM_NAME,
            probe_radius_A=probe_radius,
            method_sc=sc_method,
            warnings=overflow_warnings,
        )

    except (ImportError, FileNotFoundError, ValueError) as exc:
        # Known error types — re-raise wrapped in the result model
        logger.warning("InterfaceGeometry computation failed: %s", exc)
        return InterfaceGeometry(
            error=str(exc),
            probe_radius_A=probe_radius,
            method_bsa=_FREESASA_ALGORITHM_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in compute_interface_geometry")
        return InterfaceGeometry(
            error=str(exc),
            probe_radius_A=probe_radius,
            method_bsa=_FREESASA_ALGORITHM_NAME,
        )
    finally:
        # Always clean up temporary files
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError as exc_cleanup:
                logger.debug("Failed to delete temp file %s: %s", path, exc_cleanup)


__all__ = [
    "compute_interface_geometry",
]
