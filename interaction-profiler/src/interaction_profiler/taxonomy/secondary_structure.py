"""Secondary-structure assignment for protein chains.

Provides :func:`assign_secondary_structure` which returns a per-residue
secondary-structure map ``{(chain_id, resnum): ss_state}`` where *ss_state* is
one of ``"helix"``, ``"sheet"``, ``"loop"``, ``"turn"``, or ``"unknown"``.

Strategy (priority order)
--------------------------
1. Parse ``HELIX`` / ``SHEET`` records directly from a PDB file.
   Fast, zero extra dependencies, no subprocess.
2. If the file is mmCIF, or if step 1 found no records, attempt
   Biopython DSSP via the ``mkdssp`` binary.
3. If DSSP fails or is unavailable, return an empty dict with
   ``method_used="none"``.

The function never raises — all errors are caught and logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSSP → canonical state mapping
# ---------------------------------------------------------------------------
_DSSP_TO_STATE: dict[str, str] = {
    "H": "helix",    # α-helix
    "G": "helix",    # 3₁₀-helix
    "I": "helix",    # π-helix
    "E": "sheet",    # extended β-strand
    "B": "sheet",    # isolated β-bridge
    "T": "turn",     # hydrogen-bonded turn
    "S": "loop",     # bend
    "C": "loop",     # coil / unclassified
    " ": "loop",     # coil (blank DSSP code)
    "-": "loop",     # explicit dash used in some parsers
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_secondary_structure(
    structure_path: str,
    chain_ids: list[str],
    method: str = "auto",
) -> tuple[dict[tuple[str, int], str], str]:
    """Assign secondary structure to residues in *chain_ids*.

    Parameters
    ----------
    structure_path:
        Path to the structure file (PDB or mmCIF).
    chain_ids:
        Chain identifiers to include in the output map.  An empty list means
        all chains present in the file.
    method:
        ``"auto"``  – PDB records first, then DSSP if records are absent.
        ``"pdb_records"`` – PDB records only; return empty if none found.
        ``"dssp"``  – Skip PDB records; run DSSP directly.

    Returns
    -------
    (ss_map, method_used)
        *ss_map* – ``{(chain_id, resnum): ss_state}`` (may be empty).
        *method_used* – ``"pdb_records"`` | ``"dssp"`` | ``"none"``.
    """
    path = Path(structure_path)
    chain_set: set[str] = set(chain_ids)  # empty → accept all
    is_cif = path.suffix.lower() in (".cif", ".mmcif", ".pdbx")

    # ------------------------------------------------------------------
    # Step 1: PDB HELIX/SHEET records
    # ------------------------------------------------------------------
    if method in ("auto", "pdb_records") and not is_cif:
        ss_map, found = _parse_pdb_records(path, chain_set)
        if found:
            logger.debug(
                "assign_secondary_structure: parsed %d residues from PDB records (%s).",
                len(ss_map), path.name,
            )
            return ss_map, "pdb_records"
        if method == "pdb_records":
            logger.debug(
                "assign_secondary_structure: no HELIX/SHEET records in %s.", path.name
            )
            return {}, "none"
        # else: auto → fall through to DSSP

    if is_cif and method == "pdb_records":
        logger.debug(
            "assign_secondary_structure: method='pdb_records' requested but file is "
            "mmCIF (%s); HELIX/SHEET records do not exist in CIF format.",
            path.name,
        )
        return {}, "none"

    # ------------------------------------------------------------------
    # Step 2: Biopython DSSP
    # ------------------------------------------------------------------
    ss_map, dssp_ok = _run_dssp(path, chain_set)
    if dssp_ok:
        return ss_map, "dssp"

    # ------------------------------------------------------------------
    # Step 3: Nothing worked
    # ------------------------------------------------------------------
    logger.warning(
        "assign_secondary_structure: could not assign secondary structure for %s "
        "(no PDB records, DSSP unavailable or failed). Returning empty map.",
        path.name,
    )
    return {}, "none"


# ---------------------------------------------------------------------------
# Private: PDB HELIX / SHEET parsing
# ---------------------------------------------------------------------------

def _parse_pdb_records(
    path: Path,
    chain_set: set[str],
) -> tuple[dict[tuple[str, int], str], bool]:
    """Parse HELIX and SHEET records from a PDB file.

    PDB fixed-column format (1-based columns → 0-based Python slices):

    HELIX record
        initChainID : col 20      → ``line[19:20]``
        initSeqNum  : cols 22-25  → ``line[21:25]``
        endChainID  : col 33      → ``line[32:33]``
        endSeqNum   : cols 35-38  → ``line[34:38]``

    SHEET record
        initChainID : col 22      → ``line[21:22]``
        initSeqNum  : cols 23-26  → ``line[22:26]``
        endChainID  : col 33      → ``line[32:33]``
        endSeqNum   : cols 34-37  → ``line[33:37]``

    Returns
    -------
    (ss_map, has_records)
        *has_records* is True if at least one HELIX or SHEET record was parsed
        successfully (even if it yielded zero residues after chain filtering).
    """
    ss_map: dict[tuple[str, int], str] = {}
    has_records: bool = False

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        logger.warning("_parse_pdb_records: cannot read %s: %s", path, exc)
        return {}, False

    for line in text.splitlines():
        record = line[:6].rstrip()
        if record not in ("HELIX", "SHEET"):
            continue

        padded = line.ljust(80)

        if record == "HELIX":
            init_chain = padded[19:20].strip()  # col 20
            end_chain  = padded[32:33].strip()  # col 33
            try:
                init_seq = int(padded[21:25])   # cols 22-25
                end_seq  = int(padded[34:38])   # cols 35-38
            except ValueError:
                continue
            has_records = True
            for chain in {init_chain, end_chain}:
                if not chain:
                    continue
                if chain_set and chain not in chain_set:
                    continue
                for resnum in range(init_seq, end_seq + 1):
                    ss_map[(chain, resnum)] = "helix"

        else:  # SHEET
            init_chain = padded[21:22].strip()  # col 22
            end_chain  = padded[32:33].strip()  # col 33
            try:
                init_seq = int(padded[22:26])   # cols 23-26
                end_seq  = int(padded[33:37])   # cols 34-37
            except ValueError:
                continue
            has_records = True
            for chain in {init_chain, end_chain}:
                if not chain:
                    continue
                if chain_set and chain not in chain_set:
                    continue
                for resnum in range(init_seq, end_seq + 1):
                    ss_map[(chain, resnum)] = "sheet"

    return ss_map, has_records


# ---------------------------------------------------------------------------
# Private: Biopython DSSP
# ---------------------------------------------------------------------------

def _run_pydssp(
    path: Path,
    chain_set: set[str],
) -> tuple[dict[tuple[str, int], str], bool]:
    """Run pydssp (pure-Python DSSP) — no binary required.

    Uses MDAnalysis to extract backbone coordinates per chain and
    pydssp to assign secondary structure.

    Returns
    -------
    (ss_map, success)
    """
    try:
        import pydssp  # type: ignore[import]
    except ImportError:
        logger.debug("_run_pydssp: pydssp not installed; skipping.")
        return {}, False

    ss_map: dict[tuple[str, int], str] = {}
    try:
        import MDAnalysis as mda
        u = mda.Universe(str(path))
        # Enumerate all unique chain IDs present in the structure
        try:
            all_chain_ids: set[str] = set(u.atoms.chainIDs.tolist())
        except Exception:
            all_chain_ids = set()
        target_chain_ids = chain_set if chain_set else all_chain_ids
        for chain_id in target_chain_ids:
            # Select protein residues for this chain
            ag = u.select_atoms(f"(chainID {chain_id} or segid {chain_id}) and protein")
            if ag.n_atoms == 0:
                continue
            residues = list(ag.residues)
            resnums = [int(r.resnum) for r in residues]

            # Build (N_res, 4, 3) array: N, CA, C, O backbone coords
            try:
                coords_list = []
                for res in residues:
                    try:
                        n  = res.atoms.select_atoms("name N")[0].position
                        ca = res.atoms.select_atoms("name CA")[0].position
                        c  = res.atoms.select_atoms("name C")[0].position
                        o  = res.atoms.select_atoms("name O")[0].position
                        coords_list.append([n, ca, c, o])
                    except (IndexError, Exception):
                        coords_list.append([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])

                import numpy as np
                coords_arr = np.array(coords_list, dtype=np.float64)  # (N, 4, 3)
                ss_codes = pydssp.assign(coords_arr)

                _PYDSSP_MAP = {
                    "H": "helix", "G": "helix", "I": "helix",
                    "E": "sheet", "B": "sheet",
                    "T": "turn",
                    "S": "loop", "C": "loop", "-": "loop", " ": "loop",
                }
                for resnum, code in zip(resnums, ss_codes):
                    if chain_set and chain_id not in chain_set:
                        continue
                    state = _PYDSSP_MAP.get(str(code).strip(), "unknown")
                    ss_map[(chain_id, int(resnum))] = state
            except Exception as exc:  # noqa: BLE001
                logger.warning("_run_pydssp: coord extraction failed for chain %s: %s", chain_id, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("_run_pydssp: failed for %s: %s", path.name, exc)
        return {}, False

    logger.debug("_run_pydssp: assigned %d residues via pydssp (%s).", len(ss_map), path.name)
    return ss_map, len(ss_map) > 0


def _run_dssp(
    path: Path,
    chain_set: set[str],
) -> tuple[dict[tuple[str, int], str], bool]:
    """Run Biopython DSSP via mkdssp binary (if installed), else try pydssp.

    Priority: mkdssp binary → pydssp (pure Python).

    Returns
    -------
    (ss_map, success)
        *success* is False when both methods are unavailable or crash.
    """
    try:
        from Bio.PDB import PDBParser, DSSP  # type: ignore[import]
    except ImportError:
        logger.debug("_run_dssp: biopython not installed; trying pydssp.")
        return _run_pydssp(path, chain_set)

    try:
        parser = PDBParser(QUIET=True)
        model = parser.get_structure("s", str(path))[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("_run_dssp: Biopython PDBParser failed for %s: %s", path.name, exc)
        return _run_pydssp(path, chain_set)

    try:
        dssp = DSSP(model, str(path), dssp="mkdssp")
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "_run_dssp: mkdssp unavailable for %s (%s); falling back to pydssp.", path.name, exc,
        )
        return _run_pydssp(path, chain_set)

    ss_map: dict[tuple[str, int], str] = {}
    try:
        for key in dssp:
            chain_id, (_, resnum, _) = key
            if chain_set and chain_id not in chain_set:
                continue
            dssp_code = dssp[key][2]  # index 2 is the ss code
            state = _DSSP_TO_STATE.get(dssp_code, "unknown")
            ss_map[(chain_id, int(resnum))] = state
    except Exception as exc:  # noqa: BLE001
        logger.warning("_run_dssp: error iterating DSSP results for %s: %s", path.name, exc)
        return _run_pydssp(path, chain_set)

    logger.debug(
        "_run_dssp: assigned secondary structure to %d residues via mkdssp (%s).",
        len(ss_map), path.name,
    )
    return ss_map, True


__all__ = ["assign_secondary_structure"]
