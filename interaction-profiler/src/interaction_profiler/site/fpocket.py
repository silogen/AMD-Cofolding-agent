"""fpocket subprocess wrapper and output parser.

fpocket finds protein cavities via Voronoi tessellation + alpha-sphere
algorithm (Le Guilloux et al., 2009).  This module:

1. Locates the fpocket binary (PATH or FPOCKET_BIN env var).
2. Runs fpocket on a PDB file in a temporary directory.
3. Parses the ``*_info.txt`` output to extract per-pocket properties.
4. Returns a dataclass with ranked pockets and version info.

Graceful degradation
--------------------
If the fpocket binary is not available the function returns a
``FpocketResult`` with ``pockets=[]`` and ``error`` set — the calling
pipeline records this as a warning and continues without pocket geometry.
The service starts and serves all other modes normally.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FPOCKET_BIN: str = os.environ.get("FPOCKET_BIN", "fpocket")
_TIMEOUT: int = int(os.environ.get("FPOCKET_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FpocketPocket:
    """Properties of one fpocket-detected pocket."""
    rank: int                          # 1 = top-ranked (most druggable)
    volume_A3: float | None            # Voronoi approximation
    druggability_score: float | None   # fpocket composite druggability (0–1)
    polarity_score: float | None       # fraction of polar alpha-spheres
    hydrophobicity_score: float | None
    charge_score: float | None
    flexibility: float | None          # alpha-sphere radius variance
    residues: list[str] = field(default_factory=list)   # "RESNAME:NUM:CHAIN"
    centroid_A: list[float] = field(default_factory=list)  # [x, y, z] mean


@dataclass
class FpocketResult:
    """Parsed output from one fpocket run."""
    pockets: list[FpocketPocket] = field(default_factory=list)
    fpocket_version: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def fpocket_available() -> bool:
    """Return True if the fpocket binary is on PATH or FPOCKET_BIN is set."""
    return shutil.which(_FPOCKET_BIN) is not None


def _get_version() -> str | None:
    """Try to extract the fpocket version string."""
    try:
        result = subprocess.run(
            [_FPOCKET_BIN, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        m = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout + result.stderr)
        return m.group(1) if m else "unknown"
    except Exception:
        return None


def run_fpocket(
    pdb_path: str,
    chain_ids: list[str] | None = None,
    min_volume: float = 100.0,
    max_pockets: int = 10,
) -> FpocketResult:
    """Run fpocket on *pdb_path* and return parsed pocket data.

    Parameters
    ----------
    pdb_path:
        Absolute path to the PDB file to analyse.
    chain_ids:
        If provided, only residues from these chains are considered for
        pocket labelling (fpocket still runs on the full structure).
    min_volume:
        Skip pockets smaller than this volume in Å³.
    max_pockets:
        Return at most this many pockets (ranked by druggability).

    Returns
    -------
    FpocketResult
        ``pockets`` sorted by druggability descending (rank 1 = best).
        ``error`` is set and ``pockets=[]`` when fpocket is unavailable.
    """
    if not fpocket_available():
        return FpocketResult(
            error=(
                "fpocket binary not found. "
                "Install fpocket (https://github.com/Discngine/fpocket) or set "
                "FPOCKET_BIN to the full path. "
                "Pocket geometry will not be computed."
            )
        )

    version = _get_version()
    pdb_path = str(Path(pdb_path).resolve())

    with tempfile.TemporaryDirectory(prefix="fpocket_") as tmp:
        # fpocket writes output into a <stem>_out/ directory relative to
        # where the PDB is; copy PDB into the temp dir first.
        pdb_stem = Path(pdb_path).stem
        tmp_pdb = Path(tmp) / Path(pdb_path).name
        shutil.copy(pdb_path, tmp_pdb)

        cmd = [_FPOCKET_BIN, "-f", str(tmp_pdb)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return FpocketResult(
                fpocket_version=version,
                error=f"fpocket timed out after {_TIMEOUT}s.",
            )
        except Exception as exc:
            return FpocketResult(
                fpocket_version=version,
                error=f"fpocket subprocess failed: {exc}",
            )

        if proc.returncode != 0:
            logger.warning(
                "fpocket returned exit code %d:\n%s",
                proc.returncode, proc.stderr[-2000:],
            )

        # Locate output directory
        out_dir = Path(tmp) / f"{pdb_stem}_out"
        if not out_dir.is_dir():
            return FpocketResult(
                fpocket_version=version,
                error=f"fpocket output directory not found at {out_dir}.",
            )

        # Parse pockets
        pockets = _parse_fpocket_info(out_dir, chain_ids, min_volume)

    # Sort by druggability descending, re-rank
    pockets.sort(key=lambda p: -(p.druggability_score or 0.0))
    for i, p in enumerate(pockets[:max_pockets], start=1):
        p.rank = i

    return FpocketResult(pockets=pockets[:max_pockets], fpocket_version=version)


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

def _parse_fpocket_info(
    out_dir: Path,
    chain_ids: list[str] | None,
    min_volume: float,
) -> list[FpocketPocket]:
    """Parse fpocket's ``*_info.txt`` summary file."""
    info_files = list(out_dir.glob("*_info.txt"))
    if not info_files:
        logger.debug("No *_info.txt found under %s", out_dir)
        return []

    info_text = info_files[0].read_text(errors="replace")
    pockets: list[FpocketPocket] = []

    # fpocket _info.txt has blocks like:
    #   Pocket N :
    #     Score:              0.XXX
    #     Druggability Score: 0.XXX
    #     Volume:             XXX.XX
    #     ...
    #     Residues:  RES NUM CHAIN, ...
    pocket_blocks = re.split(r"Pocket\s+(\d+)\s*:", info_text)[1:]

    for i in range(0, len(pocket_blocks) - 1, 2):
        rank_str = pocket_blocks[i].strip()
        body = pocket_blocks[i + 1]
        try:
            rank = int(rank_str)
        except ValueError:
            continue

        def _extract(pattern: str) -> float | None:
            m = re.search(pattern, body, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
            return None

        volume = _extract(r"Volume\s*:\s*([\d.]+)")
        if volume is not None and volume < min_volume:
            continue

        druggability  = _extract(r"Druggability Score\s*:\s*([\d.]+)")
        polarity      = _extract(r"Polarity Score\s*:\s*([\d.]+)")
        hydrophobicity = _extract(r"Hydrophobicity Score\s*:\s*([\d.]+)")
        charge        = _extract(r"Charge Score\s*:\s*([\d.]+)")
        flexibility   = _extract(r"Flexibility\s*:\s*([\d.]+)")

        # Residues
        residues: list[str] = []
        res_match = re.search(r"Residue IDs\s*:\s*([^\n]+)", body, re.IGNORECASE)
        if res_match:
            raw_res = res_match.group(1).strip()
            for part in re.split(r"[,\s]+", raw_res):
                part = part.strip()
                if part:
                    # Filter by chain if requested
                    if chain_ids:
                        # fpocket labels residues as "RES:NUM:CHAIN" or "RESNUM:CHAIN"
                        parts = part.split(":")
                        chain = parts[-1] if len(parts) > 1 else ""
                        if chain and chain not in chain_ids:
                            continue
                    residues.append(part)

        pockets.append(FpocketPocket(
            rank=rank,
            volume_A3=volume,
            druggability_score=druggability,
            polarity_score=polarity,
            hydrophobicity_score=hydrophobicity,
            charge_score=charge,
            flexibility=flexibility,
            residues=residues,
        ))

    return pockets


def pocket_to_dict(pocket: FpocketPocket) -> dict[str, Any]:
    """Convert an FpocketPocket to a JSON-serializable dict."""
    return {
        "rank":                pocket.rank,
        "volume_A3":           pocket.volume_A3,
        "druggability_score":  pocket.druggability_score,
        "polarity_score":      pocket.polarity_score,
        "hydrophobicity_score": pocket.hydrophobicity_score,
        "charge_score":        pocket.charge_score,
        "flexibility":         pocket.flexibility,
        "residues":            pocket.residues,
        "centroid_A":          pocket.centroid_A,
    }
