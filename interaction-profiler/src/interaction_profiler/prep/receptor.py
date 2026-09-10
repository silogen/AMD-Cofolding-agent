"""Receptor preparation via PDBFixer.

Adds missing heavy atoms within existing residues and explicit hydrogens
at the requested pH for protein AND nucleic residues. Metals and waters
are preserved (no ``removeHeterogens``) because the detector needs them
for metal-coordination and water-bridge interactions. Crystal-gap
residues are NOT modelled in by default (extrapolated atoms bloat the
system and add no experimental support).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReceptorPrep:
    prepared_path: str | None = None
    atoms_added: int = 0
    hydrogens_added: bool = False
    ph: float = 7.4
    warnings: list[str] = field(default_factory=list)


def _ph_tag(ph: float) -> str:
    """pH as a cache-key token: 7.4 -> ``"7.40"``.

    Rebuilt from a range-checked integer count of hundredths, so the token is
    always plain digits and a decimal point even for a caller that reaches
    this module without going through the service's typed request model.
    """
    v = float(ph)
    if not 0.0 <= v <= 14.0:
        raise ValueError(f"ph must be between 0 and 14, got {ph!r}")
    return format(int(round(v * 100)) / 100, ".2f")


def _cache_target(src: Path, ph: float) -> tuple[Path, bool]:
    """Resolve where the prepared receptor is written.

    Returns ``(path, hashed)``. When ``IP_CACHE_DIR`` is set we key the file
    by a content+pH hash inside that dir (``hashed=True``) -- this keeps the
    caller's (possibly read-only) artifact directory clean, is safe under
    concurrency, and lets a warm cache be reused by content rather than
    mtime. Otherwise we fall back to ``<src>_prepared_ph<ph>.pdb`` next to the
    input: the pH is in the filename so a different pH cannot silently reuse a
    prepared file at the wrong protonation state (the non-hashed path only
    checks mtime).
    """
    cache_dir = os.environ.get("IP_CACHE_DIR")
    if not cache_dir:
        return src.with_name(f"{src.stem}_prepared_ph{_ph_tag(ph)}.pdb"), False
    h = hashlib.sha1()
    try:
        # Stream the file so hashing a large receptor does not load it all into memory.
        with src.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        h.update(str(src).encode())
    h.update(f"|ph={_ph_tag(ph)}".encode())
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{src.stem}_{h.hexdigest()[:12]}_prepared.pdb", True


def _atomic_write_pdb(topology, positions, out: Path) -> None:
    """Write a PDB atomically (temp file in the same dir + ``os.replace``) so
    concurrent preparers never observe a partially written file."""
    from openmm.app import PDBFile

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".pdb.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            PDBFile.writeFile(topology, positions, fh, keepIds=True)
        os.replace(tmp, out)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def prepare_receptor(
    input_path: str | Path,
    ph: float = 7.4,
    output_path: str | Path | None = None,
    cache: bool = True,
) -> ReceptorPrep:
    from pdbfixer import PDBFixer

    src = Path(input_path)
    prep = ReceptorPrep(ph=ph)
    if output_path:
        out, hashed = Path(output_path), False
    else:
        out, hashed = _cache_target(src, ph)

    # Reuse a cached prepared file (PDBFixer dominates runtime). A hash-keyed
    # cache is content-addressed so any existing file is valid; the
    # next-to-source fallback additionally checks mtime against the source.
    if cache and out.exists():
        fresh = hashed or (src.exists() and out.stat().st_mtime >= src.stat().st_mtime)
        if fresh:
            prep.prepared_path = str(out)
            prep.hydrogens_added = True
            prep.warnings.append("reused cached prepared receptor")
            return prep

    fixer = PDBFixer(filename=str(src))
    fixer.missingResidues = {}
    with contextlib.suppress(Exception):
        fixer.findNonstandardResidues()
    try:
        fixer.findMissingAtoms()
        prep.atoms_added = sum(len(v) for v in getattr(fixer, "missingAtoms", {}).values())
        fixer.addMissingAtoms()
    except Exception as exc:
        prep.warnings.append(f"addMissingAtoms failed: {exc}")
        fixer.missingAtoms = {}
        fixer.missingTerminals = {}

    try:
        fixer.addMissingHydrogens(pH=ph)
        prep.hydrogens_added = True
    except Exception as exc:
        prep.warnings.append(f"addMissingHydrogens failed: {exc}")

    try:
        _atomic_write_pdb(fixer.topology, fixer.positions, out)
        prep.prepared_path = str(out)
    except Exception as exc:
        prep.warnings.append(f"could not write prepared receptor: {exc}")

    return prep


__all__ = ["ReceptorPrep", "prepare_receptor"]
