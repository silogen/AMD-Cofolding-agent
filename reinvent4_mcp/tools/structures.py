# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bring a structure or a molecule file into play, like the UI's form does.

The agent could name files on the shared volume but never put one there, read
a chain out of one, or build the single-chain template Boltz folds against:
those three steps lived only in the browser, so templates and "fold the
construct of this entry" were out of its reach.  ``prepare_structure`` fetches
a PDB entry, reports the chains of a structure, and writes a template, in the
same order the form does; ``read_molecules`` reads the seed files the form can
read.  Everything written lands under ``uploads/agent/`` on the shared volume,
so the paths are the ``/shared``-relative ones every other tool takes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import requests

from reinvent4_mcp.ttp.structure import (
    STRUCTURE_SUFFIXES,
    is_cif,
    pick_chain,
    shared_root,
    structure_chains,
)
from reinvent4_mcp.ttp.template import clean_chain, template_name

logger = logging.getLogger(__name__)

#: Where the mmCIF of a PDB entry comes from.
_RCSB_FILES = "https://files.rcsb.org/download/{pdb_id}.cif"
_TIMEOUT = (5, 30)

#: Everything this tool writes, so a user can tell it from their own uploads.
_AGENT_DIR = "uploads/agent"

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_MOLECULE_SUFFIXES = (".smi", ".smiles", ".csv", ".sdf", ".mol")
_SMILES_COLUMNS = ("smiles", "canonical_smiles", "smi", "input_smiles", "mol")
#: A last-resort reading of "is this a SMILES" for an image without RDKit.
_SMILES_SHAPE = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$%/\\.*]+$")
_ELEMENTS = re.compile(r"[BCNOPSFIbcnops]|Cl|Br")

#: Molecules returned by one read_molecules call.
_MAX_MOLECULES = 2000


def prepare_structure(
    pdb_id: Optional[str] = None,
    path: Optional[str] = None,
    chain: Optional[str] = None,
    template: bool = False,
) -> dict[str, Any]:
    """Fetch, inspect, and optionally make a Boltz template of a structure.

    Parameters
    ----------
    pdb_id:
        PDB entry to download to the shared volume, if it is not there yet.
    path:
        A structure already on the shared volume, instead of *pdb_id*.
    chain:
        Author chain id to report in full; needed for a template when the file
        holds more than one protein.
    template:
        Also write the cleaned single-chain mmCIF Boltz folds against.

    Returns
    -------
    dict
        ``structure_path`` and the ``chains`` of the file, the picked chain's
        ``sequence``, and ``template`` when one was written.
    """

    if bool(pdb_id) == bool(path):
        return {"error": "Give either 'pdb_id' (an entry to fetch) or 'path' (a file on the shared volume)."}

    try:
        source = _fetch_entry(pdb_id) if pdb_id else _shared_file(str(path))
    except ValueError as exc:
        return {"error": str(exc)}

    text = source.read_text(errors="replace")
    chains = structure_chains(source.name, text)
    if not chains:
        return {
            "error": f"No protein chain in {_relative(source)}.",
            "structure_path": _relative(source),
        }

    result: dict[str, Any] = {
        "structure_path": _relative(source),
        "format": "mmcif" if is_cif(source.name, text) else "pdb",
        "chains": [
            {
                "chain_id": c.id,
                "length": len(c.sequence),
                # The deposited construct, or only the residues with coordinates.
                "sequence_from": c.source,
            }
            for c in chains
        ],
    }
    if pdb_id:
        result["pdb_id"] = pdb_id.upper()

    if not chain and not template:
        result["next"] = (
            "Set target.structure_path to this path, with target.structure_chain, "
            "to fold that chain's construct instead of pasting a sequence. Call again "
            "with template=true to build a Boltz template of it."
        )
        return result

    try:
        picked = pick_chain(chains, chain, _relative(source))
    except ValueError as exc:
        return {**result, "error": str(exc)}

    result["structure_chain"] = picked.id
    result["sequence"] = picked.sequence
    result["sequence_length"] = len(picked.sequence)

    if template:
        try:
            result["template"] = _write_template(source, text, picked.id)
        except (ValueError, OSError) as exc:
            return {**result, "error": f"Could not build the template: {exc}"}

    return result


def read_molecules(path: str, limit: int = _MAX_MOLECULES) -> dict[str, Any]:
    """Read the SMILES of a .smi, .csv, .sdf or .mol file on the shared volume.

    Returns ``smiles`` (the molecules, ready for ``seed_smiles`` or
    ``score_molecules``), how many records were ``unreadable``, and whether the
    list was cut at *limit*.
    """

    try:
        resolved = _shared_file(path, _MOLECULE_SUFFIXES)
    except ValueError as exc:
        return {"error": str(exc)}

    text = resolved.read_text(errors="replace")
    suffix = resolved.suffix.lower()
    if suffix in (".sdf", ".mol"):
        smiles, unreadable = _sdf_smiles(text)
    elif suffix == ".csv":
        smiles, unreadable = _csv_smiles(text)
    else:
        smiles, unreadable = _smi_smiles(text)

    if not smiles:
        return {
            "error": f"No SMILES in {_relative(resolved)}" + (f" ({unreadable} unreadable records)" if unreadable else "."),
        }

    capped = max(1, min(int(limit), _MAX_MOLECULES))
    result: dict[str, Any] = {
        "path": _relative(resolved),
        "count": len(smiles),
        "smiles": smiles[:capped],
    }
    if unreadable:
        result["unreadable"] = unreadable
    if len(smiles) > capped:
        result["note"] = f"{len(smiles)} molecules in the file; the first {capped} are listed."
    return result


# ---------------------------------------------------------------------------
# files on the shared volume
# ---------------------------------------------------------------------------


def _relative(path: Path) -> str:
    """The path as the other tools take it: relative to the shared volume."""
    try:
        return path.relative_to(shared_root()).as_posix()
    except ValueError:
        return str(path)


def _shared_file(path: str, suffixes: tuple[str, ...] = STRUCTURE_SUFFIXES) -> Path:
    """An existing file of the shared volume, named by a ``/shared``-relative path."""
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("Give a path to a file on the shared volume.")
    root = shared_root().resolve()
    resolved = (root / cleaned.lstrip("/")).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"'{path}' is outside the shared volume.")
    if not resolved.is_file():
        raise ValueError(
            f"'{path}' is not on the shared volume; call list_shared_assets for the paths that are."
        )
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"'{path}' is not a {', '.join(suffixes)} file.")
    return resolved


def _agent_dir() -> Path:
    directory = shared_root() / _AGENT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _fetch_entry(pdb_id: Optional[str]) -> Path:
    """The mmCIF of a PDB entry on the shared volume, downloading it once."""
    code = (pdb_id or "").strip().upper()
    if not _PDB_ID.match(code):
        raise ValueError(f"'{pdb_id}' is not a PDB id (four characters, e.g. '4ZC9').")

    target = _agent_dir() / f"{code}.cif"
    if target.is_file() and target.stat().st_size:
        return target

    try:
        response = requests.get(_RCSB_FILES.format(pdb_id=code), timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise ValueError(f"Could not reach the PDB to fetch {code}: {exc}") from None
    if response.status_code == 404:
        raise ValueError(f"The PDB has no entry {code}.")
    if not response.ok:
        raise ValueError(f"The PDB refused to serve {code}: HTTP {response.status_code}.")

    # Written beside the final name and moved, so a half-written file is never read.
    partial = target.with_suffix(".cif.part")
    partial.write_bytes(response.content)
    partial.replace(target)
    logger.info("Fetched PDB entry %s to %s", code, target)
    return target


def _write_template(source: Path, text: str, chain_id: str) -> dict[str, Any]:
    """The cleaned single chain, as a ``target.templates`` entry."""
    name = template_name(source.name, chain_id)
    cleaned = clean_chain(source.name, text, chain_id, name)
    destination = _agent_dir() / name
    destination.write_text(cleaned.text)
    logger.info("Wrote template %s (%d atoms)", destination, cleaned.atoms)
    return {
        "path": _relative(destination),
        "chain_id": chain_id,
        "atoms": cleaned.atoms,
        "residues": cleaned.residues,
        "next": (
            "Add it to the TTP as target.templates: "
            f'[{{"path": "{_relative(destination)}", "chain_id": "{chain_id}"}}]. '
            'Add "force": true with a "threshold" in angstrom to hold the cofold near it.'
        ),
    }


# ---------------------------------------------------------------------------
# molecule files
# ---------------------------------------------------------------------------


def _is_smiles(value: str) -> bool:
    """Whether *value* reads as a molecule; RDKit decides when it is installed."""
    if not value or len(value) > 600:
        return False
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return bool(_SMILES_SHAPE.match(value)) and bool(_ELEMENTS.search(value))
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(value) is not None


def _smi_smiles(text: str) -> tuple[list[str], int]:
    """A .smi file: one molecule per line, the SMILES first, a name after it."""
    smiles: list[str] = []
    unreadable = 0
    for line in text.splitlines():
        token = re.split(r"[\s,]+", line.strip())[0]
        if not token or token.startswith("#"):
            continue
        if _is_smiles(token):
            smiles.append(token)
        else:
            unreadable += 1
    return smiles, unreadable


def _csv_fields(line: str, separator: str) -> list[str]:
    """One CSV line, with quoted fields."""
    fields: list[str] = []
    field: list[str] = []
    quoted = False
    index = 0
    while index < len(line):
        char = line[index]
        if quoted:
            if char == '"' and line[index + 1 : index + 2] == '"':
                field.append('"')
                index += 1
            elif char == '"':
                quoted = False
            else:
                field.append(char)
        elif char == '"':
            quoted = True
        elif char == separator:
            fields.append("".join(field).strip())
            field = []
        else:
            field.append(char)
        index += 1
    fields.append("".join(field).strip())
    return fields


def _csv_smiles(text: str) -> tuple[list[str], int]:
    """A CSV with a SMILES column: the one named so, else the first that reads as one."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], 0
    separator = "\t" if "\t" in lines[0] else ";" if ";" in lines[0] and "," not in lines[0] else ","
    rows = [_csv_fields(line, separator) for line in lines]
    header = [cell.lower() for cell in rows[0]]
    column = next((i for i, cell in enumerate(header) if cell in _SMILES_COLUMNS), -1)
    body = rows[1:]
    if column < 0:
        sample = rows[1] if len(rows) > 1 else rows[0]
        column = next((i for i, cell in enumerate(sample) if _is_smiles(cell)), -1)
        if column < 0:
            return [], 0
        # No header: the first row is a molecule too.
        if _is_smiles(rows[0][column] if column < len(rows[0]) else ""):
            body = rows
    smiles: list[str] = []
    unreadable = 0
    for row in body:
        cell = row[column] if column < len(row) else ""
        if not cell:
            continue
        if _is_smiles(cell):
            smiles.append(cell)
        else:
            unreadable += 1
    return smiles, unreadable


def _sdf_blocks(text: str) -> list[str]:
    """The molblocks of an SD file; a .mol file is one."""
    blocks: list[str] = []
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("$$$$"):
            end = next((i for i, l in enumerate(lines) if l.startswith("M  END")), -1)
            if end >= 0:
                blocks.append("\n".join(lines[: end + 1]))
            lines = []
        else:
            lines.append(line)
    end = next((i for i, l in enumerate(lines) if l.startswith("M  END")), -1)
    if end >= 0:
        blocks.append("\n".join(lines[: end + 1]))
    return blocks


def _sdf_smiles(text: str) -> tuple[list[str], int]:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return [], len(_sdf_blocks(text))
    RDLogger.DisableLog("rdApp.*")
    smiles: list[str] = []
    unreadable = 0
    for block in _sdf_blocks(text):
        mol = Chem.MolFromMolBlock(block)
        if mol is None:
            unreadable += 1
        else:
            smiles.append(Chem.MolToSmiles(mol))
    return smiles, unreadable
