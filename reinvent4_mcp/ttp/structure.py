# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Protein chain sequences of a PDB or mmCIF file, for ``target.structure_path``.

Pure Python (the agent image has neither gemmi nor Biopython).  A chain's
sequence is the deposited construct when the file has one (mmCIF
``entity_poly`` / ``entity_poly_seq``, PDB ``SEQRES``), else the residues
with coordinates in the first model.  Common modified residues read as the
residue they modify (MSE -> M); anything else non-standard is dropped, as
``TargetConfig`` takes the 20 standard amino acids only.

Mirrors the browser's preview (services/ui/frontend/src/lib/sharedFiles.ts)
so the chain and length the form shows are the ones the server folds.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

AMINO = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

MODIFIED = {
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "CME": "C", "CAS": "C",
    "CSD": "C", "OCS": "C", "HYP": "P", "MLY": "K", "KCX": "K", "LLP": "K", "M3L": "K",
}

STRUCTURE_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")

_MIN_RESIDUES = 10
_STANDARD = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


@dataclass(frozen=True)
class ProteinChain:
    """A protein chain: its id (author chain id where the file has one) and sequence."""

    id: str
    sequence: str
    source: str  # "construct" or "atoms"


def letter(code: str) -> Optional[str]:
    code = code.upper()
    return AMINO.get(code) or MODIFIED.get(code)


def _protein(residues: list[str]) -> Optional[str]:
    """Residue names as a sequence, or None when they are mostly not amino acids."""
    letters = [l for l in (letter(r) for r in residues) if l]
    if len(letters) >= _MIN_RESIDUES and len(letters) * 2 >= len(residues):
        return "".join(letters)
    return None


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


def pdb_chains(text: str) -> list[ProteinChain]:
    """Chains of a PDB file: SEQRES when present, else ATOM records of the first model."""
    seqres: dict[str, list[str]] = {}
    atoms: dict[str, list[str]] = {}
    last: dict[str, str] = {}
    for line in text.splitlines():
        record = line[:6]
        if record == "SEQRES":
            chain = line[11:12].strip() or "_"
            seqres.setdefault(chain, []).extend(line[19:].split())
        elif record in ("ATOM  ", "HETATM"):
            resname = line[17:20].strip()
            if record == "HETATM" and resname not in MODIFIED:
                continue
            chain = line[21:22].strip() or "_"
            key = line[22:27]
            if last.get(chain) == key:
                continue
            last[chain] = key
            atoms.setdefault(chain, []).append(resname)
        elif record == "ENDMDL":
            break
    chains: list[ProteinChain] = []
    for chain in dict.fromkeys([*seqres, *atoms]):
        construct = _protein(seqres.get(chain, []))
        if construct:
            chains.append(ProteinChain(chain, construct, "construct"))
            continue
        observed = _protein(atoms.get(chain, []))
        if observed:
            chains.append(ProteinChain(chain, observed, "atoms"))
    return chains


# ---------------------------------------------------------------------------
# mmCIF
# ---------------------------------------------------------------------------

_CIF_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+""")


def _cif_tokens(text: str) -> Iterator[tuple[str, bool]]:
    """(value, quoted) tokens; a ;-delimited text field is one quoted token."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(";"):
            parts = [line[1:]]
            i += 1
            while i < len(lines) and not lines[i].startswith(";"):
                parts.append(lines[i])
                i += 1
            i += 1
            yield "\n".join(parts), True
            continue
        for match in _CIF_TOKEN.finditer(line):
            token = match.group(0)
            if token.startswith("#"):
                break
            if token[0] in "'\"":
                yield token[1:-1], True
            else:
                yield token, False
        i += 1


def _split_tag(tag: str) -> tuple[str, str]:
    category, _, field = tag[1:].partition(".")
    return category.lower(), field.lower()


def read_cif(text: str, categories: tuple[str, ...]) -> dict[str, list[dict[str, Optional[str]]]]:
    """Rows of some mmCIF categories, fields keyed in lower case; '.' and '?' read as None."""
    wanted = {c.lower() for c in categories}
    out: dict[str, list[dict[str, Optional[str]]]] = {}
    tags: list[str] = []
    values: list[Optional[str]] = []
    mode = "none"  # "tags" while reading a loop's header, "values" after
    keep = False
    pending: Optional[str] = None

    def flush() -> None:
        nonlocal tags, values, mode
        if tags and keep:
            fields = [_split_tag(t)[1] for t in tags]
            rows = out.setdefault(_split_tag(tags[0])[0], [])
            for start in range(0, len(values) - len(fields) + 1, len(fields)):
                rows.append(dict(zip(fields, values[start : start + len(fields)])))
        tags, values, mode = [], [], "none"

    for token, quoted in _cif_tokens(text):
        value = None if not quoted and token in (".", "?") else token
        if pending is not None:
            category, field = _split_tag(pending)
            if category in wanted:
                rows = out.setdefault(category, [{}])
                rows[0][field] = value
            pending = None
            continue
        if not quoted and token.lower() == "loop_":
            flush()
            mode = "tags"
            continue
        if not quoted and token.startswith("_"):
            if mode == "tags":
                tags.append(token)
                keep = _split_tag(tags[0])[0] in wanted
                continue
            flush()
            pending = token
            continue
        if not quoted and token[:5].lower() == "data_":
            flush()
            continue
        if mode in ("tags", "values"):
            mode = "values"
            if keep:
                values.append(value)
    flush()
    return out


def cif_chains(text: str) -> list[ProteinChain]:
    """Chains of an mmCIF file: entity_poly, else entity_poly_seq, else atom_site of the first model."""
    tables = read_cif(text, ("entity_poly", "entity_poly_seq", "struct_asym"))
    asyms = tables.get("struct_asym", [])

    def chains_of(entity: str, strands: Optional[str]) -> list[str]:
        if strands:
            return [s.strip() for s in strands.split(",") if s.strip()]
        return [a["id"] for a in asyms if a.get("entity_id") == entity and a.get("id")]

    poly_seq: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for row in tables.get("entity_poly_seq", []):
        entity = row.get("entity_id") or ""
        key = (entity, row.get("num") or "")
        # Microheterogeneity lists several monomers at one position: keep the first.
        if not row.get("mon_id") or key in seen:
            continue
        seen.add(key)
        poly_seq.setdefault(entity, []).append(row["mon_id"])

    chains: list[ProteinChain] = []
    polys = tables.get("entity_poly", [])
    if polys:
        for poly in polys:
            entity = poly.get("entity_id") or ""
            if poly.get("type") and "polypeptide" not in poly["type"].lower():
                continue
            canonical = _STANDARD.sub("", (poly.get("pdbx_seq_one_letter_code_can") or "").upper())
            sequence = canonical if len(canonical) >= _MIN_RESIDUES else _protein(poly_seq.get(entity, []))
            if sequence:
                chains.extend(ProteinChain(c, sequence, "construct") for c in chains_of(entity, poly.get("pdbx_strand_id")))
    else:
        for entity, residues in poly_seq.items():
            sequence = _protein(residues)
            if sequence:
                chains.extend(ProteinChain(c, sequence, "construct") for c in chains_of(entity, None))
    if chains:
        return chains

    atoms = read_cif(text, ("atom_site",)).get("atom_site", [])
    model = atoms[0].get("pdbx_pdb_model_num") if atoms else None
    residues: dict[str, list[str]] = {}
    last: dict[str, str] = {}
    for atom in atoms:
        if model is not None and atom.get("pdbx_pdb_model_num") != model:
            break
        resname = atom.get("label_comp_id") or atom.get("auth_comp_id") or ""
        if atom.get("group_pdb") == "HETATM" and resname.upper() not in MODIFIED:
            continue
        chain = atom.get("auth_asym_id") or atom.get("label_asym_id") or "_"
        key = f"{atom.get('auth_seq_id') or atom.get('label_seq_id') or ''}{atom.get('pdbx_pdb_ins_code') or ''}"
        if last.get(chain) == key:
            continue
        last[chain] = key
        residues.setdefault(chain, []).append(resname)
    for chain, names in residues.items():
        sequence = _protein(names)
        if sequence:
            chains.append(ProteinChain(chain, sequence, "atoms"))
    return chains


def is_cif(name: str, text: str) -> bool:
    lower = name.lower()
    if lower.endswith((".cif", ".mmcif")):
        return True
    if lower.endswith((".pdb", ".ent")):
        return False
    return text.lstrip().startswith("data_")


def structure_chains(name: str, text: str) -> list[ProteinChain]:
    return cif_chains(text) if is_cif(name, text) else pdb_chains(text)


def pick_chain(chains: list[ProteinChain], chain_id: Optional[str], name: str) -> ProteinChain:
    """The chain *chain_id*, or the only protein of the file when no chain is named."""
    if not chains:
        raise ValueError(f"No protein chain in {name}")
    listed = ", ".join(f"{c.id} ({len(c.sequence)} aa)" for c in chains)
    if chain_id:
        for chain in chains:
            if chain.id == chain_id:
                return chain
        raise ValueError(f"{name} has no protein chain '{chain_id}'; its chains: {listed}")
    if len({c.sequence for c in chains}) == 1:
        return chains[0]
    raise ValueError(f"{name} holds several proteins; set structure_chain to one of: {listed}")


# ---------------------------------------------------------------------------
# files on the shared volume
# ---------------------------------------------------------------------------


def shared_root() -> Path:
    """The shared volume, as list_shared_assets sees it."""
    return Path(os.environ.get("IP_SHARED_DIR", "/shared"))


def resolve_structure_path(path: str) -> Path:
    """An absolute path as is; anything else relative to the shared volume."""
    p = Path(path)
    return p if p.is_absolute() else shared_root() / path


@lru_cache(maxsize=32)
def _chain_sequence(path: str, chain_id: Optional[str], _mtime: int, _size: int) -> str:
    text = Path(path).read_text(errors="replace")
    name = Path(path).name
    return pick_chain(structure_chains(name, text), chain_id, name).sequence


def structure_sequence(path: str, chain_id: Optional[str] = None) -> str:
    """The protein sequence of a chain of a structure file under the shared volume."""
    resolved = resolve_structure_path(path)
    try:
        stat = resolved.stat()
    except OSError:
        raise ValueError(f"structure_path '{path}' not found (looked for {resolved})") from None
    return _chain_sequence(str(resolved), chain_id, stat.st_mtime_ns, stat.st_size)
