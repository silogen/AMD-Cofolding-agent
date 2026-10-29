# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One protein chain of a PDB or mmCIF file as a minimal mmCIF, for Boltz-2
to use as a template: the first model, the chain's amino acids (standard and
common modified ones), the first alt-loc of each residue, no hydrogens,
waters, ligands or ions.

Boltz reads templates with gemmi and names chains by ``label_asym_id`` (a PDB
file's chains become "A1", ...), so the output gives the chain the same author
and label id: the chain picked here is the one Boltz finds.

Mirrors the browser's template builder
(services/ui/frontend/src/lib/cleanStructure.ts) so the file the form uploads
and the file the server writes are byte for byte the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from reinvent4_mcp.ttp.structure import is_cif, letter, read_cif

HYDROGEN = {"H", "D"}

ATOM_FIELDS = (
    "group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id", "label_entity_id", "label_seq_id",
    "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id",
    "auth_atom_id", "pdbx_PDB_model_num",
)

_RESERVED = re.compile(r"(data_|loop_|save_|global_|stop_)", re.IGNORECASE)
_BARE = re.compile(r"""[^\s'"_#$;\[\]][^\s'"]*""")
_NOT_NAME = re.compile(r"[^A-Za-z0-9_-]")
_NOT_ALNUM = re.compile(r"[^A-Za-z0-9]")
_NOT_LETTER = re.compile(r"[^A-Za-z]")
_SUFFIX = re.compile(r"\.[^.]+\Z")
_WRAP = re.compile(r"(.{80})")


@dataclass(frozen=True)
class _Atom:
    record: str  # "ATOM" or "HETATM"
    name: str
    element: str
    comp: str
    num: int  # label_seq_id: position in the chain's sequence, from 1
    auth_seq: str
    ins: str | None
    x: str
    y: str
    z: str
    occupancy: str
    b: str


@dataclass(frozen=True)
class _Chain:
    sequence: dict[int, str]  # the chain's sequence as residue names, num -> name
    atoms: list[_Atom]


@dataclass(frozen=True)
class CleanedChain:
    """A template-ready mmCIF document and the size of the chain it holds."""

    text: str
    atoms: int
    residues: int


def _first(*values: str | None) -> str | None:
    """The first value that is present."""
    return next((value for value in values if value is not None), None)


def _text(*values: str | None, default: str) -> str:
    """The first value that is present, else *default*."""
    value = _first(*values)
    return default if value is None else value


def _whole(value: str | None) -> int | None:
    """*value* as a whole number, read as mmCIF writers mean it; None when it is not one."""
    try:
        number = float((value or "").strip() or "0")
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def _picker() -> Callable[[str, str, str | None, str], bool]:
    """Keeps the first residue name and alt-loc seen at each position, and each atom name once."""
    comps: dict[str, str] = {}
    alts: dict[str, str] = {}
    names: set[str] = set()

    def keep(key: str, comp: str, alt: str | None, name: str) -> bool:
        if comps.get(key, comp) != comp:
            return False
        comps[key] = comp
        if alt:
            if alts.get(key, alt) != alt:
                return False
            alts[key] = alt
        if f"{key}:{name}" in names:
            return False
        names.add(f"{key}:{name}")
        return True

    return keep


def _cif_chain(text: str, chain_id: str) -> _Chain:
    tables = read_cif(text, ("atom_site", "entity_poly", "entity_poly_seq"))
    rows = tables.get("atom_site", [])
    polys = tables.get("entity_poly", [])
    peptides = {p.get("entity_id") for p in polys if not p.get("type") or "polypeptide" in p["type"].lower()}
    model = rows[0].get("pdbx_pdb_model_num") if rows else None
    in_chain = [
        r
        for r in rows
        if (model is None or r.get("pdbx_pdb_model_num") == model)
        and _first(r.get("auth_asym_id"), r.get("label_asym_id")) == chain_id
        and r.get("label_seq_id") is not None
        and (not polys or r.get("label_entity_id") in peptides)
    ]
    amino = next((r for r in in_chain if letter(_text(r.get("label_comp_id"), r.get("auth_comp_id"), default=""))), None)
    entity = amino.get("label_entity_id") if amino is not None else None
    keep = _picker()
    atoms: list[_Atom] = []
    for r in in_chain:
        comp = _text(r.get("label_comp_id"), r.get("auth_comp_id"), default="")
        name = _text(r.get("label_atom_id"), r.get("auth_atom_id"), default="")
        element = _text(r.get("type_symbol"), default=_NOT_LETTER.sub("", name)[:1]).upper()
        num = _whole(r.get("label_seq_id"))
        if r.get("label_entity_id") != entity or not letter(comp) or element in HYDROGEN or not name or num is None:
            continue
        if not keep(str(num), comp, r.get("label_alt_id"), name):
            continue
        atoms.append(
            _Atom(
                record="HETATM" if r.get("group_pdb") == "HETATM" else "ATOM",
                name=name,
                element=element,
                comp=comp,
                num=num,
                auth_seq=_text(r.get("auth_seq_id"), default=str(num)),
                ins=r.get("pdbx_pdb_ins_code"),
                x=_text(r.get("cartn_x"), default="0"),
                y=_text(r.get("cartn_y"), default="0"),
                z=_text(r.get("cartn_z"), default="0"),
                occupancy=_text(r.get("occupancy"), default="1"),
                b=_text(r.get("b_iso_or_equiv"), default="0"),
            )
        )
    sequence: dict[int, str] = {}
    for row in tables.get("entity_poly_seq", []):
        num = _whole(row.get("num"))
        if row.get("entity_id") == entity and row.get("mon_id") and num is not None and num not in sequence:
            sequence[num] = row["mon_id"]
    if sequence and all(a.num in sequence for a in atoms):
        return _Chain(sequence, atoms)
    # Boltz indexes a template's sequence by label_seq_id, so without the full
    # sequence the observed residues have to be numbered 1..n.
    order = {num: i for i, num in enumerate(dict.fromkeys(a.num for a in atoms), 1)}
    renumbered = [replace(a, num=order.get(a.num, 0)) for a in atoms]
    return _Chain({a.num: a.comp for a in renumbered}, renumbered)


def _pdb_chain(text: str, chain_id: str) -> _Chain:
    """A PDB file's chain, numbered by the residues with coordinates (SEQRES is not aligned to them)."""
    keep = _picker()
    atoms: list[_Atom] = []
    sequence: dict[int, str] = {}
    last_key: str | None = None
    num = 0
    for line in text.splitlines():
        record = line[:6]
        if record == "ENDMDL" and atoms:
            break
        if record not in ("ATOM  ", "HETATM"):
            continue
        if (line[21:22].strip() or "_") != chain_id:
            continue
        comp = line[17:20].strip()
        name = line[12:16].strip()
        element = (line[76:78].strip() or _NOT_LETTER.sub("", name)[:1]).upper()
        key = line[22:27]
        if not letter(comp) or element in HYDROGEN or not name or not keep(key, comp, line[16:17].strip() or None, name):
            continue
        if key != last_key:
            num += 1
            last_key = key
            sequence[num] = comp
        atoms.append(
            _Atom(
                record="HETATM" if record == "HETATM" else "ATOM",
                name=name,
                element=element,
                comp=comp,
                num=num,
                auth_seq=line[22:26].strip(),
                ins=line[26:27].strip() or None,
                x=line[30:38].strip(),
                y=line[38:46].strip(),
                z=line[46:54].strip(),
                occupancy=line[54:60].strip() or "1.00",
                b=line[60:66].strip() or "0.00",
            )
        )
    return _Chain(sequence, atoms)


def cif_value(value: str | None) -> str:
    """A value as an mmCIF token, quoted when it would not read back as itself."""
    if value is None or value == "":
        return "."
    if _BARE.fullmatch(value) and not _RESERVED.match(value) and value not in (".", "?"):
        return value
    return f'"{value}"' if "'" in value else f"'{value}'"


def _loop(category: str, fields: Sequence[str], rows: Sequence[Sequence[str | None]]) -> str:
    return "\n".join(
        [
            "loop_",
            *(f"_{category}.{field}" for field in fields),
            *(" ".join(cif_value(value) for value in row) for row in rows),
            "#",
        ]
    )


def _monomer(chain: _Chain, num: int) -> str:
    """The residue name at position *num*: the sequence's, else the one its atoms carry."""
    if num in chain.sequence:
        return chain.sequence[num]
    return next((a.comp for a in chain.atoms if a.num == num), "UNK")


def _write_cif(name: str, chain_id: str, chain: _Chain) -> str:
    nums = sorted({*chain.sequence, *(a.num for a in chain.atoms)})
    monomers = [(num, _monomer(chain, num)) for num in nums]
    one_letter = _WRAP.sub(r"\1\n", "".join(letter(comp) or "X" for _, comp in monomers))
    block = _NOT_NAME.sub("_", name)
    return "\n".join(
        [
            f"data_{block}",
            "#",
            f"_entry.id {cif_value(block)}",
            "#",
            _loop("entity", ["id", "type"], [["1", "polymer"]]),
            "_entity_poly.entity_id 1",
            "_entity_poly.type 'polypeptide(L)'",
            "_entity_poly.nstd_linkage no",
            f"_entity_poly.pdbx_strand_id {cif_value(chain_id)}",
            "_entity_poly.pdbx_seq_one_letter_code_can",
            f";{one_letter}",
            ";",
            "#",
            _loop("entity_poly_seq", ["entity_id", "num", "mon_id", "hetero"], [["1", str(num), comp, "n"] for num, comp in monomers]),
            _loop("struct_asym", ["id", "entity_id"], [[chain_id, "1"]]),
            _loop(
                "atom_site",
                ATOM_FIELDS,
                [
                    [
                        a.record, str(i), a.element, a.name, None, a.comp, chain_id, "1", str(a.num),
                        a.ins, a.x, a.y, a.z, a.occupancy, a.b, a.auth_seq, a.comp, chain_id, a.name, "1",
                    ]
                    for i, a in enumerate(chain.atoms, 1)
                ],
            ),
            "",
        ]
    )


def clean_chain(path: str, text: str, chain_id: str, name: str) -> CleanedChain:
    """Chain *chain_id* (author id) of a PDB or mmCIF file as a template-ready mmCIF named *name*."""
    chain = _cif_chain(text, chain_id) if is_cif(path, text) else _pdb_chain(text, chain_id)
    if not chain.atoms:
        raise ValueError(f"Chain {chain_id} has no protein atoms.")
    return CleanedChain(_write_cif(name, chain_id, chain), len(chain.atoms), len({a.num for a in chain.atoms}))


def template_name(source: str, chain_id: str) -> str:
    """A template's file name: 1M17_A.cif for chain A of 1M17, 1M17.cif or a/b/1M17.pdb."""
    stem = _SUFFIX.sub("", source.replace("\\", "/").split("/")[-1])
    return f"{_NOT_NAME.sub('_', stem)}_{_NOT_ALNUM.sub('_', chain_id)}.cif"
