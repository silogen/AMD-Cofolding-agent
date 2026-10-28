# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load a complex with MDAnalysis and classify its components.

Splits a structure into: small-molecule ligand candidate(s), the
receptor (protein OR nucleic), metal ions, and waters. The ligand of a
triple is ALWAYS a small molecule; peptides/proteins/RNA are receptor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STANDARD_AA: frozenset[str] = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH", "LYN", "MSE", "SEC", "PYL",
})

RNA_RESNAMES: frozenset[str] = frozenset({
    "A", "U", "G", "C", "I",
    "RA", "RU", "RG", "RC", "RI",
    "A3", "A5", "U3", "U5", "G3", "G5", "C3", "C5",
})
DNA_RESNAMES: frozenset[str] = frozenset({
    "DA", "DT", "DG", "DC", "DU", "DI",
    "DA3", "DA5", "DT3", "DT5", "DG3", "DG5", "DC3", "DC5",
})
# Combined nucleic set (RNA + DNA), kept for callers that don't care.
NUCLEIC: frozenset[str] = RNA_RESNAMES | DNA_RESNAMES

WATER: frozenset[str] = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP5", "T3P", "T4P", "DOD", "H2O"})

METALS: frozenset[str] = frozenset({
    "NA", "K", "LI", "RB", "CS", "MG", "CA", "SR", "BA",
    "ZN", "FE", "FE2", "FE3", "MN", "MN3", "CU", "CU1", "CU2", "CO", "NI",
    "CD", "HG", "MO", "W", "V", "CR", "PT", "AU", "AG", "PB", "AL", "GA",
})

# Non-metal ions / crystallisation aids to exclude from ligand detection.
SKIP_LIGAND: frozenset[str] = frozenset({
    "CL", "BR", "IOD", "F", "FLC",
    "SO4", "PO4", "PI", "PO3", "PEG", "GOL", "EDO", "DMS", "ACT", "FMT",
    "CIT", "TRS", "BME", "EPE", "MES", "BOG", "LDA", "MPD", "IPA", "ACY",
    "NO3", "AZI", "SCN", "BCT", "1PE", "PG4", "P6G", "PGE",
})

# Ubiquitous cofactors / nucleotides / lipids / sugars that are usually NOT
# the ligand of interest. Excluded from auto-detection (but selectable via an
# explicit resname / keyword, e.g. for nucleotide-site profiling).
COFACTORS: frozenset[str] = frozenset({
    # nucleotides
    "GDP", "GTP", "GNP", "GSP", "ADP", "ATP", "ANP", "AGS", "ACP", "APC",
    "AMP", "CMP", "UMP", "UDP", "UTP", "TTP", "GCP",
    # redox / carriers
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "ACO",
    "TPP", "PLP", "BTN", "B12", "HEM", "HEC", "HEA", "DHE",
    # lipids / detergents / sterols
    "CLR", "PLM", "MYR", "OLA", "OLC", "STE", "PCW", "LMT", "LMN", "12P",
    "HTG", "OCT", "D10", "C8E", "PX4",
    # iron-sulfur / clusters, phosphoinositols & common sugars
    "SF4", "FES", "FS3", "IHP", "IPD", "I3P", "NAG", "BMA", "MAN", "GAL", "GLC", "FUC", "SIA",
})

MIN_LIGAND_HEAVY_ATOMS = 6
MAX_LIGAND_HEAVY_ATOMS = 200  # small molecule guardrail
# A modified residue inside a chain (PTR, SEP, KCX ...) is part of the
# receptor; only one this large (Trp has 14) carries a fused ligand, e.g.
# a covalent inhibitor deposited as one residue with its cysteine (CY0).
MIN_ADDUCT_HEAVY_ATOMS = 20
PEPTIDE_BOND_MAX = 1.75

PROTEIN_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT", "H", "HA", "HN"})
# nucleic backbone: phosphate + sugar
NUCLEIC_BACKBONE = frozenset({
    "P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
    "C2'", "O2'", "C1'",
})


@dataclass(frozen=True)
class LigandSelection:
    chain: str
    resname: str
    resnum: int
    n_heavy: int
    # Peptide-bonded to its neighbours in the chain: a modified residue that
    # carries the ligand rather than a separate HETATM group.
    in_polymer: bool = False


def _resname(res: Any) -> str:
    return (getattr(res, "resname", "") or "").strip().upper()


def _chain_of(res: Any) -> str:
    for attr in ("chainID", "segid"):
        try:
            atoms = getattr(res, "atoms", None)
            val = getattr(atoms[0], attr, None) if atoms is not None else None
        except Exception:
            val = None
        if val:
            return str(val).strip()
    seg = getattr(getattr(res, "segment", None), "segid", "") or ""
    return seg.strip()


class LoadedStructure:
    """Wraps an MDAnalysis Universe with component classification."""

    def __init__(self, universe: Any, path: str):
        self.u = universe
        self.path = path

    @property
    def chains(self) -> list[str]:
        seen: list[str] = []
        for res in self.u.residues:
            c = _chain_of(res)
            if c and c not in seen:
                seen.append(c)
        return seen

    def receptor_kind(self) -> str | None:
        """Summary receptor type across all chains (protein | rna | dna | hybrids)."""
        kinds = {k for k in self.chain_types().values() if k in ("protein", "rna", "dna")}
        if not kinds:
            return None
        # Deterministic hybrid summary, e.g. "protein+dna".
        order = ["protein", "dna", "rna"]
        return "+".join(k for k in order if k in kinds)

    def chain_types(self) -> dict[str, str]:
        """Classify each chain with a specific label.

        Polymer chains: ``protein`` / ``dna`` / ``rna`` (by any residue of
        that kind). Non-polymer chains are resolved specifically rather than
        a vague "other": ``ligand`` (a drug-like small molecule present),
        ``cofactor`` (GDP/NAD/HEM/... ), ``ion``, ``water``, or ``hetero``.
        """
        cats: dict[str, set[str]] = {}
        for res in self.u.residues:
            rn = _resname(res)
            ch = _chain_of(res)
            if not ch:
                continue
            bucket = cats.setdefault(ch, set())
            if rn in STANDARD_AA:
                bucket.add("protein")
            elif rn in DNA_RESNAMES:
                bucket.add("dna")
            elif rn in RNA_RESNAMES:
                bucket.add("rna")
            elif rn in WATER:
                bucket.add("water")
            elif rn in METALS or rn in {"CL", "BR", "IOD", "F", "NA", "K"}:
                bucket.add("ion")
            elif rn in COFACTORS:
                bucket.add("cofactor")
            elif rn in SKIP_LIGAND:
                bucket.add("buffer")
            else:
                bucket.add("ligand")

        out: dict[str, str] = {}
        # Most specific / informative label wins, in this order.
        priority = ["protein", "dna", "rna", "ligand", "cofactor", "ion", "buffer", "water"]
        for ch, kinds in cats.items():
            out[ch] = next((k for k in priority if k in kinds), "hetero")
        return out

    def detect_ligands(
        self,
        *,
        resname: str | None = None,
        chain: str | None = None,
        keyword: str | None = None,
    ) -> list[LigandSelection]:
        """Return candidate small-molecule ligands, best first.

        ``resname`` (exact) / ``chain`` / ``keyword`` (case-insensitive
        substring of the residue id) constrain the search. With no
        constraint, standard residues, waters, ions, crystallisation aids
        and ubiquitous cofactors/lipids are skipped. Candidates are ranked
        by number of receptor contacts (the true binder, not a distal
        cofactor), tie-broken by heavy-atom count.
        """
        out: list[LigandSelection] = []
        want_res = resname.strip().upper() if resname else None
        want_chain = chain.strip() if chain else None
        kw = keyword.strip().upper() if keyword else None
        for res in self.u.residues:
            rn = _resname(res)
            ch = _chain_of(res)
            if want_res and rn != want_res:
                continue
            if want_chain and ch != want_chain:
                continue
            if kw:
                resid = str(getattr(res, "resid", "") or "")
                rid = f"{rn}{resid}"
                if kw not in rn and kw not in rid:
                    continue
            auto = not (want_res or want_chain or kw)
            if auto and (rn in STANDARD_AA or rn in NUCLEIC or rn in WATER
                         or rn in METALS or rn in SKIP_LIGAND or rn in COFACTORS):
                continue
            heavy = sum(1 for a in res.atoms if (getattr(a, "element", "") or "").upper() != "H"
                        and not (getattr(a, "name", "") or "").upper().startswith("H"))
            out_of_range = heavy < MIN_LIGAND_HEAVY_ATOMS or heavy > MAX_LIGAND_HEAVY_ATOMS
            if out_of_range and auto:
                continue
            in_polymer = rn not in STANDARD_AA and self._peptide_linked(res)
            if in_polymer and auto and heavy < MIN_ADDUCT_HEAVY_ATOMS:
                continue
            out.append(LigandSelection(chain=ch, resname=rn, resnum=int(res.resid), n_heavy=heavy,
                                       in_polymer=in_polymer))
        return self._rank_by_contacts(out)

    def _peptide_linked(self, res: Any) -> bool:
        """True when *res* has a backbone peptide-bonded to another residue."""
        atoms = res.atoms
        own_n = atoms.select_atoms("name N")
        own_c = atoms.select_atoms("name C")
        if not own_n.n_atoms or not own_c.n_atoms or not atoms.select_atoms("name CA").n_atoms:
            return False
        try:
            partners = self.u.select_atoms(
                f"(name C and around {PEPTIDE_BOND_MAX} group n) or (name N and around {PEPTIDE_BOND_MAX} group c)",
                n=own_n, c=own_c,
            )
        except Exception:
            return False
        return bool((partners - atoms).n_atoms)

    def _rank_by_contacts(self, sels: list[LigandSelection]) -> list[LigandSelection]:
        """Sort ligand candidates best-first.

        Key = (coordinates a metal, receptor contact count, heavy atoms).
        Metal coordination is the strongest signal that a copy sits in the
        functional site (e.g. the Zn-bound acetazolamide in carbonic
        anhydrase, not a surface-bound duplicate), then contact count picks
        the true binder over a distal crystallisation additive.
        """
        if len(sels) <= 1:
            return sels
        metal_names = " ".join(sorted(METALS))
        scored: list[tuple[int, int, int, LigandSelection]] = []
        for s in sels:
            ref = f"resname {s.resname} and resid {s.resnum}"
            # The chain neighbours of a fused residue touch it through the peptide bonds.
            neighbours = f" and not resid {s.resnum - 1} {s.resnum + 1}" if s.in_polymer else ""
            try:
                n_contacts = int(self.u.select_atoms(f"(protein or nucleic){neighbours} and around 4.5 ({ref})").n_atoms)
            except Exception:
                n_contacts = 0
            try:
                metal_bonus = 1 if self.u.select_atoms(f"(resname {metal_names}) and around 3.0 ({ref})").n_atoms else 0
            except Exception:
                metal_bonus = 0
            scored.append((metal_bonus, n_contacts, s.n_heavy, s))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [t[3] for t in scored]


_TWO_LETTER_ELEMENTS = frozenset({
    "CL", "BR", "NA", "MG", "AL", "SI", "CA", "FE", "ZN", "CU", "MN", "CO",
    "NI", "SE", "MO", "CD", "HG", "PT", "AU", "AG", "PB", "LI", "BE",
})


def _element_from_atom(a: Any) -> str:
    """Best-effort element symbol for an MDAnalysis atom.

    Prefers the explicit ``element`` attribute; otherwise derives it from
    the atom name (handles two-letter elements like Cl/Br and strips
    PDB digit prefixes such as ``1H`` / numbering suffixes)."""
    el = (getattr(a, "element", "") or "").strip()
    if el:
        return el.capitalize()
    nm = (getattr(a, "name", "") or "").strip().upper()
    nm = "".join(c for c in nm if c.isalpha())
    if not nm:
        return ""
    if len(nm) >= 2 and nm[:2] in _TWO_LETTER_ELEMENTS:
        return nm[:2].capitalize()
    return nm[0].capitalize()


def heavy_element_counts(atoms: Any) -> dict[str, int]:
    """Heavy-atom element multiset of an MDAnalysis AtomGroup / residue."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for a in atoms:
        sym = _element_from_atom(a)
        if sym and sym != "H":
            counts[sym] += 1
    return dict(counts)


def prefer_formula_match(
    u: Any,
    sels: list[LigandSelection],
    template_formula: dict[str, int] | None,
) -> list[LigandSelection]:
    """Reorder ligand candidates so ones matching the template's heavy-atom
    formula come first (order preserved within each group).

    When a SMILES template is supplied, the user is telling us *which*
    molecule to profile; a formula match on the structure's HETATM residues
    identifies the right copy/type without relying on bond-order perception.
    Returns the input unchanged when no template formula is available.
    """
    if not template_formula or len(sels) <= 1:
        return sels
    matched: list[LigandSelection] = []
    rest: list[LigandSelection] = []
    for s in sels:
        try:
            ag = u.select_atoms(f"resname {s.resname} and resid {s.resnum}")
            (matched if heavy_element_counts(ag.atoms) == template_formula else rest).append(s)
        except Exception:
            rest.append(s)
    return matched + rest if matched else sels


def load_structure(path: str | Path) -> LoadedStructure:
    import MDAnalysis as mda

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"structure not found: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"structure file is empty: {p.name}")
    suffix = p.suffix.lower() or "?"
    if suffix not in (".pdb", ".cif", ".mmcif", ".ent", ".pdbx"):
        raise ValueError(
            f"unsupported structure format '{suffix}' for {p.name} (expected PDB/mmCIF)"
        )
    try:
        u = mda.Universe(str(p))
    except Exception as exc:
        # MDAnalysis leaks low-level IndexError/EOFError on malformed files;
        # wrap them in one precise, actionable message.
        raise ValueError(
            f"could not parse structure '{p.name}' as {suffix} "
            f"(malformed or not a valid PDB/mmCIF): {type(exc).__name__}: {exc}"
        ) from exc
    if u.atoms.n_atoms == 0:
        raise ValueError(f"structure '{p.name}' contains no atoms")
    return LoadedStructure(u, str(p))


def parse_covalent_links(path: str | Path, ligand_resname: str | None) -> set[tuple[str, int, str]]:
    """Return receptor residues covalently bonded to the ligand as
    ``(resname, resnum, chain)``.

    Parses PDB ``LINK`` records and mmCIF ``_struct_conn`` (conn_type_id
    ``covale``). Returns an empty set on any problem -- covalent detection
    then falls back to the geometric rule.
    """
    p = Path(path)
    if not p.exists():
        return set()
    want = (ligand_resname or "").strip().upper()
    if p.suffix.lower() in (".cif", ".mmcif"):
        return _parse_struct_conn(p, want)
    if p.suffix.lower() != ".pdb":
        return set()
    links: set[tuple[str, int, str]] = set()
    try:
        for line in p.read_text().splitlines():
            if not line.startswith("LINK"):
                continue
            padded = line.ljust(80)
            try:
                rn1 = padded[17:20].strip().upper()
                ch1 = padded[21:22].strip() or "A"
                rs1 = int(padded[22:26].strip() or "0")
                rn2 = padded[47:50].strip().upper()
                ch2 = padded[51:52].strip() or "A"
                rs2 = int(padded[52:56].strip() or "0")
            except ValueError:
                continue
            e1, e2 = (rn1, rs1, ch1), (rn2, rs2, ch2)
            is1, is2 = (want and rn1 == want), (want and rn2 == want)
            if is1 and not is2:
                links.add(e2)
            elif is2 and not is1:
                links.add(e1)
    except OSError:
        return set()
    return links


def _parse_struct_conn(path: Path, want: str) -> set[tuple[str, int, str]]:
    """Parse the mmCIF ``_struct_conn`` loop for covalent (``covale``) bonds."""
    links: set[tuple[str, int, str]] = set()
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return links
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            cols: list[str] = []
            j = i + 1
            while j < n and lines[j].lstrip().startswith("_struct_conn."):
                cols.append(lines[j].strip().split(".", 1)[1])
                j += 1
            if cols:
                idx = {c: k for k, c in enumerate(cols)}
                need = ["conn_type_id",
                        "ptnr1_label_comp_id", "ptnr2_label_comp_id"]
                if all(c in idx for c in need):
                    while j < n and lines[j].strip() and not lines[j].startswith(("loop_", "_", "#")):
                        toks = lines[j].split()
                        if len(toks) >= len(cols):
                            _sc_row(toks, idx, want, links)
                        j += 1
                i = j
                continue
        i += 1
    return links


def _sc_row(toks: list[str], idx: dict, want: str, links: set) -> None:
    def g(key: str, default: str = "") -> str:
        return toks[idx[key]] if key in idx else default

    if "covale" not in g("conn_type_id").lower():
        return

    def partner(pfx: str) -> tuple[str, int, str] | None:
        comp = g(f"{pfx}_label_comp_id").upper()
        chain = g(f"{pfx}_auth_asym_id") or g(f"{pfx}_label_asym_id") or "A"
        seq = g(f"{pfx}_auth_seq_id") or g(f"{pfx}_label_seq_id") or "0"
        try:
            return (comp, int(seq.replace("?", "0") or "0"), chain)
        except ValueError:
            return None

    e1, e2 = partner("ptnr1"), partner("ptnr2")
    if e1 is None or e2 is None:
        return
    is1, is2 = (want and e1[0] == want), (want and e2[0] == want)
    if is1 and not is2:
        links.add(e2)
    elif is2 and not is1:
        links.add(e1)


def is_backbone(reskind: str, atom_name: str) -> bool:
    name = (atom_name or "").strip().upper()
    if reskind == "protein":
        return name in PROTEIN_BACKBONE
    if reskind in ("rna", "dna", "nucleic"):
        return name in {a.upper() for a in NUCLEIC_BACKBONE}
    return False


def residue_kind(resname: str) -> str:
    rn = (resname or "").strip().upper()
    if rn in STANDARD_AA:
        return "protein"
    if rn in DNA_RESNAMES:
        return "dna"
    if rn in RNA_RESNAMES:
        return "rna"
    if rn in WATER:
        return "water"
    if rn in METALS:
        return "ion"
    return "other"


__all__ = [
    "LoadedStructure",
    "LigandSelection",
    "load_structure",
    "heavy_element_counts",
    "prefer_formula_match",
    "is_backbone",
    "residue_kind",
    "STANDARD_AA",
    "NUCLEIC",
    "RNA_RESNAMES",
    "DNA_RESNAMES",
    "WATER",
    "METALS",
    "COFACTORS",
]
