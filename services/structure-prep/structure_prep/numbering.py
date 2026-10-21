"""Residue-numbering maps for safe cross-service coordinate translation."""

from __future__ import annotations

import gemmi

from .models import ResidueNumberingEntry


def _is_amino_acid(residue: gemmi.Residue) -> bool:
    try:
        return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())
    except Exception:
        return False


def _one_letter(residue: gemmi.Residue) -> str:
    try:
        code = gemmi.find_tabulated_residue(residue.name).one_letter_code
    except Exception:
        code = "X"
    return code if code and len(code) == 1 else "X"


def build_residue_numbering(
    structure: gemmi.Structure,
    uniprot_offsets: dict[str, int] | None = None,
    uniprot_sequence: str | None = None,
) -> tuple[list[ResidueNumberingEntry], dict[str, str]]:
    """Build author↔sequence↔UniProt mappings from the final structure.

    If *uniprot_offsets* is not provided but *uniprot_sequence* is, the
    function tries a simple global alignment per chain to derive the offset
    automatically (useful when the structure has a non-zero author start).
    """
    if len(structure) == 0:
        return [], {}
    offsets = dict(uniprot_offsets or {})

    # Auto-derive offsets from a canonical UniProt sequence via substring match.
    if uniprot_sequence and not offsets:
        for chain in structure[0]:
            residues = [r for r in chain if _is_amino_acid(r)]
            if not residues:
                continue
            chain_seq = "".join(_one_letter(r) for r in residues)
            # Find chain_seq as a subsequence of uniprot_sequence (exact match).
            pos = uniprot_sequence.find(chain_seq[:20])  # anchor on first 20 aa
            if pos >= 0:
                # offset = UniProt 1-based position of the first chain residue
                #        - author resseq of that residue
                offsets[chain.name] = (pos + 1) - residues[0].seqid.num
    entries: list[ResidueNumberingEntry] = []
    sequences: dict[str, str] = {}
    for chain in structure[0]:
        residues = [residue for residue in chain if _is_amino_acid(residue)]
        if not residues:
            continue
        sequence: list[str] = []
        for position, residue in enumerate(residues, start=1):
            letter = _one_letter(residue)
            sequence.append(letter)
            label_seq_id = getattr(residue, "label_seq", None)
            try:
                label_seq_id = int(label_seq_id)
                if label_seq_id <= 0:
                    label_seq_id = None
            except (TypeError, ValueError):
                label_seq_id = None
            offset = offsets.get(chain.name)
            entries.append(
                ResidueNumberingEntry(
                    chain=chain.name,
                    sequence_position=position,
                    author_resseq=residue.seqid.num,
                    author_icode=str(residue.seqid.icode).strip(),
                    label_seq_id=label_seq_id,
                    resname=residue.name.strip().upper(),
                    one_letter=letter,
                    uniprot_position=(
                        residue.seqid.num + offset if offset is not None else None
                    ),
                )
            )
        sequences[chain.name] = "".join(sequence)
    return entries, sequences


def author_to_sequence_position(
    entries: list[ResidueNumberingEntry],
    chain: str,
    author_resseq: int,
    expected_resname: str | None = None,
    author_icode: str = "",
) -> int:
    """Resolve an author residue number and optionally assert its identity."""
    matches = [
        entry
        for entry in entries
        if entry.chain == chain
        and entry.author_resseq == author_resseq
        and entry.author_icode == author_icode
    ]
    if not matches:
        raise ValueError(
            f"Author residue {chain}:{author_resseq}{author_icode} is absent "
            "from the numbering map."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Author residue {chain}:{author_resseq}{author_icode} is ambiguous."
        )
    entry = matches[0]
    if expected_resname and entry.resname != expected_resname.strip().upper():
        raise ValueError(
            f"Residue identity mismatch at {chain}:{author_resseq}{author_icode}: "
            f"expected {expected_resname.upper()}, found {entry.resname}."
        )
    return entry.sequence_position
