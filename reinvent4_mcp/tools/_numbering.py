"""Shared numbering resolution for the MCP layer.

Kept separate from the structure-prep service so that it can be imported
by any MCP tool without pulling in gemmi at the module level.
"""
from __future__ import annotations

from typing import Any


def resolve_author_resseq(
    entries: list[dict[str, Any]],
    chain: str,
    author_resseq: int,
    expected_resname: str | None = None,
    author_icode: str = "",
) -> dict[str, Any]:
    """Resolve an author residue number against a numbering-map entry list.

    The *entries* are the raw dicts from residue_numbering.json (the same
    shape as ResidueNumberingEntry.model_dump()).

    Raises ValueError with a descriptive message on mismatch — callers
    should surface this to the agent rather than swallowing it silently.
    """
    matches = [
        e
        for e in entries
        if e["chain"] == chain
        and e["author_resseq"] == author_resseq
        and (e.get("author_icode") or "") == author_icode
    ]
    if not matches:
        raise ValueError(
            f"Author residue {chain}:{author_resseq}{author_icode} is absent "
            "from the numbering map.  The structure may not have been prepared "
            "with prepare_structure, or the chain/number is wrong."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Author residue {chain}:{author_resseq}{author_icode} is ambiguous "
            f"(found {len(matches)} entries).  Supply author_icode to disambiguate."
        )
    entry = matches[0]
    if expected_resname:
        found = entry["resname"].strip().upper()
        wanted = expected_resname.strip().upper()
        if found != wanted:
            raise ValueError(
                f"Residue identity mismatch at {chain}:{author_resseq}{author_icode}: "
                f"expected {wanted}, found {found}.  "
                "This usually means the author-number coordinate system differs from "
                "what was assumed (e.g. an immature form with a negative start)."
            )
    return {
        "chain": entry["chain"],
        "author_resseq": entry["author_resseq"],
        "author_icode": entry.get("author_icode", ""),
        "sequence_position": entry["sequence_position"],
        "resname": entry["resname"],
        "one_letter": entry.get("one_letter"),
        "uniprot_position": entry.get("uniprot_position"),
    }
