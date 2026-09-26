"""Construct-form detection and conservative terminal trimming."""

from __future__ import annotations

from typing import Iterable

import gemmi

from .models import ConstructAssessment, ConstructProfile


def _is_amino_acid(residue: gemmi.Residue) -> bool:
    try:
        return bool(gemmi.find_tabulated_residue(residue.name).is_amino_acid())
    except Exception:
        return False


def _polymer_residues(chain: gemmi.Chain) -> list[gemmi.Residue]:
    return [residue for residue in chain if _is_amino_acid(residue)]


def _one_letter_sequence(residues: Iterable[gemmi.Residue]) -> str:
    sequence: list[str] = []
    for residue in residues:
        try:
            code = gemmi.find_tabulated_residue(residue.name).one_letter_code
        except Exception:
            code = "X"
        sequence.append(code if code and len(code) == 1 else "X")
    return "".join(sequence)


def _terminal_tag(sequence: str, n_terminal: bool) -> str | None:
    probe = sequence[:12] if n_terminal else sequence[-12:]
    known = ("GAMSG", "GSS", "HHHHHH", "MHHHHH")
    for tag in known:
        if (probe.startswith(tag) if n_terminal else probe.endswith(tag)):
            return tag
    return None


def assess_construct(structure: gemmi.Structure) -> list[ConstructAssessment]:
    """Classify every protein chain without mutating coordinates."""
    if len(structure) == 0:
        return []

    assessments: list[ConstructAssessment] = []
    for chain in structure[0]:
        residues = _polymer_residues(chain)
        if not residues:
            continue
        start = residues[0].seqid.num
        end = residues[-1].seqid.num
        sequence = _one_letter_sequence(residues)
        if start <= 0:
            form = "immature"
        elif start == 1:
            form = "mature"
        else:
            form = "unknown"
        assessments.append(
            ConstructAssessment(
                chain=chain.name,
                author_start=start,
                author_end=end,
                n_polymer_residues=len(residues),
                non_native_n_terminal=start <= 0,
                suspected_n_tag=_terminal_tag(sequence, True),
                suspected_c_tag=_terminal_tag(sequence, False),
                form=form,
            )
        )
    return assessments


def apply_construct_policy(
    structure: gemmi.Structure,
    profile: ConstructProfile,
) -> tuple[gemmi.Structure, list[ConstructAssessment]]:
    """Apply a conservative construct policy and return an audit assessment.

    ``native`` removes only author-numbered residues at or below zero. Tag-like
    sequences are reported but never deleted heuristically. ``user_range`` is
    the explicit escape hatch for other constructs.
    """
    assessments = assess_construct(structure)
    if profile.trim_policy == "none":
        return structure, assessments

    trimmed_by_chain: dict[str, list[str]] = {}
    for model in structure:
        for chain in model:
            remove_indexes: list[int] = []
            removed: list[str] = []
            user_range = profile.user_ranges.get(chain.name)
            for index, residue in enumerate(chain):
                if not _is_amino_acid(residue):
                    continue
                remove = False
                if profile.trim_policy == "native":
                    remove = residue.seqid.num <= 0
                elif profile.trim_policy == "user_range":
                    if user_range is None:
                        continue
                    remove = not (user_range[0] <= residue.seqid.num <= user_range[1])
                if remove:
                    remove_indexes.append(index)
                    icode = str(residue.seqid.icode).strip()
                    removed.append(f"{residue.name}{residue.seqid.num}{icode}")
            for index in reversed(remove_indexes):
                del chain[index]
            if removed:
                trimmed_by_chain.setdefault(chain.name, []).extend(removed)

    result = assess_construct(structure)
    by_chain = {assessment.chain: assessment for assessment in result}
    # Preserve pre-trim evidence (e.g. author_start=-2) while recording the
    # final range in the same audit object.
    for before in assessments:
        after = by_chain.get(before.chain, before)
        after.non_native_n_terminal = before.non_native_n_terminal
        after.suspected_n_tag = before.suspected_n_tag
        after.suspected_c_tag = before.suspected_c_tag
        after.trimmed_residues = trimmed_by_chain.get(before.chain, [])
    return structure, list(by_chain.values())
