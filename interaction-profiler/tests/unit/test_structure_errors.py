"""load_structure should raise precise, actionable errors on bad input."""

from __future__ import annotations

import pytest

from interaction_profiler.io.structure import load_structure


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="structure not found"):
        load_structure(tmp_path / "nope.pdb")


def test_empty_file(tmp_path) -> None:
    p = tmp_path / "empty.pdb"
    p.write_text("")
    with pytest.raises(ValueError, match="structure file is empty"):
        load_structure(p)


def test_unsupported_format(tmp_path) -> None:
    p = tmp_path / "thing.xyz"
    p.write_text("stuff")
    with pytest.raises(ValueError, match="unsupported structure format"):
        load_structure(p)


def test_garbage_pdb_precise_error(tmp_path) -> None:
    p = tmp_path / "garbage.pdb"
    p.write_text("this is not a pdb\nrandom text\n")
    with pytest.raises(ValueError, match="could not parse structure"):
        load_structure(p)
