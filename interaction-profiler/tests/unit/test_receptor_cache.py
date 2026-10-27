# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the receptor prep cache-target resolution (no OpenMM needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from interaction_profiler.prep.receptor import _cache_target


def test_next_to_source_when_no_cache_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("IP_CACHE_DIR", raising=False)
    src = tmp_path / "rec.pdb"
    src.write_text("X")
    out, hashed = _cache_target(src, 7.4)
    assert hashed is False
    assert out == src.with_name("rec_prepared_ph7.40.pdb")


@pytest.mark.parametrize("ph", [-0.5, 14.5, "7.4/../../etc", None])
def test_rejects_non_ph_values(monkeypatch, tmp_path, ph) -> None:
    """Only a real pH reaches the filename, so nothing can inject path parts."""
    monkeypatch.delenv("IP_CACHE_DIR", raising=False)
    src = tmp_path / "rec.pdb"
    src.write_text("X")
    with pytest.raises((ValueError, TypeError)):
        _cache_target(src, ph)


def test_hash_keyed_in_cache_dir(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("IP_CACHE_DIR", str(cache))
    src = tmp_path / "rec.pdb"
    src.write_text("ATOM  content")
    out, hashed = _cache_target(src, 7.4)
    assert hashed is True
    assert Path(out).parent == cache
    assert out.name.startswith("rec_") and out.name.endswith("_prepared.pdb")
    # pH is part of the key -> different file for a different pH.
    out2, _ = _cache_target(src, 6.5)
    assert out2 != out
    # Content is part of the key -> different file for different content.
    src.write_text("ATOM  other")
    out3, _ = _cache_target(src, 7.4)
    assert out3 != out
