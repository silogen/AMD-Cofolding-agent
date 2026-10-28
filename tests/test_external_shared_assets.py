# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""list_shared_assets sees every upload folder, at any depth, and says when it looked."""

from __future__ import annotations

import os

from reinvent4_mcp.tools.external import list_shared_assets


def _touch(path, mtime: float, text: str = "data_x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_every_upload_is_listed_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("IP_SHARED_DIR", str(tmp_path))
    _touch(tmp_path / "uploads/new/3MXF.cif", 1_000)
    _touch(tmp_path / "uploads/s-20261027-103749-57b6b7/3EYG.cif", 2_000)
    _touch(tmp_path / "uploads/s-20261028-090207-229c4e/2J5E.cif", 3_000)
    _touch(tmp_path / "uploads/s-20261026-211429-495844/3MXF.cif", 500)
    _touch(tmp_path / "predictions/out_1/boltz_results_x/predictions/x/x_model_0.cif", 4_000)
    _touch(tmp_path / "msa_cache/egfr.a3m", 100, ">q\nMKV\n")

    result = list_shared_assets()

    assert result["uploads"] == [
        "uploads/s-20261028-090207-229c4e/2J5E.cif",
        "uploads/s-20261027-103749-57b6b7/3EYG.cif",
        "uploads/new/3MXF.cif",
        "uploads/s-20261026-211429-495844/3MXF.cif",
    ]
    assert "uploads/s-20261028-090207-229c4e/2J5E.cif" in result["structures"]
    assert result["cofold_outputs"] == ["predictions/out_1/boltz_results_x/predictions/x/x_model_0.cif"]
    assert result["counts"] == {"structures": 5, "uploads": 4, "msas": 1, "cofold_outputs": 1}
    assert result["msa_details"] == [{"path": "msa_cache/egfr.a3m", "query_length": 3}]
    assert result["listed_at"].endswith("+00:00")


def test_missing_shared_dir_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("IP_SHARED_DIR", str(tmp_path / "absent"))
    result = list_shared_assets()
    assert result["structures"] == [] and result["uploads"] == []
    assert "does not exist" in result["note"]
