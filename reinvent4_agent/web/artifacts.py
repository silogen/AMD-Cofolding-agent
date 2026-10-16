# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn raw MCP tool results into the data kinds the web UI can render.

Every MCP tool returns a JSON string.  The UI shows four kinds of data next
to the conversation:

* ``structure`` — a protein-ligand complex from Boltz (a CIF or PDB file on
  the shared volume), shown in the 3D viewer.
* ``molecule``  — a ligand as SMILES, drawn as a 2D graph.
* ``table``     — predictions and measurements per molecule (affinity, ADMET,
  REINVENT scores), shown as a sortable table.
* ``text``      — anything that does not match the above stays in the chat as
  the agent's own words.

This module does the classification.  It is deliberately tolerant: tools
evolve, so an unrecognised payload simply produces no panel instead of an
error, and the raw JSON stays available in the tool-result block.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Root of the shared directory every service writes artifacts to.  Only
#: files below this directory are ever served to the browser.
SHARED_ROOT = Path(os.environ.get("REINVENT4_SHARED_DIR", "/shared"))

#: File suffixes the 3D viewer understands.
STRUCTURE_SUFFIXES = (".cif", ".mmcif", ".pdb", ".ent")

#: Keys whose value is a SMILES string, in the order we prefer them.
_SMILES_KEYS = ("smiles", "SMILES", "canonical_smiles", "smi")

#: Keys that never belong in a results table even though they are scalar.
_TABLE_NOISE_KEYS = frozenset({"status", "ok", "error", "message", "version"})

#: Cap on rows and panels handed to the browser in one message.  A REINVENT
#: run can emit tens of thousands of molecules; the UI is for inspection, not
#: bulk download, and the full file stays reachable via /api/artifacts.
MAX_TABLE_ROWS = 500
MAX_PANELS = 24


def parse_tool_result(raw: str) -> Any | None:
    """Parse a tool result string as JSON, returning None if it is not JSON."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def extract_panels(tool_name: str, raw: str) -> list[dict[str, Any]]:
    """Derive UI panels from one tool result.

    Returns a list of panel dicts, each with a ``kind`` of ``structure``,
    ``molecule`` or ``table``.  An empty list means "nothing to show but the
    text", which is the normal case for discovery tools.
    """
    payload = parse_tool_result(raw)
    if payload is None:
        return []

    panels: list[dict[str, Any]] = []
    try:
        panels.extend(_structure_panels(tool_name, payload))
        panels.extend(_molecule_panels(tool_name, payload))
        panels.extend(_table_panels(tool_name, payload))
    except Exception:  # pragma: no cover - never break the chat over a panel
        logger.exception("Failed to extract panels from %s result", tool_name)
        return []

    return panels[:MAX_PANELS]


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------


def _structure_panels(tool_name: str, payload: Any) -> list[dict[str, Any]]:
    """Collect every structure file path mentioned in the payload."""
    paths: list[str] = []
    _collect_structure_paths(payload, paths)

    panels = []
    for path in _dedupe(paths):
        rel = to_relative_artifact_path(path)
        if rel is None:
            continue
        panels.append(
            {
                "kind": "structure",
                "tool": tool_name,
                "name": Path(path).name,
                "path": rel,
                "url": f"/api/artifacts/{rel}",
                "format": _structure_format(path),
            }
        )
    return panels


def _collect_structure_paths(node: Any, out: list[str]) -> None:
    """Walk the payload and record anything that looks like a structure file."""
    if isinstance(node, str):
        if node.lower().endswith(STRUCTURE_SUFFIXES):
            out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_structure_paths(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_structure_paths(value, out)


def _structure_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return "cif"
    return "pdb"


# --------------------------------------------------------------------------
# molecules
# --------------------------------------------------------------------------


def _molecule_panels(tool_name: str, payload: Any) -> list[dict[str, Any]]:
    """Collect SMILES strings, keeping the label the tool gave them."""
    found: list[tuple[str, str | None]] = []
    _collect_smiles(payload, found)

    panels = []
    seen: set[str] = set()
    for smiles, label in found:
        if smiles in seen:
            continue
        seen.add(smiles)
        panels.append(
            {
                "kind": "molecule",
                "tool": tool_name,
                "smiles": smiles,
                "label": label,
            }
        )
    return panels


def _collect_smiles(node: Any, out: list[tuple[str, str | None]], label: str | None = None) -> None:
    if isinstance(node, dict):
        name = _first_str(node, ("name", "id", "molecule_id", "compound"))
        for key in _SMILES_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                out.append((value.strip(), name or label))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        out.append((item.strip(), None))
        for child_key, child in node.items():
            if child_key in _SMILES_KEYS:
                continue
            _collect_smiles(child, out, name or label)
    elif isinstance(node, list):
        for item in node:
            _collect_smiles(item, out, label)


def _first_str(node: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _table_panels(tool_name: str, payload: Any) -> list[dict[str, Any]]:
    """Build at most one table from the payload.

    Two shapes are recognised, and they cover every scoring service in this
    repository:

    1. The uniform ``/score`` contract — parallel ``smiles`` and ``scores``
       lists, optionally with per-molecule ``metadata``.
    2. A list of records — ``[{"smiles": ..., "qed": ..., ...}, ...]`` — as
       returned by ``get_top_candidates`` and ``read_results``.
    """
    if not isinstance(payload, (dict, list)):
        return []

    rows = _rows_from_score_contract(payload) if isinstance(payload, dict) else None
    if rows is None:
        rows = _rows_from_records(payload)
    if not rows:
        return []

    columns = _columns_for(rows)
    if len(columns) < 2:
        return []

    truncated = len(rows) > MAX_TABLE_ROWS
    return [
        {
            "kind": "table",
            "tool": tool_name,
            "columns": columns,
            "rows": [{col: row.get(col) for col in columns} for row in rows[:MAX_TABLE_ROWS]],
            "totalRows": len(rows),
            "truncated": truncated,
        }
    ]


def _rows_from_score_contract(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    scores = payload.get("scores")
    if not isinstance(scores, list):
        return None

    smiles = payload.get("smiles")
    if not isinstance(smiles, list):
        smiles = []

    metadata = payload.get("metadata")
    per_molecule = metadata if isinstance(metadata, dict) else {}
    extra_columns = {
        key: value for key, value in per_molecule.items() if isinstance(value, list) and len(value) == len(scores)
    }

    rows: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        row: dict[str, Any] = {"#": index + 1}
        if index < len(smiles) and isinstance(smiles[index], str):
            row["smiles"] = smiles[index]
        row["score"] = score
        for key, values in extra_columns.items():
            row[key] = _scalar_or_none(values[index])
        rows.append(row)
    return rows


def _rows_from_records(payload: Any) -> list[dict[str, Any]]:
    """Find the longest list of uniform dict records anywhere in the payload."""
    best: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        nonlocal best
        if isinstance(node, list):
            records = [item for item in node if isinstance(item, dict)]
            if len(records) >= 2 and len(records) == len(node) and len(records) > len(best):
                best = records
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value)

    visit(payload)

    rows = []
    for index, record in enumerate(best):
        row: dict[str, Any] = {"#": index + 1}
        for key, value in record.items():
            scalar = _scalar_or_none(value)
            if scalar is not None or value is None:
                row[key] = scalar
        rows.append(row)
    return rows


def _columns_for(rows: list[dict[str, Any]]) -> list[str]:
    """Order columns so the identity of a molecule comes before its numbers."""
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen and key not in _TABLE_NOISE_KEYS:
                seen.append(key)

    def rank(column: str) -> tuple[int, int]:
        if column == "#":
            return (0, 0)
        if column in _SMILES_KEYS:
            return (1, 0)
        if column in ("name", "id", "molecule_id"):
            return (2, 0)
        return (3, seen.index(column))

    return sorted(seen, key=rank)


def _scalar_or_none(value: Any) -> Any:
    """Keep scalars as they are; drop anything a table cell cannot show."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


# --------------------------------------------------------------------------
# path safety
# --------------------------------------------------------------------------


def to_relative_artifact_path(path: str) -> str | None:
    """Return *path* relative to the shared volume, or None if it is outside.

    Most tool results carry absolute container paths, but some, such as
    ``list_shared_assets``, give paths relative to the shared volume, so a
    relative path is read from ``SHARED_ROOT``.  The browser must never be
    able to ask for a file outside the shared volume, so the path is resolved
    and checked against ``SHARED_ROOT`` before it is ever put in a URL.
    """
    try:
        root = SHARED_ROOT.resolve()
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    except (OSError, RuntimeError):
        return None

    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        logger.debug("Ignoring artifact outside the shared volume: %s", path)
        return None


def resolve_artifact(relative: str) -> Path | None:
    """Resolve a browser-supplied relative path to a readable file.

    Returns None when the path escapes the shared volume, does not exist, or
    is not a regular file.
    """
    if not relative or relative.startswith("/"):
        return None

    root = SHARED_ROOT.resolve()
    candidate = (root / relative).resolve()

    if not candidate.is_relative_to(root):
        logger.warning("Rejected artifact request escaping the shared volume: %s", relative)
        return None
    if not candidate.is_file():
        return None
    return candidate


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
