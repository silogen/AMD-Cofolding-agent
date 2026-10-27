# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse a batch of molecules from JSON or CSV.

A batch is a list of items. Each item has at minimum a ``name``. It may
carry a ``smiles`` (SMILES-only -> taxonomy only), a ``structure`` path
(PDB/mmCIF), or both. Optional fields pin the ligand and receptor and
override pH / source.

JSON forms accepted::

    [ {"name": "1OIU", "smiles": "...", "structure": "1OIU_boltz.pdb", "ph": 7.4}, ... ]
    {"batch_id": "run1", "molecules": [ {...}, ... ]}

CSV: header row with a subset of the columns
``name,smiles,structure,ligand_resname,ligand_chain,receptor_chains,source,ph``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MoleculeInput:
    """One input molecule specification."""

    name: str
    smiles: str | None = None
    structure: str | None = None
    ligand_resname: str | None = None
    ligand_chain: str | None = None
    ligand_keyword: str | None = None
    peptide_ligand_chain: str | None = None
    receptor_chains: list[str] = field(default_factory=list)
    source: str | None = None
    ph: float | None = None

    @property
    def has_structure(self) -> bool:
        return bool(self.structure) and Path(self.structure).exists()


def _coerce_item(raw: dict, base_dir: Path) -> MoleculeInput:
    name = str(raw.get("name") or raw.get("id") or raw.get("target") or "").strip()
    if not name:
        raise ValueError(f"Batch item missing 'name': {raw}")

    structure = raw.get("structure") or raw.get("structure_path") or None
    if structure:
        p = Path(structure)
        if not p.is_absolute():
            p = (base_dir / p)
        structure = str(p)

    rc = raw.get("receptor_chains")
    if isinstance(rc, str):
        receptor_chains = [c.strip() for c in rc.split(",") if c.strip()]
    elif isinstance(rc, (list, tuple)):
        receptor_chains = [str(c).strip() for c in rc if str(c).strip()]
    else:
        receptor_chains = []

    ph = raw.get("ph")
    return MoleculeInput(
        name=name,
        smiles=(raw.get("smiles") or None),
        structure=structure,
        ligand_resname=(raw.get("ligand_resname") or None),
        ligand_chain=(raw.get("ligand_chain") or None),
        ligand_keyword=(raw.get("ligand_keyword") or None),
        peptide_ligand_chain=(raw.get("peptide_ligand_chain") or None),
        receptor_chains=receptor_chains,
        source=(raw.get("source") or None),
        ph=(float(ph) if ph not in (None, "") else None),
    )


def load_batch(path: str | Path) -> tuple[str, list[MoleculeInput]]:
    """Load a batch file. Returns ``(batch_id, molecules)``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Batch file not found: {p}")

    base_dir = p.parent
    if p.suffix.lower() in {".json", ".jsonl"}:
        text = p.read_text()
        if p.suffix.lower() == ".jsonl":
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
            batch_id = p.stem
        else:
            data = json.loads(text)
            if isinstance(data, dict):
                batch_id = str(data.get("batch_id", p.stem))
                items = data.get("molecules", [])
            else:
                batch_id = p.stem
                items = data
        return batch_id, [_coerce_item(it, base_dir) for it in items]

    if p.suffix.lower() in {".csv", ".tsv"}:
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        with p.open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delim)
            items = [dict(row) for row in reader]
        return p.stem, [_coerce_item(it, base_dir) for it in items]

    raise ValueError(f"Unsupported batch file type: {p.suffix} (use .json/.jsonl/.csv)")


__all__ = ["MoleculeInput", "load_batch"]
