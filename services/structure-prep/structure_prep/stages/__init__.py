"""Stage implementations for the structure preparation pipeline.

Stages are numbered 0-9 and executed sequentially by the pipeline
orchestrator in structure_prep/pipeline.py.

    Stage 0  — inventory.py      : parse & catalog (read-only)
    Stage 1  — assembly.py       : biological assembly / model selection
    Stage 2  — roles.py          : chain / entity role classification
    Stage 3  — curation.py       : HETATM retention / removal
    Stage 4  — altlocs.py        : alternate-location resolution
    Stage 5  — missing.py        : missing atoms / loop modeling
    Stage 6  — protonation.py    : protonation state assignment
    Stage 7  — termini.py        : N/C terminus capping
    Stage 8  — ligand.py         : ligand parameterization
    Stage 9  — minimization.py   : optional geometry minimization
"""
from __future__ import annotations

__all__: list[str] = []
