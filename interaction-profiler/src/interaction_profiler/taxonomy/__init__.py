"""Taxonomy annotation modules.

Provides residue-level secondary-structure assignment and UniProt/InterPro
feature mapping.

Quick-start
-----------
::

    from interaction_profiler.taxonomy import assign_secondary_structure
    ss_map, method = assign_secondary_structure("my.pdb", chain_ids=["A"])

    from interaction_profiler.taxonomy import map_uniprot_features_to_residues
    residue_annot = map_uniprot_features_to_residues(features, "A", struct_seq, uni_seq)
"""

from __future__ import annotations

from interaction_profiler.taxonomy.secondary_structure import assign_secondary_structure
from interaction_profiler.taxonomy.annotation_mapper import (
    map_uniprot_features_to_residues,
    map_interpro_to_residues,
)

__all__ = [
    "assign_secondary_structure",
    "map_uniprot_features_to_residues",
    "map_interpro_to_residues",
]
