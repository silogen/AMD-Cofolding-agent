# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TTP (Therapeutic Target Profile) schema, storage, and mapping.

A TTP fully specifies a multi-property optimization objective:
protein target, pre-scoring descriptors, scoring objectives,
and structural constraints.
"""

from reinvent4_mcp.ttp.schema import TTP
from reinvent4_mcp.ttp.store import TTPStore
from reinvent4_mcp.ttp.mapper import (
    describe_external_objectives,
    extract_native_components,
    get_mapper_summary,
    target_sequence,
    ttp_to_native_scorer_config,
    ttp_to_scorer_config,
)

__all__ = [
    "TTP",
    "TTPStore",
    "describe_external_objectives",
    "extract_native_components",
    "get_mapper_summary",
    "target_sequence",
    "ttp_to_native_scorer_config",
    "ttp_to_scorer_config",
]
