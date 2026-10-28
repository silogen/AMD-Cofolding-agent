# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Endpoint params must be per-endpoint values that REINVENT4 can collect."""

import pytest

from reinvent4_mcp.tools.scoring import _normalize_endpoint_params


@pytest.mark.parametrize(
    "given, expected",
    [
        ({"radius": 3}, {"radius": 3}),
        ({"radius": [3]}, {"radius": 3}),
        ({"use_counts": [[True]]}, {"use_counts": True}),
        ({"smiles": ["CCO"]}, {"smiles": ["CCO"]}),
        ({"smiles": [["CCO", "CCN"]]}, {"smiles": ["CCO", "CCN"]}),
        ({"smiles": ["CCO", "CCN"]}, {"smiles": ["CCO", "CCN"]}),
    ],
)
def test_params_are_per_endpoint_values(given, expected):
    assert _normalize_endpoint_params(given) == expected


def test_reinvent_accepts_a_tanimoto_similarity_component():
    config = pytest.importorskip("reinvent.scoring.config", exc_type=ImportError)
    params = _normalize_endpoint_params(
        {"smiles": ["CC(=O)Nc1ccc(O)cc1"], "radius": [3], "use_counts": [True], "use_features": False}
    )
    components = config.get_components(
        [{"TanimotoSimilarity": {"endpoint": [{"name": "similarity", "weight": 1.0, "params": params}]}}]
    )
    assert components
