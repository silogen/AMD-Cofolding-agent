"""Models invent nearby names for the discovery tools."""

from reinvent4_agent.mcp_client import MCPClient


def _client() -> MCPClient:
    client = MCPClient("dd-mcp")
    client._tools = [
        {"name": "list_scoring_components", "description": "", "input_schema": {}},
        {"name": "list_transforms", "description": "", "input_schema": {}},
        {"name": "list_available_models", "description": "", "input_schema": {}},
        {"name": "score_molecules", "description": "", "input_schema": {}},
    ]
    return client


def test_aliases_map_onto_list_scoring_components() -> None:
    client = _client()
    for name in (
        "list_components",
        "discover_components",
        "list_available_components",
    ):
        assert client.resolve_tool_name(name) == "list_scoring_components"


def test_exact_name_is_unchanged() -> None:
    assert _client().resolve_tool_name("score_molecules") == "score_molecules"


def test_unknown_name_is_none() -> None:
    assert _client().resolve_tool_name("launch_missiles") is None


def test_unique_substring_does_not_resolve_mutating_tool() -> None:
    assert _client().resolve_tool_name("score") is None
