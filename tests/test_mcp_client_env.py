"""Environment forwarding contracts for the MCP subprocess."""

from reinvent4_agent.mcp_client import MCPClient


def test_internal_service_overrides_are_forwarded() -> None:
    required = {
        "BOLTZ_SERVICE_URL",
        "BOLTZ_SHARED_DIR",
        "SYNTHESIS_SERVICE_URL",
        "INTERACTION_PROFILER_URL",
        "SP_SERVICE_URL",
        "IP_SHARED_DIR",
        "ANNOTATION_SERVICE_URL",
    }

    assert required <= MCPClient._ENV_ALLOW_LIST
