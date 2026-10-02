"""Tests for the Block 09 MCP client wiring."""

from unittest.mock import patch

from reinvent4_mcp.tools import boltz_structure


def test_local_boltz_client_ignores_ambient_proxies() -> None:
    with patch.object(boltz_structure.httpx, "Client") as client:
        boltz_structure._client()

    client.assert_called_once_with(
        base_url=boltz_structure._BOLTZ_URL,
        timeout=boltz_structure._TIMEOUT,
        trust_env=False,
    )
