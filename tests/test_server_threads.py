# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tools run in worker threads, so a long call does not hold up the others."""

import asyncio
import time

from reinvent4_mcp import server


def test_tools_keep_their_schemas():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert len(tools) == 26
    schema = tools["run_optimization"].inputSchema
    assert "seed_smiles" in schema["properties"]
    assert "ttp_id" in schema["properties"]
    assert set(tools["prepare_structure"].inputSchema["properties"]) == {"pdb_id", "path", "chain", "template"}


def test_a_slow_tool_does_not_block_a_fast_one():
    @server.tool()
    def _slow_probe() -> str:
        time.sleep(1.0)
        return "slow"

    @server.tool()
    def _fast_probe() -> str:
        return "fast"

    async def race() -> float:
        slow = asyncio.create_task(server.mcp.call_tool("_slow_probe", {}))
        await asyncio.sleep(0.05)
        start = time.monotonic()
        await server.mcp.call_tool("_fast_probe", {})
        fast = time.monotonic() - start
        await slow
        return fast

    try:
        assert asyncio.run(race()) < 0.5
    finally:
        for name in ("_slow_probe", "_fast_probe"):
            server.mcp._tool_manager._tools.pop(name, None)
