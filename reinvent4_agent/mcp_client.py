"""MCP client for connecting to the Drug Discovery MCP server via stdio.

Manages the lifecycle of a subprocess running the MCP server, provides
methods to list tools and call them by name.  Tool schema conversion to
LLM-specific formats is handled by the backend layer, not here.
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

#: Models invent nearby names for the discovery tools.  Map those onto
#: the real MCP tools so a guessed name still runs.
_TOOL_ALIASES = {
    "list_components": "list_scoring_components",
    "discover_components": "list_scoring_components",
    "list_available_components": "list_scoring_components",
    "get_components": "list_scoring_components",
    "list_scorers": "list_scoring_components",
    "list_scoring": "list_scoring_components",
    "get_scoring_components": "list_scoring_components",
    "list_transform": "list_transforms",
    "get_transforms": "list_transforms",
    "list_model": "list_available_models",
    "list_models": "list_available_models",
    "list_available_model": "list_available_models",
}


def _canonical_tool_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


class MCPClient:
    """Async context manager for communicating with the Drug Discovery MCP server.

    Usage::

        async with MCPClient("dd-mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("list_scoring_components", {})
    """

    def __init__(
        self,
        server_command: str,
        server_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.server_command = server_command
        self.server_args = server_args or []
        self.env = env
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools: list[dict[str, Any]] | None = None

    _ENV_ALLOW_LIST: frozenset[str] = frozenset(
        {
            "PATH",
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "SHELL",
            "TMPDIR",
            "TERM",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "REINVENT4_ROOT",
            "REINVENT4_MCP_CONFIG",
            # GPU access for the MCP server + the RL worker it spawns. Under
            # Apptainer --nv/--rocm, LD_LIBRARY_PATH points at the injected
            # driver libs (libcuda.so, ...); dropping it makes torch report
            # "Found no NVIDIA driver". The *_VISIBLE_DEVICES vars carry the
            # per-vendor device pinning set by run.sh.
            "LD_LIBRARY_PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        }
    )

    async def __aenter__(self) -> MCPClient:
        if self.env is not None:
            env = self.env
        else:
            env = {
                k: v
                for k, v in os.environ.items()
                if k in self._ENV_ALLOW_LIST
            }
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=env,
        )

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_result.tools
        ]

        logger.info(
            "Connected to MCP server (%d tools available)",
            len(self._tools),
        )
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.aclose()
        self._session = None
        self._tools = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected; use 'async with'")
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return cached MCP tool definitions.

        Each entry has ``name``, ``description``, and ``input_schema``
        keys.  LLM backends convert these to their native format.
        """
        if self._tools is None:
            raise RuntimeError("MCPClient is not connected; use 'async with'")
        return list(self._tools)

    def resolve_tool_name(self, name: str) -> str | None:
        """The registered tool ``name`` refers to, or ``None`` if unknown."""
        if self._tools is None:
            raise RuntimeError("MCPClient is not connected; use 'async with'")
        available = {t["name"]: t["name"] for t in self._tools}
        if name in available:
            return name
        key = _canonical_tool_name(name)
        by_canonical = {_canonical_tool_name(n): n for n in available}
        if key in by_canonical:
            return by_canonical[key]
        alias = _TOOL_ALIASES.get(key)
        if alias and alias in available:
            return alias
        return None

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Call an MCP tool by name and return the text result.

        Raises ValueError if the tool call returns an error.
        """
        result = await self.session.call_tool(name, arguments)

        text_parts = []
        for content_block in result.content:
            if hasattr(content_block, "text"):
                text_parts.append(content_block.text)

        text = "\n".join(text_parts)

        if result.isError:
            raise ValueError(f"Tool '{name}' returned error: {text}")

        return text
