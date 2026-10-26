# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FastAPI gateway in front of the drug design agent.

The agent is a conversation, not a request/response API: one question can
trigger a chain of tool calls that runs for minutes.  So the browser holds a
WebSocket and receives the same events the terminal client prints, as they
happen.  Files the tools write (structures, results) stay on the shared
volume and are fetched over ``/api/artifacts``.

Endpoints
---------
``GET  /api/health``           liveness and readiness.
``GET  /api/session``          backend, model and available tools.
``GET  /api/artifacts/{path}`` one file from the shared volume.
``WS   /api/chat``             the conversation itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from reinvent4_agent.web.artifacts import resolve_artifact
from reinvent4_agent.web.events import WebSocketEventHandler

logger = logging.getLogger(__name__)

#: Media types for the structure formats the 3D viewer loads.  Served as text
#: so the browser hands the viewer a string rather than a download.
_ARTIFACT_MEDIA_TYPES = {
    ".cif": "chemical/x-cif",
    ".mmcif": "chemical/x-cif",
    ".pdb": "chemical/x-pdb",
    ".ent": "chemical/x-pdb",
    ".csv": "text/csv",
    ".json": "application/json",
    ".sdf": "chemical/x-mdl-sdfile",
    ".smi": "text/plain",
    ".a3m": "text/plain",
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return None


#: Browser origins that may open the chat socket, in addition to the
#: gateway's own host and to localhost. Comma-separated full origins, for
#: example ``http://my-proxy:8080``.
_EXTRA_ORIGINS_ENV = "REINVENT4_WEB_ALLOWED_ORIGINS"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def origin_allowed(origin: str | None, host: str | None) -> bool:
    """Decide if a browser page at *origin* may open the chat socket.

    Browsers do not apply the same-origin policy to WebSockets, so without
    this check any web page open in the user's browser could drive the agent
    through the SSH tunnel. Clients that send no Origin are not browsers and
    are allowed.
    """
    if not origin:
        return True
    extra = {item.strip().rstrip("/") for item in os.environ.get(_EXTRA_ORIGINS_ENV, "").split(",") if item.strip()}
    if origin.rstrip("/") in extra:
        return True
    try:
        origin_host = urlsplit(origin).hostname
        own_host = urlsplit(f"//{host}").hostname if host else None
    except ValueError:
        return False
    if origin_host is None:
        return False
    return origin_host in _LOOPBACK_HOSTS or origin_host == own_host


#: Shared secret for clusters where the gateway port is reachable by other
#: users (on LUMI, from every login node). When set, every request must carry
#: it in the ``X-DD-Token`` header; the dev proxy adds it (VITE_PROXY_TOKEN).
_TOKEN_ENV = "REINVENT4_WEB_TOKEN"
TOKEN_HEADER = "x-dd-token"


def token_ok(sent: str | None, expected: str | None) -> bool:
    """Whether a request may proceed under the configured access token."""
    if not expected:
        return True
    return sent is not None and hmac.compare_digest(sent.encode(), expected.encode())


@dataclass
class _Session:
    """State of one chat connection."""

    running: asyncio.Task[str] | None = None


@dataclass
class WebConfig:
    """Gateway settings, read from the same environment as ``dd-agent``.

    Keeping one source of configuration means the web UI and the terminal
    client always talk to the same model through the same endpoint.
    """

    backend: str = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_BACKEND", "anthropic"))
    model: str = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_MODEL", "claude-sonnet-4"))
    base_url: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_BASE_URL"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_API_KEY"))
    gateway_key: str | None = field(default_factory=lambda: os.environ.get("LLM_GATEWAY_KEY"))
    api_version: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_API_VERSION"))
    verify_ssl: bool = field(default_factory=lambda: not _env_flag("REINVENT4_AGENT_NO_VERIFY_SSL"))
    thinking_budget: int | None = field(default_factory=lambda: _env_int("REINVENT4_AGENT_THINKING_BUDGET"))
    reasoning_effort: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_REASONING_EFFORT"))
    max_tokens: int = field(default_factory=lambda: _env_int("REINVENT4_AGENT_MAX_TOKENS") or 4096)
    log_dir: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_AGENT_LOG_DIR"))
    server_command: str = field(default_factory=lambda: os.environ.get("REINVENT4_MCP_COMMAND", "dd-mcp"))

    def resolved_api_key(self) -> str | None:
        """Mirror the terminal client's key resolution, without exiting."""
        api_key = self.api_key
        if not api_key and not self.gateway_key:
            fallback = "ANTHROPIC_API_KEY" if self.backend == "anthropic" else "OPENAI_API_KEY"
            api_key = os.environ.get(fallback)
        if self.gateway_key and not api_key:
            api_key = "dummy"
        return api_key


class AgentGateway:
    """Owns the MCP connection and hands out one agent per chat session.

    The MCP server is a subprocess speaking stdio.  Starting one per browser
    tab would be wasteful and would multiply the job state, so a single client
    is started with the app and shared, exactly as the terminal client shares
    one client between its Driver and Reviewer.
    """

    def __init__(self, config: WebConfig) -> None:
        self.config = config
        self._mcp: Any | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []
        self.ready = False
        self.error: str | None = None

    async def start(self) -> None:
        # Imported inside the try: if the MCP server cannot be reached the
        # gateway must still come up and report "unavailable" on /api/health,
        # so a health check fails cleanly instead of the gateway crash-looping.
        self._stack = contextlib.AsyncExitStack()
        try:
            from reinvent4_agent.mcp_client import MCPClient

            self._mcp = await self._stack.enter_async_context(MCPClient(server_command=self.config.server_command))
            self._tools = await self._mcp.list_tools()
            self.ready = True
            logger.info(
                "MCP server ready: %d tool(s) — %s",
                len(self._tools),
                ", ".join(tool["name"] for tool in self._tools),
            )
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Failed to start the MCP server")
            await self.stop()

    async def stop(self) -> None:
        self.ready = False
        if self._stack is not None:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            self._stack = None
        self._mcp = None

    @property
    def tool_names(self) -> list[str]:
        return [tool["name"] for tool in self._tools]

    def new_agent(self, handler: WebSocketEventHandler) -> Any:
        """Build an agent for one chat session, sharing the MCP connection."""
        from reinvent4_agent.agent import Agent
        from reinvent4_agent.backends import create_backend
        from reinvent4_agent.system_prompt import SYSTEM_PROMPT

        if self._mcp is None:
            raise RuntimeError("MCP server is not running")

        backend = create_backend(
            self.config.backend,
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.resolved_api_key(),
            gateway_key=self.config.gateway_key,
            api_version=self.config.api_version,
            verify_ssl=self.config.verify_ssl,
            thinking_budget=self.config.thinking_budget,
            reasoning_effort=self.config.reasoning_effort,
            system_prompt=SYSTEM_PROMPT,
        )
        return Agent(
            self._mcp,
            backend,
            max_tokens=self.config.max_tokens,
            log_dir=self.config.log_dir,
            event_handler=handler,
        )


def create_app(config: WebConfig | None = None) -> FastAPI:
    """Build the gateway application."""
    settings = config or WebConfig()
    gateway = AgentGateway(settings)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        await gateway.start()
        try:
            yield
        finally:
            await gateway.stop()

    app = FastAPI(
        title="AI Cofolding Drug Design Agent",
        description="Web gateway for the drug design agent.",
        lifespan=lifespan,
    )

    access_token = os.environ.get(_TOKEN_ENV) or None

    if access_token:

        @app.middleware("http")
        async def require_token(request: Request, call_next: Any) -> Any:
            if not token_ok(request.headers.get(TOKEN_HEADER), access_token):
                return JSONResponse({"detail": "Missing or wrong access token"}, status_code=401)
            return await call_next(request)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        body = {
            "status": "ok" if gateway.ready else "starting",
            "mcp": "ready" if gateway.ready else "unavailable",
            "tools": len(gateway.tool_names),
        }
        if gateway.error:
            body["error"] = gateway.error
        return JSONResponse(body, status_code=200 if gateway.ready else 503)

    @app.get("/api/session")
    async def session() -> dict[str, Any]:
        return {
            "backend": settings.backend,
            "model": settings.model,
            "endpoint": settings.base_url or "default",
            "tools": gateway.tool_names,
            "ready": gateway.ready,
        }

    @app.get("/api/artifacts/{path:path}")
    async def artifact(path: str) -> FileResponse:
        resolved = resolve_artifact(path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        media_type = _ARTIFACT_MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        return FileResponse(resolved, media_type=media_type, filename=resolved.name)

    @app.websocket("/api/chat")
    async def chat(websocket: WebSocket) -> None:
        if not token_ok(websocket.headers.get(TOKEN_HEADER), access_token):
            logger.warning("Rejected chat connection without a valid access token")
            await websocket.close(code=1008)
            return
        origin = websocket.headers.get("origin")
        if not origin_allowed(origin, websocket.headers.get("host")):
            logger.warning("Rejected chat connection from origin %s", origin)
            await websocket.close(code=1008)
            return

        await websocket.accept()

        if not gateway.ready:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": gateway.error or "The agent is still starting. Try again in a moment.",
                }
            )
            await websocket.close()
            return

        handler = WebSocketEventHandler()
        try:
            agent = gateway.new_agent(handler)
        except Exception as exc:
            logger.exception("Could not create an agent for this session")
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close()
            return

        await websocket.send_json(
            {
                "type": "ready",
                "model": settings.model,
                "backend": settings.backend,
                "tools": gateway.tool_names,
            }
        )

        session = _Session()
        sender = asyncio.create_task(_forward_events(websocket, handler))
        try:
            await _receive_loop(websocket, agent, handler, session)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        finally:
            # Nobody reads the events of this agent any more, and the next
            # connection gets a new agent, so a turn left running would only
            # keep calling tools for no one.
            if session.running is not None and not session.running.done():
                logger.info("Cancelling the turn of a disconnected client")
                session.running.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await session.running
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    return app


async def _forward_events(websocket: WebSocket, handler: WebSocketEventHandler) -> None:
    """Send queued agent events to the browser until the task is cancelled."""
    while True:
        event = await handler.drain()
        try:
            await websocket.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            return


async def _receive_loop(
    websocket: WebSocket,
    agent: Any,
    handler: WebSocketEventHandler,
    session: _Session,
) -> None:
    """Read user messages and run one agent turn at a time.

    Turns are serialised per connection: the agent keeps conversation state,
    so two overlapping turns would interleave tool calls into one history.
    A message that arrives while a turn is running is rejected rather than
    queued, so the user sees why nothing happened.
    """
    while True:
        raw = await websocket.receive_text()
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            handler.emit({"type": "error", "message": "Malformed message."})
            continue

        if message.get("type") != "user_message":
            handler.emit({"type": "error", "message": f"Unknown message type: {message.get('type')!r}"})
            continue

        content = (message.get("content") or "").strip()
        if not content:
            continue

        if session.running is not None and not session.running.done():
            handler.emit(
                {
                    "type": "error",
                    "message": "The agent is still working on the previous request.",
                }
            )
            continue

        handler.emit({"type": "status", "state": "busy"})
        session.running = asyncio.create_task(_run_turn(agent, handler, content))


async def _run_turn(agent: Any, handler: WebSocketEventHandler, content: str) -> str:
    """Run one agent turn, reporting the answer or the failure."""
    try:
        answer = await agent.chat(content)
        handler.emit(
            {
                "type": "assistant_message",
                "content": answer,
                "inputTokens": getattr(agent, "total_input_tokens", 0),
                "outputTokens": getattr(agent, "total_output_tokens", 0),
            }
        )
        return answer
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Agent turn failed")
        handler.emit({"type": "error", "message": str(exc)})
        return ""
    finally:
        handler.emit({"type": "status", "state": "idle"})
