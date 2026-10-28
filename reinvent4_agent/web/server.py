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
``GET    /api/health``                       liveness and readiness.
``GET    /api/session``                      backend, model and available tools.
``GET    /api/models``                       the chat models, and which one serves.
``POST   /api/models``                       ask for another one; it is swapped.
``GET    /api/artifacts/{path}``             one file from the shared volume.
``GET    /api/sessions``                     saved conversations, newest first.
``PATCH  /api/sessions/{id}``                rename one.
``DELETE /api/sessions/{id}``                delete one.
``GET    /api/sessions/{id}/events``         every event of one, for export.
``GET    /api/sessions/{id}/data``           what one produced or used, with sizes.
``POST   /api/sessions/{id}/purge``          delete one with the data picked from that list.
``POST   /api/tools/{name}``                 call one MCP tool without the LLM.
``GET    /api/jobs/{job_id}/files/{name}``   a results, log or config file of a job.
``GET    /api/services/cofold``              what Boltz is cofolding now (see ``cofold_status.py``).
``GET    /api/jobs/{job_id}/structures``     the cofolded complex of each molecule of a run.
``POST   /api/uploads?session=&name=``       store a file the user brought (raw body).
``POST   /api/uploads/rcsb``                 fetch a PDB entry from RCSB into the same place.
``WS     /api/chat?session=&after=``         the conversation itself.

Conversations are sessions owned by the gateway (see ``sessions.py``): a turn
keeps running when the browser goes away, and the socket replays the events a
returning browser missed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from reinvent4_agent.web import artifacts as shared_files
from reinvent4_agent.web.artifacts import extract_panels, parse_tool_result, resolve_artifact
from reinvent4_agent.web import cofold_status
from reinvent4_agent.web import job_structures
from reinvent4_agent.web import models as chat_models
from reinvent4_agent.web import session_data
from reinvent4_agent.web.events import WebSocketEventHandler
from reinvent4_agent.web.sessions import SessionManager, SessionStore, valid_session_id

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
    ".mol": "chemical/x-mdl-molfile",
    ".smi": "text/plain",
    ".a3m": "text/plain",
}


#: Tools the UI panels may call directly.  Reads only, plus the job and
#: profile actions the panels offer behind a confirmation, and creating a
#: profile from the TTP builder.  Everything else (deleting profiles,
#: scoring, launching training) stays with the agent, so it is part of a
#: conversation and its record.
DIRECT_TOOLS = frozenset(
    {
        "list_jobs",
        "get_optimization_status",
        "get_top_candidates",
        "read_results",
        "get_job_alerts",
        "list_ttps",
        "get_ttp",
        "list_shared_assets",
        "list_external_services",
        "list_available_models",
        "list_scoring_components",
        "list_transforms",
        "fetch_target_info",
        "fetch_compound",
        "predict_admet",
        "predict_retrodmta",
        "stop_optimization",
        "create_ttp",
        "update_ttp",
        "run_optimization",
    }
)

#: Where the job manager keeps one ``job_<id>`` directory per job.
_JOBS_DIR_ENV = "REINVENT4_JOBS_DIR"
_DEFAULT_JOBS_DIR = "/data/reinvent4_mcp/jobs"
_JOB_ID = re.compile(r"^[0-9a-f]{6,32}$")
#: Files of a job the browser may read.  Names, not patterns, except the
#: numbered checkpoints and results REINVENT writes.
_JOB_FILES = re.compile(r"^(reinvent\.log|config\.toml|job_metadata\.json|monitor_actions\.jsonl|[\w.-]+\.csv)$")

#: Files the user may upload, by suffix, and how large they may be.  They
#: land in ``<shared>/uploads/<session>/`` where the tools can read them.
UPLOAD_SUFFIXES = frozenset({".pdb", ".cif", ".mmcif", ".ent", ".sdf", ".mol", ".smi", ".csv"})
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_UPLOAD_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{id}.cif"
RCSB_TIMEOUT_S = 30

#: Reasoning efforts a user message may ask for.
EFFORTS = frozenset({"low", "medium", "high"})


def upload_name(name: str | None) -> str | None:
    """A safe file name for an upload, or None if it cannot be one.

    Only the last path component is kept, so a name never points elsewhere,
    and only letters, digits, dots, dashes and underscores survive.
    """
    if not name:
        return None
    base = re.split(r"[\\/]", name)[-1]
    clean = _UPLOAD_NAME_CHARS.sub("_", base)[-120:].lstrip("._")
    if not clean or Path(clean).suffix.lower() not in UPLOAD_SUFFIXES:
        return None
    return clean


def _upload_session(session: Any) -> str:
    if isinstance(session, str) and (session == "new" or valid_session_id(session)):
        return session
    raise HTTPException(status_code=422, detail="Unknown session")


def _store_upload(session: str, name: str, data: bytes) -> dict[str, Any]:
    directory = shared_files.SHARED_ROOT / "uploads" / session
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    tmp = target.with_name(f".{name}.part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return {
        "path": target.as_posix(),
        "url": f"/api/artifacts/uploads/{session}/{name}",
        "name": name,
        "size": len(data),
    }


def _download_rcsb(pdb_id: str) -> bytes:
    """The mmCIF of one PDB entry; blocking, so run it in a thread."""
    url = RCSB_DOWNLOAD_URL.format(id=pdb_id)
    with urllib.request.urlopen(url, timeout=RCSB_TIMEOUT_S) as response:  # noqa: S310 - fixed https URL
        data = response.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The entry is larger than the upload limit")
    return data


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
    session_dir: str | None = field(default_factory=lambda: os.environ.get("REINVENT4_WEB_SESSION_DIR"))

    def resolved_session_dir(self) -> Path:
        """Saved conversations go next to the transcripts unless set apart."""
        if self.session_dir:
            return Path(self.session_dir).expanduser()
        if self.log_dir:
            return Path(self.log_dir).expanduser() / "web_sessions"
        return Path("~/.reinvent4_agent/web_sessions").expanduser()

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
        self.models = chat_models.ModelControl(fallback_model=config.model)
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call one tool on the shared MCP connection, outside any conversation."""
        if self._mcp is None:
            raise RuntimeError("MCP server is not running")
        return await self._mcp.call_tool(name, arguments)

    @property
    def model(self) -> str:
        """The served id of the model the agents talk to now."""
        spec = self.models.active()
        return spec.served_id if spec is not None else self.config.model

    def model_takes_effort(self) -> bool:
        """Whether the serving model has reasoning efforts to set."""
        spec = self.models.active()
        return spec.supports_effort if spec is not None else True

    def model_choice(self) -> Any:
        """The model a request goes to now, and what that model needs sent.

        Read per request rather than per agent: the endpoint serves one model
        at a time and it can be swapped under a conversation that is already
        under way.
        """
        from reinvent4_agent.backends import ModelChoice

        spec = self.models.active()
        if spec is None:
            return ModelChoice(model=self.config.model)
        return ModelChoice(
            model=spec.served_id,
            extra_body=spec.extra_body,
            supports_reasoning_effort=spec.supports_effort,
        )

    def new_agent(self, handler: WebSocketEventHandler) -> Any:
        """Build an agent for one chat session, sharing the MCP connection."""
        from reinvent4_agent.agent import Agent
        from reinvent4_agent.backends import create_backend
        from reinvent4_agent.system_prompt import SYSTEM_PROMPT

        if self._mcp is None:
            raise RuntimeError("MCP server is not running")

        backend = create_backend(
            self.config.backend,
            model=self.model,
            model_choice=self.model_choice,
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
    sessions = SessionManager(SessionStore(settings.resolved_session_dir()), gateway.new_agent)
    jobs_dir = Path(os.environ.get(_JOBS_DIR_ENV, _DEFAULT_JOBS_DIR))

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        await gateway.start()
        try:
            yield
        finally:
            await sessions.close()
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
            "model": gateway.model,
            "endpoint": settings.base_url or "default",
            "tools": gateway.tool_names,
            "ready": gateway.ready,
        }

    # -- which model serves the UI --------------------------------------------

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        return gateway.models.snapshot()

    @app.post("/api/models")
    async def choose_model(request: Request) -> JSONResponse:
        """Ask the supervisor to serve another model.

        The swap restarts vLLM, so a turn that is running anywhere would lose
        its endpoint mid-answer: those are refused rather than interrupted.
        """
        body = await _json_body(request)
        wanted = chat_models.by_key(body.get("key"))
        if wanted is None:
            raise HTTPException(status_code=400, detail=f"Unknown model: {body.get('key')!r}")
        if not gateway.models.switchable:
            raise HTTPException(status_code=503, detail="This deployment cannot switch models")
        current, state = gateway.models.status()
        if wanted.key == current and state == "ready":
            return JSONResponse(gateway.models.snapshot(), status_code=200)
        if sessions.busy_ids():
            raise HTTPException(status_code=409, detail="Wait for the running turn to finish before switching model")
        try:
            gateway.models.request(wanted.key)
        except OSError as exc:
            logger.warning("Could not ask for model %s: %s", wanted.key, exc)
            raise HTTPException(status_code=503, detail=f"Could not ask for {wanted.label}: {exc}") from exc
        return JSONResponse(gateway.models.snapshot(current=wanted.key, state="loading"), status_code=202)

    @app.get("/api/artifacts/{path:path}")
    async def artifact(path: str) -> FileResponse:
        resolved = resolve_artifact(path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        media_type = _ARTIFACT_MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
        return FileResponse(resolved, media_type=media_type, filename=resolved.name)

    # -- saved conversations -------------------------------------------------

    def _session_or_404(session_id: str) -> Any:
        if not valid_session_id(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            live = sessions.get(session_id)
        except Exception as exc:
            logger.exception("Could not load session %s", session_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if live is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return live

    run_index = session_data.RunIndex(sessions.store.root)

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict[str, Any]]:
        listed = sessions.list()
        try:
            runs = await asyncio.to_thread(run_index.runs, [s["id"] for s in listed], _data_roots())
        except Exception:
            logger.exception("Could not tell which conversations launched which runs")
            runs = {}
        return [{**s, "runs": runs[s["id"]]} if s["id"] in runs else s for s in listed]

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        title = body.get("title")
        if not isinstance(title, str) or not title.strip():
            raise HTTPException(status_code=422, detail="A non-empty title is required")
        _session_or_404(session_id)
        return sessions.rename(session_id, title) or {}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        _session_or_404(session_id)
        if not sessions.delete(session_id):
            raise HTTPException(status_code=409, detail="Stop the running turn before deleting this conversation")
        return {"deleted": session_id}

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> dict[str, Any]:
        live = _session_or_404(session_id)
        return {"session": live.summary(), "events": live.events}

    # -- what a conversation produced, and removing it with the conversation --

    def _data_roots() -> session_data.Roots:
        return session_data.Roots(shared=shared_files.SHARED_ROOT, jobs=jobs_dir)

    @app.get("/api/sessions/{session_id}/data")
    async def session_data_view(session_id: str) -> dict[str, Any]:
        live = _session_or_404(session_id)
        events = list(live.events)
        roots = _data_roots()
        store_root = sessions.store.root

        def build() -> dict[str, Any]:
            others = session_data.stored_references(store_root, roots.shared, session_id)
            record = session_data.directory_bytes(store_root / session_id)
            return session_data.inventory(session_id, events, roots, others, record)

        body = await asyncio.to_thread(build)
        body["busy"] = live.busy
        return body

    @app.post("/api/sessions/{session_id}/purge")
    async def purge_session(session_id: str, request: Request) -> dict[str, Any]:
        """Delete a conversation and the data picked from its data view.

        The conversation goes first, under the same rule as ``DELETE``: a
        running turn refuses it and then nothing at all is removed.
        """
        body = await _json_body(request)
        items = body.get("items", [])
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise HTTPException(status_code=422, detail="items must be a list of item ids")
        live = _session_or_404(session_id)
        events = list(live.events)
        roots = _data_roots()
        store_root = sessions.store.root
        record = session_data.directory_bytes(store_root / session_id)
        if not sessions.delete(session_id):
            raise HTTPException(status_code=409, detail="Stop the running turn before deleting this conversation")

        def run() -> dict[str, Any]:
            others = session_data.stored_references(store_root, roots.shared, session_id)
            return session_data.purge(session_id, events, roots, others, items)

        try:
            summary = await asyncio.to_thread(run)
        except Exception as exc:
            logger.exception("Could not delete the data of session %s", session_id)
            summary = {"removed": [], "kept": [], "freedBytes": 0, "error": str(exc)}
        return {"deleted": session_id, "conversationBytes": record, **summary}

    # -- direct tool calls and job files --------------------------------------

    @app.post("/api/tools/{name}")
    async def call_tool(name: str, request: Request) -> dict[str, Any]:
        if name not in DIRECT_TOOLS:
            raise HTTPException(status_code=403, detail=f"{name} can only be used through the agent")
        if not gateway.ready:
            raise HTTPException(status_code=503, detail="The MCP server is not ready")
        if name not in gateway.tool_names:
            raise HTTPException(status_code=404, detail=f"No tool named {name}")
        body = await _json_body(request)
        arguments = body.get("arguments", {})
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=422, detail="arguments must be an object")
        try:
            raw = await gateway.call_tool(name, arguments)
        except Exception as exc:
            logger.warning("Direct call of %s failed: %s", name, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        parsed = parse_tool_result(raw)
        return {
            "name": name,
            "result": parsed if parsed is not None else raw,
            "panels": extract_panels(name, raw, arguments),
        }

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def job_file(job_id: str, name: str) -> FileResponse:
        if not _JOB_ID.match(job_id) or not _JOB_FILES.match(name):
            raise HTTPException(status_code=404, detail="File not found")
        path = jobs_dir / f"job_{job_id}" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        media_type = _ARTIFACT_MEDIA_TYPES.get(path.suffix.lower(), "text/plain")
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/api/services/cofold")
    async def cofold_service() -> dict[str, Any]:
        return await cofold_status.cofold_status()

    @app.get("/api/jobs/{job_id}/structures")
    async def job_structure_list(job_id: str) -> list[dict[str, Any]]:
        if not _JOB_ID.match(job_id):
            raise HTTPException(status_code=404, detail="Job not found")
        found = await asyncio.to_thread(job_structures.job_structures, jobs_dir / f"job_{job_id}", shared_files.SHARED_ROOT)
        if found is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return found

    # -- files the user brings ------------------------------------------------

    @app.post("/api/uploads")
    async def upload(request: Request) -> dict[str, Any]:
        session_id = _upload_session(request.query_params.get("session"))
        name = upload_name(request.query_params.get("name"))
        if name is None:
            allowed = ", ".join(sorted(UPLOAD_SUFFIXES))
            raise HTTPException(status_code=422, detail=f"Only these files can be uploaded: {allowed}")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The file is larger than 20 MB")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="The file is larger than 20 MB")
        if not data:
            raise HTTPException(status_code=422, detail="The file is empty")
        return _store_upload(session_id, name, bytes(data))

    @app.post("/api/uploads/rcsb")
    async def upload_from_rcsb(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        session_id = _upload_session(body.get("session"))
        pdb_id = body.get("id")
        if not isinstance(pdb_id, str) or not _PDB_ID.match(pdb_id.strip()):
            raise HTTPException(status_code=422, detail="A PDB id is four characters, e.g. 5UF0")
        pdb_id = pdb_id.strip().upper()
        try:
            data = await asyncio.to_thread(_download_rcsb, pdb_id)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise HTTPException(status_code=404, detail=f"RCSB has no entry {pdb_id}") from exc
            raise HTTPException(status_code=502, detail=f"RCSB answered {exc.code}") from exc
        except (OSError, ValueError) as exc:
            logger.warning("Could not fetch %s from RCSB: %s", pdb_id, exc)
            raise HTTPException(status_code=502, detail=f"Could not fetch {pdb_id} from RCSB: {exc}") from exc
        return _store_upload(session_id, f"{pdb_id}.cif", data)

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

        requested = websocket.query_params.get("session")
        try:
            after = max(0, int(websocket.query_params.get("after") or 0))
        except ValueError:
            after = 0

        try:
            live = sessions.get(requested) if valid_session_id(requested) else None
            notice = None
            if requested and live is None:
                notice = "That conversation no longer exists, so a new one was started."
                after = 0
            if live is None:
                live = sessions.create()
        except Exception as exc:
            logger.exception("Could not open a session")
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close()
            return

        await websocket.send_json(
            {
                "type": "ready",
                "model": gateway.model,
                "backend": settings.backend,
                "reasoningEffort": settings.reasoning_effort,
                "tools": gateway.tool_names,
            }
        )
        # Subscribe before taking the snapshot, so no event falls between the
        # replay and the live stream; the sender skips what the replay sent.
        queue = live.subscribe()
        snapshot = live.last_seq
        await websocket.send_json({"type": "session", **live.summary(), "lastSeq": snapshot})
        if notice:
            await websocket.send_json({"type": "notice", "message": notice})
        for event in live.events_after(after):
            await websocket.send_json(event)

        sender = asyncio.create_task(_forward_events(websocket, queue, snapshot))
        try:
            await _receive_loop(websocket, live, gateway.model_takes_effort)
        except WebSocketDisconnect:
            logger.info("Client disconnected from session %s", live.id)
        finally:
            # The turn goes on: its events are recorded, and the browser that
            # comes back gets them on reconnect.
            live.unsubscribe(queue)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender

    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="The body must be JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="The body must be a JSON object")
    return body


async def _forward_events(websocket: WebSocket, queue: asyncio.Queue[dict[str, Any]], after: int) -> None:
    """Send the session's new events to the browser until cancelled."""
    while True:
        event = await queue.get()
        # Unnumbered events (a notice) belong to no replay, so they are never
        # the duplicates this skips.
        if "seq" in event and event["seq"] <= after:
            continue
        try:
            await websocket.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            return


async def _receive_loop(websocket: WebSocket, live: Any, takes_effort: Any = None) -> None:
    """Read the browser's messages: a new question, or stop the current turn.

    Turns are serialised per session: the agent keeps conversation state, so
    two overlapping turns would interleave tool calls into one history.  A
    question that arrives while a turn runs is refused rather than queued,
    and only this browser is told, since it is not part of the conversation.

    An effort asked for while a model without efforts is serving is dropped:
    that model thinks regardless, and the request carries the chat template
    option instead (see ``models.py``).
    """
    while True:
        raw = await websocket.receive_text()
        try:
            message = json.loads(raw)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            await websocket.send_json({"type": "notice", "message": "Malformed message."})
            continue

        kind = message.get("type")
        if kind == "cancel":
            if not live.cancel():
                await websocket.send_json({"type": "notice", "message": "Nothing is running."})
            continue
        if kind != "user_message":
            await websocket.send_json({"type": "notice", "message": f"Unknown message type: {kind!r}"})
            continue

        effort = message.get("effort")
        if effort is not None and (not isinstance(effort, str) or effort not in EFFORTS):
            await websocket.send_json({"type": "notice", "message": f"Unknown effort: {effort!r}"})
            continue
        if effort and takes_effort is not None and not takes_effort():
            effort = None
        content = (message.get("content") or "").strip()
        if content and not live.start_turn(content, effort=effort):
            await websocket.send_json(
                {"type": "notice", "message": "The agent is still working on the previous request."}
            )
