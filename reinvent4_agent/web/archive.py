# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bulk access to the shared volume: every file in it, and many as one ZIP.

``GET  /api/files``     the files on the shared volume, with size and time (reused for a few seconds).
``POST /api/archive``   ``{"paths": [...], "name": "..."}`` as a streamed ZIP.

Paths are resolved by the same rule as ``/api/artifacts`` (relative to the
shared volume, symlinks followed and refused when they leave it), plus the
job files ``/api/jobs/{id}/files/{name}`` serves, written ``job_<id>/<name>``.
One unreadable path refuses the whole archive, so a ZIP is never silently
short of a file.  The ZIP is written while it is sent, a chunk at a time.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from reinvent4_agent.web import artifacts as shared_files
from reinvent4_agent.web.artifacts import resolve_artifact

logger = logging.getLogger(__name__)

MAX_FILES = 5000
MAX_BYTES = 2 * 1024**3
#: Files the listing returns at most; the rest is reported as truncated.
MAX_LISTED = 20000
CHUNK = 1 << 20

#: The job files the gateway serves, as ``job_<id>/<name>``; the same names as ``server._JOB_FILES``.
JOB_PATH = re.compile(r"^job_([0-9a-f]{6,32})/(reinvent\.log|config\.toml|job_metadata\.json|monitor_actions\.jsonl|[\w.-]+\.csv)$")

#: Already compressed: deflating them again costs time and saves nothing.
_STORED_SUFFIXES = frozenset({".gz", ".zip", ".npz", ".bz2", ".xz", ".zst", ".pdf", ".png", ".jpg", ".jpeg", ".pt", ".ckpt"})
_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class ArchiveError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def archive_name(name: Any) -> str:
    """A safe download name ending in ``.zip``."""
    stem = _NAME_CHARS.sub("_", name if isinstance(name, str) else "")[:80].strip("._")
    stem = re.sub(r"\.zip$", "", stem, flags=re.IGNORECASE) or "assets"
    return f"{stem}.zip"


def _resolve(path: str, jobs_dir: Path) -> tuple[Path, str] | None:
    """The file and its name in the archive, or None when it may not be read."""
    clean = path.replace("\\", "/").strip()
    job = JOB_PATH.match(clean)
    if job:
        candidate = jobs_dir / f"job_{job[1]}" / job[2]
        return (candidate, f"jobs/{clean}") if candidate.is_file() else None
    resolved = resolve_artifact(clean)
    if resolved is None:
        return None
    return resolved, resolved.relative_to(shared_files.SHARED_ROOT.resolve()).as_posix()


def select_files(paths: Any, jobs_dir: Path) -> list[tuple[Path, str]]:
    """The files to archive, deduplicated, or ArchiveError for the whole request."""
    if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
        raise ArchiveError(400, "paths must be a non-empty list of file paths")
    unique = list(dict.fromkeys(paths))
    if len(unique) > MAX_FILES:
        raise ArchiveError(413, f"{len(unique)} files asked for; one archive holds at most {MAX_FILES}")
    files: list[tuple[Path, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    total = 0
    for path in unique:
        found = _resolve(path, jobs_dir)
        if found is None:
            missing.append(path)
            continue
        if found[1] in seen:
            continue
        seen.add(found[1])
        try:
            total += found[0].stat().st_size
        except OSError:
            missing.append(path)
            continue
        files.append(found)
    if missing:
        shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
        raise ArchiveError(404, f"Not found or not readable: {shown}")
    if total > MAX_BYTES:
        raise ArchiveError(413, f"The files add up to {total / 1024**3:.1f} GB; one archive holds at most {MAX_BYTES / 1024**3:.0f} GB")
    return files


class _Sink(io.RawIOBase):
    """Collects what ZipFile writes until it is sent; not seekable, so ZipFile streams."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data: Any) -> int:
        self.buffer += data
        return len(data)

    def drain(self) -> bytes:
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


def zip_chunks(files: list[tuple[Path, str]]) -> Iterator[bytes]:
    """The ZIP of ``(file, name in archive)`` pairs, a chunk at a time."""
    sink = _Sink()
    with zipfile.ZipFile(sink, "w") as archive:
        for path, arcname in files:
            info = zipfile.ZipInfo.from_file(path, arcname, strict_timestamps=False)
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w") as target:
                while chunk := source.read(CHUNK):
                    target.write(chunk)
                    if len(sink.buffer) >= CHUNK:
                        yield sink.drain()
            if sink.buffer:
                yield sink.drain()
    yield sink.drain()


def list_files(root: Path, limit: int = MAX_LISTED) -> dict[str, Any]:
    """Regular files under ``root``, relative to it; hidden entries and symlinks are left out."""
    files: list[dict[str, Any]] = []
    truncated = False
    stack = [root]
    while stack and not truncated:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if len(files) >= limit:
                        truncated = True
                        break
                    stat = entry.stat(follow_symlinks=False)
                    rel = Path(entry.path).relative_to(root).as_posix()
                    files.append({"path": rel, "size": stat.st_size, "mtime": int(stat.st_mtime)})
            except OSError:
                continue
    return {"files": files, "truncated": truncated}


#: How long a listing is reused.  The volume is on Lustre, where walking
#: thousands of entries is slow, and the tab lists it on every open.
LISTING_TTL_S = 15.0
_listing: tuple[float, dict[str, Any]] | None = None
_listing_lock = threading.Lock()


def cached_listing(root: Path) -> dict[str, Any]:
    """``list_files(root)``, reused for ``LISTING_TTL_S``; one scan at a time, later callers get its result."""
    global _listing
    with _listing_lock:
        now = time.monotonic()
        if _listing is None or now - _listing[0] > LISTING_TTL_S:
            _listing = (now, list_files(root))
        return _listing[1]


def register(app: Any, jobs_dir: Path) -> None:
    """Add the listing and archive routes to the gateway."""
    # Imported here so the module, like ``artifacts``, loads without FastAPI.
    import asyncio

    from fastapi import HTTPException, Request
    from fastapi.responses import StreamingResponse

    @app.get("/api/files")
    async def shared_listing() -> dict[str, Any]:
        root = shared_files.SHARED_ROOT.resolve()
        if not root.is_dir():
            return {"files": [], "truncated": False}
        return await asyncio.to_thread(cached_listing, root)

    @app.post("/api/archive")
    async def archive(request: Request) -> StreamingResponse:
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="The body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="The body must be a JSON object")
        try:
            files = await asyncio.to_thread(select_files, body.get("paths"), jobs_dir)
        except ArchiveError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
        name = archive_name(body.get("name"))
        logger.info("Archiving %d files as %s", len(files), name)
        return StreamingResponse(
            zip_chunks(files),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )
