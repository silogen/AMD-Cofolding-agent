# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a conversation produced or used, and removing it with the conversation.

Nothing records which files belong to which conversation, so the association
is derived from two sources:

* the conversation's recorded events: paths, job ids and TTP ids in what the
  user typed, in the agent's answers, in tool call arguments and in tool
  results (see ``collect_references``);
* the conversation's own uploads folder, ``<shared>/uploads/<session id>/``.

Results of the inventory tools (``list_shared_assets``, ``list_jobs`` ...)
are not read: they name every file and job there is, and a conversation that
merely listed them has not used them.

Deletion is deliberately narrow.  Only three kinds of target exist, each
checked against its root after resolving links:

* a file in this conversation's uploads folder;
* one Boltz run directory, ``<shared>/predictions/<run>``;
* one finished REINVENT job directory, ``<jobs>/job_<id>``, that this
  conversation launched.

Anything another stored conversation also references is kept, and so is any
job that is not finished: runs are stopped by the user, never from here.
TTPs and shared caches (``msa_cache/``, ``batches/``) are only ever listed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from reinvent4_agent.web.artifacts import STRUCTURE_SUFFIXES
from reinvent4_agent.web.sessions import valid_session_id

logger = logging.getLogger(__name__)

#: Tools whose results enumerate everything on the volume or in the job
#: manager, so their contents say nothing about what a conversation used.
INVENTORY_TOOLS = frozenset(
    {
        "list_shared_assets",
        "list_jobs",
        "get_job_alerts",
        "list_ttps",
        "list_external_services",
        "list_available_models",
        "list_scoring_components",
        "list_transforms",
    }
)

#: Tools whose result names a job this conversation started.
LAUNCH_TOOLS = frozenset({"run_optimization", "run_transfer_learning"})

#: Job states after which the job's directory is no longer written to.
FINISHED_STATES = frozenset({"completed", "failed", "stopped"})

#: Shared folders that serve every conversation and are never removed.
SHARED_CACHES = frozenset({"msa_cache", "batches"})

#: Job files offered as links; all are served by ``/api/jobs/{id}/files``.
JOB_LINK_FILES = ("results_1.csv", "reinvent.log", "config.toml")

_JOB_ID = re.compile(r"^[0-9a-f]{6,32}$")
_JOB_DIR = re.compile(r"^job_[0-9a-f]{6,32}$")
_JOB_IN_PATH = re.compile(r"(?:^|/)job_([0-9a-f]{6,32})(?=/|$)")
_TOKEN = re.compile(r"[^\s\"'`<>()\[\]{},;|*?]+")
_SHARED_MARKERS = ("/data/shared/", "/shared/")
_SKIP_KEYS = frozenset({"url", "smiles", "SMILES", "contacts", "rows"})
_MAX_LINKS = 6

GROUPS = (
    ("uploads", "Uploads"),
    ("predictions", "Boltz predictions"),
    ("structures", "Other structures used"),
    ("analyses", "PoseBusters and interaction profiles"),
    ("jobs", "REINVENT jobs"),
    ("ttps", "TTPs"),
)


@dataclass(frozen=True)
class Roots:
    """The two places data lives: the shared volume and the job directories."""

    shared: Path
    jobs: Path


@dataclass
class References:
    """What one conversation's events point at, each with when it first did."""

    paths: dict[str, str] = field(default_factory=dict)
    jobs: dict[str, str] = field(default_factory=dict)
    launched: dict[str, str] = field(default_factory=dict)
    ttps: dict[str, dict[str, Any]] = field(default_factory=dict)
    analyses: list[dict[str, Any]] = field(default_factory=list)

    def units(self) -> set[str]:
        """The deletable units these paths fall in (see ``unit_of``)."""
        return {unit for unit in map(unit_of, self.paths) if unit is not None}


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------


def shared_relative(token: str, shared_root: Path) -> str | None:
    """A path as the tools wrote it, relative to the shared volume, or None.

    Tools report container paths (``/shared/predictions/...``), host paths
    (``.../data/shared/...``) or paths already relative to the volume, so all
    three are re-rooted.  Any other absolute path is not on the volume.
    """
    text = token.replace("\\", "/").strip().rstrip(".:")
    if "/" not in text:
        return None
    root = shared_root.as_posix().rstrip("/")
    rel: str | None = None
    if root and text.startswith(root + "/"):
        rel = text[len(root) + 1 :]
    elif text.startswith("data/shared/"):
        rel = text[len("data/shared/") :]
    else:
        for marker in _SHARED_MARKERS:
            index = text.find(marker)
            if index != -1:
                rel = text[index + len(marker) :]
                break
        else:
            if not text.startswith("/") and not re.match(r"^[A-Za-z]:/", text):
                rel = text
    if rel is None:
        return None
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def unit_of(rel: str) -> str | None:
    """The unit a shared path is kept or removed as, or None if it is too broad.

    An upload is one file of an uploads folder; a prediction is one Boltz run
    directory.  A bare ``predictions/`` or ``uploads/`` names no unit, so an
    answer that mentions where outputs go does not pin every output.
    """
    parts = rel.split("/")
    if parts[0] == "uploads":
        if len(parts) == 2:
            return rel
        return "/".join(parts[:3]) if len(parts) >= 3 else None
    if parts[0] == "predictions":
        return "/".join(parts[:2]) if len(parts) >= 2 else None
    return rel


def collect_references(events: Iterable[dict[str, Any]], shared_root: Path) -> References:
    """Read what one conversation's events point at."""
    refs = References()
    calls: dict[Any, dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        ts = str(event.get("ts") or "")
        if kind in ("user_message", "assistant_message"):
            _scan_text(event.get("content"), ts, refs, shared_root)
        elif kind == "tool_call":
            arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
            calls[event.get("id")] = arguments
            _scan_value(arguments, ts, refs, shared_root)
            _ids_in_arguments(arguments, ts, refs)
        elif kind == "tool_result" and not event.get("isError"):
            name = str(event.get("name") or "")
            result = event.get("result") if isinstance(event.get("result"), str) else ""
            if name in LAUNCH_TOOLS:
                job_id = _launched_job(result)
                if job_id:
                    refs.launched.setdefault(job_id, ts)
                    refs.jobs.setdefault(job_id, ts)
            if name in ("create_ttp", "update_ttp"):
                _ttp_in_result(name, result, ts, refs)
            if name in INVENTORY_TOOLS:
                continue
            _scan_text(result, ts, refs, shared_root)
            _scan_value(event.get("panels"), ts, refs, shared_root)
            analysis = _analysis(name, event, ts)
            if analysis is not None:
                refs.analyses.append(analysis)
    return refs


def _scan_value(value: Any, ts: str, refs: References, shared_root: Path) -> None:
    if isinstance(value, str):
        _scan_text(value, ts, refs, shared_root)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in _SKIP_KEYS:
                _scan_value(item, ts, refs, shared_root)
    elif isinstance(value, list):
        for item in value:
            _scan_value(item, ts, refs, shared_root)


def _scan_text(text: Any, ts: str, refs: References, shared_root: Path) -> None:
    if not isinstance(text, str) or "/" not in text and "\\" not in text:
        return
    for token in _TOKEN.findall(text):
        clean = token.replace("\\", "/")
        if "/" not in clean:
            continue
        for match in _JOB_IN_PATH.finditer(clean):
            refs.jobs.setdefault(match.group(1), ts)
        rel = shared_relative(clean, shared_root)
        if rel is None:
            continue
        top = rel.split("/")[0]
        if top in ("uploads", "predictions") or rel.lower().endswith(STRUCTURE_SUFFIXES):
            refs.paths.setdefault(rel, ts)


def _ids_in_arguments(value: Any, ts: str, refs: References) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "job_id" and isinstance(item, str) and _JOB_ID.match(item):
                refs.jobs.setdefault(item, ts)
            elif key == "ttp_id" and isinstance(item, str) and item.strip():
                refs.ttps.setdefault(item.strip(), {"role": "used", "first": ts})
            else:
                _ids_in_arguments(item, ts, refs)
    elif isinstance(value, list):
        for item in value:
            _ids_in_arguments(item, ts, refs)


def _parse(result: str) -> Any:
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        return None


def _launched_job(result: str) -> str | None:
    payload = _parse(result)
    if isinstance(payload, dict) and not payload.get("error"):
        job_id = payload.get("job_id")
        if isinstance(job_id, str) and _JOB_ID.match(job_id):
            return job_id
    return None


def _ttp_in_result(name: str, result: str, ts: str, refs: References) -> None:
    payload = _parse(result)
    if not isinstance(payload, dict) or payload.get("error"):
        return
    ttp_id = payload.get("id")
    if not isinstance(ttp_id, str) or not ttp_id.strip():
        return
    entry = refs.ttps.setdefault(ttp_id.strip(), {"role": "used", "first": ts})
    if name == "create_ttp":
        entry["role"] = "created"
    for key in ("name", "version"):
        if payload.get(key) is not None:
            entry[key] = payload[key]


def _analysis(name: str, event: dict[str, Any], ts: str) -> dict[str, Any] | None:
    """A PoseBusters or interaction result, which lives only in the record."""
    panels = [panel for panel in event.get("panels") or [] if isinstance(panel, dict)]
    if name == "profile_interactions":
        count = sum(1 for panel in panels if panel.get("kind") == "interactions")
        label, detail = "Interaction profile", f"{count} complex{'es' if count != 1 else ''}"
    else:
        checked = [
            panel["metrics"]
            for panel in panels
            if panel.get("kind") == "structure"
            and isinstance(panel.get("metrics"), dict)
            and "posebustersTotal" in panel["metrics"]
        ]
        if not checked:
            return None
        passed = sum(1 for m in checked if m.get("posebustersPassed") == m.get("posebustersTotal"))
        label, detail = "PoseBusters checks", f"{passed} of {len(checked)} poses pass every check"
    result = event.get("result")
    return {
        "id": f"analysis:{event.get('seq', event.get('id'))}",
        "kind": "analysis",
        "name": label,
        "detail": f"{detail} ({name})",
        "bytes": len(result.encode("utf-8")) if isinstance(result, str) else 0,
        "modified": ts or None,
    }


def stored_references(store_root: Path, shared_root: Path, exclude: str) -> list[References]:
    """The references of every other stored conversation, read from disk."""
    found: list[References] = []
    if not store_root.is_dir():
        return found
    for directory in store_root.iterdir():
        if directory.name == exclude or not valid_session_id(directory.name):
            continue
        found.append(collect_references(_read_events(directory / "events.jsonl"), shared_root))
    return found


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


class RunIndex:
    """Which REINVENT jobs each conversation launched, for the conversation list.

    The list is polled every few seconds, so nothing is read again unless it
    changed: a conversation's events when its ``events.jsonl`` did, a job's
    state when its ``job_metadata.json`` did, a job's ``config.toml`` once.

    A run continued from the Jobs panel is started by a direct tool call,
    outside any conversation; a job no conversation launched belongs to the
    one that launched a job its configuration resumes from.
    """

    ACTIVE = frozenset({"running", "queued"})

    def __init__(self, store_root: Path) -> None:
        self._store_root = store_root
        self._refs: dict[str, tuple[tuple[int, int], list[str], list[str]]] = {}
        self._states: dict[str, tuple[tuple[int, int], str | None]] = {}
        self._parents: dict[str, tuple[tuple[int, int], frozenset[str]]] = {}

    def runs(self, session_ids: Iterable[str], roots: Roots) -> dict[str, dict[str, Any]]:
        """``{session id: {"running", "total", "jobs": [{"id", "state", "launched"}]}}``.

        ``total`` and ``running`` count the jobs launched there, ``jobs`` also
        lists the ones it only referenced.  Jobs that are gone are left out.
        """
        ids = [sid for sid in session_ids if valid_session_id(sid)]
        launched_by: dict[str, str] = {}
        referenced: dict[str, list[str]] = {}
        for sid in ids:
            launched, seen = self._references(sid, roots.shared)
            for job_id in launched:
                launched_by.setdefault(job_id, sid)
            referenced[sid] = seen
        self._inherit(launched_by, roots.jobs)

        out: dict[str, dict[str, Any]] = {}
        for sid in ids:
            own = [job_id for job_id, owner in launched_by.items() if owner == sid]
            jobs = [{"id": job_id, "state": self._state(job_id, roots.jobs), "launched": True} for job_id in own]
            jobs += [
                {"id": job_id, "state": self._state(job_id, roots.jobs), "launched": False}
                for job_id in referenced[sid]
                if job_id not in own
            ]
            jobs = [job for job in jobs if job["state"] is not None]
            mine = [job for job in jobs if job["launched"]]
            out[sid] = {
                "running": sum(1 for job in mine if job["state"] in self.ACTIVE),
                "total": len(mine),
                "jobs": jobs,
            }
        for stale in set(self._refs) - set(ids):
            del self._refs[stale]
        return out

    def _references(self, session_id: str, shared_root: Path) -> tuple[list[str], list[str]]:
        path = self._store_root / session_id / "events.jsonl"
        stamp = _stamp(path)
        if stamp is None:
            return [], []
        cached = self._refs.get(session_id)
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        refs = collect_references(_read_events(path), shared_root)
        launched = list(refs.launched)
        seen = [job_id for job_id in refs.jobs if job_id not in refs.launched]
        self._refs[session_id] = (stamp, launched, seen)
        return launched, seen

    def _state(self, job_id: str, jobs_root: Path) -> str | None:
        """The job's state; None when it has no readable metadata (gone, or not a job)."""
        if not _JOB_ID.match(job_id):
            return None
        path = jobs_root / f"job_{job_id}" / "job_metadata.json"
        stamp = _stamp(path)
        if stamp is None:
            return None
        cached = self._states.get(job_id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        meta = _job_metadata(path.parent)
        state = meta.get("state") if isinstance(meta.get("state"), str) else None
        self._states[job_id] = (stamp, state)
        return state

    def _inherit(self, launched_by: dict[str, str], jobs_root: Path) -> None:
        parents = self._job_parents(jobs_root)
        changed = True
        while changed:
            changed = False
            for job_id, names in parents.items():
                if job_id in launched_by:
                    continue
                owner = next((launched_by[parent] for parent in sorted(names) if parent in launched_by), None)
                if owner is not None:
                    launched_by[job_id] = owner
                    changed = True

    def _job_parents(self, jobs_root: Path) -> dict[str, frozenset[str]]:
        """The other jobs each job's config.toml names, e.g. the checkpoint it resumes."""
        found: dict[str, frozenset[str]] = {}
        try:
            directories = [d for d in jobs_root.iterdir() if _JOB_DIR.match(d.name)]
        except OSError:
            return found
        for directory in directories:
            job_id = directory.name[4:]
            path = directory / "config.toml"
            stamp = _stamp(path)
            if stamp is None:
                continue
            cached = self._parents.get(job_id)
            if cached is None or cached[0] != stamp:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                names = frozenset(m.group(1) for m in _JOB_IN_PATH.finditer(text.replace("\\", "/"))) - {job_id}
                cached = (stamp, names)
                self._parents[job_id] = cached
            if cached[1]:
                found[job_id] = cached[1]
        return found


def _read_events(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError:
        return


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------


def inventory(
    session_id: str,
    events: Iterable[dict[str, Any]],
    roots: Roots,
    others: list[References],
    conversation_bytes: int = 0,
) -> dict[str, Any]:
    """What one conversation produced or used, grouped, with sizes and dates."""
    items = _items(session_id, list(events), roots, others)
    groups = []
    for kind, label in GROUPS:
        members = [_public(item) for item in items if item["group"] == kind]
        groups.append(
            {
                "kind": kind,
                "label": label,
                "items": members,
                "bytes": sum(item["bytes"] for item in members if item["onDisk"]),
            }
        )
    on_disk = [item for item in items if item["onDisk"]]
    removable = [item for item in items if item["deletable"]]
    return {
        "session": session_id,
        "groups": groups,
        "totals": {
            "items": len(items),
            "bytes": sum(item["bytes"] for item in on_disk),
            "deletableItems": len(removable),
            "deletableBytes": sum(item["bytes"] for item in removable),
            "conversationBytes": conversation_bytes,
        },
    }


def _public(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _item(group: str, item_id: str, name: str, **extra: Any) -> dict[str, Any]:
    item = {
        "id": item_id,
        "group": group,
        "name": name,
        "detail": None,
        "path": None,
        "bytes": 0,
        "modified": None,
        "url": None,
        "links": [],
        "onDisk": False,
        "deletable": False,
        "reason": None,
    }
    item.update(extra)
    return item


def _items(session_id: str, events: list[dict[str, Any]], roots: Roots, others: list[References]) -> list[dict[str, Any]]:
    refs = collect_references(events, roots.shared)
    other_units = [other.units() for other in others]
    other_jobs = [set(other.jobs) for other in others]
    items: list[dict[str, Any]] = []
    items.extend(_upload_items(session_id, refs, roots, other_units))
    items.extend(_prediction_items(session_id, refs, roots, other_units))
    items.extend(_structure_items(refs, roots))
    for analysis in refs.analyses:
        items.append(
            _item(
                "analyses",
                analysis["id"],
                analysis["name"],
                detail=analysis["detail"],
                bytes=analysis["bytes"],
                modified=analysis["modified"],
                reason="Recorded in this conversation; removed with it",
                withConversation=True,
            )
        )
    items.extend(_job_items(refs, roots, other_jobs))
    for ttp_id, entry in sorted(refs.ttps.items()):
        version = f" v{entry['version']}" if entry.get("version") is not None else ""
        items.append(
            _item(
                "ttps",
                f"ttp:{ttp_id}",
                ttp_id,
                detail=f"{entry['role'].capitalize()} here{version}" + (f" · {entry['name']}" if entry.get("name") else ""),
                modified=entry.get("first") or None,
                reason="TTPs are shared and never deleted here",
            )
        )
    return items


def _shared_count(unit: str, other_units: list[set[str]]) -> int:
    def touches(other: str) -> bool:
        return other == unit or other.startswith(unit + "/") or unit.startswith(other + "/")

    return sum(1 for units in other_units if any(touches(other) for other in units))


def _also_used(count: int) -> str:
    return f"Also used by {count} other conversation{'s' if count != 1 else ''}"


def _artifact_url(rel: str) -> str:
    return "/api/artifacts/" + "/".join(quote(part) for part in rel.split("/"))


def _upload_items(session_id: str, refs: References, roots: Roots, other_units: list[set[str]]) -> list[dict[str, Any]]:
    own = f"uploads/{session_id}"
    units: dict[str, str | None] = {}
    folder = roots.shared / "uploads" / session_id
    if folder.is_dir() and not folder.is_symlink():
        for entry in sorted(folder.iterdir()):
            units[f"{own}/{entry.name}"] = None
    for rel, ts in refs.paths.items():
        unit = unit_of(rel)
        if unit and unit.startswith("uploads/") and unit.count("/") == 2:
            units.setdefault(unit, ts)

    items = []
    for unit, first in sorted(units.items()):
        path = roots.shared / unit
        stat = _stat(path)
        item = _item("uploads", f"upload:{unit}", unit.rsplit("/", 1)[-1], path=unit, _target=path)
        if stat is None:
            item["reason"] = "Already gone"
            items.append(item)
            continue
        item.update(onDisk=True, bytes=stat[0], modified=stat[1], url=_artifact_url(unit) if path.is_file() else None)
        item["detail"] = "Uploaded here" if unit.startswith(own + "/") else f"Used here, uploaded to {unit.split('/')[1]}"
        if not unit.startswith(own + "/"):
            item["reason"] = "Uploaded outside this conversation's folder"
        elif count := _shared_count(unit, other_units):
            item["reason"] = _also_used(count)
        else:
            item["reason"] = unsafe_target(path, roots.shared, _allowed_shared(session_id))
        item["deletable"] = item["reason"] is None
        items.append(item)
    return items


def _prediction_items(session_id: str, refs: References, roots: Roots, other_units: list[set[str]]) -> list[dict[str, Any]]:
    runs: dict[str, list[str]] = {}
    first: dict[str, str] = {}
    for rel, ts in refs.paths.items():
        unit = unit_of(rel)
        if unit and unit.startswith("predictions/"):
            runs.setdefault(unit, [])
            first.setdefault(unit, ts)
            if rel.lower().endswith(STRUCTURE_SUFFIXES):
                runs[unit].append(rel)

    items = []
    for unit, structures in sorted(runs.items()):
        path = roots.shared / unit
        stat = _stat(path)
        item = _item("predictions", f"prediction:{unit}", unit.split("/", 1)[1], path=unit, _target=path)
        if stat is None:
            item["reason"] = "Already gone"
            items.append(item)
            continue
        existing = [rel for rel in structures if (roots.shared / rel).is_file()]
        files = _file_count(path)
        item.update(
            onDisk=True,
            bytes=stat[0],
            modified=stat[1],
            detail=f"{files} file{'s' if files != 1 else ''}, {len(existing)} structure{'s' if len(existing) != 1 else ''} shown here",
            links=[{"label": rel.rsplit("/", 1)[-1], "url": _artifact_url(rel)} for rel in existing[:_MAX_LINKS]],
        )
        if count := _shared_count(unit, other_units):
            item["reason"] = _also_used(count)
        else:
            item["reason"] = unsafe_target(path, roots.shared, _allowed_shared(session_id))
        item["deletable"] = item["reason"] is None
        items.append(item)
    return items


def _structure_items(refs: References, roots: Roots) -> list[dict[str, Any]]:
    items = []
    for rel, ts in sorted(refs.paths.items()):
        top = rel.split("/")[0]
        if top in ("uploads", "predictions") or not rel.lower().endswith(STRUCTURE_SUFFIXES):
            continue
        path = roots.shared / rel
        stat = _stat(path) if path.is_file() else None
        if stat is None:
            continue
        reason = "Shared cache, never deleted here" if top in SHARED_CACHES else "Shared input, not an output of this conversation"
        items.append(
            _item(
                "structures",
                f"structure:{rel}",
                rel.rsplit("/", 1)[-1],
                path=rel,
                bytes=stat[0],
                modified=stat[1],
                url=_artifact_url(rel),
                onDisk=True,
                detail=rel,
                reason=reason,
            )
        )
    return items


def _job_items(refs: References, roots: Roots, other_jobs: list[set[str]]) -> list[dict[str, Any]]:
    configs = _job_configs(roots.jobs)
    items = []
    for job_id, first in sorted(refs.jobs.items(), key=lambda pair: pair[1]):
        if not _JOB_ID.match(job_id):
            continue
        path = roots.jobs / f"job_{job_id}"
        launched = job_id in refs.launched
        item = _item("jobs", f"job:{job_id}", job_id, path=f"job_{job_id}", modified=first or None, _target=path)
        stat = _stat(path)
        if stat is None:
            item["detail"] = "Launched here" if launched else "Referenced here"
            item["reason"] = "Already gone"
            items.append(item)
            continue
        meta = _job_metadata(path)
        state = meta.get("state") if isinstance(meta.get("state"), str) else None
        parts = ["Launched here" if launched else "Referenced here", state or "state unknown"]
        if isinstance(meta.get("ttp_id"), str) and meta["ttp_id"]:
            parts.append(f"TTP {meta['ttp_id']}")
        item.update(
            onDisk=True,
            bytes=stat[0],
            modified=meta.get("finished_at") or meta.get("created_at") or stat[1],
            detail=" · ".join(parts),
            state=state,
            links=[
                {"label": name, "url": f"/api/jobs/{job_id}/files/{name}"}
                for name in JOB_LINK_FILES
                if (path / name).is_file()
            ],
        )
        item["reason"] = _job_reason(job_id, state, launched, other_jobs, configs, path, roots.jobs)
        item["deletable"] = item["reason"] is None
        items.append(item)
    return items


def _job_reason(
    job_id: str,
    state: str | None,
    launched: bool,
    other_jobs: list[set[str]],
    configs: dict[str, str],
    path: Path,
    jobs_root: Path,
) -> str | None:
    if state in ("running", "queued"):
        return f"Still {state}: stop the run first"
    if state not in FINISHED_STATES:
        return "State unknown (no readable job_metadata.json)"
    if not launched:
        return "Launched outside this conversation"
    if count := sum(1 for jobs in other_jobs if job_id in jobs):
        return _also_used(count)
    users = sorted(other for other, text in configs.items() if other != job_id and f"job_{job_id}" in text)
    if users:
        return f"Used by job {users[0]}'s configuration (resume or prior)"
    return unsafe_target(path, jobs_root, _allowed_job)


def _job_metadata(path: Path) -> dict[str, Any]:
    try:
        data = json.loads((path / "job_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _job_configs(jobs_root: Path) -> dict[str, str]:
    """Every job's config, to see which jobs resume from or fine-tune on another."""
    configs: dict[str, str] = {}
    if not jobs_root.is_dir():
        return configs
    for directory in jobs_root.iterdir():
        if not _JOB_DIR.match(directory.name):
            continue
        try:
            configs[directory.name[4:]] = (directory / "config.toml").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return configs


# --------------------------------------------------------------------------
# sizes and safety
# --------------------------------------------------------------------------


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _stat(path: Path) -> tuple[int, str] | None:
    """Size on disk (links not followed) and last change, or None if absent."""
    try:
        info = path.lstat()
    except OSError:
        return None
    if path.is_symlink() or not path.is_dir():
        return info.st_size, _iso(info.st_mtime)
    total, latest = 0, info.st_mtime
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in files + dirs:
            try:
                child = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            latest = max(latest, child.st_mtime)
            if name in files:
                total += child.st_size
    return total, _iso(latest)


def _file_count(path: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(path, followlinks=False))


def directory_bytes(path: Path) -> int:
    stat = _stat(path)
    return stat[0] if stat else 0


def _allowed_shared(session_id: str) -> Callable[[tuple[str, ...]], bool]:
    def allowed(parts: tuple[str, ...]) -> bool:
        if parts and parts[0] in SHARED_CACHES:
            return False
        if len(parts) == 3 and parts[0] == "uploads" and parts[1] == session_id:
            return True
        return len(parts) == 2 and parts[0] == "predictions"

    return allowed


def _allowed_job(parts: tuple[str, ...]) -> bool:
    return len(parts) == 1 and _JOB_DIR.match(parts[0]) is not None


def _is_link(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(isjunction and isjunction(path))


def unsafe_target(target: Path, root: Path, allowed: Callable[[tuple[str, ...]], bool]) -> str | None:
    """Why *target* may not be deleted under *root*, or None if it may.

    The target must lie lexically below the root, no component between the
    two may be a link, it must still resolve below the resolved root, and its
    place there must be one *allowed* accepts.
    """
    try:
        real_root = root.resolve(strict=True)
    except OSError:
        return "The data folder is not reachable"
    if real_root.parent == real_root:
        return "Refusing to delete under a filesystem root"
    try:
        rel = target.relative_to(root)
    except ValueError:
        return "Outside the folders this cleanup may touch"
    if not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        return "Outside the folders this cleanup may touch"
    current = root
    for part in rel.parts:
        current = current / part
        if _is_link(current):
            return "Is or goes through a symbolic link"
    try:
        resolved = target.resolve(strict=True)
    except OSError:
        return "Already gone"
    if resolved == real_root or not resolved.is_relative_to(real_root):
        return "Resolves outside the folders this cleanup may touch"
    if not allowed(resolved.relative_to(real_root).parts):
        return "Not a folder this cleanup may touch"
    return None


# --------------------------------------------------------------------------
# deletion
# --------------------------------------------------------------------------


def purge(
    session_id: str,
    events: Iterable[dict[str, Any]],
    roots: Roots,
    others: list[References],
    selected: Iterable[str],
) -> dict[str, Any]:
    """Delete the selected data of a conversation that has just been deleted.

    Only items the inventory itself marks deletable are removed, so an id the
    browser made up removes nothing.  Every target is checked again just
    before removal.  Returns what was removed, what was kept and why, and the
    bytes freed.
    """
    wanted = {item_id for item_id in selected if isinstance(item_id, str)}
    items = _items(session_id, list(events), roots, others)
    known = {item["id"] for item in items}
    removed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    freed = 0

    for item in items:
        if item.get("withConversation"):
            continue
        brief = {"id": item["id"], "group": item["group"], "name": item["name"], "bytes": item["bytes"]}
        if item["id"] not in wanted:
            if item["onDisk"] or item["group"] == "ttps":
                kept.append({**brief, "reason": item["reason"] if not item["deletable"] else "Not selected"})
            continue
        if not item["deletable"]:
            kept.append({**brief, "reason": item["reason"] or "Not deletable"})
            continue
        target: Path = item["_target"]
        root = roots.jobs if item["group"] == "jobs" else roots.shared
        allowed = _allowed_job if item["group"] == "jobs" else _allowed_shared(session_id)
        reason = unsafe_target(target, root, allowed)
        if reason is not None:
            kept.append({**brief, "reason": reason})
            continue
        size = directory_bytes(target)
        try:
            _remove(target)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", target, exc)
            kept.append({**brief, "reason": f"Could not delete: {exc.strerror or exc}"})
            continue
        freed += size
        removed.append({**brief, "bytes": size})

    for item_id in sorted(wanted - known):
        kept.append({"id": item_id, "group": None, "name": item_id, "bytes": 0, "reason": "Not associated with this conversation"})

    _drop_empty_upload_folder(session_id, roots.shared)
    return {"removed": removed, "kept": kept, "freedBytes": freed}


def _remove(target: Path) -> None:
    if target.is_dir() and not _is_link(target):
        shutil.rmtree(target)
    else:
        target.unlink()


def _drop_empty_upload_folder(session_id: str, shared_root: Path) -> None:
    folder = shared_root / "uploads" / session_id
    if not valid_session_id(session_id) or _is_link(folder) or not folder.is_dir():
        return
    try:
        folder.rmdir()
    except OSError:
        pass
