# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FIFO scheduler over persistent Boltz-2 worker processes.

One request is split into per-molecule jobs that run on whichever workers are
idle, so the molecules of one request run in parallel. Jobs are dispatched in
strict FIFO order across requests. A molecule that runs out of GPU memory
while another molecule shared its GPU is retried once with that GPU to itself.

The scheduler runs on the asyncio event loop; the blocking pipe round-trips to
a worker run in a thread. Workers are created through a backend factory so the
scheduler can be tested without GPUs.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger("boltz2_pool")

WORKER_LOAD_TIMEOUT_SECONDS = 1200
MAX_WORKER_RESTARTS = 2


class PoolUnavailable(Exception):
    """No worker is loading or ready; the caller should use the subprocess path."""


class JobCancelled(Exception):
    """The request was cancelled before this molecule started."""


class WorkerCrashed(Exception):
    """The worker process died or exceeded the per-molecule timeout."""


class WorkerBackend(Protocol):
    def start(self) -> dict[str, Any]:
        """Start the worker and block until its models are loaded."""

    def call(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Run one molecule and block until the result arrives."""

    def stop(self) -> None:
        ...


class ProcessBackend:
    """A ``boltz_worker.worker_main`` process started with the spawn method."""

    def __init__(self, worker_id: str, device: int, cache_dir: str) -> None:
        self.worker_id = worker_id
        self.device = device
        self.cache_dir = cache_dir
        self._ctx = multiprocessing.get_context("spawn")
        self._proc: Any = None
        self._conn: Any = None

    def start(self) -> dict[str, Any]:
        from boltz_worker import worker_main

        parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self.worker_id, self.device, child_conn, self.cache_dir),
            name=f"boltz-{self.worker_id}",
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        msg = self._recv(WORKER_LOAD_TIMEOUT_SECONDS, "model load")
        if msg.get("type") != "ready":
            self.stop()
            raise RuntimeError(msg.get("error", "worker failed to load"))
        return {k: v for k, v in msg.items() if k != "type"}

    def call(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            self._conn.send(payload)
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise WorkerCrashed(f"worker process is gone ({exc})") from exc
        return self._recv(timeout, "prediction")

    def _recv(self, timeout: float, what: str) -> dict[str, Any]:
        try:
            ready = self._conn.poll(timeout)
            if not ready:
                self.stop()
                raise WorkerCrashed(f"{what} timed out after {int(timeout)}s")
            return self._conn.recv()
        except (EOFError, OSError) as exc:
            code = self._proc.exitcode if self._proc is not None else None
            self.stop()
            raise WorkerCrashed(f"worker process died during {what} (exit code {code})") from exc

    def stop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:  # noqa: BLE001
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        if self._conn is not None:
            self._conn.close()
            self._conn = None


@dataclass
class MoleculeJob:
    request_id: str
    index: int
    payload: dict[str, Any]
    future: asyncio.Future
    exclusive: bool = False
    shared: bool = False
    retried: bool = False
    cancelled: bool = False


@dataclass
class WorkerSlot:
    worker_id: str
    device: int
    state: str = "loading"  # loading | ready | busy | failed
    backend: Any = None
    job: Optional[MoleculeJob] = None
    error: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)
    restarts: int = 0
    completed: int = 0


class WorkerPool:
    def __init__(
        self,
        devices: list[int],
        workers_per_gpu: int,
        backend_factory: Callable[[str, int], WorkerBackend],
        job_timeout: float,
    ) -> None:
        self.workers = [
            WorkerSlot(worker_id=f"gpu{d}-w{k}", device=d)
            for d in devices
            for k in range(workers_per_gpu)
        ]
        self._factory = backend_factory
        self._job_timeout = job_timeout
        self._queue: deque[MoleculeJob] = deque()
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        for slot in self.workers:
            self._spawn(self._boot(slot))

    async def shutdown(self) -> None:
        for job in self._queue:
            if not job.future.done():
                job.future.set_exception(PoolUnavailable("service shutting down"))
        self._queue.clear()
        for slot in self.workers:
            if slot.backend is not None:
                await asyncio.to_thread(slot.backend.stop)
            slot.state = "failed"
            slot.error = "stopped"

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _boot(self, slot: WorkerSlot) -> None:
        slot.state = "loading"
        slot.error = None
        try:
            slot.backend = self._factory(slot.worker_id, slot.device)
            slot.info = await asyncio.to_thread(slot.backend.start)
            slot.state = "ready"
            logger.info("Worker %s ready: %s", slot.worker_id, slot.info)
        except Exception as exc:  # noqa: BLE001
            slot.state = "failed"
            slot.error = str(exc)[:500]
            logger.error("Worker %s failed to start: %s", slot.worker_id, slot.error)
            if not self.usable:
                self._fail_queue(f"all Boltz workers failed to start ({slot.error})")
        self._dispatch()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def usable(self) -> bool:
        return any(w.state != "failed" for w in self.workers)

    @property
    def status(self) -> str:
        if any(w.state in ("ready", "busy") for w in self.workers):
            return "ok"
        if any(w.state == "loading" for w in self.workers):
            return "loading"
        return "failed"

    def queue_length(self) -> int:
        return sum(1 for j in self._queue if not j.cancelled)

    def request_progress(self) -> dict[str, dict[str, int]]:
        """Queued and running molecules per request id."""
        progress: dict[str, dict[str, int]] = {}
        for job in self._queue:
            if not job.cancelled:
                progress.setdefault(job.request_id, {"queued": 0, "running": 0})["queued"] += 1
        for w in self.workers:
            if w.job is not None:
                progress.setdefault(w.job.request_id, {"queued": 0, "running": 0})["running"] += 1
        return progress

    def health(self) -> dict[str, Any]:
        workers = []
        for w in self.workers:
            entry: dict[str, Any] = {"id": w.worker_id, "device": w.device, "state": w.state}
            if w.error:
                entry["error"] = w.error
            if w.info:
                entry["memory"] = w.info
            if w.job is not None:
                entry["request_id"] = w.job.request_id
            entry["completed"] = w.completed
            workers.append(entry)
        status = self.status
        return {
            # Requests are served by the boltz-predict subprocess fallback when
            # every worker failed, so the service itself stays healthy.
            "status": "ok" if status == "failed" else status,
            "mode": "subprocess" if status == "failed" else "pool",
            "workers": workers,
            "queue_length": self.queue_length(),
        }

    # ------------------------------------------------------------------
    # Submission / cancellation
    # ------------------------------------------------------------------
    def submit(self, request_id: str, payloads: list[dict[str, Any]]) -> list[asyncio.Future]:
        if not self.usable:
            raise PoolUnavailable("no Boltz worker is loading or ready")
        loop = asyncio.get_running_loop()
        jobs = [
            MoleculeJob(request_id=request_id, index=i, payload=p, future=loop.create_future())
            for i, p in enumerate(payloads)
        ]
        self._queue.extend(jobs)
        self._dispatch()
        return [j.future for j in jobs]

    def cancel(self, request_id: str) -> int:
        """Drop queued molecules of *request_id*; running ones finish."""
        dropped = 0
        for job in list(self._queue):
            if job.request_id == request_id:
                self._queue.remove(job)
                job.cancelled = True
                if not job.future.done():
                    job.future.set_exception(JobCancelled(request_id))
                dropped += 1
        for w in self.workers:
            if w.job is not None and w.job.request_id == request_id:
                w.job.cancelled = True
        return dropped

    def _fail_queue(self, reason: str) -> None:
        while self._queue:
            job = self._queue.popleft()
            if not job.future.done():
                job.future.set_exception(PoolUnavailable(reason))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def _busy_on(self, device: int) -> list[WorkerSlot]:
        return [w for w in self.workers if w.device == device and w.state == "busy"]

    def _pick_worker(self, job: MoleculeJob) -> Optional[WorkerSlot]:
        idle = [w for w in self.workers if w.state == "ready"]
        if job.exclusive:
            idle = [w for w in idle if not self._busy_on(w.device)]
        else:
            idle = [
                w for w in idle
                if not any(b.job is not None and b.job.exclusive for b in self._busy_on(w.device))
            ]
        if not idle:
            return None
        # Spread molecules over GPUs before doubling up on one.
        return min(idle, key=lambda w: (len(self._busy_on(w.device)), w.device, w.worker_id))

    def _dispatch(self) -> None:
        while self._queue:
            job = self._queue[0]
            slot = self._pick_worker(job)
            if slot is None:
                return
            self._queue.popleft()
            busy = self._busy_on(slot.device)
            for other in busy:
                if other.job is not None:
                    other.job.shared = True
            job.shared = bool(busy)
            slot.state = "busy"
            slot.job = job
            self._spawn(self._execute(slot, job))

    async def _execute(self, slot: WorkerSlot, job: MoleculeJob) -> None:
        crashed = False
        try:
            result = await asyncio.to_thread(slot.backend.call, job.payload, self._job_timeout)
        except WorkerCrashed as exc:
            crashed = True
            result = {"status": "error", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            crashed = True
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        slot.job = None
        slot.completed += 1
        result = {**result, "device": slot.device, "worker": slot.worker_id,
                  "shared_gpu": job.shared, "retried": job.retried}

        if crashed:
            logger.error("Worker %s crashed: %s", slot.worker_id, result["error"])
            if slot.restarts < MAX_WORKER_RESTARTS:
                slot.restarts += 1
                slot.state = "loading"
                self._spawn(self._boot(slot))
            else:
                slot.state = "failed"
                slot.error = result["error"]
                if not self.usable:
                    self._fail_queue(f"all Boltz workers failed ({slot.error})")
        else:
            slot.state = "ready"

        if (
            result.get("status") == "oom"
            and job.shared
            and not job.retried
            and not job.cancelled
        ):
            logger.warning(
                "Molecule %d of request %s ran out of memory on GPU %d next to "
                "another molecule; retrying with the GPU to itself",
                job.index, job.request_id, slot.device,
            )
            job.retried = True
            job.exclusive = True
            job.shared = False
            self._queue.appendleft(job)
            if not self.usable:
                self._fail_queue("all Boltz workers failed")
        elif not job.future.done():
            job.future.set_result(result)
        self._dispatch()
