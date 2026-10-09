"""Boltz gateway — per-molecule result cache, batch coalescing, and a queue view.

Sits between ``ExternalRESTScorer`` and the Boltz replicas, speaking the same
``{smiles, params}`` -> ``{scores, metadata, artifacts}`` contract so nothing
downstream has to change. Placement stays with ``dd-agentic-boltz-proxy``
(HAProxy, least-busy with one prediction per replica); this owns the things a
proxy cannot do because they need to understand the payload:

* **Per-molecule result cache.** Keyed on
  ``sha256(canonical_smiles || protein_sequence || msa_digest || predict_params)``
  under ``/shared``. Caching per *molecule* rather than per batch is what makes
  it useful: a batch of 16 where 10 were seen before only costs 6 predictions.
  SMILES are canonicalised first because the RL config sets
  ``randomize_smiles: true``, so the same molecule arrives spelled differently
  on every step and a literal-string cache would almost never hit.
  Exploration-tier requests read and write; validation-tier requests bypass the
  read and always re-predict, so a reported result is never a stale sample.

* **Batch coalescing.** ``boltz predict <dir>`` folds a whole directory of YAMLs
  into one process, so the per-invocation cost (model load, warm-up) is paid
  once however many molecules are in it. Two measured points bound that fixed
  cost at **at least 124 s of the 386 s batch**, which is why merging concurrent
  requests that share a target and prediction params is the strongest lever
  available -- and why dropping molecules from a batch saves much less than it
  looks. Requests waiting on the same ``(protein_sequence, params)`` are merged
  into one upstream call.

* **In-flight de-duplication.** If a molecule is already being predicted, a
  second request for it waits on that same prediction instead of starting
  another. With several supervisors exploring neighbouring weight vectors over
  one target, this is where the cross-supervisor saving actually lands.

* **``GET /stats``.** Cache hit rate, coalescing ratio, queue depth, in-flight
  work, latency percentiles, healthy replicas. The hit rate in particular has
  been an unmeasured assumption in every sizing calculation so far; this is
  what finally reports it.

* **``202 Accepted`` + ``GET /result/{task_id}``.** So a caller can submit and
  poll rather than holding a connection open for an hour. A deep queue then
  stops being indistinguishable from a failure.

Env:
    BOLTZ_UPSTREAM_URL        where to send predictions (the HAProxy proxy)
    BOLTZ_PROXY_STATS_URL     HAProxy CSV stats, for healthy-replica count
    GATEWAY_CACHE_DIR         result cache root on the shared volume
    GATEWAY_CACHE_ENABLED     "0" disables the cache entirely
    COALESCE_WINDOW_SECONDS   how long to hold a batch open for more molecules
    COALESCE_MAX_MOLECULES    most molecules in one upstream call
    GATEWAY_MAX_INFLIGHT      concurrent upstream calls (0 = healthy replicas)
    UPSTREAM_TIMEOUT_SECONDS  must exceed the service's own 1800 s ceiling
    GATEWAY_RESULT_TTL_SECONDS how long an async result stays fetchable
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog("rdApp.error")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("boltz_gateway")

SHARED_DIR = Path("/shared")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


UPSTREAM_URL = os.getenv(
    "BOLTZ_UPSTREAM_URL", "http://dd-agentic-boltz-proxy:8091/score"
)
PROXY_STATS_URL = os.getenv(
    "BOLTZ_PROXY_STATS_URL", "http://dd-agentic-boltz-proxy:8404/stats;csv"
)
CACHE_DIR = Path(os.getenv("GATEWAY_CACHE_DIR") or SHARED_DIR / "boltz_results_cache")
CACHE_ENABLED = _env_flag("GATEWAY_CACHE_ENABLED", True)
COALESCE_WINDOW_SECONDS = _env_float("COALESCE_WINDOW_SECONDS", 2.0)
COALESCE_MAX_MOLECULES = _env_int("COALESCE_MAX_MOLECULES", 16)
# 0 means "as many as there are healthy replicas", which keeps every replica
# busy while everything else accumulates into bigger batches.
MAX_INFLIGHT = _env_int("GATEWAY_MAX_INFLIGHT", 0)
UPSTREAM_TIMEOUT_SECONDS = _env_float("UPSTREAM_TIMEOUT_SECONDS", 2100.0)
RESULT_TTL_SECONDS = _env_float("GATEWAY_RESULT_TTL_SECONDS", 3600.0)
REPLICA_REFRESH_SECONDS = 10.0

# Policy, not prediction: these decide cache behaviour and must stay out of the
# key and out of what is forwarded upstream.
POLICY_KEYS = frozenset({"fidelity_tier", "cache_policy"})
# Per-call plumbing rather than a property of the prediction.
PER_CALL_KEYS = frozenset({"smiles", "structure_paths", "protein_pdb_path"})

VALIDATION_TIER = "validation"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ScoreParams(BaseModel):
    """Boltz's own params plus the two cache-policy knobs.

    ``extra="allow"`` because everything here is forwarded to Boltz, which owns
    the real schema; the gateway only needs the fields it keys or routes on.
    """

    model_config = ConfigDict(extra="allow")

    protein_sequence: str = ""
    msa_path: Optional[str] = None
    # "exploration" reads and writes the cache; "validation" always re-predicts.
    fidelity_tier: str = "exploration"
    # "read_write" (default), "write_only" to force a fresh prediction that
    # still populates the cache, or "none" to bypass it in both directions.
    cache_policy: str = "read_write"


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams = Field(default_factory=ScoreParams)


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------

_canonical_cache: dict[str, Optional[str]] = {}


def canonical_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES, or None when RDKit cannot read it.

    Memoised because one RL step sends the same strings repeatedly and
    canonicalisation is pure.
    """
    if smiles in _canonical_cache:
        return _canonical_cache[smiles]
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol) if mol is not None else None
    _canonical_cache[smiles] = canonical
    return canonical


_msa_digest_cache: dict[tuple[str, int, int], str] = {}


def msa_digest(params: ScoreParams) -> str:
    """Digest of the alignment a prediction will actually use.

    Hashes the file's contents, not its path: two paths holding the same
    alignment should share cache entries, and editing an ``.a3m`` in place must
    invalidate them. When the MSA comes from the server instead, it is a pure
    function of the sequence -- already in the key -- so a marker is enough.
    """
    path = params.msa_path
    if not path:
        return "msa:derived-from-sequence"

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = SHARED_DIR / path
    try:
        stat = resolved.stat()
    except OSError:
        # Unreadable from here: fall back to the path so entries at least do
        # not collide across different alignments.
        return f"msa:path:{path}"

    fingerprint = (str(resolved), stat.st_size, int(stat.st_mtime))
    cached = _msa_digest_cache.get(fingerprint)
    if cached:
        return cached

    digest = hashlib.sha256()
    try:
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return f"msa:path:{path}"

    value = f"msa:sha256:{digest.hexdigest()}"
    _msa_digest_cache[fingerprint] = value
    return value


def prediction_params(params: ScoreParams) -> dict[str, Any]:
    """What to forward upstream: the params that decide the prediction."""
    data = params.model_dump()
    return {
        key: value
        for key, value in sorted(data.items())
        if key not in POLICY_KEYS and key not in PER_CALL_KEYS
    }


def params_digest(params: ScoreParams) -> str:
    """Identity of the prediction settings, independent of where the MSA lives.

    ``msa_path`` is excluded deliberately: the alignment is already represented
    by ``msa_digest``, which hashes its contents. Keeping the path here as well
    would mean two names for one alignment never shared a cache entry, which is
    exactly what the content hash exists to avoid.
    """
    stable = json.dumps(
        {
            key: value
            for key, value in prediction_params(params).items()
            if key != "msa_path"
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(stable.encode()).hexdigest()[:32]


def coalesce_key(params: ScoreParams) -> str:
    """Requests sharing this key can be folded into one ``boltz predict`` run."""
    sequence = (params.protein_sequence or "").strip().upper()
    return hashlib.sha256(
        f"{sequence}|{msa_digest(params)}|{params_digest(params)}".encode()
    ).hexdigest()[:32]


def molecule_key(canonical: str, params: ScoreParams) -> str:
    return hashlib.sha256(
        f"{canonical}|{(params.protein_sequence or '').strip().upper()}|"
        f"{msa_digest(params)}|{params_digest(params)}".encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

@dataclass
class MoleculeResult:
    """One molecule's prediction, cacheable and reassemblable per request."""

    score: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    structure: Optional[str] = None

    @property
    def cacheable(self) -> bool:
        """Whether this is a determinate answer rather than a failure.

        A score is obviously keepable. A ``None`` with a ``skip_reason`` is the
        wrapper's deterministic validation verdict (too many heavy atoms, say),
        which will not change, so it is keepable too. A ``None`` with no reason
        is a failed prediction and must never be cached, or one bad run would
        poison the molecule forever.
        """
        if self.score is not None:
            return True
        return bool(self.metadata.get("skip_reason"))


def _cache_path(digest: str) -> Path:
    # Sharded so one directory does not grow to a million entries.
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def cache_read(digest: str) -> Optional[MoleculeResult]:
    path = _cache_path(digest)
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    structure = payload.get("structure")
    if structure and not Path(structure).is_file():
        # The structure was cleaned up: downstream scorers re-open these paths,
        # so a hit that points at a missing file is worse than a miss.
        logger.info("cache entry %s dropped: structure %s is gone", digest, structure)
        return None
    return MoleculeResult(
        score=payload.get("score"),
        metadata=payload.get("metadata") or {},
        structure=structure,
    )


def cache_write(digest: str, canonical: str, result: MoleculeResult) -> None:
    """Write atomically, so a reader never sees a half-written entry.

    No lock: two workers racing on the same molecule write identical content,
    and ``replace`` is atomic. That differs from the MSA cache, where the point
    of the lock is that the expensive server query happens only once.
    """
    if not result.cacheable:
        return
    path = _cache_path(digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        temporary.write_text(json.dumps({
            "score": result.score,
            "metadata": result.metadata,
            "structure": result.structure,
            "canonical_smiles": canonical,
            "written": time.time(),
        }))
        temporary.replace(path)
    except OSError:
        logger.warning("could not write cache entry %s", digest, exc_info=True)


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------

def _post_upstream_blocking(payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        UPSTREAM_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode())


def _split_upstream(
    data: dict[str, Any],
    count: int,
) -> list[MoleculeResult]:
    """Turn one upstream response into per-molecule results.

    Boltz returns index-aligned lists -- ``scores``, ``artifacts.structures``,
    and a ``metadata`` dict of per-molecule lists -- so this is the inverse of
    the batching: pick index ``i`` out of each.
    """
    scores = data.get("scores") or []
    structures = (data.get("artifacts") or {}).get("structures") or []
    metadata = data.get("metadata") or {}

    results: list[MoleculeResult] = []
    for index in range(count):
        per_molecule = {
            key: values[index]
            for key, values in metadata.items()
            if isinstance(values, list) and index < len(values)
            and values[index] is not None
        }
        results.append(MoleculeResult(
            score=scores[index] if index < len(scores) else None,
            metadata=per_molecule,
            structure=structures[index] if index < len(structures) else None,
        ))
    return results


# ---------------------------------------------------------------------------
# Replica view (for admission control and /stats)
# ---------------------------------------------------------------------------

class ReplicaView:
    """Healthy replicas and proxy queue depth, scraped from HAProxy's CSV."""

    def __init__(self) -> None:
        self.healthy = 0
        self.proxy_queued = 0
        self.proxy_in_flight = 0
        self.reachable = False
        self._checked = 0.0

    def refresh(self) -> None:
        try:
            with urllib.request.urlopen(PROXY_STATS_URL, timeout=5.0) as response:
                text = response.read().decode()
        except (urllib.error.URLError, OSError):
            self.reachable = False
            return

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            self.reachable = False
            return
        columns = lines[0].lstrip("# ").split(",")
        try:
            sv, status = columns.index("svname"), columns.index("status")
            qcur, scur = columns.index("qcur"), columns.index("scur")
        except ValueError:
            self.reachable = False
            return

        healthy = 0
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) <= max(sv, status, qcur, scur):
                continue
            name = fields[sv]
            if name == "BACKEND":
                self.proxy_queued = int(fields[qcur] or 0)
                self.proxy_in_flight = int(fields[scur] or 0)
            elif name != "FRONTEND" and fields[status].startswith("UP"):
                healthy += 1
        self.healthy = healthy
        self.reachable = True

    async def maintain(self) -> None:
        while True:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(REPLICA_REFRESH_SECONDS)

    @property
    def inflight_limit(self) -> int:
        """Concurrent upstream calls to allow.

        Matching the healthy replica count keeps every GPU busy while
        everything beyond that accumulates into larger batches instead of
        queueing one-molecule calls behind each other.
        """
        if MAX_INFLIGHT > 0:
            return MAX_INFLIGHT
        return max(1, self.healthy)


replicas = ReplicaView()


# ---------------------------------------------------------------------------
# Coalescing batcher
# ---------------------------------------------------------------------------

@dataclass
class Waiter:
    digest: str
    canonical: str
    future: asyncio.Future


class Batcher:
    """Merges concurrent molecules into shared upstream calls."""

    def __init__(self) -> None:
        self._pending: dict[str, list[Waiter]] = {}
        self._params: dict[str, ScoreParams] = {}
        self._first_enqueued: dict[str, float] = {}
        self._inflight_molecules: dict[tuple[str, str], asyncio.Future] = {}
        self._dispatchers: dict[str, asyncio.Task] = {}
        # In-flight upstream calls. Held only so the tasks are not garbage
        # collected while running; concurrency is bounded by the semaphore.
        self._running: set[asyncio.Task] = set()
        self._slots: Optional[asyncio.Semaphore] = None
        self._slot_limit = 0
        self._lock = asyncio.Lock()
        # Counters for /stats.
        self.upstream_calls = 0
        self.molecules_predicted = 0
        self.molecules_deduplicated = 0
        self.latencies: list[float] = []

    def _semaphore(self) -> asyncio.Semaphore:
        limit = replicas.inflight_limit
        if self._slots is None or limit != self._slot_limit:
            # Replica count changed: a fresh semaphore is correct because
            # outstanding calls hold the old one until they finish.
            self._slots = asyncio.Semaphore(limit)
            self._slot_limit = limit
        return self._slots

    async def submit(
        self,
        key: str,
        params: ScoreParams,
        molecules: list[tuple[str, str]],
    ) -> dict[str, MoleculeResult]:
        """Predict each ``(digest, canonical)`` and return results by digest."""
        futures: dict[str, asyncio.Future] = {}

        async with self._lock:
            self._params.setdefault(key, params)
            for digest, canonical in molecules:
                if digest in futures:
                    continue
                existing = self._inflight_molecules.get((key, digest))
                if existing is not None:
                    # Already being predicted, by this request or another one.
                    self.molecules_deduplicated += 1
                    futures[digest] = existing
                    continue
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._inflight_molecules[(key, digest)] = future
                self._pending.setdefault(key, []).append(
                    Waiter(digest=digest, canonical=canonical, future=future)
                )
                self._first_enqueued.setdefault(key, time.monotonic())
                futures[digest] = future

            if key not in self._dispatchers or self._dispatchers[key].done():
                self._dispatchers[key] = asyncio.create_task(self._dispatch_loop(key))

        results = await asyncio.gather(*futures.values(), return_exceptions=True)
        resolved: dict[str, MoleculeResult] = {}
        for digest, result in zip(futures, results):
            if isinstance(result, BaseException):
                raise result
            resolved[digest] = result
        return resolved

    async def _dispatch_loop(self, key: str) -> None:
        while True:
            async with self._lock:
                waiting = self._pending.get(key) or []
                if not waiting:
                    self._pending.pop(key, None)
                    self._first_enqueued.pop(key, None)
                    self._params.pop(key, None)
                    return
                opened = self._first_enqueued.get(key, time.monotonic())
                full = len(waiting) >= COALESCE_MAX_MOLECULES
                expired = (time.monotonic() - opened) >= COALESCE_WINDOW_SECONDS

            if not (full or expired):
                # Hold the batch open a little longer: one more molecule in this
                # call is far cheaper than a whole extra invocation.
                await asyncio.sleep(min(0.25, COALESCE_WINDOW_SECONDS))
                continue

            # Acquired before the batch is taken, so the loop parks here while
            # every replica is busy instead of building a queue of its own.
            semaphore = self._semaphore()
            await semaphore.acquire()

            async with self._lock:
                waiting = self._pending.get(key) or []
                batch = waiting[:COALESCE_MAX_MOLECULES]
                remaining = waiting[len(batch):]
                if remaining:
                    self._pending[key] = remaining
                    self._first_enqueued[key] = time.monotonic()
                else:
                    self._pending.pop(key, None)
                    self._first_enqueued.pop(key, None)
                params = self._params.get(key)

            if not batch or params is None:
                semaphore.release()
                continue

            # Deliberately not awaited. There is one dispatch loop per
            # coalescing key, and a sweep puts every supervisor on the same
            # target with the same predict params -- so all of the fleet's
            # traffic shares one key. Awaiting the call here served those
            # batches strictly one at a time and left every replica but one
            # idle, whatever the semaphore allowed: measured as upstream_p50
            # 393 s against request_p50 752 s, with 16 molecules in flight and
            # 16 waiting on two healthy replicas. Concurrency is the
            # semaphore's job, not the loop's.
            task = asyncio.create_task(
                self._run_and_release(semaphore, key, params, batch)
            )
            # Hold a reference: a bare task can be garbage collected mid-flight.
            self._running.add(task)
            task.add_done_callback(self._running.discard)

    async def _run_and_release(
        self,
        semaphore: asyncio.Semaphore,
        key: str,
        params: ScoreParams,
        batch: list[Waiter],
    ) -> None:
        """Run one batch and give its replica slot back.

        Releases the semaphore it was given rather than the current one: a
        replica count change swaps in a fresh semaphore, and outstanding calls
        have to return the slot they actually took.
        """

        try:
            await self._run_batch(key, params, batch)
        finally:
            semaphore.release()

    async def _run_batch(
        self,
        key: str,
        params: ScoreParams,
        batch: list[Waiter],
    ) -> None:
        payload = {
            "smiles": [waiter.canonical for waiter in batch],
            "params": prediction_params(params),
        }
        logger.info(
            "upstream call: %d molecule(s), coalesce key %s", len(batch), key[:8]
        )
        started = time.monotonic()
        try:
            data = await asyncio.to_thread(_post_upstream_blocking, payload)
        except Exception as exc:  # noqa: BLE001 - every failure mode is the same here
            logger.exception("upstream call failed")
            for waiter in batch:
                if not waiter.future.done():
                    waiter.future.set_exception(exc)
            async with self._lock:
                for waiter in batch:
                    self._inflight_molecules.pop((key, waiter.digest), None)
            return

        elapsed = time.monotonic() - started
        self.upstream_calls += 1
        self.molecules_predicted += len(batch)
        self.latencies.append(elapsed)
        del self.latencies[:-500]
        logger.info(
            "upstream call done in %.1fs for %d molecule(s)", elapsed, len(batch)
        )

        results = _split_upstream(data, len(batch))
        write_through = params.cache_policy != "none" and CACHE_ENABLED
        for waiter, result in zip(batch, results):
            if write_through:
                await asyncio.to_thread(
                    cache_write, waiter.digest, waiter.canonical, result
                )
            if not waiter.future.done():
                waiter.future.set_result(result)
        async with self._lock:
            for waiter in batch:
                self._inflight_molecules.pop((key, waiter.digest), None)

    @property
    def queue_depth(self) -> int:
        return sum(len(waiting) for waiting in self._pending.values())

    @property
    def in_flight_molecules(self) -> int:
        return len(self._inflight_molecules) - self.queue_depth


batcher = Batcher()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class Counters:
    def __init__(self) -> None:
        self.requests = 0
        self.molecules_requested = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.duplicate_molecules = 0
        self.invalid_molecules = 0
        self.failures = 0
        self.request_latencies: list[float] = []

    def observe_request(self, elapsed: float) -> None:
        self.request_latencies.append(elapsed)
        del self.request_latencies[:-500]


counters = Counters()


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ---------------------------------------------------------------------------
# Async task registry
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    created: float
    response: Optional[ScoreResponse] = None
    error: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.response is not None or self.error is not None


tasks: dict[str, Task] = {}


def _expire_tasks() -> None:
    cutoff = time.time() - RESULT_TTL_SECONDS
    for task_id in [t for t, task in tasks.items() if task.created < cutoff]:
        tasks.pop(task_id, None)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create the cache dir and keep the replica view fresh while serving.

    The refresher is cancelled on shutdown rather than left running, so the
    process exits promptly instead of waiting on a sleep.
    """
    if CACHE_ENABLED:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("cache dir %s is not writable", CACHE_DIR, exc_info=True)

    refresher = asyncio.create_task(replicas.maintain())
    try:
        yield
    finally:
        refresher.cancel()
        with suppress(asyncio.CancelledError):
            await refresher


app = FastAPI(title="Boltz Gateway", lifespan=lifespan)


async def score_batch(request: ScoreRequest) -> ScoreResponse:
    """Serve a batch from the cache where possible and predict the rest."""
    started = time.monotonic()
    params = request.params
    key = coalesce_key(params)
    reads_cache = (
        CACHE_ENABLED
        and params.cache_policy == "read_write"
        and params.fidelity_tier != VALIDATION_TIER
    )

    counters.requests += 1
    counters.molecules_requested += len(request.smiles)

    # Canonicalise once, then collapse duplicates: the same molecule twice in
    # one batch is one prediction.
    digests: list[Optional[str]] = []
    canonicals: list[Optional[str]] = []
    seen: set[str] = set()
    for smiles in request.smiles:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            counters.invalid_molecules += 1
            digests.append(None)
            canonicals.append(None)
            continue
        digest = molecule_key(canonical, params)
        if digest in seen:
            counters.duplicate_molecules += 1
        seen.add(digest)
        digests.append(digest)
        canonicals.append(canonical)

    resolved: dict[str, MoleculeResult] = {}
    needed: dict[str, str] = {}
    for digest, canonical in zip(digests, canonicals):
        if digest is None or digest in resolved or digest in needed:
            continue
        hit = await asyncio.to_thread(cache_read, digest) if reads_cache else None
        if hit is not None:
            counters.cache_hits += 1
            resolved[digest] = hit
        else:
            counters.cache_misses += 1
            needed[digest] = canonical

    if needed:
        try:
            predicted = await batcher.submit(
                key, params, [(d, c) for d, c in needed.items()]
            )
        except Exception as exc:  # noqa: BLE001
            counters.failures += 1
            raise HTTPException(
                status_code=502,
                detail={"error": "boltz upstream failed", "detail": str(exc)[:500]},
            ) from exc
        resolved.update(predicted)

    # Reassemble in the caller's order. An unreadable SMILES scores None with a
    # reason, mirroring what the wrapper itself would have said.
    scores: list[Optional[float]] = []
    structures: list[Optional[str]] = []
    metadata_keys: list[str] = []
    per_molecule: list[dict[str, Any]] = []
    for digest in digests:
        if digest is None:
            scores.append(None)
            structures.append(None)
            per_molecule.append({"skip_reason": "not parseable by RDKit"})
            continue
        result = resolved.get(digest) or MoleculeResult(score=None)
        scores.append(result.score)
        structures.append(result.structure)
        per_molecule.append(result.metadata)

    for entry in per_molecule:
        for metadata_key in entry:
            if metadata_key not in metadata_keys:
                metadata_keys.append(metadata_key)
    metadata = {
        metadata_key: [entry.get(metadata_key) for entry in per_molecule]
        for metadata_key in metadata_keys
    }

    counters.observe_request(time.monotonic() - started)
    return ScoreResponse(
        scores=scores,
        metadata=metadata,
        artifacts={"structures": structures},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "cache_enabled": CACHE_ENABLED,
        "cache_dir": str(CACHE_DIR),
        "upstream": UPSTREAM_URL,
        "healthy_replicas": replicas.healthy,
        "upstream_reachable": replicas.reachable,
    }


@app.post("/score")
async def score(
    request: ScoreRequest,
    mode: str = Query("sync", description="'async' returns 202 and a task_id"),
) -> Any:
    """Score a batch.

    ``mode=async`` returns ``202 {task_id}`` immediately; poll
    ``GET /result/{task_id}``. A caller that holds the connection open instead
    is fine too -- that is the default, and what keeps this a drop-in for the
    blocking contract the scorer already speaks.
    """
    if mode == "async":
        _expire_tasks()
        task = Task(task_id=uuid.uuid4().hex, created=time.time())
        tasks[task.task_id] = task

        async def _run() -> None:
            try:
                task.response = await score_batch(request)
            except HTTPException as exc:
                task.error = json.dumps(exc.detail, default=str)
            except Exception as exc:  # noqa: BLE001
                task.error = str(exc)

        asyncio.create_task(_run())
        return JSONResponse(
            status_code=202,
            content={"task_id": task.task_id, "status": "accepted"},
        )

    return await score_batch(request)


@app.get("/result/{task_id}")
async def result(task_id: str) -> Any:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown task {task_id} (results are kept "
                   f"{RESULT_TTL_SECONDS:.0f}s)",
        )
    if not task.done:
        # Still queued or predicting. Deliberately not an error: that confusion
        # is what turns a deep queue into NaN downstream.
        return JSONResponse(
            status_code=202,
            content={"task_id": task_id, "status": "pending"},
        )
    if task.error:
        tasks.pop(task_id, None)
        raise HTTPException(status_code=502, detail=task.error)
    response = task.response
    tasks.pop(task_id, None)
    return response


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Queue depth, cache hit rate, coalescing, and latency.

    The conductor throttles concurrency off these numbers, and the cache hit
    rate here is the one every sizing calculation so far has had to assume.
    """
    looked_up = counters.cache_hits + counters.cache_misses
    return {
        "requests": counters.requests,
        "molecules": {
            "requested": counters.molecules_requested,
            "distinct_looked_up": looked_up,
            "duplicates_within_requests": counters.duplicate_molecules,
            "invalid": counters.invalid_molecules,
            "predicted": batcher.molecules_predicted,
            "deduplicated_in_flight": batcher.molecules_deduplicated,
        },
        "cache": {
            "enabled": CACHE_ENABLED,
            "hits": counters.cache_hits,
            "misses": counters.cache_misses,
            # The number §5.4's admission control needs and has never had.
            "hit_rate": (counters.cache_hits / looked_up) if looked_up else None,
        },
        "coalescing": {
            "upstream_calls": batcher.upstream_calls,
            "molecules_per_call": (
                batcher.molecules_predicted / batcher.upstream_calls
                if batcher.upstream_calls else None
            ),
            "window_seconds": COALESCE_WINDOW_SECONDS,
            "max_molecules": COALESCE_MAX_MOLECULES,
        },
        "queue": {
            "gateway_molecules_waiting": batcher.queue_depth,
            "gateway_in_flight_molecules": max(0, batcher.in_flight_molecules),
            "proxy_queued_requests": replicas.proxy_queued,
            "proxy_in_flight_requests": replicas.proxy_in_flight,
            "inflight_limit": replicas.inflight_limit,
        },
        "latency_seconds": {
            "upstream_p50": percentile(batcher.latencies, 0.50),
            "upstream_p95": percentile(batcher.latencies, 0.95),
            "request_p50": percentile(counters.request_latencies, 0.50),
            "request_p95": percentile(counters.request_latencies, 0.95),
        },
        "replicas": {
            "healthy": replicas.healthy,
            "stats_reachable": replicas.reachable,
        },
        "failures": counters.failures,
        "pending_async_tasks": sum(1 for task in tasks.values() if not task.done),
    }
