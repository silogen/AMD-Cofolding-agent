"""Boltz2 FastAPI wrapper — exposes cofolding + affinity prediction via REST."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog("rdApp.error")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("boltz2_wrapper")

SHARED_DIR = Path("/shared")
CACHE_DIR = Path("/cache")
# Persisted, cross-request MSA cache. The first time a given protein sequence
# is seen with ``use_msa_server: true`` we query the MSA server exactly once,
# write the alignment here as a ``.a3m``, and every subsequent request (i.e.
# every later RL step) reads from this file instead of hitting the server.
MSA_CACHE_DIR = SHARED_DIR / "msa_cache"
DEFAULT_MSA_SERVER_URL = "https://api.colabfold.com"
PREDICT_TIMEOUT_SECONDS = 1800

MAX_HEAVY_ATOMS = 56
MAX_TOTAL_ATOMS = 128
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# GPU predictions are serialized (FIFO) so concurrent /score calls cannot
# run two predictions on one GPU and OOM/corrupt each other. Concurrency is
# env-tunable and can be raised only when memory is proven sufficient.
BOLTZ_MAX_CONCURRENCY = int(os.getenv("BOLTZ_MAX_CONCURRENCY", "1"))
_predict_sem = asyncio.Semaphore(BOLTZ_MAX_CONCURRENCY)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# Boltz's optimized kernels stay disabled by default, and on ROCm they have to:
# the kernel path imports ``cuequivariance_torch``, which is CUDA-only and is
# not in this image, so ``BOLTZ_NO_KERNELS=0`` aborts inside the pairformer's
# triangular multiplicative update. The flag keeps that dependency explicit and
# lets a CUDA node -- or an image that ships cuEquivariance -- be retested with
# k8s/scripts/ab-boltz-kernels.py.
BOLTZ_NO_KERNELS = _env_flag("BOLTZ_NO_KERNELS", True)

app = FastAPI(title="Boltz2 Scoring Wrapper")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScoreParams(BaseModel):
    protein_sequence: str
    msa_path: Optional[str] = None
    use_msa_server: bool = False
    msa_server_url: str = DEFAULT_MSA_SERVER_URL
    msa_free_fallback: bool = False
    # When an explicit MSA is supplied but its query sequence differs from
    # protein_sequence, derive the sequence from the MSA (the MSA's first
    # record is the reference the alignment was built for) so the two stay
    # consistent. If False, a mismatch is reported as an explicit error.
    derive_sequence_from_msa: bool = True
    recycling_steps: int = 3
    diffusion_samples: int = 1
    sampling_steps: int = 200
    sampling_steps_affinity: int = 200
    diffusion_samples_affinity: int = 5
    affinity_mode: str = "pred_value"
    output_format: str = "pdb"
    affinity_mw_correction: bool = False


class ScoreRequest(BaseModel):
    smiles: list[str]
    params: ScoreParams


class ScoreResponse(BaseModel):
    scores: list[Optional[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_smiles(smi: str) -> tuple[bool, str]:
    """Check a single SMILES for Boltz compatibility.

    Returns (is_valid, reason) — reason is empty when valid.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False, f"not parseable/sanitizable by RDKit ({smi!r})"
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception as exc:
        return False, f"failed extended sanitization ({exc})"
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return False, (
            f"{mol.GetNumHeavyAtoms()} heavy atoms exceeds Boltz limit ({MAX_HEAVY_ATOMS})"
        )
    if mol.GetNumAtoms() > MAX_TOTAL_ATOMS:
        return False, (
            f"{mol.GetNumAtoms()} total atoms exceeds Boltz limit ({MAX_TOTAL_ATOMS})"
        )
    return True, ""


def _validate_protein(req: ScoreRequest) -> list[str]:
    """Validate protein sequence — these errors reject the whole request."""
    errors: list[str] = []
    seq = req.params.protein_sequence.upper().strip()
    bad = set(seq) - STANDARD_AA
    if bad:
        errors.append(
            f"protein_sequence contains non-standard residues {sorted(bad)!r}; "
            f"Boltz affinity is protein-only (no DNA/RNA)"
        )
    return errors


def _partition_smiles(smiles: list[str]) -> tuple[list[int], list[str]]:
    """Return (valid_indices, reasons) — reasons[i] is non-empty for invalid SMILES."""
    valid_indices: list[int] = []
    reasons: list[str] = [""] * len(smiles)
    for i, smi in enumerate(smiles):
        ok, reason = _validate_smiles(smi)
        if ok:
            valid_indices.append(i)
        else:
            reasons[i] = reason
            logger.warning("smiles[%d] skipped: %s", i, reason)
    return valid_indices, reasons


# ---------------------------------------------------------------------------
# MSA resolution / caching
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive cross-process lock so concurrent first-calls compute once."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _msa_cache_path(protein_sequence: str) -> tuple[str, Path]:
    """Return ``(digest, a3m_path)`` for a protein sequence's cached MSA."""
    seq = protein_sequence.upper().strip()
    digest = hashlib.sha1(seq.encode("utf-8")).hexdigest()  # noqa: S324
    return digest, MSA_CACHE_DIR / f"{digest}.a3m"


def _read_a3m_query_sequence(msa_path: str) -> Optional[str]:
    """Return the query (first) sequence of an a3m/fasta MSA, or None.

    The first record in an a3m is the reference/query sequence the alignment
    was built for -- exactly the sequence Boltz must fold. Relative paths
    resolve under the shared dir. Gaps are stripped and residues upper-cased.
    """
    try:
        p = Path(msa_path)
        if not p.is_absolute():
            p = SHARED_DIR / msa_path
        seq_lines: list[str] = []
        seen_header = False
        with open(p) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seen_header:
                        break  # reached the second record; query is complete
                    seen_header = True
                    continue
                if seen_header:
                    seq_lines.append(line)
        seq = "".join(seq_lines).replace("-", "").upper()
        return seq or None
    except Exception:
        logger.debug("Could not read MSA query sequence from %r", msa_path, exc_info=True)
        return None


def _call_msa_server(protein_sequence: str, msa_server_url: str) -> str:
    """Query the MSA server once for a single sequence and return a3m text.

    Isolated into its own function so the (boltz-dependent) import stays lazy
    and the network call is easy to stub in tests.
    """
    # Imported lazily: only needed on the first request for a target and
    # keeps module import cheap / decoupled from the boltz package.
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    seq = protein_sequence.upper().strip()
    work_dir = MSA_CACHE_DIR / f"{hashlib.sha1(seq.encode()).hexdigest()}_work"  # noqa: S324
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        a3m_list = run_mmseqs2(
            [seq],
            str(work_dir / "msa"),
            use_env=True,
            use_pairing=False,
            host_url=msa_server_url,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not a3m_list or not a3m_list[0].strip():
        raise RuntimeError("MSA server returned an empty alignment")
    return a3m_list[0]


def _ensure_cached_msa(protein_sequence: str, msa_server_url: str) -> str:
    """Return a path to a ``.a3m`` for *protein_sequence*, querying the MSA
    server at most once per unique sequence.

    On a cache hit the existing file is returned immediately. On a miss the
    MSA server is queried a single time, the alignment is written to the
    shared cache as a genuine ``.a3m``, and that path is returned so all
    subsequent calls read from disk.
    """
    digest, a3m_path = _msa_cache_path(protein_sequence)
    if a3m_path.is_file() and a3m_path.stat().st_size > 0:
        logger.info("MSA cache hit for target %s -> %s", digest, a3m_path)
        return str(a3m_path)

    lock_path = MSA_CACHE_DIR / f"{digest}.lock"
    with _file_lock(lock_path):
        # Re-check under the lock: another worker may have just written it.
        if a3m_path.is_file() and a3m_path.stat().st_size > 0:
            logger.info("MSA cache hit (post-lock) for target %s", digest)
            return str(a3m_path)

        logger.info(
            "MSA cache miss for target %s; querying MSA server %s (one-time)",
            digest, msa_server_url,
        )
        a3m_text = _call_msa_server(protein_sequence, msa_server_url)

        tmp_path = a3m_path.with_name(f"{digest}.a3m.tmp")
        tmp_path.write_text(a3m_text)
        tmp_path.replace(a3m_path)
        logger.info("Cached MSA for target %s -> %s", digest, a3m_path)
        return str(a3m_path)


def _resolve_msa(params: ScoreParams) -> tuple[Optional[str], bool, Optional[str]]:
    """Decide the MSA source for a request.

    Returns ``(msa_path, use_msa_server_flag, msa_error)``:

    * An explicit ``msa_path`` always wins (e.g. an ``.a3m`` named in the
      agent prompt) -> ``(path, False, None)``.
    * Otherwise, when ``use_msa_server`` is set we try to resolve/populate the
      shared cache so Boltz reads from a file -> ``(cached_path, False, None)``.
    * If the one-time server query fails we report the reason so the caller can
      surface an explicit, actionable error instead of an opaque NaN
      -> ``(None, False, "<reason>")``. (Boltz's own ``--use_msa_server`` would
      hit the same unreachable host, so we do not silently retry it.)
    * If no MSA source is configured -> ``(None, False, None)``.
    """
    if params.msa_path:
        # Resolve to an absolute path so Boltz (which runs with a different cwd)
        # can find it. A bare relative name like "brd4_bd1.a3m" lives on the
        # shared volume; without this it is written verbatim into the Boltz YAML
        # and fails with "MSA file ... not found".
        p = Path(params.msa_path)
        if not p.is_absolute():
            p = SHARED_DIR / params.msa_path
        if not p.is_file():
            return None, False, (
                f"msa_path {params.msa_path!r} not found on the shared volume "
                f"(looked at {p}). Use a path from list_shared_assets."
            )
        return str(p), False, None

    if params.use_msa_server:
        try:
            cached = _ensure_cached_msa(params.protein_sequence, params.msa_server_url)
            return cached, False, None
        except Exception as exc:
            logger.exception("MSA precompute/cache failed")
            return None, False, (
                f"MSA generation via the MSA server ({params.msa_server_url}) "
                f"failed: {exc}"
            )

    return None, False, None


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def _build_molecule_yaml(
    smiles: str,
    protein_sequence: str,
    msa_path: Optional[str],
    msa_free: bool = False,
) -> dict:
    """Build the Boltz input YAML dict for a single protein–ligand pair.

    ``msa_free`` sets ``msa: empty`` so Boltz runs single-sequence (no MSA);
    used as a last-resort fallback when no MSA file exists and the MSA server
    is unreachable.
    """
    protein_entry: dict[str, Any] = {"id": "A", "sequence": protein_sequence}
    if msa_free:
        protein_entry["msa"] = "empty"
    elif msa_path:
        protein_entry["msa"] = msa_path

    return {
        "version": 1,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }


def _write_batch_yamls(
    batch_dir: Path,
    smiles_list: list[str],
    protein_sequence: str,
    msa_path: Optional[str],
    msa_free: bool = False,
) -> list[Path]:
    """Write one YAML per SMILES into *batch_dir*; return paths."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, smi in enumerate(smiles_list):
        mol_yaml = _build_molecule_yaml(smi, protein_sequence, msa_path, msa_free)
        p = batch_dir / f"mol_{idx}.yaml"
        with open(p, "w") as fh:
            yaml.dump(mol_yaml, fh, default_flow_style=False, sort_keys=False)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------

def _build_predict_cmd(
    batch_dir: Path,
    out_dir: Path,
    params: ScoreParams,
    use_msa_server: bool,
) -> list[str]:
    cmd = [
        "boltz", "predict", str(batch_dir),
        "--out_dir", str(out_dir),
        "--cache", str(CACHE_DIR),
        "--output_format", params.output_format,
        "--override",
        "--recycling_steps", str(params.recycling_steps),
        "--diffusion_samples", str(params.diffusion_samples),
        "--sampling_steps", str(params.sampling_steps),
        "--sampling_steps_affinity", str(params.sampling_steps_affinity),
        "--diffusion_samples_affinity", str(params.diffusion_samples_affinity),
    ]
    if BOLTZ_NO_KERNELS:
        cmd.append("--no_kernels")
    if use_msa_server:
        cmd.append("--use_msa_server")
        cmd.extend(["--msa_server_url", params.msa_server_url])
    if params.affinity_mw_correction:
        cmd.append("--affinity_mw_correction")
    return cmd


class PredictionCancelled(Exception):
    """Raised inside the worker thread when the client disconnected."""


class _PredictionHandle:
    """Shared between the request coroutine and the worker thread.

    ``asyncio.to_thread`` tasks cannot be interrupted by ``task.cancel()`` --
    the thread stays blocked in ``proc.communicate()`` and boltz runs to
    completion. To actually stop on client disconnect we hand the request
    coroutine a reference to the live subprocess so it can kill the whole
    process group directly.
    """

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.cancelled: bool = False


def _run_predict(
    cmd: list[str],
    handle: Optional[_PredictionHandle] = None,
) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    # Launch in its own session/process group so the whole tree (boltz + any
    # children it spawns) can be killed as a unit on timeout/cancellation,
    # rather than leaking an orphan that keeps holding GPU memory.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if handle is not None:
        handle.proc = proc
        # Disconnect may have arrived between acquiring the semaphore and
        # launching the subprocess; honour it immediately.
        if handle.cancelled:
            _kill_process_group(proc)
            proc.communicate()
            raise PredictionCancelled()
    try:
        stdout, stderr = proc.communicate(timeout=PREDICT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        # Reap the killed children so they don't linger as zombies.
        proc.communicate()
        raise
    except BaseException:
        # Covers cancellation/interruption: never leave the group running.
        _kill_process_group(proc)
        proc.communicate()
        raise

    # The request coroutine may have killed the process group concurrently
    # (client disconnect). Surface that as a cancellation rather than a
    # generic non-zero "boltz predict failed".
    if handle is not None and handle.cancelled:
        raise PredictionCancelled()

    if stdout:
        logger.info("boltz stdout:\n%s", stdout[-4000:])
    if stderr:
        logger.warning("boltz stderr:\n%s", stderr[-4000:])
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGKILL of the child's entire process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Process already gone, or we can't signal it; nothing more to do.
        pass


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _find_results_dir(out_dir: Path, batch_dir: Path) -> Path | None:
    """Locate ``boltz_results_<batch_stem>/predictions`` under *out_dir*."""
    expected = out_dir / f"boltz_results_{batch_dir.name}" / "predictions"
    if expected.is_dir():
        return expected
    for candidate in out_dir.glob("boltz_results_*/predictions"):
        return candidate
    return None


def _parse_molecule_outputs(
    predictions_dir: Path,
    mol_idx: int,
    affinity_mode: str,
    output_format: str,
) -> tuple[Optional[float], dict[str, Any], Optional[str]]:
    """Parse affinity, confidence, and structure path for one molecule.

    Returns ``(score, metadata_fragment, structure_path)``.
    """
    mol_id = f"mol_{mol_idx}"
    mol_dir = predictions_dir / mol_id

    if not mol_dir.is_dir():
        return None, {}, None

    # --- affinity ---
    affinity_file = mol_dir / f"affinity_{mol_id}.json"
    affinity_data: dict[str, Any] = {}
    score: Optional[float] = None
    if affinity_file.is_file():
        with open(affinity_file) as fh:
            affinity_data = json.load(fh)
        score = affinity_data.get(f"affinity_{affinity_mode}")

    # --- confidence ---
    confidence_file = mol_dir / f"confidence_{mol_id}_model_0.json"
    confidence_data: dict[str, Any] = {}
    if confidence_file.is_file():
        with open(confidence_file) as fh:
            confidence_data = json.load(fh)

    meta: dict[str, Any] = {}
    for key in (
        "affinity_pred_value",
        "affinity_probability_binary",
        "affinity_pred_value1",
        "affinity_pred_value2",
        "affinity_probability_binary1",
        "affinity_probability_binary2",
    ):
        if key in affinity_data:
            meta[key] = affinity_data[key]
    for key in ("confidence_score", "ligand_iptm"):
        if key in confidence_data:
            meta[key] = confidence_data[key]

    # --- structure ---
    ext = output_format if output_format in ("pdb", "mmcif") else "pdb"
    structure_file = mol_dir / f"{mol_id}_model_0.{ext}"
    structure_path: Optional[str] = None
    if structure_file.is_file():
        structure_path = str(structure_file)

    return score, meta, structure_path


def _assemble_response(
    predictions_dir: Path,
    n_molecules: int,
    affinity_mode: str,
    output_format: str,
) -> ScoreResponse:
    scores: list[Optional[float]] = []
    metadata: dict[str, list] = {}
    structures: list[Optional[str]] = []

    for idx in range(n_molecules):
        score, meta, struct_path = _parse_molecule_outputs(
            predictions_dir, idx, affinity_mode, output_format,
        )
        scores.append(score)
        structures.append(struct_path)
        for key, value in meta.items():
            metadata.setdefault(key, [None] * idx)
            metadata[key].append(value)
        for key in metadata:
            if key not in meta:
                metadata[key].append(None)

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
    # ``no_kernels`` is reported so an A/B run can confirm which arm the live
    # replica is serving instead of inferring it from a rollout.
    return {"status": "ok", "no_kernels": BOLTZ_NO_KERNELS}


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest, http_request: Request) -> ScoreResponse:
    params = request.params

    # --- Fast validation stays on the event loop ---
    if not params.msa_path and not params.use_msa_server:
        raise HTTPException(
            status_code=400,
            detail=(
                "Either 'msa_path' or 'use_msa_server: true' must be provided. "
                "Boltz2 requires an MSA source."
            ),
        )

    protein_errors = _validate_protein(request)
    if protein_errors:
        raise HTTPException(status_code=422, detail=protein_errors)

    valid_indices, skip_reasons = _partition_smiles(request.smiles)
    n_total = len(request.smiles)

    if not valid_indices:
        logger.warning("All %d SMILES failed validation — returning None scores", n_total)
        return ScoreResponse(
            scores=[None] * n_total,
            metadata={"skip_reason": skip_reasons},
        )

    valid_smiles = [request.smiles[i] for i in valid_indices]
    logger.info(
        "Batch: %d/%d SMILES valid, %d skipped",
        len(valid_smiles), n_total, n_total - len(valid_smiles),
    )

    # --- Blocking GPU work runs off the event loop, serialized via the
    # semaphore so /health stays responsive and predictions are FIFO. ---
    async with _predict_sem:
        handle = _PredictionHandle()
        task = asyncio.create_task(
            asyncio.to_thread(
                _run_prediction,
                params, valid_smiles, valid_indices, skip_reasons, n_total,
                handle,
            )
        )
        # If the client goes away (agent exits / job killed), kill the boltz
        # subprocess directly -- cancelling the asyncio task does NOT stop the
        # worker thread. We then keep holding the semaphore until the thread
        # actually unwinds, so the GPU is never handed to the next request
        # while this boltz process is still tearing down (which would break
        # FIFO serialization and risk OOM).
        while True:
            done, _ = await asyncio.wait({task}, timeout=2.0)
            if done:
                break
            if await http_request.is_disconnected():
                logger.warning("Client disconnected — killing prediction")
                handle.cancelled = True
                if handle.proc is not None:
                    _kill_process_group(handle.proc)
                try:
                    await task
                except Exception:
                    pass
                raise HTTPException(status_code=499, detail="Client disconnected")
        return task.result()


def _run_prediction(
    params: ScoreParams,
    valid_smiles: list[str],
    valid_indices: list[int],
    skip_reasons: list[str],
    n_total: int,
    handle: Optional[_PredictionHandle] = None,
) -> ScoreResponse:
    """Blocking prediction + response assembly (runs off the event loop)."""
    batch_id = uuid.uuid4().hex[:12]
    batch_dir = SHARED_DIR / "batches" / f"batch_{batch_id}"
    out_dir = SHARED_DIR / "predictions" / f"out_{batch_id}"

    try:
        # Resolve the MSA source once per request: explicit a3m > shared cache
        # (populated from a single MSA-server query) > live server fallback.
        effective_msa_path, use_server_flag, msa_error = _resolve_msa(params)

        # No usable MSA and generation failed. Two options:
        #   * If msa_free_fallback is enabled, degrade to single-sequence
        #     (msa: empty) so a structure is still produced (reduced
        #     confidence) rather than losing the run entirely.
        #   * Otherwise abort with an explicit, actionable reason instead of a
        #     doomed run that surfaces as an opaque NaN.
        msa_free = False
        msa_note: Optional[str] = None
        if msa_error and not effective_msa_path:
            if params.msa_free_fallback:
                msa_free = True
                use_server_flag = False
                msa_note = (
                    f"MSA-free (single-sequence) mode used: the MSA server was "
                    f"unavailable ({msa_error}). Structure and affinity "
                    "confidence are reduced; provide a local .a3m "
                    "(target.msa_path) for a full-quality cofold."
                )
                logger.warning("Batch %s: %s", batch_id, msa_note)
            else:
                reason = (
                    f"Cofold aborted: {msa_error}. No precomputed MSA was "
                    "available for this target. Provide a local .a3m (set "
                    "target.msa_path, e.g. a file on the shared volume) or use "
                    "a target that already has an MSA."
                )
                logger.error("Batch %s: %s", batch_id, reason)
                return ScoreResponse(
                    scores=[None] * n_total,
                    metadata={
                        "skip_reason": [reason] * n_total,
                        "msa_error": [msa_error] * n_total,
                    },
                    artifacts={"structures": [None] * n_total},
                )

        if msa_free:
            logger.info(
                "MSA source for batch %s: NONE (single-sequence / msa: empty)",
                batch_id,
            )
        elif effective_msa_path:
            logger.info(
                "MSA source for batch %s: reading from file %s",
                batch_id, effective_msa_path,
            )
        elif use_server_flag:
            logger.info(
                "MSA source for batch %s: live MSA server %s (boltz --use_msa_server)",
                batch_id, params.msa_server_url,
            )
        else:
            logger.info("MSA source for batch %s: none configured", batch_id)

        # Reconcile the target sequence with the MSA's query sequence. Boltz
        # requires the folded sequence to match the MSA the alignment was built
        # for; a mismatch (e.g. full-length target + domain-only MSA) otherwise
        # produces an opaque NaN. On mismatch either derive the sequence from
        # the MSA (default) or abort with an explicit reason.
        protein_sequence = params.protein_sequence
        seq_note: Optional[str] = None
        if effective_msa_path and not msa_free:
            msa_seq = _read_a3m_query_sequence(effective_msa_path)
            target_seq = protein_sequence.upper().strip()
            if msa_seq and msa_seq != target_seq:
                if params.derive_sequence_from_msa:
                    seq_note = (
                        f"protein_sequence ({len(target_seq)} aa) did not match "
                        f"the MSA query ({len(msa_seq)} aa); used the MSA's query "
                        "sequence so the alignment and folded structure are "
                        "consistent."
                    )
                    logger.warning("Batch %s: %s", batch_id, seq_note)
                    protein_sequence = msa_seq
                else:
                    reason = (
                        f"MSA/sequence mismatch: the MSA is for a "
                        f"{len(msa_seq)}-aa sequence but protein_sequence is "
                        f"{len(target_seq)} aa. Provide the matching (domain) "
                        "sequence, or enable derive_sequence_from_msa."
                    )
                    logger.error("Batch %s: %s", batch_id, reason)
                    return ScoreResponse(
                        scores=[None] * n_total,
                        metadata={"skip_reason": [reason] * n_total},
                        artifacts={"structures": [None] * n_total},
                    )

        _write_batch_yamls(
            batch_dir,
            valid_smiles,
            protein_sequence,
            effective_msa_path,
            msa_free=msa_free,
        )

        cmd = _build_predict_cmd(batch_dir, out_dir, params, use_server_flag)
        result = _run_predict(cmd, handle)

        if result.returncode != 0:
            logger.error("boltz predict failed:\nstdout: %s\nstderr: %s",
                         result.stdout, result.stderr)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "boltz predict failed",
                    "detail": result.stderr[-2000:] if result.stderr else "",
                },
            )

        predictions_dir = _find_results_dir(out_dir, batch_dir)
        if predictions_dir is None:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Prediction output directory not found",
                    "detail": f"Expected results under {out_dir}",
                },
            )

        valid_response = _assemble_response(
            predictions_dir,
            len(valid_smiles),
            params.affinity_mode,
            params.output_format,
        )

        scores: list[Optional[float]] = [None] * n_total
        for batch_idx, orig_idx in enumerate(valid_indices):
            scores[orig_idx] = valid_response.scores[batch_idx]

        metadata: dict[str, Any] = {"skip_reason": skip_reasons}
        for key, values in valid_response.metadata.items():
            full: list[Any] = [None] * n_total
            for batch_idx, orig_idx in enumerate(valid_indices):
                full[orig_idx] = values[batch_idx]
            metadata[key] = full

        # Flag reduced-confidence MSA-free runs and sequence-alignment fixes so
        # the reason is visible downstream (scorer surfaces it; agent can warn
        # the user). These two notes are mutually exclusive (msa_note only when
        # running MSA-free; seq_note only when an MSA file is used).
        note = msa_note or seq_note
        if note:
            metadata["msa_note"] = [note] * n_total

        artifacts: dict[str, Any] = {}
        if "structures" in valid_response.artifacts:
            structs: list[Optional[str]] = [None] * n_total
            for batch_idx, orig_idx in enumerate(valid_indices):
                structs[orig_idx] = valid_response.artifacts["structures"][batch_idx]
            artifacts["structures"] = structs

        return ScoreResponse(scores=scores, metadata=metadata, artifacts=artifacts)

    except HTTPException:
        raise
    except PredictionCancelled:
        # Client disconnected: the request coroutine already killed the
        # subprocess and will return 499; just unwind quietly.
        logger.info("Prediction for batch %s cancelled (client disconnected)", batch_id)
        raise
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "boltz predict timed out",
                "detail": f"Timeout after {PREDICT_TIMEOUT_SECONDS}s",
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error during scoring")
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc)},
        )
    finally:
        if batch_dir.exists():
            shutil.rmtree(batch_dir, ignore_errors=True)
