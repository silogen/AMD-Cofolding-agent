"""Background job manager for REINVENT4 optimization runs.

Tracks subprocess-based RL/TL jobs, provides status polling via CSV
output parsing, and manages checkpoints for stop/resume workflows.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reinvent4_mcp.config import ProcessConfig, get_config
from reinvent4_mcp.jobs.toml_generator import (
    generate_reinvent_toml,
    generate_tl_toml,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


# States that hold one of the ``max_concurrent_jobs`` slots. QUEUED counts as
# well as RUNNING: a job is registered before its subprocess reports back, and
# ignoring that window would let two callers arriving together both pass the
# capacity check.
ACTIVE_JOB_STATES = frozenset({JobState.QUEUED, JobState.RUNNING})


def worker_alive(pid: Optional[int], job_dir: Any) -> bool:
    """Whether the recorded worker process is still running this job.

    The pid alone is not enough: pids are recycled, and mistaking an unrelated
    process for a live worker would leave a dead job holding a concurrency
    slot forever. Where ``/proc`` is available the command line is checked for
    this job's own directory, which both confirms liveness and rules out
    reuse. Elsewhere (macOS, tests) only liveness can be established, which is
    the safe direction: adopting is recoverable, burying a running job is not.
    """

    if not pid:
        return False

    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            argv = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return False
        return str(job_dir) in argv

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to someone else, which still means alive.
        return True
    except OSError:
        return False
    return True

# The aggregated score, by the names it can appear under. REINVENT writes
# "Score"; "total_score" is kept first because older CSVs and the TL output
# use it. Resolving only "total_score" meant the column was never found in an
# RL results file, so "top" candidates came back as the first N rows of the
# CSV -- the earliest and therefore worst molecules -- and the trajectory's
# score statistics were silently never computed.
SCORE_COLUMNS = ("total_score", "Score")

# REINVENT's own bookkeeping columns, which are not scoring components.
# Without excluding them, "Agent", "Prior" and "Target" are reported as
# objectives alongside the real ones.
NON_COMPONENT_COLUMNS = frozenset({
    "SMILES", "SMILES_state", "Step", "step", "batch", "Scaffold",
    "Agent", "Prior", "Target", *SCORE_COLUMNS,
})


def score_column(columns: Any) -> Optional[str]:
    """The aggregated-score column present in `columns`, if any."""

    return next((name for name in SCORE_COLUMNS if name in columns), None)


def read_results_csv(path: Any) -> Any:
    """Read a results CSV, tolerating rows wider than the header.

    REINVENT writes the header once and appends rows, so a scorer that reports
    more metadata keys on a later batch produces rows the header cannot
    describe and pandas refuses the whole file -- "Expected 24 fields in line
    18, saw 26". Losing a finished run's results to that is far worse than
    losing the surplus fields, which are per-molecule diagnostics rather than
    scores.

    The schema is held steady at the source as well (see the external scorer's
    metadata handling); this is what rescues files written before that.
    """

    # Imported here, as everywhere else in this module: pandas is a heavy
    # import and the agent process does not need it to manage jobs.
    import pandas as pd

    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as exc:
        logger.warning(
            "results CSV %s has rows wider than its header (%s); "
            "skipping those rows -- the steps they belong to will be missing",
            path, exc,
        )

    # The surplus fields are *inserted*, not appended: the metadata block grows
    # and `step` stays last, so in a wide row `step` sits beyond the header
    # width. Truncating to that width would therefore read None as the step and
    # shift every metadata value -- complete but wrong. Dropping the rows loses
    # the later steps, which is a real cost since those hold the better
    # molecules, but everything that survives is accurate.
    frame = pd.read_csv(path, on_bad_lines="skip", engine="python")

    # How much was lost, recorded on the frame rather than only in this
    # process's log. The caller is usually an agent in another pod, and a
    # frame that quietly holds 16 of 48 molecules is read as the whole run:
    # one three-step campaign was ranked and harvested on step 1 alone with
    # nothing anywhere saying so.
    try:
        with open(path) as handle:
            data_lines = max(0, sum(1 for _ in handle) - 1)
    except OSError:
        data_lines = len(frame)
    skipped = max(0, data_lines - len(frame))
    frame.attrs["rows_skipped"] = skipped
    frame.attrs["rows_in_file"] = data_lines
    if skipped:
        logger.warning(
            "results CSV %s: kept %d of %d rows, %d unreadable",
            path, len(frame), data_lines, skipped,
        )
    return frame


class JobType(str, enum.Enum):
    OPTIMIZATION = "optimization"
    TRANSFER_LEARNING = "transfer_learning"


@dataclass
class JobInfo:
    """Metadata for a single optimization or transfer learning job."""

    job_id: str
    ttp_id: str
    ttp_version: int
    model: str
    working_dir: str
    toml_path: str
    csv_prefix: str = "results"
    chkpt_file: str = "stage.chkpt"
    job_type: JobType = JobType.OPTIMIZATION

    state: JobState = JobState.QUEUED
    pid: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    max_steps: int = 300
    num_epochs: int = 0
    exit_code: Optional[int] = None
    output_model_file: Optional[str] = None

    _process: Optional[subprocess.Popen] = field(default=None, repr=False)

    @property
    def csv_path(self) -> Path:
        return Path(self.working_dir) / f"{self.csv_prefix}_1.csv"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.working_dir) / self.chkpt_file

    @property
    def log_path(self) -> Path:
        return Path(self.working_dir) / "reinvent.log"

    @property
    def metadata_path(self) -> Path:
        return Path(self.working_dir) / "job_metadata.json"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "ttp_id": self.ttp_id,
            "ttp_version": self.ttp_version,
            "model": self.model,
            "job_type": self.job_type.value,
            "state": self.state.value,
            "working_dir": self.working_dir,
            "pid": self.pid,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
        }
        if self.error is not None:
            d["error"] = self.error
        if self.job_type == JobType.OPTIMIZATION:
            d["max_steps"] = self.max_steps
            d["csv_path"] = str(self.csv_path)
            d["checkpoint_path"] = str(self.checkpoint_path)
            d["has_checkpoint"] = self.checkpoint_path.exists()
        else:
            d["num_epochs"] = self.num_epochs
            if self.output_model_file:
                d["output_model_file"] = self.output_model_file
                output_path = Path(self.working_dir) / self.output_model_file
                d["output_model_exists"] = output_path.exists()
                epoch_models = sorted(
                    Path(self.working_dir).glob(f"{self.output_model_file}.*"),
                )
                if epoch_models:
                    d["saved_epoch_models"] = [str(p) for p in epoch_models]
        return d


class JobManager:
    """Manages background REINVENT4 optimization jobs.

    Maintains an in-memory registry of jobs. Each job runs as a
    subprocess executing the ``_run_worker.py`` script which registers
    custom scoring components before delegating to the ``reinvent`` CLI.
    """

    def __init__(self, config: Optional[ProcessConfig] = None):
        self._config = config or get_config()
        self._jobs: dict[str, JobInfo] = {}
        self._lock = threading.Lock()

        self._working_root = Path(self._config.jobs.working_dir)
        self._working_root.mkdir(parents=True, exist_ok=True)

        self._restore_existing_jobs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_and_start(
        self,
        *,
        ttp_id: str,
        ttp_version: int,
        scorer_config: dict[str, Any],
        model: str = ".reinvent",
        max_steps: int = 300,
        batch_size: Optional[int] = None,
        sigma: Optional[int] = None,
        learning_rate: Optional[float] = None,
        resume_from: Optional[str] = None,
        termination: Optional[dict[str, Any]] = None,
        seed_smiles: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Create a new job, generate the TOML config, and start the subprocess.

        Refused when ``jobs.max_concurrent_jobs`` are already active. This used
        to stop every running job instead, which capped the process at one RL
        run: the stated reason was contention with external scoring services,
        and that is now the boltz gateway's job -- it places each batch on an
        idle replica and queues the rest. An RL step is mostly spent waiting on
        that HTTP call, so several jobs share a GPU comfortably; memory is the
        real limit, which is what the slot count is for.

        Returns a summary dict with job_id and working directory.
        """

        try:
            model_path = resolve_model_path(model)
        except ValueError as exc:
            return {"error": str(exc)}

        if (full := self._refuse_if_full()) is not None:
            return full

        job_id = str(uuid.uuid4())[:8]
        job_dir = self._working_root / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        smiles_file_path: Optional[str] = None
        if seed_smiles:
            seed_path = job_dir / "seed.smi"
            seed_path.write_text("\n".join(seed_smiles) + "\n")
            smiles_file_path = str(seed_path)

        rc = self._config.reinvent
        use_checkpoint = False
        agent_file = model_path
        chkpt_file = "stage.chkpt"

        if resume_from:
            resume_path = Path(resume_from)
            if not resume_path.is_absolute():
                resume_path = self._working_root / resume_from
            if resume_path.exists():
                use_checkpoint = True
                agent_file = str(resume_path)
                logger.info("Resuming from checkpoint: %s", resume_path)
            else:
                return {"error": f"Checkpoint not found: {resume_path}"}

        toml_str = generate_reinvent_toml(
            scorer_config=scorer_config,
            prior_file=model_path,
            agent_file=agent_file,
            device=rc.device,
            batch_size=batch_size or rc.batch_size,
            sigma=sigma or rc.sigma,
            learning_rate=learning_rate or rc.learning_rate,
            unique_sequences=rc.unique_sequences,
            randomize_smiles=rc.randomize_smiles,
            max_steps=max_steps,
            use_checkpoint=use_checkpoint,
            chkpt_file=chkpt_file,
            tb_logdir="tb_logs",
            diversity_filter_type=rc.diversity_filter.type,
            diversity_filter_bucket_size=rc.diversity_filter.bucket_size,
            diversity_filter_minscore=rc.diversity_filter.minscore,
            termination=termination,
            smiles_file=smiles_file_path,
        )

        toml_path = job_dir / "config.toml"
        toml_path.write_text(toml_str)

        job = JobInfo(
            job_id=job_id,
            ttp_id=ttp_id,
            ttp_version=ttp_version,
            model=model,
            working_dir=str(job_dir),
            toml_path=str(toml_path),
            max_steps=max_steps,
        )

        self._save_metadata(job)

        if (full := self._claim_slot(job)) is not None:
            # Another caller took the last slot while this one was writing its
            # config. Leave nothing behind.
            shutil.rmtree(job_dir, ignore_errors=True)
            return full

        self._start_process(job)

        return {
            "job_id": job_id,
            "state": job.state.value,
            "working_dir": str(job_dir),
            "toml_path": str(toml_path),
            "model": model,
            "max_steps": max_steps,
            "resume_from": resume_from,
        }

    def create_and_start_tl(
        self,
        *,
        model: str = ".reinvent",
        smiles: Optional[list[str]] = None,
        smiles_file: Optional[str] = None,
        num_epochs: int = 50,
        batch_size: int = 128,
        save_every_n_epochs: int = 5,
        sample_batch_size: int = 2000,
        validation_smiles: Optional[list[str]] = None,
        validation_smiles_file: Optional[str] = None,
        standardize_smiles: bool = True,
        randomize_smiles: bool = True,
        output_model_file: str = "tl_model.model",
    ) -> dict[str, Any]:
        """Create a transfer learning job, write SMILES files + TOML, and start.

        Either *smiles* (list of SMILES strings written to a .smi file)
        or *smiles_file* (path to an existing file) must be provided.

        Returns a summary dict with job_id, working directory, and
        output model path.
        """

        if not smiles and not smiles_file:
            return {"error": "Provide either 'smiles' (list) or 'smiles_file' (path)."}
        if smiles and smiles_file:
            return {"error": "Provide either 'smiles' or 'smiles_file', not both."}

        try:
            model_path = resolve_model_path(model)
        except ValueError as exc:
            return {"error": str(exc)}

        if (full := self._refuse_if_full()) is not None:
            return full

        job_id = str(uuid.uuid4())[:8]
        job_dir = self._working_root / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        if smiles:
            train_path = job_dir / "train.smi"
            train_path.write_text("\n".join(smiles) + "\n")
            smiles_file_resolved = str(train_path)
        else:
            smiles_file_resolved = str(smiles_file)

        validation_file_resolved: Optional[str] = None
        if validation_smiles:
            val_path = job_dir / "validation.smi"
            val_path.write_text("\n".join(validation_smiles) + "\n")
            validation_file_resolved = str(val_path)
        elif validation_smiles_file:
            validation_file_resolved = str(validation_smiles_file)

        rc = self._config.reinvent

        toml_str = generate_tl_toml(
            input_model_file=model_path,
            smiles_file=smiles_file_resolved,
            output_model_file=output_model_file,
            device=rc.device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            save_every_n_epochs=save_every_n_epochs,
            sample_batch_size=sample_batch_size,
            validation_smiles_file=validation_file_resolved,
            standardize_smiles=standardize_smiles,
            randomize_smiles=randomize_smiles,
            tb_logdir="tb_logs",
        )

        toml_path = job_dir / "config.toml"
        toml_path.write_text(toml_str)

        job = JobInfo(
            job_id=job_id,
            ttp_id="",
            ttp_version=0,
            model=model,
            working_dir=str(job_dir),
            toml_path=str(toml_path),
            job_type=JobType.TRANSFER_LEARNING,
            num_epochs=num_epochs,
            output_model_file=output_model_file,
        )

        self._save_metadata(job)

        if (full := self._claim_slot(job)) is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
            return full

        self._start_process(job)

        return {
            "job_id": job_id,
            "job_type": "transfer_learning",
            "state": job.state.value,
            "working_dir": str(job_dir),
            "toml_path": str(toml_path),
            "model": model,
            "num_epochs": num_epochs,
            "output_model_file": str(job_dir / output_model_file),
            "n_training_smiles": len(smiles) if smiles else None,
        }

    def submit_toml(
        self,
        toml_str: str,
        job_name: str = "generation",
        extra_files: dict[str, str] | None = None,
        max_steps: int = 0,
        model: str = "generation",
    ) -> dict[str, Any]:
        """Write *toml_str* to a new job directory and launch the worker subprocess.

        This is the block-08 generation entry-point — it accepts a pre-built TOML
        string (from ``generate_generation_toml``) instead of building it from a TTP,
        making it mode-agnostic (reinvent / libinvent / linkinvent).

        Parameters
        ----------
        extra_files:
            Optional dict of ``{filename: content}`` for auxiliary files that the
            TOML references (e.g. ``{"warheads.smi": "[*:1]c1cc...|[*:2]CC..."}``
            for LinkInvent fragment pairs).

        Returns the same lifecycle/path metadata shape as ``create_and_start``.
        """
        import uuid  # noqa: PLC0415

        if (full := self._refuse_if_full()) is not None:
            return full

        job_id = uuid.uuid4().hex[:16]
        job_dir = self._working_root / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        toml_path = str(job_dir / "config.toml")
        (job_dir / "config.toml").write_text(toml_str, encoding="utf-8")

        # Write auxiliary files (e.g. warheads.smi for LinkInvent)
        if extra_files:
            for fname, content in extra_files.items():
                (job_dir / fname).write_text(content, encoding="utf-8")
        else:
            # Auto-extract auxiliary SMILES files from TOML comments.
            import re as _re  # noqa: PLC0415

            # LinkInvent: # Fragments: SMILES1 | SMILES2  → warheads.smi
            match = _re.search(r"^# Fragments: (.+)$", toml_str, _re.MULTILINE)
            if match and "smiles_file" in toml_str:
                frags = [f.strip() for f in match.group(1).split("|")]
                if len(frags) >= 2:
                    # RNN LinkinVent vocabulary uses bare [*] not [*:n]
                    rnn_frags = [_re.sub(r'\[\*:\d+\]', '[*]', f) for f in frags]
                    (job_dir / "warheads.smi").write_text("|".join(rnn_frags), encoding="utf-8")

            # LibInvent: # Scaffold: SMILES  → scaffolds.smi
            # - RNN prior: bare [*] (no map numbers; vocabulary limitation)
            # - Transformer prior: keep [*:n] WITH map numbers so LibInvent can
            #   fill each attachment position separately (richer vocabulary)
            match2 = _re.search(r"^# Scaffold: (.+)$", toml_str, _re.MULTILINE)
            if match2 and "scaffolds.smi" in toml_str:
                scaffold = match2.group(1).strip()
                # Only treat as a transformer prior when "transformer" appears in a
                # model/prior-file assignment — not anywhere in the TOML (a stray
                # occurrence in a comment or unrelated path must not flip handling).
                is_transformer = bool(
                    _re.search(
                        r'(?:prior_file|agent_file|model_file|model)\s*=\s*'
                        r'["\'][^"\']*transformer',
                        toml_str,
                        _re.IGNORECASE,
                    )
                )
                if not is_transformer:
                    scaffold = _re.sub(r'\[\*:\d+\]', '[*]', scaffold)
                # transformer keeps [*:1], [*:2] intact
                (job_dir / "scaffolds.smi").write_text(scaffold + "\n", encoding="utf-8")

        job = JobInfo(
            job_id=job_id,
            ttp_id=job_name,
            ttp_version=1,
            model=model,
            job_type=JobType.OPTIMIZATION,
            state=JobState.QUEUED,
            working_dir=str(job_dir),
            toml_path=toml_path,
            max_steps=max_steps,
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        )

        self._save_metadata(job)

        if (full := self._claim_slot(job)) is not None:
            shutil.rmtree(job_dir, ignore_errors=True)
            return full

        self._start_process(job)

        return job.to_dict()

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Return current job status with score trajectory.

        Polls the subprocess, parses the output CSV for per-step
        metrics, and returns a comprehensive status report suitable
        for agent-driven convergence monitoring.
        """

        job = self._get_job(job_id)
        if isinstance(job, dict):
            return job

        self._poll_process(job)

        result = job.to_dict()
        monitoring = self._parse_trajectory(job)
        result["metrics_available"] = monitoring["metrics_available"]
        result["csv_readiness"] = monitoring["csv_readiness"]
        result["current_step"] = monitoring["current_step"]
        result["max_steps"] = job.max_steps
        result["last_csv_row"] = monitoring.get("last_csv_row")
        result["data_policy"] = (
            "metrics_available"
            if monitoring["metrics_available"]
            else "metrics_withheld"
        )
        if monitoring.get("reason"):
            result["metrics_unavailable_reason"] = monitoring["reason"]
        if monitoring["metrics_available"]:
            result["trajectory"] = monitoring["trajectory"]

        return result

    @staticmethod
    def _completed_a_step(job: JobInfo) -> bool:
        """Whether any step has reported results yet.

        The CSV exists from the moment the job starts and stays empty until
        the first step is written to it.
        """

        try:
            return job.csv_path.stat().st_size > 0
        except OSError:
            return False

    def _signal_and_wait(self, job: JobInfo, first_signal: int, grace: float) -> None:
        """Signal the job, then escalate to SIGTERM and SIGKILL if it hangs on.

        Returns as soon as the process is gone, so a job that checkpoints
        promptly costs nothing extra.
        """

        escalation = [(first_signal, grace)]
        if first_signal != signal.SIGTERM:
            escalation.append((signal.SIGTERM, 10.0))
        escalation.append((signal.SIGKILL, 5.0))

        for sig, timeout in escalation:
            try:
                os.kill(job.pid, sig)
            except ProcessLookupError:
                return

            try:
                job._process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Job %s still running %.0fs after %s",
                    job.job_id, timeout, signal.Signals(sig).name,
                )

        logger.error("Job %s could not be killed", job.job_id)

    def stop_job(self, job_id: str) -> dict[str, Any]:
        """Stop a running job, letting it checkpoint first when configured.

        REINVENT turns SIGUSR1 into a *controlled* stage interrupt: it saves the
        checkpoint, logs why it is terminating and exits 0. Any other signal
        becomes an uncontrolled one, which still writes the checkpoint but
        leaves the exception uncaught, so the job looks like it crashed.

        Either way it can only notice the signal while it is inside the scoring
        call, and only unwind to the checkpoint-writing handler once that call
        is abandoned, so the grace period has to span a full RL step.
        """

        job = self._get_job(job_id)
        if isinstance(job, dict):
            return job

        if job.state != JobState.RUNNING:
            return {
                "error": f"Job {job_id} is not running (state: {job.state.value})",
                "job": job.to_dict(),
            }

        checkpoint_on_stop = self._config.jobs.checkpoint_on_stop

        if job._process and job._process.poll() is None:
            if checkpoint_on_stop:
                grace = self._config.jobs.stop_grace_seconds
                logger.info(
                    "Asking job %s (pid %d) to checkpoint and stop, waiting up to %.0fs",
                    job_id, job.pid, grace,
                )
                self._signal_and_wait(job, signal.SIGUSR1, grace)
            else:
                logger.info(
                    "Terminating job %s (pid %d) without a checkpoint", job_id, job.pid
                )
                self._signal_and_wait(job, signal.SIGTERM, 10.0)

        job.state = JobState.STOPPED
        job.finished_at = datetime.now(timezone.utc).isoformat()
        if job._process:
            job.exit_code = job._process.returncode
        self._save_metadata(job)

        result = job.to_dict()
        if job.checkpoint_path.exists():
            result["checkpoint_path"] = str(job.checkpoint_path)
            result["message"] = f"Job stopped. Checkpoint saved at {job.checkpoint_path}"
        elif checkpoint_on_stop and not self._completed_a_step(job):
            # REINVENT only has an agent to save once a step has updated the
            # weights, so a job stopped inside its first scoring call has
            # nothing to checkpoint and nothing went wrong.
            result["message"] = (
                "Job stopped before it finished its first step, so there was no "
                "trained agent to checkpoint. Start again rather than resuming."
            )
        elif checkpoint_on_stop:
            result["message"] = (
                "Job stopped, but it wrote no checkpoint: it was killed before it "
                "finished the step it was on. Raise jobs.stop_grace_seconds to "
                "give it longer, and check the job log for what it was waiting on."
            )
        else:
            result["message"] = (
                "Job stopped without a checkpoint (jobs.checkpoint_on_stop is off)"
            )

        return result

    def get_top_candidates(
        self,
        job_id: str,
        top_n: int = 50,
        min_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """Extract the best molecules from a completed/stopped job's CSV output."""

        job = self._get_job(job_id)
        if isinstance(job, dict):
            return job

        if job.state not in (JobState.COMPLETED, JobState.STOPPED, JobState.RUNNING):
            return {"error": f"Job {job_id} has no results (state: {job.state.value})"}

        return self._parse_top_candidates(job, top_n, min_score)

    def get_alerts(self) -> list[dict[str, Any]]:
        """Check all running jobs for actionable conditions.

        Scans every running job for state changes, CSV-based diagnostics
        warnings, log-file errors, and stalled progress.  Returns a list
        of alert dicts, each with job_id, severity, type, message, and
        optional details.
        """
        alerts: list[dict[str, Any]] = []
        with self._lock:
            running = [
                j for j in self._jobs.values()
                if j.state in (JobState.RUNNING, JobState.COMPLETED, JobState.FAILED)
            ]

        for job in running:
            prev_state = job.state
            self._poll_process(job)

            # -- State change detection --
            if job.state != prev_state and prev_state == JobState.RUNNING:
                severity = "critical" if job.state == JobState.FAILED else "low"
                alerts.append({
                    "job_id": job.job_id,
                    "severity": severity,
                    "type": "job_failed" if job.state == JobState.FAILED else "job_completed",
                    "message": (
                        f"Job {job.job_id} {job.state.value}"
                        + (f": {job.error}" if job.error else "")
                    ),
                    "details": {"working_dir": job.working_dir},
                })

            if job.state != JobState.RUNNING:
                continue

            # -- CSV diagnostics reuse --
            trajectory = self._parse_trajectory(job)
            if trajectory:
                diag = trajectory.get("diagnostics")
                if diag:
                    for w in diag.get("warnings", []):
                        alerts.append({
                            "job_id": job.job_id,
                            "severity": w.get("severity", "medium"),
                            "type": w.get("type", "diagnostic_warning"),
                            "message": w.get("message", ""),
                            "details": {
                                "working_dir": job.working_dir,
                                "component": w.get("component"),
                            },
                        })

                # -- Stall detection --
                current_step = trajectory.get("current_step", 0)
                if current_step > 0 and job.started_at:
                    try:
                        from datetime import datetime, timezone

                        started = datetime.fromisoformat(job.started_at)
                        elapsed_s = (
                            datetime.now(timezone.utc) - started
                        ).total_seconds()
                        bs = job_batch_size(job)
                        expected_mols = trajectory.get("n_molecules_total", 0)
                        if elapsed_s > 600 and expected_mols > 0:
                            mols_per_sec = expected_mols / elapsed_s
                            recent_window = 300
                            expected_new = mols_per_sec * recent_window
                            if expected_new > bs and current_step == trajectory.get(
                                "_prev_step", current_step
                            ):
                                pass
                            steps_per_min = (current_step / elapsed_s) * 60
                            if steps_per_min < 0.1 and elapsed_s > 900:
                                alerts.append({
                                    "job_id": job.job_id,
                                    "severity": "high",
                                    "type": "stalled_progress",
                                    "message": (
                                        f"Job {job.job_id} appears stalled: "
                                        f"{current_step} steps in "
                                        f"{elapsed_s / 60:.0f} min "
                                        f"({steps_per_min:.2f} steps/min)."
                                    ),
                                    "details": {
                                        "working_dir": job.working_dir,
                                        "current_step": current_step,
                                        "elapsed_min": round(elapsed_s / 60, 1),
                                    },
                                })
                    except Exception:
                        logger.debug(
                            "Stall detection failed for job %s",
                            job.job_id, exc_info=True,
                        )

            # -- Log error scanning --
            log_errors = _scan_log_errors(job)
            for err in log_errors:
                alerts.append({
                    "job_id": job.job_id,
                    "severity": "high" if err["count"] >= 5 else "medium",
                    "type": "log_error",
                    "message": err["message"],
                    "details": {
                        "working_dir": job.working_dir,
                        "count": err["count"],
                        "pattern": err["pattern"],
                    },
                })

        return alerts

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return summary info for all tracked jobs."""
        with self._lock:
            for job in self._jobs.values():
                if job.state == JobState.RUNNING:
                    self._poll_process(job)
            return [j.to_dict() for j in self._jobs.values()]

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    def _start_process(self, job: JobInfo) -> None:
        """Launch the REINVENT4 worker subprocess."""

        from reinvent4_mcp.utils import custom_plugins_path, find_reinvent4_root

        worker_script = str(Path(__file__).parent / "_run_worker.py")

        env = os.environ.copy()
        try:
            r4_root = str(find_reinvent4_root())
            env["REINVENT4_ROOT"] = r4_root
            if r4_root not in env.get("PYTHONPATH", ""):
                env["PYTHONPATH"] = r4_root + os.pathsep + env.get("PYTHONPATH", "")
        except RuntimeError:
            pass

        # Put the custom-plugins dir (reinvent_plugins/components namespace layout)
        # on the worker's PYTHONPATH so REINVENT4 discovers ExternalRESTScorer
        # natively -- a stable sys.path entry, not a fragile runtime __path__ append.
        plugins_dir = custom_plugins_path()
        env["DD_MCP_PLUGINS_DIR"] = plugins_dir
        if plugins_dir not in env.get("PYTHONPATH", ""):
            env["PYTHONPATH"] = plugins_dir + os.pathsep + env.get("PYTHONPATH", "")

        mcp_package_dir = str(Path(__file__).resolve().parent.parent.parent)
        if mcp_package_dir not in env.get("PYTHONPATH", ""):
            env["PYTHONPATH"] = mcp_package_dir + os.pathsep + env.get("PYTHONPATH", "")

        log_path = str(job.log_path)

        try:
            proc = subprocess.Popen(
                [sys.executable, worker_script, job.toml_path, log_path],
                cwd=job.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = f"Failed to start process: {exc}"
            self._save_metadata(job)
            logger.exception("Failed to start job %s", job.job_id)
            return

        job._process = proc
        job.pid = proc.pid
        job.state = JobState.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._save_metadata(job)

        monitor = threading.Thread(
            target=self._monitor_process,
            args=(job,),
            daemon=True,
        )
        monitor.start()

        logger.info("Started job %s (pid %d) in %s", job.job_id, proc.pid, job.working_dir)

    def _monitor_process(self, job: JobInfo) -> None:
        """Background thread: wait for the subprocess to finish and update state."""

        proc = job._process
        if proc is None:
            return

        proc.wait()
        exit_code = proc.returncode

        if job.state == JobState.STOPPED:
            return

        job.exit_code = exit_code
        job.finished_at = datetime.now(timezone.utc).isoformat()

        if exit_code == 0:
            job.state = JobState.COMPLETED
        else:
            job.state = JobState.FAILED
            stderr_output = ""
            try:
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace")[-2000:]
            except Exception:
                pass
            job.error = f"Process exited with code {exit_code}"
            if stderr_output:
                job.error += f". Last stderr: {stderr_output}"

        self._save_metadata(job)
        logger.info(
            "Job %s finished: state=%s exit_code=%d",
            job.job_id, job.state.value, exit_code,
        )

    def _poll_process(self, job: JobInfo) -> None:
        """Check if a running job's subprocess is still alive."""

        if job.state != JobState.RUNNING or job._process is None:
            return

        retcode = job._process.poll()
        if retcode is not None and job.state == JobState.RUNNING:
            job.exit_code = retcode
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.state = JobState.COMPLETED if retcode == 0 else JobState.FAILED
            if retcode != 0:
                stderr_output = ""
                try:
                    stderr_output = job._process.stderr.read().decode("utf-8", errors="replace")[-2000:]
                except Exception:
                    pass
                job.error = f"Process exited with code {retcode}"
                if stderr_output:
                    job.error += f". Last stderr: {stderr_output}"
            self._save_metadata(job)

    # ------------------------------------------------------------------
    # CSV parsing for monitoring
    # ------------------------------------------------------------------

    def _parse_trajectory(self, job: JobInfo) -> dict[str, Any]:
        """Parse the output CSV to build a score trajectory for agent monitoring.

        The return shape is explicit even before REINVENT writes its first row.
        Callers must not infer metrics from a missing ``trajectory`` key.
        """

        csv_path = job.csv_path
        if not csv_path.exists():
            return {
                "metrics_available": False,
                "csv_readiness": "missing",
                "current_step": 0,
                "last_csv_row": None,
                "reason": "Results CSV has not been created yet.",
            }

        try:
            from pandas.errors import EmptyDataError
        except ImportError as exc:
            return {
                "metrics_available": False,
                "csv_readiness": "parse_error",
                "current_step": 0,
                "last_csv_row": None,
                "reason": f"pandas unavailable: {exc}",
            }
        try:
            df = read_results_csv(csv_path)
        except EmptyDataError:
            return {
                "metrics_available": False,
                "csv_readiness": "empty",
                "current_step": 0,
                "last_csv_row": None,
                "reason": "Results CSV exists but has no header or rows yet.",
            }
        except Exception as exc:
            logger.warning("Failed to parse CSV for job %s: %s", job.job_id, exc)
            return {
                "metrics_available": False,
                "csv_readiness": "parse_error",
                "current_step": 0,
                "last_csv_row": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }

        if df.empty:
            return {
                "metrics_available": False,
                "csv_readiness": "header_only",
                "current_step": 0,
                "last_csv_row": None,
                "reason": "Results CSV has a header but no molecule rows yet.",
            }

        total_col = score_column(df.columns)

        step_col = None
        for candidate in ("Step", "step", "batch"):
            if candidate in df.columns:
                step_col = candidate
                break

        if step_col is None:
            df["_step"] = df.index // max(1, job_batch_size(job))
            step_col = "_step"

        current_step = int(df[step_col].max()) + 1 if step_col != "_step" else len(df)

        smiles_state_col = "SMILES_state" if "SMILES_state" in df.columns else None
        n_total = len(df)
        n_valid = int((df[smiles_state_col] == 1).sum()) if smiles_state_col else n_total

        trajectory: dict[str, Any] = {
            "current_step": current_step,
            "max_steps": job.max_steps,
            "n_molecules_total": n_total,
            "n_valid": n_valid,
            "validity_rate": round(n_valid / n_total, 3) if n_total > 0 else 0.0,
        }

        if total_col and total_col in df.columns:
            scores = df[total_col].dropna()
            if not scores.empty:
                trajectory["mean_total_score"] = round(float(scores.mean()), 4)
                trajectory["best_total_score"] = round(float(scores.max()), 4)
                trajectory["latest_mean_score"] = _latest_window_mean(
                    df, total_col, step_col
                )

        component_stats = _per_component_stats(df)
        if component_stats:
            trajectory["per_component"] = component_stats

        score_trajectory = _step_trajectory(df, total_col, step_col)
        if score_trajectory:
            trajectory["score_trajectory"] = score_trajectory

        comp_trajectories = _per_component_trajectory(df, step_col)
        if comp_trajectories:
            trajectory["per_component_trajectory"] = comp_trajectories

        diagnostics = _compute_diagnostics(
            df, total_col, step_col, component_stats, job,
        )
        aggregate_check = _aggregate_consistency_check(df, total_col)
        if aggregate_check:
            if diagnostics is None:
                diagnostics = {}
            diagnostics["aggregate_consistency"] = aggregate_check
        if diagnostics:
            trajectory["diagnostics"] = diagnostics

        return {
            "metrics_available": True,
            "csv_readiness": "ready",
            "current_step": current_step,
            "last_csv_row": len(df) - 1,
            "trajectory": trajectory,
        }

    def _parse_top_candidates(
        self,
        job: JobInfo,
        top_n: int,
        min_score: Optional[float],
    ) -> dict[str, Any]:
        """Parse the CSV and return the top-scoring unique molecules."""

        csv_path = job.csv_path
        if not csv_path.exists():
            return {"error": f"No results CSV found at {csv_path}"}

        try:
            import pandas as pd

            df = read_results_csv(csv_path)
        except Exception as exc:
            return {"error": f"Failed to read results CSV: {exc}"}

        if df.empty:
            return {"candidates": [], "n_total": 0}

        smiles_col = "SMILES" if "SMILES" in df.columns else None
        total_col = score_column(df.columns)

        if smiles_col is None:
            return {"error": "CSV has no SMILES column"}

        if "SMILES_state" in df.columns:
            df = df[df["SMILES_state"] == 1]

        # J1: chemical validity gate — RDKit sanitize + valence check.
        # Filters out molecules that would crash Boltz or produce misleading
        # cofolding results (as happened with compound 3 in the agro run).
        def _is_chemically_valid(smi: str) -> bool:
            try:
                from rdkit import Chem  # noqa: PLC0415
                from rdkit import RDLogger  # noqa: PLC0415
                RDLogger.DisableLog("rdApp.*")
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                if mol is None:
                    return False
                Chem.SanitizeMol(mol)
                return True
            except Exception:
                return False

        if smiles_col:
            valid_mask = df[smiles_col].apply(lambda s: _is_chemically_valid(str(s)))
            n_invalid = int((~valid_mask).sum())
            df = df[valid_mask]
            if n_invalid:
                df.attrs["n_invalid_filtered"] = n_invalid

        if total_col and min_score is not None:
            df = df[df[total_col] >= min_score]

        if total_col:
            df = df.sort_values(total_col, ascending=False)

        df = df.drop_duplicates(subset=[smiles_col])
        df = df.head(top_n)

        score_columns = [
            c for c in df.columns
            if c not in NON_COMPONENT_COLUMNS
            and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]

        raw_columns = {c: c for c in score_columns if "(raw)" in c}
        transformed_columns = {
            c: c for c in score_columns if c not in raw_columns and c != total_col
        }

        candidates: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            mol: dict[str, Any] = {
                "smiles": row[smiles_col],
            }
            if total_col:
                mol["total_score"] = _safe_float(row.get(total_col))

            components: dict[str, dict[str, Any]] = {}
            for col in transformed_columns:
                comp_name = col
                comp_data: dict[str, Any] = {"transformed": _safe_float(row.get(col))}
                raw_col = f"{col} (raw)"
                if raw_col in df.columns:
                    comp_data["raw"] = _safe_float(row.get(raw_col))
                components[comp_name] = comp_data

            if components:
                mol["components"] = components
            candidates.append(mol)

        return {
            "candidates": candidates,
            "n_total": len(candidates),
            "job_id": job.job_id,
            "job_state": job.state.value,
        }

    # ------------------------------------------------------------------
    # Persistence / recovery
    # ------------------------------------------------------------------

    def _save_metadata(self, job: JobInfo) -> None:
        """Write job metadata to a JSON file in the working directory."""
        try:
            data = job.to_dict()
            Path(job.metadata_path).write_text(json.dumps(data, indent=2))
        except Exception:
            logger.debug("Failed to save metadata for job %s", job.job_id, exc_info=True)

    def _restore_existing_jobs(self) -> None:
        """Scan the working root for previous job directories and restore metadata."""

        for job_dir in sorted(self._working_root.iterdir()):
            if not job_dir.is_dir() or not job_dir.name.startswith("job_"):
                continue

            metadata_path = job_dir / "job_metadata.json"
            if not metadata_path.exists():
                continue

            try:
                data = json.loads(metadata_path.read_text())
                job_id = data["job_id"]

                raw_type = data.get("job_type", "optimization")
                try:
                    jtype = JobType(raw_type)
                except ValueError:
                    jtype = JobType.OPTIMIZATION

                job = JobInfo(
                    job_id=job_id,
                    ttp_id=data.get("ttp_id", "unknown"),
                    ttp_version=data.get("ttp_version", 1),
                    model=data.get("model", "unknown"),
                    working_dir=str(job_dir),
                    toml_path=data.get("toml_path", str(job_dir / "config.toml")),
                    job_type=jtype,
                    state=JobState(data.get("state", "failed")),
                    pid=data.get("pid"),
                    created_at=data.get("created_at", ""),
                    started_at=data.get("started_at"),
                    finished_at=data.get("finished_at"),
                    error=data.get("error"),
                    max_steps=data.get("max_steps", 300),
                    num_epochs=data.get("num_epochs", 0),
                    exit_code=data.get("exit_code"),
                    output_model_file=data.get("output_model_file"),
                )
                # A job that held a slot is usually dead with the process that
                # ran it -- but not always, because the worker is a separate
                # process whose pid is recorded right here. Any second process
                # in this container that builds a manager would otherwise bury
                # live jobs: three healthy runs mid-Boltz-call were marked
                # failed on disk by one diagnostic command, and the
                # supervisors watching them abandoned the tasks.
                #
                # Resetting QUEUED as well as RUNNING still matters where the
                # process really is gone: with the slot count enforced, a job
                # interrupted before it started would hold one forever.
                if job.state in ACTIVE_JOB_STATES:
                    if worker_alive(job.pid, job_dir):
                        logger.info(
                            "Adopting job %s: its worker (pid %s) is still "
                            "running", job_id, job.pid,
                        )
                    else:
                        job.state = JobState.FAILED
                        job.error = "Server restarted while job was active"
                        self._save_metadata(job)

                self._jobs[job_id] = job
                logger.info("Restored job %s (state: %s)", job_id, job.state.value)
            except Exception:
                logger.debug("Skipping invalid job dir: %s", job_dir, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def active_job_ids(self) -> list[str]:
        """Jobs currently holding a concurrency slot.

        The public view, for callers that want to report capacity -- the
        runner's ``/health`` among them.
        """
        with self._lock:
            return self._active_job_ids()

    def _active_job_ids(self) -> list[str]:
        """Jobs holding a concurrency slot. Caller must hold ``self._lock``."""
        return [
            j.job_id for j in self._jobs.values() if j.state in ACTIVE_JOB_STATES
        ]

    def _capacity_error(self) -> dict[str, Any]:
        """Why the job was refused, and what to stop. Caller holds the lock."""
        active = self._active_job_ids()
        return {
            "error": (
                f"Maximum concurrent jobs ({self._config.jobs.max_concurrent_jobs}) "
                f"already active: {', '.join(active)}. Stop one first, or raise "
                "jobs.max_concurrent_jobs."
            ),
            "active_jobs": active,
        }

    def _refuse_if_full(self) -> Optional[dict[str, Any]]:
        """Pre-check, so a job that cannot start does no filesystem work."""
        with self._lock:
            if len(self._active_job_ids()) >= self._config.jobs.max_concurrent_jobs:
                return self._capacity_error()
        return None

    def _claim_slot(self, job: JobInfo) -> Optional[dict[str, Any]]:
        """Register *job* unless every slot is taken.

        The count and the registration share one lock acquisition, so two
        callers arriving together cannot both get through -- which matters now
        that several agents can drive one runner. Returns an error dict when
        the slot was lost, or None once the job is registered.
        """
        with self._lock:
            if len(self._active_job_ids()) >= self._config.jobs.max_concurrent_jobs:
                return self._capacity_error()
            self._jobs[job.job_id] = job
        return None

    def _get_job(self, job_id: str) -> JobInfo | dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"Job '{job_id}' not found"}
        return job


# ---------------------------------------------------------------------------
# Module-level helpers for CSV analysis
# ---------------------------------------------------------------------------


def job_batch_size(job: JobInfo) -> int:
    """Read batch_size from the TOML config, defaulting to 16."""
    toml_path = Path(job.toml_path)
    if toml_path.exists():
        for line in toml_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("batch_size"):
                try:
                    return int(stripped.split("=")[1].strip())
                except (ValueError, IndexError):
                    pass
    return 16


def _safe_float(val: Any) -> Optional[float]:
    """Convert a value to float, returning None for NaN or missing."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 4)
    except (ValueError, TypeError):
        return None


def _latest_window_mean(
    df: Any, total_col: str, step_col: str, window: int = 10
) -> Optional[float]:
    """Compute the mean total score over the last *window* steps."""
    try:
        max_step = df[step_col].max()
        cutoff = max(0, max_step - window + 1)
        recent = df[df[step_col] >= cutoff][total_col].dropna()
        if recent.empty:
            return None
        return round(float(recent.mean()), 4)
    except Exception:
        return None


def _per_component_stats(df: Any) -> dict[str, dict[str, Any]]:
    """Compute per-component summary statistics from the CSV."""
    stats: dict[str, dict[str, Any]] = {}

    skip_cols = NON_COMPONENT_COLUMNS
    for col in df.columns:
        if col in skip_cols or "(raw)" in col:
            continue
        if df[col].dtype not in ("float64", "float32"):
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        entry: dict[str, Any] = {
            "mean": round(float(values.mean()), 4),
            "max": round(float(values.max()), 4),
            "min": round(float(values.min()), 4),
        }

        raw_col = f"{col} (raw)"
        if raw_col in df.columns:
            raw_vals = df[raw_col].dropna()
            if not raw_vals.empty:
                entry["mean_raw"] = round(float(raw_vals.mean()), 4)
                entry["best_raw"] = round(float(raw_vals.max()), 4)

        stats[col] = entry

    return stats


def _aggregate_consistency_check(
    df: Any,
    total_col: Optional[str],
) -> Optional[dict[str, Any]]:
    """Compare the latest reported total with simple component aggregates.

    REINVENT may apply weights, so a mismatch is a diagnostic rather than a
    hard failure.  It is still valuable: it prevents an agent from publishing
    a total that is arithmetically incompatible with the component values it
    displays in the same report.
    """
    if total_col is None or total_col not in df.columns or df.empty:
        return None

    skip_cols = {
        "SMILES", "SMILES_state", "Step", "step", "batch", total_col,
    }
    component_cols = [
        col
        for col in df.columns
        if col not in skip_cols
        and "(raw)" not in col
        and getattr(df[col].dtype, "kind", "") in ("f", "i", "u")
    ]
    if len(component_cols) < 2:
        return None

    latest = df.dropna(subset=[total_col]).tail(1)
    if latest.empty:
        return None

    values: list[float] = []
    for col in component_cols:
        value = _safe_float(latest.iloc[0][col])
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None

    reported = _safe_float(latest.iloc[0][total_col])
    if reported is None:
        return None

    arithmetic = sum(values) / len(values)
    if any(value <= 0 for value in values):
        geometric = 0.0
    else:
        geometric = math.exp(sum(math.log(value) for value in values) / len(values))

    tolerance = max(0.02, abs(reported) * 0.10)
    nearest_delta = min(
        abs(reported - arithmetic),
        abs(reported - geometric),
    )
    consistent = nearest_delta <= tolerance
    result: dict[str, Any] = {
        "reported_total": round(reported, 4),
        "unweighted_arithmetic_mean": round(arithmetic, 4),
        "unweighted_geometric_mean": round(geometric, 4),
        "n_components": len(values),
        "matches_unweighted_aggregate": consistent,
    }
    if not consistent:
        result["warning"] = (
            "Reported total differs from both unweighted component aggregates; "
            "verify component weights before narrating the trajectory."
        )
    return result


def _step_trajectory(
    df: Any,
    total_col: Optional[str],
    step_col: Optional[str],
    max_points: int = 50,
) -> Optional[list[dict[str, Any]]]:
    """Build a sampled step-by-step score trajectory.

    Returns at most *max_points* data points evenly spaced across
    the completed steps, suitable for agent visualization.
    """

    if total_col is None or step_col is None:
        return None
    if total_col not in df.columns or step_col not in df.columns:
        return None

    try:
        grouped = df.groupby(step_col)[total_col].agg(["mean", "max", "count"])
        if grouped.empty:
            return None

        if len(grouped) > max_points:
            indices = np.linspace(0, len(grouped) - 1, max_points, dtype=int)
            grouped = grouped.iloc[indices]

        points: list[dict[str, Any]] = []
        for step_val, row in grouped.iterrows():
            points.append({
                "step": int(step_val),
                "mean_score": round(float(row["mean"]), 4),
                "max_score": round(float(row["max"]), 4),
                "n_molecules": int(row["count"]),
            })
        return points
    except Exception:
        return None


def _per_component_trajectory(
    df: Any,
    step_col: Optional[str],
    max_points: int = 30,
) -> Optional[dict[str, list[dict[str, Any]]]]:
    """Build per-component step-by-step trajectories for agent monitoring.

    Returns a dict mapping component names to sampled trajectory lists,
    allowing the agent to see how each individual scoring component
    evolves over time (not just the aggregate total score).
    """

    if step_col is None or step_col not in df.columns:
        return None

    skip_cols = NON_COMPONENT_COLUMNS
    comp_cols = [
        c for c in df.columns
        if c not in skip_cols and "(raw)" not in c
        and df[c].dtype in ("float64", "float32")
    ]

    if not comp_cols:
        return None

    try:
        result: dict[str, list[dict[str, Any]]] = {}
        for col in comp_cols:
            grouped = df.groupby(step_col)[col].agg(["mean", "count"])
            grouped = grouped[grouped["count"] > 0]
            if grouped.empty:
                continue

            if len(grouped) > max_points:
                indices = np.linspace(0, len(grouped) - 1, max_points, dtype=int)
                grouped = grouped.iloc[indices]

            points: list[dict[str, Any]] = []
            for step_val, row in grouped.iterrows():
                mean_val = float(row["mean"])
                if np.isnan(mean_val):
                    continue
                points.append({
                    "step": int(step_val),
                    "mean_score": round(mean_val, 4),
                })
            if points:
                result[col] = points

        return result if result else None
    except Exception:
        return None


def _compute_diagnostics(
    df: Any,
    total_col: Optional[str],
    step_col: Optional[str],
    component_stats: dict[str, dict[str, Any]],
    job: JobInfo,
) -> Optional[dict[str, Any]]:
    """Compute convergence diagnostics for agent-driven monitoring.

    Analyses the score trajectory to detect plateaus, divergence,
    component failures, and validity problems, then produces
    actionable warnings and recommendations.
    """

    if df.empty or total_col is None or step_col is None:
        return None
    if total_col not in df.columns or step_col not in df.columns:
        return None

    try:
        diagnostics: dict[str, Any] = {}

        steps = sorted(df[step_col].unique())
        n_steps = len(steps)

        if n_steps < 3:
            diagnostics["convergence_status"] = "early"
            diagnostics["message"] = "Too few steps for convergence analysis"
            health = _component_health(df)
            if health:
                diagnostics["component_health"] = health
            return diagnostics

        grouped = df.groupby(step_col)[total_col].mean()

        # --- Trend analysis: compare early vs recent windows ---
        window_frac = 0.2
        window_size = max(2, int(n_steps * window_frac))

        early_steps = steps[:window_size]
        recent_steps = steps[-window_size:]
        mid_point = n_steps // 2
        mid_steps = steps[max(0, mid_point - window_size // 2):mid_point + window_size // 2]

        early_mean = float(grouped[grouped.index.isin(early_steps)].mean())
        recent_mean = float(grouped[grouped.index.isin(recent_steps)].mean())
        mid_mean = float(grouped[grouped.index.isin(mid_steps)].mean()) if mid_steps else early_mean

        diagnostics["trend"] = {
            "early_mean": round(early_mean, 4) if not np.isnan(early_mean) else None,
            "mid_mean": round(mid_mean, 4) if not np.isnan(mid_mean) else None,
            "recent_mean": round(recent_mean, 4) if not np.isnan(recent_mean) else None,
            "overall_improvement": round(recent_mean - early_mean, 4)
            if not (np.isnan(recent_mean) or np.isnan(early_mean))
            else None,
        }

        # --- Improvement rate (score change per step over recent window) ---
        recent_grouped = grouped[grouped.index.isin(recent_steps)]
        if len(recent_grouped) >= 2:
            recent_vals = recent_grouped.values
            step_indices = np.arange(len(recent_vals), dtype=float)
            if step_indices[-1] > step_indices[0]:
                slope = float(np.polyfit(step_indices, recent_vals, 1)[0])
                diagnostics["improvement_rate"] = round(slope, 6)
            else:
                diagnostics["improvement_rate"] = 0.0
        else:
            diagnostics["improvement_rate"] = None

        # --- Plateau detection ---
        plateau_threshold = 0.005
        plateau_window = max(3, int(n_steps * 0.15))
        plateau_detected = False
        plateau_since_step = None

        if n_steps >= plateau_window + 2:
            tail_steps = steps[-plateau_window:]
            tail_scores = grouped[grouped.index.isin(tail_steps)]
            if len(tail_scores) >= 2:
                score_range = float(tail_scores.max() - tail_scores.min())
                if score_range < plateau_threshold:
                    plateau_detected = True
                    plateau_since_step = int(tail_steps[0])

        diagnostics["plateau_detected"] = plateau_detected
        if plateau_since_step is not None:
            diagnostics["plateau_since_step"] = plateau_since_step

        # --- Divergence detection (scores getting worse) ---
        diverging = False
        if not np.isnan(recent_mean) and not np.isnan(mid_mean):
            if recent_mean < mid_mean - 0.02 and n_steps >= 10:
                diverging = True
        diagnostics["diverging"] = diverging

        # --- Convergence status ---
        progress = n_steps / max(1, job.max_steps)
        if job.state in (JobState.COMPLETED, JobState.STOPPED):
            diagnostics["convergence_status"] = "completed"
        elif progress < 0.1 and n_steps < 10:
            diagnostics["convergence_status"] = "early"
        elif diverging:
            diagnostics["convergence_status"] = "diverging"
        elif plateau_detected:
            diagnostics["convergence_status"] = "plateau"
        elif (diagnostics.get("improvement_rate") or 0) > 0.001:
            diagnostics["convergence_status"] = "improving"
        else:
            diagnostics["convergence_status"] = "stable"

        # --- Progress estimation ---
        diagnostics["progress_pct"] = round(progress * 100, 1)

        # --- Component health ---
        health = _component_health(df)
        if health:
            diagnostics["component_health"] = health

        # --- Validity analysis ---
        smiles_state_col = "SMILES_state" if "SMILES_state" in df.columns else None
        if smiles_state_col:
            recent_df = df[df[step_col].isin(recent_steps)]
            if not recent_df.empty:
                recent_valid = (recent_df[smiles_state_col] == 1).sum()
                recent_validity = float(recent_valid) / len(recent_df)
                diagnostics["recent_validity_rate"] = round(recent_validity, 3)

        # --- Warnings and recommendations ---
        warnings = _generate_warnings(
            diagnostics, health, df, total_col, smiles_state_col,
        )
        if warnings:
            diagnostics["warnings"] = warnings

        recommendations = _generate_recommendations(diagnostics, health)
        if recommendations:
            diagnostics["recommendations"] = recommendations

        return diagnostics
    except Exception:
        logger.debug("Failed to compute diagnostics", exc_info=True)
        return None


#: What the TTP mapper names a constraint's column (``_map_constraint``).
CONSTRAINT_COLUMN_PREFIX = "constraint_"


def rejecting_constraints(health: dict[str, dict[str, Any]]) -> list[str]:
    """Constraints rejecting so much of the batch that they explain the zeros.

    A constraint maps to a REINVENT filter, which scores 0 for a molecule it
    matches and masks every other component for it. So one over-broad SMARTS
    makes the whole scoring pipeline look broken, which is how it gets
    misdiagnosed: the symptom is every component reading zero, including
    descriptors that are pure arithmetic over the structure.
    """

    return [
        name
        for name, entry in health.items()
        if name.startswith(CONSTRAINT_COLUMN_PREFIX)
        and entry.get("zero_rate", 0.0) >= 0.8
    ]


def _component_health(df: Any) -> dict[str, dict[str, Any]]:
    """Analyse per-component NaN rates and score distributions.

    Returns a health report for each scoring component, flagging
    components with high NaN rates or degenerate score distributions
    as 'failing' or 'degraded'.
    """

    skip_cols = NON_COMPONENT_COLUMNS
    health: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        if col in skip_cols or "(raw)" in col:
            continue
        if df[col].dtype not in ("float64", "float32"):
            continue

        total = len(df[col])
        if total == 0:
            continue

        n_nan = int(df[col].isna().sum())
        nan_rate = n_nan / total
        valid_vals = df[col].dropna()
        n_zero = int((valid_vals == 0).sum()) if not valid_vals.empty else 0
        zero_rate = n_zero / total

        if nan_rate > 0.5:
            status = "failing"
        elif nan_rate > 0.2 or zero_rate >= 0.8:
            status = "degraded"
        else:
            status = "healthy"

        entry: dict[str, Any] = {
            "status": status,
            "nan_rate": round(nan_rate, 3),
        }
        if n_zero > 0:
            entry["zero_rate"] = round(zero_rate, 3)
        if not valid_vals.empty:
            entry["mean"] = round(float(valid_vals.mean()), 4)

        health[col] = entry

    return health


def _generate_warnings(
    diagnostics: dict[str, Any],
    health: dict[str, dict[str, Any]],
    df: Any,
    total_col: Optional[str],
    smiles_state_col: Optional[str],
) -> list[dict[str, str]]:
    """Generate actionable warnings based on diagnostics and component health."""

    warnings: list[dict[str, str]] = []

    recent_validity = diagnostics.get("recent_validity_rate")
    if recent_validity is not None and recent_validity < 0.5:
        severity = "high" if recent_validity < 0.25 else "medium"
        warnings.append({
            "type": "low_validity",
            "severity": severity,
            "message": (
                f"Only {recent_validity:.0%} of recently generated SMILES are valid. "
                "The model may be struggling to generate chemically valid molecules."
            ),
        })

    if diagnostics.get("plateau_detected"):
        step = diagnostics.get("plateau_since_step", "?")
        warnings.append({
            "type": "score_plateau",
            "severity": "medium",
            "message": f"Total score has plateaued since step {step}. The run may be stalled.",
        })

    if diagnostics.get("diverging"):
        warnings.append({
            "type": "diverging",
            "severity": "high",
            "message": (
                "Scores are declining -- the model may be diverging. "
                "Consider stopping and adjusting the TTP."
            ),
        })

    # A constraint is a REINVENT filter: a molecule that matches is masked, and
    # every other component reports zero for it -- including descriptors like
    # MolecularWeight that cannot be zero for a real molecule. Reported first
    # and on its own, because the alternative is one "most scores are zero"
    # warning per component, none of which names the cause. One live run lost
    # ten minutes to exactly that: a SMARTS of `[#6][#6][#6][#6]` rejected
    # every molecule, the monitor flagged `QED_pre`, and the agent stopped both
    # jobs and rewrote the TTP around a component that was never at fault.
    rejecting = rejecting_constraints(health)
    for name in rejecting:
        rate = health[name].get("zero_rate", 1.0)
        warnings.append({
            "type": "constraint_rejects_batch",
            "severity": "high",
            "component": name,
            "message": (
                f"Constraint '{name}' rejects {rate:.0%} of molecules. A "
                "rejected molecule is masked, so every other component reads "
                "zero for it and the run has no gradient. Those zeros are a "
                "consequence, not separate failures -- fix this constraint's "
                "patterns rather than the components reporting zero."
            ),
        })

    for comp_name, comp_health in health.items():
        if comp_health["status"] == "failing":
            # Always reported: masking produces zeros, never NaN, so a NaN
            # rate this high is the component's own failure either way.
            warnings.append({
                "type": "component_failure",
                "severity": "high",
                "component": comp_name,
                "message": (
                    f"Component '{comp_name}' is returning NaN for "
                    f"{comp_health['nan_rate']:.0%} of molecules."
                ),
            })
        elif comp_health["status"] == "degraded":
            if rejecting and comp_health["nan_rate"] <= 0.2:
                # Degraded only by zeros, while a constraint is rejecting the
                # batch. For the other components that is the mask; for the
                # constraint itself it is what the warning above already said.
                continue
            warnings.append({
                "type": "component_degraded",
                "severity": "medium",
                "component": comp_name,
                "message": (
                    f"Component '{comp_name}' has high NaN rate "
                    f"({comp_health['nan_rate']:.0%}) or most scores are zero."
                ),
            })

    return warnings


def _generate_recommendations(
    diagnostics: dict[str, Any],
    health: dict[str, dict[str, Any]],
) -> list[str]:
    """Produce actionable recommendations based on diagnostics."""

    recs: list[str] = []
    status = diagnostics.get("convergence_status")

    # Before anything about weights or objectives: a constraint rejecting the
    # batch means there is no gradient at all, and no weight change helps.
    rejecting = rejecting_constraints(health)
    if rejecting:
        recs.append(
            f"Fix the constraint(s) first: {', '.join(rejecting)}. They reject "
            "almost every molecule, which masks every other component to zero "
            "-- so the plateau and the zero component scores are symptoms of "
            "this, and adjusting weights or dropping components will not help. "
            "Check the SMARTS: a pattern like [#6][#6][#6][#6] matches any four "
            "connected carbons, so it fires on practically every molecule."
        )

    if status == "plateau" and not rejecting:
        recs.append(
            "Score has plateaued. Consider stopping the run and adjusting TTP "
            "weights or objectives, then resuming from checkpoint."
        )

    if status == "diverging":
        recs.append(
            "Scores are declining. Stop the run, review component weights, "
            "and consider lowering the learning rate or increasing sigma."
        )

    failing = [n for n, h in health.items() if h["status"] == "failing"]
    if failing:
        comp_list = ", ".join(failing)
        recs.append(
            f"Components returning mostly NaN: {comp_list}. "
            "Check if the external service is healthy (use list_external_services), "
            "or reduce their weight in the TTP."
        )

    recent_validity = diagnostics.get("recent_validity_rate")
    if recent_validity is not None and recent_validity < 0.3:
        recs.append(
            "Very low validity rate. The model may benefit from transfer learning "
            "on a focused SMILES set before running RL, or try a different prior model."
        )

    if status == "improving":
        progress = diagnostics.get("progress_pct", 0)
        if progress > 80:
            recs.append(
                "Run is progressing well and nearing completion. "
                "Consider letting it finish."
            )

    if status == "early":
        recs.append(
            "Too few steps to assess convergence. Continue monitoring."
        )

    return recs


# ---------------------------------------------------------------------------
# Log error scanning
# ---------------------------------------------------------------------------


def _scan_log_errors(job: JobInfo, tail_lines: int = 200) -> list[dict[str, Any]]:
    """Read the last *tail_lines* of reinvent.log and group ERRO lines.

    Returns a list of dicts with keys: pattern, message, count.
    Repeated error patterns (e.g. "posebusters returned 422") are
    grouped into a single entry with a count.
    """
    import re
    from collections import Counter

    log_path = job.log_path
    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
    except OSError:
        return []

    erro_pattern = re.compile(r"ERRO")
    error_lines: list[str] = []
    for line in tail:
        if erro_pattern.search(line):
            error_lines.append(line.strip())

    if not error_lines:
        return []

    normalizer = re.compile(r"\d+")
    counts: Counter[str] = Counter()
    exemplars: dict[str, str] = {}
    for line in error_lines:
        key = normalizer.sub("N", line)[:200]
        counts[key] += 1
        if key not in exemplars:
            exemplars[key] = line[:300]

    results: list[dict[str, Any]] = []
    for pattern, count in counts.most_common(10):
        exemplar = exemplars[pattern]
        msg = exemplar if count == 1 else f"{exemplar} (x{count})"
        results.append({"pattern": pattern, "message": msg, "count": count})

    return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_manager: Optional[JobManager] = None


def get_job_manager():
    """Return whatever owns the jobs: this process, or the runner service.

    With ``runner.enabled`` the caller gets a ``RemoteJobManager`` speaking the
    same seven methods over HTTP, so no tool has to know where REINVENT4 runs.
    Not cached, so flipping the config takes effect without a restart.
    """
    from reinvent4_mcp.runner_client import RemoteJobManager, runner_enabled

    if runner_enabled():
        return RemoteJobManager()

    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def reset_job_manager() -> None:
    """Clear the singleton (for tests)."""
    global _manager
    _manager = None
