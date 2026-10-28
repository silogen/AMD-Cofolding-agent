# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Persistent Boltz-2 worker process.

Each worker owns one GPU, loads the Boltz-2 structure and affinity models once
and then serves single-molecule predictions received over a multiprocessing
pipe. The per-molecule pipeline mirrors ``boltz predict`` (boltz 2.2.1,
``boltz/main.py``): ``process_input`` -> manifest -> ``Boltz2InferenceDataModule``
-> ``BoltzWriter``, then the affinity pass with ``BoltzAffinityWriter``, and it
writes the same ``boltz_results_<batch>/predictions/mol_<i>/`` layout.

Nothing here imports torch at module level: the device must be pinned through
the environment before torch initialises HIP/CUDA in the child.
"""

from __future__ import annotations

import contextlib
import fcntl
import gc
import io
import logging
import os
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("boltz2_worker")

# Values ``boltz predict`` uses when the corresponding flag is not passed.
MAX_PARALLEL_SAMPLES = 5
MAX_MSA_SEQS = 8192
NUM_SUBSAMPLED_MSA = 1024
BOLTZ2_STEP_SCALE = 1.5
AFFINITY_RECYCLING_STEPS = 5


def worker_main(worker_id: str, device_index: int, conn: Any, cache_dir: str) -> None:
    """Entry point of a spawned worker process."""
    os.environ["HIP_VISIBLE_DEVICES"] = str(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s boltz2_worker[{worker_id}] %(levelname)s %(message)s",
    )
    try:
        predictor = Predictor(Path(cache_dir))
    except BaseException as exc:  # noqa: BLE001
        logger.exception("Model load failed")
        conn.send({"type": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return
    conn.send({"type": "ready", **predictor.memory_info()})
    logger.info("Ready on device %s: %s", device_index, predictor.memory_info())

    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        conn.send(predictor.run(job))


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)


class Predictor:
    """Resident Boltz-2 structure + affinity models on ``cuda:0``."""

    def __init__(self, cache: Path) -> None:
        import torch
        from boltz.data.mol import load_canonicals
        from boltz.main import (
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            MSAModuleArgs,
            PairformerArgsV2,
            download_boltz2,
        )
        from boltz.model.models.boltz2 import Boltz2
        from rdkit import Chem

        warnings.filterwarnings(
            "ignore", ".*that has Tensor Cores. To properly utilize them.*"
        )
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        for key in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
            os.environ[key] = os.environ.get(key, "1")

        if not torch.cuda.is_available():
            raise RuntimeError("no GPU visible to this worker")
        self.device = torch.device("cuda", 0)

        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        # Several workers start at once; only one may download/extract.
        with _file_lock(self.cache / ".boltz_download.lock"):
            download_boltz2(self.cache)
        self.mol_dir = self.cache / "mols"
        self.ccd = load_canonicals(self.mol_dir)

        diffusion_params = Boltz2DiffusionParams()
        diffusion_params.step_scale = BOLTZ2_STEP_SCALE
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs(
            subsample_msa=False,
            num_subsampled_msa=NUM_SUBSAMPLED_MSA,
            use_paired_feature=True,
        )

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        self.structure_model = Boltz2.load_from_checkpoint(
            self.cache / "boltz2_conf.ckpt",
            strict=True,
            predict_args=self._structure_args({}),
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        self.structure_model.eval()
        self.structure_model.to(self.device)

        affinity_steering = BoltzSteeringParams()
        affinity_steering.fk_steering = False
        affinity_steering.physical_guidance_update = False
        affinity_steering.contact_guidance_update = False
        self.affinity_model = Boltz2.load_from_checkpoint(
            self.cache / "boltz2_aff.ckpt",
            strict=True,
            predict_args=self._affinity_args({}),
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(affinity_steering),
            affinity_mw_correction=False,
        )
        self.affinity_model.eval()
        self.affinity_model.to(self.device)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @staticmethod
    def _structure_args(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "recycling_steps": params.get("recycling_steps", 3),
            "sampling_steps": params.get("sampling_steps", 200),
            "diffusion_samples": params.get("diffusion_samples", 1),
            "max_parallel_samples": MAX_PARALLEL_SAMPLES,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }

    @staticmethod
    def _affinity_args(params: dict[str, Any]) -> dict[str, Any]:
        return {
            "recycling_steps": AFFINITY_RECYCLING_STEPS,
            "sampling_steps": params.get("sampling_steps_affinity", 200),
            "diffusion_samples": params.get("diffusion_samples_affinity", 5),
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

    def memory_info(self) -> dict[str, Any]:
        import torch

        free, total = torch.cuda.mem_get_info(self.device)
        return {
            "device_name": torch.cuda.get_device_name(self.device),
            "allocated_gb": round(torch.cuda.memory_allocated(self.device) / 2**30, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(self.device) / 2**30, 2),
            "device_used_gb": round((total - free) / 2**30, 2),
            "device_total_gb": round(total / 2**30, 2),
        }

    def _trainer(self, writer: Any, root_dir: Path) -> Any:
        from pytorch_lightning import Trainer

        return Trainer(
            default_root_dir=root_dir,
            strategy=_resident_strategy(self.device),
            callbacks=[writer],
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    # ------------------------------------------------------------------
    def run(self, job: dict[str, Any]) -> dict[str, Any]:
        """Predict one molecule. Returns ``{"status": ok|oom|error, ...}``."""
        import torch

        started = time.monotonic()
        timings: dict[str, float] = {}
        try:
            return self._run(job, timings)
        except BaseException as exc:  # noqa: BLE001
            if _is_oom(exc):
                logger.warning("Out of memory on %s: %s", job.get("yaml_path"), exc)
                return {"status": "oom", "error": str(exc)[:500], "timings": timings}
            logger.exception("Prediction failed for %s", job.get("yaml_path"))
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "timings": timings,
            }
        finally:
            timings["total_s"] = round(time.monotonic() - started, 2)
            gc.collect()
            torch.cuda.empty_cache()

    def _run(self, job: dict[str, Any], timings: dict[str, float]) -> dict[str, Any]:
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest, Record
        from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
        from boltz.main import process_input

        params = job.get("params", {})
        yaml_path = Path(job["yaml_path"])
        out_dir = Path(job["out_dir"])

        msa_dir = out_dir / "msa"
        processed_dir = out_dir / "processed"
        records_dir = processed_dir / "records"
        structure_dir = processed_dir / "structures"
        processed_msa_dir = processed_dir / "msa"
        constraints_dir = processed_dir / "constraints"
        templates_dir = processed_dir / "templates"
        mols_dir = processed_dir / "mols"
        predictions_dir = out_dir / "predictions"
        for d in (
            msa_dir, records_dir, structure_dir, processed_msa_dir,
            constraints_dir, templates_dir, mols_dir, predictions_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        # process_input reports failures by printing, not raising.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            process_input(
                path=yaml_path,
                ccd=self.ccd,
                msa_dir=msa_dir,
                mol_dir=self.mol_dir,
                boltz2=True,
                use_msa_server=bool(job.get("use_msa_server", False)),
                msa_server_url=job.get("msa_server_url", "https://api.colabfold.com"),
                msa_pairing_strategy="greedy",
                msa_server_username=None,
                msa_server_password=None,
                api_key_header=None,
                api_key_value=None,
                max_msa_seqs=MAX_MSA_SEQS,
                processed_msa_dir=processed_msa_dir,
                processed_constraints_dir=constraints_dir,
                processed_templates_dir=templates_dir,
                processed_mols_dir=mols_dir,
                structure_dir=structure_dir,
                records_dir=records_dir,
            )
        timings["preprocess_s"] = round(time.monotonic() - t0, 2)

        record_path = records_dir / f"{yaml_path.stem}.json"
        if not record_path.is_file():
            output = captured.getvalue().strip()
            logger.warning("Input processing failed for %s:\n%s", yaml_path, output[-4000:])
            lines = [ln for ln in output.splitlines() if ln.startswith("Failed to process")]
            reason = lines[-1] if lines else (output.splitlines()[-1] if output else "")
            return {
                "status": "error",
                "error": f"input processing failed: {reason}"[:1000],
                "timings": timings,
            }
        record = Record.load(record_path)
        manifest = Manifest([record])

        t0 = time.monotonic()
        writer = BoltzWriter(
            data_dir=structure_dir,
            output_dir=predictions_dir,
            output_format=params.get("output_format", "pdb"),
            boltz2=True,
            write_embeddings=False,
        )
        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=structure_dir,
            msa_dir=processed_msa_dir,
            mol_dir=self.mol_dir,
            num_workers=0,
            constraints_dir=constraints_dir,
            template_dir=templates_dir,
            extra_mols_dir=mols_dir,
            override_method=None,
        )
        self.structure_model.predict_args = self._structure_args(params)
        self._trainer(writer, out_dir).predict(
            self.structure_model, datamodule=data_module, return_predictions=False,
        )
        timings["structure_s"] = round(time.monotonic() - t0, 2)
        # Boltz2.predict_step turns an OOM into {"exception": True}, which the
        # writer only counts.
        if writer.failed:
            return {"status": "oom", "error": "out of memory during structure prediction", "timings": timings}

        if record.affinity:
            t0 = time.monotonic()
            affinity_writer = BoltzAffinityWriter(
                data_dir=structure_dir,
                output_dir=predictions_dir,
            )
            data_module = Boltz2InferenceDataModule(
                manifest=manifest,
                target_dir=predictions_dir,
                msa_dir=processed_msa_dir,
                mol_dir=self.mol_dir,
                num_workers=0,
                constraints_dir=constraints_dir,
                template_dir=templates_dir,
                extra_mols_dir=mols_dir,
                override_method="other",
                affinity=True,
            )
            self.affinity_model.predict_args = self._affinity_args(params)
            self.affinity_model.affinity_mw_correction = bool(
                params.get("affinity_mw_correction", False)
            )
            self._trainer(affinity_writer, out_dir).predict(
                self.affinity_model, datamodule=data_module, return_predictions=False,
            )
            timings["affinity_s"] = round(time.monotonic() - t0, 2)
            if affinity_writer.failed:
                return {"status": "oom", "error": "out of memory during affinity prediction", "timings": timings}

        return {"status": "ok", "timings": timings}


_STRATEGY_CLS: Any = None


def _resident_strategy(device: Any) -> Any:
    """SingleDeviceStrategy whose teardown keeps the model on the GPU.

    Lightning's default teardown moves the LightningModule back to the CPU
    after every ``trainer.predict``.
    """
    global _STRATEGY_CLS
    if _STRATEGY_CLS is None:
        from pytorch_lightning.strategies import SingleDeviceStrategy

        class ResidentSingleDeviceStrategy(SingleDeviceStrategy):
            def teardown(self) -> None:
                self.precision_plugin.teardown()
                if self.accelerator is not None:
                    self.accelerator.teardown()
                self.checkpoint_io.teardown()

        _STRATEGY_CLS = ResidentSingleDeviceStrategy
    return _STRATEGY_CLS(device=device)
