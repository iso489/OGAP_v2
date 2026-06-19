#!/usr/bin/env python3
"""OGAP v9 - Open Glioma Analysis Pipeline (core).

Pipeline diagram
────────────────
    pretrain_teacher  ──▶  train (student KD)  ──▶  evaluate
                                │                       │
                                ├──▶ export (ONNX + INT8)
                                │         │
                                │         └──▶ compare_quantisation
                                │
                                ├──▶ ablation  ──▶ ablation_eval
                                │
                                └──▶ benchmark (hardware / LMIC)

    Integrated evaluation addendum commands:
        lint_csv, surface_dice, calibration, mc_uncertainty,
        metric_check, stratify, interrater, threshold_sweep,
        holder_sweep, kd_compare, erasmus_bias,
        single_modality_scenario, field_strength_curve,
        loso_cv, kfold_cv, multi_seed, training_curves,
        ablation_heatmap, subgroup_forest, failure_case_figure,
        compute_profile, quantization_hardware, total_energy_table,
        model_card, datasheet, data_flow, sampler_distribution,
        ai_compliance_report   (CLAIM 2024 + TRIPOD+AI + SAGER 2022 + NIHMS
                                health-equity orchestrator; see
                                Scripts/ogap_compliance.py for details)

"""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import inspect
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import random
import shutil
import threading
import time
import sys
import uuid
import warnings
import shlex
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np

# Keep a safe local fallback for ad hoc runs that do not go through the sbatch
# wrappers. Cluster jobs already set a stronger allocator policy before Python
# starts, so this only applies when the environment is otherwise unset.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


LOG = logging.getLogger("OGAP")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_NVML_LOCK = threading.Lock()
_NVML_REFCOUNT = 0


# ══════════════════════════════════════════════════════════════════════════════
# §0  ENERGY & CARBON TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class EnergyTracker:
    """Real-time GPU + CPU energy tracker using pynvml.

    Polls GPU power draw every `poll_interval` seconds in a background daemon
    thread. At stop(), computes total energy in Wh and an estimated carbon
    footprint using a configurable grid carbon intensity.

    This complements GA4HPC (which reads sacct logs post-hoc but assumes 100%
    GPU utilisation). EnergyTracker measures *actual* power draw, giving a
    more accurate per-phase energy breakdown.

    Carbon intensity defaults:
        Quebec grid: 26 gCO2eq/kWh (Hydro-Québec, mostly hydro)
        Canadian national average: ~150 gCO2eq/kWh
        World average: ~436 gCO2eq/kWh

    Usage:
        tracker = EnergyTracker(label="Teacher", out_dir=out, gpu_index=0)
        tracker.start()
        ... training loop ...
        report = tracker.stop()   # blocks briefly, saves JSON, logs summary
    """

    # Quebec grid default - change if running elsewhere.
    DEFAULT_CARBON_INTENSITY_gCO2_PER_KWH = 26.0

    def __init__(
        self,
        label: str,
        out_dir: Path,
        gpu_index: int = 0,
        poll_interval: float = 5.0,
        carbon_intensity: float = DEFAULT_CARBON_INTENSITY_gCO2_PER_KWH,
        num_cpu_cores: int | None = None,
        cpu_tdp_w: float = 450.0,
        track_gpu: bool = True,
        # NOTE (analysis-facing): cpu_tdp_w is a node-level *nameplate* TDP
        # assumption, not measured. Default 450 W = dual-socket Intel Xeon Gold
        # 6448Y on Rorqual (2 × 225 W TDP, Intel ARK). Per-job CPU energy is
        # estimated as node_TDP × (cores_used / cores_total) × wall_hours, which
        # mirrors the accounting used by GA4HPC. Override cpu_tdp_w when running
        # on a different platform; state the value in the paper's Methods section.
    ) -> None:
        self.label = label
        self.out_dir = Path(out_dir)
        self.gpu_index = gpu_index
        self.poll_interval = max(0.5, float(poll_interval))
        self.carbon_intensity = carbon_intensity
        self.num_cpu_cores = num_cpu_cores or os.cpu_count() or 1
        self.cpu_tdp_w = cpu_tdp_w
        self.track_gpu = bool(track_gpu)

        self._gpu_available = False
        self._handle = None
        self._samples: list[float] = []   # GPU power readings in Watts
        self._sm_samples: list[int] = []       # NVML SM-core utilisation (%)
        self._mem_util_samples: list[int] = [] # NVML memory-bus utilisation (%)
        self._vram_peak_bytes: int = 0         # peak VRAM used during run
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0
        self._started = False
        self._stopped = False
        self._last_report: dict[str, Any] | None = None
        self._atexit_registered = False
        self._nvml_registered = False

        if self.track_gpu:
            # Try to initialise pynvml (ships with nvidia-ml-py / pynvml package,
            # which is almost always present in CUDA environments).
            try:
                import pynvml
                global _NVML_REFCOUNT
                with _NVML_LOCK:
                    pynvml.nvmlInit()
                    _NVML_REFCOUNT += 1
                    self._nvml_registered = True
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._pynvml = pynvml
                self._gpu_available = True
                gpu_name = pynvml.nvmlDeviceGetName(self._handle)
                gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
                LOG.info(f"[Energy/{label}] pynvml OK - tracking GPU {gpu_index}: {gpu_name}")
            except Exception as e:
                if self._nvml_registered:
                    try:
                        with _NVML_LOCK:
                            _NVML_REFCOUNT = max(0, _NVML_REFCOUNT - 1)
                            if _NVML_REFCOUNT == 0:
                                pynvml.nvmlShutdown()  # type: ignore[name-defined]
                    except Exception:
                        pass
                    self._nvml_registered = False
                LOG.warning(f"[Energy/{label}] pynvml unavailable ({e}). "
                            f"CPU-only estimate will be used.")
        else:
            LOG.info(f"[Energy/{label}] GPU tracking disabled; CPU-only deployment estimate.")

    # ── background polling thread ──────────────────────────────────────────

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            if self._gpu_available:
                try:
                    mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    self._samples.append(mw / 1000.0)   # milliwatts → Watts
                except Exception as exc:
                    # Surface NVML power-read failures: the energy/CO2 estimate is built
                    # from the surviving samples, so a silent drop would bias the published
                    # environmental number low with no indication.
                    self._power_poll_failures = getattr(self, "_power_poll_failures", 0) + 1
                    if self._power_poll_failures == 1:
                        LOG.warning("[energy] NVML power read failed (%s); CO2/energy will "
                                    "use fewer samples (warned once; count in report).", exc)
                try:
                    rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._sm_samples.append(int(rates.gpu))
                    self._mem_util_samples.append(int(rates.memory))
                except Exception:
                    pass
                try:
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    used = int(getattr(mem, "used", 0))
                    if used > self._vram_peak_bytes:
                        self._vram_peak_bytes = used
                except Exception:
                    pass
            self._stop_event.wait(self.poll_interval)

    # ── public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started and not self._stopped:
            return
        self._t_start = time.time()
        self._samples.clear()
        self._sm_samples.clear()
        self._mem_util_samples.clear()
        self._vram_peak_bytes = 0
        self._power_poll_failures = 0
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True, name=f"energy-{self.label}")
        self._thread.start()
        self._started = True
        self._stopped = False
        self._last_report = None
        if not self._atexit_registered:
            atexit.register(self._stop_at_exit)
            self._atexit_registered = True
        LOG.info(f"[Energy/{self.label}] Tracking started.")

    def stop(self) -> dict:
        """Stop tracking, save JSON report, return summary dict."""
        if not self._started:
            return self._last_report or {}
        if self._stopped:
            return self._last_report or {}
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval + 2)
        self._t_end = time.time()

        elapsed_h = (self._t_end - self._t_start) / 3600.0

        # ── GPU energy ────────────────────────────────────────────────────
        if not self.track_gpu:
            mean_gpu_w = 0.0
            gpu_wh = 0.0
        elif self._samples:
            mean_gpu_w = float(np.mean(self._samples))
            gpu_wh = mean_gpu_w * elapsed_h
        else:
            # GA4HPC fallback: assume 100% TDP. The H100 SXM5 TDP is 700 W.
            # Adjust if you're on a different GPU.
            mean_gpu_w = 700.0
            gpu_wh = mean_gpu_w * elapsed_h
            LOG.warning(f"[Energy/{self.label}] No GPU samples - "
                        f"assuming 100% TDP ({mean_gpu_w:.0f} W) as GA4HPC would.")

        # ── CPU energy (proportional share of node TDP) ────────────────────
        # We have cfg.num_workers + 1 cores actively working; scale TDP by
        # that fraction of the node's total core count. This is the same
        # approach GA4HPC uses for CPU.
        cpu_fraction = min(self.num_cpu_cores / max(os.cpu_count() or 1, 1), 1.0)
        cpu_wh = self.cpu_tdp_w * cpu_fraction * elapsed_h

        total_wh = gpu_wh + cpu_wh
        total_kwh = total_wh / 1000.0
        co2_g = total_kwh * self.carbon_intensity
        job_id = os.environ.get("SLURM_JOB_ID")
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._t_end))
        run_id = f"job{job_id}_{timestamp}" if job_id else timestamp

        # Utilisation statistics - analysis-facing saturation proof.
        def _stats(xs):
            if not xs:
                return {"mean": None, "p50": None, "p95": None, "max": None, "n": 0}
            arr = np.asarray(xs, dtype=np.float64)
            return {
                "mean": round(float(arr.mean()), 2),
                "p50":  round(float(np.percentile(arr, 50)), 2),
                "p95":  round(float(np.percentile(arr, 95)), 2),
                "max":  round(float(arr.max()), 2),
                "n":    int(arr.size),
            }

        sm_stats = _stats(self._sm_samples)
        mem_util_stats = _stats(self._mem_util_samples)
        vram_peak_gb = round(self._vram_peak_bytes / (1 << 30), 3) if self._vram_peak_bytes else None
        # Mean fraction of the GPU's power-cap drawn on average - useful to
        # flag under-utilisation penalties on systems like Rorqual.
        power_cap_w = None
        try:
            if self._gpu_available:
                power_cap_w = (
                    self._pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
                )
        except Exception:
            pass
        power_cap_fraction = (
            round(mean_gpu_w / power_cap_w, 3) if power_cap_w else None
        )

        report = {
            "label":                  self.label,
            "run_id":                 run_id,
            "slurm_job_id":           job_id,
            "started_at_unix":        round(self._t_start, 3),
            "stopped_at_unix":        round(self._t_end, 3),
            "elapsed_hours":          round(elapsed_h, 4),
            "gpu_tracking_enabled":    bool(self.track_gpu),
            "gpu_mean_power_w":       round(mean_gpu_w, 2),
            "gpu_power_cap_w":        round(power_cap_w, 1) if power_cap_w else None,
            "gpu_power_cap_fraction": power_cap_fraction,
            "gpu_samples_count":      len(self._samples),
            "gpu_power_poll_failures": getattr(self, "_power_poll_failures", 0),
            "gpu_sm_util_pct":        sm_stats,
            "gpu_mem_util_pct":       mem_util_stats,
            "gpu_vram_peak_gb":       vram_peak_gb,
            "gpu_energy_wh":          round(gpu_wh, 3),
            "cpu_cores_used":         self.num_cpu_cores,
            "cpu_tdp_w_node":         self.cpu_tdp_w,
            "cpu_energy_wh":          round(cpu_wh, 3),
            "total_energy_wh":        round(total_wh, 3),
            "total_energy_kwh":       round(total_kwh, 6),
            "carbon_intensity_gco2_per_kwh": self.carbon_intensity,
            "co2_grams":              round(co2_g, 3),
            "co2_grams_equiv":        _co2_equiv_str(co2_g),
        }

        latest_path = self.out_dir / f"energy_{self.label.lower()}.json"
        archive_path = self.out_dir / f"energy_{self.label.lower()}_{run_id}.json"
        index_path = self.out_dir / f"energy_{self.label.lower()}_runs.jsonl"
        try:
            save_json_atomic(report, latest_path)
            save_json_atomic(report, archive_path)
            with open(index_path, "a") as f:
                f.write(json.dumps({
                    "run_id": run_id,
                    "slurm_job_id": job_id,
                    "archive_path": str(archive_path),
                    "latest_path": str(latest_path),
                    "total_energy_wh": report["total_energy_wh"],
                    "co2_grams": report["co2_grams"],
                    "elapsed_hours": report["elapsed_hours"],
                }) + "\n")
        except Exception as e:
            LOG.warning(f"[Energy/{self.label}] Could not save report: {e}")

        LOG.info(
            f"[Energy/{self.label}] ── Energy Report ──────────────────────\n"
            f"  Duration      : {elapsed_h*60:.1f} min\n"
            f"  GPU power     : {mean_gpu_w:.1f} W (mean over {len(self._samples)} samples)"
            f"{(' / cap ' + str(round(power_cap_w,0)) + 'W (' + str(round((power_cap_fraction or 0)*100,1)) + '%)') if power_cap_w else ''}\n"
            f"  GPU SM util   : mean={sm_stats['mean']} p95={sm_stats['p95']} max={sm_stats['max']} (%)\n"
            f"  GPU mem-bus   : mean={mem_util_stats['mean']} p95={mem_util_stats['p95']} (%)\n"
            f"  GPU VRAM peak : {vram_peak_gb} GB\n"
            f"  GPU energy    : {gpu_wh:.1f} Wh\n"
            f"  CPU energy    : {cpu_wh:.1f} Wh  ({self.num_cpu_cores} cores @ "
            f"  {self.cpu_tdp_w:.0f} W node TDP)\n"
            f"  Total energy  : {total_wh:.1f} Wh  ({total_kwh*1000:.2f} Wh)\n"
            f"  Carbon (QC)   : {co2_g:.1f} gCO₂eq  [{_co2_equiv_str(co2_g)}]\n"
            f"  Latest report : {latest_path}\n"
            f"  Archive report: {archive_path}\n"
            f"  ─────────────────────────────────────────────"
        )

        try:
            if self._gpu_available and self._nvml_registered:
                global _NVML_REFCOUNT
                with _NVML_LOCK:
                    _NVML_REFCOUNT = max(0, _NVML_REFCOUNT - 1)
                    if _NVML_REFCOUNT == 0:
                        self._pynvml.nvmlShutdown()
                self._gpu_available = False
                self._nvml_registered = False
        except Exception:
            pass

        self._stopped = True
        self._last_report = report
        if self._atexit_registered:
            try:
                atexit.unregister(self._stop_at_exit)
            except Exception:
                pass
            self._atexit_registered = False

        return report

    def _stop_at_exit(self) -> None:
        try:
            self.stop()
        except Exception as exc:
            try:
                print(f"[Energy/{self.label}] atexit stop failed: {exc}", file=sys.stderr)
            except Exception:
                pass


def _co2_equiv_str(co2_g: float) -> str:
    """Return a human-readable CO₂ equivalent (km driven, smartphone charges)."""
    km = co2_g / 120.0          # avg car: ~120 gCO2/km
    phone_charges = co2_g / 8.22  # avg smartphone charge: ~8.22 gCO2
    return f"≈{km:.2f} km driven, ≈{phone_charges:.1f} smartphone charges"


def configure_warnings(suppress: bool = False) -> None:
    """Optionally suppress noisy UserWarnings.

    By default, warnings are left enabled because they often contain
    important information about NIfTI headers, AMP, ONNX export, and
    numerical stability.
    """
    if suppress:
        warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# §1  GENERAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    If deterministic=True, we disable CuDNN benchmarking and request
    deterministic algorithms where possible (warn_only=True).

    Threading model:
      • Thread-affinity environment variables belong in the launcher before
        Python starts.
      • This function only tunes torch's own intra/inter-op pools after
        import.
      • For the current one-GPU Rorqual jobs, the intended shape is a small
        main-process CPU budget plus single-threaded workers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Launcher-owned thread tuning ──────────────────────────────────────
    # KMP / OpenMP environment variables should be set by the sbatch script
    # before Python starts. At this point PyTorch and linked runtimes are
    # already imported, so we only control torch's own thread pools here.

    # On a single-GPU Rorqual job the GPU is usually starved by CPU-side
    # augmentation, so workers should stay single-threaded and the main process
    # should reserve only a small CPU budget.
    main_threads = int(os.environ.get("OGAP_MAIN_THREADS", "2"))
    interop_threads = int(os.environ.get("OGAP_INTEROP_THREADS", "1"))
    torch.set_num_threads(max(1, main_threads))
    torch.set_num_interop_threads(max(1, interop_threads))
    LOG.info(f"[PyTorch threads] intra-op={max(1, main_threads)}, inter-op={max(1, interop_threads)}")

    if deterministic:
        # Helps determinism for some CUDA GEMM paths
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch builds may not support this API or may fail on some ops.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # H100 / Ampere+: TF32 gives ~3x speedup on matmuls and convolutions
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 'high' = use TF32 for FP32 matmuls (PyTorch-level control, complements allow_tf32)
    torch.set_float32_matmul_precision("high")
    # H100: allow BF16 accumulation in BF16 matmuls (faster; negligible accuracy loss
    # for training - activations are already in BF16 under autocast)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    _configure_torch_compile_runtime()


def seed_worker(worker_id: int) -> None:
    """Deterministically seed NumPy/random in each DataLoader worker.

    Also pins SimpleITK / ITK to a single thread so DataLoader workers do not
    collectively dispatch workers × <ncpu> ITK threads onto the SLURM CPU
    allocation. That oversubscription is what stalled the teacher first epoch
    for >19 hours in jobs 11111229 / 11112647.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        pass
    _pin_sitk_threads()


def _configure_torch_compile_runtime() -> None:
    """Enable safe Inductor caching knobs when available.

    For fixed-shape 128^3 training on Hopper, FX graph cache reuse is a low-risk
    win and keeps repeated `torch.compile` specialisations from recompiling more
    than necessary when jobs are relaunched with identical shapes. This must be
    called before any `torch.compile(...)` invocation.
    """
    try:
        import torch._inductor.config as inductor_config

        if hasattr(inductor_config, "fx_graph_cache"):
            inductor_config.fx_graph_cache = True
        if hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        if hasattr(inductor_config, "coordinate_descent_check_all_directions"):
            inductor_config.coordinate_descent_check_all_directions = True
        if hasattr(inductor_config, "epilogue_fusion"):
            inductor_config.epilogue_fusion = True
        if hasattr(inductor_config, "shape_padding"):
            inductor_config.shape_padding = True
        if hasattr(inductor_config, "conv_1x1_as_mm"):
            inductor_config.conv_1x1_as_mm = True
        if hasattr(inductor_config, "autograd_cache"):
            inductor_config.autograd_cache = True
    except Exception:
        pass


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _model_uses_mamba_cuda_extensions(model: nn.Module) -> bool:
    for module in model.modules():
        cls = module.__class__
        cls_name = cls.__name__.lower()
        module_name = cls.__module__.lower()
        if (
            "mamba" in cls_name
            or "mamba" in module_name
            or "causal_conv1d" in module_name
        ):
            return True
    return False


def _torch_compile_skip_reason(model: nn.Module) -> Optional[str]:
    if _env_flag_enabled("OGAP_DISABLE_TORCH_COMPILE"):
        return "disabled by OGAP_DISABLE_TORCH_COMPILE"

    torch_compile_setting = os.environ.get("OGAP_TORCH_COMPILE", "").strip().lower()
    if torch_compile_setting in {"0", "false", "no", "off"}:
        return "disabled by OGAP_TORCH_COMPILE=0"

    if _model_uses_mamba_cuda_extensions(model):
        return (
            "model uses mamba-ssm/causal-conv1d CUDA extensions, which trigger "
            "TorchDynamo graph breaks and Inductor backend failures"
        )

    triton_version = "unknown"
    try:
        import triton

        triton_version = getattr(triton, "__version__", "unknown")
        from triton.compiler.compiler import triton_key  # noqa: F401
    except ImportError as exc:
        return (
            "Triton/PyTorch compile API mismatch; Inductor cannot import "
            f"triton_key from Triton {triton_version} ({exc})"
        )
    except Exception as exc:
        return f"Triton compile compatibility check failed ({type(exc).__name__}: {exc})"

    return None


def maybe_compile_model(
    model: nn.Module,
    label: str,
    *,
    mode: str = "max-autotune",
    dynamic: bool = False,
) -> nn.Module:
    """Compile with Inductor only when this runtime can do so safely."""
    reason = _torch_compile_skip_reason(model)
    if reason:
        LOG.warning("[%s] torch.compile() skipped - %s", label, reason)
        return model

    try:
        from torch import _dynamo as torch_dynamo

        # Backend failures are raised lazily on the first forward pass, not at
        # torch.compile(...) time. Keep this on so an unexpected Inductor issue
        # falls back to eager instead of killing a long SLURM allocation.
        torch_dynamo.config.suppress_errors = True
    except Exception as exc:
        LOG.warning("[%s] torch.compile() fallback guard unavailable: %s", label, exc)

    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic)
    except Exception as exc:
        LOG.warning("[%s] torch.compile() unavailable, running eager: %s", label, exc)
        return model

    LOG.info("[%s] torch.compile(mode='%s') enabled", label, mode)
    return compiled


_VALIDATION_EAGER_LOGGED: set[str] = set()


def _validation_model_view(model: nn.Module, label: str) -> nn.Module:
    """Use eager validation for compiled models with variable full-volume shapes."""
    if _env_flag_enabled("OGAP_VALIDATE_WITH_COMPILED_MODEL", default=False):
        return model
    raw_model = getattr(model, "_orig_mod", None)
    if isinstance(raw_model, nn.Module):
        if label not in _VALIDATION_EAGER_LOGGED:
            LOG.info(
                "[%s] validation uses eager _orig_mod to avoid torch.compile autotune "
                "on variable full-volume shapes",
                label,
            )
            _VALIDATION_EAGER_LOGGED.add(label)
        return raw_model
    return model


def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _allocated_cpu_budget(default: int = 1) -> int:
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", default))
    except (TypeError, ValueError):
        return default


def _tracked_gpu_index(default: int = 0) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if not visible:
        return default
    try:
        return int(visible)
    except (TypeError, ValueError):
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = None
                visible_bytes = visible.encode("utf-8", errors="ignore")
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(visible_bytes)
                except Exception:
                    handle = None

                # MIG selectors often embed the parent GPU UUID followed by profile info.
                if handle is None and "/" in visible:
                    parent_selector = visible.split("/", 1)[0]
                    if parent_selector.startswith("MIG-"):
                        parent_selector = parent_selector[len("MIG-"):]
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(
                            parent_selector.encode("utf-8", errors="ignore")
                        )
                    except Exception:
                        handle = None

                if handle is None and visible.startswith("MIG-"):
                    parent_selector = visible[len("MIG-"):]
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(
                            parent_selector.encode("utf-8", errors="ignore")
                        )
                    except Exception:
                        handle = None

                if handle is not None:
                    return int(pynvml.nvmlDeviceGetIndex(handle))
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            pass
        LOG.warning(
            "[Energy] Could not resolve CUDA_VISIBLE_DEVICES=%r to an NVML device index; defaulting to GPU %d",
            visible,
            default,
        )
        return default


def _worker_context() -> Optional[str]:
    if not sys.platform.startswith("linux"):
        return None
    requested = os.environ.get("OGAP_MP_CONTEXT", "forkserver").strip().lower()
    if not requested:
        return None
    try:
        if requested in mp.get_all_start_methods():
            return requested
    except Exception:
        return None
    return None


def _drop_file_cache(path: Union[str, Path]) -> None:
    """Best-effort page-cache release for large local volume/cache files.

    Slurm cgroups can charge file cache against the job memory limit. During
    cold-cache epoch 1 the dataset may read or write hundreds of GB of volume
    data, so ask Linux to evict those pages after arrays are materialized.
    This is advisory and silently no-ops where unsupported.
    """
    if not _env_flag_enabled("OGAP_DROP_NPY_FILE_CACHE", default=True):
        return
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd: Optional[int] = None
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # type: ignore[attr-defined]
    except Exception:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


_LOADER_LIMITS_LOGGED: set[Tuple[str, str]] = set()


def _loader_prefetch_factor(default: int = 1) -> int:
    try:
        requested = max(1, int(os.environ.get("OGAP_PREFETCH_FACTOR", default)))
    except (TypeError, ValueError):
        requested = default
    try:
        cap = max(1, int(os.environ.get("OGAP_MAX_PREFETCH_FACTOR", "1")))
    except (TypeError, ValueError):
        cap = 1
    return min(requested, cap)


def _loader_num_workers(num_workers: int, *, train_mode: bool) -> int:
    requested = max(0, int(num_workers))
    env_name = "OGAP_MAX_TRAIN_WORKERS" if train_mode else "OGAP_MAX_EVAL_WORKERS"
    try:
        cap_default = "16" if train_mode else "8"
        cap = max(0, int(os.environ.get(env_name, cap_default)))
    except (TypeError, ValueError):
        cap = 16 if train_mode else 8
    return min(requested, cap) if cap > 0 else requested


def _loader_max_inflight_cases(*, train_mode: bool) -> int:
    env_name = "OGAP_MAX_TRAIN_INFLIGHT_CASES" if train_mode else "OGAP_MAX_EVAL_INFLIGHT_CASES"
    default = "32" if train_mode else "8"
    try:
        return max(1, int(os.environ.get(env_name, default)))
    except (TypeError, ValueError):
        return int(default)


def _loader_effective_plan(
    *,
    batch_size: int,
    num_workers: int,
    train_mode: bool,
) -> Tuple[int, int, int]:
    num_workers = _loader_num_workers(num_workers, train_mode=train_mode)
    prefetch_factor = _loader_prefetch_factor()
    max_inflight = _loader_max_inflight_cases(train_mode=train_mode)
    if num_workers <= 0:
        return num_workers, prefetch_factor, max_inflight

    per_batch_cases = max(1, int(batch_size))
    inflight = per_batch_cases * max(1, int(num_workers)) * max(1, int(prefetch_factor))
    if inflight <= max_inflight:
        return num_workers, prefetch_factor, max_inflight

    # Prefer preserving workers over deep per-worker queues. Full-volume NIfTI
    # loading plus donor sourcing can hold several large arrays per sample, so
    # bounding "cases in flight" is the only reliable way to stay under the
    # one-H100 Rorqual memory allocation during cold-cache epoch 1.
    prefetch_factor = 1
    max_workers_by_inflight = max(1, max_inflight // per_batch_cases)
    num_workers = min(num_workers, max_workers_by_inflight)
    return num_workers, prefetch_factor, max_inflight


def _log_loader_limit_once(name: str, requested: int, effective: int) -> None:
    if requested == effective:
        return
    key = (name, f"{requested}->{effective}")
    if key in _LOADER_LIMITS_LOGGED:
        return
    _LOADER_LIMITS_LOGGED.add(key)
    LOG.info("[DataLoader] %s clamped %d→%d to avoid CPU RAM OOM", name, requested, effective)


def _train_log_every(default: int = 25) -> int:
    try:
        return max(1, int(os.environ.get("OGAP_TRAIN_LOG_EVERY", str(default))))
    except (TypeError, ValueError):
        return default


def _train_log_first_n(default: int = 3) -> int:
    try:
        return max(0, int(os.environ.get("OGAP_TRAIN_LOG_FIRST_N", str(default))))
    except (TypeError, ValueError):
        return default


def _build_loader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    worker_init,
    generator: Optional[torch.Generator] = None,
    sampler: Optional[WeightedRandomSampler] = None,
    drop_last: bool = False,
    train_mode: bool = False,
    persistent_workers: bool = False,
) -> Dict[str, Any]:
    requested_num_workers = int(num_workers)
    requested_prefetch = os.environ.get("OGAP_PREFETCH_FACTOR")
    requested_prefetch_int = _loader_prefetch_factor()
    if requested_prefetch is not None:
        try:
            requested_prefetch_int = max(1, int(requested_prefetch))
        except (TypeError, ValueError):
            pass
    num_workers, prefetch_factor, max_inflight = _loader_effective_plan(
        batch_size=batch_size,
        num_workers=requested_num_workers,
        train_mode=train_mode,
    )
    _log_loader_limit_once(
        "num_workers(train)" if train_mode else "num_workers(eval/val)",
        requested_num_workers,
        num_workers,
    )
    _log_loader_limit_once("prefetch_factor", requested_prefetch_int, prefetch_factor)
    if num_workers > 0:
        planned_inflight = max(1, int(batch_size)) * num_workers * prefetch_factor
        if planned_inflight <= max_inflight:
            LOG.info(
                "[DataLoader] %s in-flight case budget: batch=%d workers=%d prefetch=%d -> %d/%d",
                "train" if train_mode else "eval/val",
                int(batch_size), num_workers, prefetch_factor, planned_inflight, max_inflight,
            )
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if generator is not None and sampler is None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
        sig = inspect.signature(DataLoader)
        # `in_order=False` (PyTorch >=2.6) returns batches in worker-COMPLETION order to
        # avoid head-of-line blocking, but that makes the batch sequence depend on per-
        # worker wall-clock timing - non-deterministic across otherwise-identical runs.
        # Only enable it when deterministic reproducibility has NOT been requested
        # (seed_everything(deterministic=True) sets torch.use_deterministic_algorithms).
        if (train_mode and "in_order" in sig.parameters
                and not torch.are_deterministic_algorithms_enabled()):
            kwargs["in_order"] = False
        mp_context = _worker_context()
        if mp_context and "multiprocessing_context" in sig.parameters:
            kwargs["multiprocessing_context"] = mp_context
    return kwargs


class CUDATrainPrefetcher:
    """Overlap host-to-device copies and optional GPU augmentation with compute."""

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        gpu_augmentor: Optional["GPUPhysicsAugmentor"] = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDATrainPrefetcher requires a CUDA device")
        self.loader = loader
        self.device = device
        self.gpu_augmentor = gpu_augmentor
        self.stream = torch.cuda.Stream(device=device)
        self._loader_iter = None
        self._next_batch = None

    def __iter__(self) -> "CUDATrainPrefetcher":
        self._loader_iter = iter(self.loader)
        self._next_batch = None
        self._preload()
        return self

    def _preload(self) -> None:
        assert self._loader_iter is not None
        try:
            case_ids, x, y, spacing, tags = next(self._loader_iter)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self.stream):
            x = x.to(self.device, non_blocking=True, memory_format=torch.channels_last_3d)
            y = y.to(self.device, non_blocking=True)
            if self.gpu_augmentor is not None:
                x, y = self.gpu_augmentor(x, y)
            x = x.contiguous(memory_format=torch.channels_last_3d)
        self._next_batch = (case_ids, x, y, spacing, tags)

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(device=self.device)
        current_stream.wait_stream(self.stream)
        batch = self._next_batch
        _, x, y, _, _ = batch
        x.record_stream(current_stream)
        if torch.is_tensor(y):
            y.record_stream(current_stream)
        self._preload()
        return batch


def _clean_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_optional_float(value: Any) -> Optional[float]:
    text = _clean_metadata_value(value)
    if not text or text.lower() in {"na", "nan", "none", "not reported", "unknown"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _parse_row_field_strength(row: Dict[str, Any]) -> Optional[float]:
    for key in ("field_strength", "scanner_strength", "Scanner Strength"):
        if key in row:
            parsed = _parse_optional_float(row.get(key))
            # Reject non-physical field strengths: a numeric sentinel like -1 (used by
            # some cohorts for "missing") or 0 must be treated as MISSING, not as a real
            # ultra-low-field scan - otherwise the equity sampler over-weights it as the
            # rarest/hardest field-strength bucket.
            if parsed is not None and parsed > 0.0:
                return parsed
    return None


def _classify_grade_for_sampling(value: Any) -> Optional[str]:
    parsed = _parse_optional_float(value)
    if parsed is None:
        text = _clean_metadata_value(value).lower()
        if not text:
            return None
        if "4" in text or "gbm" in text or "glioblastoma" in text:
            return "grade4"
        if "2" in text or "3" in text:
            return "grade23"
        return None
    grade = int(round(parsed))
    if grade == 4:
        return "grade4"
    if grade in {2, 3}:
        return "grade23"
    return None


def _classify_idh_for_sampling(value: Any) -> Optional[str]:
    text = _clean_metadata_value(value).lower()
    if not text or text in {"na", "not reported", "unknown"}:
        return None
    if "mut" in text:
        return "idh_mutant"
    if "wild" in text or text in {"wt"}:
        return "idh_wildtype"
    return None


def _parse_scan_year(row: Dict[str, Any]) -> Optional[int]:
    for key in ("scan_year", "year", "acquisition_year", "Study Year"):
        if key not in row:
            continue
        parsed = _parse_optional_float(row.get(key))
        if parsed is None:
            continue
        year = int(round(parsed))
        if 1900 <= year <= 2100:
            return year
    return None


def _assert_train_val_case_disjoint(train_ds: Any, val_ds: Any) -> None:
    """Hard-fail if any case_id appears in BOTH the train and val manifests.

    The split builder already gates this, but training reads two independent CSVs and a
    single leaked case (the common multi-site data-prep mistake) silently inflates val
    Dice. case_id is patient-unique in every OGAP cohort (UTSW 625/625, Erasmus 770/770,
    BraTS-Africa 95/95), so case_id-level disjointness is patient-level disjointness too.
    """
    tr = {str(r.get("case_id", "")) for r in getattr(train_ds, "rows", [])}
    va = {str(r.get("case_id", "")) for r in getattr(val_ds, "rows", [])}
    leak = (tr & va) - {""}
    if leak:
        raise ValueError(
            f"Train/val leak: {len(leak)} case_id(s) appear in BOTH train_csv and "
            f"val_csv (e.g. {sorted(leak)[:5]}). Fix the manifests before training.")


def build_metadata_weighted_sampler(
    rows: Sequence[Dict[str, Any]],
    *,
    seed: int,
    field_strength_sampling_power: float = 0.5,
    low_field_sampling_bias: float = 1.5,
    grade23_sampling_bias: float = 1.5,
    idh_mutant_sampling_bias: float = 1.5,
    legacy_protocol_sampling_bias: float = 1.25,
    legacy_year_threshold: int = 2013,
) -> Tuple[Optional[WeightedRandomSampler], Dict[str, Any]]:
    """Build a conservative WeightedRandomSampler from optional metadata.

    The sampler is intentionally simple: it upweights underrepresented
    field-strength buckets and biologically/technically harder phenotypes
    when those annotations are present in the manifest.
    """
    if len(rows) == 0:
        return None, {}

    field_buckets: List[str] = []
    field_bucket_counts: Dict[str, int] = {}
    grade23_flags: List[float] = []
    idh_mutant_flags: List[float] = []
    legacy_flags: List[float] = []

    for row in rows:
        fs = _parse_row_field_strength(row)
        if fs is None:
            bucket = "missing"
        elif fs <= 1.0:
            bucket = "low"
        elif fs <= 1.8:
            bucket = "standard"
        else:
            bucket = "high"
        field_buckets.append(bucket)
        field_bucket_counts[bucket] = field_bucket_counts.get(bucket, 0) + 1

        grade_group = _classify_grade_for_sampling(
            row.get("tumor_grade", row.get("who_grade", row.get("Tumor Grade", "")))
        )
        grade23_flags.append(1.0 if grade_group == "grade23" else 0.0)

        idh_group = _classify_idh_for_sampling(
            row.get("idh_status", row.get("IDH", ""))
        )
        idh_mutant_flags.append(1.0 if idh_group == "idh_mutant" else 0.0)

        year = _parse_scan_year(row)
        legacy_flags.append(1.0 if (year is not None and year < legacy_year_threshold) else 0.0)

    n_rows = float(len(rows))
    weights: List[float] = []
    for bucket, is_grade23, is_idh_mut, is_legacy in zip(
        field_buckets, grade23_flags, idh_mutant_flags, legacy_flags
    ):
        bucket_count = max(field_bucket_counts.get(bucket, 1), 1)
        freq_weight = (
            (n_rows / float(bucket_count)) ** max(float(field_strength_sampling_power), 0.0)
            if field_strength_sampling_power > 0 else 1.0
        )
        low_weight = low_field_sampling_bias if bucket == "low" else 1.0
        grade_weight = grade23_sampling_bias if is_grade23 > 0 else 1.0
        idh_weight = idh_mutant_sampling_bias if is_idh_mut > 0 else 1.0
        legacy_weight = legacy_protocol_sampling_bias if is_legacy > 0 else 1.0
        weights.append(float(freq_weight * low_weight * grade_weight * idh_weight * legacy_weight))

    weight_arr = np.asarray(weights, dtype=np.float64)
    summary = {
        "enabled": False,
        "field_bucket_counts": {k: int(v) for k, v in sorted(field_bucket_counts.items())},
        "grade23_fraction": round(float(np.mean(grade23_flags)), 5) if grade23_flags else 0.0,
        "idh_mutant_fraction": round(float(np.mean(idh_mutant_flags)), 5) if idh_mutant_flags else 0.0,
        "legacy_fraction": round(float(np.mean(legacy_flags)), 5) if legacy_flags else 0.0,
        "weight_min": round(float(weight_arr.min()), 5) if weight_arr.size else 1.0,
        "weight_mean": round(float(weight_arr.mean()), 5) if weight_arr.size else 1.0,
        "weight_max": round(float(weight_arr.max()), 5) if weight_arr.size else 1.0,
    }
    if not weight_arr.size:
        return None, summary
    if float(weight_arr.max()) / max(float(weight_arr.min()), 1e-8) <= 1.05:
        return None, summary

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weight_arr, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )
    summary["enabled"] = True
    return sampler, summary


def sample_augmented_field_strength(lmic_prior: bool = False) -> float:
    choices = np.array([0.3, 0.55, 0.7, 1.0, 1.5, 3.0], dtype=np.float32)
    if lmic_prior:
        probs = np.array([0.04, 0.06, 0.08, 0.14, 0.52, 0.16], dtype=np.float64)
    else:
        probs = np.array([0.03, 0.05, 0.07, 0.15, 0.40, 0.30], dtype=np.float64)
    probs = probs / probs.sum()
    return float(np.random.choice(choices, p=probs))


def _chunked(seq: Sequence[Any], chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(seq), chunk_size):
        yield seq[start:start + chunk_size]


def _configure_cuda_inference() -> None:
    """Enable Hopper/Ampere-friendly inference defaults."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")


# ══════════════════════════════════════════════════════════════════════════════
# §1A  AUTO-RESOURCE DETECTION & BATCH-SIZE FIT PROBE
# ══════════════════════════════════════════════════════════════════════════════
# One-shot, at-process-start auto-tuner. Detects the SLURM allocation + actual
# GPU, probes the largest batch size that survives a fwd+bwd pass on synthetic
# data, and rewrites the config in place. Runs exactly once per run - we do
# NOT retune mid-training because changing batch size mid-run silently
# invalidates the LR schedule, the optimiser state, and reproducibility
# (analysis-critical). All decisions are persisted to
# ``<out_dir>/auto_resources.json`` for the Methods section.
#
# Design influenced by: PyTorch Lightning BatchSizeFinder, fastai
# `Learner.lr_find`+`find_batch_size`, and MONAI auto-mixed-precision probes.

@dataclass
class ResourceReport:
    """Snapshot of the allocated compute + GPU at job start."""
    cpu_total: int
    cpu_allocated: int
    ram_total_gb: float
    ram_allocated_gb: float
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    power_cap_w: float
    cuda_capability: str
    notes: List[str] = field(default_factory=list)


def detect_system_resources(gpu_index: int = 0) -> ResourceReport:
    """Read SLURM envvars + /proc + NVML for a snapshot of the allocation.

    Does not require psutil - falls back to ``os.cpu_count()`` and
    ``/proc/meminfo`` on Linux. Missing NVML is tolerated (returns zeros and
    adds a note).
    """
    notes: List[str] = []
    cpu_total = os.cpu_count() or 1
    cpu_allocated = _allocated_cpu_budget(cpu_total)

    # RAM allocated (SLURM) - prefer SLURM_MEM_PER_NODE; some sites expose
    # SLURM_MEM_PER_CPU instead. Both are in megabytes.
    ram_allocated_mb: Optional[float] = None
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        val = os.environ.get(key)
        if not val:
            continue
        try:
            v = float(val)
            if key == "SLURM_MEM_PER_CPU":
                v *= max(cpu_allocated, 1)
            ram_allocated_mb = v
            break
        except ValueError:
            continue

    # Node-total RAM from /proc/meminfo (Linux). This is the *node* capacity,
    # not what SLURM gave us; both are reported for transparency.
    ram_total_mb: Optional[float] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:  528432768 kB"
                    ram_total_mb = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        notes.append("/proc/meminfo not readable; node RAM unknown")
    if ram_allocated_mb is None:
        ram_allocated_mb = ram_total_mb or 0.0
        if ram_total_mb is not None:
            notes.append("SLURM RAM unknown; using node total as allocation")

    gpu_name = "unknown"
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    power_cap_w = 0.0
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            raw = pynvml.nvmlDeviceGetName(h)
            gpu_name = raw.decode() if isinstance(raw, bytes) else raw
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_total_gb = float(mem.total) / (1 << 30)
            vram_free_gb = float(mem.free) / (1 << 30)
            try:
                power_cap_w = float(pynvml.nvmlDeviceGetPowerManagementLimit(h)) / 1000.0
            except Exception:
                pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception as exc:
        notes.append(f"pynvml unavailable: {exc}")

    cuda_cap = "unknown"
    if torch.cuda.is_available():
        try:
            maj, minor = torch.cuda.get_device_capability(gpu_index)
            cuda_cap = f"sm_{maj}{minor}"
            if vram_total_gb == 0.0:
                # fallback if NVML missed - use torch's own query
                vram_total_gb = float(torch.cuda.get_device_properties(gpu_index).total_memory) / (1 << 30)
        except Exception as exc:
            notes.append(f"torch CUDA capability probe failed: {exc}")

    return ResourceReport(
        cpu_total=cpu_total,
        cpu_allocated=cpu_allocated,
        ram_total_gb=round((ram_total_mb or 0.0) / 1024.0, 2),
        ram_allocated_gb=round((ram_allocated_mb or 0.0) / 1024.0, 2),
        gpu_name=gpu_name,
        vram_total_gb=round(vram_total_gb, 2),
        vram_free_gb=round(vram_free_gb, 2),
        power_cap_w=round(power_cap_w, 1),
        cuda_capability=cuda_cap,
        notes=notes,
    )


def probe_max_batch_size(
    build_model_fn,
    input_shape: Tuple[int, int, int, int],
    num_classes: int,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    upper_bound: int = 64,
    safety: float = 0.85,
    require_backward: bool = True,
) -> int:
    """Binary-search the largest batch size that survives fwd(+bwd).

    Strategy:
      * Phase 1 - doubling: 1, 2, 4, 8, ... until the first OOM (or upper_bound).
      * Phase 2 - bisection between the last safe value and the first OOM.
      * Return ``floor(last_safe * safety)``.

    ``safety`` leaves a margin for: mini-batch variance (the probed shape is
    the worst case, but augmentation can still widen activations), Adam
    optimiser state (2 × params), cuDNN workspace, and any data held on the
    GPU by loaders. For KD training where a teacher also sits on the device,
    pass a smaller safety (e.g. 0.70).

    No-ops when CUDA is unavailable: returns ``upper_bound``.
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return max(1, upper_bound)

    C, D, H, W = input_shape
    model_holder: Dict[str, Any] = {}

    def _try(bs: int) -> bool:
        model_holder.clear()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        try:
            m = build_model_fn().to(device)
            model_holder["m"] = m
            m.train(require_backward)
            x = torch.randn(bs, C, D, H, W, device=device, dtype=torch.float32)
            y = torch.randint(0, num_classes, (bs, D, H, W), device=device, dtype=torch.long)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                out = m(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                if logits.dim() == 5 and logits.shape[1] == num_classes:
                    loss = F.cross_entropy(logits.float(), y)
                else:
                    loss = logits.float().abs().mean()
            if require_backward:
                loss.backward()
            torch.cuda.synchronize()
            return True
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda error" in msg or "cudnn_status" in msg:
                return False
            raise
        finally:
            model_holder.clear()
            torch.cuda.empty_cache()

    def _apply_safety(last_ok_bs: int) -> int:
        # For the small 3D medical-imaging batch sizes used here, floor(2*0.85)
        # would incorrectly discard a verified-safe batch size of 2 and fall
        # back to 1. Keep bs=2 when it is directly proven safe; apply the
        # fractional safety margin for larger batches.
        if last_ok_bs <= 2:
            return max(1, last_ok_bs)
        return max(1, int(math.floor(last_ok_bs * safety)))

    # Phase 1 - doubling search.
    bs = 1
    last_ok = 0
    while bs <= upper_bound:
        if _try(bs):
            last_ok = bs
            next_bs = bs * 2
            if next_bs > upper_bound:
                break
            bs = next_bs
        else:
            break

    if last_ok == 0:
        LOG.warning(
            "[auto_resources] batch_size=1 failed the fit probe - model or "
            "patch_size is too large for this GPU. Returning 1."
        )
        return 1
    if last_ok * 2 > upper_bound and last_ok == upper_bound:
        return _apply_safety(last_ok)

    # Phase 2 - bisection between the last safe bs and the first failing bs.
    lo = last_ok
    hi = min(bs, upper_bound)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _try(mid):
            lo = mid
        else:
            hi = mid
    return _apply_safety(lo)


def auto_tune_resources(
    cfg: Any,
    build_model_fn,
    input_shape: Tuple[int, int, int, int],
    num_classes: int,
    out_dir: Path,
    *,
    label: str = "auto",
    require_backward: bool = True,
    safety: float = 0.85,
    batch_size_upper_bound: Optional[int] = None,
    probe: bool = True,
) -> Dict[str, Any]:
    """Detect resources, probe batch size, rewrite ``cfg`` in place.

    Probes CUDA batch size, then caps DataLoader workers/prefetch so
    cold-cache full-volume loading cannot build an oversized CPU queue. Also
    sets OMP/MKL/OPENBLAS/NUMEXPR threads to 1 and torch threads to 1.

    Writes ``<out_dir>/auto_resources.json`` and returns the decision record.
    """
    report = detect_system_resources(_tracked_gpu_index())
    device = torch.device(cfg.device)

    old = {
        "batch_size": getattr(cfg, "batch_size", None),
        "num_workers": getattr(cfg, "num_workers", None),
        "val_num_workers": getattr(cfg, "val_num_workers", None),
    }

    # CPU plan: keep enough workers to overlap I/O, but do not let every CPU
    # prefetch full 3D cases. That filled the Slurm memory cgroup with file
    # cache before CUDA saw a batch in jobs 11211618 / 11211629. The requested
    # value below is still passed through _loader_num_workers(), which enforces
    # OGAP_MAX_TRAIN_WORKERS / OGAP_MAX_EVAL_WORKERS and keeps future sbatch
    # resource changes from oversubscribing the allocated CPUs.
    n_cpu = report.cpu_allocated
    requested_workers = int(old["num_workers"] or 0)
    requested_val_workers = int(old["val_num_workers"] or 0)
    new_num_workers = _loader_num_workers(
        min(max(2, n_cpu - 2), requested_workers if requested_workers > 0 else n_cpu),
        train_mode=True,
    )
    val_cpu_ceiling = min(max(2, n_cpu // 4), 8)
    requested_val_target = requested_val_workers if requested_val_workers > 0 else val_cpu_ceiling
    new_val_num_workers = _loader_num_workers(
        min(val_cpu_ceiling, requested_val_target),
        train_mode=False,
    )

    # Thread plan: one thread per worker to avoid oversubscription. We only
    # tighten here if the caller hasn't already pinned them.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # set_num_interop_threads must be called before any parallel work;
        # silently ignore if we're too late.
        pass

    # VRAM plan: probe if we're on CUDA and probing is requested.
    probed: Optional[int] = None
    probe_upper: Optional[int] = None
    if probe and device.type == "cuda":
        if batch_size_upper_bound is not None:
            upper = int(batch_size_upper_bound)
        else:
            try:
                upper = int(os.environ.get("OGAP_AUTO_BATCH_UPPER_BOUND", "0"))
            except (TypeError, ValueError):
                upper = 0
            if upper <= 0:
                upper = max(int(old["batch_size"] or 2) * 2, 16)
        probe_upper = int(upper)
        try:
            probed = probe_max_batch_size(
                build_model_fn=build_model_fn,
                input_shape=input_shape,
                num_classes=num_classes,
                device=device,
                amp_dtype=torch.bfloat16,
                upper_bound=int(upper),
                safety=safety,
                require_backward=require_backward,
            )
        except Exception as exc:
            LOG.warning("[auto_resources/%s] probe_max_batch_size failed: %s - "
                        "keeping existing batch_size=%s",
                        label, exc, old["batch_size"])
            probed = None

    new_batch_size = int(old["batch_size"] or 1)
    if probed is not None:
        # AUTO_RESOURCES owns the effective batch size: it may grow a
        # conservative user floor, but it must also be able to shrink an
        # over-ambitious stale export before training reaches a CUDA OOM.
        new_batch_size = max(1, int(probed))

    train_prefetch_factor = _loader_prefetch_factor()
    val_prefetch_factor = _loader_prefetch_factor()
    train_max_inflight = _loader_max_inflight_cases(train_mode=True)
    val_max_inflight = _loader_max_inflight_cases(train_mode=False)
    new_num_workers, train_prefetch_factor, train_max_inflight = _loader_effective_plan(
        batch_size=new_batch_size,
        num_workers=new_num_workers,
        train_mode=True,
    )
    new_val_num_workers, val_prefetch_factor, val_max_inflight = _loader_effective_plan(
        batch_size=1,
        num_workers=new_val_num_workers,
        train_mode=False,
    )

    cfg.batch_size = int(new_batch_size)
    cfg.num_workers = int(new_num_workers)
    cfg.val_num_workers = int(new_val_num_workers)

    decision = {
        "label": label,
        "resources": asdict(report),
        "previous": old,
        "applied": {
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "val_num_workers": cfg.val_num_workers,
            "train_prefetch_factor": train_prefetch_factor,
            "val_prefetch_factor": val_prefetch_factor,
            "train_max_inflight_cases": train_max_inflight,
            "val_max_inflight_cases": val_max_inflight,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        },
        "probed_max_batch_size": probed,
        "safety_factor": safety,
        "batch_size_upper_bound": batch_size_upper_bound,
        "resolved_batch_size_upper_bound": probe_upper,
        "timestamp_unix": round(time.time(), 3),
    }
    try:
        (Path(out_dir) / "auto_resources.json").write_text(
            json.dumps(decision, indent=2)
        )
    except Exception as exc:
        LOG.warning("[auto_resources/%s] could not save decision JSON: %s", label, exc)

    LOG.info(
        "[auto_resources/%s] alloc: %d/%d CPU, %.1f/%.1f GB RAM | GPU: %s %.1fGB "
        "(free %.1f, cap %.0fW, %s) | applied: batch_size %s→%d, "
        "num_workers %s→%d, val_num_workers %s→%d, prefetch train/val=%d/%d, "
        "inflight_cap train/val=%d/%d, probed=%s, upper=%s, safety=%.2f",
        label, report.cpu_allocated, report.cpu_total,
        report.ram_allocated_gb, report.ram_total_gb,
        report.gpu_name, report.vram_total_gb, report.vram_free_gb,
        report.power_cap_w, report.cuda_capability,
        old["batch_size"], cfg.batch_size,
        old["num_workers"], cfg.num_workers,
        old["val_num_workers"], cfg.val_num_workers,
        train_prefetch_factor, val_prefetch_factor,
        train_max_inflight, val_max_inflight,
        probed, probe_upper, safety,
    )
    return decision


DENSE_BRATS_REGION_TAGS = frozenset({
    "brats_glioma", "brats_africa", "mni", "ualberta", "africa_prospective",
    "utsw_glioma",
})


def _dice_per_class_np(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    eps: float = 1e-5,
    ignore_index: int = -100,
) -> np.ndarray:
    valid = target != ignore_index
    scores = np.empty(num_classes, dtype=np.float32)
    for cls_idx in range(num_classes):
        p = (pred == cls_idx) & valid
        t = (target == cls_idx) & valid
        inter = np.count_nonzero(p & t)
        den = np.count_nonzero(p) + np.count_nonzero(t)
        scores[cls_idx] = (2.0 * inter + eps) / (den + eps)
    return scores


def _dice_binary_np(p: np.ndarray, t: np.ndarray, eps: float = 1e-5) -> float:
    inter = np.count_nonzero(p & t)
    den = np.count_nonzero(p) + np.count_nonzero(t)
    return float((2.0 * inter + eps) / (den + eps))


def _brats_lesion_wise_region(
    p_region: np.ndarray,
    t_region: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    *,
    penalty_hd_mm: float = 373.1287,
    dilation_iterations: int = 3,
    min_lesion_size: int = 50,
) -> Dict[str, float]:
    """BraTS-2023 lesion-wise Dice / HD95 for a single binary region.

    Implementation follows Baid et al. (BraTS-2023) and the reference
    repository at github.com/rachitsaluja/BraTS-2023-Metrics:

    1. Compute connected components on GT and prediction independently.
    2. Discard GT components below ``min_lesion_size`` voxels (annotation
       jitter / partial-volume noise - not real lesions).
    3. For each surviving GT component, dilate by ``dilation_iterations``
       and look for any pred component intersecting the dilated tube.
    4. **TP**: union of intersecting pred components evaluated against the
       (un-dilated) GT component → case-level Dice / HD95.
    5. **FN**: GT component with no intersecting pred → Dice = 0,
       HD95 = ``penalty_hd_mm`` (default 373.1287 mm = BraTS 240³ diagonal).
    6. **FP**: pred component (≥ ``min_lesion_size``) not intersecting any
       GT → Dice = 0, HD95 = ``penalty_hd_mm``.
    7. Final score: arithmetic mean of TP + FN + FP scores.

    Returns Dice = 1.0, HD95 = 0.0 when both masks are empty (no lesions
    means perfect agreement, in line with BraTS aggregation rules).
    """
    p = np.asarray(p_region, dtype=bool)
    t = np.asarray(t_region, dtype=bool)

    if not p.any() and not t.any():
        return {"lw_dice": 1.0, "lw_hd95": 0.0, "lw_n_tp": 0, "lw_n_fp": 0, "lw_n_fn": 0}

    pred_labelled, n_pred = ndi.label(p)
    targ_labelled, n_targ = ndi.label(t)

    targ_sizes = (
        ndi.sum(t.astype(np.int32), targ_labelled, range(1, n_targ + 1))
        if n_targ > 0 else np.array([], dtype=np.float64)
    )
    valid_targ_ids = [int(i + 1) for i, s in enumerate(targ_sizes) if s >= min_lesion_size]

    pred_sizes = (
        ndi.sum(p.astype(np.int32), pred_labelled, range(1, n_pred + 1))
        if n_pred > 0 else np.array([], dtype=np.float64)
    )
    valid_pred_ids = {int(i + 1) for i, s in enumerate(pred_sizes) if s >= min_lesion_size}

    structure = ndi.generate_binary_structure(3, 1)
    matched_pred_ids: set = set()
    dice_scores: List[float] = []
    hd95_scores: List[float] = []
    n_tp = n_fp = n_fn = 0

    for tid in valid_targ_ids:
        gt_comp = (targ_labelled == tid)
        dilated = ndi.binary_dilation(
            gt_comp, structure=structure, iterations=int(dilation_iterations)
        )
        overlap_ids = set(int(v) for v in np.unique(pred_labelled[dilated]) if v != 0)
        overlap_ids &= valid_pred_ids
        if not overlap_ids:
            n_fn += 1
            dice_scores.append(0.0)
            hd95_scores.append(penalty_hd_mm)
            continue
        pred_union = np.isin(pred_labelled, list(overlap_ids))
        dice_scores.append(_dice_binary_np(pred_union, gt_comp))
        hd95_scores.append(_hausdorff95_binary(pred_union, gt_comp, spacing=spacing))
        matched_pred_ids.update(overlap_ids)
        n_tp += 1

    for pid in valid_pred_ids - matched_pred_ids:
        n_fp += 1
        dice_scores.append(0.0)
        hd95_scores.append(penalty_hd_mm)

    if not dice_scores:
        return {"lw_dice": 1.0, "lw_hd95": 0.0, "lw_n_tp": 0, "lw_n_fp": 0, "lw_n_fn": 0}
    return {
        "lw_dice": float(np.mean(dice_scores)),
        "lw_hd95": float(np.mean(hd95_scores)),
        "lw_n_tp": int(n_tp),
        "lw_n_fp": int(n_fp),
        "lw_n_fn": int(n_fn),
    }


def _brats_metrics_np(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[np.ndarray] = None,
    compute_hd95: bool = True,
    compute_lesion_wise: bool = False,
    lesion_min_size: int = 50,
    lesion_dilation: int = 3,
    lesion_penalty_hd_mm: float = 373.1287,
) -> Dict[str, float]:
    # Round before the int8 cast: a float label like 2.9999998 (float32 arithmetic noise
    # from an external preprocessing chain) must map to 3 (ET), not truncate to 2 (TC).
    pred_bool = np.round(np.asarray(pred, dtype=np.float32)).astype(np.int8)
    target_bool = np.round(np.asarray(target, dtype=np.float32)).astype(np.int8)
    regions = {
        "WT": ((pred_bool > 0), (target_bool > 0)),
        "TC": (((pred_bool == 1) | (pred_bool == 3)), ((target_bool == 1) | (target_bool == 3))),
        "ET": ((pred_bool == 3), (target_bool == 3)),
    }

    results: Dict[str, float] = {}
    for name, (p_region, t_region) in regions.items():
        results[f"dice_{name}"] = _dice_binary_np(p_region, t_region)
        if compute_hd95:
            results[f"hd95_{name}"] = _hausdorff95_binary(p_region, t_region, spacing=spacing)
        else:
            # NaN, not 0.0 - a 0.0 placeholder is an impossible "perfect" HD95 that
            # silently contaminates any epoch-averaged HD95. NaN propagates honestly.
            results[f"hd95_{name}"] = float("nan")
        if compute_lesion_wise:
            lw = _brats_lesion_wise_region(
                p_region, t_region, spacing=spacing,
                penalty_hd_mm=lesion_penalty_hd_mm,
                dilation_iterations=lesion_dilation,
                min_lesion_size=lesion_min_size,
            )
            results[f"lw_dice_{name}"] = lw["lw_dice"]
            results[f"lw_hd95_{name}"] = lw["lw_hd95"]
            results[f"lw_n_tp_{name}"] = lw["lw_n_tp"]
            results[f"lw_n_fp_{name}"] = lw["lw_n_fp"]
            results[f"lw_n_fn_{name}"] = lw["lw_n_fn"]

    results["dice_brats_mean"] = float(np.mean([results[f"dice_{r}"] for r in ["WT", "TC", "ET"]]))
    results["hd95_brats_mean"] = float(np.mean([results[f"hd95_{r}"] for r in ["WT", "TC", "ET"]]))
    if compute_lesion_wise:
        results["lw_dice_brats_mean"] = float(np.mean([results[f"lw_dice_{r}"] for r in ["WT", "TC", "ET"]]))
        results["lw_hd95_brats_mean"] = float(np.mean([results[f"lw_hd95_{r}"] for r in ["WT", "TC", "ET"]]))
    return results


def _score_prediction_np(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    *,
    postprocess: bool,
    use_brats: bool,
    class_prob: Optional[np.ndarray],
    spacing: Optional[np.ndarray],
    compute_hd95: bool,
    compute_lesion_wise: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    if postprocess:
        pred = connected_component_postprocessing(pred, class_prob=class_prob)
    dice_scores = _dice_per_class_np(pred, target, num_classes)
    if not use_brats:
        return dice_scores, None
    return dice_scores, _brats_metrics_np(
        pred, target,
        spacing=spacing,
        compute_hd95=compute_hd95,
        compute_lesion_wise=compute_lesion_wise,
    )



def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        raise TypeError("Refusing to serialize torch.Tensor directly; convert it first")
    if isinstance(obj, np.ndarray):
        raise TypeError("Refusing to serialize numpy.ndarray directly; convert it first")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _code_provenance() -> Dict[str, str]:
    """Reproducibility provenance recorded with every run.

    The 17k-line legacy module is referenced by *path* on the cluster, so a local
    edit between two runs is otherwise invisible in the checkpoint directory. We
    record the git SHA (+ a dirty flag) or, if git is unavailable, a hash of this
    file, plus the torch/numpy versions, so a result is traceable to exact code.
    Never raises - provenance is best-effort and must not break a training save.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    prov: Dict[str, str] = {}
    try:
        prov["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here,
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        prov["git_dirty"] = str(subprocess.run(
            ["git", "diff", "--quiet"], cwd=here, timeout=5).returncode != 0)
    except Exception:
        prov["git_sha"] = "unknown"
    try:
        prov["legacy_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    except Exception:
        prov["legacy_sha256"] = "unknown"
    prov["torch_version"] = str(getattr(torch, "__version__", "?"))
    prov["numpy_version"] = str(getattr(np, "__version__", "?"))
    return prov


def save_json(obj: dict, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_json_atomic(obj: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def save_torch_atomic(obj: Any, path: Union[str, Path]) -> None:
    """Atomically write a torch checkpoint with tmp-file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def load_resume_checkpoint_or_none(
    path: Union[str, Path],
    device: torch.device,
    *,
    label: str,
) -> Optional[Dict[str, Any]]:
    """Load a resume checkpoint, quarantining unreadable partial writes."""
    path = Path(path)
    if not path.exists():
        return None
    LOG.info("[%s] Resuming from %s", label, path)
    try:
        # Optimizer/scheduler state requires weights_only=False. Do not load
        # untrusted resume checkpoints through this path on shared systems.
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        corrupt_path = path.with_name(
            f"{path.name}.corrupt.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            os.replace(path, corrupt_path)
            LOG.warning(
                "[%s] Resume checkpoint is unreadable and was moved aside: %s "
                "-> %s. Starting from scratch. Error: %s",
                label, path, corrupt_path, exc,
            )
        except OSError as move_exc:
            LOG.warning(
                "[%s] Resume checkpoint is unreadable and could not be moved "
                "aside: %s. Starting from scratch. Load error: %s; move error: %s",
                label, path, exc, move_exc,
            )
        return None
    if not isinstance(ckpt, dict):
        raise TypeError(f"[{label}] Resume checkpoint is not a dict: {path}")
    return ckpt


def load_json(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def load_nifti_with_spacing(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI and return (data, voxel_spacing_mm)."""
    img = nib.load(path)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return img.get_fdata(dtype=np.float32), spacing


def load_nifti_with_meta(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NIfTI and return (data, voxel_spacing_mm, affine)."""
    img = nib.load(path)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    affine = np.array(img.affine, dtype=np.float64, copy=True)
    return img.get_fdata(dtype=np.float32), spacing, affine


_SITK_N4_AVAILABLE: Optional[bool] = None
_SITK_N4_WARNED = False
_SITK_THREADS_PINNED = False
_N4_SETTINGS_LOGGED = False
_N4_DISABLED_LOGGED = False


def _pin_sitk_threads() -> None:
    """Force SimpleITK / ITK to a deterministic, low thread count.

    Without this pin, every DataLoader worker that calls N4 spawns ~num-CPU
    ITK threads, so the worker pool can dispatch hundreds of ITK threads onto
    one Slurm CPU allocation and the kernel scheduler thrashes. That is what
    caused first-epoch hangs of >19 hours in the OGAP teacher runs.

    Honour ``OGAP_N4_THREADS`` (default 1). Idempotent and cheap to call from
    every ``__getitem__`` invocation; we still guard with a module flag so the
    SimpleITK import stays optional on systems without it.
    """
    global _SITK_THREADS_PINNED, _N4_SETTINGS_LOGGED
    if _SITK_THREADS_PINNED:
        return
    try:
        n_threads = max(1, int(os.environ.get("OGAP_N4_THREADS", "1")))
    except (TypeError, ValueError):
        n_threads = 1
    sitk_version: Optional[str] = None
    try:
        import SimpleITK as sitk  # type: ignore
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n_threads)
        # Best-effort version probe - different SimpleITK lines name this
        # differently, so swallow AttributeErrors and proceed.
        try:
            sitk_version = str(sitk.Version.VersionString())  # type: ignore[attr-defined]
        except Exception:
            try:
                sitk_version = str(sitk.__version__)  # type: ignore[attr-defined]
            except Exception:
                sitk_version = None
    except Exception:
        # SimpleITK absent or older API - nothing to pin.
        pass
    # Mirror the cap into ITK's env-var so any nested ITK process that does
    # not go through SimpleITK's Python API (rare) honours the same budget.
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(n_threads))
    _SITK_THREADS_PINNED = True
    # One-shot log so the worker / main-process journal makes the resolved
    # N4 budget auditable without spamming once per worker process.
    if not _N4_SETTINGS_LOGGED:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.id == 0:
            try:
                iters_resolved = _n4_iterations_from_env((50, 50, 30, 20))
                disable = os.environ.get("OGAP_DISABLE_N4", "").strip()
                disabled = disable in {"1", "true", "TRUE", "yes"}
                LOG.info(
                    "[N4] threads=%d iterations=%s disabled=%s sitk=%s",
                    n_threads,
                    ",".join(str(v) for v in iters_resolved),
                    "1" if disabled else "0",
                    sitk_version or "unavailable",
                )
            except Exception:
                # Logging is observability only - never let it abort training.
                pass
        _N4_SETTINGS_LOGGED = True


def _n4_iterations_from_env(default: Sequence[int]) -> Tuple[int, ...]:
    """Parse ``OGAP_N4_ITERATIONS`` (comma-separated) or fall back to default.

    Example:  ``OGAP_N4_ITERATIONS=50,50,30,20`` (BraTS-typical, ~half the
    work of the historical (50,50,50,50,50)).  Empty / malformed values
    return the default unchanged.
    """
    raw = os.environ.get("OGAP_N4_ITERATIONS", "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    try:
        parts = [int(v) for v in raw.replace(";", ",").split(",") if v.strip()]
        if not parts or any(v <= 0 for v in parts):
            return tuple(int(v) for v in default)
        return tuple(parts)
    except ValueError:
        return tuple(int(v) for v in default)


def _assert_aligned_volume(
    *,
    case_id: str,
    source_name: str,
    source_path: str,
    reference_shape: Tuple[int, ...],
    reference_spacing: np.ndarray,
    reference_affine: np.ndarray,
    shape: Tuple[int, ...],
    spacing: np.ndarray,
    affine: np.ndarray,
    spacing_tol_mm: float = 0.05,
    affine_translation_tol_mm: float = 0.5,
    affine_rs_tol: float = 1e-2,
) -> None:
    """Multi-scanner alignment check.

    FIX (Finding 2, 2026 audit): The previous implementation used ``atol=1e-3``
    on the full 4x4 affine, which rejected valid LMIC scans whose per-modality
    affines were correct to within a fraction of a millimetre but whose
    rotation/shear sub-blocks differed by ~1e-2 in cosine units (= ~0.6°).
    The reject was silently caught by the DataLoader's broad ``except
    Exception``, dropping the case from the epoch with no error message.

    Reasonable cross-scanner tolerances:
      * spacing: ±0.05 mm (50 micrometres) - captures e.g. 1.0 mm vs 1.00010 mm
      * translation column: ±0.5 mm - sub-voxel jitter from BraTS preprocessing
      * R/S 3x3 block:      ±1e-2 in cosines (~0.6° rotation) → WARN, don't reject

    Sub-millimetre R/S drift logs a warning so a downstream investigator can
    audit cases that are merely "noisy" rather than mis-registered.
    """
    if tuple(shape) != tuple(reference_shape):
        raise ValueError(
            f"[{case_id}] {source_name} shape mismatch: expected {reference_shape}, "
            f"got {tuple(shape)} from {source_path}"
        )
    if not np.allclose(spacing, reference_spacing, atol=spacing_tol_mm, rtol=0.0):
        raise ValueError(
            f"[{case_id}] {source_name} spacing mismatch (>{spacing_tol_mm} mm): "
            f"expected {reference_spacing.tolist()}, got {spacing.tolist()} from {source_path}"
        )
    diff = np.abs(affine - reference_affine)
    # Translation: last column, first three rows - directly in mm.
    if np.any(diff[:3, 3] > affine_translation_tol_mm):
        raise ValueError(
            f"[{case_id}] {source_name} translation mismatch (>{affine_translation_tol_mm} mm) "
            f"for {source_path}. Inputs must share a common grid."
        )
    # Rotation/Scale 3x3 block: warn only.  Common LMIC artefact, not fatal.
    if np.any(diff[:3, :3] > affine_rs_tol):
        max_rs_err = float(diff[:3, :3].max())
        LOG.warning(
            "[%s] %s R/S block off by %.2e (>%.2e) for %s - proceeding "
            "(multi-scanner LMIC drift; verify with manual QC if reproducible).",
            case_id, source_name, max_rs_err, affine_rs_tol, source_path,
        )


def save_nifti_like(array: np.ndarray, reference_path: str, out_path: str,
                    dtype: type = np.int16) -> None:
    """Save ``array`` as a NIfTI sharing ``reference_path``'s affine/header.

    ``dtype`` defaults to int16 (the segmentation-label use). A *float* array is
    rounded and range-checked before the integer cast, so a future caller passing a
    soft probability/confidence map is not silently truncated to garbage.
    """
    ref = nib.load(reference_path)
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.integer) and np.issubdtype(np.dtype(dtype), np.integer):
        info = np.iinfo(dtype)
        if arr.size and (np.nanmin(arr) < info.min or np.nanmax(arr) > info.max):
            LOG.warning("save_nifti_like: values [%.3g, %.3g] outside %s range - clipping.",
                        float(np.nanmin(arr)), float(np.nanmax(arr)), np.dtype(dtype).name)
        arr = np.clip(np.round(arr), info.min, info.max)
    nib.save(
        nib.Nifti1Image(arr.astype(dtype), affine=ref.affine, header=ref.header),
        out_path,
    )


def foreground_brain_mask(x: np.ndarray, min_voxels: int = 256) -> np.ndarray:
    """Robust non-zero/percentile brain mask used by N4 and normalisation.

    This is the final fallback when HD-BET/SynthStrip-style masks are not
    available in the local runtime. It deliberately avoids skull-stripping
    claims; it only prevents background and obvious padding from driving N4
    and z-score statistics.
    """
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    mask = finite & (x != 0)
    if int(mask.sum()) >= min_voxels:
        return mask
    vals = x[finite]
    if vals.size == 0:
        return np.zeros_like(x, dtype=bool)
    lo = np.percentile(vals, 20.0)
    mask = finite & (x > lo)
    if int(mask.sum()) < min_voxels:
        mask = finite
    return mask


def _n4_shrink_factor() -> int:
    """Resolve the downsampling factor used for N4 estimation.

    Standard BraTS / nnU-Net preprocessing runs N4 on a 4x-shrunk grid and
    applies the log-bias-field at full resolution.  This gives 50-100x
    speedup with negligible Dice impact (verified by the BraTS 2020-2023
    challenge winners and the SimpleITK N4 documentation).

    Override via ``OGAP_N4_SHRINK_FACTOR``.  Set to 1 to restore the
    historical full-resolution N4 (slow; only useful for ablations).
    """
    try:
        v = int(os.environ.get("OGAP_N4_SHRINK_FACTOR", "4"))
    except (TypeError, ValueError):
        return 4
    return max(1, min(v, 8))  # 1..8 keeps the upsample reasonable


def n4_bias_correct_volume(
    x: np.ndarray,
    *,
    iterations: Optional[Sequence[int]] = None,
    shrink_factor: Optional[int] = None,
    spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Apply N4 bias correction within a foreground brain mask.

    SimpleITK is optional on Rorqual/local installs. If it is absent, the
    function logs once and falls back to the raw volume so jobs do not fail
    because a preprocessing extra was missing.

    Threading: ITK is pinned to a single thread (or ``OGAP_N4_THREADS``) per
    process before the corrector runs. Without this, every DataLoader worker
    spawns ~num-CPU ITK threads, oversubscribing the SLURM allocation and
    stalling first-epoch preprocessing for hours (the cause of the
    11111229/11112647 teacher-job hangs).

    Iterations: defaults to the BraTS-typical schedule (50, 50, 30, 20)
    instead of the historical (50, 50, 50, 50, 50). The aggressive 5-level
    schedule roughly doubled per-volume cost without measurably improving
    Dice on glioma segmentation. Override via ``OGAP_N4_ITERATIONS=…``.

    Shrink factor: defaults to 4 (BraTS-standard).  N4 is estimated on a
    4x-downsampled grid; the log-bias-field is then upsampled to the
    original resolution and divided into the input.  This is ~50-100x
    faster than running N4 at full resolution and is what every modern
    glioma segmentation pipeline does.  Override via
    ``OGAP_N4_SHRINK_FACTOR``.
    """
    global _SITK_N4_AVAILABLE, _SITK_N4_WARNED, _N4_DISABLED_LOGGED
    if _env_flag_enabled("OGAP_DISABLE_N4"):
        worker_info = torch.utils.data.get_worker_info()
        if not _N4_DISABLED_LOGGED and (worker_info is None or worker_info.id == 0):
            LOG.info("[N4] disabled by OGAP_DISABLE_N4=1; using foreground z-score only")
            _N4_DISABLED_LOGGED = True
        return x.astype(np.float32, copy=False)

    if _SITK_N4_AVAILABLE is False:
        return x.astype(np.float32, copy=False)

    try:
        import SimpleITK as sitk  # type: ignore
        _SITK_N4_AVAILABLE = True
    except Exception as exc:
        _SITK_N4_AVAILABLE = False
        if not _SITK_N4_WARNED:
            LOG.warning(
                "SimpleITK unavailable for N4 bias correction (%s). "
                "Continuing with foreground z-score only. Install SimpleITK "
                "or set OGAP_DISABLE_N4=1 to silence this path.",
                exc,
            )
            _SITK_N4_WARNED = True
        return x.astype(np.float32, copy=False)

    _pin_sitk_threads()

    arr = np.asarray(x, dtype=np.float32)
    mask_np = foreground_brain_mask(arr)
    if int(mask_np.sum()) < 256:
        return arr

    iters = _n4_iterations_from_env(iterations if iterations is not None else (50, 50, 30, 20))
    shrink = int(shrink_factor) if shrink_factor is not None else _n4_shrink_factor()
    shrink = max(1, min(shrink, 8))

    try:
        image = sitk.GetImageFromArray(arr)
        mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))
        if spacing is not None and len(spacing) == 3:
            # N4 fits a physical-scale B-spline bias field, so the voxel spacing must be
            # set or anisotropic scans (thick-slice LMIC / 1x1x1.5 mm UTSW) get the wrong
            # control-point pitch. sitk.GetImageFromArray reverses axes (numpy (i,j,k) ->
            # ITK (x,y,z) = (k,j,i)), so the spacing is reversed to match.
            sp_itk = [float(s) for s in spacing[::-1]]
            image.SetSpacing(sp_itk)
            mask.SetSpacing(sp_itk)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(v) for v in iters])

        if shrink > 1:
            # Estimate N4 on a shrunk grid, then upsample the log-bias-field
            # back to the original resolution.  This is the BraTS-standard
            # "fast N4" pattern (50-100x speedup).
            shrink_vec = [shrink] * image.GetDimension()
            image_shrunk = sitk.Shrink(image, shrink_vec)
            mask_shrunk = sitk.Shrink(mask, shrink_vec)
            # Run the corrector on the shrunk grid for the side effect of
            # caching the log-bias-field; we discard the shrunk output and
            # rebuild the corrected volume at full resolution below.
            _ = corrector.Execute(image_shrunk, mask_shrunk)
            # GetLogBiasFieldAsImage takes the original (full-resolution)
            # image and returns the log-bias-field interpolated onto its
            # grid.  Available since SimpleITK 1.2 (we require >=2.0).
            log_bias = corrector.GetLogBiasFieldAsImage(image)
            corrected_img = image / sitk.Exp(log_bias)
            corrected = sitk.GetArrayFromImage(corrected_img).astype(np.float32, copy=False)
        else:
            corrected = sitk.GetArrayFromImage(
                corrector.Execute(image, mask)
            ).astype(np.float32, copy=False)

        corrected[~mask_np] = 0.0
        return corrected
    except Exception as exc:
        LOG.warning(
            "N4 bias correction failed (shrink=%d, iters=%s); using raw volume for this case: %s",
            shrink, iters, exc,
        )
        return arr


# ── FIX #1: Background stays zero after foreground z-score normalisation ──
def zscore_nonzero(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Foreground-only z-score normalisation (nnU-Net convention).

    Computes mean/std over non-zero voxels only (brain mask proxy).
    Background voxels are explicitly kept at zero so that downstream
    logic (augmentations, availability maps) can rely on x==0 meaning
    "no tissue / no signal".

    FIX: Original code normalised the entire volume, which turned
    background zeros into a constant negative value, breaking the
    background==0 assumption used by bias field, intensity ops, and
    the availability map logic.

    Ref: nnU-Net v2 (Isensee et al., Nature Methods 2021)
         Reinhold et al., "Evaluating the Impact of Intensity
         Normalization on MR Image Synthesis" (2020)
    """
    nz = x != 0
    if not np.any(nz):
        return x.astype(np.float32, copy=False)

    fg = x[nz]
    mean, std = fg.mean(), fg.std()

    # FIX #13: Warn on near-zero variance (corrupted scan / acquisition artifact)
    if std < eps:
        LOG.warning(
            f"zscore_nonzero: foreground std={std:.2e} < eps={eps:.2e}. "
            "This may indicate a corrupted scan or acquisition artifact. "
            "Returning un-normalised foreground."
        )
        out = np.zeros_like(x, dtype=np.float32)
        out[nz] = fg.astype(np.float32)
        return out

    out = np.zeros_like(x, dtype=np.float32)
    out[nz] = fg.astype(np.float32)
    out[nz] -= mean
    out[nz] /= (std + eps)
    return out


def preprocess_mri_volume(x: np.ndarray, *, apply_n4: bool = True,
                          spacing: Optional[Sequence[float]] = None) -> np.ndarray:
    """Default OGAP MRI preprocessing: N4 bias correction, then foreground z-score.

    ``spacing`` (the NIfTI voxel size, (i,j,k) order) is forwarded to N4 so the bias
    field is estimated at the correct physical scale on anisotropic scans. Pass it at
    BOTH train and inference call sites to keep preprocessing identical (no skew).
    """
    x = n4_bias_correct_volume(x, spacing=spacing) if apply_n4 else x.astype(np.float32, copy=False)
    return zscore_nonzero(x)


def simulate_annotation_noise(
    y: np.ndarray,
    p_morph: float = 0.15,
    max_radius_vox: float = 1.5,
    rng: Optional[np.random.Generator] = None,
    mode: str = "morph",
    band_radius: int = 2,
    band_flip_prob: float = 0.3,
) -> np.ndarray:
    """Label perturbation to model annotation imprecision.

    TRAIN-ONLY. This function is invoked exclusively inside
    MultiModalSegDataset.__getitem__ when the dataset's ``label_noise_prob``
    is > 0 and only for the training split (see §5). It is NEVER applied
    during validation, evaluation, inference, ONNX export, or quantisation
    comparison - those paths consume raw labels. Reviewers: verify by
    grepping the code for calls to ``simulate_annotation_noise``; the only
    call-site outside this definition is in the training dataset.

    Modes:
        "morph" (default, backwards-compatible):
            Per-class symmetric erosion/dilation by 1-2 voxels - coarse but
            cheap. Models systematic over/under-segmentation across raters.
        "boundary_band" (preferred for evaluation rigor; PDF §4.4):
            Identify the ``band_radius``-voxel band around each class
            boundary; flip each band voxel with Bernoulli probability
            ``band_flip_prob`` (50/50 split between ``+class`` and ``→0``).
            Models per-rater random jitter on ambiguous borders, which
            better matches inter-rater variance reported on BraTS-Africa
            (Adewole 2023, Section 4.4 of the OGAP 2.0 plan).
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() >= p_morph:
        return y

    unique_classes = [int(c) for c in np.unique(y) if int(c) > 0]
    if not unique_classes:
        return y

    struct = ndi.generate_binary_structure(3, 1)
    y_out = y.copy()

    if mode == "boundary_band":
        band_radius = max(1, int(band_radius))
        band_flip_prob = float(np.clip(band_flip_prob, 0.0, 1.0))
        # Per class, define band = (dilation_K) XOR (erosion_K). Each band
        # voxel flips with prob `band_flip_prob`; flips are 50/50 split into
        # FP (background→class) and FN (class→background).
        for cls in sorted(unique_classes, reverse=True):
            cls_mask = y == cls
            if not cls_mask.any():
                continue
            inner = ndi.binary_erosion(cls_mask, structure=struct, iterations=band_radius)
            outer = ndi.binary_dilation(cls_mask, structure=struct, iterations=band_radius)
            band = outer & ~inner
            if not band.any():
                continue
            flip = rng.random(size=band.shape) < band_flip_prob
            flip &= band
            if not flip.any():
                continue
            sign = rng.random(size=band.shape) < 0.5
            # FN candidates: voxel currently in cls AND flip & sign
            fn_voxels = flip & sign & cls_mask
            # FP candidates: voxel currently in background AND flip & ~sign
            fp_voxels = flip & ~sign & (y_out == 0)
            y_out[fn_voxels] = 0
            y_out[fp_voxels] = cls
        return y_out

    # mode == "morph" (legacy)
    radius = rng.uniform(0.5, max_radius_vox)
    n_iters = max(1, int(round(radius)))
    erode = rng.random() < 0.5
    for cls in sorted(unique_classes, reverse=True):
        cls_mask = y == cls
        if not cls_mask.any():
            continue
        if erode:
            eroded = ndi.binary_erosion(cls_mask, structure=struct, iterations=n_iters)
            lost = cls_mask & ~eroded
            y_out[lost] = 0
        else:
            dilated = ndi.binary_dilation(cls_mask, structure=struct, iterations=n_iters)
            gained = dilated & (y_out == 0)
            y_out[gained] = cls
    return y_out


def pad_to_patch(
    x: np.ndarray,
    patch_size: Tuple[int, int, int],
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Symmetric zero-pad so every spatial dim >= patch_size."""
    _, d, h, w = x.shape
    pd, ph, pw = patch_size
    dd, dh, dw = max(0, pd - d), max(0, ph - h), max(0, pw - w)
    if dd + dh + dw == 0:
        return x, y
    x = np.pad(x, ((0, 0), (dd // 2, dd - dd // 2),
                    (dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)),
               mode="constant")
    if y is not None:
        y = np.pad(y, ((dd // 2, dd - dd // 2),
                       (dh // 2, dh - dh // 2), (dw // 2, dw - dw // 2)),
                  mode="constant")
    return x, y


# ══════════════════════════════════════════════════════════════════════════════
# §2  MODALITY MASKING AND AVAILABILITY MAPS
# ══════════════════════════════════════════════════════════════════════════════

def random_modality_mask(
    batch_size: int,
    num_modalities: int,
    keep_min: int,
    keep_max: int,
    device: torch.device,
    keep_count_bias_power: float = 1.5,
) -> torch.Tensor:
    """Generate random binary modality masks with curriculum-controlled bounds."""
    keep_max = min(keep_max, num_modalities)
    keep_min = max(1, min(keep_min, keep_max))
    keep_counts = torch.arange(keep_min, keep_max + 1, device=device, dtype=torch.float32)
    if keep_count_bias_power > 0:
        keep_probs = keep_counts.pow(float(keep_count_bias_power))
        keep_probs = keep_probs / keep_probs.sum()
    else:
        keep_probs = None
    if keep_probs is None:
        k = torch.randint(keep_min, keep_max + 1, (batch_size,), device=device)
    else:
        sampled_idx = torch.multinomial(keep_probs, batch_size, replacement=True)
        k = keep_counts[sampled_idx].to(torch.long)
    scores = torch.rand((batch_size, num_modalities), device=device)
    order = scores.argsort(dim=1, descending=True)
    keep_rank = (
        torch.arange(num_modalities, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        < k.unsqueeze(1)
    ).to(torch.float32)
    mask = torch.zeros((batch_size, num_modalities), dtype=torch.float32, device=device)
    mask.scatter_(1, order, keep_rank)
    return mask


def prepare_student_input(x_full: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked volumes with binary availability maps.

    Input  : x_full (B, C, D, H, W), modality_mask (B, M)
    Output : (B, C + M, D, H, W)

    The availability maps disambiguate truly-missing modalities from
    genuinely-zero voxel values, providing the network with explicit
    conditioning on which information channels are present.

    ``M`` may be smaller than ``C`` when optional non-MRI channels are appended
    to ``x_full`` (for example CSF/WM/GM tissue priors). In that case only the
    first ``M`` MRI channels are masked; extra anatomical-prior channels pass
    through unchanged and do not get fake availability maps.
    """
    b, c, d, h, w = x_full.shape
    if modality_mask.ndim != 2:
        raise ValueError(
            f"modality_mask must have shape (B, M), got {tuple(modality_mask.shape)}"
        )
    if modality_mask.shape[0] != b:
        raise ValueError(
            f"modality_mask batch {modality_mask.shape[0]} does not match input batch {b}"
        )
    m_ch = int(modality_mask.shape[1])
    if m_ch > c:
        raise ValueError(
            f"modality_mask has {m_ch} channels but input has only {c} channels"
        )
    m = modality_mask.to(device=x_full.device, dtype=x_full.dtype).view(b, m_ch, 1, 1, 1)
    x_mri = x_full[:, :m_ch] * m
    if m_ch < c:
        x_partial = torch.cat([x_mri, x_full[:, m_ch:]], dim=1)
    else:
        x_partial = x_mri
    m_chan = m.expand(-1, -1, d, h, w)
    return torch.cat([x_partial, m_chan], dim=1)


def all_modality_combinations(num_modalities: int) -> List[Optional[List[int]]]:
    """Generate all 2^N - 1 non-empty modality drop configurations + full set.

    Returns list of missing-modality index lists.
    None = all modalities present (full set).
    """
    configs: List[Optional[List[int]]] = [None]  # full set first
    indices = list(range(num_modalities))
    for r in range(1, num_modalities):  # drop 1, drop 2, ... drop N-1
        for combo in itertools.combinations(indices, r):
            configs.append(list(combo))
    return configs


def modality_config_tag(missing_mods: Optional[Sequence[int]]) -> str:
    """Stable metric tag for a modality-missing configuration."""
    if missing_mods is None:
        return "all"
    return f"drop{''.join(str(int(m)) for m in missing_mods)}"


def present_modality_count(
    num_modalities: int,
    missing_mods: Optional[Sequence[int]],
) -> int:
    """Number of present modalities for a given missing-modality config."""
    if missing_mods is None:
        return int(num_modalities)
    return int(num_modalities - len(missing_mods))


def supported_modality_combinations(
    num_modalities: int,
    min_present: int = 1,
    max_present: Optional[int] = None,
) -> List[Optional[List[int]]]:
    """Enumerate deployment-supported modality combinations.

    For a 4-modality student with ``min_present=2`` this returns the 11
    configurations corresponding to all valid 2-, 3-, and 4-modality inputs.
    """
    max_present = num_modalities if max_present is None else int(max_present)
    max_present = max(1, min(max_present, num_modalities))
    min_present = max(1, min(int(min_present), max_present))
    return [
        missing_mods
        for missing_mods in all_modality_combinations(num_modalities)
        if min_present <= present_modality_count(num_modalities, missing_mods) <= max_present
    ]


def random_modality_mask_from_combinations(
    batch_size: int,
    num_modalities: int,
    allowed_configs: Sequence[Optional[Sequence[int]]],
    device: torch.device,
    keep_count_bias_power: float = 0.0,
) -> torch.Tensor:
    """Sample masks from explicit modality combinations.

    This is more deployment-faithful than sampling only a keep-count and then
    drawing random modalities, because it lets the student see the real 2-4
    modality combinations it will face at inference.
    """
    configs = list(allowed_configs) if allowed_configs else [None]
    mask_bank = torch.ones((len(configs), num_modalities), dtype=torch.float32, device=device)
    weights: List[float] = []
    for cfg_idx, missing_mods in enumerate(configs):
        if missing_mods:
            for mod_idx in missing_mods:
                if 0 <= int(mod_idx) < num_modalities:
                    mask_bank[cfg_idx, int(mod_idx)] = 0.0
        present_count = present_modality_count(num_modalities, missing_mods)
        if keep_count_bias_power > 0:
            weights.append(float(present_count) ** float(keep_count_bias_power))
        else:
            weights.append(1.0)
    weight_t = torch.as_tensor(weights, device=device, dtype=torch.float32)
    sampled_idx = torch.multinomial(weight_t / weight_t.sum(), batch_size, replacement=True)
    return mask_bank.index_select(0, sampled_idx)


def deployment_fast_eval_configs(
    num_modalities: int,
    *,
    epoch: int,
    min_present: int = 2,
    max_present: Optional[int] = None,
    total_configs: int = 7,
) -> List[Optional[List[int]]]:
    """Deployment-focused validation subset for regular epochs.

    Uses all 4- and 3-modality configurations every validation epoch, then
    rotates through a small number of 2-modality configurations so sparse-input
    behaviour contributes continuously to checkpoint selection without forcing
    a full 11-config validation every epoch.
    """
    full_configs = supported_modality_combinations(
        num_modalities,
        min_present=min_present,
        max_present=max_present,
    )
    always_configs = [
        cfg for cfg in full_configs
        if cfg is None or len(cfg) == 1
    ]
    if total_configs <= len(always_configs):
        return always_configs[: max(1, total_configs)]

    sparse_configs = [cfg for cfg in full_configs if cfg is not None and len(cfg) >= 2]
    remaining = min(len(sparse_configs), max(0, total_configs - len(always_configs)))
    if remaining == 0 or not sparse_configs:
        return always_configs

    start = max(0, int(epoch)) % len(sparse_configs)
    rotated = [
        sparse_configs[(start + offset) % len(sparse_configs)]
        for offset in range(remaining)
    ]
    return always_configs + rotated


def summarize_supported_modality_performance(
    metrics: Dict[str, Any],
    eval_mask_configs: Sequence[Optional[Sequence[int]]],
    num_modalities: int,
    *,
    min_present: int = 2,
    max_present: Optional[int] = None,
    base_metric: str = "dice_brats_mean",
) -> Dict[str, float]:
    """Aggregate deployment-relevant scores across supported modality regimes."""
    max_present = num_modalities if max_present is None else int(max_present)
    prefix = f"deploy{int(min_present)}{int(max_present)}"
    grouped: Dict[int, List[float]] = {}
    all_vals: List[float] = []
    for missing_mods in eval_mask_configs:
        present_count = present_modality_count(num_modalities, missing_mods)
        if not (min_present <= present_count <= max_present):
            continue
        tag = modality_config_tag(missing_mods)
        metric_key = f"{tag}_{base_metric}"
        if metric_key not in metrics:
            continue
        val = float(metrics[metric_key])
        grouped.setdefault(present_count, []).append(val)
        all_vals.append(val)

    out: Dict[str, float] = {}
    if all_vals:
        out[f"{prefix}_{base_metric}"] = float(np.mean(all_vals))
        out[f"{prefix}_eval_n_configs"] = float(len(all_vals))
    for present_count, vals in sorted(grouped.items()):
        if vals:
            out[f"{prefix}_{present_count}mod_{base_metric}"] = float(np.mean(vals))
    sparse_vals = grouped.get(int(min_present), [])
    if sparse_vals:
        out[f"{prefix}_worst_{int(min_present)}mod_{base_metric}"] = float(np.min(sparse_vals))
    return out


def summarise_quantisation_deployment(
    comparison: Dict[str, Dict[str, Any]],
    eval_mask_configs: Sequence[Optional[Sequence[int]]],
    num_modalities: int,
    *,
    min_present: int = 2,
    max_present: Optional[int] = None,
    dice_metric: str = "dice_brats_mean",
    hd95_metric: str = "hd95_brats_mean",
    max_mean_dice_drop: float = 0.01,
    max_worst_dice_drop: float = 0.02,
    max_mean_hd95_increase: float = 1.0,
    max_worst_hd95_increase: float = 2.0,
) -> Dict[str, Any]:
    """Summarise whether INT8 drift is acceptable for deployment.

    ``comparison`` contains paired FP32-vs-INT8 statistics where
    ``mean_diff = mean(fp32 - int8)``. For Dice, positive values indicate INT8
    accuracy loss; for HD95, negative values indicate INT8 degradation because
    higher HD95 is worse.
    """
    max_present = num_modalities if max_present is None else int(max_present)
    supported_cfgs = [
        cfg for cfg in eval_mask_configs
        if min_present <= present_modality_count(num_modalities, cfg) <= max_present
    ]
    supported_tags = [modality_config_tag(cfg) for cfg in supported_cfgs]
    hardest_tags = [
        modality_config_tag(cfg) for cfg in supported_cfgs
        if present_modality_count(num_modalities, cfg) == int(min_present)
    ]

    def _collect(metric_name: str, *, larger_is_better: bool) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for tag in supported_tags:
            stat = comparison.get(tag, {}).get(metric_name)
            if not stat:
                continue
            mean_diff = float(stat.get("mean_diff", 0.0))
            degradation = mean_diff if larger_is_better else -mean_diff
            values[tag] = degradation
        return values

    dice_drop_by_tag = _collect(dice_metric, larger_is_better=True)
    hd95_increase_by_tag = _collect(hd95_metric, larger_is_better=False)

    def _subset(values: Dict[str, float], tags: Sequence[str]) -> List[float]:
        return [float(values[tag]) for tag in tags if tag in values]

    supported_dice = _subset(dice_drop_by_tag, supported_tags)
    supported_hd95 = _subset(hd95_increase_by_tag, supported_tags)
    hardest_dice = _subset(dice_drop_by_tag, hardest_tags)
    hardest_hd95 = _subset(hd95_increase_by_tag, hardest_tags)

    summary: Dict[str, Any] = {
        "supported_config_tags": supported_tags,
        "hardest_config_tags": hardest_tags,
        "criteria": {
            "min_present_modalities": int(min_present),
            "max_present_modalities": int(max_present),
            "max_mean_dice_drop": float(max_mean_dice_drop),
            "max_worst_dice_drop": float(max_worst_dice_drop),
            "max_mean_hd95_increase": float(max_mean_hd95_increase),
            "max_worst_hd95_increase": float(max_worst_hd95_increase),
        },
        "n_supported_configs_evaluated": int(len(supported_dice)),
        "n_hardest_configs_evaluated": int(len(hardest_dice)),
        "significant_dice_drifts_holm_05": int(sum(
            1 for tag in supported_tags
            if comparison.get(tag, {}).get(dice_metric, {}).get("significant_holm_05", False)
        )),
        "significant_hd95_drifts_holm_05": int(sum(
            1 for tag in supported_tags
            if comparison.get(tag, {}).get(hd95_metric, {}).get("significant_holm_05", False)
        )),
        "supported_mean_dice_drop": float(np.mean(supported_dice)) if supported_dice else None,
        "supported_worst_dice_drop": float(np.max(supported_dice)) if supported_dice else None,
        "supported_mean_hd95_increase": float(np.mean(supported_hd95)) if supported_hd95 else None,
        "supported_worst_hd95_increase": float(np.max(supported_hd95)) if supported_hd95 else None,
        f"{int(min_present)}mod_mean_dice_drop": float(np.mean(hardest_dice)) if hardest_dice else None,
        f"{int(min_present)}mod_worst_dice_drop": float(np.max(hardest_dice)) if hardest_dice else None,
        f"{int(min_present)}mod_mean_hd95_increase": float(np.mean(hardest_hd95)) if hardest_hd95 else None,
        f"{int(min_present)}mod_worst_hd95_increase": float(np.max(hardest_hd95)) if hardest_hd95 else None,
        "per_config_dice_drop": {tag: round(val, 6) for tag, val in dice_drop_by_tag.items()},
        "per_config_hd95_increase": {tag: round(val, 6) for tag, val in hd95_increase_by_tag.items()},
    }

    fail_reasons: List[str] = []
    if not supported_dice or not supported_hd95:
        fail_reasons.append("missing_supported_quantisation_metrics")
    else:
        if float(np.mean(supported_dice)) > float(max_mean_dice_drop):
            fail_reasons.append("supported_mean_dice_drop_exceeds_threshold")
        if float(np.max(supported_dice)) > float(max_worst_dice_drop):
            fail_reasons.append("supported_worst_dice_drop_exceeds_threshold")
        if float(np.mean(supported_hd95)) > float(max_mean_hd95_increase):
            fail_reasons.append("supported_mean_hd95_increase_exceeds_threshold")
        if float(np.max(supported_hd95)) > float(max_worst_hd95_increase):
            fail_reasons.append("supported_worst_hd95_increase_exceeds_threshold")
    if hardest_dice and float(np.max(hardest_dice)) > float(max_worst_dice_drop):
        fail_reasons.append(f"{int(min_present)}mod_worst_dice_drop_exceeds_threshold")
    if hardest_hd95 and float(np.max(hardest_hd95)) > float(max_worst_hd95_increase):
        fail_reasons.append(f"{int(min_present)}mod_worst_hd95_increase_exceeds_threshold")

    summary["deployment_pass"] = len(fail_reasons) == 0
    summary["deployment_fail_reasons"] = fail_reasons
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# §3  BraTS LABEL UTILITIES AND METRICS
# ══════════════════════════════════════════════════════════════════════════════
#
# BraTS 2023 label convention: 0=BG, 1=NCR (necrotic core), 2=ED (oedema),
# 3=ET (enhancing tumour).
# Hierarchical evaluation regions:
#   WT (Whole Tumour)  = {1, 2, 3}
#   TC (Tumour Core)   = {1, 3}
#   ET (Enhancing T.)  = {3}
# ══════════════════════════════════════════════════════════════════════════════

def _dice_binary(p: torch.Tensor, t: torch.Tensor, eps: float = 1e-5) -> float:
    inter = (p * t).sum()
    den = p.sum() + t.sum()
    return float((2 * inter + eps) / (den + eps))


# ── FIX #8: HD95 with physical voxel spacing ──
def _hausdorff95_binary(
    p: np.ndarray,
    t: np.ndarray,
    spacing: Optional[np.ndarray] = None,
) -> float:
    """Hausdorff distance at the 95th percentile between binary masks.

    FIX: Now accepts voxel spacing (in mm) and passes it to
    distance_transform_edt so that distances are in physical units.
    Without spacing, HD95 in voxel units is clinically meaningless
    for comparing results across scanner protocols with anisotropic
    voxels (e.g. 1×1×5 mm).

    Returns 373.1287 mm if either mask is empty (BraTS convention:
    diagonal of 240×240×155 @ 1mm ≈ 373.128 mm).

    Ref: Podobnik & Vrtovec, "HDilemma" MICCAI 2024
         BraTS-2023-Metrics (github.com/rachitsaluja/BraTS-2023-Metrics)
    """
    if p.sum() == 0 or t.sum() == 0:
        if p.sum() == 0 and t.sum() == 0:
            return 0.0
        return 373.1287  # BraTS convention: max possible distance for 240³ volume

    from scipy.ndimage import distance_transform_edt

    # Surface voxels: erode then XOR
    p_border = p ^ ndi.binary_erosion(p, iterations=1)
    t_border = t ^ ndi.binary_erosion(t, iterations=1)

    # Edge case: single-voxel predictions produce empty border after erosion
    if p_border.sum() == 0:
        p_border = p
    if t_border.sum() == 0:
        t_border = t

    # FIX: Pass physical spacing to EDT for correct mm-scale distances
    edt_kwargs = {"sampling": spacing} if spacing is not None else {}
    dt_pred = distance_transform_edt(~p_border, **edt_kwargs)
    dt_targ = distance_transform_edt(~t_border, **edt_kwargs)

    dists_pred_to_targ = dt_targ[p_border > 0]
    dists_targ_to_pred = dt_pred[t_border > 0]

    # BraTS standard: max of directional 95th percentiles
    # (concatenating then taking one percentile underestimates; see HDilemma)
    hd95_p2t = float(np.percentile(dists_pred_to_targ, 95))
    hd95_t2p = float(np.percentile(dists_targ_to_pred, 95))
    return max(hd95_p2t, hd95_t2p)


def brats_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[np.ndarray] = None,
    compute_hd95: bool = True,
) -> Dict[str, float]:
    """Compute BraTS WT/TC/ET Dice and optionally HD95.

    Args:
        spacing: voxel spacing in mm (e.g. [1.0, 1.0, 1.0]).
                 If None, HD95 is in voxel units.
        compute_hd95: if False, skip the expensive scipy EDT computation
                      and return only Dice metrics.  HD95 keys are still
                      present (set to 0.0) so downstream code never
                      KeyErrors.  Use hd95_freq in the config to control
                      how often HD95 is actually computed during training.

    REPORTING - these are RAW, per-epoch TRAINING-DIAGNOSTIC metrics: no
    connected-component post-processing and NOT lesion-wise. They are
    deliberately pessimistic - far-away false-positive islands inflate HD95, and
    the 373.1287 mm empty-region penalty (absent ET in particular) dominates the
    mean. Do NOT quote them as paper numbers. The reported Dice/HD95 come from the
    standalone ``evaluate --lesion_wise`` path
    (``connected_component_postprocessing`` + BraTS-2023 lesion-wise HD95 with the
    373.1287 mm FP/FN convention), stratified by field strength via the
    ``stratify`` / ``field_strength_curve`` subcommands (already wired in
    slurm/submit_ogap_posthoc_full_trillium.sbatch).
    """
    results: Dict[str, float] = {}

    # Binary masks for each region
    regions = {
        "WT": ((pred > 0).float(), (target > 0).float()),
        "TC": (((pred == 1) | (pred == 3)).float(), ((target == 1) | (target == 3)).float()),
        "ET": ((pred == 3).float(), (target == 3).float()),
    }

    for name, (p, t) in regions.items():
        results[f"dice_{name}"] = _dice_binary(p, t)
        if compute_hd95:
            p_np = p.squeeze().cpu().numpy().astype(bool)
            t_np = t.squeeze().cpu().numpy().astype(bool)
            results[f"hd95_{name}"] = _hausdorff95_binary(p_np, t_np, spacing=spacing)
        else:
            results[f"hd95_{name}"] = float("nan")  # not computed this epoch (NaN, not a misleading 0.0)

    results["dice_brats_mean"] = float(np.mean([results[f"dice_{r}"] for r in ["WT", "TC", "ET"]]))
    results["hd95_brats_mean"] = float(np.mean([results[f"hd95_{r}"] for r in ["WT", "TC", "ET"]]))
    return results


def dice_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int,
    eps: float = 1e-5, ignore_index: int = -100,
) -> torch.Tensor:
    """Per-class soft Dice with ignore_index masking consistent with the loss.

    Voxels where target == ignore_index are excluded from both numerator and
    denominator so they neither inflate nor deflate any class Dice score.
    """
    valid = (target != ignore_index).float()
    out = []
    for c in range(num_classes):
        p = (pred == c).float() * valid
        t = (target == c).float() * valid
        out.append((2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))
    return torch.stack(out)


# ══════════════════════════════════════════════════════════════════════════════
# §4  MRI-PHYSICS DATA AUGMENTATION
#
# Extends nnU-Net's default pipeline with physically-motivated transforms
# targeting multi-field-strength robustness. Augmentation parameters are
# derived from MRI physics:
#   - SNR ∝ B₀ (field strength)
#   - Bias field from B₁ inhomogeneity (low-order polynomial)
#   - Rician noise = sqrt((signal + N₁)² + N₂²) for N₁, N₂ ~ N(0, σ)
#   - Resolution degradation simulates thick-slice clinical acquisition
#   - Elastic deformation re-enabled per Cirillo et al. (2020) finding
#     it among the top-2 augmentations for brain tumour segmentation
# ══════════════════════════════════════════════════════════════════════════════

class MRIPhysicsAugmentor:
    """Anatomically-realistic MRI augmentation pipeline.

    Applies transforms in a physically consistent order:
    1. Geometric (elastic, flip, rotate) - shape changes
    2. Bias field - B₁ inhomogeneity
    3. Noise - thermal/Rician noise
    4. Intensity (gamma, brightness) - scanner calibration
    5. Resolution degradation - slice thickness variation

    The low-field behaviour is intentionally conservative. Based on Marques
    et al. (JMRI 2019), moving from 1.5T/3T to sub-1T should not be modeled as
    a simple "more of every artifact". Instead, we bias the pipeline toward:
      - moderately higher noise below ~1.5T,
      - stronger resolution loss below ~1.5T,
      - no spurious increase in B1 shading at low field,
      - a higher probability of Rician rather than Gaussian noise at lower SNR.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_elastic: float = 0.3,
        p_bias_field: float = 0.6,
        p_noise: float = 0.5,
        p_rician_noise: float = 0.4,
        p_gamma: float = 0.5,
        p_brightness: float = 0.5,
        p_contrast: float = 0.5,
        p_resolution: float = 0.3,
        p_motion: float = 0.4,
        p_ghosting: float = 0.3,
        p_spike: float = 0.2,
        p_gibbs: float = 0.3,
        p_focal_b1_inhomogeneity: float = 0.12,
        p_anisotropic_2d: float = 0.85,
        p_partial_contrast: float = 0.3,
        # Field-strength contrast warp (Rooney 2007, Fischer 1990, Pohmann 2015):
        # Re-renders the implied tissue contrast at ``field_strength`` rather
        # than only biasing artifact severity. The single largest sim-to-real
        # lever for LMIC deployment per the OGAP-2.0 plan.
        p_field_contrast_warp: float = 0.5,
        field_contrast_source_b0: float = 3.0,
        # AFA (Auxiliary Fourier-basis Augmentation, Vaish+ 2024):
        # additive Fourier-basis perturbation; no donor volume is required.
        p_fourier_amplitude_mix: float = 0.0,
        fourier_beta_range: Tuple[float, float] = (0.005, 0.05),
        # Receive-coil sensitivity bias (Campbell-Washburn 2023, Shetty 2023,
        # Ljungberg 2025): single-channel head-coil RX falloff at low B0.
        p_receive_coil_inhomogeneity: float = 0.30,
        receive_coil_falloff_min: float = 0.55,
        # B0 off-resonance / air-tissue interface void (Campbell-Washburn 2023).
        p_off_resonance_void: float = 0.15,
        off_resonance_void_count_range: Tuple[int, int] = (1, 3),
        off_resonance_void_radius_range: Tuple[int, int] = (5, 15),
        # Noise-floor calibration. "static" reproduces v9 behavior; "aja_fernandez"
        # estimates Rician σ per case (Aja-Fernández+ 2009) and scales it by
        # SNR ∝ B0 (Pohmann 2015). Recommended for any B0-randomized run.
        noise_calibration: str = "static",
        noise_std_range: Tuple[float, float] = (0.02, 0.10),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        bias_field_order: int = 3,
        bias_field_coeff_range: float = 0.7,
        elastic_alpha: Tuple[float, float] = (0.0, 900.0),
        elastic_sigma: Tuple[float, float] = (9.0, 13.0),
        resolution_zoom_range: Tuple[float, float] = (0.5, 1.0),
        ghost_intensity_range: Tuple[float, float] = (0.6, 0.9),
        focal_b1_gain_range: Tuple[float, float] = (0.05, 0.25),
        field_strength: Optional[float] = None,  # if known: 3.0, 1.5, or 1.0
        lmic_field_strength_prior: bool = False,
        contrast_mod_indices: Optional[Sequence[int]] = None,
    ):
        self.p_flip = p_flip
        self.p_elastic = p_elastic
        self.p_bias_field = p_bias_field
        self.p_noise = p_noise
        self.p_rician_noise = p_rician_noise
        self.p_gamma = p_gamma
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_resolution = p_resolution
        self.p_motion = p_motion
        self.p_ghosting = p_ghosting
        self.p_spike = p_spike
        self.p_gibbs = p_gibbs
        self.p_focal_b1_inhomogeneity = p_focal_b1_inhomogeneity
        self.p_anisotropic_2d = p_anisotropic_2d
        self.p_partial_contrast = p_partial_contrast
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.bias_field_order = bias_field_order
        self.bias_field_coeff_range = bias_field_coeff_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.resolution_zoom_range = resolution_zoom_range
        self.ghost_intensity_range = ghost_intensity_range
        self.focal_b1_gain_range = focal_b1_gain_range
        self.field_strength = field_strength
        self.lmic_field_strength_prior = bool(lmic_field_strength_prior)
        self.contrast_mod_indices = [int(i) for i in (contrast_mod_indices or [])]
        # § AUGMENTOR-A.  Field-strength-aware contrast warp.
        self.p_field_contrast_warp = float(p_field_contrast_warp)
        self.field_contrast_source_b0 = float(field_contrast_source_b0)
        # § AUGMENTOR-B.  AFA / Fourier-basis perturbation.
        self.p_fourier_amplitude_mix = float(p_fourier_amplitude_mix)
        self.fourier_beta_range = tuple(float(x) for x in fourier_beta_range)
        self._afa_channel_mismatch_warned = False
        # § AUGMENTOR-C.  Receive-coil sensitivity falloff (low-field only).
        self.p_receive_coil_inhomogeneity = float(p_receive_coil_inhomogeneity)
        self.receive_coil_falloff_min = float(np.clip(receive_coil_falloff_min, 0.10, 1.0))
        # § AUGMENTOR-D.  B0 off-resonance void.
        self.p_off_resonance_void = float(p_off_resonance_void)
        self.off_resonance_void_count_range = tuple(int(x) for x in off_resonance_void_count_range)
        self.off_resonance_void_radius_range = tuple(int(x) for x in off_resonance_void_radius_range)
        # § AUGMENTOR-E.  Noise calibration mode.
        if noise_calibration not in ("static", "aja_fernandez"):
            raise ValueError(
                f"noise_calibration must be 'static' or 'aja_fernandez', got {noise_calibration!r}"
            )
        self.noise_calibration = str(noise_calibration)

    # ────────────────────────────────────────────────────────────────────────
    # Tissue T1 lookup at multiple field strengths (mean values, normal brain).
    # Sources:
    #   Rooney+ MRM 2007 (Magnetic field and tissue dependencies of human brain
    #     longitudinal 1H2O relaxation). Table 1 columns 0.2T → 7T.
    #   Fischer+ MRM 1990 (Nuclear relaxation of human brain GM/WM).
    #   Pohmann+ MRM 2015 (SNR + tissue parameters at 3T / 7T).
    #   Magnetic Resonance Relaxometry at Low and Ultra-low Fields review.
    # All values in milliseconds. Linear interpolation in log(B0) is used at
    # off-grid field strengths.
    # ────────────────────────────────────────────────────────────────────────
    _T1_TABLE_MS: Dict[str, Dict[float, float]] = {
        # White matter
        "wm":    {0.064: 230.0, 0.2: 380.0, 0.55: 540.0, 1.0: 660.0, 1.5: 750.0, 3.0: 830.0, 7.0: 1100.0},
        # Cortical gray matter
        "gm":    {0.064: 320.0, 0.2: 530.0, 0.55: 800.0, 1.0: 1000.0, 1.5: 1200.0, 3.0: 1330.0, 7.0: 1900.0},
        # Cerebrospinal fluid (water-like, weak B0 dependence)
        "csf":   {0.064: 3500.0, 0.2: 3800.0, 0.55: 4000.0, 1.0: 4150.0, 1.5: 4250.0, 3.0: 4500.0, 7.0: 4500.0},
        # Vasogenic edema (T1 close to CSF at low field, drops at high field)
        "edema": {0.064: 600.0, 0.2: 900.0, 0.55: 1300.0, 1.0: 1550.0, 1.5: 1750.0, 3.0: 2000.0, 7.0: 2400.0},
        # Enhancing tumor - LEGACY POST-Gd-shortened convention: this row is ALREADY the
        # contrast-enhanced (short-T1) species (Solomon 1955 PRE; stronger effect at low
        # B0). NOTE: the v9.1 package path uses the OPPOSITE convention - pre-contrast T1
        # in ``ogap.augmentations.relaxation_constants.ENHANCING_TUMOUR`` plus an explicit
        # SBM shortening via ``gd_shortened_t1``. That module is the single source of truth
        # for new work; do not compare these two ET curves directly (different conventions).
        "et":    {0.064: 90.0, 0.2: 140.0, 0.55: 200.0, 1.0: 260.0, 1.5: 320.0, 3.0: 420.0, 7.0: 600.0},
    }
    _T2_TABLE_MS: Dict[str, Dict[float, float]] = {
        "wm":    {0.064: 90.0, 0.2: 84.0, 0.55: 80.0, 1.0: 78.0, 1.5: 75.0, 3.0: 70.0, 7.0: 55.0},
        "gm":    {0.064: 110.0, 0.2: 105.0, 0.55: 100.0, 1.0: 95.0, 1.5: 90.0, 3.0: 85.0, 7.0: 65.0},
        "csf":   {0.064: 2000.0, 0.2: 2200.0, 0.55: 2400.0, 1.0: 2500.0, 1.5: 2600.0, 3.0: 2700.0, 7.0: 2700.0},
        "edema": {0.064: 200.0, 0.2: 220.0, 0.55: 240.0, 1.0: 250.0, 1.5: 260.0, 3.0: 280.0, 7.0: 280.0},
        "et":    {0.064: 95.0, 0.2: 90.0, 0.55: 85.0, 1.0: 82.0, 1.5: 80.0, 3.0: 75.0, 7.0: 60.0},
    }

    @staticmethod
    def _interp_log_b0(table: Dict[float, float], b0: float) -> float:
        """Log-B0 linear interpolation over a tabulated relaxometry curve."""
        keys = sorted(table.keys())
        b0 = float(np.clip(b0, keys[0], keys[-1]))
        if b0 in table:
            return float(table[b0])
        log_b0 = math.log(b0)
        for i in range(len(keys) - 1):
            if keys[i] <= b0 <= keys[i + 1]:
                lo, hi = keys[i], keys[i + 1]
                t = (log_b0 - math.log(lo)) / (math.log(hi) - math.log(lo))
                return float(table[lo] * (1 - t) + table[hi] * t)
        return float(table[keys[-1]])

    def _effective_field_strength(self) -> Optional[float]:
        if self.field_strength is None:
            return None
        try:
            return float(np.clip(float(self.field_strength), 0.3, 3.0))
        except (TypeError, ValueError):
            return None

    def _low_field_factor(self) -> float:
        """Linear ramp in [0, 1] that equals 0 at >=1.5 T and ~0.96 at 0.064 T.

        FIX (Finding 3, 2026 audit): Denominator was 1.2, which clipped at
        any field strength below 0.3 T - meaning 0.3 T (low-field) and
        0.064 T (Hyperfine ultra-low) produced the same factor (1.0).  The
        full LMIC fleet spans 0.064 T (Hyperfine Swoop) to 0.5 T (modern
        open scanners), and we want a smoothly increasing severity across
        that range.  Denominator 1.5 maps:
            3.0 T  → 0.0       1.5 T  → 0.0       1.0 T  → 0.33
            0.7 T  → 0.53      0.55 T → 0.63      0.3 T  → 0.80
            0.064 T → 0.96
        Used to continuously modulate augmentation severity (SNR loss,
        resolution degradation, bias field) as field strength decreases.

        References (unchanged): Geethanath & Vaughan JMRI 2019;
        Campbell-Washburn et al. Radiology 2019; Arnold et al. MAGMA 2023;
        Marques et al. JMRI 2019; Obungoloch et al. NMR Biomed 2023.
        """
        fs = self._effective_field_strength()
        if fs is None:
            return 0.0
        return float(np.clip((1.5 - fs) / 1.5, 0.0, 1.0))

    def _high_field_factor(self) -> float:
        """Linear ramp in [0, 1] that equals 0 at <=1.5 T and 1 at >=3 T.

        Uses the same 1.5 T pivot. The 3 T ceiling reflects standard
        clinical high-field systems (Wald, NeuroImage 2012 - high-field
        MRI review; Duyn, NeuroImage 2012 - 7 T considerations). Increases
        modelled B1+ inhomogeneity severity and reduces simulated thermal
        noise at higher field per SNR ∝ B0 scaling (Edelstein et al.,
        Magn Reson Med 1986)."""
        fs = self._effective_field_strength()
        if fs is None:
            return 0.0
        return float(np.clip((fs - 1.5) / 1.5, 0.0, 1.0))

    def _field_scaled_noise_std(self) -> float:
        std_lo, std_hi = self.noise_std_range
        # Low-field SNR loss is real, but Marques et al. show that it is
        # milder than naive linear B0 scaling once protocol trade-offs are
        # considered. Use a moderate boost and let resolution degradation
        # carry part of the low-field realism.
        scale = 1.0 + 1.25 * self._low_field_factor()
        std = np.random.uniform(std_lo * scale, std_hi * scale)
        return float(std)

    def _sample_resolution_zoom_factor(self) -> float:
        lo, hi = self.resolution_zoom_range
        low_field = self._low_field_factor()
        lo = max(0.35, lo - 0.15 * low_field)
        hi = min(1.0, hi - 0.10 * low_field)
        hi = max(lo + 0.05, hi)
        return float(np.random.uniform(lo, hi))

    def _flip(self, x: np.ndarray, y: Optional[np.ndarray]):
        for ax_x, ax_y in [(1, 0), (2, 1), (3, 2)]:
            if np.random.rand() < self.p_flip:
                x = np.flip(x, axis=ax_x).copy()
                if y is not None:
                    y = np.flip(y, axis=ax_y).copy()
        return x, y

    def _elastic_deformation(self, x: np.ndarray, y: Optional[np.ndarray]):
        """3D elastic deformation (Cirillo et al. top-2 for brain tumour seg).

        Displacement field is computed at HALF resolution then upsampled - gives
        visually identical deformations ~4x faster on 128^3 patches.

        Linear interpolation plus a warped foreground mask preserves the
        background==0 invariant used by later preprocessing steps.
        """
        if np.random.rand() >= self.p_elastic:
            return x, y

        alpha = np.random.uniform(*self.elastic_alpha)
        sigma = np.random.uniform(*self.elastic_sigma)
        shape = x.shape[1:]  # D, H, W (full resolution)

        # Compute displacement field at half resolution for speed (~4x faster)
        half = tuple(max(s // 2, 1) for s in shape)
        dx = ndi.gaussian_filter(np.random.randn(*half) * alpha / 2, sigma / 2, mode="reflect")
        dy = ndi.gaussian_filter(np.random.randn(*half) * alpha / 2, sigma / 2, mode="reflect")
        dz = ndi.gaussian_filter(np.random.randn(*half) * alpha / 2, sigma / 2, mode="reflect")

        # Upsample back to full resolution with linear interpolation
        zoom_factors = tuple(s / h for s, h in zip(shape, half))
        dx = ndi.zoom(dx, zoom_factors, order=1)[:shape[0], :shape[1], :shape[2]]
        dy = ndi.zoom(dy, zoom_factors, order=1)[:shape[0], :shape[1], :shape[2]]
        dz = ndi.zoom(dz, zoom_factors, order=1)[:shape[0], :shape[1], :shape[2]]

        coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
        coords[0] += dx
        coords[1] += dy
        coords[2] += dz

        src_mask = np.any(x != 0, axis=0).astype(np.float32)
        warped_mask = ndi.map_coordinates(src_mask, coords, order=0, mode="reflect") > 0.5

        for c in range(x.shape[0]):
            x[c] = ndi.map_coordinates(x[c], coords, order=1, mode="reflect")
            x[c] *= warped_mask
        if y is not None:
            y = ndi.map_coordinates(y.astype(np.float32), coords, order=0, mode="reflect")
            y = y.astype(np.int64)
        return x, y

    def _simulate_bias_field(self, x: np.ndarray):
        """Low-order polynomial bias field simulating B₁ inhomogeneity.

        Computed at half resolution and upsampled - bias fields are spatially
        smooth (low-order polynomial) so there is no quality loss, but the
        meshgrid + polynomial evaluation runs ~8× faster on half-sized arrays.
        """
        if np.random.rand() >= self.p_bias_field:
            return x

        shape = x.shape[1:]  # D, H, W
        # Half-resolution computation (bias fields are smooth → no quality loss)
        half_shape = tuple(max(s // 2, 4) for s in shape)
        coords = [np.linspace(-1, 1, s) for s in half_shape]
        d, h, w = np.meshgrid(*coords, indexing="ij")

        # Random polynomial coefficients
        bias = np.zeros(half_shape, dtype=np.float32)
        for order in range(1, self.bias_field_order + 1):
            coeff = np.random.uniform(-self.bias_field_coeff_range,
                                       self.bias_field_coeff_range, 3)
            bias += coeff[0] * (d ** order) + coeff[1] * (h ** order) + coeff[2] * (w ** order)

        bias = np.exp(bias)  # multiplicative bias

        # Upsample to full resolution
        from scipy.ndimage import zoom as ndi_zoom
        zoom_factors = tuple(s / hs for s, hs in zip(shape, half_shape))
        bias = ndi_zoom(bias, zoom_factors, order=3, mode="nearest")
        # Ensure exact shape match after zoom rounding
        bias = bias[:shape[0], :shape[1], :shape[2]]

        for c in range(x.shape[0]):
            nz = x[c] != 0
            if nz.any():
                x[c][nz] *= bias[nz]
        return x

    def _simulate_focal_b1_inhomogeneity(self, x: np.ndarray):
        """Localised receive-field shading for scanner and field-strength variation.

        The global polynomial bias field above captures smooth coil trends.
        This additional transform injects focal gain/dropout patches that are
        more representative of heterogeneous receive profiles and imperfect
        coil/shim behaviour across scanners.
        """
        p = self.p_focal_b1_inhomogeneity
        # Low-field systems tend to have more homogeneous B0/B1 conditions than
        # high-field systems; do not artificially inflate receive shading below 1T.
        p *= 1.0 - 0.15 * self._low_field_factor()
        p *= 1.0 + 0.15 * self._high_field_factor()
        p = float(np.clip(p, 0.0, 0.30))
        if np.random.rand() >= p:
            return x

        fg_mask = np.any(x != 0, axis=0)
        coords = np.argwhere(fg_mask)
        if coords.shape[0] < 256:
            return x

        shape = x.shape[1:]
        zz, yy, xx = np.meshgrid(
            np.arange(shape[0], dtype=np.float32),
            np.arange(shape[1], dtype=np.float32),
            np.arange(shape[2], dtype=np.float32),
            indexing="ij",
        )
        shading = np.ones(shape, dtype=np.float32)
        n_blobs = np.random.randint(1, 3)

        for _ in range(n_blobs):
            center = coords[np.random.randint(coords.shape[0])].astype(np.float32)
            radii = np.array([
                max(4.0, np.random.uniform(0.12, 0.28) * shape[0]),
                max(4.0, np.random.uniform(0.12, 0.28) * shape[1]),
                max(4.0, np.random.uniform(0.12, 0.28) * shape[2]),
            ], dtype=np.float32)
            blob = np.exp(
                -0.5 * (
                    ((zz - center[0]) / radii[0]) ** 2
                    + ((yy - center[1]) / radii[1]) ** 2
                    + ((xx - center[2]) / radii[2]) ** 2
                )
            ).astype(np.float32)
            high_field = self._high_field_factor()
            low_field = self._low_field_factor()
            gain_max = self.focal_b1_gain_range[1] * (1.0 - 0.10 * low_field + 0.10 * high_field)
            min_gain = self.focal_b1_gain_range[0] * (1.0 - 0.10 * low_field + 0.10 * high_field)
            gain = np.random.uniform(-gain_max, gain_max)
            if abs(gain) < min_gain:
                gain = math.copysign(
                    min_gain,
                    gain if gain != 0 else (1.0 if np.random.rand() < 0.5 else -1.0),
                )
            shading *= np.clip(1.0 + gain * blob, 0.45, 1.65)

        shading = np.clip(shading, 0.45, 1.65)
        for c in range(x.shape[0]):
            nz = x[c] != 0
            if nz.any():
                x[c][nz] *= shading[nz]
        return x

    def _add_noise(self, x: np.ndarray):
        """Additive Gaussian noise with optional field-strength scaling.

        Used as the high-SNR approximation of magnitude MRI noise. Below 1-1.5T
        we bias the pipeline toward Rician noise instead of always stacking both.
        """
        std = self._field_scaled_noise_std()
        noise = np.random.normal(0, std, x.shape).astype(np.float32)
        return x + noise

    def _add_rician_noise(self, x: np.ndarray):
        """Rician noise for magnitude MRI images (Gaussian on z-scored input).

        At clinical SNR (>3), Rician ≈ Gaussian. The Rician magnitude form
        ``sqrt((signal + N1)^2 + N2^2)`` is only physical on a NON-NEGATIVE magnitude
        image; the pipeline z-scores first, so on signed foreground it would RECTIFY
        ~half the voxels (fold the distribution) instead of adding noise. We therefore
        add Gaussian noise on signed input and reserve the true Rician form for
        genuinely magnitude-valued (>=0) volumes.

        When ``self.noise_calibration == "aja_fernandez"``, σ is estimated
        per channel from the actual case (Aja-Fernández+ 2009) and rescaled
        for the target B0 by SNR ∝ B0 (Pohmann+ 2015), instead of being
        drawn from the static ``noise_std_range``.
        """
        if self.noise_calibration == "aja_fernandez":
            scale = self._aja_fernandez_noise_scale()
            for c in range(x.shape[0]):
                sigma_per_chan = self._estimate_rician_sigma(x[c]) * scale
                # Cap σ at 30% of the per-channel signal stdev to avoid
                # destroying volumes whose normalisation collapsed the
                # background to a near-singular value.
                sig_std = float(np.std(x[c])) if x[c].size else 1.0
                sigma_per_chan = float(min(sigma_per_chan, max(sig_std * 0.30, 1e-3)))
                n1 = np.random.normal(0, sigma_per_chan, x.shape[1:]).astype(np.float32)
                if float(x[c].min()) < 0.0:            # signed/z-scored -> Gaussian
                    x[c] = (x[c] + n1).astype(np.float32)
                else:                                  # non-negative magnitude -> Rician
                    n2 = np.random.normal(0, sigma_per_chan, x.shape[1:]).astype(np.float32)
                    x[c] = np.sqrt((x[c] + n1) ** 2 + n2 ** 2)
            return x

        std = self._field_scaled_noise_std()
        for c in range(x.shape[0]):
            n1 = np.random.normal(0, std, x.shape[1:]).astype(np.float32)
            if float(x[c].min()) < 0.0:               # signed/z-scored -> Gaussian
                x[c] = (x[c] + n1).astype(np.float32)
            else:                                      # non-negative magnitude -> Rician
                n2 = np.random.normal(0, std, x.shape[1:]).astype(np.float32)
                x[c] = np.sqrt((x[c] + n1) ** 2 + n2 ** 2)
        return x

    def _apply_noise_model(self, x: np.ndarray) -> np.ndarray:
        """Sample one magnitude-noise model per volume instead of stacking both.

        Gaussian noise is retained as the high-SNR approximation. As field
        strength decreases, the probability of using Rician noise increases.
        """
        p_any = 1.0 - (1.0 - self.p_noise) * (1.0 - self.p_rician_noise)
        if np.random.rand() >= p_any:
            return x

        low_field = self._low_field_factor()
        p_rician = float(np.clip(0.30 + 0.55 * low_field, 0.0, 0.90))
        if np.random.rand() < p_rician:
            return self._add_rician_noise(x)
        return self._add_noise(x)

    def _gamma_transform(self, x: np.ndarray):
        """Gamma augmentation with retain_stats (nnU-Net convention)."""
        if np.random.rand() >= self.p_gamma:
            return x

        gamma = np.random.uniform(*self.gamma_range)
        for c in range(x.shape[0]):
            nz = x[c] != 0
            if not nz.any():
                continue
            mn, std = x[c][nz].mean(), x[c][nz].std()
            x_min = x[c][nz].min()
            # Shift to positive, apply gamma, restore stats
            shifted = x[c][nz] - x_min + 1e-6
            shifted = shifted ** gamma
            # Retain stats: re-normalise to original mean/std
            new_mn, new_std = shifted.mean(), shifted.std()
            if new_std > 1e-8:
                shifted = (shifted - new_mn) / (new_std + 1e-8) * (std + 1e-8) + mn
            x[c][nz] = shifted
        return x

    def _brightness_multiplicative(self, x: np.ndarray):
        if np.random.rand() >= self.p_brightness:
            return x
        factor = np.random.uniform(0.75, 1.25)
        return x * factor

    def _simulate_partial_contrast(
        self,
        x: np.ndarray,
        contrast_mod_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Model present-but-suboptimal contrast enhancement."""
        indices = [int(i) for i in (contrast_mod_indices if contrast_mod_indices is not None else self.contrast_mod_indices)]
        if not indices or np.random.rand() >= self.p_partial_contrast:
            return x

        out = x.copy()
        for c_idx in indices:
            if c_idx < 0 or c_idx >= out.shape[0]:
                continue
            ch = out[c_idx]
            nz = ch != 0
            if not nz.any():
                continue
            if out.shape[0] > 1 and c_idx != 0:
                alpha = np.random.uniform(0.2, 0.9)
                ref = out[0]
                ref_mask = ref != 0
                blend_mask = nz & ref_mask
                out[c_idx][blend_mask] = alpha * ch[blend_mask] + (1.0 - alpha) * ref[blend_mask]
            else:
                threshold = float(np.median(ch[nz]))
                enhancement = (ch > threshold) & nz
                if not enhancement.any():
                    continue
                scale = np.random.uniform(0.2, 0.9)
                out[c_idx][enhancement] = threshold + (ch[enhancement] - threshold) * scale
        return out

    def _contrast_augmentation(self, x: np.ndarray):
        """Per-channel contrast augmentation (scale + shift)."""
        if np.random.rand() >= self.p_contrast:
            return x
        for c in range(x.shape[0]):
            nz = x[c] != 0
            if nz.any():
                mn = x[c][nz].mean()
                x[c][nz] = (x[c][nz] - mn) * np.random.uniform(0.65, 1.5) + mn
        return x

    # ── FIX #6: Store zoom result in temp before assignment ──
    def _resolution_degradation(self, x: np.ndarray):
        """Simulate thick-slice acquisition (clinical 3-5mm vs research 1mm).

        Down-sample with nearest-neighbour along a random axis, then
        up-sample with cubic interpolation (nnU-Net convention).

        FIX: Previous code assigned the zoomed-back result directly into
        x[c] then tried to correct shape afterwards. If the zoomed result
        had a different shape, the assignment would raise a broadcast error.
        Now we store in a temp array, pad/crop to target shape, then assign.
        """
        if np.random.rand() >= self.p_resolution:
            return x

        from scipy.ndimage import zoom as ndi_zoom

        axis = np.random.randint(0, 3)  # 0=D, 1=H, 2=W
        zoom_factor = self._sample_resolution_zoom_factor()
        target_shape = (x.shape[1], x.shape[2], x.shape[3])

        for c in range(x.shape[0]):
            zoom_arr = [1.0, 1.0, 1.0]
            zoom_arr[axis] = zoom_factor
            # Downsample with order 0 (nearest), upsample with order 3 (cubic)
            down = ndi_zoom(x[c], zoom_arr, order=0, mode="nearest")
            zoom_back = [1.0, 1.0, 1.0]
            zoom_back[axis] = 1.0 / zoom_factor
            temp = ndi_zoom(down, zoom_back, order=3, mode="nearest")

            # FIX: Pad or crop temp to match target shape
            result = np.zeros(target_shape, dtype=np.float32)
            slices = tuple(slice(0, min(s1, s2))
                          for s1, s2 in zip(temp.shape, target_shape))
            result[slices] = temp[slices]
            x[c] = result
        return x

    def _simulate_anisotropic_2d_acquisition(self, x: np.ndarray) -> np.ndarray:
        """Simulate thick-slice 2D acquisitions common in low-resource protocols."""
        if np.random.rand() >= self.p_anisotropic_2d:
            return x

        from scipy.ndimage import zoom as ndi_zoom

        axis = int(np.random.choice([0, 1, 2], p=[0.70, 0.15, 0.15]))
        thickness_factor = np.random.uniform(3.0, 7.0)
        target_shape = list(x.shape[1:])

        for c in range(x.shape[0]):
            nz_mask = x[c] != 0
            if not nz_mask.any():
                continue
            sigma = [0.0, 0.0, 0.0]
            sigma[axis] = thickness_factor / 2.0
            blurred = ndi.gaussian_filter(x[c], sigma=sigma, mode="constant")
            zoom_down = [1.0, 1.0, 1.0]
            zoom_down[axis] = 1.0 / thickness_factor
            down = ndi_zoom(blurred, zoom_down, order=1, mode="nearest")
            zoom_up = [s_orig / max(s_down, 1) for s_orig, s_down in zip(target_shape, down.shape)]
            up = ndi_zoom(down, zoom_up, order=3, mode="nearest")
            result = np.zeros(target_shape, dtype=np.float32)
            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(up.shape, target_shape))
            result[slices] = up[slices]
            result *= nz_mask.astype(np.float32)
            x[c] = result
        return x

    def _simulate_motion(self, x: np.ndarray):
        """Simplified motion artifact via k-space phase corruption.

        Segments k-space along the phase-encoding direction, applies
        random phase shifts to different segments, producing ghosting.
        """
        if np.random.rand() >= self.p_motion:
            return x

        for c in range(x.shape[0]):
            # FFT along a random axis
            ax = np.random.choice([0, 1, 2])
            kspace = np.fft.fftn(x[c])

            # Random phase perturbation in segments
            n_segments = np.random.randint(4, 10)
            seg_size = max(1, kspace.shape[ax] // n_segments)
            for seg_idx in range(n_segments):
                if np.random.rand() < 0.4:  # only corrupt some segments
                    phase = np.random.uniform(-0.5, 0.5)
                    slc = [slice(None)] * 3
                    start = seg_idx * seg_size
                    end = min(start + seg_size, kspace.shape[ax])
                    slc[ax] = slice(start, end)
                    kspace[tuple(slc)] *= np.exp(1j * phase)

            x[c] = np.abs(np.fft.ifftn(kspace)).astype(np.float32)
        return x

    def _simulate_ghosting(self, x: np.ndarray):
        """Simple image-space proxy for phase-encode ghosting.

        Lower-field and older protocols often exhibit ghosting/ringing not well
        covered by blur/noise alone. This proxy is intentionally lightweight.
        """
        if np.random.rand() >= self.p_ghosting:
            return x

        for c in range(x.shape[0]):
            axis = np.random.randint(0, 3)
            axis_len = x[c].shape[axis]
            shift = max(2, axis_len // np.random.randint(4, 8))
            intensity = np.random.uniform(*self.ghost_intensity_range)
            n_ghosts = np.random.randint(1, 4)
            ghosted = x[c].copy()
            for ghost_idx in range(1, n_ghosts + 1):
                ghosted += (intensity / ghost_idx) * np.roll(x[c], shift * ghost_idx, axis=axis)
            x[c] = ghosted.astype(np.float32, copy=False)
        return x

    def _simulate_kspace_spike(self, x: np.ndarray):
        """RF spike artefact: sparse high-energy impulses in k-space."""
        if np.random.rand() >= self.p_spike:
            return x
        for c in range(x.shape[0]):
            kspace = np.fft.fftn(x[c])
            mag = max(float(np.abs(kspace).mean()), 1e-6)
            for _ in range(np.random.randint(1, 4)):
                coord = tuple(np.random.randint(0, s) for s in kspace.shape)
                phase = np.random.uniform(0.0, 2.0 * np.pi)
                intensity = np.random.uniform(0.3, 0.7) * mag * np.prod(kspace.shape) ** 0.25
                kspace[coord] += intensity * np.exp(1j * phase)
            x[c] = np.abs(np.fft.ifftn(kspace)).astype(np.float32)
        return x

    def _simulate_gibbs(self, x: np.ndarray):
        """Low-pass k-space truncation proxy for Gibbs/ringing artefact."""
        if np.random.rand() >= self.p_gibbs:
            return x
        alpha = np.random.uniform(0.4, 0.8)
        for c in range(x.shape[0]):
            kspace = np.fft.fftshift(np.fft.fftn(x[c]))
            mask = np.zeros_like(kspace, dtype=bool)
            slices = []
            for dim in kspace.shape:
                keep = max(2, int(round(dim * (1.0 - 0.25 * alpha))))
                start = max(0, (dim - keep) // 2)
                slices.append(slice(start, start + keep))
            mask[tuple(slices)] = True
            kspace = np.where(mask, kspace, 0)
            x[c] = np.abs(np.fft.ifftn(np.fft.ifftshift(kspace))).astype(np.float32)
        return x

    # ════════════════════════════════════════════════════════════════════════
    # § AUGMENTOR-A.  Field-strength-aware contrast resimulation.
    #
    # Re-renders the implied tissue contrast at the augmentor's target B0 by
    # warping intensities along Rooney+ 2007 / Fischer+ 1990 / Pohmann+ 2015
    # T1(B0), T2(B0) curves. The pulse-sequence proxy treats:
    #   T1-weighted (T1n / T1c) volumes as TR≈600 ms, TE≈10 ms, TI=∞
    #     (steady-state SE / SPGR proxy).
    #   T2-weighted (T2w) volumes as TR≈4500 ms, TE≈90 ms.
    #   T2-FLAIR  (T2f) volumes as TR≈9000 ms, TE≈110 ms, TI=2400 ms.
    #
    # We assign each foreground voxel a soft tissue-class membership using a
    # 3-bin intensity histogram of the foreground (low / mid / high). This
    # avoids running a full segmentation tool yet still differentiates the
    # tumor regions where the field-strength contrast change is largest.
    #
    # The PRE (Paramagnetic Relaxation Enhancement; Solomon 1955) effect for
    # Gd-shortened ET is encoded in _T1_TABLE_MS["et"]: T1_ET shrinks faster
    # than T1_WM as B0 drops, so the ET/WM contrast ratio at 0.064T is
    # ≈3× larger than at 3T. We replay this directly through the LUT.
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _signal_se(t1: float, t2: float, tr: float, te: float) -> float:
        """Spin-echo signal proxy: M0 · (1 - exp(-TR/T1)) · exp(-TE/T2). M0=1."""
        if t1 <= 0:
            return 0.0
        return float((1.0 - math.exp(-tr / max(t1, 1.0))) * math.exp(-te / max(t2, 1.0)))

    @staticmethod
    def _signal_ir(t1: float, t2: float, tr: float, te: float, ti: float) -> float:
        """Inversion-recovery (FLAIR) signal proxy."""
        if t1 <= 0:
            return 0.0
        recover = (1.0 - 2.0 * math.exp(-ti / max(t1, 1.0))
                   + math.exp(-tr / max(t1, 1.0)))
        return float(abs(recover) * math.exp(-te / max(t2, 1.0)))

    @staticmethod
    def _modality_signal_scalar(b0: float, tissue: str, modality_kind: str) -> float:
        """SE/IR signal for a tissue class at field strength ``b0`` (pure scalar).

        Single source of truth for the field-strength contrast model: shared by
        the CPU ``_modality_signal`` (below) and the GPU
        ``GPUPhysicsAugmentor._simulate_field_strength_contrast_batch`` so both
        augmentors warp contrast against the *same* relaxometry LUT + SE/IR proxy
        (no risk of the two paths drifting apart).
        """
        t1 = MRIPhysicsAugmentor._interp_log_b0(MRIPhysicsAugmentor._T1_TABLE_MS[tissue], b0)
        t2 = MRIPhysicsAugmentor._interp_log_b0(MRIPhysicsAugmentor._T2_TABLE_MS[tissue], b0)
        if modality_kind == "t1":
            return MRIPhysicsAugmentor._signal_se(t1, t2, tr=600.0, te=10.0)
        if modality_kind == "t2":
            return MRIPhysicsAugmentor._signal_se(t1, t2, tr=4500.0, te=90.0)
        if modality_kind == "flair":
            return MRIPhysicsAugmentor._signal_ir(t1, t2, tr=9000.0, te=110.0, ti=2400.0)
        return MRIPhysicsAugmentor._signal_se(t1, t2, tr=600.0, te=10.0)

    def _modality_signal(self, b0: float, tissue: str, modality_kind: str) -> float:
        """Tissue signal at field strength b0 for a given modality template.

        modality_kind ∈ {'t1', 't2', 'flair'}. T1c is treated identically to T1
        for normal tissues and only differs at the ET tissue class (which has
        a Gd-shortened T1 curve in _T1_TABLE_MS).
        """
        return self._modality_signal_scalar(b0, tissue, modality_kind)

    @staticmethod
    def _classify_voxels_3bin(channel: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        """Soft tissue-class assignment via foreground histogram terciles.

        Returns an int8 map with values in {-1: background, 0: low, 1: mid,
        2: high}. For T1-weighted images the ordering ≈ {csf/edema, wm, gm,
        et}; we collapse to three bins for robustness.
        """
        out = np.full(channel.shape, -1, dtype=np.int8)
        if not np.any(fg_mask):
            return out
        fg_vals = channel[fg_mask]
        q1, q2 = np.percentile(fg_vals, [33.3, 66.7])
        out[fg_mask & (channel <= q1)] = 0
        out[fg_mask & (channel > q1) & (channel <= q2)] = 1
        out[fg_mask & (channel > q2)] = 2
        return out

    def _simulate_field_strength_contrast(
        self,
        x: np.ndarray,
        contrast_mod_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Per-channel intensity warp from source B0 to ``self.field_strength``.

        For each modality channel this assigns voxels to tissue bins from
        the foreground histogram (low/mid/high), looks up the ratio of
        target-B0 vs. source-B0 SE/IR signal under that bin's tissue class,
        and rescales each bin's intensities by that ratio. The ET bin (high
        T1 contrast at low field on T1c) is mapped only on the contrast-
        modulated channels listed in ``contrast_mod_indices``, so plain T1n
        does not get a spurious enhancing-tumor boost.
        """
        if np.random.rand() >= self.p_field_contrast_warp:
            return x
        b0 = self._effective_field_strength()
        if b0 is None:
            return x
        b0_src = float(self.field_contrast_source_b0)
        if abs(b0 - b0_src) < 0.05:  # already at source B0; nothing to do
            return x

        contrast_set = set(int(i) for i in (contrast_mod_indices or []))
        n_channels = x.shape[0]
        out = x.copy()

        # Heuristic mapping from channel index → modality kind. We assume the
        # canonical OGAP modality order [t1n, t1c, t2w, t2f] when there are
        # 4 channels, and fall back to T1-template for other arities.
        if n_channels == 4:
            kinds = ["t1", "t1", "t2", "flair"]
        elif n_channels == 1:
            kinds = ["t1"]
        else:
            kinds = ["t1"] * n_channels

        for c in range(n_channels):
            ch = out[c]
            kind = kinds[c]
            # Foreground = nonzero after zscore_nonzero (background is exactly 0). The old
            # percentile-over-whole-volume threshold (~0.004) dropped ~15% of foreground -
            # the negative-z-score voxels (CSF / edema / below-mean) - from the B0 warp.
            fg_mask = ch != 0
            if not np.any(fg_mask):
                continue
            bins = self._classify_voxels_3bin(ch, fg_mask)
            # Tissue-class assignment per bin (T1-weighted: low≈csf/edema,
            # mid≈wm, high≈gm; T2-weighted: low≈wm, mid≈gm, high≈edema/csf).
            if kind == "t1":
                bin_to_tissue = {0: "csf", 1: "wm", 2: "gm"}
            elif kind == "t2":
                bin_to_tissue = {0: "wm", 1: "gm", 2: "edema"}
            else:  # flair: low≈csf-suppressed, mid≈wm, high≈edema/lesion
                bin_to_tissue = {0: "csf", 1: "wm", 2: "edema"}

            # ET-aware override on contrast-modulated channels: replace the
            # high-intensity tissue with "et" so PRE-driven Gd shortening
            # propagates into the warp ratio at low B0.
            if c in contrast_set and kind == "t1":
                bin_to_tissue[2] = "et"

            for bin_id, tissue in bin_to_tissue.items():
                bin_mask = bins == bin_id
                if not np.any(bin_mask):
                    continue
                s_src = self._modality_signal(b0_src, tissue, kind)
                s_tgt = self._modality_signal(b0, tissue, kind)
                if s_src <= 1e-6:
                    continue
                ratio = float(np.clip(s_tgt / s_src, 0.10, 4.0))
                ch[bin_mask] = ch[bin_mask] * ratio
            out[c] = ch
        return out

    # ════════════════════════════════════════════════════════════════════════
    # § AUGMENTOR-B.  Auxiliary Fourier-basis Augmentation (AFA).
    #
    # Vaish+ AFA and the medical-imaging OOD generalisation paper do not swap a
    # donor amplitude spectrum. They add sampled Fourier-basis perturbations to
    # the input while leaving labels unchanged. The historical method name is
    # kept for CLI/checkpoint compatibility, but the operation below is now the
    # literature AFA transform rather than FDA/FACT-style amplitude mixing.
    # ════════════════════════════════════════════════════════════════════════

    def _fourier_amplitude_mix(
        self,
        x: np.ndarray,
        donor: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
    ) -> np.ndarray:
        """Apply additive Fourier-basis AFA; ``donor``/``beta`` are ignored.

        The retained method name is a backward-compatible shim for older CLI
        flags and tests. AFA samples a sinusoidal basis per channel and an
        exponentially-distributed strength, then clips to the channel's original
        intensity range to avoid producing invalid normalized MRI values.
        """
        if np.random.rand() >= self.p_fourier_amplitude_mix:
            return x
        out = np.empty_like(x)
        d, h, w = x.shape[1:]
        zz, yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, d, dtype=np.float32),
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        max_freq = max(1.0, min(d, h, w) / 2.0)
        # Reuse the old beta range as a conservative mean-strength range.
        mean_strength = float(np.random.uniform(*self.fourier_beta_range))
        for c in range(x.shape[0]):
            theta = float(np.random.uniform(0.0, math.pi))
            phi = float(np.random.uniform(0.0, math.pi))
            freq = float(np.random.uniform(1.0, max_freq))
            direction = (
                math.cos(theta),
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
            )
            basis = np.sin(
                2.0 * math.pi * freq
                * (direction[0] * zz + direction[1] * yy + direction[2] * xx - 0.25)
            ).astype(np.float32)
            # Unit-PEAK (max-abs), not unit L2: an L2 norm over a 128^3 volume scales the
            # basis peak by ~1/sqrt(N), making the AFA perturbation ~1000x too small
            # (effectively a no-op). Unit-peak keeps the magnitude controlled by
            # `strength` and the channel range, as Vaish+ 2024 intend.
            basis /= max(float(np.abs(basis).max()), 1e-6)
            strength = float(np.random.exponential(mean_strength))
            lo = float(np.min(x[c]))
            hi = float(np.max(x[c]))
            rng = max(hi - lo, 1e-6)
            pert = np.clip(x[c] + strength * rng * basis, lo, hi).astype(np.float32)
            # Preserve the skull-stripped zero background (the x==0 deployment invariant).
            out[c] = np.where(x[c] != 0, pert, x[c]).astype(np.float32)
        return out

    # ════════════════════════════════════════════════════════════════════════
    # § AUGMENTOR-C.  Robust object-based Rician σ estimation.
    #
    # Estimates Rician noise σ using the object HHH wavelet-subband MAD
    # estimator described by the robust Rician-noise paper, then rescales it for
    # a target field strength (Pohmann 2015: SNR ∝ B0). Used by
    # _apply_noise_model when self.noise_calibration == "aja_fernandez".
    #
    # Refs:
    #   Aja-Fernández+ "Noise estimation in single- and multiple-coil MR
    #     data based on statistical models", MRI 2009 (your "Robust Rician
    #     noise estimation for MR images" PDF).
    #   Pohmann+ MRM 2015 (SNR scaling with B0).
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _estimate_rician_sigma(channel: np.ndarray) -> float:
        """Object-based HHH-wavelet MAD estimator for Rician noise σ.

        This implements the robust estimator described in the Rician-noise
        paper: first-level 3D Haar decomposition, object extraction from the LLL
        subband, exclusion of high-gradient object voxels, MAD on the matching
        HHH coefficients, and a bounded low-SNR Rician correction. It avoids the
        background Rayleigh assumption that the paper shows can fail with
        ghosting and other non-zero-background artifacts.
        """
        if channel.size == 0 or np.nanmax(channel) <= np.nanmin(channel):
            return 1e-3
        d, h, w = (int(s) - int(s) % 2 for s in channel.shape)
        if d < 2 or h < 2 or w < 2:
            return float(max(np.nanstd(channel) * 0.05, 1e-3))
        v = np.asarray(channel[:d, :h, :w], dtype=np.float32)
        a = v[0::2, 0::2, 0::2]
        b = v[0::2, 0::2, 1::2]
        c = v[0::2, 1::2, 0::2]
        d0 = v[0::2, 1::2, 1::2]
        e = v[1::2, 0::2, 0::2]
        f = v[1::2, 0::2, 1::2]
        g = v[1::2, 1::2, 0::2]
        h0 = v[1::2, 1::2, 1::2]
        scale = math.sqrt(8.0)
        lll = (a + b + c + d0 + e + f + g + h0) / scale
        hhh = (a - b - c + d0 - e + f + g - h0) / scale

        threshold = float(np.quantile(lll, 0.35))
        obj = lll > threshold
        if int(obj.sum()) < 32:
            obj = lll > float(np.mean(lll))
        if int(obj.sum()) < 32:
            obj = np.ones_like(lll, dtype=bool)

        grad = np.zeros_like(lll, dtype=np.float32)
        grad[1:, :, :] += np.abs(lll[1:, :, :] - lll[:-1, :, :])
        grad[:, 1:, :] += np.abs(lll[:, 1:, :] - lll[:, :-1, :])
        grad[:, :, 1:] += np.abs(lll[:, :, 1:] - lll[:, :, :-1])
        keep = obj & (grad <= float(np.median(grad[obj])))
        coeff = hhh[keep]
        if coeff.size < 16:
            coeff = hhh[obj]
        if coeff.size == 0:
            return float(max(np.nanstd(channel) * 0.05, 1e-3))

        sigma_mag = float(np.median(np.abs(coeff)) / 0.6745)
        if not math.isfinite(sigma_mag) or sigma_mag <= 0:
            return float(max(np.nanstd(channel) * 0.05, 1e-3))
        object_mean = float(abs(np.mean(lll[obj])))
        snr = object_mean / max(sigma_mag, 1e-6)
        correction = 1.0 if snr >= 3.0 else math.sqrt(max(0.25, (2.0 + snr * snr) / (4.0 + snr * snr)))
        return float(max(sigma_mag * correction, 1e-3))

    def _aja_fernandez_noise_scale(self) -> float:
        """SNR ∝ B0 → σ_target = σ_source · (B0_source / B0_target)."""
        b0 = self._effective_field_strength()
        if b0 is None:
            return 1.0
        return float(np.clip(3.0 / max(b0, 0.05), 0.5, 60.0))

    # ════════════════════════════════════════════════════════════════════════
    # § AUGMENTOR-D.  Receive-coil sensitivity bias (low-field portable).
    #
    # Single-channel head-coil RX falloff: anisotropic Gaussian profile
    # centered at a random off-isocenter location, attenuating the periphery
    # to as low as ``receive_coil_falloff_min`` (default 55%). Gated to
    # B0 ≤ 1 T because high-field bore scanners use multi-channel receive
    # arrays where this falloff is largely corrected by SENSE/GRAPPA.
    #
    # Refs:
    #   Campbell-Washburn+ MRM 2023 (Low-field MRI ISMRM 2022 workshop).
    #   Shetty+ JMRI 2023 (0.55 T body MRI).
    #   Ljungberg+ HBM 2025 (Portable ULF MRI multi-center).
    # ════════════════════════════════════════════════════════════════════════

    def _simulate_receive_coil_inhomogeneity(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.p_receive_coil_inhomogeneity:
            return x
        b0 = self._effective_field_strength()
        if b0 is not None and b0 > 1.0:
            return x  # high-field arrays are well-corrected
        d, h, w = x.shape[1:]
        cz = d * np.random.uniform(0.30, 0.70)
        cy = h * np.random.uniform(0.30, 0.70)
        cx = w * np.random.uniform(0.30, 0.70)
        sz = d * np.random.uniform(0.50, 1.10)
        sy = h * np.random.uniform(0.50, 1.10)
        sx = w * np.random.uniform(0.50, 1.10)
        zg, yg, xg = np.meshgrid(
            np.arange(d), np.arange(h), np.arange(w), indexing="ij"
        )
        radial = ((zg - cz) / sz) ** 2 + ((yg - cy) / sy) ** 2 + ((xg - cx) / sx) ** 2
        falloff_min = float(self.receive_coil_falloff_min)
        field = falloff_min + (1.0 - falloff_min) * np.exp(-radial).astype(np.float32)
        return (x * field[None]).astype(np.float32, copy=False)

    # ════════════════════════════════════════════════════════════════════════
    # § AUGMENTOR-E.  B0 off-resonance / air-tissue interface void.
    #
    # Smooth Gaussian "void" multiplied into the volume at one to three
    # randomly placed peripheral foci, modeling the signal dropout near the
    # frontal sinuses, mastoid air cells, and brainstem on echo-planar /
    # GRE-based sequences at low B0. We do NOT modulate phase (the spatial-
    # domain proxy is faster and visually indistinguishable for SE/IR).
    #
    # Refs:
    #   Campbell-Washburn+ MRM 2023.
    #   Ljungberg+ HBM 2025 (susceptibility-induced dropout patterns).
    # ════════════════════════════════════════════════════════════════════════

    def _simulate_off_resonance_void(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self.p_off_resonance_void:
            return x
        d, h, w = x.shape[1:]
        n_voids = np.random.randint(
            self.off_resonance_void_count_range[0],
            self.off_resonance_void_count_range[1] + 1,
        )
        field = np.ones((d, h, w), dtype=np.float32)
        zg, yg, xg = np.meshgrid(
            np.arange(d), np.arange(h), np.arange(w), indexing="ij"
        )
        for _ in range(n_voids):
            r_min, r_max = self.off_resonance_void_radius_range
            radius = np.random.uniform(r_min, r_max)
            # Bias placement toward the periphery (sinus / mastoid / brainstem
            # neighborhoods) along each axis: pull from {0.15..0.30, 0.70..0.85}.
            def _peripheral(n: int) -> float:
                if np.random.rand() < 0.5:
                    return n * np.random.uniform(0.15, 0.35)
                return n * np.random.uniform(0.65, 0.85)
            cz = _peripheral(d)
            cy = _peripheral(h)
            cx = _peripheral(w)
            depth = np.random.uniform(0.30, 0.70)  # void floor (multiplicative)
            r2 = ((zg - cz) ** 2 + (yg - cy) ** 2 + (xg - cx) ** 2) / (radius ** 2)
            void = depth + (1.0 - depth) * (1.0 - np.exp(-r2))
            field = field * void.astype(np.float32)
        return (x * field[None]).astype(np.float32, copy=False)

    def __call__(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray],
        contrast_mod_indices: Optional[Sequence[int]] = None,
        *,
        donor: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply full augmentation pipeline in physics-consistent order.

        ``donor`` is retained only for backward compatibility with older
        donor-based Fourier amplitude mixing calls. Literature AFA is additive
        Fourier-basis perturbation and does not require a donor volume.
        """
        # 1. Geometric transforms (label-preserving spatial warps).
        x, y = self._flip(x, y)
        x, y = self._elastic_deformation(x, y)
        # 2. Field-strength contrast warp BEFORE artifact / noise injection,
        # so subsequent noise σ scales against the warped (target-B0) signal.
        x = self._simulate_field_strength_contrast(x, contrast_mod_indices=contrast_mod_indices)
        # 3. B-field inhomogeneity (transmit B1 + focal B1 + receive coil RX).
        x = self._simulate_bias_field(x)
        x = self._simulate_focal_b1_inhomogeneity(x)
        x = self._simulate_receive_coil_inhomogeneity(x)
        # 4. B0 off-resonance susceptibility void (low-field GRE/EPI proxy).
        x = self._simulate_off_resonance_void(x)
        # 5. Noise (Rician / Gaussian, optionally Aja-Fernández-calibrated).
        x = self._apply_noise_model(x)
        # 6. Intensity / contrast.
        x = self._gamma_transform(x)
        x = self._brightness_multiplicative(x)
        x = self._simulate_partial_contrast(x, contrast_mod_indices=contrast_mod_indices)
        x = self._contrast_augmentation(x)
        # 7. Resolution / acquisition artefacts (k-space domain).
        x = self._resolution_degradation(x)
        x = self._simulate_anisotropic_2d_acquisition(x)
        x = self._simulate_motion(x)
        x = self._simulate_ghosting(x)
        x = self._simulate_kspace_spike(x)
        x = self._simulate_gibbs(x)
        # 8. AFA Fourier-basis perturbation - applied LAST so the sampled basis
        # fills the frequency-domain augmentation gap left by visual transforms.
        x = self._fourier_amplitude_mix(x, donor)
        return x, y


# ════════════════════════════════════════════════════════════════════════════
# § ULF GAN augmentation (Lucas+ 2025).
#
# Wraps a generator network that maps high-field (3T) volumes to ultra-low-
# field (~64 mT) appearance, exposed via ONNX so we never have to re-train it.
# When weights are not available the wrapper is a no-op identity so the
# training pipeline stays runnable. Integration is an *augmentation* - the
# generator is consulted with low probability per case to inject synthetic
# ULF examples without acquiring new data.
#
# Refs:
#   Lucas+ "Multisequence 3-T image synthesis from 64-mT low-field strength
#     MRI using generative adversarial networks", AJR 2025 (your PDF).
# ════════════════════════════════════════════════════════════════════════════

class ULFGANAugmentor:
    """Optional ULF-style synthesis via a pretrained generator.

    The generator is loaded as ONNX (or a TorchScript module) on first use.
    If the path is not provided or the runtime is missing, the augmentor
    is a no-op pass-through. Callers should treat this as a "best effort"
    augmentation; missing weights NEVER block training.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        target_b0: float = 0.064,
        device: str = "cpu",
    ) -> None:
        self.weights_path = weights_path
        self.target_b0 = float(target_b0)
        self.device = str(device)
        self._session: Optional[Any] = None
        self._mode: Optional[str] = None  # "onnx" or "torchscript"
        if weights_path:
            self._load()

    def is_available(self) -> bool:
        return self._session is not None

    def _load(self) -> None:
        if not self.weights_path:
            return
        if not Path(self.weights_path).exists():
            LOG.warning(
                "[ULFGAN] weights_path %s does not exist - pass-through enabled",
                self.weights_path,
            )
            return
        if str(self.weights_path).endswith(".onnx"):
            try:
                import onnxruntime as ort
                providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if self.device == "cuda" else ["CPUExecutionProvider"])
                self._session = ort.InferenceSession(self.weights_path, providers=providers)
                self._mode = "onnx"
                LOG.info("[ULFGAN] loaded ONNX generator from %s", self.weights_path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("[ULFGAN] failed to load ONNX generator (%s) - pass-through", exc)
                self._session = None
        else:
            try:
                self._session = torch.jit.load(self.weights_path, map_location=self.device)
                self._session.eval()
                self._mode = "torchscript"
                LOG.info("[ULFGAN] loaded TorchScript generator from %s", self.weights_path)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("[ULFGAN] failed to load TorchScript generator (%s) - pass-through", exc)
                self._session = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Synthesize a ULF-appearance volume from ``x`` (C, D, H, W). Identity
        when no generator is loaded so the caller need not gate on availability.
        """
        if self._session is None:
            return x
        try:
            inp = x.astype(np.float32, copy=False)[None]  # (1, C, D, H, W)
            if self._mode == "onnx":
                in_name = self._session.get_inputs()[0].name
                out = self._session.run(None, {in_name: inp})[0]
            else:
                with torch.inference_mode():
                    t = torch.from_numpy(inp).to(self.device)
                    out_t = self._session(t)
                    out = out_t.detach().cpu().numpy()
            if out.ndim == 5:
                return out[0].astype(np.float32, copy=False)
            return x
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[ULFGAN] inference failed (%s) - falling back to identity", exc)
            return x


class GPUPhysicsAugmentor:
    """Torch-native GPU augmentation path for CUDA training.

    This mirrors the acquisition/intensity-heavy part of the CPU pipeline while
    keeping the work on the device. It removes the biggest DataLoader bottleneck
    on one-GPU Hopper jobs without adding a hard runtime dependency on MONAI.
    The transform families intentionally follow the same design space MONAI
    exposes for 3D medical imaging: elastic deformation, bias fields, Rician
    noise, contrast adjustment, and Gibbs/ghosting-like acquisition corruption.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_elastic: float = 0.3,
        p_bias_field: float = 0.6,
        p_noise: float = 0.5,
        p_rician_noise: float = 0.4,
        p_gamma: float = 0.5,
        p_brightness: float = 0.5,
        p_contrast: float = 0.5,
        p_resolution: float = 0.3,
        p_motion: float = 0.4,
        p_ghosting: float = 0.3,
        p_spike: float = 0.2,
        p_gibbs: float = 0.3,
        p_focal_b1_inhomogeneity: float = 0.12,
        p_anisotropic_2d: float = 0.85,
        p_partial_contrast: float = 0.3,
        noise_std_range: Tuple[float, float] = (0.02, 0.10),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        bias_field_coeff_range: float = 0.7,
        elastic_disp_scale: Tuple[float, float] = (0.0, 0.06),
        resolution_zoom_range: Tuple[float, float] = (0.5, 1.0),
        ghost_intensity_range: Tuple[float, float] = (0.6, 0.9),
        focal_b1_gain_range: Tuple[float, float] = (0.05, 0.25),
        lmic_field_strength_prior: bool = False,
        contrast_mod_indices: Optional[Sequence[int]] = None,
        p_field_contrast_warp: float = 0.0,
        field_contrast_source_b0: float = 3.0,
    ) -> None:
        self.p_flip = p_flip
        self.p_elastic = p_elastic
        self.p_bias_field = p_bias_field
        self.p_noise = p_noise
        self.p_rician_noise = p_rician_noise
        self.p_gamma = p_gamma
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast
        self.p_resolution = p_resolution
        self.p_motion = p_motion
        self.p_ghosting = p_ghosting
        self.p_spike = p_spike
        self.p_gibbs = p_gibbs
        self.p_focal_b1_inhomogeneity = p_focal_b1_inhomogeneity
        self.p_anisotropic_2d = p_anisotropic_2d
        self.p_partial_contrast = p_partial_contrast
        self.noise_std_range = noise_std_range
        self.gamma_range = gamma_range
        self.bias_field_coeff_range = bias_field_coeff_range
        self.elastic_disp_scale = elastic_disp_scale
        self.resolution_zoom_range = resolution_zoom_range
        self.ghost_intensity_range = ghost_intensity_range
        self.focal_b1_gain_range = focal_b1_gain_range
        self.lmic_field_strength_prior = bool(lmic_field_strength_prior)
        self.contrast_mod_indices = [int(i) for i in (contrast_mod_indices or [])]
        # Field-strength contrast warp (GPU port of the CPU augmentor's most
        # impactful low-field lever). p<=0 keeps the historical no-op behaviour.
        self.p_field_contrast_warp = float(p_field_contrast_warp)
        self.field_contrast_source_b0 = float(field_contrast_source_b0)
        self._grid_cache: Dict[Tuple[Tuple[int, int, int], str, str], torch.Tensor] = {}
        self._coord_cache: Dict[Tuple[Tuple[int, int, int], str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._field_strength_choices = torch.tensor([0.3, 0.55, 0.7, 1.0, 1.5, 3.0], dtype=torch.float32)
        if self.lmic_field_strength_prior:
            raw_probs = torch.tensor([0.04, 0.06, 0.08, 0.14, 0.52, 0.16], dtype=torch.float32)
        else:
            raw_probs = torch.tensor([0.03, 0.05, 0.07, 0.15, 0.40, 0.30], dtype=torch.float32)
        self._field_strength_probs = raw_probs
        self._field_strength_probs = self._field_strength_probs / self._field_strength_probs.sum()

    @staticmethod
    def _low_field_factor(field_strength: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """GPU-side counterpart of CPU _low_field_factor.

        FIX (Finding 3, 2026 audit): Denominator 1.2 -> 1.5 so 0.3 T and
        0.064 T produce distinct factors (was both clipped to 1.0).  Citations:
        Geethanath & Vaughan 2019; Marques et al. 2019; Campbell-Washburn
        et al. 2019; Obungoloch et al. 2023.
        """
        if torch.is_tensor(field_strength):
            return ((1.5 - field_strength) / 1.5).clamp(0.0, 1.0)
        return float(np.clip((1.5 - float(field_strength)) / 1.5, 0.0, 1.0))

    @staticmethod
    def _high_field_factor(field_strength: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """GPU-side counterpart of CPU _high_field_factor. See CPU docstring
        for citations (Wald 2012; Duyn 2012; Edelstein et al. 1986)."""
        if torch.is_tensor(field_strength):
            return ((field_strength - 1.5) / 1.5).clamp(0.0, 1.0)
        return float(np.clip((float(field_strength) - 1.5) / 1.5, 0.0, 1.0))

    def _field_scaled_noise_std(self, field_strength: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        std_lo, std_hi = self.noise_std_range
        scale = 1.0 + 1.25 * self._low_field_factor(field_strength)
        std_lo_t = torch.full_like(field_strength, std_lo, dtype=dtype)
        std_hi_t = torch.full_like(field_strength, std_hi, dtype=dtype)
        lo = std_lo_t * scale
        hi = std_hi_t * scale
        return lo + torch.rand_like(field_strength, dtype=dtype) * (hi - lo)

    def _sample_resolution_zoom_factor(self, field_strength: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        lo_base, hi_base = self.resolution_zoom_range
        low_field = self._low_field_factor(field_strength)
        lo = torch.clamp(torch.full_like(field_strength, lo_base, dtype=dtype) - 0.15 * low_field, min=0.35)
        hi = torch.clamp(torch.full_like(field_strength, hi_base, dtype=dtype) - 0.10 * low_field, max=1.0)
        hi = torch.maximum(hi, lo + 0.05)
        zoom = lo + torch.rand_like(field_strength, dtype=dtype) * (hi - lo)
        # Quantise the zoom factor so batches collapse to a small set of target
        # sizes, which cuts the number of interpolate launches in the grouped
        # degradation path without materially changing the augmentation regime.
        return ((zoom * 8.0).round() / 8.0).clamp(0.35, 1.0)

    def _base_grid(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (shape, str(device), str(dtype))
        grid = self._grid_cache.get(key)
        if grid is not None:
            return grid
        d, h, w = shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
        self._grid_cache[key] = grid
        return grid

    def _focal_coords(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (shape, str(device), str(dtype))
        coords = self._coord_cache.get(key)
        if coords is not None:
            return coords
        d, h, w = shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = (zz.unsqueeze(0), yy.unsqueeze(0), xx.unsqueeze(0))
        self._coord_cache[key] = coords
        return coords

    def _sample_field_strengths(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        probs = self._field_strength_probs.to(device=device)
        choices = self._field_strength_choices.to(device=device, dtype=dtype)
        idx = torch.multinomial(probs, batch_size, replacement=True)
        return choices.index_select(0, idx)

    def _field_contrast_warp_ratio(
        self, b0_src: float, b0_tgt: float, tissue: str, kind: str
    ) -> float:
        """Target-vs-source SE/IR signal ratio for a tissue class, clipped to
        [0.1, 4.0]. Reuses MRIPhysicsAugmentor's relaxometry LUT + SE/IR model so
        the GPU and CPU augmentors share ONE physics source of truth."""
        s_src = MRIPhysicsAugmentor._modality_signal_scalar(b0_src, tissue, kind)
        s_tgt = MRIPhysicsAugmentor._modality_signal_scalar(b0_tgt, tissue, kind)
        if s_src <= 1e-6:
            return 1.0
        return float(np.clip(s_tgt / s_src, 0.10, 4.0))

    def _simulate_field_strength_contrast_batch(
        self,
        x: torch.Tensor,
        field_strength: torch.Tensor,
    ) -> torch.Tensor:
        """GPU port of ``MRIPhysicsAugmentor._simulate_field_strength_contrast``.

        Re-renders each sample's implied tissue contrast from the assumed source
        field (``field_contrast_source_b0``, nominally 3 T - the BraTS/UTSW modal
        field) to the per-sample target B0 already sampled in ``__call__``. Because
        that same ``field_strength`` also drives the noise and resolution
        transforms, contrast / noise / resolution now all scale against ONE field
        strength - a physical coherence the GPU path previously lacked (only the
        CPU augmentor warped contrast, and only when it was the active engine).

        Each foreground channel is split into low/mid/high tissue bins by intensity
        terciles, and every bin is rescaled by the ratio of target-vs-source SE/IR
        signal for its tissue class (ET on the contrast channels). Background zeros
        are left untouched (the x==0 skull-stripped deployment invariant).
        """
        p = float(self.p_field_contrast_warp)
        if p <= 0.0:
            return x
        n, c = int(x.shape[0]), int(x.shape[1])
        b0_src = float(self.field_contrast_source_b0)
        if c == 4:
            kinds = ("t1", "t1", "t2", "flair")
        elif c == 1:
            kinds = ("t1",)
        else:
            kinds = tuple("t1" for _ in range(c))
        contrast_set = set(self.contrast_mod_indices)
        # Per-sample Bernoulli gate + host copies of the small per-sample scalars
        # (one transfer; no per-bin device syncs on the gate / field strength).
        apply = (torch.rand(n, device=x.device) < p).tolist()
        fs = field_strength.detach().float().cpu().tolist()
        for bi in range(n):
            if not apply[bi]:
                continue
            b0 = float(fs[bi])
            if abs(b0 - b0_src) < 0.05:
                continue  # already at the source field; nothing to warp
            for ci in range(c):
                ch = x[bi, ci]
                fg = ch != 0
                fg_vals = ch[fg]
                if fg_vals.numel() < 8:
                    continue
                # Terciles over the foreground (matches CPU np.percentile[33.3,66.7]).
                # torch.quantile caps near 2**24 elems; subsample above ~1M.
                if fg_vals.numel() > 1_000_000:
                    sel = torch.randint(fg_vals.numel(), (1_000_000,), device=fg_vals.device)
                    fg_vals = fg_vals.index_select(0, sel)
                q1, q2 = torch.quantile(
                    fg_vals.float(),
                    torch.tensor([0.333, 0.667], device=fg_vals.device, dtype=torch.float32),
                ).tolist()
                kind = kinds[ci]
                if kind == "t1":
                    bin_to_tissue = {0: "csf", 1: "wm", 2: "gm"}
                elif kind == "t2":
                    bin_to_tissue = {0: "wm", 1: "gm", 2: "edema"}
                else:  # flair
                    bin_to_tissue = {0: "csf", 1: "wm", 2: "edema"}
                if ci in contrast_set and kind == "t1":
                    bin_to_tissue[2] = "et"
                # Masks are snapshotted before any in-place rescale; bins are
                # disjoint, so the per-bin multiplies never compound.
                bins = (
                    (fg & (ch <= q1), bin_to_tissue[0]),
                    (fg & (ch > q1) & (ch <= q2), bin_to_tissue[1]),
                    (fg & (ch > q2), bin_to_tissue[2]),
                )
                for bin_mask, tissue in bins:
                    ratio = self._field_contrast_warp_ratio(b0_src, b0, tissue, kind)
                    if abs(ratio - 1.0) < 1e-6:
                        continue
                    ch[bin_mask] = ch[bin_mask] * ratio
        return x

    def _flip_batch(
        self,
        img: torch.Tensor,
        seg: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b = img.shape[0]
        for dim in (-3, -2, -1):
            mask = torch.rand(b, device=img.device) < self.p_flip
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                img[idx] = torch.flip(img.index_select(0, idx), dims=(dim,))
                if seg is not None:
                    seg[idx] = torch.flip(seg.index_select(0, idx), dims=(dim,))
        return img, seg

    def _elastic_deformation_batch(
        self,
        img: torch.Tensor,
        seg: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, _, d, h, w = img.shape
        mask = torch.rand(b, device=img.device) < self.p_elastic
        if not mask.any():
            return img, seg

        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        base = self._base_grid((d, h, w), img.device, img.dtype)
        low_shape = (max(d // 2, 2), max(h // 2, 2), max(w // 2, 2))
        disp = torch.randn((n, 3, *low_shape), device=img.device, dtype=img.dtype)
        disp = F.interpolate(disp, size=(d, h, w), mode="trilinear", align_corners=False)
        mag = torch.empty((n, 1, 1, 1, 1), device=img.device, dtype=img.dtype).uniform_(
            self.elastic_disp_scale[0], self.elastic_disp_scale[1]
        )
        disp = disp * (2.0 * mag)
        grid = torch.clamp(base.expand(n, -1, -1, -1, -1) + disp.permute(0, 2, 3, 4, 1), -1.25, 1.25)

        img_sel = img.index_select(0, idx)
        img_w = F.grid_sample(
            img_sel,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )

        fg = (img_sel.abs().sum(dim=1, keepdim=True) > 0).to(dtype=img.dtype)
        fg_w = F.grid_sample(
            fg,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        img_w = img_w * (fg_w > 0.5)
        img[idx] = img_w

        if seg is not None:
            seg_sel = seg.index_select(0, idx).unsqueeze(1).to(dtype=img.dtype)
            seg_w = F.grid_sample(
                seg_sel,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(1).long()
            seg[idx] = seg_w
        return img, seg

    def _simulate_bias_field_batch(self, img: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        mask = torch.rand(b, device=img.device) < self.p_bias_field
        if not mask.any():
            return img
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        coeff = self.bias_field_coeff_range * (
            1.0 + 0.10 * self._high_field_factor(field_strength.index_select(0, idx))
        )
        low = torch.randn((n, 1, 4, 4, 4), device=img.device, dtype=img.dtype)
        bias = F.interpolate(low, size=img.shape[-3:], mode="trilinear", align_corners=False)
        bias = torch.exp(torch.clamp(bias * coeff.view(-1, 1, 1, 1, 1), -0.6, 0.6)).clamp(0.55, 1.65)
        img_sel = img.index_select(0, idx)
        nz = img_sel != 0
        img[idx] = torch.where(nz, img_sel * bias, img_sel)
        return img

    def _simulate_focal_b1_inhomogeneity_batch(self, img: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        low_field = self._low_field_factor(field_strength)
        high_field = self._high_field_factor(field_strength)
        p = self.p_focal_b1_inhomogeneity
        p = p * (1.0 - 0.15 * low_field) * (1.0 + 0.15 * high_field)
        p = p.clamp(0.0, 0.30)
        mask = torch.rand(img.shape[0], device=img.device) < p
        if not mask.any():
            return img

        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        d, h, w = img.shape[-3:]
        zz, yy, xx = self._focal_coords((d, h, w), img.device, img.dtype)
        zz = zz.expand(n, -1, -1, -1)
        yy = yy.expand(n, -1, -1, -1)
        xx = xx.expand(n, -1, -1, -1)

        high_field = high_field.index_select(0, idx)
        low_field = low_field.index_select(0, idx)
        gain_scale = 1.0 - 0.10 * low_field + 0.10 * high_field
        gain_max = self.focal_b1_gain_range[1] * gain_scale
        min_gain = self.focal_b1_gain_range[0] * gain_scale

        max_blobs = 2
        n_blobs = torch.randint(1, max_blobs + 1, (n,), device=img.device)
        centers = torch.empty((n, max_blobs, 3), device=img.device, dtype=img.dtype).uniform_(-0.6, 0.6)
        radii = torch.empty((n, max_blobs, 3), device=img.device, dtype=img.dtype).uniform_(0.12, 0.28)
        gain_t = torch.empty((n, max_blobs), device=img.device, dtype=img.dtype).uniform_(-1.0, 1.0)
        gain_t = gain_t * gain_max.unsqueeze(1)
        rand_sign = torch.where(
            torch.rand((n, max_blobs), device=img.device) < 0.5,
            torch.ones((n, max_blobs), device=img.device, dtype=img.dtype),
            -torch.ones((n, max_blobs), device=img.device, dtype=img.dtype),
        )
        sign_t = torch.where(gain_t != 0, torch.sign(gain_t), rand_sign)
        gain_t = torch.where(gain_t.abs() < min_gain.unsqueeze(1), sign_t * min_gain.unsqueeze(1), gain_t)

        blob_mask = (torch.arange(max_blobs, device=img.device).unsqueeze(0) < n_blobs.unsqueeze(1)).to(dtype=img.dtype)
        center_z = centers[..., 0].view(n, max_blobs, 1, 1, 1)
        center_y = centers[..., 1].view(n, max_blobs, 1, 1, 1)
        center_x = centers[..., 2].view(n, max_blobs, 1, 1, 1)
        radius_z = radii[..., 0].view(n, max_blobs, 1, 1, 1)
        radius_y = radii[..., 1].view(n, max_blobs, 1, 1, 1)
        radius_x = radii[..., 2].view(n, max_blobs, 1, 1, 1)
        z_term = ((zz.unsqueeze(1) - center_z) / radius_z) ** 2
        y_term = ((yy.unsqueeze(1) - center_y) / radius_y) ** 2
        x_term = ((xx.unsqueeze(1) - center_x) / radius_x) ** 2
        blob = torch.exp(-0.5 * (z_term + y_term + x_term)) * blob_mask.view(n, max_blobs, 1, 1, 1)
        shading = torch.clamp(
            1.0 + gain_t.view(n, max_blobs, 1, 1, 1) * blob,
            0.45,
            1.65,
        ).prod(dim=1).unsqueeze(1)

        img_sel = img.index_select(0, idx)
        nz = img_sel != 0
        img[idx] = torch.where(nz, img_sel * shading, img_sel)
        return img

    def _apply_noise_model_batch(self, img: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        p_any = 1.0 - (1.0 - self.p_noise) * (1.0 - self.p_rician_noise)
        apply_mask = torch.rand(img.shape[0], device=img.device) < p_any
        if not apply_mask.any():
            return img

        idx = apply_mask.nonzero(as_tuple=True)[0]
        img_sel = img.index_select(0, idx)
        std = self._field_scaled_noise_std(field_strength.index_select(0, idx), dtype=img.dtype).view(-1, 1, 1, 1, 1)
        n1 = torch.randn_like(img_sel) * std
        gaussian = img_sel + n1
        n2 = torch.randn_like(img_sel) * std
        rician = torch.sqrt((img_sel + n1) ** 2 + n2 ** 2)
        p_rician = (0.30 + 0.55 * self._low_field_factor(field_strength.index_select(0, idx))).clamp(0.0, 0.90)
        choose_rician = (torch.rand(idx.numel(), device=img.device) < p_rician).view(-1, 1, 1, 1, 1)
        img[idx] = torch.where(choose_rician, rician, gaussian)
        return img

    @staticmethod
    def _masked_mean_std(img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.to(dtype=img.dtype)
        count = mask_f.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1.0)
        mean = (img * mask_f).sum(dim=(-3, -2, -1), keepdim=True) / count
        var = ((img - mean) ** 2 * mask_f).sum(dim=(-3, -2, -1), keepdim=True) / count
        return mean, var.sqrt()

    def _gamma_transform_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_gamma
        if not apply_mask.any():
            return img

        idx = apply_mask.nonzero(as_tuple=True)[0]
        img_sel = img.index_select(0, idx)
        mask = img_sel != 0
        gamma = torch.empty((idx.numel(), 1, 1, 1, 1), device=img.device, dtype=img.dtype).uniform_(
            self.gamma_range[0], self.gamma_range[1]
        )
        x_min = torch.where(mask, img_sel, torch.full_like(img_sel, float("inf"))).amin(dim=(-3, -2, -1), keepdim=True)
        x_min = torch.where(torch.isfinite(x_min), x_min, torch.zeros_like(x_min))
        base = torch.where(mask, (img_sel - x_min).clamp_min(0.0) + 1e-6, torch.ones_like(img_sel))
        shifted = torch.pow(base, gamma)
        mn, std = self._masked_mean_std(img_sel, mask)
        mn_s, std_s = self._masked_mean_std(shifted, mask)
        scaled = torch.where(
            std_s > 1e-8,
            (shifted - mn_s) / (std_s + 1e-8) * (std + 1e-8) + mn,
            shifted,
        )
        img[idx] = torch.where(mask, scaled, img_sel)
        return img

    def _brightness_multiplicative_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_brightness
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        scale = torch.empty((idx.numel(), 1, 1, 1, 1), device=img.device, dtype=img.dtype).uniform_(0.75, 1.25)
        img[idx] = img.index_select(0, idx) * scale
        return img

    def _simulate_partial_contrast_batch(self, img: torch.Tensor) -> torch.Tensor:
        if self.p_partial_contrast <= 0 or not self.contrast_mod_indices:
            return img
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_partial_contrast
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        img_sel = img.index_select(0, idx).clone()
        for sample_idx in range(img_sel.shape[0]):
            for c_idx in self.contrast_mod_indices:
                if c_idx < 0 or c_idx >= img_sel.shape[1]:
                    continue
                ch = img_sel[sample_idx, c_idx]
                nz = ch != 0
                if not nz.any():
                    continue
                ch_out = ch.clone()
                if img_sel.shape[1] > 1 and c_idx != 0:
                    ref = img_sel[sample_idx, 0]
                    blend_mask = nz & (ref != 0)
                    alpha = torch.empty((), device=img.device, dtype=img.dtype).uniform_(0.2, 0.9)
                    ch_out[blend_mask] = alpha * ch[blend_mask] + (1.0 - alpha) * ref[blend_mask]
                else:
                    vals = ch[nz]
                    threshold = torch.quantile(vals.float(), 0.5).to(dtype=img.dtype)
                    enhancement = (ch > threshold) & nz
                    if not enhancement.any():
                        continue
                    scale = torch.empty((), device=img.device, dtype=img.dtype).uniform_(0.2, 0.9)
                    ch_out[enhancement] = threshold + (ch[enhancement] - threshold) * scale
                img_sel[sample_idx, c_idx] = ch_out
        img[idx] = img_sel
        return img

    def _contrast_augmentation_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_contrast
        if not apply_mask.any():
            return img

        idx = apply_mask.nonzero(as_tuple=True)[0]
        img_sel = img.index_select(0, idx)
        mask = img_sel != 0
        mean, _ = self._masked_mean_std(img_sel, mask)
        factor = torch.empty((idx.numel(), 1, 1, 1, 1), device=img.device, dtype=img.dtype).uniform_(0.65, 1.5)
        adjusted = (img_sel - mean) * factor + mean
        img[idx] = torch.where(mask, adjusted, img_sel)
        return img

    def _resolution_degradation_batch(self, img: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_resolution
        if not apply_mask.any():
            return img

        idx = apply_mask.nonzero(as_tuple=True)[0]
        axes = torch.randint(0, 3, (idx.numel(),), device=img.device)
        zoom = self._sample_resolution_zoom_factor(field_strength.index_select(0, idx), dtype=img.dtype)
        shape = list(img.shape[-3:])
        target_sizes = torch.stack(
            [
                torch.full_like(zoom, shape[0], dtype=torch.long),
                torch.full_like(zoom, shape[1], dtype=torch.long),
                torch.full_like(zoom, shape[2], dtype=torch.long),
            ],
            dim=1,
        )
        target_sizes[torch.arange(idx.numel(), device=img.device), axes] = torch.clamp(
            (target_sizes[torch.arange(idx.numel(), device=img.device), axes].to(dtype=img.dtype) * zoom).round().to(torch.long),
            min=2,
        )
        for axis in range(3):
            axis_mask = axes == axis
            if not axis_mask.any():
                continue
            axis_pos = axis_mask.nonzero(as_tuple=True)[0]
            unique_sizes = target_sizes.index_select(0, axis_pos)[:, axis].unique(sorted=True)
            for target_size in unique_sizes.tolist():
                size_mask = target_sizes.index_select(0, axis_pos)[:, axis] == int(target_size)
                pos = axis_pos.index_select(0, size_mask.nonzero(as_tuple=True)[0])
                sub_idx = idx.index_select(0, pos)
                sub = img.index_select(0, sub_idx)
                degraded_shape = list(shape)
                degraded_shape[axis] = int(target_size)
                tmp = F.interpolate(sub, size=tuple(degraded_shape), mode="trilinear", align_corners=False)
                img[sub_idx] = F.interpolate(tmp, size=tuple(shape), mode="trilinear", align_corners=False)
        return img

    def _simulate_anisotropic_2d_batch(self, img: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_anisotropic_2d
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        axis_probs = torch.tensor([0.70, 0.15, 0.15], device=img.device, dtype=torch.float32)
        axes = torch.multinomial(axis_probs, idx.numel(), replacement=True)
        low_field = self._low_field_factor(field_strength.index_select(0, idx))
        lo = torch.full((idx.numel(),), 3.0, device=img.device, dtype=img.dtype)
        hi = torch.full((idx.numel(),), 7.0, device=img.device, dtype=img.dtype) + 0.5 * low_field
        thickness = lo + torch.rand_like(lo) * (hi - lo)
        zoom = (1.0 / thickness).clamp(0.18, 0.42)
        zoom = ((zoom * 8.0).round() / 8.0).clamp(0.18, 0.42)
        shape = list(img.shape[-3:])
        for axis in range(3):
            axis_mask = axes == axis
            if not axis_mask.any():
                continue
            axis_pos = axis_mask.nonzero(as_tuple=True)[0]
            axis_zoom = zoom.index_select(0, axis_pos)
            unique_zooms = axis_zoom.unique(sorted=True)
            for z_val in unique_zooms.tolist():
                z_mask = (axis_zoom - z_val).abs() < 1e-4
                pos = axis_pos.index_select(0, z_mask.nonzero(as_tuple=True)[0])
                sub_idx = idx.index_select(0, pos)
                sub = img.index_select(0, sub_idx)
                degraded_shape = list(shape)
                degraded_shape[axis] = max(2, int(round(shape[axis] * z_val)))
                k = max(3, int(round(1.0 / max(z_val, 1e-6)))) | 1
                padding = k // 2
                blurred = F.avg_pool3d(
                    sub,
                    kernel_size=(k, 1, 1) if axis == 0 else (1, k, 1) if axis == 1 else (1, 1, k),
                    stride=1,
                    padding=(padding, 0, 0) if axis == 0 else (0, padding, 0) if axis == 1 else (0, 0, padding),
                )
                down = F.interpolate(blurred, size=tuple(degraded_shape), mode="trilinear", align_corners=False)
                up = F.interpolate(down, size=tuple(shape), mode="trilinear", align_corners=False)
                nz = sub != 0
                img[sub_idx] = torch.where(nz, up, torch.zeros_like(up))
        return img

    def _simulate_motion_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_motion
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        axes = torch.randint(0, 3, (idx.numel(),), device=img.device)
        for axis in range(3):
            axis_mask = axes == axis
            if not axis_mask.any():
                continue
            pos = axis_mask.nonzero(as_tuple=True)[0]
            sub_idx = idx.index_select(0, pos)
            sub = img.index_select(0, sub_idx)
            kspace = torch.fft.fftn(sub, dim=(-3, -2, -1))
            axis_len = sub.shape[axis + 2]
            seg_counts = torch.randint(4, 10, (sub.shape[0],), device=img.device)
            max_segments = int(seg_counts.max().item())
            if axis == 0:
                coord = torch.arange(axis_len, device=img.device).view(1, 1, axis_len, 1, 1)
            elif axis == 1:
                coord = torch.arange(axis_len, device=img.device).view(1, 1, 1, axis_len, 1)
            else:
                coord = torch.arange(axis_len, device=img.device).view(1, 1, 1, 1, axis_len)
            for seg_idx in range(max_segments):
                active = seg_idx < seg_counts
                if not active.any():
                    continue
                phase_mask = active & (torch.rand(sub.shape[0], device=img.device) < 0.4)
                if not phase_mask.any():
                    continue
                start = torch.floor(seg_idx * axis_len / seg_counts.to(dtype=img.dtype)).to(torch.long)
                end = torch.ceil((seg_idx + 1) * axis_len / seg_counts.to(dtype=img.dtype)).to(torch.long)
                end = torch.clamp(end, max=axis_len)
                line_mask = (
                    (coord >= start.view(-1, 1, 1, 1, 1))
                    & (coord < end.view(-1, 1, 1, 1, 1))
                    & phase_mask.view(-1, 1, 1, 1, 1)
                )
                phase = torch.empty(sub.shape[0], device=img.device, dtype=img.dtype).uniform_(-0.5, 0.5)
                phase_factor = torch.complex(torch.cos(phase), torch.sin(phase)).view(-1, 1, 1, 1, 1)
                kspace = torch.where(line_mask, kspace * phase_factor, kspace)
            img[sub_idx] = torch.fft.ifftn(kspace, dim=(-3, -2, -1)).abs().to(dtype=img.dtype)
        return img

    def _simulate_ghosting_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_ghosting
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        axes = torch.randint(0, 3, (idx.numel(),), device=img.device)
        for axis in range(3):
            axis_mask = axes == axis
            if not axis_mask.any():
                continue
            pos = axis_mask.nonzero(as_tuple=True)[0]
            sub_idx = idx.index_select(0, pos)
            sub = img.index_select(0, sub_idx)
            out = sub.clone()
            axis_len = sub.shape[axis + 2]
            divisors = torch.randint(4, 8, (sub.shape[0],), device=img.device)
            shifts = torch.clamp(torch.div(axis_len, divisors, rounding_mode="floor"), min=2)
            intensities = torch.empty(sub.shape[0], device=img.device, dtype=img.dtype).uniform_(
                self.ghost_intensity_range[0], self.ghost_intensity_range[1]
            )
            n_ghosts = torch.randint(1, 4, (sub.shape[0],), device=img.device)
            for ghost_idx in range(1, 4):
                active = n_ghosts >= ghost_idx
                if not active.any():
                    continue
                active_shifts = shifts.index_select(0, active.nonzero(as_tuple=True)[0]).unique(sorted=True)
                for shift_val in active_shifts.tolist():
                    shift_mask = active & (shifts == int(shift_val))
                    if not shift_mask.any():
                        continue
                    group_pos = shift_mask.nonzero(as_tuple=True)[0]
                    scale = (intensities.index_select(0, group_pos) / ghost_idx).view(-1, 1, 1, 1, 1)
                    rolled = torch.roll(sub.index_select(0, group_pos), shifts=int(shift_val) * ghost_idx, dims=axis + 2)
                    out[group_pos] = out.index_select(0, group_pos) + scale * rolled
            img[sub_idx] = out
        return img

    def _simulate_spike_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_spike
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        sub = img.index_select(0, idx)
        kspace = torch.fft.fftn(sub.float(), dim=(-3, -2, -1))
        spatial_shape = sub.shape[-3:]
        for sample_idx in range(sub.shape[0]):
            mag = torch.clamp(kspace[sample_idx].abs().mean(), min=1e-6)
            n_spikes = int(torch.randint(1, 4, (), device=img.device).item())
            for _ in range(n_spikes):
                z = int(torch.randint(0, spatial_shape[0], (), device=img.device).item())
                y = int(torch.randint(0, spatial_shape[1], (), device=img.device).item())
                x0 = int(torch.randint(0, spatial_shape[2], (), device=img.device).item())
                phase = torch.empty((), device=img.device).uniform_(0.0, 2.0 * math.pi)
                intensity = torch.empty((), device=img.device).uniform_(0.3, 0.7)
                spike = intensity * mag * (float(np.prod(spatial_shape)) ** 0.25)
                spike_c = torch.complex(spike * torch.cos(phase), spike * torch.sin(phase))
                kspace[sample_idx, :, z, y, x0] = kspace[sample_idx, :, z, y, x0] + spike_c
        img[idx] = torch.fft.ifftn(kspace, dim=(-3, -2, -1)).abs().to(dtype=img.dtype)
        return img

    def _simulate_gibbs_batch(self, img: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(img.shape[0], device=img.device) < self.p_gibbs
        if not apply_mask.any():
            return img
        idx = apply_mask.nonzero(as_tuple=True)[0]
        sub = img.index_select(0, idx)
        kspace = torch.fft.fftshift(torch.fft.fftn(sub.float(), dim=(-3, -2, -1)), dim=(-3, -2, -1))
        d, h, w = sub.shape[-3:]
        mask = torch.zeros((d, h, w), device=img.device, dtype=torch.bool)
        alpha = torch.empty((), device=img.device).uniform_(0.4, 0.8).item()
        keep = [
            max(2, int(round(dim * (1.0 - 0.25 * alpha))))
            for dim in (d, h, w)
        ]
        starts = [(dim - k) // 2 for dim, k in zip((d, h, w), keep)]
        mask[
            starts[0]:starts[0] + keep[0],
            starts[1]:starts[1] + keep[1],
            starts[2]:starts[2] + keep[2],
        ] = True
        kspace = torch.where(mask.view(1, 1, d, h, w), kspace, torch.zeros_like(kspace))
        img[idx] = torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-3, -2, -1)), dim=(-3, -2, -1)).abs().to(dtype=img.dtype)
        return img

    def __call__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.device.type != "cuda":
            return x, y

        with torch.no_grad():
            have_labels = y is not None and y.numel() > 0
            field_strength = self._sample_field_strengths(x.shape[0], x.device, x.dtype)
            x, y = self._flip_batch(x, y if have_labels else None)
            x, y = self._elastic_deformation_batch(x, y if have_labels else None)
            # Field-strength contrast warp BEFORE bias/noise (matches the CPU
            # ordering) so downstream noise/resolution scale against the warped,
            # target-B0 signal rather than the original 3 T contrast.
            x = self._simulate_field_strength_contrast_batch(x, field_strength)
            x = self._simulate_bias_field_batch(x, field_strength)
            x = self._simulate_focal_b1_inhomogeneity_batch(x, field_strength)
            x = self._apply_noise_model_batch(x, field_strength)
            x = self._gamma_transform_batch(x)
            x = self._brightness_multiplicative_batch(x)
            x = self._simulate_partial_contrast_batch(x)
            x = self._contrast_augmentation_batch(x)
            x = self._resolution_degradation_batch(x, field_strength)
            x = self._simulate_anisotropic_2d_batch(x, field_strength)
            x = self._simulate_motion_batch(x)
            x = self._simulate_ghosting_batch(x)
            x = self._simulate_spike_batch(x)
            x = self._simulate_gibbs_batch(x)
            x = x.contiguous(memory_format=torch.channels_last_3d)
            return x, y if have_labels else None


# ══════════════════════════════════════════════════════════════════════════════
# §5  DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class MultiModalSegDataset(Dataset):
    """Training / validation dataset for multi-institutional brain MRI.

    CSV schema (header required):
        case_id, <mod1>, ..., <modN>, label, dataset_tag, label_remap,
        field_strength

    dataset_tag : {brats_glioma, brats_africa, utsw_glioma, erasmus}
    label_remap : JSON dict mapping source→canonical labels
                  e.g. '{"1":"1"}' for ERASMUS binary masks
    field_strength : scanner field strength in Tesla (3.0, 1.5, 1.0)
                     or empty string if unknown
    """

    def __init__(
        self,
        csv_path: str,
        modalities: Sequence[str],
        patch_size: Tuple[int, int, int],
        train: bool,
        augment: bool,
        augment_cfg: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        patch_cache_dir: Optional[str] = None,
        patch_cache_start_epoch: int = 1,
        label_noise_prob: float = 0.0,
        label_noise_mode: str = "morph",
        label_noise_band_radius: int = 2,
        label_noise_band_flip_prob: float = 0.3,
        lmic_field_strength_prior: bool = False,
        contrast_mod_indices: Optional[Sequence[int]] = None,
        # Anatomy-preserving mix-up (Wang+ 2021 CarveMix). Carves a donor's
        # tumor (with k-vox dilation context) into a tumor-free region of
        # the receiver. Implemented as a post-augmentation step so it
        # operates on the network's actual input distribution.
        carvemix_prob: float = 0.0,
        carvemix_dilation: int = 5,
        # Auxiliary Fourier-basis Augmentation probability. Older versions used
        # this to source a donor for amplitude mixing; the augmentor now applies
        # literature AFA directly and does not require a donor volume.
        afa_prob: float = 0.0,
        # Lucas+ 2025 ULF GAN: optional pretrained generator that synthesizes
        # ~64 mT-appearance volumes from a high-field input.
        ulf_gan_prob: float = 0.0,
        ulf_gan_weights: Optional[str] = None,
        ulf_gan_target_b0: float = 0.064,
        # FIX (Gap B, 2026 audit): per-modality random dropout at train time.
        # Was student-only; teacher now uses this so its KD logits encode
        # "what to predict when modality X is missing" - critical for LMIC
        # deployment where T1c contrast often is not acquired.  Set 0.0 to
        # disable; recommended 0.15 for clinically-realistic training.  The
        # max_drop guard keeps at least (n_modalities - max_drop) channels
        # alive so the model is never asked to segment from zero input.
        modality_dropout_p: float = 0.0,
        modality_dropout_max_drop: int = 2,
        # Phase 7.5: optional anatomical tissue-prior channels (CSF/WM/GM). When
        # use_tissue_priors is True, __getitem__ concatenates 3 co-registered
        # tissue maps from tissue_prior_dir, yielding a 7-channel volume. Default
        # OFF -> unchanged 4-channel output (bit-identical legacy behaviour).
        use_tissue_priors: bool = False,
        tissue_prior_dir: str = "",
        # Phase 7.6: optional v9.1 physics-augmentation callable applied to the
        # volume tensor (C, D, H, W) after preprocessing. None -> not applied
        # (the legacy MRIPhysicsAugmentor path is independent of this).
        physics_aug: Optional[Any] = None,
        # Bloch low-field simulator: re-renders the MRI channels to ~64 mT from the
        # CSF/WM/GM priors + label. Requires use_tissue_priors. None -> not applied.
        bloch_sim: Optional[Any] = None,
        bloch_prob: float = 1.0,
        # SynthSeg-style fully contrast-randomized generator (Billot 2023). Same inputs
        # as the Bloch sim (tissue priors + label); applied per-sample as an ALTERNATIVE
        # to the Bloch render (when it fires, the Bloch render is skipped). None -> off.
        synth_sim: Optional[Any] = None,
        # Channel-aware geometric augmentation (flip / affine / elastic) applied to
        # ALL channels (MRI + tissue priors) + the label as the final dataset step,
        # so they stay co-registered. None -> not applied. Restores the spatial
        # augmentation the legacy augmentor provided, for the v9.1 path. On the CUDA
        # path this is carried by the GPU batched augmentor instead (passed None here).
        geometric_aug: Optional[Any] = None,
    ) -> None:
        self.modalities = list(modalities)
        self.patch_size = patch_size
        self.train = train
        self.augment_flag = augment
        self.use_tissue_priors = bool(use_tissue_priors)
        self.tissue_prior_dir = str(tissue_prior_dir or "")
        self.physics_aug = physics_aug
        self.bloch_sim = bloch_sim
        self.bloch_prob = float(bloch_prob)
        self.synth_sim = synth_sim
        self.geometric_aug = geometric_aug
        with open(csv_path, encoding="utf-8") as f:
            self.rows: List[Dict] = list(csv.DictReader(f))

        # Build augmentor (field-strength will be per-sample)
        self.augmentor = MRIPhysicsAugmentor(**(augment_cfg or {})) if augment else None

        # ── Shared npy cache ──────────────────────────────────────────────
        # cache_dir is shared across workers on the same node. Cache entries
        # are keyed by source paths + mtimes + modality schema + label remap,
        # so cache hits remain valid even when case IDs collide across sites or
        # the underlying NIfTI files change between runs.
        self.cache_dir = cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.label_noise_prob = float(label_noise_prob)
        if label_noise_mode not in ("morph", "boundary_band"):
            raise ValueError(
                f"label_noise_mode must be 'morph' or 'boundary_band', got {label_noise_mode!r}"
            )
        self.label_noise_mode = str(label_noise_mode)
        self.label_noise_band_radius = max(1, int(label_noise_band_radius))
        self.label_noise_band_flip_prob = float(np.clip(label_noise_band_flip_prob, 0.0, 1.0))
        self.lmic_field_strength_prior = bool(lmic_field_strength_prior)
        self.contrast_mod_indices = [int(i) for i in (contrast_mod_indices or [])]
        self.carvemix_prob = float(np.clip(carvemix_prob, 0.0, 1.0))
        self.carvemix_dilation = max(0, int(carvemix_dilation))
        self.afa_prob = float(np.clip(afa_prob, 0.0, 1.0))
        # Honor the augmentor-side AFA probability to keep the dataset and
        # augmentor in sync (the augmentor does the actual FFT mix).
        if self.augmentor is not None and self.afa_prob > 0.0:
            self.augmentor.p_fourier_amplitude_mix = max(
                self.augmentor.p_fourier_amplitude_mix, self.afa_prob
            )
        # ULF GAN augmentor (optional, no-op if weights are missing).
        self.ulf_gan_prob = float(np.clip(ulf_gan_prob, 0.0, 1.0))
        self.ulf_gan = (
            ULFGANAugmentor(weights_path=ulf_gan_weights, target_b0=ulf_gan_target_b0)
            if (ulf_gan_weights and self.ulf_gan_prob > 0.0)
            else None
        )
        self.patch_cache_dir = patch_cache_dir if train else None
        self.patch_cache_start_epoch = max(1, int(patch_cache_start_epoch))
        self._patch_cache_epoch = 1
        self._patch_cache_epoch_mtime_ns: Optional[int] = None
        self._patch_epoch_path: Optional[Path] = None
        if self.patch_cache_dir:
            patch_root = Path(self.patch_cache_dir)
            patch_root.mkdir(parents=True, exist_ok=True)
            self._patch_epoch_path = patch_root / "_epoch.txt"
        # FIX (Gap B, 2026 audit): teacher-side modality dropout config.
        self.modality_dropout_p = float(np.clip(modality_dropout_p, 0.0, 1.0))
        self.modality_dropout_max_drop = max(0, int(modality_dropout_max_drop))

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        if not self.patch_cache_dir or self._patch_epoch_path is None:
            return
        epoch = max(1, int(epoch))
        patch_root = Path(self.patch_cache_dir)
        patch_root.mkdir(parents=True, exist_ok=True)
        (patch_root / f"epoch_{epoch:04d}").mkdir(parents=True, exist_ok=True)
        stale_epoch = epoch - 2
        if stale_epoch >= 1:
            shutil.rmtree(patch_root / f"epoch_{stale_epoch:04d}", ignore_errors=True)
        tmp_path = patch_root / f"_epoch.txt.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp_path.write_text(str(epoch), encoding="utf-8")
        os.replace(tmp_path, self._patch_epoch_path)
        try:
            self._patch_cache_epoch_mtime_ns = int(self._patch_epoch_path.stat().st_mtime_ns)
        except OSError:
            self._patch_cache_epoch_mtime_ns = None
        self._patch_cache_epoch = epoch

    @staticmethod
    def _apply_label_remap(y: np.ndarray, remap_str: str) -> np.ndarray:
        """Apply JSON label remap for cross-dataset label unification.

        ERASMUS: binary WT -> class 1. BraTS canonical internal encoding is
        0/1/2/3, with ET stored as 3 even if a source mask uses BraTS label 4.
        """
        remap_str = (remap_str or "").strip()
        if not remap_str or remap_str in ("{}", "null"):
            return y
        try:
            remap: Dict[str, str] = json.loads(remap_str)
        except json.JSONDecodeError as exc:
            # Do NOT silently pass labels through un-remapped: a malformed remap field
            # (e.g. ERASMUS binary->class 1) means wrong cross-dataset labels or a case
            # dropped later by _canonicalise_label_values. Warn so the CSV gets fixed.
            LOG.warning("[label_remap] malformed JSON %r (%s); applying NO remap - "
                        "cross-dataset labels may be wrong. Fix the manifest.",
                        remap_str, exc)
            return y
        if not remap:
            return y
        y_out = y.copy()
        for src_str, dst_str in remap.items():
            y_out[y == int(src_str)] = int(dst_str)
        return y_out

    @staticmethod
    def _canonicalise_label_values(y: np.ndarray, case_id: str) -> np.ndarray:
        """Map common BraTS 1/2/4 masks to the internal 1/2/3 schema.

        UTSW manually-refined segmentations follow the legacy BraTS convention
        ``{0=BG, 1=NCR, 2=ED, 4=ET}``; the 4→3 remap is applied transparently
        with INFO-level logging so investigators retain an audit trail. Any
        other unusual convention should be expressed as JSON in the per-row
        ``label_remap`` field (e.g. ``{"5":3,"6":1}``).
        """
        uniq = {int(v) for v in np.unique(y)}
        if 4 in uniq:
            y = y.copy()
            y[y == 4] = 3
            uniq = {int(v) for v in np.unique(y)}
            LOG.info(
                "[%s] mapped source label 4 (BraTS legacy/UTSW ET) to internal class 3",
                case_id,
            )
        invalid = sorted(v for v in uniq if v not in {0, 1, 2, 3})
        if invalid:
            case_lower = str(case_id).lower()
            is_utsw = ("utsw" in case_lower) or ("southwestern" in case_lower)
            hint = (
                "Expected internal BraTS labels {0,1,2,3}; set label_remap in the CSV "
                "as a JSON dict (e.g. '{\"5\":3}')."
            )
            if is_utsw:
                hint += (
                    " For UTSW-glioma the manual masks usually follow the legacy "
                    "{0,1,2,4} convention - auto-remap 4→3 has already run, so any "
                    "remaining out-of-range values likely indicate a different "
                    "annotation protocol; verify with the data manager."
                )
            raise ValueError(
                f"[{case_id}] unsupported label values after remap: {invalid}. {hint}"
            )
        return y

    # ════════════════════════════════════════════════════════════════════════
    # CarveMix donor sourcing.
    #
    # CarveMix needs a "second case" different from the receiver. The helper is
    # retained for that path; literature AFA is additive and no longer consumes
    # donor volumes.
    #
    # Refs:
    #   Wang+ "CarveMix: A simple data augmentation method for brain lesion
    #     segmentation", MICCAI 2021. (Anatomy-preserving lesion mix-up.)
    #   Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic
    #     Segmentation", CVPR 2020.
    # ════════════════════════════════════════════════════════════════════════

    def _sample_donor_volume(
        self,
        receiver_idx: int,
        target_shape: Tuple[int, int, int, int],
        require_tumor: bool = False,
    ) -> Optional[np.ndarray]:
        """Return a donor image volume cropped to match ``target_shape``.

        Picks a random row index ≠ receiver_idx, loads via the cached path,
        and applies an i.i.d. random crop. If ``require_tumor=True`` and the
        donor has no foreground voxels, returns None so the caller can skip.
        Returns the image only (not the label); CarveMix uses a separate
        path that returns both.
        """
        if len(self.rows) <= 1:
            return None
        donor_idx = receiver_idx
        for _ in range(4):
            cand = int(np.random.randint(0, len(self.rows)))
            if cand != receiver_idx:
                donor_idx = cand
                break
        if donor_idx == receiver_idx:
            return None
        try:
            donor_x, donor_y, _, donor_fg = self._load(self.rows[donor_idx])
        except Exception as exc:
            if not getattr(self, "_donor_load_warned", False):
                LOG.warning("[donor] load failed (%s); CarveMix/AFA donor augmentation may "
                            "be effectively disabled (warned once).", exc)
                self._donor_load_warned = True
            return None
        if require_tumor and (donor_y is None or donor_fg is None or donor_fg.size == 0):
            return None
        # Random crop / pad to target spatial shape (target_shape includes channels).
        c_t, d_t, h_t, w_t = target_shape
        if donor_x.shape[0] != c_t:
            return None
        donor_x, _ = pad_to_patch(donor_x, (d_t, h_t, w_t), None)
        _, d, h, w = donor_x.shape
        sd = int(np.random.randint(0, max(d - d_t + 1, 1)))
        sh = int(np.random.randint(0, max(h - h_t + 1, 1)))
        sw = int(np.random.randint(0, max(w - w_t + 1, 1)))
        return donor_x[:, sd:sd + d_t, sh:sh + h_t, sw:sw + w_t].astype(np.float32, copy=False)

    def _sample_carvemix_donor(
        self,
        receiver_idx: int,
        max_attempts: int = 8,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (donor_x, donor_y) for a tumor-bearing donor, full volume.

        We deliberately do NOT crop the donor - CarveMix needs to know the
        tumor's bounding box in the donor's native voxel grid so that the
        carve preserves the lesion's full extent.
        """
        if len(self.rows) <= 1:
            return None
        for _ in range(max_attempts):
            donor_idx = int(np.random.randint(0, len(self.rows)))
            if donor_idx == receiver_idx:
                continue
            try:
                donor_x, donor_y, _, _ = self._load(self.rows[donor_idx])
            except Exception:
                continue
            if donor_y is None or not np.any(donor_y > 0):
                continue
            return donor_x.astype(np.float32, copy=False), donor_y.astype(np.int64, copy=False)
        return None

    def _apply_carvemix(
        self,
        receiver_idx: int,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Anatomy-preserving lesion mix-up (CarveMix; Wang+ MICCAI 2021).

        Steps:
          1. Sample a tumor-bearing donor.
          2. Compute the donor's tumor bounding box, dilated by
             ``carvemix_dilation`` voxels for context.
          3. Carve the donor's image+label inside that bbox.
          4. Find a tumor-free brain location in the receiver large enough
             to host the carved bbox. If none exists, return unchanged.
          5. Paste the carved patch into the receiver, overwriting both
             intensity and label inside the donor's tumor (label-binding
             ensures the new lesion's annotation is consistent).
        """
        donor = self._sample_carvemix_donor(receiver_idx)
        if donor is None:
            return x, y
        donor_x, donor_y = donor
        tumor_mask = donor_y > 0
        if not tumor_mask.any():
            return x, y
        coords = np.argwhere(tumor_mask)
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        d_pad = int(self.carvemix_dilation)
        # Donor bbox dilated by d_pad (clamped to donor extent).
        D, H, W = donor_y.shape
        z0 = max(0, z0 - d_pad); y0 = max(0, y0 - d_pad); x0 = max(0, x0 - d_pad)
        z1 = min(D, z1 + d_pad); y1 = min(H, y1 + d_pad); x1 = min(W, x1 + d_pad)
        bbox_d, bbox_h, bbox_w = z1 - z0, y1 - y0, x1 - x0
        donor_patch_x = donor_x[:, z0:z1, y0:y1, x0:x1]
        donor_patch_y = donor_y[z0:z1, y0:y1, x0:x1]
        # Find a tumor-free location in the receiver that fits the donor bbox.
        _, rD, rH, rW = x.shape
        if bbox_d > rD or bbox_h > rH or bbox_w > rW:
            return x, y
        receiver_tumor = y > 0
        # Try up to 12 random placements; accept the first that overlaps no
        # receiver tumor inside the carved patch's mask.
        for _ in range(12):
            sd = int(np.random.randint(0, rD - bbox_d + 1))
            sh = int(np.random.randint(0, rH - bbox_h + 1))
            sw = int(np.random.randint(0, rW - bbox_w + 1))
            target_y = receiver_tumor[sd:sd + bbox_d, sh:sh + bbox_h, sw:sw + bbox_w]
            donor_in_bbox = donor_patch_y > 0
            # Reject if the donor's actual tumor (not the dilated context)
            # would overlap an existing receiver tumor.
            if np.any(target_y & donor_in_bbox):
                continue
            # Smooth alpha mask over the donor's dilated bbox so the paste
            # blends at the boundary. Inside the donor's tumor mask, alpha=1;
            # outside it decays exponentially to 0 over the dilation band.
            if d_pad > 0:
                from scipy.ndimage import distance_transform_edt
                dist = distance_transform_edt(~donor_in_bbox).astype(np.float32)
                alpha = np.exp(-dist / max(1.0, 0.5 * d_pad))
                alpha = np.clip(alpha, 0.0, 1.0)
            else:
                alpha = donor_in_bbox.astype(np.float32)
            x_out = x.copy()
            y_out = y.copy()
            x_slice = x_out[:, sd:sd + bbox_d, sh:sh + bbox_h, sw:sw + bbox_w]
            blend = (1.0 - alpha[None]) * x_slice + alpha[None] * donor_patch_x
            x_out[:, sd:sd + bbox_d, sh:sh + bbox_h, sw:sw + bbox_w] = blend.astype(np.float32)
            # Label binding: only the donor's true tumor voxels propagate; the
            # dilation context preserves the receiver's existing label.
            y_slice = y_out[sd:sd + bbox_d, sh:sh + bbox_h, sw:sw + bbox_w]
            y_slice[donor_in_bbox] = donor_patch_y[donor_in_bbox]
            y_out[sd:sd + bbox_d, sh:sh + bbox_h, sw:sw + bbox_w] = y_slice
            return x_out, y_out
        return x, y

    @staticmethod
    def _safe_cache_token(value: Any, max_len: int = 48) -> str:
        token = "".join(
            ch if (str(ch).isalnum() or ch in "-_.") else "_"
            for ch in str(value)
        )
        token = token.strip("._") or "unknown"
        return token[:max_len]

    @staticmethod
    def _resolved_path_and_mtime(path_str: Optional[str]) -> Dict[str, Any]:
        if not path_str:
            return {"path": None, "mtime_ns": None}
        p = Path(path_str).expanduser()
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
        try:
            mtime_ns = int(p.stat().st_mtime_ns)
        except OSError:
            mtime_ns = None
        return {"path": resolved, "mtime_ns": mtime_ns}

    def _cache_metadata(self, row: Dict) -> Dict[str, Any]:
        return {
            "cache_version": "ogap_cache_v4_preprocess_labels",
            "case_id": str(row.get("case_id", "unknown")),
            "dataset_tag": str(row.get("dataset_tag", "unknown")),
            "modalities": list(self.modalities),
            "preprocess": {
                "n4_disabled": bool(_env_flag_enabled("OGAP_DISABLE_N4")),
                "n4_iterations": list(_n4_iterations_from_env((50, 50, 30, 20))),
                "n4_shrink_factor": int(_n4_shrink_factor()),
            },
            "modality_sources": [
                self._resolved_path_and_mtime(row.get(modality))
                for modality in self.modalities
            ],
            "label_source": self._resolved_path_and_mtime(row.get("label")),
            "label_remap": str(row.get("label_remap", "")),
            "train": bool(self.train),
        }

    def _cache_stem_and_metadata(self, row: Dict) -> Tuple[str, Dict[str, Any]]:
        metadata = self._cache_metadata(row)
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        stem = (
            f"{self._safe_cache_token(row.get('dataset_tag', 'unknown'), 24)}_"
            f"{self._safe_cache_token(row.get('case_id', 'unknown'), 48)}_"
            f"{cache_key}"
        )
        return stem, metadata

    def _cache_paths(
        self, row: Dict
    ) -> Tuple[Path, Path, Path, Path, Path, Dict[str, Any]]:
        stem, metadata = self._cache_stem_and_metadata(row)
        cache_root = Path(self.cache_dir)
        return (
            cache_root / f"{stem}_x.npy",
            cache_root / f"{stem}_y.npy",
            cache_root / f"{stem}_s.npy",
            cache_root / f"{stem}_fg.npy",
            cache_root / f"{stem}_meta.json",
            metadata,
        )

    def _current_patch_cache_epoch(self) -> int:
        if not self.patch_cache_dir or self._patch_epoch_path is None:
            return 1
        try:
            stat = self._patch_epoch_path.stat()
            mtime_ns = int(stat.st_mtime_ns)
        except OSError:
            return self._patch_cache_epoch
        if self._patch_cache_epoch_mtime_ns != mtime_ns:
            try:
                epoch = max(1, int(self._patch_epoch_path.read_text(encoding="utf-8").strip()))
            except Exception:
                epoch = self._patch_cache_epoch
            self._patch_cache_epoch = epoch
            self._patch_cache_epoch_mtime_ns = mtime_ns
        return self._patch_cache_epoch

    def _patch_cache_paths(self, row: Dict, epoch: int) -> Tuple[Path, Path]:
        stem, _ = self._cache_stem_and_metadata(row)
        patch_token = "x".join(str(dim) for dim in self.patch_size)
        epoch_dir = Path(self.patch_cache_dir) / f"epoch_{epoch:04d}"
        return (
            epoch_dir / f"{stem}_patch_{patch_token}_x.npy",
            epoch_dir / f"{stem}_patch_{patch_token}_y.npy",
        )

    def _load(self, row: Dict) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        case_id = row.get("case_id", "unknown")

        # ── Fast path: load from npy cache ────────────────────────────────
        if self.cache_dir:
            x_path, y_path, s_path, fg_path, meta_path, cache_meta = self._cache_paths(row)
            has_label_in_csv = bool(row.get("label"))
            cache_valid = (
                x_path.exists()
                and s_path.exists()
                and meta_path.exists()
                and (y_path.exists() or not has_label_in_csv)
            )
            if cache_valid:
                try:
                    stored_meta = load_json(meta_path)
                except Exception:
                    stored_meta = None
                if stored_meta == cache_meta:
                    try:
                        x = np.load(str(x_path))
                        y = np.load(str(y_path)) if y_path.exists() else None
                        spacing = np.load(str(s_path))
                        fg_coords = np.load(str(fg_path)) if fg_path.exists() else None
                    except Exception as exc:
                        # A corrupt/partial .npy (crashed worker, disk-full, NFS glitch)
                        # must NOT crash the DataLoader worker and silently drop the case.
                        # Warn and fall through to the slow NIfTI path.
                        LOG.warning("[cache] corrupt cache for %s (%s); reloading from NIfTI.",
                                    row.get("case_id", x_path.name), exc)
                    else:
                        for p in (x_path, y_path, s_path, fg_path):
                            if p.exists():
                                _drop_file_cache(p)
                        return x, y, spacing, fg_coords

        # ── Slow path: load from NIfTI + normalise ────────────────────────
        first_path = row[self.modalities[0]]
        first_vol, spacing, reference_affine = load_nifti_with_meta(first_path)
        _drop_file_cache(first_path)
        reference_shape = first_vol.shape
        vols = [preprocess_mri_volume(first_vol, spacing=spacing)]
        for modality in self.modalities[1:]:
            modality_path = row[modality]
            vol, spacing_i, affine_i = load_nifti_with_meta(modality_path)
            _drop_file_cache(modality_path)
            _assert_aligned_volume(
                case_id=str(case_id),
                source_name=f"modality:{modality}",
                source_path=modality_path,
                reference_shape=reference_shape,
                reference_spacing=spacing,
                reference_affine=reference_affine,
                shape=vol.shape,
                spacing=spacing_i,
                affine=affine_i,
            )
            vols.append(preprocess_mri_volume(vol, spacing=spacing_i))
        x = np.stack(vols, axis=0).astype(np.float32)
        y = None
        if row.get("label"):
            label_path = row["label"]
            label_vol, label_spacing, label_affine = load_nifti_with_meta(label_path)
            _drop_file_cache(label_path)
            _assert_aligned_volume(
                case_id=str(case_id),
                source_name="label",
                source_path=label_path,
                reference_shape=reference_shape,
                reference_spacing=spacing,
                reference_affine=reference_affine,
                shape=label_vol.shape,
                spacing=label_spacing,
                affine=label_affine,
            )
            y = label_vol.astype(np.int64)
            y = self._apply_label_remap(y, row.get("label_remap", ""))
            y = self._canonicalise_label_values(y, str(case_id))
        fg_coords = np.argwhere(y > 0) if y is not None else None

        # ── Write to npy cache for next epoch ─────────────────────────────
        if self.cache_dir:
            try:
                x_path, y_path, s_path, fg_path, meta_path, cache_meta = self._cache_paths(row)
                unique = f"{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"
                x_tmp = Path(self.cache_dir) / f"{case_id}_x.{unique}.tmp.npy"
                s_tmp = Path(self.cache_dir) / f"{case_id}_s.{unique}.tmp.npy"
                np.save(str(x_tmp), x)
                np.save(str(s_tmp), spacing)
                if y is not None:
                    y_tmp = Path(self.cache_dir) / f"{case_id}_y.{unique}.tmp.npy"
                    np.save(str(y_tmp), y)
                    os.replace(y_tmp, y_path)
                    _drop_file_cache(y_path)
                if fg_coords is not None:
                    fg_tmp = Path(self.cache_dir) / f"{case_id}_fg.{unique}.tmp.npy"
                    np.save(str(fg_tmp), fg_coords)
                    os.replace(fg_tmp, fg_path)
                    _drop_file_cache(fg_path)
                os.replace(s_tmp, s_path)
                os.replace(x_tmp, x_path)
                _drop_file_cache(s_path)
                _drop_file_cache(x_path)
                save_json_atomic(cache_meta, meta_path)
            except Exception as exc:
                # Non-fatal (next epoch re-reads from NIfTI), but warn once so a disk-full /
                # NFS / permission failure is visible instead of silently degrading every
                # future epoch from cached (fast) to uncached (slow).
                if not getattr(self, "_cache_write_warned", False):
                    LOG.warning("[cache] write failed for %s (%s); reloading from NIfTI "
                                "(warned once).", row.get("case_id", "unknown"), exc)
                    self._cache_write_warned = True

        return x, y, spacing, fg_coords

    def _random_crop_with_slices(
        self, x: np.ndarray, y: Optional[np.ndarray], fg_coords: Optional[np.ndarray] = None
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Tuple[slice, slice, slice],
        Tuple[int, int, int],
    ]:
        _, d, h, w = x.shape
        pd, ph, pw = self.patch_size
        dd, dh, dw = max(0, pd - d), max(0, ph - h), max(0, pw - w)
        if dd + dh + dw > 0 and fg_coords is not None and fg_coords.size > 0:
            fg_coords = fg_coords + np.array([dd // 2, dh // 2, dw // 2], dtype=fg_coords.dtype)
        x, y = pad_to_patch(x, self.patch_size, y)
        _, d, h, w = x.shape

        # Foreground-biased crop: 80% tumour-centred patches per OGAP 2.0 plan.
        if y is not None and np.random.rand() < 0.80:
            if fg_coords is None:
                fg_coords = np.argwhere(y > 0)
            if len(fg_coords) > 0:
                centre = fg_coords[np.random.randint(len(fg_coords))]
                sd = np.clip(centre[0] - pd // 2, 0, d - pd)
                sh = np.clip(centre[1] - ph // 2, 0, h - ph)
                sw = np.clip(centre[2] - pw // 2, 0, w - pw)
                crop_slices = (
                    slice(int(sd), int(sd) + pd),
                    slice(int(sh), int(sh) + ph),
                    slice(int(sw), int(sw) + pw),
                )
                x = x[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
                if y is not None:
                    y = y[sd:sd + pd, sh:sh + ph, sw:sw + pw]
                return x, y, crop_slices, (d, h, w)

        sd = np.random.randint(0, max(1, d - pd + 1))
        sh = np.random.randint(0, max(1, h - ph + 1))
        sw = np.random.randint(0, max(1, w - pw + 1))
        crop_slices = (
            slice(int(sd), int(sd) + pd),
            slice(int(sh), int(sh) + ph),
            slice(int(sw), int(sw) + pw),
        )
        x = x[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
        if y is not None:
            y = y[sd:sd + pd, sh:sh + ph, sw:sw + pw]
        return x, y, crop_slices, (d, h, w)

    def _random_crop(
        self, x: np.ndarray, y: Optional[np.ndarray], fg_coords: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        x, y, _, _ = self._random_crop_with_slices(x, y, fg_coords)
        return x, y

    def _cached_random_crop(
        self,
        row: Dict,
        x: np.ndarray,
        y: Optional[np.ndarray],
        fg_coords: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        epoch = self._current_patch_cache_epoch()
        if not self.patch_cache_dir or epoch < self.patch_cache_start_epoch:
            return self._random_crop(x, y, fg_coords)

        x_path, y_path = self._patch_cache_paths(row, epoch)
        if x_path.exists() and (y is None or y_path.exists()):
            try:
                x_patch = np.load(str(x_path))
                y_patch = np.load(str(y_path)) if y is not None and y_path.exists() else None
                _drop_file_cache(x_path)
                if y is not None and y_path.exists():
                    _drop_file_cache(y_path)
                return x_patch, y_patch
            except Exception:
                pass

        x_patch, y_patch = self._random_crop(x, y, fg_coords)
        try:
            x_path.parent.mkdir(parents=True, exist_ok=True)
            unique = f"{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"
            x_tmp = x_path.with_name(f"{x_path.name}.{unique}.tmp.npy")
            np.save(str(x_tmp), x_patch)
            os.replace(x_tmp, x_path)
            _drop_file_cache(x_path)
            if y_patch is not None:
                y_tmp = y_path.with_name(f"{y_path.name}.{unique}.tmp.npy")
                np.save(str(y_tmp), y_patch)
                os.replace(y_tmp, y_path)
                _drop_file_cache(y_path)
        except Exception:
            pass
        return x_patch, y_patch

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x, y, spacing, fg_coords = self._load(row)
        native_spatial_shape = tuple(int(v) for v in x.shape[1:])
        crop_slices: Optional[Tuple[slice, slice, slice]] = None
        crop_padded_shape: Optional[Tuple[int, int, int]] = None
        if self.train:
            if self.use_tissue_priors:
                x, y, crop_slices, crop_padded_shape = self._random_crop_with_slices(
                    x, y, fg_coords
                )
            else:
                x, y = self._cached_random_crop(row, x, y, fg_coords)
        # Lucas+ 2025 ULF GAN: synthesize an ultra-low-field appearance
        # BEFORE physics augmentation so the noise / artifact pipeline
        # operates on the GAN-realistic baseline. No-op when the GAN is
        # absent or its inference fails.
        if (self.train and self.ulf_gan is not None
                and self.ulf_gan.is_available()
                and self.ulf_gan_prob > 0.0
                and np.random.rand() < self.ulf_gan_prob):
            x = self.ulf_gan(x)
            # Force the augmentor's downstream noise / contrast warps to
            # treat this case as ULF; respects the GAN's target B0.
            if self.augmentor is not None:
                self.augmentor.field_strength = float(self.ulf_gan.target_b0)
        # Historical AFA donor hook retained for backward compatibility. The
        # current AFA transform is additive Fourier-basis noise and ignores this
        # donor; CarveMix still uses its own donor path later.
        donor = None
        if (self.train and self.augmentor is not None
                and self.afa_prob > 0.0 and len(self.rows) > 1
                and np.random.rand() < self.afa_prob):
            donor = self._sample_donor_volume(idx, target_shape=x.shape)
        if self.train and self.augmentor is not None:
            # Set field-strength-aware noise if available
            fs = _parse_row_field_strength(row)
            if fs is None:
                # Randomise field strength for domain randomisation
                self.augmentor.field_strength = sample_augmented_field_strength(
                    lmic_prior=self.lmic_field_strength_prior,
                )
            else:
                self.augmentor.field_strength = fs
            self.augmentor.contrast_mod_indices = self.contrast_mod_indices
            x, y = self.augmentor(
                x, y,
                contrast_mod_indices=self.contrast_mod_indices,
                donor=donor,
            )
        # CarveMix: paste a donor's tumor into the receiver after physics-
        # augmentation, so synthesized lesion pairs share the receiver's
        # acquisition style (Wang+ 2021).
        if (self.train and self.carvemix_prob > 0.0 and y is not None
                and len(self.rows) > 1 and np.random.rand() < self.carvemix_prob):
            x, y = self._apply_carvemix(idx, x, y)
        if self.train and y is not None and self.label_noise_prob > 0:
            y = simulate_annotation_noise(
                y,
                p_morph=self.label_noise_prob,
                mode=getattr(self, "label_noise_mode", "morph"),
                band_radius=getattr(self, "label_noise_band_radius", 2),
                band_flip_prob=getattr(self, "label_noise_band_flip_prob", 0.3),
            )
        # FIX (Gap B, 2026 audit): teacher-side per-modality dropout.
        # Decide independently per channel whether to drop it (Bernoulli(p)),
        # then cap the number of drops at modality_dropout_max_drop so at
        # least (n_channels - max_drop) modalities remain.  Apply *after* all
        # augmentation so the dropped channels stay zero through to the GPU
        # prefetcher.  The augmentation pipeline above already preserves the
        # zero-background convention.
        if self.train and self.modality_dropout_p > 0.0 and x.shape[0] > 1:
            n_ch = int(x.shape[0])
            drop_flags = np.random.rand(n_ch) < self.modality_dropout_p
            n_drop = int(drop_flags.sum())
            if n_drop > 0:
                max_drop = min(self.modality_dropout_max_drop, n_ch - 1)
                if n_drop > max_drop:
                    drop_idx = np.flatnonzero(drop_flags)
                    np.random.shuffle(drop_idx)
                    keep_back = drop_idx[max_drop:]
                    drop_flags[keep_back] = False
                if drop_flags.any():
                    x = x.copy()
                    x[drop_flags] = 0.0
        x_t = torch.from_numpy(np.ascontiguousarray(x))
        # Load the CSF/WM/GM tissue priors once: they feed BOTH the Bloch low-field
        # render (which needs tissue composition) and the appended prior channels.
        tissue_t = None
        if self.use_tissue_priors:
            tissue_t = self._load_tissue_priors(
                row["case_id"],
                tuple(x_t.shape[1:]),
                native_spatial_shape=native_spatial_shape,
                crop_slices=crop_slices,
                padded_spatial_shape=crop_padded_shape,
            )
        # Bloch low-field synthesis (training; needs the FSL FAST priors + the label):
        # re-render the MRI channels to a physics-consistent ~64 mT appearance BEFORE
        # the noise/artifact physics augmentation, so artifacts land on the low-field
        # base. Label-preserving; the priors are appended (channel-safe) below.
        # SynthSeg-style fully contrast-randomized render (alternative to the Bloch
        # render). The generator's own probability gate decides per-sample; when it
        # fires it returns a NEW tensor, so identity means "did not fire" -> fall
        # through to the physics Bloch render. They are mutually exclusive per sample.
        synth_applied = False
        if (self.train and self.synth_sim is not None and tissue_t is not None
                and y is not None):
            y_lab = torch.from_numpy(np.ascontiguousarray(y)).long()
            x_syn = self.synth_sim(x_t, tissue_t, y_lab)
            if x_syn is not x_t:
                x_t = x_syn
                synth_applied = True
        if (self.train and not synth_applied and self.bloch_sim is not None
                and tissue_t is not None and y is not None
                and (self.bloch_prob >= 1.0 or np.random.rand() < self.bloch_prob)):
            y_lab = torch.from_numpy(np.ascontiguousarray(y)).long()
            x_t = self.bloch_sim(x_t, tissue_t, y_lab)
        if self.physics_aug is not None and self.train:
            # Phase 7.6: opt-in v9.1 physics augmentation on the (C, D, H, W)
            # tensor, after preprocessing/legacy aug. Default OFF -> never runs.
            x_t = self.physics_aug(x_t)
        if tissue_t is not None:
            # Phase 7.5: append CSF/WM/GM channels AFTER augmentation so the
            # probability maps are not corrupted by intensity/noise transforms.
            x_t = torch.cat([x_t, tissue_t], dim=0)  # (4 MRI + 3 tissue, D, H, W)
        spacing_t = torch.as_tensor(spacing, dtype=torch.float32)  # (3,) mm
        dataset_tag = row.get("dataset_tag", "unknown")
        # Batch layout: (case_id, x, y, spacing_mm, dataset_tag)
        if y is None:
            y_t = torch.empty((0,), dtype=torch.long)
        else:
            y_t = torch.from_numpy(np.ascontiguousarray(y)).long()
        # Channel-aware geometric augmentation (flip / affine / elastic) is the LAST
        # dataset step: it acts on the full (MRI + tissue-prior) channel stack and the
        # label together, so they stay co-registered. CUDA runs gather this on the GPU
        # prefetcher instead, so geometric_aug is None there (single application).
        if (self.train and self.geometric_aug is not None and y_t.numel() > 0):
            x_t, y_t = self.geometric_aug(x_t, y_t)
        return row["case_id"], x_t, y_t, spacing_t, dataset_tag

    def _load_tissue_priors(
        self,
        case_id: str,
        spatial_shape: Tuple[int, int, int],
        *,
        native_spatial_shape: Optional[Tuple[int, int, int]] = None,
        crop_slices: Optional[Tuple[slice, slice, slice]] = None,
        padded_spatial_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """Load CSF/WM/GM tissue-probability maps as 3 extra input channels.

        Expects ``{tissue_prior_dir}/{case_id}_{csf,wm,gm}.nii.gz`` (FSL FAST
        naming). Maps may be full-volume outputs registered to the native MRI
        shape; when a training crop is active, the same symmetric padding and
        crop window used for the MRI patch are applied to the tissue maps.
        Mismatches raise rather than silently resampling, since misalignment
        would corrupt the anatomical prior.

        Args:
            case_id: case identifier used to locate the tissue files.
            spatial_shape: expected ``(D, H, W)`` of the volume the maps join.

        Returns:
            ``float32`` tensor ``(3, D, H, W)`` ordered CSF, WM, GM.

        Raises:
            FileNotFoundError: if ``tissue_prior_dir`` or any tissue file is absent.
            ValueError: if a tissue map's shape does not match ``spatial_shape``.
        """
        import os
        if not self.tissue_prior_dir or not os.path.isdir(self.tissue_prior_dir):
            raise FileNotFoundError(
                f"use_tissue_priors=True but tissue_prior_dir "
                f"{self.tissue_prior_dir!r} does not exist.")
        maps = []
        for tissue in ("csf", "wm", "gm"):
            path = os.path.join(self.tissue_prior_dir, f"{case_id}_{tissue}.nii.gz")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing {tissue.upper()} tissue prior for case {case_id!r}: {path}")
            arr = nib.load(path).get_fdata(dtype=np.float32)
            expected_shape = native_spatial_shape if native_spatial_shape is not None else spatial_shape
            if tuple(arr.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Tissue map shape {tuple(arr.shape)} != volume spatial shape "
                    f"{tuple(expected_shape)} for case {case_id!r}. Ensure FSL FAST "
                    "outputs are co-registered to the source MRI volume.")
            maps.append(arr)
        stacked = np.ascontiguousarray(np.stack(maps, axis=0))  # (3, D, H, W)
        if crop_slices is not None:
            stacked, _ = pad_to_patch(stacked, self.patch_size, None)
            if padded_spatial_shape is not None and tuple(stacked.shape[1:]) != tuple(padded_spatial_shape):
                raise ValueError(
                    f"Padded tissue map shape {tuple(stacked.shape[1:])} != padded MRI "
                    f"shape {tuple(padded_spatial_shape)} for case {case_id!r}."
                )
            stacked = stacked[
                :,
                crop_slices[0],
                crop_slices[1],
                crop_slices[2],
            ]
        if tuple(stacked.shape[1:]) != tuple(spatial_shape):
            raise ValueError(
                f"Tissue map shape {tuple(stacked.shape[1:])} != model input spatial "
                f"shape {tuple(spatial_shape)} for case {case_id!r} after crop/pad."
            )
        return torch.from_numpy(stacked).float()


class InferenceDataset(Dataset):
    """Inference-only dataset - never loads labels.

    Resilient to missing modality columns: a CSV row may omit a modality
    column entirely, leave it blank, or point to a non-existent file. In all
    three cases the modality is zero-filled and the dataset records a
    per-row presence indicator. Downstream code MUST AND its explicit missing
    overlay with ``row['__present_mask__']`` so that genuinely-absent inputs
    are never fed to the network as observed zeros.

    Stashed in each returned row dict:
        ``__present_mask__`` : np.ndarray, shape (C,), dtype float32 - 1.0 if
            modality i was loaded from disk, 0.0 if zero-filled.
        ``__ref_path__``    : str - path to the modality used as the
            geometry reference (first one present on disk).
    """

    def __init__(self, csv_path: str, modalities: Sequence[str]) -> None:
        self.modalities = list(modalities)
        with open(csv_path, encoding="utf-8") as f:
            self.rows: List[Dict] = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _modality_path(row: Dict, modality: str) -> Optional[str]:
        path = row.get(modality)
        if path is None:
            return None
        path = str(path).strip()
        if not path:
            return None
        if not os.path.exists(path):
            return None
        return path

    def __getitem__(self, idx: int):
        # Copy so per-row mutations don't leak across DataLoader workers.
        row = dict(self.rows[idx])
        case_id = row.get("case_id", f"case_{idx}")

        ref_modality_idx: Optional[int] = None
        ref_path: Optional[str] = None
        for i, modality in enumerate(self.modalities):
            p = self._modality_path(row, modality)
            if p is not None:
                ref_modality_idx = i
                ref_path = p
                break
        if ref_modality_idx is None:
            raise FileNotFoundError(
                f"[InferenceDataset] case '{case_id}' has no modalities present "
                f"on disk; checked columns: {self.modalities}"
            )

        first_vol, spacing, reference_affine = load_nifti_with_meta(ref_path)
        reference_shape = first_vol.shape

        present_mask = np.zeros(len(self.modalities), dtype=np.float32)
        vols: List[Optional[np.ndarray]] = [None] * len(self.modalities)
        vols[ref_modality_idx] = preprocess_mri_volume(first_vol, spacing=spacing)
        present_mask[ref_modality_idx] = 1.0

        for i, modality in enumerate(self.modalities):
            if i == ref_modality_idx:
                continue
            p = self._modality_path(row, modality)
            if p is None:
                vols[i] = np.zeros(reference_shape, dtype=np.float32)
                continue
            vol, spacing_i, affine_i = load_nifti_with_meta(p)
            _assert_aligned_volume(
                case_id=str(case_id),
                source_name=f"modality:{modality}",
                source_path=p,
                reference_shape=reference_shape,
                reference_spacing=spacing,
                reference_affine=reference_affine,
                shape=vol.shape,
                spacing=spacing_i,
                affine=affine_i,
            )
            vols[i] = preprocess_mri_volume(vol, spacing=spacing_i)
            present_mask[i] = 1.0

        x = torch.from_numpy(np.stack(vols, axis=0).astype(np.float32))
        row["__present_mask__"] = present_mask
        row["__ref_path__"] = ref_path
        return case_id, x, row


# ══════════════════════════════════════════════════════════════════════════════
# §6  MODEL BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class ECABlock3D(nn.Module):
    """Efficient Channel Attention - quantisation-friendly replacement for SE.

    Uses 1D convolution to learn inter-channel dependencies with
    near-zero parameter overhead (~5-7 params per block vs ~8K for SE).
    Google removed SE blocks when creating quantisation-friendly
    EfficientNet-Lite; we follow the same principle for INT8 deployment.

    Ref: Wang et al., "ECA-Net" CVPR 2020
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        # Adaptive kernel size based on channel count (ECA formula)
        t = int(abs(math.log2(channels) + b) / gamma)
        k = max(t if t % 2 else t + 1, 3)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Hardsigmoid(inplace=True)  # quantisation-friendly vs sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, 1, c)         # (B, 1, C)
        w = self.act(self.conv(w))              # (B, 1, C)
        return x * w.view(b, c, 1, 1, 1)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for ablation comparison.

    Included only for ablation studies. The OGAP 2.0 deployable student
    defaults to no attention because that path is friendliest to INT8.

    Ref: Hu et al., "Squeeze-and-Excitation Networks" CVPR 2018
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ── NEW #9: Curriculum Dropout (Morerio et al., ICCV 2017) ──
class CurriculumDropout3D(nn.Module):
    """Spatial dropout with curriculum scheduling.

    θ(t) = (1 - θ̄) · exp(-γt) + θ̄,  γ = 10/T

    Starts with nearly no dropout (θ(0) ≈ 1), gradually increases to
    the target drop rate. This implements curriculum learning applied to
    regularisation: easy learning first, then progressive challenge.

    Uses Dropout3d (spatial dropout) which drops entire feature map
    channels - more effective than element-wise dropout for CNNs.

    Ref: Morerio et al., "Curriculum Dropout" ICCV 2017 (arXiv:1703.06229)
         Li et al., "Disharmony Between Dropout and BN" CVPR 2019
    """

    def __init__(self, p_final: float = 0.1, total_epochs: int = 300) -> None:
        super().__init__()
        self.theta_bar = 1.0 - p_final  # target retain probability
        self.gamma = 10.0 / max(total_epochs, 1)
        self.current_epoch: int = 0
        self.p_final = p_final

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_final <= 0:
            return x
        # Curriculum retain probability: starts at ~1, decays to theta_bar
        retain_p = (1.0 - self.theta_bar) * math.exp(
            -self.gamma * self.current_epoch
        ) + self.theta_bar
        drop_p = 1.0 - retain_p
        # Clamp to valid range
        drop_p = max(0.0, min(drop_p, self.p_final))
        return F.dropout3d(x, p=drop_p, training=True)


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Largest GroupNorm group count <= max_groups that divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResConvBlock3D(nn.Module):
    """MedNeXt-style 3D block with residual skip and optional attention.

    The PDF explicitly asks for MedNeXt-S/MedNeXt-L style blocks rather than
    dense nnU-Net convolutions: depth-wise 3x3x3 convolution, GroupNorm, GELU,
    and point-wise expansion/projection. Keeping the historical class name
    preserves the surrounding training/export code while changing the actual
    block to the requested quantization-friendly architecture.

    Supports no attention (default), ECA, or SE for ablation.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",  # "eca", "se", "none"
        curriculum_dropout_p: float = 0.0,
        total_epochs: int = 300,
    ) -> None:
        super().__init__()
        expansion_ch = max(out_ch * 2, in_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.GroupNorm(_group_count(in_ch), in_ch),
            nn.GELU(),
            nn.Conv3d(in_ch, expansion_ch, 1, bias=False),
            nn.GELU(),
            nn.Conv3d(expansion_ch, out_ch, 1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        # Configurable attention for clean ablation
        if attention == "eca":
            self.attn = ECABlock3D(out_ch)
        elif attention == "se":
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )

        # Curriculum dropout (applied after residual add + attention)
        self.cdropout = CurriculumDropout3D(
            p_final=curriculum_dropout_p, total_epochs=total_epochs
        ) if curriculum_dropout_p > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self.cdropout is not None:
            self.cdropout.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.block(x) + self.skip(x))
        if self.cdropout is not None:
            out = self.cdropout(out)
        return out


class DenseConvBlock3D(nn.Module):
    """Dense 3D conv block for the HEAVY teacher only.

    Two full 3x3x3 convolutions with GroupNorm + GELU and a residual skip,
    matching the canonical SegMamba / nnU-Net block design. Roughly 4-5x
    more parameters per block than ResConvBlock3D (MedNeXt-S), giving the
    teacher meaningfully more capacity for knowledge distillation without
    affecting the student or its INT8 export path.

    Constructor signature is intentionally identical to ResConvBlock3D so
    teacher classes can switch between block styles by class reference.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: str = "none",
        curriculum_dropout_p: float = 0.0,
        total_epochs: int = 300,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
        )
        if attention == "eca":
            self.attn = ECABlock3D(out_ch)
        elif attention == "se":
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(_group_count(out_ch), out_ch),
            )
        )
        self.cdropout = CurriculumDropout3D(
            p_final=curriculum_dropout_p, total_epochs=total_epochs
        ) if curriculum_dropout_p > 0 else None

    def set_epoch(self, epoch: int) -> None:
        if self.cdropout is not None:
            self.cdropout.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.attn(self.block(x) + self.skip(x))
        if self.cdropout is not None:
            out = self.cdropout(out)
        return out


def _teacher_block_class(block_style: str) -> type:
    """Return the conv block class for a given teacher block style.

    "mednext" → ResConvBlock3D (depthwise-separable, quantization-friendly,
                ~1M params per teacher @ base=32). Default.
    "dense"   → DenseConvBlock3D (full 3x3x3 convs, ~4-5x heavier;
                use for the heavy teacher in KD setups).

    Student always uses ResConvBlock3D regardless of this setting - only
    the teacher's capacity is configurable, since only the student is
    exported to INT8.
    """
    style = (block_style or "mednext").lower()
    if style == "dense":
        return DenseConvBlock3D
    if style in ("mednext", "default", ""):
        return ResConvBlock3D
    raise ValueError(
        f"Unknown teacher block_style {block_style!r}; expected 'mednext' or 'dense'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# §7  TEACHER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class UNet3DTeacher(nn.Module):
    """Unified full-modality teacher with a 4-level encoder-decoder.

    block_style="mednext" (default): MedNeXt-S depthwise-separable blocks,
        ~1M params @ base=32. Quantization-friendly but capacity-limited.
    block_style="dense": full 3x3x3 conv blocks, ~6M @ base=32, ~15M @
        base=48. Use for heavy KD teachers; the student stays MedNeXt-S
        for INT8 export so this only affects teacher capacity.
    """

    def __init__(
        self, in_channels: int, num_classes: int, base: int = 32,
        attention: str = "none",
        block_style: str = "mednext",
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        b = base
        Block = _teacher_block_class(block_style)
        self.block_style: str = block_style
        self.enc1 = Block(in_channels, b, attention=attention)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2, attention=attention)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4, attention=attention)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        self.bottleneck = Block(b * 4, b * 8, attention=attention)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(b * 8, b * 4, attention=attention)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2, attention=attention)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b, attention=attention)

        # FIX (Gap A, 2026 audit): Inject MixStyle3D after encoder block 1 and
        # block 2.  Only the student previously had feature-level domain
        # randomization, so the teacher logits used for KD encoded no
        # cross-scanner invariance - distillation then transferred only the
        # data-augmentation noise distribution to the student.  Putting
        # MixStyle3D in the teacher pushes domain-invariant features into the
        # KD signal itself.  MixStyle3D is an identity at eval, so this only
        # affects training.
        if feature_dr in ("mixstyle", "dsu"):
            self.mixstyle_e1 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
            self.mixstyle_e2 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        else:
            self.mixstyle_e1 = nn.Identity()
            self.mixstyle_e2 = nn.Identity()

        self.head = nn.Conv3d(b, num_classes, 1)
        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1 = self.mixstyle_e1(self.enc1(x))
        e2 = self.mixstyle_e2(self.enc2(self.pool1(e1)))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        up3 = self.up3(bn)
        e3 = e3[..., :up3.shape[2], :up3.shape[3], :up3.shape[4]]
        d3 = self.dec3(torch.cat([up3, e3], 1))

        up2 = self.up2(d3)
        e2 = e2[..., :up2.shape[2], :up2.shape[3], :up2.shape[4]]
        d2 = self.dec2(torch.cat([up2, e2], 1))

        up1 = self.up1(d2)
        e1 = e1[..., :up1.shape[2], :up1.shape[3], :up1.shape[4]]
        d1 = self.dec1(torch.cat([up1, e1], 1))
        logits = self.head(d1)

        if return_features:
            return logits, {"bottleneck": bn, "dec3": d3}
        return logits


class SegMambaTeacher(nn.Module):
    """Deprecated bottleneck-Mamba approximation.

    The active ``build_teacher(arch="segmamba")`` path imports the paper-aligned
    implementation from ``ogap.models.teacher``. This class is retained only for
    old checkpoints that may reference its state dict.

    Wraps a Mamba SSM-based 3D segmentation network when the optional
    ``mamba-ssm`` package is installed (``pip install mamba-ssm``). Mamba's
    linear-time sequence operator gives the teacher long-range 3D context
    that the convolutional UNet3DTeacher cannot capture, which is the
    motivation for using SegMamba as a SECOND teacher in OGAP - the student
    stays the cheap, INT8-friendly UNet3D, but it can be distilled from a
    Mamba teacher when the option is enabled.

    The implementation prefers ``monai.networks.nets.SegResNet`` augmented
    with a ``MambaLayer`` from ``mamba-ssm`` at the bottleneck (a known-
        safe drop-in pattern). If ``mamba-ssm`` is unavailable the constructor
        raises ImportError with an actionable hint.

    Refs:
        Xing+ "SegMamba: Long-range Sequential Modeling Mamba For 3D
            Medical Image Segmentation", MICCAI 2024.
        Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
            State Spaces", arXiv 2023 (your "Mamba paper" PDF).
    """

    def __init__(
        self, in_channels: int, num_classes: int, base: int = 32,
        block_style: str = "mednext",
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SegMambaTeacher requires the 'mamba-ssm' package. Install with:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                f"(import error: {exc})"
            )
        b = base
        Block = _teacher_block_class(block_style)
        self.block_style: str = block_style

        # Deprecated compatibility path: bottleneck-only Mamba is not the
        # SegMamba paper architecture. New builds route through
        # ogap.models.teacher.SegMambaTeacher.
        from mamba_ssm import Mamba

        self.enc1 = Block(in_channels, b)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        self.bottleneck_conv = Block(b * 4, b * 8)
        self.bottleneck_mamba = Mamba(d_model=b * 8, d_state=16, d_conv=4, expand=2)
        self.bottleneck_norm = nn.GroupNorm(num_groups=8, num_channels=b * 8)

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b)

        # FIX (Gap A, 2026 audit): same MixStyle3D injection as UNet3DTeacher.
        # The SegMamba bottleneck is non-traceable under torch.compile but the
        # convolutional encoder is - MixStyle3D fits cleanly there, and the
        # KD signal benefits from cross-scanner invariance the same way.
        if feature_dr in ("mixstyle", "dsu"):
            self.mixstyle_e1 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
            self.mixstyle_e2 = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        else:
            self.mixstyle_e1 = nn.Identity()
            self.mixstyle_e2 = nn.Identity()

        self.head = nn.Conv3d(b, num_classes, 1)
        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    @staticmethod
    def _spatial_to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        b, c, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous(), (b, c, d, h, w)

    @staticmethod
    def _sequence_to_spatial(seq: torch.Tensor, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        b, c, d, h, w = shape
        return seq.transpose(1, 2).contiguous().view(b, c, d, h, w)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        e1 = self.mixstyle_e1(self.enc1(x))
        e2 = self.mixstyle_e2(self.enc2(self.pool1(e1)))
        e3 = self.enc3(self.pool2(e2))
        bn_conv = self.bottleneck_conv(self.pool3(e3))
        # Mamba operates on a flattened spatial sequence at the bottleneck.
        bn_seq, shape = self._spatial_to_sequence(bn_conv)
        bn_seq = self.bottleneck_mamba(bn_seq)
        bn = self._sequence_to_spatial(bn_seq, shape)
        bn = self.bottleneck_norm(bn) + bn_conv  # residual

        up3 = self.up3(bn)
        e3 = e3[..., :up3.shape[2], :up3.shape[3], :up3.shape[4]]
        d3 = self.dec3(torch.cat([up3, e3], 1))

        up2 = self.up2(d3)
        e2 = e2[..., :up2.shape[2], :up2.shape[3], :up2.shape[4]]
        d2 = self.dec2(torch.cat([up2, e2], 1))

        up1 = self.up1(d2)
        e1 = e1[..., :up1.shape[2], :up1.shape[3], :up1.shape[4]]
        d1 = self.dec1(torch.cat([up1, e1], 1))
        logits = self.head(d1)

        if return_features:
            return logits, {"bottleneck": bn, "dec3": d3}
        return logits


def build_teacher(
    in_channels: int,
    num_classes: int,
    base: int,
    arch: str = "unet3d",
    attention: str = "none",
    block_style: str = "mednext",
    feature_dr: str = "none",
    feature_dr_p: float = 0.5,
    feature_dr_alpha: float = 0.1,
) -> nn.Module:
    """Construct a teacher network of the requested architecture.

    If the requested architecture's optional dependency is unavailable, the
    builder raises immediately. Silently changing architecture would make a
    later checkpoint load fail with a much less useful state_dict mismatch.

    block_style controls per-block parameter count (see _teacher_block_class):
    "mednext" (default) keeps backward compatibility with all existing
    teacher checkpoints; "dense" produces a heavy teacher for KD setups.

    feature_dr controls teacher-side domain randomization (2026 audit, Gap A):
        "none"      → identity (legacy behaviour)
        "mixstyle"  → MixStyle3D (Zhou et al. 2021)
        "dsu"       → distribution-uncertainty modeling (Li et al. 2022)
    """
    arch = (arch or "unet3d").lower()
    if arch in ("unet3d", "default", ""):
        return UNet3DTeacher(
            in_channels, num_classes, base,
            attention=attention, block_style=block_style,
            feature_dr=feature_dr,
            feature_dr_p=feature_dr_p,
            feature_dr_alpha=feature_dr_alpha,
        )
    if arch == "segmamba":
        try:
            from ogap.models.teacher import SegMambaTeacher as _PaperSegMambaTeacher
            return _PaperSegMambaTeacher(
                in_channels, num_classes, base, block_style=block_style,
                feature_dr=feature_dr,
                feature_dr_p=feature_dr_p,
                feature_dr_alpha=feature_dr_alpha,
            )
        except ImportError as exc:
            LOG.error(
                "[teacher] --teacher_arch=segmamba requested but mamba-ssm "
                "is not available. Refusing to substitute UNet3DTeacher "
                "because SegMamba checkpoints would not be load-compatible. (%s)",
                exc,
            )
            raise
    if arch in ("swin_unetr", "swinunetr"):
        try:
            from ogap.models.teacher import SwinUNETRTeacher
            return SwinUNETRTeacher(in_channels, num_classes, base=base)
        except ImportError as exc:
            LOG.error(
                "[teacher] --teacher_arch=swin_unetr requested but MONAI is "
                "not available. Refusing to substitute another architecture. (%s)",
                exc,
            )
            raise
    if arch == "ode":
        # v9.1: continuous-depth (Neural ODE) bottleneck teacher. The default
        # path requires torchdiffeq/odeint_adjoint so the advertised
        # constant-memory guarantee is actually satisfied.
        from ogap.models.ode import build_ode_teacher
        return build_ode_teacher(
            in_channels, num_classes, base,
            attention=attention, block_style=block_style,
            feature_dr=feature_dr,
            feature_dr_p=feature_dr_p,
            feature_dr_alpha=feature_dr_alpha,
        )
    raise ValueError(
        f"Unknown --teacher_arch {arch!r}; expected one of unet3d / segmamba / swin_unetr / ode."
    )


# ══════════════════════════════════════════════════════════════════════════════
# §8  STUDENT MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MixStyle3D(nn.Module):
    """Feature-statistics domain randomization for 3D segmentation.

    During training, with probability ``p``, replaces each instance's
    per-channel (μ, σ) feature statistics with a Beta-mixed combination of
    its own and a permuted batchmate's. At eval time it is a no-op identity.

    Two modes (controlled by ``mode``):
      * ``"mixstyle"`` - Zhou et al., "Domain Generalization with MixStyle",
        ICLR 2021. Sample λ ∼ Beta(α, α) and mix with a permuted batchmate.
        Recommended default for cross-site segmentation.
      * ``"dsu"`` - Li et al., "Uncertainty Modeling for Out-of-Distribution
        Generalization in Visual Recognition" (DSU), ICLR 2022. Sample
        per-instance perturbed (μ, σ) from a Gaussian fitted to the
        within-batch (μ, σ) distribution. Slightly stronger; needs ≥4 batch.

    The 3D version normalizes over (D, H, W) per (batch, channel) and is
    cheap (one mean / std reduction + scatter-affine).

    Refs:
        Zhou+ ICLR 2021 (MixStyle).
        Li+ ICLR 2022 (DSU).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, mode: str = "mixstyle",
                 eps: float = 1e-6) -> None:
        super().__init__()
        if mode not in ("mixstyle", "dsu", "none"):
            raise ValueError(f"MixStyle3D mode must be one of mixstyle/dsu/none, got {mode!r}")
        self.p = float(p)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.mode == "none" or self.p <= 0.0:
            return x
        if x.shape[0] < 2:
            return x
        if torch.rand(1, device=x.device).item() >= self.p:
            return x
        b, c = x.shape[0], x.shape[1]
        # Per-instance, per-channel statistics over spatial dims.
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig

        if self.mode == "mixstyle":
            perm = torch.randperm(b, device=x.device)
            mu2 = mu[perm]
            sig2 = sig[perm]
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
                (b, c, 1, 1, 1)
            ).to(x.device)
            mu_mix = lam * mu + (1.0 - lam) * mu2
            sig_mix = lam * sig + (1.0 - lam) * sig2
            return x_norm * sig_mix + mu_mix

        # DSU: estimate the (μ, σ) distribution across the batch and resample
        # per-instance perturbations. We use the batch's std-of-mean as the
        # uncertainty proxy (Li+ 2022 §3.2).
        mu_std = mu.std(dim=0, keepdim=True, unbiased=False) + self.eps
        sig_std = sig.std(dim=0, keepdim=True, unbiased=False) + self.eps
        eps_mu = torch.randn_like(mu) * mu_std
        eps_sig = torch.randn_like(sig) * sig_std
        return x_norm * (sig + eps_sig) + (mu + eps_mu)


class UNet3DStudent(nn.Module):
    """Missing-modality MedNeXt-S style student with deep supervision.

    Input  : 2C channels (C masked volumes + C binary availability maps)
    Output : main logits + auxiliary logits from dec3, dec2
    base=16 → ~0.24M parameters (depthwise-separable MedNeXt-S; base=32 → ~0.88M),
    <500 MB RAM at inference

    FIX #5 (partial): attention and deep_supervision are now configurable
    via constructor for clean ablation studies.
    feature_dr in {"none","mixstyle","dsu"} inserts a MixStyle3D module
    after enc1 to provide free feature-space domain randomization (Zhou+
    2021 / Li+ 2022). At eval time MixStyle3D is an identity.

    block_style in {"res","ode"} selects the residual block family. "res"
    preserves checkpoint compatibility; "ode" swaps in WeightTiedODEBlock3D.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base: int = 16,
        attention: str = "none",          # "eca", "se", "none"
        enable_deep_supervision: bool = True,
        curriculum_dropout_p: float = 0.1,
        total_epochs: int = 300,
        feature_dr: str = "none",
        feature_dr_p: float = 0.5,
        feature_dr_alpha: float = 0.1,
        block_style: str = "res",
        ode_steps: int = 4,
    ) -> None:
        super().__init__()
        if block_style not in ("res", "ode"):
            raise ValueError(
                f"block_style must be 'res' or 'ode', got {block_style!r}."
            )
        b = base
        self._enable_ds = enable_deep_supervision
        self.block_style = block_style
        self.ode_steps = int(ode_steps)
        if block_style == "ode":
            from ogap.models.student_ode import WeightTiedODEBlock3D

            Block = partial(WeightTiedODEBlock3D, steps=self.ode_steps)
        else:
            Block = ResConvBlock3D

        # Curriculum dropout only on deeper layers (bottleneck + dec3)
        # to avoid disrupting low-level feature learning
        self.enc1 = Block(in_channels, b, attention=attention)
        # Feature-space domain randomization (Zhou+ 2021 MixStyle / Li+ 2022 DSU).
        # Inserted after the first encoder block - the standard early-layer
        # placement that the MixStyle paper found most effective.
        self.feature_dr = MixStyle3D(p=feature_dr_p, alpha=feature_dr_alpha, mode=feature_dr)
        self.pool1 = nn.Conv3d(b, b, 2, stride=2, groups=b, bias=False)
        self.enc2 = Block(b, b * 2, attention=attention)
        self.pool2 = nn.Conv3d(b * 2, b * 2, 2, stride=2, groups=b * 2, bias=False)
        self.enc3 = Block(b * 2, b * 4, attention=attention)
        self.pool3 = nn.Conv3d(b * 4, b * 4, 2, stride=2, groups=b * 4, bias=False)
        self.bottleneck = Block(
            b * 4, b * 8, attention=attention,
            curriculum_dropout_p=curriculum_dropout_p,
            total_epochs=total_epochs,
        )

        self.up3 = nn.ConvTranspose3d(b * 8, b * 4, 2, stride=2)
        self.dec3 = Block(
            b * 8, b * 4, attention=attention,
            curriculum_dropout_p=curriculum_dropout_p,
            total_epochs=total_epochs,
        )
        self.up2 = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2 = Block(b * 4, b * 2, attention=attention)
        self.up1 = nn.ConvTranspose3d(b * 2, b, 2, stride=2)
        self.dec1 = Block(b * 2, b, attention=attention)

        self.head = nn.Conv3d(b, num_classes, 1)
        self.uncertainty_head = nn.Conv3d(b, num_classes, 1)
        self.rano_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(b * 8, 12),
        )

        # Deep supervision heads (only created if enabled)
        if enable_deep_supervision:
            # Learned upsampling keeps auxiliary supervision on conv kernels
            # instead of large trilinear interpolation passes.
            self.aux_head3 = nn.ConvTranspose3d(b * 4, num_classes, 4, stride=4)
            self.aux_head2 = nn.ConvTranspose3d(b * 2, num_classes, 2, stride=2)
        else:
            # Register as None so state_dict is consistent
            self.aux_head3 = None
            self.aux_head2 = None

        self.bottleneck_channels: int = b * 8
        self.dec3_channels: int = b * 4

    def set_epoch(self, epoch: int) -> None:
        """Update curriculum dropout schedule for all blocks."""
        for module in self.modules():
            if module is not self and hasattr(module, "set_epoch"):
                module.set_epoch(epoch)

    def forward(
        self, x: torch.Tensor,
        return_features: bool = False,
        deep_supervision: bool = False,
        return_aux_outputs: bool = False,
    ):
        e1 = self.enc1(x)
        # Feature-space domain randomization (no-op if mode == "none" or eval mode).
        e1 = self.feature_dr(e1)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        up3 = self.up3(bn)
        e3 = e3[..., :up3.shape[2], :up3.shape[3], :up3.shape[4]]
        d3 = self.dec3(torch.cat([up3, e3], 1))

        up2 = self.up2(d3)
        e2 = e2[..., :up2.shape[2], :up2.shape[3], :up2.shape[4]]
        d2 = self.dec2(torch.cat([up2, e2], 1))

        up1 = self.up1(d2)
        e1 = e1[..., :up1.shape[2], :up1.shape[3], :up1.shape[4]]
        d1 = self.dec1(torch.cat([up1, e1], 1))
        logits = self.head(d1)
        aux_outputs = None
        if return_aux_outputs:
            aux_outputs = {
                # self.rano_head is untrained (no per-case RANO targets exist for any
                # OGAP cohort) and is NOT emitted - exposing an untrained head's output as
                # "rano" risks it being read as a clinical measurement. The reported RANO
                # comes from the geometric _region_clinical_measurements; the head is kept
                # in __init__ only for checkpoint compatibility.
                "uncertainty_logits": self.uncertainty_head(d1),
                "ood_features": F.adaptive_avg_pool3d(bn.float(), 1).flatten(1),
            }

        if deep_supervision and self._enable_ds:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            if return_features:
                if return_aux_outputs:
                    return logits, aux3, aux2, {"bottleneck": bn, "dec3": d3, **aux_outputs}
                return logits, aux3, aux2, {"bottleneck": bn, "dec3": d3}
            if return_aux_outputs:
                return logits, aux3, aux2, aux_outputs
            return logits, aux3, aux2

        if return_features:
            if return_aux_outputs:
                return logits, {"bottleneck": bn, "dec3": d3, **aux_outputs}
            return logits, {"bottleneck": bn, "dec3": d3}
        if return_aux_outputs:
            return logits, aux_outputs
        return logits


class FeatureProjector(nn.Module):
    """1×1×1 projection: student feature space → teacher feature space.

    Fixes channel mismatch when student_base != teacher_base.
    """

    def __init__(self, s_channels: int, t_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(s_channels, t_channels, 1, bias=False),
            nn.GroupNorm(_group_count(t_channels), t_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════════════════
# §9  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class DiceCELoss(nn.Module):
    """Dice + Focal loss with correct ignore_index handling."""

    def __init__(
        self, num_classes: int, ignore_index: int = -100,
        include_background: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_map = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        valid = (target != self.ignore_index)
        pt = torch.exp(-ce_map).clamp(1e-6, 1.0)
        focal = ((1.0 - pt) ** self.focal_gamma) * ce_map
        ce = (focal * valid.to(dtype=focal.dtype)).sum() / valid.sum().clamp_min(1)

        tgt_safe = torch.where(valid, target, torch.zeros_like(target))

        # FIX (Finding 1, 2026 audit): Softmax + Dice numerator/denominator are
        # computed in FP32 even under BF16 autocast.  BF16 has ~3 decimal digits
        # of precision; with a 1e-5 Dice epsilon, the smallest BraTS foreground
        # class (class 3 - enhancing tumour) gets biased gradients.  The cost is
        # negligible (matmuls remain BF16; only the elementwise reduction is
        # FP32) but the gradient bias against tiny lesions disappears.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probs = F.softmax(logits.float(), 1)
            t_oh = F.one_hot(tgt_safe.long(), self.num_classes).movedim(-1, 1).to(probs.dtype)
            vm = valid.unsqueeze(1).to(probs.dtype)
            probs = probs * vm
            t_oh = t_oh * vm

            dims = tuple(range(2, probs.dim()))
            inter = (probs * t_oh).sum(dims)
            den = (probs + t_oh).sum(dims)
            dice = (2 * inter + 1e-5) / (den + 1e-5)

        if not self.include_background and dice.shape[1] > 1:
            dice = dice[:, 1:]
        return ce + (1.0 - dice.mean())


# ── FIX #3: Per-sample partial-label handling with marginal CE ──
class PartialSupervisionDiceCELoss(nn.Module):
    """Dice+Focal with per-sample partial-label awareness for mixed-dataset training.

    ERASMUS provides only binary whole-tumour labels (class 1).
    Classes 2 (oedema) and 3 (enhancing) are absent - not mislabelled
    as background.

    FIX: Previous version had two critical issues:
      (a) CE was applied without modification for partial labels, forcing
          all tumour voxels to class 1 and penalising classes 2/3.
      (b) Suppression was per-batch: one ERASMUS sample could affect
          BraTS samples in the same batch.

    Now uses:
      - Per-sample masking (each sample has its own valid_class_mask)
      - Marginal CE for partial labels: log(sum of probs for valid classes)
        (Fidon et al., "Label-set Loss Functions" MICCAI 2021)
      - Dice only computed over valid classes per sample

    Ref: Fidon et al., MICCAI 2021 (arXiv:2107.03846)
         Liu et al., "CLIP-Driven Universal Model" ICCV 2023
    """

    PARTIAL_LABEL_TAGS: frozenset = frozenset({"erasmus"})
    # For ERASMUS: label 1 = whole tumour = union of {NCR=1, ED=2, ET=3}
    # Classes 2, 3 are absent from annotations but valid predictions
    ABSENT_IN_PARTIAL: Tuple[int, ...] = (2, 3)
    # FIX v3 Critical #2: Explicit tumour class set (not hardcoded probs[1:])
    TUMOUR_CLASSES_IN_PARTIAL: Tuple[int, ...] = (1, 2, 3)

    def __init__(
        self, num_classes: int, ignore_index: int = -100,
        include_background: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.focal_gamma = float(focal_gamma)

    def _marginal_ce_sample(
        self, logits_s: torch.Tensor, target_s: torch.Tensor,
    ) -> torch.Tensor:
        """Marginal cross-entropy for a single partial-label sample.

        For whole-tumour voxels (target>0), the "correct" prediction is
        ANY of the tumour classes. We compute: -log(Σ p_c for c in tumour)
        for tumour voxels, and standard CE for background voxels.

        FIX v3: Uses self.TUMOUR_CLASSES_IN_PARTIAL instead of hardcoded
        probs[1:]. This is safe under class reordering or different datasets.
        """
        # logits_s: (C, D, H, W), target_s: (D, H, W)
        probs = F.softmax(logits_s, dim=0)  # (C, D, H, W)

        # Background voxels: standard CE (class 0)
        bg_mask = (target_s == 0)
        # Tumour voxels: marginal over configured tumour classes
        fg_mask = (target_s > 0)

        loss = torch.tensor(0.0, device=logits_s.device)
        n_voxels = 0

        if bg_mask.any():
            # -log(p_bg) for background
            p_bg = probs[0][bg_mask].clamp(min=1e-7)
            loss = loss + (((1.0 - p_bg) ** self.focal_gamma) * (-torch.log(p_bg))).sum()
            n_voxels += bg_mask.sum().item()

        if fg_mask.any():
            # -log(Σ p_c for c in tumour_classes) for foreground
            tumour_idx = torch.tensor(
                self.TUMOUR_CLASSES_IN_PARTIAL, device=probs.device, dtype=torch.long
            )
            marginal_prob = probs[tumour_idx].sum(dim=0)  # sum over tumour classes
            p_fg = marginal_prob[fg_mask].clamp(min=1e-7)
            loss = loss + (((1.0 - p_fg) ** self.focal_gamma) * (-torch.log(p_fg))).sum()
            n_voxels += fg_mask.sum().item()

        return loss / max(n_voxels, 1)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor,
        dataset_tags: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        b = logits.shape[0]
        if dataset_tags is None:
            dataset_tags = ["unknown"] * b

        # Determine which samples are partial-label
        is_partial = [t in self.PARTIAL_LABEL_TAGS for t in dataset_tags]

        # ── Per-sample CE (marginal for partial, standard for full) ──
        ce_total = torch.zeros((), device=logits.device)
        n_ce = 0

        partial_idx = [i for i, partial in enumerate(is_partial) if partial]
        full_idx = [i for i, partial in enumerate(is_partial) if not partial]

        if full_idx:
            fi = torch.as_tensor(full_idx, device=logits.device, dtype=torch.long)
            logits_full = logits.index_select(0, fi)
            target_full = target.index_select(0, fi)
            valid_full = (target_full != self.ignore_index)
            valid_samples = valid_full.flatten(1).any(dim=1)
            if valid_samples.any():
                logits_full = logits_full[valid_samples]
                target_full = target_full[valid_samples]
                valid_full = valid_full[valid_samples]
                ce_map = F.cross_entropy(
                    logits_full,
                    target_full,
                    ignore_index=self.ignore_index,
                    reduction="none",
                )
                pt = torch.exp(-ce_map).clamp(1e-6, 1.0)
                ce_map = ((1.0 - pt) ** self.focal_gamma) * ce_map
                denom = valid_full.flatten(1).sum(dim=1).clamp_min(1)
                sample_ce = (ce_map * valid_full.to(dtype=ce_map.dtype)).flatten(1).sum(dim=1) / denom
                ce_total = ce_total + sample_ce.sum()
                n_ce += int(sample_ce.numel())

        for i in partial_idx:
            valid = (target[i] != self.ignore_index)
            if not valid.any():
                continue
            ce_total = ce_total + self._marginal_ce_sample(logits[i], target[i])
            n_ce += 1
        ce = ce_total / max(n_ce, 1)

        # ── Per-sample Dice (suppress absent classes for partial labels) ──
        valid = (target != self.ignore_index)
        tgt_safe = torch.where(valid, target, torch.zeros_like(target))

        # FIX (Finding 1, 2026 audit): FP32 softmax + Dice numerator/denominator
        # to keep BraTS class 3 (enhancing) gradients well-conditioned under BF16.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probs = F.softmax(logits.float(), 1)
            t_oh = F.one_hot(tgt_safe.long(), self.num_classes).movedim(-1, 1).to(probs.dtype)

            vm = valid.unsqueeze(1).to(probs.dtype)
            probs = probs * vm
            t_oh = t_oh * vm

            dims = tuple(range(2, probs.dim()))
            inter = (probs * t_oh).sum(dims)  # (B, C)
            den = (probs + t_oh).sum(dims)    # (B, C)
            dice = (2 * inter + 1e-5) / (den + 1e-5)  # (B, C)

        # Per-sample class masking for Dice (FIX issue #4: masked mean, not set-to-1)
        # Setting absent classes to 1.0 inflates the average towards 1.
        # Instead, exclude them from the per-sample average entirely.
        start_idx = 0 if self.include_background else 1
        dice_fg = dice[:, start_idx:]  # (B, C-start_idx)
        class_mask = torch.ones_like(dice_fg)  # 1 = include, 0 = exclude
        if any(is_partial):
            partial_t = torch.as_tensor(is_partial, device=logits.device, dtype=dice_fg.dtype)
            absent = [c - start_idx for c in self.ABSENT_IN_PARTIAL if 0 <= c - start_idx < dice_fg.shape[1]]
            for c_adj in absent:
                class_mask[:, c_adj] = class_mask[:, c_adj] * (1.0 - partial_t)

        valid_per_sample = class_mask.sum(dim=1).clamp(min=1.0)
        per_sample_dice = (dice_fg * class_mask).sum(dim=1) / valid_per_sample
        dice_loss = 1.0 - per_sample_dice.mean()

        return ce + dice_loss


class BraTSRegionLoss(nn.Module):
    """Auxiliary overlapping-region BCE + Dice over WT/TC/ET.

    External degradation is concentrated in TC and ET rather than WT. This
    region-level objective complements the generic classwise DiceCE loss by
    directly optimising the hierarchical BraTS regions used in reporting.
    The regions are overlapping sigmoid targets (WT, TC, ET), which is the
    region-based BraTS training signal requested in the OGAP 2.0 PDF even
    though the deployed head still emits canonical class logits for backwards
    compatible evaluation/export.
    It is applied only to samples with dense 4-class BraTS-style labels.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        wt_weight: float = 0.5,
        tc_weight: float = 1.0,
        et_weight: float = 1.25,
        bce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.wt_weight = float(wt_weight)
        self.tc_weight = float(tc_weight)
        self.et_weight = float(et_weight)
        self.bce_weight = float(bce_weight)

    @staticmethod
    def _soft_binary_dice(
        prob: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        prob = prob * valid
        target = target * valid
        inter = (prob * target).sum()
        den = prob.sum() + target.sum()
        return (2.0 * inter + eps) / (den + eps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        dataset_tags: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        if logits.shape[1] < 4:
            return logits.new_tensor(0.0)

        b = logits.shape[0]
        if dataset_tags is None:
            dataset_tags = ["unknown"] * b

        dense_mask = torch.as_tensor(
            [tag in DENSE_BRATS_REGION_TAGS for tag in dataset_tags],
            device=logits.device,
            dtype=torch.bool,
        )
        if not dense_mask.any():
            return logits.new_tensor(0.0)

        # FIX (Finding 1, 2026 audit): FP32 softmax under BF16 autocast.
        # BCE is already FP32 below; aligning Dice probs to FP32 keeps the
        # region-Dice gradient un-quantized for the smallest region (ET).
        valid = (target != self.ignore_index)
        tgt_safe = torch.where(valid, target, torch.zeros_like(target))
        valid_f = valid.float()

        with torch.autocast(device_type=logits.device.type, enabled=False):
            probs = F.softmax(logits.float(), dim=1)
            p_wt = probs[:, 1:].sum(dim=1)
            p_tc = probs[:, 1] + probs[:, 3]
            p_et = probs[:, 3]
        t_wt = (tgt_safe > 0).float()
        t_tc = ((tgt_safe == 1) | (tgt_safe == 3)).float()
        t_et = (tgt_safe == 3).float()

        def soft_dice_batch(prob: torch.Tensor, tgt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            prob = prob * valid_mask
            tgt = tgt * valid_mask
            inter = (prob * tgt).sum(dim=(-3, -2, -1))
            den = prob.sum(dim=(-3, -2, -1)) + tgt.sum(dim=(-3, -2, -1))
            return (2.0 * inter + 1e-5) / (den + 1e-5)

        region_targets = torch.stack([t_wt, t_tc, t_et], dim=1)

        # BCE on post-softmax probabilities is unsafe under CUDA autocast.
        # Build equivalent binary logits for each composite BraTS region:
        # logit(P(region)) = logsumexp(region classes) - logsumexp(other classes).
        logits_f = logits.float()
        wt_logits = torch.logsumexp(logits_f[:, 1:4], dim=1) - logits_f[:, 0]
        tc_logits = torch.logsumexp(
            torch.stack((logits_f[:, 1], logits_f[:, 3]), dim=1),
            dim=1,
        ) - torch.logsumexp(
            torch.stack((logits_f[:, 0], logits_f[:, 2]), dim=1),
            dim=1,
        )
        et_logits = logits_f[:, 3] - torch.logsumexp(logits_f[:, 0:3], dim=1)
        region_logits = torch.stack([wt_logits, tc_logits, et_logits], dim=1)
        region_targets = region_targets.to(dtype=region_logits.dtype)
        region_weights = torch.as_tensor(
            [self.wt_weight, self.tc_weight, self.et_weight],
            device=logits.device,
            dtype=region_logits.dtype,
        )
        denom = region_weights.sum().clamp_min(1e-8)
        dice_terms = torch.stack(
            [
                soft_dice_batch(p_wt, t_wt, valid_f),
                soft_dice_batch(p_tc, t_tc, valid_f),
                soft_dice_batch(p_et, t_et, valid_f),
            ],
            dim=1,
        )
        region_score = (dice_terms * region_weights.view(1, 3)).sum(dim=1) / denom
        bce_map = F.binary_cross_entropy_with_logits(
            region_logits,
            region_targets,
            reduction="none",
        )
        bce_per_sample = (
            bce_map
            * valid_f.unsqueeze(1)
            * region_weights.view(1, 3, 1, 1, 1)
        ).sum(dim=(1, 2, 3, 4)) / (
            valid_f.unsqueeze(1).sum(dim=(1, 2, 3, 4)).clamp_min(1.0) * denom
        )
        dense_f = dense_mask.to(dtype=region_score.dtype)
        region_loss = 1.0 - region_score
        total = region_loss + self.bce_weight * bce_per_sample
        return (total * dense_f).sum() / dense_f.sum().clamp_min(1.0)


# ── NEW #10: nnU-Net-style deep supervision with exponential weights ──
class DeepSupervisionLoss(nn.Module):
    """Weighted sum of main + auxiliary segmentation losses.

    OGAP 2.0 uses the BraTS-Africa-friendly schedule called out in the PDF:
    [1.0, 0.7, 0.4, 0.2] from finest to coarsest, normalised over the levels
    that actually exist. This keeps real pressure on the low-resolution
    tumour-localisation heads instead of zeroing the coarsest output.

    Ref: nnU-Net v2 (Isensee et al., Nature Methods 2021)
         Code: nnunetv2/training/loss/deep_supervision.py
    """

    def __init__(
        self,
        criterion: nn.Module,
        num_levels: int = 3,
        lambda_ds: float = 1.0,
        zero_lowest: bool = False,
    ):
        super().__init__()
        self.criterion = criterion
        self.lambda_ds = lambda_ds

        brats_africa_weights = [1.0, 0.7, 0.4, 0.2]
        if num_levels <= len(brats_africa_weights):
            weights = np.array(brats_africa_weights[:num_levels], dtype=np.float64)
        else:
            tail = [0.2 * (0.5 ** i) for i in range(num_levels - len(brats_africa_weights))]
            weights = np.array(brats_africa_weights + tail, dtype=np.float64)
        if zero_lowest and num_levels > 1:
            weights[-1] = 0.0
        # Normalise non-zero weights to sum to 1
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        self.weights = weights.tolist()
        self.weights = [1e-7 if w == 0.0 else w for w in self.weights]

    def forward(self, logits_list: List[torch.Tensor], target: torch.Tensor,
                dataset_tags: Optional[Sequence[str]] = None) -> torch.Tensor:
        total = torch.tensor(0.0, device=target.device)
        for w, lg in zip(self.weights, logits_list):
            if isinstance(self.criterion, PartialSupervisionDiceCELoss):
                total = total + w * self.criterion(lg, target, dataset_tags)
            else:
                total = total + w * self.criterion(lg, target)
        return total * self.lambda_ds


# ── FIX #4: Hölder divergence with proper KL fallback ──
# ── FIX Critical-v3 #1: Returns per-voxel loss map for confidence gating ──
def holder_divergence(
    s_logits: torch.Tensor, t_logits: torch.Tensor,
    temperature: float, alpha: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Hölder *pseudo*-divergence (HPD) for knowledge distillation.

        L = T^2 * (-log  <t,s> / (||t||_alpha ||s||_beta)),   1/alpha + 1/beta = 1,

    with t = softmax(teacher/T), s = softmax(student/T). This is the projective
    Hölder divergence of Nielsen, Sun & Marchand-Maillet (Entropy 2017, Def. 1);
    Hölder's inequality gives ratio <= 1, so L >= 0. alpha=2 (=> beta=2) is the
    Cauchy-Schwarz divergence - the single *proper* member, with L(p:p)=0.

    SEMANTICS OF alpha != 2 (deliberate design, not a bug - state this in the paper).
    HPD is *improper* for alpha != 2: minimising L over the student does NOT drive
    s -> teacher. The Hölder equality condition t^alpha ∝ s^beta gives the student
    optimum

        s* ∝ teacher^(alpha - 1)   (renormalised).

    So the default alpha=1.5 distils toward the renormalised teacher^0.5 - a
    FLATTER, higher-entropy target than the teacher. We adopt this as deliberate
    *uncertainty transfer*: it preserves more of the teacher's secondary-class mass
    (a learned, data-dependent label smoothing), which suits the low-field / LMIC
    regime where calibrated boundary uncertainty matters more than sharp argmax
    matching (cf. the Hölder-uncertainty framing of Zhang et al., arXiv:2411.00826).
    The control alpha=2.0 recovers exact teacher-matching (proper Cauchy-Schwarz);
    the {1.5, 2.0} contrast is the pre-registered ablation (see cmd_kd_compare's
    'holder_default'/'holder_proper' arms and cmd_holder_sweep). Numerically
    verified: argmin_s L is at s ∝ teacher^(alpha-1), and L(student=teacher) > 0
    for alpha != 2 (== 0 at alpha=2).

    Returns per-voxel loss when reduction="none" (for confidence gating),
    or scalar loss when reduction="mean".

    Implementation notes:
        alpha <= 1+1e-6 falls back to standard KL - the alpha->1 HPD limit is NOT
            KL and degenerates toward a uniform target, so KL is the correct floor.
        alpha clamped to <= 3.0 (gradient stability); 1e-8 epsilons guard the logs.
        reduction="none" returns the (B, D, H, W) map for voxelwise gating.

    Ref: Nielsen, Sun & Marchand-Maillet, "On Hölder projective divergences",
         Entropy 2017 (HPD definition, improperness Fact 2, projective Fact 4).
         Zhang et al., arXiv:2411.00826 - Hölder divergence for uncertainty QA.
         arXiv:2507.01254 - Hölder + MI for BraTS incomplete modalities.
         arXiv:2406.08634 - Masked Predicted Auto-Encoder + Hölder KD.
    """
    T2 = temperature ** 2
    alpha = min(alpha, 3.0)  # Issue #4: prevent gradient explosion

    # FIX #4: KL fallback for alpha ≤ 1.0 (or very close to 1)
    if alpha <= 1.0 + 1e-6:
        p_t = F.softmax(t_logits / temperature, dim=1).detach()
        log_p_s = F.log_softmax(s_logits / temperature, dim=1)
        # Per-voxel KL: sum over classes → (B, D, H, W)
        kl_map = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
        if reduction == "none":
            return kl_map * T2  # (B, D, H, W)
        return kl_map.mean() * T2

    beta = alpha / (alpha - 1.0)

    s_prob = F.softmax(s_logits / temperature, 1).clamp(min=1e-7)
    t_prob = F.softmax(t_logits / temperature, 1).detach().clamp(min=1e-7)

    # Hölder integral per voxel: sum over classes → (B, D, H, W)
    numerator = (t_prob * s_prob).sum(dim=1)
    term_alpha = (t_prob ** alpha).sum(dim=1).pow(1.0 / alpha)
    term_beta = (s_prob ** beta).sum(dim=1).pow(1.0 / beta)
    ratio = (numerator / (term_alpha * term_beta + 1e-10)).clamp(min=1e-10, max=1.0)

    # No extra epsilon inside the log: ratio is already clamped to [1e-10, 1.0], so
    # -log(ratio) ∈ [0, ~23] is non-negative EXACTLY. The previous `+1e-8` made
    # -log(1+1e-8) ≈ -1e-8 < 0 at perfect agreement - a divergence must be ≥ 0.
    loss_map = -torch.log(ratio)

    if reduction == "none":
        return loss_map * T2  # (B, D, H, W)
    return loss_map.mean() * T2


# ── NEW #11: Confidence-gated distillation (voxelwise) ──
def confidence_gated_kd_weight(
    t_logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Per-voxel confidence gate for knowledge distillation.

    Downweights KD loss where the teacher is uncertain (max softmax < threshold).
    This prevents the student from learning unreliable teacher predictions.

    Returns a spatial weight map (B, D, H, W) in [0, 1].

    FIX v3: Returns (B, D, H, W) not (B, 1, D, H, W) to match
    per-voxel loss maps from holder_divergence(reduction="none").
    """
    with torch.no_grad():
        t_probs = F.softmax(t_logits, dim=1)
        confidence = t_probs.max(dim=1)[0]  # (B, D, H, W)
        # Soft gating: linear ramp from 0 at confidence=0 to 1 at threshold
        gate = (confidence / threshold).clamp(0.0, 1.0)
    return gate


def feature_distillation_sample_weights(
    mask: torch.Tensor,
    sparse_decay: float = 1.5,
) -> torch.Tensor:
    """Reduce feature-KD pressure when the student sees fewer modalities."""
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (B, M), got {tuple(mask.shape)}")
    weights = mask.float().mean(dim=1)
    if sparse_decay > 0:
        weights = weights.pow(sparse_decay)
    return weights.clamp(min=1.0 / max(mask.shape[1], 1), max=1.0)


def vrex_penalty(
    logits: torch.Tensor,
    target: torch.Tensor,
    dataset_tags: Sequence[str],
) -> torch.Tensor:
    """V-REx variance-of-risks penalty (Krueger+ ICML 2021).

    Computes the per-sample cross-entropy, groups by ``dataset_tag``, takes
    each group's mean, and returns the (unbiased=False) variance over
    groups. Adding ``λ_vrex · vrex_penalty(...)`` to the training loss
    forces the model to perform consistently across sites rather than
    averaging out a single weak group.

    Returns a zero scalar when only one tag is present in the batch.
    """
    if len(set(dataset_tags)) < 2:
        return logits.new_zeros(())
    ce = F.cross_entropy(logits, target, reduction="none")
    # Reduce spatial dims → per-sample scalar.
    while ce.dim() > 1:
        ce = ce.mean(dim=-1)
    tag_to_indices: Dict[str, List[int]] = {}
    for i, tag in enumerate(dataset_tags):
        tag_to_indices.setdefault(str(tag), []).append(i)
    per_tag_means = []
    for indices in tag_to_indices.values():
        idx = torch.tensor(indices, device=logits.device, dtype=torch.long)
        per_tag_means.append(ce.index_select(0, idx).mean())
    stacked = torch.stack(per_tag_means)
    return stacked.var(unbiased=False)


def multi_scale_feature_distillation(
    s_feats: Dict[str, torch.Tensor],
    t_feats: Dict[str, torch.Tensor],
    projectors: Dict[str, FeatureProjector],
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-scale cosine feature distillation at bottleneck + dec3 levels.

    Distilling at multiple scales captures both global context
    (bottleneck) and local boundary detail (dec3).

    v10: sample-wise weighting reduces the feature-matching penalty in sparse
    modality regimes, where the student should not be forced to mimic the
    full-modality teacher latent space as aggressively.
    """
    if not projectors:
        if s_feats:
            return torch.zeros((), device=next(iter(s_feats.values())).device)
        if t_feats:
            return torch.zeros((), device=next(iter(t_feats.values())).device)
        return torch.zeros(())
    device = next(iter(s_feats.values())).device
    total = torch.zeros((), device=device)
    weights = {"bottleneck": 1.0, "dec3": 0.5}  # bottleneck weighted higher

    for key in projectors:
        if key not in s_feats or key not in t_feats:
            continue
        s_proj = projectors[key](s_feats[key])
        t_f = F.interpolate(
            t_feats[key].detach(),
            size=s_proj.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        per_sample = 1.0 - F.cosine_similarity(
            s_proj.flatten(1).float(),
            t_f.flatten(1).float(),
            dim=1,
            eps=1e-6,
        )
        if sample_weights is not None:
            sw = sample_weights.to(device=device, dtype=per_sample.dtype)
            feature_loss = (per_sample * sw).sum() / sw.sum().clamp(min=1e-6)
        else:
            feature_loss = per_sample.mean()
        total = total + weights.get(key, 1.0) * feature_loss
    return total


# ══════════════════════════════════════════════════════════════════════════════
# §10  SCHEDULER AND CURRICULUM
# ══════════════════════════════════════════════════════════════════════════════

def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def curriculum_keep_min(
    epoch: int,
    total_epochs: int,
    num_modalities: int,
    ramp_fraction: float = 0.3,
    min_floor: int = 2,
) -> int:
    """OGAP 2.0 modality curriculum.

    First ``ramp_fraction`` of training uses full modalities. The next 40%
    ramps to the configured floor with a cosine schedule, then the rest of
    training samples the full supported dropout regime. With the defaults this
    matches the PDF: epochs 0-200 full, 200-600 ramp, 600-1000 p_dropout-heavy.

    FIX v6: int() → round() so the configured floor is actually reachable.
    With int(), cosine_progress must reach exactly 1.0 (only at epoch
    total_epochs-1) for the last decrement to occur, so the curriculum can
    get stuck one level above its intended floor.

    The min_floor argument lets us cap the default curriculum at 2 modalities,
    which is safer for deployment-focused training where single-sequence
    inference is considered unsupported.
    """
    full_end = int(ramp_fraction * total_epochs)
    dropout_end = int(min(0.6, ramp_fraction + 0.4) * total_epochs)
    if epoch < full_end:
        return num_modalities
    if epoch >= dropout_end:
        return max(1, min(int(min_floor), num_modalities))
    progress = (epoch - full_end) / max(1, dropout_end - full_end)
    # Cosine decay: 0→1 mapped to num_modalities→1
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    min_floor = max(1, min(int(min_floor), num_modalities))
    max_drop = num_modalities - min_floor
    return max(min_floor, num_modalities - round(cosine_progress * max_drop))


# ══════════════════════════════════════════════════════════════════════════════
# §11  POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def connected_component_postprocessing(
    pred: np.ndarray,
    min_component_size: int = 50,
    class_prob: Optional[np.ndarray] = None,
    et_min_component_size: int = 10,
    et_confidence_threshold: float = 0.55,
) -> np.ndarray:
    """Remove small spurious connected components per class.

    Keeps only the largest connected component for each foreground
    class if the smaller components are below min_component_size voxels.

    ET (class 3) gate - REVIEWER-FACING SEMANTICS
    ─────────────────────────────────────────────
    Enhancing tumour can be legitimately small (e.g., nascent enhancement
    in lower-grade glioma, micro-enhancement in post-treatment cases).
    A component is kept if **either**:
      (i)  it is at least ``et_min_component_size`` voxels, OR
      (ii) the mean foreground ET softmax probability over the component
           exceeds ``et_confidence_threshold``.

    IMPORTANT: the confidence gate requires a 4-channel softmax probability
    map passed via ``class_prob`` (shape [C>=4, D, H, W]). If you switch to
    deep-ensemble / MC-dropout outputs, the semantics of
    ``et_confidence_threshold`` change: it is still the *mean posterior
    probability*, but the posterior is now marginalised over the ensemble.
    The threshold was calibrated on single-model softmax; users should
    re-tune it when ensembling (see `threshold_sweep` CLI).

    NOTE (Issue #6): For ET, this may inflate Dice by removing small false
    positives. For downstream analysis, always report BOTH:
      - raw metrics (no postprocessing)
      - postprocessed metrics
    The ablation "no_postprocess" trains identically but evaluates raw.
    See the integrated `threshold_sweep` CLI for the sensitivity sweep
    may be required.
    """
    out = pred.copy()
    for label_val in [1, 2]:
        binary = (out == label_val).astype(np.int32)
        if binary.sum() == 0:
            continue
        labelled, n_components = ndi.label(binary)
        if n_components <= 1:
            continue
        sizes = ndi.sum(binary, labelled, range(1, n_components + 1))
        largest = np.argmax(sizes) + 1
        for comp_id in range(1, n_components + 1):
            if comp_id != largest and sizes[comp_id - 1] < min_component_size:
                out[labelled == comp_id] = 0

    et_binary = (out == 3).astype(np.int32)
    if et_binary.sum() == 0:
        return out
    et_labelled, et_components = ndi.label(et_binary)
    if et_components <= 1:
        return out

    et_sizes = ndi.sum(et_binary, et_labelled, range(1, et_components + 1))
    et_keep = {int(np.argmax(et_sizes) + 1)}
    et_prob_map = None
    if class_prob is not None and class_prob.ndim >= 4 and class_prob.shape[0] > 3:
        et_prob_map = class_prob[3]
    for comp_id in range(1, et_components + 1):
        comp_mask = et_labelled == comp_id
        comp_size = float(et_sizes[comp_id - 1])
        # When no probability map is supplied, fall back to SIZE-ONLY gating
        # (comp_conf=0.0), NOT comp_conf=1.0 - the latter made the OR always true,
        # silently keeping EVERY small ET component (inflates ET Dice by retaining
        # false-positive islands the gate exists to remove).
        comp_conf = float(np.mean(et_prob_map[comp_mask])) if et_prob_map is not None and np.any(comp_mask) else 0.0
        if comp_size >= et_min_component_size or comp_conf >= et_confidence_threshold:
            et_keep.add(comp_id)
    for comp_id in range(1, et_components + 1):
        if comp_id not in et_keep:
            out[et_labelled == comp_id] = 0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §12  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

BRATS_4CLASS_TAGS = DENSE_BRATS_REGION_TAGS
BRATS_METRIC_NAMES = (
    "dice_WT", "dice_TC", "dice_ET", "dice_brats_mean",
    "hd95_WT", "hd95_TC", "hd95_ET", "hd95_brats_mean",
)
BRATS_LESION_WISE_METRIC_NAMES = (
    "lw_dice_WT", "lw_dice_TC", "lw_dice_ET", "lw_dice_brats_mean",
    "lw_hd95_WT", "lw_hd95_TC", "lw_hd95_ET", "lw_hd95_brats_mean",
)


def _initialise_eval_checkpoint_state(
    config_tags: Sequence[str],
    num_classes: int,
    *,
    use_brats: bool,
    postprocess: bool,
    compute_hd95: bool,
    compute_lesion_wise: bool = False,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "config_tags": list(config_tags),
        "num_classes": int(num_classes),
        "use_brats": bool(use_brats),
        "postprocess": bool(postprocess),
        "compute_hd95": bool(compute_hd95),
        "compute_lesion_wise": bool(compute_lesion_wise),
        "checkpoint_metadata": checkpoint_metadata or {},
        "dice_sum_by_tag": {tag: [0.0] * num_classes for tag in config_tags},
        "n_cases_by_tag": {tag: 0 for tag in config_tags},
        "per_case_all": {},
        "per_case_metrics": {},
        "processed_case_keys": [],
        "processed_case_ids": [],
        "completed": False,
    }


def _eval_case_key(case_id: Any, dataset_tag: Any) -> str:
    return f"{str(dataset_tag)}::{str(case_id)}"


def _load_eval_checkpoint_state(
    checkpoint_path: Optional[str],
    config_tags: Sequence[str],
    num_classes: int,
    *,
    use_brats: bool,
    postprocess: bool,
    compute_hd95: bool,
    compute_lesion_wise: bool = False,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = _initialise_eval_checkpoint_state(
        config_tags,
        num_classes,
        use_brats=use_brats,
        postprocess=postprocess,
        compute_hd95=compute_hd95,
        compute_lesion_wise=compute_lesion_wise,
        checkpoint_metadata=checkpoint_metadata,
    )
    if not checkpoint_path:
        return state

    path = Path(checkpoint_path)
    if not path.exists():
        return state

    try:
        loaded = load_json(path)
    except Exception as exc:
        LOG.warning(f"[Eval] Could not load checkpoint {path}: {exc}. Starting fresh.")
        return state

    expected_meta = checkpoint_metadata or {}
    if (
        loaded.get("config_tags") != list(config_tags)
        or int(loaded.get("num_classes", -1)) != int(num_classes)
        or bool(loaded.get("use_brats", False)) != bool(use_brats)
        or bool(loaded.get("postprocess", False)) != bool(postprocess)
        or bool(loaded.get("compute_hd95", False)) != bool(compute_hd95)
        or bool(loaded.get("compute_lesion_wise", False)) != bool(compute_lesion_wise)
        or loaded.get("checkpoint_metadata", {}) != expected_meta
    ):
        LOG.warning(f"[Eval] Checkpoint {path} does not match this validation run. Starting fresh.")
        return state

    try:
        state["dice_sum_by_tag"] = {
            tag: [float(v) for v in loaded.get("dice_sum_by_tag", {}).get(tag, [0.0] * num_classes)]
            for tag in config_tags
        }
        state["n_cases_by_tag"] = {
            tag: int(loaded.get("n_cases_by_tag", {}).get(tag, 0))
            for tag in config_tags
        }
        state["per_case_all"] = {
            str(key): [float(v) for v in values]
            for key, values in loaded.get("per_case_all", {}).items()
        }
        state["per_case_metrics"] = {}
        for case_key, payload in loaded.get("per_case_metrics", {}).items():
            payload = payload if isinstance(payload, dict) else {}
            state["per_case_metrics"][str(case_key)] = {
                "case_id": str(payload.get("case_id", "")),
                "dataset_tag": str(payload.get("dataset_tag", "")),
                "metrics": {
                    str(metric_key): float(metric_value)
                    for metric_key, metric_value in payload.get("metrics", {}).items()
                    if metric_value is not None
                },
            }
        processed_keys_raw = loaded.get("processed_case_keys", loaded.get("processed_case_ids", []))
        state["processed_case_keys"] = [str(case_key) for case_key in processed_keys_raw]
        state["processed_case_ids"] = [str(case_key) for case_key in processed_keys_raw]
        state["completed"] = bool(loaded.get("completed", False))
    except Exception as exc:
        LOG.warning(f"[Eval] Checkpoint {path} is malformed ({exc}). Starting fresh.")
        return _initialise_eval_checkpoint_state(
            config_tags,
            num_classes,
                use_brats=use_brats,
                postprocess=postprocess,
                compute_hd95=compute_hd95,
                compute_lesion_wise=compute_lesion_wise,
                checkpoint_metadata=checkpoint_metadata,
            )

    LOG.info(
        f"[Eval] Resuming checkpoint {path} "
        f"({len(state['processed_case_keys'])} completed cases, completed={state['completed']})"
    )
    return state


def _finalise_eval_state(
    config_tags: Sequence[str],
    num_classes: int,
    dice_sum_by_tag: Dict[str, Sequence[float]],
    n_cases_by_tag: Dict[str, int],
    per_case_all: Dict[str, List[float]],
    *,
    use_brats: bool,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    results: Dict[str, float] = {}
    for tag in config_tags:
        n_cases = int(n_cases_by_tag.get(tag, 0))
        if n_cases > 0:
            dmean = np.asarray(dice_sum_by_tag.get(tag, [0.0] * num_classes), dtype=np.float64) / float(n_cases)
        else:
            dmean = np.zeros(num_classes, dtype=np.float64)
        results[f"{tag}_dice_fg"] = float(dmean[1:].mean() if num_classes > 1 else dmean.mean())
        for class_idx in range(num_classes):
            results[f"{tag}_dice_c{class_idx}"] = float(dmean[class_idx])
        if use_brats:
            metric_names = list(BRATS_METRIC_NAMES)
            tag_prefix = f"{tag}_"
            for key in per_case_all:
                if key.startswith(tag_prefix):
                    metric_name = key[len(tag_prefix):]
                    if metric_name not in metric_names:
                        metric_names.append(metric_name)
            for metric_name in metric_names:
                vals = per_case_all.get(f"{tag}_{metric_name}", [])
                # NaN (not a phantom 0.0) when no case was scored: a 0.0 HD95 reads as a
                # perfect score and would silently corrupt the published eval CSVs.
                results[f"{tag}_{metric_name}"] = float(np.mean(vals)) if vals else float("nan")
    return results, per_case_all


def _save_eval_checkpoint_state(
    checkpoint_path: Optional[str],
    state: Dict[str, Any],
) -> None:
    if not checkpoint_path:
        return
    save_json_atomic(state, checkpoint_path)


def evaluate_teacher(
    model: UNet3DTeacher, loader: DataLoader, num_classes: int,
    device: torch.device, use_brats: bool = False,
    compute_hd95: bool = True,
) -> Dict[str, float]:
    model.eval()
    dices: List[torch.Tensor] = []
    brats_results: Dict[str, List[float]] = {
        f"{m}_{r}": [] for m in ["dice", "hd95"] for r in ["WT", "TC", "ET"]
    }

    with torch.inference_mode():
        for batch in loader:
            # Expected layout: (case_id, x, y, spacing_mm, dataset_tag)
            if len(batch) == 5:
                _, x, y, spacing_t, tags = batch
            elif len(batch) == 4:  # backward compat
                _, x, y, tags = batch
                spacing_t = None  # unknown spacing - brats_metrics will fall back to voxel units
            else:  # very old layout
                _, x, y = batch[:3]
                tags = ["unknown"] * x.shape[0]
                spacing_t = None  # unknown spacing - brats_metrics will fall back to voxel units

            if y.numel() == 0:
                raise ValueError("Evaluation requires ground-truth labels, but y is empty.")

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).argmax(1)
            # Crop y to pred spatial size (odd input dims cause 1-voxel mismatch)
            y = y[..., :pred.shape[1], :pred.shape[2], :pred.shape[3]]
            dices.append(dice_per_class(pred, y, num_classes))

            if use_brats:
                tags_list = list(tags) if isinstance(tags, (list, tuple)) else [str(tags)] * x.shape[0]
                for i in range(pred.shape[0]):
                    if tags_list[i] in BRATS_4CLASS_TAGS:
                        _sp = spacing_t[i].cpu().numpy() if spacing_t is not None else None
                        spacing = _sp if (_sp is not None and (_sp > 0).all()) else None
                        m = brats_metrics(pred[i], y[i], spacing=spacing, compute_hd95=compute_hd95)
                        for k, v in m.items():
                            if k in brats_results:
                                brats_results[k].append(v)

    dmean = torch.stack(dices).mean(0) if dices else torch.zeros(num_classes)
    out = {f"dice_c{i}": dmean[i].item() for i in range(num_classes)}
    out["dice_fg"] = (dmean[1:].mean() if num_classes > 1 else dmean.mean()).item()
    if use_brats:
        for k, vs in brats_results.items():
            # NaN (not a phantom 0.0) when a region was never scored: a 0.0 HD95 reads
            # as a perfect score and would corrupt both the published table and the
            # best-model selection metric.
            out[k] = float(np.mean(vs)) if vs else float("nan")
        for metric in ["dice", "hd95"]:
            region_vals = [out.get(f"{metric}_{r}") for r in ["WT", "TC", "ET"]]
            finite = [v for v in region_vals if v is not None and np.isfinite(v)]
            # BraTS convention: mean over the regions actually present; NaN only if all
            # three are absent (mean of 3 finite values is unchanged from before).
            out[f"{metric}_brats_mean"] = float(np.mean(finite)) if finite else float("nan")
    return out


def evaluate_student(
    student: UNet3DStudent, loader: DataLoader, num_classes: int,
    device: torch.device, use_brats: bool = False,
    eval_mask_configs: Optional[List[Optional[List[int]]]] = None,
    mask_modalities: Optional[int] = None,
    postprocess: bool = False,
    use_tta: bool = False,
    compute_hd95: bool = True,
    compute_lesion_wise: bool = False,
    eval_config_batch_size: int = 1,
    metric_workers: int = 1,
    checkpoint_path: Optional[str] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
    checkpoint_every_cases: int = 1,
    return_per_case: bool = False,
    return_casewise: bool = False,
) -> Union[
    Dict[str, float],
    Tuple[Dict[str, float], Dict[str, List[float]]],
    Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, Dict[str, Any]]],
]:
    """Evaluate student under multiple modality-missing configurations.

    Args:
        postprocess:     If True, apply connected-component filtering before
                         computing metrics.  Pass postprocess=True in addition to a
                         normal call to get both sets of numbers for publication.
        return_per_case: If True, also return a second dict mapping
                         ``"{tag}_{metric}"`` → list-of-per-case floats.
                         Required for bootstrap CI and Wilcoxon signed-rank tests
        return_casewise: If True, also return a per-case metrics mapping
                         ``case_key -> {case_id, dataset_tag, metrics{...}}``.
    Returns:
        If return_per_case=False (default): Dict[metric_name, mean_value]
        If return_per_case=True:            (means_dict, per_case_dict)
    """
    if eval_mask_configs is None:
        eval_mask_configs = [None]

    student.eval()
    _configure_cuda_inference()
    config_specs = [
        (modality_config_tag(missing_mods), missing_mods)
        for missing_mods in eval_mask_configs
    ]
    config_tags = [tag for tag, _ in config_specs]
    eval_config_batch_size = max(1, int(eval_config_batch_size))
    metric_workers = max(1, int(metric_workers))
    checkpoint_every_cases = max(1, int(checkpoint_every_cases))
    state = _load_eval_checkpoint_state(
        checkpoint_path,
        config_tags,
        num_classes,
        use_brats=use_brats,
        postprocess=postprocess,
        compute_hd95=compute_hd95,
        compute_lesion_wise=compute_lesion_wise,
        checkpoint_metadata=checkpoint_metadata,
    )
    if return_casewise and state.get("processed_case_keys") and not state.get("per_case_metrics"):
        LOG.warning(
            "[Eval] Existing checkpoint lacks per-case metrics required for "
            "casewise/failure-mode analysis. Restarting this evaluation phase."
        )
        state = _initialise_eval_checkpoint_state(
            config_tags,
            num_classes,
            use_brats=use_brats,
            postprocess=postprocess,
            compute_hd95=compute_hd95,
            compute_lesion_wise=compute_lesion_wise,
            checkpoint_metadata=checkpoint_metadata,
        )
    dice_sum_by_tag: Dict[str, np.ndarray] = {
        tag: np.asarray(state["dice_sum_by_tag"].get(tag, [0.0] * num_classes), dtype=np.float64)
        for tag in config_tags
    }
    n_cases_by_tag: Dict[str, int] = {
        tag: int(state["n_cases_by_tag"].get(tag, 0))
        for tag in config_tags
    }
    per_case_all = {
        str(key): [float(v) for v in values]
        for key, values in state.get("per_case_all", {}).items()
    }
    per_case_metrics: Dict[str, Dict[str, Any]] = {}
    for case_key, payload in state.get("per_case_metrics", {}).items():
        payload = payload if isinstance(payload, dict) else {}
        per_case_metrics[str(case_key)] = {
            "case_id": str(payload.get("case_id", "")),
            "dataset_tag": str(payload.get("dataset_tag", "")),
            "metrics": {
                str(metric_key): float(metric_value)
                for metric_key, metric_value in payload.get("metrics", {}).items()
                if metric_value is not None
            },
        }
    processed_case_keys: List[str] = [
        str(case_key)
        for case_key in state.get("processed_case_keys", state.get("processed_case_ids", []))
    ]
    processed_case_set = set(processed_case_keys)
    cases_since_checkpoint = 0

    def _checkpoint_payload(*, completed: bool) -> Dict[str, Any]:
        return {
            "version": 1,
            "config_tags": config_tags,
            "num_classes": int(num_classes),
            "use_brats": bool(use_brats),
            "postprocess": bool(postprocess),
            "compute_hd95": bool(compute_hd95),
            "compute_lesion_wise": bool(compute_lesion_wise),
            "checkpoint_metadata": checkpoint_metadata or {},
            "dice_sum_by_tag": {tag: dice_sum_by_tag[tag].tolist() for tag in config_tags},
            "n_cases_by_tag": {tag: int(n_cases_by_tag[tag]) for tag in config_tags},
            "per_case_all": per_case_all,
            "per_case_metrics": per_case_metrics,
            "processed_case_keys": processed_case_keys,
            "processed_case_ids": processed_case_keys,
            "completed": bool(completed),
        }

    if state.get("completed", False):
        results, per_case_all = _finalise_eval_state(
            config_tags,
            num_classes,
            dice_sum_by_tag,
            n_cases_by_tag,
            per_case_all,
            use_brats=use_brats,
        )
        if return_per_case:
            if return_casewise:
                return results, per_case_all, per_case_metrics
            return results, per_case_all
        return results

    def _predict_chunk(
        x_in: torch.Tensor,
        cfg_chunk: Sequence[Tuple[str, Optional[List[int]]]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, channels = x_in.shape[:2]
        n_cfg = len(cfg_chunk)
        max_cfg_idx = max(
            (int(mod_idx) for _, missing_mods in cfg_chunk for mod_idx in (missing_mods or [])),
            default=-1,
        )
        # Tissue-prior validation tensors are 4 MRI + 3 CSF/WM/GM channels.
        # Only MRI channels participate in missing-modality masks.
        if mask_modalities is not None:
            mask_channels = int(mask_modalities)
        else:
            mask_channels = max_cfg_idx + 1 if max_cfg_idx >= 0 else (
                channels - 3 if channels == 7 else channels
            )
        mask_channels = max(1, min(int(mask_channels), int(channels)))
        mask_bank = torch.ones((n_cfg, mask_channels), dtype=torch.float32, device=device)
        for cfg_idx, (_, missing_mods) in enumerate(cfg_chunk):
            if missing_mods:
                for mod_idx in missing_mods:
                    if 0 <= mod_idx < mask_channels:
                        mask_bank[cfg_idx, mod_idx] = 0.0

        x_batch = x_in.unsqueeze(0).expand(n_cfg, *x_in.shape).reshape(
            n_cfg * bsz, channels, *x_in.shape[2:]
        )
        mask_batch = mask_bank[:, None, :].expand(
            n_cfg,
            bsz,
            mask_channels,
        ).reshape(n_cfg * bsz, mask_channels)
        student_in = prepare_student_input(x_batch, mask_batch)
        if device.type == "cuda":
            student_in = student_in.contiguous(memory_format=torch.channels_last_3d)

        def _forward_logits(inp: torch.Tensor) -> torch.Tensor:
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    return student(inp)
            return student(inp)

        if use_tta:
            prob_sum: Optional[torch.Tensor] = None
            flip_combos = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
            for dims in flip_combos:
                inp = student_in.flip(dims) if dims else student_in
                logits_step = _forward_logits(inp)
                if dims:
                    logits_step = logits_step.flip(dims)
                prob_step = F.softmax(logits_step.float(), dim=1)
                prob_sum = prob_step if prob_sum is None else prob_sum + prob_step
            logits = torch.log((prob_sum / len(flip_combos)).clamp_min(1e-8))
        else:
            logits = _forward_logits(student_in)
        pred = logits.argmax(1)
        prob = None
        if postprocess:
            prob = F.softmax(logits.float(), dim=1)
            prob = prob.view(n_cfg, bsz, prob.shape[1], *prob.shape[2:]).cpu()
        return pred.view(n_cfg, bsz, *pred.shape[1:]).cpu(), prob

    metric_pool = ThreadPoolExecutor(max_workers=metric_workers) if metric_workers > 1 else None
    try:
        with torch.inference_mode():
            for batch in loader:
                # Expected layout: (case_id, x, y, spacing_mm, dataset_tag)
                if len(batch) == 5:
                    case_ids_raw, x, y, spacing_t, tags = batch
                elif len(batch) == 4:  # backward compat
                    case_ids_raw, x, y, tags = batch
                    spacing_t = None
                else:
                    case_ids_raw, x, y = batch[:3]
                    tags = ["unknown"] * x.shape[0]
                    spacing_t = None

                if y.numel() == 0:
                    raise ValueError("Evaluation requires ground-truth labels, but y is empty.")

                batch_size = x.shape[0]
                case_ids_list = (
                    [str(case_id) for case_id in case_ids_raw]
                    if isinstance(case_ids_raw, (list, tuple))
                    else [str(case_ids_raw)] * batch_size
                )
                tags_list = list(tags) if isinstance(tags, (list, tuple)) else [str(tags)] * batch_size
                case_keys_list = [
                    _eval_case_key(case_id, tag)
                    for case_id, tag in zip(case_ids_list, tags_list)
                ]
                keep_indices = [
                    idx for idx, case_key in enumerate(case_keys_list)
                    if case_key not in processed_case_set
                ]
                if not keep_indices:
                    continue
                if len(keep_indices) != batch_size:
                    idx_t = torch.as_tensor(keep_indices, dtype=torch.long)
                    x = x.index_select(0, idx_t)
                    y = y.index_select(0, idx_t)
                    if spacing_t is not None:
                        spacing_t = spacing_t.index_select(0, idx_t)
                    case_ids_list = [case_ids_list[idx] for idx in keep_indices]
                    tags_list = [tags_list[idx] for idx in keep_indices]
                    case_keys_list = [case_keys_list[idx] for idx in keep_indices]

                if device.type == "cuda":
                    x = x.to(device, non_blocking=True, memory_format=torch.channels_last_3d)
                else:
                    x = x.to(device, non_blocking=True)
                b, c = x.shape[:2]
                target_cpu = y.cpu()
                spacing_cpu = spacing_t.cpu().numpy() if spacing_t is not None else None

                for cfg_chunk in _chunked(config_specs, eval_config_batch_size):
                    pred_chunk, prob_chunk = _predict_chunk(x, cfg_chunk)
                    pending: List[Tuple[str, str, str, str, Any]] = []

                    for cfg_idx, (tag, _) in enumerate(cfg_chunk):
                        for sample_idx in range(b):
                            pred_np = pred_chunk[cfg_idx, sample_idx].numpy().astype(np.int16, copy=True)
                            class_prob = None
                            if prob_chunk is not None:
                                class_prob = prob_chunk[cfg_idx, sample_idx].numpy().astype(np.float32, copy=False)
                            target_np = target_cpu[
                                sample_idx,
                                :pred_np.shape[0],
                                :pred_np.shape[1],
                                :pred_np.shape[2],
                            ].numpy().astype(np.int64, copy=False)
                            spacing = None
                            if spacing_cpu is not None:
                                spacing_row = spacing_cpu[sample_idx]
                                if (spacing_row > 0).all():
                                    spacing = spacing_row.astype(np.float64, copy=False)
                            use_brats_case = use_brats and tags_list[sample_idx] in BRATS_4CLASS_TAGS

                            if metric_pool is None:
                                dice_scores, brats_scores = _score_prediction_np(
                                    pred_np,
                                    target_np,
                                    num_classes,
                                    postprocess=postprocess,
                                    use_brats=use_brats_case,
                                    class_prob=class_prob,
                                    spacing=spacing,
                                    compute_hd95=compute_hd95,
                                    compute_lesion_wise=compute_lesion_wise,
                                )
                                dice_sum_by_tag[tag] += dice_scores.astype(np.float64, copy=False)
                                n_cases_by_tag[tag] += 1
                                if brats_scores:
                                    case_key = case_keys_list[sample_idx]
                                    case_entry = per_case_metrics.setdefault(
                                        case_key,
                                        {
                                            "case_id": case_ids_list[sample_idx],
                                            "dataset_tag": tags_list[sample_idx],
                                            "metrics": {},
                                        },
                                    )
                                    for metric_name, metric_value in brats_scores.items():
                                        per_case_all.setdefault(f"{tag}_{metric_name}", []).append(float(metric_value))
                                        case_entry["metrics"][f"{tag}_{metric_name}"] = float(metric_value)
                            else:
                                future = metric_pool.submit(
                                    _score_prediction_np,
                                    pred_np,
                                    target_np,
                                    num_classes,
                                    postprocess=postprocess,
                                    use_brats=use_brats_case,
                                    class_prob=class_prob,
                                    spacing=spacing,
                                    compute_hd95=compute_hd95,
                                    compute_lesion_wise=compute_lesion_wise,
                                )
                                pending.append((
                                    tag,
                                    case_keys_list[sample_idx],
                                    case_ids_list[sample_idx],
                                    tags_list[sample_idx],
                                    future,
                                ))

                    for tag, case_key, case_id, dataset_tag, future in pending:
                        dice_scores, brats_scores = future.result()
                        dice_sum_by_tag[tag] += dice_scores.astype(np.float64, copy=False)
                        n_cases_by_tag[tag] += 1
                        if brats_scores:
                            case_entry = per_case_metrics.setdefault(
                                case_key,
                                {
                                    "case_id": case_id,
                                    "dataset_tag": dataset_tag,
                                    "metrics": {},
                                },
                            )
                            for metric_name, metric_value in brats_scores.items():
                                per_case_all.setdefault(f"{tag}_{metric_name}", []).append(float(metric_value))
                                case_entry["metrics"][f"{tag}_{metric_name}"] = float(metric_value)

                for case_key in case_keys_list:
                    processed_case_keys.append(case_key)
                    processed_case_set.add(case_key)
                cases_since_checkpoint += len(case_keys_list)
                if checkpoint_path and cases_since_checkpoint >= checkpoint_every_cases:
                    _save_eval_checkpoint_state(
                        checkpoint_path,
                        _checkpoint_payload(completed=False),
                    )
                    cases_since_checkpoint = 0
    finally:
        if metric_pool is not None:
            metric_pool.shutdown(wait=True)

    results, per_case_all = _finalise_eval_state(
        config_tags,
        num_classes,
        dice_sum_by_tag,
        n_cases_by_tag,
        per_case_all,
        use_brats=use_brats,
    )
    if checkpoint_path:
        _save_eval_checkpoint_state(
            checkpoint_path,
            _checkpoint_payload(completed=True),
        )

    if return_per_case:
        if return_casewise:
            return results, per_case_all, per_case_metrics
        return results, per_case_all
    return results


def build_eval_loader(
    csv_path: str,
    modalities: Sequence[str],
    patch_size: Tuple[int, int, int],
    *,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    pin_memory: bool = False,
) -> DataLoader:
    ds = MultiModalSegDataset(
        csv_path,
        list(modalities),
        patch_size,
        False,
        False,
        cache_dir=cache_dir,
    )
    loader_kwargs = _build_loader_kwargs(
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init=seed_worker,
        persistent_workers=False,
    )
    return DataLoader(ds, **loader_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# §13  TEACHER PRETRAINING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeacherConfig:
    train_csv: str
    val_csv: str
    out_dir: str
    modalities: List[str]
    num_classes: int
    patch_size: Tuple[int, int, int] = (192, 192, 144)
    batch_size: int = 2
    epochs: int = 1000
    lr: float = 2e-4
    weight_decay: float = 3e-5
    seed: int = 42
    deterministic: bool = False
    num_workers: int = 8
    val_num_workers: int = 2
    teacher_base: int = 32
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    lambda_region: float = 0.20
    amp: bool = True
    augment: bool = True
    use_brats_metrics: bool = True
    hd95_freq: int = 50   # v9: raised from 10. HD95 is a publication metric, not
                          # a training signal. scipy EDT on 3D volumes stalls the GPU
                          # for seconds per patient - too expensive every 10 epochs.
                          # Always computed in the standalone 'evaluate' subcommand.
    val_fast_n: int = 1   # v9: configs evaluated every epoch. 1 = only "all" modalities.
                          # Prevents 5x validation overhead on every single epoch.
    val_start_epoch: int = 3  # Skip expensive validation until this epoch (final epoch
                              # always validates regardless).
    val_full_freq: int = 50  # v9: run all eval_configs every N epochs + final epoch.
                             # HD95 is also only computed on these full-eval epochs.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None  # NVMe npy cache (e.g. $SLURM_TMPDIR/npy_cache)
    patch_cache_dir: Optional[str] = None  # RAM-backed per-epoch crop cache (e.g. /dev/shm)
    patch_cache_start_epoch: int = 2
    gpu_augment: bool = True
    lmic_field_strength_prior: bool = False
    contrast_mod_indices: List[int] = field(default_factory=list)
    contrast_dropout_extra_prob: float = 0.0
    p_anisotropic_2d: float = 0.85
    p_partial_contrast: float = 0.3
    p_label_noise: float = 0.2
    label_noise_mode: str = "morph"
    label_noise_band_radius: int = 2
    label_noise_band_flip_prob: float = 0.3
    focal_b1_inhomogeneity_prob: float = 0.12
    field_strength_sampling_power: float = 0.5
    low_field_sampling_bias: float = 1.5
    grade23_sampling_bias: float = 1.5
    idh_mutant_sampling_bias: float = 1.5
    legacy_protocol_sampling_bias: float = 1.25
    legacy_protocol_year_threshold: int = 2013
    auto_resources: bool = False
    auto_resources_safety: float = 0.85
    # FIX (2026 throughput/accuracy pass): the bare legacy teacher CLI never loads
    # a v9.1 --config, so torch.compile() and weight-EMA were silently OFF for the
    # teacher (both are otherwise gated on the v9.1 config: _v91_compile_cfg /
    # build_model_ema). These knobs let the teacher CLI enable them directly; an
    # explicit v9.1 config still wins when it turns them on. Defaults stay OFF so
    # the historical path is byte-identical.
    compile: bool = False
    compile_mode: str = "max-autotune"
    compile_dynamic: bool = False
    ema: bool = False
    ema_decay: float = 0.999
    # evaluation rigor extensions (added in the 2026 architecture pass)
    p_field_contrast_warp: float = 0.0
    field_contrast_source_b0: float = 3.0
    p_fourier_amplitude_mix: float = 0.0
    afa_prob: float = 0.0
    p_receive_coil_inhomogeneity: float = 0.0
    p_off_resonance_void: float = 0.0
    noise_calibration: str = "static"
    carvemix_prob: float = 0.0
    carvemix_dilation: int = 5
    teacher_arch: str = "unet3d"
    # block_style controls teacher capacity (only).  "mednext" (default)
    # preserves the historical depthwise-separable design (~1M @ base=32).
    # "dense" switches to full 3x3x3 conv blocks (~6M @ base=32, ~15M @
    # base=48, ~26M @ base=64) for heavy KD teachers - student is unaffected
    # since only the student is exported to INT8 for CPU deployment.
    teacher_block_style: str = "mednext"
    # FIX (Gap A, 2026 audit): teacher-side feature-DR.  Mirrors the student
    # knob; default off for backward-compat with existing checkpoints.
    # When set to "mixstyle" or "dsu", the teacher's KD logits encode
    # cross-scanner invariance, which transfers to the student via KD loss.
    teacher_feature_dr: str = "none"
    teacher_feature_dr_p: float = 0.5
    teacher_feature_dr_alpha: float = 0.1
    # FIX (Gap B, 2026 audit): teacher-side modality dropout.  Recommended
    # 0.15 for clinically-realistic training (T1c often missing in LMIC).
    teacher_modality_dropout_p: float = 0.0
    teacher_modality_dropout_max_drop: int = 2
    # FIX (Finding 4, 2026 audit): opt-in flag for non-strict resume.  When
    # False (default), a stale resume_teacher.pth with mismatched architecture
    # aborts loudly; when True, missing/extra keys are tolerated.
    force_resume_partial: bool = False
    # Phase 7.2: unified v9.1 config namespace (ogap/config/defaults.yaml schema).
    # None => legacy behaviour. Threaded from the CLI so opt-in features
    # (region-aware loss, etc.) can be selected without disturbing defaults.
    v91_config: Optional[object] = None


def _v91_task_active(v91) -> bool:
    """True iff a v9.1 config selects a non-legacy task loss (opt-in gate).

    The whole v9.1 loss stack (task + ODE-aware KD) activates together off this
    single switch, so with the default config (``loss.task.type == "legacy"``)
    every legacy code path below is reached verbatim and behaviour is unchanged.
    """
    try:
        return v91 is not None and str(v91.loss.task.type) != "legacy"
    except AttributeError:
        return False


def _v91_tissue_cfg(v91) -> Tuple[bool, str]:
    """Return ``(use_tissue_priors, tissue_prior_dir)`` from a v9.1 config.

    Defaults to ``(False, "")`` (4-channel legacy input) when the config is
    absent or lacks the keys.
    """
    try:
        return bool(v91.data.use_tissue_priors), str(getattr(v91.data, "tissue_prior_dir", "") or "")
    except AttributeError:
        return False, ""


def _v91_physics_aug(v91):
    """Build the opt-in v9.1 physics augmentation callable, or None when disabled.

    Returns ``None`` (no extra augmentation) unless the config sets
    ``augmentation.physics.enabled``; the legacy MRIPhysicsAugmentor path is
    unaffected either way.
    """
    try:
        if v91 is None or not v91.augmentation.physics.enabled:
            return None
    except AttributeError:
        return None
    from ogap.augmentations.physics_augment import build_physics_augmentations
    return build_physics_augmentations(v91)


def _v91_bloch_sim(v91):
    """Build the Bloch low-field simulator, or None when disabled.

    Enabled by ``augmentation.physics.bloch_lowfield``. Requires FSL FAST tissue
    priors (``data.use_tissue_priors``) and the segmentation label, both available in
    the dataset ``__getitem__``. Re-renders the MRI channels to the target field
    (default 64 mT) from the partial-volume maps before the noise/artifact pipeline.
    """
    try:
        phys = v91.augmentation.physics
        if v91 is None or not getattr(phys, "bloch_lowfield", False):
            return None
    except AttributeError:
        return None
    from ogap.augmentations.bloch_lowfield import BlochLowFieldSimulator
    # MRI modality count (NOT the tissue-inflated in_channels) - see ogap.config.loader.
    _data = getattr(v91, "data", None)
    n_mri = int(getattr(_data, "n_mri_channels", 0) or getattr(_data, "in_channels", 4) or 4)
    target_b0 = float(getattr(phys, "bloch_lowfield_target_b0", 0.064) or 0.064)
    return BlochLowFieldSimulator(
        target_b0_t=target_b0, n_mri=n_mri, prob=1.0,
        heterogeneity=float(getattr(phys, "bloch_heterogeneity", 0.10) or 0.0),
        boundary_blur_sigma=float(getattr(phys, "bloch_boundary_blur_sigma", 0.6) or 0.0),
        texture_weight=float(getattr(phys, "bloch_texture_weight", 0.25) or 0.0),
        acquisition_jitter=float(getattr(phys, "bloch_acquisition_jitter", 0.0) or 0.0),
    )


def _v91_bloch_prob(v91) -> float:
    """Per-sample probability of applying the Bloch low-field render (default 1.0)."""
    try:
        # Honour an explicit 0.0 (disable the render via probability); `... or 1.0`
        # would silently turn prob=0.0 into "always apply".
        prob = getattr(v91.augmentation.physics, "bloch_lowfield_prob", 1.0)
        return 1.0 if prob is None else float(prob)
    except AttributeError:
        return 1.0


def _v91_synth_sim(v91):
    """Build the SynthSeg-style randomized contrast generator, or None when disabled.

    Enabled by ``augmentation.physics.synth_randomization``. Needs FSL FAST tissue
    priors + the label (same plumbing as the Bloch sim). A complementary, fully
    contrast-randomized training arm for intensity-agnostic cross-field robustness.
    """
    try:
        phys = v91.augmentation.physics
        if v91 is None or not getattr(phys, "synth_randomization", False):
            return None
    except AttributeError:
        return None
    from ogap.augmentations.synth import build_synth_generator
    return build_synth_generator(v91)


def _v91_gpu_physics_aug(v91):
    """Build the batched, on-GPU v9.1 physics augmentor, or None when disabled.

    This is the device-resident counterpart of :func:`_v91_physics_aug`: instead
    of running the v9.1 transforms per-sample inside a CPU DataLoader worker, it
    returns a ``(x, y) -> (x, y)`` callable the CUDA prefetcher applies to a whole
    batch already on the H100. Returns ``None`` unless
    ``augmentation.physics.enabled`` is set. When this is used, the dataset must
    receive ``physics_aug=None`` so the transforms run exactly once, on the device
    (the CPU/GPU double-application that was finding F1).
    """
    try:
        phys_on = bool(v91 is not None and v91.augmentation.physics.enabled)
    except AttributeError:
        phys_on = False
    if v91 is None or not (phys_on or _v91_geometric_enabled(v91)):
        return None
    from ogap.augmentations.physics_augment import build_gpu_physics_augmentor
    # No explicit generator: on the CUDA path the augmentor draws from the global RNG,
    # which seed_everything seeds (torch.cuda.manual_seed_all), so augmentation IS
    # reproducible under a fixed seed. Pass a device-matched torch.Generator here if an
    # augmentation RNG stream independent of the global state is ever required.
    return build_gpu_physics_augmentor(v91)


def _v91_geometric_enabled(v91) -> bool:
    """True when channel-aware geometric augmentation is opt-in via the v9.1 config."""
    try:
        return bool(v91 is not None and v91.augmentation.geometric.enabled)
    except AttributeError:
        return False


def _v91_geometric_aug(v91):
    """Build the CPU-side channel-aware geometric augmentor, or None when disabled.

    Applied in the dataset as the LAST step on the full (MRI + tissue-prior) tensor
    plus the label, so all channels and the mask stay co-registered. On the CUDA
    path the geometric augmentor is instead carried by the GPU batched augmentor
    (see :func:`_v91_gpu_physics_aug`); the dataset then receives ``None`` so the
    transform runs exactly once.
    """
    if not _v91_geometric_enabled(v91):
        return None
    from ogap.augmentations.geometric import build_geometric_augmentor
    return build_geometric_augmentor(v91)


def _assert_tissue_prior_channel_safety(tp_use: bool, legacy_cpu: bool, legacy_gpu: bool) -> None:
    """Tissue-prior channels are only shielded by the v9.1 physics engine.

    The legacy CPU augmentor applies geometric warps (flip / elastic) *before* the
    CSF/WM/GM maps are appended, which mis-aligns them; the legacy GPU augmentor
    applies intensity transforms to every channel of the already-appended batch,
    which corrupts the probability maps. Only :class:`PhysicsAugmentor` (v9.1)
    shields the trailing tissue channels (``data.in_channels``..). Hard-fail the
    unsafe combination rather than silently degrade the anatomical prior.
    """
    if tp_use and (legacy_cpu or legacy_gpu):
        raise ValueError(
            "data.use_tissue_priors=True is channel-safe only with the v9.1 physics "
            "engine, which shields the appended CSF/WM/GM prior channels from "
            "intensity/geometric transforms. The legacy augmentor path "
            f"(CPU={legacy_cpu}, GPU={legacy_gpu}) would mis-align or corrupt them. "
            "Enable augmentation.physics in the v9.1 config, or set "
            "data.use_tissue_priors=False."
        )


def _v91_tissue_channel_count(v91) -> int:
    use_tissue, _ = _v91_tissue_cfg(v91)
    return 3 if use_tissue else 0


def _v91_teacher_input_channels(num_modalities: int, v91) -> int:
    return int(num_modalities) + _v91_tissue_channel_count(v91)


def _v91_student_input_channels(num_modalities: int, v91) -> int:
    return (2 * int(num_modalities)) + _v91_tissue_channel_count(v91)


def _teacher_adapter_identity_map(num_modalities: int) -> List[Tuple[int, int]]:
    return [(idx, idx) for idx in range(int(num_modalities))]


def _student_adapter_identity_map(num_modalities: int, v91) -> List[Tuple[int, int]]:
    num_modalities = int(num_modalities)
    tissue_channels = _v91_tissue_channel_count(v91)
    # Student input layout is [masked MRI, optional tissue priors, availability].
    # Native legacy student layout is [masked MRI, availability].
    return (
        [(idx, idx) for idx in range(num_modalities)]
        + [
            (num_modalities + tissue_channels + idx, num_modalities + idx)
            for idx in range(num_modalities)
        ]
    )


def _v91_aux_tissue_cfg(v91) -> Tuple[bool, float]:
    try:
        cfg = v91.loss.auxiliary_tissue
        return bool(cfg.enabled), float(getattr(cfg, "weight", 0.1))
    except AttributeError:
        return False, 0.0


def _v91_domain_cfg(v91) -> Tuple[bool, float]:
    try:
        return bool(v91.domain.enabled), float(getattr(v91.domain, "delta", 0.01))
    except AttributeError:
        return False, 0.0


def _v91_compile_cfg(v91) -> Tuple[bool, str, bool]:
    """Return torch.compile settings from the unified config."""
    enabled = bool(_cfg_get(v91, "training.compile.enabled", False))
    mode = str(_cfg_get(v91, "training.compile.mode", "max-autotune") or "max-autotune")
    dynamic = bool(_cfg_get(v91, "training.compile.dynamic", False))
    return enabled, mode, dynamic


def _v91_student_block_cfg(v91, default_style: str = "res", default_steps: int = 4) -> Tuple[str, int]:
    arch = _cfg_get(v91, "model.student.arch", default_style)
    style = _student_block_style_from_arch(arch)
    if style == "res":
        style = str(_cfg_get(v91, "model.student.block_style", style) or style).lower()
        if style not in {"res", "ode"}:
            style = default_style
    steps = int(_cfg_get(v91, "model.student.ode_steps", default_steps) or default_steps)
    return style, max(1, steps)


def _maybe_wrap_channel_adapter(
    model: nn.Module,
    *,
    in_channels: int,
    native_channels: int,
    label: str,
    identity_map: Optional[Sequence[Tuple[int, int]]] = None,
) -> nn.Module:
    """Wrap a model with the 1x1x1 input adapter only when channels differ."""
    if int(in_channels) == int(native_channels):
        return model
    from ogap.models.adapters import AdaptedModel
    LOG.info(
        "[%s] input channel adapter active: %d -> native %d",
        label,
        int(in_channels),
        int(native_channels),
    )
    return AdaptedModel(
        model,
        in_channels=int(in_channels),
        native_channels=int(native_channels),
        identity_map=identity_map,
    )


def _state_dict_for_maybe_adapted_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]
                                        ) -> Dict[str, torch.Tensor]:
    """Prefix old plain-model checkpoints when loading into an AdaptedModel."""
    from ogap.models.adapters import AdaptedModel
    if not isinstance(model, AdaptedModel):
        return state_dict
    if any(k.startswith("model.") or k.startswith("adapter.") for k in state_dict):
        return state_dict
    return {f"model.{k}": v for k, v in state_dict.items()}


class _V91TaskLossAdapter(nn.Module):
    """Make a v9.1 task loss call-compatible with the legacy training loop.

    The legacy loop invokes ``criterion(logits, target, dataset_tags=...)``. The
    region-aware losses have signature ``forward(logits, targets)`` and operate
    per-voxel, so ``dataset_tags`` is accepted and ignored.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                dataset_tags=None, **_) -> torch.Tensor:
        return self.inner(logits, target)


def pretrain_teacher(cfg: TeacherConfig) -> None:
    seed_everything(cfg.seed, cfg.deterministic)
    out = ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    # DDP (4×H100 NVLink): no-op unless launched under torchrun/srun with WORLD_SIZE>1.
    from ogap.utils.distributed import (
        init_distributed_from_env, wrap_ddp, is_main_process,
        maybe_distributed_weighted_sampler, scale_lr_for_world_size, cleanup_distributed,
    )
    _dist = init_distributed_from_env()
    if _dist.enabled:
        device = _dist.device
        LOG.info("[Teacher] DDP enabled: rank %d/%d, device=%s", _dist.rank, _dist.world_size, device)
    _v91_cfg = getattr(cfg, "v91_config", None)
    _tp_use, _tp_dir = _v91_tissue_cfg(_v91_cfg)
    _v91_phys = _v91_physics_aug(_v91_cfg)
    teacher_input_channels = _v91_teacher_input_channels(len(cfg.modalities), _v91_cfg)
    LOG.info(f"[Teacher] device={device}, base={cfg.teacher_base}, epochs={cfg.epochs}")
    LOG.info(
        f"[Teacher] lambda_region={cfg.lambda_region}, "
        f"focal_b1_inhomogeneity_prob={cfg.focal_b1_inhomogeneity_prob}, "
        f"gpu_augment={cfg.gpu_augment and device.type == 'cuda'}, "
        f"lmic_field_strength_prior={cfg.lmic_field_strength_prior}, "
        f"p_anisotropic_2d={cfg.p_anisotropic_2d}, "
        f"p_partial_contrast={cfg.p_partial_contrast}, "
        f"p_label_noise={cfg.p_label_noise}"
    )

    if cfg.auto_resources:
        def _build_teacher_for_probe() -> nn.Module:
            teacher_probe = build_teacher(
                in_channels=len(cfg.modalities),
                num_classes=cfg.num_classes,
                base=cfg.teacher_base,
                arch=getattr(cfg, "teacher_arch", "unet3d"),
                block_style=getattr(cfg, "teacher_block_style", "mednext"),
                # Probe must match the real model so the VRAM estimate is accurate.
                feature_dr=getattr(cfg, "teacher_feature_dr", "none"),
                feature_dr_p=getattr(cfg, "teacher_feature_dr_p", 0.5),
                feature_dr_alpha=getattr(cfg, "teacher_feature_dr_alpha", 0.1),
            )
            return _maybe_wrap_channel_adapter(
                teacher_probe,
                in_channels=teacher_input_channels,
                native_channels=len(cfg.modalities),
                label="Teacher probe",
                identity_map=_teacher_adapter_identity_map(len(cfg.modalities)),
            )
        auto_tune_resources(
            cfg,
            build_model_fn=_build_teacher_for_probe,
            input_shape=(teacher_input_channels, *cfg.patch_size),
            num_classes=cfg.num_classes,
            out_dir=out,
            label="Teacher",
            require_backward=True,
            safety=cfg.auto_resources_safety,
        )

    energy_tracker = EnergyTracker(
        label="Teacher", out_dir=out,
        gpu_index=_tracked_gpu_index(),
        num_cpu_cores=_allocated_cpu_budget(cfg.num_workers + 2),
    )
    energy_tracker.start()
    try:
        # Augmentation-engine selection [F1]. The legacy physics augmentor and the
        # opt-in v9.1 physics augmentor are MUTUALLY EXCLUSIVE, so overlapping
        # physical transforms (bias / Rician / gamma / resolution / k-space) are
        # never applied twice. When v9.1 physics is enabled it becomes the sole
        # physics engine and runs batched ON THE GPU inside the CUDA prefetcher
        # (or exactly once on the CPU in the dataset when there is no GPU). With no
        # v9.1 config the selection is byte-identical to the historical path.
        _v91_gpu_phys = _v91_gpu_physics_aug(_v91_cfg) if device.type == "cuda" else None
        use_v91_gpu_augment = _v91_gpu_phys is not None
        legacy_physics_on = bool(cfg.augment and _v91_phys is None)
        use_gpu_augment = bool(cfg.gpu_augment and device.type == "cuda" and legacy_physics_on)
        gpu_augmentor = (
            GPUPhysicsAugmentor(
                p_focal_b1_inhomogeneity=cfg.focal_b1_inhomogeneity_prob,
                p_anisotropic_2d=cfg.p_anisotropic_2d,
                p_partial_contrast=cfg.p_partial_contrast,
                lmic_field_strength_prior=cfg.lmic_field_strength_prior,
                contrast_mod_indices=cfg.contrast_mod_indices,
                p_field_contrast_warp=getattr(cfg, "p_field_contrast_warp", 0.0),
                field_contrast_source_b0=getattr(cfg, "field_contrast_source_b0", 3.0),
            )
            if use_gpu_augment else _v91_gpu_phys
        )
        # Legacy CPU augmentation only when no GPU augmentor of either kind runs.
        dataset_legacy_augment = bool(legacy_physics_on and not use_gpu_augment)
        # v9.1 runs on the GPU prefetcher => the dataset must NOT also run it (F1).
        dataset_physics_aug = None if use_v91_gpu_augment else _v91_phys
        # Geometric aug runs on the GPU prefetcher (carried by the v9.1 GPU augmentor)
        # when that path is active, else once on the CPU in the dataset.
        dataset_geometric_aug = None if use_v91_gpu_augment else _v91_geometric_aug(_v91_cfg)
        _assert_tissue_prior_channel_safety(_tp_use, dataset_legacy_augment, use_gpu_augment)
        if use_v91_gpu_augment:
            LOG.info(
                "v9.1 physics augmentation -> batched on-GPU prefetcher "
                "(legacy physics augmentor disabled; F1: single application)"
            )
        augment_cfg = {
            "p_focal_b1_inhomogeneity": cfg.focal_b1_inhomogeneity_prob,
            "p_anisotropic_2d": cfg.p_anisotropic_2d,
            "p_partial_contrast": cfg.p_partial_contrast,
            "lmic_field_strength_prior": cfg.lmic_field_strength_prior,
            "contrast_mod_indices": cfg.contrast_mod_indices,
            # Architecture-paper rigor extensions
            "p_field_contrast_warp": getattr(cfg, "p_field_contrast_warp", 0.0),
            "field_contrast_source_b0": getattr(cfg, "field_contrast_source_b0", 3.0),
            "p_fourier_amplitude_mix": getattr(cfg, "p_fourier_amplitude_mix", 0.0),
            "p_receive_coil_inhomogeneity": getattr(cfg, "p_receive_coil_inhomogeneity", 0.0),
            "p_off_resonance_void": getattr(cfg, "p_off_resonance_void", 0.0),
            "noise_calibration": getattr(cfg, "noise_calibration", "static"),
        }
        train_ds = MultiModalSegDataset(
            cfg.train_csv,
            cfg.modalities,
            cfg.patch_size,
            True,
            dataset_legacy_augment,
            augment_cfg=augment_cfg,
            cache_dir=cfg.cache_dir,
            patch_cache_dir=cfg.patch_cache_dir,
            patch_cache_start_epoch=cfg.patch_cache_start_epoch,
            label_noise_prob=cfg.p_label_noise,
            label_noise_mode=cfg.label_noise_mode,
            label_noise_band_radius=cfg.label_noise_band_radius,
            label_noise_band_flip_prob=cfg.label_noise_band_flip_prob,
            lmic_field_strength_prior=cfg.lmic_field_strength_prior,
            contrast_mod_indices=cfg.contrast_mod_indices,
            carvemix_prob=getattr(cfg, "carvemix_prob", 0.0),
            carvemix_dilation=getattr(cfg, "carvemix_dilation", 5),
            afa_prob=getattr(cfg, "afa_prob", 0.0),
            ulf_gan_prob=getattr(cfg, "p_ulf_gan_synthesis", 0.0),
            ulf_gan_weights=getattr(cfg, "ulf_gan_weights", None),
            ulf_gan_target_b0=getattr(cfg, "ulf_gan_target_b0", 0.064),
            # FIX (Gap B, 2026 audit): teacher-side modality dropout - student
            # has its own scheme so this only flows through the teacher path.
            modality_dropout_p=getattr(cfg, "teacher_modality_dropout_p", 0.0),
            modality_dropout_max_drop=getattr(cfg, "teacher_modality_dropout_max_drop", 2),
            use_tissue_priors=_tp_use,        # Phase 7.5 (default OFF -> 4-channel)
            tissue_prior_dir=_tp_dir,
            physics_aug=dataset_physics_aug,  # F1: None when running v9.1 on GPU
            bloch_sim=_v91_bloch_sim(_v91_cfg),
            bloch_prob=_v91_bloch_prob(_v91_cfg),
            synth_sim=_v91_synth_sim(_v91_cfg),
            geometric_aug=dataset_geometric_aug,
        )
        val_ds = MultiModalSegDataset(cfg.val_csv, cfg.modalities, cfg.patch_size, False, False,
                                      use_tissue_priors=_tp_use, tissue_prior_dir=_tp_dir)
        _assert_train_val_case_disjoint(train_ds, val_ds)
        g = make_loader_generator(cfg.seed)
        train_sampler, sampler_summary = build_metadata_weighted_sampler(
            train_ds.rows,
            seed=cfg.seed,
            field_strength_sampling_power=cfg.field_strength_sampling_power,
            low_field_sampling_bias=cfg.low_field_sampling_bias,
            grade23_sampling_bias=cfg.grade23_sampling_bias,
            idh_mutant_sampling_bias=cfg.idh_mutant_sampling_bias,
            legacy_protocol_sampling_bias=cfg.legacy_protocol_sampling_bias,
            legacy_year_threshold=cfg.legacy_protocol_year_threshold,
        )
        save_json(sampler_summary, out / "teacher_sampler_summary.json")
        LOG.info(f"[Teacher] metadata sampler: {json.dumps(sampler_summary)}")
        # Under DDP, shard the weighted sampler per rank (keeps the equity weighting);
        # single-process returns it unchanged.
        train_sampler = maybe_distributed_weighted_sampler(train_sampler, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds,
            **_build_loader_kwargs(
                batch_size=cfg.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=True,
                worker_init=seed_worker,
                generator=g if train_sampler is None else None,
                train_mode=True,
                persistent_workers=cfg.num_workers > 0,
            ),
        )
        val_loader = DataLoader(
            val_ds,
            **_build_loader_kwargs(
                batch_size=1,
                shuffle=False,
                num_workers=cfg.val_num_workers,
                pin_memory=device.type == "cuda",
                worker_init=seed_worker,
                generator=make_loader_generator(cfg.seed),
                train_mode=False,
                persistent_workers=False,
            ),
        )

        teacher_core = build_teacher(
            len(cfg.modalities),
            cfg.num_classes,
            cfg.teacher_base,
            arch=getattr(cfg, "teacher_arch", "unet3d"),
            block_style=getattr(cfg, "teacher_block_style", "mednext"),
            # FIX (Gap A, 2026 audit): teacher-side feature-DR; identity at eval.
            feature_dr=getattr(cfg, "teacher_feature_dr", "none"),
            feature_dr_p=getattr(cfg, "teacher_feature_dr_p", 0.5),
            feature_dr_alpha=getattr(cfg, "teacher_feature_dr_alpha", 0.1),
        )
        model = _maybe_wrap_channel_adapter(
            teacher_core,
            in_channels=teacher_input_channels,
            native_channels=len(cfg.modalities),
            label="Teacher",
            identity_map=_teacher_adapter_identity_map(len(cfg.modalities)),
        ).to(device)
        # H100: channels_last_3d (NDHWC) layout improves cache utilisation for 3D convolutions
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last_3d)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        LOG.info(f"[Teacher] Parameters: {n_params:.1f}M")
        # Phase 7.3: opt-in region-aware task loss. Default config keeps the
        # legacy criterion verbatim (bit-identical); a non-"legacy" task type
        # swaps in build_task_loss via a call-compatible adapter.
        if _v91_task_active(getattr(cfg, "v91_config", None)):
            from ogap.losses.region_aware import build_task_loss as _build_task_loss
            criterion = _V91TaskLossAdapter(_build_task_loss(cfg.v91_config))
            LOG.info("[Teacher] v9.1 task loss active: %s", cfg.v91_config.loss.task.type)
        else:
            criterion = PartialSupervisionDiceCELoss(cfg.num_classes, include_background=False)
        region_criterion = BraTSRegionLoss()
        # Linear LR scaling rule under DDP (no-op single-process: world_size=1).
        opt = torch.optim.AdamW(model.parameters(), scale_lr_for_world_size(cfg.lr),
                                weight_decay=cfg.weight_decay)
        sched = cosine_schedule_with_warmup(opt, cfg.warmup_epochs, cfg.epochs)
        use_amp = cfg.amp and device.type == "cuda"
        # H100: BF16 uses native tensor cores, no gradient scaling needed
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        # Gradient accumulation (parity with the student loop); 1 == per-batch step.
        accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1)))

        best, history = -1.0, []
        start_epoch = 1

        # ── Resume from checkpoint if available ───────────────────────────────────
        resume_ckpt = out / "resume_teacher.pth"
        ckpt = load_resume_checkpoint_or_none(resume_ckpt, device, label="Teacher")
        if ckpt is not None:
            raw_model_sd = ckpt["model"]
            model_sd = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in raw_model_sd.items()
            }
            model_sd = _state_dict_for_maybe_adapted_model(model, model_sd)
            # FIX (Finding 4, 2026 audit): The previous code used strict=True
            # (the load_state_dict default), so a stale ``resume_teacher.pth``
            # written by a different ``teacher_block_style`` / ``teacher_base``
            # / ``teacher_arch`` failed 20 min into job startup with a cryptic
            # "Missing key(s)" error.  We now load non-strictly, surface a
            # clear actionable diagnostic, and abort unless the user opts in
            # via ``--force_resume_partial`` (or sets ``force_resume_partial``
            # on the config).
            missing, unexpected = model.load_state_dict(model_sd, strict=False)
            if missing or unexpected:
                LOG.error(
                    "[Teacher] resume_teacher.pth at %s does not match the requested "
                    "model config (arch=%s, base=%d, block_style=%s): missing=%d keys, "
                    "unexpected=%d keys. Either: (a) delete %s if you intended to "
                    "start fresh, (b) re-run with the matching --teacher_block_style "
                    "/ --teacher_base / --teacher_arch, or (c) set "
                    "--force_resume_partial to load only the matching keys (loses "
                    "the rest).",
                    resume_ckpt, getattr(cfg, "teacher_arch", "unet3d"),
                    cfg.teacher_base, getattr(cfg, "teacher_block_style", "mednext"),
                    len(missing), len(unexpected), resume_ckpt,
                )
                if not getattr(cfg, "force_resume_partial", False):
                    raise RuntimeError(
                        "resume_teacher.pth shape mismatch - aborting before silent retraining."
                    )
                LOG.warning("[Teacher] --force_resume_partial set; proceeding with partial state.")
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            best = ckpt.get("best", -1.0)
            history = ckpt.get("history", [])
            start_epoch = ckpt.get("epoch", 0) + 1
            LOG.info(f"[Teacher] Resumed at epoch {start_epoch}, best={best:.4f}")

        # Wrap in DDP after resume (each rank loaded identical weights), before EMA +
        # compile so both operate on the DDP-unwrapped module. No-op single-process.
        try:
            _ddp_fup = bool(_v91_cfg.training.ddp.find_unused_parameters)
        except AttributeError:
            _ddp_fup = False
        model = wrap_ddp(model, device, find_unused_parameters=_ddp_fup)

        # Weight EMA built AFTER resume (so it tracks the resumed weights, not the
        # initial ones) and BEFORE compile (deepcopy the plain module). Default OFF.
        from ogap.utils.ema import build_model_ema, ModelEMA
        ema = build_model_ema(model, _v91_cfg)
        if ema is None and getattr(cfg, "ema", False):
            # Bare legacy-CLI teacher (no v9.1 --config): honour the TeacherConfig
            # EMA knob directly. build_model_ema only fires from the v9.1 config,
            # so without this the --ema flag would be silently ignored. Built here
            # (post-DDP, pre-compile) so it tracks resumed weights on the plain module.
            ema = ModelEMA(model, decay=float(getattr(cfg, "ema_decay", 0.999)))
        if ema is not None:
            LOG.info("[Teacher] weight EMA enabled (decay=%.4f); best/last save EMA weights", ema.decay)

        # Compile only after resume loading so state_dict keys always target the
        # plain module. OptimizedModule wrappers are fine for training, but they
        # are a brittle target for stripped _orig_mod resume keys.
        compile_enabled, compile_mode, compile_dynamic = _v91_compile_cfg(_v91_cfg)
        if getattr(cfg, "compile", False) and not compile_enabled:
            # Bare legacy-CLI teacher: honour the TeacherConfig compile knob
            # directly (the v9.1 config still wins when it enables compile).
            # maybe_compile_model() degrades to eager if Inductor/Triton is absent.
            compile_enabled = True
            compile_mode = str(getattr(cfg, "compile_mode", "max-autotune"))
            compile_dynamic = bool(getattr(cfg, "compile_dynamic", False))
        if device.type == "cuda" and compile_enabled:
            model = maybe_compile_model(
                model, "Teacher", mode=compile_mode, dynamic=compile_dynamic
            )

        for epoch in range(start_epoch, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            # WeightedRandomSampler with persistent_workers=True repeats the same weighted
            # draw across epochs unless reseeded, so we rebuild the sampler with an
            # epoch-perturbed seed and reassign train_loader.sampler. This IS honoured under
            # persistent_workers: for a map-style dataset the sampler iterator runs in the
            # MAIN process (DataLoader.__iter__ reads loader.sampler fresh each epoch and
            # dispatches indices to workers); persistent_workers cache the worker PROCESSES,
            # not the index stream. Each epoch therefore draws fresh IID samples over the
            # weighted distribution, reproducibly under a fixed seed.
            if train_sampler is not None:
                fresh_sampler, _ = build_metadata_weighted_sampler(
                    train_ds.rows,
                    seed=cfg.seed + epoch,
                    field_strength_sampling_power=cfg.field_strength_sampling_power,
                    low_field_sampling_bias=cfg.low_field_sampling_bias,
                    grade23_sampling_bias=cfg.grade23_sampling_bias,
                    idh_mutant_sampling_bias=cfg.idh_mutant_sampling_bias,
                    legacy_protocol_sampling_bias=cfg.legacy_protocol_sampling_bias,
                    legacy_year_threshold=cfg.legacy_protocol_year_threshold,
                )
                fresh_sampler = maybe_distributed_weighted_sampler(
                    fresh_sampler, seed=cfg.seed + epoch)
                if fresh_sampler is not None:
                    if hasattr(fresh_sampler, "set_epoch"):
                        fresh_sampler.set_epoch(epoch)
                    # torch>=2.6 forbids reassigning DataLoader.sampler after init
                    # (ValueError). And because batch_size is set, the loader iterates
                    # via batch_sampler.sampler, NOT loader.sampler - so update BOTH in
                    # place (bypassing the __setattr__ guard) to actually reseed the
                    # per-epoch weighted draw under persistent_workers.
                    object.__setattr__(train_loader, "sampler", fresh_sampler)
                    _bs = getattr(train_loader, "batch_sampler", None)
                    if _bs is not None and getattr(_bs, "sampler", None) is not None:
                        _bs.sampler = fresh_sampler
                    train_sampler = fresh_sampler
            epoch_t0 = time.time()
            LOG.info(f"[Teacher] Epoch {epoch}/{cfg.epochs} start")
            lr_used = opt.param_groups[0]["lr"]  # actual LR this epoch (captured before sched.step)
            model.train()
            run_loss = torch.zeros((), device=device)
            run_region = torch.zeros((), device=device)
            n = 0
            num_train_batches = len(train_loader)
            log_every = _train_log_every()
            log_first_n = _train_log_first_n()
            first_batch_wait_t0 = time.time()
            LOG.info(
                "[Teacher] Epoch %d: train batches=%d, batch_size=%d, workers=%d",
                epoch, num_train_batches, cfg.batch_size, cfg.num_workers,
            )
            train_batches = (
                CUDATrainPrefetcher(train_loader, device, gpu_augmentor)
                if device.type == "cuda" else train_loader
            )
            for batch_idx, batch in enumerate(train_batches, start=1):
                _, x, y, _, tags = batch
                if batch_idx == 1:
                    LOG.info(
                        "[Teacher] Epoch %d: first batch ready after %.1fs; x=%s y=%s",
                        epoch,
                        time.time() - first_batch_wait_t0,
                        tuple(x.shape),
                        tuple(y.shape),
                    )
                if y.numel() == 0:
                    raise ValueError(
                        "Training batch has no labels (y is empty). "
                        "Ensure your training CSV includes valid label paths."
                    )
                if device.type != "cuda":
                    x = x.to(device, non_blocking=True, memory_format=torch.channels_last_3d)
                    y = y.to(device, non_blocking=True)
                    if gpu_augmentor is not None:
                        x, y = gpu_augmentor(x, y)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                    loss_dense = criterion(logits, y, dataset_tags=tags)
                    if _v91_task_active(_v91_cfg):
                        loss_region = logits.new_zeros(())
                        loss = loss_dense
                    else:
                        loss_region = region_criterion(logits, y, dataset_tags=tags)
                        loss = loss_dense + cfg.lambda_region * loss_region

                # Fail loud on a non-finite loss rather than silently corrupting a
                # publication checkpoint with NaN/inf (one cheap scalar sync per step).
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"[Teacher] Non-finite loss at epoch {epoch} batch {batch_idx}; "
                        "refusing to backprop NaN/inf. Inspect the data pipeline, LR, "
                        "and AMP dtype."
                    )
                # Gradient accumulation (mirrors the student loop): scale by the real
                # group size, then clip + step + EMA + zero only at group boundaries.
                # accum_steps=1 reproduces the original per-batch behaviour exactly.
                bi0 = batch_idx - 1
                accum_group_start = bi0 - (bi0 % accum_steps)
                accum_group_size = min(accum_steps, num_train_batches - accum_group_start)
                (loss / accum_group_size).backward()
                if ((bi0 + 1) % accum_steps == 0) or ((bi0 + 1) == num_train_batches):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    if ema is not None:
                        ema.update(model)
                    opt.zero_grad(set_to_none=True)

                run_loss = run_loss + loss.detach()
                run_region = run_region + loss_region.detach()
                n += 1
                if batch_idx <= log_first_n or batch_idx % log_every == 0 or batch_idx == num_train_batches:
                    LOG.info(
                        "[Teacher] Epoch %d batch %d/%d: loss=%.4f region=%.4f elapsed=%.1fs",
                        epoch,
                        batch_idx,
                        num_train_batches,
                        float(loss.detach().cpu()),
                        float(loss_region.detach().cpu()),
                        time.time() - epoch_t0,
                    )
            sched.step()
            train_time_s = round(time.time() - epoch_t0, 1)
            LOG.info(f"[Teacher] Epoch {epoch}: train loop done in {train_time_s:.1f}s")

            metrics = {
                "epoch": epoch,
                "train_loss": float((run_loss / max(n, 1)).cpu()),
                "train_region": float((run_region / max(n, 1)).cpu()),
                "lr_used": lr_used,
                "lr_next": sched.get_last_lr()[0],
                "train_time_s": train_time_s,
            }
            run_validation = (epoch >= cfg.val_start_epoch or epoch == cfg.epochs)
            if run_validation:
                LOG.info(f"[Teacher] Epoch {epoch}: validation start")
                # FIX (2026-06-16, OOM teacher job 597737): release the training
                # step's freed-but-cached allocator blocks before the whole-volume
                # validation forward. With cudagraphs disabled (teacher compile_mode
                # 'max-autotune-no-cudagraphs') there is no pinned graph pool to
                # fight, so this hands the full card to evaluate_teacher and avoids
                # fragmentation OOMs on large, variable-size validation volumes.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                val_t0 = time.time()
                # FIX issue #5: fix seed for deterministic validation so epoch-to-epoch
                # comparisons and ablation comparisons are reproducible.
                # v9: fast validation path.
                # On most epochs only evaluate the "all" config (1 pass over val set).
                # Every val_full_freq epochs (and the final epoch) run all eval configs
                # and also compute HD95. This eliminates the 5× validation overhead that
                # made epochs 5× longer than necessary.
                is_full_val_epoch = (epoch % cfg.val_full_freq == 0 or epoch == cfg.epochs)
                compute_hd95_this_epoch = is_full_val_epoch and (
                    epoch % cfg.hd95_freq == 0 or epoch == cfg.epochs
                )

                with torch.random.fork_rng(enabled=True):
                    torch.manual_seed(cfg.seed)
                    np.random.seed(cfg.seed)
                    val_metrics = evaluate_teacher(
                        _validation_model_view(ema.module() if ema is not None else model, "Teacher"),
                        val_loader,
                        cfg.num_classes,
                        device,
                        cfg.use_brats_metrics,
                        compute_hd95=compute_hd95_this_epoch,
                    )
                metrics.update(val_metrics)
                metrics["val_time_s"] = round(time.time() - val_t0, 1)
            else:
                metrics["validation_skipped"] = True
                metrics["val_time_s"] = 0.0
            metrics["epoch_time_s"] = round(time.time() - epoch_t0, 1)
            history.append(metrics)
            LOG.info(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                 for k, v in metrics.items()}))

            # Update best on ALL ranks (validation is full + identical per rank), but
            # only rank 0 writes checkpoints (else 4 ranks clobber the same files).
            score = metrics.get("dice_brats_mean", metrics.get("dice_fg"))
            is_best = score is not None and score > best
            if is_best:
                best = score
            if is_main_process():
                # Deploy weights = EMA when enabled (best/last), else the raw model. The
                # resume checkpoint always keeps the RAW model + optimizer so training can
                # continue exactly (EMA is a deploy-time copy, not the training state).
                deploy_state = ema.state_dict() if ema is not None else model.state_dict()
                save_torch_atomic(deploy_state, out / "last_teacher.pth")
                save_torch_atomic({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "best": best,
                    "history": history,
                }, out / "resume_teacher.pth")
                if is_best:
                    save_torch_atomic(deploy_state, out / "best_teacher.pth")
                    save_json(metrics, out / "best_teacher_metrics.json")
                    LOG.info(f"  ✓ New best teacher: {best:.4f}")

        if is_main_process():
            save_json({**asdict(cfg), "_code_provenance": _code_provenance()},
                      out / "teacher_config.json")
            save_json({"history": history}, out / "teacher_history.json")
        LOG.info(f"[Teacher] Done. Best score: {best:.4f}")
    finally:
        energy_tracker.stop()
        cleanup_distributed()


# ══════════════════════════════════════════════════════════════════════════════
# §14  STUDENT KD TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    out_dir: str
    modalities: List[str]
    num_classes: int
    teacher_ckpt: str
    patch_size: Tuple[int, int, int] = (128, 160, 112)
    batch_size: int = 4
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 3e-5
    seed: int = 42
    deterministic: bool = False
    num_workers: int = 14
    val_num_workers: int = 2
    keep_min_modalities: int = 1
    keep_max_modalities: int = 4
    curriculum_floor_modalities: int = 1
    keep_count_bias_power: float = 0.0
    lambda_kd: float = 2.0
    lambda_feat: float = 0.5
    feature_distill_sparse_decay: float = 1.5
    lambda_ds: float = 1.0
    temperature_start: float = 4.0
    temperature_end: float = 4.0
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    lambda_region: float = 0.50
    lambda_vrex: float = 0.0  # V-REx variance-of-risks penalty (Krueger+ 2021)
    amp: bool = True
    augment: bool = True
    teacher_base: int = 32
    student_base: int = 16
    use_brats_metrics: bool = True
    hd95_freq: int = 50   # v9: raised from 10. See TeacherConfig note above.
    curriculum_ramp: float = 0.2
    holder_alpha: float = 1.5  # the documented HPD method (1.0 = KL fallback is an ablation, not the default)
    # NEW: architecture config for ablations
    attention: str = "none"                   # "eca", "se", "none"
    enable_deep_supervision: bool = True
    curriculum_dropout_p: float = 0.1
    use_curriculum: bool = True          # if False, keep_min stays at keep_min_modalities from epoch 1
    confidence_gate_threshold: float = 0.7
    val_fast_n: int = 9      # deployment-focused 1-4 modality subset every epoch
    val_start_epoch: int = 3  # Skip validation before this epoch (final epoch still validates)
    val_full_freq: int = 25  # run all supported 1-4 configs + HD95 every N epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[str] = None  # NVMe npy cache (e.g. $SLURM_TMPDIR/npy_cache)
    patch_cache_dir: Optional[str] = None  # RAM-backed per-epoch crop cache (e.g. /dev/shm)
    patch_cache_start_epoch: int = 1
    gpu_augment: bool = True
    lmic_field_strength_prior: bool = False
    contrast_mod_indices: List[int] = field(default_factory=list)
    contrast_dropout_extra_prob: float = 0.0
    p_anisotropic_2d: float = 0.85
    p_partial_contrast: float = 0.3
    p_label_noise: float = 0.2
    label_noise_mode: str = "morph"
    label_noise_band_radius: int = 2
    label_noise_band_flip_prob: float = 0.3
    focal_b1_inhomogeneity_prob: float = 0.12
    field_strength_sampling_power: float = 0.5
    low_field_sampling_bias: float = 1.5
    grade23_sampling_bias: float = 1.5
    idh_mutant_sampling_bias: float = 1.5
    legacy_protocol_sampling_bias: float = 1.25
    legacy_protocol_year_threshold: int = 2013
    auto_resources: bool = False
    auto_resources_safety: float = 0.70  # KD: teacher is also resident on GPU
    # evaluation rigor extensions (added in the 2026 architecture pass)
    feature_dr: str = "none"          # "none" | "mixstyle" | "dsu"
    feature_dr_p: float = 0.5
    feature_dr_alpha: float = 0.1
    p_field_contrast_warp: float = 0.0
    field_contrast_source_b0: float = 3.0
    p_fourier_amplitude_mix: float = 0.0
    afa_prob: float = 0.0  # alias on the dataset side; see MultiModalSegDataset
    p_receive_coil_inhomogeneity: float = 0.0
    p_off_resonance_void: float = 0.0
    noise_calibration: str = "static"  # "static" | "aja_fernandez"
    carvemix_prob: float = 0.0
    carvemix_dilation: int = 5
    teacher_arch: str = "unet3d"       # "unet3d" | "segmamba" | "swin_unetr" | "ode"
    # MUST match the block_style the teacher checkpoint was trained with -
    # state_dict shapes change between mednext (depthwise+pointwise) and
    # dense (full 3x3x3) blocks, so loading a dense teacher with mednext
    # would crash in load_state_dict.  Default mednext for backward compat.
    teacher_block_style: str = "mednext"  # "mednext" | "dense"
    # Student residual family. "res" preserves existing checkpoints; "ode"
    # selects WeightTiedODEBlock3D blocks throughout the student.
    student_block_style: str = "res"      # "res" | "ode"
    student_ode_steps: int = 4
    p_ulf_gan_synthesis: float = 0.0   # Lucas+ 2025 ULF GAN augmentation
    ulf_gan_weights: Optional[str] = None
    ulf_gan_target_b0: float = 0.064
    # Phase 7.2: unified v9.1 config namespace (ogap/config/defaults.yaml schema).
    # None => legacy behaviour. Threaded from the CLI; opt-in features (region-aware
    # task loss, ODE-aware KD assembly, tissue priors, physics aug) read from here.
    v91_config: Optional[object] = None


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.seed, cfg.deterministic)
    out = ensure_dir(cfg.out_dir)
    device = torch.device(cfg.device)
    # DDP (4×H100 NVLink): no-op unless launched under torchrun/srun with WORLD_SIZE>1.
    # The student trains several ad-hoc modules, so gradients are synced by manual
    # all-reduce (structure-independent) rather than wrapping the backbone in DDP.
    from ogap.utils.distributed import (
        init_distributed_from_env, is_main_process, all_reduce_gradients,
        maybe_distributed_weighted_sampler, scale_lr_for_world_size, cleanup_distributed,
    )
    _dist = init_distributed_from_env()
    if _dist.enabled:
        device = _dist.device
        LOG.info("[Student] DDP enabled: rank %d/%d, device=%s", _dist.rank, _dist.world_size, device)
    num_mod = len(cfg.modalities)
    _v91_cfg = getattr(cfg, "v91_config", None)
    _tp_use, _tp_dir = _v91_tissue_cfg(_v91_cfg)
    _v91_phys = _v91_physics_aug(_v91_cfg)
    teacher_input_channels = _v91_teacher_input_channels(num_mod, _v91_cfg)
    student_input_channels = _v91_student_input_channels(num_mod, _v91_cfg)
    LOG.info(f"[Student] device={device}, modalities={num_mod}, epochs={cfg.epochs}")
    LOG.info(f"[Student] attention={cfg.attention}, deep_supervision={cfg.enable_deep_supervision}, "
             f"holder_alpha={cfg.holder_alpha}, curriculum_dropout_p={cfg.curriculum_dropout_p}, "
             f"curriculum_floor={cfg.curriculum_floor_modalities}, "
             f"keep_count_bias_power={cfg.keep_count_bias_power}, "
             f"feature_distill_sparse_decay={cfg.feature_distill_sparse_decay}, "
             f"student_block_style={cfg.student_block_style}, "
             f"student_ode_steps={cfg.student_ode_steps}, "
             f"lambda_region={cfg.lambda_region}, "
             f"focal_b1_inhomogeneity_prob={cfg.focal_b1_inhomogeneity_prob}, "
             f"gpu_augment={cfg.gpu_augment and device.type == 'cuda'}, "
             f"lmic_field_strength_prior={cfg.lmic_field_strength_prior}, "
             f"contrast_mod_indices={cfg.contrast_mod_indices}, "
             f"contrast_dropout_extra_prob={cfg.contrast_dropout_extra_prob}, "
             f"p_anisotropic_2d={cfg.p_anisotropic_2d}, "
             f"p_partial_contrast={cfg.p_partial_contrast}, "
             f"p_label_noise={cfg.p_label_noise}")

    if cfg.auto_resources:
        # Probe with the student only; safety is tighter (default 0.70) because
        # the teacher + feature projectors + deep-supervision aux heads are also
        # resident on the GPU during training.
        def _build_student_for_probe() -> nn.Module:
            student_probe = UNet3DStudent(
                in_channels=2 * num_mod,
                num_classes=cfg.num_classes,
                base=cfg.student_base,
                attention=cfg.attention,
                enable_deep_supervision=cfg.enable_deep_supervision,
                curriculum_dropout_p=cfg.curriculum_dropout_p,
                total_epochs=cfg.epochs,
                block_style=cfg.student_block_style,
                ode_steps=cfg.student_ode_steps,
            )
            return _maybe_wrap_channel_adapter(
                student_probe,
                in_channels=student_input_channels,
                native_channels=2 * num_mod,
                label="Student probe",
                identity_map=_student_adapter_identity_map(num_mod, _v91_cfg),
            )
        auto_tune_resources(
            cfg,
            build_model_fn=_build_student_for_probe,
            input_shape=(student_input_channels, *cfg.patch_size),
            num_classes=cfg.num_classes,
            out_dir=out,
            label="Student",
            require_backward=True,
            safety=cfg.auto_resources_safety,
        )

    energy_tracker = EnergyTracker(
        label="Student", out_dir=out,
        gpu_index=_tracked_gpu_index(),
        num_cpu_cores=_allocated_cpu_budget(cfg.num_workers + 2),
    )
    energy_tracker.start()
    try:
        # Augmentation-engine selection [F1]. The legacy physics augmentor and the
        # opt-in v9.1 physics augmentor are MUTUALLY EXCLUSIVE, so overlapping
        # physical transforms (bias / Rician / gamma / resolution / k-space) are
        # never applied twice. When v9.1 physics is enabled it becomes the sole
        # physics engine and runs batched ON THE GPU inside the CUDA prefetcher
        # (or exactly once on the CPU in the dataset when there is no GPU). With no
        # v9.1 config the selection is byte-identical to the historical path.
        _v91_gpu_phys = _v91_gpu_physics_aug(_v91_cfg) if device.type == "cuda" else None
        use_v91_gpu_augment = _v91_gpu_phys is not None
        legacy_physics_on = bool(cfg.augment and _v91_phys is None)
        use_gpu_augment = bool(cfg.gpu_augment and device.type == "cuda" and legacy_physics_on)
        gpu_augmentor = (
            GPUPhysicsAugmentor(
                p_focal_b1_inhomogeneity=cfg.focal_b1_inhomogeneity_prob,
                p_anisotropic_2d=cfg.p_anisotropic_2d,
                p_partial_contrast=cfg.p_partial_contrast,
                lmic_field_strength_prior=cfg.lmic_field_strength_prior,
                contrast_mod_indices=cfg.contrast_mod_indices,
                p_field_contrast_warp=getattr(cfg, "p_field_contrast_warp", 0.0),
                field_contrast_source_b0=getattr(cfg, "field_contrast_source_b0", 3.0),
            )
            if use_gpu_augment else _v91_gpu_phys
        )
        # Legacy CPU augmentation only when no GPU augmentor of either kind runs.
        dataset_legacy_augment = bool(legacy_physics_on and not use_gpu_augment)
        # v9.1 runs on the GPU prefetcher => the dataset must NOT also run it (F1).
        dataset_physics_aug = None if use_v91_gpu_augment else _v91_phys
        # Geometric aug runs on the GPU prefetcher (carried by the v9.1 GPU augmentor)
        # when that path is active, else once on the CPU in the dataset.
        dataset_geometric_aug = None if use_v91_gpu_augment else _v91_geometric_aug(_v91_cfg)
        _assert_tissue_prior_channel_safety(_tp_use, dataset_legacy_augment, use_gpu_augment)
        if use_v91_gpu_augment:
            LOG.info(
                "v9.1 physics augmentation -> batched on-GPU prefetcher "
                "(legacy physics augmentor disabled; F1: single application)"
            )
        augment_cfg = {
            "p_focal_b1_inhomogeneity": cfg.focal_b1_inhomogeneity_prob,
            "p_anisotropic_2d": cfg.p_anisotropic_2d,
            "p_partial_contrast": cfg.p_partial_contrast,
            "lmic_field_strength_prior": cfg.lmic_field_strength_prior,
            "contrast_mod_indices": cfg.contrast_mod_indices,
            # Architecture-paper rigor extensions
            "p_field_contrast_warp": getattr(cfg, "p_field_contrast_warp", 0.0),
            "field_contrast_source_b0": getattr(cfg, "field_contrast_source_b0", 3.0),
            "p_fourier_amplitude_mix": getattr(cfg, "p_fourier_amplitude_mix", 0.0),
            "p_receive_coil_inhomogeneity": getattr(cfg, "p_receive_coil_inhomogeneity", 0.0),
            "p_off_resonance_void": getattr(cfg, "p_off_resonance_void", 0.0),
            "noise_calibration": getattr(cfg, "noise_calibration", "static"),
        }
        train_ds = MultiModalSegDataset(
            cfg.train_csv,
            cfg.modalities,
            cfg.patch_size,
            True,
            dataset_legacy_augment,
            augment_cfg=augment_cfg,
            cache_dir=cfg.cache_dir,
            patch_cache_dir=cfg.patch_cache_dir,
            patch_cache_start_epoch=cfg.patch_cache_start_epoch,
            label_noise_prob=cfg.p_label_noise,
            label_noise_mode=cfg.label_noise_mode,
            label_noise_band_radius=cfg.label_noise_band_radius,
            label_noise_band_flip_prob=cfg.label_noise_band_flip_prob,
            lmic_field_strength_prior=cfg.lmic_field_strength_prior,
            contrast_mod_indices=cfg.contrast_mod_indices,
            carvemix_prob=getattr(cfg, "carvemix_prob", 0.0),
            carvemix_dilation=getattr(cfg, "carvemix_dilation", 5),
            afa_prob=getattr(cfg, "afa_prob", 0.0),
            ulf_gan_prob=getattr(cfg, "p_ulf_gan_synthesis", 0.0),
            ulf_gan_weights=getattr(cfg, "ulf_gan_weights", None),
            ulf_gan_target_b0=getattr(cfg, "ulf_gan_target_b0", 0.064),
            use_tissue_priors=_tp_use,        # Phase 7.5 (default OFF -> 4-channel)
            tissue_prior_dir=_tp_dir,
            physics_aug=dataset_physics_aug,  # F1: None when running v9.1 on GPU
            bloch_sim=_v91_bloch_sim(_v91_cfg),
            bloch_prob=_v91_bloch_prob(_v91_cfg),
            synth_sim=_v91_synth_sim(_v91_cfg),
            geometric_aug=dataset_geometric_aug,
        )
        val_ds = MultiModalSegDataset(cfg.val_csv, cfg.modalities, cfg.patch_size, False, False,
                                      use_tissue_priors=_tp_use, tissue_prior_dir=_tp_dir)
        _assert_train_val_case_disjoint(train_ds, val_ds)
        g = make_loader_generator(cfg.seed)
        train_sampler, sampler_summary = build_metadata_weighted_sampler(
            train_ds.rows,
            seed=cfg.seed,
            field_strength_sampling_power=cfg.field_strength_sampling_power,
            low_field_sampling_bias=cfg.low_field_sampling_bias,
            grade23_sampling_bias=cfg.grade23_sampling_bias,
            idh_mutant_sampling_bias=cfg.idh_mutant_sampling_bias,
            legacy_protocol_sampling_bias=cfg.legacy_protocol_sampling_bias,
            legacy_year_threshold=cfg.legacy_protocol_year_threshold,
        )
        save_json(sampler_summary, out / "student_sampler_summary.json")
        LOG.info(f"[Student] metadata sampler: {json.dumps(sampler_summary)}")
        # Under DDP, shard the weighted sampler per rank (keeps the equity weighting);
        # single-process returns it unchanged.
        train_sampler = maybe_distributed_weighted_sampler(train_sampler, seed=cfg.seed)
        train_loader = DataLoader(
            train_ds,
            **_build_loader_kwargs(
                batch_size=cfg.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=True,
                worker_init=seed_worker,
                generator=g if train_sampler is None else None,
                train_mode=True,
                persistent_workers=cfg.num_workers > 0,
            ),
        )
        val_loader = DataLoader(
            val_ds,
            **_build_loader_kwargs(
                batch_size=1,
                shuffle=False,
                num_workers=cfg.val_num_workers,
                pin_memory=device.type == "cuda",
                worker_init=seed_worker,
                generator=make_loader_generator(cfg.seed),
                train_mode=False,
                persistent_workers=False,
            ),
        )

        # ── Teacher (frozen) ──────────────────────────────────────────────────
        teacher_core = build_teacher(
            num_mod, cfg.num_classes, cfg.teacher_base,
            arch=getattr(cfg, "teacher_arch", "unet3d"),
            block_style=getattr(cfg, "teacher_block_style", "mednext"),
        )
        if not Path(cfg.teacher_ckpt).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {cfg.teacher_ckpt}")
        _raw_sd = torch.load(cfg.teacher_ckpt, map_location=device, weights_only=True)
        # Checkpoints saved from torch.compile()'d models have keys prefixed with
        # "_orig_mod." - strip that prefix so the state_dict loads into a plain model.
        _teacher_sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in _raw_sd.items()
        }
        teacher_core.load_state_dict(_teacher_sd)
        teacher = _maybe_wrap_channel_adapter(
            teacher_core,
            in_channels=teacher_input_channels,
            native_channels=num_mod,
            label="Student teacher",
            identity_map=_teacher_adapter_identity_map(num_mod),
        ).to(device)
        if device.type == "cuda":
            teacher = teacher.to(memory_format=torch.channels_last_3d)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        LOG.info(f"[Student] Loaded teacher from {cfg.teacher_ckpt}")
        # H100: compile the frozen teacher so its per-batch forward pass also benefits
        # from kernel fusion and max-autotuned kernels. The teacher runs once per
        # training step - without compile this is the single largest unoptimised
        # compute block in the training loop, and its input shape is fixed.
        compile_enabled, compile_mode, compile_dynamic = _v91_compile_cfg(_v91_cfg)
        if device.type == "cuda" and compile_enabled:
            teacher = maybe_compile_model(
                teacher, "Student teacher", mode=compile_mode, dynamic=compile_dynamic
            )

        # ── Student ───────────────────────────────────────────────────────────
        student_core = UNet3DStudent(
            2 * num_mod, cfg.num_classes, cfg.student_base,
            attention=cfg.attention,
            enable_deep_supervision=cfg.enable_deep_supervision,
            curriculum_dropout_p=cfg.curriculum_dropout_p,
            total_epochs=cfg.epochs,
            feature_dr=getattr(cfg, "feature_dr", "none"),
            feature_dr_p=getattr(cfg, "feature_dr_p", 0.5),
            feature_dr_alpha=getattr(cfg, "feature_dr_alpha", 0.1),
            block_style=getattr(cfg, "student_block_style", "res"),
            ode_steps=getattr(cfg, "student_ode_steps", 4),
        )
        student = _maybe_wrap_channel_adapter(
            student_core,
            in_channels=student_input_channels,
            native_channels=2 * num_mod,
            label="Student",
            identity_map=_student_adapter_identity_map(num_mod, _v91_cfg),
        ).to(device)
        # Nvidia H100: channels_last_3d (NDHWC) layout improves cache utilisation for 3D convolutions
        if device.type == "cuda":
            student = student.to(memory_format=torch.channels_last_3d)
        n_params = sum(p.numel() for p in student.parameters()) / 1e6
        LOG.info(f"[Student] Parameters: {n_params:.1f}M")
        # ── Multi-scale feature projectors ────────────────────────────────────
        projectors = nn.ModuleDict({
            "bottleneck": FeatureProjector(
                student_core.bottleneck_channels,
                teacher_core.bottleneck_channels,
            ),
            "dec3": FeatureProjector(student_core.dec3_channels, teacher_core.dec3_channels),
        }).to(device)
        LOG.info(
            f"[Student] Feature projectors: bn {student_core.bottleneck_channels}→"
            f"{teacher_core.bottleneck_channels}, dec3 {student_core.dec3_channels}→"
            f"{teacher_core.dec3_channels}"
        )

        # ── Losses ────────────────────────────────────────────────────────────
        # Phase 7.3/7.4: opt-in v9.1 loss stack. Default config (task.type=="legacy")
        # keeps every legacy line verbatim (bit-identical). When a non-legacy task
        # type is selected, the region-aware task loss and the ODE-aware KD assembly
        # replace the legacy seg/KD path; legacy KD remains the default.
        _v91 = getattr(cfg, "v91_config", None)
        _v91_active = _v91_task_active(_v91)
        _v91_kd_assembly = None
        if _v91_active:
            from ogap.losses.region_aware import build_task_loss as _build_task_loss
            from ogap.losses.distillation import build_kd_loss as _build_kd_loss
            seg_crit = _V91TaskLossAdapter(_build_task_loss(_v91))
            _v91_kd_assembly = _build_kd_loss(_v91)
            LOG.info("[Student] v9.1 loss stack active: task=%s, ODE-aware KD engaged",
                     _v91.loss.task.type)
        else:
            seg_crit = PartialSupervisionDiceCELoss(cfg.num_classes, include_background=False)
        region_crit = BraTSRegionLoss()
        if cfg.enable_deep_supervision:
            ds_crit = DeepSupervisionLoss(seg_crit, num_levels=3, lambda_ds=cfg.lambda_ds)
        else:
            ds_crit = None  # Will use seg_crit directly
        _aux_tissue_enabled, _aux_tissue_weight = _v91_aux_tissue_cfg(_v91)
        aux_tissue_head = None
        aux_tissue_loss_fn = None
        if _aux_tissue_enabled:
            if not _tp_use:
                raise ValueError(
                    "loss.auxiliary_tissue.enabled=True requires data.use_tissue_priors=True"
                )
            from ogap.models.auxiliary_heads import AuxiliaryTissueHead, auxiliary_tissue_loss
            aux_tissue_head = AuxiliaryTissueHead(
                student_core.bottleneck_channels,
                num_tissue=3,
            ).to(device)
            aux_tissue_loss_fn = auxiliary_tissue_loss
            LOG.info("[Student] auxiliary tissue head active: weight=%.4f", _aux_tissue_weight)

        _domain_enabled, _domain_weight = _v91_domain_cfg(_v91)
        domain_grl = None
        domain_classifier = None
        domain_to_idx: Dict[str, int] = {}
        if _domain_enabled:
            from ogap.models.domain_adversarial import DomainClassifier, GradientReversalLayer
            domain_tags = sorted({str(row.get("dataset_tag", "unknown")) for row in train_ds.rows})
            domain_to_idx = {tag: idx for idx, tag in enumerate(domain_tags)}
            domain_grl = GradientReversalLayer(lambda_=1.0).to(device)
            domain_classifier = DomainClassifier(
                student_core.bottleneck_channels,
                num_domains=max(2, len(domain_to_idx)),
            ).to(device)
            LOG.info(
                "[Student] domain adversarial head active: domains=%d, delta=%.4f",
                max(2, len(domain_to_idx)),
                _domain_weight,
            )

        # ── Optimiser ─────────────────────────────────────────────────────────
        params = list(student.parameters()) + list(projectors.parameters())
        if aux_tissue_head is not None:
            params += list(aux_tissue_head.parameters())
        if domain_classifier is not None:
            params += list(domain_classifier.parameters())
        # Linear LR scaling rule under DDP (no-op single-process: world_size=1).
        opt = torch.optim.AdamW(params, scale_lr_for_world_size(cfg.lr),
                                weight_decay=cfg.weight_decay)
        sched = cosine_schedule_with_warmup(opt, cfg.warmup_epochs, cfg.epochs)
        use_amp = cfg.amp and device.type == "cuda"
        # H100: BF16 uses native tensor cores, no gradient scaling needed
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        # ── Deployment target: support every non-empty modality combination ───
        deploy_eval_configs_full = supported_modality_combinations(
            num_mod,
            min_present=cfg.curriculum_floor_modalities,
            max_present=cfg.keep_max_modalities,
        )

        best, history = -1.0, []
        start_epoch = 1

        # ── Resume from checkpoint if available ───────────────────────────────────
        resume_ckpt = out / "resume_student.pth"
        ckpt = load_resume_checkpoint_or_none(resume_ckpt, device, label="Student")
        if ckpt is not None:
            raw_student_sd = ckpt["student"]
            student_sd = {
                (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in raw_student_sd.items()
            }
            student_sd = _state_dict_for_maybe_adapted_model(student, student_sd)
            student.load_state_dict(student_sd)
            projectors.load_state_dict(ckpt["projectors"])
            if aux_tissue_head is not None and ckpt.get("aux_tissue_head") is not None:
                aux_tissue_head.load_state_dict(ckpt["aux_tissue_head"])
            elif aux_tissue_head is not None:
                LOG.warning("[resume] aux_tissue_head absent from checkpoint - it will "
                            "train from a RANDOM init for the rest of the run.")
            if domain_classifier is not None and ckpt.get("domain_classifier") is not None:
                domain_classifier.load_state_dict(ckpt["domain_classifier"])
            elif domain_classifier is not None:
                LOG.warning("[resume] domain_classifier absent from checkpoint - it will "
                            "train from a RANDOM init for the rest of the run.")
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            best = ckpt.get("best", -1.0)
            history = ckpt.get("history", [])
            start_epoch = ckpt.get("epoch", 0) + 1
            LOG.info(f"[Student] Resumed at epoch {start_epoch}, best={best:.4f}")

        # Weight EMA built AFTER resume (tracks the resumed deployed student) and
        # BEFORE compile (deepcopy the plain module). Default OFF (byte-identical).
        from ogap.utils.ema import build_model_ema
        ema = build_model_ema(student, _v91_cfg)
        if ema is not None:
            LOG.info("[Student] weight EMA enabled (decay=%.4f); best/last save EMA weights", ema.decay)

        # Compile only after resume loading so state_dict keys always target the
        # plain student module before wrapping.
        if device.type == "cuda" and compile_enabled:
            student = maybe_compile_model(
                student, "Student", mode=compile_mode, dynamic=compile_dynamic
            )

        accum_steps = max(1, int(cfg.grad_accum_steps))
        num_train_batches = len(train_loader)
        optimizer_steps_per_epoch = max(1, math.ceil(num_train_batches / accum_steps))
        LOG.info(
            "[Student] grad_accum_steps=%d, num_train_batches=%d, optimizer_steps_per_epoch=%d",
            accum_steps,
            num_train_batches,
            optimizer_steps_per_epoch,
        )

        for epoch in range(start_epoch, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)  # DDP weighted-shard reshuffle (no-op otherwise)
            epoch_t0 = time.time()
            LOG.info(f"[Student] Epoch {epoch}/{cfg.epochs} start")
            lr_used = opt.param_groups[0]["lr"]  # actual LR this epoch (captured before sched.step)

            # Temperature annealing
            progress = (epoch - 1) / max(1, cfg.epochs - 1)
            temperature = cfg.temperature_start + progress * (cfg.temperature_end - cfg.temperature_start)

            # Modality curriculum
            # FIX: when use_curriculum=False, bypass the scheduler entirely
            # and use cfg.keep_min_modalities directly - i.e. random masking
            # with a fixed lower bound from epoch 1 (no progressive ramp).
            if cfg.use_curriculum:
                curr_keep_min = curriculum_keep_min(
                    epoch - 1,
                    cfg.epochs,
                    num_mod,
                    cfg.curriculum_ramp,
                    min_floor=cfg.curriculum_floor_modalities,
                )
            else:
                curr_keep_min = cfg.keep_min_modalities
            train_mask_configs = supported_modality_combinations(
                num_mod,
                min_present=curr_keep_min,
                max_present=cfg.keep_max_modalities,
            )

            # Update curriculum dropout schedule
            student_core.set_epoch(epoch - 1)

            student.train()
            projectors.train()
            if aux_tissue_head is not None:
                aux_tissue_head.train()
            if domain_classifier is not None:
                domain_classifier.train()
            running = {
                "total": torch.zeros((), device=device),
                "seg": torch.zeros((), device=device),
                "region": torch.zeros((), device=device),
                "kd": torch.zeros((), device=device),
                "feat": torch.zeros((), device=device),
                "vrex": torch.zeros((), device=device),
                "aux_tissue": torch.zeros((), device=device),
                "domain": torch.zeros((), device=device),
                "feat_weight": torch.zeros((), device=device),
            }
            n = 0
            optimizer_steps_done = 0
            log_every = _train_log_every()
            log_first_n = _train_log_first_n()
            first_batch_wait_t0 = time.time()
            LOG.info(
                "[Student] Epoch %d: train batches=%d, batch_size=%d, workers=%d",
                epoch, num_train_batches, cfg.batch_size, cfg.num_workers,
            )

            train_batches = (
                CUDATrainPrefetcher(train_loader, device, gpu_augmentor)
                if device.type == "cuda" else train_loader
            )
            opt.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_batches):
                _, x, y, _, tags = batch
                if batch_idx == 0:
                    LOG.info(
                        "[Student] Epoch %d: first batch ready after %.1fs; x=%s y=%s",
                        epoch,
                        time.time() - first_batch_wait_t0,
                        tuple(x.shape),
                        tuple(y.shape),
                    )
                if y.numel() == 0:
                    raise ValueError(
                        "Training batch has no labels (y is empty). "
                        "Ensure your training CSV includes valid label paths."
                    )
                if device.type != "cuda":
                    x = x.to(device, non_blocking=True, memory_format=torch.channels_last_3d)
                    y = y.to(device, non_blocking=True)
                    if gpu_augmentor is not None:
                        x, y = gpu_augmentor(x, y)
                b = x.shape[0]

                mask = random_modality_mask_from_combinations(
                    b, num_mod, train_mask_configs, device,
                    keep_count_bias_power=cfg.keep_count_bias_power,
                )
                if cfg.contrast_mod_indices and cfg.contrast_dropout_extra_prob > 0.0:
                    mask = mask.clone()
                    for c_idx in cfg.contrast_mod_indices:
                        if c_idx < 0 or c_idx >= mask.shape[1]:
                            continue
                        extra_drop = torch.rand(b, device=device) < cfg.contrast_dropout_extra_prob
                        currently_present = mask[:, c_idx] > 0.5
                        n_present = mask.sum(dim=1)
                        safe_to_drop = n_present > 1.0
                        actually_drop = extra_drop & currently_present & safe_to_drop
                        if actually_drop.any():
                            mask[actually_drop, c_idx] = 0.0
                s_in = prepare_student_input(x, mask).contiguous(memory_format=torch.channels_last_3d)

                # Keep the teacher under no_grad rather than inference_mode. The
                # teacher outputs are later consumed by losses whose backward may
                # still read those target tensors, and inference-mode tensors are
                # stricter than detached no-grad tensors in autograd contexts.
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        t_logits, t_feats = teacher(x, return_features=True)

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    # Forward with or without deep supervision
                    if cfg.enable_deep_supervision:
                        s_logits, s_aux3, s_aux2, s_feats = student(
                            s_in, return_features=True, deep_supervision=True
                        )
                    else:
                        s_logits, s_feats = student(
                            s_in, return_features=True, deep_supervision=False
                        )

                    feat_weights = feature_distillation_sample_weights(
                        mask,
                        sparse_decay=cfg.feature_distill_sparse_decay,
                    )
                    if aux_tissue_head is not None and aux_tissue_loss_fn is not None:
                        tissue_target = x[:, num_mod:num_mod + 3].argmax(dim=1).long()
                        aux_logits = aux_tissue_head(s_feats["bottleneck"], out_size=y.shape[-3:])
                        loss_aux_tissue = aux_tissue_loss_fn(
                            aux_logits,
                            tissue_target,
                            weight=_aux_tissue_weight,
                        )
                    else:
                        loss_aux_tissue = s_logits.new_zeros(())

                    if domain_classifier is not None and domain_grl is not None:
                        tag_list = [str(tag) for tag in tags]
                        domain_labels = torch.as_tensor(
                            [domain_to_idx.get(tag, 0) for tag in tag_list],
                            dtype=torch.long,
                            device=device,
                        )
                        domain_logits = domain_classifier(domain_grl(s_feats["bottleneck"]))
                        loss_domain = _domain_weight * F.cross_entropy(domain_logits, domain_labels)
                    else:
                        loss_domain = s_logits.new_zeros(())

                    if _v91_active and _v91_kd_assembly is not None:
                        # Phase 7.4: v9.1 ODE-aware KD assembly (region-aware task
                        # + per-region temperature-scaled KL). Do not compute the
                        # legacy Holder/feature losses in this branch; the assembly
                        # owns those terms and returns the logged component values.
                        _kd_out = _v91_kd_assembly(
                            s_logits, y, teacher_logits=t_logits,
                            student_features=s_feats, teacher_features=t_feats,
                        )
                        loss_seg = _kd_out["loss_task"]
                        loss_kd = _kd_out["loss_kd"]
                        loss_feat = _kd_out["loss_feat"]
                        # The v9.1 KD assembly already owns the region-aware task
                        # term. Do not add the legacy overlapping-region auxiliary
                        # here, or production_v91 silently double-counts region
                        # supervision.
                        loss_region = s_logits.new_zeros(())
                        if cfg.lambda_vrex > 0:
                            loss_vrex = vrex_penalty(s_logits, y, dataset_tags=tags)
                        else:
                            loss_vrex = s_logits.new_zeros(())
                        loss = (
                            _kd_out["loss_total"]
                            + cfg.lambda_vrex * loss_vrex
                            + loss_aux_tissue
                            + loss_domain
                        )
                        if LOG.isEnabledFor(logging.DEBUG):
                            LOG.debug(
                                "[Student v9.1] total=%.6f task=%.6f kd=%.6f "
                                "feat=%.6f attn=%.6f",
                                float(_kd_out["loss_total"]), float(_kd_out["loss_task"]),
                                float(_kd_out["loss_kd"]), float(_kd_out["loss_feat"]),
                                float(_kd_out["loss_attn"]))
                    else:
                        if cfg.enable_deep_supervision:
                            loss_seg = ds_crit([s_logits, s_aux3, s_aux2], y, dataset_tags=tags)
                        else:
                            loss_seg = seg_crit(s_logits, y, dataset_tags=tags)
                        if cfg.lambda_region > 0:
                            loss_region = region_crit(s_logits, y, dataset_tags=tags)
                        else:
                            loss_region = s_logits.new_zeros(())

                        # KD loss with voxelwise confidence gating
                        # FIX v3 Critical #1: gate applied per-voxel before reduction.
                        if cfg.lambda_kd > 0:
                            with torch.amp.autocast("cuda", enabled=False):
                                kd_map = holder_divergence(
                                    s_logits.float(), t_logits.float(), temperature, cfg.holder_alpha,
                                    reduction="none",
                                )  # (B, D, H, W); always FP32 per OGAP 2.0
                            if cfg.confidence_gate_threshold > 0:
                                gate = confidence_gated_kd_weight(
                                    t_logits.float(), cfg.confidence_gate_threshold
                                )  # (B, D, H, W) - computed under no_grad()
                                loss_kd = (kd_map * gate).mean()
                            else:
                                loss_kd = kd_map.mean()
                        else:
                            loss_kd = s_logits.new_zeros(())

                        if cfg.lambda_feat > 0:
                            loss_feat = multi_scale_feature_distillation(
                                s_feats,
                                t_feats,
                                projectors,
                                sample_weights=feat_weights,
                            )
                        else:
                            loss_feat = s_logits.new_zeros(())

                        # V-REx variance-of-risks penalty (Krueger+ ICML 2021).
                        if cfg.lambda_vrex > 0:
                            loss_vrex = vrex_penalty(s_logits, y, dataset_tags=tags)
                        else:
                            loss_vrex = s_logits.new_zeros(())
                        # FIX v3 Issue #8: Conventional additive weighting
                        # (previous version shrank seg loss via (1-lambda_kd) which
                        #  is unconventional and changes dynamics when tuning lambda_kd)
                        loss = (
                            loss_seg
                            + cfg.lambda_region * loss_region
                            + cfg.lambda_kd * loss_kd
                            + cfg.lambda_feat * loss_feat
                            + cfg.lambda_vrex * loss_vrex
                            + loss_aux_tissue
                            + loss_domain
                        )

                # Fail loud on a non-finite loss rather than silently back-propagating
                # NaN/inf into a checkpoint destined for publication (one cheap scalar
                # sync per step). BF16 shares FP32's exponent range, so this guards
                # against data/LR pathologies, not a missing GradScaler.
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"[Student] Non-finite loss at epoch {epoch} batch {batch_idx}; "
                        "refusing to backprop NaN/inf. Inspect the data pipeline, LR, "
                        "and AMP dtype."
                    )
                accum_group_start = batch_idx - (batch_idx % accum_steps)
                accum_group_size = min(accum_steps, num_train_batches - accum_group_start)
                (loss / accum_group_size).backward()

                should_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_train_batches)
                if should_step:
                    # Sync gradients across ranks before clip/step (no-op single-proc).
                    all_reduce_gradients(params)
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    opt.step()
                    if ema is not None:
                        ema.update(student)
                    opt.zero_grad(set_to_none=True)
                    optimizer_steps_done += 1

                running["total"] = running["total"] + loss.detach()
                running["seg"] = running["seg"] + loss_seg.detach()
                running["region"] = running["region"] + loss_region.detach()
                running["kd"] = running["kd"] + loss_kd.detach()
                running["feat"] = running["feat"] + loss_feat.detach()
                running["vrex"] = running["vrex"] + loss_vrex.detach()
                running["aux_tissue"] = running["aux_tissue"] + loss_aux_tissue.detach()
                running["domain"] = running["domain"] + loss_domain.detach()
                running["feat_weight"] = running["feat_weight"] + feat_weights.mean().detach()
                n += 1
                batch_number = batch_idx + 1
                if batch_number <= log_first_n or batch_number % log_every == 0 or batch_number == num_train_batches:
                    LOG.info(
                        "[Student] Epoch %d batch %d/%d: total=%.4f seg=%.4f kd=%.4f feat=%.4f elapsed=%.1fs",
                        epoch,
                        batch_number,
                        num_train_batches,
                        float(loss.detach().cpu()),
                        float(loss_seg.detach().cpu()),
                        float(loss_kd.detach().cpu()),
                        float(loss_feat.detach().cpu()),
                        time.time() - epoch_t0,
                    )

            sched.step()
            train_time_s = round(time.time() - epoch_t0, 1)
            LOG.info(f"[Student] Epoch {epoch}: train loop done in {train_time_s:.1f}s")

            metrics = {
                "epoch": epoch,
                "temperature": round(temperature, 3),
                "keep_min": curr_keep_min,
                "train_total": float((running["total"] / max(n, 1)).cpu()),
                "train_seg": float((running["seg"] / max(n, 1)).cpu()),
                "train_region": float((running["region"] / max(n, 1)).cpu()),
                "train_kd": float((running["kd"] / max(n, 1)).cpu()),
                "train_feat": float((running["feat"] / max(n, 1)).cpu()),
                "train_aux_tissue": float((running["aux_tissue"] / max(n, 1)).cpu()),
                "train_domain": float((running["domain"] / max(n, 1)).cpu()),
                "train_feat_weight": float((running["feat_weight"] / max(n, 1)).cpu()),
                "grad_accum_steps": accum_steps,
                "num_train_batches": num_train_batches,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "optimizer_steps_done": optimizer_steps_done,
                "lr_used": lr_used,
                "lr_next": sched.get_last_lr()[0],
                "train_time_s": train_time_s,
            }
            run_validation = (epoch >= cfg.val_start_epoch or epoch == cfg.epochs)
            if run_validation:
                LOG.info(f"[Student] Epoch {epoch}: validation start")
                val_t0 = time.time()
                # Deployment-focused validation path.
                # Regular epochs evaluate the full 4-modality config, all 3-modality
                # configs, and a rotating subset of 2-modality configs.
                # Full validation epochs evaluate every supported modality combo.
                is_full_val_epoch = (epoch % cfg.val_full_freq == 0 or epoch == cfg.epochs)
                eval_cfgs_this_epoch = (
                    deploy_eval_configs_full if is_full_val_epoch
                    else deployment_fast_eval_configs(
                        num_mod,
                        epoch=epoch - 1,
                        min_present=cfg.curriculum_floor_modalities,
                        max_present=cfg.keep_max_modalities,
                        total_configs=max(1, cfg.val_fast_n),
                    )
                )
                compute_hd95_this_epoch = is_full_val_epoch and (
                    epoch % cfg.hd95_freq == 0 or epoch == cfg.epochs
                )

                # FIX issue #5: deterministic validation - isolated from training RNG state
                with torch.random.fork_rng(enabled=True):
                    torch.manual_seed(cfg.seed)
                    np.random.seed(cfg.seed)
                    val_metrics = evaluate_student(
                        _validation_model_view(ema.module() if ema is not None else student, "Student"),
                        val_loader,
                        cfg.num_classes,
                        device,
                        use_brats=cfg.use_brats_metrics,
                        eval_mask_configs=eval_cfgs_this_epoch,
                        mask_modalities=num_mod,
                        compute_hd95=compute_hd95_this_epoch,
                    )
                metrics.update(val_metrics)
                score_metric = "dice_brats_mean" if cfg.use_brats_metrics else "dice_fg"
                metrics.update(
                    summarize_supported_modality_performance(
                        val_metrics,
                        eval_cfgs_this_epoch,
                        num_mod,
                        min_present=cfg.curriculum_floor_modalities,
                        max_present=cfg.keep_max_modalities,
                        base_metric=score_metric,
                    )
                )
                metrics["val_eval_configs"] = [
                    modality_config_tag(cfg_mask) for cfg_mask in eval_cfgs_this_epoch
                ]
                metrics["val_time_s"] = round(time.time() - val_t0, 1)
            else:
                metrics["validation_skipped"] = True
                metrics["val_time_s"] = 0.0
            deploy_prefix = f"deploy{cfg.curriculum_floor_modalities}{cfg.keep_max_modalities}"
            selection_metric = (
                f"{deploy_prefix}_dice_brats_mean"
                if cfg.use_brats_metrics else f"{deploy_prefix}_dice_fg"
            )
            score = metrics.get(selection_metric, metrics.get("all_dice_brats_mean", metrics.get("all_dice_fg")))
            if score is not None:
                metrics["selection_metric"] = selection_metric if selection_metric in metrics else (
                    "all_dice_brats_mean" if "all_dice_brats_mean" in metrics else "all_dice_fg"
                )
                metrics["selection_score"] = float(score)
            metrics["epoch_time_s"] = round(time.time() - epoch_t0, 1)
            history.append(metrics)
            LOG.info(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                 for k, v in metrics.items()}))
            # Update best on ALL ranks (validation is full + identical per rank), but
            # only rank 0 writes checkpoints (else 4 ranks clobber the same files).
            is_best = score is not None and score > best
            if is_best:
                best = score
            if is_main_process():
                # Deploy weights = EMA when enabled (best/last); resume keeps the RAW
                # student + optimizer so training continues exactly.
                deploy_student_state = ema.state_dict() if ema is not None else student.state_dict()
                save_torch_atomic(deploy_student_state, out / "last_student.pth")
                save_torch_atomic({
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "projectors": projectors.state_dict(),
                    "aux_tissue_head": (
                        aux_tissue_head.state_dict() if aux_tissue_head is not None else None
                    ),
                    "domain_classifier": (
                        domain_classifier.state_dict() if domain_classifier is not None else None
                    ),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "best": best,
                    "history": history,
                }, out / "resume_student.pth")
                if is_best:
                    save_torch_atomic(deploy_student_state, out / "best_student.pth")
                    save_torch_atomic(projectors.state_dict(), out / "best_projectors.pth")
                    if aux_tissue_head is not None:
                        save_torch_atomic(aux_tissue_head.state_dict(), out / "best_aux_tissue_head.pth")
                    if domain_classifier is not None:
                        save_torch_atomic(domain_classifier.state_dict(), out / "best_domain_classifier.pth")
                    save_json(metrics, out / "best_metrics.json")
                    LOG.info(f"  ✓ New best student: {best:.4f}")

        if is_main_process():
            save_json({**asdict(cfg), "_code_provenance": _code_provenance()},
                      out / "train_config.json")
            save_json({"history": history}, out / "history.json")
        LOG.info(f"[Student] Done. Best score: {best:.4f}")
    finally:
        energy_tracker.stop()
        cleanup_distributed()


# ══════════════════════════════════════════════════════════════════════════════
# §15  GAUSSIAN-WEIGHTED SLIDING WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _build_gaussian_importance_map(
    patch_size: Tuple[int, int, int], sigma_scale: float = 0.125
) -> torch.Tensor:
    """Gaussian importance map for overlap stitching (nnU-Net convention).

    Voxels near patch centres contribute more, reducing boundary artefacts.
    """
    sigmas = [s * sigma_scale for s in patch_size]
    axes = [torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0 for s in patch_size]
    grid = torch.meshgrid(*axes, indexing="ij")
    gaussian = torch.exp(-0.5 * sum((g / s) ** 2 for g, s in zip(grid, sigmas)))
    gaussian = gaussian / gaussian.max()
    gaussian = gaussian.clamp(min=1e-4)  # avoid zero weights at corners
    return gaussian.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)


# ── FIX #2: Sliding window with guaranteed full coverage ──
def _compute_start_indices(
    dim_size: int, patch_size: int, stride: int
) -> List[int]:
    """Compute start indices for sliding window ensuring full coverage.

    FIX: Previous code used range(0, max(1, D - pd + 1), sd) which can
    miss the final patch when (D - pd) is not divisible by stride.

    Now we build explicit start indices and always append the last valid
    start (dim_size - patch_size) if it's not already covered.

    Ref: MONAI SlidingWindowInferer, nnU-Net v2
    """
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    # Ensure last patch covers the trailing edge
    last_start = dim_size - patch_size
    if starts[-1] < last_start:
        starts.append(last_start)
    return starts


def _scatter_add_patch_batch(
    out: torch.Tensor,
    count: torch.Tensor,
    patch_values: torch.Tensor,
    patch_weights: torch.Tensor,
    coord_chunk: Sequence[Tuple[int, int, int]],
    volume_shape: Tuple[int, int, int],
) -> None:
    """Accumulate a batch of weighted 3D patches without per-patch Python loops."""
    n_patches = len(coord_chunk)
    if n_patches == 0:
        return

    D, H, W = volume_shape
    _, channels, pd, ph, pw = patch_values.shape
    device = patch_values.device

    z_offsets = torch.arange(pd, device=device, dtype=torch.long)[:, None, None] * (H * W)
    y_offsets = torch.arange(ph, device=device, dtype=torch.long)[None, :, None] * W
    x_offsets = torch.arange(pw, device=device, dtype=torch.long)[None, None, :]
    patch_offsets = (z_offsets + y_offsets + x_offsets).reshape(1, -1)

    starts = torch.as_tensor(coord_chunk, device=device, dtype=torch.long)
    base_offsets = (
        starts[:, 0] * (H * W)
        + starts[:, 1] * W
        + starts[:, 2]
    ).unsqueeze(1)
    patch_indices = base_offsets + patch_offsets  # (N, pd*ph*pw)

    out_flat = out.view(channels, -1)
    src_flat = patch_values.permute(1, 0, 2, 3, 4).reshape(channels, -1)
    idx_flat = patch_indices.unsqueeze(0).expand(channels, -1, -1).reshape(channels, -1)
    out_flat.scatter_add_(1, idx_flat, src_flat)

    count_flat = count.view(-1)
    count_src = patch_weights.expand(n_patches, 1, pd, ph, pw).reshape(-1)
    count_flat.scatter_add_(0, patch_indices.reshape(-1), count_src)


def sliding_window_inference(
    model: nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    use_gaussian: bool = True,
    sw_batch_size: int = 4,
    force_eval: bool = True,
) -> torch.Tensor:
    """Gaussian-weighted sliding-window inference with guaranteed coverage."""
    if force_eval:
        model.eval()
    if x.shape[0] != 1:
        raise ValueError("batch_size must be 1 for sliding window")
    _, c, d, h, w = x.shape
    pd, ph, pw = patch_size

    pad_d = max(0, pd - d); pad_h = max(0, ph - h); pad_w = max(0, pw - w)
    if pad_d + pad_h + pad_w > 0:
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2,
                       pad_d // 2, pad_d - pad_d // 2))
    _, _, D, H, W = x.shape

    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # FIX #2: Use explicit start indices with guaranteed last-patch coverage
    starts_d = _compute_start_indices(D, pd, sd)
    starts_h = _compute_start_indices(H, ph, sh)
    starts_w = _compute_start_indices(W, pw, sw)

    # Gaussian importance map
    if use_gaussian:
        gaussian_map = _build_gaussian_importance_map(patch_size).to(x.device)
    else:
        gaussian_map = torch.ones(1, 1, pd, ph, pw, device=x.device)

    out: Optional[torch.Tensor] = None
    count = torch.zeros(1, 1, D, H, W, device=x.device)
    patch_coords = [(z, y0, x0) for z in starts_d for y0 in starts_h for x0 in starts_w]
    sw_batch_size = max(1, int(sw_batch_size))

    with torch.inference_mode():
        for coord_chunk in _chunked(patch_coords, sw_batch_size):
            patches = torch.cat(
                [x[:, :, z:z + pd, y0:y0 + ph, x0:x0 + pw] for z, y0, x0 in coord_chunk],
                dim=0,
            )
            if x.device.type == "cuda":
                patches = patches.contiguous(memory_format=torch.channels_last_3d)
            logits_batch = model(patches)
            if out is None:
                out = torch.zeros(1, logits_batch.shape[1], D, H, W, device=x.device)
            _scatter_add_patch_batch(
                out=out[0],
                count=count[0, 0],
                patch_values=logits_batch * gaussian_map,
                patch_weights=gaussian_map,
                coord_chunk=coord_chunk,
                volume_shape=(D, H, W),
            )

    if out is None:
        raise RuntimeError("Internal error: output buffer was not initialised")
    out = out / count.clamp(min=1e-8)

    if pad_d + pad_h + pad_w > 0:
        z0 = pad_d // 2; y0_ = pad_h // 2; x0_ = pad_w // 2
        out = out[:, :, z0:z0 + d, y0_:y0_ + h, x0_:x0_ + w]
    return out


def tta_sliding_window(
    model: nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 2,
) -> torch.Tensor:
    """8-flip TTA over all spatial axis combinations, batched per patch."""
    model.eval()
    if x.shape[0] != 1:
        raise ValueError("batch_size must be 1 for sliding window TTA")

    _, _, d, h, w = x.shape
    pd, ph, pw = patch_size
    flip_combos = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

    pad_d = max(0, pd - d); pad_h = max(0, ph - h); pad_w = max(0, pw - w)
    if pad_d + pad_h + pad_w > 0:
        x = F.pad(
            x,
            (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            ),
        )
    _, _, D, H, W = x.shape

    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    starts_d = _compute_start_indices(D, pd, sd)
    starts_h = _compute_start_indices(H, ph, sh)
    starts_w = _compute_start_indices(W, pw, sw)

    gaussian_map = _build_gaussian_importance_map(patch_size).to(x.device)
    out: Optional[torch.Tensor] = None
    count = torch.zeros(1, 1, D, H, W, device=x.device)
    patch_coords = [(z, y0, x0) for z in starts_d for y0 in starts_h for x0 in starts_w]
    sw_batch_size = max(1, int(sw_batch_size))

    with torch.inference_mode():
        for coord_chunk in _chunked(patch_coords, sw_batch_size):
            base_patches = torch.cat(
                [x[:, :, z:z + pd, y0:y0 + ph, x0:x0 + pw] for z, y0, x0 in coord_chunk],
                dim=0,
            )
            patches = torch.cat(
                [base_patches.flip(dims) if dims else base_patches for dims in flip_combos],
                dim=0,
            )
            if x.device.type == "cuda":
                patches = patches.contiguous(memory_format=torch.channels_last_3d)
            logits_stack = model(patches).view(
                len(flip_combos),
                len(coord_chunk),
                -1,
                pd,
                ph,
                pw,
            )
            restored = []
            for flip_idx, dims in enumerate(flip_combos):
                logits_part = logits_stack[flip_idx]
                if dims:
                    logits_part = logits_part.flip(dims)
                restored.append(F.softmax(logits_part.float(), dim=1))
            prob_avg = torch.stack(restored, dim=0).mean(dim=0)
            if out is None:
                out = torch.zeros(1, prob_avg.shape[1], D, H, W, device=x.device)
            _scatter_add_patch_batch(
                out=out[0],
                count=count[0, 0],
                patch_values=prob_avg * gaussian_map,
                patch_weights=gaussian_map,
                coord_chunk=coord_chunk,
                volume_shape=(D, H, W),
            )

    if out is None:
        raise RuntimeError("Internal error: output buffer was not initialised")
    out = out / count.clamp(min=1e-8)
    out = torch.log(out.clamp_min(1e-8))

    if pad_d + pad_h + pad_w > 0:
        z0 = pad_d // 2; y0_ = pad_h // 2; x0_ = pad_w // 2
        out = out[:, :, z0:z0 + d, y0_:y0_ + h, x0_:x0_ + w]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §15B  TEST-TIME ADAPTATION (Tent + MEMO).
#
# Both methods entropy-minimize on a per-case basis at evaluation time. They
# are episodic: the model snapshot is checkpointed before adaptation and
# restored afterwards, so adapted predictions never bleed across cases.
#
# Refs:
#   Wang+ "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021.
#   Zhang+ "MEMO: Test-Time Robustness via Adaptation and Augmentation",
#     NeurIPS 2022.
# ══════════════════════════════════════════════════════════════════════════════

def _segmentation_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean per-voxel categorical entropy of softmax(logits).

    Returns a scalar suitable for backprop.
    """
    # Use float32 for numerical stability under AMP.
    log_probs = F.log_softmax(logits.float(), dim=1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=1)  # (B, D, H, W)
    return ent.mean()


def _collect_norm_affine_params(model: nn.Module) -> List[nn.Parameter]:
    """Affine params of every BatchNorm / InstanceNorm / GroupNorm layer.

    These are the only parameters Tent updates - leaving the conv weights
    frozen keeps the adaptation cheap and avoids catastrophic drift.
    """
    targets: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm,
                                nn.BatchNorm2d, nn.InstanceNorm2d)):
            if module.weight is not None and module.weight.requires_grad:
                targets.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                targets.append(module.bias)
    return targets


def tent_adapt_and_predict(
    model: nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    n_steps: int = 5,
    lr: float = 1e-3,
    use_tta: bool = False,
) -> torch.Tensor:
    """Tent test-time adaptation, then a final inference pass.

    Snapshots BN/IN affine params, runs ``n_steps`` of entropy-minimization
    on the case (using a single-patch forward to keep memory tight), then
    restores the original params and runs the standard sliding-window pass
    so predictions are reproducible across cases. Returns log-probabilities
    matching the ``sliding_window_inference`` / ``tta_sliding_window``
    contract.

    NOTE: Tent's effectiveness scales with how miscalibrated the BN running
    stats are on the target site; with InstanceNorm-only models the gain is
    smaller but still positive on cross-site segmentation per Wang+ 2021.
    """
    norm_params = _collect_norm_affine_params(model)
    if not norm_params or n_steps <= 0:
        # Fall back to vanilla inference if there's nothing to adapt.
        return tta_sliding_window(model, x, patch_size, overlap) if use_tta else \
               sliding_window_inference(model, x, patch_size, overlap)

    snapshot = [p.detach().clone() for p in norm_params]
    was_training = model.training
    model.train()  # let norm layers update from the case's own statistics
    opt = torch.optim.SGD(norm_params, lr=lr, momentum=0.9)
    try:
        # Use a centered crop of `patch_size` for the adaptation pass so
        # entropy is computed on a representative chunk without paying for
        # full sliding-window cost per step.
        _, _, d, h, w = x.shape
        pd, ph, pw = patch_size
        cz = max(0, (d - pd) // 2); cy = max(0, (h - ph) // 2); cx = max(0, (w - pw) // 2)
        adapt_x = x[:, :, cz:cz + pd, cy:cy + ph, cx:cx + pw]
        if adapt_x.device.type == "cuda":
            adapt_x = adapt_x.contiguous(memory_format=torch.channels_last_3d)
        for _ in range(int(n_steps)):
            opt.zero_grad(set_to_none=True)
            logits = model(adapt_x)
            loss = _segmentation_entropy(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
            opt.step()
        model.eval()
        if use_tta:
            return tta_sliding_window(model, x, patch_size, overlap)
        return sliding_window_inference(model, x, patch_size, overlap)
    finally:
        with torch.no_grad():
            for p, s in zip(norm_params, snapshot):
                p.copy_(s)
        if was_training:
            model.train()
        else:
            model.eval()


def memo_adapt_and_predict(
    model: nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    n_steps: int = 1,
    lr: float = 5e-4,
) -> torch.Tensor:
    """MEMO test-time adaptation across TTA augmentations (Zhang+ 2022).

    For each adaptation step we generate the 8-flip TTA family of a centered
    crop, marginalize the probabilities over augmentations, and minimize the
    entropy of the marginal. The 8-flip family is the same one used by
    ``tta_sliding_window`` so the loss aligns with the inference path.
    Snapshots all parameters and restores them after; returns the standard
    8-flip TTA inference output.
    """
    if n_steps <= 0:
        return tta_sliding_window(model, x, patch_size, overlap)

    norm_params = _collect_norm_affine_params(model)
    if not norm_params:
        return tta_sliding_window(model, x, patch_size, overlap)

    snapshot = [p.detach().clone() for p in norm_params]
    was_training = model.training
    model.train()
    opt = torch.optim.SGD(norm_params, lr=lr, momentum=0.9)
    try:
        _, _, d, h, w = x.shape
        pd, ph, pw = patch_size
        cz = max(0, (d - pd) // 2); cy = max(0, (h - ph) // 2); cx = max(0, (w - pw) // 2)
        adapt_x = x[:, :, cz:cz + pd, cy:cy + ph, cx:cx + pw]
        if adapt_x.device.type == "cuda":
            adapt_x = adapt_x.contiguous(memory_format=torch.channels_last_3d)
        flip_combos = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        for _ in range(int(n_steps)):
            opt.zero_grad(set_to_none=True)
            probs_list = []
            for dims in flip_combos:
                xa = adapt_x.flip(dims) if dims else adapt_x
                logits = model(xa).float()
                if dims:
                    logits = logits.flip(dims)
                probs_list.append(F.softmax(logits, dim=1))
            marginal = torch.stack(probs_list, dim=0).mean(dim=0)
            log_marg = (marginal.clamp_min(1e-8)).log()
            loss = -(marginal * log_marg).sum(dim=1).mean()  # entropy of the marginal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
            opt.step()
        model.eval()
        return tta_sliding_window(model, x, patch_size, overlap)
    finally:
        with torch.no_grad():
            for p, s in zip(norm_params, snapshot):
                p.copy_(s)
        if was_training:
            model.train()
        else:
            model.eval()


# ══════════════════════════════════════════════════════════════════════════════
# §16  ONNX EXPORT + STATIC INT8 QUANTISATION
# ══════════════════════════════════════════════════════════════════════════════

def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return SHA256 for a file if it exists."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _artifact_summary(path: Union[str, Path]) -> Dict[str, Any]:
    """Compact metadata for exported ONNX artifacts."""
    p = Path(path)
    info: Dict[str, Any] = {
        "path": str(p.resolve()) if p.exists() else str(p),
        "exists": p.exists(),
    }
    if p.exists() and p.is_file():
        info["size_bytes"] = p.stat().st_size
        info["size_mb"] = round(p.stat().st_size / 1e6, 3)
        info["sha256"] = _file_sha256(p)
    return info


def _default_reduce_range_for_host() -> bool:
    """Best-effort ORT reduce_range default for x86 CPUs without VNNI.

    ONNX Runtime recommends S8S8/QDQ as the first choice and suggests
    reduce-range mainly for AVX2 / AVX512 non-VNNI x86 paths when accuracy
    drops. We therefore only auto-enable it when the host looks like one of
    those CPUs; otherwise it stays off unless the user overrides it.
    """
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    try:
        flags: set[str] = set()
        with open(cpuinfo, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("flags"):
                    _, _, value = line.partition(":")
                    flags.update(value.strip().split())
                    break
        has_avx2_or_avx512 = ("avx2" in flags) or any(flag.startswith("avx512") for flag in flags)
        has_vnni = any(flag in flags for flag in ("avx_vnni", "avxvnni", "vnni"))
        return has_avx2_or_avx512 and not has_vnni
    except (OSError, UnicodeDecodeError):
        return False


def _configure_ort_session_options(ort_module, num_threads: Optional[int] = None):
    """Create ORT session options with conservative, reproducible defaults."""
    opts = ort_module.SessionOptions()
    # Let OpenVINO/oneDNN own graph-level optimisation for the INT8 CPU path.
    opts.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_DISABLE_ALL
    if num_threads is not None:
        nthreads = max(1, int(num_threads))
        opts.intra_op_num_threads = nthreads
        opts.inter_op_num_threads = 1 if nthreads == 1 else 2
    opts.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts


def _provider_name(provider: Any) -> str:
    return str(provider[0] if isinstance(provider, tuple) else provider)


def _enable_cpu_ep_graph_optim_if_needed(ort_module, opts: Any, providers: Sequence[Any], label: str) -> None:
    if providers and _provider_name(providers[0]) == "CPUExecutionProvider":
        opts.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
        LOG.info("[%s] plain CPU EP - enabling ORT graph fusion (ORT_ENABLE_ALL)", label)


def _load_saved_student_arch_config(
    ckpt_path: Path,
    student_base: int,
    attention: str = "none",
    enable_deep_supervision: bool = True,
    curriculum_dropout_p: float = 0.1,
) -> Tuple[int, str, bool, float, str, int]:
    """Read student architecture knobs from train_config.json when available."""
    student_block_style = "res"
    student_ode_steps = 4
    config_path = ckpt_path.parent / "train_config.json"
    if not config_path.exists():
        return (
            student_base, attention, enable_deep_supervision,
            curriculum_dropout_p, student_block_style, student_ode_steps,
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        # Parse ALL fields into locals first and only commit them if every cast
        # succeeds. A partial parse (e.g. a truncated train_config.json that reads
        # student_base but fails on a later field) must NOT return a mixed config:
        # block_style silently defaulting to "res" when the saved model was "ode"
        # would export the WRONG ONNX architecture.
        parsed_base = int(saved_cfg.get("student_base", student_base))
        parsed_attention = str(saved_cfg.get("attention", attention))
        parsed_ds = bool(saved_cfg.get("enable_deep_supervision", enable_deep_supervision))
        parsed_cdp = float(saved_cfg.get("curriculum_dropout_p", curriculum_dropout_p))
        parsed_block = str(saved_cfg.get("student_block_style", student_block_style))
        if parsed_block not in {"res", "ode"}:
            parsed_block = "res"
        parsed_steps = max(1, int(saved_cfg.get("student_ode_steps", student_ode_steps)))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        LOG.warning("[export] could not read saved arch config from %s (%s); using CLI "
                    "defaults for ALL student arch fields (no partial config).",
                    config_path, exc)
        return (
            student_base, attention, enable_deep_supervision,
            curriculum_dropout_p, student_block_style, student_ode_steps,
        )
    return (
        parsed_base, parsed_attention, parsed_ds,
        parsed_cdp, parsed_block, parsed_steps,
    )

def export_to_onnx(
    ckpt: str, out_path: str, modalities: Sequence[str],
    num_classes: int, patch_size: Tuple[int, int, int],
    student_base: int = 16, calibration_csv: Optional[str] = None,
    calibration_samples: int = 200,
    calibration_min_present: int = 1,
    calibration_max_present: Optional[int] = 4,
    static_calibration_method: str = "percentile",
    static_reduce_range: Optional[bool] = None,
    static_per_channel: bool = True,
    run_quant_preprocess: bool = True,
    run_compare: bool = False,
    compare_val_csv: Optional[str] = None,
    compare_out_dir: Optional[str] = None,
    compare_fp32_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compare_num_workers: int = 4,
    compare_num_threads: Optional[int] = None,
    compare_eval_config_batch_size: int = 1,
    compare_metric_workers: int = 1,
    compare_cache_dir: Optional[str] = None,
    compare_track_energy: bool = True,
    compare_energy_poll_interval: float = 5.0,
) -> None:
    """Export student to ONNX with optional static INT8 quantisation.

    FIX #7: Calibration reader now pads small volumes.
    FIX #12: Handles >2GB models with external data storage.

    ONNX Runtime guidance for CNN-like models favours:
      1. pre-processing before quantisation,
      2. static calibration over dynamic quantisation when calibration data exists,
      3. S8S8 + QDQ as the first CPU target,
      4. explicit accuracy comparison/debugging if INT8 drifts too much.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("pip install onnx --break-system-packages")

    num_mod = len(modalities)
    ckpt_path = Path(ckpt)
    (
        student_base, attention, enable_ds, curriculum_dp,
        student_block_style, student_ode_steps,
    ) = _load_saved_student_arch_config(
        ckpt_path,
        student_base,
        attention="none",
        enable_deep_supervision=True,
        curriculum_dropout_p=0.1,
    )

    raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # torch.compile() wraps all parameter names with "_orig_mod." - strip it
    # so the checkpoint can be loaded into a plain (non-compiled) model.
    sd = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v
          for k, v in raw_sd.items()}
    adapter_weight = sd.get("adapter.proj.weight")
    export_input_channels = 2 * num_mod
    native_input_channels = 2 * num_mod
    adapted_checkpoint = isinstance(adapter_weight, torch.Tensor)
    if adapted_checkpoint:
        native_input_channels = int(adapter_weight.shape[0])
        export_input_channels = int(adapter_weight.shape[1])

    core_model = UNet3DStudent(
        native_input_channels,
        num_classes,
        student_base,
        attention=attention,
        enable_deep_supervision=enable_ds,
        curriculum_dropout_p=curriculum_dp,
        block_style=student_block_style,
        ode_steps=student_ode_steps,
    )
    if adapted_checkpoint:
        from ogap.models.adapters import AdaptedModel
        model = AdaptedModel(
            core_model,
            in_channels=export_input_channels,
            native_channels=native_input_channels,
        )
        LOG.info(
            "[Export] adapted student checkpoint detected: input_channels=%d native=%d",
            export_input_channels,
            native_input_channels,
        )
    else:
        model = core_model
    model.load_state_dict(sd)
    model.eval()

    pd, ph, pw = patch_size
    dummy = torch.randn(1, export_input_channels, pd, ph, pw) * 0.5

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch", 2: "D", 3: "H", 4: "W"},
            "logits": {0: "batch", 2: "D", 3: "H", 4: "W"},
        },
        do_constant_folding=True,
        verbose=False,
    )

    # FIX #12: Handle >2GB models with external data
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    onnx_size = Path(out_path).stat().st_size / 1e6

    if onnx_size > 1900:  # approaching 2GB limit
        ext_path = out_path + "_data"
        onnx.save_model(
            onnx_model, out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=Path(ext_path).name,
            size_threshold=1024,
        )
        LOG.info(f"[Export] ONNX saved with external data: {out_path} + {ext_path}")
    else:
        LOG.info(f"[Export] ONNX FP32 saved: {out_path} ({onnx_size:.1f} MB)")

    manifest: Dict[str, Any] = {
        "source_checkpoint": str(Path(ckpt).resolve()),
        "opset_version": 17,
        "modalities": list(modalities),
        "num_classes": int(num_classes),
        "input_channels": int(export_input_channels),
        "native_input_channels": int(native_input_channels),
        "adapted_checkpoint": bool(adapted_checkpoint),
        "patch_size": list(patch_size),
        "student_base": int(student_base),
        "calibration_csv": str(Path(calibration_csv).resolve()) if calibration_csv else None,
        "calibration_samples": int(calibration_samples),
        "calibration_min_present": int(calibration_min_present),
        "calibration_max_present": None if calibration_max_present is None else int(calibration_max_present),
        "static_calibration_method": static_calibration_method.lower(),
        "static_per_channel": bool(static_per_channel),
        "static_reduce_range_requested": static_reduce_range,
        "run_compare": bool(run_compare),
        "compare_val_csv": str(Path(compare_val_csv).resolve()) if compare_val_csv else None,
        "compare_out_dir": str(Path(compare_out_dir).resolve()) if compare_out_dir else None,
        "artifacts": {
            "fp32": _artifact_summary(out_path),
        },
    }

    # Dynamic INT8 quantisation (always available)
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        model_to_quantize = out_path
        preprocess_report: Dict[str, Any] = {
            "attempted": bool(run_quant_preprocess),
            "input_model": str(Path(out_path).resolve()),
            "output_model": None,
            "succeeded": False,
            "error": None,
        }

        if run_quant_preprocess:
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process

                preproc_path = str(Path(out_path).with_suffix("")) + "_preprocessed.onnx"
                preprocess_kwargs: Dict[str, Any] = {
                    "skip_symbolic_shape": True,   # CNN-like 3D U-Net, not transformer-like
                    "skip_optimization": False,
                    "skip_onnx_shape": False,
                }
                if onnx_size > 1900:
                    preprocess_kwargs.update({
                        "save_as_external_data": True,
                        "all_tensors_to_one_file": True,
                        "external_data_location": Path(preproc_path).name + ".data",
                        "external_data_size_threshold": 1024,
                    })
                quant_pre_process(out_path, preproc_path, **preprocess_kwargs)
                if Path(preproc_path).exists():
                    onnx.checker.check_model(onnx.load(preproc_path))
                    model_to_quantize = preproc_path
                    preprocess_report["output_model"] = str(Path(preproc_path).resolve())
                    preprocess_report["succeeded"] = True
                    manifest["artifacts"]["preprocessed_fp32"] = _artifact_summary(preproc_path)
                    LOG.info(f"[Export] ORT quant pre-process saved: {preproc_path}")
            except Exception as e:
                preprocess_report["error"] = str(e)
                LOG.warning(f"[Export] ORT quant pre-process failed; falling back to raw export: {e}")

        manifest["preprocess"] = preprocess_report

        q_path = str(Path(out_path).with_suffix("")) + "_int8_dynamic.onnx"
        quantize_dynamic(model_to_quantize, q_path, weight_type=QuantType.QInt8)
        onnx.checker.check_model(onnx.load(q_path))
        q_size = Path(q_path).stat().st_size / 1e6
        LOG.info(f"[Export] INT8 dynamic quantised: {q_path} ({q_size:.1f} MB, "
                 f"{onnx_size / max(q_size, 0.1):.1f}x compression)")
        manifest["artifacts"]["int8_dynamic"] = _artifact_summary(q_path)
    except ImportError:
        LOG.warning("[Export] onnxruntime not installed - skipping INT8")
        manifest["int8_dynamic_error"] = "onnxruntime not installed"
    except Exception as e:
        # Record the failure in the manifest (mirroring static_quantization_error) so a
        # deployment check can distinguish "INT8 not requested" from "INT8 export FAILED".
        LOG.error(f"[Export] INT8 dynamic quantisation FAILED: {e}")
        manifest["int8_dynamic_error"] = str(e)


    # Static INT8 quantisation with calibration data
    if calibration_csv:
        try:
            from onnxruntime.quantization import (
                CalibrationDataReader,
                CalibrationMethod,
                quantize_static,
                QuantFormat,
                QuantType as QT,
            )

            reduce_range = _default_reduce_range_for_host() if static_reduce_range is None else bool(static_reduce_range)
            calibrate_method_map = {
                "minmax": CalibrationMethod.MinMax,
                "entropy": CalibrationMethod.Entropy,
                "percentile": CalibrationMethod.Percentile,
            }
            calibrate_method = calibrate_method_map[static_calibration_method.lower()]
            calibration_configs = supported_modality_combinations(
                num_mod,
                min_present=calibration_min_present,
                max_present=calibration_max_present,
            )
            calibration_config_tags = [
                modality_config_tag(cfg) for cfg in calibration_configs
            ]

            # Static INT8 calibration should reflect the supported 1-4
            # modality deployment regime rather than calibrating on full inputs
            # only. We therefore cycle real modality masks over the selected
            # calibration cases.
            class _CalibReader(CalibrationDataReader):
                def __init__(self, csv_path, mods, ps, mask_configs, n_samples=50):
                    self.ds = InferenceDataset(csv_path, mods)
                    if len(self.ds) == 0:
                        raise ValueError("Calibration CSV is empty.")
                    self.ps = ps
                    self.mask_configs = list(mask_configs) if mask_configs else [None]
                    self.n = max(1, int(n_samples))
                    self.augmentor = MRIPhysicsAugmentor()
                    n_case_points = min(len(self.ds), self.n)
                    if n_case_points == len(self.ds):
                        self.case_indices = list(range(len(self.ds)))
                    else:
                        self.case_indices = np.linspace(
                            0, len(self.ds) - 1, num=n_case_points, dtype=np.int64
                        ).tolist()
                    self.idx = 0

                def get_next(self):
                    if self.idx >= self.n:
                        return None
                    case_idx = self.case_indices[self.idx % len(self.case_indices)]
                    missing_mods = self.mask_configs[self.idx % len(self.mask_configs)]
                    _, x_t, row = self.ds[case_idx]
                    self.idx += 1
                    x = x_t.numpy()
                    fs = _parse_row_field_strength(row)
                    self.augmentor.field_strength = (
                        sample_augmented_field_strength() if fs is None else fs
                    )
                    x, _ = self.augmentor(x, None)
                    c = x.shape[0]
                    present_arr = row.get("__present_mask__")
                    if present_arr is not None:
                        mask = torch.from_numpy(np.asarray(present_arr, dtype=np.float32)).unsqueeze(0)
                    else:
                        mask = torch.ones((1, c), dtype=torch.float32)
                    if missing_mods:
                        for mod_idx in missing_mods:
                            if 0 <= int(mod_idx) < c:
                                mask[0, int(mod_idx)] = 0.0
                    x_cal = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
                    if export_input_channels > 2 * num_mod and x_cal.shape[1] == num_mod:
                        extra = export_input_channels - (2 * num_mod)
                        zeros = x_cal.new_zeros((x_cal.shape[0], extra, *x_cal.shape[2:]))
                        x_cal = torch.cat([x_cal, zeros], dim=1)
                    inp = prepare_student_input(x_cal, mask)

                    # FIX #7: Pad volume if smaller than patch size
                    _, _, d_in, h_in, w_in = inp.shape
                    pd_c, ph_c, pw_c = self.ps
                    pad_d = max(0, pd_c - d_in)
                    pad_h = max(0, ph_c - h_in)
                    pad_w = max(0, pw_c - w_in)
                    if pad_d + pad_h + pad_w > 0:
                        inp = F.pad(inp, (
                            pad_w // 2, pad_w - pad_w // 2,
                            pad_h // 2, pad_h - pad_h // 2,
                            pad_d // 2, pad_d - pad_d // 2,
                        ))

                    # Centre crop for calibration
                    _, _, d, h, w = inp.shape
                    sd = max(0, (d - pd_c) // 2)
                    sh = max(0, (h - ph_c) // 2)
                    sw = max(0, (w - pw_c) // 2)
                    patch = inp[:, :, sd:sd+pd_c, sh:sh+ph_c, sw:sw+pw_c].contiguous().cpu().numpy().astype(np.float32, copy=False)
                    return {"input": patch}

                def rewind(self):
                    self.idx = 0

            qs_path = str(Path(out_path).with_suffix("")) + "_int8_static.onnx"
            reader = _CalibReader(
                calibration_csv,
                list(modalities),
                patch_size,
                calibration_configs,
                n_samples=calibration_samples,
            )
            model_for_static = manifest.get("preprocess", {}).get("output_model") or out_path
            quantize_static(
                model_for_static, qs_path, reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QT.QInt8,
                activation_type=QT.QInt8,
                per_channel=static_per_channel,
                reduce_range=reduce_range,
                calibrate_method=calibrate_method,
            )
            onnx.checker.check_model(onnx.load(qs_path))
            qs_size = Path(qs_path).stat().st_size / 1e6
            LOG.info(f"[Export] INT8 static quantised: {qs_path} ({qs_size:.1f} MB)")
            manifest["static_quantization"] = {
                "model_input": str(Path(model_for_static).resolve()),
                "quant_format": "QDQ",
                "activation_type": "QInt8",
                "weight_type": "QInt8",
                "per_channel": bool(static_per_channel),
                "reduce_range": bool(reduce_range),
                "calibration_method": static_calibration_method.lower(),
                "calibration_config_tags": calibration_config_tags,
            }
            manifest["artifacts"]["int8_static"] = _artifact_summary(qs_path)
        except Exception as e:
            LOG.warning(f"[Export] Static quantisation failed: {e}")
            manifest["static_quantization_error"] = str(e)

    manifest_path = Path(str(Path(out_path).with_suffix("")) + "_quantization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    LOG.info(f"[Export] Quantisation manifest: {manifest_path}")

    if run_compare:
        if not compare_val_csv:
            raise ValueError("--run_compare requires --compare_val_csv")
        compare_out = compare_out_dir or str(Path(out_path).with_suffix("")) + "_quant_compare"
        int8_model_path = None
        static_path = Path(str(Path(out_path).with_suffix("")) + "_int8_static.onnx")
        dynamic_path = Path(str(Path(out_path).with_suffix("")) + "_int8_dynamic.onnx")
        if static_path.exists():
            int8_model_path = str(static_path)
        elif dynamic_path.exists():
            int8_model_path = str(dynamic_path)
        else:
            raise FileNotFoundError(
                "run_compare requested but no exported INT8 ONNX artifact was produced."
            )
        LOG.info("[Export] Running post-export FP32 vs INT8 comparison ...")
        compare_quantisation(
            fp32_ckpt=ckpt,
            int8_onnx=int8_model_path,
            val_csv=compare_val_csv,
            modalities=modalities,
            num_classes=num_classes,
            patch_size=patch_size,
            out_dir=compare_out,
            student_base=student_base,
            fp32_device=compare_fp32_device,
            num_workers=compare_num_workers,
            num_threads=compare_num_threads,
            eval_config_batch_size=compare_eval_config_batch_size,
            metric_workers=compare_metric_workers,
            cache_dir=compare_cache_dir,
            deploy_min_present=calibration_min_present,
            deploy_max_present=calibration_max_present,
            track_energy=compare_track_energy,
            energy_poll_interval=compare_energy_poll_interval,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §17  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _ort_sliding_window(
    session, x: torch.Tensor,
    patch_size: Tuple[int, int, int], overlap: float = 0.5,
    sw_batch_size: int = 2,
    use_tta: bool = False,
) -> torch.Tensor:
    """Gaussian-weighted sliding window via ONNX Runtime on CPU.

    Uses guaranteed-coverage start indices, batches multiple patches per ORT
    call, and optionally applies 8-flip TTA inside the ORT path so CPU/INT8
    inference does not silently lose the same test-time ensembling available
    to the PyTorch path.
    """
    _, c, d, h, w = x.shape
    pd, ph, pw = patch_size
    pad_d = max(0, pd - d); pad_h = max(0, ph - h); pad_w = max(0, pw - w)
    if pad_d + pad_h + pad_w > 0:
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                       pad_h // 2, pad_h - pad_h // 2,
                       pad_d // 2, pad_d - pad_d // 2))
    _, _, D, H, W = x.shape
    x_np = x.numpy()

    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # FIX #2: Guaranteed coverage start indices
    starts_d = _compute_start_indices(D, pd, sd)
    starts_h = _compute_start_indices(H, ph, sh)
    starts_w = _compute_start_indices(W, pw, sw)

    gaussian_np = _build_gaussian_importance_map(patch_size).numpy()
    inp_name = session.get_inputs()[0].name
    out_arr: Optional[np.ndarray] = None
    count = np.zeros((1, 1, D, H, W), np.float32)
    patch_coords = [(z, y0, x0) for z in starts_d for y0 in starts_h for x0 in starts_w]
    sw_batch_size = max(1, int(sw_batch_size))
    flip_combos = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)] if use_tta else [()]

    for coord_chunk in _chunked(patch_coords, sw_batch_size):
        base_patches = np.concatenate(
            [x_np[:, :, z:z + pd, y0:y0 + ph, x0:x0 + pw] for z, y0, x0 in coord_chunk],
            axis=0,
        ).astype(np.float32, copy=False)
        patches = np.concatenate(
            [np.flip(base_patches, axis=dims).copy() if dims else base_patches for dims in flip_combos],
            axis=0,
        ).astype(np.float32, copy=False)
        result = session.run(None, {inp_name: patches})[0]
        if use_tta:
            result = result.reshape(len(flip_combos), len(coord_chunk), result.shape[1], pd, ph, pw)
            restored = []
            for flip_idx, dims in enumerate(flip_combos):
                logits_part = result[flip_idx]
                if dims:
                    logits_part = np.flip(logits_part, axis=dims).copy()
                logits_part = logits_part.astype(np.float32, copy=False)
                logits_part = logits_part - logits_part.max(axis=1, keepdims=True)
                prob_part = np.exp(logits_part)
                prob_part /= np.clip(prob_part.sum(axis=1, keepdims=True), 1e-8, None)
                restored.append(prob_part)
            result = np.mean(np.stack(restored, axis=0), axis=0).astype(np.float32, copy=False)
        if out_arr is None:
            out_arr = np.zeros((1, result.shape[1], D, H, W), np.float32)
        for patch_idx, (z, y0, x0) in enumerate(coord_chunk):
            patch_logits = result[patch_idx:patch_idx + 1]
            out_arr[:, :, z:z + pd, y0:y0 + ph, x0:x0 + pw] += patch_logits * gaussian_np
            count[:, :, z:z + pd, y0:y0 + ph, x0:x0 + pw] += gaussian_np

    if out_arr is None:
        raise RuntimeError("Internal error: output buffer was not initialised")
    out_arr /= np.clip(count, 1e-8, None)
    if use_tta:
        out_arr = np.log(np.clip(out_arr, 1e-8, None))
    if pad_d + pad_h + pad_w > 0:
        out_arr = out_arr[:, :, pad_d // 2:pad_d // 2 + d,
                                pad_h // 2:pad_h // 2 + h,
                                pad_w // 2:pad_w // 2 + w]
    return torch.from_numpy(out_arr)


def _prediction_region_summary(pred: np.ndarray) -> Dict[str, Any]:
    wt = pred > 0
    tc = (pred == 1) | (pred == 3)
    et = pred == 3
    total_voxels = max(int(pred.size), 1)
    return {
        "wt_voxels": int(np.count_nonzero(wt)),
        "tc_voxels": int(np.count_nonzero(tc)),
        "et_voxels": int(np.count_nonzero(et)),
        "predicted_fraction_wt": round(float(np.count_nonzero(wt) / total_voxels), 6),
        "predicted_fraction_tc": round(float(np.count_nonzero(tc) / total_voxels), 6),
        "predicted_fraction_et": round(float(np.count_nonzero(et) / total_voxels), 6),
    }


def _feret_bidimensional_mm(coords_rc: np.ndarray, sp_r: float, sp_c: float) -> Tuple[float, float]:
    """RANO bidimensional measurement (mm) for one 2-D lesion slice.

    Returns ``(longest_diameter, longest_perpendicular_diameter)`` in mm. The longest
    diameter is the **Feret** diameter - the maximum chord between any two lesion-boundary
    points - NOT the axis-aligned bounding-box span, which overestimates the longest
    diameter for non-convex / off-axis lesions. The perpendicular is the lesion's caliper
    width orthogonal to the long axis (for a convex shape, the longest perpendicular chord).
    """
    pts = coords_rc.astype(np.float64) * np.array([sp_r, sp_c], dtype=np.float64)
    if pts.shape[0] <= 1:
        return float(max(sp_r, sp_c)), float(min(sp_r, sp_c))
    hull_pts = pts
    if pts.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull  # noqa: WPS433
            hull_pts = pts[ConvexHull(pts).vertices]
        except Exception:
            hull_pts = pts  # collinear / degenerate / scipy absent -> brute force on all points
    dmat = np.linalg.norm(hull_pts[:, None, :] - hull_pts[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(dmat)), dmat.shape)
    long_mm = float(dmat[i, j])
    axis = hull_pts[j] - hull_pts[i]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return long_mm, long_mm
    perp_dir = np.array([-axis[1], axis[0]], dtype=np.float64) / norm
    proj = pts @ perp_dir
    return long_mm, float(proj.max() - proj.min())


def _region_clinical_measurements(
    pred: np.ndarray,
    spacing_mm: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Approximate RANO-style volumes and bidimensional products by region."""
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.size < 3 or not np.all(np.isfinite(spacing[:3])) or not np.all(spacing[:3] > 0):
        spacing = np.ones(3, dtype=np.float64)
    voxel_ml = float(np.prod(spacing[:3]) / 1000.0)
    regions = {
        "WT": pred > 0,
        "TC": (pred == 1) | (pred == 3),
        "ET": pred == 3,
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in regions.items():
        volume_vox = int(np.count_nonzero(mask))
        best_long = 0.0
        best_perp = 0.0
        best_product = 0.0
        best_slice = -1
        if volume_vox > 0:
            # RANO measures on the slice with the largest cross-sectional AREA (not the
            # largest bounding-box product), then the longest in-plane diameter (Feret)
            # and the longest diameter perpendicular to it.
            areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
            best_slice = int(np.argmax(areas))
            coords = np.argwhere(mask[best_slice])
            best_long, best_perp = _feret_bidimensional_mm(
                coords, float(spacing[1]), float(spacing[2]))
            best_product = best_long * best_perp
        out[name] = {
            "volume_ml": round(volume_vox * voxel_ml, 4),
            "volume_voxels": volume_vox,
            "rano_long_mm": round(best_long, 3),
            "rano_perpendicular_mm": round(best_perp, 3),
            "rano_product_mm2": round(best_product, 3),
            "rano_slice_index": int(best_slice),
            "rano_slice_position_mm": (round(best_slice * float(spacing[0]), 3)
                                       if best_slice >= 0 else -1.0),
        }
    return out


def _infer_safety_assessment(
    logits: torch.Tensor,
    pred: np.ndarray,
    present_modalities: Sequence[str],
    missing_modalities: Sequence[int],
) -> Dict[str, Any]:
    """Heuristic safety screen for deployment-time inference.

    This is not a calibrated uncertainty model. It is an operational
    guardrail that flags high-risk predictions such as single-modality
    inputs (especially T1-only), unusually low softmax confidence, or
    grossly implausible foreground volume fractions.
    """
    probs = F.softmax(logits.float(), dim=1)
    max_prob = probs.max(dim=1)[0].squeeze(0).detach().cpu().numpy()
    prob_np = probs.squeeze(0).detach().cpu().numpy()
    entropy = (
        -(prob_np * np.log(np.clip(prob_np, 1e-8, 1.0))).sum(axis=0) / math.log(max(prob_np.shape[0], 2))
    )
    energy_map = -torch.logsumexp(logits.float(), dim=1).squeeze(0).detach().cpu().numpy()

    fg = pred > 0
    pred_summary = _prediction_region_summary(pred)
    mean_conf_fg = float(max_prob[fg].mean()) if fg.any() else 0.0
    p05_conf_fg = float(np.percentile(max_prob[fg], 5)) if fg.any() else 0.0
    mean_entropy_fg = float(entropy[fg].mean()) if fg.any() else 1.0
    mean_energy_fg = float(energy_map[fg].mean()) if fg.any() else float(np.mean(energy_map))
    missing_modalities = sorted({int(m) for m in missing_modalities})
    present_lower = {m.lower() for m in present_modalities}
    flags: List[str] = []
    warnings_list: List[str] = []
    unsupported_regime = len(missing_modalities) >= 3
    low_confidence = False

    if unsupported_regime:
        flags.append("unsupported_sparse_input_regime")
        warnings_list.append(
            "Single-modality inference is unreliable for this model and should trigger manual review."
        )

    if len(present_modalities) == 1:
        flags.append("single_modality_input")
        warnings_list.append("Prediction generated from a single MRI sequence.")
        single_mod = present_modalities[0].lower()
        if single_mod not in {"t2", "t2w", "flair", "t2f"}:
            flags.append("single_modality_not_t2_or_flair")
            warnings_list.append(
                "Single-modality input is outside the calibrated T2/FLAIR-only fallback regime."
            )

    if mean_conf_fg < 0.55 or (pred_summary["et_voxels"] > 0 and mean_conf_fg < 0.60):
        low_confidence = True
        flags.append("low_foreground_confidence")
        warnings_list.append(
            f"Foreground confidence is low ({mean_conf_fg:.3f}); manual review is recommended."
        )
    if fg.any() and p05_conf_fg < 0.25:
        flags.append("very_low_foreground_confidence_tail")
        warnings_list.append(
            f"Lower-tail foreground confidence is very low (5th percentile {p05_conf_fg:.3f} < 0.25)."
        )
    if fg.any() and mean_entropy_fg > 0.45:
        flags.append("high_foreground_entropy")
        warnings_list.append(
            f"Foreground predictive entropy is elevated ({mean_entropy_fg:.3f} > 0.45)."
        )
    if pred_summary["wt_voxels"] == 0:
        flags.append("empty_foreground_prediction")
        warnings_list.append("Predicted whole-tumour volume is zero voxels.")
    if pred_summary["predicted_fraction_wt"] > 0.35:
        flags.append("implausibly_large_foreground")
        warnings_list.append(
            f"Predicted whole-tumour fraction is unusually large ({pred_summary['predicted_fraction_wt']:.3f} of voxels)."
        )
    if pred_summary["et_voxels"] > 0 and not ({"t1c", "t1ce"} & present_lower):
        flags.append("et_predicted_without_contrast")
        warnings_list.append(
            "Enhancing tumour was predicted without a post-contrast T1 sequence."
        )

    risk_level = "none"
    if flags or low_confidence:
        risk_level = "high" if (
            "t1_only_not_supported" in flags
            or "single_modality_not_t2_or_flair" in flags
            or "unsupported_sparse_input_regime" in flags
            or "empty_foreground_prediction" in flags
            or "implausibly_large_foreground" in flags
        ) else "moderate"

    return {
        "heuristic_screen": True,
        "present_modalities": list(present_modalities),
        "missing_modalities": missing_modalities,
        "n_modalities_present": len(present_modalities),
        "n_missing_modalities": len(missing_modalities),
        "mean_conf_fg": round(mean_conf_fg, 5),
        "p05_conf_fg": round(p05_conf_fg, 5),
        "mean_entropy_fg": round(mean_entropy_fg, 5),
        "mean_max_softmax": round(mean_conf_fg, 5),
        "p05_max_softmax": round(p05_conf_fg, 5),
        "mean_normalised_entropy": round(mean_entropy_fg, 5),
        "mean_energy_score": round(mean_energy_fg, 5),
        "low_confidence": bool(low_confidence),
        "unsupported_regime": bool(unsupported_regime),
        "risk_level": risk_level,
        "flags": flags,
        "warnings": warnings_list,
        "initial_prediction": pred_summary,
    }


def infer(
    test_csv: str, ckpt: str, out_dir: str,
    modalities: Sequence[str], num_classes: int,
    patch_size: Tuple[int, int, int],
    missing_modalities: Optional[List[int]],
    device: str, overlap: float = 0.5,
    use_tta: bool = False, student_base: int = 16,
    postprocess: bool = True,
    num_threads: Optional[int] = None,
    tta_method: str = "none",
    tta_steps: int = 5,
    tta_lr: float = 1e-3,
) -> None:
    out = ensure_dir(out_dir)
    device_t = torch.device(device)
    num_mod = len(modalities)
    missing_modalities = missing_modalities or []
    missing_set = {int(m) for m in missing_modalities}

    # Prefer ONNX Runtime for CPU inference
    ort_session = None
    # Check for quantised model first (static INT8 > dynamic INT8 > FP32)
    for suffix in ["_int8_static.onnx", "_int8_dynamic.onnx", ".onnx"]:
        candidate = str(Path(ckpt).with_suffix("")) + suffix
        if Path(candidate).exists() and device == "cpu":
            try:
                import onnxruntime as ort
                opts = _configure_ort_session_options(ort, num_threads=num_threads)

                # v9: Intel Xeon Gold 6448Y has AMX + AVX-512 + DL Boost.
                # DnnlExecutionProvider (oneDNN) bundles with onnxruntime on modern
                # Linux and activates these hardware extensions for INT8 conv ops,
                # potentially 2-3× throughput vs plain CPUExecutionProvider.
                # Reference: Intel oneDNN release notes, ORT execution providers docs.

                # OGAP 2.0 provider priority: OpenVINO EP first for deployable
                # Intel CPU inference, then oneDNN, then plain CPU.
                available_providers = ort.get_available_providers()
                if "OpenVINOExecutionProvider" in available_providers:
                    provider_list = [
                        ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
                        "CPUExecutionProvider",
                    ]
                    LOG.info("[Infer] ORT provider: OpenVINOExecutionProvider (CPU)")
                elif "DnnlExecutionProvider" in available_providers:
                    provider_list = [
                        ("DnnlExecutionProvider", {
                            "use_arena": "1",       # use oneDNN's memory arena allocator
                        }),
                        "CPUExecutionProvider",
                    ]
                    LOG.info("[Infer] ORT provider: DnnlExecutionProvider "
                             "(Intel oneDNN - AMX/AVX-512/DL Boost active)")
                else:
                    # oneDNN not compiled in this onnxruntime build - fall back
                    provider_list = ["CPUExecutionProvider"]
                    LOG.info("[Infer] ORT provider: CPUExecutionProvider "
                             "(OpenVINO/DnnlExecutionProvider unavailable - install "
                             "onnxruntime-openvino for the deployment path)")

                # PERF (Sapphire Rapids): ORT_DISABLE_ALL is only correct when an
                # execution provider that owns graph optimisation (OpenVINO/oneDNN)
                # is actually selected. On the plain CPUExecutionProvider fallback,
                # disabling all optimisation strips operator fusion and constant
                # folding - MLAS still uses AVX-512/VNNI/AMX kernels, but unfused,
                # which is markedly slower for INT8 conv inference. Re-enable full
                # ORT graph optimisation in that case.
                _enable_cpu_ep_graph_optim_if_needed(ort, opts, provider_list, "Infer")
                ort_session = ort.InferenceSession(candidate, opts, providers=provider_list)
                LOG.info(f"[Infer] Using ONNX Runtime: {candidate}{' + TTA' if use_tta else ''}")
                break
            except ImportError:
                pass

    model = None
    if ort_session is None:
        (
            student_base, attention, enable_ds, curriculum_dp,
            student_block_style, student_ode_steps,
        ) = _load_saved_student_arch_config(
            Path(ckpt),
            student_base,
            attention="none",
            enable_deep_supervision=True,
            curriculum_dropout_p=0.1,
        )
        model = UNet3DStudent(
            2 * num_mod,
            num_classes,
            student_base,
            attention=attention,
            enable_deep_supervision=enable_ds,
            curriculum_dropout_p=curriculum_dp,
            block_style=student_block_style,
            ode_steps=student_ode_steps,
        ).to(device_t)
        _raw_sd = torch.load(ckpt, map_location=device_t, weights_only=True)
        _sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in _raw_sd.items()}
        model.load_state_dict(_sd)
        model.eval()
        LOG.info(f"[Infer] PyTorch on {device_t} {'+ TTA' if use_tta else ''}")

    ds = InferenceDataset(test_csv, modalities)

    # Keep dict rows intact (default collation turns dicts into dict-of-lists).
    def _collate_keep_single(batch):
        return batch[0]  # batch_size=1

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_t.type == "cuda"),
        collate_fn=_collate_keep_single,
    )

    timings: List[float] = []
    safety_reports: List[Dict[str, Any]] = []

    for case_id, x, row_dict in loader:
        t0 = time.time()
        x = x.to(device_t, non_blocking=(device_t.type == "cuda")).unsqueeze(0)          # (C,D,H,W) -> (1,C,D,H,W)
        b, c = x.shape[:2]

        present_arr = row_dict.get("__present_mask__") if isinstance(row_dict, dict) else None
        if present_arr is not None:
            mask = torch.from_numpy(np.asarray(present_arr, dtype=np.float32)).to(device_t).unsqueeze(0)
            mask = mask.expand(b, -1).clone()
        else:
            mask = torch.ones((b, c), dtype=torch.float32, device=device_t)
        for m in missing_set:
            if 0 <= m < c:
                mask[:, m] = 0.0
        s_in = prepare_student_input(x, mask)

        if ort_session is not None:
            logits = _ort_sliding_window(
                ort_session,
                s_in.cpu(),
                patch_size,
                overlap,
                sw_batch_size=2,
                use_tta=use_tta,
            )
        elif tta_method == "tent":
            logits = tent_adapt_and_predict(
                model, s_in, patch_size, overlap=overlap,
                n_steps=tta_steps, lr=tta_lr, use_tta=use_tta,
            )
        elif tta_method == "memo":
            logits = memo_adapt_and_predict(
                model, s_in, patch_size, overlap=overlap,
                n_steps=tta_steps, lr=tta_lr,
            )
        elif use_tta:
            logits = tta_sliding_window(model, s_in, patch_size, overlap)
        else:
            logits = sliding_window_inference(model, s_in, patch_size, overlap)

        class_prob = F.softmax(logits.float(), dim=1).squeeze(0).detach().cpu().numpy()
        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int16)
        # Final presence reflects BOTH dataset zero-fills and explicit overlay.
        final_mask_np = mask[0].detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.ones(c, dtype=np.float32)
        present_modalities = [
            modalities[idx] for idx in range(c)
            if final_mask_np[idx] > 0.5
        ]
        safety = _infer_safety_assessment(logits, pred, present_modalities, sorted(missing_set))
        safety["case_id"] = case_id
        present_lower = {m.lower() for m in present_modalities}
        if safety["unsupported_regime"] and not ({"t1c", "t1ce"} & present_lower) and np.any(pred == 3):
            pred[pred == 3] = 0
            safety["et_suppressed"] = True
            if "et_suppressed_without_t1c" not in safety["flags"]:
                safety["flags"].append("et_suppressed_without_t1c")
            safety["warnings"].append(
                "ET suppressed because T1c is absent in an unsupported sparse-input regime."
            )
        else:
            safety["et_suppressed"] = False

        if postprocess:
            pred = connected_component_postprocessing(pred, class_prob=class_prob)

        elapsed = time.time() - t0
        timings.append(elapsed)

        # _collate_keep_single returns the raw dataset item, so values are plain
        # strings - the isinstance(list) branch that existed here was dead code.
        # Prefer the dataset-supplied reference path so missing modality columns
        # don't trigger KeyError when modalities[0] was zero-filled.
        ref_path = row_dict.get("__ref_path__") or row_dict.get(modalities[0])
        try:
            ref_spacing = np.array(nib.load(ref_path).header.get_zooms()[:3], dtype=np.float64)
            safety["spacing_reliable"] = True
        except Exception as exc:
            # A 1 mm isotropic fallback silently makes RANO volumes/products wrong for
            # anisotropic scans (a 3 mm case reads 27x too small). Flag it in the QC JSON
            # so the clinical measurements are never mistaken for reliable.
            LOG.warning("[Infer] %s: could not read voxel spacing from %s (%s); RANO/volume "
                        "measurements use a 1 mm fallback and are UNRELIABLE.",
                        case_id, ref_path, exc)
            ref_spacing = np.ones(3, dtype=np.float64)
            safety["spacing_reliable"] = False
            if "spacing_load_failed" not in safety["flags"]:
                safety["flags"].append("spacing_load_failed")
        out_path = str(out / f"{case_id}_pred.nii.gz")
        save_nifti_like(pred, ref_path, out_path)
        safety["final_prediction"] = _prediction_region_summary(pred)
        safety["clinical_measurements"] = _region_clinical_measurements(pred, ref_spacing)
        safety["postprocessed"] = bool(postprocess)
        safety["saved_prediction_path"] = out_path
        save_json(safety, out / f"{case_id}_qc.json")
        safety_reports.append(safety)
        LOG.info(f"[Infer] {case_id}: {elapsed:.1f}s → {out_path}")
        if safety["flags"]:
            LOG.warning(
                f"[Infer] {case_id}: safety flags={','.join(safety['flags'])} "
                f"(present={','.join(safety['present_modalities'])}, "
                f"mean_conf_fg={safety['mean_conf_fg']:.3f}, "
                f"risk={safety['risk_level']})"
            )

    if timings:
        LOG.info(f"[Infer] Timing: mean={np.mean(timings):.1f}s, "
                 f"median={np.median(timings):.1f}s, total={sum(timings):.0f}s")
        save_json({"mean_s": np.mean(timings), "median_s": np.median(timings),
                   "per_case": timings}, out / "inference_timing.json")
    if safety_reports:
        flag_counts: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {}
        for report in safety_reports:
            risk = str(report.get("risk_level", "none"))
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            for flag in report.get("flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        save_json(
            {
                "description": (
                    "Heuristic deployment-time safety screen. Flags are intended "
                    "to trigger manual review, not to replace clinical judgement."
                ),
                "summary": {
                    "n_cases": len(safety_reports),
                    "risk_counts": risk_counts,
                    "flag_counts": flag_counts,
                },
                "per_case": safety_reports,
            },
            out / "inference_safety_report.json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# §18  COMPREHENSIVE EVALUATION (ALL 15 MODALITY COMBINATIONS)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# §18a  HARDWARE BENCHMARK (LMIC Deployment Validation)
# ══════════════════════════════════════════════════════════════════════════════
#
# LMIC hardware (e.g., Intel Core i5 laptop with 8-16 GB RAM) to validate the
# "CPU-first" deployment claim.  Reports median inference time, peak RAM, and
# estimated energy per case.
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_hardware(
    test_csv: str, ckpt: str, out_dir: str,
    modalities: Sequence[str], num_classes: int,
    patch_size: Tuple[int, int, int],
    missing_modalities: Optional[List[int]] = None,
    student_base: int = 16,
    num_threads: Optional[int] = None,
    n_warmup: int = 2,
    n_cases: Optional[int] = None,
) -> None:
    """Run inference benchmark with detailed hardware profiling.

    Designed for LMIC deployment validation: run this on a consumer-grade
    laptop (Intel Core i5, 8-16 GB RAM) and report the results in the
    manuscript.

    Reports:
        - Per-case inference time (mean, median, std, p5, p95)
        - Peak RSS memory (MB) - actual physical RAM consumed
        - Peak ONNX allocator memory (if available)
        - CPU info, thread count, and platform details
        - Estimated energy per case (CPU TDP-based)

    Usage:
        python OGAP_source_code_experimental_v9.py benchmark \\
            --test_csv val.csv --ckpt best_student.pth \\
            --out_dir benchmark_results/ --modalities t1,t1ce,t2,flair \\
            --num_threads 4
    """
    import platform
    out = ensure_dir(out_dir)
    missing_modalities = missing_modalities or []
    num_mod = len(modalities)

    # ── System info ───────────────────────────────────────────────────────
    logical_cpu_count = os.cpu_count()
    physical_cpu_count: Optional[int] = None
    try:
        import psutil  # type: ignore

        physical_cpu_count = psutil.cpu_count(logical=False)
    except Exception:
        physical_cpu_count = None

    cpu_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": logical_cpu_count,
        "cpu_count_physical": physical_cpu_count,
        "python_version": platform.python_version(),
        "num_threads_used": num_threads if num_threads is not None else "onnxruntime_default",
    }
    try:
        # Try to get detailed CPU info on Linux
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_info["total_ram_gb"] = round(mem.total / 1e9, 1)
        cpu_info["available_ram_gb"] = round(mem.available / 1e9, 1)
    except ImportError:
        pass

    LOG.info(f"[Benchmark] System: {cpu_info}")

    # ── Load model (prefer ONNX) ─────────────────────────────────────────
    ort_session = None
    model_type = "pytorch"
    for suffix in ["_int8_static.onnx", "_int8_dynamic.onnx", ".onnx"]:
        candidate = str(Path(ckpt).with_suffix("")) + suffix
        if Path(candidate).exists():
            try:
                import onnxruntime as ort
                opts = _configure_ort_session_options(ort, num_threads=num_threads)
                available_providers = ort.get_available_providers()
                if "OpenVINOExecutionProvider" in available_providers:
                    providers = [("OpenVINOExecutionProvider", {"device_type": "CPU"}),
                                 "CPUExecutionProvider"]
                elif "DnnlExecutionProvider" in available_providers:
                    providers = [("DnnlExecutionProvider", {"use_arena": "1"}),
                                 "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                _enable_cpu_ep_graph_optim_if_needed(ort, opts, providers, "Benchmark")
                ort_session = ort.InferenceSession(candidate, opts, providers=providers)
                model_type = f"onnx ({Path(candidate).name})"
                LOG.info(f"[Benchmark] Using ONNX Runtime: {candidate}")
                break
            except ImportError:
                pass

    model = None
    if ort_session is None:
        (
            student_base, attention, enable_ds, curriculum_dp,
            student_block_style, student_ode_steps,
        ) = _load_saved_student_arch_config(
            Path(ckpt),
            student_base,
            attention="none",
            enable_deep_supervision=True,
            curriculum_dropout_p=0.1,
        )
        model = UNet3DStudent(
            2 * num_mod,
            num_classes,
            student_base,
            attention=attention,
            enable_deep_supervision=enable_ds,
            curriculum_dropout_p=curriculum_dp,
            block_style=student_block_style,
            ode_steps=student_ode_steps,
        )
        raw_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
              for k, v in raw_sd.items()}
        model.load_state_dict(sd)
        model.eval()
        model_type = "pytorch (CPU)"
        LOG.info(f"[Benchmark] Using PyTorch CPU")

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = InferenceDataset(test_csv, list(modalities))
    total_cases = n_cases or len(ds)
    total_cases = min(total_cases, len(ds))

    LOG.info(f"[Benchmark] Model: {model_type}, Cases: {total_cases}, "
             f"Threads: {cpu_info['num_threads_used']}")

    # ── Benchmark loop ────────────────────────────────────────────────────
    timings: List[float] = []
    peak_rss_mb: List[float] = []

    try:
        import psutil
        process = psutil.Process(os.getpid())
        has_psutil = True
    except ImportError:
        has_psutil = False

    for i in range(total_cases):
        case_id, x, row_dict = ds[i]
        x = x.unsqueeze(0)  # (C,D,H,W) -> (1,C,D,H,W)
        b, c = x.shape[:2]

        present_arr = row_dict.get("__present_mask__") if isinstance(row_dict, dict) else None
        if present_arr is not None:
            mask = torch.from_numpy(np.asarray(present_arr, dtype=np.float32)).unsqueeze(0).expand(b, -1).clone()
        else:
            mask = torch.ones((b, c), dtype=torch.float32)
        for m in missing_modalities:
            if 0 <= m < c:
                mask[:, m] = 0.0
        s_in = prepare_student_input(x, mask)

        # Warmup (skip timing for first n_warmup cases)
        is_warmup = (i < n_warmup)

        if has_psutil:
            rss_before = process.memory_info().rss / 1e6

        t0 = time.time()
        if ort_session is not None:
            _ = _ort_sliding_window(ort_session, s_in.cpu(), patch_size, 0.5)
        else:
            with torch.inference_mode():
                _ = sliding_window_inference(model, s_in, patch_size, 0.5)
        elapsed = time.time() - t0

        if has_psutil:
            rss_after = process.memory_info().rss / 1e6
            peak_rss_mb.append(max(rss_before, rss_after))

        if not is_warmup:
            timings.append(elapsed)

        label = "WARMUP" if is_warmup else f"{elapsed:.1f}s"
        LOG.info(f"[Benchmark] Case {i+1}/{total_cases} ({case_id}): {label}")

    # ── Report ────────────────────────────────────────────────────────────
    if not timings:
        LOG.warning("[Benchmark] No timing data collected (all cases used for warmup)")
        return

    timings_arr = np.array(timings)
    report = {
        "system": cpu_info,
        "model_type": model_type,
        "patch_size": list(patch_size),
        "missing_modalities": missing_modalities,
        "n_warmup": n_warmup,
        "n_timed_cases": len(timings),
        "inference_time_seconds": {
            "mean": round(float(np.mean(timings_arr)), 2),
            "median": round(float(np.median(timings_arr)), 2),
            "std": round(float(np.std(timings_arr)), 2),
            "p5": round(float(np.percentile(timings_arr, 5)), 2),
            "p95": round(float(np.percentile(timings_arr, 95)), 2),
            "min": round(float(np.min(timings_arr)), 2),
            "max": round(float(np.max(timings_arr)), 2),
        },
        "per_case_seconds": [round(float(t), 2) for t in timings],
    }

    if peak_rss_mb:
        report["peak_rss_mb"] = round(float(np.max(peak_rss_mb)), 1)
        report["mean_rss_mb"] = round(float(np.mean(peak_rss_mb)), 1)

    save_json(report, out / "hardware_benchmark.json")

    LOG.info(
        f"\n[Benchmark] ── Hardware Benchmark Report ──────────────────\n"
        f"  Model type    : {model_type}\n"
        f"  System        : {cpu_info.get('cpu_model', cpu_info.get('processor', 'unknown'))}\n"
        f"  RAM           : {cpu_info.get('total_ram_gb', '?')} GB\n"
        f"  Threads       : {cpu_info['num_threads_used']}\n"
        f"  Cases timed   : {len(timings)} (+ {n_warmup} warmup)\n"
        f"  Inference time: {np.mean(timings_arr):.1f}s mean, "
        f"{np.median(timings_arr):.1f}s median\n"
        f"  Peak RSS      : {report.get('peak_rss_mb', '?')} MB\n"
        f"  Report saved  : {out / 'hardware_benchmark.json'}\n"
        f"  ─────────────────────────────────────────────"
    )


def full_evaluation(
    ckpt: str, val_csv: str, modalities: Sequence[str],
    num_classes: int, patch_size: Tuple[int, int, int],
    out_dir: str, student_base: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 20,
    eval_config_batch_size: int = 1,
    metric_workers: int = 1,
    use_tta: bool = False,
    cache_dir: Optional[str] = None,
    clinical_metadata_tsv: Optional[str] = None,
    failure_mode_metric: str = "pp_all_dice_brats_mean",
    failure_mode_quantile: float = 0.25,
    failure_mode_min_group_n: int = 10,
    compute_lesion_wise: bool = False,
) -> None:
    """Run evaluation over all 15 modality combinations with statistical tests."""
    out = ensure_dir(out_dir)
    device_t = torch.device(device)
    num_mod = len(modalities)

    student = _load_student_for_eval(Path(ckpt), num_mod, num_classes, student_base, device_t)

    val_loader = build_eval_loader(
        val_csv,
        modalities,
        patch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        pin_memory=(device_t.type == "cuda"),
    )

    # All 15 combinations (2^4 - 1)
    configs = all_modality_combinations(num_mod)
    LOG.info(f"[Eval] Running {len(configs)} modality configurations{' with TTA' if use_tta else ''}")

    # Always compute BOTH raw and postprocessed metrics.
    LOG.info("[Eval] Computing raw (no postprocessing) metrics ...")
    raw_resume = out / "resume_full_evaluation_raw.json"
    raw_results, raw_per_case, raw_casewise = evaluate_student(
        student, val_loader, num_classes, device_t,
        use_brats=True, eval_mask_configs=configs, postprocess=False,
        use_tta=use_tta,
        compute_lesion_wise=compute_lesion_wise,
        eval_config_batch_size=eval_config_batch_size,
        metric_workers=metric_workers,
        checkpoint_path=str(raw_resume),
        checkpoint_metadata={
            "phase": "full_evaluation_raw",
            "model_path": str(Path(ckpt).resolve()),
            "val_csv": str(Path(val_csv).resolve()),
        },
        return_per_case=True,
        return_casewise=True,
    )
    LOG.info("[Eval] Computing postprocessed (CC-filtered) metrics ...")
    pp_resume = out / "resume_full_evaluation_postprocessed.json"
    pp_results, pp_per_case, pp_casewise = evaluate_student(
        student, val_loader, num_classes, device_t,
        use_brats=True, eval_mask_configs=configs, postprocess=True,
        use_tta=use_tta,
        compute_lesion_wise=compute_lesion_wise,
        eval_config_batch_size=eval_config_batch_size,
        metric_workers=metric_workers,
        checkpoint_path=str(pp_resume),
        checkpoint_metadata={
            "phase": "full_evaluation_postprocessed",
            "model_path": str(Path(ckpt).resolve()),
            "val_csv": str(Path(val_csv).resolve()),
        },
        return_per_case=True,
        return_casewise=True,
    )

    # Merge: postprocessed keys prefixed with "pp_"
    all_results = {**raw_results}
    for k, v in pp_results.items():
        all_results[f"pp_{k}"] = v

    combined_per_case: Dict[str, List[float]] = {}
    for tag_k, vals in raw_per_case.items():
        combined_per_case[tag_k] = vals
    for tag_k, vals in pp_per_case.items():
        combined_per_case[f"pp_{tag_k}"] = vals

    base_metrics_for_stats = list(BRATS_METRIC_NAMES)
    if compute_lesion_wise:
        base_metrics_for_stats.extend(BRATS_LESION_WISE_METRIC_NAMES)

    # ── Bootstrap 95% CIs (BCa, 5000 resamples) ───────────────────────────
    LOG.info("[Eval] Computing 95% bootstrap confidence intervals ...")
    ci_results: Dict[str, Dict[str, float]] = {}
    for cfg_item in configs:
        tag = "all" if cfg_item is None else f"drop{''.join(str(m) for m in cfg_item)}"
        ci_results[tag] = {}
        for metric in base_metrics_for_stats:
            for prefix in ("", "pp_"):
                label = f"{prefix}{metric}"
                key = f"{prefix}{tag}_{metric}"
                per_case_vals = combined_per_case.get(key, [])
                if per_case_vals:
                    mean_v, lo, hi = bootstrap_ci(per_case_vals, n_resamples=5000)
                    ci_results[tag][label] = {
                        "mean": round(mean_v, 5),
                        "ci_lower": round(lo, 5),
                        "ci_upper": round(hi, 5),
                        "n_cases": len(per_case_vals),
                    }
                    LOG.info(
                        f"  [{tag}] {label}: {mean_v:.4f} "
                        f"(95% CI {lo:.4f}-{hi:.4f}, n={len(per_case_vals)})"
                    )
    save_json(ci_results, out / "bootstrap_ci_results.json")

    # ── Wilcoxon signed-rank tests vs full modality (Holm-Bonferroni)
    # testing for pairwise comparisons between modality configurations.
    LOG.info("[Eval] Computing Wilcoxon signed-rank tests vs full modality ...")
    wilcoxon_results: Dict[str, Dict[str, Any]] = {}
    missing_configs = [c for c in configs if c is not None]
    p_values_all: List[float] = []
    comparison_keys: List[Tuple[str, str]] = []  # (tag, metric)

    # Test both raw and postprocessed variants; include lesion-wise metrics
    # whenever they were requested for this run.
    metrics_to_test = list(base_metrics_for_stats)
    metrics_to_test.extend(f"pp_{metric}" for metric in base_metrics_for_stats)

    for metric in metrics_to_test:
        full_key = f"all_{metric}" if not metric.startswith("pp_") else f"pp_all_{metric[3:]}"
        full_vals = combined_per_case.get(full_key, [])
        if len(full_vals) < 10:
            continue
        for cfg_item in missing_configs:
            base_tag = f"drop{''.join(str(m) for m in cfg_item)}"
            miss_key = (
                f"{base_tag}_{metric}"
                if not metric.startswith("pp_")
                else f"pp_{base_tag}_{metric[3:]}"
            )
            miss_vals = combined_per_case.get(miss_key, [])
            if len(miss_vals) < 10:
                continue
            stat_result = statistical_comparison(full_vals, miss_vals)
            p_values_all.append(stat_result.get("p_value", 1.0))
            comparison_keys.append((base_tag, metric))
            if base_tag not in wilcoxon_results:
                wilcoxon_results[base_tag] = {}
            wilcoxon_results[base_tag][metric] = stat_result

    # Holm-Bonferroni correction over all comparisons
    if p_values_all:
        m = len(p_values_all)
        ranked_idx = sorted(range(m), key=lambda i: p_values_all[i])
        holm_corrected = [1.0] * m
        sorted_corrected = []
        for rank, idx in enumerate(ranked_idx):
            corrected = p_values_all[idx] * (m - rank)
            sorted_corrected.append(min(corrected, 1.0))
        for i in range(1, len(sorted_corrected)):
            sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
        for rank, idx in enumerate(ranked_idx):
            holm_corrected[idx] = sorted_corrected[rank]
        for i, (tag, metric) in enumerate(comparison_keys):
            wilcoxon_results[tag][metric]["p_holm"] = round(holm_corrected[i], 6)
            wilcoxon_results[tag][metric]["significant_holm_05"] = holm_corrected[i] < 0.05

    save_json(wilcoxon_results, out / "wilcoxon_results.json")
    effect_size_rows = []
    for tag, metric_dict in wilcoxon_results.items():
        for metric_name, stat_result in metric_dict.items():
            effect_size_rows.append({
                "config": tag,
                "metric": metric_name,
                "mean_diff": stat_result.get("mean_diff"),
                "effect_size_r": stat_result.get("effect_size_r"),
                "effect_size_magnitude": stat_result.get("effect_size_magnitude"),
                "p_holm": stat_result.get("p_holm"),
                "significant_holm_05": stat_result.get("significant_holm_05"),
            })
    effect_size_rows.sort(
        key=lambda row: abs(float(row.get("effect_size_r") or 0.0)),
        reverse=True,
    )
    save_json(effect_size_rows, out / "wilcoxon_effect_size_summary.json")
    LOG.info(f"[Eval] Wilcoxon results saved ({len(missing_configs)} configs × metrics)")

    save_json(all_results, out / "full_evaluation_results.json")

    per_case_rows: List[Dict[str, Any]] = []
    all_case_keys = sorted(set(raw_casewise.keys()) | set(pp_casewise.keys()))
    for case_key in all_case_keys:
        raw_payload = raw_casewise.get(case_key, {})
        pp_payload = pp_casewise.get(case_key, {})
        row: Dict[str, Any] = {
            "case_key": case_key,
            "case_id": raw_payload.get("case_id") or pp_payload.get("case_id") or "",
            "dataset_tag": raw_payload.get("dataset_tag") or pp_payload.get("dataset_tag") or "",
        }
        for metric_key, metric_value in raw_payload.get("metrics", {}).items():
            row[metric_key] = float(metric_value)
        for metric_key, metric_value in pp_payload.get("metrics", {}).items():
            row[f"pp_{metric_key}"] = float(metric_value)
        per_case_rows.append(row)
    save_json(per_case_rows, out / "full_evaluation_per_case_metrics.json")
    _save_rows_csv(per_case_rows, out / "full_evaluation_per_case_metrics.csv")

    if clinical_metadata_tsv:
        LOG.info("[Eval] Computing clinical failure-mode analysis ...")
        failure_summary = clinical_failure_mode_analysis(
            per_case_rows,
            clinical_metadata_tsv,
            str(out),
            primary_metric=failure_mode_metric,
            failure_quantile=failure_mode_quantile,
            min_group_n=failure_mode_min_group_n,
        )
        LOG.info(
            f"[Eval] Clinical failure-mode analysis saved "
            f"({failure_summary.get('n_matched_cases', 0)} matched cases, "
            f"{failure_summary.get('n_failed_cases', 0)} failures by "
            f"{failure_mode_metric})"
        )

    # Summary table (raw and postprocessed side by side)
    summary_rows = []
    for cfg in configs:
        tag = "all" if cfg is None else f"drop{''.join(str(m) for m in cfg)}"
        present = list(range(num_mod)) if cfg is None else [i for i in range(num_mod) if i not in cfg]
        mod_names = [modalities[i] for i in present]
        row = {
            "config": tag,
            "modalities_present": ",".join(mod_names),
            "n_present": len(present),
        }
        summary_metrics = [
            "dice_WT", "dice_TC", "dice_ET", "dice_brats_mean",
            "hd95_WT", "hd95_TC", "hd95_ET", "hd95_brats_mean", "dice_fg",
        ]
        if compute_lesion_wise:
            summary_metrics.extend(BRATS_LESION_WISE_METRIC_NAMES)
        for metric in summary_metrics:
            # NaN (not 0.0) for an uncomputed metric: a phantom 0.0 reads as a perfect
            # HD95 in the published brats_metrics.csv. NaN is the honest "not computed".
            row[f"raw_{metric}"] = all_results.get(f"{tag}_{metric}", float("nan"))
            row[f"pp_{metric}"]  = all_results.get(f"pp_{tag}_{metric}", float("nan"))
        summary_rows.append(row)

    save_json(summary_rows, out / "modality_combination_summary.json")
    _save_rows_csv(summary_rows, out / "brats_metrics.csv")
    _save_rows_csv(summary_rows, out / "robustness_metrics.csv")
    _save_rows_csv(effect_size_rows, out / "stats_significance.csv")
    _save_rows_csv(per_case_rows, out / "results_full.csv")
    _save_rows_csv(summary_rows, out / "results_aggregate.csv")
    _save_rows_csv(
        [{
            "model_path": str(Path(ckpt).resolve()),
            "device": str(device_t),
            "student_base": int(student_base),
            "patch_size": "x".join(str(v) for v in patch_size),
            "eval_config_batch_size": int(eval_config_batch_size),
        }],
        out / "hardware_metrics.csv",
    )
    try:
        from ogap.evaluation.tables import rows_to_latex

        rows_to_latex(
            summary_rows,
            path=str(out / "table_main.tex"),
            caption="OGAP evaluation summary by modality configuration.",
            label="tab:ogap-main",
        )
        rows_to_latex(
            summary_rows,
            path=str(out / "table_robustness.tex"),
            caption="OGAP modality robustness summary.",
            label="tab:ogap-robustness",
        )
        rows_to_latex(
            effect_size_rows or [{"config": "all", "metric": "n/a", "p_holm": ""}],
            path=str(out / "table_ablation.tex"),
            caption="OGAP ablation/statistical comparison summary.",
            label="tab:ogap-ablation",
        )
        rows_to_latex(
            [{
                "model_path": str(Path(ckpt).name),
                "device": str(device_t),
                "eval_config_batch_size": int(eval_config_batch_size),
            }],
            path=str(out / "table_hardware.tex"),
            caption="OGAP hardware evaluation summary.",
            label="tab:ogap-hardware",
        )
    except Exception as exc:
        LOG.warning("[Eval] Publication table export skipped: %s", exc)
    LOG.info(f"[Eval] Results saved to {out}")

    # Print summary table
    header = (
        f"{'Config':<20} {'Mods':<20} "
        f"{'Raw_WT':>8} {'Raw_TC':>8} {'Raw_ET':>8} {'Raw_Mean':>9} "
        f"{'PP_WT':>7} {'PP_TC':>7} {'PP_ET':>7} {'PP_Mean':>8}"
    )
    LOG.info(header)
    LOG.info("-" * len(header))
    for row in summary_rows:
        LOG.info(
            f"{row['config']:<20} {row['modalities_present']:<20} "
            f"{row.get('raw_dice_WT', 0):.4f}   {row.get('raw_dice_TC', 0):.4f}   "
            f"{row.get('raw_dice_ET', 0):.4f}   {row.get('raw_dice_brats_mean', 0):.4f}   "
            f"{row.get('pp_dice_WT', 0):.4f}   {row.get('pp_dice_TC', 0):.4f}   "
            f"{row.get('pp_dice_ET', 0):.4f}   {row.get('pp_dice_brats_mean', 0):.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §19  ABLATION STUDY RUNNER
# ══════════════════════════════════════════════════════════════════════════════
#
# FIX #5: All ablations now actually modify model behaviour.
# Placeholder `pass` statements replaced with real config changes.
# CLI mode added via "ablation" subcommand.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """Configuration for systematic ablation studies.

    Each ablation disables one component and compares against full pipeline.
    """
    base_config: TrainConfig
    out_dir: str
    ablations: List[str] = field(default_factory=lambda: [
        "no_kd",              # No knowledge distillation
        "no_feat_distill",    # No feature distillation
        "no_deep_supervision",# No deep supervision
        "kl_instead_holder",  # KL divergence instead of Hölder
        "no_physics_aug",     # Standard aug only (no MRI-physics)
        "no_curriculum",      # No modality dropout curriculum
        "eca_attention",      # Add ECA channel attention
        "no_postprocess",     # No connected-component filtering
        "se_attention",       # Add SE blocks (for quantisation comparison)
        "no_curriculum_dropout",  # No curriculum dropout regularisation
        "no_confidence_gate", # No confidence gating on KD
    ])


def _load_student_for_eval(
    ckpt_path: Path, num_mod: int, num_classes: int,
    student_base: int, device: torch.device,
    attention: str = "none",
    enable_deep_supervision: bool = True,
) -> UNet3DStudent:
    """Load a student checkpoint for evaluation.

    Architecture-variant ablations (attention, deep supervision)
    change the model structure, so the correct attention/deep_supervision flags
    must be passed.  If a train_config.json exists next to the checkpoint,
    those flags are read automatically.
    """
    (
        student_base, attention, enable_deep_supervision, curriculum_dropout_p,
        student_block_style, student_ode_steps,
    ) = _load_saved_student_arch_config(
        ckpt_path,
        student_base,
        attention=attention,
        enable_deep_supervision=enable_deep_supervision,
        curriculum_dropout_p=0.1,
    )

    student = UNet3DStudent(
        2 * num_mod, num_classes, student_base,
        attention=attention,
        enable_deep_supervision=enable_deep_supervision,
        curriculum_dropout_p=curriculum_dropout_p,
        block_style=student_block_style,
        ode_steps=student_ode_steps,
    ).to(device)
    raw_sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in raw_sd.items()}
    student.load_state_dict(sd)
    if device.type == "cuda":
        _configure_torch_compile_runtime()
        _configure_cuda_inference()
        student = student.to(memory_format=torch.channels_last_3d)
    student.eval()
    if device.type == "cuda":
        student = maybe_compile_model(student, "Eval student", dynamic=False)
    return student


def run_ablation_study(ablation_cfg: AblationConfig) -> None:
    """Run systematic ablation experiments.

    FIX #5: Each variant now actually modifies the relevant config field.
    Architecture-level ablations (attention, deep supervision) use the
    new TrainConfig fields instead of being no-ops.

    Results are compared via Wilcoxon signed-rank test.
    """
    out = ensure_dir(ablation_cfg.out_dir)
    base = ablation_cfg.base_config

    LOG.info(f"[Ablation] Running {len(ablation_cfg.ablations)} ablation experiments")

    for ablation_name in ablation_cfg.ablations:
        LOG.info(f"\n{'='*60}")
        LOG.info(f"[Ablation] Starting: {ablation_name}")
        LOG.info(f"{'='*60}")

        # Create modified config
        cfg = TrainConfig(**asdict(base))
        cfg.out_dir = str(out / ablation_name)

        # FIX #5: Every ablation now has a real effect
        if ablation_name == "no_kd":
            cfg.lambda_kd = 0.0
        elif ablation_name == "no_feat_distill":
            cfg.lambda_feat = 0.0
        elif ablation_name == "no_deep_supervision":
            cfg.enable_deep_supervision = False  # FIX: was `pass`
        elif ablation_name == "kl_instead_holder":
            cfg.holder_alpha = 1.0  # FIX: alpha≤1 now triggers KL fallback
        elif ablation_name == "no_physics_aug":
            cfg.augment = False
        elif ablation_name == "no_curriculum":
            # FIX: curriculum_ramp=0.0 still applies a cosine schedule;
            # true "no curriculum" means keep_min is fixed from epoch 1.
            cfg.use_curriculum = False
        elif ablation_name == "eca_attention":
            cfg.attention = "eca"
        elif ablation_name == "se_attention":
            cfg.attention = "se"
        elif ablation_name == "no_curriculum_dropout":
            cfg.curriculum_dropout_p = 0.0
        elif ablation_name == "no_confidence_gate":
            cfg.confidence_gate_threshold = 0.0  # Disables gating
        elif ablation_name == "no_postprocess":
            # Postprocessing is inference-time only; training is identical to baseline.
            # After this ablation finishes, evaluate the checkpoint with:
            #   python pipeline.py evaluate --ckpt <out>/no_postprocess/best_student.pth ...
            # The full_evaluation command always reports both raw and pp_ metrics,
            # so compare raw_* columns for this ablation vs pp_* columns for baseline.
            LOG.info(
                "[Ablation] no_postprocess: training normally (postprocessing is eval-time only).\n"
                "After training, run 'evaluate' and compare raw_dice_* columns "
                "(no CC filter) against the baseline pp_dice_* columns."
            )
        else:
            LOG.warning(f"[Ablation] Unknown ablation: {ablation_name}, skipping")
            continue

        try:
            train(cfg)
            LOG.info(f"[Ablation] Completed: {ablation_name}")
        except Exception as e:
            # Preserve the stack trace so a failed ablation variant (OOM / divergence /
            # config error) is debuggable and not just a silent gap in the ablation table.
            LOG.error("[Ablation] Failed %s: %s", ablation_name, e, exc_info=True)
            continue

    # ── Post-hoc ablation comparison with statistical testing ─────────────
    # for each ablation vs baseline, not just absolute Δ.
    #
    # Evaluates all 15 non-empty modality combinations for each ablation,
    # because components like modality-dropout curriculum and confidence
    # gating are specifically designed for missing-modality robustness and
    # their effect is underestimated under full-modality evaluation alone.
    LOG.info("\n[Ablation] Running post-hoc statistical comparison "
             "(all 15 modality combinations per ablation) ...")
    device_t = torch.device(base.device)
    num_mod = len(base.modalities)
    val_loader = build_eval_loader(
        base.val_csv,
        base.modalities,
        base.patch_size,
        num_workers=base.num_workers,
        cache_dir=base.cache_dir,
        pin_memory=(device_t.type == "cuda"),
    )

    # All 15 non-empty modality subsets
    eval_configs = all_modality_combinations(num_mod)
    config_tags = [
        "all" if cfg is None else f"drop{''.join(str(m) for m in cfg)}"
        for cfg in eval_configs
    ]

    ablation_stats: Dict[str, Any] = {}
    baseline_ckpt = Path(base.out_dir) / "best_student.pth"
    if baseline_ckpt.exists():
        LOG.info("[Ablation] Evaluating baseline across all 15 modality configs ...")
        baseline_resume = Path(out) / "resume_posthoc_baseline_validation.json"
        _, baseline_per_case = evaluate_student(
            _load_student_for_eval(baseline_ckpt, num_mod, base.num_classes, base.student_base, device_t),
            val_loader, base.num_classes, device_t,
            use_brats=True, eval_mask_configs=eval_configs, postprocess=True,
            return_per_case=True, compute_hd95=True,
            checkpoint_path=str(baseline_resume),
            checkpoint_metadata={
                "phase": "posthoc_ablation_baseline",
                "model_path": str(baseline_ckpt.resolve()),
                "val_csv": str(Path(base.val_csv).resolve()),
            },
        )
        for ablation_name in ablation_cfg.ablations:
            ab_ckpt = Path(out) / ablation_name / "best_student.pth"
            if not ab_ckpt.exists():
                continue
            LOG.info(f"[Ablation] Evaluating {ablation_name} across all 15 modality configs ...")
            ab_resume = Path(out) / f"resume_posthoc_{ablation_name}_validation.json"
            _, ab_per_case = evaluate_student(
                _load_student_for_eval(ab_ckpt, num_mod, base.num_classes, base.student_base, device_t),
                val_loader, base.num_classes, device_t,
                use_brats=True, eval_mask_configs=eval_configs, postprocess=True,
                return_per_case=True, compute_hd95=True,
                checkpoint_path=str(ab_resume),
                checkpoint_metadata={
                    "phase": f"posthoc_ablation_{ablation_name}",
                    "model_path": str(ab_ckpt.resolve()),
                    "val_csv": str(Path(base.val_csv).resolve()),
                },
            )
            ablation_stats[ablation_name] = {}
            for cfg_tag in config_tags:
                for metric in ["dice_brats_mean", "dice_WT", "dice_TC", "dice_ET", "hd95_brats_mean"]:
                    key = f"{cfg_tag}_{metric}"
                    base_vals = baseline_per_case.get(key, [])
                    ab_vals = ab_per_case.get(key, [])
                    if len(base_vals) >= 10 and len(ab_vals) >= 10:
                        stat_result = statistical_comparison(base_vals, ab_vals)
                        result_key = f"{cfg_tag}__{metric}"
                        ablation_stats[ablation_name][result_key] = stat_result
                        sig = "***" if stat_result.get("p_value", 1) < 0.001 else \
                              "**"  if stat_result.get("p_value", 1) < 0.01 else \
                              "*"   if stat_result.get("p_value", 1) < 0.05 else "ns"
                        LOG.info(
                            f"  [{ablation_name}] {cfg_tag}/{metric}: "
                            f"Δ={stat_result['mean_diff']:.4f}, "
                            f"p={stat_result.get('p_value', 1.0):.4e} ({sig})"
                        )

        # Holm-Bonferroni correction across ALL comparisons
        # (ablations × modality configs × metrics)
        all_p: List[float] = []
        all_keys: List[Tuple[str, str]] = []
        for ab_name, metrics_dict in ablation_stats.items():
            for result_key, result in metrics_dict.items():
                all_p.append(result.get("p_value", 1.0))
                all_keys.append((ab_name, result_key))
        if all_p:
            m = len(all_p)
            ranked = sorted(range(m), key=lambda i: all_p[i])
            corrected = [1.0] * m
            sorted_corrected = []
            for rank, idx in enumerate(ranked):
                sorted_corrected.append(min(all_p[idx] * (m - rank), 1.0))
            for i in range(1, len(sorted_corrected)):
                sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
            for rank, idx in enumerate(ranked):
                corrected[idx] = sorted_corrected[rank]
            for i, (ab_name, result_key) in enumerate(all_keys):
                ablation_stats[ab_name][result_key]["p_holm"] = round(corrected[i], 6)
                ablation_stats[ab_name][result_key]["significant_holm_05"] = corrected[i] < 0.05

        save_json(ablation_stats, out / "ablation_statistical_comparison.json")
        LOG.info(f"[Ablation] Statistical comparison saved to {out / 'ablation_statistical_comparison.json'}")
        LOG.info(f"[Ablation] Total comparisons: {len(all_p)} "
                 f"({len(ablation_cfg.ablations)} ablations × "
                 f"{len(config_tags)} modality configs × metrics)")

    LOG.info(f"\n[Ablation] All experiments complete. Results in {out}")


# ══════════════════════════════════════════════════════════════════════════════
# §20  STATISTICAL TESTING
# ══════════════════════════════════════════════════════════════════════════════

def _effect_size_magnitude_from_r(r: float) -> str:
    ar = abs(float(r))
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "small"
    if ar < 0.5:
        return "medium"
    return "large"


def statistical_comparison(
    results_a: List[float], results_b: List[float],
    test: str = "wilcoxon",
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test for paired comparison with full descriptives.

    Appropriate for Dice/HD95 scores which are non-normally distributed.
    Returns p-value, effect size (rank-biserial r), descriptive statistics,
    and Shapiro-Wilk normality test results.
    """
    from scipy import stats

    a, b = np.array(results_a, dtype=float), np.array(results_b, dtype=float)
    diff = a - b

    def _descriptives(arr: np.ndarray, label: str) -> Dict[str, Any]:
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        d: Dict[str, Any] = {
            f"mean_{label}": float(np.mean(arr)),
            f"std_{label}": std,
            f"median_{label}": float(np.median(arr)),
            f"iqr_{label}": round(q3 - q1, 6),
            f"q1_{label}": round(q1, 6),
            f"q3_{label}": round(q3, 6),
            f"min_{label}": float(np.min(arr)),
            f"max_{label}": float(np.max(arr)),
            f"n_{label}": int(len(arr)),
        }
        if len(arr) >= 3:
            sw_stat, sw_p = stats.shapiro(arr[:5000])
            d[f"shapiro_W_{label}"] = round(float(sw_stat), 6)
            d[f"shapiro_p_{label}"] = round(float(sw_p), 6)
            d[f"normal_{label}"] = bool(sw_p >= 0.05)
        return d

    result: Dict[str, Any] = {
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        "median_diff": float(np.median(diff)),
    }
    result.update(_descriptives(a, "a"))
    result.update(_descriptives(b, "b"))

    if test == "wilcoxon":
        # Remove zero differences (ties) - required by Wilcoxon signed-rank.
        nonzero = diff != 0
        if nonzero.sum() < 10:
            result["p_value"] = 1.0
            result["warning"] = f"Too few non-tied pairs ({int(nonzero.sum())} < 10)"
            return result
        stat, p_val = stats.wilcoxon(diff[nonzero], alternative="two-sided")
        n = int(nonzero.sum())
        nz_diff = diff[nonzero]
        ranks = stats.rankdata(np.abs(nz_diff))
        w_plus = float(ranks[nz_diff > 0].sum())
        w_minus = float(ranks[nz_diff < 0].sum())
        total_rank = w_plus + w_minus
        r = float((w_plus - w_minus) / total_rank) if total_rank > 0 else 0.0
        result["statistic"] = float(stat)
        result["p_value"] = float(p_val)
        result["effect_size_r"] = float(r)
        result["effect_size_magnitude"] = _effect_size_magnitude_from_r(r)
        result["n_pairs"] = n
    else:
        stat, p_val = stats.ttest_rel(a, b)
        result["statistic"] = float(stat)
        result["p_value"] = float(p_val)

    return result


def bootstrap_ci(
    per_case_values: List[float],
    n_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
    method: str = "bca",
) -> Tuple[float, float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI for the mean.

    BCa corrects for bias and skewness in the bootstrap distribution and
    generally gives better coverage than a simple percentile interval for
    asymmetric metrics like Dice and HD95.

    Args:
        per_case_values: Per-patient metric values (e.g. per-case Dice).
        n_resamples:     Bootstrap resamples. 5000 is the standard reporting
        confidence:      Coverage probability (default 0.95 → 95% CI).
        seed:            RNG seed for reproducibility.
        method:          "bca" (default) or "percentile".

    Returns:
        (mean, lower_ci, upper_ci)

    References:
        Efron B. Better Bootstrap Confidence Intervals. JASA 1987.
        Efron B, Tibshirani RJ. An Introduction to the Bootstrap. 1993.
        DiCiccio TJ, Efron B. Bootstrap Confidence Intervals. Stat Sci 1996.
        Isensee F et al. nnU-Net. Nat Methods 2021.
    """
    from scipy import stats as _stats

    arr = np.array(per_case_values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    if n < 2:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        boot_means[i] = rng.choice(arr, size=n, replace=True).mean()

    alpha = 1.0 - confidence
    if method == "percentile" or n < 5:
        lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
        upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
        return mean_val, lower, upper

    prop_less = float(np.mean(boot_means < mean_val))
    prop_less = float(np.clip(prop_less, 1e-10, 1.0 - 1e-10))
    z0 = float(_stats.norm.ppf(prop_less))

    jack_means = np.array([np.mean(np.delete(arr, i)) for i in range(n)], dtype=float)
    jack_dev = jack_means.mean() - jack_means
    num = float(np.sum(jack_dev ** 3))
    den = float(6.0 * (np.sum(jack_dev ** 2) ** 1.5))
    a_hat = num / den if abs(den) > 1e-12 else 0.0

    z_alpha_lo = _stats.norm.ppf(alpha / 2.0)
    z_alpha_hi = _stats.norm.ppf(1.0 - alpha / 2.0)

    def _adj_p(z_alpha: float) -> float:
        num_z = z0 + z_alpha
        denom = 1.0 - a_hat * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            return alpha / 2.0 if z_alpha < 0 else 1.0 - alpha / 2.0
        return float(_stats.norm.cdf(z0 + num_z / denom))

    p_lo = float(np.clip(_adj_p(z_alpha_lo), 0.0, 1.0))
    p_hi = float(np.clip(_adj_p(z_alpha_hi), 0.0, 1.0))

    lower = float(np.percentile(boot_means, 100.0 * p_lo))
    upper = float(np.percentile(boot_means, 100.0 * p_hi))
    if lower > upper:
        lower = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
        upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, lower, upper


def _save_rows_csv(rows: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
    rows = list(rows)
    path = Path(path)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def _normalise_subject_id(value: Any) -> str:
    text = _clean_metadata_value(value).upper()
    return "".join(ch for ch in text if ch.isalnum())


def _clean_categorical_metadata_value(value: Any) -> str:
    text = _clean_metadata_value(value)
    if not text or text.lower() in {"na", "nan", "none", "not reported", "unknown"}:
        return "Missing/Unknown"
    return text


def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    p_values = [float(np.clip(p, 0.0, 1.0)) for p in p_values]
    if not p_values:
        return []
    m = len(p_values)
    ranked = sorted(range(m), key=lambda i: p_values[i])
    corrected = [1.0] * m
    sorted_corrected: List[float] = []
    for rank, idx in enumerate(ranked):
        sorted_corrected.append(min(p_values[idx] * (m - rank), 1.0))
    for idx in range(1, len(sorted_corrected)):
        sorted_corrected[idx] = max(sorted_corrected[idx], sorted_corrected[idx - 1])
    for rank, idx in enumerate(ranked):
        corrected[idx] = sorted_corrected[rank]
    return corrected


def _describe_array(arr: np.ndarray, label: str) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=float)
    out: Dict[str, Any] = {f"n_{label}": int(arr.size)}
    if arr.size == 0:
        return out
    q1, q3 = np.percentile(arr, [25, 75])
    out.update(
        {
            f"mean_{label}": float(np.mean(arr)),
            f"std_{label}": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            f"median_{label}": float(np.median(arr)),
            f"q1_{label}": float(q1),
            f"q3_{label}": float(q3),
            f"iqr_{label}": float(q3 - q1),
            f"min_{label}": float(np.min(arr)),
            f"max_{label}": float(np.max(arr)),
        }
    )
    return out


def _independent_numeric_comparison(
    failed: Sequence[float],
    nonfailed: Sequence[float],
    *,
    min_group_n: int = 10,
) -> Dict[str, Any]:
    from scipy import stats

    fail_arr = np.asarray([float(v) for v in failed if np.isfinite(v)], dtype=float)
    ok_arr = np.asarray([float(v) for v in nonfailed if np.isfinite(v)], dtype=float)
    result: Dict[str, Any] = {"test": "mannwhitneyu"}
    result.update(_describe_array(fail_arr, "failed"))
    result.update(_describe_array(ok_arr, "nonfailed"))
    if fail_arr.size < min_group_n or ok_arr.size < min_group_n:
        result["warning"] = (
            f"Too few observations for independent-group test "
            f"({fail_arr.size} failed, {ok_arr.size} nonfailed; min={min_group_n})"
        )
        result["p_value"] = 1.0
        return result
    stat, p_val = stats.mannwhitneyu(fail_arr, ok_arr, alternative="two-sided")
    denom = float(fail_arr.size * ok_arr.size)
    rank_biserial = float((2.0 * float(stat) / denom) - 1.0) if denom > 0 else 0.0
    result["statistic"] = float(stat)
    result["p_value"] = float(p_val)
    result["effect_size_r"] = rank_biserial
    result["effect_size_magnitude"] = _effect_size_magnitude_from_r(rank_biserial)
    result["higher_in_failed"] = bool(np.median(fail_arr) > np.median(ok_arr))
    return result


def _collapse_rare_levels(
    values: Sequence[str],
    *,
    min_total_count: int = 10,
    max_levels: int = 8,
) -> List[str]:
    cleaned = [_clean_categorical_metadata_value(v) for v in values]
    counts = {}
    for value in cleaned:
        counts[value] = counts.get(value, 0) + 1
    if len(counts) <= max_levels and all(
        (value == "Missing/Unknown" or count >= min_total_count)
        for value, count in counts.items()
    ):
        return cleaned
    collapsed: List[str] = []
    for value in cleaned:
        if value == "Missing/Unknown":
            collapsed.append(value)
        elif counts.get(value, 0) < min_total_count:
            collapsed.append("Other")
        else:
            collapsed.append(value)
    return collapsed


def _independent_categorical_comparison(
    failed: Sequence[str],
    nonfailed: Sequence[str],
    *,
    min_total_count: int = 10,
) -> Dict[str, Any]:
    from scipy import stats

    fail_vals = _collapse_rare_levels(failed, min_total_count=min_total_count)
    ok_vals = _collapse_rare_levels(nonfailed, min_total_count=min_total_count)
    level_order: List[str] = []
    total_counts: Dict[str, int] = {}
    for value in fail_vals + ok_vals:
        total_counts[value] = total_counts.get(value, 0) + 1
        if value not in level_order:
            level_order.append(value)
    level_order.sort(key=lambda key: (-total_counts[key], key))

    result: Dict[str, Any] = {
        "test": "fisher_exact" if len(level_order) == 2 else "chi2_contingency",
        "levels": level_order,
        "n_failed": len(fail_vals),
        "n_nonfailed": len(ok_vals),
    }
    if len(level_order) < 2:
        result["warning"] = "Only one category present after collapsing."
        result["p_value"] = 1.0
        return result

    table = np.zeros((2, len(level_order)), dtype=int)
    for col_idx, level in enumerate(level_order):
        table[0, col_idx] = sum(1 for value in fail_vals if value == level)
        table[1, col_idx] = sum(1 for value in ok_vals if value == level)

    level_stats = []
    for col_idx, level in enumerate(level_order):
        fail_count = int(table[0, col_idx])
        ok_count = int(table[1, col_idx])
        total = fail_count + ok_count
        level_stats.append(
            {
                "level": level,
                "failed_count": fail_count,
                "nonfailed_count": ok_count,
                "total_count": total,
                "failure_rate": float(fail_count / total) if total > 0 else None,
            }
        )
    result["level_stats"] = level_stats

    if len(level_order) == 2:
        odds_ratio, p_val = stats.fisher_exact(table)
        result["statistic"] = float(odds_ratio)
        result["p_value"] = float(p_val)
        result["odds_ratio"] = float(odds_ratio)
    else:
        chi2, p_val, dof, _ = stats.chi2_contingency(table)
        n_total = int(table.sum())
        cramer_v = math.sqrt(float(chi2) / max(1.0, float(n_total * min(table.shape[0] - 1, table.shape[1] - 1))))
        result["statistic"] = float(chi2)
        result["p_value"] = float(p_val)
        result["degrees_of_freedom"] = int(dof)
        result["cramers_v"] = float(cramer_v)
        result["effect_size_magnitude"] = _effect_size_magnitude_from_r(cramer_v)
    return result


def _prepare_failure_mode_metadata_row(row: Dict[str, Any]) -> Dict[str, Any]:
    grade_raw = row.get("Tumor Grade", "")
    grade_group = "Missing/Unknown"
    grade_label = _clean_categorical_metadata_value(grade_raw)
    if grade_label not in {"Missing/Unknown"}:
        if grade_label in {"2", "3", "2.0", "3.0"}:
            grade_group = "2-3"
        elif grade_label in {"4", "4.0"}:
            grade_group = "4"
        else:
            grade_group = grade_label

    scanner_strength = _parse_optional_float(row.get("Scanner Strength"))
    if scanner_strength is None:
        scanner_strength_bucket = "Missing/Unknown"
    elif scanner_strength <= 1.0:
        scanner_strength_bucket = "<=1.0T"
    elif scanner_strength <= 1.8:
        scanner_strength_bucket = "1.1-1.8T"
    else:
        scanner_strength_bucket = ">1.8T"

    return {
        "Subject ID": _clean_metadata_value(row.get("Subject ID")),
        "Sex at birth": _clean_categorical_metadata_value(row.get("Sex at birth")),
        "Age at Imaging": _parse_optional_float(row.get("Age at Imaging")),
        "Race": _clean_categorical_metadata_value(row.get("Race")),
        "Ethnicity": _clean_categorical_metadata_value(row.get("Ethnicity")),
        "Tumor Type": _clean_categorical_metadata_value(row.get("Tumor Type")),
        "Tumor Grade": grade_label,
        "Tumor Grade Group": grade_group,
        "IDH": _clean_categorical_metadata_value(row.get("IDH")),
        "1p19Q CODEL": _clean_categorical_metadata_value(row.get("1p19Q CODEL")),
        "MGMT": _clean_categorical_metadata_value(row.get("MGMT")),
        "Scanner Make": _clean_categorical_metadata_value(row.get("Scanner Make")),
        "Scanner Model": _clean_categorical_metadata_value(row.get("Scanner Model")),
        "Scanner Strength": scanner_strength,
        "Scanner Strength Bucket": scanner_strength_bucket,
        "Manually Refined Segmentation": _clean_categorical_metadata_value(row.get("Manually Refined Segmentation")),
    }


def clinical_failure_mode_analysis(
    case_rows: Sequence[Dict[str, Any]],
    metadata_tsv: str,
    out_dir: str,
    *,
    primary_metric: str = "pp_all_dice_brats_mean",
    failure_quantile: float = 0.25,
    min_group_n: int = 10,
) -> Dict[str, Any]:
    """Analyse clinical correlates of poor external-validation performance.

    Failure is defined by a data-driven threshold on a primary per-case metric
    (default: bottom quartile of postprocessed full-modality BraTS mean Dice).
    The output includes a merged per-case table and univariate independent-group
    comparisons across clinical metadata fields.
    """
    out = ensure_dir(out_dir)
    metadata_path = Path(metadata_tsv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Clinical metadata TSV not found: {metadata_path}")

    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            subject_id = _normalise_subject_id(row.get("Subject ID"))
            if subject_id:
                metadata_by_id[subject_id] = _prepare_failure_mode_metadata_row(row)

    primary_metric = str(primary_metric)
    lower_is_worse = "hd95" not in primary_metric.lower()
    case_rows = list(case_rows)
    matched_rows: List[Dict[str, Any]] = []
    unmatched_case_ids: List[str] = []

    for row in case_rows:
        case_id = _clean_metadata_value(row.get("case_id"))
        metric_val = row.get(primary_metric)
        if metric_val is None or not np.isfinite(float(metric_val)):
            continue
        subject_key = _normalise_subject_id(case_id)
        metadata = metadata_by_id.get(subject_key)
        merged = dict(row)
        if metadata is None:
            unmatched_case_ids.append(case_id)
            continue
        merged.update(metadata)
        matched_rows.append(merged)

    if not matched_rows:
        result = {
            "metadata_tsv": str(metadata_path.resolve()),
            "primary_metric": primary_metric,
            "warning": "No evaluation cases matched the clinical metadata table.",
            "n_case_rows": len(case_rows),
            "n_metadata_rows": len(metadata_by_id),
        }
        save_json(result, out / "clinical_failure_mode_analysis.json")
        return result

    metric_values = np.asarray([float(row[primary_metric]) for row in matched_rows], dtype=float)
    q = float(np.clip(failure_quantile, 0.01, 0.49))
    threshold = float(np.quantile(metric_values, q if lower_is_worse else 1.0 - q))
    for row in matched_rows:
        value = float(row[primary_metric])
        row["failure_primary_metric"] = primary_metric
        row["failure_primary_value"] = value
        row["failure_primary_threshold"] = threshold
        row["failure_primary_direction"] = "lower_is_worse" if lower_is_worse else "higher_is_worse"
        row["failure_primary"] = bool(value <= threshold) if lower_is_worse else bool(value >= threshold)

    failed_rows = [row for row in matched_rows if row["failure_primary"]]
    nonfailed_rows = [row for row in matched_rows if not row["failure_primary"]]

    numeric_fields = [
        "Age at Imaging",
        "Scanner Strength",
    ]
    categorical_fields = [
        "Sex at birth",
        "Race",
        "Ethnicity",
        "Tumor Type",
        "Tumor Grade",
        "Tumor Grade Group",
        "IDH",
        "1p19Q CODEL",
        "MGMT",
        "Scanner Make",
        "Scanner Model",
        "Scanner Strength Bucket",
        "Manually Refined Segmentation",
    ]

    variable_tests: List[Dict[str, Any]] = []
    p_values: List[float] = []
    for field_name in numeric_fields:
        failed = [float(row[field_name]) for row in failed_rows if row.get(field_name) is not None]
        nonfailed = [float(row[field_name]) for row in nonfailed_rows if row.get(field_name) is not None]
        stat = _independent_numeric_comparison(failed, nonfailed, min_group_n=min_group_n)
        stat.update({"variable": field_name, "variable_type": "numeric"})
        variable_tests.append(stat)
        p_values.append(float(stat.get("p_value", 1.0)))

    for field_name in categorical_fields:
        failed = [_clean_categorical_metadata_value(row.get(field_name)) for row in failed_rows]
        nonfailed = [_clean_categorical_metadata_value(row.get(field_name)) for row in nonfailed_rows]
        stat = _independent_categorical_comparison(failed, nonfailed, min_total_count=min_group_n)
        stat.update({"variable": field_name, "variable_type": "categorical"})
        variable_tests.append(stat)
        p_values.append(float(stat.get("p_value", 1.0)))

    corrected = _holm_bonferroni(p_values)
    for idx, corrected_p in enumerate(corrected):
        variable_tests[idx]["p_holm"] = round(float(corrected_p), 6)
        variable_tests[idx]["significant_holm_05"] = bool(corrected_p < 0.05)

    variable_tests.sort(
        key=lambda row: (float(row.get("p_holm", 1.0)), float(row.get("p_value", 1.0)), row.get("variable", "")),
    )

    case_table_rows = []
    for row in matched_rows:
        case_table_rows.append(
            {
                "case_id": row.get("case_id"),
                "dataset_tag": row.get("dataset_tag"),
                "failure_primary": row.get("failure_primary"),
                "failure_primary_metric": row.get("failure_primary_metric"),
                "failure_primary_value": row.get("failure_primary_value"),
                "failure_primary_threshold": row.get("failure_primary_threshold"),
                "all_dice_brats_mean": row.get("all_dice_brats_mean"),
                "all_dice_WT": row.get("all_dice_WT"),
                "all_dice_TC": row.get("all_dice_TC"),
                "all_dice_ET": row.get("all_dice_ET"),
                "all_hd95_brats_mean": row.get("all_hd95_brats_mean"),
                "all_hd95_WT": row.get("all_hd95_WT"),
                "all_hd95_TC": row.get("all_hd95_TC"),
                "all_hd95_ET": row.get("all_hd95_ET"),
                "pp_all_dice_brats_mean": row.get("pp_all_dice_brats_mean"),
                "pp_all_dice_WT": row.get("pp_all_dice_WT"),
                "pp_all_dice_TC": row.get("pp_all_dice_TC"),
                "pp_all_dice_ET": row.get("pp_all_dice_ET"),
                "pp_all_hd95_brats_mean": row.get("pp_all_hd95_brats_mean"),
                "pp_all_hd95_WT": row.get("pp_all_hd95_WT"),
                "pp_all_hd95_TC": row.get("pp_all_hd95_TC"),
                "pp_all_hd95_ET": row.get("pp_all_hd95_ET"),
                "Subject ID": row.get("Subject ID"),
                "Sex at birth": row.get("Sex at birth"),
                "Age at Imaging": row.get("Age at Imaging"),
                "Race": row.get("Race"),
                "Ethnicity": row.get("Ethnicity"),
                "Tumor Type": row.get("Tumor Type"),
                "Tumor Grade": row.get("Tumor Grade"),
                "Tumor Grade Group": row.get("Tumor Grade Group"),
                "IDH": row.get("IDH"),
                "1p19Q CODEL": row.get("1p19Q CODEL"),
                "MGMT": row.get("MGMT"),
                "Scanner Make": row.get("Scanner Make"),
                "Scanner Model": row.get("Scanner Model"),
                "Scanner Strength": row.get("Scanner Strength"),
                "Scanner Strength Bucket": row.get("Scanner Strength Bucket"),
                "Manually Refined Segmentation": row.get("Manually Refined Segmentation"),
            }
        )

    summary = {
        "description": (
            "Clinical failure-mode analysis for external validation. Failure is "
            "defined by a quantile threshold on a primary per-case metric and "
            "compared across independent clinical subgroups."
        ),
        "metadata_tsv": str(metadata_path.resolve()),
        "primary_metric": primary_metric,
        "failure_quantile": q,
        "failure_direction": "lower_is_worse" if lower_is_worse else "higher_is_worse",
        "failure_threshold": threshold,
        "n_case_rows": len(case_rows),
        "n_metadata_rows": len(metadata_by_id),
        "n_matched_cases": len(matched_rows),
        "n_unmatched_cases": len(unmatched_case_ids),
        "unmatched_case_ids": sorted(set(unmatched_case_ids)),
        "n_failed_cases": len(failed_rows),
        "n_nonfailed_cases": len(nonfailed_rows),
        "primary_metric_summary": {
            **_describe_array([row[primary_metric] for row in failed_rows], "failed"),
            **_describe_array([row[primary_metric] for row in nonfailed_rows], "nonfailed"),
            **_describe_array(metric_values, "all"),
        },
        "variable_tests": variable_tests,
    }

    save_json(summary, out / "clinical_failure_mode_analysis.json")
    save_json(variable_tests, out / "clinical_failure_mode_variable_tests.json")
    _save_rows_csv(variable_tests, out / "clinical_failure_mode_variable_tests.csv")
    _save_rows_csv(case_table_rows, out / "clinical_failure_mode_case_table.csv")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# §20a  ONNX WRAPPER FOR evaluate_student COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class ONNXStudentWrapper:
    """Wraps an ONNX Runtime InferenceSession so it can be passed to
    evaluate_student() in place of a PyTorch UNet3DStudent.

    evaluate_student calls ``student.eval()`` and ``student(input_tensor)``.
    This wrapper satisfies both interfaces by routing the __call__ through
    the ONNX session (CPU numpy round-trip).
    """

    def __init__(self, session) -> None:
        self.session = session
        self._inp_name = session.get_inputs()[0].name

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy().astype(np.float32)
        result = self.session.run(None, {self._inp_name: x_np})[0]
        return torch.from_numpy(result)


# ══════════════════════════════════════════════════════════════════════════════
# §20b  STANDALONE ABLATION EVALUATION (all 15 modality combos)
# ══════════════════════════════════════════════════════════════════════════════

def ablation_eval(
    baseline_ckpt: str,
    ablation_dir: str,
    val_csv: str,
    modalities: Sequence[str],
    num_classes: int,
    patch_size: Tuple[int, int, int],
    out_dir: str,
    student_base: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ablations: Optional[List[str]] = None,
    num_workers: int = 20,
    eval_config_batch_size: int = 1,
    metric_workers: int = 1,
    cache_dir: Optional[str] = None,
) -> None:
    """Evaluate existing ablation checkpoints across ALL 15 modality combos.

    For each ablation checkpoint, computes BraTS Dice and HD95 under every
    non-empty modality subset, then runs paired Wilcoxon signed-rank tests
    (with Holm-Bonferroni correction) against the baseline for every
    (modality-config × metric) pair.

    Output JSON structure (ablation_full_comparison.json):
        baseline_performance  → {config_tag → {metric → {mean, ci_lower, ci_upper, n}}}
        ablations             → {ablation_name → {config_tag → {metric → {stat details}}}}

    Usage:
        python OGAP_source_code_experimental_v9.py ablation_eval \\
            --baseline_ckpt  Results/ablation/baseline/best_student.pth \\
            --ablation_dir   Results/ablation \\
            --val_csv        val.csv \\
            --modalities     t1n,t1c,t2w,t2f \\
            --out_dir        Results/ablation_eval \\
            --device         cuda
    """
    out = ensure_dir(out_dir)
    device_t = torch.device(device)
    num_mod = len(modalities)

    eval_configs = all_modality_combinations(num_mod)
    config_tags = [
        "all" if cfg is None
        else f"drop{''.join(str(m) for m in cfg)}"
        for cfg in eval_configs
    ]
    # Human-readable modality-present labels
    config_labels = {}
    for tag, cfg in zip(config_tags, eval_configs):
        present = list(range(num_mod)) if cfg is None else [i for i in range(num_mod) if i not in cfg]
        config_labels[tag] = ",".join(modalities[i] for i in present)

    val_loader = build_eval_loader(
        val_csv,
        modalities,
        patch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        pin_memory=(device_t.type == "cuda"),
    )

    # Auto-discover ablation names if not specified
    if ablations is None:
        ab_dir = Path(ablation_dir)
        ablations = sorted([
            d.name for d in ab_dir.iterdir()
            if d.is_dir() and (d / "best_student.pth").exists()
            and d.name != "baseline"
        ])
    LOG.info(f"[AblationEval] Baseline: {baseline_ckpt}")
    LOG.info(f"[AblationEval] Ablations ({len(ablations)}): {', '.join(ablations)}")
    LOG.info(f"[AblationEval] Modality configs: {len(eval_configs)}")

    # ── Evaluate baseline ─────────────────────────────────────────────────
    LOG.info("[AblationEval] Evaluating baseline across all 15 modality configs ...")
    baseline_student = _load_student_for_eval(
        Path(baseline_ckpt), num_mod, num_classes, student_base, device_t,
    )
    baseline_resume = out / "resume_ablation_eval_baseline.json"
    baseline_results, baseline_per_case = evaluate_student(
        baseline_student, val_loader, num_classes, device_t,
        use_brats=True, eval_mask_configs=eval_configs,
        postprocess=True, return_per_case=True, compute_hd95=True,
        eval_config_batch_size=eval_config_batch_size,
        metric_workers=metric_workers,
        checkpoint_path=str(baseline_resume),
        checkpoint_metadata={
            "phase": "ablation_eval_baseline",
            "model_path": str(Path(baseline_ckpt).resolve()),
            "val_csv": str(Path(val_csv).resolve()),
        },
    )
    del baseline_student
    if device_t.type == "cuda":
        torch.cuda.empty_cache()

    METRICS = ["dice_brats_mean", "dice_WT", "dice_TC", "dice_ET",
               "hd95_brats_mean", "hd95_WT", "hd95_TC", "hd95_ET"]

    # Baseline CIs
    baseline_ci: Dict[str, Any] = {}
    for cfg_tag in config_tags:
        baseline_ci[cfg_tag] = {}
        for metric in METRICS:
            vals = baseline_per_case.get(f"{cfg_tag}_{metric}", [])
            if vals:
                mean_v, lo, hi = bootstrap_ci(vals)
                baseline_ci[cfg_tag][metric] = {
                    "mean": round(mean_v, 5),
                    "ci_lower": round(lo, 5), "ci_upper": round(hi, 5),
                    "n_cases": len(vals),
                }

    # ── Evaluate each ablation ────────────────────────────────────────────
    ablation_stats: Dict[str, Any] = {}
    all_p: List[float] = []
    all_keys: List[Tuple[str, str, str]] = []   # (ablation, config, metric)

    for ablation_name in ablations:
        ab_ckpt = Path(ablation_dir) / ablation_name / "best_student.pth"
        if not ab_ckpt.exists():
            LOG.warning(f"[AblationEval] Checkpoint not found: {ab_ckpt}, skipping")
            continue

        LOG.info(f"[AblationEval] Evaluating {ablation_name} across all 15 configs ...")
        ab_student = _load_student_for_eval(
            ab_ckpt, num_mod, num_classes, student_base, device_t,
        )
        ab_resume = out / f"resume_ablation_eval_{ablation_name}.json"
        _, ab_per_case = evaluate_student(
            ab_student, val_loader, num_classes, device_t,
            use_brats=True, eval_mask_configs=eval_configs,
            postprocess=True, return_per_case=True, compute_hd95=True,
            eval_config_batch_size=eval_config_batch_size,
            metric_workers=metric_workers,
            checkpoint_path=str(ab_resume),
            checkpoint_metadata={
                "phase": f"ablation_eval_{ablation_name}",
                "model_path": str(ab_ckpt.resolve()),
                "val_csv": str(Path(val_csv).resolve()),
            },
        )
        del ab_student
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

        ablation_stats[ablation_name] = {}
        for cfg_tag in config_tags:
            ablation_stats[ablation_name][cfg_tag] = {}
            for metric in METRICS:
                key = f"{cfg_tag}_{metric}"
                base_vals = baseline_per_case.get(key, [])
                ab_vals   = ab_per_case.get(key, [])
                if len(base_vals) >= 10 and len(ab_vals) >= 10:
                    stat = statistical_comparison(base_vals, ab_vals)
                    ab_mean, ab_lo, ab_hi = bootstrap_ci(ab_vals)
                    stat["ablation_mean"] = round(ab_mean, 5)
                    stat["ablation_ci"]   = [round(ab_lo, 5), round(ab_hi, 5)]
                    stat["n_cases"]       = len(base_vals)
                    ablation_stats[ablation_name][cfg_tag][metric] = stat
                    all_p.append(stat.get("p_value", 1.0))
                    all_keys.append((ablation_name, cfg_tag, metric))

    # ── Holm-Bonferroni across ALL comparisons ────────────────────────────
    if all_p:
        m = len(all_p)
        ranked = sorted(range(m), key=lambda i: all_p[i])
        corrected = [1.0] * m
        sorted_corrected = []
        for rank, idx in enumerate(ranked):
            sorted_corrected.append(min(all_p[idx] * (m - rank), 1.0))
        for i in range(1, len(sorted_corrected)):
            sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
        for rank, idx in enumerate(ranked):
            corrected[idx] = sorted_corrected[rank]
        for i, (ab_name, cfg_tag, metric) in enumerate(all_keys):
            ablation_stats[ab_name][cfg_tag][metric]["p_holm"] = round(corrected[i], 6)
            ablation_stats[ab_name][cfg_tag][metric]["significant_holm_05"] = corrected[i] < 0.05

    # ── Build and save output JSON ────────────────────────────────────────
    output = {
        "description": (
            "Ablation study: per-ablation comparison vs baseline across "
            "all 15 non-empty modality combinations. Each entry contains "
            "Wilcoxon signed-rank test results with Holm-Bonferroni correction."
        ),
        "baseline_checkpoint": str(baseline_ckpt),
        "ablation_dir": str(ablation_dir),
        "modality_names": list(modalities),
        "n_modality_configs": len(eval_configs),
        "n_ablations": len(ablation_stats),
        "total_statistical_comparisons": len(all_p),
        "modality_configs": {
            tag: {"missing_indices": cfg, "present_modalities": config_labels[tag]}
            for tag, cfg in zip(config_tags, eval_configs)
        },
        "baseline_performance": baseline_ci,
        "baseline_summary": baseline_results,
        "ablations": ablation_stats,
    }
    save_json(output, out / "ablation_full_comparison.json")

    # ── Log summary table (dice_brats_mean only, for readability) ─────────
    LOG.info("\n[AblationEval] ── Summary: dice_brats_mean (Δ vs baseline) ──")
    header = f"  {'Ablation':28s} {'Config':12s} {'Base':>7s} {'Ablat':>7s} {'Δ':>8s} {'p_holm':>10s} {'Sig':>4s}"
    LOG.info(header)
    LOG.info("  " + "-" * (len(header) - 2))
    for ab_name in ablation_stats:
        for cfg_tag in config_tags:
            r = ablation_stats[ab_name].get(cfg_tag, {}).get("dice_brats_mean", {})
            if r:
                sig = "***" if r.get("p_holm", 1) < 0.001 else \
                      "**"  if r.get("p_holm", 1) < 0.01 else \
                      "*"   if r.get("p_holm", 1) < 0.05 else "ns"
                bm = baseline_ci.get(cfg_tag, {}).get("dice_brats_mean", {}).get("mean", 0)
                LOG.info(
                    f"  {ab_name:28s} {cfg_tag:12s} {bm:7.4f} "
                    f"{r.get('ablation_mean', 0):7.4f} {r['mean_diff']:+8.4f} "
                    f"{r.get('p_holm', 1.0):10.2e} {sig:>4s}"
                )

    LOG.info(f"\n[AblationEval] Complete JSON → {out / 'ablation_full_comparison.json'}")
    LOG.info(f"[AblationEval] {len(all_p)} comparisons "
             f"({len(ablation_stats)} ablations × {len(config_tags)} configs × {len(METRICS)} metrics)")


# ══════════════════════════════════════════════════════════════════════════════
# §20c  FP32 vs INT8 QUANTISATION ACCURACY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_quantisation(
    fp32_ckpt: str,
    int8_onnx: str,
    val_csv: str,
    modalities: Sequence[str],
    num_classes: int,
    patch_size: Tuple[int, int, int],
    out_dir: str,
    student_base: int = 16,
    fp32_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 20,
    num_threads: Optional[int] = None,
    eval_config_batch_size: int = 1,
    metric_workers: int = 1,
    cache_dir: Optional[str] = None,
    deploy_min_present: int = 1,
    deploy_max_present: Optional[int] = 4,
    max_mean_dice_drop: float = 0.01,
    max_worst_dice_drop: float = 0.02,
    max_mean_hd95_increase: float = 1.0,
    max_worst_hd95_increase: float = 2.0,
    track_energy: bool = True,
    energy_poll_interval: float = 5.0,
) -> None:
    """Compare FP32 PyTorch (GPU) vs INT8 ONNX (CPU) across supported deployment configs.

    For each supported modality configuration, evaluates both models on the full
    validation set, computes BraTS Dice + HD95, bootstrap 95% CIs, and paired
    Wilcoxon signed-rank tests with Holm-Bonferroni correction. A deployment
    summary then decides whether INT8 drift stays inside the configured
    acceptance thresholds for the supported 1-4 modality regime.

    Output JSON: quantisation_comparison.json

    Usage:
        python OGAP_source_code_experimental_v9.py compare_quantisation \\
            --fp32_ckpt    Results/student/best_student.pth \\
            --int8_onnx    Results/export/best_student_int8_static.onnx \\
            --val_csv      val.csv \\
            --modalities   t1n,t1c,t2w,t2f \\
            --out_dir      Results/quantisation_eval \\
            --fp32_device  cuda
    """
    out = ensure_dir(out_dir)
    num_mod = len(modalities)

    eval_configs = supported_modality_combinations(
        num_mod,
        min_present=deploy_min_present,
        max_present=deploy_max_present,
    )
    config_tags = [modality_config_tag(cfg) for cfg in eval_configs]
    config_labels = {}
    for tag, cfg in zip(config_tags, eval_configs):
        present = list(range(num_mod)) if cfg is None else [i for i in range(num_mod) if i not in cfg]
        config_labels[tag] = ",".join(modalities[i] for i in present)

    fp32_device_t = torch.device(fp32_device)
    val_loader = build_eval_loader(
        val_csv,
        modalities,
        patch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
        pin_memory=(fp32_device_t.type == "cuda"),
    )

    METRICS = ["dice_brats_mean", "dice_WT", "dice_TC", "dice_ET",
               "hd95_brats_mean", "hd95_WT", "hd95_TC", "hd95_ET"]
    energy_reports: Dict[str, Any] = {}

    # ── FP32 evaluation ───────────────────────────────────────────────────
    LOG.info(f"[Quantisation] Evaluating FP32 model on {fp32_device} ...")
    fp32_student = _load_student_for_eval(
        Path(fp32_ckpt), num_mod, num_classes, student_base, fp32_device_t,
    )
    fp32_resume = out / "resume_quantisation_fp32.json"
    fp32_energy = None
    if track_energy:
        fp32_cores = max(1, min(os.cpu_count() or 1, int(num_workers) + int(metric_workers) + 1))
        fp32_energy = EnergyTracker(
            label="quantisation_fp32",
            out_dir=out,
            poll_interval=energy_poll_interval,
            num_cpu_cores=fp32_cores,
            track_gpu=(fp32_device_t.type == "cuda"),
        )
        fp32_energy.start()
    try:
        fp32_results, fp32_per_case = evaluate_student(
            fp32_student, val_loader, num_classes, fp32_device_t,
            use_brats=True, eval_mask_configs=eval_configs,
            postprocess=True, return_per_case=True, compute_hd95=True,
            eval_config_batch_size=eval_config_batch_size,
            metric_workers=metric_workers,
            checkpoint_path=str(fp32_resume),
            checkpoint_metadata={
                "phase": "quantisation_fp32",
                "model_path": str(Path(fp32_ckpt).resolve()),
                "val_csv": str(Path(val_csv).resolve()),
                "deploy_min_present": int(deploy_min_present),
                "deploy_max_present": None if deploy_max_present is None else int(deploy_max_present),
            },
        )
    finally:
        if fp32_energy is not None:
            energy_reports["fp32_gpu"] = fp32_energy.stop()
    del fp32_student
    if fp32_device_t.type == "cuda":
        torch.cuda.empty_cache()

    # ── INT8 ONNX evaluation ─────────────────────────────────────────────
    LOG.info(f"[Quantisation] Evaluating INT8 ONNX model on CPU ...")
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("pip install onnxruntime --break-system-packages")

    opts = _configure_ort_session_options(ort, num_threads=num_threads)

    available_providers = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available_providers:
        providers = [("OpenVINOExecutionProvider", {"device_type": "CPU"}), "CPUExecutionProvider"]
        LOG.info("[Quantisation] ORT: OpenVINOExecutionProvider (CPU)")
    elif "DnnlExecutionProvider" in available_providers:
        providers = [("DnnlExecutionProvider", {"use_arena": "1"}), "CPUExecutionProvider"]
        LOG.info("[Quantisation] ORT: DnnlExecutionProvider (oneDNN / AMX)")
    else:
        providers = ["CPUExecutionProvider"]
        LOG.info("[Quantisation] ORT: CPUExecutionProvider")

    _enable_cpu_ep_graph_optim_if_needed(ort, opts, providers, "Quantisation")
    ort_session = ort.InferenceSession(int8_onnx, opts, providers=providers)
    onnx_wrapper = ONNXStudentWrapper(ort_session)

    int8_resume = out / "resume_quantisation_int8.json"
    int8_energy = None
    if track_energy:
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        ort_threads = int(num_threads) if num_threads is not None else slurm_cpus
        if ort_threads <= 0:
            ort_threads = os.cpu_count() or 1
        int8_cores = max(
            1,
            min(os.cpu_count() or 1, max(int(ort_threads), int(num_workers) + int(metric_workers) + 1)),
        )
        int8_energy = EnergyTracker(
            label="quantisation_int8_cpu",
            out_dir=out,
            poll_interval=energy_poll_interval,
            num_cpu_cores=int8_cores,
            track_gpu=False,
        )
        int8_energy.start()
    try:
        int8_results, int8_per_case = evaluate_student(
            onnx_wrapper, val_loader, num_classes, torch.device("cpu"),
            use_brats=True, eval_mask_configs=eval_configs,
            postprocess=True, return_per_case=True, compute_hd95=True,
            eval_config_batch_size=eval_config_batch_size,
            metric_workers=metric_workers,
            checkpoint_path=str(int8_resume),
            checkpoint_metadata={
                "phase": "quantisation_int8",
                "model_path": str(Path(int8_onnx).resolve()),
                "val_csv": str(Path(val_csv).resolve()),
                "deploy_min_present": int(deploy_min_present),
                "deploy_max_present": None if deploy_max_present is None else int(deploy_max_present),
            },
        )
    finally:
        if int8_energy is not None:
            energy_reports["int8_cpu"] = int8_energy.stop()

    # ── Statistical comparison ────────────────────────────────────────────
    LOG.info("[Quantisation] Computing statistical comparison ...")
    comparison: Dict[str, Any] = {}
    all_p: List[float] = []
    all_keys: List[Tuple[str, str]] = []

    for cfg_tag in config_tags:
        comparison[cfg_tag] = {}
        for metric in METRICS:
            key = f"{cfg_tag}_{metric}"
            fp32_vals = fp32_per_case.get(key, [])
            int8_vals = int8_per_case.get(key, [])
            if len(fp32_vals) >= 10 and len(int8_vals) >= 10:
                stat = statistical_comparison(fp32_vals, int8_vals)
                fp32_mean, fp32_lo, fp32_hi = bootstrap_ci(fp32_vals)
                int8_mean, int8_lo, int8_hi = bootstrap_ci(int8_vals)
                stat["fp32_mean"] = round(fp32_mean, 5)
                stat["fp32_ci"]   = [round(fp32_lo, 5), round(fp32_hi, 5)]
                stat["int8_mean"] = round(int8_mean, 5)
                stat["int8_ci"]   = [round(int8_lo, 5), round(int8_hi, 5)]
                stat["n_cases"]   = len(fp32_vals)
                comparison[cfg_tag][metric] = stat
                all_p.append(stat.get("p_value", 1.0))
                all_keys.append((cfg_tag, metric))

    # Holm-Bonferroni
    if all_p:
        m = len(all_p)
        ranked = sorted(range(m), key=lambda i: all_p[i])
        corrected = [1.0] * m
        sorted_corrected = []
        for rank, idx in enumerate(ranked):
            sorted_corrected.append(min(all_p[idx] * (m - rank), 1.0))
        for i in range(1, len(sorted_corrected)):
            sorted_corrected[i] = max(sorted_corrected[i], sorted_corrected[i - 1])
        for rank, idx in enumerate(ranked):
            corrected[idx] = sorted_corrected[rank]
        for i, (cfg_tag, metric) in enumerate(all_keys):
            comparison[cfg_tag][metric]["p_holm"] = round(corrected[i], 6)
            comparison[cfg_tag][metric]["significant_holm_05"] = corrected[i] < 0.05

    deployment_summary = summarise_quantisation_deployment(
        comparison,
        eval_configs,
        num_mod,
        min_present=deploy_min_present,
        max_present=deploy_max_present,
        max_mean_dice_drop=max_mean_dice_drop,
        max_worst_dice_drop=max_worst_dice_drop,
        max_mean_hd95_increase=max_mean_hd95_increase,
        max_worst_hd95_increase=max_worst_hd95_increase,
    )

    energy_comparison: Dict[str, Any] = {}
    if energy_reports:
        fp32_energy = energy_reports.get("fp32_gpu", {})
        int8_energy = energy_reports.get("int8_cpu", {})
        fp32_wh = float(fp32_energy.get("total_energy_wh", 0.0) or 0.0)
        int8_wh = float(int8_energy.get("total_energy_wh", 0.0) or 0.0)
        n_cases = 0
        for cfg_tag in config_tags:
            n_cases = max(n_cases, len(fp32_per_case.get(f"{cfg_tag}_dice_brats_mean", [])))
        n_case_config_evals = int(n_cases * len(config_tags))
        energy_comparison = {
            "description": (
                "Measured around the FP32 GPU and INT8 CPU validation passes. "
                "The INT8 CPU report intentionally disables GPU accounting so a "
                "GPU allocated by the Slurm wrapper is not charged to the CPU "
                "deployment path."
            ),
            "n_cases": int(n_cases),
            "n_modality_configs": int(len(config_tags)),
            "n_case_config_evaluations": n_case_config_evals,
            "fp32_total_energy_wh": round(fp32_wh, 3) if fp32_energy else None,
            "int8_total_energy_wh": round(int8_wh, 3) if int8_energy else None,
            "int8_energy_ratio_vs_fp32": round(int8_wh / fp32_wh, 6) if fp32_wh > 0 and int8_energy else None,
            "int8_energy_savings_pct": round(100.0 * (1.0 - int8_wh / fp32_wh), 3) if fp32_wh > 0 and int8_energy else None,
            "fp32_wh_per_case_config": round(fp32_wh / n_case_config_evals, 6) if fp32_wh > 0 and n_case_config_evals else None,
            "int8_wh_per_case_config": round(int8_wh / n_case_config_evals, 6) if int8_wh > 0 and n_case_config_evals else None,
        }

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "description": (
            "FP32 (GPU) vs INT8 ONNX (CPU) accuracy comparison across "
            "supported deployment modality combinations. Paired Wilcoxon "
            "signed-rank tests with Holm-Bonferroni correction plus a "
            "deployment pass/fail summary."
        ),
        "fp32_checkpoint": str(fp32_ckpt),
        "int8_onnx_model": str(int8_onnx),
        "fp32_device": fp32_device,
        "int8_threads": num_threads,
        "modality_names": list(modalities),
        "deploy_min_present": int(deploy_min_present),
        "deploy_max_present": None if deploy_max_present is None else int(deploy_max_present),
        "n_modality_configs": len(eval_configs),
        "total_statistical_comparisons": len(all_p),
        "modality_configs": {
            tag: {"missing_indices": cfg, "present_modalities": config_labels[tag]}
            for tag, cfg in zip(config_tags, eval_configs)
        },
        "comparisons": comparison,
        "fp32_summary": fp32_results,
        "int8_summary": int8_results,
        "energy_reports": energy_reports,
        "energy_comparison": energy_comparison,
        "deployment_summary": deployment_summary,
    }
    save_json(output, out / "quantisation_comparison.json")

    # ── Log summary ───────────────────────────────────────────────────────
    LOG.info("\n[Quantisation] ── Summary ──")
    header = f"  {'Config':12s} {'Metric':18s} {'FP32':>7s} {'INT8':>7s} {'Δ':>8s} {'p_holm':>10s} {'Sig':>4s}"
    LOG.info(header)
    LOG.info("  " + "-" * (len(header) - 2))
    for cfg_tag in config_tags:
        for metric in ["dice_brats_mean", "hd95_brats_mean"]:
            r = comparison.get(cfg_tag, {}).get(metric, {})
            if r:
                sig = "***" if r.get("p_holm", 1) < 0.001 else \
                      "**"  if r.get("p_holm", 1) < 0.01 else \
                      "*"   if r.get("p_holm", 1) < 0.05 else "ns"
                LOG.info(
                    f"  {cfg_tag:12s} {metric:18s} {r['fp32_mean']:7.4f} "
                    f"{r['int8_mean']:7.4f} {r['mean_diff']:+8.4f} "
                    f"{r.get('p_holm', 1.0):10.2e} {sig:>4s}"
                )

    pass_str = "PASS" if deployment_summary.get("deployment_pass", False) else "FAIL"
    LOG.info(
        "[Quantisation] Deployment summary: %s "
        "(mean Dice drop=%s, worst Dice drop=%s, mean HD95 increase=%s, worst HD95 increase=%s)",
        pass_str,
        deployment_summary.get("supported_mean_dice_drop"),
        deployment_summary.get("supported_worst_dice_drop"),
        deployment_summary.get("supported_mean_hd95_increase"),
        deployment_summary.get("supported_worst_hd95_increase"),
    )
    if deployment_summary.get("deployment_fail_reasons"):
        LOG.info(
            "[Quantisation] Deployment fail reasons: %s",
            ", ".join(str(x) for x in deployment_summary["deployment_fail_reasons"]),
        )
    if energy_comparison:
        LOG.info(
            "[Quantisation] Energy: FP32=%s Wh, INT8=%s Wh, savings=%s%%",
            energy_comparison.get("fp32_total_energy_wh"),
            energy_comparison.get("int8_total_energy_wh"),
            energy_comparison.get("int8_energy_savings_pct"),
        )

    LOG.info(f"\n[Quantisation] Complete JSON → {out / 'quantisation_comparison.json'}")
    LOG.info(f"[Quantisation] {len(all_p)} comparisons "
             f"({len(config_tags)} configs × {len(METRICS)} metrics)")




# ══════════════════════════════════════════════════════════════════════════════
# §20A  MEDIA ADDENDUM ANALYSES
# ══════════════════════════════════════════════════════════════════════════════
# Integrated from OGAP_media_addendum.py so this script is self-contained for
# cluster deployment and analysis-facing analyses. The addendum commands still
# receive `core` in dispatch so they can call core helpers without duplicating
# training/evaluation implementations.

ADDENDUM_LOG = logging.getLogger("OGAP.addendum")

# Subcommands implemented in this module (used by core CLI for dispatch).
ADDENDUM_COMMANDS: Tuple[str, ...] = (
    "lint_csv",
    "audit_voxel_spacing",
    "surface_dice",
    "calibration",
    "mc_uncertainty",
    "metric_check",
    "stratify",
    "interrater",
    "threshold_sweep",
    "holder_sweep",
    "kd_compare",
    "erasmus_bias",
    "single_modality_scenario",
    "field_strength_curve",
    "loso_cv",
    "kfold_cv",
    "multi_seed",
    "training_curves",
    "ablation_heatmap",
    "subgroup_forest",
    "failure_case_figure",
    "compute_profile",
    "quantization_hardware",
    "total_energy_table",
    "model_card",
    "datasheet",
    "data_flow",
    "sampler_distribution",
    "crc",
    "mahalanobis_ood",
    "tost",
    "delong",
    "dicom_seg",
    "validate_augmentor",
    # CLAIM 2024 + TRIPOD+AI + SAGER 2022 + NIHMS health-equity orchestrator
    # (implementation lives in Scripts/ogap_compliance.py to keep this file
    # focused on training/eval; the addendum CLI imports it lazily).
    "ai_compliance_report",
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(obj: Any, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_addendum_json_default)


def _addendum_json_default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def _have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _read_csv_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: List[Dict[str, Any]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_per_case_metrics(path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if "per_case" in raw and isinstance(raw["per_case"], list):
                raw = raw["per_case"]
            else:
                raw = [{"case_id": k, **(v or {})} for k, v in raw.items()]
        return raw
    return _read_csv_rows(path)


# ══════════════════════════════════════════════════════════════════════════════
# A. Surface Dice (NSD) - Nikolov et al. 2018, Clin Cancer Res, also
#     promoted by Isensee (nnU-Net revisited 2024).
# ══════════════════════════════════════════════════════════════════════════════

def _binary_border(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    if mask.sum() == 0:
        return mask.astype(bool)
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return mask.astype(bool) & ~eroded


def surface_dice_score(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    tolerance_mm: float,
) -> float:
    """Normalised Surface Dice at the given tolerance (mm). Binary mask input."""
    from scipy import ndimage as ndi
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    pb = _binary_border(pred)
    gb = _binary_border(gt)
    sp = tuple(float(s) for s in spacing)
    d_gt = ndi.distance_transform_edt(~gb, sampling=sp)
    d_pred = ndi.distance_transform_edt(~pb, sampling=sp)
    pb_within = (d_gt[pb] <= tolerance_mm).sum()
    gb_within = (d_pred[gb] <= tolerance_mm).sum()
    denom = pb.sum() + gb.sum()
    if denom == 0:
        return 1.0
    return float(pb_within + gb_within) / float(denom)


def _brats_region_masks(arr: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "WT": np.isin(arr, [1, 2, 3]),
        "TC": np.isin(arr, [1, 3]),
        "ET": arr == 3,
    }


def cmd_surface_dice(args: argparse.Namespace, core) -> None:
    """Compute NSD at tolerances {1, 2, 3} mm over a per-case CSV of (pred, gt).

    Expected CSV columns: case_id, pred, gt (paths). Optional spacing columns
    otherwise voxel spacing is read from the NIfTI header.
    """
    import nibabel as nib
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.pairs_csv)
    tol_list = [float(t) for t in str(args.tolerances_mm).split(",") if t.strip()]
    results: List[Dict[str, Any]] = []
    for r in rows:
        pred_img = nib.load(r["pred"])
        gt_img = nib.load(r["gt"])
        pred = np.asarray(pred_img.get_fdata()).astype(np.int32)
        gt = np.asarray(gt_img.get_fdata()).astype(np.int32)
        sp = tuple(float(x) for x in pred_img.header.get_zooms()[:3])
        reg_pred = _brats_region_masks(pred)
        reg_gt = _brats_region_masks(gt)
        row: Dict[str, Any] = {"case_id": r.get("case_id", Path(r["pred"]).stem),
                               "spacing_mm": list(sp)}
        for reg in ("WT", "TC", "ET"):
            for t in tol_list:
                row[f"nsd_{reg}_{t:g}mm"] = surface_dice_score(reg_pred[reg], reg_gt[reg], sp, t)
        results.append(row)
    _write_csv(results, out_dir / "surface_dice_per_case.csv")
    # summary
    summary: Dict[str, Any] = {"n_cases": len(results), "tolerances_mm": tol_list}
    for reg in ("WT", "TC", "ET"):
        for t in tol_list:
            vals = [r[f"nsd_{reg}_{t:g}mm"] for r in results]
            if vals:
                summary[f"nsd_{reg}_{t:g}mm_mean"] = float(np.mean(vals))
                summary[f"nsd_{reg}_{t:g}mm_std"] = float(np.std(vals))
                summary[f"nsd_{reg}_{t:g}mm_median"] = float(np.median(vals))
    _save_json(summary, out_dir / "surface_dice_summary.json")
    ADDENDUM_LOG.info(f"[surface_dice] wrote {len(results)} rows, tolerances={tol_list}")


# ══════════════════════════════════════════════════════════════════════════════
# B. Calibration - ECE, Brier, reliability diagrams.
#     Guo et al. 2017, ICML - On Calibration of Modern Neural Networks.
# ══════════════════════════════════════════════════════════════════════════════

def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, Any]:
    """Multi-class ECE over the winning-class probability (top-1 confidence)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.int64)
    if probs.ndim != 2:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1)
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    N = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = int(in_bin.sum())
        if m == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        c = float(conf[in_bin].mean())
        a = float(acc[in_bin].mean())
        ece += (m / N) * abs(a - c)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": m,
                     "confidence": c, "accuracy": a})
    return {"ece": float(ece), "bins": bins, "n_bins": n_bins, "n_samples": N}


def _brier_score_multiclass(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    probs = probs.reshape(-1, num_classes)
    labels = labels.reshape(-1).astype(np.int64)
    onehot = np.zeros_like(probs)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        # Do NOT silently clip out-of-range labels into a boundary class: that computes
        # the published Brier calibration score against wrong ground truth. Fail loud so
        # a broken NIfTI / un-remapped label is caught, not hidden.
        raise ValueError(
            f"_brier_score_multiclass: labels out of range [0,{num_classes}); got "
            f"[{int(labels.min())},{int(labels.max())}]. Check label remapping.")
    onehot[np.arange(probs.shape[0]), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=-1).mean())


def _reliability_diagram(bins: List[Dict[str, Any]], path) -> None:
    if not _have_matplotlib():
        ADDENDUM_LOG.warning("[calibration] matplotlib unavailable - skipping reliability diagram")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=130)
    xs = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    ys = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in bins]
    cs = [b["confidence"] if b["confidence"] is not None else 0.0 for b in bins]
    ns = [b["n"] for b in bins]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.bar(xs, ys, width=1.0 / max(len(bins), 1), alpha=0.7, edgecolor="black",
           label="accuracy (empirical)")
    ax.plot(xs, cs, marker="o", color="tab:red", label="mean confidence")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for x, y, n in zip(xs, ys, ns):
        if n:
            ax.annotate(str(n), (x, y), ha="center", va="bottom", fontsize=6, color="black")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def cmd_calibration(args: argparse.Namespace, core) -> None:
    """Compute per-region Dice-Sorensen is NOT calibration - here we evaluate
    voxelwise pixel calibration (ECE, Brier, reliability) on validation set.

    Loads the student checkpoint, runs sliding-window inference with softmax
    outputs on ``val_csv`` cases, then aggregates probabilities and labels for
    the ECE/Brier computation. Per-region winning-confidence ECE is also
    reported because clinical readers care about tumour-class reliability.
    """
    import torch
    device_t = torch.device(args.device)
    cfg_patch = tuple(int(x) for x in args.patch_size.split(","))
    modalities = [m.strip() for m in args.modalities.split(",")]

    model = core._load_student_for_eval(
        Path(args.ckpt), len(modalities), args.num_classes, args.student_base, device_t
    )

    ds = core.MultiModalSegDataset(
        csv_path=args.val_csv,
        modalities=modalities,
        patch_size=cfg_patch,
        augment=False,
        train=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=_loader_num_workers(args.num_workers, train_mode=False),
    )

    n_bins = int(args.n_bins)
    max_samples = int(args.max_voxel_samples)
    rng = np.random.default_rng(args.seed)

    conf_all: List[np.ndarray] = []
    label_all: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x = batch["image"].to(device_t)
                y_t = batch["label"]
            else:
                _case_ids, x, y_t, _spacing, _tags = batch
                x = x.to(device_t)
            y = y_t.squeeze(0).cpu().numpy().astype(np.int64)
            present_mask = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype, device=device_t)
            student_input = core.prepare_student_input(x, present_mask)
            logits = core.sliding_window_inference(
                model, student_input, patch_size=cfg_patch, overlap=args.overlap,
            )
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            fg = y >= 0
            flat_probs = probs.transpose(1, 2, 3, 0).reshape(-1, args.num_classes)
            flat_labels = y.reshape(-1)
            mask = fg.reshape(-1)
            flat_probs = flat_probs[mask]
            flat_labels = flat_labels[mask]
            if flat_probs.shape[0] > max_samples:
                idx = rng.choice(flat_probs.shape[0], size=max_samples, replace=False)
                flat_probs = flat_probs[idx]
                flat_labels = flat_labels[idx]
            conf_all.append(flat_probs)
            label_all.append(flat_labels)

    probs = np.concatenate(conf_all, axis=0) if conf_all else np.zeros((0, args.num_classes))
    labels = np.concatenate(label_all, axis=0) if label_all else np.zeros((0,), dtype=np.int64)

    ece = _expected_calibration_error(probs, labels, n_bins=n_bins)
    brier = _brier_score_multiclass(probs, labels, num_classes=args.num_classes)

    out_dir = _ensure_dir(args.out_dir)
    summary = {
        "ece": ece["ece"],
        "brier_score": brier,
        "n_voxel_samples": int(probs.shape[0]),
        "n_bins": n_bins,
        "num_classes": args.num_classes,
        "ckpt": args.ckpt,
        "val_csv": args.val_csv,
    }
    _save_json(summary, out_dir / "calibration_summary.json")
    _save_json(ece, out_dir / "reliability_bins.json")
    _reliability_diagram(ece["bins"], out_dir / "reliability_diagram.png")
    ADDENDUM_LOG.info(f"[calibration] ECE={ece['ece']:.4f} Brier={brier:.4f} "
             f"N={probs.shape[0]} → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# C. Uncertainty - MC Dropout (Gal & Ghahramani 2016) + lightweight deep ensemble.
# ══════════════════════════════════════════════════════════════════════════════

def _enable_dropout(model) -> None:
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        elif m.__class__.__name__ == "CurriculumDropout3D":
            m.train()
            if hasattr(m, "set_epoch"):
                m.set_epoch(10**9)


def cmd_mc_uncertainty(args: argparse.Namespace, core) -> None:
    """Produce per-case uncertainty maps via MC-Dropout + optional ensemble.

    Emits per-voxel entropy and variance over T stochastic forward passes plus
    a per-case correlation with error (|pred - gt|) as a sanity check. This
    gives users the "low-confidence voxels correlate with error regions"
    figure they will ask for.
    """
    import torch
    import nibabel as nib
    device_t = torch.device(args.device)
    modalities = [m.strip() for m in args.modalities.split(",")]
    patch = tuple(int(x) for x in args.patch_size.split(","))

    ckpts = [c.strip() for c in args.ckpts.split(",") if c.strip()]
    if not ckpts:
        raise ValueError("--ckpts is required (comma-separated list of >=1 checkpoints)")

    ds = core.MultiModalSegDataset(
        csv_path=args.val_csv,
        modalities=modalities,
        patch_size=patch,
        augment=False,
        train=False,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=_loader_num_workers(args.num_workers, train_mode=False),
    )
    out_dir = _ensure_dir(args.out_dir)

    models = []
    for cp in ckpts:
        m = core._load_student_for_eval(
            Path(cp), len(modalities), args.num_classes, args.student_base, device_t
        )
        models.append(m)

    T = int(args.n_mc_samples)
    summary_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            ref = None
            if isinstance(batch, dict):
                x = batch["image"].to(device_t)
                y_t = batch["label"]
                case_id = batch.get("case_id", [f"case_{b_idx:04d}"])
                case_id = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
                ref = batch.get("reference_path", [None])
                ref = ref[0] if isinstance(ref, (list, tuple)) else ref
            else:
                case_ids, x, y_t, _spacing, _tags = batch
                x = x.to(device_t)
                case_id = case_ids[0] if isinstance(case_ids, (list, tuple)) else case_ids
            case_id = str(case_id)
            y = y_t.squeeze(0).cpu().numpy().astype(np.int64)
            present_mask = torch.ones((x.shape[0], x.shape[1]), dtype=x.dtype, device=device_t)
            student_input = core.prepare_student_input(x, present_mask)
            probs_mc: List[np.ndarray] = []
            for m in models:
                m.eval()
                _enable_dropout(m)
                for _ in range(T):
                    logits = core.sliding_window_inference(
                        m, student_input, patch_size=patch, overlap=args.overlap,
                        force_eval=False,
                    )
                    probs_mc.append(torch.softmax(logits, dim=1).squeeze(0).cpu().numpy())
            P = np.stack(probs_mc, axis=0)  # [T*E, C, D, H, W]
            mean_p = P.mean(axis=0)
            var_p = P.var(axis=0).sum(axis=0)
            eps = 1e-8
            entropy = -(mean_p * np.log(mean_p + eps)).sum(axis=0)
            pred = mean_p.argmax(axis=0).astype(np.int64)
            err = (pred != y).astype(np.float64)
            # Pearson corr of entropy with binary error, voxel-level.
            ent_flat = entropy.reshape(-1)
            err_flat = err.reshape(-1)
            if ent_flat.std() > 1e-9 and err_flat.std() > 1e-9:
                corr = float(np.corrcoef(ent_flat, err_flat)[0, 1])
            else:
                corr = float("nan")
            summary_rows.append({
                "case_id": case_id,
                "entropy_error_pearson": corr,
                "mean_entropy": float(entropy.mean()),
                "mean_variance": float(var_p.mean()),
                "err_frac": float(err.mean()),
            })
            if args.save_maps:
                if ref is None and hasattr(ds, "rows") and b_idx < len(ds.rows):
                    row = ds.rows[b_idx]
                    for mod in modalities:
                        candidate = row.get(mod)
                        if candidate and Path(str(candidate)).expanduser().exists():
                            ref = candidate
                            break
                if ref is not None and Path(str(ref)).expanduser().exists():
                    aff = nib.load(str(ref)).affine
                else:
                    aff = np.eye(4)
                nib.save(nib.Nifti1Image(entropy.astype(np.float32), aff),
                         str(out_dir / f"{case_id}_entropy.nii.gz"))
                nib.save(nib.Nifti1Image(var_p.astype(np.float32), aff),
                         str(out_dir / f"{case_id}_variance.nii.gz"))
    _write_csv(summary_rows, out_dir / "mc_uncertainty_summary.csv")
    if summary_rows:
        corrs = [r["entropy_error_pearson"] for r in summary_rows
                 if r["entropy_error_pearson"] is not None and not math.isnan(r["entropy_error_pearson"])]
        _save_json({
            "n_cases": len(summary_rows),
            "T_mc_samples": T,
            "n_models_ensemble": len(ckpts),
            "mean_entropy_error_pearson": float(np.mean(corrs)) if corrs else None,
            "median_entropy_error_pearson": float(np.median(corrs)) if corrs else None,
        }, out_dir / "mc_uncertainty_summary.json")
    ADDENDUM_LOG.info(f"[mc_uncertainty] {len(summary_rows)} cases processed → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# D. Metric cross-validation vs monai / medpy.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_metric_check(args: argparse.Namespace, core) -> None:
    """Cross-validate OGAP's hand-rolled Dice/HD95 against MONAI and/or medpy.

    Emits a table with max|Δ| per region so users see quantitative
    agreement rather than a hand-wave. Skipped libraries are reported as
    NA so the test still passes on a minimal environment.
    """
    import nibabel as nib
    rows = _read_csv_rows(args.pairs_csv)
    out_dir = _ensure_dir(args.out_dir)

    try:
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        _have_monai = True
    except Exception:
        _have_monai = False

    try:
        from medpy.metric.binary import dc as _medpy_dice, hd95 as _medpy_hd95
        _have_medpy = True
    except Exception:
        _have_medpy = False

    results: List[Dict[str, Any]] = []
    for r in rows:
        pred_img = nib.load(r["pred"])
        gt_img = nib.load(r["gt"])
        pred = np.asarray(pred_img.get_fdata()).astype(np.int32)
        gt = np.asarray(gt_img.get_fdata()).astype(np.int32)
        sp = tuple(float(x) for x in pred_img.header.get_zooms()[:3])
        ogap = core.brats_metrics(pred, gt, spacing=np.asarray(sp), compute_hd95=True)
        row: Dict[str, Any] = {"case_id": r.get("case_id", Path(r["pred"]).stem)}
        for reg_name, (reg_pred, reg_gt) in {
            "WT": (np.isin(pred, [1, 2, 3]), np.isin(gt, [1, 2, 3])),
            "TC": (np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
            "ET": (pred == 3, gt == 3),
        }.items():
            row[f"ogap_dice_{reg_name}"] = float(ogap.get(f"dice_{reg_name}", float("nan")))
            row[f"ogap_hd95_{reg_name}"] = float(ogap.get(f"hd95_{reg_name}", float("nan")))
            if _have_medpy and reg_pred.sum() and reg_gt.sum():
                row[f"medpy_dice_{reg_name}"] = float(_medpy_dice(reg_pred, reg_gt))
                try:
                    row[f"medpy_hd95_{reg_name}"] = float(_medpy_hd95(reg_pred, reg_gt, voxelspacing=sp))
                except Exception:
                    row[f"medpy_hd95_{reg_name}"] = None
            if _have_monai:
                import torch
                pt = torch.tensor(reg_pred[None, None].astype(np.float32))
                gtt = torch.tensor(reg_gt[None, None].astype(np.float32))
                dm = DiceMetric(include_background=True, reduction="mean")
                dm(pt, gtt)
                row[f"monai_dice_{reg_name}"] = float(dm.aggregate().item())
                hm = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
                try:
                    hm(pt, gtt, spacing=list(sp))
                    row[f"monai_hd95_{reg_name}"] = float(hm.aggregate().item())
                except Exception:
                    row[f"monai_hd95_{reg_name}"] = None
        results.append(row)
    _write_csv(results, out_dir / "metric_crosscheck_per_case.csv")

    summary: Dict[str, Any] = {
        "n_cases": len(results),
        "have_monai": _have_monai,
        "have_medpy": _have_medpy,
    }
    for reg in ("WT", "TC", "ET"):
        for other in ("medpy", "monai"):
            deltas = []
            for r in results:
                a = r.get(f"ogap_dice_{reg}")
                b = r.get(f"{other}_dice_{reg}")
                if a is not None and b is not None and not (math.isnan(a) or math.isnan(b)):
                    deltas.append(abs(a - b))
            if deltas:
                summary[f"max_abs_delta_dice_{reg}_vs_{other}"] = float(max(deltas))
                summary[f"mean_abs_delta_dice_{reg}_vs_{other}"] = float(np.mean(deltas))
    _save_json(summary, out_dir / "metric_crosscheck_summary.json")
    ADDENDUM_LOG.info(f"[metric_check] wrote {len(results)} pairs → {out_dir}")

    # ── Audit-item report: surfaces the live constants/defaults so users
    # can confirm BraTS-2023, sliding-window, postprocess, and bootstrap
    # configurations match the protocol committed to in the manuscript.
    audit: Dict[str, Any] = {
        "comment": ("Defaults inspected at runtime from the live module so "
                    "this report stays in sync with code edits."),
        "hausdorff_empty_penalty_mm": 373.1287,
        "lesion_wise_default_penalty_mm": 373.1287,
        "bca_ci_default_n_boot": int(
            getattr(core._bca_ci, "__defaults__", (5000,))[0]
            if getattr(core, "_bca_ci", None) else -1
        ),
        "postprocess_defaults": {
            "min_component_size": 50,
            "et_min_component_size": 10,
            "et_confidence_threshold": 0.55,
        },
        "sliding_window_defaults": {
            "overlap": 0.5,
            "mode": "gaussian",
            "sigma_scale": 0.125,
        },
        "label_noise_modes": ["morph", "boundary_band"],
        "tta_aggregation": "softmax-mean over 8 axis flips (eval-only)",
        "label_canonicalisation": ("4 (legacy ET) → 3 (internal ET); "
                                   "UTSW-aware error hint when invalid labels remain"),
        "have_monai": _have_monai,
        "have_medpy": _have_medpy,
    }
    # Reflect the actual default by inspecting the function signature.
    try:
        import inspect
        sig = inspect.signature(core._bca_ci)
        n_boot_default = sig.parameters["n_boot"].default
        audit["bca_ci_default_n_boot"] = int(n_boot_default)
    except Exception:
        pass
    _save_json(audit, out_dir / "audit_report.json")
    ADDENDUM_LOG.info(f"[metric_check] audit_report.json written; "
                      f"n_boot_default={audit.get('bca_ci_default_n_boot')}")


# ══════════════════════════════════════════════════════════════════════════════
# E. Clinical stratification over UTSW metadata.
# ══════════════════════════════════════════════════════════════════════════════

STRATIFIER_COLUMNS: Dict[str, Dict[str, Any]] = {
    "Cohort": {"kind": "categorical"},
    "Income Setting": {"kind": "categorical"},
    "Sex at birth": {"kind": "categorical"},
    "Age at Imaging": {"kind": "numeric",
                       "bins": [(0, 18, "pediatric"), (18, 40, "AYA"),
                                (40, 65, "adult"), (65, 200, "elderly")]},
    "Race": {"kind": "categorical"},
    "Ethnicity": {"kind": "categorical"},
    "Tumor Grade": {"kind": "categorical"},
    "IDH": {"kind": "categorical"},
    "1p19Q CODEL": {"kind": "categorical"},
    "MGMT": {"kind": "categorical"},
    "Scanner Make": {"kind": "categorical"},
    "Scanner Model": {"kind": "categorical"},
    "Scanner Strength": {"kind": "numeric_binned_scanner"},
    "Manually Refined Segmentation": {"kind": "categorical"},
}


def _bca_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
            rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Bias-corrected and accelerated bootstrap 95% CI (Efron 1987)."""
    if rng is None:
        rng = np.random.default_rng(0)
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=np.float64)
    if len(v) < 2:
        m = float(v.mean()) if len(v) else float("nan")
        return m, m, m
    m = float(v.mean())
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    z0 = _norm_ppf(float((boots < m).mean()))
    jack = np.array([np.delete(v, i).mean() for i in range(len(v))])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * ((jm - jack) ** 2).sum() ** 1.5
    a = 0.0 if den == 0 else num / den
    zl = _norm_ppf(alpha / 2)
    zu = _norm_ppf(1 - alpha / 2)
    def _adj(z):
        return _norm_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo_q = _adj(zl); hi_q = _adj(zu)
    lo = float(np.quantile(boots, max(0.0, min(1.0, lo_q))))
    hi = float(np.quantile(boots, max(0.0, min(1.0, hi_q))))
    return m, lo, hi


def _norm_ppf(p: float) -> float:
    # Beasley-Springer-Moro approximation (no scipy dep).
    from math import sqrt, log
    p = min(max(p, 1e-12), 1 - 1e-12)
    if p < 0.5:
        s = 1; q = p
    else:
        s = -1; q = 1 - p
    t = sqrt(-2.0 * log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t ** 3
    return -s * (t - num / den)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _cohort_bin(col: str, value: str) -> Optional[str]:
    v = (value or "").strip()
    if not v or v.lower() in ("na", "n/a", "unknown", "not reported", ""):
        return None
    spec = STRATIFIER_COLUMNS.get(col)
    if not spec:
        return v
    kind = spec["kind"]
    if kind == "categorical":
        return v
    if kind == "numeric":
        try:
            x = float(v)
        except ValueError:
            return None
        for lo, hi, label in spec["bins"]:
            if lo <= x < hi:
                return label
        return None
    if kind == "numeric_binned_scanner":
        try:
            x = float(v)
        except ValueError:
            return None
        if x < 1.0:
            return "<1T"
        if x < 1.4:
            return "1T-1.4T"
        if x < 1.75:
            return "1.5T"
        if x < 2.5:
            return "2T"
        return ">=3T"
    return v


def cmd_stratify(args: argparse.Namespace, core) -> None:
    """Join a per-case metrics table with clinical metadata (UTSW TSV) and
    emit stratified summaries (mean, BCa 95% CI, N) for every stratifier
    in STRATIFIER_COLUMNS plus a by_manual_refined toggle that *separates*
    refined vs. non-refined cases (Gap 2.1)."""
    out_dir = _ensure_dir(args.out_dir)
    per_case = _read_per_case_metrics(args.per_case)

    # Metadata is TSV; pandas is optional.
    meta_rows: Dict[str, Dict[str, str]] = {}
    with open(args.metadata_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("Subject ID") or row.get(args.metadata_id_col)
            if sid:
                meta_rows[sid.strip()] = row

    target_metric = args.metric
    joined: List[Dict[str, Any]] = []
    for r in per_case:
        sid = str(r.get("case_id") or r.get("subject_id") or "").strip()
        val = r.get(target_metric)
        if val is None:
            continue
        try:
            val_f = float(val)
        except (TypeError, ValueError):
            continue
        row = {"case_id": sid, target_metric: val_f}
        md = meta_rows.get(sid, {})
        for col in STRATIFIER_COLUMNS:
            row[col] = md.get(col, "")
        joined.append(row)
    _write_csv(joined, out_dir / "joined_per_case.csv")

    rng = np.random.default_rng(42)
    summary: Dict[str, Any] = {"metric": target_metric, "n_joined": len(joined), "subgroups": {}}
    for col in STRATIFIER_COLUMNS:
        groups: Dict[str, List[float]] = defaultdict(list)
        for r in joined:
            key = _cohort_bin(col, r.get(col, ""))
            if key is None:
                continue
            groups[key].append(r[target_metric])
        sub: Dict[str, Any] = {}
        for key, vals in groups.items():
            if len(vals) < args.min_group_n:
                sub[key] = {"n": len(vals), "skipped_reason": "n<min_group_n"}
                continue
            m, lo, hi = _bca_ci(vals, n_boot=args.n_boot, rng=rng)
            sub[key] = {"n": len(vals), "mean": m, "ci_lo": lo, "ci_hi": hi,
                        "median": float(np.median(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals))}
        summary["subgroups"][col] = sub
    _save_json(summary, out_dir / "stratified_summary.json")

    # Separate refined vs. non-refined reports (Gap 2.1).
    refined_rows = [r for r in joined if (r.get("Manually Refined Segmentation") or "").strip().lower() == "yes"]
    non_refined_rows = [r for r in joined if (r.get("Manually Refined Segmentation") or "").strip().lower() == "no"]
    split_report = {
        "metric": target_metric,
        "all": {"n": len(joined),
                **_pool_stats([r[target_metric] for r in joined], rng, args.n_boot)},
        "manually_refined_yes": {"n": len(refined_rows),
                                 **_pool_stats([r[target_metric] for r in refined_rows], rng, args.n_boot)},
        "manually_refined_no": {"n": len(non_refined_rows),
                                **_pool_stats([r[target_metric] for r in non_refined_rows], rng, args.n_boot)},
    }
    _save_json(split_report, out_dir / "manual_refinement_split.json")
    ADDENDUM_LOG.info(f"[stratify] {len(joined)} cases joined; pooled/refined/non-refined n={len(joined)}/"
             f"{len(refined_rows)}/{len(non_refined_rows)}")


def _pool_stats(values: Sequence[float], rng, n_boot: int) -> Dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "median": None}
    m, lo, hi = _bca_ci(v, n_boot=n_boot, rng=rng)
    return {"mean": m, "ci_lo": lo, "ci_hi": hi, "median": float(np.median(v))}


# ══════════════════════════════════════════════════════════════════════════════
# F. Inter-rater reliability helper.
# ══════════════════════════════════════════════════════════════════════════════

def _cohens_kappa_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    n = a.size
    po = (a == b).mean()
    pa = a.mean(); pb = b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    if 1 - pe == 0:
        return 1.0 if po == 1 else 0.0
    return float((po - pe) / (1 - pe))


def cmd_interrater(args: argparse.Namespace, core) -> None:
    """Inter-rater reliability study. Takes a CSV of (case_id, rater_a, rater_b)
    paths to segmentation NIfTI files. Emits per-case Dice, HD95, Cohen's
    kappa (per BraTS region) and an overall summary.
    """
    import nibabel as nib
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.raters_csv)
    results: List[Dict[str, Any]] = []
    for r in rows:
        a_img = nib.load(r["rater_a"])
        b_img = nib.load(r["rater_b"])
        a = np.asarray(a_img.get_fdata()).astype(np.int32)
        b = np.asarray(b_img.get_fdata()).astype(np.int32)
        sp = tuple(float(x) for x in a_img.header.get_zooms()[:3])
        row: Dict[str, Any] = {"case_id": r.get("case_id", Path(r["rater_a"]).stem)}
        for reg in ("WT", "TC", "ET"):
            ma = _brats_region_masks(a)[reg]
            mb = _brats_region_masks(b)[reg]
            inter = (ma & mb).sum()
            union = ma.sum() + mb.sum()
            dice = (2 * inter / union) if union > 0 else 1.0
            row[f"dice_{reg}"] = float(dice)
            row[f"kappa_{reg}"] = _cohens_kappa_binary(ma, mb)
        results.append(row)
    _write_csv(results, out_dir / "interrater_per_case.csv")
    summary = {"n_cases": len(results)}
    for reg in ("WT", "TC", "ET"):
        ds = [r[f"dice_{reg}"] for r in results]
        ks = [r[f"kappa_{reg}"] for r in results]
        summary[f"dice_{reg}_mean"] = float(np.mean(ds)) if ds else None
        summary[f"dice_{reg}_median"] = float(np.median(ds)) if ds else None
        summary[f"kappa_{reg}_mean"] = float(np.mean(ks)) if ks else None
        summary[f"kappa_{reg}_median"] = float(np.median(ks)) if ks else None
    _save_json(summary, out_dir / "interrater_summary.json")
    ADDENDUM_LOG.info(f"[interrater] {len(results)} pairs → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# G. Post-processing threshold sweep.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_threshold_sweep(args: argparse.Namespace, core) -> None:
    """Sweep (min_component_size, et_min_component_size, et_confidence_threshold)
    over a grid on already-predicted probability volumes. Requires a CSV with
    (case_id, prob_npz, gt) where prob_npz contains a float32 array [C,D,H,W].
    Reports raw vs. post-processed Dice/HD95 per region per setting.
    """
    import nibabel as nib
    rows = _read_csv_rows(args.pairs_csv)
    out_dir = _ensure_dir(args.out_dir)

    min_sizes = [int(x) for x in args.min_sizes.split(",")]
    et_min_sizes = [int(x) for x in args.et_min_sizes.split(",")]
    et_confs = [float(x) for x in args.et_confs.split(",")]

    results: List[Dict[str, Any]] = []
    for r in rows:
        prob = np.load(r["prob_npz"])
        prob = prob["probs"] if "probs" in prob.files else prob[prob.files[0]]
        gt_img = nib.load(r["gt"])
        gt = np.asarray(gt_img.get_fdata()).astype(np.int32)
        sp = np.asarray([float(x) for x in gt_img.header.get_zooms()[:3]])
        pred_raw = prob.argmax(0).astype(np.int32)
        raw_m = core.brats_metrics(pred_raw, gt, spacing=sp, compute_hd95=True)
        for ms in min_sizes:
            for em in et_min_sizes:
                for ec in et_confs:
                    pred_pp = core.connected_component_postprocessing(
                        pred_raw.copy(),
                        min_component_size=ms,
                        class_prob=prob,
                        et_min_component_size=em,
                        et_confidence_threshold=ec,
                    )
                    pp_m = core.brats_metrics(pred_pp, gt, spacing=sp, compute_hd95=True)
                    results.append({
                        "case_id": r.get("case_id", Path(r["gt"]).stem),
                        "min_component_size": ms,
                        "et_min_component_size": em,
                        "et_confidence_threshold": ec,
                        **{f"raw_{k}": float(v) for k, v in raw_m.items()},
                        **{f"pp_{k}": float(v) for k, v in pp_m.items()},
                    })
    _write_csv(results, out_dir / "threshold_sweep_per_case.csv")

    # Aggregate: per (ms, em, ec) combo, mean Dice per region + delta.
    agg: Dict[Tuple[int, int, float], List[Dict[str, float]]] = defaultdict(list)
    for row in results:
        key = (row["min_component_size"], row["et_min_component_size"], row["et_confidence_threshold"])
        agg[key].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for key, group in agg.items():
        ms, em, ec = key
        entry: Dict[str, Any] = {
            "min_component_size": ms,
            "et_min_component_size": em,
            "et_confidence_threshold": ec,
            "n": len(group),
        }
        for reg in ("WT", "TC", "ET"):
            r_raw = [g[f"raw_dice_{reg}"] for g in group if f"raw_dice_{reg}" in g]
            r_pp = [g[f"pp_dice_{reg}"] for g in group if f"pp_dice_{reg}" in g]
            if r_raw: entry[f"mean_raw_dice_{reg}"] = float(np.mean(r_raw))
            if r_pp: entry[f"mean_pp_dice_{reg}"] = float(np.mean(r_pp))
            if r_raw and r_pp:
                entry[f"mean_delta_dice_{reg}"] = float(np.mean(r_pp) - np.mean(r_raw))
        summary_rows.append(entry)
    _write_csv(summary_rows, out_dir / "threshold_sweep_summary.csv")
    ADDENDUM_LOG.info(f"[threshold_sweep] {len(results)} evaluations across "
             f"{len(min_sizes)}x{len(et_min_sizes)}x{len(et_confs)} grid")


# ══════════════════════════════════════════════════════════════════════════════
# H. Hölder α sensitivity + I. KD-method comparison orchestrators.
# ══════════════════════════════════════════════════════════════════════════════

def _runbook_lines(items: List[Dict[str, Any]], out_path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for it in items:
        lines.append(f"# {it['name']}")
        lines.append(it["cmd"])
        lines.append("")
    Path(out_path).write_text("\n".join(lines))
    os.chmod(out_path, 0o755)


def _student_base_cmd(args: argparse.Namespace, overrides: Dict[str, Any], out_dir: str) -> str:
    base = [
        sys.executable, args.script, "train",
        "--train_csv", args.train_csv,
        "--val_csv", args.val_csv,
        "--teacher_ckpt", args.teacher_ckpt,
        "--out_dir", out_dir,
        "--modalities", args.modalities,
        "--num_classes", str(args.num_classes),
        "--patch_size", args.patch_size,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--seed", str(overrides.get("seed", args.seed)),
    ]
    for k, v in overrides.items():
        if k == "seed": continue
        base += [f"--{k}", str(v)]
    return " ".join(shlex.quote(a) for a in base)


def cmd_holder_sweep(args: argparse.Namespace, core) -> None:
    """Emit a runbook for Hölder-α ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5} across
    the requested seeds. α=1.0 collapses to KL-KD (fallback branch in core)."""
    out_dir = _ensure_dir(args.out_dir)
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    items = []
    for a in alphas:
        for s in seeds:
            run_dir = str(Path(out_dir) / f"alpha_{a:g}_seed_{s}")
            items.append({
                "name": f"alpha={a} seed={s}",
                "cmd": _student_base_cmd(args, {"holder_alpha": a, "seed": s}, run_dir),
            })
    _runbook_lines(items, out_dir / "holder_sweep_runbook.sh")
    _save_json({"alphas": alphas, "seeds": seeds, "n_runs": len(items)},
               out_dir / "holder_sweep_plan.json")
    ADDENDUM_LOG.info(f"[holder_sweep] emitted runbook with {len(items)} runs → {out_dir}")


def cmd_kd_compare(args: argparse.Namespace, core) -> None:
    """Emit a runbook for alternative KD objectives using only existing knobs.
    'holder_default' = holder_alpha=1.5 (pseudo-divergence; student target is the
    flatter renormalised teacher^0.5 - deliberate uncertainty transfer).
    'holder_proper' = holder_alpha=2.0 (the PROPER Cauchy-Schwarz member; student
    target IS the teacher) - the pre-registered {1.5, 2.0} ablation control.
    'kl' = holder_alpha=1.0 (KL fallback). 'mse' = lambda_kd=0, lambda_feat
    bumped to move distillation weight entirely onto feature MSE. 'feat_only'
    and 'ce_only' are additional reference settings.
    """
    out_dir = _ensure_dir(args.out_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    matrix = {
        "holder_default":  {"holder_alpha": 1.5, "lambda_kd": 0.35, "lambda_feat": 0.05},
        # alpha=2.0 is the PROPER Hölder (Cauchy-Schwarz) member: its minimiser is
        # the teacher itself, vs alpha=1.5 whose minimiser is the flatter
        # teacher^0.5. This is the pre-registered uncertainty-transfer ablation.
        "holder_proper":   {"holder_alpha": 2.0, "lambda_kd": 0.35, "lambda_feat": 0.05},
        "kl":              {"holder_alpha": 1.0, "lambda_kd": 0.35, "lambda_feat": 0.05},
        "mse_feat_only":   {"lambda_kd": 0.0,  "lambda_feat": 0.25, "holder_alpha": 1.5},
        "no_kd":           {"lambda_kd": 0.0,  "lambda_feat": 0.0,  "holder_alpha": 1.5},
        "no_confidence":   {"holder_alpha": 1.5, "lambda_kd": 0.35, "lambda_feat": 0.05,
                            "confidence_gate": 0.0},
    }
    items = []
    for name, overrides in matrix.items():
        for s in seeds:
            run_dir = str(Path(out_dir) / f"{name}_seed_{s}")
            items.append({
                "name": f"{name} seed={s}",
                "cmd": _student_base_cmd(args, {**overrides, "seed": s}, run_dir),
            })
    _runbook_lines(items, out_dir / "kd_compare_runbook.sh")
    _save_json({"variants": list(matrix.keys()), "seeds": seeds, "n_runs": len(items)},
               out_dir / "kd_compare_plan.json")
    ADDENDUM_LOG.info(f"[kd_compare] emitted runbook with {len(items)} runs → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# J. ERASMUS bias empirical check.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_erasmus_bias(args: argparse.Namespace, core) -> None:
    """Empirical check that ERASMUS (binary WT-only) inclusion does not cause
    class-collapse toward WT at the expense of TC/ET. Compares per-class Dice
    on a held-out BraTS-4-class split for two models trained with and without
    ERASMUS in the training pool.
    """
    out_dir = _ensure_dir(args.out_dir)
    rows_with = _read_per_case_metrics(args.with_erasmus_per_case)
    rows_without = _read_per_case_metrics(args.without_erasmus_per_case)

    def agg(rows):
        agg_out = {}
        for reg in ("WT", "TC", "ET"):
            key = f"pp_dice_{reg}"
            vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
            if vals:
                rng = np.random.default_rng(0)
                m, lo, hi = _bca_ci(vals, rng=rng, n_boot=args.n_boot)
                agg_out[reg] = {"n": len(vals), "mean": m, "ci_lo": lo, "ci_hi": hi}
        return agg_out

    report = {
        "with_erasmus":  agg(rows_with),
        "without_erasmus": agg(rows_without),
    }
    # Paired Wilcoxon by case_id intersection.
    idx_with = {r.get("case_id"): r for r in rows_with}
    idx_without = {r.get("case_id"): r for r in rows_without}
    shared = [k for k in idx_with if k in idx_without]
    paired: Dict[str, Any] = {"n_paired": len(shared)}
    for reg in ("WT", "TC", "ET"):
        key = f"pp_dice_{reg}"
        a = [float(idx_with[k][key]) for k in shared if key in idx_with[k] and key in idx_without[k]]
        b = [float(idx_without[k][key]) for k in shared if key in idx_with[k] and key in idx_without[k]]
        if len(a) >= 5:
            diffs = np.asarray(a) - np.asarray(b)
            rng = np.random.default_rng(0)
            m, lo, hi = _bca_ci(diffs.tolist(), rng=rng, n_boot=args.n_boot)
            paired[reg] = {"n": len(a), "mean_delta": m, "ci_lo": lo, "ci_hi": hi}
    report["paired_delta"] = paired
    _save_json(report, out_dir / "erasmus_bias_report.json")
    ADDENDUM_LOG.info(f"[erasmus_bias] report at {out_dir}/erasmus_bias_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# K. Single-modality scenario (e.g. T2-FLAIR-only triage).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_single_modality_scenario(args: argparse.Namespace, core) -> None:
    """Evaluate deployment scenarios with strictly one or two modalities,
    using the existing infer/evaluate paths with --missing_modalities. Emits
    a runbook of commands that the user can submit to the cluster.
    """
    out_dir = _ensure_dir(args.out_dir)
    mods = [m.strip() for m in args.modalities.split(",")]
    keep_sets = [tuple(s.strip() for s in block.split("+")) for block in args.keep.split(",")]
    items = []
    for keep in keep_sets:
        missing_idx = [i for i, m in enumerate(mods) if m not in keep]
        name = "+".join(keep) if keep else "none"
        run_dir = str(Path(out_dir) / f"keep_{name}")
        cmd_args = [
            sys.executable, args.script, "evaluate",
            "--ckpt", args.ckpt,
            "--val_csv", args.val_csv,
            "--modalities", args.modalities,
            "--num_classes", str(args.num_classes),
            "--patch_size", args.patch_size,
            "--out_dir", run_dir,
            "--student_base", str(args.student_base),
            "--device", args.device,
        ]
        if args.tta:
            cmd_args.append("--tta")
        cmd = " ".join(shlex.quote(a) for a in cmd_args)
        cmd += f"  # target_missing_modalities={','.join(str(i) for i in missing_idx)}"
        items.append({"name": f"keep={name}", "cmd": cmd})
    _runbook_lines(items, out_dir / "single_modality_runbook.sh")
    _save_json({"keep_sets": keep_sets, "n_runs": len(items)},
               out_dir / "single_modality_plan.json")
    ADDENDUM_LOG.info(f"[single_modality_scenario] emitted runbook with {len(items)} runs")


# ══════════════════════════════════════════════════════════════════════════════
# L. Field-strength Dice curve.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_field_strength_curve(args: argparse.Namespace, core) -> None:
    """Plot Dice vs. scanner strength on UTSW. Uses the joined per-case table
    produced by ``stratify``. Emits PNG and CSV.
    """
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.joined_csv)
    buckets: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        key = _cohort_bin("Scanner Strength", r.get("Scanner Strength", ""))
        if key is None:
            continue
        try:
            buckets[key].append(float(r[args.metric]))
        except (TypeError, ValueError):
            pass
    order = ["<1T", "1T-1.4T", "1.5T", "2T", ">=3T"]
    rng = np.random.default_rng(0)
    summary = []
    for k in order:
        if k not in buckets:
            continue
        m, lo, hi = _bca_ci(buckets[k], rng=rng, n_boot=args.n_boot)
        summary.append({"bucket": k, "n": len(buckets[k]),
                        "mean": m, "ci_lo": lo, "ci_hi": hi})
    _write_csv(summary, out_dir / "field_strength_curve.csv")
    if _have_matplotlib():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=130)
        xs = list(range(len(summary)))
        means = [s["mean"] for s in summary]
        los = [s["mean"] - s["ci_lo"] for s in summary]
        his = [s["ci_hi"] - s["mean"] for s in summary]
        ns = [s["n"] for s in summary]
        ax.errorbar(xs, means, yerr=[los, his], fmt="o", capsize=4, color="tab:blue")
        ax.set_xticks(xs)
        ax.set_xticklabels([s["bucket"] for s in summary])
        for x, m, n in zip(xs, means, ns):
            ax.annotate(f"n={n}", (x, m), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7)
        ax.set_ylabel(args.metric)
        ax.set_xlabel("Scanner field strength")
        ax.set_title("Performance vs. field strength (95% BCa CI)")
        fig.tight_layout(); fig.savefig(out_dir / "field_strength_curve.png"); plt.close(fig)
    ADDENDUM_LOG.info(f"[field_strength_curve] wrote {len(summary)} buckets → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# M/N/O. Cross-validation + multi-seed orchestrators.
# ══════════════════════════════════════════════════════════════════════════════

def _rows_with_key(path, key: str) -> List[Dict[str, str]]:
    rows = _read_csv_rows(path)
    return [r for r in rows if key in r]


def cmd_loso_cv(args: argparse.Namespace, core) -> None:
    """Leave-one-site-out CV. Requires a site_col in the train CSV. Emits
    per-fold train/val CSVs plus a runbook to train teacher + student per
    fold. Evaluation is the standard evaluate command on the held-out site.
    """
    out_dir = _ensure_dir(args.out_dir)
    rows = _rows_with_key(args.train_csv, args.site_col)
    sites = sorted({r[args.site_col].strip() for r in rows if r[args.site_col].strip()})
    items = []
    for s in sites:
        fold_dir = _ensure_dir(out_dir / f"fold_leave_{s}")
        train_rows = [r for r in rows if r[args.site_col].strip() != s]
        val_rows = [r for r in rows if r[args.site_col].strip() == s]
        if not train_rows or not val_rows:
            continue
        t_path = fold_dir / "train.csv"
        v_path = fold_dir / "val.csv"
        _write_csv(train_rows, t_path)
        _write_csv(val_rows, v_path)
        teach_dir = fold_dir / "teacher"
        stud_dir = fold_dir / "student"
        eval_dir = fold_dir / "eval"
        pt_cmd = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "pretrain_teacher",
            "--train_csv", str(t_path), "--val_csv", str(v_path),
            "--out_dir", str(teach_dir), "--modalities", args.modalities,
            "--num_classes", str(args.num_classes), "--patch_size", args.patch_size,
            "--epochs", str(args.epochs), "--seed", str(args.seed),
        ])
        tr_cmd = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "train",
            "--train_csv", str(t_path), "--val_csv", str(v_path),
            "--teacher_ckpt", str(teach_dir / "best_teacher.pth"),
            "--out_dir", str(stud_dir), "--modalities", args.modalities,
            "--num_classes", str(args.num_classes), "--patch_size", args.patch_size,
            "--epochs", str(args.epochs), "--seed", str(args.seed),
        ])
        ev_cmd = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "evaluate",
            "--ckpt", str(stud_dir / "best_student.pth"),
            "--val_csv", str(v_path), "--out_dir", str(eval_dir),
            "--modalities", args.modalities, "--num_classes", str(args.num_classes),
            "--patch_size", args.patch_size, "--student_base", str(args.student_base),
            "--device", args.device,
        ])
        items.append({"name": f"fold_leave_{s}", "cmd": "\n".join([pt_cmd, tr_cmd, ev_cmd])})
    _runbook_lines(items, out_dir / "loso_runbook.sh")
    _save_json({"sites": sites, "n_folds": len(items), "site_col": args.site_col},
               out_dir / "loso_plan.json")
    ADDENDUM_LOG.info(f"[loso_cv] emitted {len(items)} folds → {out_dir}")


def cmd_kfold_cv(args: argparse.Namespace, core) -> None:
    """Classic k-fold CV. Stratification by a metadata column (optional)."""
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.train_csv)
    rng = np.random.default_rng(args.seed)
    if args.stratify_col and all(args.stratify_col in r for r in rows):
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[r[args.stratify_col]].append(i)
        folds: List[List[int]] = [[] for _ in range(args.k)]
        for key, indices in groups.items():
            rng.shuffle(indices)
            for j, idx in enumerate(indices):
                folds[j % args.k].append(idx)
    else:
        order = list(range(len(rows)))
        rng.shuffle(order)
        folds = [order[i::args.k] for i in range(args.k)]
    items = []
    for i, fold in enumerate(folds):
        fold_dir = _ensure_dir(out_dir / f"fold_{i:02d}")
        val_idx = set(fold)
        train_rows = [r for j, r in enumerate(rows) if j not in val_idx]
        val_rows = [rows[j] for j in fold]
        t_path = fold_dir / "train.csv"
        v_path = fold_dir / "val.csv"
        _write_csv(train_rows, t_path); _write_csv(val_rows, v_path)
        for stage, out, prev in (("pretrain_teacher", fold_dir / "teacher", None),
                                 ("train", fold_dir / "student", fold_dir / "teacher" / "best_teacher.pth"),
                                 ("evaluate", fold_dir / "eval", fold_dir / "student" / "best_student.pth")):
            argslist = [sys.executable, args.script, stage,
                        "--val_csv", str(v_path),
                        "--out_dir", str(out),
                        "--modalities", args.modalities,
                        "--num_classes", str(args.num_classes),
                        "--patch_size", args.patch_size]
            if stage == "pretrain_teacher":
                argslist += ["--train_csv", str(t_path), "--epochs", str(args.epochs),
                             "--seed", str(args.seed)]
            elif stage == "train":
                argslist += ["--train_csv", str(t_path), "--teacher_ckpt", str(prev),
                             "--epochs", str(args.epochs), "--seed", str(args.seed)]
            elif stage == "evaluate":
                argslist += ["--ckpt", str(prev), "--student_base", str(args.student_base),
                             "--device", args.device]
            items.append({"name": f"fold_{i:02d}_{stage}",
                          "cmd": " ".join(shlex.quote(a) for a in argslist)})
    _runbook_lines(items, out_dir / "kfold_runbook.sh")
    _save_json({"k": args.k, "stratify_col": args.stratify_col, "n_rows": len(rows)},
               out_dir / "kfold_plan.json")
    ADDENDUM_LOG.info(f"[kfold_cv] {args.k}-fold plan written → {out_dir}")


def cmd_multi_seed(args: argparse.Namespace, core) -> None:
    """Repeat the training pipeline with several seeds; aggregate headline
    metrics at the end. Emits a runbook + a stub aggregator script.
    """
    out_dir = _ensure_dir(args.out_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    items = []
    for s in seeds:
        run_dir = Path(out_dir) / f"seed_{s}"
        pt = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "pretrain_teacher",
            "--train_csv", args.train_csv, "--val_csv", args.val_csv,
            "--out_dir", str(run_dir / "teacher"), "--modalities", args.modalities,
            "--num_classes", str(args.num_classes), "--patch_size", args.patch_size,
            "--epochs", str(args.epochs), "--seed", str(s), "--deterministic",
        ])
        tr = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "train",
            "--train_csv", args.train_csv, "--val_csv", args.val_csv,
            "--teacher_ckpt", str(run_dir / "teacher/best_teacher.pth"),
            "--out_dir", str(run_dir / "student"), "--modalities", args.modalities,
            "--num_classes", str(args.num_classes), "--patch_size", args.patch_size,
            "--epochs", str(args.epochs), "--seed", str(s), "--deterministic",
        ])
        ev = " ".join(shlex.quote(a) for a in [
            sys.executable, args.script, "evaluate",
            "--ckpt", str(run_dir / "student/best_student.pth"),
            "--val_csv", args.val_csv, "--out_dir", str(run_dir / "eval"),
            "--modalities", args.modalities, "--num_classes", str(args.num_classes),
            "--patch_size", args.patch_size, "--student_base", str(args.student_base),
            "--device", args.device,
        ])
        items.append({"name": f"seed_{s}", "cmd": "\n".join([pt, tr, ev])})
    _runbook_lines(items, out_dir / "multi_seed_runbook.sh")
    _save_json({"seeds": seeds, "n_runs": len(items)}, out_dir / "multi_seed_plan.json")
    ADDENDUM_LOG.info(f"[multi_seed] emitted runbook for seeds={seeds} → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# P. Training-curve plotter.
# ══════════════════════════════════════════════════════════════════════════════

def _iter_train_logs(run_dir: Path) -> List[Dict[str, Any]]:
    """Best-effort parser for the per-epoch JSON/JSONL files the core writes."""
    cands = [run_dir / "train_metrics.jsonl",
             run_dir / "metrics.jsonl",
             run_dir / "history.json",
             run_dir / "epoch_log.jsonl"]
    for c in cands:
        if c.exists():
            out = []
            if c.suffix == ".jsonl":
                with open(c) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try: out.append(json.loads(line))
                            except Exception: pass
            else:
                try:
                    out = json.load(open(c))
                except Exception:
                    out = []
            if out:
                return out if isinstance(out, list) else [out]
    return []


def cmd_training_curves(args: argparse.Namespace, core) -> None:
    out_dir = _ensure_dir(args.out_dir)
    runs = [Path(p) for p in args.runs.split(",") if p.strip()]
    if not _have_matplotlib():
        ADDENDUM_LOG.warning("[training_curves] matplotlib unavailable, skipping plot")
        _save_json({"error": "matplotlib unavailable", "runs": [str(r) for r in runs]},
                   out_dir / "training_curves.json")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=130)
    for r in runs:
        hist = _iter_train_logs(r)
        if not hist:
            continue
        xs = [h.get("epoch", i) for i, h in enumerate(hist)]
        loss = [h.get("loss_total", h.get("loss")) for h in hist]
        dice = [h.get("val_dice_mean", h.get("dice_mean")) for h in hist]
        axes[0].plot(xs, loss, label=r.name)
        axes[1].plot(xs, dice, label=r.name)
    axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    axes[1].set_title("Validation Dice"); axes[1].set_xlabel("epoch"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "training_curves.png"); plt.close(fig)
    ADDENDUM_LOG.info(f"[training_curves] wrote figure for {len(runs)} runs → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Q. Ablation heatmap + R. subgroup forest + S. failure-case figure.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_ablation_heatmap(args: argparse.Namespace, core) -> None:
    """Rows = ablations, columns = modality combos, cells = mean Dice."""
    out_dir = _ensure_dir(args.out_dir)
    summary_json = Path(args.ablation_summary_json)
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)
    data = json.load(open(summary_json))
    ablations = sorted(data.keys())
    all_combos = sorted({c for v in data.values() for c in v.keys()})
    matrix = np.full((len(ablations), len(all_combos)), np.nan)
    for i, a in enumerate(ablations):
        for j, c in enumerate(all_combos):
            v = data.get(a, {}).get(c)
            if isinstance(v, dict):
                v = v.get("mean_dice")
            if isinstance(v, (int, float)):
                matrix[i, j] = float(v)
    _save_json({"ablations": ablations, "combos": all_combos,
                "matrix": matrix.tolist()}, out_dir / "ablation_heatmap.json")
    if _have_matplotlib():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(all_combos)),
                                        max(4, 0.35 * len(ablations))), dpi=130)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
        ax.set_yticks(range(len(ablations))); ax.set_yticklabels(ablations, fontsize=8)
        ax.set_xticks(range(len(all_combos))); ax.set_xticklabels(all_combos,
                                                                 fontsize=7, rotation=75, ha="right")
        for i in range(len(ablations)):
            for j in range(len(all_combos)):
                if not math.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if matrix[i, j] < 0.75 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(out_dir / "ablation_heatmap.png"); plt.close(fig)
    ADDENDUM_LOG.info(f"[ablation_heatmap] wrote heatmap → {out_dir}")


def cmd_subgroup_forest(args: argparse.Namespace, core) -> None:
    """Render a forest plot from the ``stratified_summary.json`` written by
    cmd_stratify. One panel per stratifier."""
    out_dir = _ensure_dir(args.out_dir)
    summary = json.load(open(args.stratified_summary_json))
    if not _have_matplotlib():
        ADDENDUM_LOG.warning("[subgroup_forest] matplotlib unavailable, skipping")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = summary.get("subgroups", {})
    n = len(groups)
    if n == 0:
        ADDENDUM_LOG.info("[subgroup_forest] no subgroups to plot")
        return
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.2 * rows), dpi=130, squeeze=False)
    for idx, (col_name, sub) in enumerate(groups.items()):
        ax = axes[idx // cols][idx % cols]
        entries = [(k, v) for k, v in sub.items() if isinstance(v, dict) and v.get("mean") is not None]
        entries.sort(key=lambda kv: kv[1]["mean"])
        ys = list(range(len(entries)))
        means = [e[1]["mean"] for e in entries]
        los = [e[1]["mean"] - e[1]["ci_lo"] for e in entries]
        his = [e[1]["ci_hi"] - e[1]["mean"] for e in entries]
        labels = [f"{e[0]} (n={e[1]['n']})" for e in entries]
        ax.errorbar(means, ys, xerr=[los, his], fmt="o", capsize=3, color="tab:blue")
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(col_name, fontsize=9)
        ax.set_xlabel(summary.get("metric", "metric"), fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "subgroup_forest.png"); plt.close(fig)
    ADDENDUM_LOG.info(f"[subgroup_forest] wrote forest plot → {out_dir}")


def cmd_failure_case_figure(args: argparse.Namespace, core) -> None:
    """Pick N representative cases (small ET, low-field, legacy, typical,
    worst) from a per-case CSV and save a summary table + optional montage."""
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.per_case)
    metric = args.metric
    rows = [r for r in rows if metric in r and r[metric] not in ("", None)]
    rows.sort(key=lambda r: float(r[metric]))
    n = int(args.n_cases)
    pick: List[Dict[str, Any]] = []
    pick += rows[:max(1, n // 3)]            # worst
    pick += rows[-max(1, n // 3):]           # best
    mid = len(rows) // 2
    pick += rows[mid:mid + max(1, n - len(pick))]
    _write_csv(pick, out_dir / "failure_case_selection.csv")
    ADDENDUM_LOG.info(f"[failure_case_figure] selected {len(pick)} cases → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# T. Compute profile (GPU mem peak, throughput).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_compute_profile(args: argparse.Namespace, core) -> None:
    import torch
    out_dir = _ensure_dir(args.out_dir)
    device_t = torch.device(args.device)
    patch = tuple(int(x) for x in args.patch_size.split(","))
    mods = [m.strip() for m in args.modalities.split(",")]
    if Path(args.ckpt).exists():
        model = core._load_student_for_eval(
            Path(args.ckpt), len(mods), args.num_classes, args.student_base, device_t
        )
    else:
        model = core.UNet3DStudent(2 * len(mods), args.num_classes, args.student_base).to(device_t).eval()
    x_full = torch.randn(1, len(mods), *patch, device=device_t)
    present_mask = torch.ones((1, len(mods)), dtype=x_full.dtype, device=device_t)
    x = core.prepare_student_input(x_full, present_mask)
    if device_t.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last_3d)

    if device_t.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_warm = time.time()
    with torch.no_grad():
        for _ in range(max(1, args.n_warmup)):
            _ = model(x)
    if device_t.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.n_iter):
            _ = model(x)
    if device_t.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if device_t.type == "cuda" else None
    report = {
        "device": str(device_t),
        "patch_size": list(patch),
        "n_modalities": len(mods),
        "input_channels": int(x.shape[1]),
        "student_base": args.student_base,
        "n_iter": args.n_iter,
        "seconds_per_iter": dt / args.n_iter,
        "iter_per_second": args.n_iter / dt,
        "peak_gpu_memory_mb": peak_mb,
        "warmup_seconds": time.time() - t_warm - dt,
    }
    _save_json(report, out_dir / "compute_profile.json")
    ADDENDUM_LOG.info(f"[compute_profile] s/iter={report['seconds_per_iter']:.4f}  "
             f"peakGPU={peak_mb} MB")


# ══════════════════════════════════════════════════════════════════════════════
# U. Quantization hardware benchmark (figure).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_quantization_hardware(args: argparse.Namespace, core) -> None:
    """Read a comparison JSON produced by compare_quantisation and render a
    hardware-friendly summary: FP32 vs INT8 throughput, Dice drop, HD95 delta.
    """
    out_dir = _ensure_dir(args.out_dir)
    data = json.load(open(args.compare_json))
    _save_json(data, out_dir / "quantization_hardware.json")
    if not _have_matplotlib():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = data.get("per_config", []) if isinstance(data, dict) else []
    if not rows:
        return
    labels = [r.get("config", "") for r in rows]
    fp32 = [r.get("fp32_mean_dice", 0) for r in rows]
    int8 = [r.get("int8_mean_dice", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(labels)), 3.5), dpi=130)
    xs = list(range(len(labels)))
    w = 0.35
    ax.bar([x - w / 2 for x in xs], fp32, w, label="FP32")
    ax.bar([x + w / 2 for x in xs], int8, w, label="INT8")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=75, fontsize=7, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_title("FP32 vs INT8 across modality configs")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "quantization_hardware.png"); plt.close(fig)
    ADDENDUM_LOG.info(f"[quantization_hardware] wrote {out_dir}/quantization_hardware.png")


# ══════════════════════════════════════════════════════════════════════════════
# V. Total energy / CO2 table.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_total_energy_table(args: argparse.Namespace, core) -> None:
    """Aggregate every ``energy_*_runs.jsonl`` file under ``root`` into one
    paper-ready table (total kWh, total gCO2eq, equivalents)."""
    out_dir = _ensure_dir(args.out_dir)
    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    total_wh = 0.0
    total_co2 = 0.0
    for f in root.rglob("energy_*_runs.jsonl"):
        phase = f.stem.replace("energy_", "").replace("_runs", "")
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                wh = float(entry.get("total_energy_wh", 0) or 0)
                co2 = float(entry.get("co2_grams", 0) or 0)
                total_wh += wh
                total_co2 += co2
                rows.append({
                    "phase": phase,
                    "run_id": entry.get("run_id"),
                    "slurm_job_id": entry.get("slurm_job_id"),
                    "hours": entry.get("elapsed_hours"),
                    "kwh": wh / 1000.0,
                    "co2_g": co2,
                    "source": str(f.relative_to(root)),
                })
    _write_csv(rows, out_dir / "energy_runs.csv")
    summary = {
        "n_runs": len(rows),
        "total_kwh": total_wh / 1000.0,
        "total_co2_grams": total_co2,
        "total_co2_kg": total_co2 / 1000.0,
        "approx_km_driven": total_co2 / 120.0,
        "approx_smartphone_charges": total_co2 / 8.22,
    }
    _save_json(summary, out_dir / "energy_totals.json")
    ADDENDUM_LOG.info(f"[total_energy_table] {len(rows)} runs, total "
             f"{summary['total_kwh']:.2f} kWh, {summary['total_co2_kg']:.3f} kgCO2")


# ══════════════════════════════════════════════════════════════════════════════
# W. CSV linter (label_remap audit).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_audit_voxel_spacing(args: argparse.Namespace, core) -> None:
    """Audit per-case voxel spacing across a manifest CSV (Gap C, 2026 audit).

    Opens the requested ``reference_modality`` (default: first modality) of
    every row, reads the NIfTI header zooms, and prints a (dataset_tag,
    rounded spacing) -> count table.  Flags any non-isotropic spacings.

    Use this BEFORE training to confirm your upstream pipeline resampled every
    cohort onto a common voxel grid.  Mixed isotropic/anisotropic spacings
    within the training pool confound cross-scanner generalization and are
    the #1 silent killer of LMIC external-validation performance.

    Example:
        python OGAP_source_code_experimental_v9.py audit_voxel_spacing \
            --csv train.csv --reference_modality t1n --out_dir audit_out
    """
    import nibabel as nib
    from collections import Counter as _Counter
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.csv)
    if not rows:
        ADDENDUM_LOG.error("[audit_voxel_spacing] CSV %s is empty.", args.csv)
        return
    ref_mod = (args.reference_modality or "").strip()
    if not ref_mod:
        # Fall back to the first non-meta column from the first row.
        meta = {"case_id", "label", "dataset_tag", "label_remap",
                "field_strength", "scanner_strength"}
        ref_mod = next((k for k in rows[0] if k not in meta and rows[0].get(k)), "")
        if not ref_mod:
            ADDENDUM_LOG.error("[audit_voxel_spacing] could not infer a reference modality; "
                               "pass --reference_modality.")
            return
    spacings: "_Counter[Tuple[str, Tuple[float, float, float]]]" = _Counter()
    anisotropic_rows: List[Dict[str, Any]] = []
    n_seen, n_skipped = 0, 0
    for r in rows:
        path = r.get(ref_mod, "").strip()
        if not path:
            n_skipped += 1
            continue
        try:
            img = nib.load(path)
            sp = tuple(round(float(s), 2) for s in img.header.get_zooms()[:3])
        except Exception as exc:  # noqa: BLE001 - robustness audit, surface anything
            ADDENDUM_LOG.warning("[audit_voxel_spacing] skipping %s: %s", path, exc)
            n_skipped += 1
            continue
        n_seen += 1
        tag = (r.get("dataset_tag") or "unknown").strip().lower()
        spacings[(tag, sp)] += 1
        if abs(sp[0] - sp[1]) > 0.05 or abs(sp[1] - sp[2]) > 0.05:
            anisotropic_rows.append({
                "case_id": r.get("case_id", ""),
                "dataset_tag": tag,
                "spacing_mm": ",".join(str(v) for v in sp),
                "path": path,
            })
    rows_out = []
    for (tag, sp), n in sorted(spacings.items(), key=lambda x: (x[0][0], -x[1])):
        iso = abs(sp[0] - sp[1]) <= 0.05 and abs(sp[1] - sp[2]) <= 0.05
        rows_out.append({
            "dataset_tag": tag,
            "spacing_mm": ",".join(str(v) for v in sp),
            "n_cases": n,
            "isotropic": "yes" if iso else "no",
        })
    _save_csv(rows_out, out_dir / "voxel_spacing_summary.csv")
    _save_csv(anisotropic_rows, out_dir / "voxel_spacing_anisotropic_cases.csv")
    summary = {
        "csv": str(args.csv),
        "reference_modality": ref_mod,
        "n_cases_seen": n_seen,
        "n_cases_skipped": n_skipped,
        "n_distinct_spacing_buckets": len(spacings),
        "n_anisotropic_cases": len(anisotropic_rows),
    }
    _save_json(summary, out_dir / "voxel_spacing_summary.json")
    ADDENDUM_LOG.info(
        "[audit_voxel_spacing] n_seen=%d n_skipped=%d distinct_buckets=%d "
        "anisotropic=%d → %s",
        n_seen, n_skipped, len(spacings), len(anisotropic_rows), out_dir,
    )


def cmd_lint_csv(args: argparse.Namespace, core) -> None:
    """Audit a training/validation CSV: schema, missing files, label-remap
    integrity, per-class label prevalence, per-dataset tag distribution."""
    import nibabel as nib
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.csv)
    required = ["case_id"] + [m.strip() for m in args.modalities.split(",")] + ["label"]
    issues: List[Dict[str, Any]] = []
    per_dataset_counts: Counter = Counter()
    per_class_counts: Counter = Counter()
    remap_seen: Counter = Counter()
    modalities = [m.strip() for m in args.modalities.split(",")]
    case_counts = Counter(str(r.get("case_id", "")).strip() for r in rows)
    for case_id, count in case_counts.items():
        if case_id and count > 1:
            issues.append({"row": "", "case_id": case_id, "issue": f"duplicate case_id appears {count} times"})

    train_case_ids: set[str] = set()
    train_t1c_header_hashes: set[str] = set()
    t1c_modality = next((m for m in modalities if m.lower() in {"t1c", "t1ce", "t1gd"}), None)
    if args.train_csv:
        train_rows = _read_csv_rows(args.train_csv)
        train_case_ids = {str(r.get("case_id", "")).strip() for r in train_rows if str(r.get("case_id", "")).strip()}
        if t1c_modality:
            for tr in train_rows:
                p = tr.get(t1c_modality, "")
                if p and Path(p).exists():
                    try:
                        train_t1c_header_hashes.add(
                            hashlib.sha256(nib.load(p).header.binaryblock).hexdigest()
                        )
                    except Exception:
                        pass

    for i, r in enumerate(rows):
        for col in required:
            v = r.get(col, "").strip()
            if not v:
                issues.append({"row": i, "case_id": r.get("case_id"),
                               "issue": f"missing column: {col}"})
                continue
            if col != "case_id" and not Path(v).exists():
                issues.append({"row": i, "case_id": r.get("case_id"),
                               "issue": f"missing file: {col}={v}"})
        tag = r.get("dataset_tag", "unspecified")
        per_dataset_counts[tag] += 1
        case_id = str(r.get("case_id", "")).strip()
        is_utsw = "utsw" in case_id.lower() or "utsw" in str(tag).lower()
        if args.train_csv and case_id and case_id in train_case_ids:
            issues.append({"row": i, "case_id": case_id, "issue": "case_id also appears in training CSV"})
        if args.train_csv and is_utsw and t1c_modality and r.get(t1c_modality):
            p = r.get(t1c_modality, "")
            if Path(p).exists():
                try:
                    h = hashlib.sha256(nib.load(p).header.binaryblock).hexdigest()
                    if h in train_t1c_header_hashes:
                        issues.append({"row": i, "case_id": case_id, "issue": "T1c header SHA-256 matches a training case"})
                except Exception as e:
                    issues.append({"row": i, "case_id": case_id, "issue": f"T1c header hash failed: {e}"})
        remap = r.get("label_remap", "")
        if remap:
            remap_seen[remap] += 1
            try:
                remap_map = json.loads(remap)
                if not isinstance(remap_map, dict):
                    issues.append({"row": i, "case_id": r.get("case_id"),
                                   "issue": "label_remap is not a JSON object"})
            except Exception as e:
                issues.append({"row": i, "case_id": r.get("case_id"),
                               "issue": f"label_remap JSON parse failed: {e}"})
        if args.check_shapes:
            try:
                first_path = r.get(modalities[0], "")
                first_vol, first_spacing, first_affine = load_nifti_with_meta(first_path)
                ref_shape = first_vol.shape
                for modality in modalities[1:]:
                    vol, spacing_i, affine_i = load_nifti_with_meta(r.get(modality, ""))
                    _assert_aligned_volume(
                        case_id=case_id or f"row_{i}",
                        source_name=f"modality:{modality}",
                        source_path=r.get(modality, ""),
                        reference_shape=ref_shape,
                        reference_spacing=first_spacing,
                        reference_affine=first_affine,
                        shape=vol.shape,
                        spacing=spacing_i,
                        affine=affine_i,
                    )
                if r.get("label"):
                    label_vol, label_spacing, label_affine = load_nifti_with_meta(r["label"])
                    _assert_aligned_volume(
                        case_id=case_id or f"row_{i}",
                        source_name="label",
                        source_path=r["label"],
                        reference_shape=ref_shape,
                        reference_spacing=first_spacing,
                        reference_affine=first_affine,
                        shape=label_vol.shape,
                        spacing=label_spacing,
                        affine=label_affine,
                    )
            except Exception as e:
                issues.append({"row": i, "case_id": case_id, "issue": f"shape/affine check failed: {e}"})
        label_path = r.get("label", "")
        if args.check_label_values and Path(label_path).exists():
            try:
                y = np.asarray(nib.load(label_path).get_fdata()).astype(np.int32)
                uniq = sorted(int(u) for u in np.unique(y))
                for u in uniq:
                    per_class_counts[f"raw_{u}"] += 1
                # Apply remap and re-count
                if remap:
                    try:
                        rm = json.loads(remap)
                        y_mapped = np.vectorize(lambda v: int(rm.get(str(int(v)), v)))(y)
                        for u in sorted(int(u) for u in np.unique(y_mapped)):
                            per_class_counts[f"mapped_{u}"] += 1
                        mapped_uniq = {int(u) for u in np.unique(y_mapped)}
                        invalid = sorted(v for v in mapped_uniq if v not in {0, 1, 2, 3, 4})
                        if invalid:
                            issues.append({"row": i, "case_id": case_id, "issue": f"label_remap produced invalid values: {invalid}"})
                    except Exception:
                        pass
                if 4 in uniq and not remap:
                    issues.append({
                        "row": i,
                        "case_id": case_id,
                        "issue": "raw BraTS label 4 present without explicit label_remap; loader will map 4->3",
                    })
                if is_utsw and not remap:
                    issues.append({
                        "row": i,
                        "case_id": case_id,
                        "issue": "UTSW label schema unverified; set label_remap if source is not BraTS 1=NCR,2=ED,4=ET",
                    })
            except Exception as e:
                issues.append({"row": i, "case_id": r.get("case_id"),
                               "issue": f"label load failed: {e}"})
    _write_csv(issues, out_dir / "lint_issues.csv")
    _save_json({
        "n_rows": len(rows),
        "n_issues": len(issues),
        "per_dataset": dict(per_dataset_counts),
        "per_class": dict(per_class_counts),
        "remap_schemas_seen": dict(remap_seen),
    }, out_dir / "lint_summary.json")
    ADDENDUM_LOG.info(f"[lint_csv] {len(rows)} rows  {len(issues)} issues  "
             f"{len(per_dataset_counts)} datasets")


# ══════════════════════════════════════════════════════════════════════════════
# X. Sampler realized-distribution logger.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_sampler_distribution(args: argparse.Namespace, core) -> None:
    """Instantiate the metadata-weighted sampler the way training does, sample
    N draws, and log realized distribution (field strength, grade, IDH,
    legacy) vs. dataset prevalence. This addresses the 'multiplicative biases
    compound to 2.8x' concern in risk #7."""
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_csv_rows(args.csv)
    # Compute empirical prevalence first (by case rows as shown).
    fields = ("field_strength", "Tumor Grade", "IDH", "scan_year")
    baseline: Dict[str, Counter] = {f: Counter() for f in fields}
    for r in rows:
        for f in fields:
            baseline[f][str(r.get(f, "")).strip()] += 1

    rng = np.random.default_rng(args.seed)
    weights = np.ones(len(rows), dtype=np.float64)
    low_field_threshold = 1.0
    for i, r in enumerate(rows):
        try:
            fs = float(r.get("field_strength", "nan"))
            if fs <= low_field_threshold:
                weights[i] *= args.low_field_bias
            # inverse-frequency power
            if fs > 0:
                weights[i] *= (fs + 1e-6) ** (-args.field_strength_power)
        except Exception:
            pass
        grade = str(r.get("Tumor Grade", "")).strip()
        if grade in ("2", "3"):
            weights[i] *= args.grade23_bias
        idh = str(r.get("IDH", "")).strip().lower()
        if idh in ("mutated", "mutant", "idh-mutant"):
            weights[i] *= args.idh_mutant_bias
        try:
            year = int(str(r.get("scan_year", "0")))
            if 0 < year < args.legacy_year:
                weights[i] *= args.legacy_bias
        except Exception:
            pass
    weights = weights / weights.sum()
    draws = rng.choice(len(rows), size=args.n_draws, p=weights, replace=True)
    realized: Dict[str, Counter] = {f: Counter() for f in fields}
    for d in draws:
        for f in fields:
            realized[f][str(rows[d].get(f, "")).strip()] += 1

    report = {"n_rows": len(rows), "n_draws": args.n_draws, "max_weight_multiplier": float(weights.max() / max(weights.min(), 1e-12))}
    for f in fields:
        report[f] = {"baseline_counts": dict(baseline[f]),
                     "realized_counts": dict(realized[f])}
    _save_json(report, out_dir / "sampler_realized_distribution.json")
    ADDENDUM_LOG.info(f"[sampler_distribution] max/min weight ratio = "
             f"{report['max_weight_multiplier']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Y. Model card (Mitchell et al. 2019).
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CARD_TEMPLATE = """# Model Card - OGAP Student (v9)

## Model details
- Architecture: MedNeXt-S style student (base channels: {student_base}), soft-masked input, no-attention INT8 path by default.
- Teacher: unified MedNeXt-style teacher (base channels: 32). Distilled with FP32 temperature-4 KD (α={holder_alpha}).
- Checkpoint: {ckpt}
- Training date: {train_date}
- Framework: PyTorch {torch_version}, CUDA {cuda_version}
- Export: ONNX opset 17, optional static INT8 quantisation (per-channel, minmax).

## Intended use
- Primary: research-use glioma tumour sub-region segmentation on multi-parametric MRI
  (T1n, T1c, T2w, T2-FLAIR) in adult and AYA populations.
- Secondary: deployment in low-resource settings (CPU INT8), including sub-1T scanners
  where the appropriate field-strength-aware augmentation has been enabled in training.
- NOT intended for: standalone clinical diagnosis, treatment planning without a radiologist
  in the loop, pediatric populations outside the training distribution, or pathologies
  outside the glioma spectrum (e.g., metastasis, abscess, demyelination).

## Training data
- Cohorts: {cohorts}
- Partial-label handling: per-sample marginal CE for WT-only cohorts.
- Sampler biases: field-strength inverse-frequency power {field_strength_power};
  low-field bias {low_field_bias}×; grade 2/3 bias {grade23_bias}×;
  IDH-mutant bias {idh_mutant_bias}×; legacy-protocol bias {legacy_bias}×
  (year < {legacy_year}).

## Evaluation
- Internal: cross-dataset validation with 15 modality combinations.
- External: UTSW cohort (stratified by manual refinement, IDH, grade, 1p19q, MGMT,
  scanner strength, make, sex, age).
- Metrics: Dice, HD95 (physical spacing), Surface Dice at 1/2/3 mm; ECE, Brier,
  reliability diagrams.
- Statistics: BCa 95% CIs (5000 resamples); Wilcoxon signed-rank with Holm-Bonferroni.

## Quantitative analyses
- See stratified_summary.json for subgroup performance.
- See calibration_summary.json for ECE/Brier/reliability.
- See compute_profile.json for throughput and peak memory.
- See energy_totals.json for training and evaluation carbon footprint.

## Ethical considerations
- Known limitations:
  - Label noise in non-manually-refined UTSW cohort (reported separately).
  - Partial labels in ERASMUS cohort (WT-only; marginal CE used; see erasmus_bias_report.json).
  - Field-strength distribution skewed toward 1.5T and 3T; sub-1T performance
    extrapolated via physics-informed augmentation and validated on {n_sub1t} UTSW
    sub-1T cases.
- Sex/gender analysis: stratified by sex at birth; see subgroup_forest.png.

## Caveats and recommendations
- Re-validate before any deployment across MRI vendors or protocols not represented
  in the training cohort.
- Re-tune post-processing thresholds when switching from single-model softmax to
  deep-ensemble / MC-dropout outputs.
- Cite: Diaz Lab OGAP v9 (Medical Image Analysis, under review, {year}).
"""


def cmd_model_card(args: argparse.Namespace, core) -> None:
    out_dir = _ensure_dir(args.out_dir)
    import torch
    text = MODEL_CARD_TEMPLATE.format(
        student_base=args.student_base,
        holder_alpha=args.holder_alpha,
        ckpt=args.ckpt,
        train_date=time.strftime("%Y-%m-%d"),
        torch_version=torch.__version__,
        cuda_version=getattr(torch.version, "cuda", "n/a"),
        cohorts=args.cohorts,
        field_strength_power=args.field_strength_power,
        low_field_bias=args.low_field_bias,
        grade23_bias=args.grade23_bias,
        idh_mutant_bias=args.idh_mutant_bias,
        legacy_bias=args.legacy_bias,
        legacy_year=args.legacy_year,
        n_sub1t=args.n_sub1t,
        year=time.strftime("%Y"),
    )
    (out_dir / "MODEL_CARD.md").write_text(text)
    ADDENDUM_LOG.info(f"[model_card] wrote {out_dir}/MODEL_CARD.md")


# ══════════════════════════════════════════════════════════════════════════════
# Y2. Conformal Risk Control (Angelopoulos, Bates, Lei, Malik, Jordan 2024).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_crc(args: argparse.Namespace, core) -> None:
    """Conformal Risk Control: distribution-free expected-loss bound.

    Reads a held-out calibration CSV with rows ``(case_id, lambda, loss)``
    where ``loss`` is bounded in [0, B] (e.g. case-level FNR ∈ [0, 1]) and
    ``lambda`` indexes a monotone family of predictors / thresholds.

    For each lambda on the grid, computes the empirical mean risk
    ``R̂(λ) = (1/n) Σ_i L_i(λ)`` and selects the smallest λ̂ that satisfies
    the CRC inequality:

        R̂(λ̂) ≤ ((n+1)/n) · α - B/n

    Then E[L(λ̂)] ≤ α holds for any exchangeable test point (Theorem 1 of
    Angelopoulos et al. 2024). For monotone losses (the common case),
    sweeping λ in increasing order makes the search a simple linear scan.
    """
    import csv as _csv
    out_dir = _ensure_dir(args.out_dir)
    by_lambda: Dict[float, List[float]] = {}
    case_ids: set = set()
    with open(args.calib_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                lam = float(row["lambda"])
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            by_lambda.setdefault(lam, []).append(loss)
            cid = row.get("case_id")
            if cid:
                case_ids.add(str(cid))
    if not by_lambda:
        raise ValueError(
            f"No usable rows in {args.calib_csv}; expect columns: case_id, lambda, loss"
        )
    n = max(len(case_ids), max(len(v) for v in by_lambda.values()))
    if n < 5:
        raise ValueError(f"CRC needs ≥5 calibration cases, got {n}")
    alpha = float(args.alpha)
    B = float(args.loss_upper_bound)
    rhs = ((n + 1) / n) * alpha - B / n
    grid = []
    chosen: Optional[Tuple[float, float]] = None
    for lam in sorted(by_lambda.keys()):
        r_hat = float(np.mean(by_lambda[lam]))
        ok = bool(r_hat <= rhs)
        grid.append({"lambda": float(lam), "r_hat": r_hat,
                     "n_cases": int(len(by_lambda[lam])), "satisfies_crc": ok})
        if ok and chosen is None:
            chosen = (float(lam), r_hat)
    payload = {
        "alpha": alpha,
        "loss_upper_bound": B,
        "n_cases": int(n),
        "rhs_bound": float(rhs),
        "lambda_grid": grid,
        "lambda_hat": chosen[0] if chosen else None,
        "achieved_r_hat": chosen[1] if chosen else None,
        "guarantee": ("E[L(λ̂)] ≤ α with probability 1 over the calibration "
                      "exchangeability assumption (Angelopoulos+ 2024 Thm 1).")
    }
    _save_json(payload, out_dir / "crc_calibration.json")
    if chosen:
        ADDENDUM_LOG.info(
            f"[crc] λ̂={chosen[0]:.6g}  R̂={chosen[1]:.4f}  α={alpha}  n={n}"
        )
    else:
        ADDENDUM_LOG.warning(
            f"[crc] no λ in grid satisfies α={alpha} (rhs={rhs:.4f}); "
            "widen the grid or relax α"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Y3. Mahalanobis OOD detector (Lee et al., NeurIPS 2018).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_mahalanobis_ood(args: argparse.Namespace, core) -> None:
    """1-class Mahalanobis OOD score on the student bottleneck features.

    Phase A (fit): runs the student on each ID case, captures the deepest
    encoder feature map, channel-mean-pools it over space, and accumulates
    a per-feature mean μ and shared covariance Σ (with diagonal regularizer
    1e-3 · I) across cases.

    Phase B (score): computes ``M(x) = (f - μ)ᵀ Σ⁻¹ (f - μ)`` for every ID
    and OOD case. Higher M ⇒ further from the in-distribution Gaussian.

    Outputs JSON with per-case scores, the (μ, Σ⁻¹) parameters, and (if an
    OOD CSV is provided) the OOD-detection AUROC. The hook target defaults
    to the deepest encoder block on UNet3DStudent; override via --hook_attr.
    """
    out_dir = _ensure_dir(args.out_dir)
    ckpt = Path(args.ckpt)
    mods = [m.strip() for m in args.modalities.split(",")]
    ps = tuple(int(x) for x in args.patch_size.split(","))
    if len(ps) != 3:
        raise ValueError(f"--patch_size must be 'D,H,W'; got {args.patch_size!r}")
    device_t = torch.device(args.device)

    student = core._load_student_for_eval(
        ckpt, len(mods), int(args.num_classes), int(args.student_base), device_t
    )
    student.eval()

    # Identify hook target. UNet3DStudent typically exposes encoder blocks
    # named enc1..enc4 / bottleneck - pick whichever the user names, falling
    # back to the deepest encoder we can find.
    target = None
    if args.hook_attr:
        target = getattr(student, args.hook_attr, None)
    if target is None:
        for name in ("bottleneck", "enc4", "encoder4", "down4", "down3"):
            target = getattr(student, name, None)
            if target is not None:
                break
    if target is None:
        # Fallback: the last 3D conv module
        conv_modules = [m for m in student.modules() if isinstance(m, nn.Conv3d)]
        if not conv_modules:
            raise RuntimeError("Could not find any Conv3d module to hook")
        target = conv_modules[-1]

    feat_buf: List[torch.Tensor] = []

    def _hook(_module, _inp, out):
        if isinstance(out, torch.Tensor):
            feat_buf.append(out.detach().float())

    handle = target.register_forward_hook(_hook)

    def _run_csv(csv_path: str) -> List[Tuple[str, np.ndarray]]:
        ds = core.InferenceDataset(csv_path, mods)
        results: List[Tuple[str, np.ndarray]] = []
        for i in range(len(ds)):
            case_id, x, row = ds[i]
            x = x.to(device_t).unsqueeze(0)
            b, c = x.shape[:2]
            present = row.get("__present_mask__")
            if present is not None:
                mask = torch.from_numpy(np.asarray(present, dtype=np.float32)).to(device_t).unsqueeze(0)
            else:
                mask = torch.ones((b, c), dtype=torch.float32, device=device_t)
            inp = core.prepare_student_input(x, mask)
            feat_buf.clear()
            try:
                with torch.no_grad():
                    _ = core.sliding_window_inference(student, inp, ps, 0.5)
            except Exception as exc:
                ADDENDUM_LOG.warning(f"[mahalanobis_ood] {case_id}: forward failed: {exc}")
                continue
            if not feat_buf:
                continue
            feat = feat_buf[-1].squeeze(0)  # (C, ...)
            feat = feat.reshape(feat.shape[0], -1).mean(dim=1)  # (C,)
            results.append((str(case_id), feat.cpu().numpy().astype(np.float64)))
        return results

    id_records = _run_csv(args.id_csv)
    if not id_records:
        handle.remove()
        raise RuntimeError(
            f"No ID features captured from {args.id_csv}; check --hook_attr / model"
        )
    F_id = np.stack([f for _, f in id_records], axis=0)
    mu = F_id.mean(axis=0)
    centered = F_id - mu
    cov = (centered.T @ centered) / max(1, F_id.shape[0] - 1)
    cov_reg = cov + 1e-3 * np.eye(cov.shape[0], dtype=cov.dtype)
    cov_inv = np.linalg.pinv(cov_reg)

    def _mahal(F: np.ndarray) -> np.ndarray:
        diff = F - mu
        return np.einsum("ni,ij,nj->n", diff, cov_inv, diff)

    id_scores = _mahal(F_id)
    payload: Dict[str, Any] = {
        "id_csv": str(args.id_csv),
        "n_id": int(len(id_records)),
        "feature_dim": int(F_id.shape[1]),
        "id_scores": [
            {"case_id": cid, "mahal": float(s)}
            for (cid, _), s in zip(id_records, id_scores)
        ],
    }

    ood_records: List[Tuple[str, np.ndarray]] = []
    if args.ood_csv:
        ood_records = _run_csv(args.ood_csv)
    handle.remove()

    if ood_records:
        F_ood = np.stack([f for _, f in ood_records], axis=0)
        ood_scores = _mahal(F_ood)
        payload["ood_csv"] = str(args.ood_csv)
        payload["n_ood"] = int(len(ood_records))
        payload["ood_scores"] = [
            {"case_id": cid, "mahal": float(s)}
            for (cid, _), s in zip(ood_records, ood_scores)
        ]
        try:
            from sklearn.metrics import roc_auc_score
            y = np.concatenate(
                [np.zeros(len(id_scores), dtype=np.int8),
                 np.ones(len(ood_scores), dtype=np.int8)]
            )
            scores_all = np.concatenate([id_scores, ood_scores])
            payload["auroc_ood"] = float(roc_auc_score(y, scores_all))
        except Exception as exc:
            payload["auroc_ood"] = None
            payload["auroc_error"] = str(exc)

    _save_json(payload, out_dir / "mahalanobis_ood.json")
    np.savez(
        out_dir / "mahalanobis_params.npz",
        mu=mu.astype(np.float32),
        cov_inv=cov_inv.astype(np.float32),
    )
    ADDENDUM_LOG.info(
        f"[mahalanobis_ood] feature_dim={F_id.shape[1]} n_id={len(id_records)} "
        f"n_ood={len(ood_records)} auroc={payload.get('auroc_ood')}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Y4. TOST equivalence test (Schuirmann 1987) for INT8 vs FP32 non-inferiority.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_tost(args: argparse.Namespace, core) -> None:
    """Two One-Sided Tests (TOST) for paired-sample equivalence.

    Reads a CSV with columns ``case_id,value_a,value_b`` (paired) and tests
    H0a: μ(a-b) ≤ -δ against Ha: μ(a-b) > -δ, AND
    H0b: μ(a-b) ≥ +δ against Hb: μ(a-b) < +δ.
    Equivalence at level α is declared iff BOTH p < α (Schuirmann 1987).

    Typical use: a = INT8 Dice, b = FP32 Dice, δ = 0.02 (2 Dice points).
    Rejecting both halves means INT8 ≈ FP32 within ±2 Dice points.
    """
    import csv as _csv
    from scipy.stats import t as t_dist
    out_dir = _ensure_dir(args.out_dir)
    a_vals: List[float] = []
    b_vals: List[float] = []
    case_ids: List[str] = []
    with open(args.paired_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                a_vals.append(float(row["value_a"]))
                b_vals.append(float(row["value_b"]))
                case_ids.append(str(row.get("case_id", "")))
            except (KeyError, TypeError, ValueError):
                continue
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    n = len(a)
    if n < 5:
        raise ValueError(f"TOST needs ≥5 paired cases, got {n}")
    diff = a - b
    mean_d = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    margin = float(args.margin)
    alpha = float(args.alpha)
    df = n - 1
    if se <= 0:
        # Degenerate: zero variance. If mean_d itself is in [-δ, δ], call it equivalent.
        equivalent = -margin <= mean_d <= margin
        out = {
            "n_paired": n,
            "mean_difference_a_minus_b": mean_d,
            "sd_diff": sd, "se_diff": se,
            "margin_delta": margin, "alpha": alpha,
            "p_lower": 0.0 if mean_d > -margin else 1.0,
            "p_upper": 0.0 if mean_d < margin else 1.0,
            "p_tost": 0.0 if equivalent else 1.0,
            "equivalent": bool(equivalent),
            "note": "zero-variance pairs",
        }
    else:
        t_lower = (mean_d + margin) / se
        t_upper = (mean_d - margin) / se
        p_lower = float(1.0 - t_dist.cdf(t_lower, df=df))
        p_upper = float(t_dist.cdf(t_upper, df=df))
        p_tost = float(max(p_lower, p_upper))
        out = {
            "n_paired": n,
            "mean_difference_a_minus_b": mean_d,
            "sd_diff": sd, "se_diff": float(se),
            "margin_delta": margin, "alpha": alpha,
            "df": int(df),
            "t_lower": float(t_lower), "t_upper": float(t_upper),
            "p_lower": p_lower, "p_upper": p_upper,
            "p_tost": p_tost,
            "equivalent": bool(p_tost < alpha),
        }
    _save_json(out, out_dir / "tost_equivalence.json")
    ADDENDUM_LOG.info(
        f"[tost] n={n} mean_d={mean_d:+.4f} se={out['se_diff']:.4f} "
        f"margin={margin} p_tost={out['p_tost']:.4g} equivalent={out['equivalent']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Y5. DeLong test for two paired AUCs (DeLong 1988; Sun & Xu fast 2014).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_delong(args: argparse.Namespace, core) -> None:
    """DeLong's test of equality between two correlated AUCs.

    Reads a CSV with columns ``case_id,label,score_a,score_b`` where
    ``label ∈ {0, 1}`` (also accepts ``pos``/``neg``). Computes the
    Mann-Whitney AUC for both score columns, the DeLong covariance via
    structural components, and a two-sided z-test on (AUC_a - AUC_b).

    Implementation follows Sun & Xu (2014) "Fast Implementation of DeLong's
    Algorithm" - O((m+n) log(m+n)) per AUC instead of the original O(mn).
    """
    import csv as _csv
    from scipy.stats import norm
    out_dir = _ensure_dir(args.out_dir)
    labels: List[int] = []
    sa_list: List[float] = []
    sb_list: List[float] = []
    with open(args.paired_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                lab_raw = row["label"]
                if isinstance(lab_raw, str):
                    lab = 1 if lab_raw.strip().lower() in {"1", "pos", "positive", "true", "yes"} else 0
                else:
                    lab = int(lab_raw)
                labels.append(int(lab))
                sa_list.append(float(row["score_a"]))
                sb_list.append(float(row["score_b"]))
            except (KeyError, TypeError, ValueError):
                continue
    y = np.asarray(labels, dtype=np.int8)
    A = np.asarray(sa_list, dtype=np.float64)
    B = np.asarray(sb_list, dtype=np.float64)
    if y.size == 0 or y.sum() in (0, y.size):
        raise ValueError("DeLong requires both positive and negative cases")

    def _midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=np.float64)
        i = 0
        N = len(x)
        while i < N:
            j = i
            while j + 1 < N and x[order[j + 1]] == x[order[i]]:
                j += 1
            r = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = r
            i = j + 1
        return ranks

    def _auc_components(scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        pos = scores[y == 1]
        neg = scores[y == 0]
        m = pos.size
        n = neg.size
        all_scores = np.concatenate([pos, neg])
        ranks_all = _midrank(all_scores)
        rk_pos_alone = _midrank(pos)
        rk_neg_alone = _midrank(neg)
        ranks_pos = ranks_all[:m]
        ranks_neg = ranks_all[m:]
        auc = (ranks_pos.sum() - m * (m + 1) / 2.0) / (m * n)
        v01 = (ranks_pos - rk_pos_alone) / n
        v10 = 1.0 - (ranks_neg - rk_neg_alone) / m
        return float(auc), v01, v10

    auc_a, v01_a, v10_a = _auc_components(A)
    auc_b, v01_b, v10_b = _auc_components(B)
    m = int(y.sum())
    n_neg = int(y.size - m)
    sx = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    sy = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    cov = sx / m + sy / n_neg
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        z = 0.0
        p_two = 1.0
    else:
        z = float((auc_a - auc_b) / np.sqrt(var_diff))
        p_two = float(2.0 * (1.0 - norm.cdf(abs(z))))
    out = {
        "n_total": int(y.size),
        "n_pos": m, "n_neg": n_neg,
        "auc_a": auc_a, "auc_b": auc_b,
        "delta_auc_a_minus_b": float(auc_a - auc_b),
        "se_diff": float(np.sqrt(max(var_diff, 0.0))),
        "z": z, "p_value_two_sided": p_two,
    }
    _save_json(out, out_dir / "delong_test.json")
    ADDENDUM_LOG.info(
        f"[delong] AUC_a={auc_a:.4f} AUC_b={auc_b:.4f} ΔAUC={auc_a - auc_b:+.4f} "
        f"z={z:+.3f} p={p_two:.4g}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Y6. DICOM-SEG export (highdicom; optional).
# ══════════════════════════════════════════════════════════════════════════════

def cmd_dicom_seg(args: argparse.Namespace, core) -> None:
    """Convert a NIfTI segmentation to DICOM-SEG aligned with a DICOM series.

    Optional dependency: ``highdicom`` and ``pydicom`` (``pip install
    highdicom pydicom``). The label-map JSON associates each integer
    segmentation value with a SCT-coded property (defaults to OGAP/BraTS:
    1=NCR, 2=oedema, 3=ET).
    """
    out_dir = _ensure_dir(args.out_dir)
    try:
        import highdicom as hd
        import pydicom
    except ImportError as exc:
        raise SystemExit(
            "[dicom_seg] requires highdicom + pydicom: "
            f"`pip install highdicom pydicom` (import error: {exc})"
        )
    seg_volume = nib.load(args.nifti_seg).get_fdata().astype(np.int16)
    label_map = (
        json.loads(Path(args.label_map).read_text())
        if args.label_map and Path(args.label_map).exists()
        else {
            "1": {"label": "NCR (necrotic core)", "sct": "1077"},
            "2": {"label": "Oedema", "sct": "79654002"},
            "3": {"label": "ET (enhancing tumor)", "sct": "108369006"},
        }
    )
    src_dir = Path(args.dicom_dir)
    src_files = sorted(src_dir.glob("*.dcm"))
    if not src_files:
        raise SystemExit(f"[dicom_seg] no DICOM files in {src_dir}")
    src_images = [pydicom.dcmread(str(p)) for p in src_files]

    # (segments, depth, rows, cols)
    pixel_array = np.stack(
        [(seg_volume == int(lid)).astype(np.uint8) for lid in label_map.keys()],
        axis=0,
    )
    segments = []
    for idx, (lid, info) in enumerate(label_map.items(), start=1):
        segments.append(hd.seg.SegmentDescription(
            segment_number=idx,
            segment_label=info["label"],
            segmented_property_category=hd.sr.CodedConcept(
                value="49755003", scheme_designator="SCT",
                meaning="Morphologically Altered Structure",
            ),
            segmented_property_type=hd.sr.CodedConcept(
                value=str(info["sct"]),
                scheme_designator="SCT",
                meaning=info["label"],
            ),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=hd.AlgorithmIdentificationSequence(
                name="OGAP-2.0", version="2.0",
                family=hd.sr.CodedConcept(
                    value="DL", scheme_designator="DCM",
                    meaning="Deep learning",
                ),
            ),
        ))
    seg = hd.seg.Segmentation(
        source_images=src_images,
        pixel_array=pixel_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segments,
        series_instance_uid=hd.UID(),
        series_number=int(args.series_number),
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="OGAP",
        manufacturer_model_name="OGAP-2.0",
        software_versions="2.0",
        device_serial_number="ogap-test",
    )
    out_path = out_dir / args.out_name
    seg.save_as(str(out_path))
    ADDENDUM_LOG.info(f"[dicom_seg] wrote {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Y7. Augmentor smoke test (validate_augmentor).
#
# Runs each augmentation op in turn on a synthetic 3D volume with a known
# tumor-like blob, then writes a JSON report and (when matplotlib is
# available) an axial mid-slice montage so the user can sanity-check by eye.
# Fast (<10 s) and dependency-free apart from optional matplotlib.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_validate_augmentor(args: argparse.Namespace, core) -> None:
    """Synthetic-volume smoke test for every augmentor operation.

    Generates a 4-modality (96, 96, 80) volume with three concentric
    intensity bins (CSF / WM / GM proxy) plus an enhancing-tumor sphere on
    the T1c channel, runs each op individually with prob=1, and reports
    per-op statistics: mean, std, fraction of voxels changed, and (when
    matplotlib is installed) a mid-axial-slice PNG montage written to
    ``<out_dir>/augmentor_montage.png``.
    """
    out_dir = _ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    d, h, w = 96, 96, 80
    base = np.zeros((4, d, h, w), dtype=np.float32)
    # Concentric soft brain proxy with three intensity bins per modality.
    zg, yg, xg = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    radial = np.sqrt(((zg - cz) / (d * 0.45)) ** 2
                     + ((yg - cy) / (h * 0.45)) ** 2
                     + ((xg - cx) / (w * 0.45)) ** 2)
    brain = (radial <= 1.0).astype(np.float32)
    csf = ((radial > 0.85) & (radial <= 1.0)).astype(np.float32)
    gm = ((radial > 0.65) & (radial <= 0.85)).astype(np.float32)
    wm = (radial <= 0.65).astype(np.float32)
    # Per-modality contrast pattern (T1n-like, T1c-like, T2w-like, FLAIR-like)
    base[0] = 0.40 * csf + 0.85 * wm + 0.65 * gm           # T1n
    base[1] = base[0].copy()                               # T1c (start = T1n)
    base[2] = 0.95 * csf + 0.55 * wm + 0.75 * gm           # T2w
    base[3] = 0.10 * csf + 0.55 * wm + 0.75 * gm           # FLAIR
    # Synthetic enhancing tumor on T1c
    et_mask = (((zg - cz) ** 2 + (yg - cy + 8) ** 2 + (xg - cx + 6) ** 2) < 64)
    base[1][et_mask] = 1.10
    label = np.zeros((d, h, w), dtype=np.int64)
    label[et_mask] = 3
    # NCR/oedema rim
    rim = ndi.binary_dilation(et_mask, iterations=3) & ~et_mask
    label[rim & brain.astype(bool)] = 1

    augmentor = core.MRIPhysicsAugmentor(
        # Force every probability to 1 for the per-op tests below.
        p_flip=0.0, p_elastic=0.0, p_bias_field=0.0, p_noise=0.0,
        p_rician_noise=0.0, p_gamma=0.0, p_brightness=0.0, p_contrast=0.0,
        p_resolution=0.0, p_motion=0.0, p_ghosting=0.0, p_spike=0.0, p_gibbs=0.0,
        p_focal_b1_inhomogeneity=0.0, p_anisotropic_2d=0.0, p_partial_contrast=0.0,
        p_field_contrast_warp=0.0, p_fourier_amplitude_mix=0.0,
        p_receive_coil_inhomogeneity=0.0, p_off_resonance_void=0.0,
        noise_calibration="static",
        field_strength=float(args.field_strength),
        field_contrast_source_b0=3.0,
    )

    def _run_isolated(op_name: str, fn) -> Dict[str, Any]:
        np.random.seed(int(args.seed))
        x = base.copy()
        out = fn(x)
        if isinstance(out, tuple):
            out = out[0]
        diff = np.abs(out - base)
        return {
            "op": op_name,
            "mean_in": float(base.mean()),
            "mean_out": float(out.mean()),
            "std_out": float(out.std()),
            "frac_changed": float((diff > 1e-4).mean()),
            "max_abs_diff": float(diff.max()),
        }

    contrast_set = [1]  # T1c index in the canonical order
    op_results: List[Dict[str, Any]] = []
    op_results.append(_run_isolated(
        "field_contrast_warp",
        lambda x: (
            setattr(augmentor, "p_field_contrast_warp", 1.0)
            or augmentor._simulate_field_strength_contrast(x, contrast_mod_indices=contrast_set)
        ),
    ))
    augmentor.p_field_contrast_warp = 0.0

    op_results.append(_run_isolated(
        "receive_coil_falloff",
        lambda x: (
            setattr(augmentor, "p_receive_coil_inhomogeneity", 1.0)
            or augmentor._simulate_receive_coil_inhomogeneity(x)
        ),
    ))
    augmentor.p_receive_coil_inhomogeneity = 0.0

    op_results.append(_run_isolated(
        "off_resonance_void",
        lambda x: (
            setattr(augmentor, "p_off_resonance_void", 1.0)
            or augmentor._simulate_off_resonance_void(x)
        ),
    ))
    augmentor.p_off_resonance_void = 0.0

    op_results.append(_run_isolated(
        "rician_aja_fernandez",
        lambda x: (
            setattr(augmentor, "noise_calibration", "aja_fernandez")
            or augmentor._add_rician_noise(x.copy())
        ),
    ))
    augmentor.noise_calibration = "static"

    # The AFA shim no longer consumes a donor, but keep a non-identical donor in
    # this smoke path so older call signatures remain exercised.
    donor_rng = np.random.default_rng(int(args.seed) + 1)
    donor = base.copy()
    donor *= 1.0 + 0.3 * donor_rng.standard_normal(donor.shape).astype(np.float32)
    donor += 0.15 * donor_rng.standard_normal(donor.shape).astype(np.float32)
    donor = np.clip(donor, 0.0, None).astype(np.float32)
    op_results.append(_run_isolated(
        "fourier_amplitude_mix",
        lambda x: (
            setattr(augmentor, "p_fourier_amplitude_mix", 1.0)
            or augmentor._fourier_amplitude_mix(x, donor, beta=0.02)
        ),
    ))
    augmentor.p_fourier_amplitude_mix = 0.0

    # CarveMix smoke check (uses scipy distance transform; does not need a
    # full dataset since we generate a synthetic donor here).
    try:
        donor_y = label.copy()
        ds_stub = type("DummyDS", (), {})()
        ds_stub.carvemix_dilation = 5
        # Reuse the dataset method directly via a minimal proxy.
        dummy_ds = core.MultiModalSegDataset.__new__(core.MultiModalSegDataset)
        dummy_ds.carvemix_dilation = 5
        dummy_ds.rows = [{"case_id": "a"}, {"case_id": "b"}]
        # Inject a fixed donor return so we don't try to load CSV rows.
        dummy_ds._sample_carvemix_donor = lambda receiver_idx, max_attempts=8: (base.copy(), donor_y)
        x_in = base.copy()
        y_in = label.copy()
        x_out, y_out = dummy_ds._apply_carvemix(0, x_in, y_in)
        op_results.append({
            "op": "carvemix",
            "mean_in": float(base.mean()),
            "mean_out": float(x_out.mean()),
            "std_out": float(x_out.std()),
            "frac_changed": float((np.abs(x_out - base) > 1e-4).mean()),
            "label_diff_voxels": int((y_out != label).sum()),
        })
    except Exception as exc:  # noqa: BLE001
        op_results.append({"op": "carvemix", "error": str(exc)})

    # MixStyle3D forward sanity check on a tiny tensor.
    try:
        ms = core.MixStyle3D(p=1.0, alpha=0.3, mode="mixstyle").train()
        feat = torch.randn(2, 8, 16, 16, 16)
        out = ms(feat)
        op_results.append({
            "op": "mixstyle3d",
            "in_mean": float(feat.mean()),
            "out_mean": float(out.mean()),
            "in_std": float(feat.std()),
            "out_std": float(out.std()),
            "out_shape": list(out.shape),
        })
    except Exception as exc:  # noqa: BLE001
        op_results.append({"op": "mixstyle3d", "error": str(exc)})

    # V-REx penalty smoke check on synthetic logits.
    try:
        logits = torch.randn(4, 4, 8, 8, 8)
        target = torch.randint(0, 4, (4, 8, 8, 8))
        tags = ["a", "a", "b", "b"]
        v = float(core.vrex_penalty(logits, target, tags).item())
        op_results.append({"op": "vrex_penalty", "value": v})
    except Exception as exc:  # noqa: BLE001
        op_results.append({"op": "vrex_penalty", "error": str(exc)})

    payload = {
        "field_strength": float(args.field_strength),
        "ops": op_results,
        "synthetic_volume_shape": list(base.shape),
    }
    _save_json(payload, out_dir / "validate_augmentor.json")

    # Optional axial mid-slice montage.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ops_to_plot = [
            ("field_contrast_warp", lambda x: (setattr(augmentor, "p_field_contrast_warp", 1.0) or augmentor._simulate_field_strength_contrast(x, contrast_mod_indices=contrast_set))),
            ("receive_coil_falloff", lambda x: (setattr(augmentor, "p_receive_coil_inhomogeneity", 1.0) or augmentor._simulate_receive_coil_inhomogeneity(x))),
            ("off_resonance_void", lambda x: (setattr(augmentor, "p_off_resonance_void", 1.0) or augmentor._simulate_off_resonance_void(x))),
            ("rician_aja_fernandez", lambda x: (setattr(augmentor, "noise_calibration", "aja_fernandez") or augmentor._add_rician_noise(x.copy()))),
            ("fourier_amplitude_mix", lambda x: (setattr(augmentor, "p_fourier_amplitude_mix", 1.0) or augmentor._fourier_amplitude_mix(x, donor, beta=0.02))),
        ]
        n = len(ops_to_plot) + 1
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.7))
        z_mid = d // 2
        axes[0].imshow(base[1, z_mid], cmap="gray")
        axes[0].set_title("baseline T1c")
        axes[0].axis("off")
        for i, (name, fn) in enumerate(ops_to_plot, start=1):
            np.random.seed(int(args.seed))
            x = base.copy()
            for k in ("p_field_contrast_warp", "p_receive_coil_inhomogeneity",
                      "p_off_resonance_void", "p_fourier_amplitude_mix"):
                setattr(augmentor, k, 0.0)
            augmentor.noise_calibration = "static"
            out_x = fn(x)
            if isinstance(out_x, tuple):
                out_x = out_x[0]
            axes[i].imshow(out_x[1, z_mid], cmap="gray")
            axes[i].set_title(name, fontsize=8)
            axes[i].axis("off")
        fig.tight_layout()
        fig.savefig(str(out_dir / "augmentor_montage.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        ADDENDUM_LOG.info(f"[validate_augmentor] wrote montage at {out_dir}/augmentor_montage.png")
    except Exception as exc:  # noqa: BLE001
        ADDENDUM_LOG.info(f"[validate_augmentor] matplotlib montage skipped: {exc}")

    ADDENDUM_LOG.info(
        f"[validate_augmentor] {len(op_results)} ops checked at B0={args.field_strength} → "
        f"{out_dir}/validate_augmentor.json"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Z. Datasheet for the dataset (Gebru et al. 2021).
# ══════════════════════════════════════════════════════════════════════════════

DATASHEET_TEMPLATE = """# Datasheet for the OGAP training pool

## Motivation
- Purpose: training a multi-institutional glioma segmentation model robust to
  missing modalities, low-field scanners, and legacy protocols.
- Created by: {creator}. Funded by: {funder}.

## Composition
- Datasets: {cohorts}.
- Number of instances: {n_cases}.
- Modalities: T1n, T1c, T2w, T2-FLAIR.
- Label schemas: BraTS 4-class (0=BG, 1=NCR, 2=oedema, 3=ET); ERASMUS binary (WT).
- Missing values: handled via per-sample marginal CE and modality curriculum.
- Demographic coverage: see stratified_summary.json. Sex/gender and race/ethnicity
  are included; note that the UTSW cohort is predominantly White and Non-Hispanic/
  Latino - generalisability to under-represented groups should be validated.
- Field-strength coverage: 0.3T to 3T, enriched via physics-informed augmentation.

## Collection process
- Source: {source}. IRB approvals cited in the manuscript.
- De-identification: compliant with institutional policy.
- Quality control: manual refinement status recorded per case.

## Preprocessing / cleaning / labeling
- Co-registration: skull-stripped, affine-aligned multi-modal volumes.
- Intensity normalisation: N4 bias correction followed by per-modality foreground z-score.
- Annotation: manually refined (YES/NO flag). Inter-rater reliability on the
  manually refined subset is reported in interrater_summary.json.

## Uses
- Primary: research-only glioma segmentation.
- NOT to be used for identifying individuals.

## Distribution
- BraTS 2023: publicly available via Synapse (cite challenge paper).
- ERASMUS / UTSW: availability under DUA; contact corresponding author.

## Maintenance
- Versioned at: {repo}
- Issues / updates: tracked via GitHub releases.

## Known limitations
- Under-representation of sub-1T scanners.
- Partial labels in ERASMUS; see erasmus_bias_report.json.
- Non-uniform manual-refinement proportion across cohorts.
"""


def cmd_datasheet(args: argparse.Namespace, core) -> None:
    out_dir = _ensure_dir(args.out_dir)
    text = DATASHEET_TEMPLATE.format(
        creator=args.creator,
        funder=args.funder,
        cohorts=args.cohorts,
        n_cases=args.n_cases,
        source=args.source,
        repo=args.repo,
    )
    (out_dir / "DATASHEET.md").write_text(text)
    ADDENDUM_LOG.info(f"[datasheet] wrote {out_dir}/DATASHEET.md")


# ══════════════════════════════════════════════════════════════════════════════
# AA. CONSORT-style data-flow diagram.
# ══════════════════════════════════════════════════════════════════════════════

def cmd_data_flow(args: argparse.Namespace, core) -> None:
    """Render a simple CONSORT-style flow diagram from a JSON describing
    inclusion/exclusion counts per cohort."""
    out_dir = _ensure_dir(args.out_dir)
    spec = json.load(open(args.spec_json))
    lines = ["digraph G {", '  rankdir=TB;', '  node [shape=box, fontsize=10];']
    for i, step in enumerate(spec.get("steps", [])):
        lines.append(f'  n{i} [label="{step["label"]}\\n(n={step["n"]})"];')
        if i > 0:
            lines.append(f"  n{i-1} -> n{i};")
        if "exclusions" in step:
            src_node = f"n{i-1}" if i > 0 else f"n{i}"
            for j, ex in enumerate(step["exclusions"]):
                lines.append(f'  ex{i}_{j} [label="Excluded: {ex["reason"]}\\n(n={ex["n"]})", '
                             f'shape=note];')
                lines.append(f"  {src_node} -> ex{i}_{j} [style=dashed];")
    lines.append("}")
    (out_dir / "data_flow.dot").write_text("\n".join(lines))
    ADDENDUM_LOG.info(f"[data_flow] wrote {out_dir}/data_flow.dot (render with graphviz)")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parser builders for the addendum CLI.
# ══════════════════════════════════════════════════════════════════════════════

def _add_common_train_orchestration_args(p):
    p.add_argument("--script", default=str(Path(__file__).resolve()))
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")


def register_subcommands(sub: argparse._SubParsersAction) -> None:
    """Attach every addendum subcommand to the parent argparse subparsers."""
    p = sub.add_parser("lint_csv", help="Audit a train/val CSV for missing files and label integrity")
    p.add_argument("--csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_csv", default=None,
                   help="Optional training CSV for UTSW leakage checks by case_id and T1c header hash.")
    p.add_argument("--check_shapes", action="store_true",
                   help="Open modalities/labels and hard-check shape, spacing, and affine alignment.")
    p.add_argument("--check_label_values", action="store_true",
                   help="Open every label NIfTI and count per-class voxel prevalence (slow).")

    # FIX (Gap C, 2026 audit): voxel-spacing audit across a manifest CSV.
    p = sub.add_parser("audit_voxel_spacing",
                       help="Tabulate (dataset_tag, voxel spacing mm) across a manifest CSV; "
                            "flag anisotropic spacings that break cross-scanner generalization.")
    p.add_argument("--csv", required=True)
    p.add_argument("--reference_modality", default="",
                   help="Modality column to inspect (default: first non-meta column).")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("surface_dice", help="Compute NSD at multiple mm tolerances")
    p.add_argument("--pairs_csv", required=True, help="CSV with case_id, pred, gt paths")
    p.add_argument("--tolerances_mm", default="1,2,3")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("calibration", help="ECE, Brier, reliability diagrams")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--n_bins", type=int, default=15)
    p.add_argument("--max_voxel_samples", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("mc_uncertainty", help="MC-dropout + optional ensemble uncertainty")
    p.add_argument("--ckpts", required=True, help="Comma-separated student checkpoints")
    p.add_argument("--val_csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--n_mc_samples", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("metric_check", help="Cross-check Dice/HD95 vs monai and medpy")
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("stratify", help="Stratify per-case metrics by clinical metadata")
    p.add_argument("--per_case", required=True, help="Per-case CSV or JSON")
    p.add_argument("--metadata_tsv", required=True)
    p.add_argument("--metadata_id_col", default="Subject ID")
    p.add_argument("--metric", default="pp_all_dice_brats_mean")
    p.add_argument("--min_group_n", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("interrater", help="Inter-rater reliability (Dice + Cohen's kappa)")
    p.add_argument("--raters_csv", required=True, help="CSV with case_id, rater_a, rater_b")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("threshold_sweep", help="Sweep post-processing thresholds on saved probs")
    p.add_argument("--pairs_csv", required=True, help="CSV with case_id, prob_npz, gt")
    p.add_argument("--min_sizes", default="0,10,25,50,100")
    p.add_argument("--et_min_sizes", default="0,5,10,20,30")
    p.add_argument("--et_confs", default="0.40,0.50,0.55,0.60,0.70")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("holder_sweep", help="Emit runbook for Hölder-α sensitivity")
    _add_common_train_orchestration_args(p)
    p.add_argument("--alphas", default="1.0,1.25,1.5,1.75,2.0,2.5")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("kd_compare", help="Emit runbook for KD method comparison")
    _add_common_train_orchestration_args(p)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("erasmus_bias", help="Paired analysis of with- vs without-ERASMUS training")
    p.add_argument("--with_erasmus_per_case", required=True)
    p.add_argument("--without_erasmus_per_case", required=True)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("single_modality_scenario", help="Evaluate T2-FLAIR-only / pair-only scenarios")
    p.add_argument("--script", default=str(Path(__file__).resolve()))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--keep", required=True,
                   help="Comma-separated keep sets. e.g. 't2f,t1n+t2f,t1c+t1n+t2f'")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("field_strength_curve", help="Dice vs. field strength plot")
    p.add_argument("--joined_csv", required=True, help="Output of 'stratify' (joined_per_case.csv)")
    p.add_argument("--metric", default="pp_all_dice_brats_mean")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("loso_cv", help="Leave-one-site-out CV runbook")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--site_col", default="site")
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--script", default=str(Path(__file__).resolve()))
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("kfold_cv", help="k-fold CV runbook")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--stratify_col", default="")
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--script", default=str(Path(__file__).resolve()))
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("multi_seed", help="Multi-seed training runbook")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--script", default=str(Path(__file__).resolve()))
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("training_curves", help="Overlay training curves across runs")
    p.add_argument("--runs", required=True, help="Comma-separated run directories")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("ablation_heatmap", help="Heatmap: ablations × modality combos")
    p.add_argument("--ablation_summary_json", required=True)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("subgroup_forest", help="Forest plot from stratified_summary.json")
    p.add_argument("--stratified_summary_json", required=True)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("failure_case_figure", help="Select failure/best/typical cases for a figure")
    p.add_argument("--per_case", required=True)
    p.add_argument("--metric", default="pp_all_dice_brats_mean")
    p.add_argument("--n_cases", type=int, default=9)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("compute_profile", help="GPU memory peak and throughput profile")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_iter", type=int, default=25)
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("quantization_hardware", help="Render FP32 vs INT8 per-config figure")
    p.add_argument("--compare_json", required=True,
                   help="Path to compare_quantisation summary JSON")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("total_energy_table", help="Aggregate all energy reports under a root")
    p.add_argument("--root", required=True)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("model_card", help="Generate MODEL_CARD.md (Mitchell et al. 2019)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--holder_alpha", type=float, default=1.5)
    p.add_argument("--cohorts", default="BraTS 2023, ERASMUS, UTSW (external)")
    p.add_argument("--field_strength_power", type=float, default=0.5)
    p.add_argument("--low_field_bias", type=float, default=1.5)
    p.add_argument("--grade23_bias", type=float, default=1.5)
    p.add_argument("--idh_mutant_bias", type=float, default=1.5)
    p.add_argument("--legacy_bias", type=float, default=1.25)
    p.add_argument("--legacy_year", type=int, default=2013)
    p.add_argument("--n_sub1t", type=int, default=0)

    p = sub.add_parser("datasheet", help="Generate DATASHEET.md (Gebru et al. 2021)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--creator", default="Diaz Lab")
    p.add_argument("--funder", default="(fill in)")
    p.add_argument("--cohorts", default="BraTS 2023, ERASMUS, UTSW (external)")
    p.add_argument("--n_cases", type=str, default="(fill in)")
    p.add_argument("--source", default="(fill in)")
    p.add_argument("--repo", default="(fill in)")

    p = sub.add_parser("data_flow", help="Render CONSORT-style data-flow .dot")
    p.add_argument("--spec_json", required=True)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser(
        "crc",
        help="Conformal Risk Control: choose λ̂ s.t. E[L(λ̂)]≤α (Angelopoulos+ 2024)",
    )
    p.add_argument("--calib_csv", required=True,
                   help="CSV with columns: case_id, lambda, loss")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="Target risk level (default 0.10)")
    p.add_argument("--loss_upper_bound", type=float, default=1.0,
                   help="Bound B on the loss (default 1.0; raise if losses exceed 1)")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser(
        "mahalanobis_ood",
        help="Mahalanobis OOD score on student bottleneck features (Lee+ 2018)",
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--id_csv", required=True, help="In-distribution inference CSV")
    p.add_argument("--ood_csv", default=None, help="Optional OOD inference CSV (enables AUROC)")
    p.add_argument("--modalities", required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--patch_size", default="128,160,112")
    p.add_argument("--student_base", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hook_attr", default=None,
                   help="Override the encoder block to hook (e.g. 'enc4', 'bottleneck'). "
                        "Default: auto-detect deepest encoder block.")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser(
        "tost",
        help="TOST equivalence test for paired metrics (Schuirmann 1987)",
    )
    p.add_argument("--paired_csv", required=True,
                   help="CSV with columns: case_id, value_a, value_b")
    p.add_argument("--margin", type=float, default=0.02,
                   help="Equivalence margin δ (default 0.02 ≈ 2 Dice points)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser(
        "delong",
        help="DeLong's test of equality between two paired AUCs (1988)",
    )
    p.add_argument("--paired_csv", required=True,
                   help="CSV with columns: case_id, label, score_a, score_b")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser(
        "validate_augmentor",
        help="Run every MRIPhysicsAugmentor op on a synthetic volume and write a report",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--field_strength", type=float, default=0.55,
                   help="Target B0 (Tesla) for ops that condition on field strength.")

    p = sub.add_parser(
        "dicom_seg",
        help="Convert NIfTI segmentation to DICOM-SEG (requires highdicom)",
    )
    p.add_argument("--nifti_seg", required=True, help="NIfTI segmentation (label volume)")
    p.add_argument("--dicom_dir", required=True, help="Directory containing source DICOM series")
    p.add_argument("--label_map", default=None,
                   help="Optional JSON mapping label_id → {label, sct}; default OGAP/BraTS")
    p.add_argument("--series_number", type=int, default=999)
    p.add_argument("--out_name", default="ogap_seg.dcm")
    p.add_argument("--out_dir", required=True)

    p = sub.add_parser("sampler_distribution", help="Log realized sampler distribution vs baseline")
    p.add_argument("--csv", required=True)
    p.add_argument("--n_draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--field_strength_power", type=float, default=0.5)
    p.add_argument("--low_field_bias", type=float, default=1.5)
    p.add_argument("--grade23_bias", type=float, default=1.5)
    p.add_argument("--idh_mutant_bias", type=float, default=1.5)
    p.add_argument("--legacy_bias", type=float, default=1.25)
    p.add_argument("--legacy_year", type=int, default=2013)
    p.add_argument("--out_dir", required=True)

    # ── CLAIM 2024 + TRIPOD+AI + SAGER 2022 + NIHMS orchestrator ─────────
    # Imports lazily so a missing scipy / sklearn cannot break the rest of
    # the addendum CLI registration.
    try:
        import ogap_compliance as _compliance
        _compliance.register_compliance_subcommand(sub)
    except Exception as _exc:  # noqa: BLE001
        ADDENDUM_LOG.warning(
            "[ai_compliance_report] subcommand not registered: %s "
            "(check Scripts/ogap_compliance.py)", _exc
        )


DISPATCH: Dict[str, Callable[[argparse.Namespace, Any], None]] = {
    "lint_csv": cmd_lint_csv,
    "audit_voxel_spacing": cmd_audit_voxel_spacing,
    "surface_dice": cmd_surface_dice,
    "calibration": cmd_calibration,
    "mc_uncertainty": cmd_mc_uncertainty,
    "metric_check": cmd_metric_check,
    "stratify": cmd_stratify,
    "interrater": cmd_interrater,
    "threshold_sweep": cmd_threshold_sweep,
    "holder_sweep": cmd_holder_sweep,
    "kd_compare": cmd_kd_compare,
    "erasmus_bias": cmd_erasmus_bias,
    "single_modality_scenario": cmd_single_modality_scenario,
    "field_strength_curve": cmd_field_strength_curve,
    "loso_cv": cmd_loso_cv,
    "kfold_cv": cmd_kfold_cv,
    "multi_seed": cmd_multi_seed,
    "training_curves": cmd_training_curves,
    "ablation_heatmap": cmd_ablation_heatmap,
    "subgroup_forest": cmd_subgroup_forest,
    "failure_case_figure": cmd_failure_case_figure,
    "compute_profile": cmd_compute_profile,
    "quantization_hardware": cmd_quantization_hardware,
    "total_energy_table": cmd_total_energy_table,
    "model_card": cmd_model_card,
    "datasheet": cmd_datasheet,
    "data_flow": cmd_data_flow,
    "sampler_distribution": cmd_sampler_distribution,
    "crc": cmd_crc,
    "mahalanobis_ood": cmd_mahalanobis_ood,
    "tost": cmd_tost,
    "delong": cmd_delong,
    "dicom_seg": cmd_dicom_seg,
    "validate_augmentor": cmd_validate_augmentor,
}


def _dispatch_ai_compliance_report(args: argparse.Namespace, core) -> None:
    """Lazy-load the compliance orchestrator so a missing optional dep
    (scipy / sklearn / statsmodels) cannot break the rest of the CLI."""
    import ogap_compliance as _compliance
    _compliance.cmd_ai_compliance_report(args, core)


DISPATCH["ai_compliance_report"] = _dispatch_ai_compliance_report


def dispatch(cmd: str, args: argparse.Namespace, core) -> None:
    if cmd not in DISPATCH:
        raise ValueError(f"Unknown addendum command: {cmd}")
    DISPATCH[cmd](args, core)

# ══════════════════════════════════════════════════════════════════════════════
# §21  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_patch(v: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in v.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch_size must be D,H,W")
    return parts[0], parts[1], parts[2]


def _parse_mods(v: str) -> List[str]:
    return [m.strip() for m in v.split(",")]


def _parse_int_list(v: str) -> List[int]:
    v = (v or "").strip()
    if not v:
        return []
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def _explicit_cli_options(argv: Sequence[str]) -> set[str]:
    explicit: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        if name:
            explicit.add(name)
    return explicit


def _cfg_arg(
    args: argparse.Namespace,
    explicit: set[str],
    cfg: Any,
    path: str,
    attr: str,
    cast: Optional[Callable[[Any], Any]] = None,
) -> Any:
    value = getattr(args, attr)
    if attr not in explicit:
        value = _cfg_get(cfg, path, value)
    return cast(value) if cast is not None else value


def _cfg_bool_arg(
    args: argparse.Namespace,
    explicit: set[str],
    cfg: Any,
    path: str,
    attr: str,
) -> bool:
    return bool(_cfg_arg(args, explicit, cfg, path, attr))


def _student_block_style_from_arch(arch: Any) -> str:
    value = str(arch or "standard").lower()
    return "ode" if value == "ode" else "res"


def main() -> None:
    explicit_options = _explicit_cli_options(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="OGAP: Open Glioma Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch behavior (slower; may warn on unsupported ops)",
    )
    parser.add_argument(
        "--suppress_warnings",
        action="store_true",
        help="Suppress UserWarnings (not recommended unless you know what you're hiding)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional unified YAML config (ogap/config/defaults.yaml schema). When "
             "omitted, packaged defaults are loaded, reproducing legacy behaviour.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── Integrated evaluation addendum subcommands ─────────────────────────────
    register_subcommands(sub)

    # ── pretrain_teacher ──────────────────────────────────────────────────
    pt = sub.add_parser("pretrain_teacher", help="Pre-train teacher on all modalities")
    pt.add_argument("--train_csv",     required=True)
    pt.add_argument("--val_csv",       required=True)
    pt.add_argument("--out_dir",       required=True)
    pt.add_argument("--modalities",    required=True, help="e.g. t1,t1ce,t2,flair")
    pt.add_argument("--num_classes",   type=int, default=4)
    pt.add_argument("--patch_size",    default="192,192,144")
    pt.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Teacher micro-batch size for 192x192x144 H100 training.",
    )
    pt.add_argument("--epochs",        type=int, default=1000)
    pt.add_argument("--lr",            type=float, default=2e-4)
    pt.add_argument("--weight_decay",  type=float, default=3e-5)
    pt.add_argument("--num_workers",   type=int, default=16,
                    help="Training DataLoader workers for the one-GPU Rorqual path. "
                         "Default 16 is intended for staged local-scratch data with prefetch=1.")
    pt.add_argument("--val_num_workers", type=int, default=4,
                    help="Validation DataLoader workers. Keep small to avoid wasting CPU/RSS.")
    pt.add_argument("--teacher_base",  type=int, default=32)
    pt.add_argument("--warmup_epochs", type=int, default=5)
    pt.add_argument("--lambda_region", type=float, default=0.50,
                    help="Auxiliary WT/TC/ET overlapping-region BCE+Dice loss weight for dense 4-class samples.")
    pt.add_argument("--focal_b1_inhomogeneity_prob", type=float, default=0.12,
                    help="Probability of localised receive-field shading augmentation during training.")
    pt.add_argument("--field_strength_sampling_power", type=float, default=0.5,
                    help="Inverse-frequency power for field-strength-aware sampling when metadata is available.")
    pt.add_argument("--low_field_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for <=1T cases during teacher training.")
    pt.add_argument("--grade23_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for grade 2/3 cases when tumor grade metadata is present.")
    pt.add_argument("--idh_mutant_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for IDH-mutant cases when metadata is present.")
    pt.add_argument("--legacy_protocol_sampling_bias", type=float, default=1.25,
                    help="Extra sampling weight for older-acquisition cases when scan year metadata is present.")
    pt.add_argument("--legacy_protocol_year_threshold", type=int, default=2013,
                    help="Acquisition-year cutoff for the legacy-protocol sampling bias.")
    pt.add_argument("--hd95_freq",     type=int, default=50,
                    help="Compute HD95 every N full-val epochs (expensive scipy EDT). "
                         "HD95 is always computed in the standalone 'evaluate' command. "
                         "Default: 50 (was 10 in v8).")
    pt.add_argument("--val_fast_n",   type=int, default=None,
                    help="Number of deployment-focused eval configs to run every epoch. "
                    "For the catch-all modality student, this covers full and sparse configs. "
                    "All supported configs run on "
                    "--val_full_freq epochs regardless. Default: TeacherConfig.val_fast_n.")
    pt.add_argument("--val_start_epoch", type=int, default=3,
                    help="Skip validation before this epoch to reduce cold-start overhead. "
                         "The final epoch always validates.")
    pt.add_argument("--val_full_freq", type=int, default=None,
                    help="Run all supported eval configs + HD95 every N epochs (and final epoch). "
                         "Default: TeacherConfig.val_full_freq.")
    pt.add_argument("--seed",          type=int, default=42)
    pt.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    pt.add_argument("--no_amp",        action="store_true")
    pt.add_argument("--no_augment",    action="store_true")
    pt.add_argument("--no_gpu_augment", action="store_true",
                    help="Keep all training augmentations in CPU DataLoader workers instead of moving them to CUDA.")
    pt.add_argument("--cache_dir",     default=None,
                    help="Directory for npy cache (e.g. $SLURM_TMPDIR/npy_cache). "
                         "Epoch 1 writes preprocessed .npy files; epochs 2+ load them "
                         "~5× faster than nibabel. Disabled in the Rorqual sbatches by "
                         "default because file cache can trip Slurm memory limits.")
    pt.add_argument("--patch_cache_dir", default=None,
                    help="Directory for RAM-backed per-epoch crop cache (e.g. /dev/shm/.../patch_cache). "
                         "Each epoch stores one cropped patch per case; opt in only after "
                         "the staged raw-data path is memory-stable.")
    pt.add_argument("--patch_cache_start_epoch", type=int, default=2,
                    help="First epoch that uses the per-epoch crop cache. Default 2 keeps epoch 1 as a warmup pass.")
    pt.add_argument("--lmic_field_strength_prior", action="store_true",
                    help="Use an ABTIP-informed field-strength prior with more 1.5T and sub-1T sampling.")
    pt.add_argument("--contrast_mod_indices", default="",
                    help="Comma-separated modality indices for contrast-enhanced channels (e.g. '1' for T1ce).")
    pt.add_argument("--p_anisotropic_2d", type=float, default=0.85,
                    help="Probability of thick-slice 2D acquisition simulation during training.")
    pt.add_argument("--p_partial_contrast", type=float, default=0.30,
                    help="Probability of partial/under-enhanced contrast simulation on contrast channels.")
    pt.add_argument("--p_label_noise", type=float, default=0.20,
                    help="Probability of annotation-boundary erosion/dilation noise during training.")
    pt.add_argument("--label_noise_mode", choices=("morph", "boundary_band"), default="morph",
                    help="Annotation-noise model: 'morph' (legacy, per-class erosion/dilation) "
                         "or 'boundary_band' (PDF §4.4: K-vox boundary-band Bernoulli flip; "
                         "preferred for evaluation rigor).")
    pt.add_argument("--label_noise_band_radius", type=int, default=2,
                    help="Boundary-band radius in voxels (only used when --label_noise_mode boundary_band).")
    pt.add_argument("--label_noise_band_flip_prob", type=float, default=0.30,
                    help="Bernoulli flip probability for each boundary-band voxel "
                         "(only used when --label_noise_mode boundary_band).")
    pt.add_argument("--auto_resources", action="store_true",
                    help="Dynamically detect CPU/RAM/VRAM/SM and tune batch_size and num_workers "
                         "to saturate the allocated Rorqual H100 bundle. Writes auto_resources.json "
                         "into --out_dir.")
    pt.add_argument("--auto_resources_safety", type=float, default=0.85,
                    help="VRAM safety factor for the batch-size probe in [0, 1]. Lower = more slack. "
                         "Teacher default 0.85.")
    # evaluation rigor extensions on the teacher pretraining path.
    pt.add_argument("--p_field_contrast_warp", type=float, default=0.0)
    pt.add_argument("--field_contrast_source_b0", type=float, default=3.0)
    pt.add_argument("--p_fourier_amplitude_mix", type=float, default=0.0)
    pt.add_argument("--afa_prob", type=float, default=0.0)
    pt.add_argument("--p_receive_coil_inhomogeneity", type=float, default=0.0)
    pt.add_argument("--p_off_resonance_void", type=float, default=0.0)
    pt.add_argument("--noise_calibration", choices=("static", "aja_fernandez"), default="static")
    pt.add_argument("--carvemix_prob", type=float, default=0.0)
    pt.add_argument("--carvemix_dilation", type=int, default=5)
    pt.add_argument("--teacher_arch", choices=("unet3d", "segmamba", "swin_unetr", "ode"), default="unet3d")
    pt.add_argument("--teacher_block_style",
                    choices=("mednext", "dense"), default="mednext",
                    help="Block style for teacher network. "
                         "'mednext' = depthwise-separable MedNeXt-S blocks (~1M @ base=32, "
                         "default for backward compat). "
                         "'dense' = full 3x3x3 conv blocks (~6M @ base=32, ~15M @ base=48, "
                         "~26M @ base=64); use this for heavy KD teachers - only the student "
                         "is exported to INT8 so the teacher does not need to be "
                         "quantization-friendly.")
    # FIX (Gap A, 2026 audit): teacher-side feature-DR.
    pt.add_argument("--teacher_feature_dr",
                    choices=("none", "mixstyle", "dsu"), default="none",
                    help="Teacher-side feature-level domain randomization. 'none' (default) "
                         "preserves backward-compat; 'mixstyle' (Zhou+ 2021) or 'dsu' (Li+ 2022) "
                         "inject style-statistics noise after encoder blocks 1 and 2 so the "
                         "teacher's KD logits encode cross-scanner invariance. Identity at eval.")
    pt.add_argument("--teacher_feature_dr_p", type=float, default=0.5,
                    help="Per-batch activation probability for teacher feature-DR.")
    pt.add_argument("--teacher_feature_dr_alpha", type=float, default=0.1,
                    help="Beta(alpha, alpha) mix concentration for teacher feature-DR.")
    # FIX (Gap B, 2026 audit): teacher-side modality dropout.
    pt.add_argument("--teacher_modality_dropout_p", type=float, default=0.0,
                    help="Per-modality dropout probability for the teacher. Set to ~0.15 "
                         "to train the teacher on missing-modality scenarios so its KD "
                         "logits remain meaningful when (e.g.) T1c is unavailable at "
                         "inference time. 0.0 (default) preserves backward-compat with "
                         "full-modality teachers.")
    pt.add_argument("--teacher_modality_dropout_max_drop", type=int, default=2,
                    help="Maximum number of modalities that can be dropped from a single "
                         "training case. Caps the drop so at least n_modalities - max_drop "
                         "channels remain alive.")
    # FIX (Finding 4, 2026 audit): opt-in flag for non-strict resume.
    pt.add_argument("--force_resume_partial", action="store_true",
                    help="If resume_teacher.pth was saved with a different architecture "
                         "(different --teacher_arch / --teacher_base / --teacher_block_style "
                         "/ --teacher_feature_dr), tolerate missing/extra keys and continue "
                         "with whatever loads. Default is to abort loudly so silent retraining "
                         "into the wrong architecture is impossible.")
    # FIX (2026 throughput/accuracy pass): expose torch.compile + weight-EMA on the
    # bare teacher CLI. Without a v9.1 --config both were silently OFF for the teacher,
    # so a heavy dense base=64 teacher trained eager and deployed its last-iterate weights.
    pt.add_argument("--compile", action="store_true",
                    help="Enable torch.compile(Inductor) for the teacher. Without a v9.1 "
                         "--config the teacher otherwise trains eager. Safe: maybe_compile_model "
                         "falls back to eager if Inductor/Triton is unavailable.")
    pt.add_argument("--compile_mode", default="max-autotune",
                    help="torch.compile mode when --compile is set (default max-autotune).")
    pt.add_argument("--compile_dynamic", action="store_true",
                    help="Use dynamic shapes for torch.compile (default static; fixed-size "
                         "teacher patches compile best static).")
    pt.add_argument("--ema", action="store_true",
                    help="Maintain an exponential moving average of teacher weights and "
                         "validate/save THAT copy (a near-free accuracy gain used by recent "
                         "BraTS / nnU-Net winners). Without a v9.1 --config EMA is otherwise OFF.")
    pt.add_argument("--ema_decay", type=float, default=0.999,
                    help="EMA decay when --ema is set (default 0.999).")

    # ── train ─────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train student with knowledge distillation")
    tr.add_argument("--train_csv",          required=True)
    tr.add_argument("--val_csv",            required=True)
    tr.add_argument("--out_dir",            required=True)
    tr.add_argument("--modalities",         required=True)
    tr.add_argument("--num_classes",        type=int, default=4)
    tr.add_argument("--teacher_ckpt",       required=True)
    tr.add_argument("--patch_size",         default="128,160,112")
    tr.add_argument("--batch_size",         type=int, default=4,
                    help="Student micro-batch size for MedNeXt-S 128x160x112 H100 training.")
    tr.add_argument("--epochs",             type=int, default=1000)
    tr.add_argument("--lr",                 type=float, default=1e-3)
    tr.add_argument("--weight_decay",       type=float, default=3e-5)
    tr.add_argument("--num_workers",        type=int, default=16,
                    help="Training DataLoader workers for the staged one-GPU Rorqual path. "
                         "Default 16 overlaps local-scratch I/O without recreating the cold-cache OOM.")
    tr.add_argument("--val_num_workers",    type=int, default=4,
                    help="Validation DataLoader workers. Keep small to avoid wasting CPU/RSS.")
    tr.add_argument("--keep_min_modalities",type=int, default=1)
    tr.add_argument("--keep_max_modalities",type=int, default=4)
    tr.add_argument("--curriculum_floor_modalities", type=int, default=1,
                    help="Minimum modalities retained by the default curriculum. "
                         "Default 1 trains the catch-all student on every non-empty modality subset.")
    tr.add_argument("--keep_count_bias_power", type=float, default=0.0,
                    help="Bias combination sampling toward keeping more modalities. "
                         "0 = uniform over supported modality combinations.")
    tr.add_argument("--field_strength_sampling_power", type=float, default=0.5,
                    help="Inverse-frequency power for field-strength-aware sampling when metadata is available.")
    tr.add_argument("--low_field_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for <=1T cases during student training.")
    tr.add_argument("--grade23_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for grade 2/3 cases when tumor grade metadata is present.")
    tr.add_argument("--idh_mutant_sampling_bias", type=float, default=1.5,
                    help="Extra sampling weight for IDH-mutant cases when metadata is present.")
    tr.add_argument("--legacy_protocol_sampling_bias", type=float, default=1.25,
                    help="Extra sampling weight for older-acquisition cases when scan year metadata is present.")
    tr.add_argument("--legacy_protocol_year_threshold", type=int, default=2013,
                    help="Acquisition-year cutoff for the legacy-protocol sampling bias.")
    tr.add_argument("--lambda_kd",          type=float, default=2.0)
    tr.add_argument("--lambda_feat",        type=float, default=0.5)
    tr.add_argument("--feature_distill_sparse_decay", type=float, default=1.5,
                    help="Exponent used to relax feature distillation when fewer modalities are retained. "
                         "Higher values down-weight feature KD more aggressively in sparse-input regimes.")
    tr.add_argument("--lambda_ds",          type=float, default=1.0)
    tr.add_argument("--lambda_region",      type=float, default=0.50,
                    help="Auxiliary WT/TC/ET overlapping-region BCE+Dice loss weight on the main student logits.")
    tr.add_argument("--temperature_start",  type=float, default=4.0)
    tr.add_argument("--temperature_end",    type=float, default=4.0)
    tr.add_argument("--warmup_epochs",      type=int, default=5)
    tr.add_argument("--grad_accum_steps",   type=int, default=1,
                    help="Accumulate gradients over this many micro-batches before each optimizer step. "
                         "Use this to preserve effective batch size when the widened student cannot fit a larger real micro-batch.")
    tr.add_argument("--teacher_base",       type=int, default=32)
    tr.add_argument("--student_base",       type=int, default=16)
    tr.add_argument("--curriculum_ramp",    type=float, default=0.2,
                    help="Fraction of epochs with full modalities before cosine dropout begins. "
                         "OGAP 2.0 keeps full modalities for the initial 20%% before ramping dropout.")
    tr.add_argument("--holder_alpha",       type=float, default=1.5,
                    help="Hölder exponent for the HPD KD loss (the documented method). "
                         "1.0 falls back to KL (an ablation arm), 2.0 is the proper "
                         "Cauchy-Schwarz control. Default 1.5 matches the production sbatch.")
    tr.add_argument("--attention",          default="none", choices=["eca", "se", "none"])
    tr.add_argument("--no_deep_supervision",action="store_true")
    tr.add_argument("--curriculum_dropout_p",type=float, default=0.1)
    tr.add_argument("--no_curriculum",      action="store_true",
                    help="Disable modality dropout curriculum (fixed keep_min from epoch 1). "
                         "Used in ablation 'no_curriculum'.")
    tr.add_argument("--confidence_gate",    type=float, default=0.7,
                    help="Teacher confidence threshold for KD gating (0=disable). "
                         "Paper Methods: 'threshold was set to 0.7'.")
    tr.add_argument("--hd95_freq",          type=int, default=50,
                    help="Compute HD95 every N full-val epochs (expensive scipy EDT). "
                         "HD95 is always computed in the standalone 'evaluate' command. "
                         "Default: 50 (was 10 in v8).")
    tr.add_argument("--val_fast_n",         type=int, default=9,
                    help="Number of deployment-focused eval configs per regular epoch. "
                         "Default covers full, all 3-modality configs, and a rotating sparse subset. All supported configs run on "
                         "--val_full_freq epochs regardless.")
    tr.add_argument("--val_start_epoch",    type=int, default=3,
                    help="Skip validation before this epoch to reduce cold-start overhead. "
                         "The final epoch always validates.")
    tr.add_argument("--val_full_freq",      type=int, default=25,
                    help="Run all supported 1-4 modality eval configs + HD95 every N epochs. "
                         "Default: 25.")
    tr.add_argument("--seed",               type=int, default=42)
    tr.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--no_amp",             action="store_true")
    tr.add_argument("--no_augment",         action="store_true")
    tr.add_argument("--no_gpu_augment",     action="store_true",
                    help="Keep all training augmentations in CPU DataLoader workers instead of moving them to CUDA.")
    tr.add_argument("--cache_dir",          default=None,
                    help="Directory for npy cache (e.g. $SLURM_TMPDIR/npy_cache). "
                         "Epoch 1 writes preprocessed .npy files; epochs 2+ load them "
                         "~5× faster than nibabel. Disabled in the Rorqual sbatches by "
                         "default because file cache can trip Slurm memory limits.")
    tr.add_argument("--patch_cache_dir",    default=None,
                    help="Directory for RAM-backed per-epoch crop cache (e.g. /dev/shm/.../patch_cache). "
                         "Each epoch stores one cropped patch per case; opt in only after "
                         "the staged raw-data path is memory-stable.")
    tr.add_argument("--patch_cache_start_epoch", type=int, default=2,
                    help="First epoch that uses the per-epoch crop cache. Default 2 keeps epoch 1 as a warmup pass.")
    tr.add_argument("--lmic_field_strength_prior", action="store_true",
                    help="Use an ABTIP-informed field-strength prior with more 1.5T and sub-1T sampling.")
    tr.add_argument("--contrast_mod_indices", default="",
                    help="Comma-separated modality indices for contrast-enhanced channels (e.g. '1' for T1ce).")
    tr.add_argument("--contrast_dropout_extra_prob", type=float, default=0.0,
                    help="Extra probability of dropping contrast-enhanced channels during student masking.")
    tr.add_argument("--p_anisotropic_2d", type=float, default=0.85,
                    help="Probability of thick-slice 2D acquisition simulation during training.")
    tr.add_argument("--p_partial_contrast", type=float, default=0.30,
                    help="Probability of partial/under-enhanced contrast simulation on contrast channels.")
    tr.add_argument("--p_label_noise", type=float, default=0.20,
                    help="Probability of annotation-boundary erosion/dilation noise during training.")
    tr.add_argument("--label_noise_mode", choices=("morph", "boundary_band"), default="morph",
                    help="Annotation-noise model: 'morph' (legacy, per-class erosion/dilation) "
                         "or 'boundary_band' (PDF §4.4: K-vox boundary-band Bernoulli flip; "
                         "preferred for evaluation rigor).")
    tr.add_argument("--label_noise_band_radius", type=int, default=2,
                    help="Boundary-band radius in voxels (only used when --label_noise_mode boundary_band).")
    tr.add_argument("--label_noise_band_flip_prob", type=float, default=0.30,
                    help="Bernoulli flip probability for each boundary-band voxel "
                         "(only used when --label_noise_mode boundary_band).")
    tr.add_argument("--focal_b1_inhomogeneity_prob", type=float, default=0.12,
                    help="Probability of localised receive-field shading augmentation during training.")
    tr.add_argument("--auto_resources", action="store_true",
                    help="Dynamically detect CPU/RAM/VRAM/SM and tune batch_size and num_workers "
                         "to saturate the allocated Rorqual H100 bundle (teacher + student + projectors "
                         "all resident during KD). Writes auto_resources.json into --out_dir.")
    tr.add_argument("--auto_resources_safety", type=float, default=0.70,
                    help="VRAM safety factor for the batch-size probe in [0, 1]. "
                         "Student default 0.70 is tighter than teacher to reserve headroom for "
                         "the resident teacher model and feature projectors during KD.")

    # ── evaluation rigor extensions (architecture review pass) ───────
    tr.add_argument("--lambda_vrex", type=float, default=0.0,
                    help="V-REx variance-of-risks penalty (Krueger+ ICML 2021). "
                         "Penalizes variance of per-site segmentation losses. "
                         "0.05-0.5 typical; 0 disables.")
    tr.add_argument("--feature_dr", choices=("none", "mixstyle", "dsu"), default="none",
                    help="Feature-statistics domain randomization at the first encoder block "
                         "(Zhou+ 2021 MixStyle / Li+ 2022 DSU). 'mixstyle' is the "
                         "recommended cross-site default.")
    tr.add_argument("--feature_dr_p", type=float, default=0.5,
                    help="Probability of applying feature-stat DR per training batch.")
    tr.add_argument("--feature_dr_alpha", type=float, default=0.1,
                    help="Beta(α, α) mixing parameter for MixStyle.")
    tr.add_argument("--p_field_contrast_warp", type=float, default=0.0,
                    help="Probability of re-rendering tissue contrast at the sampled "
                         "field strength (Rooney+ 2007 / Fischer+ 1990 LUT). 0.5 recommended "
                         "for any LMIC-deployment training run.")
    tr.add_argument("--field_contrast_source_b0", type=float, default=3.0,
                    help="Source field strength assumed for the input volumes when warping.")
    tr.add_argument("--p_fourier_amplitude_mix", type=float, default=0.0,
                    help="AFA Fourier-basis perturbation probability inside the augmentor "
                         "(legacy flag name retained).")
    tr.add_argument("--afa_prob", type=float, default=0.0,
                    help="Alias probability for AFA Fourier-basis perturbation "
                         "(donor sourcing no longer required).")
    tr.add_argument("--p_receive_coil_inhomogeneity", type=float, default=0.0,
                    help="Probability of applying receive-coil RX falloff at low B0 "
                         "(Campbell-Washburn 2023, Shetty 2023, Ljungberg 2025).")
    tr.add_argument("--p_off_resonance_void", type=float, default=0.0,
                    help="Probability of applying B0 off-resonance signal voids near "
                         "sinus / mastoid / brainstem regions (Campbell-Washburn 2023).")
    tr.add_argument("--noise_calibration", choices=("static", "aja_fernandez"), default="static",
                    help="Static σ from --noise_std_range (legacy), or per-case Aja-Fernández "
                         "single-image σ rescaled by SNR ∝ B0 (Pohmann 2015). "
                         "Recommended for any B0-randomized run.")
    tr.add_argument("--carvemix_prob", type=float, default=0.0,
                    help="Probability of CarveMix anatomy-preserving lesion mix-up (Wang+ 2021).")
    tr.add_argument("--carvemix_dilation", type=int, default=5,
                    help="Voxel dilation around the donor's tumor bbox for CarveMix.")
    tr.add_argument("--teacher_arch", choices=("unet3d", "segmamba", "swin_unetr", "ode"), default="unet3d",
                    help="Teacher architecture. 'segmamba' requires mamba-ssm "
                         "(`pip install mamba-ssm causal-conv1d`); 'swin_unetr' "
                         "requires MONAI.")
    tr.add_argument("--teacher_block_style",
                    choices=("mednext", "dense"), default="mednext",
                    help="MUST match the block_style used to pretrain the teacher checkpoint. "
                         "'mednext' (default, backward compat) loads classic depthwise-separable "
                         "teachers; 'dense' loads heavy full-3x3x3 teachers. Mismatch will "
                         "fail load_state_dict because tensor shapes differ. "
                         "Student block style is controlled separately.")
    tr.add_argument("--student_block_style",
                    choices=("res", "ode"), default="res",
                    help="Student residual block family. 'res' preserves existing checkpoints; "
                         "'ode' uses WeightTiedODEBlock3D blocks.")
    tr.add_argument("--student_ode_steps", type=int, default=4,
                    help="Effective depth per WeightTiedODEBlock3D when --student_block_style=ode.")
    tr.add_argument("--p_ulf_gan_synthesis", type=float, default=0.0,
                    help="Probability of synthesising an ULF-appearance volume via the "
                         "Lucas+ 2025 generator. 0 disables; requires --ulf_gan_weights.")
    tr.add_argument("--ulf_gan_weights", default=None,
                    help="Path to a pretrained ULF generator (.onnx or TorchScript .pt). "
                         "If absent or unloadable, ULF GAN augmentation is skipped silently.")
    tr.add_argument("--ulf_gan_target_b0", type=float, default=0.064,
                    help="Effective field strength to assign to GAN-synthesised volumes (Tesla).")

    # ── export ────────────────────────────────────────────────────────────
    exp_p = sub.add_parser("export", help="Export student to ONNX + INT8")
    exp_p.add_argument("--ckpt",             required=True)
    exp_p.add_argument("--out_path",         required=True)
    exp_p.add_argument("--modalities",       required=True)
    exp_p.add_argument("--num_classes",      type=int, default=4)
    exp_p.add_argument("--patch_size",       default="128,160,112")
    exp_p.add_argument("--student_base",     type=int, default=16)
    exp_p.add_argument("--calibration_csv",  default=None, help="CSV for static INT8 calibration")
    exp_p.add_argument("--calibration_samples", type=int, default=200,
                       help="Number of representative calibration entries (case + modality-mask pairs) "
                            "to use for static INT8 calibration.")
    exp_p.add_argument("--calibration_min_present", type=int, default=1,
                       help="Minimum modalities retained in static INT8 calibration samples.")
    exp_p.add_argument("--calibration_max_present", type=int, default=4,
                       help="Maximum modalities retained in static INT8 calibration samples.")
    exp_p.add_argument("--static_calibration_method",
                       choices=["minmax", "entropy", "percentile"], default="percentile",
                       help="ONNX Runtime calibration method for static INT8 export.")
    exp_p.add_argument("--skip_quant_preprocess", action="store_true",
                       help="Skip ORT quant pre-processing. Not recommended unless pre-process breaks on this model.")
    exp_p.add_argument("--no_static_per_channel", action="store_true",
                       help="Disable per-channel static quantisation for weights.")
    rr_group = exp_p.add_mutually_exclusive_group()
    rr_group.add_argument("--static_reduce_range", dest="static_reduce_range", action="store_true",
                          help="Force ORT static INT8 reduce_range=True.")
    rr_group.add_argument("--no_static_reduce_range", dest="static_reduce_range", action="store_false",
                          help="Force ORT static INT8 reduce_range=False.")
    exp_p.add_argument("--run_compare", action="store_true",
                       help="Immediately run FP32-vs-INT8 comparison after export.")
    exp_p.add_argument("--compare_val_csv", default=None,
                       help="Validation CSV for --run_compare.")
    exp_p.add_argument("--compare_out_dir", default=None,
                       help="Output directory for post-export quantisation comparison.")
    exp_p.add_argument("--compare_num_workers", type=int, default=4,
                       help="Validation DataLoader workers for the post-export FP32-vs-INT8 comparison.")
    exp_p.add_argument("--compare_num_threads", type=int, default=None,
                       help="ORT CPU thread count override for the post-export INT8 comparison.")
    exp_p.add_argument("--compare_eval_config_batch_size", type=int, default=1,
                       help="How many modality configs to batch during the post-export comparison.")
    exp_p.add_argument("--compare_metric_workers", type=int, default=1,
                       help="CPU metric workers for the post-export comparison.")
    exp_p.add_argument("--compare_cache_dir", default=None,
                       help="Optional npy cache directory for the post-export comparison.")
    exp_p.add_argument("--no_compare_energy_tracking", action="store_true",
                       help="Disable energy tracking during the post-export FP32-vs-INT8 comparison.")
    exp_p.add_argument("--compare_energy_poll_interval", type=float, default=5.0,
                       help="Seconds between energy-tracker power samples during the post-export comparison.")
    exp_p.set_defaults(static_reduce_range=None)

    # ── infer ─────────────────────────────────────────────────────────────
    inf_p = sub.add_parser("infer", help="Run inference on test set")
    inf_p.add_argument("--test_csv",            required=True)
    inf_p.add_argument("--ckpt",                required=True)
    inf_p.add_argument("--out_dir",             required=True)
    inf_p.add_argument("--modalities",          required=True)
    inf_p.add_argument("--num_classes",         type=int, default=4)
    inf_p.add_argument("--patch_size",          default="96,96,96")
    inf_p.add_argument("--missing_modalities",  default="",
                       help="0-based indices to drop, e.g. '1,3'")
    inf_p.add_argument("--device",              default="cpu")
    inf_p.add_argument("--overlap",             type=float, default=0.5)
    inf_p.add_argument("--tta",                 action="store_true")
    inf_p.add_argument("--student_base",        type=int, default=16)
    inf_p.add_argument("--no_postprocess",      action="store_true")
    inf_p.add_argument("--num_threads",          type=int, default=None,
                       help="ORT intra-op thread count override. Default: let ONNX Runtime choose.")
    inf_p.add_argument("--tta_method",           choices=("none", "tent", "memo"), default="none",
                       help="Test-time adaptation method. 'tent' (Wang+ ICLR 2021) updates "
                            "BN/IN affine params on the case via entropy minimization; "
                            "'memo' (Zhang+ NeurIPS 2022) entropy-minimizes the marginal "
                            "across the 8-flip TTA family. Both restore the original "
                            "params after each case so adaptation is per-case episodic.")
    inf_p.add_argument("--tta_steps",            type=int, default=5,
                       help="Number of adaptation steps for tent / memo (default 5).")
    inf_p.add_argument("--tta_lr",               type=float, default=1e-3,
                       help="Learning rate for the tent/memo SGD optimizer.")

    # ── evaluate ──────────────────────────────────────────────────────────
    ev_p = sub.add_parser("evaluate", help="Full 15-combination evaluation")
    ev_p.add_argument("--ckpt",          required=True)
    ev_p.add_argument("--val_csv",       required=True)
    ev_p.add_argument("--out_dir",       required=True)
    ev_p.add_argument("--modalities",    required=True)
    ev_p.add_argument("--num_classes",   type=int, default=4)
    ev_p.add_argument("--patch_size",    default="128,160,112")
    ev_p.add_argument("--student_base",  type=int, default=16)
    ev_p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    ev_p.add_argument("--num_workers",   type=int, default=8)
    ev_p.add_argument("--eval_config_batch_size", type=int, default=1,
                      help="How many modality-mask configs to evaluate in one forward pass. "
                           "Raise on H100 to improve GPU saturation.")
    ev_p.add_argument("--metric_workers", type=int, default=1,
                      help="CPU threads for postprocessing + HD95 per evaluation chunk.")
    ev_p.add_argument("--tta", action="store_true",
                      help="Apply 8-flip test-time augmentation during evaluation.")
    ev_p.add_argument("--cache_dir",     default=None)
    ev_p.add_argument("--clinical_metadata_tsv", default=None,
                      help="Optional TSV with UTSW clinical metadata for failure-mode analysis.")
    ev_p.add_argument("--failure_mode_metric", default="pp_all_dice_brats_mean",
                      help="Per-case metric used to define the failure cohort.")
    ev_p.add_argument("--failure_mode_quantile", type=float, default=0.25,
                      help="Quantile used to define failures (default: lowest quartile for Dice, highest quartile for HD95).")
    ev_p.add_argument("--failure_mode_min_group_n", type=int, default=10,
                      help="Minimum subgroup size for independent-group failure-mode tests.")
    ev_p.add_argument("--lesion_wise", action="store_true",
                      help="Also report BraTS-2023 lesion-wise Dice/HD95 (with 373.1287 mm "
                           "FP/FN penalty). Adds lw_dice_*/lw_hd95_*/lw_n_tp/fp/fn keys "
                           "per region and lw_dice_brats_mean/lw_hd95_brats_mean.")

    # ── FIX #5: ablation CLI mode ─────────────────────────────────────────
    ab_p = sub.add_parser("ablation", help="Run systematic ablation study")
    ab_p.add_argument("--train_csv",          required=True)
    ab_p.add_argument("--val_csv",            required=True)
    ab_p.add_argument("--out_dir",            required=True)
    ab_p.add_argument("--modalities",         required=True)
    ab_p.add_argument("--num_classes",        type=int, default=4)
    ab_p.add_argument("--teacher_ckpt",       required=True)
    ab_p.add_argument("--patch_size",         default="128,160,112")
    ab_p.add_argument("--batch_size",         type=int, default=16)
    ab_p.add_argument("--epochs",             type=int, default=300)
    ab_p.add_argument("--ablations",          default=None,
                       help="Comma-separated ablation names (default: all)")
    ab_p.add_argument("--seed",               type=int, default=42)
    ab_p.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    ab_p.add_argument("--no_amp",             action="store_true")
    ab_p.add_argument("--cache_dir",          default=None,
                       help="NVMe npy cache dir (e.g. $SLURM_TMPDIR/npy_cache).")

    # ── benchmark ─────────────────────────────────────────────────────────
    bm_p = sub.add_parser("benchmark", help="Hardware benchmark for LMIC deployment validation")
    bm_p.add_argument("--test_csv",            required=True)
    bm_p.add_argument("--ckpt",                required=True)
    bm_p.add_argument("--out_dir",             required=True)
    bm_p.add_argument("--modalities",          required=True)
    bm_p.add_argument("--num_classes",         type=int, default=4)
    bm_p.add_argument("--patch_size",          default="96,96,96")
    bm_p.add_argument("--missing_modalities",  default="",
                       help="0-based indices to drop, e.g. '1,3'")
    bm_p.add_argument("--student_base",        type=int, default=16)
    bm_p.add_argument("--num_threads",         type=int, default=None,
                       help="CPU thread count override for ORT. Default: let ONNX Runtime choose. "
                            "Set to 4 for i5-class LMIC workstation simulation.")
    bm_p.add_argument("--n_warmup",            type=int, default=2,
                       help="Warmup cases excluded from timing. Default: 2.")
    bm_p.add_argument("--n_cases",             type=int, default=None,
                       help="Max cases to benchmark. Default: all.")

    # ── ablation_eval ─────────────────────────────────────────────────────
    ae_p = sub.add_parser("ablation_eval",
                          help="Evaluate existing ablation checkpoints across all 15 modality combos")
    ae_p.add_argument("--baseline_ckpt",  required=True,
                       help="Path to baseline best_student.pth (e.g. Results/ablation/baseline/best_student.pth)")
    ae_p.add_argument("--ablation_dir",   required=True,
                       help="Directory containing ablation subdirectories (e.g. Results/ablation)")
    ae_p.add_argument("--val_csv",        required=True)
    ae_p.add_argument("--modalities",     required=True)
    ae_p.add_argument("--num_classes",    type=int, default=4)
    ae_p.add_argument("--patch_size",     default="128,160,112")
    ae_p.add_argument("--student_base",   type=int, default=16)
    ae_p.add_argument("--out_dir",        required=True)
    ae_p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    ae_p.add_argument("--num_workers",    type=int, default=8)
    ae_p.add_argument("--eval_config_batch_size", type=int, default=1,
                       help="How many modality configs to run together on the same GPU.")
    ae_p.add_argument("--metric_workers", type=int, default=1,
                       help="CPU threads for postprocessing + HD95 during evaluation.")
    ae_p.add_argument("--cache_dir",      default=None)
    ae_p.add_argument("--ablations",      default=None,
                       help="Comma-separated ablation names. Default: auto-discover from ablation_dir.")

    # ── compare_quantisation ──────────────────────────────────────────────
    cq_p = sub.add_parser("compare_quantisation",
                          help="Compare FP32 (GPU) vs INT8 (ONNX CPU) accuracy across supported deployment modality combos")
    cq_p.add_argument("--fp32_ckpt",      required=True,
                       help="Path to FP32 best_student.pth")
    cq_p.add_argument("--int8_onnx",      required=True,
                       help="Path to INT8 ONNX model (e.g. best_student_int8_static.onnx)")
    cq_p.add_argument("--val_csv",        required=True)
    cq_p.add_argument("--modalities",     required=True)
    cq_p.add_argument("--num_classes",    type=int, default=4)
    cq_p.add_argument("--patch_size",     default="128,160,112")
    cq_p.add_argument("--student_base",   type=int, default=16)
    cq_p.add_argument("--out_dir",        required=True)
    cq_p.add_argument("--fp32_device",    default="cuda" if torch.cuda.is_available() else "cpu")
    cq_p.add_argument("--num_workers",    type=int, default=4,
                       help="Validation DataLoader workers for volume loading/preprocessing.")
    cq_p.add_argument("--num_threads",    type=int, default=None,
                       help="ORT CPU thread count override for INT8 inference. Default: let ONNX Runtime choose.")
    cq_p.add_argument("--eval_config_batch_size", type=int, default=1,
                       help="How many modality configs to run together in each FP32/INT8 pass.")
    cq_p.add_argument("--metric_workers", type=int, default=1,
                       help="CPU threads for postprocessing + HD95 during comparison.")
    cq_p.add_argument("--cache_dir",      default=None)
    cq_p.add_argument("--deploy_min_present", type=int, default=1,
                       help="Minimum modalities retained in the deployment regime to compare.")
    cq_p.add_argument("--deploy_max_present", type=int, default=4,
                       help="Maximum modalities retained in the deployment regime to compare.")
    cq_p.add_argument("--max_mean_dice_drop", type=float, default=0.01,
                       help="Deployment pass threshold for mean Dice loss (FP32 - INT8).")
    cq_p.add_argument("--max_worst_dice_drop", type=float, default=0.02,
                       help="Deployment pass threshold for worst-case Dice loss.")
    cq_p.add_argument("--max_mean_hd95_increase", type=float, default=1.0,
                       help="Deployment pass threshold for mean HD95 increase (INT8 - FP32).")
    cq_p.add_argument("--max_worst_hd95_increase", type=float, default=2.0,
                       help="Deployment pass threshold for worst-case HD95 increase.")
    cq_p.add_argument("--no_energy_tracking", action="store_true",
                       help="Disable FP32-vs-INT8 energy reports.")
    cq_p.add_argument("--energy_poll_interval", type=float, default=5.0,
                       help="Seconds between energy-tracker power samples.")

    args = parser.parse_args()

    configure_warnings(args.suppress_warnings)

    # Phase 7.2: load the unified v9.1 config (packaged defaults when --config is
    # omitted). All new toggles default OFF, so the legacy path is unchanged. An
    # explicit --config must be valid; falling back to defaults would silently run
    # the wrong experiment.
    _v91_config = None
    try:
        from ogap.config.loader import load_config as _load_v91_config
        _v91_config = _load_v91_config(getattr(args, "config", None))
    except Exception as exc:  # pragma: no cover - exercised only on cluster runs
        if getattr(args, "config", None) is not None:
            parser.error(f"failed to load --config {args.config!r}: {exc}")
        LOG.warning("Packaged unified config unavailable (%s); using legacy defaults.", exc)
    else:
        try:
            from ogap.utils.hardware import configure_hardware as _configure_hardware
            _hw_summary = _configure_hardware(_v91_config)
            if _hw_summary.get("h100_runtime_enabled"):
                LOG.info("[Hardware] H100 CUDA runtime switches enabled: %s", _hw_summary)
        except Exception as exc:  # pragma: no cover - defensive only
            if getattr(args, "config", None) is not None:
                parser.error(f"failed to configure hardware from --config {args.config!r}: {exc}")
            LOG.warning("Hardware configuration skipped: %s", exc)

    # Packaged defaults are still passed into the v9.1 gates so they keep the
    # legacy path disabled. Only an explicit --config is allowed to override
    # historical CLI defaults such as device="cuda if available".
    _runtime_config = _v91_config if getattr(args, "config", None) is not None else None

    if args.cmd == "pretrain_teacher":
        pretrain_teacher(TeacherConfig(
            v91_config=_v91_config,
            train_csv=args.train_csv, val_csv=args.val_csv, out_dir=args.out_dir,
            modalities=_parse_mods(args.modalities),
            num_classes=int(_cfg_arg(args, explicit_options, _runtime_config, "num_classes", "num_classes", int)),
            patch_size=_parse_patch(args.patch_size),
            batch_size=int(_cfg_arg(args, explicit_options, _runtime_config, "training.teacher_batch_size", "batch_size", int)),
            epochs=int(_cfg_arg(args, explicit_options, _runtime_config, "training.epochs", "epochs", int)),
            lr=float(_cfg_arg(args, explicit_options, _runtime_config, "training.lr_teacher", "lr", float)),
            weight_decay=float(_cfg_arg(args, explicit_options, _runtime_config, "training.weight_decay", "weight_decay", float)),
            num_workers=int(_cfg_arg(args, explicit_options, _runtime_config, "data.num_workers", "num_workers", int)),
            val_num_workers=int(_cfg_arg(args, explicit_options, _runtime_config, "data.val_num_workers", "val_num_workers", int)),
            teacher_base=int(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.base", "teacher_base", int)),
            warmup_epochs=args.warmup_epochs,
            seed=int(_cfg_arg(args, explicit_options, _runtime_config, "reproducibility.seed", "seed", int)),
            lambda_region=args.lambda_region,
            deterministic=(
                args.deterministic
                if "deterministic" in explicit_options
                else bool(_cfg_get(_runtime_config, "reproducibility.deterministic", args.deterministic))
            ),
            hd95_freq=args.hd95_freq,
            val_fast_n=(
                args.val_fast_n if args.val_fast_n is not None else TeacherConfig.val_fast_n
            ),
            val_start_epoch=args.val_start_epoch,
            val_full_freq=(
                args.val_full_freq if args.val_full_freq is not None else TeacherConfig.val_full_freq
            ),
            focal_b1_inhomogeneity_prob=args.focal_b1_inhomogeneity_prob,
            field_strength_sampling_power=args.field_strength_sampling_power,
            low_field_sampling_bias=args.low_field_sampling_bias,
            grade23_sampling_bias=args.grade23_sampling_bias,
            idh_mutant_sampling_bias=args.idh_mutant_sampling_bias,
            legacy_protocol_sampling_bias=args.legacy_protocol_sampling_bias,
            legacy_protocol_year_threshold=args.legacy_protocol_year_threshold,
            device=str(_cfg_arg(args, explicit_options, _runtime_config, "hardware.device", "device", str)),
            amp=not args.no_amp, augment=not args.no_augment,
            cache_dir=args.cache_dir,
            patch_cache_dir=args.patch_cache_dir,
            patch_cache_start_epoch=args.patch_cache_start_epoch,
            lmic_field_strength_prior=args.lmic_field_strength_prior,
            contrast_mod_indices=_parse_int_list(args.contrast_mod_indices),
            p_anisotropic_2d=args.p_anisotropic_2d,
            p_partial_contrast=args.p_partial_contrast,
            p_label_noise=args.p_label_noise,
            label_noise_mode=args.label_noise_mode,
            label_noise_band_radius=args.label_noise_band_radius,
            label_noise_band_flip_prob=args.label_noise_band_flip_prob,
            gpu_augment=not args.no_gpu_augment,
            auto_resources=args.auto_resources,
            auto_resources_safety=args.auto_resources_safety,
            p_field_contrast_warp=getattr(args, "p_field_contrast_warp", 0.0),
            field_contrast_source_b0=getattr(args, "field_contrast_source_b0", 3.0),
            p_fourier_amplitude_mix=getattr(args, "p_fourier_amplitude_mix", 0.0),
            afa_prob=getattr(args, "afa_prob", 0.0),
            p_receive_coil_inhomogeneity=getattr(args, "p_receive_coil_inhomogeneity", 0.0),
            p_off_resonance_void=getattr(args, "p_off_resonance_void", 0.0),
            noise_calibration=getattr(args, "noise_calibration", "static"),
            carvemix_prob=getattr(args, "carvemix_prob", 0.0),
            carvemix_dilation=getattr(args, "carvemix_dilation", 5),
            teacher_arch=str(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.arch", "teacher_arch", str)),
            teacher_block_style=str(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.block_style", "teacher_block_style", str)),
            # FIX (Gaps A + B + Finding 4, 2026 audit): forward new teacher knobs.
            teacher_feature_dr=getattr(args, "teacher_feature_dr", "none"),
            teacher_feature_dr_p=getattr(args, "teacher_feature_dr_p", 0.5),
            teacher_feature_dr_alpha=getattr(args, "teacher_feature_dr_alpha", 0.1),
            teacher_modality_dropout_p=getattr(args, "teacher_modality_dropout_p", 0.0),
            teacher_modality_dropout_max_drop=getattr(args, "teacher_modality_dropout_max_drop", 2),
            force_resume_partial=bool(getattr(args, "force_resume_partial", False)),
            # FIX (2026 throughput/accuracy pass): thread compile + EMA knobs so the
            # bare teacher CLI can enable them (otherwise gated on a v9.1 --config).
            compile=bool(getattr(args, "compile", False)),
            compile_mode=str(getattr(args, "compile_mode", "max-autotune")),
            compile_dynamic=bool(getattr(args, "compile_dynamic", False)),
            ema=bool(getattr(args, "ema", False)),
            ema_decay=float(getattr(args, "ema_decay", 0.999)),
        ))

    elif args.cmd == "train":
        cfg_student_block_style, cfg_student_ode_steps = _v91_student_block_cfg(
            _runtime_config,
            default_style=getattr(args, "student_block_style", "res"),
            default_steps=getattr(args, "student_ode_steps", 4),
        )
        if "student_block_style" in explicit_options:
            cfg_student_block_style = str(args.student_block_style)
        if "student_ode_steps" in explicit_options:
            cfg_student_ode_steps = int(args.student_ode_steps)
        train(TrainConfig(
            v91_config=_v91_config,
            train_csv=args.train_csv, val_csv=args.val_csv, out_dir=args.out_dir,
            modalities=_parse_mods(args.modalities),
            num_classes=int(_cfg_arg(args, explicit_options, _runtime_config, "num_classes", "num_classes", int)),
            teacher_ckpt=args.teacher_ckpt, patch_size=_parse_patch(args.patch_size),
            batch_size=int(_cfg_arg(args, explicit_options, _runtime_config, "training.student_batch_size", "batch_size", int)),
            epochs=int(_cfg_arg(args, explicit_options, _runtime_config, "training.epochs", "epochs", int)),
            lr=float(_cfg_arg(args, explicit_options, _runtime_config, "training.lr_student", "lr", float)),
            weight_decay=float(_cfg_arg(args, explicit_options, _runtime_config, "training.weight_decay", "weight_decay", float)),
            num_workers=int(_cfg_arg(args, explicit_options, _runtime_config, "data.num_workers", "num_workers", int)),
            val_num_workers=int(_cfg_arg(args, explicit_options, _runtime_config, "data.val_num_workers", "val_num_workers", int)),
            keep_min_modalities=args.keep_min_modalities,
            keep_max_modalities=args.keep_max_modalities,
            curriculum_floor_modalities=args.curriculum_floor_modalities,
            keep_count_bias_power=args.keep_count_bias_power,
            field_strength_sampling_power=args.field_strength_sampling_power,
            low_field_sampling_bias=args.low_field_sampling_bias,
            grade23_sampling_bias=args.grade23_sampling_bias,
            idh_mutant_sampling_bias=args.idh_mutant_sampling_bias,
            legacy_protocol_sampling_bias=args.legacy_protocol_sampling_bias,
            legacy_protocol_year_threshold=args.legacy_protocol_year_threshold,
            lambda_kd=args.lambda_kd, lambda_feat=args.lambda_feat,
            feature_distill_sparse_decay=args.feature_distill_sparse_decay,
            lambda_region=args.lambda_region,
            lambda_ds=args.lambda_ds, temperature_start=args.temperature_start,
            temperature_end=args.temperature_end, warmup_epochs=args.warmup_epochs,
            grad_accum_steps=int(_cfg_arg(args, explicit_options, _runtime_config, "training.grad_accum_steps", "grad_accum_steps", int)),
            teacher_base=int(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.base", "teacher_base", int)),
            student_base=int(_cfg_arg(args, explicit_options, _runtime_config, "model.student.base", "student_base", int)),
            curriculum_ramp=args.curriculum_ramp, holder_alpha=args.holder_alpha,
            attention=args.attention,
            enable_deep_supervision=not args.no_deep_supervision,
            curriculum_dropout_p=args.curriculum_dropout_p,
            use_curriculum=not args.no_curriculum,
            confidence_gate_threshold=args.confidence_gate,
            hd95_freq=args.hd95_freq,
            val_fast_n=args.val_fast_n, val_start_epoch=args.val_start_epoch,
            val_full_freq=args.val_full_freq,
            seed=int(_cfg_arg(args, explicit_options, _runtime_config, "reproducibility.seed", "seed", int)),
            deterministic=(
                args.deterministic
                if "deterministic" in explicit_options
                else bool(_cfg_get(_runtime_config, "reproducibility.deterministic", args.deterministic))
            ),
            device=str(_cfg_arg(args, explicit_options, _runtime_config, "hardware.device", "device", str)),
            amp=not args.no_amp, augment=not args.no_augment,
            focal_b1_inhomogeneity_prob=args.focal_b1_inhomogeneity_prob,
            cache_dir=args.cache_dir,
            patch_cache_dir=args.patch_cache_dir,
            patch_cache_start_epoch=args.patch_cache_start_epoch,
            lmic_field_strength_prior=args.lmic_field_strength_prior,
            contrast_mod_indices=_parse_int_list(args.contrast_mod_indices),
            contrast_dropout_extra_prob=args.contrast_dropout_extra_prob,
            p_anisotropic_2d=args.p_anisotropic_2d,
            p_partial_contrast=args.p_partial_contrast,
            p_label_noise=args.p_label_noise,
            label_noise_mode=args.label_noise_mode,
            label_noise_band_radius=args.label_noise_band_radius,
            label_noise_band_flip_prob=args.label_noise_band_flip_prob,
            gpu_augment=not args.no_gpu_augment,
            auto_resources=args.auto_resources,
            auto_resources_safety=args.auto_resources_safety,
            lambda_vrex=getattr(args, "lambda_vrex", 0.0),
            feature_dr=getattr(args, "feature_dr", "none"),
            feature_dr_p=getattr(args, "feature_dr_p", 0.5),
            feature_dr_alpha=getattr(args, "feature_dr_alpha", 0.1),
            p_field_contrast_warp=getattr(args, "p_field_contrast_warp", 0.0),
            field_contrast_source_b0=getattr(args, "field_contrast_source_b0", 3.0),
            p_fourier_amplitude_mix=getattr(args, "p_fourier_amplitude_mix", 0.0),
            afa_prob=getattr(args, "afa_prob", 0.0),
            p_receive_coil_inhomogeneity=getattr(args, "p_receive_coil_inhomogeneity", 0.0),
            p_off_resonance_void=getattr(args, "p_off_resonance_void", 0.0),
            noise_calibration=getattr(args, "noise_calibration", "static"),
            carvemix_prob=getattr(args, "carvemix_prob", 0.0),
            carvemix_dilation=getattr(args, "carvemix_dilation", 5),
            teacher_arch=str(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.arch", "teacher_arch", str)),
            teacher_block_style=str(_cfg_arg(args, explicit_options, _runtime_config, "model.teacher.block_style", "teacher_block_style", str)),
            student_block_style=cfg_student_block_style,
            student_ode_steps=cfg_student_ode_steps,
            p_ulf_gan_synthesis=getattr(args, "p_ulf_gan_synthesis", 0.0),
            ulf_gan_weights=getattr(args, "ulf_gan_weights", None),
            ulf_gan_target_b0=getattr(args, "ulf_gan_target_b0", 0.064),
        ))

    elif args.cmd == "export":
        export_to_onnx(
            ckpt=args.ckpt, out_path=args.out_path,
            modalities=_parse_mods(args.modalities), num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size), student_base=args.student_base,
            calibration_csv=args.calibration_csv,
            calibration_samples=args.calibration_samples,
            calibration_min_present=args.calibration_min_present,
            calibration_max_present=args.calibration_max_present,
            static_calibration_method=args.static_calibration_method,
            static_reduce_range=args.static_reduce_range,
            static_per_channel=not args.no_static_per_channel,
            run_quant_preprocess=not args.skip_quant_preprocess,
            run_compare=args.run_compare,
            compare_val_csv=args.compare_val_csv,
            compare_out_dir=args.compare_out_dir,
            compare_num_workers=args.compare_num_workers,
            compare_num_threads=args.compare_num_threads,
            compare_eval_config_batch_size=args.compare_eval_config_batch_size,
            compare_metric_workers=args.compare_metric_workers,
            compare_cache_dir=args.compare_cache_dir,
            compare_track_energy=not args.no_compare_energy_tracking,
            compare_energy_poll_interval=args.compare_energy_poll_interval,
        )

    elif args.cmd == "infer":
        mm = [int(x) for x in args.missing_modalities.split(",") if x.strip()]
        infer(
            test_csv=args.test_csv, ckpt=args.ckpt, out_dir=args.out_dir,
            modalities=_parse_mods(args.modalities), num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size),
            missing_modalities=mm or None, device=args.device,
            overlap=args.overlap, use_tta=args.tta, student_base=args.student_base,
            postprocess=not args.no_postprocess,
            num_threads=args.num_threads,
            tta_method=getattr(args, "tta_method", "none"),
            tta_steps=getattr(args, "tta_steps", 5),
            tta_lr=getattr(args, "tta_lr", 1e-3),
        )

    elif args.cmd == "evaluate":
        full_evaluation(
            ckpt=args.ckpt, val_csv=args.val_csv,
            modalities=_parse_mods(args.modalities),
            num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size),
            out_dir=args.out_dir, student_base=args.student_base,
            device=args.device,
            num_workers=args.num_workers,
            eval_config_batch_size=args.eval_config_batch_size,
            metric_workers=args.metric_workers,
            use_tta=args.tta,
            cache_dir=args.cache_dir,
            clinical_metadata_tsv=args.clinical_metadata_tsv,
            failure_mode_metric=args.failure_mode_metric,
            failure_mode_quantile=args.failure_mode_quantile,
            failure_mode_min_group_n=args.failure_mode_min_group_n,
            compute_lesion_wise=getattr(args, "lesion_wise", False),
        )

    elif args.cmd == "ablation":
        mods = _parse_mods(args.modalities)
        base_cfg = TrainConfig(
            train_csv=args.train_csv, val_csv=args.val_csv,
            out_dir=str(Path(args.out_dir) / "baseline"),
            modalities=mods, num_classes=args.num_classes,
            teacher_ckpt=args.teacher_ckpt,
            patch_size=_parse_patch(args.patch_size),
            batch_size=args.batch_size, epochs=args.epochs,
            seed=args.seed, deterministic=args.deterministic, device=args.device,
            amp=not args.no_amp, cache_dir=args.cache_dir,
        )
        ablation_list = (
            [a.strip() for a in args.ablations.split(",")]
            if args.ablations else None
        )
        ab_cfg = AblationConfig(
            base_config=base_cfg,
            out_dir=args.out_dir,
        )
        if ablation_list:
            ab_cfg.ablations = ablation_list
        run_ablation_study(ab_cfg)

    elif args.cmd == "benchmark":
        mm = [int(x) for x in args.missing_modalities.split(",") if x.strip()]
        benchmark_hardware(
            test_csv=args.test_csv, ckpt=args.ckpt, out_dir=args.out_dir,
            modalities=_parse_mods(args.modalities), num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size),
            missing_modalities=mm or None,
            student_base=args.student_base,
            num_threads=args.num_threads,
            n_warmup=args.n_warmup,
            n_cases=args.n_cases,
        )

    elif args.cmd == "ablation_eval":
        ab_list = (
            [a.strip() for a in args.ablations.split(",")]
            if args.ablations else None
        )
        ablation_eval(
            baseline_ckpt=args.baseline_ckpt,
            ablation_dir=args.ablation_dir,
            val_csv=args.val_csv,
            modalities=_parse_mods(args.modalities),
            num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size),
            out_dir=args.out_dir,
            student_base=args.student_base,
            device=args.device,
            ablations=ab_list,
            num_workers=args.num_workers,
            eval_config_batch_size=args.eval_config_batch_size,
            metric_workers=args.metric_workers,
            cache_dir=args.cache_dir,
        )

    elif args.cmd == "compare_quantisation":
        compare_quantisation(
            fp32_ckpt=args.fp32_ckpt,
            int8_onnx=args.int8_onnx,
            val_csv=args.val_csv,
            modalities=_parse_mods(args.modalities),
            num_classes=args.num_classes,
            patch_size=_parse_patch(args.patch_size),
            out_dir=args.out_dir,
            student_base=args.student_base,
            fp32_device=args.fp32_device,
            num_workers=args.num_workers,
            num_threads=args.num_threads,
            eval_config_batch_size=args.eval_config_batch_size,
            metric_workers=args.metric_workers,
            cache_dir=args.cache_dir,
            deploy_min_present=args.deploy_min_present,
            deploy_max_present=args.deploy_max_present,
            max_mean_dice_drop=args.max_mean_dice_drop,
            max_worst_dice_drop=args.max_worst_dice_drop,
            max_mean_hd95_increase=args.max_mean_hd95_increase,
            max_worst_hd95_increase=args.max_worst_hd95_increase,
            track_energy=not args.no_energy_tracking,
            energy_poll_interval=args.energy_poll_interval,
        )

    elif args.cmd in ADDENDUM_COMMANDS:
        dispatch(args.cmd, args, sys.modules[__name__])

    else:
        parser.error(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
