#!/usr/bin/env python3
"""OGAP v9.1 production pipeline launcher.

This file is an orchestrator only. It shells out to supported ``python -m ogap``
subcommands and keeps global legacy options, especially ``--config``, before the
subcommand where argparse expects them.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional


DEFAULT_RESULTS_DIR = "results/production_v91"
DEFAULT_TEACHER_DIR = f"{DEFAULT_RESULTS_DIR}/teacher"
DEFAULT_STUDENT_DIR = f"{DEFAULT_RESULTS_DIR}/student"
DEFAULT_EXPORT_DIR = f"{DEFAULT_RESULTS_DIR}/export"
DEFAULT_EVAL_DIR = f"{DEFAULT_RESULTS_DIR}/eval"
DEFAULT_PARETO_JSON = f"{DEFAULT_RESULTS_DIR}/nas/pareto_front.json"
DEFAULT_ONNX_OUT = f"{DEFAULT_EXPORT_DIR}/model_int8.onnx"
DEFAULT_OOD_DIR = f"{DEFAULT_RESULTS_DIR}/ood"
DEFAULT_LONGITUDINAL_DIR = f"{DEFAULT_RESULTS_DIR}/longitudinal"


@dataclass(frozen=True)
class Stage:
    stage_id: int
    name: str
    command: Callable[[argparse.Namespace], List[str]]
    artifact: Callable[[argparse.Namespace], Optional[Path]]
    heavy: bool = False


def _py_ogap(args: argparse.Namespace, subcommand: str) -> List[str]:
    cmd = [sys.executable, "-m", "ogap"]
    if subcommand != "nas-search":
        cmd.extend(["--config", str(Path(args.config))])
    cmd.append(subcommand)
    return cmd


def _require(args: argparse.Namespace, attr: str, stage: str) -> str:
    value = getattr(args, attr)
    if not value:
        raise ValueError(f"Stage {stage} requires --{attr.replace('_', '-')}")
    return str(value)


def _common_train_args(args: argparse.Namespace, stage: str = "training") -> List[str]:
    return [
        "--train_csv", _require(args, "train_csv", stage),
        "--val_csv", _require(args, "val_csv", stage),
        "--modalities", args.modalities,
        "--num_classes", str(args.num_classes),
    ]


def _teacher_ckpt(args: argparse.Namespace) -> Path:
    return Path(args.teacher_ckpt) if args.teacher_ckpt else Path(args.teacher_out_dir) / "best_teacher.pth"


def _student_ckpt(args: argparse.Namespace) -> Path:
    return Path(args.student_ckpt) if args.student_ckpt else Path(args.student_out_dir) / "best_student.pth"


def _teacher_cmd(args: argparse.Namespace) -> List[str]:
    cmd = (
        _py_ogap(args, "pretrain_teacher")
        + _common_train_args(args, "pretrain_teacher")
        + [
            "--out_dir", str(args.teacher_out_dir),
            "--teacher_arch", args.teacher_arch,
            "--teacher_block_style", args.teacher_block_style,
            "--patch_size", args.teacher_patch_size,
            "--batch_size", str(args.teacher_batch_size),
            "--num_workers", str(args.num_workers),
            "--val_num_workers", str(args.val_num_workers),
            "--epochs", str(args.teacher_epochs),
        ]
    )
    if args.auto_resources:
        cmd.append("--auto_resources")
    return cmd


def _student_cmd(args: argparse.Namespace) -> List[str]:
    cmd = (
        _py_ogap(args, "train")
        + _common_train_args(args, "train")
        + [
            "--out_dir", str(args.student_out_dir),
            "--teacher_ckpt", str(_teacher_ckpt(args)),
            "--teacher_arch", args.teacher_arch,
            "--teacher_block_style", args.teacher_block_style,
            "--student_block_style", args.student_block_style,
            "--student_ode_steps", str(args.student_ode_steps),
            "--patch_size", args.student_patch_size,
            "--batch_size", str(args.student_batch_size),
            "--num_workers", str(args.num_workers),
            "--val_num_workers", str(args.val_num_workers),
            "--epochs", str(args.student_epochs),
        ]
    )
    if args.auto_resources:
        cmd.append("--auto_resources")
    return cmd


def _nas_cmd(args: argparse.Namespace) -> List[str]:
    return _py_ogap(args, "nas-search") + [
        "--strategy", "evolution",
        "--n", str(args.nas_iterations),
        "--in-channels", str(args.nas_in_channels),
        "--num-classes", str(args.num_classes),
        "--patch", args.nas_patch,
        "--max-int8-mb", str(args.max_int8_mb),
        "--max-latency-ms", str(args.max_latency_ms),
        "--out", str(args.pareto_json),
    ]


def _export_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _py_ogap(args, "export") + [
        "--ckpt", str(_student_ckpt(args)),
        "--out_path", str(args.onnx_out),
        "--modalities", args.modalities,
        "--num_classes", str(args.num_classes),
        "--patch_size", args.student_patch_size,
        "--student_base", str(args.student_base),
    ]
    if args.calibration_csv:
        cmd.extend([
            "--calibration_csv", str(args.calibration_csv),
            "--calibration_samples", str(args.calibration_samples),
        ])
    if args.run_compare:
        cmd.extend([
            "--run_compare",
            "--compare_val_csv", _require(args, "val_csv", "export comparison"),
            "--compare_out_dir", str(Path(args.export_dir) / "compare"),
            "--compare_num_workers", str(args.val_num_workers),
            "--compare_eval_config_batch_size", str(args.eval_config_batch_size),
            "--compare_metric_workers", str(args.metric_workers),
        ])
    return cmd


def _fit_ood_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _py_ogap(args, "mahalanobis_ood") + [
        "--ckpt", str(_student_ckpt(args)),
        "--id_csv", _require(args, "val_csv", "fit_ood"),
        "--modalities", args.modalities,
        "--num_classes", str(args.num_classes),
        "--patch_size", args.student_patch_size,
        "--student_base", str(args.student_base),
        "--device", args.ood_device,
        "--out_dir", str(args.ood_out_dir),
    ]
    if args.ood_csv:
        cmd.extend(["--ood_csv", str(args.ood_csv)])
    return cmd


def _fit_longitudinal_cmd(args: argparse.Namespace) -> List[str]:
    raise ValueError(
        "Stage fit_longitudinal is disabled until a real longitudinal training "
        "dataset/command is wired. Use --enable-longitudinal-fit only after adding "
        "a fitted LatentODERANOTracker training path."
    )


def _evaluate_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _py_ogap(args, "evaluate") + [
        "--ckpt", str(_student_ckpt(args)),
        "--val_csv", _require(args, "val_csv", "evaluate"),
        "--out_dir", str(args.eval_out_dir),
        "--modalities", args.modalities,
        "--num_classes", str(args.num_classes),
        "--patch_size", args.student_patch_size,
        "--student_base", str(args.student_base),
        "--num_workers", str(args.val_num_workers),
        "--eval_config_batch_size", str(args.eval_config_batch_size),
        "--metric_workers", str(args.metric_workers),
        "--lesion_wise",
    ]
    if not args.no_tta:
        cmd.append("--tta")
    return cmd


def _ablation_cmd(args: argparse.Namespace) -> List[str]:
    return (
        _py_ogap(args, "ablation")
        + _common_train_args(args, "run_ablation")
        + [
            "--out_dir", str(args.ablation_out_dir),
            "--teacher_ckpt", str(_teacher_ckpt(args)),
            "--patch_size", args.student_patch_size,
            "--batch_size", str(args.student_batch_size),
            "--epochs", str(args.ablation_epochs),
        ]
    )


def _stage_artifact(path_attr: str) -> Callable[[argparse.Namespace], Optional[Path]]:
    return lambda args: Path(getattr(args, path_attr))


STAGES: List[Stage] = [
    Stage(1, "pretrain_teacher", _teacher_cmd, lambda args: Path(args.teacher_out_dir) / "best_teacher.pth"),
    Stage(2, "train", _student_cmd, lambda args: Path(args.student_out_dir) / "best_student.pth"),
    # NAS is an OFFLINE design tool: it writes an advisory Pareto front to JSON.
    # No downstream stage consumes pareto_json, and the searched OFA family is not
    # the deployed UNet3DStudent — read it to inform a student config, don't expect
    # it to be auto-applied. Skipped by --no-heavy. [audit S2-B]
    Stage(3, "nas_search", _nas_cmd, _stage_artifact("pareto_json"), heavy=True),
    Stage(4, "export", _export_cmd, _stage_artifact("onnx_out")),
    Stage(5, "fit_ood", _fit_ood_cmd, lambda args: Path(args.ood_out_dir) / "mahalanobis_params.npz", heavy=True),
    Stage(6, "fit_longitudinal", _fit_longitudinal_cmd, lambda args: Path(args.longitudinal_out_dir) / "tracker.pt", heavy=True),
    Stage(7, "evaluate", _evaluate_cmd, lambda args: Path(args.eval_out_dir) / "results_full.csv"),
    Stage(8, "run_ablation", _ablation_cmd, lambda args: Path(args.ablation_out_dir) / "ablation_statistical_comparison.json", heavy=True),
]


def _pipeline_env() -> dict:
    env = os.environ.copy()
    # Stage-agnostic torch/Inductor knobs only.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    # NOTE: thread affinity (OMP/MKL/KMP) and NCCL settings are intentionally NOT
    # set here. Thread counts are stage-specific — small for GPU training so CPU
    # cores serve the dataloaders, large for CPU INT8 inference — and belong in the
    # sbatch scripts (see docs/HARDWARE_TUNING.md §2). There is no DDP path in the
    # training loop, so the previous NCCL_* settings were dead. [audit S1-D / S3-D]
    return env


def _normalise_output_paths(args: argparse.Namespace) -> None:
    if not args.output_dir:
        return
    base = Path(args.output_dir)
    args.results_dir = str(base)
    args.teacher_out_dir = str(base / "teacher")
    args.student_out_dir = str(base / "student")
    args.export_dir = str(base / "export")
    args.eval_out_dir = str(base / "eval")
    args.ablation_out_dir = str(base / "ablation")
    args.pareto_json = str(base / "nas" / "pareto_front.json")
    args.onnx_out = str(base / "export" / "model_int8.onnx")
    args.ood_out_dir = str(base / "ood")
    args.longitudinal_out_dir = str(base / "longitudinal")


def _ensure_common_dirs(args: argparse.Namespace) -> None:
    paths = [
        args.results_dir,
        args.teacher_out_dir,
        args.student_out_dir,
        args.export_dir,
        args.eval_out_dir,
        args.ablation_out_dir,
        Path(args.pareto_json).parent,
        Path(args.onnx_out).parent,
        args.ood_out_dir,
        args.longitudinal_out_dir,
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def run_stage(stage: Stage, args: argparse.Namespace, logger: logging.Logger) -> None:
    if stage.stage_id < args.from_stage:
        logger.info("Stage %d (%s): skipped by --from-stage", stage.stage_id, stage.name)
        return
    if args.no_heavy and stage.heavy:
        logger.info("Stage %d (%s): skipped by --no-heavy", stage.stage_id, stage.name)
        return
    if stage.name == "fit_ood" and not args.enable_ood_fit:
        logger.info("Stage %d (%s): skipped; pass --enable-ood-fit to run Mahalanobis OOD fitting", stage.stage_id, stage.name)
        return
    if stage.name == "fit_longitudinal" and not args.enable_longitudinal_fit:
        logger.info("Stage %d (%s): skipped; longitudinal fitting has no production training path yet", stage.stage_id, stage.name)
        return
    artifact = stage.artifact(args)
    if artifact is not None and artifact.exists() and not args.force:
        logger.info("Stage %d (%s): artifact exists, skipping: %s", stage.stage_id, stage.name, artifact)
        return
    cmd = stage.command(args)
    logger.info("Stage %d (%s): %s", stage.stage_id, stage.name, " ".join(cmd))
    if args.dry_run:
        return
    _ensure_common_dirs(args)
    result = subprocess.run(cmd, env=_pipeline_env(), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/baseline_v91.yaml",
                        help="Validated default profile. Pass "
                             "configs/production_v91.yaml for the full experimental stack.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--calibration-csv", default=None)
    parser.add_argument("--modalities", default="t1,t1ce,t2,flair")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, len(STAGES) + 1))
    parser.add_argument("--force", action="store_true", help="Run stages even when artifacts exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print stage commands without running them.")
    parser.add_argument("--no-heavy", action="store_true", help="Skip NAS, OOD, longitudinal, and ablation stages.")
    parser.add_argument("--enable-ood-fit", action="store_true",
                        help="Run the real Mahalanobis OOD fitting stage instead of skipping it.")
    parser.add_argument("--enable-longitudinal-fit", action="store_true",
                        help="Attempt longitudinal fitting. Currently fails fast until a real fitted tracker path is wired.")
    parser.add_argument("--no-tta", action="store_true", help="Do not pass --tta to the evaluate stage.")
    parser.add_argument("--run-compare", action="store_true", help="Run FP32-vs-INT8 comparison during export.")
    parser.add_argument("--auto-resources", action="store_true", help="Pass --auto_resources to training stages.")

    parser.add_argument("--output-dir", default=None, help="Base output directory for all stage artifacts.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--teacher-out-dir", default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--student-out-dir", default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--eval-out-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--ablation-out-dir", default=f"{DEFAULT_RESULTS_DIR}/ablation")
    parser.add_argument("--teacher-ckpt", default=None)
    parser.add_argument("--student-ckpt", default=None)
    parser.add_argument("--pareto-json", default=DEFAULT_PARETO_JSON)
    parser.add_argument("--onnx-out", default=DEFAULT_ONNX_OUT)
    parser.add_argument("--ood-out-dir", default=DEFAULT_OOD_DIR)
    parser.add_argument("--ood-csv", default=None, help="Optional OOD CSV for AUROC in the Mahalanobis OOD stage.")
    parser.add_argument("--ood-device", default="cuda")
    parser.add_argument("--longitudinal-out-dir", default=DEFAULT_LONGITUDINAL_DIR)

    # Default to the VALIDATED path (conv UNet teacher + residual student). The
    # ODE teacher/student are opt-in until their KD-lift is benchmarked. [audit S1-B]
    parser.add_argument("--teacher-arch", default="unet3d", choices=("unet3d", "segmamba", "swin_unetr", "ode"))
    parser.add_argument("--teacher-block-style", default="dense", choices=("mednext", "dense"))
    parser.add_argument("--student-block-style", default="res", choices=("res", "ode"))
    parser.add_argument("--student-ode-steps", type=int, default=4)
    parser.add_argument("--teacher-patch-size", default="192,192,144")
    parser.add_argument("--student-patch-size", default="128,160,112")
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=1000)
    parser.add_argument("--student-epochs", type=int, default=1000)
    parser.add_argument("--ablation-epochs", type=int, default=300)
    parser.add_argument("--student-base", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--val-num-workers", type=int, default=4)

    parser.add_argument("--nas-iterations", type=int, default=50)
    # 11 = student input channels: 4 modalities + 4 availability-mask channels
    # (2*num_modalities) + 3 tissue-prior channels (CSF/WM/GM). [audit S3-E]
    parser.add_argument("--nas-in-channels", type=int, default=11)
    parser.add_argument("--nas-patch", default="32,32,32")
    parser.add_argument("--max-int8-mb", type=float, default=5.0)
    parser.add_argument("--max-latency-ms", type=float, default=800.0)
    parser.add_argument("--calibration-samples", type=int, default=200)
    # 11 = number of non-empty modality-dropout combinations evaluated for
    # modality-robustness (NOT a GPU minibatch size). [audit S3-E]
    parser.add_argument("--eval-config-batch-size", type=int, default=11)
    parser.add_argument("--metric-workers", type=int, default=1)
    args = parser.parse_args()
    _normalise_output_paths(args)
    return args


def main() -> None:
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)
    log_path = Path("logs") / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logger = logging.getLogger("ogap.pipeline")
    logger.info("OGAP v9.1 pipeline: config=%s from_stage=%d", args.config, args.from_stage)
    for stage in STAGES:
        run_stage(stage, args, logger)
    logger.info("Pipeline complete. Log: %s", log_path)


if __name__ == "__main__":
    main()
