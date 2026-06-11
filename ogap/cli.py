"""Unified OGAP command-line entrypoint (`python -m ogap ...`).

Backward compatibility first: any command that the legacy core understands is
delegated **unchanged** to :func:`ogap.legacy.main`, which parses ``sys.argv``
exactly as the original monolith did. So ``python -m ogap train ...`` and the
compat shim ``python OGAP_source_code_experimental_v9.py train ...`` are
identical.

On top of that, this CLI adds the new, additive capabilities that live in the
clean subpackages. Currently:

* ``nas-search`` — run the hardware-aware multi-objective architecture search
  (:mod:`ogap.nas`) and write the Pareto front to JSON.

The new commands are namespaced with a hyphen so they can never collide with a
legacy subcommand name.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional

NEW_COMMANDS = {"nas-search"}


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


def _bool_word(value: Any) -> str:
    return "active" if bool(value) else "inactive"


def _cmd_validate_config(argv: List[str]) -> int:
    from .config.loader import load_config

    p = argparse.ArgumentParser(
        prog="ogap --validate-config",
        description="Validate the unified OGAP v9.1 YAML config and print gate status.",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--validate-config", action="store_true")
    args, extra = p.parse_known_args(argv)
    if extra:
        p.error(f"unrecognized arguments: {' '.join(extra)}")

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        p.error(f"invalid config: {exc}")
    compile_enabled = _cfg_get(
        cfg,
        "training.compile.enabled",
        _cfg_get(cfg, "training.compile", False),
    )
    dtype = str(
        _cfg_get(cfg, "training.dtype", _cfg_get(cfg, "training.amp_dtype", ""))
    ).lower()
    fp8_enabled = bool(_cfg_get(cfg, "training.use_fp8", False)) or bool(
        _cfg_get(cfg, "model.transformer_engine_fp8.enabled", False)
    )
    task_type = str(_cfg_get(cfg, "loss.task.type", "legacy"))
    gates = [
        ("BF16", dtype in {"bf16", "bfloat16"}),
        ("FP8", fp8_enabled),
        ("compile", compile_enabled),
        ("tissue_priors", _cfg_get(cfg, "data.use_tissue_priors", False)),
        ("task_loss", task_type != "legacy"),
        ("distillation", _cfg_get(cfg, "distillation.enabled", False)),
        (
            "per_region_temperature",
            _cfg_get(cfg, "distillation.per_region_temperature_enabled", False),
        ),
        ("auxiliary_tissue", _cfg_get(cfg, "loss.auxiliary_tissue.enabled", False)),
        ("domain_adaptation", _cfg_get(cfg, "domain.enabled", False)),
        ("physics_augmentation", _cfg_get(cfg, "augmentation.physics.enabled", False)),
        ("tta", _cfg_get(cfg, "inference.tta.enabled", False)),
        ("ood", _cfg_get(cfg, "inference.ood.enabled", False)),
        ("longitudinal", _cfg_get(cfg, "inference.longitudinal.enabled", False)),
        ("int8_export", _cfg_get(cfg, "export.int8.enabled", False)),
    ]
    label = args.config or "packaged defaults"
    print(f"OGAP config valid: {label}")
    print("Gate summary:")
    for name, enabled in gates:
        print(f"  {name}: {_bool_word(enabled)}")
    print(f"  loss.task.type: {task_type}")
    print(f"  model.teacher.arch: {_cfg_get(cfg, 'model.teacher.arch', 'unet3d')}")
    print(f"  model.student.arch: {_cfg_get(cfg, 'model.student.arch', 'standard')}")
    return 0


def _cmd_nas_search(argv: List[str]) -> int:
    from .nas import Budget, SearchConfig, random_search, evolutionary_search

    p = argparse.ArgumentParser(prog="ogap nas-search",
                                description="Hardware-aware multi-objective NAS for the OGAP student.")
    p.add_argument("--in-channels", type=int, default=8,
                   help="student input channels (2*num_modalities for masked+availability).")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--patch", type=str, default="32,32,32", help="D,H,W proxy patch for cost/proxy estimation.")
    p.add_argument("--strategy", choices=["random", "evolution"], default="random")
    p.add_argument("--n", type=int, default=16, help="candidates (random) / iterations (evolution).")
    p.add_argument("--proxy", default="jacob_cov",
                   choices=["jacob_cov", "synflow", "snip", "grad_norm", "param_count", "diswot"])
    p.add_argument("--teacher-arch", default=None,
                   choices=["unet3d", "segmamba", "swin_unetr", "ode"],
                   help="Build a random-initialized teacher for teacher-aware DisWOT scoring.")
    p.add_argument("--teacher-base", type=int, default=32)
    p.add_argument("--max-int8-mb", type=float, default=1e9)
    p.add_argument("--max-latency-ms", type=float, default=1e9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write Pareto front + full results JSON here.")
    args = p.parse_args(argv)

    d, h, w = (int(v) for v in args.patch.split(","))
    teacher = None
    if args.proxy == "diswot":
        if args.teacher_arch is None:
            p.error("--proxy diswot requires --teacher-arch")
        if args.teacher_arch == "ode":
            from .models import build_ode_teacher
            teacher = build_ode_teacher(args.in_channels, args.num_classes, args.teacher_base)
        else:
            from .models import build_teacher
            teacher = build_teacher(
                args.in_channels, args.num_classes, args.teacher_base,
                arch=args.teacher_arch,
            )

    cfg = SearchConfig(
        in_channels=args.in_channels, num_classes=args.num_classes,
        input_shape=(1, args.in_channels, d, h, w),
        proxy_key=args.proxy, seed=args.seed,
        teacher_model=teacher,
        budget=Budget(max_int8_mb=args.max_int8_mb, max_latency_ms=args.max_latency_ms),
    )
    if args.strategy == "random":
        res = random_search(cfg, n=args.n)
        pool = res["evaluated"]
    else:
        res = evolutionary_search(cfg, iterations=args.n)
        pool = res["history"]

    def row(c):
        return {"arch": c.arch.to_dict(), "accuracy_proxy": c.accuracy, "cost": c.cost.to_dict()}

    report = {
        "strategy": args.strategy,
        "proxy": args.proxy,
        "budget": {"max_int8_mb": args.max_int8_mb, "max_latency_ms": args.max_latency_ms},
        "n_evaluated": len(pool),
        "pareto_front": [row(c) for c in res["pareto"]],
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[nas-search] wrote {len(report['pareto_front'])} Pareto-optimal architectures to {args.out}")
    else:
        print(text)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--validate-config" in argv:
        return _cmd_validate_config(argv)
    if argv and argv[0] in NEW_COMMANDS:
        if argv[0] == "nas-search":
            return _cmd_nas_search(argv[1:])
    # Everything else is a legacy command — delegate verbatim.
    from . import legacy
    legacy.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
