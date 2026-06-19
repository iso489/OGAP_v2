#!/usr/bin/env python3
"""Verification harness for the OGAP v2 environment.

Imports every dependency the experimental v9 script touches and prints a
green/red status line for each. Optional dependencies that are missing
yield a yellow warning (the corresponding subcommand will simply be
disabled at runtime); they do not cause this script to fail.

Run on Trillium after ``setup_ogap_env_v2.sh``:

    python verify_ogap_env.py

Exit codes:
    0 - all REQUIRED packages importable.
    1 - at least one required import failed.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import List, Tuple

# ANSI color codes (stripped automatically when stdout isn't a tty).
def _color(code: str) -> str:
    return code if sys.stdout.isatty() else ""

GREEN = _color("\033[32m")
YELLOW = _color("\033[33m")
RED = _color("\033[31m")
DIM = _color("\033[2m")
RESET = _color("\033[0m")
CHECK = "✓"
CROSS = "✗"
WARN = "○"

EXPECTED_TORCH_PUBLIC_VERSION = os.environ.get("OGAP_EXPECTED_TORCH_PUBLIC_VERSION", "2.6.0")
REQUIRE_MAMBA = os.environ.get("OGAP_REQUIRE_MAMBA", "0") == "1"
REQUIRE_ODE = os.environ.get("OGAP_REQUIRE_ODE", "1") == "1"


# (import_name, label, required, feature_when_missing)
DEPS: List[Tuple[str, str, bool, str]] = [
    # ── Hard requirements ────────────────────────────────────────────
    ("numpy", "numpy", True, ""),
    ("scipy", "scipy", True, ""),
    ("scipy.ndimage", "scipy.ndimage", True, ""),
    ("scipy.stats", "scipy.stats", True, ""),
    ("torch", "torch", True, ""),
    ("triton", "triton", True, ""),
    ("nibabel", "nibabel", True, ""),
    ("matplotlib", "matplotlib", True, ""),
    ("pandas", "pandas", True, ""),
    ("einops", "einops", True, ""),
    ("threadpoolctl", "threadpoolctl", True, ""),
    ("onnx", "onnx", True, ""),
    ("onnxruntime", "onnxruntime", True, ""),
    ("psutil", "psutil", True, ""),
    ("sklearn", "scikit-learn", True, ""),
    ("SimpleITK", "SimpleITK (N4 bias correction)", True, ""),
    ("pydicom", "pydicom", True, ""),
    # ── Optional rigor extensions ────────────────────────────────────
    ("monai", "MONAI", False, "metric_check MONAI cross-validation"),
    ("medpy", "medpy", False, "metric_check medpy cross-validation"),
    ("highdicom", "highdicom", False, "dicom_seg subcommand"),
    # huggingface_hub / transformers are needed by mamba-ssm __init__ at import time.
    # List them before mamba_ssm so the error message names the missing dep precisely.
    ("huggingface_hub", "huggingface_hub", REQUIRE_MAMBA, "mamba-ssm import (PyTorchModelHubMixin)"),
    ("transformers", "transformers", REQUIRE_MAMBA, "mamba-ssm import (AutoConfig/modeling)"),
    ("tokenizers", "tokenizers", REQUIRE_MAMBA, "transformers runtime dep (fast tokenizers)"),
    ("mamba_ssm", "mamba-ssm", REQUIRE_MAMBA, "SegMamba teacher (--teacher_arch segmamba)"),
    ("causal_conv1d", "causal-conv1d", REQUIRE_MAMBA, "SegMamba teacher (mamba-ssm dep)"),
    ("torchdiffeq", "torchdiffeq", REQUIRE_ODE, "Neural ODE adjoint teacher (--teacher_arch ode)"),
    # ── Hardware probing ─────────────────────────────────────────────
    ("pynvml", "pynvml (= nvidia_ml_py)", False, "GPU energy tracking"),
]


def _version_problem(import_name: str, version: str) -> str:
    """Return a human-readable version problem, or an empty string."""
    public_version = version.split("+", 1)[0]
    if import_name == "torch" and public_version != EXPECTED_TORCH_PUBLIC_VERSION:
        return f"expected torch {EXPECTED_TORCH_PUBLIC_VERSION}; rerun setup"
    if import_name == "numpy":
        try:
            major = int(public_version.split(".", 1)[0])
        except ValueError:
            return ""
        if major >= 2:
            return "expected scipy-stack numpy 1.x; local numpy is shadowing it"
    return ""


def _try_import(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", "?")
    return True, version


def main() -> int:
    print(f"{DIM}OGAP v2 environment verification - Python "
          f"{sys.version.split()[0]} on {sys.platform}{RESET}")
    print()

    ok_required = 0
    fail_required = 0
    ok_optional = 0
    miss_optional = 0
    missing_features: List[str] = []

    for import_name, label, required, feature in DEPS:
        ok, info = _try_import(import_name)
        problem = _version_problem(import_name, info) if ok else ""
        if ok and not problem:
            color = GREEN
            mark = CHECK
            tail = f"{DIM}{info}{RESET}"
            if required:
                ok_required += 1
            else:
                ok_optional += 1
        elif ok and required:
            color = RED
            mark = CROSS
            tail = f"{info} - {problem}"
            fail_required += 1
        elif ok:
            color = YELLOW
            mark = WARN
            tail = f"{info} - {problem}"
            miss_optional += 1
            missing_features.append(feature)
        elif required:
            color = RED
            mark = CROSS
            tail = info
            fail_required += 1
        else:
            color = YELLOW
            mark = WARN
            tail = f"missing - disables: {feature}"
            miss_optional += 1
            missing_features.append(feature)

        print(f"  {color}{mark}{RESET} {label:<35s} {tail}")

    # ── Bonus: check torch.cuda + bf16 availability ─────────────────────
    print()
    try:
        import torch  # noqa: WPS433
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"  {GREEN}{CHECK}{RESET} torch.cuda                        "
                  f"{DIM}{name} (cc {cap[0]}.{cap[1]}){RESET}")
            if torch.cuda.is_bf16_supported():
                print(f"  {GREEN}{CHECK}{RESET} bfloat16 support                  "
                      f"{DIM}available{RESET}")
            else:
                print(f"  {YELLOW}{WARN}{RESET} bfloat16 support                  "
                      f"{DIM}unavailable; AMP runs will use fp16{RESET}")
        else:
            print(f"  {YELLOW}{WARN}{RESET} torch.cuda                        "
                  f"{DIM}unavailable - env still works on CPU{RESET}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}{CROSS}{RESET} torch.cuda check                  {exc}")

    if REQUIRE_MAMBA:
        try:
            from mamba_ssm import Mamba  # noqa: WPS433, F401
            print(f"  {GREEN}{CHECK}{RESET} mamba_ssm.Mamba                 "
                  f"{DIM}importable{RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED}{CROSS}{RESET} mamba_ssm.Mamba                 "
                  f"failed: {type(exc).__name__}: {exc}")
            fail_required += 1

        # torch.compile / Inductor availability. The SegMamba env ships a Triton
        # whose Inductor API may be absent; OGAP then runs Mamba jobs eager.
        try:
            import triton  # noqa: WPS433
            triton_version = getattr(triton, "__version__", "unknown")
            from triton.compiler.compiler import triton_key  # noqa: WPS433, F401
            print(f"  {GREEN}{CHECK}{RESET} torch.compile/Inductor            "
                  f"{DIM}Triton {triton_version} API OK{RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {YELLOW}{WARN}{RESET} torch.compile/Inductor            "
                  f"{DIM}unavailable ({type(exc).__name__}); Mamba runs eager{RESET}")

    # ── Bonus: try importing the OGAP source itself ─────────────────────
    print()
    import os
    # verify_ogap_env.py lives in setup/; the shim is at the repo root, one level up.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, "OGAP_source_code_experimental_v9.py")
    if os.path.exists(src_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_ogap_v9_check", src_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_ogap_v9_check"] = mod
            spec.loader.exec_module(mod)
            print(f"  {GREEN}{CHECK}{RESET} OGAP_source_code_experimental_v9  "
                  f"{DIM}module imported cleanly{RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {RED}{CROSS}{RESET} OGAP_source_code_experimental_v9  "
                  f"failed: {type(exc).__name__}: {exc}")
            fail_required += 1
    else:
        print(f"  {YELLOW}{WARN}{RESET} OGAP_source_code_experimental_v9  "
              f"{DIM}not found next to verify_ogap_env.py - skipped{RESET}")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print(f"{DIM}─────────────────────────────────────────────────────────{RESET}")
    print(f"  required: {ok_required} ok / {fail_required} fail")
    print(f"  optional: {ok_optional} ok / {miss_optional} missing")
    if missing_features:
        print(f"  {YELLOW}features disabled by missing optional packages:{RESET}")
        for feat in missing_features:
            print(f"    {YELLOW}{WARN}{RESET} {feat}")
    print()
    if fail_required:
        print(f"{RED}FAIL - at least one required package is missing.{RESET}")
        return 1
    print(f"{GREEN}OK - environment ready for OGAP v2 training and evaluation.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
