#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  OGAP v2 SegMamba environment bootstrapper for Rorqual.
#
#  Creates a separate env:
#    ~/ogap/envs/ogap_env_v2_mamba
#
#  This intentionally does NOT modify the stable ogap_env_v2 torch-2.6 env.
#  Compute Canada's current mamba-ssm wheel is pinned to torch ~=2.5.0, so this
#  env uses torch 2.5.1 and is meant only for --teacher_arch segmamba runs.
#
#  Usage:
#    bash setup_ogap_env_v2_mamba.sh
#    bash setup_ogap_env_v2_mamba.sh --verify-only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2_mamba}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[mamba-setup] unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION:-2.5.1+computecanada}"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_VERSION%%+*}"
TRITON_PACKAGE="${TRITON_PACKAGE:-triton}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.4.0}"
MAMBA_SSM_VERSION="${MAMBA_SSM_VERSION:-2.2.4}"
VERIFY_SCRIPT="${SCRIPTS_DIR}/verify_ogap_env.py"

python_module_version() {
  local module="$1"
  python - "${module}" <<'PY'
import importlib
import sys
try:
    mod = importlib.import_module(sys.argv[1])
except Exception:
    print("")
else:
    print(getattr(mod, "__version__", ""))
PY
}

find_compatible_cc_wheel() {
  local dist="$1"
  python - "${dist}" "${EXPECTED_TORCH_PUBLIC_VERSION}" <<'PY'
from __future__ import annotations

import glob
import os
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

dist = sys.argv[1].replace("-", "_")
dist_canon = canonicalize_name(sys.argv[1])
torch_version = Version(sys.argv[2])
supported_tags = set(sys_tags())
roots = [
    "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4",
    "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3",
    "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic",
    "/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic",
]

candidates = []
for root in roots:
    for path in glob.glob(os.path.join(root, f"{dist}-*.whl")):
        base = os.path.basename(path)
        try:
            wheel_name, version, _build, wheel_tags = parse_wheel_filename(base)
        except Exception:
            continue
        if canonicalize_name(wheel_name) != dist_canon:
            continue
        if not wheel_tags.intersection(supported_tags):
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                meta_name = next(
                    n for n in zf.namelist()
                    if n.endswith(".dist-info/METADATA")
                )
                metadata = zf.read(meta_name).decode("utf-8", "replace")
        except Exception:
            continue
        torch_req = None
        actual_name = dist_canon
        for line in metadata.splitlines():
            if line.startswith("Name:"):
                actual_name = canonicalize_name(line.split(":", 1)[1].strip())
            if line.startswith("Requires-Dist:"):
                try:
                    req = Requirement(line.split(":", 1)[1].strip())
                except Exception:
                    continue
                if canonicalize_name(req.name) == "torch":
                    torch_req = req
        if actual_name != dist_canon:
            continue
        if torch_req is not None and torch_version not in torch_req.specifier:
            continue
        candidates.append((version, path))

if candidates:
    print(sorted(candidates)[-1][1])
PY
}

ensure_torch_pin() {
  local torch_version
  torch_version="$(python_module_version torch)"
  if [[ "${torch_version%%+*}" != "${EXPECTED_TORCH_PUBLIC_VERSION}" ]]; then
    echo "[mamba-setup] enforcing torch ${EXPECTED_TORCH_VERSION} ..."
    pip install --no-index --force-reinstall \
      torch=="${EXPECTED_TORCH_VERSION}" \
      nvidia_ml_py
  fi
}

ensure_triton() {
  if ! python -c "import triton" 2>/dev/null; then
    echo "[mamba-setup] installing Triton from CC wheelhouse (${TRITON_PACKAGE}) ..."
    pip install --no-index --no-deps "${TRITON_PACKAGE}"
  fi
}

check_torch_compile_api() {
  python - <<'PY'
try:
    import triton
    triton_version = getattr(triton, "__version__", "unknown")
    from triton.compiler.compiler import triton_key  # noqa: F401
except Exception as exc:
    print(
        "[mamba-setup] note: torch.compile/Inductor is unavailable in this "
        f"SegMamba env with Triton {locals().get('triton_version', 'unknown')} "
        f"({type(exc).__name__}: {exc}). OGAP disables torch.compile for "
        "Mamba jobs and runs eager instead."
    )
else:
    print("[mamba-setup] torch.compile/Inductor Triton API check OK.")
PY
}

echo "[mamba-setup] loading Compute Canada modules ..."
module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a

if [[ ! -d "${ENV_PATH}" ]]; then
  echo "[mamba-setup] creating new venv at ${ENV_PATH}"
  mkdir -p "$(dirname "${ENV_PATH}")"
  virtualenv --no-download "${ENV_PATH}"
else
  echo "[mamba-setup] reusing existing venv at ${ENV_PATH}"
fi

# shellcheck disable=SC1091
source "${ENV_PATH}/bin/activate"

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  echo "[mamba-setup] --verify-only requested; skipping all installs."
else
  pip install --no-index --upgrade pip setuptools wheel >/dev/null

  echo "[mamba-setup][A] installing shared OGAP dependencies ..."
  pip install --no-index \
    numpy==1.26.4+computecanada \
    scipy \
    matplotlib \
    pandas \
    nibabel \
    einops \
    ninja \
    filelock \
    fsspec \
    jinja2 \
    markupsafe \
    networkx \
    packaging \
    pyyaml \
    regex \
    tqdm \
    threadpoolctl \
    typing_extensions \
    sympy \
    mpmath \
    protobuf \
    flatbuffers \
    ml_dtypes \
    humanfriendly \
    coloredlogs \
    click \
    shellingham \
    pygments \
    markdown_it_py \
    mdurl

  echo "[mamba-setup][B] installing torch ${EXPECTED_TORCH_VERSION} stack ..."
  pip uninstall -y mamba-ssm causal-conv1d torch triton nvidia_ml_py \
    huggingface_hub transformers tokenizers >/dev/null 2>&1 || true
  pip install --no-index --force-reinstall \
    torch=="${EXPECTED_TORCH_VERSION}" \
    nvidia_ml_py
  ensure_torch_pin
  ensure_triton

  echo "[mamba-setup][C] installing remaining OGAP dependencies ..."
  pip install --no-index \
    onnx \
    onnxruntime \
    onnxscript \
    onnx-ir \
    safetensors \
    hf_xet \
    psutil \
    scikit-learn \
    pydicom \
    SimpleITK

  pip install --no-index --no-deps monai 2>/dev/null \
    || pip install --no-deps monai --quiet \
    || echo "[mamba-setup][C] monai unavailable."
  pip install --no-index --no-deps medpy 2>/dev/null \
    || pip install --no-deps medpy --quiet \
    || echo "[mamba-setup][C] medpy unavailable."
  pip install --no-deps highdicom --quiet \
    || echo "[mamba-setup][C] highdicom unavailable."

  echo "[mamba-setup][D] installing SegMamba stack ..."
  export CUDA_HOME="${EBROOTCUDA:-${CUDA_HOME:-}}"
  if [[ -n "${CUDA_HOME}" ]]; then
    echo "[mamba-setup][D] CUDA_HOME=${CUDA_HOME}"
  fi

  causal_wheel="$(find_compatible_cc_wheel causal-conv1d || true)"
  if [[ -n "${causal_wheel}" ]]; then
    echo "[mamba-setup][D] using CC causal-conv1d wheel: ${causal_wheel}"
    pip install --no-index --no-deps "${causal_wheel}"
  else
    echo "[mamba-setup][D] no CC causal-conv1d wheel matches torch ${EXPECTED_TORCH_PUBLIC_VERSION}."
    echo "[mamba-setup][D] trying PyPI source build causal-conv1d==${CAUSAL_CONV1D_VERSION} ..."
    pip install --no-deps --no-build-isolation --no-binary=causal-conv1d \
      causal-conv1d=="${CAUSAL_CONV1D_VERSION}"
  fi
  ensure_torch_pin
  ensure_triton

  mamba_wheel="$(find_compatible_cc_wheel mamba-ssm || true)"
  if [[ -n "${mamba_wheel}" ]]; then
    echo "[mamba-setup][D] using CC mamba-ssm wheel: ${mamba_wheel}"
    pip install --no-index --no-deps "${mamba_wheel}"
  else
    echo "[mamba-setup][D] no CC mamba-ssm wheel matches torch ${EXPECTED_TORCH_PUBLIC_VERSION}."
    echo "[mamba-setup][D] trying PyPI source build mamba-ssm==${MAMBA_SSM_VERSION} ..."
    pip install --no-deps --no-build-isolation --no-binary=mamba-ssm \
      mamba-ssm=="${MAMBA_SSM_VERSION}"
  fi
  ensure_torch_pin
  ensure_triton

  # mamba-ssm was installed --no-deps to prevent torch churn, but its __init__.py
  # imports from huggingface_hub (PyTorchModelHubMixin) and transformers at module
  # load time — so `from mamba_ssm import Mamba` crashes without them.
  # Install these HuggingFace deps from the CC wheelhouse.  We install tokenizers
  # and requests explicitly so transformers' --no-deps doesn't silently break it,
  # and we guard with ensure_torch_pin/triton afterwards to catch any accidental churn.
  echo "[mamba-setup][D] installing mamba-ssm HuggingFace runtime deps ..."
  pip install --no-index huggingface_hub \
    || pip install huggingface_hub --quiet
  # transformers pulls tokenizers (Rust wheel) — use --no-deps then add tokenizers
  # explicitly so we never let transformers silently upgrade numpy.
  pip install --no-index --no-deps transformers \
    || pip install --no-deps transformers --quiet
  pip install --no-index tokenizers \
    || pip install tokenizers --quiet
  pip install --no-index requests \
    || true   # requests is usually already present via the system python
  ensure_torch_pin
  ensure_triton

  # mamba-ssm 2.2.4's __init__.py eagerly imports MambaLMHeadModel, which in turn
  # imports `GreedySearchDecoderOnlyOutput` from `transformers.generation`.  That
  # symbol was REMOVED in transformers 5.x (CC wheelhouse only ships 5.6.2+),
  # so `from mamba_ssm import Mamba` raises ImportError at module load.
  # OGAP only needs the Mamba/Mamba2 SSM blocks — never MambaLMHeadModel — so
  # we comment that single import out.  Idempotent: already-commented lines are
  # not re-touched because the regex requires a leading non-`#` character.
  echo "[mamba-setup][D] patching mamba_ssm/__init__.py for transformers 5.x compat ..."
  MAMBA_INIT_PY="${ENV_PATH}/lib/python3.11/site-packages/mamba_ssm/__init__.py"
  if [[ -f "${MAMBA_INIT_PY}" ]]; then
    sed -i 's|^from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel$|# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # disabled: needs transformers<5 (GreedySearchDecoderOnlyOutput removed in 5.x)|' "${MAMBA_INIT_PY}"
    if grep -q "^# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel" "${MAMBA_INIT_PY}"; then
      echo "[mamba-setup][D] mamba_ssm/__init__.py patched."
    else
      echo "[mamba-setup][D] WARNING: expected import line not found in ${MAMBA_INIT_PY}; mamba-ssm may have changed structure."
    fi
  else
    echo "[mamba-setup][D] WARNING: ${MAMBA_INIT_PY} not found; skipping patch."
  fi
fi

check_torch_compile_api

echo
echo "[mamba-setup] running SegMamba verification ..."
OGAP_EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_PUBLIC_VERSION}" \
OGAP_REQUIRE_MAMBA=1 \
python "${VERIFY_SCRIPT}"
echo
echo "[mamba-setup] done. Activate with:"
echo "    module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a"
echo "    source ${ENV_PATH}/bin/activate"
