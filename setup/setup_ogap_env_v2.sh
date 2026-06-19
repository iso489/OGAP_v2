#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  OGAP v2 Alliance (Trillium) environment bootstrapper.
#
#  Creates ~/ogap/envs/ogap_env_v2 with everything
#  needed by Scripts/OGAP_source_code_experimental_v9.py, including the
#  MedIA / IEEE TMI rigor extensions added in the 2026 architecture pass:
#    • Field-strength-aware contrast resimulation
#    • AFA Fourier amplitude mixing
#    • CarveMix anatomy-preserving lesion mix-up
#    • MixStyle3D / DSU feature-space domain randomization
#    • V-REx variance-of-risks penalty
#    • Tent / MEMO test-time adaptation
#    • Conformal Risk Control, Mahalanobis OOD, TOST, DeLong, DICOM-SEG
#    • Optional SegMamba teacher (mamba-ssm)
#    • Optional Lucas-2025 ULF GAN augmentation
#
#  Usage on Trillium:
#    cd /project/def-rdiaz/ilyaso/OGAP/Scripts
#    bash setup_ogap_env_v2.sh                 # install/repair default UNet3D env
#    bash setup_ogap_env_v2.sh --with-mamba    # try CUDA-built SegMamba deps
#    bash setup_ogap_env_v2.sh --verify-only   # only run the verification
#
#  Layout:
#    ENV_PATH ← override target dir
#  Re-run is idempotent: existing env is reused; only missing packages
#  are installed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── User-tunable paths ───────────────────────────────────────────────────────
ENV_PATH="${ENV_PATH:-${HOME}/ogap/envs/ogap_env_v2}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WITH_MAMBA=0
SKIP_OPTIONAL=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-mamba)    WITH_MAMBA=1 ;;
    --skip-mamba)    WITH_MAMBA=0 ;;
    --skip-optional) SKIP_OPTIONAL=1 ;;
    --verify-only)   VERIFY_ONLY=1 ;;
    --help|-h)
      grep "^#" "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[setup] unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

EXPECTED_TORCH_VERSION="2.6.0+computecanada"
EXPECTED_TORCH_PUBLIC_VERSION="${EXPECTED_TORCH_VERSION%%+*}"
EXPECTED_NUMPY_MAJOR_MAX=1

find_script_file() {
  # Some terminal/clipboard paths were pasted as markdown links, e.g.
  # verify_ogap_[env.py](http://env.py). Prefer the canonical file, but fall
  # back to the corrupted copy so setup can still run and print a useful hint.
  local canonical="$1"
  local fallback_glob="$2"
  if [[ -f "${SCRIPTS_DIR}/${canonical}" ]]; then
    printf '%s\n' "${SCRIPTS_DIR}/${canonical}"
    return 0
  fi
  local match
  match="$(find "${SCRIPTS_DIR}" -maxdepth 1 -type f -name "${fallback_glob}" -print -quit 2>/dev/null || true)"
  if [[ -n "${match}" ]]; then
    echo "[setup] WARNING: expected ${canonical}, but found markdown-corrupted filename:" >&2
    echo "[setup]          ${match}" >&2
    echo "[setup]          Rename it on Trillium after setup finishes." >&2
    printf '%s\n' "${match}"
    return 0
  fi
  printf '%s\n' "${SCRIPTS_DIR}/${canonical}"
}

VERIFY_SCRIPT="$(find_script_file "verify_ogap_env.py" "verify_ogap_*env.py*")"

python_version_ok() {
  python - <<'PY'
import sys
sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

python_module_version() {
  local module="$1"
  python - "${module}" <<'PY'
import importlib
import sys
module = sys.argv[1]
try:
    mod = importlib.import_module(module)
except Exception:
    print("")
else:
    print(getattr(mod, "__version__", ""))
PY
}

python_module_origin() {
  local module="$1"
  python - "${module}" <<'PY'
import importlib.util
import sys
spec = importlib.util.find_spec(sys.argv[1])
print(spec.origin if spec and spec.origin else "")
PY
}

python_dist_location() {
  local dist="$1"
  python - "${dist}" <<'PY'
import importlib.metadata as metadata
import sys
try:
    dist = metadata.distribution(sys.argv[1])
except metadata.PackageNotFoundError:
    print("")
else:
    print(dist.locate_file(""))
PY
}

uninstall_if_local() {
  local dist="$1"
  local module="${2:-$1}"
  local origin origin_real dist_location dist_location_real env_real
  origin="$(python_module_origin "${module}")"
  dist_location="$(python_dist_location "${dist}")"
  origin_real="$(readlink -f "${origin}" 2>/dev/null || printf '%s' "${origin}")"
  dist_location_real="$(readlink -f "${dist_location}" 2>/dev/null || printf '%s' "${dist_location}")"
  env_real="$(readlink -f "${ENV_PATH}" 2>/dev/null || printf '%s' "${ENV_PATH}")"
  if [[ "${origin}" == "${ENV_PATH}"/* ||
        "${origin}" == "${env_real}"/* ||
        "${origin_real}" == "${env_real}"/* ||
        "${dist_location}" == "${ENV_PATH}"/* ||
        "${dist_location}" == "${env_real}"/* ||
        "${dist_location_real}" == "${env_real}"/* ]]; then
    echo "[setup][repair] removing local ${dist} from previous broken install"
    pip uninstall -y "${dist}" >/dev/null || true
  fi
}

repair_known_bad_packages() {
  echo "[setup][repair] checking for locally-shadowed numpy/torch packages ..."
  local numpy_version torch_version
  numpy_version="$(python_module_version numpy)"
  if [[ "${numpy_version}" =~ ^([0-9]+)\. ]] && (( BASH_REMATCH[1] > EXPECTED_NUMPY_MAJOR_MAX )); then
    uninstall_if_local numpy numpy
  fi

  torch_version="$(python_module_version torch)"
  if [[ -n "${torch_version}" && "${torch_version%%+*}" != "${EXPECTED_TORCH_PUBLIC_VERSION}" ]]; then
    uninstall_if_local torch torch
  fi

  # These optional packages are the ones that pulled incompatible torch pins.
  uninstall_if_local causal-conv1d causal_conv1d
  uninstall_if_local mamba-ssm mamba_ssm
}

ensure_torch_pin() {
  local torch_version
  torch_version="$(python_module_version torch)"
  if [[ "${torch_version%%+*}" != "${EXPECTED_TORCH_PUBLIC_VERSION}" ]]; then
    echo "[setup][repair] enforcing torch ${EXPECTED_TORCH_VERSION} ..."
    pip install --no-index --force-reinstall --no-deps \
      torch=="${EXPECTED_TORCH_VERSION}" \
      triton==3.2.0+computecanada \
      nvidia_ml_py
  fi
}

# ── 1. Compute Canada module stack ───────────────────────────────────────────
echo "[setup] loading Compute Canada modules ..."
module --force purge
module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a
# Optional: gcc 12 is what cuda/12.6 expects. StdEnv/2023 handles this.

# ── 2. Create / reuse the venv ───────────────────────────────────────────────
if [[ ! -d "${ENV_PATH}" ]]; then
  echo "[setup] creating new venv at ${ENV_PATH}"
  mkdir -p "$(dirname "${ENV_PATH}")"
  virtualenv --no-download "${ENV_PATH}"
else
  echo "[setup] reusing existing venv at ${ENV_PATH}"
fi
# shellcheck disable=SC1091
source "${ENV_PATH}/bin/activate"
python_version_ok || {
  echo "[setup] ERROR: expected Python 3.11 after module load and venv activation." >&2
  python --version >&2
  exit 1
}

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  echo "[setup] --verify-only requested; skipping all installs."
else

repair_known_bad_packages
pip install --no-index --upgrade pip setuptools wheel >/dev/null

# ── 3. Phase A - core CC wheelhouse packages (matches v1 + missing v2) ───────
# Everything here is installed from the local Compute Canada wheelhouse with
# `--no-index`. These are the +computecanada wheels that ship with CC.
echo "[setup][A] installing core wheels from CC wheelhouse ..."
pip install --no-index \
  numpy \
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

# Torch + CUDA stack (same versions as v1 baseline). Use --no-deps so the
# resolver cannot replace the pinned CC torch wheel.  Do NOT use
# --force-reinstall here: reinstalling torch when it is already correct wastes
# ~4 GB of free-space headroom (uninstall + re-write) and fails on full
# project quotas.  ensure_torch_pin() below handles the repair case - it only
# force-reinstalls when the detected version actually differs from the pin.
_torch_now="$(python_module_version torch)"
if [[ "${_torch_now%%+*}" != "${EXPECTED_TORCH_PUBLIC_VERSION}" ]]; then
  pip install --no-index --force-reinstall --no-deps \
    torch==2.6.0+computecanada \
    triton==3.2.0+computecanada \
    nvidia_ml_py
else
  # Correct version already installed; just make sure triton + nvidia_ml_py are present.
  pip install --no-index --no-deps \
    triton==3.2.0+computecanada \
    nvidia_ml_py
fi
ensure_torch_pin

# ONNX stack.
pip install --no-index \
  onnx \
  onnxruntime \
  onnxscript \
  onnx-ir \
  safetensors \
  hf_xet

# Hardware probing + ML metrics (REQUIRED additions over v1).
pip install --no-index \
  psutil \
  scikit-learn \
  pydicom \
  SimpleITK

# ── 4. Phase B - optional MedIA-rigor packages (some need PyPI) ──────────────
if [[ "${SKIP_OPTIONAL}" != "1" ]]; then
  echo "[setup][B] installing optional MedIA-rigor packages ..."

  # MONAI + medpy: try CC wheelhouse first, fall back to PyPI.
  pip install --no-index --no-deps monai 2>/dev/null \
    || pip install --no-deps monai --quiet \
    || echo "[setup][B] monai unavailable - metric_check MONAI cross-check disabled."

  pip install --no-index --no-deps medpy 2>/dev/null \
    || pip install --no-deps medpy --quiet \
    || echo "[setup][B] medpy unavailable - metric_check medpy cross-check disabled."

  # highdicom: not usually in CC wheelhouse. Install without deps so PyPI does
  # not shadow scipy-stack's numpy with an incompatible local numpy.
  pip install --no-deps highdicom --quiet \
    || echo "[setup][B] highdicom unavailable - dicom_seg subcommand disabled."
fi
ensure_torch_pin

# ── 5. Phase C - optional CUDA-built SegMamba dependency ─────────────────────
# mamba-ssm + causal-conv1d are optional and must not be allowed to resolve
# torch themselves: available wheels currently advertise incompatible torch
# pins. If explicitly requested, build/source-install them against the active
# pinned torch + CUDA toolkit. This requires:
#   • a node with `nvcc` (the cuda/12.6 module provides it)
#   • a recent ninja (>= 1.11) - CC's wheel covers it
#   • CUDA_HOME pointing at $EBROOTCUDA so pip's build_ext finds it
#
# If you only need the UNet3D teacher path (default), leave this disabled.
if [[ "${WITH_MAMBA}" == "1" ]]; then
  if ! python -c "import mamba_ssm" 2>/dev/null; then
    echo "[setup][C] building mamba-ssm + causal-conv1d from PyPI source ..."
    export CUDA_HOME="${EBROOTCUDA:-${CUDA_HOME:-}}"
    if [[ -z "${CUDA_HOME}" ]]; then
      echo "[setup][C] WARNING: CUDA_HOME not set - mamba-ssm build will likely fail."
    else
      echo "[setup][C] CUDA_HOME=${CUDA_HOME}"
    fi
    # causal-conv1d must come first; mamba-ssm imports it at install time.
    # --no-deps is intentional: CC's available wheels currently request
    # incompatible torch versions, so torch must remain pinned above.
    pip install --no-deps --no-build-isolation --no-binary=causal-conv1d causal-conv1d --quiet \
      || echo "[setup][C] causal-conv1d build failed; SegMamba teacher disabled."
    ensure_torch_pin
    pip install --no-deps --no-build-isolation --no-binary=mamba-ssm mamba-ssm --quiet \
      || echo "[setup][C] mamba-ssm build failed; SegMamba teacher disabled."
    ensure_torch_pin
  else
    echo "[setup][C] mamba-ssm already importable; skipping."
  fi
else
  echo "[setup][C] default UNet3D path; skipping optional SegMamba dependency."
  echo "[setup][C] pass --with-mamba only if you explicitly need --teacher_arch segmamba."
fi

fi  # end VERIFY_ONLY guard

# ── 6. Verification ──────────────────────────────────────────────────────────
echo
echo "[setup] running verification ..."
python "${VERIFY_SCRIPT}"
echo
echo "[setup] done. Activate with:"
echo "    module load StdEnv/2023 python/3.11.5 cuda/12.6 scipy-stack/2024a"
echo "    source ${ENV_PATH}/bin/activate"
