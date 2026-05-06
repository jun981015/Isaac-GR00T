#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-groot-smoke}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-/home/junhyeong/Value/robocasa}"
HF_HOME_DIR="${HF_HOME_DIR:-/home/junhyeong/.cache/huggingface}"
ENV_FILE="${REPO_ROOT}/envs/groot-smoke-environment.yml"
REQ_FILE="${REPO_ROOT}/envs/groot-smoke-requirements.txt"

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

# Install package metadata/code from a clean source copy. Installing directly
# from REPO_ROOT can make setuptools scan large Docker/runtime outputs such as
# local_outputs/** and fail on root-owned files.
CLEAN_SRC="$(mktemp -d /tmp/groot-smoke-src.XXXXXX)"
cleanup_clean_src() {
  rm -rf "${CLEAN_SRC}"
}
trap cleanup_clean_src EXIT
rsync -a \
  --exclude '.git' \
  --exclude '.cache' \
  --exclude '.mypy_cache' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '.tox' \
  --exclude '.nox' \
  --exclude '.venv' \
  --exclude '.ipynb_checkpoints' \
  --exclude '__pycache__' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude '*.egg-info' \
  --exclude 'local_outputs' \
  --exclude 'wandb' \
  --exclude 'envs/groot-smoke-build.log' \
  --exclude 'envs/groot-smoke-resume.log' \
  "${REPO_ROOT}/" "${CLEAN_SRC}/"
conda run -n "${ENV_NAME}" python -m pip install "${CLEAN_SRC}" --no-deps

# Match the active OpenCV import observed in the Docker image: cv2==4.8.0.
conda run -n "${ENV_NAME}" python -m pip install --force-reinstall --no-deps opencv-python==4.8.0.74

# Docker image explicitly removed transformer-engine before installing flash-attn.
conda run -n "${ENV_NAME}" python -m pip uninstall -y transformer-engine || true

# Must be installed after torch is importable. Build isolation can pull mismatched torch.
CONDA_PREFIX_PATH="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
CUDA_HOME="${CONDA_PREFIX_PATH}" MAX_JOBS="${MAX_JOBS:-4}" conda run -n "${ENV_NAME}" python -m pip install \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  flash-attn==2.7.1.post4

ACTIVATE_DIR="${CONDA_PREFIX_PATH}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX_PATH}/etc/conda/deactivate.d"
PURELIB="$(conda run -n "${ENV_NAME}" python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/groot-smoke.sh" <<EOF
export _GROOT_SMOKE_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${PURELIB}/nvidia/cuda_nvrtc/lib:${PURELIB}/nvidia/nvjitlink/lib:${PURELIB}/nvidia/npp/lib:${PURELIB}/nvidia/cuda_runtime/lib:${PURELIB}/nvidia/cublas/lib:${PURELIB}/nvidia/cudnn/lib:${PURELIB}/nvidia/cusparse/lib:${PURELIB}/nvidia/cusolver/lib:${PURELIB}/nvidia/nccl/lib:\${LD_LIBRARY_PATH:-}"
export _GROOT_SMOKE_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export PYTHONPATH="${REPO_ROOT}:${ROBOCASA_ROOT}:\${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_DIR}"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}/hub"
EOF
cat > "${DEACTIVATE_DIR}/groot-smoke.sh" <<'EOF'
export LD_LIBRARY_PATH="${_GROOT_SMOKE_OLD_LD_LIBRARY_PATH:-}"
export PYTHONPATH="${_GROOT_SMOKE_OLD_PYTHONPATH:-}"
unset _GROOT_SMOKE_OLD_LD_LIBRARY_PATH
unset _GROOT_SMOKE_OLD_PYTHONPATH
EOF

echo "Created conda env: ${ENV_NAME}"
echo "Suggested runtime env:"
echo "  export PYTHONPATH=${REPO_ROOT}:${ROBOCASA_ROOT}:\${PYTHONPATH:-}"
echo "  export HF_HOME=${HF_HOME_DIR}"
echo "  export TRANSFORMERS_CACHE=${HF_HOME_DIR}/hub"
