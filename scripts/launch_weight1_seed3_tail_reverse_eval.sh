#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_weight1_alpha0_50k_100ep_actionseed123_env8_conda_800step_20260505}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
PORT="${PORT:-18161}"
GPU="${GPU:-2}"
SEED="${SEED:-3}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"

TASKS=(
  TurnSinkSpout
  TurnOnStove
  TurnOnSinkFaucet
  TurnOnMicrowave
  TurnOffStove
)

schedule_for_task() {
  local env_name="$1"
  find "${SCHEDULE_BASE}" -type f -name "${env_name}_${N_EPISODES}eps.json" | sort | head -n 1
}

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"

for env_name in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${env_name}")"
  if [[ -z "${schedule_path}" || ! -f "${schedule_path}" ]]; then
    echo "[missing_schedule] env=${env_name}" | tee -a "${OUTPUT_ROOT}/logs/seed3_tail_reverse.log"
    continue
  fi

  echo "[eval_start] env=${env_name} seed=${SEED} gpu=${GPU} schedule=${schedule_path}" | tee -a "${OUTPUT_ROOT}/logs/seed3_tail_reverse.log"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV="${CONDA_ENV}" \
  MODEL_HOST="${MODEL_HOST}" \
  PORT="${PORT}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/action_seed_${SEED}" \
  SCHEDULE_PATH="${schedule_path}" \
  ENV_NAME="${env_name}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ENVS="${N_ENVS}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  N_ACTION_STEPS=16 \
  VIDEO_SOURCE=obs \
  VIDEO_RENDER_SIZE=0 \
  VIDEO_SCALE=1 \
  VIDEO_STEPS_PER_RENDER=4 \
  WRITE_VIDEO=1 \
  STREAM_VIDEO=1 \
  SKIP_EXISTING=1 \
  bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${env_name}_seed${SEED}_gpu${GPU}_tail_reverse.log"
  echo "[eval_done] env=${env_name} seed=${SEED} gpu=${GPU}" | tee -a "${OUTPUT_ROOT}/logs/seed3_tail_reverse.log"
done

echo "[all_done] seed=${SEED} tail reverse output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/seed3_tail_reverse.log"
