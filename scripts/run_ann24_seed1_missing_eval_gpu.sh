#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
GPU_ID="${GPU_ID:?Set GPU_ID, e.g. 2 or 3}"
TASKS_CSV="${TASKS_CSV:?Set TASKS_CSV to comma-separated RoboCasa env names}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
PORT="${PORT:-18131}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_r10_50k_existing_sched_20260503_005351}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/ann24_atomic_missing_100ep_envseed1_20260503/seed1}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_r10_seed1_missing_nenv${N_ENVS}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S)/logs}"

export PATH="/home/junhyeong/miniconda3/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_EGL_DEVICE_ID="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
echo "[ann24_missing_start] gpu=${GPU_ID} seed=${SEED} n_envs=${N_ENVS} tasks=${TASKS_CSV}" | tee -a "${LOG_ROOT}/manager.log"

for ENV_NAME in "${TASKS[@]}"; do
  SCHEDULE_PATH="${SCHEDULE_DIR}/${ENV_NAME}_${N_EPISODES}eps.json"
  if [[ ! -f "${SCHEDULE_PATH}" ]]; then
    echo "[ann24_missing_skip_missing_schedule] ${ENV_NAME} ${SCHEDULE_PATH}" | tee -a "${LOG_ROOT}/manager.log"
    continue
  fi

  echo "[ann24_missing_task_start] ${ENV_NAME}" | tee -a "${LOG_ROOT}/manager.log"
  CONDA_ENV="${CONDA_ENV}" \
    MODEL_HOST="${MODEL_HOST}" \
    PORT="${PORT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SCHEDULE_PATH="${SCHEDULE_PATH}" \
    ENV_NAME="${ENV_NAME}" \
    SEED="${SEED}" \
    N_EPISODES="${N_EPISODES}" \
    N_ENVS="${N_ENVS}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}" \
    N_ACTION_STEPS="${N_ACTION_STEPS:-16}" \
    VIDEO_SOURCE="${VIDEO_SOURCE:-obs}" \
    VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-0}" \
    VIDEO_SCALE="${VIDEO_SCALE:-1}" \
    VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}" \
    WRITE_VIDEO="${WRITE_VIDEO:-1}" \
    STREAM_VIDEO="${STREAM_VIDEO:-1}" \
    SKIP_EXISTING="${SKIP_EXISTING:-1}" \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh" \
    2>&1 | tee "${LOG_ROOT}/${ENV_NAME}_seed${SEED}_gpu${GPU_ID}.log"
  echo "[ann24_missing_task_done] ${ENV_NAME}" | tee -a "${LOG_ROOT}/manager.log"
done

echo "[ann24_missing_all_done] gpu=${GPU_ID}" | tee -a "${LOG_ROOT}/manager.log"
