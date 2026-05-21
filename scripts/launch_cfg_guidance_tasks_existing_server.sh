#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO}/local_outputs/robocasa_benchmark/schedules}"
TASKS_CSV="${TASKS_CSV:?TASKS_CSV is required}"
EVAL_GPU="${EVAL_GPU:-2}"
PORT="${PORT:?PORT is required}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
CONDA_EVAL_ENV="${CONDA_EVAL_ENV:-robocasa-eval}"
WRITE_VIDEO="${WRITE_VIDEO:-1}"
STREAM_VIDEO="${STREAM_VIDEO:-1}"
SAVE_ROLLOUT_HDF5="${SAVE_ROLLOUT_HDF5:-1}"

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
}

schedule_for_task() {
  local env_name="$1"
  find "${SCHEDULE_BASE}" -type f -name "${env_name}_${N_EPISODES}eps.json" | sort | head -n 1
}

wait_for_server() {
  python - "${PORT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            print("server ready", flush=True)
            raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit("server not ready")
PY
}

log "output=${OUTPUT_ROOT}"
log "using_existing_server=127.0.0.1:${PORT}"
log "eval_gpu=${EVAL_GPU} seed=${SEED} n_envs=${N_ENVS} n_episodes=${N_EPISODES} max_steps=${MAX_EPISODE_STEPS}"
log "write_video=${WRITE_VIDEO} stream_video=${STREAM_VIDEO} save_rollout_hdf5=${SAVE_ROLLOUT_HDF5}"
log "tasks=${TASKS[*]}"
wait_for_server

for task in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${task}")"
  if [[ -z "${schedule_path}" || ! -s "${schedule_path}" ]]; then
    log "missing schedule task=${task}"
    exit 1
  fi
  log "eval start task=${task} seed=${SEED} gpu=${EVAL_GPU} schedule=${schedule_path}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${EVAL_GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV="${CONDA_EVAL_ENV}" \
  MODEL_HOST=127.0.0.1 \
  PORT="${PORT}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/action_seed_${SEED}" \
  SCHEDULE_PATH="${schedule_path}" \
  ENV_NAME="${task}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ENVS="${N_ENVS}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  N_ACTION_STEPS=16 \
  VIDEO_SOURCE=obs \
  VIDEO_RENDER_SIZE=0 \
  VIDEO_SCALE=1 \
  VIDEO_STEPS_PER_RENDER=4 \
  WRITE_VIDEO="${WRITE_VIDEO}" \
  STREAM_VIDEO="${STREAM_VIDEO}" \
  SAVE_ROLLOUT_HDF5="${SAVE_ROLLOUT_HDF5}" \
  SKIP_EXISTING=1 \
  bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh \
    > "${OUTPUT_ROOT}/logs/${task}_seed${SEED}_gpu${EVAL_GPU}.log" 2>&1
  log "eval done task=${task}"
done

log "all done output=${OUTPUT_ROOT}"
