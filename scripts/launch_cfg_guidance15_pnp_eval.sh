#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/junhyeong/Value/Isaac-GR00T}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO}/local_outputs/robocasa_cfg_retrain/ann24_plus_weight1eval_failuretag_drop02_b64_50000step_gpu3_20260519_172000/checkpoint-40000}"
RUN_ID="${RUN_ID:-cfg_failuretag40k_guidance15_pnp_seed1_env8_800step_novideo_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO}/local_outputs/robocasa_benchmark/schedules}"
POLICY_GPU="${POLICY_GPU:-2}"
EVAL_GPU="${EVAL_GPU:-2}"
PORT="${PORT:-18194}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
CFG_GUIDANCE_SCALE="${CFG_GUIDANCE_SCALE:-1.5}"
CONDA_POLICY_ENV="${CONDA_POLICY_ENV:-gr00t}"
CONDA_EVAL_ENV="${CONDA_EVAL_ENV:-robocasa-eval}"
WRITE_VIDEO="${WRITE_VIDEO:-0}"
STREAM_VIDEO="${STREAM_VIDEO:-1}"
SAVE_ROLLOUT_HDF5="${SAVE_ROLLOUT_HDF5:-0}"
TASKS_CSV="${TASKS_CSV:-}"

TASKS=(
  PnPCabToCounter
  PnPCounterToCab
  PnPCounterToMicrowave
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
  PnPSinkToCounter
  PnPStoveToCounter
)

if [[ -n "${TASKS_CSV}" ]]; then
  IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
fi

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
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            print("server ready", flush=True)
            raise SystemExit(0)
    except OSError:
        time.sleep(5)
raise SystemExit("server not ready")
PY
}

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO}"

log "output=${OUTPUT_ROOT}"
log "checkpoint=${CHECKPOINT_DIR}"
log "policy_gpu=${POLICY_GPU} eval_gpu=${EVAL_GPU} port=${PORT}"
log "seed=${SEED} n_envs=${N_ENVS} n_episodes=${N_EPISODES} max_steps=${MAX_EPISODE_STEPS}"
log "outcome_conditioning=guidance outcome_prompt_style=failure_tag cfg_guidance_scale=${CFG_GUIDANCE_SCALE}"
log "write_video=${WRITE_VIDEO} stream_video=${STREAM_VIDEO} save_rollout_hdf5=${SAVE_ROLLOUT_HDF5}"
log "tasks=${TASKS[*]}"

for task in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${task}")"
  if [[ -z "${schedule_path}" || ! -s "${schedule_path}" ]]; then
    log "missing schedule task=${task}"
    exit 1
  fi
  log "schedule task=${task} path=${schedule_path}"
done

SERVER_SESSION="cfg_guidance15_pnp_server_${RUN_ID}"
if ! tmux has-session -t "${SERVER_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SERVER_SESSION}" \
    "bash -lc 'cd ${REPO}; \
      export CUDA_VISIBLE_DEVICES=${POLICY_GPU}; \
      export PYTHONPATH=${REPO}:/home/junhyeong/Value/robocasa:\${PYTHONPATH:-}; \
      export NO_ALBUMENTATIONS_UPDATE=1; \
      conda run --no-capture-output -n ${CONDA_POLICY_ENV} python scripts/inference_service.py \
        --server \
        --host 127.0.0.1 \
        --port ${PORT} \
        --model_path ${CHECKPOINT_DIR} \
        --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        --outcome-conditioning guidance \
        --outcome-prompt-style failure_tag \
        --cfg-guidance-scale ${CFG_GUIDANCE_SCALE} \
        > ${OUTPUT_ROOT}/logs/server.log 2>&1'"
fi
log "server_session=${SERVER_SESSION}"
wait_for_server

for task in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${task}")"
  log "eval start task=${task} seed=${SEED} gpu=${EVAL_GPU}"
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
