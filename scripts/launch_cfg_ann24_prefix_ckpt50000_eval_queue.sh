#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/local_outputs/robocasa_cfg_retrain/ann24_plus_weight1eval_cfgdrop02_b64_50k_gpu2_20260519_031000/checkpoint-50000}"
RUN_ID="${RUN_ID:-cfg_prefix_ann24_plus_weight1eval_ckpt50000_100ep_seed123_env8_800step_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules}"
PORT="${PORT:-18191}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
POLICY_GPU="${POLICY_GPU:-2}"
EVAL_GPU_LIST_CSV="${EVAL_GPU_LIST_CSV:-2}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
CONDA_POLICY_ENV="${CONDA_POLICY_ENV:-gr00t}"
CONDA_EVAL_ENV="${CONDA_EVAL_ENV:-robocasa-eval}"
OUTCOME_CONDITIONING="${OUTCOME_CONDITIONING:-success}"
OUTCOME_PROMPT_STYLE="${OUTCOME_PROMPT_STYLE:-prefix}"
CFG_GUIDANCE_SCALE="${CFG_GUIDANCE_SCALE:-1.0}"
SAVE_ROLLOUT_HDF5="${SAVE_ROLLOUT_HDF5:-0}"
TASK_ORDER="${TASK_ORDER:-forward}"
DRY_RUN="${DRY_RUN:-0}"

TASKS=(
  CloseDoubleDoor
  CloseDrawer
  CloseSingleDoor
  CoffeePressButton
  CoffeeServeMug
  CoffeeSetupMug
  OpenDoubleDoor
  OpenDrawer
  OpenSingleDoor
  PnPCabToCounter
  PnPCounterToCab
  PnPCounterToMicrowave
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
  PnPSinkToCounter
  PnPStoveToCounter
  TurnOffMicrowave
  TurnOffSinkFaucet
  TurnOffStove
  TurnOnMicrowave
  TurnOnSinkFaucet
  TurnOnStove
  TurnSinkSpout
)

if [[ "${TASK_ORDER}" == "reverse" ]]; then
  REVERSED_TASKS=()
  for ((idx=${#TASKS[@]}-1; idx>=0; idx--)); do
    REVERSED_TASKS+=("${TASKS[$idx]}")
  done
  TASKS=("${REVERSED_TASKS[@]}")
elif [[ "${TASK_ORDER}" != "forward" ]]; then
  echo "TASK_ORDER must be forward or reverse, got: ${TASK_ORDER}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
}

schedule_for_task() {
  local env_name="$1"
  find "${SCHEDULE_BASE}" -type f -name "${env_name}_${N_EPISODES}eps.json" | sort | head -n 1
}

wait_for_server() {
  python - "$MODEL_HOST" "$PORT" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print("ready")
            raise SystemExit(0)
    except OSError:
        time.sleep(5)
raise SystemExit(1)
PY
}

cd "${REPO_DIR}"
IFS=',' read -r -a EVAL_GPUS <<< "${EVAL_GPU_LIST_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if [[ "${#EVAL_GPUS[@]}" -lt 1 ]]; then
  echo "EVAL_GPU_LIST_CSV must contain at least one GPU id" >&2
  exit 1
fi

missing=0
for task in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${task}")"
  if [[ -z "${schedule_path}" || ! -s "${schedule_path}" ]]; then
    log "missing schedule task=${task}"
    missing=$((missing + 1))
  else
    log "schedule task=${task} path=${schedule_path}"
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "missing schedules: ${missing}" >&2
  exit 1
fi

log "checkpoint=${CHECKPOINT_DIR}"
log "output=${OUTPUT_ROOT}"
log "policy_gpu=${POLICY_GPU} eval_gpus=${EVAL_GPU_LIST_CSV} port=${PORT}"
log "seeds=${SEEDS_CSV} n_envs=${N_ENVS} n_episodes=${N_EPISODES} max_steps=${MAX_EPISODE_STEPS}"
log "outcome_conditioning=${OUTCOME_CONDITIONING} outcome_prompt_style=${OUTCOME_PROMPT_STYLE} cfg_guidance_scale=${CFG_GUIDANCE_SCALE}"
log "save_rollout_hdf5=${SAVE_ROLLOUT_HDF5}"
log "task_order=${TASK_ORDER}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry run complete"
  exit 0
fi

SERVER_SESSION="cfg_prefix_eval_server_${RUN_ID}"
if ! tmux has-session -t "${SERVER_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SERVER_SESSION}" \
    "bash -lc 'cd ${REPO_DIR}; \
      export CUDA_VISIBLE_DEVICES=${POLICY_GPU}; \
      export PYTHONPATH=${REPO_DIR}:/home/junhyeong/Value/robocasa:\${PYTHONPATH:-}; \
      export NO_ALBUMENTATIONS_UPDATE=1; \
      conda run --no-capture-output -n ${CONDA_POLICY_ENV} python scripts/inference_service.py \
        --server \
        --host ${MODEL_HOST} \
        --port ${PORT} \
        --model_path ${CHECKPOINT_DIR} \
        --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        --outcome-conditioning ${OUTCOME_CONDITIONING} \
        --outcome-prompt-style ${OUTCOME_PROMPT_STYLE} \
        --cfg-guidance-scale ${CFG_GUIDANCE_SCALE} \
        > ${OUTPUT_ROOT}/logs/server.log 2>&1'"
fi
log "server_session=${SERVER_SESSION}"

if ! wait_for_server; then
  log "server did not open ${MODEL_HOST}:${PORT}"
  exit 1
fi
log "server ready"

active_jobs=0
gpu_idx=0

run_one() {
  local seed="$1"
  local env_name="$2"
  local gpu="$3"
  local schedule_path="$4"
  local output_dir="${OUTPUT_ROOT}/action_seed_${seed}"
  local log_path="${OUTPUT_ROOT}/logs/${env_name}_seed${seed}_gpu${gpu}.log"

  log "eval start task=${env_name} seed=${seed} gpu=${gpu} schedule=${schedule_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${gpu}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV="${CONDA_EVAL_ENV}" \
  MODEL_HOST="${MODEL_HOST}" \
  PORT="${PORT}" \
  OUTPUT_DIR="${output_dir}" \
  SCHEDULE_PATH="${schedule_path}" \
  ENV_NAME="${env_name}" \
  SEED="${seed}" \
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
  SAVE_ROLLOUT_HDF5="${SAVE_ROLLOUT_HDF5}" \
  SKIP_EXISTING=1 \
  bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh \
    > "${log_path}" 2>&1
  log "eval done task=${env_name} seed=${seed} gpu=${gpu}"
}

for seed in "${SEEDS[@]}"; do
  for env_name in "${TASKS[@]}"; do
    gpu="${EVAL_GPUS[$gpu_idx]}"
    gpu_idx=$(( (gpu_idx + 1) % ${#EVAL_GPUS[@]} ))
    schedule_path="$(schedule_for_task "${env_name}")"
    run_one "${seed}" "${env_name}" "${gpu}" "${schedule_path}" &
    active_jobs=$((active_jobs + 1))
    if [[ "${active_jobs}" -ge "${#EVAL_GPUS[@]}" ]]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
done

wait
log "all done output=${OUTPUT_ROOT}"
