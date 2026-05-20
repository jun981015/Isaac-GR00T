#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/local_outputs/robocasa_awr_retrain/ann24_annotations_new_r5_alpha100_clip2_b64_50k_gpu3_20260512_210601/checkpoint-50000}"
RUN_ID="${RUN_ID:-ann24_newanno_mixed_r5_ckpt50000_seed1_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules}"
PORT="${PORT:-18177}"
GPU="${GPU:-3}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
CONDA_POLICY_ENV="${CONDA_POLICY_ENV:-gr00t}"
CONDA_EVAL_ENV="${CONDA_EVAL_ENV:-robocasa-eval}"

TASKS=(
  CloseDoubleDoor
  CloseDrawer
  CloseSingleDoor
  CoffeePressButton
  CoffeeSetupMug
  MicrowaveThawing
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
  PrepareCoffee
  TurnOffMicrowave
  TurnOffSinkFaucet
  TurnOffStove
  TurnOnMicrowave
  TurnOnSinkFaucet
  TurnOnStove
)

mkdir -p "${OUTPUT_ROOT}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
}

schedule_for_task() {
  local env_name="$1"
  find "${SCHEDULE_BASE}" -type f -name "${env_name}_${N_EPISODES}eps.json" | sort | head -n 1
}

cd "${REPO_DIR}"
log "checkpoint=${CHECKPOINT_DIR}"
log "output=${OUTPUT_ROOT}"
log "gpu=${GPU} port=${PORT} seed=${SEED} n_envs=${N_ENVS}"

SERVER_SESSION="server_${RUN_ID}"
if ! tmux has-session -t "${SERVER_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SERVER_SESSION}" \
    "bash -lc 'cd ${REPO_DIR}; \
      export CUDA_VISIBLE_DEVICES=${GPU}; \
      export PYTHONPATH=${REPO_DIR}:/home/junhyeong/Value/robocasa:\${PYTHONPATH:-}; \
      export NO_ALBUMENTATIONS_UPDATE=1; \
      conda run --no-capture-output -n ${CONDA_POLICY_ENV} python scripts/inference_service.py \
        --server \
        --host 127.0.0.1 \
        --port ${PORT} \
        --model_path ${CHECKPOINT_DIR} \
        --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        > ${OUTPUT_ROOT}/logs/server.log 2>&1'"
fi

log "server session=${SERVER_SESSION}"
for _ in $(seq 1 180); do
  if ss -ltn | grep -q ":${PORT} "; then
    log "server ready"
    break
  fi
  sleep 10
done

if ! ss -ltn | grep -q ":${PORT} "; then
  log "server did not open port ${PORT}"
  exit 1
fi

for env_name in "${TASKS[@]}"; do
  schedule_path="$(schedule_for_task "${env_name}")"
  if [[ -z "${schedule_path}" || ! -f "${schedule_path}" ]]; then
    log "missing schedule env=${env_name}"
    continue
  fi

  log "eval start env=${env_name} schedule=${schedule_path}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV="${CONDA_EVAL_ENV}" \
  MODEL_HOST=127.0.0.1 \
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
    > "${OUTPUT_ROOT}/logs/${env_name}_seed${SEED}_gpu${GPU}.log" 2>&1
  log "eval done env=${env_name}"
done

log "all done"
