#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
PORT="${PORT:-18161}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_weight1_alpha0_50k_100ep_actionseed123_env8_conda_800step_$(date +%Y%m%d_%H%M%S)}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
GPU_LIST_CSV="${GPU_LIST_CSV:-2}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3}"
SCHEDULE_BASE="${SCHEDULE_BASE:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules}"

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

schedule_for_task() {
  local env_name="$1"
  find "${SCHEDULE_BASE}" -type f -name "${env_name}_${N_EPISODES}eps.json" | sort | head -n 1
}

IFS=',' read -r -a GPUS <<< "${GPU_LIST_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_LIST_CSV must contain at least one GPU id" >&2
  exit 1
fi
if [[ "${#SEEDS[@]}" -lt 1 ]]; then
  echo "SEEDS_CSV must contain at least one seed" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"

active_jobs=0
gpu_idx=0

run_one() {
  local seed="$1"
  local env_name="$2"
  local gpu="$3"
  local schedule_path="$4"
  local output_dir="${OUTPUT_ROOT}/action_seed_${seed}"
  local log_path="${OUTPUT_ROOT}/logs/${env_name}_seed${seed}_gpu${gpu}.log"

  if [[ -z "${schedule_path}" || ! -f "${schedule_path}" ]]; then
    echo "[missing_schedule] env=${env_name}" | tee -a "${OUTPUT_ROOT}/logs/missing_schedule.log"
    return 0
  fi

  echo "[eval_start] env=${env_name} seed=${seed} gpu=${gpu} schedule=${schedule_path}" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${gpu}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV="${CONDA_ENV}" \
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
  SKIP_EXISTING=1 \
  bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh \
    2>&1 | tee "${log_path}"
  echo "[eval_done] env=${env_name} seed=${seed} gpu=${gpu}" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
}

for seed in "${SEEDS[@]}"; do
  for env_name in "${TASKS[@]}"; do
    gpu="${GPUS[$gpu_idx]}"
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    schedule_path="$(schedule_for_task "${env_name}")"
    run_one "${seed}" "${env_name}" "${gpu}" "${schedule_path}" &
    active_jobs=$((active_jobs + 1))

    if [[ "${active_jobs}" -ge "${#GPUS[@]}" ]]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
done

wait
echo "[all_done] output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
