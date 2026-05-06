#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
PORT="${PORT:-18131}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_r10_50k_seed123_existing_sched_20260503}"
LEGACY_SEED1_OUTPUT="${LEGACY_SEED1_OUTPUT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_r10_50k_existing_sched_20260503_005351}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-4}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
GPU_LIST_CSV="${GPU_LIST_CSV:-2,3}"
SEEDS_CSV="${SEEDS_CSV:-1,2,3}"

TASKS=(
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
  PnPCounterToMicrowave
  CoffeeSetupMug
  CoffeePressButton
  OpenSingleDoor
  CloseSingleDoor
  TurnOnMicrowave
)

schedule_for_task() {
  case "$1" in
    PnPCounterToSink|PnPCounterToStove|PnPMicrowaveToCounter)
      printf '%s\n' "${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/${1}_${N_EPISODES}eps.json"
      ;;
    PnPCounterToMicrowave|CoffeeSetupMug|CoffeePressButton|OpenSingleDoor|CloseSingleDoor|TurnOnMicrowave)
      printf '%s\n' "${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430/seed1/${1}_${N_EPISODES}eps.json"
      ;;
    *)
      return 1
      ;;
  esac
}

output_for_seed() {
  if [[ "$1" == "1" ]]; then
    printf '%s\n' "${LEGACY_SEED1_OUTPUT}"
  else
    printf '%s\n' "${OUTPUT_ROOT}/action_seed_${1}"
  fi
}

IFS=',' read -r -a GPUS <<< "${GPU_LIST_CSV}"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "GPU_LIST_CSV must contain at least one GPU id" >&2
  exit 1
fi
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
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
  local output_dir="$5"
  local log_path="${OUTPUT_ROOT}/logs/${env_name}_seed${seed}_gpu${gpu}.log"

  if [[ ! -f "${schedule_path}" ]]; then
    echo "[missing_schedule] env=${env_name} schedule=${schedule_path}" | tee -a "${OUTPUT_ROOT}/logs/missing_schedule.log"
    return 0
  fi

  echo "[eval_start] env=${env_name} seed=${seed} gpu=${gpu} output=${output_dir}" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
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
  SCHEDULE_DIR="$(dirname "$(dirname "${schedule_path}")")" \
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
    output_dir="$(output_for_seed "${seed}")"

    run_one "${seed}" "${env_name}" "${gpu}" "${schedule_path}" "${output_dir}" &
    active_jobs=$((active_jobs + 1))

    if [[ "${active_jobs}" -ge "${#GPUS[@]}" ]]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
done

wait
echo "[all_done] output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/manager.log"
