#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
GPU_ID="${GPU_ID:?Set GPU_ID, e.g. 2 or 3}"
TASKS_CSV="${TASKS_CSV:?Set TASKS_CSV to comma-separated RoboCasa env names}"
WAIT_SESSION="${WAIT_SESSION:-}"
N_ENVS="${N_ENVS:-4}"
SEEDS_CSV="${SEEDS_CSV:-2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/ann24_r10_missing_seed23_nenv${N_ENVS}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs_gpu${GPU_ID}}"

export PATH="/home/junhyeong/miniconda3/bin:${PATH}"

mkdir -p "${LOG_ROOT}"

if [[ -n "${WAIT_SESSION}" ]]; then
  echo "[wait_start] ${WAIT_SESSION}" | tee -a "${LOG_ROOT}/manager.log"
  while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep "${WAIT_POLL_SEC:-60}"
  done
  echo "[wait_done] ${WAIT_SESSION}" | tee -a "${LOG_ROOT}/manager.log"
fi

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
for seed in "${SEEDS[@]}"; do
  seed_output="${OUTPUT_ROOT}/action_seed_${seed}"
  seed_log="${LOG_ROOT}/seed${seed}"
  mkdir -p "${seed_output}" "${seed_log}"
  echo "[seed_start] seed=${seed} gpu=${GPU_ID} output=${seed_output}" | tee -a "${LOG_ROOT}/manager.log"
  GPU_ID="${GPU_ID}" \
    TASKS_CSV="${TASKS_CSV}" \
    SEED="${seed}" \
    N_ENVS="${N_ENVS}" \
    OUTPUT_DIR="${seed_output}" \
    LOG_ROOT="${seed_log}" \
    bash "${REPO_DIR}/scripts/run_ann24_seed1_missing_eval_gpu.sh"
  echo "[seed_done] seed=${seed} gpu=${GPU_ID}" | tee -a "${LOG_ROOT}/manager.log"
done

echo "[seed23_missing_all_done] gpu=${GPU_ID}" | tee -a "${LOG_ROOT}/manager.log"
