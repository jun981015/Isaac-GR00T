#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
RUN_NAME="${RUN_NAME:-robocasa_n15_eval_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_NAME}}"
GPU_DEVICE="${GPU_DEVICE:-0}"
PORT="${PORT:-8011}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"

TASKS=(
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
)

for task in "${TASKS[@]}"; do
  task_container="isaac-gr00t-robocasa-bench-${RUN_NAME}-${task}"
  ENV_NAME="${task}" \
  CONTAINER_NAME="${task_container}" \
  OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
  GPU_DEVICE="${GPU_DEVICE}" \
  PORT="${PORT}" \
  MODEL_HOST="${MODEL_HOST}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  REPLACE="${REPLACE:-0}" \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
done

echo "Launched 3 task eval containers."
echo "Output root: ${BASE_OUTPUT_DIR}"
