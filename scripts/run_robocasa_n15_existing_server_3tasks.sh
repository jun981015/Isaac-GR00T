#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
LABEL="${LABEL:?set LABEL}"
CKPT_STEP="${CKPT_STEP:?set CKPT_STEP}"
GPU_DEVICE="${GPU_DEVICE:?set GPU_DEVICE}"
PORT="${PORT:?set PORT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCHEDULE_DIR="${SCHEDULE_DIR:?set SCHEDULE_DIR}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

TASKS=(
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
)

for task in "${TASKS[@]}"; do
  bench_name="isaac-gr00t-robocasa-bench-${LABEL}-ckpt${CKPT_STEP}-${task}-seed${SEED}"
  run_out="${OUTPUT_ROOT}/${LABEL}/ckpt${CKPT_STEP}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LABEL} ckpt${CKPT_STEP}: eval ${task}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${GPU_DEVICE}" \
  MODEL_HOST=127.0.0.1 \
  PORT="${PORT}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ACTION_STEPS="${N_ACTION_STEPS}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  VIDEO_FPS="${VIDEO_FPS}" \
  VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE}" \
  VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER}" \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  CONTAINER_NAME="${bench_name}" \
  OUTPUT_DIR="${run_out}" \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
  docker wait "${bench_name}" >/dev/null
  docker logs "${bench_name}" > "${run_out}/docker_${task}.log" 2>&1 || true
  docker rm -f "${bench_name}" >/dev/null 2>&1 || true
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LABEL} ckpt${CKPT_STEP}: done"
