#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/pm16_alpha50_50k_ckpt30000_seed1_50ep_20260428}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
GPU_DEVICE="${GPU_DEVICE:-3}"
PORT="${PORT:-8043}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_50k_gpu3/checkpoint-30000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"
VIDEO_SOURCE="${VIDEO_SOURCE:-obs}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

HTTP_CONTAINER="isaac-gr00t-robocasa-http-pm16-alpha50-50k-ckpt30000-seed1"
tasks=(PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter)

mkdir -p "${OUTPUT_ROOT}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] start HTTP ${HTTP_CONTAINER}"
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
GPU_DEVICE="${GPU_DEVICE}" \
PORT="${PORT}" \
CONTAINER_NAME="${HTTP_CONTAINER}" \
OUTPUT_DIR="${OUTPUT_ROOT}/http_server" \
REPLACE=1 \
BOOTSTRAP_DEPS=1 \
  bash "${REPO_DIR}/scripts/run_groot_robocasa_http_server.sh"

for task in "${tasks[@]}"; do
  cname="isaac-gr00t-robocasa-bench-pm16-alpha50-50k-ckpt30000-${task}"
  output_dir="${OUTPUT_ROOT}/pm16_alpha50_50k/ckpt30000"
  schedule_path="${SCHEDULE_DIR}/seed${SEED}/${task}_${N_EPISODES}eps.json"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch ${cname}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${GPU_DEVICE}" \
  MODEL_HOST=127.0.0.1 \
  PORT="${PORT}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ACTION_STEPS=16 \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  VIDEO_FPS=20 \
  VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE}" \
  VIDEO_SOURCE="${VIDEO_SOURCE}" \
  VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER}" \
  STREAM_VIDEO=1 \
  SKIP_EXISTING=1 \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  SCHEDULE_PATH="${schedule_path}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${output_dir}" \
  REPLACE=1 \
  BOOTSTRAP_DEPS=1 \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] wait ${cname}"
  status="$(docker wait "${cname}")"
  docker logs "${cname}" > "${output_dir}/docker_${task}.log" 2>&1 || true
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${cname} status=${status}"
  [[ "${status}" == "0" ]]
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] pm16 alpha50 50k ckpt30000 eval complete"
