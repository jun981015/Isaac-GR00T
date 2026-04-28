#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCHEDULE_DIR="${SCHEDULE_DIR:?set SCHEDULE_DIR}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${REPO_DIR}/local_outputs/robocasa_awr_retrain}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

launch_bench() {
  local label="$1"
  local gpu="$2"
  local port="$3"
  local step="$4"
  local task="$5"
  local cname="isaac-gr00t-robocasa-bench-${label}-ckpt${step}-${task}-seed${SEED}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] launch ${cname}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${gpu}" \
  MODEL_HOST=127.0.0.1 \
  PORT="${port}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ACTION_STEPS=16 \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  VIDEO_FPS=20 \
  VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE}" \
  VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER}" \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${label}/ckpt${step}" \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
}

wait_bench() {
  local label="$1"
  local step="$2"
  local task="$3"
  local cname="isaac-gr00t-robocasa-bench-${label}-ckpt${step}-${task}-seed${SEED}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] wait ${cname}"
  local status
  status="$(docker wait "${cname}")"
  docker logs "${cname}" > "${OUTPUT_ROOT}/${label}/ckpt${step}/docker_${task}.log" 2>&1 || true
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${cname} status=${status}"
  [[ "${status}" == "0" ]]
}

start_server() {
  local label="$1"
  local policy_dir="$2"
  local gpu="$3"
  local port="$4"
  local step="$5"
  local sname="isaac-gr00t-robocasa-http-${label}-ckpt${step}-seed${SEED}"
  CHECKPOINT_DIR="${BASE_CKPT_DIR}/${policy_dir}/checkpoint-${step}" \
  GPU_DEVICE="${gpu}" \
  PORT="${port}" \
  CONTAINER_NAME="${sname}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${label}/http_ckpt${step}" \
  BOOTSTRAP_DEPS=1 \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_groot_robocasa_http_server.sh"
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "server ${sname} failed health check" >&2
  return 1
}

stop_server() {
  local label="$1"
  local step="$2"
  docker rm -f "isaac-gr00t-robocasa-http-${label}-ckpt${step}-seed${SEED}" >/dev/null 2>&1 || true
}

labels=(awr10 awr15 baseline)
policy_dirs=(awr_alpha10_20k awr_alpha15_20k baseline_noawr_20k)
gpus=(1 2 3)
ports=(8031 8032 8033)

# Current state: ckpt20000 Sink has already been launched manually.
for i in "${!labels[@]}"; do
  wait_bench "${labels[$i]}" 20000 PnPCounterToSink &
done
wait

for task in PnPCounterToStove PnPMicrowaveToCounter; do
  for i in "${!labels[@]}"; do
    launch_bench "${labels[$i]}" "${gpus[$i]}" "${ports[$i]}" 20000 "${task}" &
  done
  wait
  for i in "${!labels[@]}"; do
    wait_bench "${labels[$i]}" 20000 "${task}" &
  done
  wait
done

for i in "${!labels[@]}"; do
  stop_server "${labels[$i]}" 20000
done

for i in "${!labels[@]}"; do
  start_server "${labels[$i]}" "${policy_dirs[$i]}" "${gpus[$i]}" "${ports[$i]}" 10000 &
done
wait

for task in PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter; do
  for i in "${!labels[@]}"; do
    launch_bench "${labels[$i]}" "${gpus[$i]}" "${ports[$i]}" 10000 "${task}" &
  done
  wait
  for i in "${!labels[@]}"; do
    wait_bench "${labels[$i]}" 10000 "${task}" &
  done
  wait
done

for i in "${!labels[@]}"; do
  stop_server "${labels[$i]}" 10000
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] all evals complete"
