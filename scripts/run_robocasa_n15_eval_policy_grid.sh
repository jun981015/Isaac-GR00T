#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
BASE_CKPT_DIR="${BASE_CKPT_DIR:-${REPO_DIR}/local_outputs/robocasa_awr_retrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/n15_policy_grid_seed1_50ep_hires_$(date +%Y%m%d_%H%M%S)}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed${SEED:-1}_${N_EPISODES:-50}ep}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"

TASKS=(
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
)

POLICY_NAMES=(awr_alpha10_20k awr_alpha15_20k baseline_noawr_20k)
POLICY_LABELS=(awr10 awr15 baseline)
POLICY_GPUS=(1 2 3)
POLICY_PORTS=(8031 8032 8033)
CHECKPOINT_STEPS=(20000 10000)

mkdir -p "${OUTPUT_ROOT}" "${SCHEDULE_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_health() {
  local port="$1"
  local deadline=$((SECONDS + 900))
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; do
    if (( SECONDS > deadline )); then
      echo "server on port ${port} did not become healthy" >&2
      return 1
    fi
    sleep 5
  done
}

generate_schedules() {
  log "Generating/reusing schedules in ${SCHEDULE_DIR}"
  for task in "${TASKS[@]}"; do
    local schedule_path="${SCHEDULE_DIR}/seed${SEED}/${task}_${N_EPISODES}eps.json"
    if [[ -f "${schedule_path}" ]]; then
      log "Schedule exists: ${schedule_path}"
      continue
    fi
    local cname="isaac-gr00t-robocasa-bench-schedule-${task}-seed${SEED}-${N_EPISODES}ep"
    log "Generating schedule ${task}"
    ENV_NAME="${task}" \
    GPU_DEVICE=3 \
    SEED="${SEED}" \
    N_EPISODES="${N_EPISODES}" \
    SCHEDULE_DIR="${SCHEDULE_DIR}" \
    GENERATE_SCHEDULE_ONLY=1 \
    REGENERATE_SCHEDULE=1 \
    CONTAINER_NAME="${cname}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/_schedule_${task}" \
    REPLACE=1 \
      bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
    docker wait "${cname}" >/dev/null
    docker logs "${cname}" > "${OUTPUT_ROOT}/schedule_${task}.log" 2>&1 || true
    docker rm -f "${cname}" >/dev/null 2>&1 || true
  done
}

run_policy_pipeline() {
  local idx="$1"
  local policy_name="${POLICY_NAMES[$idx]}"
  local label="${POLICY_LABELS[$idx]}"
  local gpu="${POLICY_GPUS[$idx]}"
  local port="${POLICY_PORTS[$idx]}"
  local policy_root="${OUTPUT_ROOT}/${label}"
  mkdir -p "${policy_root}"

  log "Policy ${label}: gpu=${gpu}, port=${port}"
  for step in "${CHECKPOINT_STEPS[@]}"; do
    local ckpt="${BASE_CKPT_DIR}/${policy_name}/checkpoint-${step}"
    if [[ ! -d "${ckpt}" ]]; then
      echo "missing checkpoint: ${ckpt}" >&2
      return 1
    fi
    local server_name="isaac-gr00t-robocasa-http-${label}-ckpt${step}-seed${SEED}"
    local server_out="${policy_root}/http_ckpt${step}"
    log "Policy ${label}: starting server checkpoint-${step}"
    CHECKPOINT_DIR="${ckpt}" \
    GPU_DEVICE="${gpu}" \
    PORT="${port}" \
    CONTAINER_NAME="${server_name}" \
    OUTPUT_DIR="${server_out}" \
    BOOTSTRAP_DEPS=1 \
    REPLACE=1 \
      bash "${REPO_DIR}/scripts/run_groot_robocasa_http_server.sh"
    wait_health "${port}"

    for task in "${TASKS[@]}"; do
      local bench_name="isaac-gr00t-robocasa-bench-${label}-ckpt${step}-${task}-seed${SEED}"
      local run_out="${policy_root}/ckpt${step}"
      log "Policy ${label}: eval ${task} checkpoint-${step}"
      ENV_NAME="${task}" \
      GPU_DEVICE="${gpu}" \
      MODEL_HOST=127.0.0.1 \
      PORT="${port}" \
      SEED="${SEED}" \
      N_EPISODES="${N_EPISODES}" \
      N_ACTION_STEPS="${N_ACTION_STEPS}" \
      MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
      VIDEO_FPS="${VIDEO_FPS}" \
      VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE}" \
      SCHEDULE_DIR="${SCHEDULE_DIR}" \
      CONTAINER_NAME="${bench_name}" \
      OUTPUT_DIR="${run_out}" \
      REPLACE=1 \
        bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
      docker wait "${bench_name}" >/dev/null
      docker logs "${bench_name}" > "${run_out}/docker_${task}.log" 2>&1 || true
      docker rm -f "${bench_name}" >/dev/null 2>&1 || true
    done

    log "Policy ${label}: stopping server checkpoint-${step}"
    docker rm -f "${server_name}" >/dev/null 2>&1 || true
  done
  log "Policy ${label}: done"
}

generate_schedules

pids=()
for idx in "${!POLICY_NAMES[@]}"; do
  (
    run_policy_pipeline "${idx}"
  ) > "${OUTPUT_ROOT}/${POLICY_LABELS[$idx]}.pipeline.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

log "All pipelines finished with status=${status}. Output: ${OUTPUT_ROOT}"
exit "${status}"
