#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
RUN_ID="${RUN_ID:-n15_20k_100ep_actionseed123_env8_20260429_1045}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_sink_stove_gpu23_logs}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

TASKS=(PnPCounterToSink PnPCounterToStove)
ACTION_SEEDS=(1 2 3)

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_DIR}/launcher.log"
}

wait_for_server_ready() {
  local cname="$1"
  local timeout_sec="${2:-900}"
  local start
  start="$(date +%s)"
  while true; do
    if docker logs "${cname}" 2>&1 | grep -q "Server is ready"; then
      log "server ready ${cname}"
      return 0
    fi
    if ! docker inspect --format '{{.State.Running}}' "${cname}" 2>/dev/null | grep -q true; then
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      log "server exited before ready ${cname}"
      return 1
    fi
    if (( $(date +%s) - start > timeout_sec )); then
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      log "server ready timeout ${cname}"
      return 1
    fi
    sleep 5
  done
}

start_server() {
  local policy="$1"
  local checkpoint="$2"
  local gpu="$3"
  local port="$4"
  local cname="robocasa-zmq-sinkstove-${policy}-gpu${gpu}"

  log "start server policy=${policy} gpu=${gpu} port=${port}"
  CHECKPOINT_DIR="${checkpoint}" \
  GPU_DEVICE="${gpu}" \
  PORT="${port}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${policy}/_server_sinkstove_gpu${gpu}" \
  BOOTSTRAP_DEPS=0 \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_groot_robocasa_zmq_server.sh" \
      > "${LOG_DIR}/${cname}.launch.log" 2>&1
  wait_for_server_ready "${cname}"
}

run_policy() {
  local policy="$1"
  local checkpoint_label="$2"
  local gpu="$3"
  local port="$4"

  for task in "${TASKS[@]}"; do
    local schedule_path="${SCHEDULE_DIR}/seed1/${task}_${N_EPISODES}eps.json"
    test -s "${schedule_path}"
    for action_seed in "${ACTION_SEEDS[@]}"; do
      local cname="robocasa-bench-sinkstove-${policy}-as${action_seed}-${task}-g${gpu}"
      local out_dir="${OUTPUT_ROOT}/${policy}/${checkpoint_label}/action_seed_${action_seed}"

      log "launch policy=${policy} task=${task} action_seed=${action_seed} gpu=${gpu}"
      CONTAINER_NAME="${cname}" \
      OUTPUT_DIR="${out_dir}" \
      GPU_DEVICE="${gpu}" \
      MODEL_HOST=127.0.0.1 \
      PORT="${port}" \
      ENV_NAME="${task}" \
      SEED="${action_seed}" \
      N_EPISODES="${N_EPISODES}" \
      N_ENVS="${N_ENVS}" \
      N_ACTION_STEPS=16 \
      MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
      VIDEO_FPS=20 \
      VIDEO_SOURCE=obs \
      VIDEO_RENDER_SIZE=0 \
      VIDEO_SCALE=1 \
      VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER}" \
      CAMERA_WIDTH=256 \
      CAMERA_HEIGHT=256 \
      POLICY_IMAGE_SIZE=128 \
      STREAM_VIDEO=1 \
      WRITE_VIDEO=1 \
      SKIP_EXISTING=1 \
      REGENERATE_SCHEDULE=0 \
      BOOTSTRAP_DEPS=0 \
      REPLACE=1 \
      SCHEDULE_DIR="${SCHEDULE_DIR}" \
      SCHEDULE_PATH="${schedule_path}" \
        bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
          > "${LOG_DIR}/${cname}.launch.log" 2>&1

      docker wait "${cname}" > "${LOG_DIR}/${cname}.exit"
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      local code
      code="$(cat "${LOG_DIR}/${cname}.exit")"
      docker rm "${cname}" >/dev/null 2>&1 || true
      log "done policy=${policy} task=${task} action_seed=${action_seed} code=${code}"
      [[ "${code}" == "0" ]]
    done
  done
}

main() {
  cd "${REPO_DIR}"
  log "run_id=${RUN_ID}"
  log "output_root=${OUTPUT_ROOT}"
  log "schedule_dir=${SCHEDULE_DIR}"

  start_server \
    baseline_noawr_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000" \
    2 \
    8102
  run_policy baseline_noawr_20k checkpoint-20000 2 8102

  start_server \
    awr_pm8_alpha25_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000" \
    3 \
    8103
  run_policy awr_pm8_alpha25_20k checkpoint-20000 3 8103

  start_server \
    awr_pm16_alpha50_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000" \
    2 \
    8104
  run_policy awr_pm16_alpha50_20k checkpoint-20000 2 8104

  log "all sink/stove done"
}

main "$@"
