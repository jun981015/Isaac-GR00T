#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
RUN_ID="${RUN_ID:-n15_20k_100ep_actionseed123_env8_$(date +%Y%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_launcher_logs}"

TASKS=(PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter)
ACTION_SEEDS=(1 2 3)
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "${SCHEDULE_DIR}/seed1"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

generate_schedule_for_task() {
  local task="$1"
  local gpu="$2"
  local schedule_path="${SCHEDULE_DIR}/seed1/${task}_${N_EPISODES}eps.json"
  local cname="isaac-gr00t-robocasa-schedule-${RUN_ID}-${task}"
  local out_dir="${OUTPUT_ROOT}/_schedule_gen/${task}"

  if [[ -s "${schedule_path}" ]]; then
    local count
    count="$(python - "${schedule_path}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data.get("episodes", [])))
PY
)"
    if [[ "${count}" == "${N_EPISODES}" ]]; then
      log "schedule exists: ${schedule_path}"
      return 0
    fi
  fi

  log "generate schedule task=${task} gpu=${gpu}"
  ENV_NAME="${task}" \
  N_EPISODES="${N_EPISODES}" \
  SEED=1 \
  N_ENVS=1 \
  GPU_DEVICE="${gpu}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${out_dir}" \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  SCHEDULE_PATH="${schedule_path}" \
  GENERATE_SCHEDULE_ONLY=1 \
  REGENERATE_SCHEDULE=1 \
  BOOTSTRAP_DEPS=0 \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
      > "${LOG_DIR}/${cname}.launch.log" 2>&1

  docker wait "${cname}" > "${LOG_DIR}/${cname}.exit"
  docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
  local code
  code="$(cat "${LOG_DIR}/${cname}.exit")"
  docker rm "${cname}" >/dev/null || true
  if [[ "${code}" != "0" ]]; then
    log "schedule failed task=${task} code=${code}"
    return 1
  fi
}

verify_schedules() {
  for task in "${TASKS[@]}"; do
    local schedule_path="${SCHEDULE_DIR}/seed1/${task}_${N_EPISODES}eps.json"
    python - "${schedule_path}" "${N_EPISODES}" <<'PY'
import json, sys
path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
count = len(data.get("episodes", []))
if count != expected:
    raise SystemExit(f"{path}: expected {expected}, got {count}")
print(f"{path}: {count} episodes")
PY
  done
}

wait_for_server_ready() {
  local cname="$1"
  local timeout_sec="${2:-900}"
  local start
  start="$(date +%s)"
  while true; do
    if docker logs "${cname}" 2>&1 | grep -q "Server is ready"; then
      return 0
    fi
    if ! docker inspect --format '{{.State.Running}}' "${cname}" 2>/dev/null | grep -q true; then
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      log "server exited before ready: ${cname}"
      return 1
    fi
    if (( $(date +%s) - start > timeout_sec )); then
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      log "server ready timeout: ${cname}"
      return 1
    fi
    sleep 5
  done
}

run_policy_group() {
  local policy_name="$1"
  local checkpoint_dir="$2"
  local gpu="$3"
  local port="$4"
  local server_name="isaac-gr00t-robocasa-zmq-${RUN_ID}-${policy_name}"

  log "start server policy=${policy_name} gpu=${gpu} port=${port}"
  CHECKPOINT_DIR="${checkpoint_dir}" \
  GPU_DEVICE="${gpu}" \
  PORT="${port}" \
  CONTAINER_NAME="${server_name}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${policy_name}/_server" \
  BOOTSTRAP_DEPS=0 \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_groot_robocasa_zmq_server.sh" \
      > "${LOG_DIR}/${server_name}.launch.log" 2>&1

  wait_for_server_ready "${server_name}"

  for action_seed in "${ACTION_SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      local bench_name="isaac-gr00t-robocasa-bench-${RUN_ID}-${policy_name}-as${action_seed}-${task}"
      local out_dir="${OUTPUT_ROOT}/${policy_name}/checkpoint-20000/action_seed_${action_seed}"
      local schedule_path="${SCHEDULE_DIR}/seed1/${task}_${N_EPISODES}eps.json"

      log "eval policy=${policy_name} action_seed=${action_seed} task=${task}"
      CONTAINER_NAME="${bench_name}" \
      OUTPUT_DIR="${out_dir}" \
      GPU_DEVICE="${gpu}" \
      MODEL_HOST=127.0.0.1 \
      PORT="${port}" \
      ENV_NAME="${task}" \
      SEED="${action_seed}" \
      N_EPISODES="${N_EPISODES}" \
      N_ENVS="${N_ENVS}" \
      MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
      VIDEO_FPS="${VIDEO_FPS}" \
      VIDEO_SCALE=2 \
      VIDEO_RENDER_SIZE=256 \
      VIDEO_SOURCE=obs \
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
          > "${LOG_DIR}/${bench_name}.launch.log" 2>&1

      docker wait "${bench_name}" > "${LOG_DIR}/${bench_name}.exit"
      docker logs "${bench_name}" > "${LOG_DIR}/${bench_name}.docker.log" 2>&1 || true
      local code
      code="$(cat "${LOG_DIR}/${bench_name}.exit")"
      docker rm "${bench_name}" >/dev/null || true
      if [[ "${code}" != "0" ]]; then
        log "eval failed policy=${policy_name} action_seed=${action_seed} task=${task} code=${code}"
        docker stop "${server_name}" >/dev/null || true
        return 1
      fi
    done
  done

  docker logs "${server_name}" > "${LOG_DIR}/${server_name}.docker.log" 2>&1 || true
  docker stop "${server_name}" >/dev/null || true
  log "done policy=${policy_name}"
}

main() {
  cd "${REPO_DIR}"
  log "run_id=${RUN_ID}"
  log "output_root=${OUTPUT_ROOT}"
  log "schedule_dir=${SCHEDULE_DIR}"

  generate_schedule_for_task PnPCounterToSink 1 &
  generate_schedule_for_task PnPCounterToStove 2 &
  generate_schedule_for_task PnPMicrowaveToCounter 3 &
  wait
  verify_schedules

  run_policy_group \
    baseline_noawr_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000" \
    1 \
    8081 &
  run_policy_group \
    awr_pm8_alpha25_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000" \
    2 \
    8082 &
  run_policy_group \
    awr_pm16_alpha50_20k \
    "${REPO_DIR}/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000" \
    3 \
    8083 &
  wait
  log "all done"
}

main "$@"
