#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedule_logs/critical8_remaining_6task_100ep_envseed1_20260430}"
GPU_DEVICE="${GPU_DEVICE:-1}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"

TASKS=(
  CoffeeSetupMug
  CoffeePressButton
  OpenSingleDoor
  CloseSingleDoor
  PnPCounterToMicrowave
  TurnOnMicrowave
)

mkdir -p "${SCHEDULE_DIR}/seed${SEED}" "${OUTPUT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${OUTPUT_ROOT}/launcher.log"
}

wait_existing_container() {
  local cname="$1"
  local task="$2"
  if docker ps -a --format '{{.Names}}' | grep -qx "${cname}"; then
    log "wait existing ${task}: ${cname}"
    docker wait "${cname}" > "${OUTPUT_ROOT}/${task}.exit" || true
    docker logs "${cname}" > "${OUTPUT_ROOT}/${task}.docker.log" 2>&1 || true
    docker rm "${cname}" >/dev/null 2>&1 || true
  fi
}

run_task() {
  local task="$1"
  local schedule_path="${SCHEDULE_DIR}/seed${SEED}/${task}_${N_EPISODES}eps.json"
  local cname="robocasa-schedule-critical8-${task}-100ep-envseed${SEED}"

  if [[ -s "${schedule_path}" ]]; then
    log "skip existing schedule ${task}: ${schedule_path}"
    return 0
  fi

  wait_existing_container "${cname}" "${task}"
  if [[ -s "${schedule_path}" ]]; then
    log "completed by existing container ${task}: ${schedule_path}"
    return 0
  fi

  log "start schedule ${task}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${GPU_DEVICE}" \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ENVS=1 \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  WRITE_VIDEO=0 \
  GENERATE_SCHEDULE_ONLY=1 \
  REGENERATE_SCHEDULE=0 \
  BOOTSTRAP_DEPS=0 \
  REPLACE=1 \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  SCHEDULE_PATH="${schedule_path}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${task}" \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
      > "${OUTPUT_ROOT}/${task}.launch.log" 2>&1

  docker wait "${cname}" > "${OUTPUT_ROOT}/${task}.exit"
  docker logs "${cname}" > "${OUTPUT_ROOT}/${task}.docker.log" 2>&1 || true
  local code
  code="$(cat "${OUTPUT_ROOT}/${task}.exit")"
  docker rm "${cname}" >/dev/null 2>&1 || true
  log "done schedule ${task} code=${code}"
  [[ "${code}" == "0" ]]
  test -s "${schedule_path}"
}

main() {
  cd "${REPO_DIR}"
  log "schedule_dir=${SCHEDULE_DIR}"
  for task in "${TASKS[@]}"; do
    run_task "${task}"
  done
  log "all 6 schedules complete"
}

main "$@"
