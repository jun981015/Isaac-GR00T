#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/junhyeong/Value/Isaac-GR00T}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO}/local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/local_outputs/robocasa_benchmark/atomic8_envseed1_actionseed123_env8_20260501}"
LOG_ROOT="${LOG_ROOT:-${REPO}/local_outputs/robocasa_benchmark/atomic8_queue_logs_20260501}"
RUN_TAG="${RUN_TAG:-20260501}"

mkdir -p "${LOG_ROOT}" "${SCHEDULE_DIR}/seed1"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_ROOT}/queue.log"
}

wait_container() {
  local cname="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${cname}"; then
    log "wait ${cname}"
    docker wait "${cname}" > "${LOG_ROOT}/${cname}.exit" || true
    docker logs "${cname}" > "${LOG_ROOT}/${cname}.docker.log" 2>&1 || true
    log "done ${cname} exit=$(cat "${LOG_ROOT}/${cname}.exit" 2>/dev/null || echo unknown)"
  else
    log "skip wait missing container ${cname}"
  fi
}

wait_eval_batch() {
  local task_slug="$1"
  wait_container "robocasa-bench-atomic-${task_slug}-baseline-as1-${RUN_TAG}"
  wait_container "robocasa-bench-atomic-${task_slug}-baseline-as2-${RUN_TAG}"
  wait_container "robocasa-bench-atomic-${task_slug}-baseline-as3-${RUN_TAG}"
  wait_container "robocasa-bench-atomic-${task_slug}-critical8-as1-${RUN_TAG}"
  wait_container "robocasa-bench-atomic-${task_slug}-critical8-as2-${RUN_TAG}"
  wait_container "robocasa-bench-atomic-${task_slug}-critical8-as3-${RUN_TAG}"
}

ensure_schedule() {
  local task="$1"
  local schedule_path="${SCHEDULE_DIR}/seed1/${task}_100eps.json"
  if [[ -s "${schedule_path}" ]]; then
    log "schedule exists ${task}: ${schedule_path}"
    return 0
  fi

  local cname="robocasa-schedule-critical8-${task}-100ep-envseed1-${RUN_TAG}"
  log "launch schedule ${task}: ${cname}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${SCHEDULE_GPU_DEVICE:-3}" \
  SEED=1 \
  N_EPISODES=100 \
  N_ENVS=1 \
  MAX_EPISODE_STEPS=1000 \
  WRITE_VIDEO=0 \
  GENERATE_SCHEDULE_ONLY=1 \
  REGENERATE_SCHEDULE=0 \
  BOOTSTRAP_DEPS=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MUJOCO_EGL_DEVICE_ID=0 \
  PYOPENGL_PLATFORM=egl \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  SCHEDULE_PATH="${schedule_path}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${LOG_ROOT}/schedule_${task}" \
    bash "${REPO}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
      > "${LOG_ROOT}/${cname}.launch.log" 2>&1

  wait_container "${cname}"
  test -s "${schedule_path}"
}

launch_eval_batch() {
  local task="$1"
  local task_slug="$2"
  local schedule_path="${SCHEDULE_DIR}/seed1/${task}_100eps.json"
  local common_env=(
    N_EPISODES=100
    N_ENVS=8
    MAX_EPISODE_STEPS=1000
    N_ACTION_STEPS=16
    VIDEO_FPS=20
    VIDEO_SOURCE=obs
    VIDEO_RENDER_SIZE=256
    VIDEO_STEPS_PER_RENDER=4
    CAMERA_WIDTH=256
    CAMERA_HEIGHT=256
    POLICY_IMAGE_SIZE=128
    WRITE_VIDEO=1
    STREAM_VIDEO=1
    SKIP_EXISTING=1
    BOOTSTRAP_DEPS=0
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    MUJOCO_EGL_DEVICE_ID=0
    PYOPENGL_PLATFORM=egl
    ENV_NAME="${task}"
    SCHEDULE_DIR="${SCHEDULE_DIR}"
    SCHEDULE_PATH="${schedule_path}"
  )

  log "launch eval batch task=${task}"
  for seed in 1 2 3; do
    local gpu=$(( (seed % 3) + 1 ))
    env "${common_env[@]}" \
      CONTAINER_NAME="robocasa-bench-atomic-${task_slug}-baseline-as${seed}-${RUN_TAG}" \
      GPU_DEVICE="${gpu}" \
      PORT=8091 \
      SEED="${seed}" \
      OUTPUT_DIR="${OUTPUT_ROOT}/baseline_8task_b50k/checkpoint-50000/action_seed_${seed}" \
      bash "${REPO}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
        > "${LOG_ROOT}/launch_${task_slug}_baseline_as${seed}.log" 2>&1
  done

  for seed in 1 2 3; do
    local gpu=$(( ((seed + 1) % 3) + 1 ))
    env "${common_env[@]}" \
      CONTAINER_NAME="robocasa-bench-atomic-${task_slug}-critical8-as${seed}-${RUN_TAG}" \
      GPU_DEVICE="${gpu}" \
      PORT=8092 \
      SEED="${seed}" \
      OUTPUT_DIR="${OUTPUT_ROOT}/awr_critical8_groupalpha_clip18_50k/checkpoint-50000/action_seed_${seed}" \
      bash "${REPO}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
        > "${LOG_ROOT}/launch_${task_slug}_critical8_as${seed}.log" 2>&1
  done
}

main() {
  cd "${REPO}"

  log "wait current CoffeeSetupMug batch before queueing more evals"
  wait_eval_batch "setupmug"

  local tasks=(
    "CoffeePressButton:pressbutton"
    "OpenSingleDoor:opensingle"
    "CloseSingleDoor:closesingle"
    "PnPCounterToMicrowave:pnpctomicrowave"
    "TurnOnMicrowave:turnonmicrowave"
  )

  for item in "${tasks[@]}"; do
    local task="${item%%:*}"
    local slug="${item##*:}"
    ensure_schedule "${task}"
    launch_eval_batch "${task}" "${slug}"
    wait_eval_batch "${slug}"
  done

  log "all queued atomic eval batches complete"
}

main "$@"
