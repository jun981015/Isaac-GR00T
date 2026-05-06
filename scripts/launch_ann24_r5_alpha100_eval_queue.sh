#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/local_outputs/robocasa_benchmark/ann24_r5_alpha100_50k_100ep_actionseed123_env8_20260504}"
LOG_ROOT="${LOG_ROOT:-${REPO}/local_outputs/robocasa_benchmark/ann24_r5_alpha100_eval_logs_20260504}"
RUN_TAG="${RUN_TAG:-20260504_r5a100_v2}"
PORT="${PORT:-8125}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

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

schedule_path_for_task() {
  local task="$1"
  case "${task}" in
    PnPCounterToSink|PnPCounterToStove|PnPMicrowaveToCounter)
      printf '%s\n' "${REPO}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/${task}_100eps.json"
      ;;
    CloseSingleDoor|CoffeePressButton|CoffeeSetupMug|OpenSingleDoor|PnPCounterToMicrowave|TurnOnMicrowave)
      printf '%s\n' "${REPO}/local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430/seed1/${task}_100eps.json"
      ;;
    *)
      printf '%s\n' "${REPO}/local_outputs/robocasa_benchmark/schedules/ann24_atomic_missing_100ep_envseed1_20260503/seed1/${task}_100eps.json"
      ;;
  esac
}

launch_eval() {
  local task="$1"
  local seed="$2"
  local schedule_path
  schedule_path="$(schedule_path_for_task "${task}")"
  test -s "${schedule_path}"

  local gpu=$(( (seed % 3) + 1 ))
  local cname="robocasa-bench-ann24-r5a100-${task}-as${seed}-${RUN_TAG}"

  if docker ps -a --format '{{.Names}}' | grep -qx "${cname}"; then
    log "container exists, skip launch task=${task} action_seed=${seed}: ${cname}"
    return 0
  fi

  log "launch task=${task} action_seed=${seed} gpu=${gpu} schedule=${schedule_path}"
  ENV_NAME="${task}" \
  GPU_DEVICE="${gpu}" \
  PORT="${PORT}" \
  SEED="${seed}" \
  N_EPISODES=100 \
  N_ENVS=8 \
  MAX_EPISODE_STEPS=1000 \
  N_ACTION_STEPS=16 \
  VIDEO_FPS=20 \
  VIDEO_SOURCE=obs \
  VIDEO_RENDER_SIZE=256 \
  VIDEO_STEPS_PER_RENDER=4 \
  CAMERA_WIDTH=256 \
  CAMERA_HEIGHT=256 \
  POLICY_IMAGE_SIZE=128 \
  WRITE_VIDEO=1 \
  STREAM_VIDEO=1 \
  SKIP_EXISTING=1 \
  BOOTSTRAP_DEPS=0 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MUJOCO_EGL_DEVICE_ID=0 \
  PYOPENGL_PLATFORM=egl \
  SCHEDULE_DIR="${REPO}/local_outputs/robocasa_benchmark/schedules" \
  SCHEDULE_PATH="${schedule_path}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/checkpoint-50000/action_seed_${seed}" \
  CONTAINER_NAME="${cname}" \
    bash "${REPO}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
      > "${LOG_ROOT}/${cname}.launch.log" 2>&1
}

wait_eval() {
  local task="$1"
  local seed="$2"
  wait_container "robocasa-bench-ann24-r5a100-${task}-as${seed}-${RUN_TAG}"
}

main() {
  cd "${REPO}"
  log "start ann24 r5 alpha100 checkpoint-50000 eval queue"
  log "output=${OUTPUT_ROOT}"
  log "policy=127.0.0.1:${PORT}"

  local tasks=(
    CloseDoubleDoor
    CloseDrawer
    CloseSingleDoor
    CoffeePressButton
    CoffeeServeMug
    CoffeeSetupMug
    OpenDoubleDoor
    OpenDrawer
    OpenSingleDoor
    PnPCabToCounter
    PnPCounterToCab
    PnPCounterToMicrowave
    PnPCounterToSink
    PnPCounterToStove
    PnPMicrowaveToCounter
    PnPSinkToCounter
    PnPStoveToCounter
    TurnOffMicrowave
    TurnOffSinkFaucet
    TurnOffStove
    TurnOnMicrowave
    TurnOnSinkFaucet
    TurnOnStove
    TurnSinkSpout
  )

  local batch=()
  for task in "${tasks[@]}"; do
    batch+=("${task}")
    if [[ "${#batch[@]}" -eq 2 ]]; then
      for t in "${batch[@]}"; do
        for seed in 1 2 3; do
          launch_eval "${t}" "${seed}"
        done
      done
      for t in "${batch[@]}"; do
        for seed in 1 2 3; do
          wait_eval "${t}" "${seed}"
        done
      done
      batch=()
    fi
  done

  for t in "${batch[@]}"; do
    for seed in 1 2 3; do
      launch_eval "${t}" "${seed}"
    done
  done
  for t in "${batch[@]}"; do
    for seed in 1 2 3; do
      wait_eval "${t}" "${seed}"
    done
  done

  log "complete ann24 r5 alpha100 checkpoint-50000 eval queue"
}

main "$@"
