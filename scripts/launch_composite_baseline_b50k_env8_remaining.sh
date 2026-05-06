#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/_launcher_logs}"
GPU_DEVICE="${GPU_DEVICE:-1}"
PORT="${PORT:-8091}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

TASKS=(PrepareCoffee MicrowaveThawing)
ACTION_SEEDS=(1 2 3)

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_DIR}/baseline_env8_remaining.log"
}

main() {
  cd "${REPO_DIR}"
  for action_seed in "${ACTION_SEEDS[@]}"; do
    for task in "${TASKS[@]}"; do
      local_schedule="${SCHEDULE_DIR}/seed1/${task}_${N_EPISODES}eps.json"
      test -s "${local_schedule}"
      cname="robocasa-bench-b50k-${task}-envseed1-actionseed${action_seed}-100ep-env8"
      out_dir="${OUTPUT_ROOT}/baseline_8task_b50k/checkpoint-50000/action_seed_${action_seed}"
      log "launch baseline task=${task} action_seed=${action_seed} n_envs=${N_ENVS}"
      CONTAINER_NAME="${cname}" \
      OUTPUT_DIR="${out_dir}" \
      GPU_DEVICE="${GPU_DEVICE}" \
      MODEL_HOST=127.0.0.1 \
      PORT="${PORT}" \
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
      SCHEDULE_PATH="${local_schedule}" \
        bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
          > "${LOG_DIR}/${cname}.launch.log" 2>&1
      docker wait "${cname}" > "${LOG_DIR}/${cname}.exit"
      docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
      code="$(cat "${LOG_DIR}/${cname}.exit")"
      docker rm "${cname}" >/dev/null 2>&1 || true
      log "done baseline task=${task} action_seed=${action_seed} code=${code}"
      [[ "${code}" == "0" ]]
    done
  done
  log "baseline remaining complete"
}

main "$@"
