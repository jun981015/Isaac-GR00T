#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
RUN_ID="${RUN_ID:-n15_20k_100ep_actionseed123_env8_20260429_1045}"
COMPOSITE_RUN_ID="${COMPOSITE_RUN_ID:-composite_awr2_envseed1_actionseed123_env8_20260430_0915}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/manual_eval_launch_logs}"
RUN_TAG="${RUN_TAG:-pm16_stove_all_critical_prepare_s3_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_TAG}}"

PNP_SCHEDULE_DIR="${PNP_SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep}"
COMPOSITE_SCHEDULE_DIR="${COMPOSITE_SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430}"

PM16_OUTPUT_ROOT="${PM16_OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}/awr_pm16_alpha50_20k/checkpoint-20000}"
CRITICAL_OUTPUT_ROOT="${CRITICAL_OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${COMPOSITE_RUN_ID}/awr_critical8_groupalpha_clip18_50k/checkpoint-50000}"

PM16_PORT="${PM16_PORT:-8102}"
CRITICAL_PORT="${CRITICAL_PORT:-8092}"
PNP_EVAL_GPU="${PNP_EVAL_GPU:-3}"
COMPOSITE_EVAL_GPU="${COMPOSITE_EVAL_GPU:-2}"
N_ENVS="${N_ENVS:-8}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_DIR}/launcher.log"
}

run_eval() {
  local label="$1"
  local task="$2"
  local action_seed="$3"
  local port="$4"
  local gpu="$5"
  local max_steps="$6"
  local output_root="$7"
  local schedule_dir="$8"
  local schedule_path="${schedule_dir}/seed1/${task}_100eps.json"
  local cname="robocasa-bench-manual-${label}-${task}-as${action_seed}-${RUN_TAG}"
  local out_dir="${output_root}/action_seed_${action_seed}"

  if [[ ! -s "${schedule_path}" ]]; then
    log "missing schedule: ${schedule_path}"
    return 1
  fi

  log "launch label=${label} task=${task} action_seed=${action_seed} gpu=${gpu} port=${port}"
  log "output=${out_dir}"

  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${out_dir}" \
  GPU_DEVICE="${gpu}" \
  MODEL_HOST=127.0.0.1 \
  PORT="${port}" \
  ENV_NAME="${task}" \
  SEED="${action_seed}" \
  N_EPISODES=100 \
  N_ENVS="${N_ENVS}" \
  N_ACTION_STEPS=16 \
  MAX_EPISODE_STEPS="${max_steps}" \
  VIDEO_FPS=20 \
  VIDEO_SOURCE=obs \
  VIDEO_RENDER_SIZE=0 \
  VIDEO_SCALE=1 \
  VIDEO_STEPS_PER_RENDER=4 \
  CAMERA_WIDTH=256 \
  CAMERA_HEIGHT=256 \
  POLICY_IMAGE_SIZE=128 \
  STREAM_VIDEO=1 \
  WRITE_VIDEO=1 \
  SKIP_EXISTING=1 \
  REGENERATE_SCHEDULE=0 \
  BOOTSTRAP_DEPS=0 \
  REPLACE=0 \
  SCHEDULE_DIR="${schedule_dir}" \
  SCHEDULE_PATH="${schedule_path}" \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_zmq_parallel_eval_container.sh" \
      > "${LOG_DIR}/${cname}.launch.log" 2>&1

  docker wait "${cname}" > "${LOG_DIR}/${cname}.exit"
  docker logs "${cname}" > "${LOG_DIR}/${cname}.docker.log" 2>&1 || true
  local code
  code="$(cat "${LOG_DIR}/${cname}.exit")"
  log "done label=${label} task=${task} action_seed=${action_seed} code=${code}"
  [[ "${code}" == "0" ]]
}

main() {
  cd "${REPO_DIR}"
  log "run_tag=${RUN_TAG}"
  log "log_dir=${LOG_DIR}"
  log "n_envs=${N_ENVS}"

  run_eval pm16alpha50-stove PnPCounterToStove 1 "${PM16_PORT}" "${PNP_EVAL_GPU}" 800 "${PM16_OUTPUT_ROOT}" "${PNP_SCHEDULE_DIR}"
  run_eval pm16alpha50-stove PnPCounterToStove 2 "${PM16_PORT}" "${PNP_EVAL_GPU}" 800 "${PM16_OUTPUT_ROOT}" "${PNP_SCHEDULE_DIR}"
  run_eval pm16alpha50-stove PnPCounterToStove 3 "${PM16_PORT}" "${PNP_EVAL_GPU}" 800 "${PM16_OUTPUT_ROOT}" "${PNP_SCHEDULE_DIR}"
  run_eval critical8-prepare PrepareCoffee 3 "${CRITICAL_PORT}" "${COMPOSITE_EVAL_GPU}" 1000 "${CRITICAL_OUTPUT_ROOT}" "${COMPOSITE_SCHEDULE_DIR}"

  log "complete"
}

main "$@"
