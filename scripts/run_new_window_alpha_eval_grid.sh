#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"
VIDEO_SOURCE="${VIDEO_SOURCE:-obs}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"

labels=(pm8_alpha25 pm16_alpha50)
gpus=(1 2)
ports=(8041 8042)
tasks=(PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter)

launch_bench() {
  local label="$1"
  local gpu="$2"
  local port="$3"
  local task="$4"
  local cname="isaac-gr00t-robocasa-bench-${label}-ckpt20000-${task}-newwindow"
  local output_dir="${OUTPUT_ROOT}/${label}/ckpt20000"
  local schedule_path="${SCHEDULE_DIR}/seed${SEED}/${task}_${N_EPISODES}eps.json"

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
  VIDEO_SOURCE="${VIDEO_SOURCE}" \
  VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER}" \
  STREAM_VIDEO=1 \
  SKIP_EXISTING=1 \
  SCHEDULE_DIR="${SCHEDULE_DIR}" \
  SCHEDULE_PATH="${schedule_path}" \
  CONTAINER_NAME="${cname}" \
  OUTPUT_DIR="${output_dir}" \
  REPLACE=1 \
    bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_container.sh"
}

wait_bench() {
  local label="$1"
  local task="$2"
  local cname="isaac-gr00t-robocasa-bench-${label}-ckpt20000-${task}-newwindow"
  local output_dir="${OUTPUT_ROOT}/${label}/ckpt20000"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] wait ${cname}"
  local status
  status="$(docker wait "${cname}")"
  docker logs "${cname}" > "${output_dir}/docker_${task}.log" 2>&1 || true
  docker rm -f "${cname}" >/dev/null 2>&1 || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${cname} status=${status}"
  [[ "${status}" == "0" ]]
}

for task in "${tasks[@]}"; do
  for i in "${!labels[@]}"; do
    launch_bench "${labels[$i]}" "${gpus[$i]}" "${ports[$i]}" "${task}" &
  done
  wait

  for label in "${labels[@]}"; do
    wait_bench "${label}" "${task}" &
  done
  wait
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] new-window alpha eval complete"
