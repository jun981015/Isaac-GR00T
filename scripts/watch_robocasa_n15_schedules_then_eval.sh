#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
SCHEDULE_DIR="${SCHEDULE_DIR:?set SCHEDULE_DIR}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-512}"

containers=(
  isaac-gr00t-robocasa-bench-schedule-PnPCounterToSink-seed${SEED}-${N_EPISODES}ep
  isaac-gr00t-robocasa-bench-schedule-PnPCounterToStove-seed${SEED}-${N_EPISODES}ep
  isaac-gr00t-robocasa-bench-schedule-PnPMicrowaveToCounter-seed${SEED}-${N_EPISODES}ep
)

for container in "${containers[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting ${container}"
  status="$(docker wait "${container}")"
  docker logs "${container}" > "${OUTPUT_ROOT}/${container}.log" 2>&1 || true
  docker rm -f "${container}" >/dev/null 2>&1 || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] done ${container} status=${status}"
  if [[ "${status}" != "0" ]]; then
    echo "Schedule container failed: ${container}" >&2
    exit 1
  fi
done

OUTPUT_ROOT="${OUTPUT_ROOT}" \
SCHEDULE_DIR="${SCHEDULE_DIR}" \
SEED="${SEED}" \
N_EPISODES="${N_EPISODES}" \
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE}" \
  bash "${REPO_DIR}/scripts/run_robocasa_n15_eval_policy_grid.sh"
