#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/junhyeong/Value/Isaac-GR00T}"
RUN_ID="${RUN_ID:-ann24_atomic_newanno_r5_ckpt50000_seed1_eval_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:-${REPO}/local_outputs/robocasa_awr_retrain/ann24_atomic_newanno_r5_alpha100_clip2_b64_50k_gpu2_20260514_161117/checkpoint-50000}"
OUT="${OUT:-${REPO}/local_outputs/robocasa_benchmark/${RUN_ID}}"
PORT="${PORT:-18176}"
GPU="${GPU:-3}"
SEEDS_CSV="${SEEDS_CSV:-1}"
N_ENVS="${N_ENVS:-4}"
N_EPISODES="${N_EPISODES:-100}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

mkdir -p "${OUT}/logs"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT}/logs/watcher.log"
}

log "wait checkpoint=${CKPT}"
until [[ -s "${CKPT}/model.safetensors.index.json" || -s "${CKPT}/config.json" ]]; do
  log "checkpoint not ready; sleep ${WAIT_SECONDS}s"
  sleep "${WAIT_SECONDS}"
done
log "checkpoint ready"

SERVER_SESSION="server_${RUN_ID}"
if ! tmux has-session -t "${SERVER_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SERVER_SESSION}" \
    "bash -lc 'cd ${REPO}; \
      export CUDA_VISIBLE_DEVICES=${GPU}; \
      export PYTHONPATH=${REPO}:/home/junhyeong/Value/robocasa:\${PYTHONPATH:-}; \
      export NO_ALBUMENTATIONS_UPDATE=1; \
      conda run --no-capture-output -n gr00t python scripts/inference_service.py \
        --server \
        --host 127.0.0.1 \
        --port ${PORT} \
        --model_path ${CKPT} \
        --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
        --embodiment_tag new_embodiment \
        --denoising_steps 4 \
        > ${OUT}/logs/server.log 2>&1'"
fi
log "server session=${SERVER_SESSION} port=${PORT}"

for _ in $(seq 1 180); do
  if ss -ltn | grep -q ":${PORT} "; then
    log "server ready"
    break
  fi
  sleep 10
done

if ! ss -ltn | grep -q ":${PORT} "; then
  log "server did not open port ${PORT}"
  exit 1
fi

cd "${REPO}"
log "start eval output=${OUT} seeds=${SEEDS_CSV} gpu=${GPU} n_envs=${N_ENVS}"
PORT="${PORT}" \
MODEL_HOST=127.0.0.1 \
OUTPUT_ROOT="${OUT}" \
SEEDS_CSV="${SEEDS_CSV}" \
GPU_LIST_CSV="${GPU}" \
N_EPISODES="${N_EPISODES}" \
N_ENVS="${N_ENVS}" \
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
CONDA_ENV=robocasa-eval \
bash scripts/launch_ann24_weight1_alpha0_conda_eval_queue.sh \
  > "${OUT}/logs/eval_queue.log" 2>&1

log "eval complete"
