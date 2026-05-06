#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha100_b64_50000step_gpu3_20260502_224019/checkpoint-50000}"
DATA_CONFIG_DIR="${DATA_CONFIG_DIR:-/home/junhyeong/Value/robocasa}"
CONDA_TRAIN_ENV="${CONDA_TRAIN_ENV:-gr00t-train}"
CONDA_EVAL_ENV="${CONDA_EVAL_ENV:-robocasa-eval}"
SERVER_GPU="${SERVER_GPU:-1}"
PORT="${PORT:-18141}"
N_ENVS="${N_ENVS:-8}"
N_EPISODES="${N_EPISODES:-100}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-ann24_r1_alpha100_50k_seed123_nenv${N_ENVS}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/${RUN_ID}}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${REPO_DIR}/local_outputs/robocasa_zmq_server}"

export PATH="/home/junhyeong/miniconda3/bin:${PATH}"

mkdir -p "${OUTPUT_ROOT}" "${SERVER_LOG_DIR}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

server_session="${RUN_ID}_server_gpu${SERVER_GPU}"
server_log="${SERVER_LOG_DIR}/${RUN_ID}_server.log"

tmux new-session -d -s "${server_session}" \
  "cd '${REPO_DIR}' && \
   export PATH=\"/home/junhyeong/miniconda3/bin:\${PATH}\" && \
   export PYTHONPATH='${REPO_DIR}:${DATA_CONFIG_DIR}:\${PYTHONPATH:-}' && \
   export CUDA_VISIBLE_DEVICES='${SERVER_GPU}' && \
   export TRANSFORMERS_CACHE='/home/junhyeong/.cache/huggingface' && \
   conda run --no-capture-output -n '${CONDA_TRAIN_ENV}' python scripts/inference_service.py \
     --server \
     --host 0.0.0.0 \
     --port '${PORT}' \
     --model_path '${CHECKPOINT_DIR}' \
     --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
     --embodiment_tag new_embodiment \
     --denoising_steps 4 \
     2>&1 | /usr/bin/tee '${server_log}'"

echo "[server_launched] session=${server_session} gpu=${SERVER_GPU} port=${PORT} log=${server_log}" | tee -a "${OUTPUT_ROOT}/launcher.log"

for pair in 1:1 2:2 3:3; do
  seed="${pair%%:*}"
  gpu="${pair##*:}"
  worker_session="${RUN_ID}_seed${seed}_gpu${gpu}"
  tmux new-session -d -s "${worker_session}" \
    "cd '${REPO_DIR}' && \
     GPU_ID='${gpu}' \
     SEED='${seed}' \
     PORT='${PORT}' \
     N_ENVS='${N_ENVS}' \
     N_EPISODES='${N_EPISODES}' \
     CONDA_ENV='${CONDA_EVAL_ENV}' \
     OUTPUT_ROOT='${OUTPUT_ROOT}' \
     bash scripts/run_ann24_existing_schedule_eval_gpu.sh \
     2>&1 | tee '${OUTPUT_ROOT}/seed${seed}_gpu${gpu}.tmux.log'"
  echo "[worker_launched] session=${worker_session} gpu=${gpu} seed=${seed}" | tee -a "${OUTPUT_ROOT}/launcher.log"
done

echo "[launch_done] output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/launcher.log"
