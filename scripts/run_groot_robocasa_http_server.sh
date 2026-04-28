#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
DATA_CONFIG_DIR="${DATA_CONFIG_DIR:-/home/junhyeong/Value/robocasa}"
IMAGE_NAME="${IMAGE_NAME:-isaac-gr00t-robocasa:smoke}"
CONTAINER_NAME="${CONTAINER_NAME:-isaac-gr00t-robocasa-http-n15}"
GPU_DEVICE="${GPU_DEVICE:-0}"
PORT="${PORT:-8011}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?set CHECKPOINT_DIR to a GR00T checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_http_server/${CONTAINER_NAME}}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
BOOTSTRAP_DEPS="${BOOTSTRAP_DEPS:-0}"
REPLACE="${REPLACE:-0}"

mkdir -p "${OUTPUT_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if [[ "${REPLACE}" == "1" ]]; then
    docker rm -f "${CONTAINER_NAME}"
  else
    echo "Container ${CONTAINER_NAME} already exists. Set REPLACE=1 to remove it first." >&2
    exit 1
  fi
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --network=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/workspace/output/runtime_home \
  -e PYTHONUSERBASE=/workspace/output/runtime_home/.local \
  -e PYTHONPATH="/workspace/Isaac-GR00T:/workspace/robocasa_config:${PYTHONPATH:-}" \
  -v "${REPO_DIR}:/workspace/Isaac-GR00T" \
  -v "${DATA_CONFIG_DIR}:/workspace/robocasa_config:ro" \
  -v "${CHECKPOINT_DIR}:/workspace/model_checkpoint:ro" \
  -v "${OUTPUT_DIR}:/workspace/output" \
  -w /workspace/Isaac-GR00T \
  "${IMAGE_NAME}" \
  bash -lc "
    set -euo pipefail
    export PATH=\"/workspace/output/runtime_home/.local/bin:\$PATH\"
    if [[ '${BOOTSTRAP_DEPS}' == '1' ]]; then
      python -m pip install --user --no-cache-dir -q fastapi uvicorn json-numpy requests
    fi
    python scripts/inference_service.py \
    --server \
    --http-server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model_path /workspace/model_checkpoint \
    --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
    --embodiment_tag new_embodiment \
    --denoising_steps ${DENOISING_STEPS} \
    2>&1 | tee /workspace/output/server.log
  "

echo "Started ${CONTAINER_NAME} on GPU ${GPU_DEVICE}, port ${PORT}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Log: ${OUTPUT_DIR}/server.log"
