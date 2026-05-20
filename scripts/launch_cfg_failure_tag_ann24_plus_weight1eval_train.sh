#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
DATA_ROOT="${DATA_ROOT:-/home/junhyeong/data/robocasa_lerobot_flat}"
EVAL_TRAJ_DATASET="${EVAL_TRAJ_DATASET:-${REPO_DIR}/local_outputs/lerobot_datasets/robocasa_weight1_seed1_128_rollout_lerobot_train_ready_20260519}"
GPU="${GPU:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-10000}"
DROPOUT="${DROPOUT:-0.2}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_cfg_retrain/ann24_plus_weight1eval_failuretag_drop${DROPOUT//./}_b${BATCH_SIZE}_${MAX_STEPS}step_gpu${GPU}_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}}"
WANDB_PROJECT="${WANDB_PROJECT:-robocasa-cfg-finetune}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME="${WANDB_NAME:-cfg_failuretag_ann24_plus_weight1eval_drop${DROPOUT}_b${BATCH_SIZE}_${MAX_STEPS}step_gpu${GPU}_${TIMESTAMP}}"
DETACH="${DETACH:-1}"

TASKS=(
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

DATASETS=()
for task in "${TASKS[@]}"; do
  DATASETS+=("${DATA_ROOT}/${task}")
done
DATASETS+=("${EVAL_TRAJ_DATASET}")

mkdir -p "${LOG_DIR}" "${WANDB_DIR}"
cd "${REPO_DIR}"

echo "output_dir=${OUTPUT_DIR}"
echo "log=${LOG_DIR}/train.log"
echo "gpu=${GPU}"
echo "batch_size=${BATCH_SIZE}"
echo "max_steps=${MAX_STEPS}"
echo "datasets=${#DATASETS[@]}"
echo "wandb_dir=${WANDB_DIR}"
echo "wandb_project=${WANDB_PROJECT}"
echo "wandb_mode=${WANDB_MODE}"
echo "wandb_name=${WANDB_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="/home/junhyeong/Value/robocasa:${PYTHONPATH:-}"
export WANDB_DIR
export WANDB_PROJECT
export WANDB_MODE
export WANDB_NAME
export PYTHONUNBUFFERED=1

CMD=(
  conda run -n gr00t python scripts/robocasa_cfg_finetune.py
  --dataset-path "${DATASETS[@]}"
  --output-dir "${OUTPUT_DIR}"
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --outcome-prefix
  --outcome-prompt-style failure_tag
  --outcome-prefix-dropout-prob "${DROPOUT}"
  --expert-outcome success
  --tune-projector
  --tune-diffusion-model
  --no-tune-llm
  --no-tune-visual
  --report-to wandb
)

if [[ "${DETACH}" == "1" ]]; then
  nohup "${CMD[@]}" > "${LOG_DIR}/train.log" 2>&1 &
  echo "$!" > "${LOG_DIR}/train.pid"
  echo "pid=$(cat "${LOG_DIR}/train.pid")"
else
  echo "$$" > "${LOG_DIR}/train.pid"
  exec "${CMD[@]}" > "${LOG_DIR}/train.log" 2>&1
fi
