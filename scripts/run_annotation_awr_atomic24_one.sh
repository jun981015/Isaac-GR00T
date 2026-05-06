#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <gpu_id> <annotation_r> <stamp>" >&2
  exit 2
fi

GPU_ID="$1"
ANNOTATION_R="$2"
STAMP="$3"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-64}"
AWR_ALPHA="${AWR_ALPHA:-100}"
AWR_CLIP_MAX="${AWR_CLIP_MAX:-2.0}"

REPO_DIR=/home/junhyeong/Value/Isaac-GR00T
CONDA=/home/junhyeong/miniconda3/bin/conda
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
RUN_NAME="ann24_awrsrc_annotation_r${ANNOTATION_R}_alpha${AWR_ALPHA}_b${BATCH_SIZE}_${MAX_STEPS}step_gpu${GPU_ID}_${STAMP}"
OUT="${REPO_DIR}/local_outputs/robocasa_awr_retrain/${RUN_NAME}"

DATASET_PATHS=(
  /home/junhyeong/data/robocasa_lerobot_flat/CloseDoubleDoor
  /home/junhyeong/data/robocasa_lerobot_flat/CloseDrawer
  /home/junhyeong/data/robocasa_lerobot_flat/CloseSingleDoor
  /home/junhyeong/data/robocasa_lerobot_flat/CoffeePressButton
  /home/junhyeong/data/robocasa_lerobot_flat/CoffeeServeMug
  /home/junhyeong/data/robocasa_lerobot_flat/CoffeeSetupMug
  /home/junhyeong/data/robocasa_lerobot_flat/OpenDoubleDoor
  /home/junhyeong/data/robocasa_lerobot_flat/OpenDrawer
  /home/junhyeong/data/robocasa_lerobot_flat/OpenSingleDoor
  /home/junhyeong/data/robocasa_lerobot_flat/PnPCabToCounter
  /home/junhyeong/data/robocasa_lerobot_flat/PnPCounterToCab
  /home/junhyeong/data/robocasa_lerobot_flat/PnPCounterToMicrowave
  /home/junhyeong/data/robocasa_lerobot_flat/PnPCounterToSink
  /home/junhyeong/data/robocasa_lerobot_flat/PnPCounterToStove
  /home/junhyeong/data/robocasa_lerobot_flat/PnPMicrowaveToCounter
  /home/junhyeong/data/robocasa_lerobot_flat/PnPSinkToCounter
  /home/junhyeong/data/robocasa_lerobot_flat/PnPStoveToCounter
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOffMicrowave
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOffSinkFaucet
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOffStove
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOnMicrowave
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOnSinkFaucet
  /home/junhyeong/data/robocasa_lerobot_flat/TurnOnStove
  /home/junhyeong/data/robocasa_lerobot_flat/TurnSinkSpout
)

cd "${REPO_DIR}"

echo "[launch] $(date -Is) ${RUN_NAME}"
echo "[launch] cwd=$(pwd)"
echo "[launch] CUDA_VISIBLE_DEVICES=${GPU_ID}"
echo "[launch] env=gr00t-train"
echo "[launch] awr_source=annotation annotation_r=${ANNOTATION_R} alpha=${AWR_ALPHA} clip=${AWR_CLIP_MAX}"
echo "[launch] excluded composite tasks: ArrangeVegetables MicrowaveThawing PreSoakPan PrepareCoffee RestockPantry"
echo "[launch] excluded navigation task: NavigateKitchen"
echo "[launch] output=${OUT}"

export PATH=/home/junhyeong/miniconda3/bin:${PATH}
export PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:${PYTHONPATH:-}
export HF_HOME=/home/junhyeong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/junhyeong/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=RwHlabs
export WANDB_PROJECT=groot_robocasa_awr_annotation
export WANDB_RUN_GROUP="ann24_alpha${AWR_ALPHA}_r_grid_${STAMP}"
export WANDB_NAME="${RUN_NAME}"

exec "${CONDA}" run --no-capture-output -n gr00t-train \
  python scripts/robocasa_awr_finetune.py \
  --dataset-path "${DATASET_PATHS[@]}" \
  --output-dir "${OUT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --num-gpus 1 \
  --save-steps "${SAVE_STEPS}" \
  --base-model-path "${BASE}" \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --lora-rank 0 \
  --dataloader-num-workers 12 \
  --dataloader-prefetch-factor 4 \
  --gradient-accumulation-steps 1 \
  --report-to wandb \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --awr-source annotation \
  --annotation-root /home/junhyeong/Value/annotations \
  --annotation-camera robot0_agentview_left \
  --annotation-r "${ANNOTATION_R}" \
  --awr-alpha "${AWR_ALPHA}" \
  --awr-clip-max "${AWR_CLIP_MAX}" \
  --task-alpha-map ""
