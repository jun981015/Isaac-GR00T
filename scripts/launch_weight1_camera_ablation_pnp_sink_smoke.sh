#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/local_outputs/robocasa_benchmark/weight1_camera_ablation_pnp_sink_20ep_$(date +%Y%m%d_%H%M%S)}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToSink_100eps.json}"
GPU="${GPU:-2}"
PORT="${PORT:-18161}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-4}"

CONDITIONS=(
  none
  no_hand
  no_static
  hand_only
  static_only
)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"

for condition in "${CONDITIONS[@]}"; do
  echo "[camera_ablation_start] condition=${condition} task=PnPCounterToSink seed=${SEED}" | tee -a "${OUTPUT_ROOT}/logs/run.log"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID="${GPU}" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  CONDA_ENV=robocasa-eval \
  MODEL_HOST=127.0.0.1 \
  PORT="${PORT}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${condition}" \
  SCHEDULE_PATH="${SCHEDULE_PATH}" \
  ENV_NAME=PnPCounterToSink \
  SEED="${SEED}" \
  N_EPISODES="${N_EPISODES}" \
  N_ENVS="${N_ENVS}" \
  MAX_EPISODE_STEPS=800 \
  N_ACTION_STEPS=16 \
  VIDEO_SOURCE=obs \
  VIDEO_RENDER_SIZE=0 \
  VIDEO_SCALE=1 \
  VIDEO_STEPS_PER_RENDER=4 \
  WRITE_VIDEO=1 \
  STREAM_VIDEO=1 \
  CAMERA_ABLATION="${condition}" \
  bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${condition}.log"
  echo "[camera_ablation_done] condition=${condition}" | tee -a "${OUTPUT_ROOT}/logs/run.log"
done

echo "[all_done] output_root=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/logs/run.log"
