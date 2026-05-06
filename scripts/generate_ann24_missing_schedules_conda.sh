#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
CONDA="${CONDA:-/home/junhyeong/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
ROBOCASA_DIR="${ROBOCASA_DIR:-/home/junhyeong/workspace/robocasa}"
GPU_DEVICE="${GPU_DEVICE:-2}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules/ann24_atomic_missing_100ep_envseed1_20260503}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-100}"
CAMERA_WIDTH="${CAMERA_WIDTH:-64}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-64}"
OBJ_INSTANCE_SPLIT="${OBJ_INSTANCE_SPLIT:-A}"
LAYOUT_STYLE_IDS="${LAYOUT_STYLE_IDS:-1:1,2:2,4:4,6:9,7:10}"

TASKS=(
  CloseDoubleDoor
  CloseDrawer
  CoffeeServeMug
  OpenDoubleDoor
  OpenDrawer
  PnPCabToCounter
  PnPCounterToCab
  PnPSinkToCounter
  PnPStoveToCounter
  TurnOffMicrowave
  TurnOffSinkFaucet
  TurnOffStove
  TurnOnSinkFaucet
  TurnOnStove
  TurnSinkSpout
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_DEVICE}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_DEVICE}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_DIR}:${ROBOCASA_DIR}:${PYTHONPATH:-}"

mkdir -p "${SCHEDULE_DIR}/seed${SEED}"
cd "${REPO_DIR}"

for env_name in "${TASKS[@]}"; do
  schedule_path="${SCHEDULE_DIR}/seed${SEED}/${env_name}_${N_EPISODES}eps.json"
  if [[ -f "${schedule_path}" ]]; then
    echo "[skip] ${schedule_path}"
    continue
  fi

  echo "[schedule] start env=${env_name} path=${schedule_path}"
  "${CONDA}" run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/robocasa_n15_zmq_parallel_eval.py \
    --env_name "${env_name}" \
    --host 127.0.0.1 \
    --port 1 \
    --output_dir /tmp/robocasa_schedule_only_unused \
    --schedule_path "${schedule_path}" \
    --seed "${SEED}" \
    --n_episodes "${N_EPISODES}" \
    --n_envs 1 \
    --max_episode_steps 1 \
    --camera_width "${CAMERA_WIDTH}" \
    --camera_height "${CAMERA_HEIGHT}" \
    --policy_image_size 64 \
    --no_camera_obs \
    --no_offscreen_renderer \
    --obj_instance_split "${OBJ_INSTANCE_SPLIT}" \
    --layout_style_ids "${LAYOUT_STYLE_IDS}" \
    --generate_schedule_only \
    --no_video
  echo "[schedule] done env=${env_name}"
done

echo "[schedule] all done root=${SCHEDULE_DIR}"
