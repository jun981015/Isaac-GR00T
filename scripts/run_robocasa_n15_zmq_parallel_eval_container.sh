#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T-eval-zmq}"
ROBOCASA_DIR="${ROBOCASA_DIR:-/home/junhyeong/workspace/robocasa}"
ROBOSUITE_DIR="${ROBOSUITE_DIR:-/home/junhyeong/workspace/robosuite}"
IMAGE_NAME="${IMAGE_NAME:-isaac-gr00t-robocasa:benchmark}"
CONTAINER_NAME="${CONTAINER_NAME:-isaac-gr00t-robocasa-bench-zmq-parallel-n15}"
GPU_DEVICE="${GPU_DEVICE:-3}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/${CONTAINER_NAME}}"
SCHEDULE_DIR="${SCHEDULE_DIR:-/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules}"
RUNTIME_STATE_HOST_DIR="${RUNTIME_STATE_HOST_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark_runtime}"
ENV_NAME="${ENV_NAME:-PnPCounterToSink}"
SEED="${SEED:-1}"
N_EPISODES="${N_EPISODES:-50}"
N_ENVS="${N_ENVS:-4}"
POLICY_BATCH_MODE="${POLICY_BATCH_MODE:-lockstep}"
POLICY_BATCH_WAIT_SEC="${POLICY_BATCH_WAIT_SEC:-0.05}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_SCALE="${VIDEO_SCALE:-2}"
VIDEO_RENDER_SIZE="${VIDEO_RENDER_SIZE:-256}"
VIDEO_SOURCE="${VIDEO_SOURCE:-obs}"
VIDEO_STEPS_PER_RENDER="${VIDEO_STEPS_PER_RENDER:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-256}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-256}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-128}"
WRITE_VIDEO="${WRITE_VIDEO:-1}"
STREAM_VIDEO="${STREAM_VIDEO:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
OBJ_INSTANCE_SPLIT="${OBJ_INSTANCE_SPLIT:-A}"
LAYOUT_STYLE_IDS="${LAYOUT_STYLE_IDS:-1:1,2:2,4:4,6:9,7:10}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${SCHEDULE_DIR}/seed${SEED}/${ENV_NAME}_${N_EPISODES}eps.json}"
REGENERATE_SCHEDULE="${REGENERATE_SCHEDULE:-0}"
GENERATE_SCHEDULE_ONLY="${GENERATE_SCHEDULE_ONLY:-0}"
BOOTSTRAP_DEPS="${BOOTSTRAP_DEPS:-1}"
REPLACE="${REPLACE:-0}"

mkdir -p "${OUTPUT_DIR}" "${SCHEDULE_DIR}" "${RUNTIME_STATE_HOST_DIR}"

CONTAINER_SCHEDULE_PATH="/workspace/schedules/seed${SEED}/${ENV_NAME}_${N_EPISODES}eps.json"
if [[ -n "${SCHEDULE_PATH}" ]]; then
  schedule_dir_real="$(realpath -m "${SCHEDULE_DIR}")"
  schedule_path_real="$(realpath -m "${SCHEDULE_PATH}")"
  if [[ "${schedule_path_real}" == "${schedule_dir_real}"/* ]]; then
    schedule_rel="${schedule_path_real#${schedule_dir_real}/}"
    CONTAINER_SCHEDULE_PATH="/workspace/schedules/${schedule_rel}"
  fi
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if [[ "${REPLACE}" == "1" ]]; then
    docker rm -f "${CONTAINER_NAME}"
  else
    echo "Container ${CONTAINER_NAME} already exists. Set REPLACE=1 to remove it first." >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ "${REGENERATE_SCHEDULE}" == "1" ]]; then
  EXTRA_ARGS+=(--regenerate_schedule)
fi
if [[ "${GENERATE_SCHEDULE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--generate_schedule_only)
fi
if [[ "${STREAM_VIDEO}" == "1" ]]; then
  EXTRA_ARGS+=(--stream_video)
fi
if [[ "${WRITE_VIDEO}" == "0" ]]; then
  EXTRA_ARGS+=(--no_video)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --network=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e MUJOCO_GL=egl \
  -e HOME=/workspace/runtime_home \
  -e PYTHONUSERBASE=/workspace/runtime_home/.local \
  -e PYTHONPATH="/workspace/Isaac-GR00T:/workspace/robocasa:/workspace/robosuite:${PYTHONPATH:-}" \
  -v "${REPO_DIR}:/workspace/Isaac-GR00T" \
  -v "${ROBOCASA_DIR}:/workspace/robocasa" \
  -v "${ROBOSUITE_DIR}:/workspace/robosuite:ro" \
  -v "${OUTPUT_DIR}:/workspace/output" \
  -v "${SCHEDULE_DIR}:/workspace/schedules" \
  -v "${RUNTIME_STATE_HOST_DIR}:/workspace/runtime_home" \
  -w /workspace/Isaac-GR00T \
  "${IMAGE_NAME}" \
  bash -lc "
    set -euo pipefail
    export PATH=\"/workspace/runtime_home/.local/bin:\$PATH\"
    if [[ '${BOOTSTRAP_DEPS}' == '1' ]]; then
      python -m pip install --user --no-cache-dir -q \
        numpy==1.23.5 mujoco==3.2.6 numba==0.57.1 \
        pyzmq msgpack pandas imageio imageio-ffmpeg termcolor \
        opencv-python opencv-python-headless \
        scipy h5py lxml pygame pynput hidapi \
        'qpsolvers[quadprog]' mink tianshou
    fi
    python scripts/robocasa_n15_zmq_parallel_eval.py \
      --env_name '${ENV_NAME}' \
      --host '${MODEL_HOST}' \
      --port '${PORT}' \
      --output_dir /workspace/output \
      --schedule_path '${CONTAINER_SCHEDULE_PATH}' \
      --seed '${SEED}' \
      --n_episodes '${N_EPISODES}' \
      --n_envs '${N_ENVS}' \
      --policy_batch_mode '${POLICY_BATCH_MODE}' \
      --policy_batch_wait_sec '${POLICY_BATCH_WAIT_SEC}' \
      --n_action_steps '${N_ACTION_STEPS}' \
      --max_episode_steps '${MAX_EPISODE_STEPS}' \
      --video_fps '${VIDEO_FPS}' \
      --video_scale '${VIDEO_SCALE}' \
      --video_render_size '${VIDEO_RENDER_SIZE}' \
      --video_source '${VIDEO_SOURCE}' \
      --video_steps_per_render '${VIDEO_STEPS_PER_RENDER}' \
      --camera_width '${CAMERA_WIDTH}' \
      --camera_height '${CAMERA_HEIGHT}' \
      --policy_image_size '${POLICY_IMAGE_SIZE}' \
      --obj_instance_split '${OBJ_INSTANCE_SPLIT}' \
      --layout_style_ids '${LAYOUT_STYLE_IDS}' \
      ${EXTRA_ARGS[*]} \
      2>&1 | tee /workspace/output/eval_${ENV_NAME}.log
  "

echo "Started parallel ZMQ ${CONTAINER_NAME} on GPU ${GPU_DEVICE}"
echo "Task: ${ENV_NAME}, episodes: ${N_EPISODES}, n_envs: ${N_ENVS}, seed: ${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "Schedule: ${SCHEDULE_PATH}"
