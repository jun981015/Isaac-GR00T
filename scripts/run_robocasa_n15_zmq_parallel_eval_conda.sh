#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/junhyeong/Value/Isaac-GR00T}"
ROBOCASA_DIR="${ROBOCASA_DIR:-/home/junhyeong/workspace/robocasa}"
CONDA_ENV="${CONDA_ENV:-robocasa-eval}"
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
PORT="${PORT:-8011}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/conda_${CONDA_ENV}_$(date +%Y%m%d_%H%M%S)}"
SCHEDULE_DIR="${SCHEDULE_DIR:-${REPO_DIR}/local_outputs/robocasa_benchmark/schedules}"
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
USE_CAMERA_OBS="${USE_CAMERA_OBS:-1}"
HAS_OFFSCREEN_RENDERER="${HAS_OFFSCREEN_RENDERER:-1}"
OBJ_INSTANCE_SPLIT="${OBJ_INSTANCE_SPLIT:-A}"
LAYOUT_STYLE_IDS="${LAYOUT_STYLE_IDS:-1:1,2:2,4:4,6:9,7:10}"
SCHEDULE_PATH="${SCHEDULE_PATH:-${SCHEDULE_DIR}/seed${SEED}/${ENV_NAME}_${N_EPISODES}eps.json}"
REGENERATE_SCHEDULE="${REGENERATE_SCHEDULE:-0}"
GENERATE_SCHEDULE_ONLY="${GENERATE_SCHEDULE_ONLY:-0}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-300}"
PING_TIMEOUT_MS="${PING_TIMEOUT_MS:-5000}"
ACTION_TIMEOUT_MS="${ACTION_TIMEOUT_MS:-120000}"
API_TOKEN="${API_TOKEN:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
  else
    export MUJOCO_EGL_DEVICE_ID="0"
  fi
else
  export MUJOCO_EGL_DEVICE_ID
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_DIR}:${ROBOCASA_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${SCHEDULE_DIR}"

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
if [[ "${USE_CAMERA_OBS}" == "0" ]]; then
  EXTRA_ARGS+=(--no_camera_obs)
fi
if [[ "${HAS_OFFSCREEN_RENDERER}" == "0" ]]; then
  EXTRA_ARGS+=(--no_offscreen_renderer)
fi
if [[ -n "${API_TOKEN}" ]]; then
  EXTRA_ARGS+=(--api_token "${API_TOKEN}")
fi

cd "${REPO_DIR}"
echo "Starting conda RoboCasa parallel eval"
echo "Conda env: ${CONDA_ENV}"
echo "Task: ${ENV_NAME}, episodes: ${N_EPISODES}, n_envs: ${N_ENVS}, seed: ${SEED}"
echo "Policy: ${MODEL_HOST}:${PORT}"
echo "Output: ${OUTPUT_DIR}"
echo "Schedule: ${SCHEDULE_PATH}"

conda run --no-capture-output -n "${CONDA_ENV}" python scripts/robocasa_n15_zmq_parallel_eval.py \
  --env_name "${ENV_NAME}" \
  --host "${MODEL_HOST}" \
  --port "${PORT}" \
  --output_dir "${OUTPUT_DIR}" \
  --schedule_path "${SCHEDULE_PATH}" \
  --seed "${SEED}" \
  --n_episodes "${N_EPISODES}" \
  --n_envs "${N_ENVS}" \
  --policy_batch_mode "${POLICY_BATCH_MODE}" \
  --policy_batch_wait_sec "${POLICY_BATCH_WAIT_SEC}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" \
  --video_fps "${VIDEO_FPS}" \
  --video_scale "${VIDEO_SCALE}" \
  --video_render_size "${VIDEO_RENDER_SIZE}" \
  --video_source "${VIDEO_SOURCE}" \
  --video_steps_per_render "${VIDEO_STEPS_PER_RENDER}" \
  --camera_width "${CAMERA_WIDTH}" \
  --camera_height "${CAMERA_HEIGHT}" \
  --policy_image_size "${POLICY_IMAGE_SIZE}" \
  --obj_instance_split "${OBJ_INSTANCE_SPLIT}" \
  --layout_style_ids "${LAYOUT_STYLE_IDS}" \
  --server_timeout_sec "${SERVER_TIMEOUT_SEC}" \
  --ping_timeout_ms "${PING_TIMEOUT_MS}" \
  --action_timeout_ms "${ACTION_TIMEOUT_MS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/eval_${ENV_NAME}.log"
