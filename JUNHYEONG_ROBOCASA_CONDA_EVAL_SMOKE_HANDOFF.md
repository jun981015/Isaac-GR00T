# RoboCasa Conda Eval Smoke Handoff

Last updated: 2026-05-01 KST

This file records the conda-based RoboCasa eval deployment smoke tests. It is
intended for reproducing the setup on another server and for debugging the same
class of EGL / schedule / ZMQ issues later.

## Goal

Run the existing N1.5 RoboCasa ZMQ parallel eval without the benchmark Docker
container:

```text
gr00t-train policy server -> ZMQ -> robocasa-eval simulator client
```

The `gr00t-train` env owns GR00T checkpoint loading and torch/CUDA inference.
The `robocasa-eval` env owns RoboCasa env reset/step, MuJoCo/EGL rendering,
schedule replay, episode JSON, and video writing.

## Files Used

```text
envs/robocasa-eval-environment.yml
envs/robocasa-eval-requirements.txt
scripts/install_robocasa_eval_conda.sh
scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
scripts/smoke/robocasa_eval_render_smoke.py
JUNHYEONG_ROBOCASA_CONDA_EVAL_SETUP.md
```

Policy server reference:

```text
GROOT_CONDA_FINETUNE_HANDOFF.md
```

## Verified Envs

```text
policy env:  gr00t-train
eval env:    robocasa-eval
repo:        /home/junhyeong/Value/Isaac-GR00T
robocasa:    /home/junhyeong/workspace/robocasa
robosuite:   installed into robocasa-eval from /home/junhyeong/workspace/robosuite
```

Important eval versions observed:

```text
mujoco 3.2.6
robosuite 1.5.2
PyOpenGL 3.1.10
```

## Isolated Render Smoke

The first debugging step was to remove GR00T/ZMQ entirely and test only
RoboCasa + robosuite + MuJoCo offscreen rendering.

Command shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/workspace/robocasa:${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES=2 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n robocasa-eval \
  python scripts/smoke/robocasa_eval_render_smoke.py \
  --camera-width 256 \
  --camera-height 256 \
  --render-width 256 \
  --render-height 256
```

Results:

```text
CUDA_VISIBLE_DEVICES=2, 128x128: OK
CUDA_VISIBLE_DEVICES=2, 256x256: OK
CUDA_VISIBLE_DEVICES=0, 256x256: OK
```

Observed successful output includes:

```text
robot0_agentview_left_image (256, 256, 3) uint8
robot0_agentview_right_image (256, 256, 3) uint8
robot0_eye_in_hand_image (256, 256, 3) uint8
render_shape= (256, 256, 3) dtype= uint8
OK
```

Conclusion: the conda eval env's MuJoCo/EGL stack can render RoboCasa image
observations. The earlier error was not a simple missing-EGL install.

## Policy Server Smoke

A conda policy server was launched from `gr00t-train` with:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

export PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1
export NO_ALBUMENTATIONS_UPDATE=1

conda run --no-capture-output -n gr00t-train \
  python scripts/inference_service.py \
  --server \
  --host 127.0.0.1 \
  --port 18021 \
  --model_path /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000 \
  --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --embodiment_tag new_embodiment \
  --denoising_steps 4
```

Notes:

```text
Foreground conda run worked.
An earlier nohup + conda run background attempt exited with an empty server.log.
For robust long-running conda serving, make a wrapper with explicit logging or use tmux/systemd-style supervision.
```

## ZMQ Eval Smoke: N_ENVS=1

Command shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv1_1ep80_20260501_001611 \
CONDA_ENV=robocasa-eval \
PORT=18021 \
ENV_NAME=PnPCounterToSink \
N_EPISODES=1 \
N_ENVS=1 \
MAX_EPISODE_STEPS=80 \
VIDEO_RENDER_SIZE=256 \
CAMERA_WIDTH=256 \
CAMERA_HEIGHT=256 \
VIDEO_STEPS_PER_RENDER=4 \
STREAM_VIDEO=1 \
WRITE_VIDEO=1 \
SKIP_EXISTING=0 \
CUDA_VISIBLE_DEVICES=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
```

Result:

```text
[server] healthy: zmq://127.0.0.1:18021
[policy_batch] size=1 episodes=[0] calls=[0..4]
[episode] PnPCounterToSink ep=000 success=0 env_steps=80 policy_calls=5 env_step_sec=4.2 render_sec=0.0
[summary] PnPCounterToSink: 0/1 success_rate=0.000
```

Output:

```text
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv1_1ep80_20260501_001611
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv1_1ep80_20260501_001611/PnPCounterToSink/summary.json
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv1_1ep80_20260501_001611/PnPCounterToSink/videos/ep000_seed1_composite_outcome0.mp4
```

Summary JSON key values:

```text
n_episodes=1
n_envs=1
total_env_steps=80
total_policy_calls=5
total_policy_wait_sec=1.838
total_env_step_sec=4.192
```

## ZMQ Eval Smoke: N_ENVS=2

The first `N_ENVS=2` attempt failed before rollout because the default schedule
directory was not writable from conda:

```text
PermissionError: [Errno 13] Permission denied:
local_outputs/robocasa_benchmark/schedules/seed1/PnPCounterToSink_2eps.json
```

Likely cause: the shared schedules root contains Docker/root-owned paths from
previous benchmark-container runs.

Retry command used a run-local writable schedule directory:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

RUN_ROOT=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828

OUTPUT_DIR="${RUN_ROOT}" \
SCHEDULE_DIR="${RUN_ROOT}/schedules" \
CONDA_ENV=robocasa-eval \
PORT=18021 \
ENV_NAME=PnPCounterToSink \
N_EPISODES=2 \
N_ENVS=2 \
MAX_EPISODE_STEPS=80 \
VIDEO_RENDER_SIZE=256 \
CAMERA_WIDTH=256 \
CAMERA_HEIGHT=256 \
VIDEO_STEPS_PER_RENDER=4 \
STREAM_VIDEO=1 \
WRITE_VIDEO=1 \
SKIP_EXISTING=0 \
CUDA_VISIBLE_DEVICES=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
```

Result:

```text
[server] healthy: zmq://127.0.0.1:18021
[schedule] PnPCounterToSink ep=000 seed=1 layout=4 style=4
[schedule] PnPCounterToSink ep=001 seed=257 layout=6 style=9
[policy_batch] size=2 episodes=[0, 1] calls=[0, 0]
[policy_batch] size=2 episodes=[0, 1] calls=[1, 1]
[policy_batch] size=2 episodes=[0, 1] calls=[2, 2]
[policy_batch] size=2 episodes=[0, 1] calls=[3, 3]
[policy_batch] size=2 episodes=[0, 1] calls=[4, 4]
[episode] PnPCounterToSink ep=000 success=0 env_steps=80 policy_calls=5 env_step_sec=4.1 render_sec=0.0
[episode] PnPCounterToSink ep=001 success=0 env_steps=80 policy_calls=5 env_step_sec=4.9 render_sec=0.0
[summary] PnPCounterToSink: 0/2 success_rate=0.000
```

Output:

```text
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828/PnPCounterToSink/summary.json
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828/PnPCounterToSink/videos/ep000_seed1_composite_outcome0.mp4
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828/PnPCounterToSink/videos/ep001_seed257_composite_outcome0.mp4
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828/schedules/seed1/PnPCounterToSink_2eps.json
```

Summary JSON key values:

```text
n_episodes=2
n_envs=2
total_env_steps=160
total_policy_calls=10
total_policy_wait_sec=10.219
total_env_step_sec=9.017
```

## Errors Encountered

### Offscreen Framebuffer Error

Observed during an earlier GPU0 run:

```text
FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
```

Meaning: robosuite/MuJoCo failed to create or use the offscreen framebuffer
needed for image observations.

What was checked:

```text
GR00T policy server health check had passed.
RoboCasa env creation started.
GR00T-1.6 check_sim_eval_ready.py passed basic EGL headless checks.
Isolated RoboCasa render smoke later passed on GPU0 and GPU2.
GPU0 had about 75GB / 80GB VRAM used at the time.
```

Current interpretation:

```text
The conda env is not fundamentally broken.
The failure was likely tied to runtime GPU/EGL resource state or GPU selection.
Use explicit CUDA_VISIBLE_DEVICES and avoid heavily occupied GPUs for eval rendering.
```

### Schedule Permission Error

Observed during `N_ENVS=2`:

```text
PermissionError: [Errno 13] Permission denied:
local_outputs/robocasa_benchmark/schedules/seed1/PnPCounterToSink_2eps.json
```

Fix:

```text
Use a schedule directory writable by the current user.
For smoke tests, prefer SCHEDULE_DIR="${OUTPUT_DIR}/schedules".
For real reproducible eval, pre-create schedule JSONs with correct ownership or use existing read-only schedule JSONs via SCHEDULE_PATH.
```

### nohup + conda run Empty Log

Backgrounding:

```bash
nohup conda run --no-capture-output -n gr00t-train python scripts/inference_service.py ...
```

exited immediately with a 0-byte log in this smoke. Foreground execution worked.
Before using conda policy server in production, add a dedicated wrapper that:

```text
sets PYTHONPATH and CUDA_VISIBLE_DEVICES
writes stdout/stderr to a known log file
records PID
checks port health
does not rely on fragile shell/nohup behavior
```

## GPU Masking Notes

Conda does not isolate GPUs as strongly as Docker `--gpus device=<id>`.
Use environment variables:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_id>
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

If MuJoCo/EGL device selection is suspicious, try:

```bash
MUJOCO_EGL_DEVICE_ID=0
```

When `CUDA_VISIBLE_DEVICES=2`, `MUJOCO_EGL_DEVICE_ID=0` means "visible GPU 0",
which is physical GPU2 after masking.

Observed tests:

```text
render smoke with CUDA_VISIBLE_DEVICES=2: OK
render smoke with CUDA_VISIBLE_DEVICES=0: OK
ZMQ eval with CUDA_VISIBLE_DEVICES=1: N_ENVS=1 OK, N_ENVS=2 OK
```

Concurrent eval CPU thread policy:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Use these limits for each conda eval or eval container when running multiple
RoboCasa evals simultaneously. This prevents CPU oversubscription; expected
speed loss is small because eval is usually bounded by MuJoCo/env stepping,
rendering, and policy communication rather than BLAS/OpenMP work. Keep
concurrent evals bounded, around 6 at a time. The current atomic eval used this
policy and completed normally.

EGL GPU selection policy:

```text
CUDA_VISIBLE_DEVICES=<physical_eval_gpu>
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=<physical_eval_gpu for conda direct eval>
```

For conda direct eval, set `MUJOCO_EGL_DEVICE_ID` to the same physical GPU id as
`CUDA_VISIBLE_DEVICES`; for example, `CUDA_VISIBLE_DEVICES=2` with
`MUJOCO_EGL_DEVICE_ID=2`. The robosuite binding asserts that
`MUJOCO_EGL_DEVICE_ID` appears in `CUDA_VISIBLE_DEVICES`.

For Docker/container eval, the wrapper uses `--gpus device=<physical_eval_gpu>`
and internal `CUDA_VISIBLE_DEVICES=0`, so `MUJOCO_EGL_DEVICE_ID=0` is correct
inside the container. This avoids accidental render load on physical GPU0 during
concurrent evals.

For strict verification on another server, run `nvidia-smi pmon -c 1` while the
eval is active and confirm the eval Python PID appears on the intended GPU.

## Current Status

The conda split deployment is smoke-validated through:

```text
PnPCounterToSink
N_ENVS=1, N_EPISODES=1, MAX_EPISODE_STEPS=80
N_ENVS=2, N_EPISODES=2, MAX_EPISODE_STEPS=80
video writing enabled
ZMQ batched policy calls enabled
```

Not yet validated in this smoke:

```text
N_ENVS=4 or N_ENVS=8
full MAX_EPISODE_STEPS=800
100 episode real eval
conda policy server wrapper launched robustly in background
object placement equivalence against old Docker schedule JSONs
```

## Recommended Next Tests

1. Add or use a robust conda policy server launch wrapper.
2. Run `N_ENVS=4`, `N_EPISODES=4`, `MAX_EPISODE_STEPS=80`.
3. Run `N_ENVS=8`, `N_EPISODES=8`, `MAX_EPISODE_STEPS=80` if GPU/CPU load is acceptable.
4. Run one real schedule-backed task with an existing 100ep schedule JSON and `SKIP_EXISTING=1`.
5. Compare one episode's first-frame object placement against Docker eval output using the same schedule JSON.
