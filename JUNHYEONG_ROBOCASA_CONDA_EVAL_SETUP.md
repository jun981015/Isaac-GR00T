# RoboCasa Eval Conda Setup

This document records how to reproduce the RoboCasa simulator side of the
GR00T N1.5 eval pipeline without Docker. The GR00T policy server remains a
separate process/env.

## Process Split

Current Docker eval already uses two processes:

```text
RoboCasa eval process
  - creates / steps RoboCasa envs
  - renders videos and writes episode JSON
  - sends numpy observation dicts over ZMQ
  - receives action dicts

GR00T policy server process
  - loads GR00T checkpoint
  - runs torch/CUDA inference
  - owns action/noise torch seeding
```

The conda migration keeps the same split:

```text
conda env robocasa-eval -> localhost ZMQ -> conda/docker GR00T policy server
```

The RoboCasa eval env does not need GR00T weights, flash-attn, or CUDA torch.
It imports only the ZMQ client and lightweight wrapper code from this repo.

## Files Added

- `envs/robocasa-eval-environment.yml`
- `envs/robocasa-eval-requirements.txt`
- `scripts/install_robocasa_eval_conda.sh`
- `scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh`

## Host Prerequisites

Install system libraries if the target host does not already provide them:

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs ffmpeg \
  libgl1 libegl1 libglvnd0 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libosmesa6
```

For normal GPU-backed offscreen rendering, use EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If EGL is unavailable, `MUJOCO_GL=osmesa` is a CPU fallback but can be slower.

## Required Source Checkouts

Default paths on this machine:

```text
/home/junhyeong/Value/Isaac-GR00T
/home/junhyeong/workspace/robocasa
/home/junhyeong/workspace/robosuite
```

The installer uses local source installs with `--no-deps`:

```bash
pip install --no-deps --force-reinstall /path/to/robosuite
pip install --no-deps -e /path/to/robocasa
```

`--no-deps` is intentional. The local `robocasa` and `robosuite` setup files
declare different `mujoco` constraints, while the Docker eval path has worked
by pinning runtime deps explicitly.

Non-editable install is intentional for `robosuite`: numba's on-disk function
cache can fail to locate source files through a PEP660 editable import hook.
RoboCasa is installed editable because its wheel can miss XML / texture assets
needed by kitchen scenes.

If you edit robosuite source, rerun the installer. If you edit RoboCasa source,
the editable install should reflect it immediately.

The installer also writes:

```text
<conda-env>/lib/python*/site-packages/robosuite/macros_private.py
```

with:

```python
CACHE_NUMBA = False
```

This avoids `RuntimeError: cannot cache function ... no locator available`
during RoboSuite import.

Isaac-GR00T itself is not installed into the eval env. The conda eval launcher
sets:

```bash
export PYTHONPATH=/path/to/Isaac-GR00T:/path/to/robocasa:$PYTHONPATH
```

This avoids scanning large / permission-restricted experiment output folders
during editable install, while still allowing imports of `gr00t.eval.*` and
`scripts.*`. The RoboCasa source path keeps scene XML / texture assets visible.

## Create / Update Env

```bash
cd /home/junhyeong/Value/Isaac-GR00T

REPO_DIR=/home/junhyeong/Value/Isaac-GR00T \
ROBOCASA_DIR=/home/junhyeong/workspace/robocasa \
ROBOSUITE_DIR=/home/junhyeong/workspace/robosuite \
ENV_NAME=robocasa-eval \
bash scripts/install_robocasa_eval_conda.sh
```

On another server, change only the three path variables.

## Run With Existing Policy Server

Start a GR00T policy server first. This can be the existing Docker server or a
future `gr00t-policy` conda server.

Then run a 1 episode smoke:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

CONDA_ENV=robocasa-eval \
MODEL_HOST=127.0.0.1 \
PORT=8011 \
ENV_NAME=PnPCounterToSink \
N_EPISODES=1 \
N_ENVS=1 \
MAX_EPISODE_STEPS=100 \
WRITE_VIDEO=1 \
STREAM_VIDEO=1 \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/conda_smoke_sink1 \
bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
```

For conda smoke runs that generate new schedules, prefer a writable run-local
schedule directory:

```bash
RUN_ROOT=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/conda_eval_smoke

OUTPUT_DIR="${RUN_ROOT}" \
SCHEDULE_DIR="${RUN_ROOT}/schedules" \
CONDA_ENV=robocasa-eval \
MODEL_HOST=127.0.0.1 \
PORT=8011 \
ENV_NAME=PnPCounterToSink \
N_EPISODES=2 \
N_ENVS=2 \
MAX_EPISODE_STEPS=80 \
CUDA_VISIBLE_DEVICES=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
```

Reason: the shared `local_outputs/robocasa_benchmark/schedules` tree may
contain Docker/root-owned directories from previous benchmark-container runs.
If conda needs to create a new schedule JSON there, it can fail with
`PermissionError`. For real reproducible eval, use a known existing
`SCHEDULE_PATH` or pre-create schedules with the correct ownership.

For normal 100 episode parallel eval:

```bash
CONDA_ENV=robocasa-eval \
MODEL_HOST=127.0.0.1 \
PORT=8011 \
ENV_NAME=PnPCounterToSink \
SEED=1 \
N_EPISODES=100 \
N_ENVS=8 \
MAX_EPISODE_STEPS=800 \
VIDEO_FPS=20 \
VIDEO_STEPS_PER_RENDER=4 \
WRITE_VIDEO=1 \
STREAM_VIDEO=1 \
SKIP_EXISTING=1 \
SCHEDULE_PATH=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToSink_100eps.json \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/conda_eval_test \
bash scripts/run_robocasa_n15_zmq_parallel_eval_conda.sh
```

The wrapper defaults thread limits to reduce CPU oversubscription:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## GPU Selection Under Conda

Conda does not isolate GPUs as strongly as Docker `--gpus device=<id>`. Always
set GPU visibility explicitly:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu_id>
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

Most torch code treats the visible GPU as `cuda:0` after masking. MuJoCo/EGL can
be more sensitive to device selection. If a host shows EGL device confusion, add:

```bash
MUJOCO_EGL_DEVICE_ID=0
```

With `CUDA_VISIBLE_DEVICES=2`, `MUJOCO_EGL_DEVICE_ID=0` means "the first visible
GPU", i.e. physical GPU2 after masking.

Use a GPU with enough free VRAM for both policy inference and RoboCasa offscreen
render contexts when both run on the same card. If possible, keep the policy
server and eval render on the same masked GPU for a smoke test, then measure.

## Verified Smoke On This Server

Detailed smoke notes are in:

```text
JUNHYEONG_ROBOCASA_CONDA_EVAL_SMOKE_HANDOFF.md
```

Validated on 2026-05-01 KST:

```text
robocasa-eval isolated render smoke:
  CUDA_VISIBLE_DEVICES=2, 128x128: OK
  CUDA_VISIBLE_DEVICES=2, 256x256: OK
  CUDA_VISIBLE_DEVICES=0, 256x256: OK

gr00t-train policy server + robocasa-eval ZMQ eval:
  GPU mask: CUDA_VISIBLE_DEVICES=1
  task: PnPCounterToSink
  N_ENVS=1, N_EPISODES=1, MAX_EPISODE_STEPS=80: OK
  N_ENVS=2, N_EPISODES=2, MAX_EPISODE_STEPS=80: OK
  video writing enabled: OK
```

Successful smoke outputs:

```text
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv1_1ep80_20260501_001611
local_outputs/robocasa_benchmark/conda_eval_smoke_gpu1_nenv2_2ep80_retry_20260501_001828
```

The initial `N_ENVS=2` attempt failed only because the default shared schedule
directory was not writable by the conda user. Re-running with
`SCHEDULE_DIR="${OUTPUT_DIR}/schedules"` passed.

An earlier run hit:

```text
FatalError('Offscreen framebuffer is not complete, error 0x8cdd')
```

This is a MuJoCo/robosuite offscreen framebuffer creation failure. Later
isolated render and ZMQ eval smoke passed, so the conda env was not generally
broken. Treat this as a runtime GPU/EGL resource or GPU-selection issue first:
choose a less loaded GPU, set `CUDA_VISIBLE_DEVICES`, and optionally test
`MUJOCO_EGL_DEVICE_ID=0`.

## Schedule Reproducibility

Use the same schedule JSONs already used by Docker eval. Important roots:

```text
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep
local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430
local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430
```

The schedule JSON stores episode metadata for deterministic environment reset.
Policy action randomness is controlled by the ZMQ `get_action_seeded` endpoint
on the GR00T policy server.

## Notes

- The eval env may have torch installed, but the eval path should not perform
  policy torch inference. Torch seeding belongs to the policy server.
- `tianshou` is intentionally not installed in this env because it pulls torch
  wheels. It is not used by the ZMQ eval path.
- `gr00t/eval/robot.py` currently contains both client and server classes. A
  future cleanup should split them into torch-free client and server modules.
- The conda wrapper intentionally mirrors
  `scripts/run_robocasa_n15_zmq_parallel_eval_container.sh` to keep settings
  comparable.
