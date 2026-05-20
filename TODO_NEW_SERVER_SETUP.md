# TODO: New Server RoboCasa / GR00T Setup

Purpose: reproduce the current RoboCasa + GR00T training/eval environment on a new server without copying heavy experiment outputs into git.

## 1. Code

- Clone/pull this fork branch:
  - remote: `git@github-jun981015:jun981015/Isaac-GR00T.git`
  - branch: `Isaac-GR00T`
- Important latest commit:
  - `dce7056 Add RoboCasa CFG eval and dataset tooling`
- Do not rely on `local_outputs/` from git. It is ignored and must be copied separately if needed.

## 2. Required External Paths

Prepare these paths on the new server, or override paths in scripts:

```text
/home/junhyeong/Value/Isaac-GR00T
/home/junhyeong/Value/robocasa
/home/junhyeong/workspace/robocasa
/home/junhyeong/workspace/robosuite
/home/junhyeong/data/robocasa_lerobot
/home/junhyeong/data/robocasa_lerobot_flat
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B
```

Notes:

- `Value/robocasa` is used by the GR00T train/policy env for helper config imports.
- `workspace/robocasa` and `workspace/robosuite` are used by the RoboCasa eval env for simulator/assets.
- The HF base cache is important for compact checkpoints and for avoiding repeated downloads.

## 3. GR00T Train / Policy Env

Main docs:

```text
GROOT_CONDA_FINETUNE_HANDOFF.md
GROOT_SMOKE_CONDA_DEPENDENCIES.md
```

Install shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

ENV_NAME=groot-smoke \
REPO_ROOT=/home/junhyeong/Value/Isaac-GR00T \
ROBOCASA_ROOT=/home/junhyeong/Value/robocasa \
HF_HOME_DIR=/home/junhyeong/.cache/huggingface \
bash envs/install_groot_smoke_conda.sh
```

Verify:

```bash
conda run -n groot-smoke python envs/verify_groot_smoke_env.py
```

## 4. RoboCasa Eval Env

Main docs:

```text
JUNHYEONG_ROBOCASA_CONDA_EVAL_SETUP.md
JUNHYEONG_ROBOCASA_CONDA_EVAL_SMOKE_HANDOFF.md
JUNHYEONG_ROBOCASA_ZMQ_PARALLEL_EVAL_HANDOFF.md
```

Install shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T

REPO_DIR=/home/junhyeong/Value/Isaac-GR00T \
ROBOCASA_DIR=/home/junhyeong/workspace/robocasa \
ROBOSUITE_DIR=/home/junhyeong/workspace/robosuite \
ENV_NAME=robocasa-eval \
bash scripts/install_robocasa_eval_conda.sh
```

Render smoke:

```bash
PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/workspace/robocasa:${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES=<gpu> \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
conda run --no-capture-output -n robocasa-eval \
  python scripts/smoke/robocasa_eval_render_smoke.py \
  --camera-width 256 \
  --camera-height 256 \
  --render-width 256 \
  --render-height 256
```

## 5. Weights / Schedules

Copy separately with `rsync`:

```text
local_outputs/robocasa_awr_retrain/<run>/checkpoint-*
local_outputs/robocasa_cfg_retrain/<run>/checkpoint-*
local_outputs/robocasa_benchmark/schedules
```

Do not copy full `local_outputs/` unless intentionally moving videos/hdf5/eval outputs.

Important recent CFG checkpoint examples:

```text
local_outputs/robocasa_cfg_retrain/ann24_plus_weight1eval_cfgdrop02_b64_50k_gpu2_20260519_031000/checkpoint-50000
local_outputs/robocasa_cfg_retrain/ann24_plus_weight1eval_failuretag_drop02_b64_50000step_gpu3_20260519_172000/checkpoint-40000
```

## 6. Eval Architecture Reminder

```text
GR00T policy server env: groot-smoke
  - loads checkpoint
  - owns torch/CUDA inference
  - serves actions over ZMQ

RoboCasa eval env: robocasa-eval
  - creates/resets/steps RoboCasa env
  - renders observations/videos
  - requests actions from policy server
```

Use fixed schedule JSONs for comparable eval across checkpoints.

## 7. Immediate Follow-Ups

- Confirm new server has matching NVIDIA driver for the GR00T torch/cu124 stack.
- Confirm `robocasa_n15_data_config.py` import works from the train/policy env.
- Confirm RoboCasa render smoke works before launching long eval.
- Copy only the checkpoints needed for the next experiment.
- Keep `local_outputs/`, videos, hdf5, safetensors, wandb logs out of git.
