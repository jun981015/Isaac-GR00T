# GR00T RoboCasa Conda Finetune Handoff

Last reviewed: 2026-04-30 KST

This document is the operational handoff for running Isaac-GR00T RoboCasa finetuning with the conda env `gr00t-train`, replacing the previous Docker-based GR00T training workflow.

This file is intended to be self-contained for day-to-day finetuning. For the full dependency audit and Docker-to-conda reconstruction details, also read:

```text
GROOT_SMOKE_CONDA_DEPENDENCIES.md
```

For advanced operational tips, especially compact checkpoints that avoid repeatedly storing frozen VLM/base tensors, checkpoint recovery, and eval/server caveats, also skim:

```text
JUNHYEONG_ROBOCASA_AWR_EVAL_HANDOFF.md
to_codex_robocasa_awr_eval_handoff.md
ROBOCASA_TASK_FINETUNE_HANDOFF.md
```

This file focuses on how to actually launch, monitor, resume, checkpoint, and sanity-check finetuning.

## 0. End-To-End Flow

If you are starting from a fresh shell on the current server, the normal sequence is:

```text
1. cd /home/junhyeong/Value/Isaac-GR00T
2. Build or activate conda env gr00t-train.
3. Verify env packages, GR00T import, RoboCasa data config import, HF base snapshot, and video decode.
4. Run a 1-step smoke on one GPU if this is a new machine or changed setup.
5. Choose standard BC or AWR/reward-weighted finetuning.
6. On a multi-GPU host, launch with explicit CUDA_VISIBLE_DEVICES + IS_TORCHRUN=1 + torchrun.
7. Watch first logs for data loading, trainable flags, dataloader length, GPU memory, and first loss.
8. Monitor WandB/TensorBoard and checkpoint directories.
9. After training, inspect trainer_state.json and checkpoint loadability.
10. Optionally serve a checkpoint with scripts/inference_service.py for downstream RoboCasa eval.
```

Minimum current-server quick start for the preferred multi-GPU standard BC run:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

conda run -n gr00t-train python envs/verify_groot_smoke_env.py

CUDA_VISIBLE_DEVICES=1,2 \
IS_TORCHRUN=1 \
WANDB_ENTITY=RwHlabs \
WANDB_PROJECT=groot_robocasa_finetune_3_task \
torchrun --standalone --nproc_per_node=2 --nnodes=1 \
  scripts/gr00t_finetune.py \
  --num-gpus 2 \
  --dataset-path \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove \
    /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter \
  --output-dir /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_conda_finetune/<run_name> \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 32 \
  --max-steps 20000 \
  --save-steps 5000 \
  --base-model-path /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
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
  --video-backend torchcodec
```

Replace `CUDA_VISIBLE_DEVICES=1,2` and `<run_name>` before launching.

## 1. Scope

Use this env for:

```text
GR00T N1.5 checkpoint loading
RoboCasa LeRobot data loading
standard behavior cloning finetuning
AWR / reward-weighted finetuning
offline smoke checks
ZMQ/HTTP policy serving from trained checkpoints
WandB or TensorBoard logging
```

Do not assume this env handles:

```text
RoboCasa simulator rollout
MuJoCo/EGL rendering
robosuite benchmark execution
```

Simulator eval remains a separate environment/workflow.

## 2. Required Paths

Current server paths:

```text
repo:          /home/junhyeong/Value/Isaac-GR00T
conda env:     gr00t-train
robocasa cfg:  /home/junhyeong/Value/robocasa
data root:     /home/junhyeong/data/robocasa_lerobot
HF cache:      /home/junhyeong/.cache/huggingface
base model:    /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Core RoboCasa GR00T data config:

```text
/home/junhyeong/Value/robocasa/robocasa_n15_data_config.py
robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
```

The data config uses 3 cameras:

```text
video.robot0_agentview_left
video.robot0_agentview_right
video.robot0_eye_in_hand
```

Current RoboCasa N1.5 shape under this config:

```text
action horizon: 16
raw state dim observed by config/data: 53
raw action dim observed by config/data: 12
GR00T padded state dim: 64
GR00T padded action dim: 32
```

The 3-camera setup is intentional. Do not reduce to a single camera unless creating a separate ablation.

## 3. Environment Build And Verify

Prerequisites:

```text
conda or miniconda is installed
host NVIDIA driver supports the CUDA 12.4 PyTorch wheel stack
repo files under envs/ are present
/home/junhyeong/Value/robocasa exists or ROBOCASA_ROOT is overridden
HF cache exists or the machine can download nvidia/GR00T-N1.5-3B
```

Build:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
ENV_NAME=gr00t-train envs/install_groot_smoke_conda.sh
```

The build script creates the env, installs pinned torch/cu124 packages, installs GR00T from a clean source copy, force-aligns OpenCV, installs `flash-attn` after torch, installs the NPP/FFmpeg-compatible stack needed by `torchcodec`, and writes activation hooks for `PYTHONPATH`, `HF_HOME`, `TRANSFORMERS_CACHE`, and CUDA wheel library paths.

On a new machine with different paths:

```bash
cd /path/to/Isaac-GR00T
ENV_NAME=gr00t-train \
REPO_ROOT=/path/to/Isaac-GR00T \
ROBOCASA_ROOT=/path/to/robocasa \
HF_HOME_DIR=/path/to/huggingface_cache \
envs/install_groot_smoke_conda.sh
```

Activate:

```bash
conda activate gr00t-train
```

Verify packages:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda run -n gr00t-train python envs/verify_groot_smoke_env.py
```

Expected critical versions:

```text
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
flash_attn==2.7.1.post4
transformers==4.51.3
diffusers==0.30.2
accelerate==1.2.1
numpy==1.26.4
cv2==4.8.0
torchcodec==0.1.0+cu124
torch.cuda.is_available=True
torch.version.cuda=12.4
```

Verify local repo and RoboCasa helper import:

```bash
PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:${PYTHONPATH:-} \
conda run -n gr00t-train python -c 'import gr00t, robocasa_n15_data_config; print("imports ok", gr00t.__file__)'
```

The install script writes activation hooks for `PYTHONPATH`, `HF_HOME`, `TRANSFORMERS_CACHE`, and CUDA wheel `LD_LIBRARY_PATH`. If `torchcodec` fails with `libnvrtc.so.12` or `libnppicc.so.12`, activate the env again and inspect:

```bash
echo "$CONDA_PREFIX"
ls "/etc/conda/activate.d/gr00t-train.sh"
```

Check the HF base snapshot with symlinks resolved:

```bash
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
du -shL "$BASE"
find -L "$BASE" -maxdepth 1 \( -name 'model*.safetensors' -o -name 'model.safetensors.index.json' \) -print
```

The snapshot is HuggingFace-cache symlink based. `du -sh "$BASE"` can look small; use `du -shL`.

## 4. Data Sets

Primary 3-task training set:

```text
/home/junhyeong/data/robocasa_lerobot/PnPCounterToSink
/home/junhyeong/data/robocasa_lerobot/PnPCounterToStove
/home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter
```

Other converted RoboCasa LeRobot tasks observed locally:

```text
/home/junhyeong/data/robocasa_lerobot/PnPCabToCounter
/home/junhyeong/data/robocasa_lerobot/PnPCounterToCab
/home/junhyeong/data/robocasa_lerobot/PnPCounterToMicrowave
/home/junhyeong/data/robocasa_lerobot/PnPSinkToCounter
```

Before a new task mix, check that each task has:

```text
meta/info.json
meta/modality.json
meta/episodes.jsonl
data/chunk-*/episode_*.parquet
videos/chunk-*/*
```

Quick check:

```bash
for task in PnPCounterToSink PnPCounterToStove PnPMicrowaveToCounter; do
  test -f "/home/junhyeong/data/robocasa_lerobot/${task}/meta/modality.json" && echo "OK ${task}"
done
```

## 5. Intended Trainable Parts

Original user intent:

```text
LoRA 미사용.
VLM/backbone freeze.
Action diffusion transformer / DiT 학습.
RoboCasa 3-camera setup 유지.
```

Current local preference as of 2026-05-01 KST:

```text
For new RoboCasa finetuning runs, do not train the action projector unless explicitly requested.
Use --no-tune-projector and --tune-diffusion-model.
The already-running pnp7_bc_100k_b64_gpu1_20260501_015619 run was launched before this preference was clarified and trains projector + DiT.
```

VRAM check baseline as of 2026-05-01 KST:

```text
run: pnp7_bc_100k_b64_gpu1_20260501_015619
purpose: use as the default VRAM stability check before launching similar training on another GPU
env: gr00t-train
tmux: pnp7_bc_100k_gpu1
GPU: physical GPU1
datasets: 7 PnP LeRobot datasets under /home/junhyeong/data/robocasa_lerobot
batch size: 64
max steps: 100000
save steps: 100000
trainable: action projector + action diffusion/DiT
observed VRAM: about 72.9GB / 80GB on A100 80GB
W&B: https://wandb.ai/RwHlabs/groot_robocasa_finetune_pnp/runs/joxa2yoe
log: local_outputs/training_logs/pnp7_bc_100k_b64_gpu1_20260501_015619.log
launcher: local_outputs/training_launchers/run_pnp7_bc_100k_b64_gpu1_20260501_015619.sh
```

When testing another GPU's usable VRAM, this exact projector+DiT 100k-step,
save-at-100k setup is the reference workload. If it fits and remains stable,
then lower-memory future defaults such as DiT-only should be safe.

Standard official-like setting:

```text
--no-tune-llm
--no-tune-visual
--tune-projector
--tune-diffusion-model
--lora-rank 0
```

DiT-only setting:

```text
--no-tune-llm
--no-tune-visual
--no-tune-projector
--tune-diffusion-model
--lora-rank 0
```

Known important correction from earlier work:

```text
Do not freeze the action diffusion/DiT by mistake.
The action side is the part that must learn for this RoboCasa finetune.
```

Expected log lines for DiT-only:

```text
Tune backbone vision tower: False
Tune backbone LLM: False
Tune action head projector: False
Tune action head DiT: True
Warning: No backbone trainable parameters found.
Tune action head diffusion model: True
```

Expected log lines for official-like projector+DiT:

```text
Tune backbone vision tower: False
Tune backbone LLM: False
Tune action head projector: True
Tune action head DiT: True
Warning: No backbone trainable parameters found.
Tune action head diffusion model: True
```

## 6. Standard BC Finetuning

Script:

```text
scripts/gr00t_finetune.py
```

This is the normal GR00T behavior cloning path. It uses `LeRobotSingleDataset` for one task and `LeRobotMixtureDataset` for multiple task paths.

Objective:

```text
standard flow-matching MSE over action trajectories
action_mask controls valid padded action dimensions
no simulator reward
no loss_weight
```

If `loss_weight` appears in samples, the action head can apply weighted loss. That is not the clean standard BC baseline for this project.

Single-GPU 3-task official-like command:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

RUN_NAME=robocasa_3task_bc_$(date +%Y%m%d_%H%M%S)
OUT=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_conda_finetune/${RUN_NAME}
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

CUDA_VISIBLE_DEVICES=0 \
WANDB_ENTITY=RwHlabs \
WANDB_PROJECT=groot_robocasa_finetune_3_task \
python scripts/gr00t_finetune.py \
  --dataset-path \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove \
    /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter \
  --output-dir "${OUT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 64 \
  --max-steps 20000 \
  --num-gpus 1 \
  --save-steps 5000 \
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
  --video-backend torchcodec
```

Single-GPU DiT-only variant:

```text
Change --tune-projector to --no-tune-projector.
Keep --tune-diffusion-model.
```

This DiT-only variant is useful when the explicit goal is to freeze both VLM and projector and train only the action diffusion/DiT. The official-like GR00T finetune default in `scripts/gr00t_finetune.py` trains both projector and action diffusion.

TensorBoard-only variant:

```text
Change --report-to wandb to --report-to tensorboard.
```

The standard script does not accept `--report-to none`; use TensorBoard if WandB is not desired.

## 7. AWR / Reward-Weighted Finetuning

Script:

```text
scripts/robocasa_awr_finetune.py
```

Use this only when intentionally training with AWR-style loss weighting. It adds per-sample `loss_weight` based on detected task events.

Do not use this script for a clean standard BC baseline.

Important: this AWR implementation is not simulator-return AWR. It is heuristic critical-frame upweighting from demonstration trajectories. It detects events such as pick/place/press/door interaction and increases flow-matching loss near those windows.

Current formula in the implementation:

```text
progress_delta[t] = mass assigned around detected critical event windows
mean_delta = mean(progress_delta[t : t + action_chunk_delta_count])
loss_weight = clip(exp(task_alpha * mean_delta), 1.0, awr_clip_max)
loss = mean(per_sample_flow_matching_loss * loss_weight)
```

There is no down-weighting in the current implementation because the lower bound is `1.0`.

3-task AWR command shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

RUN_NAME=robocasa_3task_awr_pm16_alpha50_$(date +%Y%m%d_%H%M%S)
OUT=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/${RUN_NAME}
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

CUDA_VISIBLE_DEVICES=0 \
WANDB_ENTITY=RwHlabs \
WANDB_PROJECT=groot_robocasa_finetune_3_task \
python scripts/robocasa_awr_finetune.py \
  --dataset-path \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove \
    /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter \
  --output-dir "${OUT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 64 \
  --max-steps 20000 \
  --num-gpus 1 \
  --save-steps 5000 \
  --base-model-path "${BASE}" \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --tune-diffusion-model \
  --lora-rank 0 \
  --awr-alpha 50 \
  --awr-clip-max 1.8 \
  --critical-radius 16 \
  --critical-mass 0.8 \
  --dataloader-num-workers 12 \
  --dataloader-prefetch-factor 4 \
  --gradient-accumulation-steps 1 \
  --report-to wandb \
  --embodiment-tag new_embodiment \
  --video-backend torchcodec
```

AWR script supports:

```text
--report-to none
--compact-checkpoints-on-save
--compact-base-model-path
--compact-link-base-model-path
```

Checkpoint compaction can save disk, but only use it after confirming the base model path is stable and recoverable.

Known completed AWR/BC local output examples:

```text
local_outputs/robocasa_awr_retrain/baseline_noawr_20k                         step 20000, train_loss about 0.02314
local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k                        step 20000, train_loss about 0.02628
local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k                       step 20000, train_loss about 0.02902
local_outputs/robocasa_awr_retrain/awr_critical8_groupalpha_clip18_50k_gpu2   step 50000, train_loss about 0.02658
local_outputs/robocasa_awr_retrain/baseline_coffee_microwave_door_8task_50k_gpu1  step 50000, train_loss about 0.02307
```

Observed speed for batch 64 on one A100 80GB in these runs:

```text
about 0.55 steps/sec
about 4.8-5.2 sec/step
20K steps: about 10 hours
50K steps: about 25 hours
```

## 8. GPU Selection And Batch Size

A100 80GB observations from this project:

```text
batch 1: smoke test only
batch 16: safe but conservative
batch 32: safe on one A100 80GB
batch 64: viable on one A100 80GB, about 50GB VRAM in corrected setup
```

Recommended default:

```text
multi-GPU server default: use explicit torchrun with CUDA_VISIBLE_DEVICES and IS_TORCHRUN=1
single A100 80GB fallback: batch 64, gradient_accumulation_steps 1
unknown GPU or shared GPU: start with batch 1 smoke, then 16, then 32/64
```

Before launching:

```bash
nvidia-smi
```

Monitor:

```bash
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu --format=csv'
```

For single GPU, external `CUDA_VISIBLE_DEVICES=<physical_gpu>` is OK. The script then maps that visible GPU to local `0`.

Example:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/gr00t_finetune.py --num-gpus 1 ...
```

Inside the process, the script sets `CUDA_VISIBLE_DEVICES=0`, which refers to the already-masked physical GPU 2.

## 9. Multi-GPU On Host Conda

This is the preferred path on a shared multi-GPU server.

Be careful: the finetune scripts were originally convenient inside Docker, where the container only saw selected GPUs. In host conda, the built-in multi-GPU branch deletes `CUDA_VISIBLE_DEVICES` before launching `torchrun`.

Avoid this host command:

```bash
CUDA_VISIBLE_DEVICES=1,2 python scripts/gr00t_finetune.py --num-gpus 2 ...
```

Reason:

```text
The script may remove CUDA_VISIBLE_DEVICES before torchrun, causing torchrun to see all host GPUs.
```

Safe host conda multi-GPU standard BC pattern:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

RUN_NAME=robocasa_3task_bc_ddp_$(date +%Y%m%d_%H%M%S)
OUT=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_conda_finetune/${RUN_NAME}
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

CUDA_VISIBLE_DEVICES=1,2 \
IS_TORCHRUN=1 \
WANDB_ENTITY=RwHlabs \
WANDB_PROJECT=groot_robocasa_finetune_3_task \
torchrun --standalone --nproc_per_node=2 --nnodes=1 \
  scripts/gr00t_finetune.py \
  --num-gpus 2 \
  --dataset-path \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove \
    /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter \
  --output-dir "${OUT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 32 \
  --max-steps 20000 \
  --save-steps 5000 \
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
  --video-backend torchcodec
```

For two A100 80GB:

```text
per-GPU batch: 32
global batch: 64
```

Use the same `IS_TORCHRUN=1 torchrun ...` pattern for `scripts/robocasa_awr_finetune.py`.

Safe host conda multi-GPU AWR pattern is identical except:

```text
script: scripts/robocasa_awr_finetune.py
add AWR args such as --awr-alpha, --awr-clip-max, --critical-radius, --critical-mass
keep --num-gpus equal to --nproc_per_node
```

Rule of thumb:

```text
Do not use the script's auto torchrun branch on a shared host.
Always pick physical GPUs with CUDA_VISIBLE_DEVICES.
Always set IS_TORCHRUN=1 when calling torchrun directly.
Always make --num-gpus match --nproc_per_node.
```

## 10. Smoke Test Before A Real Run

Use a smoke test before any new PC, new data mix, new base model, or new script variant.

Minimal smoke command using the standard script writes a small final model unless interrupted by `max_steps=1`; for pure dependency validation use the previously validated smoke result in `GROOT_SMOKE_CONDA_DEPENDENCIES.md`.

Practical 1-step training smoke:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

OUT=/tmp/groot_robocasa_smoke_$(date +%Y%m%d_%H%M%S)
BASE=/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e

CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
python scripts/gr00t_finetune.py \
  --dataset-path /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
  --output-dir "${OUT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 1 \
  --max-steps 1 \
  --num-gpus 1 \
  --save-steps 100000 \
  --base-model-path "${BASE}" \
  --no-tune-llm \
  --no-tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --lora-rank 0 \
  --dataloader-num-workers 1 \
  --dataloader-prefetch-factor 2 \
  --gradient-accumulation-steps 1 \
  --report-to tensorboard \
  --embodiment-tag new_embodiment \
  --video-backend torchcodec
```

Check:

```bash
python - "${OUT}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "trainer_state.json"
d = json.load(open(p))
print("global_step", d.get("global_step"))
print("last_log", d.get("log_history", [])[-1])
PY
```

Expected:

```text
global_step 1
train_loss present
no import errors
no torchcodec errors
```

Historical no-save smoke note:

```text
A temporary no-save copy of scripts/gr00t_finetune.py was used once to validate the conda env without writing large checkpoints.
That temporary script has been removed intentionally.
For future smoke tests, use the real training scripts with max_steps=1 and a disposable output_dir.
```

## 11. Resume

The scripts expose:

```text
--resume / --no-resume
```

Use resume only when the same output directory already contains a Trainer checkpoint.

Example:

```bash
python scripts/gr00t_finetune.py \
  --output-dir /path/to/existing_run \
  --resume \
  ...
```

Before resume, inspect:

```bash
python - <<'PY'
import json, pathlib
run = pathlib.Path("/path/to/existing_run")
for p in sorted(run.glob("checkpoint-*/trainer_state.json")):
    d = json.load(open(p))
    print(p.parent.name, d.get("global_step"), d.get("max_steps"))
PY
```

Batch-size warning:

```text
If VRAM usage after resume is much lower than expected, check whether Trainer loaded old training_args.
For example, a run that was restarted with batch 64 can still behave like an older batch 32 run if the checkpoint/training_args state wins.
Inspect trainer_state.json, command logs, and training_args.bin before trusting the resumed run.
```

Do not change semantics casually while resuming:

```text
Do not switch standard BC <-> AWR in the same output directory.
Do not change task mix unless intentionally continuing a changed experiment.
Do not assume a batch-size change took effect; inspect logs/trainer_state.
Prefer a new output_dir for materially different settings.
```

## 12. Outputs And Checkpoints

Each run writes under `--output-dir`.

Expected files:

```text
trainer_state.json
training_args.bin
config.json
model.safetensors.index.json
model-*.safetensors
experiment_cfg/metadata.json
runs/<tensorboard event dir>
checkpoint-<step>/
```

Checkpoint frequency:

```text
--save-steps controls intermediate checkpoints.
save_total_limit=5 in the scripts, so older checkpoints may be pruned.
At training end, Trainer also saves final model/state to output_dir.
```

Inspect last loss:

```bash
python - /path/to/output_dir <<'PY'
import json, pathlib, sys
state = pathlib.Path(sys.argv[1]) / "trainer_state.json"
d = json.load(open(state))
print("global_step", d.get("global_step"))
print("max_steps", d.get("max_steps"))
for row in d.get("log_history", [])[-5:]:
    print(row)
PY
```

Check checkpoint steps:

```bash
find /path/to/output_dir -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
```

Disk warning:

```text
Full GR00T checkpoints are large.
Confirm free disk before long runs.
Use compact checkpoint tooling only when you understand the symlink/base-model dependency.
```

Compact checkpoint warning:

```text
Compact checkpoints depend on base model shard symlinks.
Do not delete or move the HF base snapshot used as compact_base_model_path/link target.
Always verify compact_summary.json and symlink targets before treating a compact checkpoint as portable.
```

## 13. Logging

WandB:

```bash
export WANDB_ENTITY=RwHlabs
export WANDB_PROJECT=groot_robocasa_finetune_3_task
```

Then use:

```text
--report-to wandb
```

If WandB credentials are in `/home/junhyeong/.netrc`, conda can use them directly on host. Check:

```bash
conda run -n gr00t-train wandb status
```

Known previous WandB project:

```text
entity:  RwHlabs
project: groot_robocasa_finetune_3_task
example run: https://wandb.ai/RwHlabs/groot_robocasa_finetune_3_task/runs/f1kw1sz2
```

Disable WandB for smoke:

```bash
WANDB_MODE=disabled
--report-to tensorboard
```

TensorBoard:

```bash
tensorboard --logdir /path/to/output_dir/runs --host 0.0.0.0 --port 6006
```

## 14. Policy Server From A Checkpoint

Docker server script exists:

```text
scripts/run_groot_robocasa_zmq_server.sh
```

Conda-equivalent server command:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate gr00t-train

CHECKPOINT=/path/to/output_dir/checkpoint-20000
CUDA_VISIBLE_DEVICES=0 \
python scripts/inference_service.py \
  --server \
  --host 0.0.0.0 \
  --port 8011 \
  --model-path "${CHECKPOINT}" \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --embodiment-tag new_embodiment \
  --denoising-steps 4
```

This starts the GR00T side only. RoboCasa simulator rollout remains separate.

## 15. Choosing The Correct Finetune Mode

Use standard BC when:

```text
You want the clean official-like GR00T finetune baseline.
You do not want hand-designed reward/loss weighting.
You are comparing against prior "baseline_noawr" runs.
```

Use AWR when:

```text
You intentionally want pick/place or critical-event weighting.
You are comparing against awr_pm8, awr_pm16, critical8, or alpha sweep runs.
```

Use 8-task or composite task sets only when:

```text
The data paths are explicitly listed.
The task_alpha_map and event detection rules are appropriate for those tasks.
You have checked that each task's modality/action schema matches RobocasaKitchenPnPDataConfig or a matching config.
```

Composite task decomposition note:

```text
PreSoakPan:
  PnPCounterToSink x2
  TurnOnSinkFaucet single x3

PrepareCoffee:
  CoffeeServeMug
  CoffeeSetupMug
  CoffeePressButton
  user note: two single components are CoffeeSetupMug and CoffeePressButton

RestockPantry:
  PnPCounterToCab x2

ArrangeVegetables:
  PnPSinkToCounter x2

MicrowaveThawing:
  OpenSingleDoor
  PnPCounterToMicrowave
  CloseSingleDoor
  TurnOnMicrowave
  single x4
```

Do not infer run semantics from directory names alone. Always inspect:

```text
command log
trainer_state.json
experiment_cfg/metadata.json
script name
WandB config
```

## 16. Pre-Launch Checklist

Before starting a long run:

```text
conda activate gr00t-train
env verify passes
nvidia-smi shows enough free VRAM
base model path exists
all dataset paths exist
robocasa_n15_data_config import succeeds
chosen script matches experiment intent: gr00t_finetune.py vs robocasa_awr_finetune.py
tune_diffusion_model is True
LoRA rank is 0 unless intentionally using LoRA
WandB project/entity are set or TensorBoard is selected
output_dir is new or resume is intentional
disk has enough free space for checkpoints
```

For multi-GPU host conda, also check:

```text
Use direct torchrun, not the script auto torchrun branch.
Set CUDA_VISIBLE_DEVICES to the intended physical GPU list.
Set IS_TORCHRUN=1.
Set --nproc_per_node equal to --num-gpus.
Use per-GPU batch size in --batch-size.
```

For standard 3-task BC, also check:

```text
No reward or loss_weight path is being used.
No AWR script is being used.
```

For AWR, also check:

```text
task_alpha_map covers the task family or fallback behavior is acceptable.
critical_radius, critical_mass, awr_alpha, and awr_clip_max are intentional.
```

## 17. Post-Launch Checklist

In the first few minutes, confirm logs contain:

```text
Loading external config: robocasa_n15_data_config.RobocasaKitchenPnPDataConfig
Initialized dataset <task> with EmbodimentTag.NEW_EMBODIMENT
Loaded <N> datasets
Loading pretrained dual brain from <base model>
Tune backbone vision tower: False
Tune backbone LLM: False
Tune action head DiT: True
train dataloader length: <positive number>
train dataset length: <positive number>
GPU memory before training: <number> GB
```

During training:

```text
loss is logged every 10 steps
GPU memory stays stable
no checkpoint write errors
WandB or TensorBoard receives logs
```

At each save point:

```bash
test -f /path/to/output_dir/checkpoint-5000/trainer_state.json
```

After training:

```bash
python - <<'PY'
import json, pathlib
run = pathlib.Path("/path/to/output_dir")
d = json.load(open(run / "trainer_state.json"))
print(d.get("global_step"), d.get("max_steps"))
print(d.get("log_history", [])[-1])
PY
```

## 18. Common Failures

`ModuleNotFoundError: robocasa_n15_data_config`:

```text
Activation hook did not set PYTHONPATH or ROBOCASA_ROOT was wrong during env build.
Set PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:$PYTHONPATH.
```

`torchcodec` native library error:

```text
Activate the env.
Check ffmpeg=6.*, nvidia-npp-cu12, and LD_LIBRARY_PATH hook.
Run envs/verify_groot_smoke_env.py again.
```

Video decode smoke:

```bash
PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:${PYTHONPATH:-} \
conda run -n gr00t-train python -c 'from gr00t.utils.video import get_frames_by_timestamps; p="/home/junhyeong/data/robocasa_lerobot/PnPCounterToSink/videos/chunk-000/observation.images.robot0_agentview_right/episode_000015.mp4"; f=get_frames_by_timestamps(p,[0.0],video_backend="torchcodec")[0]; print(f.shape, f.dtype)'
```

Expected current data output:

```text
(128, 128, 3) uint8
```

Wrong GPU used in multi-GPU:

```text
Use IS_TORCHRUN=1 torchrun with CUDA_VISIBLE_DEVICES set.
Do not rely on the script's internal multi-GPU branch on host conda.
```

OOM:

```text
Reduce batch size.
Start from batch 1 smoke, then 16, then 32, then 64.
Check other users' processes with nvidia-smi.
```

WandB 404 or wrong project:

```text
Set WANDB_ENTITY=RwHlabs.
Set WANDB_PROJECT=groot_robocasa_finetune_3_task.
Confirm wandb login/status in the conda env.
```

Action side not learning:

```text
Check logs for Tune action head DiT: True.
Check command contains --tune-diffusion-model.
Do not pass --no-tune-diffusion-model.
```

## 19. Known Good Reference Results

Conda env package verify passed on this server.

One-step conda finetune smoke passed:

```text
3-task data loaded.
GR00T N1.5 checkpoint loaded.
Backbone/VLM frozen.
Projector frozen in that smoke.
Action diffusion/DiT trainable.
global_step=1
train_loss=0.6214311122894287
```

Known local 20K/50K runs under `local_outputs/robocasa_awr_retrain` show expected Trainer outputs and speeds around:

```text
batch 64
0.54-0.55 steps/sec
35 samples/sec
20K steps in about 10 hours
```

Known older standard BC checkpoint record from Docker-era work:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

If that host path is missing, the previous handoff recorded a possible recovery source:

```text
container: isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000
path in container: /workspace/model_checkpoint
```
