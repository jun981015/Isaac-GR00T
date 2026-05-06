# GR00T Smoke Conda Build Handoff

Last reviewed: 2026-04-30 KST

This document explains how to reproduce the Docker image `isaac-gr00t-robocasa:smoke` as a conda environment for Isaac-GR00T RoboCasa finetuning, smoke tests, offline checks, and ZMQ/HTTP policy serving.

It intentionally does not cover RoboCasa/MuJoCo simulator rollout. Simulator eval should use the separate eval environment prepared for RoboCasa.

## 1. Goal

Build a conda env that can replace this Docker image for the GR00T side:

```text
image: isaac-gr00t-robocasa:smoke
id:    5c354ecbd400
size:  19GB
```

Expected use cases:

```text
GR00T N1.5 checkpoint loading
RoboCasa LeRobot dataset loading
GR00T finetuning
offline smoke/debug checks
ZMQ/HTTP policy service
WandB/TensorBoard logging
```

Out of scope:

```text
RoboCasa simulator rollout
MuJoCo/EGL rendering
robosuite benchmark execution
```

## 2. Important Paths

Current server paths:

```text
GR00T repo:       /home/junhyeong/Value/Isaac-GR00T
RoboCasa helpers: /home/junhyeong/Value/robocasa
LeRobot data:     /home/junhyeong/data/robocasa_lerobot
HF cache:         /home/junhyeong/.cache/huggingface
conda env name:   groot-smoke
conda env path:   /home/junhyeong/miniconda3/envs/groot-smoke
```

GR00T N1.5 local snapshot used here:

```text
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Important build files:

```text
envs/groot-smoke-environment.yml
envs/groot-smoke-requirements.txt
envs/install_groot_smoke_conda.sh
envs/verify_groot_smoke_env.py
```

Audit/log files:

```text
envs/groot-smoke-build.log
envs/groot-smoke-resume.log
envs/groot-smoke-npp-fix.log
envs/groot-smoke-ffmpeg-fix.log
envs/groot-smoke-pip-freeze-full.txt
envs/groot-smoke-conda-list-export.txt
envs/groot-smoke-apt-manual.txt
```

## 3. One-Command Build

From the Isaac-GR00T repo root:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
chmod +x envs/install_groot_smoke_conda.sh
ENV_NAME=groot-smoke envs/install_groot_smoke_conda.sh
```

On another machine, override paths without editing the script:

```bash
ENV_NAME=groot-smoke \
REPO_ROOT=/path/to/Isaac-GR00T \
ROBOCASA_ROOT=/path/to/robocasa \
HF_HOME_DIR=/path/to/huggingface_cache \
envs/install_groot_smoke_conda.sh
```

If rebuilding from scratch:

```bash
conda env remove -n groot-smoke
ENV_NAME=groot-smoke envs/install_groot_smoke_conda.sh
```

The current script includes all follow-up fixes found during debugging. A clean build should not require manual extra installs if the target machine has working conda, network access, and a compatible NVIDIA driver.

## 4. Activation

Use either:

```bash
conda activate groot-smoke
```

or:

```bash
conda run -n groot-smoke python ...
```

The install script writes conda activation hooks:

```text
$CONDA_PREFIX/etc/conda/activate.d/groot-smoke.sh
$CONDA_PREFIX/etc/conda/deactivate.d/groot-smoke.sh
```

These hooks set:

```text
PYTHONPATH=<Isaac-GR00T repo>:<Value/robocasa>:$PYTHONPATH
HF_HOME=<HF cache>
TRANSFORMERS_CACHE=<HF cache>/hub
LD_LIBRARY_PATH=<pip CUDA wheel library dirs>:$LD_LIBRARY_PATH
```

The `LD_LIBRARY_PATH` hook is required for CUDA wheel native libraries used by `torchcodec`, especially `libnvrtc.so.12` and `libnppicc.so.12`.

## 5. What The Build Script Does

`envs/install_groot_smoke_conda.sh` intentionally follows this order:

```text
1. Create conda env from envs/groot-smoke-environment.yml.
2. Upgrade pip/setuptools/wheel.
3. Install pinned Python packages from envs/groot-smoke-requirements.txt.
4. Copy the repo to a clean temporary source tree.
5. Install GR00T package from the clean source with --no-deps.
6. Force active OpenCV import to opencv-python==4.8.0.74.
7. Remove transformer-engine if present.
8. Install flash-attn==2.7.1.post4 after torch is importable.
9. Write activation/deactivation hooks.
```

Do not move `flash-attn` into the bulk requirements install. It must be installed after torch and with:

```text
--no-build-isolation --no-deps --force-reinstall
```

Otherwise pip may build against or resolve a mismatched torch.

## 6. Docker Image Versions Matched

Observed in `isaac-gr00t-robocasa:smoke`:

```text
Python:        3.11.10
pip:           26.0.1
conda:         24.9.2
CUDA_VERSION:  12.4.1
PyTorch:       2.5.1+cu124
torchvision:   0.20.1+cu124
torchaudio:    2.5.1+cu124
flash-attn:    2.7.1.post4
transformers:  4.51.3
diffusers:     0.30.2
accelerate:    1.2.1
numpy:         1.26.4
nvidia-npp:    12.2.5.30
gr00t:         1.1.0
```

The conda env currently verifies these critical versions:

```text
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
flash_attn==2.7.1.post4
transformers==4.51.3
diffusers==0.30.2
accelerate==1.2.1
numpy==1.26.4
tensorflow==2.15.0
cv2==4.8.0
decord==0.6.0
pytorch3d==0.7.6
torchcodec==0.1.0+cu124
zmq==27.1.0
wandb==0.18.0
torch.version.cuda=12.4
```

`gr00t` imports successfully; the package version attribute may print as `unknown` in the verify script even though conda/pip metadata reports `gr00t 1.1.0`.

## 7. Host Requirements

Required outside conda:

```text
Linux x86_64
conda or miniconda
working NVIDIA driver
GPU visible through nvidia-smi for CUDA training
git and rsync available on the host
network access for package/model download unless caches are pre-populated
```

Driver warning:

```text
This env uses PyTorch CUDA 12.4 wheels.
On Docker-free conda, do not rely on Docker CUDA compatibility packages.
Use a driver new enough for CUDA 12.4 wheels; a recent 550-series driver is the safer practical target.
```

This server currently has driver `535.288.01` and CUDA runtime shown by `nvidia-smi` as `12.2`, but the local verification still works because the installed wheel stack and driver combination are sufficient here. Re-check on a new PC.

Basic driver check:

```bash
nvidia-smi
conda run -n groot-smoke python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 8. Why Some Versions Differ From Docker Apt

Docker apt provided:

```text
ffmpeg version 4.4.2-0ubuntu0.22.04.1
```

The conda env pins:

```text
ffmpeg=6.*
```

Reason: in conda-only testing, `torchcodec==0.1.0+cu124` attempted to load FFmpeg ABI 5/6/7 libraries such as `libavutil.so.57/58/59`. FFmpeg 4.4 only provides the older ABI and caused import failure. FFmpeg 6.1.1 fixed the issue.

Docker also provided NPP through CUDA apt packages. In conda, this is represented by:

```text
nvidia-npp-cu12==12.2.5.30
```

This fixes `libnppicc.so.12` failures in `torchcodec`.

## 9. Verification

Run after build:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda run -n groot-smoke python envs/verify_groot_smoke_env.py
```

Expected important output:

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

Expected harmless warnings:

```text
TensorFlow cuDNN/cuFFT/cuBLAS registration warnings
TensorFlow TF-TRT not found warning
Transformers TRANSFORMERS_CACHE deprecation warning
Albumentations update warning
Matplotlib pyparsing deprecation warnings
```

Additional import check:

```bash
PYTHONPATH=/home/junhyeong/Value/Isaac-GR00T:/home/junhyeong/Value/robocasa:${PYTHONPATH:-} \
conda run -n groot-smoke python -c 'import gr00t, robocasa_n15_data_config; print("imports ok", gr00t.__file__)'
```

Inference service CLI check:

```bash
conda run -n groot-smoke python scripts/inference_service.py --help
```

## 10. Finetune Smoke Already Validated

On 2026-04-30 KST, this env successfully ran a real GR00T RoboCasa finetune path for one optimizer step.

Smoke parameters:

```text
env: groot-smoke
GPU: CUDA_VISIBLE_DEVICES=1
tasks:
  /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink
  /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove
  /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter
data config: robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
base model: /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
batch_size: 1
max_steps: 1
gradient_accumulation_steps: 1
LoRA: off
tune_llm: False
tune_visual: False
tune_projector: False
tune_diffusion_model: True
video_backend: torchcodec
```

Observed:

```text
3 RoboCasa LeRobot datasets loaded.
GR00T checkpoint shards loaded.
Backbone/VLM frozen.
Projector frozen.
Action diffusion/DiT trainable.
One train step completed.
train_loss: 0.6214311122894287
```

Output:

```text
local_outputs/conda_smoke/groot_smoke_1step_20260430_223117.log
local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/trainer_state.json
local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/experiment_cfg/metadata.json
local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/runs/
```

No `model*.safetensors`, `optimizer.pt`, or large checkpoint files were written during that smoke run.

## 11. Normal Finetuning Command Shape

Use the standard or project-specific finetuning script, not the dependency smoke verify script.

For the current 3 RoboCasa tasks, command shape:

```bash
cd /home/junhyeong/Value/Isaac-GR00T
conda activate groot-smoke

CUDA_VISIBLE_DEVICES=0 \
WANDB_ENTITY=RwHlabs \
WANDB_PROJECT=groot_robocasa_finetune_3_task \
python scripts/gr00t_finetune.py \
  --dataset-path \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToSink \
    /home/junhyeong/data/robocasa_lerobot/PnPCounterToStove \
    /home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter \
  --output-dir /path/to/output_run \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --batch-size 64 \
  --max-steps 20000 \
  --num-gpus 1 \
  --save-steps 5000 \
  --base-model-path /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --tune-diffusion-model \
  --lora-rank 0 \
  --dataloader-num-workers 12 \
  --gradient-accumulation-steps 1 \
  --report-to wandb \
  --embodiment-tag new_embodiment \
  --video-backend torchcodec
```

For AWR/reward-weighted finetuning, use:

```text
scripts/robocasa_awr_finetune.py
```

Do not mix AWR/reward-weighted scripts with standard BC runs unless that is explicitly intended. Standard BC should not add `loss_weight` or `reward` fields.

## 12. Common Failure Modes And Fixes

Editable install or package install fails under `local_outputs`:

```text
Cause: setuptools scanned Docker/runtime outputs or root-owned files.
Fix: use the current install script; it copies a clean source tree and excludes local_outputs.
```

`ModuleNotFoundError: flash_attn`:

```text
Cause: flash-attn was skipped or failed after torch install.
Fix: rerun install script, or manually install after torch:
CUDA_HOME=$CONDA_PREFIX MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps --force-reinstall flash-attn==2.7.1.post4
```

`torchcodec` cannot load `libnvrtc.so.12`:

```text
Cause: pip CUDA wheel library dirs are not on LD_LIBRARY_PATH.
Fix: activate the env so the hook runs, or inspect $CONDA_PREFIX/etc/conda/activate.d/groot-smoke.sh.
```

`torchcodec` cannot load `libnppicc.so.12`:

```text
Cause: NPP runtime missing.
Fix: ensure nvidia-npp-cu12==12.2.5.30 is installed and the activation hook includes site-packages/nvidia/npp/lib.
```

`torchcodec` complains about FFmpeg ABI:

```text
Cause: FFmpeg 4.x ABI is too old for torchcodec 0.1.0+cu124.
Fix: conda install ffmpeg=6.* or rebuild with current environment.yml.
```

`cv2` version is not `4.8.0`:

```text
Cause: opencv-python-headless may shadow the intended full opencv-python wheel.
Fix: script force-reinstalls opencv-python==4.8.0.74 with --no-deps.
```

CUDA unavailable:

```text
Cause: no visible GPU, incompatible driver, or CUDA_VISIBLE_DEVICES masking.
Fix: check nvidia-smi, driver version, and CUDA_VISIBLE_DEVICES.
```

## 13. Review Checklist

Before handing this env to another user or PC, verify:

```text
conda env exists: conda info --envs | grep groot-smoke
activation hook exists under $CONDA_PREFIX/etc/conda/activate.d
PYTHONPATH includes Isaac-GR00T and Value/robocasa after activation
HF cache contains or can download nvidia/GR00T-N1.5-3B
envs/verify_groot_smoke_env.py passes
import gr00t and robocasa_n15_data_config succeeds
scripts/inference_service.py --help succeeds
one-step finetune smoke succeeds if GPU is available
```

The current server passed:

```text
env verify script
gr00t + robocasa_n15_data_config import
inference_service.py --help
3-task batch=1 max_steps=1 finetune smoke
```

