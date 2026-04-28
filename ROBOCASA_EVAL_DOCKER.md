# RoboCasa Eval Docker Status

Last recovered: 2026-04-26 KST

This document records the RoboCasa online evaluation setup that was built for GR00T checkpoints, including the strict fixed-scene evaluation method that controls object / layout / style across checkpoint comparisons.

This file was recovered from session memory after the original `ROBOCASA_EVAL_DOCKER.md` and related local folder contents were lost. The Docker images still existed at recovery time:

- `isaac-gr00t-robocasa:benchmark`
- `isaac-gr00t-robocasa:http-server`
- `isaac-gr00t-robocasa:smoke`

## Purpose

The online eval path runs RoboCasa simulation in one container and the GR00T checkpoint server in another container:

```text
RoboCasa benchmark container -> HTTP -> GR00T model server container
```

The main use cases are:

- quick RoboCasa smoke eval for a finetuned GR00T checkpoint
- fixed-task eval for `PickPlaceCounterToSink`
- strict checkpoint A/B comparison where the scene must be identical across models

This document also records the related training setup, because the Docker images, mounted folders, data config, checkpoints, and eval method are tightly coupled.

The main tasks used during bring-up were:

- `robocasa/PickPlaceCounterToSink`
- `robocasa/PickPlaceCounterToStove`
- `robocasa/PickPlaceMicrowaveToCounter`

## Immediate Recovery Warning

The training output folder was observed missing after the local folder loss:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke
```

The best known final official-like 20K checkpoint path before the loss was:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

At one point, a surviving HTTP inference container still had that checkpoint mounted:

```text
container: isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000
image:     isaac-gr00t-robocasa:http-server
model:     /workspace/model_checkpoint
```

If that container still exists and the host checkpoint is missing, copy the mounted checkpoint before stopping the container:

```bash
mkdir -p /home/junhyeong/Value/Isaac-GR00T/recovered_outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1
docker cp isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000:/workspace/model_checkpoint \
  /home/junhyeong/Value/Isaac-GR00T/recovered_outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

After copying, check for checkpoint files such as:

- `model.safetensors`
- `trainer_state.json`
- config files
- optimizer/scheduler states if present
- action head weights

Do not stop checkpoint-serving HTTP containers until the checkpoint source has been backed up.

## Folder Structure

The working folder layout used during the session was:

```text
/home/junhyeong/Value/Isaac-GR00T
  GR00T repo and training/eval scripts

/home/junhyeong/Value/robocasa
  helper scripts and data config used by GR00T
  robocasa_n15_data_config.py
  convert_robocasa_to_groot_lerobot_v21.py
  robocasa_n15_conversion_notes.md

/home/junhyeong/Value/robocasa_official_v1
  official RoboCasa checkout used for simulator rollout

/home/junhyeong/workspace/robosuite
  robosuite checkout used by RoboCasa simulator

/home/junhyeong/data/robocasa_lerobot
  converted LeRobot-format RoboCasa training data

/home/junhyeong/data/robocasa
  original HDF5 RoboCasa demos
```

Important output roots:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke
  standard BC training outputs, checkpoints, logs

/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark
  simulator eval videos, benchmark logs, schedule JSONs

/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_http_server
  HTTP server runtime/output folders

/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain
  AWR/reward-weighted experiments, not standard BC
```

Keep standard BC, AWR, HTTP server outputs, and benchmark videos in separate folders. Mixing these makes later recovery and comparison difficult.

## Data

Converted LeRobot data root:

```text
/home/junhyeong/data/robocasa_lerobot
```

Container mount:

```text
/home/junhyeong/data/robocasa_lerobot -> /workspace/data/robocasa_lerobot:ro
```

Standard 3-task training set:

```text
/workspace/data/robocasa_lerobot/PnPCounterToSink
/workspace/data/robocasa_lerobot/PnPCounterToStove
/workspace/data/robocasa_lerobot/PnPMicrowaveToCounter
```

Other converted RoboCasa tasks that existed and may be useful later:

```text
/home/junhyeong/data/robocasa_lerobot/PnPCabToCounter
/home/junhyeong/data/robocasa_lerobot/PnPCounterToCab
/home/junhyeong/data/robocasa_lerobot/PnPCounterToMicrowave
/home/junhyeong/data/robocasa_lerobot/PnPCounterToSink
/home/junhyeong/data/robocasa_lerobot/PnPCounterToStove
/home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter
/home/junhyeong/data/robocasa_lerobot/PnPSinkToCounter
```

Original HDF5 sources seen during the session:

```text
/home/junhyeong/data/robocasa/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5
```

Observed 3-task train split summary:

```text
PnPCounterToSink:        39 train episodes, 16536 visible steps
PnPCounterToStove:       39 train episodes, 11940 visible steps
PnPMicrowaveToCounter:   39 train episodes, 11735 visible steps
train dataset length:    40211
train dataloader length: 629 with batch 64
```

Observed mixture weights:

```text
PnPCounterToSink:        0.41123075775285367
PnPCounterToStove:       0.29693367486508665
PnPMicrowaveToCounter:   0.2918355673820596
```

Training samples are shuffled by sample/start index. Each sample still preserves its internal temporal structure:

- observation at sampled time
- 3 camera images
- state
- language
- action chunk target after that time

It is not training on full videos in sequential order.

## Model And Training Setup

Base model:

```text
nvidia/GR00T-N1.5-3B
```

Local HF snapshot used:

```text
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Container path:

```text
/workspace/hf_cache/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Intended official-like finetune:

```text
LoRA: not used
VLM/LLM backbone: frozen
Vision tower: frozen
Action head projector: trainable
Action diffusion model / DiT: trainable
Embodiment tag: new_embodiment
Data config: robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
```

Important log lines confirming the intended trainable state:

```text
Tune backbone vision tower: False
Tune backbone LLM: False
Tune action head projector: True
Tune action head DiT: True
Tune backbone llm: False
Tune backbone visual: False
Warning: No backbone trainable parameters found.
Tune action head projector: True
Tune action head diffusion model: True
```

Critical correction from the session:

```text
Early experiments likely froze the action diffusion/DiT unintentionally.
The corrected official-like setup trains the action diffusion model and projector while freezing VLM/backbone.
```

Do not compare early frozen-action runs directly against corrected official-like BC runs.

## RoboCasa Data Config

The data config module:

```text
/home/junhyeong/Value/robocasa/robocasa_n15_data_config.py
```

Command reference:

```text
--data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
```

Make sure `PYTHONPATH` includes:

```text
/workspace/Isaac-GR00T:/workspace/Value/robocasa
```

Known details to verify after recovery:

```text
RobocasaKitchenPnPDataConfig
3 camera streams
state groups total around 53D
RoboCasa action around 12D
GR00T max_action_dim 32 with padding/mask
action horizon/chunk effectively 16 for GR00T N1.5 action head
```

The exact camera/action key order matters for rollout quality, especially:

- eef position
- eef rotation
- gripper
- base
- control mode

Sanity checks before serious retraining:

1. Load one sample from each task.
2. Print image keys/shapes for all camera streams.
3. Print state/action shapes.
4. Verify action mask and `max_action_dim`.
5. Verify action chunk horizon matches the GR00T transform/data config.
6. Confirm standard BC samples do not include `reward` or `loss_weight`.

## Training Docker Image

Training/offline checks used:

```text
isaac-gr00t-robocasa:smoke
```

This image was for GR00T finetuning and offline checks. Do not assume it contains the full RoboCasa/MuJoCo simulator runtime for video rollout. Simulator rollout used the `benchmark` and `http-server` images.

The long-lived standard training container used:

```text
name:  isaac-gr00t-robocasa-smoke
image: isaac-gr00t-robocasa:smoke
cmd:   tail -f /dev/null
```

Important training mounts:

```text
/home/junhyeong/data/robocasa_lerobot -> /workspace/data/robocasa_lerobot:ro
/home/junhyeong/.cache/huggingface -> /workspace/hf_cache
/home/junhyeong/Value/Isaac-GR00T -> /workspace/Isaac-GR00T
/home/junhyeong/Value/robocasa -> /workspace/Value/robocasa:ro
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke -> /workspace/outputs
```

WandB mounts that should be included:

```text
-v /home/junhyeong/.netrc:/workspace/outputs/home/.netrc:ro
-v /home/junhyeong/.config/wandb:/workspace/outputs/home/.config/wandb:ro
```

## Training Script Inventory

The standard BC training script used during the session was:

```text
scripts/robocasa_multitask_finetune.py
```

If this script is missing after recovery, recreate it before launching a new official-like BC run.

Purpose:

- train GR00T N1.5 on selected RoboCasa LeRobot task directories
- support multi-task manifest/split
- freeze VLM/backbone
- train projector + action diffusion model
- support resume from explicit checkpoint
- support single-GPU and DDP

Important flags:

```text
--split-manifest
--output-dir
--robocasa-helper-dir
--data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
--base-model-path
--gpu-index
--batch-size
--gradient-accumulation-steps
--max-steps
--save-steps
--save-total-limit
--dataloader-num-workers
--tune-diffusion-model / --no-tune-diffusion-model
--balance-dataset-weights / --no-balance-dataset-weights
--balance-trajectory-weights / --no-balance-trajectory-weights
--expected-task-names
--resume-from-checkpoint
--report-to tensorboard|wandb|none
```

Important defaults:

```text
tune_diffusion_model=True
balance_dataset_weights=True
balance_trajectory_weights=True
```

Important implementation guards:

- `set_single_gpu_visible(args.gpu_index)` should only run when world size is 1
- manifest copy/log printing should be rank-gated where needed
- explicit `--resume-from-checkpoint` should override auto-discovery
- print trainable intent before training

## Final Training Command

The final corrected continuation used a command equivalent to:

```bash
cd /workspace/Isaac-GR00T
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
export WANDB_DIR=/workspace/outputs/home/wandb
export WANDB_CACHE_DIR=/workspace/outputs/home/.cache/wandb
export WANDB_PROJECT=groot_robocasa_finetune_3_task
export WANDB_ENTITY=RwHlabs

python scripts/robocasa_multitask_finetune.py \
  --split-manifest /workspace/outputs/robocasa_3tasks_split.json \
  --output-dir /workspace/outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1 \
  --robocasa-helper-dir /workspace/Value/robocasa \
  --data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --base-model-path /workspace/hf_cache/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
  --gpu-index 1 \
  --batch-size 64 \
  --gradient-accumulation-steps 1 \
  --max-steps 20000 \
  --save-steps 5000 \
  --save-total-limit 20 \
  --dataloader-num-workers 0 \
  --tune-diffusion-model \
  --balance-dataset-weights \
  --balance-trajectory-weights \
  --report-to wandb \
  --resume-from-checkpoint /workspace/outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-8000
```

Known final log path:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/train_single_gpu_bs64_gpu2_from_ckpt8000_to20k_wandb_RwHlabs_groot_robocasa_finetune_3_task.log
```

## Training Timeline

Early low-value smoke runs:

```text
robocasa_smoke_step1
  checkpoint-1
  first one-step smoke

robocasa_train_gpu2_run1
  checkpoint-30, checkpoint-40, checkpoint-50
  single-task early validation
```

Early multi-task baseline before the action-head fix:

```text
robocasa_multitask_10k_run1
  checkpoint-8000, checkpoint-9000, checkpoint-10000
  checkpoint-8000 loss around 0.0565
  checkpoint-9000 loss around 0.0446
  checkpoint-10000 loss around 0.079
  caveat: action DiT was probably frozen
```

Corrected official-like run:

```text
robocasa_multitask_ddp_bs32x2_official_32k_run1
```

Despite the name, the final 8K to 20K continuation was single-GPU batch 64.

Saved checkpoints:

```text
checkpoint-2000
checkpoint-4000
checkpoint-6000
checkpoint-8000
checkpoint-10000
checkpoint-12000
checkpoint-14000
checkpoint-16000
checkpoint-20000
```

Final recorded training stats:

```text
global step: 20000 / 20000
final train_loss: 0.00845177420154214
final step loss at 20000: about 0.0087
train_runtime: 60236.8694 sec
train_samples_per_second: 21.249
train_steps_per_second: 0.332
epoch: 31.8
```

Recent final losses:

```text
19940: 0.0117
19950: 0.0121
19960: 0.0083
19970: 0.0103
19980: 0.0108
19990: 0.0088
20000: 0.0087
```

## Training Hardware Notes

Observed hardware:

```text
4 x NVIDIA A100 80GB PCIe
MIG disabled
```

Training observations:

```text
single GPU, batch 64: about 50.5GB VRAM
step speed: about 4.8-5.2 sec/step
batch 64 was viable on one A100 80GB
```

Two-GPU DDP was used briefly:

```text
host GPUs: 1,2
per-GPU batch: 32
global batch: 64
saved: checkpoint-2000
```

The final 8K to 20K continuation was single GPU batch 64.

## Trainer Resume Caveat

HuggingFace Trainer may load `train_batch_size` from `checkpoint-*/trainer_state.json` when resuming.

Incident from the session:

```text
checkpoint-2000/trainer_state.json had train_batch_size=32.
We tried to resume single-GPU batch 64.
VRAM stayed around 35GB, indicating actual batch 32.
The run produced a checkpoint-2020 probe.
We stopped it, deleted checkpoint-2020, backed up trainer_state.json, and edited train_batch_size 32 -> 64.
Then true batch 64 used about 50.5GB.
```

Future rule:

1. Before changing batch size on resume, inspect `trainer_state.json`.
2. Check `train_batch_size`, `save_steps`, `max_steps`, and `global_step`.
3. Confirm VRAM after restart.
4. On A100 80GB, true batch 64 should be around 50GB, not around 35GB.

## Save Step Caveat

`save_steps` is Trainer global optimizer step, not sample count or epoch count.

Resume continues from `global_step`, so checkpoint names are global.

If `save_total_limit` is too small, older checkpoints may be auto-deleted. Keep it high while doing eval comparisons.

Observed save behavior:

```text
User asked for 10K, 15K, 20K during one continuation.
Actual observed saves included 10K, 12K, 14K, 16K, 20K because resume metadata and save_steps interacted.
```

Trust `checkpoint-*/trainer_state.json` for exact global step.

## WandB

Final standard BC run:

```text
entity:  RwHlabs
project: groot_robocasa_finetune_3_task
run id:  f1kw1sz2
url:     https://wandb.ai/RwHlabs/groot_robocasa_finetune_3_task/runs/f1kw1sz2
```

WandB settings:

```bash
export WANDB_ENTITY=RwHlabs
export WANDB_PROJECT=groot_robocasa_finetune_3_task
export WANDB_DIR=/workspace/outputs/home/wandb
export WANDB_CACHE_DIR=/workspace/outputs/home/.cache/wandb
```

Mistakes corrected:

```text
Initial run accidentally went to junhyeong/robocasa_groot.
RwHlabs-org failed because WandB does not allow logging directly to organization entity.
Correct team entity was RwHlabs.
Correct project name was groot_robocasa_finetune_3_task.
```

Large checkpoint artifact upload was not needed. Keep artifact upload disabled unless intentionally archiving checkpoints.

## Standard BC vs AWR

Do not mix standard BC and AWR/reward-weighted experiments.

Standard BC:

```text
script: scripts/robocasa_multitask_finetune.py
data:   3-task LeRobot data
loss:   plain BC/action prediction objective
```

AWR/reward-weighted experiments:

```text
script: scripts/robocasa_awr_finetune.py
output: local_outputs/robocasa_awr_retrain
```

AWR-related files observed:

```text
scripts/robocasa_awr_finetune.py
gr00t/data/robocasa_awr.py
gr00t/model/action_head/flow_matching_action_head.py
gr00t/model/transforms.py
```

Important distinction:

```text
Standard 3-task BC samples did not include reward/loss_weight keys.
AWR alpha, clipping, and phase weights can overweight pick/place segments.
Always label AWR outputs clearly and keep output_dir separate.
```

Known AWR container/output examples:

```text
isaac-gr00t-robocasa-train-awr-alpha10-gpu2
  output: /workspace/outputs/awr_alpha10_20k
  host output root: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain

isaac-gr00t-robocasa-train-awr-alpha15-gpu3
  output: /workspace/outputs/awr_alpha15_20k
  host output root: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain
```

## Core Lesson

**The important trick for strict eval is not just setting the same seed.**

`split=pretrain` fixes only the allowed object / layout / style pool. It does not guarantee that two separate eval runs will sample the exact same object, layout, style, fixture references, or object configs.

The strict comparison method is:

1. Sample the task episodes once.
2. Save each episode's `seed` and RoboCasa `ep_meta` into a trial schedule JSON.
3. Reuse that same JSON for every checkpoint.
4. During eval, replay each stored `ep_meta` through RoboCasa `set_ep_meta(...)`.

That is what makes object / layout / style match across checkpoints.

## When To Use Strict Eval

Use strict eval only when exact scene identity matters:

- checkpoint A/B comparison
- reporting fair side-by-side success rates
- debugging whether a policy change really improved behavior on the same scene set
- comparing videos for the same object / layout / style sequence

For quick training progress checks, broad smoke tests, or rough ID sanity checks, normal `split=pretrain` eval is usually enough.

## Strict Eval Rules

When exact object / layout / style control matters, keep all of these fixed:

1. `ROBOCASA_EVAL_SPLIT=pretrain`
2. `N_ENVS=1`
3. Same `ROBOCASA_TRIAL_SCHEDULE_PATH`
4. Same `N_EPISODES`
5. Same `MAX_EPISODE_STEPS`
6. Same `N_ACTION_STEPS`
7. Same video/render settings
8. Change only the checkpoint path

If any of these change, the run is no longer a strict same-scene comparison.

## Trial Schedule JSON

The canonical schedule location used in the session was:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules
```

Canonical schedules that were generated:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceCounterToSink_10eps.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceCounterToStove_10eps.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceMicrowaveToCounter_10eps.json
```

Each schedule stores:

- `env_name`
- `split`
- `base_seed`
- `episodes`
- per-episode `seed`
- per-episode `ep_meta`

The `ep_meta` is the source of truth for strict object / layout / style replay. It includes fields such as:

- `layout_id`
- `style_id`
- `object_cfgs`
- `fixture_refs`
- `cam_configs`
- `init_robot_base_pos`
- `init_robot_base_ori`

## Required Code Support

Strict replay requires these code paths. If the repo was reset or the local folder was lost, check that these files still contain the corresponding functionality.

### `gr00t/eval/simulation.py`

Required behavior:

- `SimulationConfig` has `seed: Optional[int]`
- `SimulationConfig` has `trial_schedule_path: Optional[str]`
- client loads `trial_schedule_path` JSON when provided
- eval requires `n_envs=1` for schedule replay
- before each reset, it finds the underlying RoboCasa env and calls `set_ep_meta(ep_meta)`
- reset uses the stored episode seed when present
- HTTP client path adapts RoboCasa observations/actions for GR00T

The important implementation idea:

```python
target_env.set_ep_meta(ep_meta)
obs, info = vector_env.reset(seed=episode_spec.get("seed"))
```

### `scripts/simulation_service.py`

Required behavior:

- accepts `--trial_schedule_path`
- passes it to `SimulationConfig(trial_schedule_path=...)`
- supports `--http-server`
- supports `--seed`
- supports `--steps_per_render`
- supports `--n_action_steps`
- supports `--max_episode_steps`

### `scripts/generate_robocasa_eval_schedule.py`

Purpose:

- create a fixed trial schedule JSON from RoboCasa by sampling episodes once
- save `seed + ep_meta` for each episode

Example:

```bash
python /home/junhyeong/Value/Isaac-GR00T/scripts/generate_robocasa_eval_schedule.py \
  --env_name robocasa/PickPlaceCounterToSink \
  --n_episodes 10 \
  --split pretrain \
  --seed 12345 \
  --output_path /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceCounterToSink_10eps.json
```

### `scripts/check_robocasa_schedule_replay.py`

Purpose:

- verify that replaying a schedule entry creates the same scene digest twice
- use it before trusting a new schedule for strict comparison

Example:

```bash
python /home/junhyeong/Value/Isaac-GR00T/scripts/check_robocasa_schedule_replay.py \
  --schedule_path /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceCounterToSink_10eps.json
```

Expected result:

```text
MATCH True
```

### `scripts/check_robocasa_benchmark_readiness.py`

Purpose:

- quickly check benchmark prerequisites
- useful after Docker/runtime folder changes

Expected checks:

- repo paths exist
- Docker image exists
- Python imports are available
- optional HTTP `/health` check
- optional `gym.make(...)` env creation check

### `robocasa/wrappers/gym_wrapper.py`

Required local patch:

- default split should be `pretrain`
- constructor should forward `seed` into `create_env(...)`

The important behavior:

```python
seed = kwargs.pop("seed", None)
self.env = create_env(..., seed=seed, split=split, ...)
```

This does not by itself guarantee strict same-scene comparison, but it prevents constructor-time randomness from silently ignoring the requested seed.

## Docker Images

Images used:

```text
isaac-gr00t-robocasa:http-server
isaac-gr00t-robocasa:benchmark
isaac-gr00t-robocasa:smoke
```

The model server image runs:

```text
scripts/inference_service.py --server --http-server
```

The benchmark image runs:

```text
scripts/simulation_service.py --client --http-server
```

The benchmark container connects to the model server over HTTP:

```text
GET  /health
GET  /modality_config
POST /act
```

## Runtime Mounting Policy

The goal was to avoid baking checkpoints, datasets, output videos, or local repos into images.

Preferred pattern:

- mount checkpoint read-only
- mount GR00T repo read-only
- mount RoboCasa repo read-only
- mount robosuite repo read-only
- mount output folder
- mount a shared runtime state folder for Python bootstrap/cache

This avoids rebuilding Docker images or reinstalling Python packages every run.

## Shared Runtime State

Two shared runtime roots were introduced to avoid expensive reinstall loops:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_http_server_runtime
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark_runtime
```

or, in later ad hoc runs:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/_shared_runtime
```

These are disposable runtime folders. They may contain:

- `_delete_me_runtime_home`
- `_delete_me_runtime_cache`
- `.runtime_code`
- `.runtime_repos`
- pip user installs
- generated RoboCasa/robosuite writable copies

They are safe to delete when intentionally clearing bootstrap state, but deleting them means the next eval run may reinstall dependencies.

## Important Bootstrap Fixes

Problems encountered:

- benchmark `OUTPUT_DIR` was changed per run, which caused a fresh `HOME/.local` each time
- benchmark repeatedly reinstalled RoboCasa/robosuite/mujoco/opencv
- video wrapper required `av`, but `av` was not always installed
- model server readiness check was too weak and could miss `tyro` / `inference_service` dependency failures
- benchmark readiness check was too weak and could miss `simulation_service` dependency failures

Fixes:

- move runtime state outside per-run output folders
- set `PYTHONUSERBASE`
- make model server readiness check import the real `scripts.inference_service`
- make benchmark readiness check import the real `scripts.simulation_service`
- include `av` in benchmark bootstrap because video recording imports it
- skip benchmark-side GR00T pip install and use mounted source via `PYTHONPATH`

## Expected Launcher Variables

### Model server

Common variables:

```bash
IMAGE_NAME=isaac-gr00t-robocasa:http-server
CONTAINER_NAME=isaac-gr00t-robocasa-http-gpu1-example
GPU_DEVICE=1
PORT=8011
CHECKPOINT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/.../checkpoint-XXXX
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_http_server/example
RUNTIME_STATE_HOST_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_http_server_runtime
```

Expected script:

```bash
bash /home/junhyeong/Value/Isaac-GR00T/scripts/run_groot_robocasa_http_server.sh
```

### Benchmark

Common variables:

```bash
IMAGE_NAME=isaac-gr00t-robocasa:benchmark
CONTAINER_NAME=isaac-gr00t-robocasa-bench-gpu1-example
GPU_DEVICE=1
PORT=8011
MODEL_HOST=host.docker.internal
ENV_NAME=robocasa/PickPlaceCounterToSink
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/example
RUNTIME_STATE_HOST_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark_runtime
ROBOCASA_EVAL_SPLIT=pretrain
ROBOCASA_EVAL_SEED=12345
ROBOCASA_TRIAL_SCHEDULE_PATH=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/pretrain_seed12345/PickPlaceCounterToSink_10eps.json
N_EPISODES=5
N_ENVS=1
N_ACTION_STEPS=16
MAX_EPISODE_STEPS=400
VIDEO_STEPS_PER_RENDER=1
```

Expected script:

```bash
bash /home/junhyeong/Value/Isaac-GR00T/scripts/run_robocasa_benchmark_container.sh
```

## Video Output

The eval wrapper was adjusted to produce:

- one composite video per episode
- optional per-camera videos
- final names ending in `_outcome0.mp4` or `_outcome1.mp4`

The word `success` was intentionally removed from generated video names because it was visually misleading when a task failed.

Preferred cleanup after each eval:

```bash
find "$OUTPUT_DIR/videos" -maxdepth 1 -type f -name '*.mp4' ! -name '*composite*_outcome*.mp4' -delete
```

This leaves only the side-by-side composite videos.

## Known Result Folders

Historical result folders from the session included:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/ckpt20000_pretrain_3tasks_10eps_400step_20260423_gpu3
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/ckpt2000_pretrain_3tasks_3eps_400step_20260423_gpu2
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/ckpt8000_pretrain_3tasks_3eps_400step_20260424_gpu1
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/awr5000_pretrain_3tasks_10eps_400step_20260424_gpu1
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/awr5000_fixedsink5_20260424_gpu1
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/ckpt6000_fixedsink5_20260424_gpu1
```

Some folders may be missing if local outputs were deleted with the lost folder.

## Known Results

Recorded results from the session:

```text
checkpoint-20000, 3 tasks x 10 episodes:
PickPlaceCounterToSink        2/10
PickPlaceCounterToStove       2/10
PickPlaceMicrowaveToCounter   0/10

checkpoint-2000, 3 tasks x 3 episodes:
PickPlaceCounterToSink        0/3
PickPlaceCounterToStove       0/3
PickPlaceMicrowaveToCounter   0/3

checkpoint-8000, 3 tasks x 3 episodes:
PickPlaceCounterToSink        0/3
PickPlaceCounterToStove       2/3
PickPlaceMicrowaveToCounter   0/3

AWR checkpoint-5000, 3 tasks x 10 episodes:
PickPlaceCounterToSink        4/10
PickPlaceCounterToStove       1/10
PickPlaceMicrowaveToCounter   0/10

Strict fixed Sink 5 episode comparison:
AWR checkpoint-5000           1/5
checkpoint-6000               0/5
```

## Fixed Sink 5-Episode Comparison

The final strict comparison used:

```text
Task: robocasa/PickPlaceCounterToSink
Schedule: PickPlaceCounterToSink_10eps.json
Episodes: first 5
Split: pretrain
N_ENVS: 1
N_ACTION_STEPS: 16
MAX_EPISODE_STEPS: 400
VIDEO_STEPS_PER_RENDER: 1
GPU: 1
```

Compared checkpoints:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_AWR_bs64_20k_20260423_215731/checkpoint-5000
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-6000
```

Result:

```text
AWR checkpoint-5000: 1/5
checkpoint-6000:     0/5
```

This eval confirmed that the schedule JSON method controlled objects well across checkpoints.

## GPU Cleanup

After evals, remove eval containers explicitly.

Common patterns:

```bash
docker rm -f isaac-gr00t-robocasa-bench-...
docker rm -f isaac-gr00t-robocasa-http-...
```

Known stale container that was removed during the session:

```text
isaac-gr00t-robocasa-http-gpu1-awr-ckpt10000
```

Always check before removing containers:

```bash
docker ps --format '{{.Names}}\t{{.Status}}'
nvidia-smi
```

Do not remove unrelated training containers unless explicitly requested.

## Operational Checklist

Use this checklist before starting, stopping, or comparing eval/training runs.

Before stopping GPU containers:

- Check both `docker ps` and `nvidia-smi`; an HTTP inference container can keep a checkpoint and GPU memory alive even if the host output folder was deleted or recreated.
- If the host-side final BC checkpoint is missing, recover it from the live HTTP container before stopping it:

```bash
docker cp isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000:/workspace/model_checkpoint <recovery_target>
```

- Do not identify runs only by port. Ports observed during the session included `8014`, `8015`, `8017`, and `8020`, but the only reliable source is the container command and mounted checkpoint path.
- Do not stop unrelated training containers. Only remove benchmark/server containers whose name, port, and checkpoint path match the current eval.

Before training resume:

- Inspect `checkpoint-*/trainer_state.json`.
- Verify `global_step`, `max_steps`, `save_steps`, and `train_batch_size`.
- If intended batch 64 uses only about `35GB` on A100 80GB, suspect stale `train_batch_size=32` loaded from the checkpoint state.
- Keep `save_total_limit` high while comparing checkpoints, or older eval targets can be deleted automatically.
- Trust `trainer_state.json.global_step` over folder names such as `official_32k`.

Before strict eval:

- Use `N_ENVS=1`.
- Use the same trial schedule JSON.
- Keep `ROBOCASA_EVAL_SPLIT=pretrain` unless intentionally testing OOD/generalization.
- Keep `N_EPISODES`, `MAX_EPISODE_STEPS`, `N_ACTION_STEPS`, render settings, video settings, and task list fixed.
- Change only the checkpoint path for A/B comparisons.
- If object/layout/style equality matters, do not rely on seed alone; require schedule JSON replay with `ep_meta`.

Common failure checks:

- Client cannot reach server: check `--network host`, server bind address `0.0.0.0`, host, and port.
- Readiness falsely passes: import the actual entrypoints, `scripts.inference_service` and `scripts.simulation_service`.
- Video fails: verify `av` is installed inside the benchmark runtime.
- Simulator fails: verify RoboCasa/robosuite/MuJoCo, EGL/GLU, and kitchen assets are present inside the benchmark container/runtime.
- HF model load fails: verify Hugging Face token/cache availability in the model-server runtime.
- Unexpected low success: confirm action transform/rotation convention, camera mapping, `n_action_steps`, episode horizon, and exact object schedule before blaming training.

## Current Recovery Note

At recovery time, `/home/junhyeong/Value/Isaac-GR00T` existed but the previous `docker/` directory and this MD file were absent. The Docker images still existed.

If full functionality is needed again, the likely files to restore or reapply are:

- `docker/groot_robocasa_http_server.Dockerfile`
- `docker/robocasa_benchmark.Dockerfile`
- `docker/robocasa_benchmark.compose.yaml`
- `scripts/run_groot_robocasa_http_server.sh`
- `scripts/run_robocasa_benchmark_container.sh`
- `scripts/generate_robocasa_eval_schedule.py`
- `scripts/check_robocasa_schedule_replay.py`
- `scripts/check_robocasa_benchmark_readiness.py`
- `gr00t/eval/simulation.py` strict schedule replay support
- `gr00t/eval/wrappers/video_recording_wrapper.py` composite/outcome video support
- `robocasa/wrappers/gym_wrapper.py` local seed/split patch in `/home/junhyeong/Value/robocasa_official_v1`

The most important non-obvious behavior to preserve is:

```text
strict checkpoint comparison = same fixed trial schedule JSON + n_envs=1 + checkpoint path is the only variable
```

## Rebuilt N1.5 HTTP Eval Scripts

The restored repo now has a separate N1.5-specific eval path. GR00T-1.6 eval code was used only as a reference for metadata/video/schedule structure because the 1.6 `AutoProcessor` server and `ROBOCASA_PANDA_OMRON` embodiment do not directly load our N1.5 fine-tuned checkpoints.

Files:

- `gr00t/eval/wrappers/robocasa_n15_wrapper.py`
- `scripts/robocasa_n15_http_eval.py`
- `scripts/run_groot_robocasa_http_server.sh`
- `scripts/run_robocasa_n15_eval_container.sh`
- `scripts/run_robocasa_n15_eval_3tasks.sh`

Model server example:

```bash
CHECKPOINT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_alpha10_20k/checkpoint-20000 \
GPU_DEVICE=2 \
PORT=8011 \
CONTAINER_NAME=isaac-gr00t-robocasa-http-awr10-20k \
REPLACE=1 \
bash /home/junhyeong/Value/Isaac-GR00T/scripts/run_groot_robocasa_http_server.sh
```

Single task eval example:

```bash
ENV_NAME=PnPCounterToSink \
GPU_DEVICE=3 \
MODEL_HOST=127.0.0.1 \
PORT=8011 \
SEED=1 \
N_EPISODES=50 \
MAX_EPISODE_STEPS=800 \
VIDEO_FPS=20 \
CONTAINER_NAME=isaac-gr00t-robocasa-bench-awr10-sink \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/awr10_ckpt20000_seed1_50ep \
REPLACE=1 \
bash /home/junhyeong/Value/Isaac-GR00T/scripts/run_robocasa_n15_eval_container.sh
```

Three task eval example:

```bash
RUN_NAME=awr10_ckpt20000_seed1_50ep \
GPU_DEVICE=3 \
MODEL_HOST=127.0.0.1 \
PORT=8011 \
SEED=1 \
N_EPISODES=50 \
REPLACE=1 \
bash /home/junhyeong/Value/Isaac-GR00T/scripts/run_robocasa_n15_eval_3tasks.sh
```

Behavior:

- Uses HTTP `/act` from `scripts/inference_service.py`.
- Creates RoboCasa with `robosuite.make`, `PandaOmron`, three cameras, and the N1.5 RoboCasa data config keys.
- Default object split is `A`, matching RoboCasa demo/pretrain object collection code.
- Generates or reuses a schedule JSON with per-episode `seed` and `ep_meta`.
- Replays each episode with `set_ep_meta(ep_meta)` before reset, so checkpoint A/B comparisons should reuse identical object/layout/style settings.
- Saves only composite videos: `videos/ep{ID}_seed{SEED}_composite_outcome{0|1}.mp4`.
- Saves per-episode metadata under `episodes/ep{ID}.json` and aggregate stats in `summary.json`.

Validation performed after writing these scripts:

```bash
python -c "import pathlib; [compile(pathlib.Path(p).read_text(), p, 'exec') for p in ['scripts/robocasa_n15_http_eval.py','gr00t/eval/wrappers/robocasa_n15_wrapper.py']]; print('syntax ok')"
bash -n scripts/run_groot_robocasa_http_server.sh scripts/run_robocasa_n15_eval_container.sh scripts/run_robocasa_n15_eval_3tasks.sh
```

Not yet validated in this restored session:

- Full MuJoCo/RoboCasa import inside the benchmark container after bootstrap.
- One-episode end-to-end rollout against a live N1.5 HTTP server.
