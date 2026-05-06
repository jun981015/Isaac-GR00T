# RoboCasa 3-Task GR00T Finetune Handoff

Last reconstructed: 2026-04-26 KST

This file was reconstructed from the active Codex session memory after the working files under `Value/Isaac-GR00T` were lost/recreated. Treat exact paths and commands as the best known record from the session, and verify surviving artifacts before deleting any remaining containers.

## 0. Immediate Recovery Priority

The original training output directory appears to be missing from the current host repo:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke
```

However, a surviving HTTP inference container still has the final 20K checkpoint mounted:

```text
container: isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000
image:     isaac-gr00t-robocasa:http-server
status:    running
model:     /workspace/model_checkpoint
host source recorded by docker inspect:
  /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

If the host checkpoint directory is gone, do not stop this HTTP container until attempting to recover `/workspace/model_checkpoint` from inside it.

This is the highest-priority recovery action. Docker may still keep the mounted checkpoint visible to the running process even when the host tree was recreated or hidden. Copy first, then decide what to stop/remove.

Suggested recovery command:

```bash
mkdir -p /home/junhyeong/Value/Isaac-GR00T/recovered_outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1
docker cp isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000:/workspace/model_checkpoint \
  /home/junhyeong/Value/Isaac-GR00T/recovered_outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

After copying, check for files such as `model.safetensors`, `trainer_state.json`, optimizer/scheduler states, config files, and action-head weights.

## 1. Original Goal

Initial objective from the user, preserved in spirit:

```text
Isaac Groot를 Robocasa 데이터에 파인튜닝.
LoRA 미사용.
VLM 백본은 freeze.
그냥 학습.
전체 데이터가 아니라 task 3개의 데이터만 train/eval.
데이터는 LeRobot 형식.
data/robocasa_lerobot 아래 데이터 사용.
학습 및 eval task:
  PnPCounterToSink
  PnPCounterToStove
  PnPMicrowaveToCounter
Isaac Groot 모델은 Value/Isaac-GR00T 참고.
n1.5 release 사용.
Value/robocasa 안의 md 및 python 파일 참고.
우선 smoke로 loss가 떨어지는지 확인 후, multi-task 학습과 eval 준비.
```

Later clarification:

```text
사용자가 의도한 것은 거의 System2/VLM만 freeze하고,
action side인 diffusion transformer(System1/action head)는 학습하는 세팅.
LoRA는 쓰지 않음.
```

## 2. Hardware And Runtime Context

Original GPU snapshot supplied by user:

```text
NVIDIA-SMI 535.288.01
CUDA Version: 12.2
4 x NVIDIA A100 80GB PCIe
MIG Disabled
```

Earlier usage at the time:

```text
GPU0: A100 80GB, VLLM EngineCore, about 77.8GB used
GPU1: A100 80GB, python, about 34.0GB used
GPU2: A100 80GB, python, about 34.2GB used
GPU3: A100 80GB, python, about 33.9GB used
```

Training experiments showed:

```text
single GPU, batch 64: about 50.5GB VRAM on A100 80GB
step speed: about 4.8-5.2 sec/step
batch 64 was viable on one A100 80GB
```

Two-GPU DDP was considered and briefly used:

```text
per-GPU batch 32
global batch 64
host GPUs 1,2
```

But the final 8K -> 20K continuation was single-GPU batch 64.

## 3. Data

Host data root:

```text
/home/junhyeong/data/robocasa_lerobot
```

Container mount:

```text
/home/junhyeong/data/robocasa_lerobot -> /workspace/data/robocasa_lerobot:ro
```

Tasks used:

```text
/workspace/data/robocasa_lerobot/PnPCounterToSink
/workspace/data/robocasa_lerobot/PnPCounterToStove
/workspace/data/robocasa_lerobot/PnPMicrowaveToCounter
```

Other RoboCasa LeRobot task directories that existed locally and may be useful later:

```text
/home/junhyeong/data/robocasa_lerobot/PnPCabToCounter
/home/junhyeong/data/robocasa_lerobot/PnPCounterToCab
/home/junhyeong/data/robocasa_lerobot/PnPCounterToMicrowave
/home/junhyeong/data/robocasa_lerobot/PnPCounterToSink
/home/junhyeong/data/robocasa_lerobot/PnPCounterToStove
/home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter
/home/junhyeong/data/robocasa_lerobot/PnPSinkToCounter
```

Do not confuse this handoff with a possible 7-task extension. The completed standard BC run in this session was the 3-task setup above.

Original HDF5 sources also survived at the time of audit:

```text
/home/junhyeong/data/robocasa/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams.hdf5
/home/junhyeong/data/robocasa/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5
```

Conversion/helper files that survived:

```text
/home/junhyeong/Value/robocasa/convert_robocasa_to_groot_lerobot_v21.py
/home/junhyeong/Value/robocasa/robocasa_n15_data_config.py
/home/junhyeong/Value/robocasa/robocasa_n15_conversion_notes.md
```

Observed train split summary from logs:

```text
PnPCounterToSink:        39 train episodes, 16536 visible steps
PnPCounterToStove:       39 train episodes, 11940 visible steps
PnPMicrowaveToCounter:   39 train episodes, 11735 visible steps
train dataset length:    40211
train dataloader length: 629 with batch 64
```

The full converted task directories were believed to contain 50 episodes each at 20 FPS. The training split exposed 39 train episodes per task in the run above.

Mixture sampling weights observed:

```text
PnPCounterToSink:        0.41123075775285367
PnPCounterToStove:       0.29693367486508665
PnPMicrowaveToCounter:   0.2918355673820596
```

Important data interpretation:

```text
Training samples are shuffled at sample/start-index level.
Each sample keeps its internal temporal structure:
  observation at a sampled time
  3 camera images/state/language
  action chunk target after that time
It is not frame-by-frame video order training.
Validation/offline eval would normally sample from held-out split similarly.
Simulator rollout eval is separate and was not part of the training loop.
```

## 4. Model And Intended Trainable Parts

Base model:

```text
nvidia/GR00T-N1.5-3B
local HF snapshot:
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
container path:
/workspace/hf_cache/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

The intended official-like finetune configuration:

```text
LoRA: not used
VLM/LLM backbone: frozen
Vision tower: frozen
Action head projector: trainable
Action diffusion model / DiT: trainable
Embodiment tag: new_embodiment
Data config: robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
```

Log lines confirming intended trainable state:

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
Trainable intent: tune_llm=False, tune_visual=False, tune_projector=True, tune_diffusion_model=True
```

Important correction made during the session:

```text
Early experiments likely froze the action diffusion/DiT unintentionally.
After reviewing GR00T README/code/paper discussion, we changed the default/command path so action diffusion model is trained.
The corrected setting is the one used for the official-like multi-task 20K run.
```

## 5. 3-Camera RoboCasa Assumption

RoboCasa data already uses multiple camera streams. The concern was whether GR00T data config/action transform assumed a different camera count or modality naming.

Final practical conclusion:

```text
Keeping 3 cameras is reasonable because inference on RoboCasa had already worked with the same multi-camera observation structure.
The model/data config can accept the mapped image keys as long as modality.json/config and transforms align.
The main risk is not "3 cameras are impossible"; the risk is mismatched modality names, ordering, shape, or normalization.
```

Checks that should be redone after file recovery:

```text
1. Load one sample from each task.
2. Print image keys/shapes for all camera streams.
3. Print state/action shapes.
4. Verify action mask and max_action_dim.
5. Verify action chunk horizon matches GR00TTransform/data config.
6. Confirm no reward/loss_weight key in standard BC samples.
```

Known RoboCasa data config details to re-verify from `/home/junhyeong/Value/robocasa/robocasa_n15_data_config.py`:

```text
RobocasaKitchenPnPDataConfig
3 camera streams
state groups total around 53D
RoboCasa action around 12D
GR00T max_action_dim 32 with padding/mask
action horizon/chunk effectively 16 for GR00T N1.5 action head
```

The exact camera/action key order matters for rollout quality, especially eef position, eef rotation, gripper, base, and control mode ordering.

## 6. Code That Was Added Or Modified

The current repo may have lost these files. This is the reconstruction inventory.

### 6.1 `scripts/robocasa_multitask_finetune.py`

Purpose:

```text
Train GR00T N1.5 on selected RoboCasa LeRobot task directories.
Support multi-task manifest/split.
Run official-like Trainer finetune while freezing VLM and training projector + action diffusion model.
Support resume from explicit checkpoint.
Support single-GPU and DDP.
```

Known behavior/flags:

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

Important code details:

```text
BooleanOptionalAction was used for tune_diffusion_model, balance_dataset_weights, balance_trajectory_weights.
Default should be tune_diffusion_model=True.
Default should be balance_dataset_weights=True and balance_trajectory_weights=True.
```

Functions/guards that were added:

```text
_rank()
_world_size()
_validate_expected_tasks()
```

DDP/single-GPU behavior:

```text
set_single_gpu_visible(args.gpu_index) should only run when _world_size() == 1.
Rank-gate manifest copying/log printing where needed.
Print trainable intent clearly before training.
```

Resume behavior:

```text
--resume-from-checkpoint should override auto-discovery.
Do not rely only on Trainer auto-resume when batch-size metadata changed.
```

### 6.2 `scripts/run_robocasa_smoke_container.sh`

Purpose:

```text
Start the smoke/development container with data, repo, model cache, outputs, and Value/robocasa mounted.
Support GPU_DEVICE as a single GPU or comma-separated list.
Map host GPU selection into container CUDA_VISIBLE_DEVICES correctly.
```

Important mounts used:

```text
/home/junhyeong/data/robocasa_lerobot -> /workspace/data/robocasa_lerobot:ro
/home/junhyeong/.cache/huggingface -> /workspace/hf_cache
/home/junhyeong/Value/Isaac-GR00T -> /workspace/Isaac-GR00T
/home/junhyeong/Value/robocasa -> /workspace/Value/robocasa:ro
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke -> /workspace/outputs
```

WandB mounts to include in future:

```text
-v /home/junhyeong/.netrc:/workspace/outputs/home/.netrc:ro
-v /home/junhyeong/.config/wandb:/workspace/outputs/home/.config/wandb:ro
```

### 6.3 `scripts/run_robocasa_ddp_bs32x2_official_like.sh`

Purpose:

```text
Official-like 2-GPU DDP launcher.
Used host GPUs 1,2.
Per-GPU batch 32, global batch 64.
```

### 6.4 `robocasa_n15_data_config`

The command used:

```text
--data-config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
```

This module likely came from helper path:

```text
/workspace/Value/robocasa
/home/junhyeong/Value/robocasa
```

Make sure `PYTHONPATH` includes:

```text
/workspace/Isaac-GR00T:/workspace/Value/robocasa
```

### 6.5 AWR/Reward Files Not Part Of Standard BC

Another developer had reward/AWR work in the same repo/container ecosystem:

```text
scripts/robocasa_awr_finetune.py
gr00t/data/robocasa_awr.py
gr00t/model/action_head/flow_matching_action_head.py
```

The action head had optional `loss_weight` support:

```text
if "loss_weight" in action_input:
    ...
```

Important distinction:

```text
Our standard 3-task BC run used scripts/robocasa_multitask_finetune.py.
Sample inspection showed no loss_weight key and no reward key.
Therefore reward-weighted/AWR BC was not active in our standard run.
But the shared repo was not clean, so future reproducibility should isolate BC and AWR work.
```

Additional AWR detail from review:

```text
robocasa_awr_finetune.py used heuristic pseudo reward / loss weighting, not the plain official-like BC objective.
The heuristic was related to gripper/action/qpos-style signals, not a simulator rollout reward in the standard BC loop.
AWR alpha, clipping, and phase weights can overweight specific pick/place segments.
Always label AWR runs clearly as AWR/heuristic and keep output_dir separate from standard BC.
```

Surviving AWR-local files after the loss:

```text
scripts/robocasa_awr_finetune.py
gr00t/model/transforms.py local change: passes loss_weight through GR00TTransform
gr00t/model/action_head/flow_matching_action_head.py local change: per-sample weighted loss
local_outputs/robocasa_awr_retrain
```

AWR alpha retrain containers observed after the loss:

```text
isaac-gr00t-robocasa-train-awr-alpha10-gpu2
  image: isaac-gr00t-robocasa:smoke
  output: /workspace/outputs/awr_alpha10_20k
  host output root: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain

isaac-gr00t-robocasa-train-awr-alpha15-gpu3
  image: isaac-gr00t-robocasa:smoke
  output: /workspace/outputs/awr_alpha15_20k
  host output root: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain
```

These are not the completed standard 3-task BC result.

### 6.6 LeRobot Policy Port Note

One review note mentioned a possible separate LeRobot policy integration under another path such as `/home/junhyeong/latest/lerobot`. That was not the path used for the completed run described in this document.

For this session's completed 20K result, the authoritative path was:

```text
Isaac-GR00T repo script: scripts/robocasa_multitask_finetune.py
GR00T data config:       robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
Data root:               /home/junhyeong/data/robocasa_lerobot
Container image:         isaac-gr00t-robocasa:smoke
```

If future work moves to a `lerobot-train --policy.type=groot` integration, document it as a new pipeline and do not mix its checkpoint layout with the Isaac-GR00T Trainer checkpoints here.

## 7. Docker Images And Containers

Images verified on 2026-04-26:

```text
isaac-gr00t-robocasa:smoke        5c354ecbd400   19GB
isaac-gr00t-robocasa:http-server  2a8a19b2461c   14.5GB
isaac-gr00t-robocasa:benchmark    80f8c730bc42   13.8GB
topreward:cu128                   271e3abba653   31.9GB
```

The standard 3-task BC training container we used:

```text
name:  isaac-gr00t-robocasa-smoke
image: isaac-gr00t-robocasa:smoke
cmd:   tail -f /dev/null
```

This container was intentionally stopped and removed after the 20K run completed to save space:

```text
docker stop isaac-gr00t-robocasa-smoke
docker rm isaac-gr00t-robocasa-smoke
```

Important runtime distinction:

```text
isaac-gr00t-robocasa:smoke was used for finetuning/offline checks.
It should not be assumed to contain a complete RoboCasa/MuJoCo simulator runtime for rollout video.
Rollout/video eval used or should use separate simulator-capable benchmark/http containers.
```

Do not confuse with:

```text
isaac-gr00t-robocasa-smoke-gpu3
```

That was a similar long-lived smoke container used by another workflow/developer and was not the standard 3-task BC training container we removed.

Surviving relevant containers as of 2026-04-26:

```text
isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000
  image: isaac-gr00t-robocasa:http-server
  status: running
  purpose: HTTP inference server for official-like 20K checkpoint
  port inside command: 8017
  model_path: /workspace/model_checkpoint
  data_config: robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
  embodiment_tag: new_embodiment
  denoising_steps: 4
  recovery priority: highest; copy /workspace/model_checkpoint before stopping

isaac-gr00t-robocasa-http-gpu1-robocasa_AWR10-ckpt10000
  image: isaac-gr00t-robocasa:http-server
  status: running
  purpose: AWR-related inference, not standard BC
  observed port: 8020

isaac-gr00t-robocasa-http-gpu2-awr20-ckpt10000
  image: isaac-gr00t-robocasa:http-server
  status: running
  purpose: AWR-related inference, not standard BC
  observed port: 8015

isaac-gr00t-robocasa-http-gpu1-awr10-ckpt20000-seed1check
  image: isaac-gr00t-robocasa:http-server
  status: running
  purpose: AWR-related inference, not standard BC
  observed port: 8014

isaac-gr00t-robocasa-train-awr-alpha15-gpu3
  image: isaac-gr00t-robocasa:smoke
  status at inspection: exited
  purpose: AWR alpha15 retrain, not standard BC

isaac-gr00t-robocasa-train-awr-alpha10-gpu2
  image: isaac-gr00t-robocasa:smoke
  status at inspection: exited
  purpose: AWR alpha10 retrain, not standard BC
```

Important HTTP server command from surviving standard BC container:

```bash
python scripts/inference_service.py \
  --server \
  --http-server \
  --host 0.0.0.0 \
  --port 8017 \
  --model_path /workspace/model_checkpoint \
  --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --embodiment_tag new_embodiment \
  --denoising_steps 4
```

## 8. WandB

Host login files:

```text
/home/junhyeong/.netrc
/home/junhyeong/.config/wandb/settings
```

Container home used by original smoke container:

```text
/workspace/outputs/home
```

Writable WandB dirs used:

```text
WANDB_DIR=/workspace/outputs/home/wandb
WANDB_CACHE_DIR=/workspace/outputs/home/.cache/wandb
```

WandB file ownership caveat:

```text
Container users may leave /workspace/outputs/home/wandb or cache files owned by numeric/nobody users on the host.
Use explicit writable mounted dirs and avoid putting WandB output under read-only repo mounts.
Large checkpoint artifact upload was not needed for this workflow; keep artifact upload disabled unless intentionally archiving checkpoints.
```

Final standard BC run was logged to:

```text
entity:  RwHlabs
project: groot_robocasa_finetune_3_task
run id:  f1kw1sz2
url:     https://wandb.ai/RwHlabs/groot_robocasa_finetune_3_task/runs/f1kw1sz2
```

Mistakes/corrections during setup:

```text
Initial run was accidentally sent to junhyeong/robocasa_groot.
RwHlabs-org failed because WandB does not allow logging directly to organization entity.
Correct team entity was RwHlabs.
Correct project name requested by user: groot_robocasa_finetune_3_task.
```

Probe runs may exist in WandB:

```text
RwHlabs/robocasa_groot: entity_probe_codex, etc.
RwHlabs/groot_robocasa_finetune: intermediate mistaken project.
RwHlabs/groot_robocasa_finetune_3_task: final desired project.
```

## 9. Training Timeline And Results

### 9.1 Early Smoke / Single-Task

Initial experiments:

```text
robocasa_smoke_step1
  path then: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_smoke_step1
  checkpoint: checkpoint-1
  size then: about 19G
  character: first 1-step smoke, low preservation value

robocasa_train_gpu2_run1
  path then: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_train_gpu2_run1
  checkpoints: checkpoint-30, checkpoint-40, checkpoint-50
  losses then: about 0.1976, 0.18, 0.2514
  character: single-task 50-step early validation, low preservation value
```

These were later considered deletable.

### 9.2 Multi-Task Baseline Before Fix

```text
robocasa_multitask_10k_run1
  path then: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_10k_run1
  checkpoints: checkpoint-8000, checkpoint-9000, checkpoint-10000
  losses then:
    checkpoint-8000 around 0.0565
    checkpoint-9000 around 0.0446
    checkpoint-10000 around 0.079
  caveat: action DiT was probably frozen, so not the intended final setup
```

This was later considered deletable.

### 9.3 Corrected Official-Like Run

Main output directory then:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1
```

Despite the name containing `ddp_bs32x2_official_32k`, the final completed continuation was single-GPU batch 64 from `checkpoint-8000` to `checkpoint-20000`.

Early DDP segment:

```text
host GPUs: 1,2
per-GPU batch: 32
global batch: 64
saved: checkpoint-2000
```

Single-GPU true batch 64 segment:

```text
host GPU1 first, then host GPU2 for later continuation
batch-size: 64
gradient_accumulation_steps: 1
VRAM: about 50.5GB on A100 80GB
speed: about 5 sec/step
```

Saved checkpoints from corrected run:

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

Note on desired save intervals:

```text
User asked for 10K, 15K, 20K during one continuation.
Actual observed saves included 10K, 12K, 14K, 16K, 20K, because trainer_state/resume save_steps metadata interacted with configured save_steps.
This should be treated carefully in future runs.
```

Trainer save semantics:

```text
save_steps is Trainer global optimizer step, not sample count or epoch count.
Resume continues from global_step, so checkpoint names are global.
Checkpoint saving can briefly stall throughput.
If save_total_limit is too small, older checkpoints may be auto-deleted. Keep it high while doing eval comparisons.
```

Final completed result:

```text
global step: 20000 / 20000
final train_loss: 0.00845177420154214
final step loss at 20000: about 0.0087
train_runtime: 60236.8694 sec
train_samples_per_second: 21.249
train_steps_per_second: 0.332
epoch: 31.8
wandb run: https://wandb.ai/RwHlabs/groot_robocasa_finetune_3_task/runs/f1kw1sz2
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

## 10. Commands Used For Final Continuation

The final corrected run used a command equivalent to:

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

Final log path then:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/train_single_gpu_bs64_gpu2_from_ckpt8000_to20k_wandb_RwHlabs_groot_robocasa_finetune_3_task.log
```

## 11. HF Trainer Resume Caveat

Critical issue discovered:

```text
When resuming, HuggingFace Trainer may load train_batch_size from checkpoint-*/trainer_state.json.
This can override the intended dataloader batch size.
```

Actual incident:

```text
checkpoint-2000/trainer_state.json had train_batch_size=32.
We tried to resume single-GPU batch 64, but VRAM stayed around 35GB, indicating actual batch 32.
The run produced a checkpoint-2020 probe.
We stopped it, deleted checkpoint-2020, backed up trainer_state.json, and edited train_batch_size 32 -> 64.
Then true batch 64 used about 50.5GB.
```

Files then:

```text
backup:
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-2000/trainer_state.json.bs32_backup

edited:
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-2000/trainer_state.json
```

Future rule:

```text
Before changing batch size on resume, inspect trainer_state.json.
Check train_batch_size, save_steps, max_steps, global_step.
Confirm VRAM after restart.
For A100 80GB, true batch 64 should be roughly 50GB, not 35GB.
```

## 12. Multi-Task / Normalization Caveat

Concern discussed:

```text
GR00T/LeRobot metadata might combine task datasets into one metadata/normalization context.
If so, action/state normalization may not be task-specific.
```

Clarified interpretation:

```text
Multi-task mixing is a legitimate VLA objective.
The concern is not that multi-task is conceptually wrong.
The concern is whether this code path accidentally treats separate task metadata as one dataset in a way that causes unintended normalization or modality mismatch.
```

Practical stance:

```text
3-task multi-task training was acceptable for smoke/BC adaptation.
For rigorous results, verify per-task metadata, modality.json, normalization stats, and split manifests.
If using task-specific normalization intentionally, implement and document it explicitly.
Aggregate train loss is not enough to judge task success; run task-wise offline eval and simulator rollout.
Single-task loss and multi-task loss are from different distributions and should not be directly compared without context.
```

Settings that influence the mixture:

```text
balance_dataset_weights=True
balance_trajectory_weights=True
metadata percentile/stat mixing may be weighted-average depending on the dataset/config code path
```

## 13. Eval / Inference Status

Training loop did not include simulator eval:

```text
No periodic eval inside Trainer.
No simulator rollout in the main training process.
Eval/video was handled or planned as separate HTTP server + RoboCasa benchmark rollout.
```

Known inference/HTTP setup:

```text
server container:
isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000

server command:
python scripts/inference_service.py --server --http-server --host 0.0.0.0 --port 8017 \
  --model_path /workspace/model_checkpoint \
  --data_config robocasa_n15_data_config:RobocasaKitchenPnPDataConfig \
  --embodiment_tag new_embodiment \
  --denoising_steps 4
```

User reported that even earlier/frozen-action-ish settings could make the robot move toward intended behavior and grasp objects, though motion was slower/gummier than expert data.

Rotation representation concern:

```text
There was a concern about rotation representation/action transform correctness.
No obvious failure was seen in quick rollout behavior, but this still needs explicit verification against RoboCasa action convention and GR00T transform inverse.
```

## 14. Loss Curve / Monitoring

TensorBoard logs were written under:

```text
/workspace/outputs/robocasa_multitask_ddp_bs32x2_official_32k_run1/runs
```

Host path then:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/runs
```

PNG loss curves were generated during earlier runs, including:

```text
loss_curve_single_gpu_bs64_to10k.png
```

For future runs, prefer WandB from the start:

```text
WANDB_ENTITY=RwHlabs
WANDB_PROJECT=groot_robocasa_finetune_3_task
--report-to wandb
```

## 15. Cleanup Already Done

Deleted or considered low-value:

```text
robocasa_smoke_step1
robocasa_train_gpu2_run1
robocasa_multitask_10k_run1
checkpoint-2020 probe
robocasa_multitask_dryrun
robocasa_smoke_run
pycache_check
local_outputs/*/home
robocasa_http_server* test folders
robocasa_benchmark/smoke_*
pickplace_counter_to_sink_multicam_gpu1_128
miniconda3/envs/robocasa
```

Important:

```text
The final 20K standard BC checkpoint should not have been deleted intentionally.
If now missing, recover from the surviving HTTP container before stopping it.
```

## 16. Current Repo State After Loss

As of 2026-04-26, current `/home/junhyeong/Value/Isaac-GR00T` appears to be a recreated or partially restored repo.

Observed current facts:

```text
/home/junhyeong/Value/Isaac-GR00T.deleted_shell_20260426_150148 exists but contains only .codex and .git at shallow inspection.
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke is missing.
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain exists.
scripts/robocasa_multitask_finetune.py is currently absent.
scripts/robocasa_awr_finetune.py is present.
git status shows modified gr00t/model/action_head/flow_matching_action_head.py and gr00t/model/transforms.py, plus untracked local_outputs and scripts/robocasa_awr_finetune.py.
```

Do not assume current repo exactly matches the code used for the standard BC run. Recreate and review the standard BC script before launching another official-like BC training run.

Current upstream/repo audit from an agent:

```text
current repo appears to be upstream NVIDIA/Isaac-GR00T detached HEAD
HEAD observed: 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96
remote observed: https://github.com/NVIDIA/Isaac-GR00T.git
```

Permission caveat:

```text
Some local output/cache dirs may be owned by nobody/nogroup because they were written from Docker.
Examples seen around robocasa_awr_retrain/home and wandb artifact cache.
Use Docker or proper permissions before cleanup.
```

## 17. Rebuild Checklist

1. Recover or recreate `scripts/robocasa_multitask_finetune.py`.
2. Recreate container launcher script with mounts and WandB mounts.
3. Confirm `robocasa_n15_data_config:RobocasaKitchenPnPDataConfig` imports from `/home/junhyeong/Value/robocasa`.
4. Load one sample from each of the 3 tasks and print keys/shapes.
5. Confirm standard BC dataloader samples contain no `reward` and no `loss_weight`.
6. Confirm model trainable flags: VLM frozen, projector trainable, DiT/action diffusion trainable.
7. Do a batch 1 CPU/GPU smoke if possible.
8. Do a short GPU smoke with `--batch-size 1` and verify loss decreases.
9. For batch 64 resume, inspect `trainer_state.json` before launch.
10. Start full run with WandB entity `RwHlabs` and project `groot_robocasa_finetune_3_task`.
11. After each checkpoint, verify directory exists and `trainer_state.json.global_step` matches.
12. For eval, run HTTP server against a checkpoint and launch RoboCasa rollout separately.
13. For recovery from a live HTTP server, copy `/workspace/model_checkpoint` before stopping the container.
14. For future checkpoint layout docs, record whether it is an Isaac-GR00T Trainer checkpoint or a separate LeRobot policy checkpoint.

## 18. Known Good Final Checkpoint Path

Best known final checkpoint path before file loss:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_smoke/robocasa_multitask_ddp_bs32x2_official_32k_run1/checkpoint-20000
```

If missing on host, likely recovery source:

```text
docker cp isaac-gr00t-robocasa-http-gpu3-robocasa_official32k-ckpt20000:/workspace/model_checkpoint <recovery_target>
```

Best known WandB record:

```text
https://wandb.ai/RwHlabs/groot_robocasa_finetune_3_task/runs/f1kw1sz2
```

## 19. Practical Tips

```text
Use A100 80GB single GPU batch 64 for this setup if only one GPU is available.
Expected VRAM for true batch 64 is about 50GB.
If VRAM is around 35GB after resume, suspect Trainer loaded old train_batch_size.
Keep docker HTTP checkpoint containers alive until checkpoint directories are backed up.
Do not mix AWR/reward scripts with standard BC unless explicitly intended.
Use WandB from the beginning; retrofitting is possible through TensorBoard sync but noisier.
Do not rely on folder names like official_32k to infer what actually ran; inspect logs and trainer_state.
For exact checkpoint step, trust checkpoint-*/trainer_state.json global_step.
For standard BC, check that no sample contains loss_weight/reward.
```

## 20. Conda Finetune Smoke After Docker Replacement

On 2026-04-30 KST, the Docker-equivalent conda env `gr00t-train` was validated with an actual GR00T RoboCasa finetune path.

Historical note: the one-step check used a temporary smoke-only copy of `scripts/gr00t_finetune.py` that skipped checkpoint/final model saving. That temporary script was intentionally removed after validation; use `scripts/gr00t_finetune.py` or `scripts/robocasa_awr_finetune.py` for actual training.

Run scope:

```text
env: gr00t-train
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
report_to: tensorboard
```

Observed result:

```text
3 datasets initialized and mixed.
GR00T checkpoint shards loaded.
Backbone/VLM frozen.
Projector frozen.
Action diffusion/DiT trainable.
One train step completed.
train_loss: 0.6214311122894287
final model save skipped.
```

Smoke output:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/conda_smoke/groot_smoke_1step_20260430_223117.log
/home/junhyeong/Value/Isaac-GR00T/local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/trainer_state.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/experiment_cfg/metadata.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/conda_smoke/groot_smoke_1step_20260430_223117/runs/
```

No `model*.safetensors`, `optimizer.pt`, or large checkpoint files were written. The output directory was about 68KB.
