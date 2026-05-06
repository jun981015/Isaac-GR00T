# Codex Handoff: RoboCasa GR00T AWR Training and Parallel Eval

Last updated: 2026-04-30 KST

This is a detailed handoff for a future Codex session. The core goals are:

```text
1. Train baseline and AWR variants of GR00T N1.5 on RoboCasa LeRobot tasks.
2. Evaluate policies under reproducible, replayable RoboCasa environment schedules.
3. Use parallel ZMQ eval to make 100ep x multiple-action-seed evaluation tractable.
4. Keep output roots stable so partial evals can be resumed with SKIP_EXISTING=1.
```

## Repo and Data Paths

```text
main repo:
  /home/junhyeong/Value/Isaac-GR00T

RoboCasa helper/config repo:
  /home/junhyeong/Value/robocasa

LeRobot data:
  /home/junhyeong/data/robocasa_lerobot

HF base model cache:
  /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Important markdowns:

```text
JUNHYEONG_ROBOCASA_AWR_EVAL_HANDOFF.md
JUNHYEONG_ROBOCASA_ZMQ_PARALLEL_EVAL_HANDOFF.md
100ep_eval_3_8tasks.md
ROBOCASA_TASK_FINETUNE_HANDOFF.md
JUNHYEONG_GIT_TODO.md
```

## Git / Worktree Situation

Main worktree:

```text
/home/junhyeong/Value/Isaac-GR00T
```

Known related worktrees:

```text
/home/junhyeong/Value/Isaac-GR00T-compact-eval
```

There was also ZMQ eval work that has largely been merged/copied into the main repo. Do not assume the tree is clean. Before editing:

```bash
git status --short
```

Do not revert user or other-agent changes. There are active eval outputs under `local_outputs/`.

## Training Families

### 3-Task Models

Tasks:

```text
PnPCounterToSink
PnPCounterToStove
PnPMicrowaveToCounter
```

Primary checkpoints:

```text
local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000
local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000
local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000
```

Meaning:

```text
baseline_noawr_20k:
  regular flow matching loss, no AWR weight

awr_pm8_alpha25_20k:
  pick/place critical window radius 8, alpha 25

awr_pm16_alpha50_20k:
  pick/place critical window radius 16, alpha 50
```

### 8-Task / Composite Models

Tasks:

```text
PrepareCoffee
MicrowaveThawing
CoffeeSetupMug
CoffeePressButton
OpenSingleDoor
CloseSingleDoor
PnPCounterToMicrowave
TurnOnMicrowave
```

Primary checkpoints:

```text
local_outputs/robocasa_awr_retrain/baseline_8task_b50k/checkpoint-50000
local_outputs/robocasa_awr_retrain/awr_critical8_groupalpha_clip18_50k_gpu2/checkpoint-50000
```

The exact baseline path may be mounted through a running server or located under `local_outputs/robocasa_awr_retrain`. Verify before launching.

## AWR Implementation

Main script:

```text
scripts/robocasa_awr_finetune.py
```

Model/data plumbing:

```text
gr00t/model/transforms.py
gr00t/model/action_head/flow_matching_action_head.py
```

The intended loss behavior:

```text
If loss_weight exists in the batch/action_input, multiply the per-sample flow matching loss by that weight.
If no loss_weight exists, training is regular baseline flow matching.
```

The AWR weight is heuristic:

```text
progress_delta is assigned over an episode.
critical events receive a specified total progress mass.
non-critical frames share the remaining progress mass.
For each training sample, average progress deltas over the action chunk.
Then use exp(alpha * mean_delta), clipped by awr_clip_max.
```

Generic formula:

```text
mean_delta = mean(progress_delta[t : t + action_chunk_delta_count])
loss_weight = clip(exp(alpha * mean_delta), 0, awr_clip_max)
```

3-task PnP heuristic:

```text
pick:
  first sustained closing gripper action segment
  detect first stuck gripper qpos window inside the closing segment
  stuck window length around 10 frames
  qpos stuck threshold around 0.001

place:
  first sustained opening action after pick
  physical opening lags action by roughly 0.5s
  final used offset for retraining was place center = opening_start + 8 frames
```

3-task progress mass:

```text
pick total mass = 0.3
place total mass = 0.3
remaining non-critical mass = 0.4
```

Composite critical-frame intent:

```text
PrepareCoffee:
  mug pick
  mug place
  button press

MicrowaveThawing:
  door open
  object pick
  object place
  door close
  button press

CoffeeSetupMug:
  pick/place

CoffeePressButton:
  press

OpenSingleDoor / CloseSingleDoor:
  door contact/open/close critical point

PnPCounterToMicrowave:
  pick/place

TurnOnMicrowave:
  press
```

Composite AWR used grouped alpha/clip settings. The most important current checkpoint is:

```text
awr_critical8_groupalpha_clip18_50k_gpu2/checkpoint-50000
```

## Eval Reproducibility

Do not rely on just `seed=1`. Use schedule replay.

A schedule JSON stores per episode:

```text
episode_idx
seed
RoboCasa ep_meta
```

Eval should:

```text
1. Load schedule JSON.
2. Use the episode's stored seed.
3. Apply schedule_episode["ep_meta"] to the env.
4. Reset.
5. Run policy.
```

This is what should align across policies:

```text
object category / asset
object placement
layout/style
fixture refs
language prompt
initial scene
```

Important caveat:

```text
Do not use ep_meta copied out of completed episode JSON as the canonical schedule source.
RoboCasa may mutate ep_meta during reset/rollout.
Use the original schedule JSON under local_outputs/robocasa_benchmark/schedules.
```

## Parallel ZMQ Eval

Core files:

```text
scripts/robocasa_n15_zmq_eval.py
scripts/robocasa_n15_zmq_parallel_eval.py
scripts/run_groot_robocasa_zmq_server.sh
scripts/run_robocasa_n15_zmq_eval_container.sh
scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
gr00t/eval/robot.py
gr00t/eval/service.py
gr00t/eval/wrappers/robocasa_n15_wrapper.py
```

Architecture:

```text
ZMQ policy server:
  loads one GR00T checkpoint once
  serves policy calls on a port

Parallel eval container:
  starts N RoboCasa env workers
  workers collect obs and step envs
  main process batches policy requests
  action chunks are distributed back to workers
```

Default current eval parameters:

```text
N_EPISODES=100
N_ENVS=8
N_ACTION_STEPS=16
policy_batch_mode=lockstep
policy_batch_wait_sec=0.05
CAMERA_WIDTH=256
CAMERA_HEIGHT=256
POLICY_IMAGE_SIZE=128
VIDEO_SOURCE=obs
VIDEO_RENDER_SIZE=0
VIDEO_SCALE=1
VIDEO_STEPS_PER_RENDER=4
STREAM_VIDEO=1
WRITE_VIDEO=1
SKIP_EXISTING=1
```

Max episode steps:

```text
3-task PnP eval:
  MAX_EPISODE_STEPS=800

Composite eval:
  MAX_EPISODE_STEPS=1000
```

Action seeds:

```text
SEED=1,2,3
```

These are action/noise seeds, not environment schedule seeds. The environment schedule seed is fixed by the schedule JSON path.

## Schedule Roots

3-task 100ep schedules:

```text
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPMicrowaveToCounter_100eps.json
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToSink_100eps.json
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToStove_100eps.json
```

Composite schedules:

```text
local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430/seed1/PrepareCoffee_100eps.json
local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430/seed1/MicrowaveThawing_100eps.json
```

## Output Roots

3-task 100ep root:

```text
local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045
```

Composite baseline root:

```text
local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020
```

Composite AWR root:

```text
local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915
```

Expected output pattern:

```text
<output_root>/<model_name>/<checkpoint_label>/action_seed_<1|2|3>/<task>/episodes/ep*.json
<output_root>/<model_name>/<checkpoint_label>/action_seed_<1|2|3>/<task>/videos/*.mp4
```

Avoid creating new timestamped output roots unless the eval setting intentionally changes.

## Current Eval Snapshot

Snapshot from 2026-04-30 around 15:38 KST.

3-task:

```text
baseline_noawr_20k/checkpoint-20000:
  PnPMicrowaveToCounter:
    seed1 26/100
    seed2 30/100
    seed3 28/100
  PnPCounterToSink:
    seed1 51/100
    seed2 50/100 complete
    seed3 17/32 running
  PnPCounterToStove:
    not started or not yet recorded in this campaign

awr_pm8_alpha25_20k/checkpoint-20000:
  PnPMicrowaveToCounter:
    seed1 29/100
    seed2 26/100
    seed3 37/100
  PnPCounterToSink:
    seed1 40/100 complete
    seed2 47/92 running
  PnPCounterToStove:
    not started or not yet recorded in this campaign

awr_pm16_alpha50_20k/checkpoint-20000:
  PnPMicrowaveToCounter:
    seed1 27/100
    seed2 31/100
    seed3 31/100
  PnPCounterToSink:
    seed1 52/100 complete
    seed2 51/89 running
  PnPCounterToStove:
    not started or not yet recorded in this campaign
```

Composite:

```text
baseline_8task_b50k/checkpoint-50000:
  PrepareCoffee:
    seed1 3/100
    seed2 0/100
  MicrowaveThawing:
    seed1 13/100

awr_critical8_groupalpha_clip18_50k/checkpoint-50000:
  PrepareCoffee:
    seed1 4/100
    seed2 1/65 running
  MicrowaveThawing:
    seed1 6/54 running
```

## Composite Sub-Task Diagnostics

Episode JSONs include `success_diagnostics` for composite tasks.

PrepareCoffee keys:

```text
contact_check
coffee_machine_turned_on
gripper_obj_far
gripper_button_far
```

MicrowaveThawing keys:

```text
obj_in_microwave
button_pressed
gripper_obj_far
```

Use this to report not only final success but also failure mode.

Sub-condition summary command:

```bash
python - <<'PY'
import json, pathlib, collections
roots=[
 pathlib.Path('local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915'),
 pathlib.Path('local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020'),
]
for root in roots:
 print('\\nROOT', root.name)
 if not root.exists(): continue
 for epdir in sorted(root.rglob('episodes')):
  files=sorted(epdir.glob('ep*.json'))
  if not files: continue
  if 'awr_pm16_alpha70_cm06_20k' in str(epdir): continue
  cond_total=collections.Counter(); cond_true=collections.Counter(); succ=0
  for f in files:
   d=json.loads(f.read_text()); succ += int(bool(d.get('success')))
   diag=d.get('success_diagnostics') or {}
   for k,v in diag.items():
    if k in {'env_name','success','missing_conditions','error'}: continue
    if isinstance(v, bool):
     cond_total[k]+=1; cond_true[k]+=int(v)
  print(f'{epdir.relative_to(root)} | episodes={len(files)} | success={succ}/{len(files)} ({succ/len(files):.3f})')
  for k in sorted(cond_total):
   print(f'  {k}: {cond_true[k]}/{cond_total[k]} ({cond_true[k]/cond_total[k]:.3f})')
PY
```

## Progress Check Command

Use from repo root:

```bash
python - <<'PY'
import json, pathlib, time
roots=[
 pathlib.Path('local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045'),
 pathlib.Path('local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915'),
 pathlib.Path('local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020'),
]
now=time.time()
for root in roots:
 print('\\nROOT', root)
 if not root.exists(): continue
 for epdir in sorted(root.rglob('episodes')):
  files=sorted(epdir.glob('ep*.json'))
  if not files: continue
  if 'awr_pm16_alpha70_cm06_20k' in str(epdir): continue
  succ=0; maxep=-1; mt=0; first=10**18
  for f in files:
   d=json.loads(f.read_text())
   succ += int(bool(d.get('success')))
   maxep=max(maxep, int(d.get('episode_idx', -1)))
   st=f.stat(); mt=max(mt, st.st_mtime); first=min(first, st.st_mtime)
  n=len(files); age=(now-mt)/60
  if n<100 or age < 180:
   span=mt-first if n>1 else 0
   speed='' if n<=1 else f' avg={(span/(n-1))/60:.2f}min/ep'
   print(f'{epdir.relative_to(root)} | {n}/100 | success={succ} ({succ/n:.3f}) | max_ep={maxep} | last={time.strftime("%H:%M:%S", time.localtime(mt))} | age={age:.1f}min{speed}')
PY
```

## Current Running Containers Snapshot

Snapshot from 2026-04-30 around 15:38 KST:

```text
robocasa-bench-sinkstove-baseline_noawr_20k-as3-PnPCounterToSink-g2
robocasa-bench-critical8-ckpt50000-MicrowaveThawing-as1-env8
robocasa-bench-sinkstove-extra-awr_pm16_alpha50_20k-as2-PnPCounterToSink-g3
robocasa-bench-sinkstove-extra-awr_pm8_alpha25_20k-as2-PnPCounterToSink-g3
robocasa-bench-critical8-ckpt50000-PrepareCoffee-as2-env8
robocasa-zmq-sinkstove-extra-awr_pm8_alpha25_20k-gpu3
robocasa-zmq-sinkstove-extra-awr_pm16_alpha50_20k-gpu3
robocasa-zmq-sinkstove-baseline_noawr_20k-gpu2
robocasa-zmq-composite-awr-critical8-test-gpu2
robocasa-zmq-composite-b50k-smoke-gpu1
```

Do not stop these without checking whether the user wants to preserve current eval progress.

## Launch Pattern

Example 3-task eval with an existing ZMQ policy server:

```bash
CONTAINER_NAME=robocasa-bench-sinkstove-extra-awr_pm16_alpha50_20k-as2-PnPCounterToSink-g3 \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045/awr_pm16_alpha50_20k/checkpoint-20000/action_seed_2 \
GPU_DEVICE=3 \
MODEL_HOST=127.0.0.1 \
PORT=8106 \
ENV_NAME=PnPCounterToSink \
SEED=2 \
N_EPISODES=100 \
N_ENVS=8 \
N_ACTION_STEPS=16 \
MAX_EPISODE_STEPS=800 \
VIDEO_FPS=20 \
VIDEO_SOURCE=obs \
VIDEO_RENDER_SIZE=0 \
VIDEO_SCALE=1 \
VIDEO_STEPS_PER_RENDER=4 \
CAMERA_WIDTH=256 \
CAMERA_HEIGHT=256 \
POLICY_IMAGE_SIZE=128 \
STREAM_VIDEO=1 \
WRITE_VIDEO=1 \
SKIP_EXISTING=1 \
REGENERATE_SCHEDULE=0 \
BOOTSTRAP_DEPS=0 \
REPLACE=1 \
SCHEDULE_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep \
SCHEDULE_PATH=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToSink_100eps.json \
bash scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
```

Example composite eval with existing AWR critical8 server:

```bash
CONTAINER_NAME=robocasa-bench-critical8-ckpt50000-MicrowaveThawing-as1-env8 \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915/awr_critical8_groupalpha_clip18_50k/checkpoint-50000/action_seed_1 \
GPU_DEVICE=1 \
MODEL_HOST=127.0.0.1 \
PORT=8092 \
ENV_NAME=MicrowaveThawing \
SEED=1 \
N_EPISODES=100 \
N_ENVS=8 \
N_ACTION_STEPS=16 \
MAX_EPISODE_STEPS=1000 \
VIDEO_FPS=20 \
VIDEO_SOURCE=obs \
VIDEO_RENDER_SIZE=0 \
VIDEO_SCALE=1 \
VIDEO_STEPS_PER_RENDER=4 \
CAMERA_WIDTH=256 \
CAMERA_HEIGHT=256 \
POLICY_IMAGE_SIZE=128 \
STREAM_VIDEO=1 \
WRITE_VIDEO=1 \
SKIP_EXISTING=1 \
REGENERATE_SCHEDULE=0 \
BOOTSTRAP_DEPS=0 \
REPLACE=1 \
SCHEDULE_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430 \
SCHEDULE_PATH=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430/seed1/MicrowaveThawing_100eps.json \
bash scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
```

## Compact Checkpoint Work

Scripts:

```text
scripts/compact_groot_checkpoint.py
scripts/compact_nonfinal_checkpoints.py
```

Compact eval worktree:

```text
/home/junhyeong/Value/Isaac-GR00T-compact-eval
```

Compact checkpoint concept:

```text
Changed fine-tuned tensors are stored in model-changed.safetensors.
Unchanged base tensors are mapped to symlinked base model shards from the HF cache.
model.safetensors.index.json is rewritten so from_pretrained can still load the checkpoint.
```

Important:

```text
Do not delete the HF cache snapshot while compact checkpoints are needed.
Docker eval needs the HF cache mounted at the same absolute path.
Use compact-aware server wrappers from the compact worktree if evaluating compact checkpoints.
```

Known tested result:

```text
compact checkpoint load worked in Docker with GR00T_N1_5.from_pretrained
single ckpt size approximately 13G -> 2.9G
robocasa_awr_retrain total approximately 561G -> 215G after compacting non-final ckpts
```

## Known Ignore Path

This was a mistaken eval: a 3-task model was evaluated on `PrepareCoffee`.

```text
local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915/awr_pm16_alpha70_cm06_20k/checkpoint-20000/action_seed_1/PrepareCoffee
```

Exclude it from composite statistics.

## Immediate Next Steps

1. Let the currently running eval containers finish unless the user asks to stop them.
2. Re-run the progress check and update `100ep_eval_3_8tasks.md` after current runs complete.
3. Continue missing 3-task evals:
   `PnPCounterToSink seed3` for all 3 models and `PnPCounterToStove seed1/2/3` for all 3 models.
4. Continue composite eval:
   `PrepareCoffee seed2/3` and `MicrowaveThawing seed1/2/3` for baseline/AWR as needed.
5. Always use existing output roots and `SKIP_EXISTING=1`.
6. When reporting composite, include both final success and sub-condition success.
