# Junhyeong RoboCasa AWR Training / Eval Handoff

This file records the current Isaac-GR00T RoboCasa AWR training and evaluation setup so the next Codex/Claude session can continue without reconstructing context.

## Repo And Important Paths

```text
repo:        /home/junhyeong/Value/Isaac-GR00T
data config: /home/junhyeong/Value/robocasa/robocasa_n15_data_config.py
dataset:     /home/junhyeong/data/robocasa_lerobot
HF cache:    /home/junhyeong/.cache/huggingface
```

Main scripts:

```text
scripts/robocasa_awr_finetune.py
scripts/run_groot_robocasa_http_server.sh
scripts/run_robocasa_n15_eval_container.sh
scripts/robocasa_n15_http_eval.py
scripts/run_new_window_alpha_eval_grid.sh
```

Important model-side AWR plumbing:

```text
gr00t/model/transforms.py
gr00t/model/action_head/flow_matching_action_head.py
```

## Training Data Tasks

All current training runs use exactly these three LeRobot RoboCasa tasks:

```text
/workspace/data/robocasa_lerobot/PnPCounterToSink
/workspace/data/robocasa_lerobot/PnPCounterToStove
/workspace/data/robocasa_lerobot/PnPMicrowaveToCounter
```

Host equivalents:

```text
/home/junhyeong/data/robocasa_lerobot/PnPCounterToSink
/home/junhyeong/data/robocasa_lerobot/PnPCounterToStove
/home/junhyeong/data/robocasa_lerobot/PnPMicrowaveToCounter
```

All AWR runs use:

```text
base model: /workspace/hf_cache/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
data config: robocasa_n15_data_config:RobocasaKitchenPnPDataConfig
embodiment: new_embodiment
batch size: 64
trainable: diffusion model / DiT only
frozen: LLM, visual encoder, action projector
```

## AWR Labeling And Weighting

The AWR implementation is in `scripts/robocasa_awr_finetune.py`.

Training sample weight:

```text
progress_delta per episode:
  pick event total mass  = 0.3
  place event total mass = 0.3
  non-event total mass   = 0.4

chunk mean:
  mean_delta = mean(progress_delta[t : t + action_chunk_delta_count])
  action_chunk_delta_count = 15

loss weight:
  weight = clip(exp(alpha * mean_delta), 0, awr_clip_max)
  awr_clip_max = 2.0
```

Pick/place heuristic:

```text
pick:
  first sustained closing gripper action segment
  detect first gripper gap stuck window inside closing segment
  stuck window = 10 frames
  stuck threshold = 0.001

place:
  first sustained opening gripper action after pick
  place center = opening_start + place_offset
  place_offset = 8 frames
```

Window variants:

```text
old awr_alpha10_20k:
  alpha=10
  pick_radius/place_radius default old setting

old awr_alpha15_20k:
  alpha=15
  pick_radius/place_radius default old setting

new pm8_alpha25:
  alpha=25
  pick_radius=8
  place_radius=8
  max_steps=20000
  save_steps=5000

new pm16_alpha50:
  alpha=50
  pick_radius=16
  place_radius=16
  max_steps=20000
  save_steps=5000
```

Currently also running as temporary GPU occupancy jobs:

```text
temporary_pm8_alpha25_100k_gpu1_save200k
temporary_pm16_alpha50_100k_gpu2_save200k
```

These temporary jobs are not primary experiment outputs.

## Training Outputs

Primary completed checkpoints:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_alpha10_20k/checkpoint-20000
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_alpha15_20k/checkpoint-20000
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000
```

Long-running / temporary:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_50k_gpu3
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/temporary_pm8_alpha25_100k_gpu1_save200k
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/temporary_pm16_alpha50_100k_gpu2_save200k
```

## Eval Tasks And Fixed Settings

Eval tasks:

```text
PnPCounterToSink
PnPCounterToStove
PnPMicrowaveToCounter
```

Eval runtime settings:

```text
seed: 1
n_episodes: 50
episode indices: 0-49
n_action_steps: 16
max_episode_steps: 800
video_fps: 20
video_render_size: 512
video_steps_per_render: 4
video_source: obs
stream_video: 1
obj_instance_split: A
layout_style_ids: 1:1,2:2,4:4,6:9,7:10
```

For old policy eval, outputs are organized under:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/n15_policy_grid_seed1_50ep_hires_20260427_0030
```

Layout:

```text
{weight}/ckpt20000/{task}/episodes
{weight}/ckpt20000/{task}/videos
```

where `{weight}` is:

```text
awr10
awr15
baseline
```

For new window/alpha eval:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428
```

Layout:

```text
pm8_alpha25/ckpt20000/{task}/episodes
pm8_alpha25/ckpt20000/{task}/videos
pm16_alpha50/ckpt20000/{task}/episodes
pm16_alpha50/ckpt20000/{task}/videos
```

## How Eval Reproducibility Is Fixed

The key mechanism is strict schedule replay. Do not rely only on `seed=1`.

Strict schedule files:

```text
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPCounterToSink_50eps.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPCounterToStove_50eps.json
/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPMicrowaveToCounter_50eps.json
```

Each schedule stores per-episode:

```text
episode_idx
seed
RoboCasa ep_meta
```

During eval, `scripts/robocasa_n15_http_eval.py`:

```text
1. loads the schedule
2. creates or reuses RoboCasa env
3. calls reseed_env(env, seed)
4. calls set_ep_meta(env, schedule_episode["ep_meta"])
5. runs env.reset()
```

This is what aligns:

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
Do not use ep_meta copied from eval result episodes/epXXX.json as the canonical replay source.
RoboCasa may mutate ep_meta during reset/rollout.
Use the schedule JSON as the source of truth.
```

The eval wrapper was patched to support:

```text
SKIP_EXISTING=1
SCHEDULE_PATH=/explicit/path/to/*_50eps.json
```

This allows interrupted evals to resume while preserving the fixed scene set.

## Completed Old Eval Results

All old ckpt20000 evals below are complete for 50 episodes.

```text
Task: PnPCounterToSink
  awr10     24/50 = 48.0%
  awr15     24/50 = 48.0%
  baseline  23/50 = 46.0%

Task: PnPCounterToStove
  awr10     28/50 = 56.0%
  awr15     25/50 = 50.0%
  baseline  24/50 = 48.0%

Task: PnPMicrowaveToCounter
  awr10      9/50 = 18.0%
  awr15     10/50 = 20.0%
  baseline  11/50 = 22.0%
```

Microwave old results were originally produced under `microwave_fast_reuse_obs_*` folders, then copied into the grid root above for easier task grouping.

## Current New Window/Alpha Eval Status

Currently evaluating:

```text
pm8_alpha25  checkpoint-20000
pm16_alpha50 checkpoint-20000
```

As of the last check:

```text
pm8_alpha25:
  PnPCounterToSink done 27/50, success 12/27 = 44.4%
  PnPCounterToStove done 0/50
  PnPMicrowaveToCounter done 0/50

pm16_alpha50:
  PnPCounterToSink done 28/50, success 15/28 = 53.6%
  PnPCounterToStove done 0/50
  PnPMicrowaveToCounter done 0/50
```

Current running containers observed:

```text
isaac-gr00t-robocasa-bench-pm8_alpha25-ckpt20000-PnPCounterToSink-newwindow
isaac-gr00t-robocasa-bench-pm16_alpha50-ckpt20000-PnPCounterToSink-newwindow
isaac-gr00t-robocasa-http-pm8-alpha25-ckpt20000-seed1
isaac-gr00t-robocasa-http-pm16-alpha50-ckpt20000-seed1
```

The intended pipeline is:

```text
PnPCounterToSink -> PnPCounterToStove -> PnPMicrowaveToCounter
```

Script:

```bash
bash scripts/run_new_window_alpha_eval_grid.sh
```

Note: a previous `nohup` attempt wrote PID `3` and did not persist. The foreground run did start the current eval and is waiting on Docker containers. If the controlling Codex session dies, check Docker containers and output files before relaunching.

## Useful Status Commands

Eval progress:

```bash
python - <<'PY'
import json
from pathlib import Path
root=Path('/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428')
for label in ['pm8_alpha25','pm16_alpha50']:
    print('\\n'+label)
    for task in ['PnPCounterToSink','PnPCounterToStove','PnPMicrowaveToCounter']:
        p=root/label/'ckpt20000'/task/'episodes'
        items=[]
        for f in sorted(p.glob('ep*.json')) if p.exists() else []:
            d=json.loads(f.read_text())
            items.append((int(f.stem[2:]), bool(d.get('success'))))
        succ=sum(s for _,s in items)
        print(task, len(items), succ, f'{succ/len(items):.3f}' if items else 'nan')
PY
```

Docker status:

```bash
docker ps --format '{{.Names}} {{.Status}}' | grep -E 'newwindow|http-pm|temporary|train-awr'
```

GPU status:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

## Rerun / Resume Eval Commands

Start policy servers for new window/alpha checkpoints:

```bash
CHECKPOINT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000 \
GPU_DEVICE=1 PORT=8041 \
CONTAINER_NAME=isaac-gr00t-robocasa-http-pm8-alpha25-ckpt20000-seed1 \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428/pm8_alpha25/http_ckpt20000 \
BOOTSTRAP_DEPS=1 REPLACE=1 \
bash scripts/run_groot_robocasa_http_server.sh

CHECKPOINT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000 \
GPU_DEVICE=2 PORT=8042 \
CONTAINER_NAME=isaac-gr00t-robocasa-http-pm16-alpha50-ckpt20000-seed1 \
OUTPUT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428/pm16_alpha50/http_ckpt20000 \
BOOTSTRAP_DEPS=1 REPLACE=1 \
bash scripts/run_groot_robocasa_http_server.sh
```

Resume all new window/alpha evals:

```bash
bash scripts/run_new_window_alpha_eval_grid.sh
```

Because `SKIP_EXISTING=1` is set inside that script, completed episodes are skipped.

## Download From Local PC

SSH config alias is `junhyeong50`.

Old 3-policy grid:

```bash
mkdir -p n15_policy_50ep_3task
rsync -avP \
  junhyeong50:/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/n15_policy_grid_seed1_50ep_hires_20260427_0030/ \
  ./n15_policy_50ep_3task/
```

New window/alpha eval:

```bash
mkdir -p new_window_alpha_eval
rsync -avP \
  junhyeong50:/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428/ \
  ./new_window_alpha_eval/
```
