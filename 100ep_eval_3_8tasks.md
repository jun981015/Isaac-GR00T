# 100ep Eval: 3task / 8task Models

Last updated: 2026-04-30 10:45 KST

Purpose: keep the 100-episode, 3-action-seed RoboCasa eval layout reproducible and prevent creating new output roots for the same experiment.

## Fixed Rules

- Env schedule seed is fixed to `seed1`.
- Action/noise seeds are `1`, `2`, `3`; these are stored as `action_seed_1`, `action_seed_2`, `action_seed_3`.
- Use `N_EPISODES=100`, `N_ENVS=8`, `N_ACTION_STEPS=16`.
- 3task max env step is `800`.
- Use existing output roots below. Do not create new timestamped roots unless the eval setting intentionally changes.
- Use `SKIP_EXISTING=1` when resuming so completed episode JSONs are not regenerated.

## Schedule Roots

3task schedules:

```text
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPMicrowaveToCounter_100eps.json
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToSink_100eps.json
local_outputs/robocasa_benchmark/schedules/n15_seed1_100ep/seed1/PnPCounterToStove_100eps.json
```

Composite 8task schedules currently available:

```text
local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430/seed1/PrepareCoffee_100eps.json
local_outputs/robocasa_benchmark/schedules/composite_baseline50k_100ep_envseed1_original_20260430/seed1/MicrowaveThawing_100eps.json
```

Critical8 atomic schedule currently available:

```text
local_outputs/robocasa_benchmark/schedules/critical8_100ep_envseed1_20260430/seed1/CoffeeSetupMug_100eps.json
```

## Output Roots

3task eval output root:

```text
local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045
```

Composite baseline output root:

```text
local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020
```

Composite AWR output root:

```text
local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915
```

Expected folder pattern:

```text
<output_root>/<model_name>/<checkpoint_label>/action_seed_<1|2|3>/<task>/episodes/ep*.json
```

Example:

```text
local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045/awr_pm16_alpha50_20k/checkpoint-20000/action_seed_1/PnPCounterToSink/episodes
```

## 3task Models

Tasks:

```text
PnPMicrowaveToCounter
PnPCounterToSink
PnPCounterToStove
```

Models:

| model_name | checkpoint | notes |
|---|---|---|
| `baseline_noawr_20k` | `local_outputs/robocasa_awr_retrain/baseline_noawr_20k/checkpoint-20000` | no AWR, regular flow matching |
| `awr_pm8_alpha25_20k` | `local_outputs/robocasa_awr_retrain/awr_pm8_alpha25_20k/checkpoint-20000` | 3task AWR, pick/place window pm8, alpha 25 |
| `awr_pm16_alpha50_20k` | `local_outputs/robocasa_awr_retrain/awr_pm16_alpha50_20k/checkpoint-20000` | 3task AWR, pick/place window pm16, alpha 50 |

Current 3task status snapshot:

| model | task | action_seed | progress | success |
|---|---|---:|---:|---:|
| `baseline_noawr_20k` | `PnPMicrowaveToCounter` | 1 | 100/100 | 26 |
| `baseline_noawr_20k` | `PnPMicrowaveToCounter` | 2 | 100/100 | 30 |
| `baseline_noawr_20k` | `PnPMicrowaveToCounter` | 3 | 100/100 | 28 |
| `baseline_noawr_20k` | `PnPCounterToSink` | 1 | 20/100 | 11 |
| `awr_pm8_alpha25_20k` | `PnPMicrowaveToCounter` | 1 | 100/100 | 29 |
| `awr_pm8_alpha25_20k` | `PnPMicrowaveToCounter` | 2 | 100/100 | 26 |
| `awr_pm8_alpha25_20k` | `PnPMicrowaveToCounter` | 3 | 100/100 | 37 |
| `awr_pm8_alpha25_20k` | `PnPCounterToSink` | 1 | 9/100 | 4 |
| `awr_pm16_alpha50_20k` | `PnPMicrowaveToCounter` | 1 | 100/100 | 27 |
| `awr_pm16_alpha50_20k` | `PnPMicrowaveToCounter` | 2 | 100/100 | 31 |
| `awr_pm16_alpha50_20k` | `PnPMicrowaveToCounter` | 3 | 100/100 | 31 |
| `awr_pm16_alpha50_20k` | `PnPCounterToSink` | 1 | 11/100 | 8 |

Remaining 3task eval work:

- Finish `PnPCounterToSink` action seeds `1,2,3` for all three models.
- Run `PnPCounterToStove` action seeds `1,2,3` for all three models.
- Keep using the same 3task output root.

## 8task / Composite Models

Composite tasks currently being tested:

```text
PrepareCoffee
MicrowaveThawing
```

8task/critical models:

| model_name | checkpoint | notes |
|---|---|---|
| `baseline_8task_b50k` | baseline 8task checkpoint-50000 server on GPU1, port 8091 | composite baseline |
| `awr_critical8_groupalpha_clip18_50k` | `local_outputs/robocasa_awr_retrain/awr_critical8_groupalpha_clip18_50k_gpu2/checkpoint-50000` | critical 8task AWR |

Current composite status snapshot:

| model | task | action_seed | progress | success |
|---|---|---:|---:|---:|
| `baseline_8task_b50k` | `PrepareCoffee` | 1 | 100/100 | 3 |
| `baseline_8task_b50k` | `MicrowaveThawing` | 1 | 100/100 | 13 |
| `baseline_8task_b50k` | `PrepareCoffee` | 2 | 24/100 | 0 |
| `awr_critical8_groupalpha_clip18_50k` | `PrepareCoffee` | 1 | 25/100 | 3 |

Remaining composite eval work:

- Continue `baseline_8task_b50k` for `PrepareCoffee` action seed 2, then seed 3.
- Continue `baseline_8task_b50k` for `MicrowaveThawing` action seeds 2 and 3.
- Continue `awr_critical8_groupalpha_clip18_50k` for `PrepareCoffee` action seeds 1, 2, 3.
- Add `MicrowaveThawing` for `awr_critical8_groupalpha_clip18_50k` only if this model is intended to be compared on composite tasks.

## Ignore / Do Not Use

This path was produced by a mistaken eval: `awr_pm16_alpha70_cm06_20k` is a 3task model but was accidentally evaluated on `PrepareCoffee`.

```text
local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915/awr_pm16_alpha70_cm06_20k/checkpoint-20000/action_seed_1/PrepareCoffee
```

Do not include it in composite statistics.

## Current Running Containers

Snapshot at 2026-04-30 10:45 KST:

| GPU | role | container | model | task | action_seed | num_envs |
|---|---|---|---|---|---:|---:|
| GPU1 | server | `robocasa-zmq-composite-b50k-smoke-gpu1` | `baseline_8task_b50k` | composite server | - | - |
| GPU1 | eval | `robocasa-bench-baseline-b50k-PrepareCoffee-as2-env8` | `baseline_8task_b50k` | `PrepareCoffee` | 2 | 8 |
| GPU2 | server | `robocasa-zmq-composite-awr-critical8-test-gpu2` | `awr_critical8_groupalpha_clip18_50k` | composite server | - | - |
| GPU2 | eval | `robocasa-bench-critical8-ckpt50000-PrepareCoffee-as1-env8` | `awr_critical8_groupalpha_clip18_50k` | `PrepareCoffee` | 1 | 8 |
| GPU2 | server | `robocasa-zmq-sinkstove-baseline_noawr_20k-gpu2` | `baseline_noawr_20k` | 3task server | - | - |
| GPU2 | eval | `robocasa-bench-sinkstove-baseline_noawr_20k-as1-PnPCounterToSink-g2` | `baseline_noawr_20k` | `PnPCounterToSink` | 1 | 8 |
| GPU3 | server | `robocasa-zmq-sinkstove-extra-awr_pm16_alpha50_20k-gpu3` | `awr_pm16_alpha50_20k` | 3task server | - | - |
| GPU3 | eval | `robocasa-bench-sinkstove-extra-awr_pm16_alpha50_20k-as1-PnPCounterToSink-g3` | `awr_pm16_alpha50_20k` | `PnPCounterToSink` | 1 | 8 |
| GPU3 | server | `robocasa-zmq-sinkstove-extra-awr_pm8_alpha25_20k-gpu3` | `awr_pm8_alpha25_20k` | 3task server | - | - |
| GPU3 | eval | `robocasa-bench-sinkstove-extra-awr_pm8_alpha25_20k-as1-PnPCounterToSink-g3` | `awr_pm8_alpha25_20k` | `PnPCounterToSink` | 1 | 8 |

## Resume Commands

Use the existing wrapper scripts. The important part is to keep `OUTPUT_DIR`, `SCHEDULE_DIR`, `SCHEDULE_PATH`, `SEED`, and `SKIP_EXISTING=1` consistent.

3task eval wrapper:

```bash
bash scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
```

3task server wrapper:

```bash
bash scripts/run_groot_robocasa_zmq_server.sh
```

3task sequential launcher currently used for remaining Sink/Stove:

```bash
setsid bash scripts/launch_20k_sink_stove_gpu23_seq.sh \
  > local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045/_sink_stove_gpu23_logs/nohup.log \
  2>&1 < /dev/null &
```

Do not launch this blindly if manual extra eval containers are already running for the same model/task/seed. First check Docker and episode folders.

## Progress Check

Use this command from repo root:

```bash
python - <<'PY'
import json, pathlib, time
roots=[
 pathlib.Path('local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045'),
 pathlib.Path('local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915'),
 pathlib.Path('local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020'),
]
for root in roots:
 print('\\nROOT', root)
 if not root.exists():
  continue
 for epdir in sorted(root.rglob('episodes')):
  files=sorted(epdir.glob('ep*.json'))
  if not files:
   continue
  succ=0; maxep=-1; mt=0
  for f in files:
   d=json.loads(f.read_text())
   succ += int(bool(d.get('success')))
   maxep=max(maxep, int(d.get('episode_idx', -1)))
   mt=max(mt, f.stat().st_mtime)
  print(f'{epdir.relative_to(root)} | {len(files)}/100 | success={succ} | max_ep={maxep} | last={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt))}')
PY
```

## Output Hygiene

- Reuse the three output roots listed above.
- Do not create per-retry roots such as `*_rerun_*`, `*_test_*`, or new timestamps unless the setting changed.
- If a container fails halfway, relaunch with the exact same `OUTPUT_DIR` and `SKIP_EXISTING=1`.
- If a wrong model/task combination is launched, stop it and mark the generated folder as ignored in this file instead of mixing it into statistics.
