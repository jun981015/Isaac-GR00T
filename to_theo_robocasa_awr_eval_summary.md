# RoboCasa GR00T AWR / Eval Summary for Theo

Last updated: 2026-04-30 KST

## One-Line Summary

We built a RoboCasa GR00T N1.5 finetuning/evaluation pipeline focused on comparing baseline behavior cloning against AWR-style flow-matching loss weighting, and we made the simulator eval reproducible and faster through schedule replay plus parallel ZMQ rollout.

## What We Trained

The first experiment family uses three RoboCasa pick-and-place tasks:

```text
PnPCounterToSink
PnPCounterToStove
PnPMicrowaveToCounter
```

The models currently compared for this 3-task setting are:

```text
baseline_noawr_20k/checkpoint-20000
awr_pm8_alpha25_20k/checkpoint-20000
awr_pm16_alpha50_20k/checkpoint-20000
```

The later composite/critical-frame experiment expanded to eight tasks:

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

The main composite AWR checkpoint is:

```text
awr_critical8_groupalpha_clip18_50k/checkpoint-50000
```

There is also an 8-task baseline:

```text
baseline_8task_b50k/checkpoint-50000
```

## AWR Idea

The AWR implementation is not simulator-reward AWR. It is a heuristic weighting of the GR00T flow-matching loss.

For each trajectory, we identify important frames such as pick, place, door contact, or button press. We then assign progress mass around those critical events. The per-training-sample weight is based on the average progress delta inside the action chunk:

```text
mean_delta = mean(progress_delta over action chunk)
loss_weight = clip(exp(alpha * mean_delta), max=clip_max)
```

For the 3-task PnP setting, pick/place are detected from gripper action and gripper qpos:

```text
pick:
  sustained closing action, then first stuck gripper qpos segment

place:
  sustained opening action after pick, with an offset for delayed physical opening
```

For composite tasks, critical frames are task-specific:

```text
PrepareCoffee:
  mug pick, mug place, coffee-machine button press

MicrowaveThawing:
  door open, object pick, object place, door close, microwave button press

Atomic tasks:
  pick/place for PnP-like tasks
  press/contact points for button/door tasks
```

## What Changed in Eval

The original eval was too slow and not reproducible enough. The key changes were:

1. We stopped relying on seed alone.
2. We now use strict schedule replay.
3. We added a parallel ZMQ eval path.

The schedule JSON stores the fixed per-episode environment state:

```text
episode_idx
seed
RoboCasa ep_meta
```

Using the same schedule is the main mechanism for comparing two policies on the same scene, object, asset, prompt, and placement.

The parallel eval architecture is:

```text
one policy server
multiple RoboCasa env workers
batched policy calls through ZMQ
episode results saved as JSON
videos saved from observation frames
```

The current standard eval setting is:

```text
N_EPISODES=100
N_ENVS=8
N_ACTION_STEPS=16
action seeds = 1, 2, 3
3-task max env steps = 800
composite max env steps = 1000
```

## Why Action Seeds Matter

We found that action sampling/noise can change rollout outcomes even under the same environment schedule. Therefore evaluation is done across three action seeds:

```text
action_seed_1
action_seed_2
action_seed_3
```

The environment schedule seed is fixed separately. The intended comparison is:

```text
same env schedule
same episode set
same task
different policy checkpoint
multiple action seeds
```

## Current Eval State Snapshot

Snapshot from 2026-04-30 around 15:38 KST.

3-task eval root:

```text
local_outputs/robocasa_benchmark/n15_20k_100ep_actionseed123_env8_20260429_1045
```

Current active or recently updated 3-task results:

```text
baseline_noawr_20k / PnPCounterToSink / seed2:
  100/100, success 50

baseline_noawr_20k / PnPCounterToSink / seed3:
  32/100, success 17

awr_pm8_alpha25_20k / PnPCounterToSink / seed1:
  100/100, success 40

awr_pm8_alpha25_20k / PnPCounterToSink / seed2:
  92/100, success 47

awr_pm16_alpha50_20k / PnPCounterToSink / seed1:
  100/100, success 52

awr_pm16_alpha50_20k / PnPCounterToSink / seed2:
  89/100, success 51
```

Completed `PnPMicrowaveToCounter` 100ep results:

```text
baseline_noawr_20k:
  seed1 26/100
  seed2 30/100
  seed3 28/100

awr_pm8_alpha25_20k:
  seed1 29/100
  seed2 26/100
  seed3 37/100

awr_pm16_alpha50_20k:
  seed1 27/100
  seed2 31/100
  seed3 31/100
```

Composite eval roots:

```text
local_outputs/robocasa_benchmark/composite_baseline50k_100ep_envseed1_actionseed123_original_20260430_0020
local_outputs/robocasa_benchmark/composite_awr2_envseed1_actionseed123_env8_20260430_0915
```

Composite snapshot:

```text
baseline_8task_b50k / PrepareCoffee / seed1:
  100/100, success 3

baseline_8task_b50k / PrepareCoffee / seed2:
  100/100, success 0

baseline_8task_b50k / MicrowaveThawing / seed1:
  100/100, success 13

awr_critical8_groupalpha_clip18_50k / PrepareCoffee / seed1:
  100/100, success 4

awr_critical8_groupalpha_clip18_50k / PrepareCoffee / seed2:
  65/100, success 1

awr_critical8_groupalpha_clip18_50k / MicrowaveThawing / seed1:
  54/100, success 6
```

For composite tasks, we also record sub-condition diagnostics. Example:

```text
PrepareCoffee:
  contact_check
  coffee_machine_turned_on
  gripper_obj_far
  gripper_button_far

MicrowaveThawing:
  obj_in_microwave
  button_pressed
  gripper_obj_far
```

## Storage Work

GR00T checkpoints are large because frozen base tensors are repeatedly stored. We added compact checkpoint tooling that stores only changed tensors and symlinks unchanged base shards to the local HuggingFace cache.

Important scripts:

```text
scripts/compact_groot_checkpoint.py
scripts/compact_nonfinal_checkpoints.py
```

The compact worktree is:

```text
/home/junhyeong/Value/Isaac-GR00T-compact-eval
```

The important warning is that compact checkpoints require the base HF cache snapshot to remain available:

```text
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

## Main Files to Read

Human-facing/context files:

```text
JUNHYEONG_ROBOCASA_AWR_EVAL_HANDOFF.md
JUNHYEONG_ROBOCASA_ZMQ_PARALLEL_EVAL_HANDOFF.md
100ep_eval_3_8tasks.md
ROBOCASA_TASK_FINETUNE_HANDOFF.md
```

Core code:

```text
scripts/robocasa_awr_finetune.py
scripts/robocasa_n15_zmq_parallel_eval.py
scripts/run_groot_robocasa_zmq_server.sh
scripts/run_robocasa_n15_zmq_parallel_eval_container.sh
gr00t/eval/robot.py
gr00t/eval/service.py
gr00t/eval/wrappers/robocasa_n15_wrapper.py
```

## Remaining Work

The main remaining work is:

```text
1. Finish 100ep x 3 action-seed eval for all 3-task models.
2. Finish composite PrepareCoffee/MicrowaveThawing eval for baseline and AWR.
3. Report success both at final-task level and sub-condition level.
4. Keep all comparisons on the same schedule JSONs.
5. Avoid creating new timestamped output roots unless the eval setting intentionally changes.
6. Continue cleaning/checkpoint compaction only after confirming currently running evals do not need full checkpoints.
```
