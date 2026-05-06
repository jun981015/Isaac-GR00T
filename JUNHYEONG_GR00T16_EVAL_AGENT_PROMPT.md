# Prompt For New Codex Agent: Port GR00T 1.6-Style RoboCasa Eval

You are working on `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`, a git worktree branched from `/home/junhyeong/Value/Isaac-GR00T` commit `c5b45b4 Add RoboCasa AWR training and eval tooling`.

## Goal

Port the faster GR00T 1.6-style RoboCasa evaluation environment into the N1.5 RoboCasa eval workflow without breaking the existing HTTP eval pipeline.

The user wants the eval to preserve these properties:

- Policy comparisons must run on the same RoboCasa eval setting across different checkpoints.
- Same seed/schedule should reproduce the same layout/style/object asset/task language and, as much as possible, object placement.
- Eval should be faster than the current HTTP/FastAPI path.
- Existing N1.5 HTTP eval scripts should remain usable.
- Do this work in the ZMQ worktree, not the main worktree.

## Current Repos And Worktrees

- Main repo: `/home/junhyeong/Value/Isaac-GR00T`
- Main branch: `junhyeong-awr-eval-base`
- Main base commit: `c5b45b4 Add RoboCasa AWR training and eval tooling`
- ZMQ eval worktree: `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`
- ZMQ branch: `junhyeong-eval-zmq`

The ZMQ worktree currently has experimental untracked files:

- `scripts/robocasa_n15_zmq_eval.py`
- `scripts/run_groot_robocasa_zmq_server.sh`
- `scripts/run_robocasa_n15_zmq_eval_container.sh`

Review these, but do not assume they are correct.

## Important Existing Files

Read these first:

- `/home/junhyeong/Value/Isaac-GR00T/JUNHYEONG_ROBOCASA_AWR_EVAL_HANDOFF.md`
- `/home/junhyeong/Value/Isaac-GR00T/ROBOCASA_EVAL_DOCKER.md`
- `/home/junhyeong/Value/Isaac-GR00T/JUNHYEONG_GIT_TODO.md`
- `scripts/robocasa_n15_http_eval.py`
- `scripts/run_robocasa_n15_eval_container.sh`
- `scripts/run_groot_robocasa_http_server.sh`
- `gr00t/eval/service.py`
- `gr00t/eval/robot.py`
- `scripts/inference_service.py`

Reference GR00T 1.6 material is under:

- `/home/junhyeong/Value/GR00T-1.6`

Use that repo only as a reference for eval structure and policy communication. Do not blindly overwrite this N1.5 repo with 1.6 code.

## Known Current Pipeline

The current N1.5 eval path uses:

- RoboCasa env container.
- Separate policy HTTP server container.
- FastAPI/HTTP with JSON numpy payloads.
- `scripts/robocasa_n15_http_eval.py` for the env-side rollout.
- Docker launchers in `scripts/run_*eval*.sh`.

This works but is slow. Earlier investigation suggested major bottlenecks include:

- Per-policy-call HTTP serialization/deserialization.
- Render/video overhead.
- Past env recreation, now mostly env reuse/reset.

GR00T 1.6 uses ZeroMQ/msgpack-style policy communication and supports a faster eval pattern. N1.5 already contains generic ZMQ service/client code in `gr00t/eval/service.py`, `gr00t/eval/robot.py`, and `scripts/inference_service.py`.

## Eval Reproducibility Context

Canonical schedule files are:

- `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPCounterToSink_50eps.json`
- `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPCounterToStove_50eps.json`
- `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/schedules/n15_seed1_50ep/seed1/PnPMicrowaveToCounter_50eps.json`

Current strict schedule replay in `scripts/robocasa_n15_http_eval.py`:

1. Load schedule episode.
2. Reuse/create RoboCasa env.
3. Reseed env.
4. Set `ep_meta`.
5. Call `env.reset()`.

Important caveat:

- Schedule JSON stores `ep_meta`, layout/style, seed, task language, etc.
- It does not currently store exact simulator object poses after reset.
- Exact object placement replay across all code paths may require saving/restoring object placements or sim state after reset.

Do not claim exact pose-level replay unless you verify it.

## Current Eval Settings To Preserve

For policy-grid evals:

- `seed=1`
- `n_episodes=50`
- episode indices `0-49`
- `n_action_steps=16`
- `max_episode_steps=800`
- `video_fps=20`
- prefer obs/video size 128 when speed matters
- only composite/necessary video unless user asks otherwise
- `obj_instance_split=A`
- layout/style ids: `1:1,2:2,4:4,6:9,7:10`

Output roots currently used:

- Old AWR/baseline grid: `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/n15_policy_grid_seed1_50ep_hires_20260427_0030`
- New window alpha eval: `/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/new_window_alpha_eval_seed1_50ep_20260428`

Do not commit or move `local_outputs`, checkpoints, videos, or runtime cache files.

## Desired Implementation Direction

Work in `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`.

Investigate and implement the smallest viable ZMQ eval path:

- A policy server launcher that loads a checkpoint once and serves actions through the existing N1.5 ZMQ service.
- A RoboCasa eval script or modification that talks to that ZMQ policy client instead of HTTP.
- Keep schedule replay behavior equivalent to the HTTP eval path.
- Keep env reuse/reset behavior, not per-episode full recreation unless required.
- Keep video writing minimal and configurable.
- Make it easy to run one-task smoke eval first, then 50-episode task evals.

## Verification Needed

Do not start with a full 50-episode run. Verify in this order:

1. Static code inspection: identify exact request/response observation keys expected by ZMQ policy.
2. One-episode smoke eval on a small task.
3. Confirm a video/outcome JSON is produced.
4. Confirm same seed/schedule rerun restores the same task language/layout/style/object asset.
5. Measure rough seconds per episode and compare with the HTTP path.

## Communication Style

Report concise findings with file paths and line references. Avoid long generic summaries. If you find that exact object placement replay is not guaranteed, say what field/state must be stored to guarantee it.
