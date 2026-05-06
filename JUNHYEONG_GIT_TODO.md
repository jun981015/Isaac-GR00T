# Junhyeong Git / TODO

## Git

- Main worktree: `/home/junhyeong/Value/Isaac-GR00T`
- Main branch: `junhyeong-awr-eval-base`
- Base commit: `c5b45b4 Add RoboCasa AWR training and eval tooling`
- Committed scope: AWR loss/training code, RoboCasa eval wrappers, HTTP eval launchers, schedule/replay helpers, handoff markdowns.
- Excluded from commit: `.codex/`, `local_outputs/`, checkpoints, eval videos, training outputs.
- ZMQ eval worktree: `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`
- ZMQ worktree branch: `junhyeong-eval-zmq`
- ZMQ worktree base: `c5b45b4`
- ZMQ worktree current untracked files: `scripts/robocasa_n15_zmq_eval.py`, `scripts/run_groot_robocasa_zmq_server.sh`, `scripts/run_robocasa_n15_zmq_eval_container.sh`

## TODO

- Port GR00T 1.6-style eval flow into the ZMQ worktree: use `/home/junhyeong/Value/Isaac-GR00T-eval-zmq`, keep the main AWR/eval baseline commit unchanged.
- For new RoboCasa tasks, design AWR labeling and critical-frame detection in a separate worktree before touching the committed main workflow.
- New critical-frame work should cover pick/place-like events where applicable, plus non-PnP events for coffee, door, and microwave tasks.
