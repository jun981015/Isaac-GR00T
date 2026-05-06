# Compact Checkpoint Eval Notes

Worktree:

```text
/home/junhyeong/Value/Isaac-GR00T-compact-eval
```

Compact checkpoints keep changed fine-tuned tensors locally and point unchanged base tensors to:

```text
/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

Use the server wrappers in this worktree when evaluating compact checkpoints:

```bash
CHECKPOINT_DIR=/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/<run>/checkpoint-<step> \
GPU_DEVICE=<gpu> \
PORT=<port> \
CONTAINER_NAME=<name> \
bash /home/junhyeong/Value/Isaac-GR00T-compact-eval/scripts/run_groot_robocasa_zmq_server.sh
```

The wrappers mount `HF_CACHE_DIR` into Docker at the same absolute path, so symlinked base shards resolve inside the container.

Default `REPO_DIR` is derived from the script location, so running scripts from this worktree mounts this worktree into Docker rather than the active eval repo.

Preflight behavior:

- If `compact_summary.json` exists in `CHECKPOINT_DIR`, the wrapper checks that `HF_CACHE_DIR` exists.
- It also checks `base-*.safetensors` symlink targets before starting Docker.

Do not delete the HF cache snapshot while compact checkpoints are needed.
