# Junhyeong RoboCasa AWR HF Checkpoint Upload

This document records the Hugging Face upload metadata for the RoboCasa AWR
checkpoint selected for cross-server reuse.

The practical transport policy is:

- Do not re-upload the original GR00T base model into this repo.
- Treat the base model as an external dependency: `nvidia/GR00T-N1.5-3B`.
- For this transfer, upload the full selected checkpoint for convenience.
- For future transfers, prefer a delta-style package that stores only weights
  changed by fine-tuning plus a reconstruction procedure from the base model.

## Target Run

```text
run name: ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1
local run dir: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1
checkpoint: /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000
local size: 7.1G
```

## Hugging Face Destination

```text
HF repo: RLobot-jun/groot-robocasa-checkpoints
repo type: model
visibility: public
uploaded at: 2026-05-19
HF path: server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000
HF URL: https://huggingface.co/RLobot-jun/groot-robocasa-checkpoints/tree/main/server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000
```

## Base Model Dependency

The original GR00T base model is not duplicated in this upload.

```text
base model: nvidia/GR00T-N1.5-3B
local server 50 cache: /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
```

On a new server, download or cache `nvidia/GR00T-N1.5-3B` through the normal
Hugging Face path used by Isaac-GR00T. Public redistribution of derived
weights should still respect the base model license and usage restrictions.

## Uploaded Files

The uploaded checkpoint contains the files required for inference or weight
reuse:

```text
checkpoint-50000/config.json
checkpoint-50000/model.safetensors.index.json
checkpoint-50000/model-00001-of-00002.safetensors
checkpoint-50000/model-00002-of-00002.safetensors
checkpoint-50000/experiment_cfg/metadata.json
checkpoint-50000/trainer_state.json
```

The two `.safetensors` shards and `model.safetensors.index.json` must be kept
together.

## Upload Command

```bash
conda run -n groot-smoke hf upload \
  RLobot-jun/groot-robocasa-checkpoints \
  /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000 \
  server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000 \
  --repo-type model \
  --commit-message "Upload ann24 AWR weight1 checkpoint-50000"
```

If the default Xet cache is not writable on the shared server, use a user-owned
Xet cache path:

```bash
mkdir -p /home/junhyeong/.cache/huggingface_jun_upload/xet

conda run -n groot-smoke env \
  HF_XET_CACHE=/home/junhyeong/.cache/huggingface_jun_upload/xet \
  hf upload \
  RLobot-jun/groot-robocasa-checkpoints \
  /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000 \
  server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000 \
  --repo-type model \
  --commit-message "Upload ann24 AWR weight1 checkpoint-50000"
```

## Compact / Fine-Tune-Only Option

This run currently has a full checkpoint. It is acceptable for one-off
cross-server transport, but it duplicates many base-model weights.

For future uploads, prefer a delta package so unchanged base weights are not
stored again. The existing compact checkpoint utility can identify and extract
changed tensors:

```bash
conda run -n groot-smoke python scripts/compact_groot_checkpoint.py \
  --checkpoint /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000 \
  --base /home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
  --output /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000-compact
```

The compact output writes:

```text
model-changed.safetensors
model.safetensors.index.json
base-*.safetensors symlinks to the base snapshot
compact_summary.json
```

Only `model-changed.safetensors`, metadata, and the index are unique to the
fine-tuned run. The current compact format is useful for local storage savings,
but it is not a standalone public HF distribution format: the `base-*` symlinks
must point to a valid base snapshot on the target server, or a reconstruction
script must materialize a full checkpoint from:

```text
nvidia/GR00T-N1.5-3B@869830fc749c35f34771aa5209f923ac57e4564e
model-changed.safetensors
model.safetensors.index.json
compact_summary.json
```

Therefore, use the full checkpoint for this one-off transport. Use the
compact/delta path only when the restore procedure is also documented and
tested.

## Download Command

Use this command on another server to restore the checkpoint under the same
local run directory layout:

```bash
mkdir -p /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1

conda run -n groot-smoke hf download \
  RLobot-jun/groot-robocasa-checkpoints \
  --repo-type model \
  --include "server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000/**" \
  --local-dir /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1
```

The download command preserves the `server_50/.../checkpoint-50000` prefix
inside `--local-dir`. If a script expects `checkpoint-50000` directly under the
run directory, move or rsync the downloaded checkpoint folder into place:

```bash
mkdir -p /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000

rsync -a \
  /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/server_50/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000/ \
  /home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_awr_retrain/ann24_awrsrc_annotation_r1_alpha0_b64_50000step_gpu1_20260504_weight1/checkpoint-50000/
```

## Resume Limitations

This upload is sufficient for inference and weight reuse, but it is not a full
Trainer resume package. The local checkpoint does not contain optimizer,
scheduler, or RNG state files such as:

```text
optimizer.pt
scheduler.pt
rng_state*.pth
```

Tokenizer or processor files are also not present in this checkpoint directory;
use the base model/repo configuration expected by the Isaac-GR00T code path.

## Git Policy

`local_outputs/` and checkpoint binaries are intentionally excluded from git.
This markdown records only the upload provenance and restoration commands.
