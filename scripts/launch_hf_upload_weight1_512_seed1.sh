#!/usr/bin/env bash
set -euo pipefail

REPO_FAILURE="${REPO_FAILURE:-RLobot-jun/robocasa-weight1-512-seed1-failure-vlm-labels}"
REPO_SUCCESS="${REPO_SUCCESS:-RLobot-jun/robocasa-weight1-512-seed1-success-vlm-labels}"
ROOT="${ROOT:-/home/junhyeong/Value/Isaac-GR00T}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/local_outputs/hf_datasets}"
CONDA_ENV="${CONDA_ENV:-gr00t}"
BATCH_SIZE="${BATCH_SIZE:-200}"

cd "${ROOT}"
export HF_HUB_DISABLE_XET=1

conda run -n "${CONDA_ENV}" python - <<PY
from huggingface_hub import HfApi, HfFolder
api = HfApi(token=HfFolder.get_token())
for repo_id in ["${REPO_FAILURE}", "${REPO_SUCCESS}"]:
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"public repo ready: {repo_id}", flush=True)
PY

conda run -n "${CONDA_ENV}" python scripts/upload_hf_dataset_batched.py \
  --folder "${DATA_ROOT}/robocasa_weight1_512_seed1_failure_vlm_labels" \
  --repo-id "${REPO_FAILURE}" \
  --batch-size "${BATCH_SIZE}" \
  --sleep-sec 0 \
  --retry-sleep-sec 0 \
  --max-retries 1 \
  2>&1 | tee "${DATA_ROOT}/robocasa_weight1_512_seed1_failure_vlm_labels/hf_upload_batched_fast.log"

conda run -n "${CONDA_ENV}" python scripts/upload_hf_dataset_batched.py \
  --folder "${DATA_ROOT}/robocasa_weight1_512_seed1_success_vlm_labels" \
  --repo-id "${REPO_SUCCESS}" \
  --batch-size "${BATCH_SIZE}" \
  --sleep-sec 0 \
  --retry-sleep-sec 0 \
  --max-retries 1 \
  2>&1 | tee "${DATA_ROOT}/robocasa_weight1_512_seed1_success_vlm_labels/hf_upload_batched_fast.log"
