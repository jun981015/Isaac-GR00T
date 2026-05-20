#!/usr/bin/env python3
"""Upload a local dataset folder to Hugging Face in rate-limit-safe batches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, HfFolder


def iter_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() or p.is_symlink())


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"uploaded": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def upload_batch(
    api: HfApi,
    *,
    repo_id: str,
    folder: Path,
    rel_paths: list[str],
    commit_message: str,
) -> None:
    ops = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(folder / rel))
        for rel in rel_paths
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=commit_message,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--sleep-sec", type=int, default=360)
    parser.add_argument("--retry-sleep-sec", type=int, default=420)
    parser.add_argument("--state-file", default="")
    parser.add_argument("--max-retries", type=int, default=1000000)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    folder = Path(args.folder).resolve()
    state_file = Path(args.state_file) if args.state_file else folder / ".hf_upload_state.json"
    api = HfApi(token=HfFolder.get_token())
    state = load_state(state_file)
    uploaded = set(state.get("uploaded", []))

    files = iter_files(folder)
    rels = [p.relative_to(folder).as_posix() for p in files]
    rels = [rel for rel in rels if rel not in {state_file.name, "hf_upload.log"} and not rel.endswith(".log")]

    priority = [rel for rel in ["README.md", "dataset_info.json", "metadata.jsonl"] if rel in rels]
    rest = [rel for rel in rels if rel not in priority]
    ordered = priority + rest
    pending = [rel for rel in ordered if rel not in uploaded]

    print(
        f"[start] repo={args.repo_id} folder={folder} total={len(ordered)} "
        f"uploaded={len(uploaded)} pending={len(pending)} batch={args.batch_size}",
        flush=True,
    )

    batch_idx = 0
    while True:
        pending = [rel for rel in ordered if rel not in uploaded]
        if not pending:
            print("[done] all files uploaded", flush=True)
            return 0

        batch_idx += 1
        batch = pending[: args.batch_size]
        retries = 0
        while True:
            try:
                print(
                    f"[batch {batch_idx}] upload {len(batch)} files "
                    f"remaining_before={len(pending)} first={batch[0]}",
                    flush=True,
                )
                upload_batch(
                    api,
                    repo_id=args.repo_id,
                    folder=folder,
                    rel_paths=batch,
                    commit_message=f"Upload batch {batch_idx} ({len(batch)} files)",
                )
                uploaded.update(batch)
                state["uploaded"] = sorted(uploaded)
                state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                save_state(state_file, state)
                print(
                    f"[batch {batch_idx}] ok uploaded_total={len(uploaded)} "
                    f"sleep={args.sleep_sec}s",
                    flush=True,
                )
                time.sleep(args.sleep_sec)
                break
            except Exception as exc:
                retries += 1
                print(
                    f"[batch {batch_idx}] error retry={retries}/{args.max_retries}: {exc!r}",
                    flush=True,
                )
                if retries >= args.max_retries:
                    raise
                print(f"[batch {batch_idx}] sleep_retry={args.retry_sleep_sec}s", flush=True)
                time.sleep(args.retry_sleep_sec)


if __name__ == "__main__":
    raise SystemExit(main())
