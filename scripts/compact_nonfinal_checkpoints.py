#!/usr/bin/env python3
"""Compact all non-final checkpoints under local_outputs/robocasa_awr_retrain."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path


def real_file_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "local_outputs/robocasa_awr_retrain"
    base = Path(
        "/home/junhyeong/.cache/huggingface/models--nvidia--GR00T-N1.5-3B/"
        "snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
    )
    compact_script = repo / "scripts/compact_groot_checkpoint.py"
    log = root / "compact_mid_checkpoints_20260430.log"

    def logmsg(message: str) -> None:
        msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")

    # Clean up interrupted temp dirs only when safe.
    for backup in sorted(root.glob("*/checkpoint-*.full_backup_tmp")):
        ckpt = backup.with_name(backup.name.replace(".full_backup_tmp", ""))
        if (ckpt / "compact_summary.json").exists() and (
            ckpt / "model.safetensors.index.json"
        ).exists():
            logmsg(f"remove interrupted backup {backup}")
            shutil.rmtree(backup)
        else:
            raise SystemExit(f"unsafe stale backup without compact replacement: {backup}")

    for tmp in sorted(root.glob("*/checkpoint-*.compact_tmp")):
        logmsg(f"remove stale compact tmp {tmp}")
        shutil.rmtree(tmp)

    targets: list[tuple[str, int, int, Path]] = []
    for run_dir in sorted(root.iterdir()):
        if (
            not run_dir.is_dir()
            or run_dir.name.endswith("_compact_test")
            or run_dir.name.startswith(".")
        ):
            continue
        checkpoints: list[tuple[int, Path]] = []
        for ckpt in run_dir.glob("checkpoint-*"):
            if ckpt.name.endswith((".compact_tmp", ".full_backup_tmp")):
                continue
            try:
                step = int(ckpt.name.split("-")[-1])
            except ValueError:
                continue
            if (ckpt / "model.safetensors.index.json").exists():
                checkpoints.append((step, ckpt))
        if not checkpoints:
            continue
        final_step = max(step for step, _ in checkpoints)
        for step, ckpt in sorted(checkpoints):
            if step < final_step and not (ckpt / "compact_summary.json").exists():
                targets.append((run_dir.name, final_step, step, ckpt))

    logmsg(f"total remaining targets={len(targets)}")
    for idx, (run_name, final_step, _step, ckpt) in enumerate(targets, 1):
        tmp = ckpt.with_name(ckpt.name + ".compact_tmp")
        backup = ckpt.with_name(ckpt.name + ".full_backup_tmp")
        if tmp.exists() or backup.exists():
            raise SystemExit(f"temp path exists for {ckpt}: {tmp} {backup}")

        before = real_file_size(ckpt)
        logmsg(
            f"[{idx}/{len(targets)}] compact start {run_name}/{ckpt.name} "
            f"final_keep=checkpoint-{final_step} before_real={before / 1024**3:.2f}GiB"
        )
        subprocess.run(
            [
                sys.executable,
                str(compact_script),
                "--checkpoint",
                str(ckpt),
                "--base",
                str(base),
                "--output",
                str(tmp),
            ],
            check=True,
            cwd=repo,
        )
        if not (tmp / "compact_summary.json").exists() or not (
            tmp / "model.safetensors.index.json"
        ).exists():
            raise SystemExit(f"compact output missing files: {tmp}")

        ckpt.rename(backup)
        tmp.rename(ckpt)
        after = real_file_size(ckpt)
        shutil.rmtree(backup)
        logmsg(
            f"[{idx}/{len(targets)}] compact done {run_name}/{ckpt.name} "
            f"after_real={after / 1024**3:.2f}GiB saved~={(before - after) / 1024**3:.2f}GiB"
        )

    logmsg("all requested non-final checkpoints compacted")


if __name__ == "__main__":
    main()
