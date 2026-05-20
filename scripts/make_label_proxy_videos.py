#!/usr/bin/env python3
"""Create sped-up proxy videos for the RoboCasa labeling UI."""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_one(src: Path, eval_root: Path, speed: float, fps: int, crf: int, overwrite: bool) -> tuple[str, bool, str]:
    dst = src.parent.parent / "videos_fast" / src.name
    if dst.exists() and not overwrite:
        return (dst.relative_to(eval_root).as_posix(), False, "exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"setpts={1.0 / speed:.8f}*PTS,fps={fps}",
        "-an",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(dst)
        return (dst.relative_to(eval_root).as_posix(), True, "ok")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return (src.relative_to(eval_root).as_posix(), False, f"error: {exc}")


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    videos = sorted(eval_root.glob("action_seed_*/*/videos/*outcome0*.mp4"))
    print(f"found={len(videos)} eval_root={eval_root}", flush=True)
    done = 0
    created = 0
    skipped = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(convert_one, video, eval_root, args.speed, args.fps, args.crf, args.overwrite)
            for video in videos
        ]
        for future in as_completed(futures):
            rel, did_create, status = future.result()
            done += 1
            if did_create:
                created += 1
            elif status == "exists":
                skipped += 1
            else:
                failed += 1
                print(f"[failed] {rel}: {status}", flush=True)
            if done == 1 or done % 25 == 0 or done == len(videos):
                print(f"progress={done}/{len(videos)} created={created} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
