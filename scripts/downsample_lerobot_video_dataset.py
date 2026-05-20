#!/usr/bin/env python3
"""Create a LeRobot dataset copy with videos resized to a fixed resolution."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args()


def copy_non_video_files(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] == "videos":
            continue
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)


def update_info_json(dst: Path, size: int) -> None:
    info_path = dst / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    for feature in info.get("features", {}).values():
        if feature.get("dtype") == "video":
            feature["shape"] = [size, size, 3]
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def transcode_one(src_file: Path, src: Path, dst: Path, size: int, crf: int, preset: str) -> str:
    rel = src_file.relative_to(src)
    out_file = dst / rel
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0:
        return "skip"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_file),
        "-vf",
        f"scale={size}:{size}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    return "ok"


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    copy_non_video_files(src, dst)
    update_info_json(dst, args.size)

    videos = sorted((src / "videos").rglob("*.mp4"))
    print(f"transcoding {len(videos)} videos: {src} -> {dst} ({args.size}x{args.size})")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                transcode_one,
                video,
                src,
                dst,
                args.size,
                args.crf,
                args.preset,
            )
            for video in videos
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(videos):
                print(f"{done}/{len(videos)} videos")


if __name__ == "__main__":
    main()
