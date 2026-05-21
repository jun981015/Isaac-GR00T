#!/usr/bin/env python3
"""Create an official LeRobot v2.1-compatible copy of a GR00T LeRobot dataset.

The RoboCasa rollout converter originally produced a dataset that GR00T can
load directly, but official lerobot 0.3.x expects `meta/episodes_stats.jsonl`
when `meta/info.json` declares `codebase_version: v2.1`. This script creates a
hardlinked copy and adds per-episode numeric stats.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = dst / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            copy_file(path, out, mode)
        elif path.is_symlink():
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.symlink_to(os.readlink(path))


def as_2d(values: list[Any]) -> np.ndarray:
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values).astype(np.float32)
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array.astype(np.float32)


def feature_stats(array: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def episode_stats(parquet_path: Path, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    stats: dict[str, Any] = {}
    for key, spec in features.items():
        if key not in df.columns:
            continue
        dtype = spec.get("dtype")
        if dtype in {"video", "image", "string"}:
            continue
        values = df[key].tolist()
        if not values:
            continue
        stats[key] = feature_stats(as_2d(values))
    return stats


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() and not (dst / "meta" / "info.json").exists():
        raise RuntimeError(f"destination exists but is not a dataset: {dst}")

    copy_tree(src, dst, args.copy_mode)

    info_path = dst / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    info["codebase_version"] = "v2.1"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    episodes_path = dst / "meta" / "episodes.jsonl"
    stats_path = dst / "meta" / "episodes_stats.jsonl"
    if stats_path.exists() and not args.overwrite_metadata:
        print(f"exists: {stats_path}")
        return

    episodes = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            episodes.append(json.loads(line))

    tmp_path = stats_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        for idx, episode in enumerate(episodes, start=1):
            ep_idx = int(episode["episode_index"])
            chunk = ep_idx // int(info["chunks_size"])
            rel = info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
            stats = episode_stats(dst / rel, info["features"])
            row = {"episode_index": ep_idx, "stats": stats}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % args.progress_every == 0 or idx == len(episodes):
                print(f"[episodes_stats] {idx}/{len(episodes)}", flush=True)
    tmp_path.replace(stats_path)
    print(f"[done] wrote {stats_path}")


if __name__ == "__main__":
    main()
