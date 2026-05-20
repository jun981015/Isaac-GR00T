#!/usr/bin/env python3
"""Prepare RoboCasa eval videos as HF dataset folders split by success."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


VIDEO_RE = re.compile(r"ep(?P<ep>\d+)_seed(?P<env_seed>-?\d+)_composite_outcome(?P<outcome>[01])\.mp4$")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ensure_empty_or_metadata_only(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    allowed = {"README.md", "dataset_info.json", "metadata.jsonl", ".hf_upload_state.json"}
    existing = [p for p in path.rglob("*") if p.is_file() or p.is_symlink()]
    unexpected = [
        p for p in existing
        if p.relative_to(path).parts[0] not in allowed
    ]
    if unexpected:
        raise RuntimeError(f"output folder already has dataset files: {path}")


def symlink_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def load_episode(video: Path) -> dict:
    task_dir = video.parents[1]
    match = VIDEO_RE.match(video.name)
    if not match:
        raise RuntimeError(f"unexpected video name: {video}")
    ep = int(match.group("ep"))
    episode_path = task_dir / "episodes" / f"ep{ep:03d}.json"
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)
    return json.loads(episode_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--copy", action="store_true", help="copy videos instead of symlinking")
    args = parser.parse_args()

    eval_root = Path(args.eval_root).resolve()
    out_root = Path(args.out_root).resolve()
    success_dir = out_root / f"{args.prefix}_success_vlm_labels"
    failure_dir = out_root / f"{args.prefix}_failure_vlm_labels"
    ensure_empty_or_metadata_only(success_dir)
    ensure_empty_or_metadata_only(failure_dir)

    rows = {True: [], False: []}
    videos = sorted(eval_root.glob("action_seed_*/*/videos/*.mp4"))
    if not videos:
        raise RuntimeError(f"no videos found under {eval_root}")

    for video in videos:
        action_seed_dir = video.parents[2].name
        action_seed = int(action_seed_dir.removeprefix("action_seed_"))
        task = video.parents[1].name
        match = VIDEO_RE.match(video.name)
        if not match:
            raise RuntimeError(f"unexpected video name: {video}")
        ep = int(match.group("ep"))
        env_seed = int(match.group("env_seed"))
        outcome = int(match.group("outcome"))
        episode = load_episode(video)
        success = bool(episode.get("success", bool(outcome)))
        split_dir = success_dir if success else failure_dir
        rel_video = Path("videos") / f"{task}_actionseed{action_seed}_ep{ep:03d}_envseed{env_seed}_outcome{outcome}_512.mp4"
        symlink_or_copy(video, split_dir / rel_video, args.copy)

        rows[success].append({
            "video": rel_video.as_posix(),
            "source_video_path": str(video),
            "task": task,
            "episode_idx": ep,
            "action_seed": action_seed,
            "env_seed": env_seed,
            "success": success,
            "outcome_from_filename": outcome,
            "prompt": episode.get("ep_meta", {}).get("lang"),
            "label_target": None,
            "label_options": [
                "pick_fail",
                "drop_after_pick",
                "place_fail",
                "navigation_or_alignment_fail",
                "button_or_fixture_interaction_fail",
                "other_fail",
                "no_count",
            ],
            "video_resolution": "1536x512",
            "source_eval_root": str(eval_root),
        })

    for success, split_dir in [(False, failure_dir), (True, success_dir)]:
        metadata_path = split_dir / "metadata.jsonl"
        with metadata_path.open("w") as f:
            for row in rows[success]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_name = "success" if success else "failure"
        write_json(split_dir / "dataset_info.json", {
            "dataset": split_dir.name,
            "source_eval_root": str(eval_root),
            "split": split_name,
            "num_videos": len(rows[success]),
            "video_resolution": "1536x512",
            "video_fps": 20,
        })
        (split_dir / "README.md").write_text(
            f"# RoboCasa Weight1 512px {split_name.title()} Videos for VLM Labeling\n\n"
            f"This dataset contains RoboCasa evaluation videos from the 512px seed1 weight1 alpha0 50k run.\n\n"
            "Each row in `metadata.jsonl` includes the relative mp4 path, task, episode/action/env seed, "
            "success flag, and `prompt` extracted from `episode_json.ep_meta.lang`.\n\n"
            "Videos are 3-view composite mp4s at 1536x512 and 20 fps.\n"
        )

    print(f"success videos: {len(rows[True])} -> {success_dir}")
    print(f"failure videos: {len(rows[False])} -> {failure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
