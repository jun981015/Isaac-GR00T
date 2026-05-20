#!/usr/bin/env python3
"""Convert RoboCasa eval rollout HDF5 files into a LeRobot-style dataset.

This is intended for eval rollouts recorded with
``scripts/robocasa_n15_zmq_parallel_eval.py --save_rollout_hdf5``.
Images are not stored in the HDF5; the eval composite mp4 is split into the
three RoboCasa camera videos expected by the existing GR00T RoboCasa data
config.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


STATE_KEYS = [
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qvel",
    "state.end_effector_position_absolute",
    "state.end_effector_rotation_absolute",
    "state.joint_position",
    "state.joint_position_cos",
    "state.joint_position_sin",
    "state.joint_velocity",
]

VIDEO_KEYS = {
    "observation.images.robot0_agentview_left": (0, "robot0_agentview_left"),
    "observation.images.robot0_agentview_right": (1, "robot0_agentview_right"),
    "observation.images.robot0_eye_in_hand": (2, "robot0_eye_in_hand"),
}


@dataclass(frozen=True)
class EpisodeSource:
    task: str
    source_episode_idx: int
    seed: int
    hdf5_path: Path
    episode_json_path: Path | None


class RunningStats:
    def __init__(self) -> None:
        self._state_chunks: list[np.ndarray] = []
        self._action_chunks: list[np.ndarray] = []

    def add(self, state: np.ndarray, action: np.ndarray) -> None:
        self._state_chunks.append(np.asarray(state, dtype=np.float32))
        self._action_chunks.append(np.asarray(action, dtype=np.float32))

    def to_json(self) -> dict[str, Any]:
        return {
            "observation.state": self._array_stats(np.concatenate(self._state_chunks, axis=0)),
            "action": self._array_stats(np.concatenate(self._action_chunks, axis=0)),
        }

    @staticmethod
    def _array_stats(x: np.ndarray) -> dict[str, list[float]]:
        return {
            "mean": x.mean(axis=0).astype(float).tolist(),
            "std": x.std(axis=0).astype(float).tolist(),
            "min": x.min(axis=0).astype(float).tolist(),
            "max": x.max(axis=0).astype(float).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).astype(float).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).astype(float).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_sources(input_root: Path, tasks: list[str] | None, limit: int | None) -> list[EpisodeSource]:
    selected_tasks = set(tasks) if tasks else None
    sources: list[EpisodeSource] = []
    for task_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        task = task_dir.name
        if selected_tasks is not None and task not in selected_tasks:
            continue
        hdf5_dir = task_dir / "hdf5"
        if not hdf5_dir.exists():
            continue
        for hdf5_path in sorted(hdf5_dir.glob("ep*_seed*_rollout_io.hdf5")):
            match = re.search(r"ep(\d+)_seed(\d+)_", hdf5_path.name)
            if match is None:
                continue
            ep_idx = int(match.group(1))
            seed = int(match.group(2))
            episode_json_path = task_dir / "episodes" / f"ep{ep_idx:03d}.json"
            sources.append(
                EpisodeSource(
                    task=task,
                    source_episode_idx=ep_idx,
                    seed=seed,
                    hdf5_path=hdf5_path,
                    episode_json_path=episode_json_path if episode_json_path.exists() else None,
                )
            )
            if limit is not None and len(sources) >= limit:
                return sources
    return sources


def read_text_dataset(group: h5py.File, key: str) -> str:
    value = group[key][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_rollout(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    with h5py.File(hdf5_path, "r") as f:
        obs_group = f["env_steps"]["obs"]
        state_parts = [np.asarray(obs_group[key], dtype=np.float32) for key in STATE_KEYS]
        state = np.concatenate(state_parts, axis=1)
        action = np.asarray(f["env_steps"]["raw_action"], dtype=np.float32)
        reward = np.asarray(f["env_steps"]["reward"], dtype=np.float32)
        done = np.asarray(f["env_steps"]["done"], dtype=bool)
        prompt = read_text_dataset(f, "prompt")
        video_path = read_text_dataset(f, "video_path")
    if state.shape[0] != action.shape[0]:
        raise ValueError(f"state/action length mismatch in {hdf5_path}: {state.shape} vs {action.shape}")
    return state, action, reward, done, prompt, video_path


def load_episode_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_episode_parquet(
    path: Path,
    state: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    prompt_task_index: int,
    episode_index: int,
    global_start_index: int,
    fps: float,
) -> None:
    length = state.shape[0]
    if done.shape[0] == length:
        next_done = done.astype(bool).copy()
        next_done[-1] = True
    else:
        next_done = np.zeros(length, dtype=bool)
        next_done[-1] = True
    df = pd.DataFrame(
        {
            "observation.state": [row for row in state.astype(np.float32)],
            "action": [row for row in action.astype(np.float32)],
            "timestamp": (np.arange(length, dtype=np.float32) / np.float32(fps)),
            "annotation.human.action.task_description": np.full(length, prompt_task_index, dtype=np.int64),
            "task_index": np.full(length, prompt_task_index, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "index": np.arange(global_start_index, global_start_index + length, dtype=np.int64),
            "next.reward": reward.astype(np.float32),
            "next.done": next_done,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def split_composite_video(
    input_video: Path,
    output_paths: dict[str, Path],
    video_size: int,
    expected_frames: int,
    resume: bool,
) -> None:
    if resume and all(path.exists() and path.stat().st_size > 0 for path in output_paths.values()):
        return
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-filter_complex",
        (
            f"[0:v]trim=end_frame={expected_frames},setpts=PTS-STARTPTS,split=3[v0][v1][v2];"
            f"[v0]crop={video_size}:{video_size}:0:0[left];"
            f"[v1]crop={video_size}:{video_size}:{video_size}:0[right];"
            f"[v2]crop={video_size}:{video_size}:{2 * video_size}:0[hand]"
        ),
        "-map",
        "[left]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_paths["observation.images.robot0_agentview_left"]),
        "-map",
        "[right]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_paths["observation.images.robot0_agentview_right"]),
        "-map",
        "[hand]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(output_paths["observation.images.robot0_eye_in_hand"]),
    ]
    subprocess.run(cmd, check=True)


def make_info(total_episodes: int, total_frames: int, total_tasks: int, chunks_size: int, fps: float, video_size: int) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [53],
            "names": [f"state_{i}" for i in range(53)],
        },
        "action": {
            "dtype": "float32",
            "shape": [12],
            "names": [f"action_{i}" for i in range(12)],
        },
        "timestamp": {"dtype": "float32", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    for video_key in VIDEO_KEYS:
        features[video_key] = {
            "dtype": "video",
            "shape": [video_size, video_size, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v2.1",
        "robot_type": "PandaMobile",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def copy_modality_json(output_dir: Path) -> None:
    modality = {
        "state": {
            "gripper_qpos": {"start": 0, "end": 2, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "base_position": {"start": 2, "end": 5, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "base_rotation": {"start": 5, "end": 9, "dtype": "float32", "absolute": True, "original_key": "observation.state", "rotation_type": "quaternion"},
            "end_effector_position_relative": {"start": 9, "end": 12, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "end_effector_rotation_relative": {"start": 12, "end": 16, "dtype": "float32", "absolute": True, "original_key": "observation.state", "rotation_type": "quaternion"},
            "gripper_qvel": {"start": 16, "end": 18, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "end_effector_position_absolute": {"start": 18, "end": 21, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "end_effector_rotation_absolute": {"start": 21, "end": 25, "dtype": "float32", "absolute": True, "original_key": "observation.state", "rotation_type": "quaternion"},
            "joint_position": {"start": 25, "end": 32, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "joint_position_cos": {"start": 32, "end": 39, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "joint_position_sin": {"start": 39, "end": 46, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
            "joint_velocity": {"start": 46, "end": 53, "dtype": "float32", "absolute": True, "original_key": "observation.state"},
        },
        "action": {
            "eef_position": {"start": 0, "end": 3, "dtype": "float32", "absolute": False, "original_key": "action"},
            "eef_rotation": {"start": 3, "end": 6, "dtype": "float32", "absolute": False, "original_key": "action", "rotation_type": "axis_angle"},
            "gripper": {"start": 6, "end": 7, "dtype": "float32", "absolute": False, "original_key": "action"},
            "base": {"start": 7, "end": 11, "dtype": "float32", "absolute": False, "original_key": "action"},
            "control_mode": {"start": 11, "end": 12, "dtype": "float32", "absolute": False, "original_key": "action"},
        },
        "video": {
            "robot0_agentview_left": {"original_key": "observation.images.robot0_agentview_left"},
            "robot0_agentview_right": {"original_key": "observation.images.robot0_agentview_right"},
            "robot0_eye_in_hand": {"original_key": "observation.images.robot0_eye_in_hand"},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }
    write_json(output_dir / "meta" / "modality.json", modality)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    sources = collect_sources(args.input_root, args.tasks, args.limit)
    if not sources:
        raise SystemExit(f"No HDF5 episodes found under {args.input_root}")
    if args.dry_run:
        print(f"Found {len(sources)} episodes")
        for src in sources[:10]:
            print(src)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompt_to_index: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    stats = RunningStats()
    total_frames = 0

    for episode_index, src in enumerate(sources):
        state, action, reward, done, prompt, video_path_text = read_rollout(src.hdf5_path)
        if prompt not in prompt_to_index:
            prompt_to_index[prompt] = len(prompt_to_index)
            task_rows.append({"task_index": prompt_to_index[prompt], "task": prompt})
        task_index = prompt_to_index[prompt]

        chunk_idx = episode_index // args.chunks_size
        parquet_path = args.output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
        if not (args.resume and parquet_path.exists()):
            write_episode_parquet(
                parquet_path,
                state,
                action,
                reward,
                done,
                task_index,
                episode_index,
                total_frames,
                args.fps,
            )

        source_video = Path(video_path_text)
        output_videos = {
            video_key: args.output_dir
            / "videos"
            / f"chunk-{chunk_idx:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
            for video_key in VIDEO_KEYS
        }
        if not args.skip_videos:
            split_composite_video(source_video, output_videos, args.video_size, state.shape[0], args.resume)

        episode_json = load_episode_json(src.episode_json_path)
        success = bool(episode_json.get("success", src.hdf5_path.stem.endswith("outcome1")))
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt],
                "length": int(state.shape[0]),
                "source_task": src.task,
                "source_episode_idx": src.source_episode_idx,
                "source_seed": src.seed,
                "success": success,
                "source_hdf5": str(src.hdf5_path),
                "source_video": str(source_video),
            }
        )
        stats.add(state, action)
        total_frames += int(state.shape[0])
        if (episode_index + 1) % 25 == 0 or episode_index + 1 == len(sources):
            print(f"[convert] {episode_index + 1}/{len(sources)} episodes, frames={total_frames}", flush=True)

    write_json(args.output_dir / "meta" / "info.json", make_info(len(sources), total_frames, len(task_rows), args.chunks_size, args.fps, args.video_size))
    copy_modality_json(args.output_dir)
    write_json(args.output_dir / "meta" / "stats.json", stats.to_json())
    write_jsonl(args.output_dir / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(args.output_dir / "meta" / "episodes.jsonl", episode_rows)
    readme = (
        "# RoboCasa eval rollouts converted to LeRobot\n\n"
        f"- Source: `{args.input_root}`\n"
        "- States/actions come from per-env-step rollout HDF5.\n"
        "- Videos are split from 3-view composite mp4 into left/right/eye-in-hand views.\n"
        "- This directory is prepared for upload or downstream LeRobot-style loading; it is not uploaded by this script.\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[done] wrote {len(sources)} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
