#!/usr/bin/env python3
"""Measure policy action sensitivity when hand camera is swapped from another episode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from gr00t.eval.robot import RobotInferenceClient


STATE_SLICES = {
    "state.gripper_qpos": (0, 2),
    "state.base_position": (2, 5),
    "state.base_rotation": (5, 9),
    "state.end_effector_position_relative": (9, 12),
    "state.end_effector_rotation_relative": (12, 16),
    "state.gripper_qvel": (16, 18),
    "state.end_effector_position_absolute": (18, 21),
    "state.end_effector_rotation_absolute": (21, 25),
    "state.joint_position": (25, 32),
    "state.joint_position_cos": (32, 39),
    "state.joint_position_sin": (39, 46),
    "state.joint_velocity": (46, 53),
}

VIDEO_KEYS = {
    "video.robot0_agentview_left": "robot0_agentview_left",
    "video.robot0_agentview_right": "robot0_agentview_right",
    "video.robot0_eye_in_hand": "robot0_eye_in_hand",
}

ACTION_GROUPS = {
    "eef_position": "action.eef_position",
    "eef_rotation": "action.eef_rotation",
    "gripper": "action.gripper",
    "base": "action.base",
    "control_mode": "action.control_mode",
}


def read_tasks(dataset: Path) -> dict[int, str]:
    tasks = {}
    with open(dataset / "meta/tasks.jsonl") as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = str(item["task"])
    return tasks


def video_path(dataset: Path, episode_idx: int, camera: str) -> Path:
    return dataset / "videos/chunk-000" / f"observation.images.{camera}" / f"episode_{episode_idx:06d}.mp4"


def read_frame(path: Path, frame_idx: int) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frame = reader.get_data(frame_idx)
    finally:
        reader.close()
    return np.asarray(frame, dtype=np.uint8)


def load_episode(dataset: Path, episode_idx: int) -> pd.DataFrame:
    path = dataset / "data/chunk-000" / f"episode_{episode_idx:06d}.parquet"
    return pd.read_parquet(path)


def choose_samples(df: pd.DataFrame, n_samples: int, margin: int) -> list[int]:
    if len(df) <= 2 * margin:
        indices = np.linspace(0, len(df) - 1, n_samples).round().astype(int)
    else:
        indices = np.linspace(margin, len(df) - margin - 1, n_samples).round().astype(int)
    return [int(x) for x in indices]


def build_obs_batch(
    dataset: Path,
    src_ep: int,
    swap_ep: int,
    frame_indices: list[int],
    swap_hand: bool,
) -> dict[str, Any]:
    src_df = load_episode(dataset, src_ep)
    swap_df = load_episode(dataset, swap_ep)
    tasks = read_tasks(dataset)

    obs: dict[str, Any] = {}
    states = np.stack([np.asarray(src_df.iloc[i]["observation.state"], dtype=np.float32) for i in frame_indices])
    for key, (start, end) in STATE_SLICES.items():
        obs[key] = states[:, None, start:end]

    for policy_key, camera in VIDEO_KEYS.items():
        frames = []
        for i in frame_indices:
            ep = swap_ep if swap_hand and camera == "robot0_eye_in_hand" else src_ep
            frame_idx = min(i, len(swap_df if ep == swap_ep else src_df) - 1)
            frames.append(read_frame(video_path(dataset, ep, camera), frame_idx))
        obs[policy_key] = np.stack(frames, axis=0)[:, None]

    task_indices = [int(src_df.iloc[i]["task_index"]) for i in frame_indices]
    obs["annotation.human.action.task_description"] = np.asarray([tasks[idx] for idx in task_indices])
    return obs


def take_obs(obs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    return {key: value[start:end] if isinstance(value, np.ndarray) else value[start:end] for key, value in obs.items()}


def repeat_obs(obs: dict[str, Any], repeats: int) -> dict[str, Any]:
    repeated: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            repeated[key] = np.repeat(value, repeats, axis=0)
        else:
            repeated[key] = np.repeat(np.asarray(value), repeats, axis=0)
    return repeated


def flatten_action(action: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    groups = {}
    flat_parts = []
    for short, key in ACTION_GROUPS.items():
        arr = np.asarray(action[key], dtype=np.float32)
        # Expected shape: B x H x D.
        arr = arr.reshape(arr.shape[0], -1)
        groups[short] = arr
        flat_parts.append(arr)
    return np.concatenate(flat_parts, axis=1), groups


def l2_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/junhyeong/data/robocasa_lerobot_flat/PnPCounterToSink")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18161)
    parser.add_argument("--src_ep", type=int, default=0)
    parser.add_argument("--swap_ep", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_noise", type=int, default=32)
    parser.add_argument("--batch_action_seed", type=int, default=0)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        default="/home/junhyeong/Value/Isaac-GR00T/local_outputs/robocasa_benchmark/weight1_hand_swap_action_norm",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src_df = load_episode(dataset, args.src_ep)
    frame_indices = choose_samples(src_df, args.n_samples, args.margin)
    original_obs = build_obs_batch(dataset, args.src_ep, args.swap_ep, frame_indices, swap_hand=False)
    swapped_obs = build_obs_batch(dataset, args.src_ep, args.swap_ep, frame_indices, swap_hand=True)

    client = RobotInferenceClient(host=args.host, port=args.port, timeout_ms=120000)

    if args.max_batch_size < args.n_noise:
        raise ValueError("--max_batch_size must be >= --n_noise")

    samples_per_call = max(1, args.max_batch_size // args.n_noise)
    sample_diffs = []
    sample_orig_norms = []
    sample_swap_norms = []
    sample_group_diffs: dict[str, list[np.ndarray]] = {name: [] for name in ACTION_GROUPS}
    raw_rows = []
    for start in range(0, args.n_samples, samples_per_call):
        end = min(args.n_samples, start + samples_per_call)
        chunk_n = end - start
        orig_chunk = repeat_obs(take_obs(original_obs, start, end), args.n_noise)
        swap_chunk = repeat_obs(take_obs(swapped_obs, start, end), args.n_noise)

        # Same action_seed and same batch shape aligns the sampled diffusion noise
        # between original and hand-swapped inputs while still producing diverse
        # noise across batch elements.
        orig_action = client.get_action_seeded(orig_chunk, action_seed=args.batch_action_seed)
        swap_action = client.get_action_seeded(swap_chunk, action_seed=args.batch_action_seed)
        orig_flat, orig_groups = flatten_action(orig_action)
        swap_flat, swap_groups = flatten_action(swap_action)
        diff = swap_flat - orig_flat
        diff_norm = l2_rows(diff).reshape(chunk_n, args.n_noise)
        orig_norm = l2_rows(orig_flat).reshape(chunk_n, args.n_noise)
        swap_norm = l2_rows(swap_flat).reshape(chunk_n, args.n_noise)
        sample_diffs.append(diff_norm)
        sample_orig_norms.append(orig_norm)
        sample_swap_norms.append(swap_norm)

        group_norms = {}
        for group in ACTION_GROUPS:
            group_diff = l2_rows(swap_groups[group] - orig_groups[group]).reshape(chunk_n, args.n_noise)
            sample_group_diffs[group].append(group_diff)
            group_norms[group] = group_diff

        for local_sample in range(chunk_n):
            global_sample = start + local_sample
            for noise_idx in range(args.n_noise):
                row = {
                    "sample": global_sample,
                    "noise_idx": noise_idx,
                    "src_ep": args.src_ep,
                    "swap_ep": args.swap_ep,
                    "frame_idx": int(frame_indices[global_sample]),
                    "time_sec": float(frame_indices[global_sample] / 20.0),
                    "action_diff_l2": float(diff_norm[local_sample, noise_idx]),
                    "original_action_l2": float(orig_norm[local_sample, noise_idx]),
                    "swapped_action_l2": float(swap_norm[local_sample, noise_idx]),
                    "relative_diff_to_original_norm": float(
                        diff_norm[local_sample, noise_idx] / max(orig_norm[local_sample, noise_idx], 1e-8)
                    ),
                }
                for group in ACTION_GROUPS:
                    row[f"{group}_diff_l2"] = float(group_norms[group][local_sample, noise_idx])
                raw_rows.append(row)

    diff_arr = np.concatenate(sample_diffs, axis=0)
    orig_norm_arr = np.concatenate(sample_orig_norms, axis=0)
    swap_norm_arr = np.concatenate(sample_swap_norms, axis=0)
    group_diff_arr = {
        group: np.concatenate(values, axis=0) for group, values in sample_group_diffs.items()
    }
    sample_rows = []
    for sample_i, frame_idx in enumerate(frame_indices):
        sample_rows.append(
            {
                "sample": sample_i,
                "src_ep": args.src_ep,
                "swap_ep": args.swap_ep,
                "frame_idx": int(frame_idx),
                "time_sec": float(frame_idx / 20.0),
                "mean_action_diff_l2_over_noise": float(diff_arr[sample_i].mean()),
                "std_action_diff_l2_over_noise": float(diff_arr[sample_i].std()),
                "mean_original_action_l2_over_noise": float(orig_norm_arr[sample_i].mean()),
                "mean_swapped_action_l2_over_noise": float(swap_norm_arr[sample_i].mean()),
                "relative_diff_to_original_norm": float(
                    diff_arr[sample_i].mean() / max(orig_norm_arr[sample_i].mean(), 1e-8)
                ),
            }
        )

    raw_csv = output_dir / "noise_level_action_norm.csv"
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    sample_csv = output_dir / "sample_level_action_norm.csv"
    with open(sample_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = {
        "dataset": str(dataset),
        "src_ep": args.src_ep,
        "swap_ep": args.swap_ep,
        "n_samples": args.n_samples,
        "n_noise": args.n_noise,
        "batch_action_seed": args.batch_action_seed,
        "max_batch_size": args.max_batch_size,
        "samples_per_call": samples_per_call,
        "frame_indices": frame_indices,
        "mean_action_diff_l2": float(diff_arr.mean()),
        "std_action_diff_l2": float(diff_arr.std()),
        "median_action_diff_l2": float(np.median(diff_arr)),
        "mean_original_action_l2": float(orig_norm_arr.mean()),
        "mean_swapped_action_l2": float(swap_norm_arr.mean()),
        "relative_diff_to_original_norm": float(diff_arr.mean() / max(orig_norm_arr.mean(), 1e-8)),
        "group_mean_diff_l2": {
            group: float(values.mean()) for group, values in group_diff_arr.items()
        },
        "raw_csv": str(raw_csv),
        "sample_csv": str(sample_csv),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
