#!/usr/bin/env python3
"""Probe simple action/state events for RoboCasa LeRobot-flat tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = [
    "PrepareCoffee",
    "MicrowaveThawing",
    "CoffeeSetupMug",
    "CoffeePressButton",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "PnPCounterToMicrowave",
    "TurnOnMicrowave",
]


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    result = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                result.append((start, idx - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        result.append((start, len(mask) - 1))
    return result


def load_episode(task_dir: Path, episode_idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
    modality = json.loads((task_dir / "meta/modality.json").read_text())
    parquet = task_dir / "data/chunk-000" / f"episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(parquet)
    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    return action, state, modality


def first_center(spans: list[tuple[int, int]]) -> int | None:
    if not spans:
        return None
    start, end = spans[0]
    return (start + end) // 2


def last_center(spans: list[tuple[int, int]]) -> int | None:
    if not spans:
        return None
    start, end = spans[-1]
    return (start + end) // 2


def summarize_episode(task_dir: Path, episode_idx: int) -> dict:
    action, state, modality = load_episode(task_dir, episode_idx)
    length = len(action)
    gripper_cfg = modality["action"]["gripper"]
    eef_cfg = modality["action"]["eef_position"]
    base_cfg = modality["action"]["base"]
    qpos_cfg = modality["state"]["gripper_qpos"]
    g = action[:, gripper_cfg["start"] : gripper_cfg["end"]].reshape(-1)
    eef = action[:, eef_cfg["start"] : eef_cfg["end"]]
    base = action[:, base_cfg["start"] : base_cfg["end"]]
    qpos = state[:, qpos_cfg["start"] : qpos_cfg["end"]]
    gap = np.abs(qpos[:, 0] - qpos[:, 1])
    dg = np.r_[0.0, np.abs(np.diff(gap))]

    closing = runs(g > 0, min_len=3)
    opening = runs(g < 0, min_len=3)
    eef_norm = np.linalg.norm(eef, axis=1)
    base_norm = np.linalg.norm(base, axis=1)
    top_eef = np.argsort(-eef_norm)[:5].tolist()
    top_base = np.argsort(-base_norm)[:5].tolist()
    top_gap_change = np.argsort(-dg)[:5].tolist()
    return {
        "episode": episode_idx,
        "length": length,
        "first_close": first_center(closing),
        "last_close": last_center(closing),
        "num_close_runs": len(closing),
        "close_runs": closing[:8],
        "first_open": first_center(opening),
        "last_open": last_center(opening),
        "num_open_runs": len(opening),
        "open_runs": opening[:8],
        "top_eef": top_eef,
        "top_base": top_base,
        "top_gap_change": top_gap_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/junhyeong/data/robocasa_lerobot_flat")
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    all_results = {}
    for task in args.tasks:
        task_dir = data_root / task
        results = [summarize_episode(task_dir, ep) for ep in range(args.episodes)]
        all_results[task] = results
        print(f"\n## {task}")
        for key in ["num_close_runs", "num_open_runs"]:
            vals = [r[key] for r in results]
            uniq = {v: vals.count(v) for v in sorted(set(vals))}
            print(f"{key}: {uniq}")
        for key in ["first_close", "last_open", "top_eef", "top_base"]:
            sample = [(r["episode"], r[key]) for r in results[:5]]
            print(f"{key} sample: {sample}")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
