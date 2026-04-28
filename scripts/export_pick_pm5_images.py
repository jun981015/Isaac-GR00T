#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TASKS = ("PnPCounterToSink", "PnPCounterToStove", "PnPMicrowaveToCounter")
OFFSETS = tuple(range(-5, 6))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            result.append((start, idx - 1))
            start = None
    if start is not None:
        result.append((start, len(mask) - 1))
    return result


def detect_pick_frame(
    gripper_action: np.ndarray,
    gripper_gap: np.ndarray,
    stuck_window: int,
    stuck_threshold: float,
) -> int:
    closing_runs = [(s, e) for s, e in runs(gripper_action > 0) if e - s + 1 >= 3]
    if not closing_runs:
        return len(gripper_action) // 3

    close_start, close_end = closing_runs[0]
    pick_frame = close_end
    last_start = close_end - stuck_window + 1
    for frame in range(close_start, max(close_start, last_start) + 1):
        window = gripper_gap[frame : frame + stuck_window]
        if len(window) == stuck_window and window.max() - window.min() <= stuck_threshold:
            pick_frame = frame
            break
    return int(pick_frame)


def episode_arrays(dataset_path: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    modality = json.loads((dataset_path / "meta/modality.json").read_text())
    action_cfg = modality["action"]["gripper"]
    state_cfg = modality["state"]["gripper_qpos"]
    parquet_path = dataset_path / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
    df = pd.read_parquet(parquet_path)

    action = np.stack(df[action_cfg["original_key"]].to_numpy())
    state = np.stack(df[state_cfg["original_key"]].to_numpy())
    gripper_action = action[:, action_cfg["start"] : action_cfg["end"]].reshape(-1)
    gripper_qpos = state[:, state_cfg["start"] : state_cfg["end"]]
    gripper_gap = np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1])
    return gripper_action, gripper_gap


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def make_grid(
    items: list[tuple[int, int, int, Image.Image]],
    output_path: Path,
    cell_size: int,
) -> None:
    cols, rows = 3, 3
    label_h = 26
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for idx, (episode_id, pick_frame, frame_idx, image) in enumerate(items):
        row, col = divmod(idx, cols)
        x = col * cell_size
        y = row * (cell_size + label_h)
        image = image.resize((cell_size, cell_size), Image.Resampling.BILINEAR)
        canvas.paste(image, (x, y + label_h))
        draw.text(
            (x + 4, y + 4),
            f"ep{episode_id:02d} pick={pick_frame} frame={frame_idx}",
            fill=(0, 0, 0),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def offset_dir(offset: int) -> str:
    if offset < 0:
        return f"minus_{abs(offset)}"
    if offset > 0:
        return f"plus_{offset}"
    return "center"


def export_task(args: argparse.Namespace, task: str) -> None:
    dataset_path = Path(args.dataset_root) / task
    video_dir = dataset_path / "videos/chunk-000/observation.images.robot0_eye_in_hand"
    episode_files = sorted((dataset_path / "data/chunk-000").glob("episode_*.parquet"))
    episode_ids = [int(path.stem.split("_")[-1]) for path in episode_files]

    picks: dict[int, int] = {}
    for episode_id in episode_ids:
        gripper_action, gripper_gap = episode_arrays(dataset_path, episode_id)
        picks[episode_id] = detect_pick_frame(
            gripper_action,
            gripper_gap,
            stuck_window=args.stuck_window,
            stuck_threshold=args.stuck_threshold,
        )

    for offset in OFFSETS:
        frames: list[tuple[int, int, int, Image.Image]] = []
        for episode_id in episode_ids:
            pick_frame = picks[episode_id]
            gripper_action, _ = episode_arrays(dataset_path, episode_id)
            frame_idx = int(np.clip(pick_frame + offset, 0, len(gripper_action) - 1))
            video_path = video_dir / f"episode_{episode_id:06d}.mp4"
            frames.append((episode_id, pick_frame, frame_idx, read_frame(video_path, frame_idx)))

        out_dir = Path(args.output_root) / task / "pick_pm5" / offset_dir(offset)
        for chunk_idx in range(0, len(frames), 9):
            grid_items = frames[chunk_idx : chunk_idx + 9]
            make_grid(
                grid_items,
                out_dir / f"image{chunk_idx // 9 + 1}.png",
                cell_size=args.cell_size,
            )

    print(f"{task}: exported {len(episode_ids)} episodes to {Path(args.output_root) / task / 'pick_pm5'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/junhyeong/data/robocasa_lerobot")
    parser.add_argument("--output_root", default="/home/junhyeong/images")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--stuck_window", type=int, default=10)
    parser.add_argument("--stuck_threshold", type=float, default=0.001)
    parser.add_argument("--cell_size", type=int, default=256)
    args = parser.parse_args()

    for task in args.tasks:
        export_task(args, task)


if __name__ == "__main__":
    main()
