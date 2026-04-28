#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TASKS = ("PnPCounterToSink", "PnPCounterToStove", "PnPMicrowaveToCounter")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    result = []
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


def detect_pick_place(
    gripper_action: np.ndarray,
    gripper_gap: np.ndarray,
    stuck_window: int,
    stuck_threshold: float,
) -> tuple[int, int]:
    closing_runs = [(s, e) for s, e in runs(gripper_action > 0) if e - s + 1 >= 3]
    opening_runs = [(s, e) for s, e in runs(gripper_action < 0) if e - s + 1 >= 5]
    pick_frame = len(gripper_action) // 3
    if closing_runs:
        close_start, close_end = closing_runs[0]
        pick_frame = close_end
        last_start = close_end - stuck_window + 1
        for frame in range(close_start, max(close_start, last_start) + 1):
            window = gripper_gap[frame : frame + stuck_window]
            if len(window) == stuck_window and window.max() - window.min() <= stuck_threshold:
                pick_frame = frame
                break
    place_open_frame = int(len(gripper_action) * 0.8)
    for open_start, _ in opening_runs:
        if open_start > pick_frame:
            place_open_frame = open_start
            break
    return int(pick_frame), int(place_open_frame)


def progress_delta(
    length: int,
    pick_frame: int,
    place_open_frame: int,
    pick_radius: int,
    place_radius: int,
    pick_mass: float,
    place_mass: float,
    other_mass: float,
    place_offset: int,
) -> np.ndarray:
    event_delta = np.zeros(length, dtype=np.float32)
    pick_frames = np.arange(pick_frame - pick_radius, pick_frame + pick_radius + 1)
    place_center = place_open_frame + place_offset
    place_frames = np.arange(place_center - place_radius, place_center + place_radius + 1)
    pick_frames = pick_frames[(0 <= pick_frames) & (pick_frames < length)]
    place_frames = place_frames[(0 <= place_frames) & (place_frames < length)]
    if len(pick_frames) > 0:
        event_delta[pick_frames] += pick_mass / len(pick_frames)
    if len(place_frames) > 0:
        event_delta[place_frames] += place_mass / len(place_frames)
    non_event = event_delta == 0
    if non_event.any():
        event_delta[non_event] = other_mass / non_event.sum()
    return event_delta


def chunk_mean(delta: np.ndarray, chunk_delta_count: int) -> np.ndarray:
    values = []
    length = len(delta)
    for base_index in range(length):
        idx = np.minimum(np.arange(base_index, base_index + chunk_delta_count), length - 1)
        values.append(float(delta[idx].mean()))
    return np.asarray(values, dtype=np.float32)


def weights(mean_delta: np.ndarray, alpha: float, clip_max: float) -> np.ndarray:
    return np.clip(np.exp(alpha * mean_delta), 0.0, clip_max)


def plot_episode(args: argparse.Namespace, task: str, episode_id: int) -> dict[str, float]:
    dataset_path = Path(args.dataset_root) / task
    gripper_action, gripper_gap = episode_arrays(dataset_path, episode_id)
    length = len(gripper_action)
    pick_frame, place_open_frame = detect_pick_place(
        gripper_action,
        gripper_gap,
        stuck_window=args.stuck_window,
        stuck_threshold=args.stuck_threshold,
    )
    place_frame = min(length - 1, place_open_frame + args.place_offset)

    series_w: list[tuple[str, np.ndarray]] = []
    series_d: list[tuple[str, np.ndarray]] = []
    stats = {}
    for pick_radius in args.pick_radii:
        for place_radius in args.place_radii:
            delta = progress_delta(
                length,
                pick_frame,
                place_open_frame,
                pick_radius,
                place_radius,
                args.pick_mass,
                args.place_mass,
                args.other_mass,
                args.place_offset,
            )
            mean_delta = chunk_mean(delta, args.chunk_delta_count)
            label_base = f"p±{pick_radius}/pl±{place_radius}"
            series_d.append((label_base, mean_delta))
            for alpha in args.alphas:
                w = weights(mean_delta, alpha, args.clip_max)
                label = f"{label_base}, a={alpha:g}"
                series_w.append((label, w))
                stats[f"{label}_min"] = float(w.min())
                stats[f"{label}_max"] = float(w.max())
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_ep{episode_id:06d}_weight_combinations.png"
    draw_two_panel_plot(
        out_path,
        title=f"{task} ep{episode_id:06d}: weight = clip(exp(alpha * mean_delta), max={args.clip_max})",
        top_title="final weight",
        top_series=series_w,
        top_ylim=(0.95, args.clip_max + 0.05),
        bottom_title="chunk mean_delta",
        bottom_series=series_d,
        bottom_ylim=None,
        vlines=[(pick_frame, "pick", (0, 150, 0)), (place_frame, "place", (210, 40, 40))],
        x_max=length - 1,
    )
    print(f"saved {out_path}")
    return stats


def palette() -> list[tuple[int, int, int]]:
    return [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (20, 80, 160),
        (200, 90, 20),
    ]


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, np.ndarray]],
    ylim: tuple[float, float] | None,
    x_max: int,
    vlines: list[tuple[int, str, tuple[int, int, int]]],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    colors = palette()
    draw.rectangle(box, outline=(180, 180, 180), width=1)
    draw.text((left, top - 22), title, fill=(0, 0, 0), font=font)

    vals = np.concatenate([v for _, v in series]) if series else np.asarray([0.0, 1.0])
    if ylim is None:
        y_min = float(vals.min())
        y_max = float(vals.max())
        pad = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = ylim

    for frac in np.linspace(0, 1, 5):
        y = int(bottom - frac * (bottom - top))
        draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
        value = y_min + frac * (y_max - y_min)
        draw.text((left - 58, y - 8), f"{value:.4f}", fill=(80, 80, 80), font=font)

    def xy(idx: int, value: float) -> tuple[int, int]:
        x = int(left + (idx / max(x_max, 1)) * (right - left))
        y = int(bottom - ((value - y_min) / max(y_max - y_min, 1e-9)) * (bottom - top))
        return x, y

    for frame, label, color in vlines:
        x = xy(frame, y_min)[0]
        draw.line((x, top, x, bottom), fill=color, width=2)
        draw.text((x + 4, top + 4), label, fill=color, font=font)

    for i, (label, values) in enumerate(series):
        color = colors[i % len(colors)]
        points = [xy(idx, float(value)) for idx, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        legend_x = right + 18
        legend_y = top + i * 18
        draw.line((legend_x, legend_y + 8, legend_x + 24, legend_y + 8), fill=color, width=3)
        draw.text((legend_x + 30, legend_y), label, fill=(0, 0, 0), font=font)


def draw_two_panel_plot(
    output_path: Path,
    title: str,
    top_title: str,
    top_series: list[tuple[str, np.ndarray]],
    top_ylim: tuple[float, float] | None,
    bottom_title: str,
    bottom_series: list[tuple[str, np.ndarray]],
    bottom_ylim: tuple[float, float] | None,
    vlines: list[tuple[int, str, tuple[int, int, int]]],
    x_max: int,
) -> None:
    width, height = 1800, 1000
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(24)
    font = load_font(13)
    draw.text((30, 20), title, fill=(0, 0, 0), font=title_font)
    plot_left, plot_right = 100, 1250
    draw_panel(
        draw,
        (plot_left, 90, plot_right, 470),
        top_title,
        top_series,
        top_ylim,
        x_max,
        vlines,
        font,
    )
    draw_panel(
        draw,
        (plot_left, 570, plot_right, 930),
        bottom_title,
        bottom_series,
        bottom_ylim,
        x_max,
        vlines,
        font,
    )
    draw.text((plot_left + 480, 950), "base frame index", fill=(0, 0, 0), font=font)
    img.save(output_path)


def aggregate_stats(args: argparse.Namespace) -> list[dict[str, float | str]]:
    rows = []
    for task in args.tasks:
        dataset_path = Path(args.dataset_root) / task
        episode_files = sorted((dataset_path / "data/chunk-000").glob("episode_*.parquet"))
        episode_ids = [int(path.stem.split("_")[-1]) for path in episode_files]
        for pick_radius in args.pick_radii:
            for place_radius in args.place_radii:
                all_mean_delta = []
                for episode_id in episode_ids:
                    gripper_action, gripper_gap = episode_arrays(dataset_path, episode_id)
                    pick_frame, place_open_frame = detect_pick_place(
                        gripper_action,
                        gripper_gap,
                        stuck_window=args.stuck_window,
                        stuck_threshold=args.stuck_threshold,
                    )
                    delta = progress_delta(
                        len(gripper_action),
                        pick_frame,
                        place_open_frame,
                        pick_radius,
                        place_radius,
                        args.pick_mass,
                        args.place_mass,
                        args.other_mass,
                        args.place_offset,
                    )
                    all_mean_delta.append(chunk_mean(delta, args.chunk_delta_count))
                mean_delta = np.concatenate(all_mean_delta)
                for alpha in args.alphas:
                    w = weights(mean_delta, alpha, args.clip_max)
                    rows.append(
                        {
                            "task": task,
                            "pick_radius": pick_radius,
                            "place_radius": place_radius,
                            "alpha": alpha,
                            "mean_delta_min": float(mean_delta.min()),
                            "mean_delta_max": float(mean_delta.max()),
                            "weight_min": float(w.min()),
                            "weight_p50": float(np.percentile(w, 50)),
                            "weight_p95": float(np.percentile(w, 95)),
                            "weight_max": float(w.max()),
                        }
                    )
    return rows


def plot_summary(args: argparse.Namespace, rows: list[dict[str, float | str]]) -> None:
    out_path = Path(args.output_root) / "summary_weight_distribution.png"
    p50 = np.asarray([row["weight_p50"] for row in rows], dtype=np.float32)
    p95 = np.asarray([row["weight_p95"] for row in rows], dtype=np.float32)
    mx = np.asarray([row["weight_max"] for row in rows], dtype=np.float32)
    draw_two_panel_plot(
        out_path,
        title="AWR weight distribution after exp(alpha * mean_delta) and clipping",
        top_title="weight percentiles",
        top_series=[("p50", p50), ("p95", p95), ("max", mx)],
        top_ylim=(0.95, args.clip_max + 0.05),
        bottom_title="mean_delta max by combination",
        bottom_series=[
            ("mean_delta_max", np.asarray([row["mean_delta_max"] for row in rows], dtype=np.float32))
        ],
        bottom_ylim=None,
        vlines=[],
        x_max=len(rows) - 1,
    )
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="/home/junhyeong/data/robocasa_lerobot")
    parser.add_argument("--output_root", default="/home/junhyeong/images/awr_weight_graphs")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--alphas", nargs="*", type=float, default=[10.0, 15.0, 20.0])
    parser.add_argument("--pick_radii", nargs="*", type=int, default=[2, 5])
    parser.add_argument("--place_radii", nargs="*", type=int, default=[2, 5])
    parser.add_argument("--clip_max", type=float, default=2.0)
    parser.add_argument("--pick_mass", type=float, default=0.3)
    parser.add_argument("--place_mass", type=float, default=0.3)
    parser.add_argument("--other_mass", type=float, default=0.4)
    parser.add_argument("--place_offset", type=int, default=8)
    parser.add_argument("--chunk_delta_count", type=int, default=15)
    parser.add_argument("--stuck_window", type=int, default=10)
    parser.add_argument("--stuck_threshold", type=float, default=0.001)
    args = parser.parse_args()

    for task in args.tasks:
        plot_episode(args, task, args.episode_id)
    rows = aggregate_stats(args)
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_weight_stats.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"saved {csv_path}")
    plot_summary(args, rows)


if __name__ == "__main__":
    main()
