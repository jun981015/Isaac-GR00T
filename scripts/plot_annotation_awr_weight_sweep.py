#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = "/home/junhyeong/data/robocasa_lerobot_flat"
DEFAULT_ANNOTATION_ROOT = "/home/junhyeong/Value/annotations"
DEFAULT_OUTPUT_ROOT = "local_outputs/annotation_awr_weight_sweep"
DEFAULT_RS = [1.0, 2.0, 3.0, 5.0, 10.0]
DEFAULT_ALPHAS = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0]


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def task_to_annotation_name(task: str) -> str:
    return task.lower()


def episode_ids(dataset_path: Path) -> list[int]:
    return sorted(
        int(path.stem.split("_")[-1])
        for path in (dataset_path / "data/chunk-000").glob("episode_*.parquet")
    )


def load_timestamps(dataset_path: Path, episode_id: int) -> np.ndarray:
    parquet_path = dataset_path / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    return df["timestamp"].to_numpy(dtype=np.float64)


def load_annotation(annotation_root: Path, task: str, episode_id: int, camera: str) -> dict | None:
    ann_task = task_to_annotation_name(task)
    path = (
        annotation_root
        / ann_task
        / "validated"
        / f"{ann_task}_episode_{episode_id:06d}_{camera}.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text())


def segment_lengths(tasks_time: list[dict], critical_tasks: set[str]) -> tuple[float, float]:
    l_crit = 0.0
    l_noncrit = 0.0
    for seg in tasks_time:
        length = float(seg["end_sec"]) - float(seg["start_sec"])
        if seg["task"] in critical_tasks:
            l_crit += length
        else:
            l_noncrit += length
    return l_crit, l_noncrit


def slopes_for_r(l_crit: float, l_noncrit: float, r: float) -> tuple[float, float]:
    if l_crit <= 0.0 and l_noncrit <= 0.0:
        raise ValueError("annotation has zero total duration")
    if l_noncrit <= 0.0:
        return 1.0 / l_crit, 0.0
    if l_crit <= 0.0:
        return 0.0, 1.0 / l_noncrit
    w_n = 1.0 / (r * l_crit + l_noncrit)
    w_c = r * w_n
    return w_c, w_n


def progress_delta_from_annotation(
    annotation: dict,
    timestamps: np.ndarray,
    r: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    tasks_time = annotation["tasks_time"]
    critical_tasks = set(annotation["critical_tasks"])
    duration = float(annotation["duration_sec"])
    l_crit, l_noncrit = segment_lengths(tasks_time, critical_tasks)
    w_c, w_n = slopes_for_r(l_crit, l_noncrit, r)

    if len(timestamps) == 0:
        raise ValueError("empty timestamp array")
    if len(timestamps) == 1:
        frame_ends = np.asarray([duration], dtype=np.float64)
    else:
        frame_ends = np.empty_like(timestamps, dtype=np.float64)
        frame_ends[:-1] = timestamps[1:]
        frame_ends[-1] = duration
    frame_starts = np.clip(timestamps, 0.0, duration)
    frame_ends = np.clip(np.maximum(frame_ends, frame_starts), 0.0, duration)

    delta = np.zeros(len(timestamps), dtype=np.float64)
    critical_mask = np.zeros(len(timestamps), dtype=np.float64)
    for seg in tasks_time:
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        is_critical = seg["task"] in critical_tasks
        slope = w_c if is_critical else w_n
        overlap = np.maximum(0.0, np.minimum(frame_ends, end) - np.maximum(frame_starts, start))
        delta += slope * overlap
        if is_critical:
            critical_mask = np.maximum(critical_mask, (overlap > 0.0).astype(np.float64))

    total = float(delta.sum())
    if total > 0:
        # Keep the r-fixed shape, but remove small timestamp/duration drift.
        delta = delta / total
    progress = np.concatenate([[0.0], np.cumsum(delta)])[:-1]
    meta = {
        "duration_sec": duration,
        "L_crit_sec": l_crit,
        "L_noncrit_sec": l_noncrit,
        "w_c": w_c,
        "w_n": w_n,
        "delta_sum_before_norm": total,
    }
    return delta.astype(np.float32), progress.astype(np.float32), meta


def chunk_mean_delta(delta: np.ndarray, chunk_delta_count: int) -> np.ndarray:
    length = len(delta)
    means = np.empty(length, dtype=np.float32)
    offsets = np.arange(chunk_delta_count)
    for base_index in range(length):
        idx = np.minimum(base_index + offsets, length - 1)
        means[base_index] = float(delta[idx].mean())
    return means


def weights(mean_delta: np.ndarray, alpha: float, clip_max: float) -> np.ndarray:
    return np.clip(np.exp(alpha * mean_delta), 1.0, clip_max)


def percentile_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(np.max(values)),
    }


def aggregate_stats(args: argparse.Namespace, tasks: list[str]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for task in tasks:
        dataset_path = Path(args.dataset_root) / task
        per_r_means: dict[float, list[np.ndarray]] = {r: [] for r in args.rs}
        l_crit_values = []
        l_noncrit_values = []
        missing = 0
        for episode_id in episode_ids(dataset_path):
            annotation = load_annotation(Path(args.annotation_root), task, episode_id, args.annotation_camera)
            if annotation is None:
                missing += 1
                continue
            timestamps = load_timestamps(dataset_path, episode_id)
            for r in args.rs:
                delta, _progress, meta = progress_delta_from_annotation(annotation, timestamps, r)
                per_r_means[r].append(chunk_mean_delta(delta, args.chunk_delta_count))
                if r == args.rs[0]:
                    l_crit_values.append(meta["L_crit_sec"])
                    l_noncrit_values.append(meta["L_noncrit_sec"])

        for r in args.rs:
            if not per_r_means[r]:
                continue
            mean_delta = np.concatenate(per_r_means[r])
            for alpha in args.alphas:
                weight = weights(mean_delta, alpha, args.clip_max)
                row: dict[str, float | str | int] = {
                    "task": task,
                    "episodes_used": len(per_r_means[r]),
                    "annotations_missing": missing,
                    "r": r,
                    "alpha": alpha,
                    "clip_max": args.clip_max,
                    "chunk_delta_count": args.chunk_delta_count,
                    "L_crit_sec_mean": float(np.mean(l_crit_values)) if l_crit_values else 0.0,
                    "L_noncrit_sec_mean": float(np.mean(l_noncrit_values)) if l_noncrit_values else 0.0,
                    "weight_frac_gt_1p1": float(np.mean(weight > 1.1)),
                    "weight_frac_gt_1p5": float(np.mean(weight > 1.5)),
                    "weight_frac_clipped": float(np.mean(weight >= args.clip_max)),
                }
                row.update(percentile_stats(mean_delta, "mean_delta"))
                row.update(percentile_stats(weight, "weight"))
                rows.append(row)
    return rows


def draw_summary_plot(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    width, height = 1800, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    font = load_font(13)
    draw.text((30, 20), "Annotation AWR weight sweep summary", fill=(0, 0, 0), font=title_font)
    draw.text((30, 52), "Each point is task-aggregated stats for one (r, alpha).", fill=(70, 70, 70), font=font)

    sorted_rows = sorted(rows, key=lambda row: (str(row["task"]), float(row["r"]), float(row["alpha"])))
    series = [
        ("weight_p50", np.asarray([row["weight_p50"] for row in sorted_rows], dtype=np.float32), (31, 119, 180)),
        ("weight_p95", np.asarray([row["weight_p95"] for row in sorted_rows], dtype=np.float32), (255, 127, 14)),
        ("weight_max", np.asarray([row["weight_max"] for row in sorted_rows], dtype=np.float32), (214, 39, 40)),
        ("frac_clipped", np.asarray([row["weight_frac_clipped"] for row in sorted_rows], dtype=np.float32), (44, 160, 44)),
    ]

    def panel(box, name, values, color, y_min, y_max):
        left, top, right, bottom = box
        draw.rectangle(box, outline=(180, 180, 180), width=1)
        draw.text((left, top - 22), name, fill=(0, 0, 0), font=font)
        for frac in np.linspace(0, 1, 5):
            y = int(bottom - frac * (bottom - top))
            draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
            val = y_min + frac * (y_max - y_min)
            draw.text((left - 62, y - 8), f"{val:.3f}", fill=(80, 80, 80), font=font)
        points = []
        for i, value in enumerate(values):
            x = int(left + i / max(len(values) - 1, 1) * (right - left))
            y = int(bottom - (float(value) - y_min) / max(y_max - y_min, 1e-9) * (bottom - top))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

    panel((100, 110, 1680, 300), *series[0], 1.0, max(1.05, float(np.max(series[0][1])) * 1.02))
    panel((100, 370, 1680, 560), *series[1], 1.0, max(1.05, float(np.max(series[1][1])) * 1.02))
    panel((100, 630, 1680, 820), *series[2], 1.0, max(1.05, float(np.max(series[2][1])) * 1.02))
    panel((100, 870, 1680, 980), *series[3], 0.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_episode_plot(args: argparse.Namespace, task: str, episode_id: int, output_path: Path) -> None:
    dataset_path = Path(args.dataset_root) / task
    annotation = load_annotation(Path(args.annotation_root), task, episode_id, args.annotation_camera)
    if annotation is None:
        print(f"skip plot, missing annotation: {task} ep{episode_id:06d}")
        return
    timestamps = load_timestamps(dataset_path, episode_id)
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    font = load_font(13)
    draw.text(
        (30, 20),
        f"{task} ep{episode_id:06d}: annotation r-fixed AWR weights",
        fill=(0, 0, 0),
        font=title_font,
    )

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
    ]

    def draw_panel(box, title, series, y_min=None, y_max=None):
        left, top, right, bottom = box
        draw.rectangle(box, outline=(180, 180, 180), width=1)
        draw.text((left, top - 22), title, fill=(0, 0, 0), font=font)
        values = np.concatenate([arr for _label, arr in series])
        if y_min is None:
            y_min = float(values.min())
        if y_max is None:
            y_max = float(values.max())
        pad = max((y_max - y_min) * 0.06, 1e-6)
        y_min -= pad
        y_max += pad
        for frac in np.linspace(0, 1, 5):
            y = int(bottom - frac * (bottom - top))
            draw.line((left, y, right, y), fill=(230, 230, 230), width=1)
            val = y_min + frac * (y_max - y_min)
            draw.text((left - 70, y - 8), f"{val:.4f}", fill=(80, 80, 80), font=font)

        x_max = max(len(timestamps) - 1, 1)

        def xy(idx, value):
            x = int(left + idx / x_max * (right - left))
            y = int(bottom - (float(value) - y_min) / max(y_max - y_min, 1e-9) * (bottom - top))
            return x, y

        critical = set(annotation["critical_tasks"])
        for seg in annotation["tasks_time"]:
            if seg["task"] not in critical:
                continue
            start = int(np.searchsorted(timestamps, float(seg["start_sec"]), side="left"))
            end = int(np.searchsorted(timestamps, float(seg["end_sec"]), side="left"))
            x0 = xy(min(start, x_max), y_min)[0]
            x1 = xy(min(end, x_max), y_min)[0]
            draw.rectangle((x0, top, x1, bottom), fill=(245, 232, 220))

        for i, (label, arr) in enumerate(series):
            color = colors[i % len(colors)]
            points = [xy(idx, value) for idx, value in enumerate(arr)]
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            legend_x = right + 20
            legend_y = top + i * 18
            draw.line((legend_x, legend_y + 8, legend_x + 24, legend_y + 8), fill=color, width=3)
            draw.text((legend_x + 30, legend_y), label, fill=(0, 0, 0), font=font)

    progress_series = []
    delta_series = []
    weight_series = []
    fixed_alpha = args.plot_alpha
    for r in args.rs:
        delta, progress, _meta = progress_delta_from_annotation(annotation, timestamps, r)
        mean_delta = chunk_mean_delta(delta, args.chunk_delta_count)
        progress_series.append((f"r={r:g}", progress))
        delta_series.append((f"r={r:g}", mean_delta))
        weight_series.append((f"r={r:g}, a={fixed_alpha:g}", weights(mean_delta, fixed_alpha, args.clip_max)))

    draw_panel((100, 100, 1280, 380), "progress P(t), critical intervals shaded", progress_series, 0.0, 1.0)
    draw_panel((100, 470, 1280, 740), f"chunk mean_delta, chunk={args.chunk_delta_count}", delta_series)
    draw_panel((100, 830, 1280, 1040), f"weight at alpha={fixed_alpha:g}, clip={args.clip_max:g}", weight_series, 1.0, args.clip_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--annotation-root", default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--rs", nargs="*", type=float, default=DEFAULT_RS)
    parser.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--plot-alpha", type=float, default=100.0)
    parser.add_argument("--clip-max", type=float, default=2.0)
    parser.add_argument("--chunk-delta-count", type=int, default=16)
    parser.add_argument("--annotation-camera", default="robot0_agentview_left")
    parser.add_argument("--episode-id", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    tasks = args.tasks or sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    rows = aggregate_stats(args, tasks)

    output_root = Path(args.output_root)
    write_csv(output_root / "annotation_awr_weight_sweep_stats.csv", rows)
    draw_summary_plot(rows, output_root / "annotation_awr_weight_sweep_summary.png")
    for task in tasks:
        draw_episode_plot(args, task, args.episode_id, output_root / "episodes" / f"{task}_ep{args.episode_id:06d}.png")
    print(f"wrote {output_root / 'annotation_awr_weight_sweep_stats.csv'}")
    print(f"wrote {output_root / 'annotation_awr_weight_sweep_summary.png'}")
    print(f"wrote episode plots under {output_root / 'episodes'}")


if __name__ == "__main__":
    main()
