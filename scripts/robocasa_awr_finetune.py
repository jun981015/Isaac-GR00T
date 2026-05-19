# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import tyro
from transformers import TrainerCallback, TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, GR00TTransform
from gr00t.utils.peft import get_lora_model
from scripts.gr00t_finetune import _copy_partial_action_expert_weights


@dataclass
class ArgsConfig:
    dataset_path: List[str]
    output_dir: str = "/tmp/gr00t_robocasa_awr"
    data_config: str = "robocasa_n15_data_config:RobocasaKitchenPnPDataConfig"
    batch_size: int = 64
    max_steps: int = 20000
    num_gpus: int = 1
    save_steps: int = 5000
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = False
    tune_diffusion_model: bool = True
    resume: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False
    dataloader_num_workers: int = 12
    gradient_accumulation_steps: int = 1
    dataloader_prefetch_factor: int = 4
    report_to: Literal["wandb", "tensorboard", "azure_ml", "none"] = "wandb"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    awr_source: Literal["heuristic", "annotation"] = "heuristic"
    annotation_root: str = ""
    annotation_version: str = "v1"
    annotation_split: str = "validated"
    annotation_camera: str = "robot0_agentview_left"
    annotation_r: float = 5.0
    awr_alpha: float = 20.0
    awr_clip_max: float = 1.8
    task_alpha_map: str = (
        "PrepareCoffee=50,MicrowaveThawing=70,"
        "CoffeeSetupMug=40,PnPCounterToMicrowave=40,"
        "CoffeePressButton=20,OpenSingleDoor=20,CloseSingleDoor=20,TurnOnMicrowave=20"
    )
    critical_mass: float = 0.8
    critical_radius: int = 16
    pick_mass: float = 0.3
    place_mass: float = 0.3
    other_mass: float = 0.4
    gripper_stuck_window: int = 10
    gripper_stuck_threshold: float = 0.001
    action_chunk_delta_count: int = 16
    place_offset: int = 8
    pick_radius: int = 2
    place_radius: int = 2
    compact_checkpoints_on_save: bool = False
    compact_base_model_path: str = ""
    compact_link_base_model_path: str = ""
    compact_final_model: bool = True
    compact_fail_fast: bool = True


def compact_model_weights_in_place(
    checkpoint_dir: Path,
    base_model_path: Path,
    *,
    link_base_model_path: Path | None = None,
    fail_fast: bool = True,
) -> bool:
    """Replace only model safetensors with compact symlink-backed shards.

    Non-model training files such as optimizer.pt, scheduler.pt, rng states,
    trainer_state.json, and experiment_cfg are intentionally left untouched.
    """

    checkpoint_dir = checkpoint_dir.resolve()
    base_model_path = base_model_path.resolve()
    if link_base_model_path is not None:
        link_base_model_path = link_base_model_path.expanduser()
    if (checkpoint_dir / "compact_summary.json").exists():
        print(f"[compact] skip already compact: {checkpoint_dir}", flush=True)
        return True
    if not (checkpoint_dir / "model.safetensors.index.json").exists():
        print(f"[compact] skip missing model index: {checkpoint_dir}", flush=True)
        return False
    if not (base_model_path / "model.safetensors.index.json").exists():
        message = f"[compact] base model index missing: {base_model_path}"
        if fail_fast:
            raise FileNotFoundError(message)
        print(message, flush=True)
        return False

    repo_dir = Path(__file__).resolve().parents[1]
    compact_script = repo_dir / "scripts" / "compact_groot_checkpoint.py"
    tmp_dir = checkpoint_dir / ".compact_tmp"
    backup_dir = checkpoint_dir / ".model_full_backup_tmp"
    if tmp_dir.exists() or backup_dir.exists():
        message = f"[compact] temp path exists for {checkpoint_dir}: {tmp_dir} {backup_dir}"
        if fail_fast:
            raise FileExistsError(message)
        print(message, flush=True)
        return False

    try:
        print(f"[compact] start model weight compact: {checkpoint_dir}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(compact_script),
                "--checkpoint",
                str(checkpoint_dir),
                "--base",
                str(base_model_path),
                *(
                    ["--link-base", str(link_base_model_path)]
                    if link_base_model_path is not None
                    else []
                ),
                "--output",
                str(tmp_dir),
            ],
            check=True,
            cwd=repo_dir,
        )
        required = [
            tmp_dir / "model.safetensors.index.json",
            tmp_dir / "model-changed.safetensors",
            tmp_dir / "compact_summary.json",
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(f"compact output missing required model files: {tmp_dir}")

        backup_dir.mkdir()
        old_model_files = list(checkpoint_dir.glob("model*.safetensors")) + [
            checkpoint_dir / "model.safetensors.index.json"
        ]
        for path in old_model_files:
            if path.exists():
                path.rename(backup_dir / path.name)

        compact_model_files = (
            list(tmp_dir.glob("base-*.safetensors"))
            + [
                tmp_dir / "model-changed.safetensors",
                tmp_dir / "model.safetensors.index.json",
                tmp_dir / "compact_summary.json",
            ]
        )
        for path in compact_model_files:
            if path.exists() or path.is_symlink():
                path.rename(checkpoint_dir / path.name)

        shutil.rmtree(tmp_dir)
        shutil.rmtree(backup_dir)
        print(f"[compact] done model weight compact: {checkpoint_dir}", flush=True)
        return True
    except Exception as exc:
        if backup_dir.exists():
            for path in backup_dir.iterdir():
                target = checkpoint_dir / path.name
                if not target.exists():
                    path.rename(target)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)
        if fail_fast:
            raise
        print(f"[compact] failed for {checkpoint_dir}: {exc}", flush=True)
        return False


class CompactModelWeightsCallback(TrainerCallback):
    def __init__(
        self,
        base_model_path: str,
        link_base_model_path: str = "",
        fail_fast: bool = True,
    ):
        self.base_model_path = Path(base_model_path).expanduser()
        self.link_base_model_path = (
            Path(link_base_model_path).expanduser() if link_base_model_path else None
        )
        self.fail_fast = fail_fast

    def on_save(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        compact_model_weights_in_place(
            checkpoint_dir,
            self.base_model_path,
            link_base_model_path=self.link_base_model_path,
            fail_fast=self.fail_fast,
        )


class AWRLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        awr_alpha: float,
        awr_clip_max: float,
        task_alpha_map: str,
        critical_mass: float,
        critical_radius: int,
        pick_mass: float,
        place_mass: float,
        other_mass: float,
        gripper_stuck_window: int,
        gripper_stuck_threshold: float,
        action_chunk_delta_count: int,
        place_offset: int,
        pick_radius: int,
        place_radius: int,
        awr_source: str,
        annotation_root: str,
        annotation_version: str,
        annotation_split: str,
        annotation_camera: str,
        annotation_r: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.awr_alpha = awr_alpha
        self.awr_clip_max = awr_clip_max
        self.task_alpha_map = self._parse_task_alpha_map(task_alpha_map)
        self.task_name = self.dataset_path.name
        self.task_alpha = self.task_alpha_map.get(self.task_name, awr_alpha)
        self.critical_mass = critical_mass
        self.critical_radius = critical_radius
        self.pick_mass = pick_mass
        self.place_mass = place_mass
        self.other_mass = other_mass
        self.gripper_stuck_window = gripper_stuck_window
        self.gripper_stuck_threshold = gripper_stuck_threshold
        self.action_chunk_delta_count = action_chunk_delta_count
        self.place_offset = place_offset
        self.pick_radius = pick_radius
        self.place_radius = place_radius
        self.awr_source = awr_source
        self.annotation_root = Path(annotation_root).expanduser() if annotation_root else None
        self.annotation_version = annotation_version
        self.annotation_split = annotation_split
        self.annotation_camera = annotation_camera
        self.annotation_r = annotation_r
        self._loss_weights = self._precompute_loss_weights()

    @staticmethod
    def _parse_task_alpha_map(raw: str) -> dict[str, float]:
        mapping = {}
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", 1)
            mapping[key.strip()] = float(value)
        return mapping

    @staticmethod
    def _annotation_task_name(task_name: str) -> str:
        return task_name.lower()

    @staticmethod
    def _slopes_for_r(critical_length: float, noncritical_length: float, r: float) -> tuple[float, float]:
        if critical_length <= 0.0 and noncritical_length <= 0.0:
            raise ValueError("annotation has zero total duration")
        if noncritical_length <= 0.0:
            return 1.0 / critical_length, 0.0
        if critical_length <= 0.0:
            return 0.0, 1.0 / noncritical_length
        noncritical_slope = 1.0 / (r * critical_length + noncritical_length)
        critical_slope = r * noncritical_slope
        return critical_slope, noncritical_slope

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx - 1))
                start = None
        if start is not None:
            runs.append((start, len(mask) - 1))
        return runs

    def _episode_arrays(self, trajectory_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=self.get_episode_chunk(trajectory_id),
            episode_index=trajectory_id,
        )
        df = pd.read_parquet(parquet_path)
        modality = json.loads((self.dataset_path / "meta/modality.json").read_text())
        action_cfg = modality["action"]["gripper"]
        state_cfg = modality["state"]["gripper_qpos"]
        eef_rel_cfg = modality["state"].get("end_effector_position_relative")
        action = np.stack(df[action_cfg["original_key"]].to_numpy())
        state = np.stack(df[state_cfg["original_key"]].to_numpy())
        gripper_action = action[:, action_cfg["start"] : action_cfg["end"]].reshape(-1)
        gripper_qpos = state[:, state_cfg["start"] : state_cfg["end"]]
        gripper_gap = np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1])
        if eef_rel_cfg is None:
            eef_x_relative = np.zeros(len(action), dtype=np.float32)
        else:
            eef_x_relative = state[:, eef_rel_cfg["start"]].astype(np.float32)
        return gripper_action, gripper_gap, eef_x_relative

    def _detect_pick_place(self, gripper_action: np.ndarray, gripper_gap: np.ndarray) -> tuple[int, int]:
        closing_runs = [(s, e) for s, e in self._runs(gripper_action > 0) if e - s + 1 >= 3]
        opening_runs = [(s, e) for s, e in self._runs(gripper_action < 0) if e - s + 1 >= 5]
        pick_frame = len(gripper_action) // 3
        if closing_runs:
            close_start, close_end = closing_runs[0]
            pick_frame = close_end
            last_start = close_end - self.gripper_stuck_window + 1
            for frame in range(close_start, max(close_start, last_start) + 1):
                window = gripper_gap[frame : frame + self.gripper_stuck_window]
                if len(window) == self.gripper_stuck_window and window.max() - window.min() <= self.gripper_stuck_threshold:
                    pick_frame = frame
                    break
        place_open_frame = int(len(gripper_action) * 0.8)
        for open_start, _ in opening_runs:
            if open_start > pick_frame:
                place_open_frame = open_start
                break
        return pick_frame, place_open_frame

    def _first_stuck_gap(self, gripper_gap: np.ndarray, start: int, end: int) -> int:
        for frame in range(start, max(start, end - self.gripper_stuck_window + 2)):
            window = gripper_gap[frame : frame + self.gripper_stuck_window]
            if len(window) == self.gripper_stuck_window and window.max() - window.min() <= self.gripper_stuck_threshold:
                return frame
        return end

    @staticmethod
    def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, np.ones(window, dtype=np.float32) / window, mode="valid")

    @staticmethod
    def _local_peaks(values: np.ndarray, min_prominence: float = 0.01) -> list[int]:
        peaks = []
        for idx in range(1, len(values) - 1):
            if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
                left = max(0, idx - 10)
                right = min(len(values), idx + 11)
                prominence = values[idx] - min(values[left : idx + 1].min(), values[idx:right].min())
                if prominence >= min_prominence:
                    peaks.append(idx)
        return peaks

    def _prepare_coffee_press(self, eef_x_relative: np.ndarray) -> int:
        smoothed_x = self._smooth(eef_x_relative, window=9)
        min_frame = int(round((len(eef_x_relative) - 1) * 0.85))
        late_peaks = [peak for peak in self._local_peaks(smoothed_x, min_prominence=0.01) if peak >= min_frame]
        if late_peaks:
            return int(late_peaks[-1])
        return int(min_frame + np.argmax(smoothed_x[min_frame:]))

    def _detect_task_events(
        self,
        trajectory_id: int,
        gripper_action: np.ndarray,
        gripper_gap: np.ndarray,
        eef_x_relative: np.ndarray,
    ) -> list[tuple[str, int]]:
        closing_runs = [(s, e) for s, e in self._runs(gripper_action > 0) if e - s + 1 >= 3]
        opening_runs_3 = [(s, e) for s, e in self._runs(gripper_action < 0) if e - s + 1 >= 3]
        opening_runs_5 = [(s, e) for s, e in self._runs(gripper_action < 0) if e - s + 1 >= 5]
        length = len(gripper_action)

        def first_pick() -> int:
            return self._first_stuck_gap(gripper_gap, *closing_runs[0]) if closing_runs else length // 3

        def first_open_after(frame: int, min_len_5: bool = False) -> int:
            openings = opening_runs_5 if min_len_5 else opening_runs_3
            return int(next((start for start, _ in openings if start > frame), openings[-1][0] if openings else min(length - 1, frame + 1)))

        if self.task_name == "PrepareCoffee":
            pick = first_pick()
            place = first_open_after(pick)
            return [("pick", pick), ("place", place), ("press", self._prepare_coffee_press(eef_x_relative))]

        if self.task_name == "CoffeeSetupMug":
            pick = first_pick()
            return [("pick", pick), ("place", first_open_after(pick))]

        if self.task_name == "CoffeePressButton":
            press = opening_runs_3[-1][0] if opening_runs_3 else length - 1
            return [("press", int(press))]

        if self.task_name == "MicrowaveThawing":
            open_door_pick = first_pick()
            if int(trajectory_id) == 10 and len(closing_runs) >= 3:
                pnp_close = closing_runs[2]
            elif len(closing_runs) >= 2:
                pnp_close = closing_runs[1]
            else:
                pnp_close = closing_runs[-1] if closing_runs else (length // 3, length // 3)
            object_pick = self._first_stuck_gap(gripper_gap, *pnp_close)
            object_place = first_open_after(object_pick)
            close_frame = int(object_place + np.argmin(eef_x_relative[object_place:]))
            press = length - 1
            return [
                ("open_door_first_pick", open_door_pick),
                ("pick_after_door", object_pick),
                ("place_after_door", object_place),
                ("close", close_frame),
                ("press", press),
            ]

        if self.task_name == "OpenSingleDoor":
            return [("door_pick", first_pick())]

        if self.task_name == "CloseSingleDoor":
            return [("close", int(np.argmin(eef_x_relative)))]

        if self.task_name == "PnPCounterToMicrowave":
            pick = first_pick()
            return [("pick", pick), ("place", first_open_after(pick))]

        if self.task_name == "TurnOnMicrowave":
            return [("press", int(np.argmax(eef_x_relative)))]

        # Backward-compatible fallback for old PnP tasks.
        pick, place_open = self._detect_pick_place(gripper_action, gripper_gap)
        return [("pick", pick), ("place", place_open + self.place_offset)]

    def _annotation_path(self, trajectory_id: int) -> Path:
        if self.annotation_root is None:
            raise FileNotFoundError("annotation_root is empty")
        task = self._annotation_task_name(self.task_name)
        new_path = (
            self.annotation_root
            / task
            / self.annotation_version
            / self.annotation_split
            / f"{task}_episode_{trajectory_id:06d}.json"
        )
        if new_path.exists():
            return new_path

        validated_dir = self.annotation_root / task / self.annotation_split
        cameras = [
            self.annotation_camera,
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        seen = set()
        for camera in cameras:
            if camera in seen:
                continue
            seen.add(camera)
            path = validated_dir / f"{task}_episode_{trajectory_id:06d}_{camera}.json"
            if path.exists():
                return path
        return new_path

    def _load_annotation(self, trajectory_id: int) -> dict:
        path = self._annotation_path(trajectory_id)
        if not path.exists():
            raise FileNotFoundError(f"missing annotation for AWR: {path}")
        return json.loads(path.read_text())

    def _episode_timestamps(self, trajectory_id: int) -> np.ndarray:
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=self.get_episode_chunk(trajectory_id),
            episode_index=trajectory_id,
        )
        df = pd.read_parquet(parquet_path, columns=["timestamp"])
        return df["timestamp"].to_numpy(dtype=np.float64)

    @staticmethod
    def _annotation_segments(annotation: dict) -> tuple[list[tuple[float, float, bool]], float]:
        duration = float(annotation["duration_sec"])
        if "tasks_time" in annotation:
            critical_tasks = set(annotation["critical_tasks"])
            return [
                (
                    max(0.0, min(float(segment["start_sec"]), duration)),
                    max(0.0, min(float(segment["end_sec"]), duration)),
                    segment["task"] in critical_tasks,
                )
                for segment in annotation["tasks_time"]
            ], duration

        critical_intervals = []
        for idx, item in enumerate(annotation.get("critical_tasks") or []):
            start = max(0.0, min(float(item.get("start_sec", 0.0)), duration))
            end = max(start, min(float(item.get("end_sec", start)), duration))
            if end > start:
                critical_intervals.append((start, end, idx))
        critical_intervals.sort()

        segments = []
        cursor = 0.0
        for start, end, _idx in critical_intervals:
            if start > cursor:
                segments.append((cursor, start, False))
            segments.append((start, end, True))
            cursor = max(cursor, end)
        if cursor < duration:
            segments.append((cursor, duration, False))
        return segments, duration

    def _annotation_progress_delta(self, trajectory_id: int) -> np.ndarray:
        annotation = self._load_annotation(trajectory_id)
        timestamps = self._episode_timestamps(trajectory_id)
        if len(timestamps) == 0:
            raise ValueError(f"empty timestamps for {self.dataset_path} episode {trajectory_id}")

        segments, duration = self._annotation_segments(annotation)
        critical_length = sum(end - start for start, end, is_critical in segments if is_critical)
        noncritical_length = sum(end - start for start, end, is_critical in segments if not is_critical)
        critical_slope, noncritical_slope = self._slopes_for_r(
            critical_length, noncritical_length, self.annotation_r
        )

        if len(timestamps) == 1:
            frame_ends = np.asarray([duration], dtype=np.float64)
        else:
            frame_ends = np.empty_like(timestamps, dtype=np.float64)
            frame_ends[:-1] = timestamps[1:]
            frame_ends[-1] = duration
        frame_starts = np.clip(timestamps, 0.0, duration)
        frame_ends = np.clip(np.maximum(frame_ends, frame_starts), 0.0, duration)

        delta = np.zeros(len(timestamps), dtype=np.float64)
        for start, end, is_critical in segments:
            slope = critical_slope if is_critical else noncritical_slope
            overlap = np.maximum(0.0, np.minimum(frame_ends, end) - np.maximum(frame_starts, start))
            delta += slope * overlap

        total = float(delta.sum())
        if total <= 0.0:
            raise ValueError(f"zero annotation progress delta for {self.dataset_path} episode {trajectory_id}")
        return (delta / total).astype(np.float32)

    def _progress_delta(self, length: int, events: list[tuple[str, int]]) -> np.ndarray:
        event_delta = np.zeros(length, dtype=np.float32)
        if not events:
            return event_delta
        event_mass = self.critical_mass / len(events)
        for _name, center in events:
            frames = np.arange(center - self.critical_radius, center + self.critical_radius + 1)
            frames = frames[(0 <= frames) & (frames < length)]
            if len(frames) > 0:
                event_delta[frames] += event_mass / len(frames)
        return event_delta

    def _precompute_loss_weights(self) -> dict[tuple[int, int], float]:
        weights = {}
        raw_means = []
        final_weights = []
        annotation_count = 0
        heuristic_count = 0
        for trajectory_id, length in zip(self.trajectory_ids, self.trajectory_lengths):
            gripper_action, gripper_gap, eef_x_relative = self._episode_arrays(int(trajectory_id))
            if self.awr_source == "annotation":
                progress_delta = self._annotation_progress_delta(int(trajectory_id))
                annotation_count += 1
            else:
                events = self._detect_task_events(int(trajectory_id), gripper_action, gripper_gap, eef_x_relative)
                progress_delta = self._progress_delta(int(length), events)
                heuristic_count += 1
            for base_index in range(int(length)):
                idx = np.minimum(
                    np.arange(base_index, base_index + self.action_chunk_delta_count),
                    int(length) - 1,
                )
                mean_delta = float(progress_delta[idx].mean())
                weight = float(np.clip(np.exp(self.task_alpha * mean_delta), 1.0, self.awr_clip_max))
                weights[(int(trajectory_id), base_index)] = weight
                raw_means.append(mean_delta)
                final_weights.append(weight)
        print(
            f"[AWR] {self.dataset_name}: mean_delta min/max={min(raw_means):.6f}/{max(raw_means):.6f}, "
            f"weight mean/min/max={np.mean(final_weights):.6f}/{min(final_weights):.6f}/{max(final_weights):.6f}, "
            f"alpha={self.task_alpha}, clip_max={self.awr_clip_max}, task={self.task_name}, "
            f"critical_mass={self.critical_mass}, critical_radius={self.critical_radius}, "
            f"annotation_count={annotation_count}, heuristic_count={heuristic_count}"
        )
        return weights

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        data["loss_weight"] = np.asarray(
            self._loss_weights[(int(trajectory_id), int(base_index))], dtype=np.float32
        )
        return data


def build_dataset(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    def make_one(path: str):
        return AWRLeRobotSingleDataset(
            dataset_path=path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            awr_alpha=config.awr_alpha,
            awr_clip_max=config.awr_clip_max,
            task_alpha_map=config.task_alpha_map,
            critical_mass=config.critical_mass,
            critical_radius=config.critical_radius,
            pick_mass=config.pick_mass,
            place_mass=config.place_mass,
            other_mass=config.other_mass,
            gripper_stuck_window=config.gripper_stuck_window,
            gripper_stuck_threshold=config.gripper_stuck_threshold,
            action_chunk_delta_count=config.action_chunk_delta_count,
            place_offset=config.place_offset,
            pick_radius=config.pick_radius,
            place_radius=config.place_radius,
            awr_source=config.awr_source,
            annotation_root=config.annotation_root,
            annotation_version=config.annotation_version,
            annotation_split=config.annotation_split,
            annotation_camera=config.annotation_camera,
            annotation_r=config.annotation_r,
        )

    if len(config.dataset_path) == 1:
        return make_one(config.dataset_path[0]), data_config_cls, transforms

    single_datasets = [make_one(path) for path in config.dataset_path]
    dataset = LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in single_datasets],
        mode="train",
        balance_dataset_weights=config.balance_dataset_weights,
        balance_trajectory_weights=config.balance_trajectory_weights,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    print(f"Loaded {len(single_datasets)} AWR datasets: {config.dataset_path}")
    return dataset, data_config_cls, transforms


def main(config: ArgsConfig):
    train_dataset, data_config_cls, transforms = build_dataset(config)
    data_action_horizon = len(data_config_cls.action_indices)
    assert hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    last_transform = transforms.transforms[-1]
    assert isinstance(last_transform, GR00TTransform)
    data_max_action_dim = last_transform.max_action_dim

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim
    if action_horizon_mismatch or action_dim_mismatch:
        old_action_dim = model.action_head.config.action_dim
        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim
        from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

        new_action_head = FlowmatchingActionHead(new_action_head_config)
        if not action_dim_mismatch:
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            new_action_head.state_dict().update(
                _copy_partial_action_expert_weights(
                    model.action_head.state_dict(),
                    new_action_head.state_dict(),
                    old_action_dim,
                    data_max_action_dim,
                )
            )
        model.action_head = new_action_head
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=Path(config.output_dir).name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to=None if config.report_to == "none" else config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    compact_base_model_path = config.compact_base_model_path or config.base_model_path
    compact_link_base_model_path = config.compact_link_base_model_path
    if config.compact_checkpoints_on_save:
        experiment.trainer.add_callback(
            CompactModelWeightsCallback(
                base_model_path=compact_base_model_path,
                link_base_model_path=compact_link_base_model_path,
                fail_fast=config.compact_fail_fast,
            )
        )
    experiment.train()
    if config.compact_checkpoints_on_save and config.compact_final_model:
        compact_model_weights_in_place(
            Path(config.output_dir),
            Path(compact_base_model_path),
            link_base_model_path=(
                Path(compact_link_base_model_path) if compact_link_base_model_path else None
            ),
            fail_fast=config.compact_fail_fast,
        )


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    print("\n" + "=" * 50)
    print("ROBOCASA AWR FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus
    assert config.num_gpus > 0
    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    elif os.environ.get("IS_TORCHRUN", "0") == "1":
        main(config)
    else:
        script_path = Path(__file__).absolute()
        raw_args_list = sys.argv[1:]
        env = os.environ.copy()
        env["IS_TORCHRUN"] = "1"
        if "CUDA_VISIBLE_DEVICES" in env:
            del env["CUDA_VISIBLE_DEVICES"]
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={config.num_gpus}",
            "--nnodes=1",
            str(script_path),
            *raw_args_list,
        ]
        print("Running torchrun command:", cmd)
        sys.exit(subprocess.run(cmd, env=env).returncode)
