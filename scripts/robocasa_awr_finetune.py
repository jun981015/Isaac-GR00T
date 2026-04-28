# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
import tyro
from transformers import TrainingArguments

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
    awr_alpha: float = 20.0
    awr_clip_max: float = 2.0
    pick_mass: float = 0.3
    place_mass: float = 0.3
    other_mass: float = 0.4
    gripper_stuck_window: int = 10
    gripper_stuck_threshold: float = 0.001
    action_chunk_delta_count: int = 15
    place_offset: int = 8
    pick_radius: int = 2
    place_radius: int = 2


class AWRLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(
        self,
        *args,
        awr_alpha: float,
        awr_clip_max: float,
        pick_mass: float,
        place_mass: float,
        other_mass: float,
        gripper_stuck_window: int,
        gripper_stuck_threshold: float,
        action_chunk_delta_count: int,
        place_offset: int,
        pick_radius: int,
        place_radius: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.awr_alpha = awr_alpha
        self.awr_clip_max = awr_clip_max
        self.pick_mass = pick_mass
        self.place_mass = place_mass
        self.other_mass = other_mass
        self.gripper_stuck_window = gripper_stuck_window
        self.gripper_stuck_threshold = gripper_stuck_threshold
        self.action_chunk_delta_count = action_chunk_delta_count
        self.place_offset = place_offset
        self.pick_radius = pick_radius
        self.place_radius = place_radius
        self._loss_weights = self._precompute_loss_weights()

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

    def _episode_arrays(self, trajectory_id: int) -> tuple[np.ndarray, np.ndarray]:
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=self.get_episode_chunk(trajectory_id),
            episode_index=trajectory_id,
        )
        df = pd.read_parquet(parquet_path)
        modality = json.loads((self.dataset_path / "meta/modality.json").read_text())
        action_cfg = modality["action"]["gripper"]
        state_cfg = modality["state"]["gripper_qpos"]
        action = np.stack(df[action_cfg["original_key"]].to_numpy())
        state = np.stack(df[state_cfg["original_key"]].to_numpy())
        gripper_action = action[:, action_cfg["start"] : action_cfg["end"]].reshape(-1)
        gripper_qpos = state[:, state_cfg["start"] : state_cfg["end"]]
        gripper_gap = np.abs(gripper_qpos[:, 0] - gripper_qpos[:, 1])
        return gripper_action, gripper_gap

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

    def _progress_delta(self, length: int, pick_frame: int, place_open_frame: int) -> np.ndarray:
        event_delta = np.zeros(length, dtype=np.float32)
        pick_frames = np.arange(pick_frame - self.pick_radius, pick_frame + self.pick_radius + 1)
        place_center = place_open_frame + self.place_offset
        place_frames = np.arange(place_center - self.place_radius, place_center + self.place_radius + 1)
        pick_frames = pick_frames[(0 <= pick_frames) & (pick_frames < length)]
        place_frames = place_frames[(0 <= place_frames) & (place_frames < length)]
        if len(pick_frames) > 0:
            event_delta[pick_frames] += self.pick_mass / len(pick_frames)
        if len(place_frames) > 0:
            event_delta[place_frames] += self.place_mass / len(place_frames)
        non_event = event_delta == 0
        if non_event.any():
            event_delta[non_event] = self.other_mass / non_event.sum()
        return event_delta

    def _precompute_loss_weights(self) -> dict[tuple[int, int], float]:
        weights = {}
        raw_means = []
        final_weights = []
        for trajectory_id, length in zip(self.trajectory_ids, self.trajectory_lengths):
            gripper_action, gripper_gap = self._episode_arrays(int(trajectory_id))
            pick_frame, place_open_frame = self._detect_pick_place(gripper_action, gripper_gap)
            progress_delta = self._progress_delta(int(length), pick_frame, place_open_frame)
            for base_index in range(int(length)):
                idx = np.minimum(
                    np.arange(base_index, base_index + self.action_chunk_delta_count),
                    int(length) - 1,
                )
                mean_delta = float(progress_delta[idx].mean())
                weight = float(np.clip(np.exp(self.awr_alpha * mean_delta), 0.0, self.awr_clip_max))
                weights[(int(trajectory_id), base_index)] = weight
                raw_means.append(mean_delta)
                final_weights.append(weight)
        print(
            f"[AWR] {self.dataset_name}: mean_delta min/max={min(raw_means):.6f}/{max(raw_means):.6f}, "
            f"weight min/max={min(final_weights):.6f}/{max(final_weights):.6f}, alpha={self.awr_alpha}, "
            f"pick_radius={self.pick_radius}, place_radius={self.place_radius}"
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
            pick_mass=config.pick_mass,
            place_mass=config.place_mass,
            other_mass=config.other_mass,
            gripper_stuck_window=config.gripper_stuck_window,
            gripper_stuck_threshold=config.gripper_stuck_threshold,
            action_chunk_delta_count=config.action_chunk_delta_count,
            place_offset=config.place_offset,
            pick_radius=config.pick_radius,
            place_radius=config.place_radius,
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
    experiment.train()


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
