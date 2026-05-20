#!/usr/bin/env python3
"""GR00T fine-tuning with RoboCasa outcome-conditioned prompts.

This keeps the original LeRobot datasets untouched. At sample time it rewrites
the task language as:

    success: <task prompt>
    failure: <task prompt>

For classifier-free guidance style training over the outcome condition, it can
randomly drop only that outcome prefix, leaving the task prompt intact.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME, LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for outcome-conditioned RoboCasa GR00T fine-tuning."""

    dataset_path: List[str]
    """LeRobot dataset directory or directories."""

    output_dir: str = "/tmp/gr00t_cfg"
    data_config: str = "robocasa_n15_data_config:RobocasaKitchenPnPDataConfig"

    batch_size: int = 32
    max_steps: int = 10000
    num_gpus: int = 1
    save_steps: int = 1000

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
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
    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True

    # Outcome-conditioning parameters.
    outcome_prefix: bool = True
    """If True, add success:/failure: prefixes to language prompts."""

    outcome_prompt_style: Literal["prefix", "failure_tag"] = "prefix"
    """How to encode episode outcome in the prompt."""

    outcome_prefix_dropout_prob: float = 0.1
    """Probability of dropping only the success/failure prefix, not the task prompt."""

    expert_outcome: Literal["success", "none"] = "success"
    """Outcome assigned to datasets whose episodes.jsonl has no success field."""


class CFGLeRobotSingleDataset(LeRobotSingleDataset):
    """LeRobot dataset that dynamically adds outcome prefixes to language."""

    def __init__(
        self,
        *args,
        outcome_prefix: bool = True,
        outcome_prompt_style: str = "prefix",
        outcome_prefix_dropout_prob: float = 0.1,
        expert_outcome: str = "success",
        **kwargs,
    ):
        self.outcome_prefix = outcome_prefix
        self.outcome_prompt_style = outcome_prompt_style
        self.outcome_prefix_dropout_prob = outcome_prefix_dropout_prob
        self.expert_outcome = expert_outcome
        self._episode_success: dict[int, bool | None] = {}
        self._has_success_metadata = False
        super().__init__(*args, **kwargs)
        self._load_episode_success_metadata()

    def _load_episode_success_metadata(self) -> None:
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with episode_path.open("r", encoding="utf-8") as f:
            for line in f:
                episode = json.loads(line)
                ep_idx = int(episode["episode_index"])
                if "success" in episode:
                    self._has_success_metadata = True
                    success = episode["success"]
                    if success is None:
                        self._episode_success[ep_idx] = None
                    else:
                        self._episode_success[ep_idx] = bool(success)
                else:
                    self._episode_success[ep_idx] = None
        if self._has_success_metadata:
            success_count = sum(v is True for v in self._episode_success.values())
            failure_count = sum(v is False for v in self._episode_success.values())
            none_count = sum(v is None for v in self._episode_success.values())
            print(
                f"[CFG] {self.dataset_name}: success={success_count} "
                f"failure={failure_count} none={none_count}"
            )
        else:
            print(f"[CFG] {self.dataset_name}: no success metadata, expert_outcome={self.expert_outcome}")

    def _outcome_for_trajectory(self, trajectory_id: int) -> str | None:
        success = self._episode_success.get(int(trajectory_id))
        if success is True:
            return "success"
        if success is False:
            return "failure"
        if self.expert_outcome == "success":
            return "success"
        return None

    def _maybe_prefix_prompt(self, trajectory_id: int, prompt: str) -> str:
        if not self.outcome_prefix:
            return prompt
        outcome = self._outcome_for_trajectory(trajectory_id)
        if outcome is None:
            return prompt
        if self.outcome_prefix_dropout_prob > 0.0:
            if random.random() < self.outcome_prefix_dropout_prob:
                return prompt
        if self.outcome_prompt_style == "failure_tag":
            failure_state = "detected" if outcome == "failure" else "none"
            return f"{prompt}\n<failure> {failure_state} </failure>"
        return f"{outcome}: {prompt}"

    def get_language(self, trajectory_id: int, key: str, base_index: int) -> list[str]:
        prompts = super().get_language(trajectory_id, key, base_index)
        return [self._maybe_prefix_prompt(trajectory_id, prompt) for prompt in prompts]


def _copy_partial_action_expert_weights(old_dict, new_dict, old_dim, new_dim):
    total_params = copied_params = random_params = 0
    for key, old_tensor in old_dict.items():
        if key not in new_dict:
            continue
        new_tensor = new_dict[key]
        total_params += new_tensor.numel()
        if old_tensor.shape == new_tensor.shape:
            new_tensor.copy_(old_tensor)
            copied_params += new_tensor.numel()
        elif "action_encoder" in key and "W1.weight" in key:
            new_tensor[:, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        elif "action_decoder" in key and ("weight" in key or "bias" in key):
            if old_tensor.dim() == 1:
                new_tensor[:old_dim] = old_tensor
            elif old_tensor.dim() == 2:
                new_tensor[:, :old_dim] = old_tensor
            elif old_tensor.dim() == 3:
                new_tensor[:, :, :old_dim] = old_tensor
            copied_params += old_tensor.numel()
            random_params += new_tensor.numel() - old_tensor.numel()
        else:
            random_params += new_tensor.numel()
    assert total_params == copied_params + random_params, "Parameter count mismatch"
    random_percentage = (random_params / total_params) * 100 if total_params > 0 else 0
    print(
        f"Weight copy stats: {copied_params:,} copied, {random_params:,} random "
        f"({random_percentage:.1f}% randomly initialized)"
    )
    print(f"Action dimensions {old_dim + 1}-{new_dim} will be learned from scratch")
    return new_dict


def build_dataset(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    def make_one(path: str):
        assert os.path.exists(path), f"Dataset path {path} does not exist"
        return CFGLeRobotSingleDataset(
            dataset_path=path,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            outcome_prefix=config.outcome_prefix,
            outcome_prompt_style=config.outcome_prompt_style,
            outcome_prefix_dropout_prob=config.outcome_prefix_dropout_prob,
            expert_outcome=config.expert_outcome,
        )

    def align_video_metadata_to_reference(single_datasets: list[CFGLeRobotSingleDataset]) -> None:
        """Make mixed 128/512 video datasets mergeable.

        GR00T transforms handle the actual image resize. LeRobotMixtureDataset
        merges metadata before transform metadata is applied, so all video
        modality configs must be identical at merge time.
        """
        if not single_datasets:
            return
        reference_video = single_datasets[0].metadata.modalities.video
        for dataset in single_datasets[1:]:
            for key, video_cfg in dataset.metadata.modalities.video.items():
                if key not in reference_video:
                    continue
                ref_cfg = reference_video[key]
                video_cfg.resolution = ref_cfg.resolution
                video_cfg.channels = ref_cfg.channels
                video_cfg.fps = ref_cfg.fps

    def allow_mixed_video_input_resolutions(transform) -> None:
        """Let 128x128 expert demos and 512x512 eval rollouts share one transform."""
        for sub_transform in getattr(transform, "transforms", []):
            if sub_transform.__class__.__name__ == "VideoToTensor":
                sub_transform.original_resolutions = {}

    if len(config.dataset_path) == 1:
        return make_one(config.dataset_path[0]), data_config_cls, transforms

    single_datasets = [make_one(path) for path in config.dataset_path]
    align_video_metadata_to_reference(single_datasets)
    train_dataset = LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in single_datasets],
        mode="train",
        balance_dataset_weights=config.balance_dataset_weights,
        balance_trajectory_weights=config.balance_trajectory_weights,
        seed=42,
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
    allow_mixed_video_input_resolutions(transforms)
    print(f"Loaded {len(single_datasets)} CFG datasets: {config.dataset_path}")
    return train_dataset, data_config_cls, transforms


def main(config: ArgsConfig):
    train_dataset, data_config_cls, transforms = build_dataset(config)

    data_action_horizon = len(data_config_cls.action_indices)
    assert hasattr(transforms, "transforms") and len(transforms.transforms) > 0, "No transforms found"
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
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
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} "
            f"(was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")
        import copy

        from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim
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
        run_name=None,
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
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    print(f"TrainingArguments.report_to: {training_args.report_to}", flush=True)

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
    print("GR00T ROBOCASA CFG FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                    else:
                        cmd.append(f"--no-{key}")
                elif isinstance(value, list):
                    for item in value:
                        cmd.extend([f"--{key}", str(item)])
                else:
                    cmd.extend([f"--{key}", str(value)])
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
