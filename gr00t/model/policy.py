# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        outcome_conditioning: Literal["none", "success", "guidance"] = "none",
        outcome_prompt_style: Literal["prefix", "failure_tag"] = "prefix",
        cfg_guidance_scale: float = 1.0,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device
        self.outcome_conditioning = outcome_conditioning
        self.outcome_prompt_style = outcome_prompt_style
        self.cfg_guidance_scale = float(cfg_guidance_scale)

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        if self.outcome_conditioning == "success":
            obs_copy = self._rewrite_language(obs_copy, "success")

        if self.outcome_conditioning == "guidance":
            normalized_action = self._get_cfg_guided_action(obs_copy)
        else:
            normalized_input = self.apply_transforms(obs_copy)
            normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    @staticmethod
    def _language_key() -> str:
        return "annotation.human.action.task_description"

    def _format_outcome_prompt(self, prompt: str, outcome: str) -> str:
        if self.outcome_prompt_style == "failure_tag":
            failure_state = "detected" if outcome == "failure" else "none"
            return f"{prompt}\n<failure> {failure_state} </failure>"
        return f"{outcome}: {prompt}"

    def _rewrite_language(self, obs: Dict[str, Any], outcome: str | None) -> Dict[str, Any]:
        if outcome is None:
            return obs
        key = self._language_key()
        if key not in obs:
            return obs

        obs = obs.copy()
        language = np.asarray(obs[key])
        flat = language.reshape(-1)
        rewritten = [self._format_outcome_prompt(str(prompt), outcome) for prompt in flat]
        obs[key] = np.asarray(rewritten, dtype=object).reshape(language.shape)
        return obs

    def _make_cfg_pair_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Interleave task-only and success-conditioned observations: [u0, c0, u1, c1, ...]."""
        key = self._language_key()
        paired: Dict[str, Any] = {}
        for obs_key, value in obs.items():
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)

            if obs_key == key:
                prompts = value.reshape(value.shape[0], -1)[:, 0]
                values = []
                for prompt in prompts:
                    prompt = str(prompt)
                    values.append(prompt)
                    values.append(self._format_outcome_prompt(prompt, "success"))
                if value.ndim == 1:
                    paired[obs_key] = np.asarray(values, dtype=object)
                else:
                    paired[obs_key] = np.asarray(values, dtype=object).reshape(
                        value.shape[0] * 2, *value.shape[1:]
                    )
            else:
                paired[obs_key] = np.repeat(value, repeats=2, axis=0)
        return paired

    def _get_cfg_guided_action(self, obs_copy: Dict[str, Any]) -> torch.Tensor:
        paired_obs = self._make_cfg_pair_observations(obs_copy)
        normalized_input = self.apply_transforms(paired_obs)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            normalized_action = self._get_cfg_guided_action_from_normalized_input(
                normalized_input, guidance_scale=self.cfg_guidance_scale
            )
        return normalized_action.float()

    def _get_cfg_guided_action_from_normalized_input(
        self, normalized_input: Dict[str, Any], guidance_scale: float
    ) -> torch.Tensor:
        """Run flow-matching CFG by mixing task-only/success velocities at each denoising step."""
        backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
        backbone_output = self.model.backbone(backbone_inputs)
        action_head = self.model.action_head
        backbone_output = action_head.process_backbone_output(backbone_output)

        vl_embs = backbone_output.backbone_features
        pair_batch = vl_embs.shape[0]
        if pair_batch % 2 != 0:
            raise ValueError(f"CFG paired batch must be even, got {pair_batch}")
        batch_size = pair_batch // 2
        device = vl_embs.device
        embodiment_id = action_inputs.embodiment_id
        state_features = action_head.state_encoder(action_inputs.state, embodiment_id)
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        num_steps = action_head.num_inference_timesteps
        dt = 1.0 / num_steps
        scale = float(guidance_scale)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * action_head.num_timestep_buckets)
            paired_actions = torch.repeat_interleave(actions, repeats=2, dim=0)
            timesteps_tensor = torch.full(
                size=(pair_batch,), fill_value=t_discretized, device=device
            )
            action_features = action_head.action_encoder(
                paired_actions, timesteps_tensor, embodiment_id
            )
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
                pair_batch, -1, -1
            )
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            uncond_velocity = pred_velocity[0::2]
            cond_velocity = pred_velocity[1::2]
            guided_velocity = uncond_velocity + scale * (cond_velocity - uncond_velocity)
            actions = actions + dt * guided_velocity

        return actions

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
