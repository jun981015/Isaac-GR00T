"""RoboCasa helpers for GR00T N1.5 fine-tuned kitchen PnP policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATE_KEY_MAP = {
    "robot0_gripper_qpos": "state.gripper_qpos",
    "robot0_base_pos": "state.base_position",
    "robot0_base_quat": "state.base_rotation",
    "robot0_base_to_eef_pos": "state.end_effector_position_relative",
    "robot0_base_to_eef_quat": "state.end_effector_rotation_relative",
    "robot0_gripper_qvel": "state.gripper_qvel",
    "robot0_eef_pos": "state.end_effector_position_absolute",
    "robot0_eef_quat": "state.end_effector_rotation_absolute",
    "robot0_joint_pos": "state.joint_position",
    "robot0_joint_pos_cos": "state.joint_position_cos",
    "robot0_joint_pos_sin": "state.joint_position_sin",
    "robot0_joint_vel": "state.joint_velocity",
}

VIDEO_KEY_MAP = {
    "robot0_agentview_left_image": "video.robot0_agentview_left",
    "robot0_agentview_right_image": "video.robot0_agentview_right",
    "robot0_eye_in_hand_image": "video.robot0_eye_in_hand",
}

ACTION_KEYS = [
    "action.eef_position",
    "action.eef_rotation",
    "action.gripper",
    "action.base",
    "action.control_mode",
]


@dataclass(frozen=True)
class RoboCasaEvalConfig:
    env_name: str
    seed: int | None = None
    obj_instance_split: str | None = "A"
    layout_and_style_ids: tuple[tuple[int, int], ...] = (
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 9),
        (7, 10),
    )
    camera_width: int = 128
    camera_height: int = 128
    randomize_cameras: bool = False
    generative_textures: str | None = None
    reward_shaping: bool = False


def create_robocasa_env(config: RoboCasaEvalConfig):
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(
        controller=None,
        robot="PandaOmron",
    )
    return robosuite.make(
        env_name=config.env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=config.camera_width,
        camera_heights=config.camera_height,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=config.seed,
        obj_instance_split=config.obj_instance_split,
        generative_textures=config.generative_textures,
        randomize_cameras=config.randomize_cameras,
        layout_and_style_ids=config.layout_and_style_ids,
        translucent_robot=False,
        reward_shaping=config.reward_shaping,
    )


def _language(env) -> str:
    try:
        return str(env.get_ep_meta().get("lang", ""))
    except Exception:
        return ""


def resize_image(image: np.ndarray, size: int | None) -> np.ndarray:
    if size is None or size <= 0 or image.shape[0] == size and image.shape[1] == size:
        return image
    try:
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        y_idx = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
        x_idx = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
        return image[y_idx][:, x_idx]


def obs_to_policy(obs: dict[str, Any], env, image_size: int | None = None) -> dict[str, Any]:
    policy_obs: dict[str, Any] = {}
    for robocasa_key, gr00t_key in STATE_KEY_MAP.items():
        policy_obs[gr00t_key] = np.asarray(obs[robocasa_key])[None]
    for robocasa_key, gr00t_key in VIDEO_KEY_MAP.items():
        image = np.asarray(obs[robocasa_key])
        image = resize_image(image, image_size)
        policy_obs[gr00t_key] = np.flip(image, axis=0)[None]
    policy_obs["annotation.human.action.task_description"] = np.asarray([_language(env)])
    return policy_obs


def action_dict_to_robosuite(action: dict[str, Any], action_idx: int) -> np.ndarray:
    elems = []
    for key in ACTION_KEYS:
        value = np.asarray(action[key])
        if value.ndim == 3:
            value = value[0]
        if value.ndim == 1:
            value = value[None]
        step_value = value[min(action_idx, value.shape[0] - 1)]
        if key in {"action.gripper", "action.control_mode"}:
            step_value = np.where(step_value > 0, 1, -1)
        elems.append(step_value)
    return np.concatenate(elems, axis=-1)


def success_from_env(env) -> bool:
    succ = env._check_success()
    if isinstance(succ, dict):
        return bool(succ.get("task", False))
    return bool(succ)


def get_ep_meta(env) -> dict[str, Any]:
    return env.get_ep_meta()


def set_ep_meta(env, ep_meta: dict[str, Any]) -> None:
    env.set_ep_meta(ep_meta)


def reseed_env(env, seed: int) -> None:
    env.seed = seed
    env.rng = np.random.default_rng(seed)


def render_composite_from_obs(obs: dict[str, Any]) -> np.ndarray:
    frames = []
    for key in (
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
    ):
        frames.append(np.flip(np.asarray(obs[key]), axis=0))
    return np.concatenate(frames, axis=1)


def render_highres_composite(env, height: int = 512, width: int = 512) -> np.ndarray:
    frames = []
    for camera_name in (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ):
        frame = env.sim.render(camera_name=camera_name, height=height, width=width)[::-1]
        frames.append(np.asarray(frame, dtype=np.uint8))
    return np.concatenate(frames, axis=1)
