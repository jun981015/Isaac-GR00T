"""RoboCasa helpers for GR00T N1.5 fine-tuned kitchen PnP policies."""

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import xml.etree.ElementTree as ET
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
    has_offscreen_renderer: bool = True
    use_camera_obs: bool = True
    randomize_cameras: bool = False
    generative_textures: str | None = None
    reward_shaping: bool = False


def create_robocasa_env(config: RoboCasaEvalConfig):
    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    _patch_robocasa_readonly_mjcf_objects()

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
        has_offscreen_renderer=config.has_offscreen_renderer,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=config.use_camera_obs,
        camera_depths=False,
        seed=config.seed,
        obj_instance_split=config.obj_instance_split,
        generative_textures=config.generative_textures,
        randomize_cameras=config.randomize_cameras,
        layout_and_style_ids=config.layout_and_style_ids,
        translucent_robot=False,
        reward_shaping=config.reward_shaping,
    )


def _patch_robocasa_readonly_mjcf_objects() -> None:
    """Allow RoboCasa object XML postprocessing when asset folders are read-only.

    RoboCasa writes a short-lived modified XML next to the source object XML.
    That fails when the RoboCasa checkout is mounted read-only. Writing the
    temporary XML under /tmp requires converting relative mesh/texture paths to
    absolute paths first.
    """

    try:
        import robocasa.environments.kitchen.kitchen as kitchen_mod
        import robocasa.models.objects.objects as objects_mod
        from robosuite.models.objects import MujocoXMLObject
    except Exception:
        return

    if getattr(objects_mod.MJCFObject, "_gr00t_readonly_patch", False):
        return

    def patched_init(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
        priority=None,
    ):
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, (tuple, list)):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception(f"got invalid scale: {scale}")
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin
        self.priority = priority
        self.rgba = rgba

        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        xml_str = self.postprocess_model_xml(xml_str)

        patched_root = ET.fromstring(xml_str)
        asset = patched_root.find("asset")
        if asset is not None:
            for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
                file_path = elem.get("file")
                if file_path and not os.path.isabs(file_path):
                    elem.set("file", os.path.abspath(os.path.join(folder, file_path)))
        xml_str = ET.tostring(patched_root, encoding="utf8").decode("utf8")

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        try:
            tmp.write(xml_str)
            tmp.close()
            MujocoXMLObject.__init__(
                self,
                fname=tmp.name,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=False,
                scale=scale,
            )
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass

    objects_mod.MJCFObject.__init__ = patched_init
    objects_mod.MJCFObject._gr00t_readonly_patch = True
    kitchen_mod.MJCFObject = objects_mod.MJCFObject


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


def success_diagnostics_from_env(env) -> dict[str, Any]:
    """Return task-specific success subconditions for debugging eval failures."""
    diagnostics: dict[str, Any] = {
        "env_name": getattr(env, "_env_name", env.__class__.__name__),
        "success": success_from_env(env),
    }
    try:
        from robocasa.utils import object_utils as OU
    except Exception as exc:  # pragma: no cover - only used in eval containers
        diagnostics["error"] = f"failed to import object_utils: {exc!r}"
        return diagnostics

    env_name = env.__class__.__name__
    try:
        if env_name == "PrepareCoffee":
            contact_check = env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")
            gripper_obj_far = OU.gripper_obj_far(env)
            turned_on = bool(env.coffee_machine._turned_on)
            gripper_button_far = env.coffee_machine.gripper_button_far(env)
            diagnostics.update(
                {
                    "contact_check": bool(contact_check),
                    "gripper_obj_far": bool(gripper_obj_far),
                    "coffee_machine_turned_on": turned_on,
                    "gripper_button_far": bool(gripper_button_far),
                    "missing_conditions": [
                        name
                        for name, value in {
                            "contact_check": contact_check,
                            "gripper_obj_far": gripper_obj_far,
                            "coffee_machine_turned_on": turned_on,
                            "gripper_button_far": gripper_button_far,
                        }.items()
                        if not bool(value)
                    ],
                }
            )
        elif env_name == "MicrowaveThawing":
            obj_in_microwave = OU.obj_inside_of(env, "obj", env.microwave)
            gripper_obj_far = OU.gripper_obj_far(env)
            button_pressed = bool(env.microwave.get_state()["turned_on"])
            diagnostics.update(
                {
                    "obj_in_microwave": bool(obj_in_microwave),
                    "gripper_obj_far": bool(gripper_obj_far),
                    "button_pressed": button_pressed,
                    "missing_conditions": [
                        name
                        for name, value in {
                            "obj_in_microwave": obj_in_microwave,
                            "gripper_obj_far": gripper_obj_far,
                            "button_pressed": button_pressed,
                        }.items()
                        if not bool(value)
                    ],
                }
            )
    except Exception as exc:  # pragma: no cover - only used in eval containers
        diagnostics["error"] = repr(exc)
    return diagnostics


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
