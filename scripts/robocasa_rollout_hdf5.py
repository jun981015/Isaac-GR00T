"""RoboCasa-like HDF5 rollout writer for policy eval trajectories.

The writer intentionally keeps camera images out of HDF5 by default. Videos stay
as mp4 files and are referenced from demo attrs. This keeps contact / state
analysis lightweight while preserving a RoboCasa-compatible structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


OBS_KEYS = [
    "object",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
]

IMAGE_KEYS = [
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
]


def json_dumps(value: Any) -> str:
    def default(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return str(obj)

    return json.dumps(value, default=default, ensure_ascii=False, indent=2)


def sim_state(env) -> np.ndarray:
    state = env.sim.get_state()
    if hasattr(state, "flatten"):
        return np.asarray(state.flatten(), dtype=np.float64)
    return np.asarray(state, dtype=np.float64).reshape(-1)


def model_xml(env) -> str:
    model = getattr(getattr(env, "sim", None), "model", None)
    if model is None:
        return ""
    get_xml = getattr(model, "get_xml", None)
    if callable(get_xml):
        try:
            return str(get_xml())
        except Exception:
            pass
    return ""


def geom_name(env, geom_id: int) -> str:
    try:
        name = env.sim.model.geom_id2name(int(geom_id))
    except Exception:
        name = None
    return "" if name is None else str(name)


def object_names_from_ep_meta(ep_meta: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for cfg in ep_meta.get("object_cfgs", []) or []:
        name = cfg.get("name") if isinstance(cfg, dict) else None
        if name:
            names.append(str(name))
    return names


def is_gripper_geom(name: str) -> bool:
    lower = name.lower()
    return "gripper" in lower or "finger" in lower or "right_hand" in lower


def is_robot_geom(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("robot0") or "gripper" in lower or "finger" in lower


def is_object_geom(name: str, object_names: list[str]) -> bool:
    lower = name.lower()
    if lower.startswith("obj") or "_obj" in lower or "object" in lower:
        return True
    return any(obj.lower() in lower for obj in object_names)


def contact_summary(env, ep_meta: dict[str, Any]) -> dict[str, Any]:
    object_names = object_names_from_ep_meta(ep_meta)
    ncon = int(getattr(env.sim.data, "ncon", 0))
    pairs = []
    robot_obj = False
    obj_fixture = False
    gripper_obj = False
    gripper_fixture = False

    for idx in range(ncon):
        c = env.sim.data.contact[idx]
        g1 = geom_name(env, c.geom1)
        g2 = geom_name(env, c.geom2)
        dist = float(getattr(c, "dist", 0.0))
        g1_robot, g2_robot = is_robot_geom(g1), is_robot_geom(g2)
        g1_grip, g2_grip = is_gripper_geom(g1), is_gripper_geom(g2)
        g1_obj, g2_obj = is_object_geom(g1, object_names), is_object_geom(g2, object_names)
        g1_fixture = not g1_robot and not g1_obj
        g2_fixture = not g2_robot and not g2_obj

        robot_obj = robot_obj or ((g1_robot and g2_obj) or (g2_robot and g1_obj))
        obj_fixture = obj_fixture or ((g1_obj and g2_fixture) or (g2_obj and g1_fixture))
        gripper_obj = gripper_obj or ((g1_grip and g2_obj) or (g2_grip and g1_obj))
        gripper_fixture = gripper_fixture or ((g1_grip and g2_fixture) or (g2_grip and g1_fixture))
        pairs.append({"geom1": g1, "geom2": g2, "dist": dist})

    return {
        "ncon": ncon,
        "pairs": pairs,
        "robot_obj_contact": robot_obj,
        "obj_fixture_contact": obj_fixture,
        "gripper_obj_contact": gripper_obj,
        "gripper_fixture_contact": gripper_fixture,
    }


class RolloutRecorder:
    def __init__(self, env, ep_meta: dict[str, Any], include_images: bool = False):
        self.ep_meta = ep_meta
        self.include_images = include_images
        self.model_file = model_xml(env)
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.states: list[np.ndarray] = []
        self.obs: dict[str, list[np.ndarray]] = {}
        self.contacts: list[dict[str, Any]] = []

    def record_step(
        self,
        env,
        obs: dict[str, Any],
        action: np.ndarray,
        reward: float,
        done: bool,
        state: np.ndarray | None = None,
    ) -> None:
        keys = list(OBS_KEYS)
        if self.include_images:
            keys.extend(IMAGE_KEYS)
        for key in keys:
            if key in obs:
                self.obs.setdefault(key, []).append(np.asarray(obs[key]))
        self.states.append(sim_state(env) if state is None else np.asarray(state, dtype=np.float64))
        self.actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.contacts.append(contact_summary(env, self.ep_meta))

    @property
    def num_samples(self) -> int:
        return len(self.actions)

    def write(
        self,
        path: str | Path,
        env_name: str,
        episode_idx: int,
        seed: int,
        success: bool,
        video_path: str | None,
        policy_checkpoint: str | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actions = np.asarray(self.actions, dtype=np.float64)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.int64)
        states = np.asarray(self.states, dtype=np.float64)

        with h5py.File(path, "w") as f:
            data = f.create_group("data")
            data.attrs["env_args"] = json_dumps(
                {
                    "env_name": env_name,
                    "env_version": "rollout_eval",
                    "type": 1,
                    "env_kwargs": {
                        "control_freq": 20,
                        "robots": "PandaOmron",
                        "has_offscreen_renderer": True,
                        "use_camera_obs": True,
                    },
                }
            )
            data.attrs["total"] = int(self.num_samples)

            demo = data.create_group(f"demo_{episode_idx:03d}")
            demo.attrs["ep_meta"] = json_dumps(self.ep_meta)
            demo.attrs["model_file"] = self.model_file
            demo.attrs["num_samples"] = int(self.num_samples)
            demo.attrs["policy_checkpoint"] = "" if policy_checkpoint is None else str(policy_checkpoint)
            demo.attrs["success"] = int(bool(success))
            demo.attrs["seed"] = int(seed)
            demo.attrs["video_path"] = "" if video_path is None else str(video_path)

            demo.create_dataset("actions", data=actions)
            demo.create_dataset("actions_abs", data=actions)
            demo.create_dataset("rewards", data=rewards)
            demo.create_dataset("dones", data=dones)
            demo.create_dataset("states", data=states)

            action_dict = demo.create_group("action_dict")
            action_dict.create_dataset("rel_pos", data=actions[:, 0:3].astype(np.float32))
            action_dict.create_dataset("rel_rot_axis_angle", data=actions[:, 3:6].astype(np.float32))
            action_dict.create_dataset("gripper", data=actions[:, 6:7].astype(np.float32))
            action_dict.create_dataset("abs_pos", data=self._obs_array("robot0_eef_pos", width=3, dtype=np.float32))
            action_dict.create_dataset("abs_rot_axis_angle", data=actions[:, 3:6].astype(np.float32))
            action_dict.create_dataset("rel_rot_6d", data=np.zeros((self.num_samples, 6), dtype=np.float32))
            action_dict.create_dataset("abs_rot_6d", data=np.zeros((self.num_samples, 6), dtype=np.float32))

            obs_group = demo.create_group("obs")
            for key, values in sorted(self.obs.items()):
                obs_group.create_dataset(key, data=np.asarray(values))

            contacts = demo.create_group("contacts")
            contacts.create_dataset("ncon", data=np.asarray([c["ncon"] for c in self.contacts], dtype=np.int32))
            for key in [
                "robot_obj_contact",
                "obj_fixture_contact",
                "gripper_obj_contact",
                "gripper_fixture_contact",
            ]:
                contacts.create_dataset(key, data=np.asarray([c[key] for c in self.contacts], dtype=np.bool_))
            string_dtype = h5py.string_dtype(encoding="utf-8")
            contacts.create_dataset(
                "pairs_json",
                data=np.asarray([json_dumps(c["pairs"]) for c in self.contacts], dtype=object),
                dtype=string_dtype,
            )

            mask = f.create_group("mask")
            demo_name = np.asarray([f"demo_{episode_idx:03d}".encode("utf-8")])
            mask.create_dataset("train", data=demo_name)
            mask.create_dataset("valid", data=demo_name[:0])

    def _obs_array(self, key: str, width: int, dtype: np.dtype) -> np.ndarray:
        if key in self.obs:
            return np.asarray(self.obs[key], dtype=dtype)
        return np.zeros((self.num_samples, width), dtype=dtype)
