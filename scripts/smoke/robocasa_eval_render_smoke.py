#!/usr/bin/env python3
"""Isolated RoboCasa offscreen render smoke for conda eval debugging."""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="PnPCounterToSink")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    print("env CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("env MUJOCO_EGL_DEVICE_ID=", os.environ.get("MUJOCO_EGL_DEVICE_ID"))
    print("env MUJOCO_GL=", os.environ.get("MUJOCO_GL"))
    print("env PYOPENGL_PLATFORM=", os.environ.get("PYOPENGL_PLATFORM"))

    import robocasa  # noqa: F401
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    from gr00t.eval.wrappers.robocasa_n15_wrapper import _patch_robocasa_readonly_mjcf_objects

    _patch_robocasa_readonly_mjcf_objects()
    controller_config = load_composite_controller_config(controller=None, robot="PandaOmron")

    t0 = time.time()
    env = robosuite.make(
        env_name=args.env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_widths=args.camera_width,
        camera_heights=args.camera_height,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_gpu_device_id=args.render_gpu_device_id,
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=args.seed,
        obj_instance_split="A",
        randomize_cameras=False,
        layout_and_style_ids=((1, 1),),
        translucent_robot=False,
        reward_shaping=False,
    )
    print(f"make_sec={time.time() - t0:.3f}")

    t1 = time.time()
    obs = env.reset()
    print(f"reset_sec={time.time() - t1:.3f}")
    for key in (
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
    ):
        print(key, getattr(obs.get(key), "shape", None), getattr(obs.get(key), "dtype", None))

    t2 = time.time()
    img = env.sim.render(
        camera_name="robot0_agentview_left",
        width=args.render_width,
        height=args.render_height,
        depth=False,
    )
    print(f"render_sec={time.time() - t2:.3f}", "render_shape=", img.shape, "dtype=", img.dtype)
    env.close()
    print("OK")


if __name__ == "__main__":
    main()
