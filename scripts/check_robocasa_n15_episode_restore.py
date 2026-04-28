#!/usr/bin/env python3
"""Check that a saved RoboCasa episode metadata restores deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.eval.wrappers.robocasa_n15_wrapper import (
    RoboCasaEvalConfig,
    create_robocasa_env,
    render_composite_from_obs,
    render_highres_composite,
    reseed_env,
    set_ep_meta,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def digest_array(value: Any) -> str:
    array = np.asarray(value)
    h = hashlib.sha256()
    h.update(str(array.shape).encode("utf-8"))
    h.update(str(array.dtype).encode("utf-8"))
    h.update(array.tobytes())
    return h.hexdigest()


def obs_digest(obs: dict[str, Any]) -> dict[str, str]:
    keys = [
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_joint_pos",
    ]
    return {key: digest_array(obs[key]) for key in keys if key in obs}


def restore_once(metadata: dict[str, Any], render_size: int) -> tuple[dict[str, str], np.ndarray]:
    env_name = metadata["env_name"]
    seed = int(metadata["seed"])
    ep_meta = metadata["ep_meta"]
    config = RoboCasaEvalConfig(
        env_name=env_name,
        seed=seed,
        obj_instance_split="A",
        layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        camera_width=128,
        camera_height=128,
    )
    env = create_robocasa_env(config)
    try:
        reseed_env(env, seed)
        set_ep_meta(env, ep_meta)
        obs = env.reset()
        frame = (
            render_highres_composite(env, height=render_size, width=render_size)
            if render_size > 0
            else render_composite_from_obs(obs)
        )
        return obs_digest(obs), frame
    finally:
        env.close()


def write_ppm(path: Path, frame: np.ndarray) -> None:
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 frame, got {frame.shape}")
    header = f"P6\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + frame.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--render_size", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    episode_json = Path(args.episode_json)
    output_dir = Path(args.output_dir)
    metadata = read_json(episode_json)

    digest_a, frame_a = restore_once(metadata, args.render_size)
    digest_b, frame_b = restore_once(metadata, args.render_size)
    match = digest_a == digest_b

    output_dir.mkdir(parents=True, exist_ok=True)
    write_ppm(output_dir / f"{args.label}_restore_a.ppm", frame_a)
    write_ppm(output_dir / f"{args.label}_restore_b.ppm", frame_b)
    write_json(
        output_dir / f"{args.label}_restore_check.json",
        {
            "label": args.label,
            "episode_json": str(episode_json),
            "env_name": metadata["env_name"],
            "episode_idx": metadata["episode_idx"],
            "seed": metadata["seed"],
            "match": match,
            "digest_a": digest_a,
            "digest_b": digest_b,
            "scene_signature": metadata.get("scene_signature"),
            "lang": metadata.get("ep_meta", {}).get("lang", ""),
        },
    )
    print(f"[restore] {args.label} ep={metadata['episode_idx']:03d} match={int(match)}")


if __name__ == "__main__":
    main()
