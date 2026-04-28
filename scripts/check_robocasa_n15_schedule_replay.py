#!/usr/bin/env python3
"""Check whether a RoboCasa N1.5 schedule replays identical initial scenes."""

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
    reseed_env,
    set_ep_meta,
)


def canonical(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "groups_containing_sampled_obj" and isinstance(item, list):
                result[key] = sorted(item)
            else:
                result[key] = canonical(item)
        return result
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def digest(value: Any) -> str:
    payload = json.dumps(canonical(value), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()[:16]


def reset_once(args: argparse.Namespace, episode: dict[str, Any]) -> dict[str, str]:
    config = RoboCasaEvalConfig(
        env_name=args.env_name,
        seed=int(episode["seed"]),
        obj_instance_split=args.obj_instance_split,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )
    env = create_robocasa_env(config)
    try:
        reseed_env(env, int(episode["seed"]))
        set_ep_meta(env, episode["ep_meta"])
        obs = env.reset()
        meta = env.get_ep_meta()
        frame = render_composite_from_obs(obs)
        return {
            "meta": digest(meta),
            "frame": image_digest(frame),
            "lang": meta.get("lang", ""),
            "layout": str(meta.get("layout_id")),
            "style": str(meta.get("style_id")),
        }
    finally:
        env.close()


def reset_on_env(env, episode: dict[str, Any]) -> dict[str, str]:
    reseed_env(env, int(episode["seed"]))
    set_ep_meta(env, episode["ep_meta"])
    obs = env.reset()
    meta = env.get_ep_meta()
    frame = render_composite_from_obs(obs)
    return {
        "meta": digest(meta),
        "frame": image_digest(frame),
        "lang": meta.get("lang", ""),
        "layout": str(meta.get("layout_id")),
        "style": str(meta.get("style_id")),
    }


def scene_signature(episode: dict[str, Any]) -> tuple[Any, Any]:
    ep_meta = episode.get("ep_meta", {})
    return ep_meta.get("layout_id"), ep_meta.get("style_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule_path", required=True)
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--obj_instance_split", default="A")
    parser.add_argument("--camera_width", type=int, default=128)
    parser.add_argument("--camera_height", type=int, default=128)
    parser.add_argument("--check_reuse_env", action="store_true")
    parser.add_argument("--scene_scoped_reuse", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    schedule = json.loads(Path(args.schedule_path).read_text())
    episodes = schedule["episodes"][: args.max_episodes] if args.max_episodes > 0 else schedule["episodes"]
    all_same = True
    reuse_env = None
    active_signature = None
    if args.check_reuse_env and episodes and not args.scene_scoped_reuse:
        reuse_env = create_robocasa_env(
            RoboCasaEvalConfig(
                env_name=args.env_name,
                seed=int(episodes[0]["seed"]),
                obj_instance_split=args.obj_instance_split,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
            )
        )
    for episode in episodes:
        first = reset_once(args, episode)
        second = reset_once(args, episode)
        if args.check_reuse_env and args.scene_scoped_reuse:
            signature = scene_signature(episode)
            if reuse_env is None or signature != active_signature:
                if reuse_env is not None:
                    reuse_env.close()
                reuse_env = create_robocasa_env(
                    RoboCasaEvalConfig(
                        env_name=args.env_name,
                        seed=int(episode["seed"]),
                        obj_instance_split=args.obj_instance_split,
                        camera_width=args.camera_width,
                        camera_height=args.camera_height,
                    )
                )
                active_signature = signature
        reuse = reset_on_env(reuse_env, episode) if reuse_env is not None else first
        same = first == second and first == reuse
        all_same &= same
        print(
            f"ep{episode['episode_idx']:03d} seed={episode['seed']} same={same} "
            f"meta={first['meta']}/{second['meta']} frame={first['frame']}/{second['frame']} "
            f"reuse_meta={reuse['meta']} reuse_frame={reuse['frame']} "
            f"layout={first['layout']} style={first['style']} lang={first['lang']}"
        )
    if reuse_env is not None:
        reuse_env.close()
    print(f"ALL_SAME {all_same}")
    if not all_same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
