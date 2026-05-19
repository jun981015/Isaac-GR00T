#!/usr/bin/env python3
"""Evaluate GR00T N1.5 RoboCasa policies through the ZMQ inference server."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v2 as imageio

from gr00t.eval.robot import RobotInferenceClient

from gr00t.eval.wrappers.robocasa_n15_wrapper import (
    RoboCasaEvalConfig,
    STATE_KEY_MAP,
    action_dict_to_robosuite,
    create_robocasa_env,
    get_ep_meta,
    obs_to_policy,
    render_composite_from_obs,
    render_highres_composite,
    reseed_env,
    set_ep_meta,
    success_from_env,
)

TASKS = {
    "sink": "PnPCounterToSink",
    "stove": "PnPCounterToStove",
    "microwave": "PnPMicrowaveToCounter",
    "prepare_coffee": "PrepareCoffee",
    "microwave_thawing": "MicrowaveThawing",
    "coffee_setup_mug": "CoffeeSetupMug",
    "coffee_press_button": "CoffeePressButton",
    "coffee_serve_mug": "CoffeeServeMug",
    "open_single_door": "OpenSingleDoor",
    "close_single_door": "CloseSingleDoor",
    "open_double_door": "OpenDoubleDoor",
    "close_double_door": "CloseDoubleDoor",
    "open_drawer": "OpenDrawer",
    "close_drawer": "CloseDrawer",
    "cab_to_counter": "PnPCabToCounter",
    "counter_to_cab": "PnPCounterToCab",
    "counter_to_microwave": "PnPCounterToMicrowave",
    "sink_to_counter": "PnPSinkToCounter",
    "stove_to_counter": "PnPStoveToCounter",
    "turn_on_microwave": "TurnOnMicrowave",
    "turn_off_microwave": "TurnOffMicrowave",
    "turn_on_sink_faucet": "TurnOnSinkFaucet",
    "turn_off_sink_faucet": "TurnOffSinkFaucet",
    "turn_sink_spout": "TurnSinkSpout",
    "turn_on_stove": "TurnOnStove",
    "turn_off_stove": "TurnOffStove",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_layouts(value: str) -> tuple[tuple[int, int], ...]:
    if not value:
        return ()
    pairs = []
    for item in value.split(","):
        layout, style = item.split(":")
        pairs.append((int(layout), int(style)))
    return tuple(pairs)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def env_seed(base_seed: int, episode_idx: int) -> int:
    return int(base_seed + episode_idx * 256)


def make_eval_config(args: argparse.Namespace, episode_idx: int) -> RoboCasaEvalConfig:
    return RoboCasaEvalConfig(
        env_name=args.env_name,
        seed=env_seed(args.seed, episode_idx),
        obj_instance_split=args.obj_instance_split,
        layout_and_style_ids=parse_layouts(args.layout_style_ids),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        has_offscreen_renderer=args.has_offscreen_renderer,
        use_camera_obs=args.use_camera_obs,
        randomize_cameras=args.randomize_cameras,
        generative_textures=args.generative_textures,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_signature(episode: dict[str, Any]) -> tuple[Any, Any]:
    ep_meta = episode.get("ep_meta", {})
    return ep_meta.get("layout_id"), ep_meta.get("style_id")


def generate_schedule(args: argparse.Namespace) -> dict[str, Any]:
    episodes = []
    start = time.time()
    for episode_idx in range(args.n_episodes):
        config = make_eval_config(args, episode_idx)
        env = create_robocasa_env(config)
        try:
            env.reset()
            ep_meta = get_ep_meta(env)
        finally:
            env.close()
        episodes.append(
            {
                "episode_idx": episode_idx,
                "seed": config.seed,
                "ep_meta": ep_meta,
            }
        )
        print(
            f"[schedule] {args.env_name} ep={episode_idx:03d} seed={config.seed} "
            f"layout={ep_meta.get('layout_id')} style={ep_meta.get('style_id')} "
            f"lang={ep_meta.get('lang', '')}"
        )
    schedule = {
        "version": 1,
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - start,
        "env_name": args.env_name,
        "base_seed": args.seed,
        "n_episodes": args.n_episodes,
        "obj_instance_split": args.obj_instance_split,
        "layout_style_ids": parse_layouts(args.layout_style_ids),
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "randomize_cameras": args.randomize_cameras,
        "episodes": episodes,
    }
    write_json(Path(args.schedule_path), schedule)
    return schedule


def load_or_generate_schedule(args: argparse.Namespace) -> dict[str, Any]:
    schedule_path = Path(args.schedule_path)
    if schedule_path.exists() and not args.regenerate_schedule:
        return read_json(schedule_path)
    return generate_schedule(args)


def wait_for_server(args: argparse.Namespace) -> RobotInferenceClient:
    deadline = time.time() + args.server_timeout_sec
    last_error = None
    while time.time() < deadline:
        client = RobotInferenceClient(
            host=args.host,
            port=args.port,
            timeout_ms=args.ping_timeout_ms,
            api_token=args.api_token,
        )
        try:
            if client.ping():
                print(f"[server] healthy: zmq://{args.host}:{args.port}")
                return RobotInferenceClient(
                    host=args.host,
                    port=args.port,
                    timeout_ms=args.action_timeout_ms,
                    api_token=args.api_token,
                )
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"ZMQ inference server not ready at {args.host}:{args.port}: {last_error}")


def request_action(client: RobotInferenceClient, obs: dict[str, Any]) -> dict[str, Any]:
    return client.get_action(obs)


def scale_video_frame(frame: np.ndarray, scale: int | float) -> np.ndarray:
    if scale <= 1:
        return frame
    if not float(scale).is_integer():
        try:
            import cv2

            height = int(round(frame.shape[0] * scale))
            width = int(round(frame.shape[1] * scale))
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            scale = round(scale)
    scale = int(scale)
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def render_video_frame(args: argparse.Namespace, obs: dict[str, Any], env) -> np.ndarray:
    if args.video_source == "render" and args.video_render_size > 0:
        return render_highres_composite(
            env,
            height=args.video_render_size,
            width=args.video_render_size,
        )
    frame = render_composite_from_obs(obs)
    if args.video_render_size > 0:
        return scale_video_frame(frame, args.video_render_size / frame.shape[0])
    return scale_video_frame(frame, args.video_scale)


def obs_to_state_policy(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """GR00T state-only observation at one env step; images are stored via mp4."""
    return {
        gr00t_key: np.asarray(obs[robocasa_key])[None]
        for robocasa_key, gr00t_key in STATE_KEY_MAP.items()
    }


class StreamingVideo:
    def __init__(self, path: Path, fps: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=fps)

    def append(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()


def _squeeze_batch(array: Any) -> np.ndarray:
    value = np.asarray(array)
    if value.shape[:1] == (1,):
        return value[0]
    return value


class RolloutIORecorder:
    """Collect exact policy inputs / outputs for later offline training."""

    def __init__(self, *, enabled: bool, path: Path, metadata: dict[str, Any]):
        self.enabled = enabled
        self.path = path
        self.metadata = metadata
        self.policy_obs: dict[str, list[np.ndarray]] = {}
        self.policy_actions: dict[str, list[np.ndarray]] = {}
        self.policy_call_env_steps: list[int] = []
        self.policy_call_indices: list[int] = []
        self.policy_seeds: list[int] = []
        self.step_obs: dict[str, list[np.ndarray]] = {}
        self.raw_actions: list[np.ndarray] = []
        self.action_indices: list[int] = []
        self.action_policy_call_indices: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.successes: list[bool] = []

    def record_policy_call(
        self,
        *,
        policy_obs: dict[str, Any],
        action: dict[str, Any],
        env_step: int,
        policy_call_idx: int,
        policy_seed: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        for key, value in policy_obs.items():
            if key == "annotation.human.action.task_description":
                continue
            if key.startswith("video."):
                continue
            self.policy_obs.setdefault(key, []).append(_squeeze_batch(value).copy())
        for key, value in action.items():
            self.policy_actions.setdefault(key, []).append(_squeeze_batch(value).copy())
        self.policy_call_env_steps.append(int(env_step))
        self.policy_call_indices.append(int(policy_call_idx))
        self.policy_seeds.append(-1 if policy_seed is None else int(policy_seed))

    def record_step_obs(self, *, obs: dict[str, Any]) -> None:
        if not self.enabled:
            return
        for key, value in obs.items():
            if key.startswith("video."):
                continue
            self.step_obs.setdefault(key, []).append(_squeeze_batch(value).copy())

    def record_env_step(
        self,
        *,
        raw_action: np.ndarray,
        action_idx: int,
        policy_call_idx: int,
        reward: float,
        done: bool,
        success: bool,
    ) -> None:
        if not self.enabled:
            return
        self.raw_actions.append(np.asarray(raw_action, dtype=np.float32).copy())
        self.action_indices.append(int(action_idx))
        self.action_policy_call_indices.append(int(policy_call_idx))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.successes.append(bool(success))

    def write(self, final_metadata: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("--save_rollout_hdf5 requires h5py in the eval environment") from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(self.path, "w") as h5:
            h5.attrs["format"] = "gr00t_robocasa_eval_rollout_io_v1"
            h5.attrs["description"] = (
                "Policy-call-level GR00T inputs and predicted action chunks, "
                "plus per-env-step raw actions actually executed in RoboCasa. "
                "Image observations are intentionally omitted; use metadata/video_path mp4 instead."
            )
            meta = h5.create_group("metadata")
            combined = {**self.metadata, **final_metadata}
            for key, value in combined.items():
                if key == "ep_meta":
                    meta.attrs[key] = json.dumps(jsonable(value), ensure_ascii=False)
                elif isinstance(value, (dict, list, tuple)):
                    meta.attrs[key] = json.dumps(jsonable(value), ensure_ascii=False)
                elif value is None:
                    meta.attrs[key] = ""
                else:
                    meta.attrs[key] = value

            prompt = str(combined.get("ep_meta", {}).get("lang", ""))
            h5.create_dataset("prompt", data=prompt, dtype=string_dtype)
            h5.create_dataset("video_path", data=str(combined.get("video_path", "")), dtype=string_dtype)

            policy = h5.create_group("policy")
            policy.create_dataset("call_env_step", data=np.asarray(self.policy_call_env_steps, dtype=np.int32))
            policy.create_dataset("call_index", data=np.asarray(self.policy_call_indices, dtype=np.int32))
            policy.create_dataset("call_seed", data=np.asarray(self.policy_seeds, dtype=np.int64))
            obs_group = policy.create_group("obs")
            for key, values in sorted(self.policy_obs.items()):
                obs_group.create_dataset(key, data=np.stack(values), compression="lzf")
            action_group = policy.create_group("action_pred")
            for key, values in sorted(self.policy_actions.items()):
                action_group.create_dataset(key, data=np.stack(values), compression="lzf")

            steps = h5.create_group("env_steps")
            step_obs_group = steps.create_group("obs")
            for key, values in sorted(self.step_obs.items()):
                step_obs_group.create_dataset(key, data=np.stack(values), compression="lzf")
            if self.raw_actions:
                steps.create_dataset("raw_action", data=np.stack(self.raw_actions), compression="lzf")
            else:
                steps.create_dataset("raw_action", data=np.zeros((0, 0), dtype=np.float32))
            steps.create_dataset("action_idx", data=np.asarray(self.action_indices, dtype=np.int32))
            steps.create_dataset(
                "policy_call_idx",
                data=np.asarray(self.action_policy_call_indices, dtype=np.int32),
            )
            steps.create_dataset("reward", data=np.asarray(self.rewards, dtype=np.float32))
            steps.create_dataset("done", data=np.asarray(self.dones, dtype=np.bool_))
            steps.create_dataset("success", data=np.asarray(self.successes, dtype=np.bool_))
        return self.path


def rollout_episode(
    args: argparse.Namespace,
    client: RobotInferenceClient,
    episode: dict[str, Any],
    task_dir: Path,
    env=None,
    env_recreated: bool = True,
) -> dict[str, Any]:
    episode_idx = int(episode["episode_idx"])
    config = make_eval_config(args, episode_idx)
    owns_env = env is None
    if owns_env:
        env = create_robocasa_env(config)
    frames: list[np.ndarray] = []
    success = False
    policy_calls = 0
    env_steps = 0
    policy_sec = 0.0
    env_step_sec = 0.0
    render_sec = 0.0
    start = time.time()
    reset_start = time.time()
    tmp_video_path = task_dir / "videos" / f"ep{episode_idx:03d}_seed{config.seed}_streaming_tmp.mp4"
    stream = StreamingVideo(tmp_video_path, args.video_fps) if args.write_video and args.stream_video else None
    hdf5_path = task_dir / "hdf5" / f"ep{episode_idx:03d}_seed{config.seed}_rollout_io.hdf5"
    recorder = RolloutIORecorder(
        enabled=args.save_rollout_hdf5,
        path=hdf5_path,
        metadata={
            "env_name": args.env_name,
            "episode_idx": episode_idx,
            "seed": config.seed,
            "ep_meta": episode["ep_meta"],
            "n_action_steps": args.n_action_steps,
            "policy_image_size": args.policy_image_size,
        },
    )
    try:
        reseed_env(env, config.seed)
        set_ep_meta(env, episode["ep_meta"])
        obs = env.reset()
        reset_sec = time.time() - reset_start
        rollout_start = time.time()
        if args.write_video:
            render_start = time.time()
            frame = render_video_frame(args, obs, env)
            render_sec += time.time() - render_start
            if stream is not None:
                stream.append(frame)
            else:
                frames.append(frame)
        while env_steps < args.max_episode_steps:
            policy_obs = obs_to_policy(obs, env, image_size=args.policy_image_size)
            policy_start = time.time()
            action = request_action(client, policy_obs)
            policy_sec += time.time() - policy_start
            recorder.record_policy_call(
                policy_obs=policy_obs,
                action=action,
                env_step=env_steps,
                policy_call_idx=policy_calls,
            )
            policy_calls += 1
            for action_idx in range(args.n_action_steps):
                recorder.record_step_obs(obs=obs_to_state_policy(obs))
                raw_action = action_dict_to_robosuite(action, action_idx)
                env_step_start = time.time()
                obs, reward, done, _info = env.step(raw_action)
                env_step_sec += time.time() - env_step_start
                env_steps += 1
                success = success or success_from_env(env)
                recorder.record_env_step(
                    raw_action=raw_action,
                    action_idx=action_idx,
                    policy_call_idx=policy_calls - 1,
                    reward=float(reward),
                    done=bool(done),
                    success=bool(success),
                )
                if args.write_video and (env_steps % args.video_steps_per_render) == 0:
                    render_start = time.time()
                    frame = render_video_frame(args, obs, env)
                    render_sec += time.time() - render_start
                    if stream is not None:
                        stream.append(frame)
                    else:
                        frames.append(frame)
                if done or success or env_steps >= args.max_episode_steps:
                    break
            if done or success:
                break
        rollout_sec = time.time() - rollout_start
    finally:
        if stream is not None:
            stream.close()
        if owns_env:
            env.close()

    video_sec = 0.0
    video_path = None
    if args.write_video:
        video_start = time.time()
        video_name = f"ep{episode_idx:03d}_seed{config.seed}_composite_outcome{int(success)}.mp4"
        video_path = task_dir / "videos" / video_name
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if stream is not None:
            tmp_video_path.replace(video_path)
        else:
            imageio.mimsave(video_path, frames, fps=args.video_fps)
        video_sec = time.time() - video_start

    metadata = {
        "env_name": args.env_name,
        "episode_idx": episode_idx,
        "seed": config.seed,
        "success": success,
        "env_steps": env_steps,
        "policy_calls": policy_calls,
        "elapsed_sec": time.time() - start,
        "reset_sec": reset_sec,
        "rollout_sec": rollout_sec,
        "policy_sec": policy_sec,
        "env_step_sec": env_step_sec,
        "render_sec": render_sec,
        "video_sec": video_sec,
        "env_recreated": env_recreated,
        "scene_signature": scene_signature(episode),
        "video_written": args.write_video,
        "video_path": str(video_path) if video_path is not None else None,
        "rollout_hdf5_written": args.save_rollout_hdf5,
        "rollout_hdf5_path": str(hdf5_path) if args.save_rollout_hdf5 else None,
        "ep_meta": episode["ep_meta"],
    }
    recorder.write(metadata)
    write_json(task_dir / "episodes" / f"ep{episode_idx:03d}.json", metadata)
    print(
        f"[episode] {args.env_name} ep={episode_idx:03d} "
        f"success={int(success)} env_steps={env_steps} policy_calls={policy_calls} "
        f"policy_sec={policy_sec:.1f} env_step_sec={env_step_sec:.1f} "
        f"render_sec={render_sec:.1f} video={video_path.name if video_path is not None else 'none'}"
    )
    return metadata


def existing_episode_metadata(task_dir: Path, episode_idx: int) -> dict[str, Any] | None:
    metadata_path = task_dir / "episodes" / f"ep{episode_idx:03d}.json"
    if not metadata_path.exists():
        return None
    metadata = read_json(metadata_path)
    if metadata.get("video_written") is False:
        return metadata
    video_path = Path(str(metadata.get("video_path", "")))
    if not video_path.is_absolute():
        video_path = task_dir / "videos" / video_path.name
    if not video_path.exists():
        return None
    return metadata


def run_eval(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    set_seed(args.seed)
    client = wait_for_server(args)
    schedule = load_or_generate_schedule(args)

    task_dir = Path(args.output_dir) / args.env_name
    write_json(
        task_dir / "run_config.json",
        {
            "env_name": args.env_name,
            "seed": args.seed,
            "schedule_path": args.schedule_path,
            "host": args.host,
            "port": args.port,
            "transport": "zmq",
            "n_episodes": args.n_episodes,
            "n_action_steps": args.n_action_steps,
            "max_episode_steps": args.max_episode_steps,
            "video_fps": args.video_fps,
            "video_scale": args.video_scale,
            "video_render_size": args.video_render_size,
            "video_source": args.video_source,
            "video_steps_per_render": args.video_steps_per_render,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "policy_image_size": args.policy_image_size,
            "save_rollout_hdf5": args.save_rollout_hdf5,
            "write_video": args.write_video,
            "stream_video": args.stream_video,
            "reuse_env": args.reuse_env,
            "obj_instance_split": args.obj_instance_split,
            "layout_style_ids": parse_layouts(args.layout_style_ids),
        },
    )

    results = []
    episodes = schedule["episodes"][: args.n_episodes]
    if args.reuse_env:
        base_config = make_eval_config(args, int(episodes[0]["episode_idx"]) if episodes else 0)
        env = create_robocasa_env(base_config) if episodes else None
        try:
            for episode in episodes:
                episode_idx = int(episode["episode_idx"])
                if args.skip_existing:
                    metadata = existing_episode_metadata(task_dir, episode_idx)
                    if metadata is not None and args.save_rollout_hdf5:
                        hdf5_path = Path(str(metadata.get("rollout_hdf5_path", "")))
                        if not hdf5_path.exists():
                            metadata = None
                    if metadata is not None:
                        print(
                            f"[skip] {args.env_name} ep={episode_idx:03d} "
                            f"success={int(bool(metadata.get('success')))}"
                        )
                        results.append(metadata)
                        continue
                results.append(
                    rollout_episode(
                        args,
                        client,
                        episode,
                        task_dir,
                        env=env,
                        env_recreated=False,
                    )
                )
        finally:
            if env is not None:
                env.close()
    else:
        for episode in episodes:
            episode_idx = int(episode["episode_idx"])
            if args.skip_existing:
                metadata = existing_episode_metadata(task_dir, episode_idx)
                if metadata is not None and args.save_rollout_hdf5:
                    hdf5_path = Path(str(metadata.get("rollout_hdf5_path", "")))
                    if not hdf5_path.exists():
                        metadata = None
                if metadata is not None:
                    print(
                        f"[skip] {args.env_name} ep={episode_idx:03d} "
                        f"success={int(bool(metadata.get('success')))}"
                    )
                    results.append(metadata)
                    continue
            results.append(rollout_episode(args, client, episode, task_dir))

    successes = [bool(item["success"]) for item in results]
    summary = {
        "env_name": args.env_name,
        "n_episodes": len(results),
        "successes": successes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "total_env_steps": int(sum(item["env_steps"] for item in results)),
        "total_policy_calls": int(sum(item["policy_calls"] for item in results)),
        "total_policy_sec": float(sum(item.get("policy_sec", 0.0) for item in results)),
        "total_env_step_sec": float(sum(item.get("env_step_sec", 0.0) for item in results)),
        "total_render_sec": float(sum(item.get("render_sec", 0.0) for item in results)),
        "total_video_sec": float(sum(item.get("video_sec", 0.0) for item in results)),
        "results": results,
    }
    write_json(task_dir / "summary.json", summary)
    print(
        f"[summary] {args.env_name}: {sum(successes)}/{len(successes)} "
        f"success_rate={summary['success_rate']:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", required=True, choices=sorted(TASKS.values()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schedule_path", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--max_episode_steps", type=int, default=800)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--video_scale", type=int, default=2)
    parser.add_argument(
        "--video_render_size",
        type=int,
        default=256,
        help="Output video height per camera. With --video_source render this is true sim.render size; with obs it upscales policy obs.",
    )
    parser.add_argument("--video_source", choices=("obs", "render"), default="obs")
    parser.add_argument("--video_steps_per_render", type=int, default=4)
    parser.add_argument("--policy_image_size", type=int, default=128)
    parser.add_argument(
        "--save_rollout_hdf5",
        action="store_true",
        help="Save policy inputs, predicted action chunks, and executed raw actions for every episode.",
    )
    parser.add_argument("--no_video", dest="write_video", action="store_false")
    parser.add_argument("--stream_video", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_reuse_env", dest="reuse_env", action="store_false")
    parser.set_defaults(reuse_env=True)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--no_offscreen_renderer", dest="has_offscreen_renderer", action="store_false")
    parser.add_argument("--no_camera_obs", dest="use_camera_obs", action="store_false")
    parser.set_defaults(has_offscreen_renderer=True, use_camera_obs=True)
    parser.add_argument("--obj_instance_split", default="A")
    parser.add_argument("--generative_textures", default=None)
    parser.add_argument(
        "--layout_style_ids",
        default="1:1,2:2,4:4,6:9,7:10",
        help="Comma-separated layout:style pairs.",
    )
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--regenerate_schedule", action="store_true")
    parser.add_argument("--generate_schedule_only", action="store_true")
    parser.add_argument("--server_timeout_sec", type=int, default=300)
    parser.add_argument("--ping_timeout_ms", type=int, default=5000)
    parser.add_argument("--action_timeout_ms", type=int, default=120000)
    parser.add_argument("--api_token", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.generate_schedule_only:
        os.environ.setdefault("MUJOCO_GL", "egl")
        set_seed(args.seed)
        generate_schedule(args)
        return
    run_eval(args)


if __name__ == "__main__":
    main()
