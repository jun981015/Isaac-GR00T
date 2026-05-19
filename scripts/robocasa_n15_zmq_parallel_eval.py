#!/usr/bin/env python3
"""Parallel RoboCasa eval with fixed schedule replay and seeded ZMQ policy calls."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.robocasa_n15_wrapper import (
    action_dict_to_robosuite,
    create_robocasa_env,
    obs_to_policy,
    reseed_env,
    set_ep_meta,
    success_diagnostics_from_env,
    success_from_env,
)
from scripts.robocasa_n15_zmq_eval import (
    RolloutIORecorder,
    TASKS,
    StreamingVideo,
    existing_episode_metadata,
    env_seed,
    generate_schedule,
    jsonable,
    load_or_generate_schedule,
    make_eval_config,
    obs_to_state_policy,
    parse_layouts,
    read_json,
    render_video_frame,
    scene_signature,
    set_seed,
    wait_for_server,
    write_json,
)


def action_seed(base_seed: int, episode_idx: int, policy_call_idx: int) -> int:
    return int(base_seed + episode_idx * 100_000 + policy_call_idx)


STATIC_CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
)
HAND_CAMERA_KEYS = ("video.robot0_eye_in_hand",)


def mask_policy_cameras(policy_obs: dict[str, Any], mode: str) -> dict[str, Any]:
    """Ablate policy camera inputs without changing env observations or videos."""
    if mode == "none":
        return policy_obs
    if mode == "no_hand":
        keys = HAND_CAMERA_KEYS
    elif mode == "no_static":
        keys = STATIC_CAMERA_KEYS
    elif mode == "hand_only":
        keys = STATIC_CAMERA_KEYS
    elif mode == "static_only":
        keys = HAND_CAMERA_KEYS
    else:
        raise ValueError(f"Unsupported camera_ablation mode: {mode}")

    for key in keys:
        if key in policy_obs:
            policy_obs[key] = np.zeros_like(policy_obs[key])
    return policy_obs


def rollout_worker_episode(
    args: argparse.Namespace,
    episode: dict[str, Any],
    task_dir: Path,
    env,
    conn,
    env_recreated: bool,
) -> dict[str, Any]:
    episode_idx = int(episode["episode_idx"])
    config = make_eval_config(args, episode_idx)
    frames: list[np.ndarray] = []
    success = False
    policy_calls = 0
    env_steps = 0
    policy_wait_sec = 0.0
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
            policy_obs = mask_policy_cameras(policy_obs, args.camera_ablation)
            wait_start = time.time()
            conn.send(
                {
                    "type": "need_action",
                    "episode_idx": episode_idx,
                    "policy_call_idx": policy_calls,
                    "policy_obs": policy_obs,
                }
            )
            action = conn.recv()
            policy_wait_sec += time.time() - wait_start
            if not isinstance(action, dict) or "action" not in action:
                raise RuntimeError(f"Unexpected action response: {type(action)}")
            action_response = action
            action = action_response["action"]
            recorder.record_policy_call(
                policy_obs=policy_obs,
                action=action,
                env_step=env_steps,
                policy_call_idx=policy_calls,
                policy_seed=action_response.get("policy_seed"),
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
        "success_diagnostics": success_diagnostics_from_env(env),
        "env_steps": env_steps,
        "policy_calls": policy_calls,
        "elapsed_sec": time.time() - start,
        "reset_sec": reset_sec,
        "rollout_sec": rollout_sec,
        "policy_wait_sec": policy_wait_sec,
        "env_step_sec": env_step_sec,
        "render_sec": render_sec,
        "video_sec": video_sec,
        "env_recreated": env_recreated,
        "scene_signature": scene_signature(episode),
        "video_written": args.write_video,
        "video_path": str(video_path) if video_path is not None else None,
        "rollout_hdf5_written": args.save_rollout_hdf5,
        "rollout_hdf5_path": str(hdf5_path) if args.save_rollout_hdf5 else None,
        "camera_ablation": args.camera_ablation,
        "ep_meta": episode["ep_meta"],
    }
    recorder.write(metadata)
    write_json(task_dir / "episodes" / f"ep{episode_idx:03d}.json", metadata)
    return metadata


def worker_main(worker_idx: int, args: argparse.Namespace, task_dir: str, conn) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    random.seed(args.seed + worker_idx)
    np.random.seed(args.seed + worker_idx)
    task_path = Path(task_dir)
    env = None
    try:
        while True:
            msg = conn.recv()
            if msg["type"] == "stop":
                break
            if msg["type"] != "episode":
                raise RuntimeError(f"Unknown worker message: {msg['type']}")
            episode = msg["episode"]
            episode_idx = int(episode["episode_idx"])
            if env is None:
                env = create_robocasa_env(make_eval_config(args, episode_idx))
                env_recreated = True
            else:
                env_recreated = False
            metadata = rollout_worker_episode(
                args=args,
                episode=episode,
                task_dir=task_path,
                env=env,
                conn=conn,
                env_recreated=env_recreated,
            )
            conn.send({"type": "done", "metadata": metadata})
    except BaseException as exc:
        conn.send({"type": "error", "worker_idx": worker_idx, "error": repr(exc)})
    finally:
        if env is not None:
            env.close()
        conn.close()


def batch_policy_observations(requests: list[dict[str, Any]]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    keys = requests[0]["policy_obs"].keys()
    for key in keys:
        values = [request["policy_obs"][key] for request in requests]
        if all(isinstance(value, np.ndarray) for value in values):
            batched[key] = np.stack(values, axis=0)
        else:
            batched[key] = values
    return batched


def split_action_batch(action: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    split_actions = [dict() for _ in range(batch_size)]
    for key, value in action.items():
        array = np.asarray(value)
        if array.shape[0] != batch_size:
            raise RuntimeError(
                f"Expected batched action key {key!r} to have leading dim {batch_size}, "
                f"got shape {array.shape}"
            )
        for idx in range(batch_size):
            split_actions[idx][key] = array[idx : idx + 1]
    return split_actions


def batch_seed(base_seed: int, requests: list[dict[str, Any]]) -> int:
    first_episode = min(int(request["episode_idx"]) for request in requests)
    first_call = min(int(request["policy_call_idx"]) for request in requests)
    batch_size = len(requests)
    return int(base_seed + first_episode * 100_000 + first_call * 1_000 + batch_size)


def request_seeded_action_batch(
    client: RobotInferenceClient,
    requests: list[dict[str, Any]],
    base_seed: int,
) -> tuple[list[dict[str, Any]], int]:
    if len(requests) == 1:
        request = requests[0]
        seed = action_seed(
            base_seed,
            int(request["episode_idx"]),
            int(request["policy_call_idx"]),
        )
        action = client.get_action_seeded(
            request["policy_obs"],
            action_seed=seed,
        )
        return [action], seed

    batched_obs = batch_policy_observations(requests)
    seed = batch_seed(base_seed, requests)
    action = client.get_action_seeded(
        batched_obs,
        action_seed=seed,
    )
    return split_action_batch(action, len(requests)), seed


def write_summary(args: argparse.Namespace, task_dir: Path, results: list[dict[str, Any]]) -> None:
    results = sorted(results, key=lambda item: int(item["episode_idx"]))
    successes = [bool(item["success"]) for item in results]
    summary = {
        "env_name": args.env_name,
        "n_episodes": len(results),
        "n_envs": args.n_envs,
        "successes": successes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "total_env_steps": int(sum(item["env_steps"] for item in results)),
        "total_policy_calls": int(sum(item["policy_calls"] for item in results)),
        "total_policy_wait_sec": float(sum(item.get("policy_wait_sec", 0.0) for item in results)),
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
            "transport": "zmq_parallel_seeded",
            "n_envs": args.n_envs,
            "policy_batch_mode": args.policy_batch_mode,
            "policy_batch_wait_sec": args.policy_batch_wait_sec,
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
            "camera_ablation": args.camera_ablation,
            "write_video": args.write_video,
            "stream_video": args.stream_video,
            "obj_instance_split": args.obj_instance_split,
            "layout_style_ids": parse_layouts(args.layout_style_ids),
            "deterministic_action_seed": True,
        },
    )

    all_episodes = schedule["episodes"][: args.n_episodes]
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for episode in all_episodes:
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
        pending.append(episode)

    if not pending:
        write_summary(args, task_dir, results)
        return

    ctx = mp.get_context("spawn")
    worker_count = min(args.n_envs, len(pending))
    workers = []
    conn_to_worker: dict[Any, int] = {}
    worker_busy: dict[int, bool] = {}
    next_episode = 0
    try:
        for worker_idx in range(worker_count):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=worker_main,
                args=(worker_idx, args, str(task_dir), child_conn),
                daemon=False,
            )
            proc.start()
            child_conn.close()
            workers.append((worker_idx, proc, parent_conn))
            conn_to_worker[parent_conn] = worker_idx
            worker_busy[worker_idx] = False

        for worker_idx, _proc, conn in workers:
            episode = pending[next_episode]
            next_episode += 1
            conn.send({"type": "episode", "episode": episode})
            worker_busy[worker_idx] = True

        completed_new = 0
        action_requests: list[tuple[Any, dict[str, Any]]] = []
        while completed_new < len(pending):
            active_conns = [conn for _idx, _proc, conn in workers if worker_busy[_idx]]
            ready = wait(active_conns)
            for conn in ready:
                msg = conn.recv()
                worker_idx = conn_to_worker[conn]
                if msg["type"] == "need_action":
                    action_requests.append((conn, msg))
                elif msg["type"] == "done":
                    metadata = msg["metadata"]
                    results.append(metadata)
                    completed_new += 1
                    print(
                        f"[episode] {args.env_name} ep={int(metadata['episode_idx']):03d} "
                        f"success={int(bool(metadata['success']))} "
                        f"env_steps={metadata['env_steps']} policy_calls={metadata['policy_calls']} "
                        f"env_step_sec={metadata.get('env_step_sec', 0.0):.1f} "
                        f"render_sec={metadata.get('render_sec', 0.0):.1f}"
                    )
                    if next_episode < len(pending):
                        episode = pending[next_episode]
                        next_episode += 1
                        conn.send({"type": "episode", "episode": episode})
                    else:
                        worker_busy[worker_idx] = False
                        conn.send({"type": "stop"})
                elif msg["type"] == "error":
                    raise RuntimeError(f"worker {msg['worker_idx']} failed: {msg['error']}")
                else:
                    raise RuntimeError(f"Unknown message from worker: {msg['type']}")

            if action_requests:
                deadline = time.time() + args.policy_batch_wait_sec
                while True:
                    if args.policy_batch_mode == "lockstep":
                        remaining = None
                    else:
                        remaining = max(0.0, deadline - time.time())
                    active_conns = [
                        conn
                        for _idx, _proc, conn in workers
                        if worker_busy[_idx]
                        and all(conn is not request_conn for request_conn, _ in action_requests)
                    ]
                    if not active_conns:
                        break
                    if remaining is not None and remaining <= 0:
                        break
                    extra_ready = wait(active_conns, timeout=remaining)
                    if not extra_ready:
                        break
                    for conn in extra_ready:
                        msg = conn.recv()
                        worker_idx = conn_to_worker[conn]
                        if msg["type"] == "need_action":
                            action_requests.append((conn, msg))
                        elif msg["type"] == "done":
                            metadata = msg["metadata"]
                            results.append(metadata)
                            completed_new += 1
                            print(
                                f"[episode] {args.env_name} ep={int(metadata['episode_idx']):03d} "
                                f"success={int(bool(metadata['success']))} "
                                f"env_steps={metadata['env_steps']} policy_calls={metadata['policy_calls']} "
                                f"env_step_sec={metadata.get('env_step_sec', 0.0):.1f} "
                                f"render_sec={metadata.get('render_sec', 0.0):.1f}"
                            )
                            if next_episode < len(pending):
                                episode = pending[next_episode]
                                next_episode += 1
                                conn.send({"type": "episode", "episode": episode})
                            else:
                                worker_busy[worker_idx] = False
                                conn.send({"type": "stop"})
                        elif msg["type"] == "error":
                            raise RuntimeError(f"worker {msg['worker_idx']} failed: {msg['error']}")
                        else:
                            raise RuntimeError(f"Unknown message from worker: {msg['type']}")

                action_conns = [conn for conn, _msg in action_requests]
                request_payloads = [msg for _conn, msg in action_requests]
                actions, policy_seed = request_seeded_action_batch(client, request_payloads, args.seed)
                print(
                    f"[policy_batch] size={len(actions)} "
                    f"episodes={[int(msg['episode_idx']) for msg in request_payloads]} "
                    f"calls={[int(msg['policy_call_idx']) for msg in request_payloads]} "
                    f"seed={policy_seed}"
                )
                for conn, action in zip(action_conns, actions):
                    conn.send({"action": action, "policy_seed": policy_seed})
                action_requests.clear()
    finally:
        for _worker_idx, proc, conn in workers:
            if proc.is_alive():
                try:
                    conn.send({"type": "stop"})
                except Exception:
                    pass
            conn.close()
        for _worker_idx, proc, _conn in workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

    write_summary(args, task_dir, results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", required=True, choices=sorted(TASKS.values()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schedule_path", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--policy_batch_mode", choices=("lockstep", "coalesce"), default="lockstep")
    parser.add_argument("--policy_batch_wait_sec", type=float, default=0.05)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--max_episode_steps", type=int, default=800)
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--video_scale", type=int, default=2)
    parser.add_argument("--video_render_size", type=int, default=256)
    parser.add_argument("--video_source", choices=("obs", "render"), default="obs")
    parser.add_argument("--video_steps_per_render", type=int, default=4)
    parser.add_argument("--policy_image_size", type=int, default=128)
    parser.add_argument(
        "--save_rollout_hdf5",
        action="store_true",
        help="Save policy inputs, predicted action chunks, and executed raw actions for every episode.",
    )
    parser.add_argument(
        "--camera_ablation",
        choices=("none", "no_hand", "no_static", "hand_only", "static_only"),
        default="none",
        help="Zero selected camera inputs before sending observations to the policy.",
    )
    parser.add_argument("--no_video", dest="write_video", action="store_false")
    parser.add_argument("--stream_video", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--no_offscreen_renderer", dest="has_offscreen_renderer", action="store_false")
    parser.add_argument("--no_camera_obs", dest="use_camera_obs", action="store_false")
    parser.add_argument("--obj_instance_split", default="A")
    parser.add_argument("--generative_textures", default=None)
    parser.add_argument("--layout_style_ids", default="1:1,2:2,4:4,6:9,7:10")
    parser.add_argument("--randomize_cameras", action="store_true")
    parser.add_argument("--regenerate_schedule", action="store_true")
    parser.add_argument("--generate_schedule_only", action="store_true")
    parser.add_argument("--server_timeout_sec", type=int, default=300)
    parser.add_argument("--ping_timeout_ms", type=int, default=5000)
    parser.add_argument("--action_timeout_ms", type=int, default=120000)
    parser.add_argument("--api_token", default=None)
    parser.set_defaults(write_video=True, has_offscreen_renderer=True, use_camera_obs=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_envs < 1:
        raise ValueError("--n_envs must be >= 1")
    if args.generate_schedule_only:
        os.environ.setdefault("MUJOCO_GL", "egl")
        set_seed(args.seed)
        generate_schedule(args)
        return
    run_eval(args)


if __name__ == "__main__":
    main()
