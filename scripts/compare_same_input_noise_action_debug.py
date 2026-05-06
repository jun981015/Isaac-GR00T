#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from gr00t.eval.robot import RobotInferenceClient
from gr00t.eval.wrappers.robocasa_n15_wrapper import create_robocasa_env, obs_to_policy
from scripts.robocasa_n15_zmq_eval import make_eval_config, set_seed


def summarize_action(action):
    return {key: np.asarray(value) for key, value in action.items()}


def main() -> None:
    out_path = Path(
        "/workspace/Isaac-GR00T/local_outputs/robocasa_benchmark/noise_debug_b50k/"
        "same_input_noise_action_compare.json"
    )
    args = argparse.Namespace(
        env_name="PrepareCoffee",
        seed=1,
        obj_instance_split="A",
        layout_style_ids="1:1,2:2,4:4,6:9,7:10",
        randomize_cameras=False,
        camera_width=256,
        camera_height=256,
        generative_textures=None,
    )

    set_seed(1)
    env = create_robocasa_env(make_eval_config(args, 0))
    try:
        obs = env.reset()
        policy_obs = obs_to_policy(obs, env, image_size=128)
    finally:
        env.close()

    client = RobotInferenceClient(host="127.0.0.1", port=8092, timeout_ms=120000)
    action_seed1_a = summarize_action(client.get_action_seeded(policy_obs, action_seed=1))
    action_seed1_b = summarize_action(client.get_action_seeded(policy_obs, action_seed=1))
    action_seed2 = summarize_action(client.get_action_seeded(policy_obs, action_seed=2))

    key_report = {}
    seed1_repeat_equal = True
    seed1_vs_seed2_any_diff = False
    for key in sorted(action_seed1_a):
        a = action_seed1_a[key]
        b = action_seed1_b[key]
        c = action_seed2[key]
        same_repeat = bool(np.array_equal(a, b))
        same_seed2 = bool(np.array_equal(a, c))
        seed1_repeat_equal = seed1_repeat_equal and same_repeat
        seed1_vs_seed2_any_diff = seed1_vs_seed2_any_diff or (not same_seed2)
        key_report[key] = {
            "shape": list(a.shape),
            "seed1_repeat_array_equal": same_repeat,
            "seed1_repeat_max_abs_diff": float(np.max(np.abs(a - b))) if a.size else 0.0,
            "seed1_vs_seed2_array_equal": same_seed2,
            "seed1_vs_seed2_max_abs_diff": float(np.max(np.abs(a - c))) if a.size else 0.0,
            "seed1_first8": a.reshape(-1)[:8].astype(float).tolist(),
            "seed1_repeat_first8": b.reshape(-1)[:8].astype(float).tolist(),
            "seed2_first8": c.reshape(-1)[:8].astype(float).tolist(),
        }

    payload = {
        "env_name": "PrepareCoffee",
        "episode_idx": 0,
        "env_seed": 1,
        "same_policy_input_reused": True,
        "calls": [
            {"action_seed": 1, "label": "seed1_first"},
            {"action_seed": 1, "label": "seed1_repeat"},
            {"action_seed": 2, "label": "seed2_control"},
        ],
        "seed1_repeat_all_action_arrays_equal": seed1_repeat_equal,
        "seed1_vs_seed2_any_action_array_differs": seed1_vs_seed2_any_diff,
        "action_keys": key_report,
        "noise_hook_source": "docker logs robocasa-zmq-noise-debug-b50k-gpu1",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2)[:8000])


if __name__ == "__main__":
    main()
