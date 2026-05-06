#!/usr/bin/env python3
"""Create a compact GR00T checkpoint by reusing unchanged base shards.

The output checkpoint remains loadable by HuggingFace `from_pretrained`: changed
tensors are stored locally, while unchanged tensors point to symlinked base
checkpoint shard files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DTYPE_SIZE = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


def load_index(path: Path) -> dict:
    return json.loads((path / "model.safetensors.index.json").read_text())


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * DTYPE_SIZE.get(str(tensor.dtype), tensor.element_size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--link-base", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--changed-shard-name", default="model-changed.safetensors")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base = args.base.resolve()
    link_base = args.link_base.resolve() if args.link_base is not None else base
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)

    ft_index = load_index(checkpoint)
    base_index = load_index(base)
    ft_map = ft_index["weight_map"]
    base_map = base_index["weight_map"]
    if set(ft_map) != set(base_map):
        raise SystemExit("fine-tuned and base checkpoint key sets differ")

    for filename in ["config.json", "trainer_state.json", "training_args.bin"]:
        src = checkpoint / filename
        if src.exists():
            shutil.copy2(src, output / filename)
    if (checkpoint / "experiment_cfg").exists():
        shutil.copytree(checkpoint / "experiment_cfg", output / "experiment_cfg")

    base_handles: dict[Path, safe_open] = {}
    ft_handles: dict[Path, safe_open] = {}

    def handle(cache: dict, path: Path):
        if path not in cache:
            cache[path] = safe_open(path, framework="pt", device="cpu")
        return cache[path]

    changed_tensors: dict[str, torch.Tensor] = {}
    new_weight_map: dict[str, str] = {}
    linked_base_shards: set[str] = set()
    total_size = 0
    changed_size = 0
    unchanged_size = 0

    for key in sorted(ft_map):
        ft_file = checkpoint / ft_map[key]
        base_file = base / base_map[key]
        ft_tensor = handle(ft_handles, ft_file).get_tensor(key)
        base_tensor = handle(base_handles, base_file).get_tensor(key)

        size = tensor_bytes(ft_tensor)
        total_size += size
        same = ft_tensor.shape == base_tensor.shape and torch.equal(
            ft_tensor.to(base_tensor.dtype), base_tensor
        )
        if same:
            symlink_name = f"base-{base_map[key]}"
            linked_base_shards.add(base_map[key])
            new_weight_map[key] = symlink_name
            unchanged_size += size
        else:
            changed_tensors[key] = ft_tensor
            new_weight_map[key] = args.changed_shard_name
            changed_size += size

    for base_shard in sorted(linked_base_shards):
        link_path = output / f"base-{base_shard}"
        link_path.symlink_to(link_base / base_shard)

    save_file(changed_tensors, output / args.changed_shard_name)
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": new_weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    summary = {
        "source_checkpoint": str(checkpoint),
        "base_checkpoint": str(base),
        "link_base_checkpoint": str(link_base),
        "output_checkpoint": str(output),
        "num_changed_tensors": len(changed_tensors),
        "num_unchanged_tensors": len(new_weight_map) - len(changed_tensors),
        "changed_size_gib": changed_size / 1024**3,
        "unchanged_size_gib": unchanged_size / 1024**3,
        "linked_base_shards": sorted(linked_base_shards),
        "changed_shard": args.changed_shard_name,
    }
    (output / "compact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
