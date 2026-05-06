import importlib
import sys

MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "flash_attn",
    "transformers",
    "diffusers",
    "accelerate",
    "numpy",
    "tensorflow",
    "cv2",
    "decord",
    "pytorch3d",
    "torchcodec",
    "zmq",
    "wandb",
    "av",
    "h5py",
    "pyarrow",
    "fastparquet",
    "ray",
    "tianshou",
    "timm",
    "peft",
    "onnx",
    "gymnasium",
    "matplotlib",
    "pandas",
    "scipy",
    "imageio",
    "kornia",
    "albumentations",
    "gr00t",
]


def version_of(module):
    version = getattr(module, "__version__", None)
    if version is not None:
        return version
    return "unknown"


for name in MODULES:
    module = importlib.import_module(name)
    print(f"{name}=={version_of(module)}")

import torch
import cv2
import flash_attn
import flash_attn_2_cuda
import numpy
import transformers
import diffusers
import accelerate

EXPECTED = {
    "torch": "2.5.1+cu124",
    "torchvision": "0.20.1+cu124",
    "flash_attn": "2.7.1.post4",
    "numpy": "1.26.4",
    "transformers": "4.51.3",
    "diffusers": "0.30.2",
    "accelerate": "1.2.1",
    "cv2": "4.8.0",
}

actual = {
    "torch": torch.__version__,
    "torchvision": importlib.import_module("torchvision").__version__,
    "flash_attn": flash_attn.__version__,
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
    "accelerate": accelerate.__version__,
    "cv2": cv2.__version__,
}

failed = False
for key, expected in EXPECTED.items():
    got = actual[key]
    if got != expected:
        failed = True
        print(f"VERSION_MISMATCH {key}: expected {expected}, got {got}", file=sys.stderr)

print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"flash_attn_2_cuda={flash_attn_2_cuda.__file__}")

if torch.version.cuda != "12.4":
    failed = True
    print(f"VERSION_MISMATCH torch.version.cuda: expected 12.4, got {torch.version.cuda}", file=sys.stderr)

if failed:
    raise SystemExit(1)
