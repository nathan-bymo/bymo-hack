"""Non-training preflight for a private Kaggle T4 x2 notebook."""

import os
import shutil
import subprocess
from pathlib import Path

import torch

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPU devices:", names, flush=True)
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("nvidia-smi unavailable", flush=True)
if len(names) != 2 or any("T4" not in name.upper() for name in names):
    raise RuntimeError("This experiment requires two NVIDIA T4 GPUs")

for device in range(2):
    result = (torch.ones(128, device=f"cuda:{device}") * 2).sum().item()
    assert result == 256
print("Both GPUs passed a compute check.", flush=True)

data_path = Path("/kaggle/input/competitions/hackapertus-galaxus")
if not data_path.exists():
    data_path = Path("/kaggle/input/hackapertus-galaxus")
print("Competition data mounted:", data_path.exists(), flush=True)
print("Network configuration available:", bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")), flush=True)
if not data_path.exists():
    raise RuntimeError("Competition data is not mounted")
