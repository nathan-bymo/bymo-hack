"""Build a private self-contained Kaggle notebook from local experiment files."""

import argparse
import json
import runpy
from pathlib import Path


def cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--vq-size", type=int, choices=(256, 512), required=True)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--holdout-size", type=int, default=600)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    prompt = runpy.run_path(str(source.parent.parent / "training" / "common" / "prompt.py"))["SHARED_JSON_PROMPT"]
    files = {name: (source / name).read_text(encoding="utf-8")
             for name in ("adapter.py", "supervision.py", "train_projector.py")}
    for name in ("prompt.py", "splits.py"):
        files[f"common/{name}"] = (source.parent.parent / "training" / "common" / name).read_text(encoding="utf-8")
    files["system_prompt.txt"] = prompt
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite files in {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_check = '''import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPU:", names, flush=True)
if len(names) != 2 or any("T4" not in name.upper() for name in names):
    raise RuntimeError("Two Tesla T4 GPUs are required; stopping before training.")
'''
    token_check = '''import os
from kaggle_secrets import UserSecretsClient
try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    raise RuntimeError("Add and enable the HF_TOKEN Kaggle secret for this notebook.") from None
'''
    install = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate>=1.10,<2", "bitsandbytes>=0.46,<1", "pandas>=2.1", "transformers[vision] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"], check=True)
'''
    write_files = "from pathlib import Path\nroot = Path('/kaggle/working/vadapter')\nroot.mkdir(exist_ok=True)\n(root / 'common').mkdir(exist_ok=True)\n"
    for name, content in files.items():
        write_files += f"(root / {name!r}).write_text({content!r}, encoding='utf-8')\n"
    run = f'''import subprocess, sys
from pathlib import Path
data = Path("/kaggle/input/competitions/hackapertus-galaxus")
if not data.exists():
    data = Path("/kaggle/input/hackapertus-galaxus")
if not (data / "train.csv").exists():
    raise RuntimeError("Competition train.csv not mounted")
base = [sys.executable, str(root / "train_projector.py"), "--data-dir", str(data), "--system-prompt", str(root / "system_prompt.txt"), "--vq-size", "{args.vq_size}"]
print("Running 8-image smoke test before the full experiment", flush=True)
subprocess.run(base + ["--train-size", "8", "--holdout-size", "4", "--output-dir", "/kaggle/working/smoke-{args.vq_size}"], check=True)
print("Smoke test passed; starting the full experiment", flush=True)
subprocess.run(base + ["--train-size", "{args.train_size}", "--holdout-size", "{args.holdout_size}", "--output-dir", "/kaggle/working/run-{args.vq_size}"], check=True)
'''
    notebook = {"cells": [cell(gpu_check), cell(token_check), cell(install), cell(write_files), cell(run)],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4, "nbformat_minor": 5}
    name = f"apertus-vadapter-{args.vq_size}"
    (args.output_dir / "experiment.ipynb").write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"{args.owner}/{name}", "title": f"Apertus Vadapter {args.vq_size}",
                "code_file": "experiment.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
                "competition_sources": ["hackapertus-galaxus"], "dataset_sources": [], "kernel_sources": [], "model_sources": []}
    (args.output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared private {args.vq_size} experiment in {args.output_dir}")


if __name__ == "__main__":
    main()
