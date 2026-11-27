"""Builds lora_train.ipynb, the Kaggle notebook (GPU T4 x2) that trains the Apertus v1.5 8B LoRA.

The notebook carries the code of this folder and of src/common as %%writefile cells, so it runs on Kaggle without
the repository, and the repository stays the only source of the code. Rebuild after any change:

    python src/apertus-8B-benchmark/lora/build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
CODE_FILES = [
    ("common/images.py", SRC / "common" / "images.py"),
    ("common/prompt.py", SRC / "common" / "prompt.py"),
    ("common/splits.py", SRC / "common" / "splits.py"),
    ("apertus_lora.py", HERE / "apertus_lora.py"),
    ("encode.py", HERE / "encode.py"),
    ("train.py", HERE / "train.py"),
]

INTRO = """# Apertus v1.5 8B: QLoRA training on Kaggle (GPU T4 x2)

Team Bymo, Hack Apertus x Galaxus. Built by `src/apertus-8B-benchmark/lora/build_notebook.py`: edit the code in the
repository, not here.

1. Checks the Hugging Face secret and the two GPUs, installs the Apertus fork of transformers.
2. Downloads Apertus v1.5 8B (gated: Kaggle secret `HF_TOKEN`).
3. Computes the VQ codes of the training images once, on both GPUs.
4. A short preflight (a few steps on 64 rows), then the QLoRA training on both GPUs (fp16, nf4).
5. Writes the LoRA adapters to `/kaggle/working/run/adapters/` (snapshots for checkpoint averaging, and `final`).

Phase 1 of `src/AGENTS.md`: training on `common.splits.training_rows`; the `coverage` set is never used here.
The run fits itself into `SESSION_HOURS`, the GPU quota left, by measuring its own step time."""

SETTINGS = """import os, sys, time, subprocess, shutil
from pathlib import Path

START = time.time()
SESSION_HOURS = 5.4        # whole session, install to saved adapters; the weekly GPU quota left must cover it
SIDE = 256                 # image area about SIDE x SIDE pixels: (SIDE/16)^2 VQ codes
PART = "training"          # "training" = phase 1 (src/AGENTS.md), "all" = phase 2
EPOCHS = 3                 # upper bound; the deadline may stop earlier
PREFLIGHT = True           # a few steps on 64 rows before the long encoding and training

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
TRANSFORMERS = "transformers[vision] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"
PINS = ["peft==0.22.0", "accelerate==1.16.0", "bitsandbytes"]

CODE = Path("/kaggle/working/code")
RUN = Path("/kaggle/working/run")
SCRATCH = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")
MODEL_DIR = SCRATCH / "apertus-v1.5-8b"
os.environ["HF_HOME"] = str(SCRATCH / "hf")
for folder in (CODE / "common", RUN):
    folder.mkdir(parents=True, exist_ok=True)


def run(command, env=None):
    \"\"\"Runs a command and streams its output into the notebook log.\"\"\"
    print(">", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               cwd=CODE, env={**os.environ, **(env or {})})
    for line in process.stdout:
        if "warn" not in line.lower():
            print(line, end="", flush=True)
    assert process.wait() == 0, f"failed: {command[:3]}"


def elapsed():
    return f"{(time.time() - START) / 60:.0f} min"
"""

CHECKS = """# The Hugging Face token comes from the Kaggle secret HF_TOKEN (Add-ons > Secrets), never from the notebook text.
from kaggle_secrets import UserSecretsClient

try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN secret: attached")
except Exception as error:
    raise SystemExit(f"HF_TOKEN secret not available ({type(error).__name__}): attach it in Add-ons > Secrets") from None

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "GPUs", torch.cuda.device_count())
assert torch.cuda.device_count() == 2, "this notebook needs GPU T4 x2"
DATA_DIR = sorted(Path("/kaggle/input").glob("**/train.csv"), key=lambda p: len(p.parts))[0].parent
print("data:", DATA_DIR, "| scratch disk free:", shutil.disk_usage(SCRATCH).free // 2**30, "GB")"""

INSTALL = """run([sys.executable, "-m", "pip", "install", "-q", TRANSFORMERS, *PINS])
versions = subprocess.run([sys.executable, "-c", "import transformers, peft, accelerate, bitsandbytes, torch; "
                           "print({m.__name__: m.__version__ for m in (transformers, peft, accelerate, bitsandbytes, torch)})"],
                          capture_output=True, text=True)
print(versions.stdout or versions.stderr[-2000:])
(RUN / "versions.txt").write_text(versions.stdout)
print("install done after", elapsed())"""

DOWNLOAD = """from huggingface_hub import snapshot_download

snapshot_download(MODEL_ID, local_dir=MODEL_DIR, token=os.environ["HF_TOKEN"])
print(sorted(p.name for p in MODEL_DIR.iterdir()))
print("download done after", elapsed(), "| disk free:", shutil.disk_usage(SCRATCH).free // 2**30, "GB")"""

ENCODE_FN = """def encode(limit=0):
    \"\"\"VQ codes of the training images and mirrors, one process per GPU; resumes from what is already there.\"\"\"
    processes = []
    for shard in (0, 1):
        command = [sys.executable, "encode.py", "--model-dir", MODEL_DIR, "--data-dir", DATA_DIR, "--side", SIDE,
                   "--part", PART, "--shard", shard, "--num-shards", 2, "--out", RUN / "codes"]
        if limit:
            command += ["--limit", limit]
        log = open(RUN / f"encode_{shard}.log", "a")
        processes.append(subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, cwd=CODE,
                                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}))
    codes = [p.wait() for p in processes]
    for shard in (0, 1):
        print("".join(line for line in open(RUN / f"encode_{shard}.log") if "warn" not in line.lower())[-1500:])
    assert codes == [0, 0], f"encoding failed: {codes}"


def train(*extra):
    run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", 2, "train.py", "--model-dir", MODEL_DIR, "--data-dir", DATA_DIR,
         "--codes", RUN / "codes", "--side", SIDE, "--part", PART, "--epochs", EPOCHS, *extra],
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"})"""

PREFLIGHT_CELL = """# A few steps on 64 rows: catches errors (memory, fp16, DDP) before the long encoding.
if PREFLIGHT:
    encode(limit=64)
    train("--limit", 64, "--max-steps", 12, "--out", SCRATCH / "preflight",
          "--deadline", time.time() + 1800)
    print("preflight done after", elapsed())"""

ENCODE_CELL = """encode()
print("encoding done after", elapsed())"""

TRAIN_CELL = """deadline = START + SESSION_HOURS * 3600
print(f"{(deadline - time.time()) / 3600:.2f} h left for training")
train("--out", RUN, "--deadline", deadline)
print("training done after", elapsed())"""

SUMMARY_CELL = """import json

summary = json.loads((RUN / "train_summary.json").read_text())
print({k: summary[k] for k in ("part", "side", "rows", "examples_per_epoch", "steps", "epochs", "minutes",
                               "mlp_scale", "peak_memory_gb")})
for entry in summary["log_history"][-5:]:
    print(entry)
shutil.rmtree(RUN / "trainer", ignore_errors=True)  # optimizer states: not needed once the adapters are saved
for path in sorted(RUN.rglob("*")):
    if path.is_file():
        print(f"{path.stat().st_size / 2**20:9.1f} MB  {path.relative_to(RUN)}")"""


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main():
    cells = [markdown(INTRO), markdown("## Settings"), code(SETTINGS), markdown("## Checks"), code(CHECKS),
             markdown("## Install"), code(INSTALL), markdown("## Code\n\nThe files of the repository, written to "
                                                              "`/kaggle/working/code`.")]
    for name, path in CODE_FILES:
        cells.append(code(f"%%writefile /kaggle/working/code/{name}\n" + path.read_text()))
    cells += [markdown("## Model"), code(DOWNLOAD), markdown("## Encoding and training"), code(ENCODE_FN),
              code(PREFLIGHT_CELL), code(ENCODE_CELL), code(TRAIN_CELL), markdown("## Result"), code(SUMMARY_CELL)]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "lora_train.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
