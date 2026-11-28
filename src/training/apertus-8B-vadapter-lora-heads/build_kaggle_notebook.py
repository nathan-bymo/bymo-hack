"""Builds a private Kaggle notebook (GPU T4 x2) that runs predict.py: dev submission, final images, or a check.

Inference only, no training: on T4 (no bf16) Apertus runs in float16 with the MLP overflow guard of model.py,
the decoder on the first GPU and VQ, SigLIP and OCR on the second. The checkpoint comes from the private HF repo.

  python src/training/apertus-8B-vadapter-lora-heads/build_kaggle_notebook.py --owner <kaggle user> \
      --checkpoint final-abt/model/best --target check
  uvx kaggle kernels push -p data/kaggle/vlh-predict

Then, once, in the Kaggle editor: Add-ons > Secrets > HF_TOKEN (read access to the gated Apertus and to the
private repo), and Settings > Accelerator > GPU T4 x2. Targets:
  check  the first 300 images of prompt-test, scored: compare with the H200 run of the same checkpoint (fp16 vs bf16)
  dev    the same check, then dev.csv -> /kaggle/working/submission.csv, checked against sample_submission.csv
         (submitting stays manual)
  final  the final images, once the passphrase is out: set FINAL_PASSPHRASE in the settings cell
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE_FILES = [ROOT / "src" / "training" / "common" / name for name in ("__init__.py", "images.py", "prompt.py", "splits.py")] + \
             [HERE / name for name in ("data.py", "encoders.py", "model.py", "predict.py", "calibrate.py")]

SETTINGS = '''import os, sys, time, subprocess, shutil, json
from pathlib import Path

START = time.time()
CHECKPOINT = "@@CHECKPOINT@@"        # folder of the private HF repo
MODEL_REPO = "@@MODEL_REPO@@"
TARGET = "@@TARGET@@"                # "check", "dev" or "final"
FINAL_PASSPHRASE = ""                # published on Sat 28 Nov, 13:00 (Discord / Kaggle discussion)
BATCH = 4                            # the fp16 decoder takes about 12.7 of the 14.6 GB of a T4
TRANSFORMERS = "transformers[vision] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"
PINS = ["peft==0.22.0", "accelerate==1.16.0", "python-doctr==1.1.0", "py7zr"]

CODE = Path("/kaggle/working/code")
OUT = Path("/kaggle/working")
SCRATCH = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")
os.environ["HF_HOME"] = str(SCRATCH / "hf")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
FOLDER = CODE / "src" / "training" / "apertus-8B-vadapter-lora-heads"
FOLDER.mkdir(parents=True, exist_ok=True)
(CODE / "src" / "training" / "common").mkdir(parents=True, exist_ok=True)


def run(command):
    print(">", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               cwd=CODE)
    for line in process.stdout:
        if "warn" not in line.lower() and "it/s]" not in line:
            print(line, end="", flush=True)
    assert process.wait() == 0, f"failed: {command[:3]}"


def elapsed():
    return f"{(time.time() - START) / 60:.1f} min"'''

CHECKS = '''from kaggle_secrets import UserSecretsClient

try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as error:
    raise SystemExit(f"HF_TOKEN secret not available ({type(error).__name__}): Add-ons > Secrets") from None
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import torch
print("torch", torch.__version__, "GPUs", torch.cuda.device_count())
assert torch.cuda.device_count() == 2, "this notebook needs GPU T4 x2 (Settings > Accelerator)"
DATA_DIR = sorted(Path("/kaggle/input").glob("**/train.csv"), key=lambda p: len(p.parts))[0].parent
print("data:", DATA_DIR, sorted(p.name for p in DATA_DIR.iterdir()))'''

INSTALL = '''run([sys.executable, "-m", "pip", "install", "-q", TRANSFORMERS, *PINS])
print(subprocess.run([sys.executable, "-c", "import transformers, peft, doctr, torch; print(transformers.__version__, "
                      "peft.__version__, doctr.__version__, torch.__version__)"], capture_output=True, text=True).stdout)
print("install done after", elapsed())'''

DOWNLOAD = '''from huggingface_hub import snapshot_download

local = Path(snapshot_download(MODEL_REPO, allow_patterns=[f"{CHECKPOINT}/*", f"{CHECKPOINT}/lora/*"],
                               local_dir=SCRATCH / "checkpoint"))
CHECKPOINT_DIR = local / CHECKPOINT
print(json.loads((CHECKPOINT_DIR / "settings.json").read_text()))
sys.path[:0] = [str(FOLDER), str(CODE / "src" / "training")]
import encoders
encoders.model_dir()  # Apertus (gated) into HF_HOME, about 17 GB
print("download done after", elapsed(), "| disk free", shutil.disk_usage(SCRATCH).free // 2**30, "GB")'''

FINAL = '''def final_rows():
    """Extracts final_encrypted.bin with the passphrase and returns the csv of the final images."""
    import py7zr

    assert FINAL_PASSPHRASE, "set FINAL_PASSPHRASE in the settings cell"
    target = SCRATCH / "final"
    with py7zr.SevenZipFile(DATA_DIR / "final_encrypted.bin", mode="r", password=FINAL_PASSPHRASE) as archive:
        archive.extractall(path=target)
    candidates = [p for p in target.rglob("*.csv") if "image_path" in p.read_text().splitlines()[0]]
    assert len(candidates) == 1, f"expected one csv with image_path, found {candidates}"
    print("final csv:", candidates[0], "| images root:", candidates[0].parent)
    return candidates[0]'''

PREDICT = '''common = ["--checkpoint", CHECKPOINT_DIR, "--dtype", "float16", "--encoder-device", "cuda:1",
          "--batch-size", BATCH]
if TARGET in ("check", "dev"):  # float16 on T4 against bf16 on H200: the first 300 images of prompt-test
    run([sys.executable, FOLDER / "predict.py", *common, "--data-dir", DATA_DIR, "--part", "test", "--limit", 300,
         "--out", OUT / "check_test300.csv"])
if TARGET == "dev":
    run([sys.executable, FOLDER / "predict.py", *common, "--data-dir", DATA_DIR, "--csv", DATA_DIR / "dev.csv",
         "--out", OUT / "submission.csv"])
elif TARGET == "final":
    csv = final_rows()
    run([sys.executable, FOLDER / "predict.py", *common, "--data-dir", csv.parent, "--csv", csv,
         "--out", OUT / "submission_final.csv"])
print("prediction done after", elapsed())'''

VERIFY = '''import pandas as pd

if TARGET in ("dev", "final"):
    name = "submission.csv" if TARGET == "dev" else "submission_final.csv"
    submission = pd.read_csv(OUT / name, dtype=str, keep_default_na=False)
    columns = ["image_id", "subject", "background", "perspective", "individuals", "text_classification", "text_language"]
    assert list(submission.columns) == columns, submission.columns
    assert (submission[columns[1:]] != "").all().all(), "empty cells"
    assert not submission.image_id.duplicated().any()
    if TARGET == "dev":
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype=str, keep_default_na=False)
        assert set(sample.image_id) == set(submission.image_id), "ids differ from sample_submission.csv"
    print(name, "ok:", len(submission), "rows; submitting stays a manual step")
report = sorted(OUT.glob("*.report.json"))
for path in report:
    print(path.name, json.loads(path.read_text()))'''


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--owner", required=True, help="Kaggle user name")
    parser.add_argument("--checkpoint", default="phase1-ocr/model/best")
    parser.add_argument("--model-repo", default="nathan-monzani-bymo/galaxus-vlh-private")
    parser.add_argument("--target", default="check", choices=("check", "dev", "final"))
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "kaggle" / "vlh-predict")
    args = parser.parse_args()

    settings = (SETTINGS.replace("@@CHECKPOINT@@", args.checkpoint).replace("@@MODEL_REPO@@", args.model_repo)
                .replace("@@TARGET@@", args.target))
    cells = [markdown("# Apertus 8B + vision adapter + heads: inference on GPU T4 x2\n\nBuilt by "
                      "`src/training/apertus-8B-vadapter-lora-heads/build_kaggle_notebook.py`: edit the repository, not here."),
             code(settings), code(CHECKS), code(INSTALL)]
    for path in CODE_FILES:
        target = f"/kaggle/working/code/{path.relative_to(ROOT)}"
        source = path.read_text() if path.exists() else ""
        if not source.strip():  # Jupyter rejects an empty %%writefile cell
            source = "# package marker\n"
        cells.append(code(f"%%writefile {target}\n{source}"))
    cells += [code(DOWNLOAD), code(FINAL), code(PREDICT), code(VERIFY)]
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"},
                             "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vlh-predict.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    metadata = {"id": f"{args.owner}/apertus-vlh-predict", "title": "Apertus VLH Predict", "code_file": "vlh-predict.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "competition_sources": ["hackapertus-galaxus"],
                "dataset_sources": [], "kernel_sources": [], "model_sources": []}
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"{args.out}: {len(cells)} cells, target {args.target}, checkpoint {args.checkpoint}")


if __name__ == "__main__":
    main()
