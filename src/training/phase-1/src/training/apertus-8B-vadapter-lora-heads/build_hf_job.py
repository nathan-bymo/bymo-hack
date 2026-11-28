"""Builds data/hf_jobs/hf_job_vlh.py, a self-contained Hugging Face Jobs script (H200) for this model.

The generated file carries the code of src/ and the SHA-256 of train.csv; it holds no image ids or labels, but it
is written under data/ (git-ignored) like the other jobs. The job downloads the competition data and Apertus onto
its own disk and uploads checkpoints, predictions and logs to a PRIVATE repository. The data comes from a PRIVATE
Hugging Face dataset (DATA_REPO: only HF_TOKEN is needed) or else from Kaggle. To fill DATA_REPO once:

  python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py --upload-data <user>/galaxus-data-private

  PHASE=1 (default)  encode training_rows -> smoke test -> train (internal validation picks the checkpoint)
                     -> end-to-end prediction of the coverage set, scored with metric.py -> optional dev predictions
  PHASE=2            encode all labeled images -> train EPOCHS epochs, no validation -> dev predictions

Environment of the job (-e NAME=value):
  UPLOAD_REPO      required, e.g. <user>/galaxus-vlh-private (created private if missing)
  DATA_REPO        private HF dataset with train.csv, dev.csv, metric.py, train_images.tar, dev_images.tar
  TRAIN_ARGS       extra arguments for train.py, e.g. "--epochs 3 --class-alpha effective"
  EPOCHS           phase 2 only: the epochs chosen in phase 1
  SIDE             VQ image side, default 256
  SIGLIP           another frozen image encoder, e.g. google/siglip2-so400m-patch16-512 (with SOFT_SIDE, default 14:
                   soft tokens per side); its projector starts from zero when INIT_FROM used the default encoder
  REUSE_CACHE      with SIGLIP: cache name in UPLOAD_REPO/cache/ (e.g. all_but_test_256) whose VQ codes and OCR
                   texts are reused, so only the new encoder runs
  INIT_PROJECTOR   optional "<repo>:<file>" of a projector of apertus-vadapter-benchmark, read with HF_TOKEN
  DEV_ONLY         1 with PREDICT_FROM: only dev.csv -> predictions/dev_submission.csv (no training images)
  PREDICT_DEV      1 to also predict dev.csv (a file in the private repo; nothing is submitted)
  NO_CONTEXT_CHECK 1 to also predict coverage without product context (about 3 more GPU minutes)
  PREDICT_TTA      1 to also predict coverage with the mirror test-time augmentation (predict.py --tta-flip)
  Budget: TRAIN_ARGS="--epochs 2 --max-minutes 45" fits the schedule into 45 training minutes; the job timeout is
  the hard cap on the cost (H200: 5 $/h on 27 November 2026, `hf jobs hardware`).
  FINAL            1 with PREDICT_FROM: decrypts the final images and predicts them -> predictions/final.csv. Needs
                   FINAL_PASSPHRASE (given at launch, never written to a file). FINAL_ARCHIVE: path in UPLOAD_REPO of a
                   test archive for a rehearsal; default final_encrypted.bin from Kaggle, checked against its SHA-256.
                   The images stay on the job's disk: only the predictions are uploaded.
  CHECK_ONLY       1 to only check secrets, data and model access and the real tokenizer (a CPU is enough)
  OCR              1: OCR text (docTR) of every image in the prompt, read once into the cache
  TARGETED_CONTINUATION  1: use prompt-test for early stopping and scale only the selected rare text classes
  PART             training (default of phase 1), all_but_test (everything except prompt-test, scored on
                   prompt-test) or all (phase 2)
  INIT_FROM        a checkpoint folder of UPLOAD_REPO (e.g. phase1/model/best): continue its training
  SYNTHETIC        path in UPLOAD_REPO of a zip holding synthetic/synthetic.csv and synthetic/images/ (made by
                   src/scripts/make_synthetic_text.py): added to the training rows only
  RUN_NAME         folder of this run in UPLOAD_REPO (default phase1 / phase2): never overwrite a model to keep
  XIELU            1 to build the CUDA xIELU kernel (needs a -devel image: see --upload-job below); kept only if it
                   matches the Python version

  With XIELU=1 the job runs in pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel (it has nvcc), from the private repo:
  python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py --upload-job <user>/galaxus-vlh-private
  hf jobs run --flavor h200 --timeout 90m --secrets HF_TOKEN -d -e XIELU=1 -e DATA_REPO=... -e UPLOAD_REPO=... \\
      pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel bash -c "$(cat data/hf_jobs/launch_devel.sh)"

  python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py
  export HF_TOKEN=$(cat ~/.cache/huggingface/token) KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
  # or, with the classic ~/.kaggle/credentials.json: pass --secrets KAGGLE_USERNAME --secrets KAGGLE_KEY instead
  hf jobs uv run --flavor cpu-upgrade --timeout 20m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \\
      -e CHECK_ONLY=1 data/hf_jobs/hf_job_vlh.py
  hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \\
      -e UPLOAD_REPO=<user>/galaxus-vlh-private -e PREDICT_DEV=1 data/hf_jobs/hf_job_vlh.py

The timeout caps the cost: the job is stopped when it expires (the best checkpoint is uploaded as soon as it is saved).
"""

import base64
import hashlib
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = "src/training/apertus-8B-vadapter-lora-heads"
FILES = ["src/training/common/images.py", "src/training/common/prompt.py", "src/training/common/splits.py", "src/scripts/eval_prompt.py",
         *(f"{FOLDER}/{name}" for name in ("data.py", "encoders.py", "model.py", "encode.py", "train.py", "predict.py",
                                            "calibrate.py"))]

TEMPLATE = '''# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "transformers[vision] @ git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4",
#   "peft==0.22.0",
#   "accelerate==1.16.0",
#   "safetensors",
#   "huggingface_hub",
#   "pandas>=2.1",
#   "pillow",
#   "python-doctr==1.1.0",
#   "kaggle==2.2.5",
#   "six",
# ]
# ///
"""Generated by @@FOLDER@@/build_hf_job.py: do not edit, do not commit."""

import base64
import hashlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

START = time.time()
WORK = Path(os.environ.get("WORK", "/tmp/work"))
DATA, RUN = WORK / "data", WORK / "data" / "run"
FOLDER = "@@FOLDER@@"
PHASE = os.environ.get("PHASE", "1")
SIDE = os.environ.get("SIDE", "256")
SIGLIP = os.environ.get("SIGLIP", "")  # another image encoder; empty: the default of encoders.py
SOFT_SIDE = os.environ.get("SOFT_SIDE", "14")
UPLOAD_REPO = os.environ.get("UPLOAD_REPO", "")
DATA_REPO = os.environ.get("DATA_REPO", "")  # private HF dataset with the competition data (see the builder)
PREFIX = os.environ.get("RUN_NAME", f"phase{PHASE}")
PREDICT_FROM = os.environ.get("PREDICT_FROM", "")  # e.g. phase1/model/best in UPLOAD_REPO: predictions only
INIT_FROM = os.environ.get("INIT_FROM", "")  # e.g. phase1/model/best in UPLOAD_REPO: continue its training
OCR = bool(os.environ.get("OCR"))  # OCR text of the image in the prompt
SYNTHETIC = os.environ.get("SYNTHETIC", "")  # zip in UPLOAD_REPO with extra training images
TRAIN_SHA256 = "@@TRAIN_SHA256@@"
FINAL_SHA256 = "4b37a260c9fcc0325e8ee43ec5db1519dea7ff6ced0d918fb691c1c563506465"  # the Kaggle data page
BUNDLE = "@@BUNDLE@@"


def log(*parts):
    print(f"[{(time.time() - START) / 60:6.1f} min]", *parts, flush=True)


def upload(folder, path_in_repo, patterns):
    if not UPLOAD_REPO:
        return
    from huggingface_hub import HfApi

    HfApi().upload_folder(repo_id=UPLOAD_REPO, folder_path=str(folder), path_in_repo=f"{PREFIX}/{path_in_repo}",
                          allow_patterns=patterns, ignore_patterns=["README.md"], commit_message=f"{PREFIX}: {path_in_repo}")
    log("uploaded", path_in_repo, "to the private repository")


def run(command, log_name):
    """Runs a step from WORK, streams its output and keeps it in RUN/log_name."""
    log(">", " ".join(map(str, command)))
    with open(RUN / log_name, "a") as keep:
        process = subprocess.Popen([sys.executable, *map(str, command)], cwd=WORK, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            keep.write(line)
            print(line, end="", flush=True)
    if process.wait() != 0:
        upload(RUN, "logs", ["*.log"])
        raise SystemExit(f"step failed, see {log_name}")


def unpack():
    RUN.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as archive:
        archive.extractall(WORK, filter="data")
    sys.path[:0] = [str(WORK / FOLDER), str(WORK / "src" / "training")]


def download_data():
    """From the PRIVATE HF dataset DATA_REPO (only HF_TOKEN needed), or else from Kaggle."""
    if DATA_REPO:
        from huggingface_hub import HfApi, hf_hub_download

        assert HfApi().repo_info(DATA_REPO, repo_type="dataset").private, f"{DATA_REPO} is not private"
        names = ("train.csv", "dev.csv", "metric.py", "train_images.tar", "dev_images.tar")
        if os.environ.get("DEV_ONLY"):  # dev predictions only: the training images are not needed
            names = tuple(n for n in names if n != "train_images.tar")
        for name in names:
            path = Path(hf_hub_download(DATA_REPO, name, repo_type="dataset", local_dir=DATA))
            if name.endswith(".tar"):
                with tarfile.open(path) as archive:
                    archive.extractall(DATA, filter="data")
                path.unlink()
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("hackapertus-galaxus", path=str(DATA), quiet=True, unzip=True)
        (DATA / "final_encrypted.bin").unlink(missing_ok=True)  # the final images are never used here
    assert hashlib.sha256((DATA / "train.csv").read_bytes()).hexdigest() == TRAIN_SHA256, "train.csv changed"
    log("competition data downloaded")


def download_synthetic():
    """The extra training images, unpacked next to the competition data; returns the arguments of encode and train."""
    if not SYNTHETIC:
        return []
    import zipfile

    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(UPLOAD_REPO, SYNTHETIC, local_dir=WORK / "synthetic_zip")
    with zipfile.ZipFile(archive) as content:
        content.extractall(DATA)
    table = DATA / "synthetic" / "synthetic.csv"
    assert table.exists(), f"{SYNTHETIC} has no synthetic/synthetic.csv"
    log("synthetic training data unpacked")
    return ["--extra-csv", table]


def final_data():
    """The folder with final.csv and final_images/, decrypted on the job's disk. The passphrase is never printed."""
    passphrase = os.environ.get("FINAL_PASSPHRASE", "")
    assert passphrase, "FINAL=1 needs FINAL_PASSPHRASE"
    store = DATA / "final_download"
    store.mkdir(parents=True, exist_ok=True)
    test_archive = os.environ.get("FINAL_ARCHIVE", "")
    if test_archive:
        from huggingface_hub import hf_hub_download

        archive = Path(hf_hub_download(UPLOAD_REPO, test_archive, local_dir=store))
        log("REHEARSAL with the test archive", test_archive)
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file("hackapertus-galaxus", "final_encrypted.bin", path=str(store), quiet=True)
        archive = store / "final_encrypted.bin"
        if not archive.exists():  # Kaggle sometimes wraps a single file in a zip
            import zipfile

            with zipfile.ZipFile(store / "final_encrypted.bin.zip") as wrapper:
                wrapper.extractall(store)
        digest = hashlib.sha256()
        with open(archive, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        assert digest.hexdigest() == FINAL_SHA256, "final_encrypted.bin does not match the published SHA-256"
        log("final_encrypted.bin downloaded, SHA-256 matches")
    if not shutil.which("7z"):
        subprocess.run("apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null", shell=True, check=True)
    target = DATA / "final"
    result = subprocess.run(["7z", "x", f"-p{passphrase}", f"-o{target}", "-y", str(archive)], capture_output=True,
                            text=True)
    if result.returncode:  # the command line holds the passphrase: only the tool's own message is shown
        raise SystemExit("7z could not open the archive: " + (result.stderr or result.stdout)[-300:].replace(passphrase, "***"))
    tables = sorted(target.rglob("final.csv"))
    assert tables, "the archive holds no final.csv"
    log("final images decrypted")
    return tables[0].parent


def predict_final():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(3) as pool:  # the three downloads together: every minute counts after 13:00
        folder = pool.submit(final_data)
        checkpoint = pool.submit(fetch_checkpoint, PREDICT_FROM)
        pool.submit(download_model).result()
        folder, checkpoint = folder.result(), checkpoint.result()
    outputs = RUN / "predictions"
    run([f"{FOLDER}/predict.py", "--checkpoint", checkpoint, "--data-dir", folder, "--csv", folder / "final.csv",
         "--out", outputs / "final.csv"], "predict_final.log")
    upload(outputs, "predictions", ["final*"])
    upload(RUN, "logs", ["*.log"])
    log("ALL DONE")


def download_model():
    import encoders
    from huggingface_hub import snapshot_download

    encoders.model_dir()
    snapshot_download(SIGLIP or encoders.SIGLIP_ID)
    log("models downloaded")


XIELU_COMMIT = "2a55f6b9efa64954bf173e63297a2b2f99741b69"
XIELU_CHECK = """
import time, torch
from transformers.activations import XIELUActivation
act = XIELUActivation().cuda()
assert act._xielu_cuda_obj is not None, "the CUDA kernel did not load"
x = (torch.randn(8, 450, 4096, device="cuda", dtype=torch.bfloat16) * 3).requires_grad_()
fast = act(x); (g_fast,) = torch.autograd.grad(fast.float().sum(), x)
slow = act._xielu_python(x); (g_slow,) = torch.autograd.grad(slow.float().sum(), x)
forward = ((fast.float() - slow.float()).abs().max() / slow.float().abs().max()).item()
backward = ((g_fast.float() - g_slow.float()).abs().max() / g_slow.float().abs().max()).item()
def timed(function):
    torch.cuda.synchronize(); start = time.time()
    for _ in range(50):
        function(x)
    torch.cuda.synchronize(); return (time.time() - start) / 50 * 1000
with torch.no_grad():
    ms_fast, ms_slow = timed(act), timed(act._xielu_python)
print(f"XIELU check: max relative error forward {forward:.1e}, backward {backward:.1e}; "
      f"{ms_fast:.2f} ms CUDA vs {ms_slow:.2f} ms Python per call")
assert forward < 2e-2 and backward < 2e-2, "the CUDA kernel differs from the Python xIELU"
"""


def install_xielu():
    """XIELU=1: builds the CUDA xIELU kernel (needs nvcc, e.g. a pytorch/pytorch:*-devel image) and keeps it only
    if it matches the Python xIELU, forward and backward. On any failure the job goes on with the Python xIELU."""
    if not os.environ.get("XIELU"):
        return
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not (Path(cuda) / "bin" / "nvcc").exists():
        log(f"XIELU: no nvcc in {cuda} (use a -devel image): Python xIELU")
        return
    env = {**os.environ, "CUDA_HOME": cuda}
    uv = ["uv", "pip", "install", "--python", sys.executable, "-q"]
    try:
        subprocess.run([*uv, "cmake>=3.30", "setuptools", "wheel", "nvtx"], check=True, env=env)
        subprocess.run([*uv, "--no-build-isolation", "--no-deps",
                        f"xielu @ https://github.com/nickjbrowning/XIELU/archive/{XIELU_COMMIT}.tar.gz"],
                       check=True, env=env)
        check = subprocess.run([sys.executable, "-c", XIELU_CHECK], capture_output=True, text=True, env=env)
        log((check.stdout + check.stderr[-1500:]).strip())
        if check.returncode:
            raise RuntimeError("numeric check failed")
        log("XIELU: CUDA kernel in use")
    except Exception as error:
        log(f"XIELU: not used ({error}); Python xIELU")
        subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, "xielu"], env=env)


def init_projector():
    value = os.environ.get("INIT_PROJECTOR", "")
    if not value:
        return []
    from huggingface_hub import hf_hub_download

    repo, file = value.split(":", 1)
    return ["--init-projector", hf_hub_download(repo, file)]


def check():
    """Everything that can fail before the GPU: secrets, data access, gated model, real tokenizer and template."""
    import encoders
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if DATA_REPO:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(DATA_REPO, repo_type="dataset", files_metadata=True)
        assert info.private, f"{DATA_REPO} is not private"
        files = {f.rfilename for f in info.siblings}
        assert {"train.csv", "dev.csv", "metric.py", "train_images.tar", "dev_images.tar"} <= files, files
        log("private data repository ok:", sorted(files))
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log("kaggle access ok:", len(api.competition_list_files("hackapertus-galaxus").files) > 0)
    directory = encoders.model_dir()
    config = AutoConfig.from_pretrained(directory)
    builder = encoders.PromptBuilder(AutoTokenizer.from_pretrained(directory), config.image_token_offset)
    ids = builder.ids({"product_title": "Test"}, (2, 3), torch.arange(6, dtype=torch.int32))
    assert ids.count(encoders.PREFIX_SLOT) == encoders.POOLED_SIDE ** 2
    assert sum(config.image_token_offset <= i < config.image_token_offset + 131072 for i in ids) == 6
    log("gated model and tokenizer ok:", len(ids), "tokens for a 2x3 image,", config.text_config.num_hidden_layers,
        "layers")
    init_projector()
    if UPLOAD_REPO:
        from huggingface_hub import HfApi

        HfApi().create_repo(UPLOAD_REPO, private=True, exist_ok=True)
    log("CHECK OK")


def ensure_opencv():
    """docTR imports OpenCV, which needs libGL; slim images lack it (the job runs as root)."""
    if subprocess.run([sys.executable, "-c", "import cv2"], capture_output=True).returncode:
        log("installing libGL for OpenCV")
        subprocess.run("apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null",
                       shell=True, check=True)


def fetch_checkpoint(path):
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(UPLOAD_REPO, allow_patterns=[f"{path}/*", f"{path}/lora/*"],
                                    local_dir=WORK / "checkpoint")) / path
    assert (folder / "settings.json").exists(), f"no checkpoint at {path} in {UPLOAD_REPO}"
    return folder


def fetch_cache(cache, name):
    """Cache of an earlier job (shards, OCR) from UPLOAD_REPO/cache/<name>: encode.py then only adds what is missing."""
    import shutil

    from huggingface_hub import HfApi, snapshot_download

    if not any(f.startswith(f"cache/{name}/") for f in HfApi().list_repo_files(UPLOAD_REPO)):
        return
    local = Path(snapshot_download(UPLOAD_REPO, allow_patterns=[f"cache/{name}/*"], local_dir=WORK / "repo"))
    cache.mkdir(parents=True, exist_ok=True)
    for item in (local / "cache" / name).iterdir():
        shutil.move(str(item), cache / item.name)
    log(f"cache {name} taken from the private repository")


def store_cache(cache, name):
    from huggingface_hub import HfApi

    HfApi().upload_folder(repo_id=UPLOAD_REPO, folder_path=str(cache), path_in_repo=f"cache/{name}",
                          allow_patterns=["shard_*.pt", "*.json"], commit_message=f"cache {name}")
    log(f"cache {name} stored in the private repository")


def predict_only():
    """A saved checkpoint of UPLOAD_REPO on the internal validation (for calibrate.py) and on coverage."""
    folder = fetch_checkpoint(PREDICT_FROM)
    predict = [f"{FOLDER}/predict.py", "--checkpoint", folder]
    outputs = RUN / "predictions"
    if os.environ.get("DEV_ONLY"):  # dev.csv -> dev_submission.csv only (nothing is submitted)
        run([*predict, "--csv", DATA / "dev.csv", "--out", outputs / "dev_submission.csv"], "predict_dev.log")
        upload(outputs, "predictions", ["dev_submission*"])
        upload(RUN, "logs", ["*.log"])
        log("ALL DONE")
        return
    run([*predict, "--part", "validation", "--out", outputs / "validation.csv"], "predict_validation.log")
    run([*predict, "--part", "coverage", "--out", outputs / "coverage.csv"], "predict_coverage.log")
    upload(outputs, "predictions", ["*.csv", "*.json", "*.pt"])  # saved before the optional steps
    scored = [outputs / "coverage.csv"]
    if os.environ.get("PREDICT_TTA"):
        run([*predict, "--part", "coverage", "--tta-flip", "--out", outputs / "coverage_tta.csv"],
            "predict_coverage_tta.log")
        scored.append(outputs / "coverage_tta.csv")
    upload(outputs, "predictions", ["*.csv", "*.json", "*.pt"])
    run(["src/scripts/eval_prompt.py", *scored, "--split", "coverage", "--classes", "--bootstrap", "200"],
        "scores.log")
    if os.environ.get("PREDICT_DEV"):
        run([*predict, "--csv", DATA / "dev.csv", "--out", outputs / "dev_submission.csv"], "predict_dev.log")
    upload(outputs, "predictions", ["*.csv", "*.json", "*.pt"])
    upload(RUN, "logs", ["*.log"])
    log("ALL DONE")


def main():
    unpack()
    kaggle = os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    assert os.environ.get("HF_TOKEN"), "secret HF_TOKEN"
    assert DATA_REPO or kaggle, "set DATA_REPO (private HF dataset) or a Kaggle secret"
    if os.environ.get("CHECK_ONLY"):
        return check()
    import torch

    assert torch.cuda.is_available(), "no GPU visible"
    install_xielu()
    ensure_opencv()
    assert PHASE in ("1", "2"), "PHASE is 1 or 2"
    assert PHASE == "1" or os.environ.get("EPOCHS"), "phase 2 needs EPOCHS, chosen in phase 1"
    log("phase", PHASE, "| torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
    assert UPLOAD_REPO, "set UPLOAD_REPO: without it the results are lost when the job ends"
    if os.environ.get("FINAL"):
        assert PREDICT_FROM, "FINAL=1 needs PREDICT_FROM, the checkpoint to use"
        return predict_final()
    from huggingface_hub import HfApi

    HfApi().create_repo(UPLOAD_REPO, private=True, exist_ok=True)
    assert HfApi().repo_info(UPLOAD_REPO).private, f"{UPLOAD_REPO} is not private"
    errors = []

    def guarded(step):
        try:
            step()
        except Exception as error:
            errors.append(f"{step.__name__}: {error}")

    threads = [threading.Thread(target=guarded, args=(step,)) for step in (download_data, download_model)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors

    if PREDICT_FROM:
        return predict_only()
    part = os.environ.get("PART") or ("training" if PHASE == "1" else "all")
    assert part in ("training", "all_but_test", "all"), f"unknown PART {part}"
    targeted = os.environ.get("TARGETED_CONTINUATION") == "1"
    if targeted and (PHASE != "1" or part != "all_but_test" or not INIT_FROM or not OCR or SIDE != "256"):
        raise ValueError("TARGETED_CONTINUATION requires phase 1, all_but_test, INIT_FROM, OCR and SIDE=256")
    targeted_flags = ["--prompt-test-validation", "--rare-text-only"] if targeted else []
    val_fraction = "0" if part == "all" else "0.1"
    suffix = f"_{SIGLIP.split('/')[-1]}_s{SOFT_SIDE}" if SIGLIP else ""
    name = f"{part}_{SIDE}{suffix}"
    cache = DATA / "cache" / name
    fetch_cache(cache, name)
    for other in (f"all_but_test_{SIDE}{suffix}", f"training_{SIDE}{suffix}"):  # they cover most of the images
        if part != "training" and not cache.exists():
            fetch_cache(cache, other)
    encoder_flags = []
    if SIGLIP:
        encoder_flags = ["--siglip-id", SIGLIP, "--pooled-side", SOFT_SIDE]
        if os.environ.get("REUSE_CACHE"):
            reuse = DATA / "cache" / "reuse"
            fetch_cache(reuse, os.environ["REUSE_CACHE"])
            assert reuse.exists(), "REUSE_CACHE is not in UPLOAD_REPO"
            encoder_flags += ["--reuse-cache", reuse]
    synthetic = download_synthetic()
    run([f"{FOLDER}/encode.py", "--part", part, "--side", SIDE, "--out", cache, "--val-fraction", val_fraction,
         *targeted_flags, *synthetic, *encoder_flags, *(["--ocr"] if OCR else [])], "encode.log")
    if not targeted:
        store_cache(cache, name)
    extra = shlex.split(os.environ.get("TRAIN_ARGS", "")) + init_projector() + synthetic
    if targeted:
        extra += [*targeted_flags, "--early-stop", "--spatial", "--attention-pool"]
    if OCR:
        extra.append("--ocr")
    if INIT_FROM:
        extra += ["--init-checkpoint", fetch_checkpoint(INIT_FROM)]
    if PHASE == "2":
        extra = ["--part", "all", "--val-fraction", "0", "--epochs", os.environ["EPOCHS"], *extra]
    elif part != "training":
        extra = ["--part", part, "--val-fraction", val_fraction, *extra]
    train = [f"{FOLDER}/train.py", "--cache", cache, *extra]
    run([*train, "--out", DATA / "smoke", "--limit", "96", "--max-steps", "3", "--evals-per-epoch", "1"], "smoke.log")
    out = RUN / "model"
    run([*train, "--out", out, "--upload-repo", UPLOAD_REPO, "--upload-prefix", f"{PREFIX}/model"], "train.log")
    checkpoint = out / ("best" if PHASE == "1" else "last")
    upload(RUN, "logs", ["*.log"])

    predict = [f"{FOLDER}/predict.py", "--checkpoint", checkpoint]
    outputs = RUN / "predictions"
    if PHASE == "1" and part == "all_but_test":  # coverage was partly trained on: score on prompt-test only
        if os.environ.get("PREDICT_VALIDATION"):
            run([*predict, "--part", "validation", "--train-part", part, "--out", outputs / "validation.csv"],
                "predict_validation.log")
        run([*predict, "--part", "test", "--out", outputs / "test.csv"], "predict_test.log")
        upload(outputs, "predictions", ["*.csv", "*.json", "*.pt"])
        run(["src/scripts/eval_prompt.py", outputs / "test.csv", "--split", "test", "--classes", "--bootstrap", "200"],
            "scores.log")
    elif PHASE == "1":
        run([*predict, "--part", "validation", "--out", outputs / "validation.csv"], "predict_validation.log")
        run([*predict, "--part", "coverage", "--out", outputs / "coverage.csv"], "predict_coverage.log")
        scored = [outputs / "coverage.csv"]
        if os.environ.get("PREDICT_TTA"):
            run([*predict, "--part", "coverage", "--tta-flip", "--out", outputs / "coverage_tta.csv"],
                "predict_coverage_tta.log")
            scored.append(outputs / "coverage_tta.csv")
        if os.environ.get("NO_CONTEXT_CHECK"):
            run([*predict, "--part", "coverage", "--no-context", "--out", outputs / "coverage_no_context.csv"],
                "predict_coverage_no_context.log")
            scored.append(outputs / "coverage_no_context.csv")
        run(["src/scripts/eval_prompt.py", *scored, "--split", "coverage", "--classes", "--bootstrap", "200"],
            "scores.log")
    if PHASE == "2" or os.environ.get("PREDICT_DEV"):
        run([*predict, "--csv", DATA / "dev.csv", "--out", outputs / "dev_submission.csv"], "predict_dev.log")
    upload(outputs, "predictions", ["*.csv", "*.json", "*.pt"])
    upload(RUN, "logs", ["*.log"])
    log("ALL DONE")


if __name__ == "__main__":
    main()
'''


def upload_data(repo):
    """Copies the competition data (never the final images) to a PRIVATE HF dataset, as two tar archives."""
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="dataset").private, f"{repo} is not private: nothing uploaded"
    data = ROOT / "data"
    with tempfile.TemporaryDirectory(dir=data) as scratch:
        for folder in ("train_images", "dev_images"):
            archive_path = Path(scratch) / f"{folder}.tar"
            with tarfile.open(archive_path, "w") as archive:  # images are already compressed
                archive.add(data / folder, arcname=folder)
            print(f"{archive_path.name}: {archive_path.stat().st_size / 1024**3:.2f} GB", flush=True)
        for name in ("train.csv", "dev.csv", "metric.py"):
            (Path(scratch) / name).write_bytes((data / name).read_bytes())
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=scratch,
                          commit_message="competition data (private, team only)")
    print(f"uploaded to the private dataset {repo}")


def main():
    import sys

    if sys.argv[1:2] == ["--upload-data"]:
        return upload_data(sys.argv[2])
    upload_job = sys.argv[2] if sys.argv[1:2] == ["--upload-job"] else None
    members = {name: (ROOT / name).read_bytes() for name in FILES}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    out = (Path(sys.argv[2]) if sys.argv[1:2] == ["--out"] else
           ROOT / "data" / "hf_jobs" / "hf_job_vlh.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    train_hash = hashlib.sha256((ROOT / "data" / "train.csv").read_bytes()).hexdigest()
    out.write_text(TEMPLATE.replace("@@FOLDER@@", FOLDER).replace("@@TRAIN_SHA256@@", train_hash)
                   .replace("@@BUNDLE@@", base64.b64encode(buffer.getvalue()).decode()))
    print(f"{out}: {len(members)} files, {out.stat().st_size // 1024} KB")
    if upload_job:  # for images without uv: the job script is read from the private repo
        from huggingface_hub import HfApi

        assert HfApi().repo_info(upload_job).private, f"{upload_job} is not private"
        HfApi().upload_file(path_or_fileobj=str(out), path_in_repo="jobs/hf_job_vlh.py", repo_id=upload_job,
                            commit_message="job script")
        launch = out.parent / "launch_devel.sh"
        launch.write_text(
            "set -e; (command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)); "
            "pip install -q uv huggingface_hub; "
            f"hf download {upload_job} jobs/hf_job_vlh.py --local-dir /tmp/job; uv run /tmp/job/jobs/hf_job_vlh.py\n")
        print(f"uploaded to {upload_job}/jobs/hf_job_vlh.py; launch command in {launch}")


if __name__ == "__main__":
    main()
