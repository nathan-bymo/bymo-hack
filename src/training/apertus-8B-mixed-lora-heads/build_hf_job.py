"""Build a self-contained run for the mixed encoder experiment on private data."""

import argparse
import base64
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FOLDER = HERE.relative_to(ROOT).as_posix()
SIGLIP = "src/training/apertus-8B-vadapter-lora-heads"
DINO = "src/training/apertus-8B-dinov3-lora-heads"

ENTRY = '''
def run_mixed_final():
    """FINAL=1: decrypts the final images (or a test archive, FINAL_ARCHIVE) and predicts them with the mixed
    checkpoint PREDICT_FROM of UPLOAD_REPO. Only predictions leave the job."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    if not (UPLOAD_REPO and PREDICT_FROM) or not HfApi().repo_info(UPLOAD_REPO).private:
        raise ValueError("FINAL=1 needs PREDICT_FROM, a checkpoint of the private repository UPLOAD_REPO.")
    sys.path.insert(0, str(WORK / "@@MIXED@@"))
    from mixed_model import E, FUSION_NAME

    ensure_opencv()
    with ThreadPoolExecutor(3) as pool:  # the downloads together: every minute counts after 13:00
        folder = pool.submit(final_data)
        checkpoint = pool.submit(fetch_checkpoint, PREDICT_FROM)
        pool.submit(E.model_dir).result()
        folder, checkpoint = folder.result(), checkpoint.result()
    if not (checkpoint / FUSION_NAME).is_file():
        raise FileNotFoundError(f"{PREDICT_FROM} is not a mixed checkpoint")
    outputs = RUN / "predictions"
    run(["@@MIXED@@/train_mixed.py", "--checkpoint", checkpoint, "--data-dir", folder, "--out", outputs,
         "--predict-csv", folder / "final.csv"], "predict_final.log")
    upload(outputs, "predictions", ["final*"])
    upload(RUN, "logs", ["*.log"])
    log("ALL DONE")


def run_mixed():
    import json
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    unpack()
    sys.path.insert(0, str(WORK / "@@MIXED@@"))
    from mixed_model import DINO_ID, E, FUSION_NAME, PATCH_NAME

    if os.environ.get("FINAL"):
        return run_mixed_final()
    vision_rank = os.environ.get("VISION_LORA", "")  # continue a mixed checkpoint, training a LoRA on SigLIP too
    if not (UPLOAD_REPO and INIT_FROM and (DATA_REPO or os.environ.get("KAGGLE_API_TOKEN"))):
        raise ValueError("A private output repo, the starting checkpoint path and the data (DATA_REPO or Kaggle) are required.")
    api = HfApi()
    source_repo = os.environ.get("SOURCE_REPO") or UPLOAD_REPO
    source_token = os.environ.get("SOURCE_HF_TOKEN")
    source_api = HfApi(token=source_token)
    source = source_api.repo_info(source_repo)
    if not api.repo_info(UPLOAD_REPO).private or not source.private:
        raise ValueError("The starting and output model repositories must be private.")

    revision = os.environ.get("DINO_REVISION")
    if not revision:
        ref = os.environ.get("DINO_SETTINGS", "dino-synthetic-modal-20261128-015048/model/best/settings.json")
        file = hf_hub_download(source_repo, ref, revision=source.sha, token=source_token)
        revision = json.loads(Path(file).read_text())["dino_revision"]
    hf_hub_download(DINO_ID, "config.json", revision=revision)
    initial = Path(snapshot_download(source_repo, token=source_token, revision=source.sha,
                                    allow_patterns=[f"{INIT_FROM}/*", f"{INIT_FROM}/lora/*"],
                                    local_dir=WORK / "source")) / INIT_FROM
    needed = ["settings.json", "trainable.pt", "lora/adapter_model.safetensors"]
    if vision_rank:
        needed.append(FUSION_NAME)
    for required in needed:
        if not (initial / required).is_file():
            raise FileNotFoundError(initial / required)

    ensure_opencv()
    with ThreadPoolExecutor(2) as pool:
        data = pool.submit(download_data)
        base = pool.submit(E.model_dir)
        data.result()
        base.result()

    extra = []
    synthetic = os.environ.get("MIXED_SYNTHETIC_FILE")
    if synthetic:
        archive = hf_hub_download(DATA_REPO, synthetic, repo_type="dataset")
        with tarfile.open(archive) as stream:
            stream.extractall(DATA, filter="data")
        extra.extend(["--synthetic", DATA / "synthetic/synthetic.csv"])

    ocr_expected = {"detector": E.OCR_DETECTOR, "recognizer": E.OCR_RECOGNIZER,
                    "revision": E.OCR_RECOGNIZER_REVISION, "min_confidence": E.OCR_MIN_CONFIDENCE,
                    "max_chars": E.OCR_MAX_CHARS}
    files = set(source_api.list_repo_files(source_repo, revision=source.sha))
    for name in ("all_but_test_256", "training_256"):
        prefix = f"cache/{name}"
        if {f"{prefix}/ocr_meta.json", f"{prefix}/ocr.json"} <= files:
            metadata = hf_hub_download(source_repo, f"{prefix}/ocr_meta.json", revision=source.sha, token=source_token)
            if json.loads(Path(metadata).read_text()) == ocr_expected:
                extra += ["--ocr-cache", hf_hub_download(source_repo, f"{prefix}/ocr.json", revision=source.sha,
                                                         token=source_token)]
                break

    wall = float(os.environ.get("WALL_MINUTES") or 125) * 60 - (time.time() - START)
    if wall < 15 * 60:
        raise TimeoutError("Preparation has consumed most of the training allocation.")
    if vision_rank:
        extra += ["--vision-lora-rank", vision_rank, "--vision-epochs", os.environ.get("VISION_EPOCHS") or "2",
                  "--lr-vision", os.environ.get("LR_VISION") or "5e-5"]
    run(["@@MIXED@@/train_mixed.py", "--checkpoint" if vision_rank else "--initial", initial,
         "--data-dir", DATA, "--out", RUN / "mixed",
         "--dino-revision", revision, "--source-id", f"{source_repo}/{INIT_FROM}", "--source-revision", source.sha,
         "--wall-seconds", str(wall), "--upload-repo", UPLOAD_REPO,
         "--prefix", PREFIX, "--predict-dev", *extra], "mixed.log")

    best = RUN / "mixed" / "best"
    upload(best, "model/best", ["settings.json", "trainable.pt", FUSION_NAME, PATCH_NAME, "lora/*", "vision_lora/*"])
    selected = json.loads((best / "settings.json").read_text())
    required = {"model/best/settings.json", "model/best/trainable.pt", f"model/best/{FUSION_NAME}",
                "model/best/lora/adapter_model.safetensors", "model/training.json",
                "predictions/prompt_test.json", "predictions/dev.csv"}
    if selected.get("unfrozen_blocks"):
        required.add(f"model/best/{PATCH_NAME}")
    if selected.get("vision_lora_rank"):
        required.add("model/best/vision_lora/adapter_model.safetensors")
    paths = set(api.list_repo_files(UPLOAD_REPO))
    missing = {f"{PREFIX}/{name}" for name in required} - paths
    if missing:
        raise RuntimeError(f"Missing private artifacts: {sorted(missing)}")
    upload(RUN, "logs", ["*.log"])
    log("Mixed encoder experiment complete")


if __name__ == "__main__":
    run_mixed()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/hf_jobs/hf_job_mixed.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("siglip_hf_builder", ROOT / SIGLIP / "build_hf_job.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    members = [*base.FILES, *(f"{DINO}/{name}" for name in ("pipeline.py", "experiment.py", "base_weights.py")),
               *(f"{FOLDER}/{name}" for name in ("mixed_model.py", "train_mixed.py"))]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in members:
            contents = (ROOT / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    template, old = base.TEMPLATE.rsplit('if __name__ == "__main__":', 1)
    if old.strip() != "main()":
        raise ValueError("The original HF job entry point has changed.")
    script = (template + ENTRY).replace("@@FOLDER@@", SIGLIP).replace("@@MIXED@@", FOLDER)
    script = script.replace("@@TRAIN_SHA256@@", hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest())
    script = script.replace("@@BUNDLE@@", base64.b64encode(buffer.getvalue()).decode())
    compile(script, str(args.out), "exec")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(script)
    print(f"Generated {args.out} ({len(members)} bundled sources).")


if __name__ == "__main__":
    main()
