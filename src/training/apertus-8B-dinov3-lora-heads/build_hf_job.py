"""Bundle the DINOv3 experiment with the existing private HF Jobs infrastructure."""

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
BASE_BUILDER = HERE.parent / "apertus-8B-vadapter-lora-heads" / "build_hf_job.py"

ENTRY = '''
def run_dino():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig

    unpack()
    sys.path.insert(0, str(WORK / "@@DINO_FOLDER@@"))
    from pipeline import DINO_ID, E

    source_repo = os.environ.get("SOURCE_REPO") or UPLOAD_REPO
    source_token = os.environ.get("SOURCE_HF_TOKEN")
    if not UPLOAD_REPO or not source_repo or not INIT_FROM:
        raise ValueError("Set UPLOAD_REPO, SOURCE_REPO and INIT_FROM.")
    api = HfApi()
    if not api.repo_info(UPLOAD_REPO).private:
        raise ValueError("Weights and predictions require a private HF repository.")

    # Check both existing weights and access to the gated DINO configuration first.
    config_path = Path(hf_hub_download(DINO_ID, "config.json"))
    dino_revision = config_path.parent.name
    dino_config = AutoConfig.from_pretrained(DINO_ID, revision=dino_revision)
    if dino_config.model_type != "dinov3_vit" or dino_config.hidden_size != 768:
        raise ValueError("Expected the pinned DINOv3 ViT-B/16 model.")
    from huggingface_hub import snapshot_download

    source_api = HfApi(token=source_token)
    if not source_api.repo_info(source_repo).private:
        raise ValueError("The existing Galaxus checkpoint must come from a private repository.")
    source_revision = source_api.repo_info(source_repo).sha
    initial = Path(snapshot_download(source_repo, revision=source_revision, token=source_token,
                                    allow_patterns=[f"{INIT_FROM}/*", f"{INIT_FROM}/lora/*"],
                                    local_dir=WORK / "source")) / INIT_FROM
    for required in ("settings.json", "trainable.pt", "lora/adapter_model.safetensors"):
        if not (initial / required).is_file():
            raise FileNotFoundError(initial / required)
    log("DINO model access and source checkpoint verified")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The experiment requires CUDA.")
    ensure_opencv()
    with ThreadPoolExecutor(2) as pool:
        data = pool.submit(download_data)
        model = pool.submit(E.model_dir)
        data.result()
        model.result()

    # This cache comes from the same dataset and OCR configuration.
    cache_args = []
    expected = {"detector": E.OCR_DETECTOR, "recognizer": E.OCR_RECOGNIZER,
                "revision": E.OCR_RECOGNIZER_REVISION, "min_confidence": E.OCR_MIN_CONFIDENCE,
                "max_chars": E.OCR_MAX_CHARS}
    import json
    files = set(source_api.list_repo_files(source_repo, revision=source_revision))
    for cache_name in ("all_but_test_256", "training_256"):
        prefix = f"cache/{cache_name}"
        if {f"{prefix}/ocr_meta.json", f"{prefix}/ocr.json"} <= files:
            metadata = hf_hub_download(source_repo, f"{prefix}/ocr_meta.json", revision=source_revision, token=source_token)
            if json.loads(Path(metadata).read_text()) == expected:
                cache_args = ["--ocr-cache", hf_hub_download(source_repo, f"{prefix}/ocr.json",
                                                            revision=source_revision, token=source_token)]
                break

    # The training deadline includes the new image encoding, and leaves time for predictions.
    wall = int(os.environ.get("DINO_TOTAL_MINUTES", "150")) * 60 - (time.time() - START)
    if wall < 15 * 60:
        raise TimeoutError("Not enough time remains for DINO encoding and training.")
    cadence = ["--eval-once-per-epoch"] if os.environ.get("DINO_EVAL_ONCE") == "1" else []
    run(["@@DINO_FOLDER@@/experiment.py", "--data-dir", DATA, "--out", RUN / "dino",
         "--initial", initial, "--dino-revision", dino_revision, "--epochs", os.environ.get("DINO_EPOCHS", "5"),
         "--source-id", f"{source_repo}/{INIT_FROM}", "--source-revision", source_revision,
         "--max-minutes", os.environ.get("DINO_MAX_MINUTES", "110"), "--wall-seconds", str(wall),
         "--upload-repo", UPLOAD_REPO, "--prefix", PREFIX, "--predict-dev", *cache_args, *cadence], "dino.log")
    if not api.file_exists(UPLOAD_REPO, f"{PREFIX}/model/best/lora/adapter_model.safetensors"):
        upload(RUN / "dino" / "best", "model/best", ["settings.json", "trainable.pt", "lora/*"])
    required = ("model/best/settings.json", "model/best/trainable.pt", "model/best/lora/adapter_model.safetensors",
                "model/training.json", "predictions/prompt_test.json", "predictions/dev.csv")
    missing = set(f"{PREFIX}/{path}" for path in required) - set(api.list_repo_files(UPLOAD_REPO))
    if missing:
        raise RuntimeError(f"Missing artifacts in the private repository: {sorted(missing)}")
    upload(RUN, "logs", ["*.log"])
    log("DINO experiment complete")


if __name__ == "__main__":
    run_dino()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/hf_jobs/hf_job_dinov3.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("base_hf_builder", BASE_BUILDER)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    files = [*base.FILES, *(f"{FOLDER}/{name}" for name in ("pipeline.py", "experiment.py"))]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in files:
            contents = (ROOT / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))

    template, entry = base.TEMPLATE.rsplit('if __name__ == "__main__":', 1)
    if entry.strip() != "main()":
        raise ValueError("The upstream HF job entry point changed.")
    template += ENTRY
    template = template.replace("@@FOLDER@@", base.FOLDER).replace("@@DINO_FOLDER@@", FOLDER)
    template = template.replace("@@TRAIN_SHA256@@", hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest())
    template = template.replace("@@BUNDLE@@", base64.b64encode(buffer.getvalue()).decode())
    compile(template, str(args.out), "exec")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template)
    print(f"Generated {args.out} ({len(files)} bundled sources).")


if __name__ == "__main__":
    main()
