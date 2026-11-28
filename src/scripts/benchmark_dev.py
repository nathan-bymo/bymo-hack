"""Benchmark all three checkpoints on a single HF Jobs GPU.

Build a self-contained job with --build OUTPUT. The read/write /benchmark bucket
must contain weights/{p11,vlh}. Only aggregate results are persisted there.
"""

# /// script
# requires-python = ">=3.12,<3.13"
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

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

BUNDLE = ""
DEV_SHA256 = ""
MODELS = {
    "vlh": "apertus-8B-vadapter-lora-heads/predict.py",
    "fullft": "apertus-8B-fullft-heads/ft_predict.py",
    "p11": "apertus-8B-lora-heads/predict.py",
}
FULLFT_REPO = "StoryofLegend/apertus-8b-galaxus-fullft"
FULLFT_REV = "aac1375d3efb93460a564faaef00a60430d84443"
FULLFT_PATH = "phase2/model/last"


def build(output):
    root = Path(__file__).resolve().parents[2]
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for folder in ("common", *(entry.split("/")[0] for entry in MODELS.values())):
            for path in sorted((root / "src" / folder).glob("*.py")):
                if not path.name.startswith(("test_", "build_")):
                    archive.add(path, arcname=str(path.relative_to(root)))
    sha = hashlib.sha256((root / "data/hackapertus-galaxus/dev.csv").read_bytes()).hexdigest()
    script = Path(__file__).read_text().replace('BUNDLE = ""', f'BUNDLE = "{base64.b64encode(blob.getvalue()).decode()}"', 1)
    script = script.replace('DEV_SHA256 = ""', f'DEV_SHA256 = "{sha}"', 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script)


def download_dev(data):
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files("hackapertus-galaxus", path=str(data), quiet=True)
    archive_path = data / "hackapertus-galaxus.zip"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extract("dev.csv", data)
        csv = data / "dev.csv"
        if hashlib.sha256(csv.read_bytes()).hexdigest() != DEV_SHA256:
            raise ValueError("Remote dev.csv differs from the local benchmark input")
        rows = pd.read_csv(csv, dtype=str, keep_default_na=False)
        paths = [Path(name) for name in rows.image_path]
        if len(rows) != 5261 or rows.image_id.duplicated().any():
            raise ValueError("Unexpected dev row count or duplicate image IDs")
        if any(path.is_absolute() or path.parts[0] != "dev_images" or ".." in path.parts for path in paths):
            raise ValueError("Unexpected dev image path")
        for path in set(paths):
            archive.extract(path.as_posix(), data)
    archive_path.unlink()
    return rows


def predict(name, script, checkpoint, data, output, batch_size, limit):
    cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--data-dir", str(data),
           "--csv", str(data / "dev.csv"), "--out", str(output), "--batch-size", str(batch_size)]
    if limit:
        cmd += ["--limit", str(limit)]
    started = time.perf_counter()
    with output.with_suffix(".log").open("w") as log, subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as process:
        for line in process.stdout:
            log.write(line)
            if re.fullmatch(r"\d+/\d+ images, \d+\.\d+ images/s\n", line):
                print(f"{name}: {line}", end="", flush=True)
        code = process.wait()
    if code:
        log_text = output.with_suffix(".log").read_text()
        if "model loaded in" not in log_text:
            # The predictor has not processed any image at this point.
            print(log_text[-2500:], flush=True)
        errors = re.findall(r"^(?:[a-zA-Z_]\w*\.)*([A-Z]\w*(?:Error|Exception)):", log_text, re.M)
        reason = f" ({errors[-1]})" if errors else ""
        raise RuntimeError(f"{name} batch {batch_size} failed with exit code {code}{reason}")
    report = json.loads(output.with_suffix(".report.json").read_text())
    count = report.get("predicted", report["images"])
    if count != report["images"] or (limit and count != limit):
        raise ValueError(f"{name} batch {batch_size}: incomplete predictions")
    report["process_seconds"] = round(time.perf_counter() - started, 2)
    report["ms_per_image"] = round(report["seconds"] * 1000 / count, 2)
    print(f"{name} batch {batch_size}: {count} images, {report['images_per_second']} img/s, "
          f"{report['ms_per_image']} ms/image", flush=True)
    return report


def diagnostics(outputs, rows):
    import pandas as pd
    import torch
    from common.prompt import DIMENSIONS, TAXONOMY

    ids = pd.Index(rows.image_id)
    labels, confidence = {}, {}
    for name, path in outputs.items():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("image_id")
        data = torch.load(path.with_suffix(".probs.pt"), map_location="cpu", weights_only=True)
        prob_ids = pd.Index(data["image_ids"])
        if not (frame.index.is_unique and prob_ids.is_unique and len(ids) == len(frame) == len(prob_ids)
                and ids.isin(frame.index).all() and ids.isin(prob_ids).all()):
            raise ValueError(f"{name}: predictions do not cover the exact dev set")
        if tuple(data["dimensions"]) != tuple(DIMENSIONS):
            raise ValueError(f"{name}: unexpected dimension order")
        labels[name] = frame.loc[ids]
        confidence[name] = {}
        for dimension, probabilities in zip(DIMENSIONS, data["probabilities"], strict=True):
            if not torch.isfinite(probabilities).all():
                raise ValueError(f"{name}: non-finite {dimension} probabilities")
            confidence[name][dimension] = probabilities.max(dim=1).values.numpy()[prob_ids.get_indexer(ids)]

    stats = {}
    for name, frame in labels.items():
        others = [other for other_name, other in labels.items() if other_name != name]
        stats[name] = {}
        for dimension, classes in TAXONOMY.items():
            value = frame[dimension]
            first, second = [other[dimension] for other in others]
            split = (value != first) | (value != second)
            outvoted = (first == second) & (value != first)
            class_stats = {}
            for label in classes:
                mask = (value == label).to_numpy()
                count = int(mask.sum())
                scores = confidence[name][dimension][mask]
                alternatives = first[outvoted & (value == label)].value_counts().head(3)
                class_stats[label] = {
                    "predicted_count": count,
                    "mean_confidence": round(float(scores.mean()), 4) if count else None,
                    "below_0_6": int((scores < 0.6).sum()),
                    "any_disagreement": int(split[mask].sum()),
                    "both_peers_agree_on_another_class": int(outvoted[mask].sum()),
                    "top_peer_agreement": {str(k): int(v) for k, v in alternatives.items()},
                }
            stats[name][dimension] = class_stats
    return stats


def ensure_ocr():
    """Use the same OpenCV system dependency check as the vision-adapter HF job."""
    if subprocess.run([sys.executable, "-c", "import cv2"], capture_output=True).returncode:
        print("Installing OpenCV system libraries", flush=True)
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "libgl1", "libglib2.0-0"], check=True)
    subprocess.run([sys.executable, "-c", "from doctr.models import from_hub, ocr_predictor"], check=True)


def benchmark(work, bucket, run, skip_single_image):
    from huggingface_hub import snapshot_download
    import torch

    if not BUNDLE or not DEV_SHA256:
        raise ValueError("Build the HF job first")
    if not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This benchmark requires a Hugging Face H200")
    ensure_ocr()
    data = work / "data"
    data.mkdir()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE))) as archive:
        archive.extractall(work, filter="data")
    sys.path.insert(0, str(work / "src"))
    from common.prompt import DIMENSIONS, TAXONOMY

    if tuple(TAXONOMY) != tuple(DIMENSIONS) or sum(map(len, TAXONOMY.values())) != 41:
        raise ValueError("Unexpected benchmark taxonomy")

    print("Downloading official dev input", flush=True)
    rows = download_dev(data)
    print("Verified 5,261 dev rows; downloading pinned full fine-tune", flush=True)
    snapshot = Path(snapshot_download(FULLFT_REPO, revision=FULLFT_REV, allow_patterns=f"{FULLFT_PATH}/*"))
    checkpoints = {"fullft": snapshot / FULLFT_PATH}
    for name in ("p11", "vlh"):
        checkpoints[name] = work / name
        shutil.copytree(bucket / "weights" / name, checkpoints[name])

    result_dir = bucket / "results" / run
    result_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    outputs = {}
    for name, entry in MODELS.items():
        script = work / "src" / entry
        outputs[name] = work / f"{name}.csv"
        print(f"Starting {name}", flush=True)
        results[name] = {
            "dev_batch_32": predict(name, script, checkpoints[name], data, outputs[name], 32, 0),
        }
        (result_dir / "timings.json").write_text(json.dumps(results, indent=2))
        if not skip_single_image:
            results[name]["dev_first_128_batch_1"] = predict(
                name, script, checkpoints[name], data, work / f"{name}-single.csv", 1, 128)
            (result_dir / "timings.json").write_text(json.dumps(results, indent=2))
    (result_dir / "class_diagnostics.json").write_text(json.dumps(diagnostics(outputs, rows), indent=2))
    print(f"Published aggregate benchmark results in results/{run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--bucket", type=Path, default=Path("/benchmark"))
    parser.add_argument("--run", default="h200")
    parser.add_argument("--skip-single-image", action="store_true")
    args = parser.parse_args()
    if args.build:
        build(args.build)
    else:
        with tempfile.TemporaryDirectory(prefix="apertus-dev-") as work:
            benchmark(Path(work), args.bucket, args.run, args.skip_single_image)


if __name__ == "__main__":
    main()
