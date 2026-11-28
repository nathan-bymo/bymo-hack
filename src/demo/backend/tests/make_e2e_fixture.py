"""Fixture dei test E2E: immagini sintetiche (Pillow) e registrazioni **segnaposto** in una cartella temporanea.

Solo per i test: i valori sono inventati e non vanno mai mostrati come risultati. Uso:
    python make_e2e_fixture.py <cartella>   ->   <cartella>/files/*, <cartella>/data/recorded/…
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from demo_api.images_io import context_hash  # noqa: E402
from demo_api.repo_code import ensure_src_on_path  # noqa: E402

ensure_src_on_path()
from common.prompt import DIMENSIONS, TAXONOMY  # noqa: E402

COLORS = [(200, 40, 40), (40, 160, 60), (50, 70, 200)]


def jpeg(color, size=(96, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def dims(labels: dict, conf: float) -> dict:
    out = {}
    for d in DIMENSIONS:
        label = labels.get(d, TAXONOMY[d][0])
        out[d] = {"label": label, "confidence": conf, "top_k": [{"label": label, "p": conf}],
                  "confidence_method": "head_softmax"}
    return out


def main(root: Path) -> None:
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    shas = []
    for i, color in enumerate(COLORS):
        data = jpeg(color)
        (files / f"synthetic-{i + 1}.jpg").write_bytes(data)
        shas.append(hashlib.sha256(data).hexdigest())
    (files / "bad.png").write_bytes(b"this is not an image")
    samples = root / "data" / "samples"   # foto di esempio: la prima ha le registrazioni segnaposto qui sotto
    samples.mkdir(parents=True, exist_ok=True)
    (samples / "01-sample red box.jpg").write_bytes(jpeg(COLORS[0]))
    (samples / "02-sample green box.jpg").write_bytes(jpeg(COLORS[1]))
    (samples / "labels.json").write_text(json.dumps({"01-sample red box": {   # etichetta di riferimento segnaposto (come train.csv)
        "subject": {"label": "full_product", "human": True}, "background": {"label": "white", "human": True},
        "perspective": {"label": "front_head_on", "human": True}, "individuals": {"label": "none", "human": True},
        "text_classification": {"label": "none", "human": True}, "text_language": {"label": "none", "human": False}}}))
    ctx = context_hash({})
    runs = {
        "apertus-8b-lora-heads-p11": ("run-e2e-a", "TEST-GPU-A", "left_side"),
        "apertus-8b-fullft-heads": ("run-e2e-b", "TEST-GPU-B", "right_side"),
    }
    for model_id, (run_id, gpu, perspective) in runs.items():
        run = root / "data" / "recorded" / model_id / run_id
        run.mkdir(parents=True, exist_ok=True)
        (run / "run.json").write_text(json.dumps({
            "model_id": model_id, "model_revision": "e2e-placeholder", "gpu_name": gpu, "batch_size": 8,
            "images_per_second": 10.0, "started_at": "2026-11-28T08:00:00Z", "finished_at": "2026-11-28T08:01:00Z",
            "use_context": True}))
        lines = []
        for k, sha in enumerate(shas):
            labels = {"perspective": perspective if k == 0 else "front_head_on"}
            lines.append(json.dumps({"image_sha256": sha, "context_hash": ctx, "use_context": True,
                                     "dimensions": dims(labels, 0.55 + 0.1 * k), "valid": True, "raw_output": None,
                                     "timing_ms": {"total": 50.0}}))
        (run / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    print(root)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
