"""Foto di esempio: elenco, file e modelli con risposta già registrata."""

from __future__ import annotations

import hashlib
import io
import json

from PIL import Image

from demo_api import images_io


def jpeg(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_no_samples_folder_content(client):
    assert client.get("/api/samples").json() == {"samples": []}


def test_samples_list_file_and_recorded_models(client):
    storage = client.app.state.storage
    data = jpeg((10, 120, 200))
    (storage.samples_dir / "02-left angle view.jpg").write_bytes(data)
    (storage.samples_dir / "notes.txt").write_text("non è una foto")
    sha = images_io.ingest(data, "image/jpeg", 10_000_000, 50_000_000).sha256

    run = storage.recorded_dir / "apertus-8b-lora-heads-p11" / "run-test"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"model_id": "apertus-8b-lora-heads-p11", "gpu_name": "TEST-GPU"}))
    (run / "predictions.jsonl").write_text(json.dumps({"image_sha256": sha, "context_hash": "x", "dimensions": {}}) + "\n")
    assert client.post("/api/recorded/rescan").json()["runs"] == 1

    samples = client.get("/api/samples").json()["samples"]
    assert len(samples) == 1                                   # il file di testo non conta
    sample = samples[0]
    assert sample["title"] == "Left angle view" and sample["sha256"] == sha
    assert sample["recorded_models"] == ["apertus-8b-lora-heads-p11"]
    served = client.get(sample["file_url"])
    assert served.status_code == 200 and hashlib.sha256(served.content).hexdigest() == hashlib.sha256(data).hexdigest()
    assert client.get("/api/samples/non-esiste/file").status_code == 404
    assert client.get("/api/samples/..%2Fdemo.sqlite/file").status_code == 404   # nessun percorso libero
