import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from imgs import make

MODEL = "apertus-8b-lora-heads-p11"


def dims(label_map=None):
    base = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
            "individuals": "none", "text_classification": "none", "text_language": "none"}
    out = {}
    for d, label in {**base, **(label_map or {})}.items():
        out[d] = {"label": label, "confidence": 0.5, "top_k": [{"label": label, "p": 0.5}],
                  "confidence_method": "head_softmax"}
    return out


@pytest.fixture
def session(client):
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    files = [("files", (f"{i}.jpg", make("JPEG", color=(i * 50, 5, 5)), "image/jpeg")) for i in range(2)]
    return sid, client.post("/api/images", data={"session_id": sid}, files=files).json()["images"]


def write_run(settings, images_to_record, run_id="run-a"):
    run = settings.data_dir / "recorded" / MODEL / run_id
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "model_id": MODEL, "model_revision": "test-rev", "gpu_name": "TEST-GPU", "batch_size": 8,
        "images_per_second": 12.5, "peak_memory_gb": 20.0, "started_at": "2026-11-28T08:00:00Z",
        "finished_at": "2026-11-28T08:01:00Z", "use_context": True}))
    lines = [json.dumps({"image_sha256": i["sha256"], "context_hash": i["context_hash"], "use_context": True,
                         "dimensions": dims(), "valid": True, "raw_output": None, "timing_ms": {"total": 80.0}})
             for i in images_to_record]
    (run / "predictions.jsonl").write_text("\n".join(lines) + "\n")


def test_recorded_prediction_flow(client, settings, session):
    sid, imgs = session
    write_run(settings, imgs[:1])
    assert client.post("/api/recorded/rescan").json() == {"runs": 1}
    model = client.get(f"/api/models/{MODEL}").json()
    assert model["modes"] == ["recorded"] and model["recorded_runs"][0]["gpu_name"] == "TEST-GPU"
    r = client.post("/api/predict", json={"model_id": MODEL, "image_ids": [imgs[0]["image_id"]], "mode": "recorded"})
    assert r.status_code == 202
    with client.stream("GET", r.json()["events_url"]) as s:
        list(s.iter_lines())
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    result = job["results"][0]
    assert result["mode"] == "recorded" and result["recorded"]["gpu_name"] == "TEST-GPU"
    assert result["recorded"]["images_per_second"] == 12.5
    assert result["dimensions"]["subject"]["confidence_method"] == "head_softmax"
    assert job["batch"]["gpu_name"] == "TEST-GPU" and job["batch"]["recorded"]["run_id"] == "run-a"


def test_missing_recording_is_recorded_missing(client, settings, session):
    sid, imgs = session
    write_run(settings, imgs[:1])
    client.post("/api/recorded/rescan")
    r = client.post("/api/predict", json={"model_id": MODEL, "image_ids": [imgs[1]["image_id"]], "mode": "recorded"})
    with client.stream("GET", r.json()["events_url"]) as s:
        text = "\n".join(s.iter_lines())
    assert "RECORDED_MISSING" in text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["errors"][0]["error"]["code"] == "RECORDED_MISSING" and job["status"] == "error"


def test_recorded_mode_without_any_run_is_unavailable(client, session):
    sid, imgs = session
    r = client.post("/api/predict", json={"model_id": MODEL, "image_ids": [imgs[0]["image_id"]], "mode": "recorded"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_export_session_writes_manifest_without_names(client, settings, session):
    sid, imgs = session
    script = Path(__file__).resolve().parents[2] / "scripts" / "export_session.py"
    spec = importlib.util.spec_from_file_location("exp", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.export(settings.data_dir, sid)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["images"]) == 2 and "original_filename" not in json.dumps(manifest)
    for item in manifest["images"]:
        data = (out / "images" / f"{item['sha256']}.{item['ext']}").read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
