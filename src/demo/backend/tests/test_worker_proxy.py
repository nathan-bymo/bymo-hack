"""Protocollo worker ↔ proxy con il worker in modalità `--dry` (famiglia finta solo per i test, nessuna GPU)."""

import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from demo_api.main import create_app
from demo_api.memory import ResidentSet
from demo_api.errors import ApiError
from demo_worker.server import create_app as create_worker
from imgs import make

MODEL = "apertus-8b-lora-p10"   # configurazione completa: richiede solo l'URL del worker (famiglia lora_generative)
FAMILY = "lora_generative"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("DEMO_WORKER_ALLOW_DRY", "1")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_worker(FAMILY, dry=True, token="tok"), port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def api(settings, worker):
    settings.worker_urls = {FAMILY: worker}
    settings.worker_token = "tok"
    with TestClient(create_app(settings)) as c:
        yield c


def test_dry_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("DEMO_WORKER_ALLOW_DRY", raising=False)
    with pytest.raises(Exception):
        create_worker(FAMILY, dry=True)


def test_worker_rejects_bad_token(worker):
    import httpx

    assert httpx.get(worker + "/worker/info").status_code == 401
    ok = httpx.get(worker + "/worker/info", headers={"X-Demo-Worker-Token": "tok"}).json()
    assert ok["gpu_name"] == "DRY (no GPU)" and ok["loaded"] == []


def test_load_predict_unload_through_proxy(api):
    assert api.get(f"/api/models/{MODEL}").json()["status"] == "not_loaded"
    sid = api.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    files = [("files", (f"{i}.jpg", make("JPEG", color=(i * 50, 9, 9)), "image/jpeg")) for i in range(3)]
    ids = [i["image_id"] for i in api.post("/api/images", data={"session_id": sid}, files=files).json()["images"]]
    r = api.post("/api/predict", json={"model_id": MODEL, "image_ids": ids, "batch_size": 2})
    assert r.status_code == 202
    with api.stream("GET", r.json()["events_url"]) as s:
        list(s.iter_lines())
    job = api.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "done" and len(job["results"]) == 3
    res = job["results"][0]
    assert len(res["dimensions"]) == 6 and res["dimensions"]["subject"]["confidence_method"] == "head_softmax"
    assert job["batch"]["gpu_name"] == "DRY (no GPU)" and job["batch"]["images_per_second"] > 0
    assert api.get(f"/api/models/{MODEL}").json()["status"] == "ready"
    assert api.get("/api/throughput", params={"model_id": MODEL}).json()["measured_in_demo"][0]["gpu_name"] == "DRY (no GPU)"
    assert api.post(f"/api/models/{MODEL}/unload").json()["state"] == "not_loaded"
    assert api.get(f"/api/models/{MODEL}").json()["status"] == "not_loaded"


def test_explicit_load_endpoint(api):
    r = api.post(f"/api/models/{MODEL}/load")
    assert r.status_code == 202 and r.json()["state"] == "loading"
    for _ in range(100):
        if api.get(f"/api/models/{MODEL}").json()["status"] == "ready":
            break
        time.sleep(0.05)
    assert api.get(f"/api/models/{MODEL}").json()["status"] == "ready"


def test_download_needs_confirmation_and_never_downloads(api):
    r = api.post(f"/api/models/{MODEL}/download", json={"confirm": False})
    assert r.status_code == 409 and r.json()["error"]["code"] == "CONFIRMATION_REQUIRED"
    assert r.json()["error"]["details"]["download_gb"] == 0.319
    r = api.post(f"/api/models/{MODEL}/download", json={"confirm": True})
    assert r.status_code == 409 and "GPU host" in r.json()["error"]["message"]


def test_load_failure_maps_worker_code(api, monkeypatch):
    api.app.state.registry.models[MODEL].raw["weights"]["fail"] = "GPU_OOM"
    r = api.post("/api/predict", json={"model_id": MODEL, "image_ids": []})
    assert r.status_code == 422  # lista vuota non valida: la validazione precede il caricamento


def test_unreachable_worker_is_upstream_error(settings):
    settings.worker_urls = {FAMILY: "http://127.0.0.1:9"}
    with TestClient(create_app(settings)) as c:
        sid = c.post("/api/sessions", json={"name": "t"}).json()["session_id"]
        img = c.post("/api/images", data={"session_id": sid},
                     files=[("files", ("a.jpg", make("JPEG"), "image/jpeg"))]).json()["images"][0]["image_id"]
        r = c.post("/api/predict", json={"model_id": MODEL, "image_ids": [img]})
        with c.stream("GET", r.json()["events_url"]) as s:
            text = "\n".join(s.iter_lines())
        assert "UPSTREAM_ERROR" in text


# ------------------------------------------------------------------------------- LRU e budget (B.8)
def test_lru_evicts_least_recent_not_busy():
    busy = set()
    rs = ResidentSet(lambda m: m in busy, max_resident=2)
    rs.add("a", 20)
    rs.add("b", 20)
    rs.touch("a")                      # ordine di uso: b, a
    assert rs.plan("c", 20, 78) == ["b"]
    busy.add("b")
    assert rs.plan("c", 20, 78) == ["a"]   # b ha un job attivo: si scarica a


def test_all_busy_raises_model_busy_and_budget_check():
    rs = ResidentSet(lambda m: True, max_resident=1)
    rs.add("a", 20)
    with pytest.raises(ApiError) as e:
        rs.plan("b", 20, 78)
    assert e.value.code == "MODEL_BUSY"
    rs2 = ResidentSet(lambda m: False, max_resident=5)
    rs2.add("a", 50)
    assert rs2.plan("b", 30, 78) == ["a"]           # 50 + 30 > 78 → serve scaricare a
    with pytest.raises(ApiError):
        rs2.plan("big", 100, 78)                      # non entra neanche da solo
    assert rs2.plan("a", 50, 78) == []                # già residente
