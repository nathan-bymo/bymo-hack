import json

import pytest

from demo_api.repo_code import eval_prompt_module
from fakes import fake_complete
from imgs import make


@pytest.fixture
def ready(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete(calls))
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    files = [("files", (f"{i}.jpg", make("JPEG", color=(i * 40, 10, 10)), "image/jpeg")) for i in range(3)]
    imgs = client.post("/api/images", data={"session_id": sid}, files=files).json()["images"]
    return sid, [i["image_id"] for i in imgs], calls


def run_job(client, response):
    events = []
    with client.stream("GET", response["events_url"]) as stream:
        name = None
        for line in stream.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    return events


def test_predict_job_events_and_results(client, ready):
    sid, ids, calls = ready
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ids})
    assert r.status_code == 202
    body = r.json()
    assert body["total"] == 3 and body["queued"] == 3 and body["cached"] == 0
    events = run_job(client, body)
    names = [n for n, _ in events]
    assert names.count("result") == 3 and names[-1] == "done"
    done = events[-1][1]
    assert done["status"] == "done" and done["batch"]["images"] == 3 and done["batch"]["gpu_name"] == "gateway remoto"
    job = client.get(f"/api/jobs/{body['job_id']}").json()
    assert job["status"] == "done" and job["done"] == 3 and len(job["results"]) == 3
    result = job["results"][0]
    assert len(result["dimensions"]) == 6 and result["valid"] and result["mode"] == "live"
    assert result["dimensions"]["subject"]["erp_label"] == "Full Product View"
    assert result["dimensions"]["subject"]["applicable"] is True
    assert len(calls) == 21  # 3 immagini x 7 richieste


def test_second_identical_predict_is_cached_without_network(client, ready):
    sid, ids, calls = ready
    first = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ids}).json()
    run_job(client, first)
    before = len(calls)
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ids})
    assert r.status_code == 200 and r.json()["cached"] == 3 and r.json()["queued"] == 0
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "done" and all(x["cached"] for x in job["results"])
    assert len(calls) == before


def test_context_change_invalidates_cache(client, ready):
    sid, ids, calls = ready
    run_job(client, client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2",
                                                      "image_ids": ids[:1]}).json())
    client.put(f"/api/images/{ids[0]}/context", json={"brand": "Acme"})
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ids[:1]})
    assert r.status_code == 202 and r.json()["queued"] == 1
    run_job(client, r.json())
    assert len(client.get("/api/predictions", params={"image_id": ids[0]}).json()["predictions"]) == 2


def test_image_detail_lists_latest_prediction(client, ready):
    sid, ids, _ = ready
    run_job(client, client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2",
                                                      "image_ids": ids[:1]}).json())
    detail = client.get(f"/api/images/{ids[0]}").json()
    assert [p["model_id"] for p in detail["predictions"]] == ["apertus-8b-zeroshot-v2"]


def test_unavailable_model_and_unknown_ids(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    img = client.post("/api/images", data={"session_id": sid},
                      files=[("files", ("a.jpg", make("JPEG"), "image/jpeg"))]).json()["images"][0]["image_id"]
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": [img]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "LLM_API_KEY" in r.json()["error"]["details"]["reason"]
    assert client.post("/api/predict", json={"model_id": "nope", "image_ids": [img]}).status_code == 404
    monkeypatch.setenv("LLM_API_KEY", "k")
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ["img-demo-x"]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    assert client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": []}).status_code == 422


def test_upstream_failure_reported_per_image(client, monkeypatch, ready):
    sid, ids, _ = ready
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete(fail="HTTP 500: boom"))
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": ids[:2]})
    events = run_job(client, r.json())
    errors = [d for n, d in events if n == "error"]
    assert len(errors) == 2 and errors[0]["error"]["code"] == "UPSTREAM_ERROR"
    assert events[-1][1]["status"] == "error"
