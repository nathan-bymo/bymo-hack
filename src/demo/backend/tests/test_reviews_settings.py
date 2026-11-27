import json

import pytest

from demo_api.repo_code import eval_prompt_module
from fakes import fake_complete
from imgs import make


@pytest.fixture
def predicted(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete())
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    img = client.post("/api/images", data={"session_id": sid},
                      files=[("files", ("a.jpg", make("JPEG"), "image/jpeg"))]).json()["images"][0]
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": [img["image_id"]]}).json()
    with client.stream("GET", r["events_url"]) as s:
        list(s.iter_lines())
    pred = client.get(f"/api/jobs/{r['job_id']}").json()["results"][0]
    return img["image_id"], pred


def values_of(pred, **override):
    out = {d: v["label"] for d, v in pred["dimensions"].items()}
    out.update(override)
    return out


def test_review_unchanged_is_reviewed(client, predicted):
    image_id, pred = predicted
    r = client.put(f"/api/images/{image_id}/review",
                   json={"prediction_id": pred["prediction_id"], "values": values_of(pred)})
    assert r.status_code == 200 and r.json()["status"] == "reviewed" and r.json()["changed_dimensions"] == []


def test_review_with_change_is_corrected(client, predicted):
    image_id, pred = predicted
    r = client.put(f"/api/images/{image_id}/review", json={
        "prediction_id": pred["prediction_id"], "values": values_of(pred, perspective="left_side"), "note": "x"})
    assert r.json()["status"] == "corrected" and r.json()["changed_dimensions"] == ["perspective"]
    detail = client.get(f"/api/images/{image_id}").json()
    assert detail["review"]["status"] == "corrected" and detail["review"]["values"]["perspective"] == "left_side"
    assert len(client.get(f"/api/images/{image_id}/reviews").json()["reviews"]) == 1


def test_review_validates_values(client, predicted):
    image_id, pred = predicted
    bad = client.put(f"/api/images/{image_id}/review",
                     json={"prediction_id": pred["prediction_id"], "values": values_of(pred, subject="banana")})
    assert bad.status_code == 422 and bad.json()["error"]["details"]["invalid"] == {"subject": "banana"}
    short = client.put(f"/api/images/{image_id}/review", json={"values": {"subject": "logo"}})
    assert short.status_code == 422 and "background" in short.json()["error"]["details"]["missing"]


def test_no_training_export_endpoint(client):
    for path in ("/api/reviews", "/api/export/training", "/api/training"):
        assert client.get(path).status_code == 404


def test_settings_defaults_and_partial_update(client):
    s = client.get("/api/settings").json()
    assert s["usd_per_hour"] == 2.5 and s["show_ground_truth"] is False and s["zero_shot_parallel_images"] == 8
    s2 = client.put("/api/settings", json={"usd_per_hour": 5.0, "gpu_label": "H200"}).json()
    assert s2["usd_per_hour"] == 5.0 and s2["gpu_label"] == "H200" and s2["presentation_mode"] is False
    assert client.put("/api/settings", json={"bogus": 1}).status_code == 422


def test_image_list_carries_review_status(client, predicted):
    image_id, pred = predicted
    sid = client.get(f"/api/images/{image_id}").json()["session_id"]
    assert client.get("/api/images", params={"session_id": sid}).json()["images"][0]["review_status"] is None
    client.put(f"/api/images/{image_id}/review",
               json={"prediction_id": pred["prediction_id"], "values": values_of(pred, background="color")})
    item = client.get("/api/images", params={"session_id": sid}).json()["images"][0]
    assert item["review_status"] == "corrected" and item["review_changed"] == ["background"]
