import pytest

from imgs import make


def test_dev_and_final_rejected(client):
    for bad in ("dev", "final", "bogus"):
        r = client.get("/api/metrics/apertus-8b-lora-heads-p11", params={"set": bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    r = client.get("/api/metrics/compare", params={"models": "apertus-8b-lora-heads-p11", "set": "dev"})
    assert r.status_code == 422


def test_full_ft_has_no_valid_results_and_contamination_note(client):
    body = client.get("/api/metrics/apertus-8b-fullft-heads", params={"set": "coverage"}).json()
    assert body["results"] == [] and body["excluded_invalid"] == 1
    assert "11.876" in body["contamination_note"] and body["kaggle_dev"]["score"] == 0.80473
    shown = client.get("/api/metrics/apertus-8b-fullft-heads", params={"set": "coverage", "include_invalid": True}).json()
    row = shown["results"][0]
    assert row["validity"] == "seen_in_training" and row["macro_f1"] is None
    assert row["badges"] == ["seen during training — invalid"] and row["badge_tooltip"]


def test_p11_coverage_badges_and_numbers(client):
    body = client.get("/api/metrics/apertus-8b-lora-heads-p11", params={"set": "coverage"}).json()
    r = body["results"][0]
    assert r["macro_f1"] == 0.781 and r["badges"] == ["phase-1 internal set (coverage)", "reported only"]
    assert r["source"].endswith("results_full_h200.md:23") and "9.344" in r["trained_on"]
    assert r["confusion"]["perspective"]["labels"][0] == "front_head_on"
    assert body["kaggle_dev"]["score"] == 0.796 and body["kaggle_dev"]["badge"] == "Kaggle dev (public)"
    assert body["kaggle_dev"]["source"] == "src/demo/config/sources/EXPERIMENTS.md:78"


def test_zeroshot_prompt_test_is_held_out(client):
    r = client.get("/api/metrics/apertus-70b-zeroshot-v2", params={"set": "prompt-test"}).json()["results"][0]
    assert r["macro_f1"] == 0.59 and r["badges"] == ["held-out", "reported only"]


def test_compare_excludes_seen_in_training_by_default(client):
    ids = "apertus-8b-lora-heads-p11,apertus-8b-fullft-heads,apertus-8b-zeroshot-v2"
    body = client.get("/api/metrics/compare", params={"models": ids, "set": "coverage"}).json()
    assert body["models"]["apertus-8b-fullft-heads"]["results"] == []
    assert len(body["models"]["apertus-8b-lora-heads-p11"]["results"]) == 1
    inc = client.get("/api/metrics/compare", params={"models": ids, "set": "coverage", "include_invalid": True}).json()
    assert len(inc["models"]["apertus-8b-fullft-heads"]["results"]) == 1
    assert client.get("/api/metrics/nope").status_code == 404


def test_p11_validation_450_is_invalid(client):
    r = client.get("/api/metrics/apertus-8b-lora-heads-p11", params={"set": "validation_450", "include_invalid": True}).json()
    assert [x["validity"] for x in r["results"]] == ["seen_in_training"]


def test_throughput_reported_cost_matches_repo_control(client):
    body = client.get("/api/throughput", params={"model_id": "apertus-8b-lora-heads-p11"}).json()
    rep = body["reported"][0]
    assert rep["value"] == 13.2 and rep["gpu_label"] == "H200 (HF)" and rep["usd_per_hour"] == 5.0
    assert rep["cost_per_1k_images_usd"] == pytest.approx(0.105, abs=5e-4)  # results_full_h200.md:159-160
    assert body["cost_per_1k_images_usd"] == pytest.approx(0.105, abs=5e-4) and body["target_images_per_second"] == 2
    p10 = client.get("/api/throughput", params={"model_id": "apertus-8b-lora-p10"}).json()
    assert p10["reported"][0]["cost_per_1k_images_usd"] is None and p10["cost_per_1k_images_usd"] is None


def test_throughput_gateway_has_no_cost(client, monkeypatch):
    from demo_api.repo_code import eval_prompt_module
    from fakes import fake_complete

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete())
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    img = client.post("/api/images", data={"session_id": sid},
                      files=[("files", ("a.jpg", make("JPEG"), "image/jpeg"))]).json()["images"][0]
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": [img["image_id"]]}).json()
    with client.stream("GET", r["events_url"]) as s:
        list(s.iter_lines())
    body = client.get("/api/throughput", params={"model_id": "apertus-8b-zeroshot-v2"}).json()
    m = body["measured_in_demo"][0]
    assert m["gpu_name"] == "gateway remoto" and m["cost_per_1k_images_usd"] is None and m["mode"] == "live"
    assert "not public" in m["cost_note"] and body["cost_per_1k_images_usd"] is None


def test_export_csv(client, monkeypatch):
    from demo_api.repo_code import eval_prompt_module
    from fakes import fake_complete

    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete())
    sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
    files = [("files", (f"{i}.jpg", make("JPEG", color=(i * 60, 1, 1)), "image/jpeg")) for i in range(2)]
    imgs = client.post("/api/images", data={"session_id": sid, "source_kind": "train_coverage"}, files=files).json()["images"]
    r = client.post("/api/predict", json={"model_id": "apertus-8b-zeroshot-v2", "image_ids": [i["image_id"] for i in imgs]}).json()
    with client.stream("GET", r["events_url"]) as s:
        list(s.iter_lines())
    pred = client.get(f"/api/jobs/{r['job_id']}").json()["results"][0]
    values = {d: v["label"] for d, v in pred["dimensions"].items()}
    values["perspective"] = "top_down"
    client.put(f"/api/images/{pred['image_id']}/review", json={"prediction_id": pred["prediction_id"], "values": values})
    resp = client.get("/api/export.csv", params={"session_id": sid})
    assert resp.headers["content-disposition"].startswith("attachment") and "X-Demo-Warning" in resp.headers
    lines = resp.text.strip().splitlines()
    assert lines[0] == "image_id,subject,background,perspective,individuals,text_classification,text_language,status,min_confidence,model_id,mode"
    by = {l.split(",")[0]: l.split(",") for l in lines[1:]}
    assert by[pred["image_id"]][3] == "top_down" and by[pred["image_id"]][7] == "corrected"
    other = next(v for k, v in by.items() if k != pred["image_id"])
    assert other[7] == "ai_suggested" and other[3] == "front_head_on"
    assert "original_filename" not in lines[0]
    assert "original_filename" in client.get("/api/export.csv", params={"session_id": sid, "include_filenames": True}).text
