import pytest
import yaml

from demo_api.main import create_app
from demo_api.registry import Registry, RegistryError
from demo_api.settings import REPO_ROOT


def test_models_without_env_are_unavailable_with_readable_reasons(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    body = client.get("/api/models").json()
    by_id = {m["model_id"]: m for m in body["models"]}
    assert len(by_id) == 7
    for zid in ("apertus-8b-zeroshot-v2", "apertus-70b-zeroshot-v2"):
        assert by_id[zid]["status"] == "unavailable" and "LLM_API_KEY" in by_id[zid]["status_reason"]
    assert "weights.repo" in by_id["apertus-8b-lora-heads-p11"]["status_reason"]
    for mid in ("apertus-8b-lora-p10", "apertus-8b-fullft-heads"):
        assert by_id[mid]["status"] == "unavailable" and "GPU worker" in by_id[mid]["status_reason"]
    for mid in ("apertus-8b-frozen-heads-p8", "apertus-8b-frozen-heads-p9"):
        assert by_id[mid]["status"] == "unavailable" and "predittore" in by_id[mid]["status_reason"]
    assert all(m["modes"] == [] for m in by_id.values())
    assert body["reference_line"]["value"] == 0.917


def test_zeroshot_ready_with_key(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    m = client.get("/api/models/apertus-8b-zeroshot-v2").json()
    assert m["status"] == "ready" and m["modes"] == ["live"]
    assert "test-key" not in str(m)


def test_headline_eval_and_detail(client):
    m = client.get("/api/models/apertus-8b-lora-heads-p11").json()
    assert m["headline_eval"]["macro_f1"] == 0.781 and m["headline_eval"]["validity"] == "phase1_internal"
    assert len(m["evals"]) == 3 and m["vram_estimate_kind"] == "measured"
    full = client.get("/api/models/apertus-8b-fullft-heads").json()
    assert full["headline_eval"] is None and full["contamination_note"]
    assert full["kaggle_dev"]["score"] is None


def test_model_not_found(client):
    r = client.get("/api/models/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "MODEL_NOT_FOUND"


def _write(tmp_path, mutate):
    data = yaml.safe_load((REPO_ROOT / "src" / "demo" / "models.yaml").read_text())
    mutate(data)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_missing_source_file_fails_startup(settings, tmp_path):
    def mutate(d):
        d["models"][0]["evals"][0]["source"] = "src/does/not/exist.md:3"
    settings.models_yaml = _write(tmp_path, mutate)
    with pytest.raises(RegistryError, match="non esiste"):
        Registry(settings)


def test_invalid_kaggle_score_fails(settings, tmp_path):
    def mutate(d):
        d["models"][0]["kaggle_dev"]["score"] = 1.5
    settings.models_yaml = _write(tmp_path, mutate)
    with pytest.raises(RegistryError, match="kaggle_dev"):
        Registry(settings)


def test_app_refuses_to_start_with_bad_registry(settings, tmp_path):
    from fastapi.testclient import TestClient

    settings.models_yaml = _write(tmp_path, lambda d: d["models"][1]["evals"][0].update(source="x/y.md:1"))
    with pytest.raises(RegistryError):
        with TestClient(create_app(settings)):
            pass


def test_worker_family_without_placeholders_needs_worker_url(settings, monkeypatch):
    reg = Registry(settings)
    assert reg.live_status(reg.get("apertus-8b-lora-p10"))[0] == "unavailable"
    settings.worker_urls["lora_generative"] = "http://127.0.0.1:8011"
    assert reg.live_status(reg.get("apertus-8b-lora-p10")) == ("not_loaded", None)
