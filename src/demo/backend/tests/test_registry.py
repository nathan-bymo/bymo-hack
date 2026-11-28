import pytest
import yaml

from demo_api.main import create_app
from demo_api.registry import Registry, RegistryError
from demo_api.settings import REPO_ROOT


def test_models_without_env_are_unavailable_with_readable_reasons(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    body = client.get("/api/models").json()
    by_id = {m["model_id"]: m for m in body["models"]}
    assert len(by_id) == 8
    for zid in ("apertus-8b-zeroshot-v2", "apertus-70b-zeroshot-v2"):
        assert by_id[zid]["status"] == "unavailable" and "LLM_API_KEY" in by_id[zid]["status_reason"]
    for mid in ("apertus-8b-lora-p10", "apertus-8b-lora-heads-p11", "apertus-8b-fullft-heads",
                "apertus-8b-vadapter-lora-heads-final"):
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
    assert full["kaggle_dev"]["score"] == 0.80473 and full["memory"]["gb"] == 18.7


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


def test_parity_flag_in_summary(client):
    by = {m["model_id"]: m for m in client.get("/api/models").json()["models"]}
    assert by["apertus-8b-zeroshot-v2"]["parity"]["verified"] is True
    for mid in ("apertus-8b-lora-p10", "apertus-8b-lora-heads-p11", "apertus-8b-fullft-heads"):
        assert by[mid]["parity"]["verified"] is False and "parity" in by[mid]["parity"]["scope"]


def test_scores_speeds_and_cost_per_100k(client):
    """Ogni punteggio porta il suo set; il costo si calcola solo da velocità end to end su una GPU nota."""
    body = client.get("/api/models").json()
    by_id = {m["model_id"]: m for m in body["models"]}
    final = by_id["apertus-8b-vadapter-lora-heads-final"]
    assert [(s["set"], s["macro_f1"]) for s in final["scores"]] == [("kaggle_dev", 0.83177), ("prompt-test", 0.821)]
    assert final["featured"] and final["weights"]["local_path"].endswith("apertus-8B-vadapter-lora-heads-final")
    costs = {s["gpu_label"]: s["cost_per_100k_usd"] for s in final["speeds"]}
    assert costs == {"H200 (HF)": round(5.0 / (6.2 * 3600) * 100_000, 2), "L40S (HF)": round(1.8 / (4.9 * 3600) * 100_000, 2)}
    p11 = by_id["apertus-8b-lora-heads-p11"]
    assert p11["speeds"][0]["cost_per_100k_usd"] == 10.52 and p11["memory"]["kind"] == "measured"
    ft = by_id["apertus-8b-fullft-heads"]
    assert ft["speeds"][0]["cost_per_100k_usd"] == 7.27 and ft["scores"][0]["set"] == "kaggle_dev"
    zs = by_id["apertus-8b-zeroshot-v2"]
    assert zs["speeds"][0]["cost_per_100k_usd"] is None and "gateway" in zs["speeds"][0]["cost_note"]
    p10 = by_id["apertus-8b-lora-p10"]
    assert p10["speeds"][0]["cost_per_100k_usd"] is None and "end to end" in p10["speeds"][0]["cost_note"]
    assert [r["score"] for r in body["kaggle_references"]] == [0.917, 0.834, 0.806]


def test_final_model_prompt_test_read_from_italian_report(client):
    """0,821 (virgola decimale) riletto dalla riga del report di Nathan; coverage segnato come visto in training."""
    r = client.get("/api/metrics/apertus-8b-vadapter-lora-heads-final", params={"set": "prompt-test"}).json()["results"][0]
    assert r["macro_f1"] == 0.821 and r["weighted_f1"] == 0.917 and r["validity"] == "held_out"
    assert r["per_dimension"]["perspective"]["macro_f1"] == 0.755
    cov = client.get("/api/metrics/apertus-8b-vadapter-lora-heads-final", params={"set": "coverage", "include_invalid": True}).json()
    assert [x["validity"] for x in cov["results"]] == ["seen_in_training"]


def test_final_model_has_a_worker_family_without_gpu():
    """Il modello finale ha la sua famiglia di worker; senza GPU il caricamento fallisce con un errore chiaro."""
    import pytest as _pytest

    from demo_worker.families import get_family
    from demo_worker.families.base import WorkerError

    family = get_family("vadapter_heads", False)
    assert family.name == "vadapter_heads" and family.loaded() == []
    torch = _pytest.importorskip("torch")
    _pytest.importorskip("transformers")
    if torch.cuda.is_available():
        _pytest.skip("GPU presente: il caricamento reale si verifica con scripts/parity.py")
    with _pytest.raises(WorkerError):
        family.load("apertus-8b-vadapter-lora-heads-final", {"weights": {}, "base": {}}, lambda s, m: None)
