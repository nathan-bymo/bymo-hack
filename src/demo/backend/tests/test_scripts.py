"""Script offline (evaluate, record, fetch_weights) con la famiglia finta e un dataset sintetico: nessuna GPU, nessun dato reale."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from imgs import make

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
METRIC = Path(__file__).resolve().parents[4] / "data" / "hackapertus-galaxus" / "metric.py"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIMS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
LABELS = {"subject": "full_product", "background": "white", "perspective": "front_head_on", "individuals": "none",
          "text_classification": "none", "text_language": "none"}


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """train.csv sintetico con 12 prodotti nel bucket 'test' (prompt-test) e le loro immagini."""
    from common.splits import bucket

    kaggle = tmp_path / "kaggle"
    (kaggle / "train_images").mkdir(parents=True)
    rows, n = [], 0
    while len([r for r in rows if bucket(r["product_id"]) == 1]) < 12:
        pid = f"prd_syn{n}"
        n += 1
        rid = f"syn_{n:05d}"
        (kaggle / "train_images" / f"{rid}.jpg").write_bytes(make("JPEG", color=(n * 7 % 255, 20, 20)))
        row = {"image_id": rid, "product_id": pid, "image_path": f"train_images/{rid}.jpg", "product_title": "T",
               "product_type": "", "brand": "", "category": "", "subcategory": ""}
        for d in DIMS:
            row.update({d: LABELS[d], f"{d}_source": "human", f"{d}_relevant": "true"})
        rows.append(row)
    pd.DataFrame(rows).to_csv(kaggle / "train.csv", index=False)
    monkeypatch.setenv("DEMO_KAGGLE_DIR", str(kaggle))
    monkeypatch.setenv("DEMO_METRIC_PATH", str(METRIC))
    monkeypatch.setenv("DEMO_DATA_DIR", str(tmp_path / "demo"))
    monkeypatch.setenv("DEMO_WORKER_ALLOW_DRY", "1")
    return tmp_path


def test_evaluate_refuses_dev_final_unknown_and_contaminated(synthetic):
    ev = script("evaluate")
    for bad in ("dev", "final", "bogus"):
        with pytest.raises(SystemExit) as e:
            ev.main(["--model", "apertus-8b-lora-p10", "--set", bad, "--dry"])
        assert "rifiutato" in str(e.value) or "sconosciuto" in str(e.value)
    with pytest.raises(SystemExit) as e:
        ev.main(["--model", "apertus-8b-fullft-heads", "--set", "coverage", "--dry"])
    assert "non valida" in str(e.value)
    with pytest.raises(SystemExit):
        ev.main(["--model", "nope", "--set", "coverage"])


@pytest.mark.skipif(not METRIC.exists(), reason="metric.py ufficiale non presente")
def test_evaluate_writes_outputs_and_api_reads_them(synthetic, capsys):
    ev = script("evaluate")
    ev.main(["--model", "apertus-8b-lora-p10", "--set", "prompt-test", "--dry", "--batch-size", "4"])
    out = synthetic / "demo" / "eval" / "apertus-8b-lora-p10" / "prompt-test"
    assert {p.name for p in out.iterdir()} == {"predictions.csv", "probs.json", "metrics.json", "run.json"}
    run = json.loads((out / "run.json").read_text())
    assert run["validity"] == "phase1_internal" and run["images"] == 12 and run["predicted"] == 12
    metrics = json.loads((out / "metrics.json").read_text())
    assert 0 <= metrics["overall"]["macro_f1"] <= 1 and "confusion" in metrics and "calibration" in metrics
    assert "macro F1" in capsys.readouterr().out
    # l'API legge i risultati ricalcolati e li mette per primi
    from fastapi.testclient import TestClient

    from demo_api.main import create_app
    from demo_api.settings import Settings

    with TestClient(create_app(Settings.from_env())) as client:
        r = client.get("/api/metrics/apertus-8b-lora-p10", params={"set": "prompt-test"}).json()["results"]
        assert r[0]["provenance"] == "offline_script" and r[0]["badges"] == ["phase-1 internal set (coverage)", "recomputed"]
        assert r[0]["confusion"]["subject"]["labels"][0] == "full_product"


@pytest.mark.skipif(not METRIC.exists(), reason="metric.py ufficiale non presente")
def test_evaluate_allow_contaminated_marks_invalid(synthetic):
    ev = script("evaluate")
    ev.main(["--model", "apertus-8b-fullft-heads", "--set", "prompt-test", "--dry", "--allow-contaminated"])
    run = json.loads((synthetic / "demo" / "eval" / "apertus-8b-fullft-heads" / "prompt-test" / "run.json").read_text())
    assert run["validity"] == "seen_in_training" and run["notes"]


def test_export_record_rescan_predict_chain(synthetic, tmp_path):
    from fastapi.testclient import TestClient

    from demo_api.main import create_app
    from demo_api.settings import Settings

    settings = Settings.from_env()
    with TestClient(create_app(settings)) as client:
        sid = client.post("/api/sessions", json={"name": "t"}).json()["session_id"]
        files = [("files", (f"{i}.jpg", make("JPEG", color=(i * 70, 5, 5)), "image/jpeg")) for i in range(2)]
        imgs = client.post("/api/images", data={"session_id": sid}, files=files).json()["images"]
        export = script("export_session").export(settings.data_dir, sid)
        script("record").main(["--model", "apertus-8b-lora-p10", "--export", str(export), "--dry", "--run-id", "r1"])
        assert client.post("/api/recorded/rescan").json() == {"runs": 1}
        r = client.post("/api/predict", json={"model_id": "apertus-8b-lora-p10", "mode": "recorded",
                                              "image_ids": [i["image_id"] for i in imgs]})
        with client.stream("GET", r.json()["events_url"]) as s:
            list(s.iter_lines())
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert job["status"] == "done" and len(job["results"]) == 2 and job["results"][0]["mode"] == "recorded"
        assert job["results"][0]["recorded"]["gpu_name"] == "DRY (no GPU)"


def test_fetch_weights_plan_never_downloads(capsys, monkeypatch):
    fw = script("fetch_weights")
    from demo_api.registry import Registry
    from demo_api.settings import Settings

    reg = Registry(Settings())
    items = fw.plan(reg, "apertus-8b-lora-p10")
    assert [i["repo"] for i in items] == ["swiss-ai/Apertus-v1.5-8B", "StoryofLegend/apertus-8b-galaxus-lora"]
    assert items[1]["patterns"] == ["prototype-256/adapters/step-00738/*"] and items[1]["gb"] == 0.319
    p11 = fw.plan(reg, "apertus-8b-lora-heads-p11")   # pesi nel repo (Git LFS): si scarica solo la base
    assert [i["repo"] for i in p11] == ["swiss-ai/Apertus-v1.5-8B"]
    with pytest.raises(SystemExit):
        fw.plan(reg, "apertus-8b-zeroshot-v2")         # nessun peso
    monkeypatch.setattr("sys.argv", ["fetch_weights", "--model", "apertus-8b-lora-p10"])
    fw.main()
    assert "Nessun download eseguito" in capsys.readouterr().out
