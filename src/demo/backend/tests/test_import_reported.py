import json

import pytest

from demo_api.metrics.reported import ReportedImportError, build_rows, import_reported
from demo_api.registry import Registry
from demo_api.settings import REPO_ROOT


@pytest.fixture(scope="module")
def rows():
    from demo_api.settings import Settings

    reg = Registry(Settings())
    return build_rows({k: v.raw for k, v in reg.models.items()}, REPO_ROOT)


def find(rows, model, set_name, validity=None):
    hits = [r for r in rows if r["model_id"] == model and r["set_name"] == set_name
            and (validity is None or r["validity"] == validity)]
    assert len(hits) == 1, (model, set_name, len(hits))
    return hits[0]


def test_p11_coverage_headline_and_perspective(rows):
    r = find(rows, "apertus-8b-lora-heads-p11", "coverage")
    assert r["macro_f1"] == 0.781 and r["weighted_f1"] == 0.905
    assert r["per_dimension"]["perspective"]["macro_f1"] == 0.625
    assert r["source"].endswith("results_full_h200.md:23") and r["provenance"] == "reported_only"


def test_p11_confusion_matrix_10x10(rows):
    c = find(rows, "apertus-8b-lora-heads-p11", "coverage")["confusion"]["perspective"]
    assert len(c["matrix"]) == 10 and all(len(r) == 10 for r in c["matrix"])
    assert c["labels"][0] == "front_head_on" and sum(c["matrix"][0]) == 614
    assert c["source"].endswith(":125-136") or ":1" in c["source"]


def test_p11_per_class(rows):
    pc = find(rows, "apertus-8b-lora-heads-p11", "coverage")["per_class"]["data"]
    assert set(pc) == {"subject", "background", "perspective", "individuals", "text_classification", "text_language"}
    full = pc["subject"]["full_product"]
    assert full["support"] == 1363 and full["f1"] == 0.95 and full["precision"] == 0.94


def test_zeroshot_8b_coverage(rows):
    r = find(rows, "apertus-8b-zeroshot-v2", "coverage")
    assert r["macro_f1"] == 0.528 and r["per_dimension"]["subject"]["macro_f1"] == 0.471
    data = r["per_class"]["data"]["subject"]["full_product"]
    assert data["f1"] == 0.71 and data["support"] == 1363 and data["precision"] == 0.82 and data["recall"] == 0.62
    assert find(rows, "apertus-70b-zeroshot-v2", "coverage")["macro_f1"] == 0.583


def test_p10_from_json(rows):
    r = find(rows, "apertus-8b-lora-p10", "coverage")
    assert r["macro_f1"] == pytest.approx(0.7341235429748102) and r["checkpoint"] == "final"
    assert r["per_dimension"]["subject"]["support"] > 0
    assert "full_product" in r["per_class"]["data"]["subject"]
    val = find(rows, "apertus-8b-lora-p10", "validation_450")
    assert val["macro_f1"] == pytest.approx(0.696193658322792) and val["validity"] == "held_out"


def test_p8_p9_from_confronto(rows):
    assert find(rows, "apertus-8b-frozen-heads-p8", "validation_450")["macro_f1"] == 0.542
    assert find(rows, "apertus-8b-frozen-heads-p9", "validation_450")["per_dimension"]["perspective"]["macro_f1"] == 0.234


def test_full_ft_has_only_invalid_rows_without_numbers(rows):
    ft = [r for r in rows if r["model_id"] == "apertus-8b-fullft-heads"]
    assert {r["set_name"] for r in ft} == {"coverage", "base", "prompt-test", "validation_450"}
    assert all(r["validity"] == "seen_in_training" and r["macro_f1"] is None for r in ft)


def test_p11_validation_450_is_invalid(rows):
    r = find(rows, "apertus-8b-lora-heads-p11", "validation_450")
    assert r["validity"] == "seen_in_training" and r["macro_f1"] is None


def test_mismatch_between_yaml_and_source_fails(settings):
    reg = Registry(settings)
    models = {k: v.raw for k, v in reg.models.items()}
    models["apertus-8b-zeroshot-v2"] = json.loads(json.dumps(models["apertus-8b-zeroshot-v2"]))
    models["apertus-8b-zeroshot-v2"]["evals"][0]["macro_f1"] = 0.999
    with pytest.raises(ReportedImportError):
        build_rows(models, REPO_ROOT)


def test_import_writes_eval_results(client):
    from demo_api.metrics.reported import import_reported as run

    db = client.app.state.db
    n = run(db, {k: v.raw for k, v in client.app.state.registry.models.items()}, REPO_ROOT)
    assert n == db.one("SELECT COUNT(*) AS c FROM eval_results")["c"] and n > 15
    run(db, {k: v.raw for k, v in client.app.state.registry.models.items()}, REPO_ROOT)  # idempotente
    assert db.one("SELECT COUNT(*) AS c FROM eval_results")["c"] == n
