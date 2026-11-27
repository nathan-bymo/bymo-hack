import pytest

from demo_api import taxonomy
from demo_api.repo_code import load_metric, prompt_module

EXPECTED = {
    "subject": ["full_product", "detail_closeup", "accessories", "packaging", "package_and_product",
                "multipack_trade_unit", "delivery_scope", "product_grouping", "multiview", "energy_label",
                "infographic", "logo", "placeholder"],
    "background": ["white", "color", "scene_lifestyle"],
    "perspective": ["front_head_on", "front_angle_left", "front_angle_right", "back_head_on", "back_angle_left",
                    "back_angle_right", "left_side", "right_side", "top_down", "bottom_up"],
    "individuals": ["individual_present", "none"],
    "text_classification": ["certification_seal", "technical_specs", "promotional_text", "nutritional_info",
                            "other", "none"],
    "text_language": ["German", "English", "French", "Italian", "Dutch", "other", "none"],
}


def test_41_classes_in_6_dimensions_match_expected_strings():
    assert list(taxonomy.dimensions()) == list(EXPECTED)
    assert {d: list(taxonomy.classes(d)) for d in EXPECTED} == EXPECTED
    assert sum(len(v) for v in EXPECTED.values()) == 41


def test_equal_to_prompt_taxonomy():
    assert {d: tuple(v) for d, v in EXPECTED.items()} == dict(prompt_module().TAXONOMY)


def test_equal_to_official_metric_when_present(settings):
    from pathlib import Path

    from demo_api.settings import REPO_ROOT

    path = REPO_ROOT / "data" / "hackapertus-galaxus" / "metric.py"
    if not path.exists():
        pytest.skip("metric.py dei dati Kaggle non presente su questa macchina")
    metric = load_metric(path)
    assert tuple(metric.DIMENSIONS) == taxonomy.dimensions()
    for d in EXPECTED:
        assert set(metric.VALID_CLASSES[d]) == set(EXPECTED[d])


def test_order_and_letters():
    payload = taxonomy.taxonomy_payload()
    subject = payload["dimensions"][0]
    assert subject["key"] == "subject" and subject["erp_label"] == "Subject"
    assert [c["code"] for c in subject["classes"]][:3] == ["A", "B", "C"]
    assert subject["classes"][0]["index"] == 0 and subject["classes"][12]["value"] == "placeholder"


def test_erp_labels_known_and_auto():
    assert taxonomy.erp_label("subject", "full_product") == ("Full Product View", "screenshot")
    assert taxonomy.erp_label("subject", "detail_closeup") == ("Detail Closeup", "auto")
    assert taxonomy.erp_label("text_language", "none") == ("-", "screenshot")
    assert taxonomy.erp_label("text_language", "German") == ("German", "auto")
    assert taxonomy.erp_label("perspective", "front_angle_left")[0] == "Front – Angled Left"


def test_taxonomy_endpoint(client):
    body = client.get("/api/taxonomy").json()
    assert len(body["dimensions"]) == 6
    assert sum(len(d["classes"]) for d in body["dimensions"]) == 41
    assert body["confidence_bins"]["low_below"] == 0.60 and body["relevance_rules"]
