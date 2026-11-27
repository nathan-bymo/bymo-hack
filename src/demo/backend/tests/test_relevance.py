from demo_api.relevance import applicable

BASE = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
        "individuals": "none", "text_classification": "none", "text_language": "none"}


def with_(**kw):
    return {**BASE, **kw}


def test_everything_applicable_for_plain_product():
    assert all(applicable(with_(text_classification="promotional_text")).values())
    assert all(applicable(BASE).values())  # full_product: la lingua resta applicabile


def test_logo_placeholder_energy_label_hide_four_dimensions():
    for subject in ("logo", "placeholder", "energy_label"):
        r = applicable(with_(subject=subject))
        assert [d for d, ok in r.items() if not ok][:4] == ["background", "perspective", "individuals",
                                                            "text_classification"]
        assert r["subject"] is True


def test_infographic_hides_three():
    r = applicable(with_(subject="infographic", text_classification="technical_specs"))
    assert (r["background"], r["perspective"], r["individuals"]) == (False, False, False)
    assert r["text_classification"] and r["text_language"]


def test_perspective_hidden_for_three_subjects():
    for subject in ("delivery_scope", "detail_closeup", "product_grouping"):
        r = applicable(with_(subject=subject))
        assert r["perspective"] is False and r["background"] is True


def test_scene_lifestyle_hides_subject_and_perspective():
    r = applicable(with_(background="scene_lifestyle"))
    assert (r["subject"], r["perspective"]) == (False, False)


def test_language_rules():
    assert applicable(with_(subject="packaging"))["text_language"] is True  # senza overlay
    assert applicable(with_(subject="accessories", text_classification="none"))["text_language"] is False
    assert applicable(with_(subject="accessories", text_classification="other"))["text_language"] is True


def test_invalid_labels_keep_dimensions_applicable():
    r = applicable({d: None for d in BASE})
    assert all(r.values())
