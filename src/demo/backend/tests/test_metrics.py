"""L'esempio calcolato a mano di piano §9.2 (5 immagini finte). I numeri attesi sono quelli del piano."""

import numpy as np
import pandas as pd
import pytest

from demo_api.metrics.compute import EMPTY, calibration, compute
from demo_api.repo_code import load_metric
from demo_api.settings import REPO_ROOT

DIMS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
METRIC_PATH = REPO_ROOT / "data" / "hackapertus-galaxus" / "metric.py"

IDS = ["e1", "e2", "e3", "e4", "e5"]
BG_LABEL = ["white", "white", "white", "color", "scene_lifestyle"]
BG_PRED = ["white", "white", "color", "color", "white"]
IND_LABEL = ["none", "none", "none", None, None]          # e4, e5: non valutate
IND_PRED = ["none", "individual_present", None, "none", "none"]  # e3: vuota


def frames():
    solution = pd.DataFrame({"image_id": IDS, **{d: [None] * 5 for d in DIMS}})
    submission = pd.DataFrame({"image_id": IDS, **{d: [None] * 5 for d in DIMS}})
    solution["background"], submission["background"] = BG_LABEL, BG_PRED
    solution["individuals"], submission["individuals"] = IND_LABEL, IND_PRED
    return solution, submission


def reference_f1(pairs, valid_classes):
    """Seconda implementazione, solo nei test (stessa convenzione di scikit-learn): serve a controllare metric.py."""
    labels = {l for l, _ in pairs}
    classes = sorted(labels | ({p for _, p in pairs if p} & valid_classes))
    f1s, weighted = [], 0.0
    for c in classes:
        tp = sum(1 for l, p in pairs if l == c and p == c)
        pp = sum(1 for _, p in pairs if p == c)
        support = sum(1 for l, _ in pairs if l == c)
        prec, rec = (tp / pp if pp else 0.0), (tp / support if support else 0.0)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        weighted += f1 * support
    return sum(f1s) / len(f1s), weighted / len(pairs)


def test_reference_implementation_gives_plan_numbers():
    bg = [(l, p) for l, p in zip(BG_LABEL, BG_PRED)]
    ind = [(l, p) for l, p in zip(IND_LABEL, IND_PRED) if l is not None]
    bg_macro, bg_w = reference_f1(bg, {"white", "color", "scene_lifestyle"})
    ind_macro, ind_w = reference_f1(ind, {"individual_present", "none"})
    assert bg_macro == pytest.approx(0.4444, abs=1e-4) and bg_w == pytest.approx(0.5333, abs=1e-4)
    assert ind_macro == pytest.approx(0.25) and ind_w == pytest.approx(0.5)
    assert (bg_macro + ind_macro) / 2 == pytest.approx(0.3472, abs=1e-4)
    assert (bg_w * 5 + ind_w * 3) / 8 == pytest.approx(0.5208, abs=1e-4)


@pytest.mark.skipif(not METRIC_PATH.exists(), reason="metric.py dei dati Kaggle non presente: resta il test di riferimento")
def test_official_metric_on_hand_example():
    solution, submission = frames()
    out = compute(load_metric(METRIC_PATH), solution, submission)
    assert out["overall"]["macro_f1"] == pytest.approx(0.3472, abs=1e-4)
    assert out["overall"]["weighted_f1"] == pytest.approx(0.5208, abs=1e-4)
    assert out["overall"]["support"] == 8
    bg, ind = out["per_dimension"]["background"], out["per_dimension"]["individuals"]
    assert bg["macro_f1"] == pytest.approx(0.4444, abs=1e-4) and bg["weighted_f1"] == pytest.approx(0.5333, abs=1e-4)
    assert ind["macro_f1"] == pytest.approx(0.25) and ind["weighted_f1"] == pytest.approx(0.5)
    assert bg["accuracy"] == pytest.approx(0.6) and ind["accuracy"] == pytest.approx(1 / 3)
    assert set(out["per_dimension"]) == {"background", "individuals"}  # le altre non hanno celle valutate


@pytest.mark.skipif(not METRIC_PATH.exists(), reason="metric.py dei dati Kaggle non presente")
def test_confusion_of_hand_example():
    solution, submission = frames()
    conf = compute(load_metric(METRIC_PATH), solution, submission)["confusion"]
    bg = conf["background"]
    assert bg["labels"] == ["white", "color", "scene_lifestyle"] and bg["columns"][-1] == EMPTY
    assert bg["matrix"] == [[2, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    ind = conf["individuals"]
    # righe: individual_present, none; colonne: individual_present, none, (empty)
    assert ind["matrix"] == [[0, 0, 0], [1, 1, 1]]
    assert sum(map(sum, ind["matrix"])) == 3  # solo le celle valutate


def test_calibration_bins():
    solution, submission = frames()
    conf = submission.copy()
    for d in DIMS:
        conf[d] = np.nan
    conf["background"] = [0.95, 0.85, 0.55, 0.65, 0.15]
    out = calibration(solution, submission, conf)
    assert out["n"] == 5 and len(out["bins"]) == 10
    assert sorted(out["correct"]) == [0.65, 0.85, 0.95] and sorted(out["wrong"]) == [0.15, 0.55]
    assert out["bins"][9]["accuracy"] == 1.0 and out["bins"][5]["accuracy"] == 0.0
