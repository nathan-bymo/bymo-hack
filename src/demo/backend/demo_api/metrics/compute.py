"""Calcolo delle metriche con il `metric.py` ufficiale dei dati Kaggle (mai copiato né reimplementato).

`metric.evaluate` dà F1/precision/recall per classe e per dimensione; qui si aggiungono solo l'accuratezza, le
matrici di confusione e i dati di calibrazione, che non fanno parte della metrica ufficiale.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ..repo_code import eval_prompt_module, prompt_module

EMPTY = "(empty)"


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def solution_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Soluzione con le sole celle valutate (label umana in una cella rilevante): `eval_prompt.solution_frame`."""
    return eval_prompt_module().solution_frame(rows)


def compute(metric, solution: pd.DataFrame, submission: pd.DataFrame, id_column: str = "image_id") -> dict:
    """`metric` = modulo `metric.py` ufficiale. Restituisce overall, per dimensione (con accuratezza) e confusioni."""
    p = prompt_module()
    result = metric.evaluate(solution, submission, id_column)
    labels = solution.set_index(id_column)
    preds = submission.set_index(id_column)
    per_dimension, confusion = {}, {}
    for dimension in p.DIMENSIONS:
        if dimension not in result:
            continue
        pairs = [(_clean(labels.at[i, dimension]), _clean(preds.at[i, dimension]) if i in preds.index else None)
                 for i in labels.index]
        pairs = [(l, q) for l, q in pairs if l is not None]
        hits = sum(1 for l, q in pairs if l == q)
        r = result[dimension]
        per_dimension[dimension] = {
            "macro_f1": r["f1_macro"], "weighted_f1": r["f1_weighted"], "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"], "support": r["support"], "accuracy": hits / len(pairs),
            "per_class": r["per_class"],
        }
        classes = list(p.TAXONOMY[dimension])
        columns = classes + [EMPTY]
        matrix = [[0] * len(columns) for _ in classes]
        for label, prediction in pairs:
            if label in classes:
                matrix[classes.index(label)][columns.index(prediction if prediction in classes else EMPTY)] += 1
        confusion[dimension] = {"labels": classes, "columns": columns, "matrix": matrix, "source": "recomputed"}
    overall = result["overall"]
    return {
        "overall": {"macro_f1": overall["f1_macro"], "weighted_f1": overall["f1_weighted"],
                    "support": overall["support"]},
        "per_dimension": per_dimension, "confusion": confusion,
    }


def calibration(solution: pd.DataFrame, submission: pd.DataFrame, confidences: pd.DataFrame,
                id_column: str = "image_id", bins: int = 10) -> dict:
    """Confidenza contro correttezza per le celle valutate: 10 fasce di affidabilità e due istogrammi.

    `confidences` ha le stesse colonne di `submission` con la probabilità della classe predetta (NaN = assente).
    """
    p = prompt_module()
    labels, preds, conf = (f.set_index(id_column) for f in (solution, submission, confidences))
    correct, wrong, rows = [], [], []
    for dimension in p.DIMENSIONS:
        for i in labels.index:
            label, prediction, c = _clean(labels.at[i, dimension]), _clean(preds.at[i, dimension]), conf.at[i, dimension]
            if label is None or c is None or (isinstance(c, float) and math.isnan(c)):
                continue
            rows.append((float(c), prediction == label))
            (correct if prediction == label else wrong).append(float(c))
    table = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in rows if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        table.append({"lo": lo, "hi": hi, "n": len(inside),
                      "accuracy": (sum(ok for _, ok in inside) / len(inside)) if inside else None,
                      "mean_confidence": (sum(c for c, _ in inside) / len(inside)) if inside else None})
    return {"bins": table, "correct": correct, "wrong": wrong, "n": len(rows)}
