"""Shared helpers for the dataset analyses: rows, splits, scored cells and the leaderboard macro F1.

Every analysis prints aggregate numbers only (counts, scores, intervals). No image or product id is printed or saved.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "scripts"), str(ROOT / "src" / "training" / "phase-3" / "src" / "training")]
from build_teacher_targets import macro_f1  # noqa: E402  same definition as data/metric.py
from common import splits  # noqa: E402
from ensemble import DIMENSIONS, TAXONOMY  # noqa: E402

SEED = 3407


def train_rows():
    return pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False)


def prompt_test(rows):
    return splits.split_rows(rows, "test")


def outside_prompt_test(rows):
    """Every labelled row whose product is not in prompt-test: the fit pool of the phase-three models."""
    test_products = set(prompt_test(rows).product_id)
    return rows[~rows.product_id.isin(test_products)].reset_index(drop=True)


def scored(rows):
    return splits.scored_cells(rows)


def overall(rows, predictions, sample=None):
    """Leaderboard macro F1 (mean over dimensions) and the per-dimension values, optionally on a row sample."""
    sample = np.arange(len(rows)) if sample is None else sample
    mask = scored(rows)
    values = {}
    for d in DIMENSIONS:
        keep = mask[d].to_numpy()[sample]
        values[d] = macro_f1(rows[d].to_numpy()[sample][keep], np.asarray(predictions[d])[sample][keep], set(TAXONOMY[d]))
    return float(np.mean(list(values.values()))), values


def product_bootstrap(products, count, seed=SEED):
    """Row-index samples drawn by resampling whole products with replacement."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(products, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    for _ in range(count):
        picked = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([members[i] for i in picked])


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in body]
    return "\n".join(lines)
