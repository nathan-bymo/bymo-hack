"""Rows, splits, labels and training items. Pure pandas/torch: no model, runs on a CPU.

Parts of train.csv (src/AGENTS.md):
  training   phase 1: common.splits.training_rows, never a product of the coverage set;
  all        phase 2: every labeled image, with the settings chosen in phase 1;
  coverage   evaluation set of phase 1, only predicted at the end;
  a csv      a smaller split with the columns of train.csv (--train-csv); it must not touch coverage either.

Repeats and mirrors follow the LoRA baseline (apertus-8B-benchmark/lora/apertus_lora.py), so that the two methods
see the same items: rare classes are shown up to 3 times (target 200), and an image with a human left/right
perspective and no text also gets one mirrored copy with left and right swapped. Repeats are identical copies:
cropping them would turn a full_product into a detail_closeup without changing its label.
"""

import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from common import prompt, splits  # noqa: E402

DIMENSIONS = prompt.DIMENSIONS
TAXONOMY = prompt.TAXONOMY
CLASS_COUNTS = [len(TAXONOMY[d]) for d in DIMENSIONS]
PERSPECTIVE = DIMENSIONS.index("perspective")

MIRRORED = {
    "front_angle_left": "front_angle_right", "front_angle_right": "front_angle_left",
    "back_angle_left": "back_angle_right", "back_angle_right": "back_angle_left",
    "left_side": "right_side", "right_side": "left_side",
}
_INDEX = {d: {name: i for i, name in enumerate(TAXONOMY[d])} for d in DIMENSIONS}
MIRROR_PERMUTATION = torch.tensor([_INDEX["perspective"][MIRRORED.get(name, name)]
                                   for name in TAXONOMY["perspective"]])

RARE_CLASS_TARGET = 200  # a row is repeated until its rarest human class gets about this many items per epoch
MAX_REPEATS = 3
LABEL_CELLS = ("human", "scored")


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def select(rows, part):
    if part == "training":
        return splits.training_rows(rows)
    if part == "coverage":
        return splits.split_rows(rows, "coverage")
    if part == "all":
        return rows.reset_index(drop=True)
    raise ValueError(f"unknown part {part!r}")


def assert_clean(rows, all_rows):
    """Phase 1: no training product may be in the coverage set."""
    reserved = set(select(all_rows, "coverage").product_id)
    shared = set(rows.product_id) & reserved
    if shared:
        raise ValueError(f"{len(shared)} training products are in the coverage set")


def internal_split(rows, fraction, salt="vlh-val:"):
    """(fit, validation) by product hash; fraction 0 leaves the validation empty."""
    is_validation = rows.product_id.map(lambda p: int(splits.digest(salt + p), 16) % 1000 < 1000 * fraction)
    return rows[~is_validation].reset_index(drop=True), rows[is_validation].reset_index(drop=True)


def label_matrix(rows, cells="scored"):
    """LongTensor [rows, 6] of class indices, -1 where the cell does not reach the loss.

    "scored": human label in a relevant cell, as the metric counts it. "human": every human label, relevant or
    not, which is what the LoRA baseline trains on.
    """
    if cells == "scored":
        keep = splits.scored_cells(rows)
    elif cells == "human":
        keep = {d: rows[f"{d}_source"] == "human" for d in DIMENSIONS}
    else:
        raise ValueError(f"unknown label cells {cells!r}")
    columns = [[_INDEX[d][value] if known else -1 for value, known in zip(rows[d], keep[d])] for d in DIMENSIONS]
    return torch.tensor(columns, dtype=torch.long).T.contiguous()


def mirror_labels(labels):
    """Labels of the mirrored image: left and right perspectives swapped."""
    labels = labels.clone()
    known = labels[..., PERSPECTIVE] >= 0
    labels[..., PERSPECTIVE][known] = MIRROR_PERMUTATION[labels[..., PERSPECTIVE][known]]
    return labels


def is_mirrorable(row):
    """A human left/right perspective on an image without text, which a mirror would garble."""
    return (row["perspective_source"] == "human" and row["perspective"] in MIRRORED
            and row["text_classification"] == "none" and row["text_language"] == "none")


def key(image_id, variant=""):
    """Cache key of an image variant: "" original, "m" mirrored."""
    return f"{image_id}|{variant}" if variant else image_id


def repeat_counts(records):
    """How often each row is shown per epoch, from the rarest of its human classes."""
    human = [[d for d in DIMENSIONS if r[f"{d}_source"] == "human"] for r in records]
    counts = Counter((d, r[d]) for r, dimensions in zip(records, human) for d in dimensions)
    result = []
    for r, dimensions in zip(records, human):
        rarest = min((counts[(d, r[d])] for d in dimensions), default=RARE_CLASS_TARGET)
        result.append(min(MAX_REPEATS, max(1, math.ceil(RARE_CLASS_TARGET / rarest))))
    return result


def training_items(rows, repeats=True, mirrors=True):
    """(row index, variant) for one epoch: every row, its repeats if its class is rare, and its mirrored copy."""
    records = rows.to_dict("records")
    counts = repeat_counts(records) if repeats else [1] * len(records)
    items = []
    for index, (row, count) in enumerate(zip(records, counts)):
        items.extend([(index, "")] * count)
        if mirrors and is_mirrorable(row):
            items.append((index, "m"))
    return items


def is_mirrored(variant):
    return variant == "m"


def item_labels(labels, items):
    """Labels of the items of one epoch (repeats and mirrors included): what the loss actually sees."""
    return torch.stack([mirror_labels(labels[i]) if is_mirrored(v) else labels[i] for i, v in items])


def class_alpha(labels, beta=0.999, cap=10.0):
    """Optional per-class weights from the effective number of samples (Cui et al., CVPR 2019), normalised so
    that the average weight per labeled example is 1. The largest weight is at most `cap` times the smallest."""
    alphas = []
    for task, size in enumerate(CLASS_COUNTS):
        column = labels[:, task]
        counts = torch.bincount(column[column >= 0], minlength=size).double()
        weight = (1 - beta) / (1 - beta ** counts.clamp(min=1))
        if counts.sum() > 0:
            weight = weight.clamp(max=cap * weight[counts > 0].min())
            weight = weight * counts.sum() / (counts * weight).sum()
        alphas.append(weight.float())
    return alphas


def solution_frame(rows):
    """Scored cells only, as metric.py expects them (None elsewhere)."""
    solution = rows[["image_id"]].copy()
    scored = splits.scored_cells(rows)
    for dimension in DIMENSIONS:
        solution[dimension] = rows[dimension].where(scored[dimension], None)
    return solution


def load_metric(data_dir):
    import importlib.util

    spec = importlib.util.spec_from_file_location("competition_metric", Path(data_dir) / "metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
