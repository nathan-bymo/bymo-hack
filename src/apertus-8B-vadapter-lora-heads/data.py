"""Rows, splits, labels and training items. Pure pandas/torch: no model, runs on a CPU.

Parts of train.csv (src/AGENTS.md):
  training   phase 1: common.splits.training_rows, never a product of the coverage set;
  all        phase 2: every labeled image, with the settings chosen in phase 1;
  coverage   evaluation set of phase 1, only predicted at the end.

Inside the training part a fraction of the products, chosen by hash, is the internal validation that picks the
checkpoint (--val-fraction). The labels the loss sees are the scored cells only: human labels in relevant cells.
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

RARE_CLASS_TARGET = 300  # a row is repeated until its rarest scored class gets about this many items per epoch
MAX_REPEATS = 4          # original + at most 3 augmented variants


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


def internal_split(rows, fraction, salt="vlh-val:"):
    """(fit, validation) by product hash; fraction 0 leaves the validation empty (phase 2)."""
    is_validation = rows.product_id.map(lambda p: int(splits.digest(salt + p), 16) % 1000 < 1000 * fraction)
    return rows[~is_validation].reset_index(drop=True), rows[is_validation].reset_index(drop=True)


def label_matrix(rows):
    """LongTensor [rows, 6] of class indices, -1 where the cell is not scored (it never reaches the loss)."""
    scored = splits.scored_cells(rows)
    columns = [[_INDEX[d][value] if keep else -1 for value, keep in zip(rows[d], scored[d])] for d in DIMENSIONS]
    return torch.tensor(columns, dtype=torch.long).T.contiguous()


def mirror_labels(labels):
    """Labels of the mirrored image: left and right perspectives swapped."""
    labels = labels.clone()
    known = labels[..., PERSPECTIVE] >= 0
    labels[..., PERSPECTIVE][known] = MIRROR_PERMUTATION[labels[..., PERSPECTIVE][known]]
    return labels


def mirror_safe(row):
    """No text in the image: a mirror would garble it."""
    return row["text_classification"] == "none" and row["text_language"] == "none"


def is_mirrorable(row):
    """A human left/right perspective without text: gets one extra mirrored copy (left/right balance)."""
    return row["perspective_source"] == "human" and row["perspective"] in MIRRORED and mirror_safe(row)


def key(image_id, variant=""):
    """Cache key of an image variant: "" original, "m" mirrored, "a1".."a3" mild crop, "a1m" crop + mirror."""
    return f"{image_id}|{variant}" if variant else image_id


def repeat_counts(labels):
    """Times each row is shown per epoch: enough to lift its rarest scored class towards RARE_CLASS_TARGET."""
    counts = Counter((task, int(value)) for row in labels for task, value in enumerate(row) if value >= 0)
    result = []
    for row in labels:
        rarest = min((counts[(task, int(value))] for task, value in enumerate(row) if value >= 0),
                     default=RARE_CLASS_TARGET)
        result.append(min(MAX_REPEATS, max(1, math.ceil(RARE_CLASS_TARGET / rarest))))
    return result


def training_items(rows, labels, repeats=True, mirrors=True):
    """(row index, variant) for one epoch.

    Every row once as it is. Rows with a rare class are repeated, but each repeat is a different variant: a mild
    crop (variant "a<k>"), mirrored as well on odd k when the image has no text. Left/right perspectives without
    text get one more, plain mirrored copy ("m"). Labels follow the mirror (left <-> right).
    """
    counts = repeat_counts(labels) if repeats else [1] * len(rows)
    items = []
    for index, (row, count) in enumerate(zip(rows.to_dict("records"), counts)):
        items.append((index, ""))
        for k in range(1, count):
            items.append((index, f"a{k}" + ("m" if mirrors and mirror_safe(row) and k % 2 else "")))
        if mirrors and is_mirrorable(row):
            items.append((index, "m"))
    return items


def is_mirrored(variant):
    return variant.endswith("m")


def item_labels(labels, items):
    """Labels of the items of one epoch (repeats and mirrors included): what the loss actually sees."""
    return torch.stack([mirror_labels(labels[i]) if is_mirrored(v) else labels[i] for i, v in items])


def class_alpha(labels, beta=0.999, cap=10.0):
    """Per-class focal alpha from the effective number of samples (Cui et al., CVPR 2019).

    Pass the item labels (item_labels), so that the repeats are not compensated twice. Normalised so that the
    average weight per scored example is 1 (the loss keeps its scale). The largest weight is at most `cap` times
    the smallest, before normalising: a class with one example (Dutch) must not crush the others.
    """
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


def prediction_frame(image_ids, probabilities):
    """Argmax class of every dimension; never an empty cell."""
    frame = {"image_id": list(image_ids)}
    for dimension, probs in zip(DIMENSIONS, probabilities):
        frame[dimension] = [TAXONOMY[dimension][i] for i in probs.argmax(-1).tolist()]
    return pd.DataFrame(frame)


def load_metric(data_dir):
    import importlib.util

    spec = importlib.util.spec_from_file_location("competition_metric", Path(data_dir) / "metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
