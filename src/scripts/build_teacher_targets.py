"""Soft teacher targets on the unlabeled dev images, for the distillation pilot (see
src/training/apertus-8B-mixed-lora-heads/RESULTS.md, section 4).

The teacher is an equal-weight average of existing model probabilities, chosen per dimension among the subsets of
the given models. A subset replaces the default (every model listed in --default) only if it beats the default on
BOTH halves of prompt-test, split by product hash: the choice never rests on a single noisy comparison.

For every dimension a confidence threshold is chosen on prompt-test: the lowest value at which the teacher's argmax
is right on at least --precision of the scored cells above it. Dev cells below the threshold, and cells that the
scoring rules would not score given the teacher's own subject/background/text predictions, get mask False.

  .venv/bin/python src/scripts/build_teacher_targets.py \\
      --model mixed=data/teacher/src/mixed-modal-20261128-033303/predictions/{split}.probs.pt \\
      --model dino=data/teacher/src/dino-synthetic-modal-20261128-015048/predictions/{split}.probs.pt \\
      --model siglip=data/share_probs/vlh_final-abt_{split}_probs.pt \\
      --default mixed,dino --out data/teacher/dev_teacher_v1.pt

{split} is replaced by prompt_test and dev. Output (torch.save, loads with weights_only=True):
  image_ids, dimensions, probabilities [tensor N x C per dimension, TAXONOMY order], mask (bool N x 6),
  thresholds, members (per dimension), report (prompt-test numbers).
Only aggregate numbers are printed.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "training" / "phase-3" / "src" / "training"))
from common import splits  # noqa: E402
from ensemble import DIMENSIONS, TAXONOMY, load_probabilities  # noqa: E402

UNSCORED_BY_SUBJECT = {
    "logo": {"background", "perspective", "individuals", "text_classification"},
    "placeholder": {"background", "perspective", "individuals", "text_classification"},
    "energy_label": {"background", "perspective", "individuals", "text_classification"},
    "infographic": {"background", "perspective", "individuals"},
    "delivery_scope": {"perspective"},
    "detail_closeup": {"perspective"},
    "product_grouping": {"perspective"},
}
LANGUAGE_SUBJECTS = {"infographic", "packaging", "package_and_product", "full_product"}


def macro_f1(labels, predictions, valid):
    """Same definition as data/metric.py: classes = labels present plus valid classes predicted."""
    classes = set(labels) | (set(predictions) & valid)
    scores = []
    for name in classes:
        hit = np.sum((labels == name) & (predictions == name))
        predicted, actual = np.sum(predictions == name), np.sum(labels == name)
        precision = hit / predicted if predicted else 0.0
        recall = hit / actual if actual else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def rule_mask(names):
    """Bool [N, 6]: cells the scoring rules keep, applied to predicted class names (dict dimension -> array)."""
    n = len(names["subject"])
    mask = np.ones((n, len(DIMENSIONS)), dtype=bool)
    index = {d: i for i, d in enumerate(DIMENSIONS)}
    for row in range(n):
        for dimension in UNSCORED_BY_SUBJECT.get(names["subject"][row], ()):
            mask[row, index[dimension]] = False
        if names["background"][row] == "scene_lifestyle":
            mask[row, index["subject"]] = mask[row, index["perspective"]] = False
        if names["text_classification"][row] == "none" and names["subject"][row] not in LANGUAGE_SUBJECTS:
            mask[row, index["text_language"]] = False
    return mask


def aligned(models, ids):
    out = {}
    for name, (model_ids, probs) in models.items():
        position = {image_id: i for i, image_id in enumerate(model_ids)}
        missing = [i for i in ids if i not in position]
        if missing:
            raise ValueError(f"{name}: {len(missing)} images missing")
        order = [position[i] for i in ids]
        out[name] = {d: probs[d][order] for d in DIMENSIONS}
    return out


def mix(models, members, dimension):
    return sum(models[m][dimension] for m in members) / len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="name=path with {split}")
    parser.add_argument("--default", required=True, help="comma-separated members of the default teacher")
    parser.add_argument("--precision", type=float, default=0.90)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(item.split("=", 1) for item in args.model)
    default = tuple(args.default.split(","))
    rows = pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False)
    test = splits.split_rows(rows, "test")
    scored = splits.scored_cells(test)
    test_ids = list(test.image_id)
    dev_ids = list(pd.read_csv(ROOT / "data" / "dev.csv", dtype=str, keep_default_na=False).image_id)

    pt = aligned({n: load_probabilities(p.format(split="prompt_test")) for n, p in paths.items()}, test_ids)
    dev = aligned({n: load_probabilities(p.format(split="dev")) for n, p in paths.items()}, dev_ids)
    fold = test.product_id.map(lambda p: int(splits.digest("teacher-fold:" + p), 16) % 2).to_numpy()
    products = test.product_id.to_numpy()

    def dim_score(probs, dimension, rows_mask):
        keep = scored[dimension].to_numpy() & rows_mask
        labels = test[dimension].to_numpy()[keep]
        predictions = np.array(TAXONOMY[dimension])[probs.argmax(1)][keep]
        return macro_f1(labels, predictions, set(TAXONOMY[dimension]))

    subsets = [s for k in range(1, len(paths) + 1) for s in itertools.combinations(paths, k)]
    everyone = np.ones(len(test), dtype=bool)
    members, report = {}, {"dimensions": {}}
    for dimension in DIMENSIONS:
        base = mix(pt, default, dimension)
        base_folds = [dim_score(base, dimension, fold == f) for f in (0, 1)]
        best, best_gain = default, 0.0
        for subset in subsets:
            probs = mix(pt, subset, dimension)
            gains = [dim_score(probs, dimension, fold == f) - base_folds[f] for f in (0, 1)]
            if min(gains) > 0 and sum(gains) / 2 > best_gain:
                best, best_gain = subset, sum(gains) / 2
        members[dimension] = list(best)
        report["dimensions"][dimension] = {
            "members": list(best), "default": dim_score(base, dimension, everyone),
            "chosen": dim_score(mix(pt, best, dimension), dimension, everyone),
            "single": {m: dim_score(pt[m][dimension], dimension, everyone) for m in paths}}

    # Overall prompt-test numbers with the official metric, and a paired product bootstrap vs the first model.
    def frame(models_probs, chooser):
        return pd.DataFrame({"image_id": test_ids, **{
            d: np.array(TAXONOMY[d])[chooser(models_probs, d).argmax(1)] for d in DIMENSIONS}})

    sys.path.insert(0, str(ROOT / "data"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("metric", ROOT / "data" / "metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    solution = test[["image_id"]].copy()
    for d in DIMENSIONS:
        solution[d] = test[d].where(scored[d], None)
    candidates = {"teacher": lambda p, d: mix(p, members[d], d), "default": lambda p, d: mix(p, default, d),
                  **{m: (lambda p, d, m=m: p[m][d]) for m in paths}}
    official = {name: metric.evaluate(solution, frame(pt, fn), "image_id")["overall"]["f1_macro"]
                for name, fn in candidates.items()}
    report["prompt_test_macro_f1"] = official

    reference = list(paths)[0]
    predicted = {name: {d: np.array(TAXONOMY[d])[fn(pt, d).argmax(1)] for d in DIMENSIONS}
                 for name, fn in (("teacher", candidates["teacher"]), (reference, candidates[reference]))}
    unique_products = np.unique(products)
    by_product = {p: np.flatnonzero(products == p) for p in unique_products}
    generator = np.random.default_rng(3407)
    deltas = []
    for _ in range(args.bootstrap):
        sample = np.concatenate([by_product[p] for p in generator.choice(unique_products, len(unique_products))])
        values = {}
        for name, preds in predicted.items():
            per_dim = []
            for d in DIMENSIONS:
                keep = scored[d].to_numpy()[sample]
                per_dim.append(macro_f1(test[d].to_numpy()[sample][keep], preds[d][sample][keep], set(TAXONOMY[d])))
            values[name] = np.mean(per_dim)
        deltas.append(values["teacher"] - values[reference])
    report["bootstrap_teacher_minus_" + reference] = [float(np.percentile(deltas, q)) for q in (2.5, 50, 97.5)]

    # Confidence thresholds on prompt-test, then the dev targets.
    thresholds, coverage = {}, {}
    teacher_pt = {d: mix(pt, members[d], d) for d in DIMENSIONS}
    for dimension in DIMENSIONS:
        keep = scored[dimension].to_numpy()
        confidence = teacher_pt[dimension].max(1)[keep]
        correct = (np.array(TAXONOMY[dimension])[teacher_pt[dimension].argmax(1)] == test[dimension].to_numpy())[keep]
        chosen = 1.0
        for t in np.round(np.arange(0.30, 1.0, 0.01), 2):
            above = confidence >= t
            if above.sum() >= 30 and correct[above].mean() >= args.precision:
                chosen = float(t)
                break
        above = confidence >= chosen
        thresholds[dimension] = chosen
        coverage[dimension] = {"prompt_test_kept": float(above.mean()),
                               "prompt_test_precision_kept": float(correct[above].mean()) if above.any() else None}
    report["thresholds"] = thresholds
    report["coverage"] = coverage

    teacher_dev = {d: mix(dev, members[d], d) for d in DIMENSIONS}
    names = {d: np.array(TAXONOMY[d])[teacher_dev[d].argmax(1)] for d in DIMENSIONS}
    mask = rule_mask(names)
    for i, d in enumerate(DIMENSIONS):
        mask[:, i] &= teacher_dev[d].max(1) >= thresholds[d]
    report["dev_cells_kept"] = {d: float(mask[:, i].mean()) for i, d in enumerate(DIMENSIONS)}
    report["dev_images"] = len(dev_ids)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": dev_ids, "dimensions": list(DIMENSIONS),
                "probabilities": [torch.tensor(teacher_dev[d], dtype=torch.float32) for d in DIMENSIONS],
                "mask": torch.tensor(mask), "thresholds": thresholds, "members": members,
                "sources": paths, "report": report}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
