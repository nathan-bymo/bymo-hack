"""Per-class logit biases that maximise the macro F1, fitted on the internal validation only (no GPU).

The heads are cautious on rare classes (high precision, low recall). A bias b_c added to log p_c before the
argmax moves that balance without training. Per dimension, coordinate ascent over a grid of biases, on the
probabilities predict.py saved for the internal validation (--part validation). The coverage set is never used to
fit: with --apply-to the biases are only applied to it, to measure the gain honestly.

  python src/apertus-8B-vadapter-lora-heads/calibrate.py --validation data/runs/vlh_bench/validation.probs.pt \
      --out data/runs/vlh_bench/bias.json --apply-to data/runs/vlh_bench/coverage.probs.pt

Guards against fitting noise on 941 images: classes with fewer than --min-support scored validation examples keep
bias 0, biases stay within +-MAX_BIAS, and a change is kept only if it raises the F1 by more than --min-gain.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

ROOT = HERE.parents[1]
GRID = [x / 4 for x in range(-12, 13)]  # -3 .. +3 in log probability
MAX_BIAS = 3.0


def apply(probabilities, biases):
    """Decision scores log p + b per dimension (biases: {dimension: {class: bias}})."""
    scores = []
    for dimension, probs in zip(D.DIMENSIONS, probabilities):
        bias = torch.tensor([biases.get(dimension, {}).get(name, 0.0) for name in D.TAXONOMY[dimension]])
        scores.append(probs.clamp(min=1e-12).log() + bias)
    return scores


def macro_f1(labels, predicted, size):
    """Macro F1 of one dimension as metric.py: classes present in the labels or predicted, scored cells only."""
    known = labels >= 0
    labels, predicted = labels[known], predicted[known]
    classes = torch.unique(torch.cat([labels, predicted]))
    f1 = []
    for c in classes.tolist():
        hits = int(((predicted == c) & (labels == c)).sum())
        precision_den, recall_den = int((predicted == c).sum()), int((labels == c).sum())
        f1.append(0.0 if hits == 0 else 2 * hits / (precision_den + recall_den))
    return sum(f1) / len(f1) if f1 else 0.0


def fit_dimension(log_probs, labels, min_support, min_gain, passes=3):
    size = log_probs.shape[1]
    bias = torch.zeros(size)
    support = torch.bincount(labels[labels >= 0], minlength=size)
    tunable = [c for c in range(size) if support[c] >= min_support]
    best = macro_f1(labels, (log_probs + bias).argmax(1), size)
    start = best
    for _ in range(passes):
        changed = False
        for c in tunable:
            for value in GRID:
                trial = bias.clone()
                trial[c] = value
                score = macro_f1(labels, (log_probs + trial).argmax(1), size)
                if score > best + min_gain:
                    best, bias, changed = score, trial, True
        if not changed:
            break
    return bias.clamp(-MAX_BIAS, MAX_BIAS), start, best


def rows_for(image_ids):
    rows = D.read_csv(ROOT / "data" / "train.csv").set_index("image_id", drop=False)
    return rows.loc[list(image_ids)].reset_index(drop=True)


def evaluate(saved, biases, metric):
    rows = rows_for(saved["image_ids"])
    probabilities = saved["probabilities"]
    decision = apply(probabilities, biases) if biases else probabilities
    predictions = D.prediction_frame(rows.image_id, decision)
    results = metric.evaluate(D.solution_frame(rows), predictions, "image_id")
    return results, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", type=Path, required=True, help="validation.probs.pt of predict.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--apply-to", type=Path, help="probs.pt of another set, scored with and without biases")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-gain", type=float, default=0.002)
    args = parser.parse_args()

    saved = torch.load(args.validation, weights_only=False)
    rows = rows_for(saved["image_ids"])
    coverage_products = set(D.select(D.read_csv(ROOT / "data" / "train.csv"), "coverage").product_id)
    assert not rows.product_id.isin(coverage_products).any(), "the calibration set touches the coverage set"
    labels = D.label_matrix(rows)
    biases, report = {}, {}
    for task, (dimension, probs) in enumerate(zip(D.DIMENSIONS, saved["probabilities"])):
        bias, before, after = fit_dimension(probs.clamp(min=1e-12).log(), labels[:, task], args.min_support,
                                            args.min_gain)
        biases[dimension] = {name: round(float(b), 2) for name, b in zip(D.TAXONOMY[dimension], bias) if b != 0}
        report[dimension] = (before, after)
        print(f"{dimension:20} validation macro F1 {before:.3f} -> {after:.3f} | biases {biases[dimension]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(biases, indent=2))
    print(f"biases in {args.out} (fitted on {len(rows)} internal validation images)")

    if args.apply_to:
        metric = D.load_metric(ROOT / "data")
        target = torch.load(args.apply_to, weights_only=False)
        plain, plain_frame = evaluate(target, None, metric)
        calibrated, calibrated_frame = evaluate(target, biases, metric)
        stem = args.apply_to.name.removesuffix(".probs.pt")
        plain_frame.to_csv(args.apply_to.with_name(f"{stem}_plain.csv"), index=False)
        calibrated_frame.to_csv(args.apply_to.with_name(f"{stem}_calibrated.csv"), index=False)
        print(f"\n{stem} ({len(target['image_ids'])} images, never used to fit): macro F1 "
              f"{plain['overall']['f1_macro']:.4f} -> {calibrated['overall']['f1_macro']:.4f}")
        for dimension in D.DIMENSIONS:
            print(f"  {dimension:20} {plain[dimension]['f1_macro']:.3f} -> {calibrated[dimension]['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
