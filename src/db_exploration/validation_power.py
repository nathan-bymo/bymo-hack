"""How much can a local validation split tell us? Class supports, bootstrap intervals and detectable differences.

1. Supports of the rarest classes in train.csv, in prompt-test and in what a larger holdout would leave for training.
2. Product bootstrap of the delivered model's prompt-test macro F1 (overall, per dimension, rarest classes).
3. Paired product bootstrap of the difference between two models: the smallest gap prompt-test can resolve.

  .venv/bin/python src/db_exploration/validation_power.py [--bootstrap 2000]
"""

import argparse

import numpy as np
import torch

from shared import (DIMENSIONS, ROOT, TAXONOMY, macro_f1, outside_prompt_test, overall, product_bootstrap,
                    prompt_test, scored, table, train_rows)

RUNS = ROOT / "data" / "teacher" / "src"
MODELS = {
    "mixed (delivered)": RUNS / "mixed-modal-20261128-033303/predictions/prompt_test.probs.pt",
    "final-abt": RUNS / "final-abt/predictions/test.probs.pt",
    "DINO-only": RUNS / "dino-synthetic-modal-20261128-015048/predictions/prompt_test.probs.pt",
}


def class_counts(rows):
    mask = scored(rows)
    return {(d, name): int(((rows[d] == name) & mask[d]).sum()) for d in DIMENSIONS for name in TAXONOMY[d]}


def predictions(path, rows):
    saved = torch.load(path, weights_only=True)
    order = {image: i for i, image in enumerate(saved["image_ids"])}
    index = np.array([order[image] for image in rows.image_id])
    return {d: np.array(TAXONOMY[d])[saved["probabilities"][i].argmax(1).numpy()][index] for i, d in enumerate(DIMENSIONS)}


def class_f1(labels, guesses, name):
    hit = np.sum((labels == name) & (guesses == name))
    predicted, actual = np.sum(guesses == name), np.sum(labels == name)
    if not actual:
        return float("nan")
    precision, recall = (hit / predicted if predicted else 0.0), hit / actual
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def interval(values):
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return f"[{low:.3f}, {high:.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    rows = train_rows()
    test = prompt_test(rows)
    total, in_test, pool = class_counts(rows), class_counts(test), class_counts(outside_prompt_test(rows))
    rare = sorted(total, key=total.get)[:10]

    print("## 1. Supports of the ten rarest classes (human, relevant cells)\n")
    body = []
    for key in rare:
        n = total[key]
        body.append([f"{key[0]}: {key[1]}", n, in_test[key], pool[key], round(0.2 * n), n - round(0.2 * n)])
    print(table(["class", "train.csv", "prompt-test", "left for training", "a 20% holdout would hold",
                 "and leave for training"], body))
    print(f"\nprompt-test: {len(test)} images, {test.product_id.nunique()} products "
          f"({len(test) / len(rows):.1%} of train.csv).")

    preds = {name: predictions(path, test) for name, path in MODELS.items()}
    products = test.product_id.to_numpy()
    samples = list(product_bootstrap(products, args.bootstrap))
    labels = {d: test[d].to_numpy() for d in DIMENSIONS}
    mask = scored(test)

    print(f"\n## 2. Product bootstrap of the delivered model on prompt-test ({args.bootstrap} resamples)\n")
    point, per_dim = overall(test, preds["mixed (delivered)"])
    draws = [overall(test, preds["mixed (delivered)"], s) for s in samples]
    body = [["overall macro F1", f"{point:.4f}", interval([o for o, _ in draws])]]
    body += [[d, f"{per_dim[d]:.4f}", interval([v[d] for _, v in draws])] for d in DIMENSIONS]
    print(table(["", "point", "95% interval"], body))

    print("\nRarest classes present in prompt-test, delivered model:\n")
    body = []
    for d, name in rare:
        if not in_test[(d, name)]:
            continue
        keep = mask[d].to_numpy()
        f1 = class_f1(labels[d][keep], preds["mixed (delivered)"][d][keep], name)
        boot = []
        for s in samples:
            k = keep[s]
            boot.append(class_f1(labels[d][s][k], preds["mixed (delivered)"][d][s][k], name))
        body.append([f"{d}: {name}", in_test[(d, name)], f"{f1:.2f}", interval(boot)])
    print(table(["class", "support", "F1", "95% interval"], body))

    print("\n## 3. Paired bootstrap of model differences on prompt-test\n")
    body = []
    for name in ("mixed (delivered)", "DINO-only"):
        diffs = [overall(test, preds[name], s)[0] - overall(test, preds["final-abt"], s)[0] for s in samples]
        gap = overall(test, preds[name])[0] - overall(test, preds["final-abt"])[0]
        body.append([f"{name} - final-abt", f"{gap:+.4f}", interval(diffs), f"{1.96 * np.std(diffs):.4f}"])
    print(table(["difference", "point", "95% interval", "half-width (≈ smallest resolvable gap)"], body))


if __name__ == "__main__":
    main()
