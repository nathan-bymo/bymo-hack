"""Perspective flip TTA on prompt-test, from a fast_infer.py run with --tta-perspective all.

Compares: no TTA, TTA on every image, and selective TTA (only rows whose two most likely perspective classes are a
left/right pair with margin < --margin, fixed BEFORE looking at the result: 0.15). Paired product bootstrap of the
overall macro F1 against no TTA. Aggregate numbers only.

  .venv/bin/python src/scripts/eval_tta.py --probs <pt_fast_tta.probs.pt> --flip <pt_fast_tta.flip.probs.pt>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "scripts"), str(ROOT / "src" / "training" / "phase-3" / "src" / "training")]
from build_teacher_targets import macro_f1  # noqa: E402
from common import splits  # noqa: E402
from ensemble import DIMENSIONS, TAXONOMY  # noqa: E402

PAIRS = {frozenset(p) for p in (("front_angle_left", "front_angle_right"), ("back_angle_left", "back_angle_right"),
                                ("left_side", "right_side"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--flip", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    base = torch.load(args.probs, weights_only=True)
    flip = torch.load(args.flip, weights_only=True)
    assert base["image_ids"] == flip["image_ids"]
    rows = pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False).set_index("image_id")
    rows = rows.loc[base["image_ids"]].reset_index()
    scored = splits.scored_cells(rows)
    names = np.array(TAXONOMY["perspective"])
    p = base["probabilities"][DIMENSIONS.index("perspective")].float()
    f = flip["perspective_flip"].float()
    top = p.topk(2, dim=-1)
    selective = np.array([frozenset((names[a], names[b])) in PAIRS and float(x - y) < args.margin
                          for (a, b), (x, y) in zip(top.indices.tolist(), top.values.tolist())])
    variants = {"none": p, "all": (p + f) / 2, "selective": torch.where(torch.tensor(selective)[:, None], (p + f) / 2, p)}
    preds = {d: np.array(TAXONOMY[d])[base["probabilities"][i].argmax(1).numpy()] for i, d in enumerate(DIMENSIONS)}
    keep = scored["perspective"].to_numpy()
    labels = rows["perspective"].to_numpy()
    products = rows["product_id"].to_numpy()

    def overall(perspective, sample):
        values = []
        for d in DIMENSIONS:
            k = scored[d].to_numpy()[sample]
            guess = perspective[sample] if d == "perspective" else preds[d][sample]
            values.append(macro_f1(rows[d].to_numpy()[sample][k], guess[k], set(TAXONOMY[d])))
        return np.mean(values)

    everything = np.arange(len(rows))
    print(f"selective rows: {selective.mean():.1%} of the images")
    left_right = {"front_angle_left", "front_angle_right", "back_angle_left", "back_angle_right", "left_side", "right_side"}
    unique = np.unique(products)
    by_product = {q: np.flatnonzero(products == q) for q in unique}
    generator = np.random.default_rng(3407)
    samples = [np.concatenate([by_product[q] for q in generator.choice(unique, len(unique))]) for _ in range(args.bootstrap)]
    for name, probs in variants.items():
        guess = names[probs.argmax(1).numpy()]
        side = keep & np.isin(labels, list(left_right))
        line = (f"{name:9s} perspective {macro_f1(labels[keep], guess[keep], set(names)):.4f} | "
                f"left/right accuracy {np.mean(guess[side] == labels[side]):.4f} | overall {overall(guess, everything):.5f}")
        if name != "none":
            reference = names[variants["none"].argmax(1).numpy()]
            deltas = [overall(guess, s) - overall(reference, s) for s in samples]
            line += f" | delta vs none {np.median(deltas):+.4f} [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]"
        print(line)


if __name__ == "__main__":
    main()
