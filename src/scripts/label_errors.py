"""Suspected label errors in train.csv, for the bonus list sent to the Galaxus hosts (never commit the csv).

A human, scored cell is suspected only when models that never trained on its product agree on another class:
  - prompt-test products: at least --min-agree of the strong models (--strong) predict the same other class with
    probability >= --confidence each, and their mean probability of the human label is <= --max-label-prob;
  - other coverage products (seen only by the phase-1 models): every --weak model predicts the same other class with
    probability >= --weak-confidence.

  .venv/bin/python src/scripts/label_errors.py \\
      --strong mixed=data/teacher/src/mixed-modal-20261128-033303/predictions/prompt_test.probs.pt \\
      --strong dino=data/teacher/src/dino-synthetic-modal-20261128-015048/predictions/prompt_test.probs.pt \\
      --strong siglip=data/share_probs/vlh_final-abt_prompt_test_probs.pt \\
      --weak phase1=data/teacher/src/phase1/predictions/coverage.probs.pt \\
      --weak phase1-ocr=data/teacher/src/phase1-ocr/predictions/coverage.probs.pt \\
      --weak bench1=data/teacher/src/bench1/predictions/coverage.probs.pt \\
      --out data/label_errors/suspected_label_errors.csv

Writes image_id, dimension, label, suggested_label (plus an evidence column in a separate file); prints counts only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "training"))
from common import splits  # noqa: E402
from ensemble import DIMENSIONS, TAXONOMY, load_probabilities  # noqa: E402


def load(items):
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        ids, probs = load_probabilities(path)
        out[name] = ({image_id: i for i, image_id in enumerate(ids)}, probs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strong", action="append", required=True)
    parser.add_argument("--weak", action="append", default=[])
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--max-label-prob", type=float, default=0.10)
    parser.add_argument("--weak-confidence", type=float, default=0.90)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False)
    scored = splits.scored_cells(rows)
    test_products = set(splits.split_rows(rows, "test").product_id)
    coverage_products = set(splits.split_rows(rows, "coverage").product_id)
    strong, weak = load(args.strong), load(args.weak)

    found, evidence = [], []
    for r, row in rows.iterrows():
        in_test = row.product_id in test_products
        in_coverage = row.product_id in coverage_products
        if not in_coverage:
            continue
        for d in DIMENSIONS:
            if not scored.at[r, d]:
                continue
            label = TAXONOMY[d].index(row[d])
            if in_test:
                vectors = [probs[d][pos[row.image_id]] for pos, probs in strong.values() if row.image_id in pos]
                if len(vectors) < args.min_agree:
                    continue
                votes = [int(v.argmax()) for v in vectors if v.max() >= args.confidence and v.argmax() != label]
                if not votes:
                    continue
                choice = max(set(votes), key=votes.count)
                if votes.count(choice) < args.min_agree or np.mean([v[label] for v in vectors]) > args.max_label_prob:
                    continue
                source = "strong"
            elif weak:
                vectors = [probs[d][pos[row.image_id]] for pos, probs in weak.values() if row.image_id in pos]
                if len(vectors) < len(weak):
                    continue
                picks = {int(v.argmax()) for v in vectors}
                if len(picks) != 1 or label in picks or min(v.max() for v in vectors) < args.weak_confidence:
                    continue
                choice, source = picks.pop(), "weak"
            else:
                continue
            found.append({"image_id": row.image_id, "dimension": d, "label": row[d],
                          "suggested_label": TAXONOMY[d][choice]})
            evidence.append({**found[-1], "evidence": source,
                             "mean_prob_suggested": float(np.mean([v[choice] for v in vectors])),
                             "mean_prob_label": float(np.mean([v[label] for v in vectors]))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(found, columns=["image_id", "dimension", "label", "suggested_label"])
    frame.to_csv(args.out, index=False)
    pd.DataFrame(evidence).to_csv(args.out.with_name(args.out.stem + "_evidence.csv"), index=False)
    detail = pd.DataFrame(evidence)
    print(f"{len(frame)} suspected cells written (not for the repository)")
    if len(detail):
        print(detail.groupby(["evidence", "dimension"]).size().to_string())


if __name__ == "__main__":
    main()
