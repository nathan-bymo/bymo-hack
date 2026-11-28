"""The final Kaggle file: the chosen dev predictions plus the predictions of the final images. Never submits.

  python src/scripts/final_submission.py --dev <dev predictions.csv> --final <final.csv predictions> \
      --out data/submissions/final.csv

Fails unless every row of sample_submission.csv gets a prediction and every cell holds a valid class.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / "src" / "training" / "phase-3" / "src" / "training"  # the delivered phase: one copy of the code
sys.path[:0] = [str(PHASE / "apertus-8B-lora-heads"), str(PHASE)]
import data as D  # noqa: E402
import make_submission as S  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sample = D.read_csv(args.sample)
    columns = ["image_id", *D.DIMENSIONS]
    final = D.read_csv(args.final)[columns]
    dev = D.read_csv(args.dev)[columns]
    dev = dev[~dev.image_id.isin(set(final.image_id))]  # a full earlier submission works too: its final rows are replaced
    submission, counts = S.build(sample, [dev, final], set(sample.image_id))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(counts)
    for dimension in D.DIMENSIONS:
        print(f"  {dimension}: {submission[dimension].nunique()} classes used")
    print(f"written {args.out} (not submitted)")


if __name__ == "__main__":
    main()
