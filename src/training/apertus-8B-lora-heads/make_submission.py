"""Builds the Kaggle file from sample_submission.csv and one or more prediction CSVs. It never submits anything.

The prediction files only need image_id and the six dimensions, so they can come from any method. Rows without a
prediction keep the answer of sample_submission.csv; every dev row must have a prediction unless --allow-missing
says otherwise. Before 13:00 CET on 28 November there is only the dev file; after it, pass dev and final:

  python src/training/apertus-8B-lora-heads/make_submission.py --predictions data/runs/x/dev.csv --require data/dev.csv \\
      --out data/submissions/submission_dev.csv
  python src/training/apertus-8B-lora-heads/make_submission.py --predictions data/runs/x/dev.csv data/runs/x/final.csv \\
      --require data/dev.csv data/final/final.csv --out data/submissions/submission_final.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402

ROOT = HERE.parents[2]


def build(sample, predictions, required_ids, allow_missing=0):
    """(submission, counts). Raises ValueError for anything Kaggle would reject or that looks wrong."""
    columns = ["image_id", *D.DIMENSIONS]
    if list(sample.columns) != columns or sample.image_id.duplicated().any():
        raise ValueError(f"the sample submission must have the columns {columns} and unique ids")
    merged = pd.concat(predictions, ignore_index=True)
    missing_columns = set(columns) - set(merged.columns)
    if missing_columns:
        raise ValueError(f"the predictions miss the columns {sorted(missing_columns)}")
    merged = merged[columns]
    if merged.image_id.duplicated().any():
        raise ValueError("an image_id is predicted twice")
    unknown = set(merged.image_id) - set(sample.image_id)
    if unknown:
        raise ValueError(f"{len(unknown)} predicted ids are not in the sample submission")
    missing = required_ids - set(merged.image_id)
    if len(missing) > allow_missing:
        raise ValueError(f"{len(missing)} required images have no prediction (allowed: {allow_missing})")
    submission = sample.set_index("image_id").copy()
    submission.loc[merged.image_id, list(D.DIMENSIONS)] = merged[list(D.DIMENSIONS)].to_numpy()
    submission = submission.reset_index()
    for dimension in D.DIMENSIONS:
        bad = sorted(set(submission[dimension]) - set(D.TAXONOMY[dimension]))
        if bad:
            raise ValueError(f"{dimension}: unknown or empty class {bad[:3]}")
    counts = {"rows": len(submission), "predicted": len(merged), "kept_from_sample": len(submission) - len(merged),
              "required_missing": len(missing)}
    return submission, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--require", type=Path, nargs="+", required=True,
                        help="dev.csv (and final.csv): every one of their images needs a prediction")
    parser.add_argument("--allow-missing", type=int, default=0, help="unreadable images that may keep the sample answer")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    required = set().union(*(set(D.read_csv(path).image_id) for path in args.require))
    try:
        submission, counts = build(D.read_csv(args.sample), [D.read_csv(p) for p in args.predictions], required,
                                   args.allow_missing)
    except ValueError as error:
        sys.exit(f"not written: {error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"{args.out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("Nothing was submitted. Upload it yourself on Kaggle and select it for the final ranking.")


if __name__ == "__main__":
    main()
