"""Stage the colleague's first 600 prompt-dev images privately for v2 evaluation."""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(next(q for p in Path(__file__).resolve().parents for q in (p, p / "training") if (q / "common" / "prompt.py").is_file())))
from common.splits import split_rows, training_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty staging directory")
    source = Path(__file__).resolve().parent
    original = pd.read_csv(args.data_dir / "train.csv", dtype=str, keep_default_na=False)
    rows = split_rows(original, "dev").head(args.limit)
    if len(rows) != args.limit:
        raise ValueError("Not enough prompt-dev images")
    if set(rows.product_id) & set(training_rows(original).product_id):
        raise ValueError("Evaluation product overlaps training pool")
    for relative in map(Path, rows.image_path):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe image path")
        dest = args.output_dir / "data" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data_dir / relative, dest)
    for name in ("train.csv", "metric.py"):
        shutil.copy2(args.data_dir / name, args.output_dir / "data" / name)
    project = args.output_dir / "src"
    experiment = project / "apertus-vadapter-benchmark"
    experiment.mkdir(parents=True)
    (project / "common").mkdir()
    for name in ("adapter.py", "supervision.py", "train_projector.py", "eval_describe_codes.py", "run_eval_v2.sh"):
        shutil.copy2(source / name, experiment / name)
    for name in ("splits.py", "prompt.py"):
        shutil.copy2(source.parent.parent / "training" / "common" / name, project / "common" / name)
    (args.output_dir / "outputs").mkdir()
    print(f"Prepared {len(rows)} prompt-dev images for a private HF job")


if __name__ == "__main__":
    main()
