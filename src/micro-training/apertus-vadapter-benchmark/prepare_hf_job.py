"""Stage only selected competition images and code for a private HF Job volume."""

import argparse
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_projector import select_manifest_rows, select_split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--holdout-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing staging directory")
    if (args.train_manifest is None) != (args.validation_manifest is None):
        parser.error("provide both manifests or neither")
    source = Path(__file__).resolve().parent
    frame = pd.read_csv(args.data_dir / "train.csv", dtype=str, keep_default_na=False)
    if args.train_manifest:
        full, evaluation = select_manifest_rows(frame, args.train_manifest, args.validation_manifest,
                                                 args.train_size, args.holdout_size, args.seed)
        smoke, _ = select_manifest_rows(frame, args.train_manifest, args.validation_manifest, 8, 4, args.seed)
    else:
        full, evaluation = select_split_rows(frame, args.train_size, args.holdout_size, args.seed)
        smoke, _ = select_split_rows(frame, 8, 4, args.seed)
    relative_paths = {Path(row["image_path"]) for row in (*full, *smoke, *evaluation)}
    if any(path.is_absolute() or ".." in path.parts for path in relative_paths):
        raise ValueError("Unsafe image path in dataset")
    root = args.output_dir
    for relative in relative_paths:
        dest = root / "data" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data_dir / relative, dest)
    for name in ("train.csv", "metric.py"):
        shutil.copy2(args.data_dir / name, root / "data" / name)
    if args.train_manifest:
        (root / "data" / "prototype_splits").mkdir()
        shutil.copy2(args.train_manifest, root / "data" / "prototype_splits" / "train_1500.csv")
        shutil.copy2(args.validation_manifest, root / "data" / "prototype_splits" / "validation_450.csv")
    project = root / "src"
    (project / "apertus-vadapter-benchmark").mkdir(parents=True)
    (project / "common").mkdir()
    (project / "docs").mkdir()
    for name in ("adapter.py", "supervision.py", "train_projector.py", "run_h200.sh"):
        shutil.copy2(source / name, project / "apertus-vadapter-benchmark" / name)
    for name in ("splits.py", "prompt.py"):
        shutil.copy2(source.parent.parent / "training" / "common" / name, project / "common" / name)
    (project / "docs" / "system_prompt.txt").write_text(
        runpy.run_path(str(source.parent.parent / "training" / "common" / "prompt.py"))["SHARED_JSON_PROMPT"],
        encoding="utf-8")
    (root / "outputs").mkdir()
    print(f"Staged {len(relative_paths)} images; train={len(full)}, holdout={len(evaluation)}, plus smoke samples")


if __name__ == "__main__":
    main()
