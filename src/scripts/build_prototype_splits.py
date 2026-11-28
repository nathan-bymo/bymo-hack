"""Create private, product-disjoint development manifests under data/."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "training" / "phase-3" / "src" / "training"))
from common.splits import scored_cells, split_rows, training_rows


def counts(rows, dimension):
    scored = scored_cells(rows)[dimension]
    return rows.loc[scored, dimension].value_counts().sort_index().to_dict()


def balanced_validation(pool, size, seed):
    """Greedily diversify scored labels, with one image per validation product.

    Choosing a validation product removes all its images from the training pool.
    Retain at least half of the products for every represented class.
    """
    scored = scored_cells(pool)
    dimensions = list(scored.columns)
    classes = [(d, c) for d in dimensions for c in sorted(pool.loc[scored[d], d].unique())]
    class_counts = {d: sum(dim == d for dim, _ in classes) for d in dimensions}
    matrix = np.column_stack([(scored[d].to_numpy() & (pool[d].to_numpy() == c)) for d, c in classes])
    product_codes, product_names = pd.factorize(pool.product_id, sort=True)
    product_features = np.zeros((len(product_names), len(classes)), dtype=bool)
    np.maximum.at(product_features, product_codes, matrix)
    product_supply = product_features.sum(axis=0).astype(int)
    minimum_supply = (product_supply + 1) // 2
    available_products = np.ones(len(product_names), dtype=bool)
    selected = []
    selected_counts = np.zeros(len(classes), dtype=int)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(pool)) * 1e-9
    # Some subject classes make most other dimensions irrelevant; compensate so
    # those examples are not systematically disfavored by a multi-label score.
    dimension_weights = np.array([(3.0 if d == "subject" else 1.0) / class_counts[d] for d, _ in classes])
    for _ in range(size):
        protected = product_supply <= minimum_supply
        allowed_products = available_products & ~(product_features[:, protected].any(axis=1))
        weights = dimension_weights / (1 + selected_counts)
        scores = matrix @ weights + tie_break
        scores[~allowed_products[product_codes]] = -1
        index = int(np.argmax(scores))
        if scores[index] < 0:
            raise RuntimeError("Cannot select enough validation products while retaining each class in training")
        product = product_codes[index]
        selected.append(index)
        selected_counts += matrix[index]
        product_supply -= product_features[product]
        available_products[product] = False
    return pool.iloc[selected].copy(), set(product_names[~available_products])


def representative_training(reference, remaining, size, seed, candidates=200):
    """Choose the closest of 200 random samples to the original class proportions."""
    pool_mask = scored_cells(reference)
    remaining_mask = scored_cells(remaining)
    dimensions = list(pool_mask.columns)
    matrices, targets = [], []
    for dimension in dimensions:
        labels = sorted(reference.loc[pool_mask[dimension], dimension].unique())
        original = np.array([(pool_mask[dimension] & (reference[dimension] == label)).sum() for label in labels])
        targets.append(original / original.sum())
        matrices.append(np.column_stack([(remaining_mask[dimension].to_numpy() &
                                          (remaining[dimension].to_numpy() == label)) for label in labels]))
    rng = np.random.default_rng(seed)
    best, best_score = None, (float("inf"), float("inf"))
    for _ in range(candidates):
        indices = rng.choice(len(remaining), size=size, replace=False)
        distance = 0.0
        missing_classes = 0
        for matrix, target in zip(matrices, targets, strict=True):
            histogram = matrix[indices].sum(axis=0)
            missing_classes += int(((histogram == 0) & matrix.any(axis=0)).sum())
            distance += 0.5 * np.abs(histogram / max(1, histogram.sum()) - target).sum()
        score = (missing_classes, distance)
        if score < best_score:
            best, best_score = indices, score
    return remaining.iloc[best].copy(), best_score[1] / len(dimensions), best_score[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "prototype_splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--validation-size", type=int, default=450)
    parser.add_argument("--replace-generated", action="store_true")
    args = parser.parse_args()
    expected = {"train_1500.csv", "validation_450.csv", "distribution.json", "README.md"}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and (
        not args.replace_generated or {p.name for p in args.output_dir.iterdir()} != expected
    ):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    source = args.data_dir / "train.csv"
    rows = pd.read_csv(source, dtype=str, keep_default_na=False)
    pool = training_rows(rows)
    validation, validation_products = balanced_validation(pool, args.validation_size, args.seed)
    remaining = pool[~pool.product_id.isin(validation_products)]
    if len(remaining) < args.train_size:
        raise ValueError("Not enough rows remain for the requested training size")
    train, mean_distance, missing_classes = representative_training(rows, remaining, args.train_size, args.seed)
    reserved = set(split_rows(rows, "coverage").product_id)
    assert not (set(train.product_id) & set(validation.product_id))
    assert not ((set(train.product_id) | set(validation.product_id)) & reserved)
    for path in pd.concat([train.image_path, validation.image_path]):
        if not (args.data_dir / path).is_file():
            raise FileNotFoundError(f"Missing image: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output_dir / "train_1500.csv", index=False)
    validation.to_csv(args.output_dir / "validation_450.csv", index=False)
    report = {
        "seed": args.seed,
        "source": "data/train.csv",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "pool": "common.splits.training_rows; entire coverage reserved",
        "validation_selection": "greedy label diversification, with subject weighted 3x; one image per product; retain at least half of the pool products per class",
        "training_selection": "best of 200 random image samples, prioritizing class coverage then mean total variation against original train.csv label distributions",
        "mean_training_total_variation_distance": mean_distance,
        "training_classes_missing_from_remaining_pool": missing_classes,
        "images": {"eligible_pool": len(pool), "train": len(train), "validation": len(validation)},
        "products": {"train": train.product_id.nunique(), "validation": validation.product_id.nunique(),
                     "overlap": len(set(train.product_id) & set(validation.product_id))},
        "human_relevant_class_counts": {
            d: {"original": counts(rows, d), "eligible_pool": counts(pool, d), "train": counts(train, d), "validation": counts(validation, d)}
            for d in scored_cells(pool).columns
        },
    }
    (args.output_dir / "distribution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "README.md").write_text(
        "# Prototype splits\n\n"
        "`train_1500.csv` and `validation_450.csv` retain all original columns. "
        "Their `image_path` values are relative to the parent `data/` directory; images are not duplicated. "
        "Both exclude the team's reserved coverage products and do not share product IDs. "
        "The training sample is chosen from 200 candidates to approximate the original label distribution. "
        "The validation set is deliberately label-balanced where feasible and is not representative of the original distribution; "
        "its balance is approximate because labels co-occur, rare classes may be unavailable, and product separation is enforced. "
        "Only human and relevant cells are used when balancing or reporting counts. "
        "See `distribution.json` for counts, seed and reproducibility details. Keep all data private.\n",
        encoding="utf-8",
    )
    print(f"Created {len(train)} training and {len(validation)} validation rows in {args.output_dir}")


if __name__ == "__main__":
    main()
