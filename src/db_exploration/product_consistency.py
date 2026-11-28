"""Do the images of one product share their labels? An upper bound on what product-level aggregation could add.

For every dimension, over products with at least two scored images in train.csv:
  all-same      share of products whose scored images all carry the same label
  sibling vote  accuracy of predicting each image's label as the majority label of the OTHER images of its product
                (ties broken by overall class frequency): what a perfect product-level vote could reach
text_language is also reported on images that carry text (label not "none").

  .venv/bin/python src/db_exploration/product_consistency.py
"""

from collections import Counter

import pandas as pd

from shared import DIMENSIONS, ROOT, scored, table, train_rows


def stats(frame, dimension):
    groups = frame.groupby("product_id")[dimension]
    sizes = groups.size()
    multi = sizes[sizes > 1].index
    if not len(multi):
        return None
    frame = frame[frame.product_id.isin(multi)]
    prior = Counter(frame[dimension])
    same = (groups.nunique().loc[multi] == 1).mean()
    hits = 0
    for _, group in frame.groupby("product_id"):
        labels = group[dimension].tolist()
        for i, label in enumerate(labels):
            votes = Counter(labels[:i] + labels[i + 1:])
            guess = max(votes, key=lambda name: (votes[name], prior[name]))
            hits += guess == label
    return len(multi), len(frame), same, hits / len(frame)


def main():
    rows = train_rows()
    mask = scored(rows)
    print(f"train.csv: {len(rows)} images of {rows.product_id.nunique()} products "
          f"({len(rows) / rows.product_id.nunique():.2f} images per product).\n")
    body = []
    for d in DIMENSIONS:
        frame = rows[mask[d]][["product_id", d]]
        result = stats(frame, d)
        body.append([d, *result[:2], f"{result[2]:.1%}", f"{result[3]:.1%}"])
        if d == "text_language":
            result = stats(frame[frame[d] != "none"], d)
            body.append(["text_language, images with text", *result[:2], f"{result[2]:.1%}", f"{result[3]:.1%}"])
    print(table(["dimension", "products with ≥2 scored images", "their images", "all-same", "sibling vote"], body))
    dev = pd.read_csv(ROOT / "data" / "dev.csv", dtype=str)
    sizes = dev.product_id.map(dev.product_id.value_counts())
    print(f"\ndev.csv: {(sizes > 1).mean():.1%} of the images have at least one sibling image of the same product.")
    train_mix = rows.category.value_counts(normalize=True)
    dev_mix = dev.category.value_counts(normalize=True)
    distance = 0.5 * train_mix.subtract(dev_mix, fill_value=0).abs().sum()
    unseen = (~dev.product_type.isin(set(rows.product_type))).mean()
    print(f"Category mix, train vs dev: total-variation distance {distance:.3f}; "
          f"dev images of a product type absent from train: {unseen:.1%}.")


if __name__ == "__main__":
    main()
