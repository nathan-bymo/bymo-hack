"""Evaluation and training splits of train.csv. Everything is by product and derived from hashes, so every machine
gets the same split and no list of ids has to be committed.

  prompt-dev   bucket 0 of 10: iterate on prompts here.
  prompt-test  bucket 1 of 10: confirm decisions here.
  extra        whole products from the other buckets, added until every class has enough scored examples.
  coverage     prompt-dev + prompt-test + extra: the evaluation set.
  training     everything else. Train, tune and select checkpoints on these rows only:

      from common.splits import training_rows

prompt-dev + prompt-test keep the class distribution of train.csv. "extra" adds mostly rare classes, so a score on
"coverage" is good for comparing methods, not for estimating the leaderboard score.
"""

import hashlib

from common.prompt import DIMENSIONS

BUCKETS = {"dev": (0,), "test": (1,), "base": (0, 1)}
TARGET_PER_CLASS = 30
MINIMUM_PER_CLASS = 20


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bucket(product_id):
    return int(digest(product_id), 16) % 10


def scored_cells(rows):
    """Boolean frame: True where the cell counts for the score (human label in a relevant dimension)."""
    return rows[[]].assign(**{
        d: (rows[f"{d}_source"] == "human") & (rows[f"{d}_relevant"].str.lower() == "true") for d in DIMENSIONS
    })


def class_goal(total):
    """Examples a class should have: 30, but leaving a third for training, and never aiming below 20."""
    return min(TARGET_PER_CLASS, max(MINIMUM_PER_CLASS, 2 * total // 3), total)


def extra_products(rows):
    """Products added to prompt-dev and prompt-test so that every class reaches its goal, rarest classes first."""
    scored = scored_cells(rows)
    selected = set(rows.product_id[rows.product_id.map(bucket).isin(BUCKETS["base"])])
    base = set(selected)
    classes = [(len(rows[scored[d] & (rows[d] == name)]), d, name)
               for d in DIMENSIONS for name in sorted(rows[d][scored[d]].unique())]
    for total, dimension, name in sorted(classes):
        mask = scored[dimension] & (rows[dimension] == name)
        per_product = rows.product_id[mask].value_counts().to_dict()
        have = sum(count for product, count in per_product.items() if product in selected)
        for product in sorted(set(per_product) - selected, key=lambda p: digest("coverage:" + p)):
            if have >= class_goal(total):
                break
            selected.add(product)
            have += per_product[product]
    return selected - base


def split_rows(rows, name):
    """The rows of "dev", "test", "base" or "coverage", in a fixed pseudo-random order of whole products."""
    if name == "coverage":
        extra = extra_products(rows)
        mask = rows.product_id.map(bucket).isin(BUCKETS["base"]) | rows.product_id.isin(extra)
    else:
        mask = rows.product_id.map(bucket).isin(BUCKETS[name])
    picked = rows[mask]
    picked = picked.assign(order=picked.product_id.map(lambda p: digest("order:" + p)))
    return picked.sort_values(["order", "image_id"]).reset_index(drop=True)


def training_rows(rows):
    """The rows that may be used for training: no product of the evaluation set."""
    evaluation = set(split_rows(rows, "coverage").product_id)
    return rows[~rows.product_id.isin(evaluation)].reset_index(drop=True)
