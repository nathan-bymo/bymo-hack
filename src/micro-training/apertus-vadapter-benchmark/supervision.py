"""Build supervision without learning from unverified or irrelevant labels."""

import json
import random


DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")


def split_by_product(rows, holdout_fraction=0.2, seed=0):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    products = sorted({str(row["product_id"]) for row in rows})
    if len(products) < 2:
        raise ValueError("at least two products are required")
    random.Random(seed).shuffle(products)
    n_holdout = max(1, min(len(products) - 1, round(len(products) * holdout_fraction)))
    holdout = set(products[:n_holdout])
    return ([r for r in rows if str(r["product_id"]) not in holdout],
            [r for r in rows if str(r["product_id"]) in holdout])


def supervised_answer(row, tokenizer):
    """Tokenize JSON and mask tokens touching labels that are not human and relevant.

    Return input IDs and loss labels for the answer only. The caller must mask the
    entire prompt, prepend visual-token masks and mask any padding separately.
    """
    pieces = ["{"]
    masked_spans = []
    offset = 1
    for i, dimension in enumerate(DIMENSIONS):
        prefix = (", " if i else "") + json.dumps(dimension) + ": "
        value = json.dumps(str(row[dimension]), ensure_ascii=False)
        pieces.extend((prefix, value))
        start = offset + len(prefix)
        relevant = str(row[f"{dimension}_relevant"]).lower() in ("true", "1")
        if str(row[f"{dimension}_source"]).lower() != "human" or not relevant:
            masked_spans.append((start, start + len(value)))
        offset = start + len(value)
    pieces.append("}")
    answer = "".join(pieces)
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    labels = [
        -100 if end <= start or any(start < b and end > a for a, b in masked_spans) else token
        for token, (start, end) in zip(ids, encoded["offset_mapping"], strict=True)
    ]
    return answer, ids, labels
