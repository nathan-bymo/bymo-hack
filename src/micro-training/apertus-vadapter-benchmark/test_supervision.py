import json

import pytest

from supervision import DIMENSIONS, split_by_product, supervised_answer


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_masks_only_unverified_or_irrelevant_label_values():
    row = {d: "value_" + d for d in DIMENSIONS}
    row.update({f"{d}_source": "human" for d in DIMENSIONS})
    row.update({f"{d}_relevant": "true" for d in DIMENSIONS})
    row["background_source"] = "model"
    row["perspective_relevant"] = "false"
    answer, ids, labels = supervised_answer(row, CharacterTokenizer())
    assert json.loads(answer) == {d: row[d] for d in DIMENSIONS}
    for dimension in DIMENSIONS:
        start = answer.index('"' + row[dimension] + '"')
        end = start + len(row[dimension]) + 2
        if dimension in ("background", "perspective"):
            assert all(value == -100 for value in labels[start:end])
        else:
            assert labels[start:end] == ids[start:end]


def test_split_is_repeatable_and_products_do_not_cross():
    rows = [{"product_id": str(i // 2)} for i in range(20)]
    train, holdout = split_by_product(rows, seed=13)
    assert not ({row["product_id"] for row in train} & {row["product_id"] for row in holdout})
    assert (train, holdout) == split_by_product(rows, seed=13)
    with pytest.raises(ValueError):
        split_by_product([{"product_id": "only"}])
