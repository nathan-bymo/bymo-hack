"""How much do the product context fields alone predict, without looking at the image?

Fits on every labelled row outside prompt-test (the fit pool of the delivered model) and scores prompt-test with the
leaderboard macro F1. Context = product_title, product_type, brand, category, subcategory.

  majority        the most frequent class of each dimension
  product type    the majority class of the image's product_type, else of its category, else overall
  tf-idf + LR     logistic regression on word 1-2 grams of the concatenated context, class-balanced

  .venv/bin/python src/db_exploration/context_baseline.py
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from shared import DIMENSIONS, outside_prompt_test, overall, prompt_test, scored, table, train_rows

FIELDS = ["product_title", "product_type", "brand", "category", "subcategory"]
DELIVERED = {"subject": 0.868, "background": 0.875, "perspective": 0.776, "individuals": 0.970,
             "text_classification": 0.769, "text_language": 0.798}  # weights/apertus-8B-mixed-lora-heads-final


def context(rows):
    return rows[FIELDS].apply(lambda r: " | ".join(r.values), axis=1)


def majority(fit, test):
    mask = scored(fit)
    return {d: np.full(len(test), fit[d][mask[d]].value_counts().idxmax(), dtype=object) for d in DIMENSIONS}


def by_product_type(fit, test):
    mask, out = scored(fit), {}
    for d in DIMENSIONS:
        labelled = fit[mask[d]]
        overall_top = labelled[d].value_counts().idxmax()
        tables = [labelled.groupby(key)[d].agg(lambda s: s.value_counts().idxmax()).to_dict()
                  for key in ("product_type", "category")]
        out[d] = np.array([tables[0].get(t, tables[1].get(c, overall_top))
                           for t, c in zip(test.product_type, test.category)], dtype=object)
    return out


def tfidf_lr(fit, test):
    mask, out = scored(fit), {}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_fit, x_test = vectorizer.fit_transform(context(fit)), None
    x_test = vectorizer.transform(context(test))
    for d in DIMENSIONS:
        keep = mask[d].to_numpy()
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
        model.fit(x_fit[keep], fit[d].to_numpy()[keep])
        out[d] = model.predict(x_test)
    return out


def main():
    rows = train_rows()
    fit, test = outside_prompt_test(rows), prompt_test(rows)
    print(f"Fit on {len(fit)} images outside prompt-test, scored on prompt-test ({len(test)} images).\n")
    body = []
    for name, method in (("majority", majority), ("product type", by_product_type), ("tf-idf + LR", tfidf_lr)):
        score, per_dim = overall(test, method(fit, test))
        body.append([name, f"{score:.3f}"] + [f"{per_dim[d]:.3f}" for d in DIMENSIONS])
    body.append(["delivered model (image + context + OCR)", f"{np.mean(list(DELIVERED.values())):.3f}"]
                + [f"{DELIVERED[d]:.3f}" for d in DIMENSIONS])
    print(table(["context-only predictor", "macro F1"] + list(DIMENSIONS), body))


if __name__ == "__main__":
    main()
