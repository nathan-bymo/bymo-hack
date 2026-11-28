# Dataset exploration

Three analyses of `train.csv` and of our local evaluation split. They print aggregate numbers only (counts, scores,
intervals) and never write image or product ids. Results: [RESULTS.md](RESULTS.md).

| Script | Question |
|---|---|
| [validation_power.py](validation_power.py) | How many examples of the rarest classes exist at all, and how precisely can prompt-test measure a model and a difference between two models? |
| [context_baseline.py](context_baseline.py) | How much do the product context fields predict without looking at the image? |
| [product_consistency.py](product_consistency.py) | Do the images of one product share their labels, and how much could a product-level vote add? |

[shared.py](shared.py) reuses the repository's own split (`src/training/common/splits.py`) and the leaderboard macro F1
(`src/scripts/build_teacher_targets.py`, same definition as `data/metric.py`).

## Run

From the repository root, with the authorised `data/train.csv` and `data/dev.csv` in place:

```sh
.venv/bin/python src/db_exploration/product_consistency.py
.venv/bin/python src/db_exploration/context_baseline.py
.venv/bin/python src/db_exploration/validation_power.py --bootstrap 2000
```

They need pandas, numpy, scikit-learn and, for `validation_power.py`, torch. CPU only, a few minutes in total.
`validation_power.py` also reads saved prompt-test probabilities of three models (mixed, final-abt, DINO-only) under
`data/teacher/src/`; these files are private and not in the repository. The bootstrap seed is 3407.
