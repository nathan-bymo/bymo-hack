# Dataset exploration: results (28 November 2026)

Aggregate numbers only. Human, relevant cells are the cells the leaderboard scores. prompt-test is bucket 1 of 10 of
the product hash in `src/training/phase-3/src/training/common/splits.py`: 1,193 images of 713 products, 10.0% of `train.csv`.

## 0. Dataset facts used in the technical report

22,429 images from 14,118 products: 11,876 labelled training images, 5,261 public-dev images, 5,292 final images;
41 classes in six dimensions. Images were pre-classified by a model and then reviewed by people; about one fifth of
label cells remain model prelabels, and text language is the most affected (59% of its relevant labels are human).
Splits: coverage 2,532 images (prompt-dev + prompt-test + 169 rare-class images), base 2,363, prompt-test 1,193; the
delivered model trained on the 10,683 images outside prompt-test. Source: `src/README.md` §1.1–1.2,
`src/training/EXPERIMENTS.md` (split table), `paper/paper.tex` (data section, 59%),
`training.json` of [nathan-monzani-bymo/apertus-8B-mixed](https://huggingface.co/nathan-monzani-bymo/apertus-8B-mixed) (10,683).

## 1. Validation power (`validation_power.py`, 2,000 product-bootstrap resamples, seed 3407)

**The rarest classes are rare in the whole dataset, not only in our split.**

| class | train.csv | prompt-test | left for training | a 20% holdout would hold | and leave for training |
|---|---|---|---|---|---|
| text_language: Dutch | 21 | 4 | 17 | 4 | 17 |
| text_language: Italian | 47 | 4 | 43 | 9 | 38 |
| text_classification: certification_seal | 60 | 6 | 54 | 12 | 48 |
| subject: product_grouping | 91 | 10 | 81 | 18 | 73 |
| text_language: other | 110 | 10 | 100 | 22 | 88 |
| subject: multipack_trade_unit | 139 | 8 | 131 | 28 | 111 |
| subject: placeholder | 178 | 13 | 165 | 36 | 142 |
| subject: multiview | 184 | 20 | 164 | 37 | 147 |
| subject: accessories | 205 | 15 | 190 | 41 | 164 |
| perspective: back_angle_right | 211 | 26 | 185 | 42 | 169 |

Prompt-test holds roughly its proportional share of each rare class. A larger holdout would add only a few rare
examples to the evaluation and remove them from training, where they are scarcest (Italian 43 → 38).

**Precision of the delivered model's score on prompt-test** (mixed, selected checkpoint):

| | point | 95% interval |
|---|---|---|
| overall macro F1 | 0.8428 | [0.816, 0.864] |
| subject | 0.8681 | [0.821, 0.899] |
| background | 0.8749 | [0.823, 0.918] |
| perspective | 0.7764 | [0.731, 0.813] |
| individuals | 0.9695 | [0.936, 0.994] |
| text_classification | 0.7695 | [0.688, 0.839] |
| text_language | 0.7985 | [0.684, 0.871] |

Per class, the rarest classes have very wide intervals: Italian (n = 4) F1 0.67 [0.00, 1.00], certification seal
(n = 6) 0.40 [0.00, 0.80], Dutch (n = 4) 0.86 [0.67, 1.00].

**Resolvable differences** (paired bootstrap, same images for both models):

| difference | point | 95% interval | half-width |
|---|---|---|---|
| mixed − final-abt | +0.0214 | [+0.006, +0.038] | 0.016 |
| DINO-only − final-abt | −0.0137 | [−0.054, +0.015] | 0.034 |

Between closely related checkpoints, prompt-test resolves differences of about 0.016 in macro F1; between less
similar models, about 0.03. Smaller gaps, and gaps driven by the rarest classes, need a second signal: we used the
public dev leaderboard (5,261 images). The delivered model passes both: +0.021 on prompt-test and +0.010 on public dev
over final-abt. The mixed checkpoint was selected on prompt-test among five epochs, so its prompt-test score carries
a small selection effect.

## 2. Context-only baselines (`context_baseline.py`)

Fit on the 10,683 images outside prompt-test, scored on prompt-test. Context = title, product type, brand, category,
subcategory.

| predictor | macro F1 | subject | background | perspective | individuals | text_cl. | text_lang. |
|---|---|---|---|---|---|---|---|
| majority class | 0.187 | 0.056 | 0.305 | 0.053 | 0.486 | 0.150 | 0.075 |
| majority by product type | 0.376 | 0.285 | 0.370 | 0.329 | 0.689 | 0.333 | 0.251 |
| tf-idf + logistic regression | 0.438 | 0.340 | 0.474 | 0.335 | 0.613 | 0.349 | 0.518 |
| delivered model (image + context + OCR) | 0.843 | 0.868 | 0.875 | 0.776 | 0.970 | 0.769 | 0.798 |

The context alone carries real signal, mostly for text language (0.518, the language of the title) and through
product-type priors. It is far below the image model on every dimension, so the image does most of the work.
For comparison, Apertus 8B zero-shot with a rule-based JSON prompt scored 0.225 on prompt-test.

## 3. Label consistency within a product (`product_consistency.py`)

train.csv: 1.64 images per product. dev.csv: 48.4% of the images have at least one sibling of the same product.

| dimension | products with ≥ 2 scored images | their images | all-same | sibling vote |
|---|---|---|---|---|
| subject | 1,424 | 5,558 | 41.1% | 64.4% |
| background | 1,391 | 5,560 | 73.6% | 86.4% |
| perspective | 1,185 | 4,100 | 20.0% | 33.8% |
| individuals | 1,392 | 5,562 | 85.8% | 93.0% |
| text_classification | 1,447 | 5,975 | 70.6% | 87.0% |
| text_language | 782 | 2,611 | 63.2% | 75.8% |
| text_language, images with text | 436 | 1,250 | 78.0% | 82.7% |

"Sibling vote" predicts each image's label as the majority label of the other images of its product, using the true
labels: an upper bound for a product-level vote. Images of one product deliberately show different views, so
perspective and subject vary within a product. Background, individuals and text are shared more often, but even a
perfect vote with true labels gets 13–36% of the images wrong on those dimensions. Product-level aggregation could
therefore only be a tie-breaker, mainly for text language on images with text; we did not use it.

## 4. Shift between train and dev (`product_consistency.py`)

Dev and train have nearly the same category mix (total-variation distance 0.026), and only 0.3% of dev images belong
to a product type absent from train. A distribution shift does not explain why some prompt-test gains did not carry
over to public dev; the size of the rarest classes does (section 1).
