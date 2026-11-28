# Phase 2 — results

Scores below are macro F1. Coverage (2,532 images, with a rare-class supplement), prompt-test (1,193 images) and public Kaggle dev are distinct sets. Per-class tables and complete run logs remain in the linked original reports; this file consolidates their main results and experimental conditions.

## P12: LoRA + heads, spatial features and pooling

P12 combined 512² images, left/right and top/bottom differences, 2D attention pooling and two training epochs on 9,344 images. P11 used 256² and three epochs. P12 obtained 0.8056 on coverage, 0.7913 on prompt-test and 0.806 on public dev, versus P11's 0.7808, 0.7909 and 0.796. Coverage perspective rose from 0.625 to 0.744 and text language from 0.521 to 0.611; subject and background declined. Multiple settings changed at once, preventing attribution to individual additions. Reported H200 training time was 122 minutes. P12 had 91.6 M trainable parameters (8 attention-pooling queries); paired 95% intervals versus P11: coverage +0.025 [+0.007, +0.044], prompt-test +0.000 [−0.022, +0.025]; public dev +0.010 (0.80566 exact). Full job 168 minutes, about $14 (estimate); inference 3.9 images/s on H200, 32.5 GB. Source: `src/training/apertus-8B-lora-heads/results_p12_512_pooling.md`, `src/training/benchmark/README.md`.

## Visual adapter: OCR and spatial continuation

| Checkpoint | Training images | Prompt-test macro F1 | Prompt-test weighted F1 | Coverage macro F1 | Public dev |
|---|---:|---:|---:|---:|---:|
| Initial SigLIP + VQ | 8,403 | 0.750 | 0.887 | 0.742 | — |
| OCR continuation | 8,403 | 0.741 | 0.891 | 0.748 | — |
| `final-abt`, OCR + spatial pooling | 9,645 | 0.8214 | 0.9166 | Not held out | 0.83177 |

The final checkpoint had prompt-test dimension scores 0.846 subject, 0.881 background, 0.755 perspective, 0.959 individuals, 0.766 text classification and 0.720 language. Training data, architecture and duration changed together. OCR alone did not improve prompt-test macro F1. The first augmentation could crop products and was replaced with scale-only augmentation. Initial OCR training also had an out-of-memory attempt. The first two checkpoints were evaluated on coverage; the final checkpoint trained on some coverage products, so that comparison is invalid for it. Source: `src/training/apertus-8B-vadapter-lora-heads/RESULTS.md`, `src/training/apertus-8B-vadapter-lora-heads/RISULTATI.md`.

Further facts from the same reports:

- **OCR stage** (`phase1-ocr`): prompt-test dimensions 0.816 subject, 0.808 background, 0.544 perspective, 0.955 individuals, 0.771 text classification, 0.554 language (macro 0.750 → 0.741). On coverage OCR helped languages with training data (German 0.79 → 0.86, French 0.63 → 0.69), not Dutch. 19 min training, about $2.70 plus about $2.40 for an out-of-memory attempt.
- **Spatial readouts:** with mean-pooled image tokens only, the model chose the correct left/right side in 55% of cases; features that change sign under mirroring plus 2D attention pooling took prompt-test perspective from 0.54 to 0.76 (together with more data and epochs in the same step).
- **Random crops:** the product touches the image border in 269 of 300 sampled images; a 10% crop cut content in 231 of 300, turning some `full_product` views into `detail_closeup`. Replaced by scale-only augmentation from the OCR stage on.
- **Post-hoc calibration:** per-class biases on coverage 0.743 → 0.738; one logit-adjustment parameter per dimension 0.743 → 0.745. Not adopted.
- **Flip TTA (all images) on `final-abt`:** +0.019 perspective, −0.009 individuals, −17% speed; not adopted.
- **`final-abt` weakest prompt-test classes:** certification seal 0.40 (6 examples), other text 0.55, Dutch 0.40 and Italian 0.67 (4 examples each).
- **Speed:** 6.2 images/s on H200 (prompt-test), 4.9 images/s on L40S (dev); 6.91 images/s on the full dev set on H200 (see [phase 1 results](../phase-1/RESULTS.md#h200-benchmark-of-the-three-8b-models-on-public-dev)). Final step: 63 min training, 73 min job, about $6.10 (estimate).

HF job `6b09e0f06c5d61fff189ab6a` confirms that `final-abt` encoded `all_but_test` at side 256 with OCR and validation fraction 0.1. The recorded training command used three epochs, batch 8 with accumulation 2, learning rates 1e-4 for LoRA, 5e-5 for the projector and 5e-4 for heads, spatial features, attention pooling, OCR and initialization from `phase1-ocr/model/best`. It confirms 9,645 training images, 1,038 validation images, 14,459 augmented items per epoch and 2,712 steps. The best internal-validation macro F1 was 0.8264 at epoch 3; prompt-test was 0.8214. HF job metadata identifies the original bucket-mounted job script; the recovered bundle differs from the current phase in 9 of its 11 bundled files. Exact dependency and input environment parity remains unverified.

## P13–P15: subsequent variants

| Run | Change | Prompt-test macro F1 | Public dev |
|---|---|---:|---:|
| `final-abt` | Reference | 0.821 | 0.83177 |
| P13 | 9,645 real + 939 synthetic training examples, one additional epoch | 0.847 | 0.829 |
| P14 | SigLIP 2 so400m 512, 196 visual tokens, 9,645 images | 0.833 | Not measured |
| P15 | One additional epoch on all 11,876 labeled images | Not held out | 0.83144 |

P13 improved prompt-test language F1 from 0.720 to 0.840, but additional training and synthetic data were combined. P14 also changed the encoder and resolution; its prompt-test gain over the reference had a reported 95% interval of −0.001 to +0.027. The P15 checkpoint cannot be evaluated on prompt-test as a held-out set because it trained on all labeled images. Details:

- **P13:** prompt-test +0.026 [95% CI +0.004, +0.045], weighted +0.005, public dev −0.003 (0.82872 exact). Without classes of fewer than 10 examples the prompt-test gain is about +0.010; the gain sits in the rarest classes (Dutch 0.40 → 0.86, Italian 0.67 → 0.86, seal 0.40 → 0.50, 4–6 examples each). The 939 synthetic images are those of the 1,040 generated whose product is in this training split. One epoch, 22.9 min, about 73 min job on Modal H200; 6.67 images/s on prompt-test.
- **P14:** `google/siglip2-so400m-patch16-512`, 196 tokens; P14 − `final-abt` = +0.012 [95% CI −0.001, +0.027] on prompt-test; 6.23 images/s and **80.7 GB** peak inference memory on H200; stopped before dev prediction.
- **P15:** public dev 0.83144 vs 0.83177 (−0.0003, a tie); one epoch, 23.0 min, about 46 min job; 7.49 images/s on dev.

Source: `src/training/apertus-8B-vadapter-lora-heads/results_p13_synthetic.md`, `src/training/apertus-8B-vadapter-lora-heads/results_p14_p15.md`.

## Trainable SigLIP LoRA on `final-abt` (run `siglip-lora-1128-0815`)

Rank-16 LoRA on all SigLIP layers (2.65 M parameters), trained together with the decoder LoRA, projector and heads, 10,683 images outside prompt-test, two epochs, no synthetic data. The checkpoint was selected on prompt-test.

| Set | Starting point (`final-abt`) | SigLIP LoRA | Gain |
|---|---:|---:|---:|
| prompt-test (reloaded checkpoint) | 0.8214 (0.8207 re-read in-model) | 0.8356 | about +0.015 |
| public dev (submission 59043423) | 0.83177 | **0.83526** | +0.0035 |

Prompt-test dimensions: 0.860 subject, 0.873 background, 0.777 perspective, 0.970 individuals, 0.793 text classification, 0.740 language. Not an ablation (other parts trained too; no frozen-encoder control). Training 37.1 min, job 63.9 min, about $5.96; 6.75 images/s (prompt-test) and 7.43 (dev), 79.8 GB. Not part of the delivered model. Source: `src/training/apertus-8B-vadapter-lora-heads/results_siglip_lora.md`, `src/training/benchmark/README.md`.

## Provenance limit

Phase code is a current implementation, not the original immutable job bundle. Compare original source archives, model and dataset revisions, complete settings and dependencies before asserting exact reproduction. See [README](README.md).
