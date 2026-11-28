# Phase 2 — results

Scores below are macro F1. Coverage (2,532 images, with a rare-class supplement), prompt-test (1,193 images) and public Kaggle dev are distinct sets. Per-class tables and complete run logs remain in the linked original reports; this file consolidates their main results and experimental conditions.

## P12: LoRA + heads, spatial features and pooling

P12 combined 512² images, left/right and top/bottom differences, 2D attention pooling and two training epochs on 9,344 images. P11 used 256² and three epochs. P12 obtained 0.8056 on coverage, 0.7913 on prompt-test and 0.806 on public dev, versus P11's 0.7808, 0.7909 and 0.796. Coverage perspective rose from 0.625 to 0.744 and text language from 0.521 to 0.611; subject and background declined. Multiple settings changed at once, preventing attribution to individual additions. Reported H200 training time was 122 minutes. [Full P12 results and class tables](../apertus-8B-lora-heads/results_p12_512_pooling.md).

## Visual adapter: OCR and spatial continuation

| Checkpoint | Training images | Prompt-test macro F1 | Prompt-test weighted F1 | Coverage macro F1 | Public dev |
|---|---:|---:|---:|---:|---:|
| Initial SigLIP + VQ | 8,403 | 0.750 | 0.887 | 0.742 | — |
| OCR continuation | 8,403 | 0.741 | 0.891 | 0.748 | — |
| `final-abt`, OCR + spatial pooling | 9,645 | 0.8214 | 0.9166 | Not held out | 0.83177 |

The final checkpoint had prompt-test dimension scores 0.846 subject, 0.881 background, 0.755 perspective, 0.959 individuals, 0.766 text classification and 0.720 language. Training data, architecture and duration changed together. OCR alone did not improve prompt-test macro F1. The first augmentation could crop products and was replaced with scale-only augmentation. Initial OCR training also had an out-of-memory attempt. The first two checkpoints were evaluated on coverage; the final checkpoint trained on some coverage products, so that comparison is invalid for it. [Full chronology and limitations](../apertus-8B-vadapter-lora-heads/RESULTS.md), [detailed recorded results](../apertus-8B-vadapter-lora-heads/RISULTATI.md).

HF job `6b09e0f06c5d61fff189ab6a` confirms that `final-abt` encoded `all_but_test` at side 256 with OCR and validation fraction 0.1. The recorded training command used three epochs, batch 8 with accumulation 2, learning rates 1e-4 for LoRA, 5e-5 for the projector and 5e-4 for heads, spatial features, attention pooling, OCR and initialization from `phase1-ocr/model/best`. It confirms 9,645 training images, 1,038 validation images, 14,459 augmented items per epoch and 2,712 steps. The best internal-validation macro F1 was 0.8264 at epoch 3; prompt-test was 0.8214. HF job metadata identifies the original bucket-mounted job script; the recovered bundle differs from the current phase in 9 of its 11 bundled files. Exact dependency and input environment parity remains unverified.

## P13–P15: subsequent variants

| Run | Change | Prompt-test macro F1 | Public dev |
|---|---|---:|---:|
| `final-abt` | Reference | 0.821 | 0.83177 |
| P13 | 9,645 real + 939 synthetic training examples, one additional epoch | 0.847 | 0.829 |
| P14 | SigLIP 2 so400m 512, 196 visual tokens, 9,645 images | 0.833 | Not measured |
| P15 | One additional epoch on all 11,876 labeled images | Not held out | 0.83144 |

P13 improved prompt-test language F1 from 0.720 to 0.840, but additional training and synthetic data were combined. P14 also changed the encoder and resolution; its prompt-test gain over the reference had a reported 95% interval of −0.001 to +0.027. The P15 checkpoint cannot be evaluated on prompt-test as a held-out set because it trained on all labeled images. See [P13 detail and per-class tables](../apertus-8B-vadapter-lora-heads/results_p13_synthetic.md) and [P14/P15 detail and per-class tables](../apertus-8B-vadapter-lora-heads/results_p14_p15.md).

## Provenance limit

Phase code is a current implementation, not the original immutable job bundle. Compare original source archives, model and dataset revisions, complete settings and dependencies before asserting exact reproduction. See [README](README.md).
