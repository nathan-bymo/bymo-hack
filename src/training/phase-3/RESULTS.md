# Phase 3 — DINOv3 and mixed results

The scores below are macro F1. Prompt-test contains 1,193 images; Kaggle public dev is a separate unlabeled set. The original [mixed run report](../apertus-8B-mixed-lora-heads/RESULTS.md) retains detailed configurations, per-class TTA analysis, throughput and artifact paths.

## DINOv3 and mixed training

The DINO-only 81-token model scored 0.8077 on prompt-test, which was also used to select its checkpoint. The mixed model combines native VQ, SigLIP 2, DINOv3, OCR, a single 24-layer Apertus decoder, LoRA and six heads. Run `mixed-modal-20261128-033303` started from `final-abt`, trained with 10,683 real and 1,040 synthetic images, and used 1,193 validation images according to the saved `training.json`. The available original-era and current code select prompt-test for validation. Thus prompt-test was excluded from fitting but used for checkpoint selection; the older report's claim of 1,038 internal validation images is inconsistent with the checkpoint metadata and code.

| Mixed training stage | Validation macro F1 (1,193 images) |
|---|---:|
| Starting checkpoint | 0.82116 |
| DINO fusion only, 0.5 epoch | 0.82872 |
| Joint epoch 1 | 0.83321 |
| Joint epoch 2 | 0.83661 |
| Joint epoch 3, selected | 0.84308 |
| Joint epoch 4 | 0.83661 |

The selected model scored 0.84282 macro F1 and 0.9235 weighted F1 on prompt-test, and 0.84215 on public dev, submission 59038887. The reported prompt-test dimension scores were 0.868 subject, 0.875 background, 0.776 perspective, 0.970 individuals, 0.769 text classification and 0.798 language. The reported dev throughput was 5.98 images/s for the original export. The last two decoder blocks were not unfrozen in this run.

## Combinations and distillation

A 50/50 mixed and DINO probability combination scored 0.8557 on prompt-test. A teacher using mixed subject predictions and the combined predictions for other dimensions scored 0.8603. This teacher selection used prompt-test and is therefore optimistic. Soft teacher targets were prepared for development images, but the documented distillation pilot had not been launched; these targets do not establish a trained student result.

## Inference optimizations

| Configuration | Set | Macro F1 | Images/s |
|---|---|---:|---:|
| Original export | prompt-test | 0.84282 | 6.23 |
| Fast path | prompt-test | 0.84318 | 7.51 |
| Fast path | public dev images | Not measured locally | 8.53 |
| Fast + TF32, H100 | prompt-test | 0.84242 | 13.83 |
| Fast + TF32, H100 | public dev images | Not measured locally | 14.34 |

TF32 changed approximately 0.7–0.9% of VQ tokens, so it was not bitwise equivalent. In the separate fast-path TTA test, selective perspective flipping applied to 3.1% of images and raised macro F1 from 0.84318 to 0.84496, with reported 95% interval for the gain of +0.0001 to +0.0039 and approximately 3% throughput cost. A later HF artifact, `infer-dev_final-20261128-084357/infer/dev_final.json`, records the combined fast + TF32 + selective TTA preset on 5,261 dev images: H200, batch 32, 11.68 images/s, 29.2 GiB peak, 209 TTA images and no replacements. This verifies a combined dev inference run, not its Kaggle score or the final-image submission. The suspected-label-error list was not submitted because the reported investigation did not establish useful corrections.

## Provenance limit

The original training ran on Modal. The present phase contains a later copy of the code; the HF launch is an unverified adaptation. Git history shows that the mixed builder, training and inference files changed after the initial mixed implementation. Recover the original Modal source bundle, dependency lock, inputs and effective configuration before claiming an exact reproduction. See [README](README.md).
