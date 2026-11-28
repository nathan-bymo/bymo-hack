# Phase 3 — DINOv3 and mixed results

The scores below are macro F1. Prompt-test contains 1,193 images; Kaggle public dev is a separate unlabeled set. Facts below are copied from the original run reports, each with a Source note (aggregates only).

## Delivered architecture (mixed)

Frozen Apertus VQ tokens (about 256 px, about 256 tokens) + frozen SigLIP 2 (`google/siglip2-base-patch16-224`, pooled 7 × 7 × 768) + frozen DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, revision `5931719e…`, 288 px, pooled to 7 × 7 × 768) fused through a zero-initialised LayerNorm + Linear 768 × 768 residual, so step 0 reproduces `final-abt` exactly (asserted on GPU). Projector 768 → 4096 → 4096 gives 49 soft tokens. docTR OCR text goes into the prompt. Apertus 1.5 8B truncated to 24 of 32 layers, LoRA rank 32, α 32, dropout 0.05; readout = last token ‖ mean VQ ‖ mean soft tokens + spatial differences + 2D attention pooling; 512-unit hidden layer; six heads. About 98.5 M trainable parameters. Source: `settings.json` of [nathan-monzani-bymo/apertus-8B-mixed](https://huggingface.co/nathan-monzani-bymo/apertus-8B-mixed), `README.md` (original report, git history); the 98.5 M count is stated in `paper/paper.tex` (mixed section), not in an older run report.

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

The selected model scored 0.84282 macro F1 and 0.9235 weighted F1 on prompt-test, and 0.84215 on public dev, submission 59038887. The reported prompt-test dimension scores were 0.868 subject, 0.875 background, 0.776 perspective, 0.970 individuals, 0.769 text classification and 0.798 language. The reported dev throughput was 5.98 images/s for the original export. The last two decoder blocks were not unfrozen in this run. Weighted F1 recorded at the selected epoch: 0.9239 (`training.json`); 0.9235 for the re-exported predictions; the technical report rounds to 0.924. At joint epoch 4 validation loss rose from 0.305 to 0.316 (overfitting). Training time: 23 minutes to build the feature cache and about 18 minutes per joint epoch, about 2 hours in all on one H200 (Modal); peak training memory 61.25 GiB (`training.json`). Source: `RESULTS.md` (original report, git history) §1, `training.json` of [nathan-monzani-bymo/apertus-8B-mixed](https://huggingface.co/nathan-monzani-bymo/apertus-8B-mixed).

Note on the prompt-test score: the technical report's 0.843 is the selected checkpoint on prompt-test; recorded values are 0.84308 (validation in `training.json`) and 0.84282 (re-exported predictions). Both round to 0.843.

**DINO-only (81 tokens) prompt-test dimensions:** 0.772 subject, 0.847 background, 0.783 perspective, 0.944 individuals, 0.719 text classification, 0.781 language. It fit 10,683 images plus 521 rare-text 320 px views (11,204 items per epoch). Source: `RESULTS.md` (original report, git history) §2, `README.md` (original report, git history).

**Public leaderboard (28 November morning).** Best team score: two-decoder ensemble (50/50 SigLIP + VQ and DINOv3 81 tokens), **0.84700**, submission 59032216; not delivered because a runtime ensemble (+0.005 over the mixed model) halves the speed. Organiser references: Qwen 3.5 9B 0.917, Gemini 4 Argon 0.834, Gemini 3.8 Flash 0.806. Source: `RESULTS.md` (original report, git history) §1 and §3, `README.md` (original report, git history).

## Combinations and distillation

A 50/50 mixed and DINO probability combination scored 0.8557 on prompt-test. A teacher using mixed subject predictions and the combined predictions for other dimensions scored 0.8603. This teacher selection used prompt-test and is therefore optimistic. Soft teacher targets were prepared for development images, but the documented distillation pilot had not been launched; these targets do not establish a trained student result. Teacher vs mixed: +0.017 [+0.006, +0.030] (paired product bootstrap). Teacher targets cover the 5,261 dev images; cells are masked by the scoring rules and a per-dimension confidence threshold (teacher precision ≥ 0.90 on prompt-test), keeping subject 93%, background 81%, perspective 59%, individuals 81%, text classification 91%, text language 76% of cells. The pilot code (7 CPU tests pass) was **not run**; no result is reported. Promotion rule: +0.008 macro F1 on prompt-test with positive paired bootstrap, weighted F1 within 0.002, at least 90% of throughput. Source: `RESULTS.md` (original report, git history) §2 and §4.

Zero-shot 70B as a teacher was not pursued: on coverage, even at confidence ≥ 0.90 its perspective answers are right only 63.4% of the time. Source: `src/plain-benchmark/apertus-8B-benchmark/teacher_student_zeroshot.md` (git history, commit 36d3aff; deleted from the working tree).

## Inference optimizations

| Configuration | Set | Macro F1 | Images/s |
|---|---|---:|---:|
| Original export | prompt-test | 0.84282 | 6.23 |
| Fast path | prompt-test | 0.84318 | 7.51 |
| Fast path | public dev images | Not measured locally | 8.53 |
| Fast + TF32, H100 | prompt-test | 0.84242 | 13.83 |
| Fast + TF32, H100 | public dev images | Not measured locally | 14.34 |

Measured image to CSV, model loading excluded; H200 except the TF32 rows (H200s were unavailable, so H100 80 GB, batch reduced to 16 automatically). Original export 6.23 images/s on prompt-test here (5.93 images/s in the training-job export) and **5.98 images/s on dev**. Fast path = merged LoRA, images sorted by VQ size, batch 32, image decoding in a thread pool, OCR in its own thread one batch ahead. Peak memory: 47 GB original export, 79–88 GB fast path, **23 GB** fast + TF32 at batch 16 on H100. Parity with the original export: fast path 99.5–99.9% per dimension on dev; fast + TF32 99.1–100% on dev. With TF32 the VQ stage fell from 84 to 18 s on prompt-test and OCR became the critical path. Source: `RESULTS.md` (original report, git history) §5.

TF32 changed approximately 0.7–0.9% of VQ tokens, so it was not bitwise equivalent; macro F1 effect −0.0008 versus the fast path on prompt-test (−0.0004 versus the original export). In the separate fast-path TTA test, selective perspective flipping applied to 3.1% of images and raised macro F1 from 0.84318 to 0.84496 (+0.0017; perspective 0.7781 → 0.7887; measured without TF32), with reported 95% interval for the gain of +0.0001 to +0.0039 and approximately 3% throughput cost. A later HF artifact, `infer-dev_final-20261128-084357/infer/dev_final.json`, records the combined fast + TF32 + selective TTA preset on 5,261 dev images: H200, batch 32, 11.68 images/s, 29.2 GiB peak, 209 TTA images (4.0%) and no replacements; model load 76 s; OCR busy 433 s of 450 (critical path). Agreement with the submitted predictions (59038887): 99.4–100% per dimension, 97.5% for perspective (expected, since TTA changes mirrored images on purpose). The first attempt stopped at its first batch because the selective TTA updated in place a tensor created in `inference_mode`; fixed in commit `37aea4b`. Mirroring every image instead: +0.0019 [−0.0031, +0.0071] at 4.16 images/s. This verifies a combined dev inference run, not its Kaggle score or the final-image submission. Source: `RESULTS.md` (original report, git history) §6 and §8.

**Cost estimates (derived from the measured rates at $5/h per H200, not invoices):** 100,000 images at 11.68 images/s ≈ 2.4 H200 GPU-hours ≈ $12/day; at 14.34 images/s on H100 without TTA ≈ 1.9 GPU-hours; the 5,292 final images ≈ 8 minutes at the dev rate. If H200s are unavailable the inference job falls back to H100 or A100 80 GB; the A100 path never ran this model.

## Label-error search

`src/scripts/label_errors.py` flagged train cells where at least two models that had not seen the product agree with high confidence against the human label: **57 candidates** (kept outside the repository). Visual check of the most extreme candidate per dimension: **0 of 6 is a certain error** (4 human labels correct, 2 ambiguous). Confident disagreements are shared blind spots of the models, not label errors; the list was not sent. Source: `RESULTS.md` (original report, git history) §7.

## Provenance limit

The original training ran on Modal. The present phase contains a later copy of the code; the HF launch is an unverified adaptation. Git history shows that the mixed builder, training and inference files changed after the initial mixed implementation. Recover the original Modal source bundle, dependency lock, inputs and effective configuration before claiming an exact reproduction. See [README](README.md).
