# Apertus 8B: visual adapter, LoRA, heads and OCR

## Architecture and distinction

The final model combines frozen SigLIP 2, a trainable projector providing 49 visual tokens, the frozen native Apertus VQ tokenizer, OCR text from frozen docTR models, and Apertus 1.5 8B truncated to 24 of 32 layers. Rank-32 LoRA adapts Apertus; six heads classify the outputs. The final version also uses left/right and top/bottom spatial differences and attention pooling with 2D positions. This is LoRA fine-tuning, not full fine-tuning of the Apertus decoder.

Base model: `swiss-ai/Apertus-v1.5-8B`, revision `1519de0ae7f3d114879291810a58259966f64c99`. The local final configuration records `google/siglip2-base-patch16-224`, OCR, spatial features and attention pooling.

## Chronology of the actual training

| Step | Checkpoint | Training images | Change | Training duration | Reported cost |
|---|---|---:|---|---:|---:|
| 1 | `phase1/model/best` | 8,403 | Projector, LoRA and heads; two epochs; original crop augmentation | 31 min; 58 min job | ~$4.80 |
| 2 | `phase1-ocr/model/best` | 8,403 | Resumed step 1; OCR added, corrected scale augmentation; one additional epoch | 19 min | ~$2.70, plus ~$2.40 for an earlier out-of-memory attempt |
| 3 | `final-abt/model/best` | 9,645 | Resumed step 2; all labeled products except prompt-test, spatial features and 2D attention pooling; three additional epochs | 63 min; 73 min job | ~$6.10 |

The documented steps used H200 and BF16. Step 3 had 14,459 augmented items per epoch. The runs used focal loss γ=2, effective-number class weights, rare-class oversampling, clipping and a cosine schedule. Steps 2–3 used learning rates 1e-4 for LoRA, 5e-5 for the projector and 5e-4 for the heads. Reported costs are estimates. The earlier crop augmentation could cut off products and invalidate labels; a scale-only approach replaced it. An initial OCR attempt ran out of memory before the successful continuation.

## Evaluation: keep the sets separate

Prompt-test contains 1,193 images and was reported as excluded from training and checkpoint selection throughout these steps. It is the relevant held-out comparison for the final checkpoint. Coverage contains 2,532 images, but the final run trained on some of its products; therefore a final-checkpoint coverage score would not be valid.

| Checkpoint | Set | Macro F1 | Weighted F1 | Subject | Background | Perspective | Individuals | Text classification | Text language |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 1 | prompt-test | 0.750 | 0.887 | 0.828 | 0.803 | 0.586 | 0.960 | 0.772 | 0.550 |
| Step 2 + OCR | prompt-test | 0.741 | 0.891 | 0.816 | 0.808 | 0.544 | 0.955 | 0.771 | 0.554 |
| Final | prompt-test | **0.821** | **0.917** | 0.846 | 0.881 | 0.755 | 0.959 | 0.766 | 0.720 |

For coverage, step 1 scored 0.742 macro F1, step 2 0.748, and step 2 with flip TTA 0.749. These are not final-model coverage scores. The public Kaggle dev score reported for `final-abt/model/best` was **0.83177**; final-image submission rows still had defaults. Dev inference ran on an L40S at 4.9 images/s, approximately 20 minutes for the job.

## Interpretation and remaining weaknesses

OCR alone did not improve prompt-test macro F1 in step 2. Step 3 improved prompt-test, particularly perspective and language, but it also changed training data, spatial features, pooling and training duration together. The improvement cannot be assigned to any one change. Coverage analysis reported better German and French with OCR, but this does not establish its isolated effect across all settings.

Average image-token features were associated with poor left/right discrimination; the final spatial and positional features were designed to preserve that information. The final model still had weak results on certification seals (F1 0.40, six examples), other added text (0.55), Dutch (0.40, four examples), Italian (0.67, four examples), and certain angled perspectives. Rare-class estimates are noisy. Flip TTA improved perspective in one test but reduced individuals accuracy and throughput; it was not adopted.

On H200, end-to-end throughput was 13.8 images/s for step 1 without OCR and 6.2 images/s for the final model on prompt-test. The reported final timing breakdown included VQ and OCR, both substantial contributors. Post-hoc calibration showed limited or negative gains in the reported experiments.

## Checkpoints and evidence

The recorded private HF repository is `nathan-monzani-bymo/galaxus-vlh-private`, path `final-abt/model/best`. A local copy is in `weights/apertus-8B-vadapter-lora-heads-final/`. Detailed run notes are retained in `RISULTATI.md` and data/augmentation details in `DATI.md`. The report records job IDs for some stages but not all exact revisions, commands or complete raw logs; preserve the private job artifacts to audit the chronology and reproduce it exactly.
