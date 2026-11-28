# Apertus 8B: LoRA and classification heads

## Run and data

The P11 run jointly trained rank-32 LoRA adapters on all 32 Apertus layers and six classification heads. It used the native VQ image representation at approximately 256², product context, cross-entropy without class weights and human labels, including human-labeled cells that were not relevant to evaluation. Approximately 83.9 million parameters were trainable. No SigLIP adapter or OCR was used.

The training set was `training_rows`: 9,344 images, excluding all products in coverage. Rare-class repetition and mirroring produced 12,491 items per epoch. The team fixed three epochs, selected the final checkpoint without a validation-based decision and used one seed (3407). Batch size was 8 with two accumulation steps; learning rates were 2e-4 for LoRA and 1e-3 for the heads. Base model: `swiss-ai/Apertus-v1.5-8B`, revision `1519de0ae7f3d114879291810a58259966f64c99`.

## Outcome

| Evaluation set | Images | Macro F1 | Weighted F1 | Subject | Background | Perspective | Individuals | Text classification | Text language |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coverage | 2,532 | 0.781 | 0.905 | 0.841 | 0.911 | 0.625 | 0.981 | 0.806 | 0.521 |
| base | 2,363 | 0.786 | 0.913 | 0.848 | 0.914 | 0.628 | 0.990 | 0.817 | 0.521 |
| prompt-test | 1,193 | 0.791 | 0.915 | 0.860 | 0.894 | 0.620 | 0.990 | 0.805 | 0.577 |

The reported public Kaggle dev score was **0.796**. It is a separate evaluation from coverage. Predictions were reportedly recomputed with the official metric and agreed with the job log. After downloading the model from its private HF repository, the team reported identical coverage predictions to the training job.

Training took 2,343 steps and 46.7 minutes on an H200, at 13.4 examples/s, with 139 GB peak reserved GPU memory. Coverage prediction took 191.5 seconds, approximately 13.2 images/s end to end and 22.0 GB peak allocated memory. The complete job was 62.3 minutes; the reported $5.19 is an estimate, not a verified invoice.

## What worked and what remained difficult

The six heads avoid invalid JSON and yielded strong subject, background and individuals scores. Perspective remained difficult: left and right front angles scored 0.59 and 0.52, and back-angle-left 0.28 on coverage. Text language was the weakest dimension (0.521); Dutch had only two human-labeled training cells and scored zero on its 20 coverage examples, while Italian scored 0.28 on 30 examples. These are properties of this split and run, not a general measurement of language ability.

The earlier generative LoRA scored 0.734 on coverage but used only 1,500 training images and different settings. The difference cannot be attributed to the classification heads alone. There is one training seed and no controlled comparison against full fine-tuning at the same dataset size.

## Checkpoints and evidence

The adapter, heads and configuration are available under `weights/lora_heads_256_p11/`; the documented private HF archive path is `archive/p11_lora_heads_256_coverage0781`. The exact HF repository identifier is not recorded in the local weight README. Detailed per-class scores and training notes are retained in `results_full_h200.md`; the documented Kaggle submission ID is 59018375. Raw predictions are private and are not in this repository. Confirm the HF repository and original run/configuration with the colleague if the remote artifact needs to be audited independently.
