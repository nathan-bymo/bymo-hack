# Apertus 8B: full decoder fine-tuning and classification heads

## Run and data

The P12 run trained the first 24 of 32 Apertus decoder layers and six classification heads; it did not use LoRA or SigLIP. The VQ tokenizer and embedding weights remained frozen, so “full fine-tuning” refers to the retained decoder layers, not every model parameter. Approximately 5.24 billion parameters were trained with FP32 master weights, BF16 autocast and 8-bit AdamW.

The run skipped a separate phase-1 comparison and trained on all 11,876 labeled images: 16,560 items per epoch after rare-class repetition and mirroring, three epochs and 3,105 steps. It used VQ images around 256², product context, batch 16, seed 3407, focal loss γ=2 with effective-number class weights, learning rate 1e-5 for the decoder and 1e-3 for the heads. The loss used relevant human-labeled cells.

## Outcome and what the run can establish

The reported public Kaggle dev score was **0.80473** (rounded to 0.805 in the run report), from submission 59021011. Since all labeled training images were used, coverage and the local training set cannot provide a valid held-out score for this checkpoint. Kaggle supplied an overall score, not per-dimension or per-class F1. One seed and one training run were reported.

The job completed on an A100 80 GB: approximately 165 minutes of training and 3 h 21 min for the complete job, with 57.1 GB peak memory. The reported cost was approximately $8.40, not a verified invoice. Epoch-1 and epoch-2 checkpoints were uploaded during training, and the final checkpoint was saved. Prediction on dev ran at 9.55 images/s end to end, with 18.7 GB reported memory and 5,261 valid output rows.

The P11 LoRA + heads run scored 0.796 on public dev, but trained on 9,344 images rather than 11,876 and used a different model configuration. The roughly 0.009 difference does not isolate the effect of full fine-tuning.

## What was observed

Training was reported stable, with no non-finite loss. Analysis of predicted probabilities suggests left/right perspective uncertainty: for `front_angle_left` the mean chosen probability was 0.51 and the opposite direction averaged 0.40; for `front_angle_right` they were 0.47 and 0.38. The model predicted Dutch 108 times, often with German as the second choice. These are unlabeled dev predictions and probability patterns, not measured per-class errors or proof of their cause. Overweighting rare classes and limited visual detail were proposed as possible explanations, not established findings.

No held-out set was available to choose among epochs or calibrate thresholds after this full-data run. The report suggests evaluating earlier checkpoints, flip TTA or a phase-1 run, but does not establish their outcomes.

## Checkpoints and evidence

The recorded private HF repository is `StoryofLegend/apertus-8b-galaxus-fullft`, revision `aac1375d3efb93460a564faaef00a60430d84443`, under `phase2/model/{epoch-1,epoch-2,last}`. The job ID was `6b09cdb46c5d61fff189842a`. Detailed logs and probability-derived observations are described in `results_phase2.md`; raw predictions and logs are not present here. Access to the private repository or a colleague's export is needed to independently check them.
