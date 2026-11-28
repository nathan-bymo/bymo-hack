# Phase 4 — final fit of the mixed model on all labelled images

Phase 4 continues the phase-3 mixed checkpoint (`mixed-modal-20261128-033303/model/best`, public dev 0.84215) on
**all 11,876 labelled images**, prompt-test included, plus the 1,040 synthetic text overlays. Unlike the other phase
folders, it has no code copy: it uses the live implementation in
[`src/training/apertus-8B-mixed-lora-heads/`](../apertus-8B-mixed-lora-heads/) with the `--fit-all` mode
(`train_mixed.py`) and the `FIT_ALL` variable of the job generator (`build_hf_job.py`).

## Why this data

The [dataset exploration](../../db_exploration/RESULTS.md) shows that the rarest classes are rare in the whole dataset:
21 human-labelled Dutch cells, 47 Italian, 60 certification seals. Prompt-test holds 4, 4 and 6 of them. Once the model
and its epoch have been chosen (phase 3), the standard last step is to return the held-out images to training: for
Dutch this is +24% of the available examples. The exploration found no confirmed label errors (0 of 6 inspected) and no
shift between train and dev (category total-variation distance 0.026), so no labels or images are removed or changed.

## Recipe

| | Phase 3 (source) | Phase 4 |
|---|---|---|
| Start | final-abt (SigLIP 2 + VQ) | phase-3 mixed, joint epoch 3 |
| Real images | 10,683 (outside prompt-test) | 11,876 (all) |
| Rare-text rescaled views (`a2`) | 521 | 579 |
| Synthetic overlays (text dimensions, weight 0.25) | 1,040 | 1,040 |
| Views per epoch | 12,244 | 13,495 |
| Stages | fusion 0.5 epoch, joint up to 4 epochs | joint, 1 epoch (`JOINT_EPOCHS`) |
| Learning rates (fusion / projector + heads / LoRA) | 6e-5 / 4e-5 / 1.5e-5 | halved: 3e-5 / 2e-5 / 7.5e-6 |
| Checkpoint | best on prompt-test | **last** (prompt-test is now training data) |
| Seed | 3407 global, 3408 joint | same |

Half learning rates and a single epoch, because the phase-3 run already lost validation score at its fourth joint
epoch at full rates. The prompt-test score of the phase-4 model is measured on images it has seen: it is reported as
a sanity check only. **The decision rests on public dev alone.**

**Encoders stay frozen.** A trainable SigLIP LoRA on final-abt gave +0.0035 on public dev for about 64 minutes of H200,
and it cannot use the cached frozen features. Training the encoders in the same run would also confound two changes
in the one attempt we can afford before the deadline.

## Decision rule (28 November, deadline 14:30)

- Phase 4 replaces the phase-3 model only if its public dev score is **above 0.84215**.
- The 13:00 final run uses the phase-3 model regardless, so a valid submission exists. If phase 4 wins, the final images
  are predicted again with the phase-4 checkpoint, and its weights are published in `weights/` before 14:30.

## Launch on Hugging Face Jobs (one H200, about 75 minutes, at most about 9 $)

Prerequisites for the account that launches the job:
- `HF_TOKEN`: that account, with accepted access to `swiss-ai/Apertus-v1.5-8B` and
  `facebook/dinov3-vitb16-pretrain-lvd1689m`, and a **private** model repository for the output (`UPLOAD_REPO`).
- `KAGGLE_API_TOKEN`: a Kaggle account in the competition (the real images are downloaded from Kaggle).
- `SOURCE_HF_TOKEN`: a fine-grained read token of the account that owns the source checkpoint and the synthetic data
  (`nathan-monzani-bymo/galaxus-vlh-private`, `nathan-monzani-bymo/galaxus-data-private`). Revoke it after the run.

From the repository root, with the authorised `data/train.csv` in place:

```sh
python src/training/apertus-8B-mixed-lora-heads/build_hf_job.py      # writes data/hf_jobs/hf_job_mixed.py
export HF_TOKEN=… KAGGLE_API_TOKEN=… SOURCE_HF_TOKEN=…
hf jobs uv run --flavor h200 --timeout 110m -d \
  --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN --secrets SOURCE_HF_TOKEN \
  -e UPLOAD_REPO=YOUR_ACCOUNT/galaxus-phase4-private \
  -e SOURCE_REPO=nathan-monzani-bymo/galaxus-vlh-private \
  -e INIT_FROM=mixed-modal-20261128-033303/model/best \
  -e DINO_REVISION=5931719e67bbdb9737e363e781fb0c67687896bc \
  -e SYNTHETIC_REPO=nathan-monzani-bymo/galaxus-data-private \
  -e MIXED_SYNTHETIC_FILE=synthetic/dino_text_1040.tar \
  -e FIT_ALL=1 -e JOINT_EPOCHS=1 -e WALL_MINUTES=80 -e RUN_NAME=phase4-fit-all \
  data/hf_jobs/hf_job_mixed.py
hf jobs logs JOB_ID      # follow the run
```

`WALL_MINUTES=80` bounds the training part; prompt-test and dev predictions follow. If the epoch cannot finish in
time, the job keeps the unchanged phase-3 checkpoint (nothing is lost). Outputs in `UPLOAD_REPO`:
`phase4-fit-all/model/best/` (weights), `phase4-fit-all/model/training.json`,
`phase4-fit-all/predictions/dev.csv` (for the Kaggle check) and `prompt_test.json`.

## Status

Prepared and tested on CPU on 28 November (7 mixed tests pass; the fit-all data path was checked on the real
`train.csv`: 11,876 + 579 + 1,040 = 13,495 views). Not yet run on GPU. Results will be added to `RESULTS.md`.
