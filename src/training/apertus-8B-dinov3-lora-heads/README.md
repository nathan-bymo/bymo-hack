# Apertus 8B with DINOv3, OCR and six heads

This experiment replaces SigLIP and native VQ image tokens with a frozen DINOv3 ViT-B/16 encoder. A 288 px
full-frame view yields 81 position-preserving visual tokens after pooling. The projector starts fresh.
The run loads LoRA, the six heads and 2D attention pooling from the 24-layer SigLIP + OCR checkpoint
`final-abt/model/best`. The existing head widths are retained for transfer. Features that belonged to VQ
are zero. OCR reads the original images.

- Fit on all 10,683 images outside prompt-test. Select on the 1,193 original prompt-test images.
- For each image with a relevant human label in one of the five target classes, add at most one full-frame
  320 px DINO view. The classes are Dutch, Italian and `other` for language, and `certification_seal`
  and `other` for added text. The current data produce 521 additional views and 11,204 items per epoch.
- Adapt the projector, pooling and heads with LoRA frozen for half an epoch. Enable the original LoRA rank 32
  at a lower learning rate for joint training.
- Evaluate every half epoch. Save the best DINO checkpoint by macro F1. Stop on sustained regression or
  a plateau, with ceilings of five epochs and 110 minutes of optimization. The 180-minute HF timeout
  also covers downloads, image encoding, reporting and prediction. The job saves the best checkpoint
  as soon as it is available.
- Report the official metric with per-class detail and end-to-end throughput on prompt-test.
  Export predictions and probabilities for dev to the private output repository.

The original model's comparison points are 0.8214 on prompt-test and 0.83177 on public Kaggle dev.
Prompt-test is now the checkpoint selection set for this experiment. A dev comparison is needed for
independent confirmation.

## Launch on the colleague's account

Use the colleague's HF account and an existing **private** output model repo. The account must have accepted
access to [Apertus](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) and
[DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
Use an already authorized private copy of the competition data or a Kaggle credential for that competition.
Export secrets through the environment. The launcher passes only their names on the command line.

`SOURCE_REPO` defaults to `OUTPUT_REPO`. If the source checkpoint is in a separate private repo,
set `SOURCE_REPO` to that repo and grant read access to the job account or supply a separate read token in
`SOURCE_HF_TOKEN`. Set `SOURCE_PATH` if its location differs from `final-abt/model/best`.

```sh
export OUTPUT_REPO=colleague/galaxus-dino-private
export SOURCE_REPO=team/existing-galaxus-model-private
export DATA_REPO=team/existing-galaxus-data-private
bash src/training/apertus-8B-dinov3-lora-heads/launch.sh --dry-run
bash src/training/apertus-8B-dinov3-lora-heads/launch.sh
```

For the Kaggle data route, omit `DATA_REPO` and export `KAGGLE_API_TOKEN` or
`KAGGLE_USERNAME` plus `KAGGLE_KEY`. Budget: one H200 at the [published rate](https://huggingface.co/docs/hub/en/jobs-pricing)
with a 180-minute ceiling and one attempt. The GPU preflight runs forward at 288 and 320 px and exercises
backward through the actual model before encoding the entire split.

## Launch on Modal

Install the Modal CLI and authenticate it against the correct workspace with `modal token new`.
Modal requires a valid payment method to use GPUs. In that workspace create a secret called
`galaxus-dino` with `HF_TOKEN` for gated model access and writes to the private output repo. Include
`SOURCE_HF_TOKEN` if access to the source checkpoint uses a separate read token. If using Kaggle
instead of a private HF dataset, include `KAGGLE_API_TOKEN` or `KAGGLE_USERNAME` and `KAGGLE_KEY`.

Generate the same job bundle, then use the CPU-only access check. It verifies readable resources
and that the repositories are private. Write access is exercised when the first checkpoint is uploaded.

```sh
python3 src/training/apertus-8B-dinov3-lora-heads/build_hf_job.py
modal run src/training/apertus-8B-dinov3-lora-heads/modal_app.py \
  --output-repo nathan-monzani-bymo/galaxus-vlh-private \
  --data-repo nathan-monzani-bymo/galaxus-data-private --preflight
```

Use the same arguments for the GPU run, replacing `--preflight` with `--gpu H200` or `--gpu H100`
and adding `--detach` after `modal run`. The job uses one container and no retries. The default
timeout is 180 minutes. `--long-run` allows up to 10 epochs and 255 minutes total, with the first
validation after half an epoch and one validation per epoch after that. It keeps the same early
stopping rule and reports macro F1, weighted F1, loss and per-dimension F1 at each validation.
H200 is recommended for its memory headroom. The bundle contains source and a dataset checksum.
The original images and labels are downloaded directly from the authorized private source.
For this private HF dataset route, `--local-hf-auth` can inject the authenticated local HF token
only into the selected Modal function instead of maintaining a named Modal secret.

## Fine-tuning method

This experiment keeps the original LoRA scaling and tensors to isolate the change in image representation.
[DoRA](https://arxiv.org/abs/2402.09353) is the first follow-up if the decoder adaptation appears limiting.
It adds learnable weight magnitudes, so it needs its own warm-start equivalence test and matched comparison.
[rsLoRA](https://arxiv.org/abs/2312.03732) is relevant if we raise rank, with scaling adjusted so the transferred
checkpoint is not amplified at step zero. [PiSSA and LoRA-GA](https://huggingface.co/docs/peft/package_reference/lora)
are most relevant when initializing a new adapter. RandLoRA, FourierFT, LoRA-XS and LR-LoRA change the adapter
parameterization and are separate experiments.

## Local checks

```sh
python -m pytest -q src/training/apertus-8B-dinov3-lora-heads/test_dino.py
python src/training/apertus-8B-dinov3-lora-heads/build_hf_job.py
```
