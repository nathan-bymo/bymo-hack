# Phase 2 — LoRA and visual-adapter improvements

This folder is a standalone working root with local copies of both implementations and their helpers. It does not depend on code in another phase or the parent repository. The copies reflect the current code and are not frozen historical snapshots. See [RESULTS.md](RESULTS.md) for consolidated evidence.

## Setup and split

Run commands from this folder. Place the authorized challenge `train.csv` and `metric.py` in `data/`, configure a private HF output repository, GPU access and gated model access, and export `HF_TOKEN` and `KAGGLE_API_TOKEN` securely. The generated bundle checks the training CSV hash against the remote data. Kaggle data must be accessible to the job. A private data repository can be used by the visual-adapter builder through `DATA_REPO`.

- **P12 LoRA + heads:** the 9,344-image `training_rows` split excludes coverage products. Input 512²; two epochs; spatial features and 2D attention pooling. Coverage contains 2,532 images and prompt-test is its 1,193-image subset.
- **Visual adapter `final-abt`:** 9,645 images after excluding prompt-test products and an internal product-level validation set; prompt-test remains held out. Resume from the earlier OCR checkpoint. P13 uses 9,645 real images plus 939 synthetic images; P14 uses a larger encoder; P15 trains on all 11,876 labels and has no valid prompt-test holdout.

## Hugging Face launch patterns

```sh
python src/training/apertus-8B-lora-heads/build_hf_job.py
```

P12 needs the exact historical 512², spatial and pooling `TRAIN_ARGS`; those arguments and the original immutable job bundle have not yet been recovered, so a command claimed to reproduce P12 is not available. The generated job supports `SPLIT=full`; see its source for current options. P12 results alone do not identify the historical command.

For a visual-adapter continuation, replace placeholders with the authorized private repository, previous checkpoint path and verified training arguments:

```sh
python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO -e RUN_NAME=replication-vlh \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PART=all_but_test -e OCR=1 -e INIT_FROM=YOUR_CHECKPOINT_PATH \
  -e 'TRAIN_ARGS=--epochs 3 --max-minutes 65 --batch-size 8 --accumulation 2 --lr-lora 1e-4 --lr-projector 5e-5 --lr-heads 5e-4 --spatial --attention-pool' data/hf_jobs/hf_job_vlh.py
```

The training flags above match the effective training command recorded in the later job log; the command still uses current code and does not establish historical equivalence. Set `INIT_FROM` to the historical `phase1-ocr/model/best` checkpoint at a verified revision. The original job ID for the later `final-abt` run is `6b09e0f06c5d61fff189ab6a`; `hf jobs logs JOB_ID` can inspect it when the account has access. The continuation example is not a verified exact recreation of the full multi-stage run. No GPU launch was performed during this restructuring.

## Historical verification

The log of HF job `6b09e0f06c5d61fff189ab6a` confirms the recorded `final-abt` data split and the training flags shown above, including the `phase1-ocr/model/best` starting checkpoint. Git shows later changes to the adapter implementation and job builder; identical current source, dependencies, and checkpoint revision have not been established. P12 and P13–P15 need their original job bundles and logs for equivalent verification.

The historical P12 seed is 3407 and its GPU was H200. The final visual-adapter HF log confirms the exact training flags above and an H200; the retained older visual-adapter bundle differs from current code. The phase copy and current builder therefore cannot be described as the original run without recovering and comparing the immutable job bundle.

## Recovered original visual-adapter continuation

HF job `6b09e0f06c5d61fff189ab6a` mounted the script at bucket path `20261127T195838-fe34f7/hf_job_vlh.py` on one H200. The exact recovered script SHA-256 is `2908f88b347cd2a19a5a9cb8b38e0c2a1ca2f85bae265f2ce8640c16bf3a3ece`; its local copy is retained under the git-ignored `data/historical_hf/phase-2/`. The script says not to commit it. Authorized users can retrieve it with:

```sh
mkdir -p data/historical
hf buckets cp hf://buckets/YOUR_BUCKET_OWNER/jobs-artifacts/20261127T195838-fe34f7/hf_job_vlh.py data/historical/hf_job_vlh.py
shasum -a 256 data/historical/hf_job_vlh.py
```

Job metadata confirms `PART=all_but_test`, `OCR=1`, `INIT_FROM=phase1-ocr/model/best`, `RUN_NAME=final-abt`, and the training flags documented above. The starting checkpoint, data and private output location must be available to the new job. Use a new output repository or run name; do not overwrite historic artifacts. The bundle embeds source but not every transitive dependency version or all input hashes.
