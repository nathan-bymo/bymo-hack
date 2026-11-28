# Phase 1 — initial training directions

This folder is a standalone working root. It contains local copies of the code used by its three branches, including their shared helpers; it does not import code from other phase folders or the parent repository. These copies are from the current working tree, not verified historical snapshots. The [consolidated results](RESULTS.md) preserve the recorded results and their sources.

## Setup and split

From this folder, create `data/` and supply the authorized challenge `data/train.csv` and `data/metric.py`. Remote training also needs the challenge data from Kaggle or an appropriately authorized private HF dataset, gated model access, private output storage, credentials and GPU quota. Install a compatible Python environment and Hugging Face CLI. Export `HF_TOKEN` and, for the Kaggle examples, `KAGGLE_API_TOKEN` in your shell without committing them. The generated scripts bind the job to the local train CSV checksum. The precise dependency pins are in the generated scripts; preserve the original job bundle for historical replication.

- **LoRA + heads (P11):** `SPLIT=full` means 9,344 `training_rows` images, excluding products in the 2,532-image coverage set. Three epochs; prompt-test is a 1,193-image subset of coverage. No internal checkpoint selection in P11.
- **Decoder fine-tuning:** `PHASE=2` trains on all 11,876 labeled images; no valid held-out coverage score is available for that checkpoint. This variable is unrelated to the present folder numbering.
- **Visual adapter initial run:** 8,403 training images, 941 internally held-out images grouped by product; coverage is held out. The 9,645-image all-but-prompt-test continuation belongs to the later sequence.

## Generate and launch on Hugging Face

Run from this folder. Replace `YOUR_PRIVATE_MODEL_REPO` with a private HF model repository you can write to. The following are launch patterns for the current source; historical reproduction requires original configs and artifacts. GPU jobs incur charges.

```sh
python src/training/apertus-8B-lora-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e MODE=train -e SPLIT=full -e RUN_NAME=replication-lora \
  -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO -e 'TRAIN_ARGS=--max-minutes 80' \
  data/hf_jobs/hf_job_lora_heads.py
```

```sh
python src/training/apertus-8B-fullft-heads/build_hf_job.py
hf jobs uv run --flavor a100-large --timeout 420m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e PHASE=2 -e EPOCHS=3 -e JOB_TIMEOUT_MINUTES=420 \
  -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO data/hf_jobs/hf_job_fullft.py
```

```sh
python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO -e PREDICT_DEV=1 \
  -e 'TRAIN_ARGS=--epochs 2 --max-minutes 45' data/hf_jobs/hf_job_vlh.py
```

For a visual-adapter access check, use `--flavor cpu-upgrade --timeout 20m` with `-e CHECK_ONLY=1` in place of the H200 launch settings. `SPLIT=full` is supported by the local builder; small-split jobs additionally require the original split manifests. The original adapter job bundle and first-run flags have now been recovered; see below. The HF commands are documented launch patterns and have not been executed during this restructuring.

## Historical verification

HF job `6b09af0354072e8b716a1d36` records `training`, 256-side encoding, two epochs, `--max-minutes 45`, 8,403 fit images and 941 validation images. These settings agree with the reported initial visual-adapter run. The original mounted bundle was recovered and compared: 2 of its 10 files match the current phase copy, and 8 differ. The full resolved dependency environment is still required. LoRA and decoder historical jobs were not available in the currently authenticated account during this audit.

Historical training seeds are 3407 for P11, decoder fine-tuning and the visual-adapter run; use H200 for P11 and the initial visual adapter, and A100 80 GB for full decoder fine-tuning. Git comparison at the repository reorganization shows the LoRA and full-decoder training/model files unchanged through the current revision, but the LoRA job builder changed. The actual original visual-adapter job bundle differs from the current phase source in 8 of 10 bundled files.

## Recovered original visual-adapter job

HF job `6b09af0354072e8b716a1d36` mounted the original script from bucket path `20261127T162537-bf5aed/hf_job_vlh.py` on one H200. The recovered file has SHA-256 `7d4e7c41d105d22a75fcce71d928c63c6eadf605fba99a5b8e2f7b4b9a14f802`. A local copy is retained under the repository's git-ignored `data/historical_hf/phase-1/`; the original script says not to commit it. An authorized reader can retrieve it from the private bucket, using their authorized bucket owner:

```sh
mkdir -p data/historical
hf buckets cp hf://buckets/YOUR_BUCKET_OWNER/jobs-artifacts/20261127T162537-bf5aed/hf_job_vlh.py data/historical/hf_job_vlh.py
shasum -a 256 data/historical/hf_job_vlh.py
```

The historical non-secret job settings were `PREDICT_DEV=1` and `TRAIN_ARGS=--epochs 2 --max-minutes 45`. The script embeds the original source and checks the original train CSV hash. A new run still needs authorized data and model access, an H200, a separate private output repository, and the original resolved software environment; several dependency versions and GPU-level determinism are not locked. Do not overwrite the historical output repository.
