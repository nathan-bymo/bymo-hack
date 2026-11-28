# Phase 1 — frozen Apertus and heads

This standalone working root contains the frozen-feature pipeline, its local support code and a reconstructed notebook. Run commands from this directory. [Results](RESULTS.md).

## Historical setup

P8 used one H200, Apertus 8B at 512², feature extraction batch 16, classification-head batch 256 and seed 3407. The `train_1500` manifest was divided into 1,199 examples for head training and 301 for internal layer/epoch selection; a separate balanced `validation_450` was used for evaluation. Preserve product separation. The original manifests for this particular run are not available at the historical path in this checkout. Do not substitute another split without verifying row identities and hashes.

## Hugging Face job

Place the original `train_1500.csv` and `validation_450.csv` in `data/heads_split/` and the authorized original `train.csv` in `data/`. `heads_split` is a portable name for the historical directory; changing the directory name does not change its data. The original manifests contain competition IDs and must remain private. Configure gated model access, private output repo, GPU quota and `HF_TOKEN` / `KAGGLE_API_TOKEN` as environment secrets.

```sh
python src/micro-training/apertus-8B-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 1h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e SPLIT=small -e BATCH=16 -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO \
  data/hf_jobs/hf_job_heads.py
```

The generated script contains split IDs and must stay private. Its seed 3407 and job configuration should be checked against the original bundle before claiming exact replication. H200 GPU execution was not rerun during this audit. The current `data/prototype_splits` manifests used by other experiments are not proven identical to the missing heads manifests.
