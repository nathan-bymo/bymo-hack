# Phase 2 — generative LoRA

This standalone working root contains the generator, training and evaluation code, local support code and a reconstructed notebook. Run from this directory. [Results](RESULTS.md).

## Historical setup and inputs

P10 used H200, 256² VQ, LoRA rank and alpha 32, three epochs, batch 8 with accumulation 2, and training seed 3407. The product-separated prototype split contains 1,500 training images and 450 balanced validation images; split construction used seed 42. The 1,500 examples produce 3,952 augmented items per epoch. Coverage is reserved from training. The original source `train.csv` SHA-256 is `936a330d207f4ded5cf8e3834682ce8f6f219f0372ab8dafaceed2245b8c15ab`. The currently available prototype manifests have SHA-256 values `1320177fe6bf2485eac544314c08121b5e0c6e553dda38753a06602a194bb51d` (training) and `d3d544d6c65a5dc910d05066605b3307c5736e9d1a1d6212cbca83a8961e551c` (validation); the historical manifest checksums were not independently recorded, so historical byte identity remains unverified. Keep image IDs private.

## Hugging Face job

Place authorized original manifests in `data/prototype_splits/` and configure access to the competition data, gated model, a private output model repository, H200 quota and `HF_TOKEN` / `KAGGLE_API_TOKEN`. Generate the job, then pack the manifests as a secret:

```sh
python src/micro-training/lora-generativo/build_hf_job.py
export PROTOTYPE_IDS="$(python src/micro-training/lora-generativo/build_hf_job.py --pack-prototype data/prototype_splits)"
hf jobs uv run --flavor h200 --timeout 3h \
  --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN --secrets PROTOTYPE_IDS \
  -e PART=prototype_train -e SIDE=256 -e TRAIN_HOURS=2 -e BATCH=8 -e ACCUMULATION=2 \
  -e UPLOAD_REPO=YOUR_PRIVATE_MODEL_REPO src/micro-training/lora-generativo/hf_job.py
```

Do not print or commit `PROTOTYPE_IDS` or the generated job. Validate original data, effective flags, checkpoint and source bundle before claiming a historical rerun. This remote launch was not executed during the audit.
