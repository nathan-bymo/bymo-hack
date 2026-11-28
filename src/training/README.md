# Model training and Hugging Face Jobs

For setup, verified checks and outstanding provenance, start with the [reproducibility guide](REPRODUCIBILITY.md).

## Browse by experimental phase

- [Phase 1](phase-1/README.md): the three initial training directions.
- [Phase 2](phase-2/README.md): subsequent LoRA and visual-adapter improvements.
- [Phase 3](phase-3/README.md): DINOv3 and mixed experiments.

Each phase folder contains its own copy of the code needed for that phase and runs as a standalone working root. These copies reflect the current implementation, not historical snapshots. Historical runs require their original revisions and artifacts.


The model directories under each phase's `src/training/` contain the actual training, encoding, prediction and Hugging Face job-generation code. Run all commands from inside the relevant phase folder (for example `cd src/training/phase-1`); `data/` is relative to that folder. The job scripts package code and are generated under `data/hf_jobs/`, which is ignored by Git. They are not notebooks.

| Model | Training code (in `phase-N/src/training/`) | Results | Generated job |
|---|---|---|---|
| LoRA + heads | `apertus-8B-lora-heads/` | [phase 1](phase-1/RESULTS.md) (P11), [phase 2](phase-2/RESULTS.md) (P12) | `hf_job_lora_heads.py` |
| Fine-tuned decoder + heads | `apertus-8B-fullft-heads/` | [phase 1](phase-1/RESULTS.md) | `hf_job_fullft.py` |
| Visual adapter + LoRA + heads + OCR | `apertus-8B-vadapter-lora-heads/` | [phase 1](phase-1/RESULTS.md), [phase 2](phase-2/RESULTS.md) | `hf_job_vlh.py` |
| DINOv3 and mixed (delivered) | `apertus-8B-dinov3-lora-heads/`, `apertus-8B-mixed-lora-heads/` | [phase 3](phase-3/RESULTS.md) | `hf_job_dinov3.py`, `hf_job_mixed.py` |

Each job uploads checkpoints and logs to the private `UPLOAD_REPO`; training and data acquisition require appropriately configured HF and Kaggle access. Use private repositories for competition data and artifacts. Set credentials as local environment variables or Hugging Face Jobs secrets; do not put them in source files. The commands below create scripts or show launch patterns; they do not submit competition predictions.

## Generate the jobs locally

```sh
cd src/training/phase-1   # from the repository root
python src/training/apertus-8B-lora-heads/build_hf_job.py
python src/training/apertus-8B-fullft-heads/build_hf_job.py
python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py
```

A local copy of the competition `data/train.csv` is needed to bind the generated jobs to the data version. The LoRA job additionally embeds IDs if both private `data/nathan_split/` manifests are available; otherwise it can be generated for `SPLIT=full` only. Do not commit the generated scripts: the small-split job can contain competition IDs.

## LoRA + heads

The historical P11 training used `MODE=train`, `SPLIT=full` (9,344 training images, not all labeled images), 256² and H200. Example launch after configuring secrets and repository access:

```sh
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e MODE=train -e SPLIT=full -e RUN_NAME=<unique-run-name> \
  -e UPLOAD_REPO=<user>/<private-model-repo> -e 'TRAIN_ARGS=--max-minutes 80' \
  data/hf_jobs/hf_job_lora_heads.py
```

The job supports `MODE=check`, `smoke`, `train` and `predict`. For a small-split job the generator needs both `train_1500.csv` and `validation_450.csv` under `data/nathan_split/`. Specify a unique `RUN_NAME` and an explicit `MODEL_PATH` when downloading a checkpoint: the default prediction path may not match newly named runs.

## Fine-tuned decoder + heads

The observed P12 run trained phase 2 directly on all labeled images. The earlier checkpoint-selection phase can be run separately using `PHASE=1`; its result is not part of P12. Example for the documented phase-2 configuration:

```sh
hf jobs uv run --flavor a100-large --timeout 420m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e PHASE=2 -e EPOCHS=3 -e JOB_TIMEOUT_MINUTES=420 \
  -e UPLOAD_REPO=<user>/<private-model-repo> data/hf_jobs/hf_job_fullft.py
```

The full fine-tuning implementation imports shared data, encoder and head code from the visual-adapter directory. It also supports `CHECK_ONLY=1`. Check the generator's header and training options before running; do not interpret training-set or coverage scores as held-out evaluation after `PHASE=2`.

## Visual adapter + LoRA + heads + OCR

The job supports a private dataset through `DATA_REPO` or Kaggle, checkpoint continuation with `INIT_FROM`, separate output paths with `RUN_NAME`, `OCR=1`, and `PART=training|all_but_test|all`. The historical final model is a **sequence** of runs, not a single fresh run. See [phase 1](phase-1/RESULTS.md) and [phase 2](phase-2/RESULTS.md) results for the actual sequence, data and limits. The following is a continuation pattern, not a claim that it exactly reproduces all historical settings:

```sh
hf jobs uv run --flavor h200 --timeout 3h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
  -e UPLOAD_REPO=<user>/<private-model-repo> -e RUN_NAME=<unique-run-name> \
  -e PART=all_but_test -e OCR=1 -e INIT_FROM=<previous-checkpoint-path> \
  -e 'TRAIN_ARGS=<verified-training-arguments>' data/hf_jobs/hf_job_vlh.py
```

Use `DATA_REPO` when a suitable private HF dataset is already available. `build_hf_job.py --upload-data` uploads competition data and must only be used when the destination and data permissions are appropriate. The generated job supports `CHECK_ONLY=1` for a CPU preflight, `PREDICT_FROM` and `DEV_ONLY=1` for inference, and uploads the best checkpoints to the configured private repository. Verify the exact historical flags against the private logs before claiming an exact reproduction.

GPU type, available quota and cost must be checked for a new run; historical runtimes and prices in the phase `RESULTS.md` files are not current quotes.
