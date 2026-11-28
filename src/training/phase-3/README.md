# Phase 3 — DINOv3 and mixed training

This folder is a standalone working root containing DINOv3, mixed, and the visual-adapter support code they require, including their local shared helpers. It has no source-code dependency on another phase or the parent repository. Its code is a copy of the current implementations, not an immutable historical job bundle. See [RESULTS.md](RESULTS.md).

## Setup and split

Run from this folder. Place authorized `train.csv` and `metric.py` in `data/`; the generators bind job bundles to the local CSV checksum. Configure gated access to Apertus and DINOv3, a private HF output model repository, an authorized private data repository or Kaggle credentials, and GPU quota. Keep tokens out of files and source control.

- **DINO-only:** 10,683 images outside prompt-test; the 1,193-image prompt-test was used for checkpoint selection and is therefore not an independent final test.
- **Mixed:** 10,683 real images outside prompt-test plus 1,040 training-derived synthetic examples; the saved training metadata records 1,193 validation images. The available implementation selects prompt-test for validation, and the saved training record is consistent with that behavior. Prompt-test was excluded from fitting but was used for checkpoint selection; it is not an independent test. This contradicts the older run report, which says 1,038 internal validation images. Data and source checkpoints must be separately accessible.

## DINOv3 on Hugging Face

The launch script generates its own bundle and requires the `OUTPUT_REPO`, `SOURCE_REPO` and `DATA_REPO` environment variables, plus a securely exported `HF_TOKEN`. Replace the example values before running; it requests one H200 for up to 180 minutes.

```sh
export OUTPUT_REPO=YOUR_PRIVATE_OUTPUT_REPO
export SOURCE_REPO=YOUR_PRIVATE_SOURCE_REPO
export DATA_REPO=YOUR_PRIVATE_DATASET_REPO
bash src/training/apertus-8B-dinov3-lora-heads/launch.sh
```

The checkpoint defaults to `final-abt/model/best`; set `SOURCE_PATH` if needed. Use the GPU launcher only after verifying permissions, cost and data rights. The command was not remotely executed during this restructuring.

## Mixed bundle and HF launch

The historical mixed training was run on Modal. Its builder also creates an HF Jobs UV script. The following is a **proposed HF adaptation**, not an exact historical reproduction or a remotely verified command. It requires `final-abt/model/best` and the listed DINO revision; `MIXED_SYNTHETIC_FILE` is relative to the private dataset repository.

```sh
python src/training/apertus-8B-mixed-lora-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 150m --secrets HF_TOKEN \
  -e UPLOAD_REPO=YOUR_PRIVATE_OUTPUT_REPO -e SOURCE_REPO=YOUR_PRIVATE_SOURCE_REPO \
  -e DATA_REPO=YOUR_PRIVATE_DATASET_REPO -e INIT_FROM=final-abt/model/best \
  -e DINO_REVISION=5931719e67bbdb9737e363e781fb0c67687896bc \
  -e MIXED_SYNTHETIC_FILE=synthetic/dino_text_1040.tar -e RUN_NAME=replication-mixed \
  data/hf_jobs/hf_job_mixed.py
```

The exact historical Modal source bundle and resolved settings are still needed. For inference, the `build_infer_job.py` generator is also included; the combined TF32 + selective-TTA preset ran on dev (`dev_final`, H200, 11.68 images/s) but has no labelled score of its own.

## Historical verification

The saved mixed checkpoint `training.json` records `real_fit=10683`, `synthetic_fit=1040` and `validation=1193`. Both the current code and the initial mixed implementation in Git select prompt-test for validation. These facts contradict the older report claiming 1,038 internal validation examples. The available code and Git history cannot prove identity with the original Modal bundle; it and its resolved environment and logs are still required. The HF command above is a proposed adaptation, not the historical Modal launch.

The mixed model uses global seed 3407 and recorded shuffle seeds 3407, 3408 and 3409 for successive stages. Historical training used H200. The reported TF32 inference measurement used H100, not H200; choose the GPU explicitly when reproducing inference measurements. A retained DINO job bundle differs from the current phase in six visual-adapter support files. A local mixed bundle matches the current source but was generated later, so it does not prove historical identity.
