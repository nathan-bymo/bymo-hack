# Phase 3 — visual-adapter prototype

This standalone working root contains the SigLIP 2 projector prototype, local helpers and a reconstructed notebook. [Results](RESULTS.md). Run commands from this directory.

## Historical setup and data

Both successful runs used H200, frozen Apertus 8B and SigLIP 2 base patch16 224, 49 continuous visual tokens, native VQ at approximately 256² and seed 42. Only the projector was trained. The training split had 1,500 images and the balanced validation set 450, disjoint by product and outside coverage. Epoch 1 used batch 1; epoch 2 resumed epoch-1 projector weights with batch 4 **and a newly initialized optimizer**. A continuous two-epoch run is not equivalent. The source `train.csv` and current prototype manifest checksums are documented in [phase 2](../phase-2/README.md); the epoch-1 manifests recovered from the original private HF job bucket are byte-identical to the currently available manifests, verified by SHA-256.

## Prepare and run on Hugging Face

Supply the original authorized `data/train.csv`, `data/metric.py`, images and two private manifests. The staging command copies images and manifests into a private staging directory; it must not be committed or made public:

```sh
python src/micro-training/apertus-vadapter-benchmark/prepare_hf_job.py \
  --data-dir data --output-dir /tmp/adapter-staging --train-size 1500 --holdout-size 450 --seed 42 \
  --train-manifest data/prototype_splits/train_1500.csv \
  --validation-manifest data/prototype_splits/validation_450.csv
```

Copy the second-stage launcher into the staging directory, then mount it as a private writable HF Job volume. The CLI first syncs local volumes to a private bucket. Confirm how to retrieve the output checkpoint from that bucket before starting; the example has not been executed remotely. Each run requests the historically used H200 image and needs `HF_TOKEN` with model and bucket access.

```sh
cp src/micro-training/apertus-vadapter-benchmark/run_h200_batch4.sh \
  /tmp/adapter-staging/src/apertus-vadapter-benchmark/run_h200_batch4.sh
hf jobs run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  -v /tmp/adapter-staging:/workspace:rw \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash /workspace/src/apertus-vadapter-benchmark/run_h200.sh
```

After retrieving the epoch-1 `projector.pt` into `/tmp/adapter-staging/outputs/run-256/projector.pt`, the second stage must run separately with its new optimizer and batch 4:

```sh
hf jobs run --flavor h200 --timeout 2h --secrets HF_TOKEN \
  -v /tmp/adapter-staging:/workspace:rw \
  -e WORKSPACE_ROOT=/workspace -e DATA_ROOT=/workspace/data \
  -e INITIAL_PROJECTOR=/workspace/outputs/run-256/projector.pt \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash /workspace/src/apertus-vadapter-benchmark/run_h200_batch4.sh
```

The mounted data and generated outputs are private. The commands express the recorded two-stage configuration but the original HF mounts and training source files have been recovered, but the commands above use the current portable code and are not the exact historical launch. Do not substitute Kaggle T4 for H200 when claiming replication. Remote execution was not tested here.

## Recovered historical evidence

The original H200 job volumes contain the source files, projectors and private manifests for both stages. The epoch-1 and epoch-2 training scripts differ from each other and from the current portable implementation: epoch 1 differs by 53 inserted and 20 deleted lines; epoch 2 differs by one line. Original source files are retained locally under the git-ignored `data/historical_hf/micro-adapter/`. Original job IDs are `6b098e2354072e8b7169f354` and `6b09959e54072e8b7169ffc8`. Full source staging, image bytes and resolved dependencies must still be verified before claiming complete replication.

HF job metadata confirms that epoch 1 mounted bucket path `bymo-h200-prototype-45079690` at `/workspace` read-write. Epoch 2 mounted the same path read-only at `/dataset`, and `bymo-batch4-code-c57271dc` read-write at `/runner`. Both used H200 and `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`. These mount paths describe historical storage, not account-independent public URLs.
