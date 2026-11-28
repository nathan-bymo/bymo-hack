# Apertus 8B mixed LoRA heads

This model runs SigLIP 2, DINOv3 and the native Apertus VQ tokenizer, with OCR and a single Apertus
decoder. DINOv3 patch features are aligned to the 7×7 SigLIP grid. A trainable residual maps those
features into the existing SigLIP pathway, with a zero-initialized output. The initial forward pass
therefore reproduces the SigLIP + VQ checkpoint `final-abt/model/best`. DINOv3 uses the weight
revision recorded in the successful DINO-only run.

The native VQ sequence and the 49 projected visual tokens remain in the Apertus input. OCR is read
once per image from the original, including for scale variants. The frozen visual encoders are
precomputed for training and run live for the reported inference speed.

## Training and selection

- Keep prompt-test entirely out of training, split by product. Compare locally to SigLIP + VQ
  (0.82140) and the two-model probability ensemble (0.84970).
- Train the DINO residual alone for half an epoch. Then train fusion, projector, pooling, heads
  and LoRA for up to four epochs.
- If joint training has two validations without at least 0.002 improvement in macro F1, reload the
  best checkpoint and try up to two epochs with the final two Apertus blocks unfrozen. Stage
  transitions based on sustained regression stop the run and retain the best checkpoint.
- Use the previously validated training-only overlays if `--synthetic-file` is supplied. They
  contribute at 0.25 weight to the two text dimensions only. Neither overlays nor their products
  enter prompt-test.
- Keep FP32 master weights for unfrozen blocks and record actual weight updates. Before exporting,
  reload the winner and reproduce its probabilities. Model artifacts stay on the private HF repo.
  Export includes prompt-test metrics and dev predictions, with measured end-to-end speed.

## Launch on Modal

Build the local source bundle first. The wrapper is configured for one H200 with a 150-minute
timeout, no retries and an internal deadline that leaves time for inference and exports.

```sh
python3 src/training/apertus-8B-mixed-lora-heads/build_hf_job.py
modal run --detach src/training/apertus-8B-mixed-lora-heads/modal_app.py \
  --output-repo nathan-monzani-bymo/galaxus-vlh-private \
  --data-repo nathan-monzani-bymo/galaxus-data-private \
  --source-path final-abt/model/best \
  --synthetic-file synthetic/dino_text_1040.tar \
  --local-hf-auth
```

Use `--secret` to use a named Modal secret instead of local HF authentication. The GPU preflight
compares mixed and original logits exactly and verifies a finite nonzero gradient through the DINO
residual before computing the full training cache.

## Continue with a LoRA on SigLIP

Starts from a mixed checkpoint and also trains a LoRA (rank 16) on SigLIP; DINOv3 stays frozen. The first
validation is the starting checkpoint and must reproduce its prompt-test score.

```sh
python3 src/training/apertus-8B-mixed-lora-heads/build_hf_job.py
MIXED_MINUTES=120 modal run --detach src/training/apertus-8B-mixed-lora-heads/modal_app.py \
  --output-repo <private model repo> --data-repo <private data repo> \
  --source-path mixed-modal-20261128-033303/model/best \
  --dino-revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --vision-lora 16 --vision-epochs 2 --wall-minutes 85 --local-hf-auth
```

## Final images

`--final` decrypts the final images on the job's disk (the passphrase is asked at launch and never written to a
file) and predicts them with the mixed checkpoint at `--source-path` of the output repo. An unreadable image is
read as a blank one, so every row gets a prediction; only `predictions/final*` is uploaded. Nothing is submitted.

```sh
python3 src/training/apertus-8B-mixed-lora-heads/build_hf_job.py
MIXED_MINUTES=60 modal run --detach src/training/apertus-8B-mixed-lora-heads/modal_app.py \
  --output-repo <private model repo> --data-repo <private data repo> \
  --source-path <checkpoint folder>/model/best \
  --dino-revision 5931719e67bbdb9737e363e781fb0c67687896bc --final --local-hf-auth
```

Rehearsal before 13:00: make a test archive with `src/scripts/make_test_archive.py`, upload it to the private
output repo, add `--final-archive <path in the repo>` and use the password the script prints.

Then join dev and final predictions of the same model with `src/scripts/final_submission.py`.

## Distillation pilot

Use a saved mixed checkpoint and the private [teacher artifact](RESULTS.md#4-distillazione-sulle-immagini-dev-f1-target-pronti-training-non-lanciato). Build the bundle
as above, then pass `--source-path`, `--teacher-file`, `--teacher-sha256` and `--source-revision`
to the Modal entry point. The source revision must contain both the checkpoint and the teacher.
Pass the same synthetic archive as the source run to retain its supervised augmentation in both
branches. The pilot has one H200, a 90-minute cap and no automatic retries.

The distillation branch runs half an epoch. The supervised control starts from the same checkpoint
and uses the same number of sampled views and optimizer updates, replacing dev views with supervised
views. Human targets retain the focal loss and teacher targets contribute masked soft cross-entropy
at weight 0.5. The job checks the teacher against the dev CSV before loading the model.

The private export reports both branches on prompt-test, a paired bootstrap by product, and measured
image-to-CSV throughput before and after the pilot. `model/training.json` records the promotion gate.
The pilot reads dev images for teacher supervision and exports no dev score.
