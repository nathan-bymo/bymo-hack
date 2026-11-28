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
