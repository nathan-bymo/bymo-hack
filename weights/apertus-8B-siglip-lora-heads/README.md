# Apertus 8B + trainable SigLIP LoRA (no DINOv3) + LoRA + heads + OCR

Best checkpoint of run `siglip-lora-1128-0815` (epoch 2): public Kaggle dev **0.83526** (submission 59043423) and
prompt-test macro F1 0.836 on 1,193 images. The checkpoint was selected on prompt-test, so that number is not an
independent test. The mixed model in `../apertus-8B-mixed-lora-heads-final/` scores higher on dev (0.84215).
Speed: 6.75 (prompt-test) and 7.43 (dev) images/s from image to CSV on one H200, peak 79.8 GB.

Files: `lora/` (Apertus LoRA rank 32 on 24 layers), `vision_lora/` (SigLIP 2 LoRA rank 16), `trainable.pt`
(projector, heads, spatial features, attention pooling), `settings.json`, `training.json` (history) and
`prompt_test_metrics.json`. Apertus, SigLIP 2 and the OCR models are loaded separately.

Base: `swiss-ai/Apertus-v1.5-8B`, revision `1519de0ae7f3d114879291810a58259966f64c99`. SigLIP:
`google/siglip2-base-patch16-224`. Started from `../apertus-8B-vadapter-lora-heads-final/`. Recipe and results:
[phase-2 results](../../src/training/phase-2/RESULTS.md#trainable-siglip-lora-on-final-abt-run-siglip-lora-1128-0815).

```sh
python src/training/apertus-8B-vadapter-lora-heads/predict.py \
  --checkpoint weights/apertus-8B-siglip-lora-heads --csv data/dev.csv --out data/runs/siglip_lora/dev.csv
```
