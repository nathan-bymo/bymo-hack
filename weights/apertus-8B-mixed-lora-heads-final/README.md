# Apertus 8B mixed (SigLIP 2 + DINOv3 + VQ) + LoRA + heads + OCR

Best checkpoint of run `mixed-modal-20261128-033303/model/best` (joint stage, epoch 3): prompt-test macro F1
**0.843** (weighted 0.924) on 1,193 images, against 0.821 for the SigLIP + VQ final model in
`weights/apertus-8B-vadapter-lora-heads-final/`. Prompt-test was kept out of training but chose the epoch, so it is not
an independent test. Public Kaggle dev: **0.84215** (submission 59038887).

Per dimension (prompt-test macro F1): subject 0.868, background 0.875, perspective 0.776, individuals 0.970,
text_classification 0.769, text_language 0.799. Trained on 10,683 real images outside prompt-test plus 1,040
synthetic text overlays (weight 0.25, text dimensions only). Speed: 5.9 images/s end to end on one H200 with the
original export, 11.7 images/s with `fast_infer.py` (merged LoRA, TF32, selective perspective TTA); peak training
memory 61 GiB.

Files: `lora/` (LoRA rank 32 on 24 layers), `trainable.pt` (projector, heads, spatial features, attention pooling),
`mixed_fusion.pt` (DINOv3 residual into the SigLIP pathway), `settings.json`, plus `training.json` (history) and
`prompt_test_metrics.json` (per-class metrics). OCR and the frozen vision models are loaded separately.

Base: `swiss-ai/Apertus-v1.5-8B`, revision `1519de0ae7f3d114879291810a58259966f64c99`. SigLIP:
`google/siglip2-base-patch16-224`. DINOv3: `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. Code: `src/training/apertus-8B-mixed-lora-heads/`. Private Hugging
Face copy: `nathan-monzani-bymo/galaxus-vlh-private`, `mixed-modal-20261128-033303/model/best`.
