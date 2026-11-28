# Apertus 8B + visual adapter + LoRA + heads + OCR

Final checkpoint `final-abt/model/best`: public Kaggle dev score 0.83177 and held-out prompt-test macro F1 0.821. The final run trained on 9,645 labeled images excluding prompt-test, after two earlier training stages. It includes the LoRA adapter, visual projector, classification heads, spatial features and attention pooling; OCR and the frozen vision models must be loaded separately.

Base: `swiss-ai/Apertus-v1.5-8B`, revision `1519de0ae7f3d114879291810a58259966f64c99`. SigLIP: `google/siglip2-base-patch16-224`. Code: `src/training/apertus-8B-vadapter-lora-heads/`; see the [phase-2 results](../../src/training/phase-2/RESULTS.md) for training history and evaluation limitations. Private Hugging Face checkpoint: `nathan-monzani-bymo/galaxus-vlh-private`, `final-abt/model/best`.
