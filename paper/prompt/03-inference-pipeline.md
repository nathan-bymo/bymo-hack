# 03 — Delivery inference pipeline

**Output:** `paper/figures/inference-pipeline.png` (landscape, about 1900 × 750 px)

## Prompt

Create a clean, flat, vector-style pipeline diagram on a pure white background, for a machine-learning paper.
Title at the top in bold sans-serif: "Delivery inference: from images to the submission CSV".

Draw a horizontal pipeline from left to right in **three swim lanes** (thin horizontal bands with small grey labels
on the left edge): "CPU threads", "GPU", "Output".

**CPU threads lane:**
- Box "Image decoding (thread pool, one batch ahead)".
- Box "Sort by VQ grid size".
- A long box running in parallel underneath both, labelled "OCR thread (docTR), one batch ahead". Put a small orange
  tag on it: "critical path".

**GPU lane** (each box light blue #CFE2F7, frozen weights):
- Box "VQ tokenizer (TF32)".
- Box "SigLIP 2 (TF32)" and directly below it box "DINOv3", with a small "+" circle joining them into box "Fusion +
  projector".
- Box "Apertus decoder, 24 layers, LoRA merged" (slightly larger).
- Box "Six classification heads".
Arrows connect: sorted batch → VQ tokenizer and → SigLIP 2 / DINOv3; VQ tokenizer, fusion + projector and the OCR
thread → Apertus decoder → six heads.

**Decision diamond** after the heads: "Top-2 perspective classes are a left/right pair with margin < 0.15?"
- "no" arrow (the main path, most images) → Output lane.
- "yes" arrow → a small loop box "Mirror the image, re-score, swap left ↔ right, average perspective only", which then
  joins the Output lane. Label this loop "≈ 4% of images".

**Output lane:** box "Predictions CSV (six dimensions)".

Add a compact legend at the bottom: light blue = "frozen model weights", grey = "CPU work", orange tag = "slowest
stage". No throughput numbers, no other numbers than "24", "0.15" and "4%", no logos, no 3D, no shadows. Thin black
arrows, rounded rectangles with thin dark-grey borders, black sans-serif text. All text must be spelled exactly as
given.
