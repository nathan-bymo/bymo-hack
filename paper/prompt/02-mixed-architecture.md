# 02 — Mixed model architecture (SigLIP 2 + DINOv3 + VQ + OCR, one decoder)

**Output:** `paper/figures/mixed-architecture.png` (landscape, about 1900 × 950 px)

## Prompt

Create a clean, flat, vector-style architecture diagram on a pure white background, for a machine-learning paper.
Title at the top in bold sans-serif: "Mixed model: three visual streams, one Apertus decoder".

Colour code (show it in a legend strip at the bottom):
- light blue fill (#CFE2F7) with "(frozen)" written in blue = frozen weights,
- light pink fill (#F6CFD3) with "(trainable)" written in dark red = trainable weights,
- very light grey fill (#F2F2F2) = data or tokens,
- a box with a **dashed dark border** = "new in phase three".

Layout from left to right:

**Inputs (far left):** a small flat icon of a product photo (a simple black tube on a white square) labelled
"Product image", and below it a grey box "Product context (title, brand, category)".

**Four branches from the product image**, stacked vertically:
1. Top branch: blue box "VQ visual tokenizer (frozen)" → grey box "≈256 discrete image tokens".
2. Second branch: blue box "SigLIP 2 base, 224 px (frozen)" → small grey box "7 × 7 × 768".
3. Third branch: blue box "DINOv3 ViT-B/16, 288 px (frozen)" → small grey box "18 × 18 → pooled 7 × 7 × 768" →
   pink box with dashed border "Fusion: LayerNorm + Linear 768 × 768, zero-initialised (trainable)".
4. Bottom branch: blue box "docTR OCR (frozen)" → grey box "OCR text, ≤300 characters".

The outputs of branch 2 and of the Fusion box meet at a circle with a "+" sign (element-wise addition), annotated in
small italic text "residual: at step 0 the model equals final-abt". From the "+" circle an arrow goes to a pink box
"Projector (trainable)" → grey box "49 soft tokens".

**Token sequence (centre):** a long horizontal grey strip divided into six labelled segments, in this order:
"chat template" | "49 soft tokens" | "VQ image tokens" | "product context" | "OCR text" | "task".
Arrows lead into the matching segments from: the 49 soft tokens box, the discrete image tokens box, the product
context box and the OCR text box.

**Decoder (centre right):** a large blue box "Apertus 1.5 8B decoder: first 24 of 32 layers (frozen)" with a small
pink tab attached on top: "LoRA rank 32 (trainable)". A thin note under the decoder: "layers 25–32 removed".

**Readout (right):** from the decoder, three arrows go into three small pink boxes stacked vertically:
- "Last token + mean of image tokens",
- "Spatial features: left − right, top − bottom",
- "2D attention pooling: 8 queries".
These three merge into a pink box "Shared hidden layer, 512 (trainable)", which fans out into six small pink boxes
labelled exactly: "Subject", "Background", "Perspective", "Individuals", "Text classification", "Text language".
Group the six boxes under the caption "Six classification heads (trainable)".

Style: rounded rectangles with thin dark-grey borders, thin black arrows, black sans-serif text (Inter/Helvetica
style), generous white space, no 3D, no shadows, no logos, no other numbers than the ones written above. All text
must be spelled exactly as given.
