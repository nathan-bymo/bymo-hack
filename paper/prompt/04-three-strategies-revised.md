# 04 — (Optional) Corrected version of the three-strategies figure

**Output:** `paper/figures/apertus-training-phase-1.png` (overwrites the current file; landscape, 1752 × 897 px)

The current figure is correct in substance, but the audit found two inaccuracies:
1. In panels B and D, "first 24 of 32 layers" does not show that layers 25–32 are **removed**, not frozen.
2. In panel D, the dashed box draws OCR → spatial features → attention pooling as a chain. In reality, OCR text enters
   the prompt as text tokens, while spatial features and pooling act on the decoder output, before the heads.

Attach the current `paper/figures/apertus-training-phase-1.png` to ChatGPT together with this prompt.

## Prompt

I am attaching a scientific diagram. Recreate it **identically** (same layout, title, four panels A–D, colours, fonts,
icons, legend and footer strip), and apply only these changes:

1. In panel B, under the box "Apertus 8B decoder: first 24 of 32 layers (trainable)", add a small grey note in italics:
   "layers 25–32 removed".
2. In panel D, under the box "Apertus 8B decoder: first 24 of 32 layers (frozen)", add the same small grey note:
   "layers 25–32 removed".
3. In panel D, replace the dashed box "Added in later iterations" with two separate elements, both drawn with a dashed
   border and both labelled with the small heading "added later":
   - a grey box "OCR text" placed next to the "Text tokens (frozen)" box, with an arrow into the "Text tokens" box;
   - a pink box "Spatial features + 2D attention pooling (trainable)" placed between the decoder and the
     "Six classification heads (trainable)" box, on the arrow that connects them.
   Remove the arrows that chained OCR → Spatial features → Attention pooling.

Do not change anything else. Keep every other label exactly as in the attached image, with identical spelling.
