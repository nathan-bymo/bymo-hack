# 01 — Lineage of the phase-two checkpoints

**Output:** `paper/figures/phase2-lineage.png` (landscape, about 1800 × 800 px)

## Prompt

Create a clean, flat, vector-style scientific diagram on a pure white background, for a machine-learning paper.
Title at the top in bold sans-serif: "Phase two: how the checkpoints evolved".

The diagram is a left-to-right directed graph with **two horizontal lanes**, each inside a rounded panel with a light
grey title bar.

**Top lane**, panel title "LoRA + heads (Apertus 8B, 32 layers, VQ tokens only)". Two rounded boxes connected by an arrow:
1. Box "P11" with a smaller second line "256 px · 3 epochs".
2. Box "P12" with a smaller second line "512 px · spatial features · attention pooling · 2 epochs".

**Bottom lane**, panel title "Visual adapter (Apertus 8B, first 24 layers, SigLIP 2 + VQ tokens)". A chain of three
boxes connected by arrows:
1. Box "First checkpoint" with second line "8,403 images · 2 epochs".
2. Box "+ OCR" with second line "docTR text in the prompt · +1 epoch".
3. Box "final-abt" with second line "+ spatial features · + attention pooling · 9,645 images · +3 epochs". Make this
   box visually emphasised: a thicker dark border and slightly larger.

From "final-abt", four arrows fan out to the right to four smaller boxes stacked vertically:
- "P13" with second line "+ 939 synthetic text images · +1 epoch"
- "P14" with second line "SigLIP 2 so400m · 512 px · 196 tokens"
- "P15" with second line "all 11,876 images · +1 epoch"
- "SigLIP LoRA" with second line "trainable encoder (LoRA rank 16) · 2 epochs"

Also from "final-abt", one thick arrow goes further right to a box with a dashed border and light pink fill labelled
"Phase three: DINOv3 + mixed model".

Add one **dotted curved arrow** from the "final-abt" box up to the "P12" box in the top lane, with the small italic
label "spatial features + pooling ported".

Colour the arrow lines by the kind of change, and show a small legend in the bottom-right corner:
- dark blue arrow = "architecture change",
- orange arrow = "more or different data",
- grey arrow = "more training epochs".
Use dark blue for: First checkpoint → + OCR, + OCR → final-abt, final-abt → P14, final-abt → SigLIP LoRA, P11 → P12.
Use orange for: final-abt → P13, final-abt → P15.
Arrows that combine several changes may be drawn as two thin parallel lines in the two colours
(+ OCR → final-abt: blue and orange; P11 → P12: blue and grey).

Style: rounded rectangles with thin dark-grey borders, black sans-serif text (Inter/Helvetica style), boxes filled
with very light grey (#F2F2F2); "final-abt" filled with light blue (#CFE2F7). No numbers other than the ones written
above, no scores, no logos, no icons, no 3D, no shadows. All text must be spelled exactly as given.
