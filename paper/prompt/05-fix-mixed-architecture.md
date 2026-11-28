# 05 — Correzione di `mixed-architecture.png` (necessaria)

**Output:** sovrascrivi `paper/figures/mixed-architecture.png`. Allega a ChatGPT l'immagine attuale.

Errori nell'immagine generata: i token VQ entrano nel nodo "+" (devono andare nella sequenza); la Fusion ha una
freccia verso "chat template"; "49 soft tokens" entra nel segmento "task"; "OCR text" non è collegato; il
"Product context" entra anche nel segmento "VQ image tokens".

## Prompt

I am attaching an architecture diagram. Recreate it **identically** (same title, boxes, colours, fonts, icons,
positions and legend) and fix only these connections:

1. Remove the arrow from "≈256 discrete image tokens" to the "+" circle. Instead, draw an arrow from
   "≈256 discrete image tokens" down to the segment "VQ image tokens" of the token-sequence strip.
2. The "+" circle must receive exactly two inputs: the arrow from "7 × 7 × 768" (SigLIP branch) and the arrow from
   the "Fusion: LayerNorm + Linear 768 × 768, zero-initialised (trainable)" box.
3. Remove the arrow from the Fusion box down to the "chat template" segment. The Fusion box has only one outgoing
   arrow, into the "+" circle.
4. The arrow from "49 soft tokens" must end in the segment "49 soft tokens" of the strip, not in "task".
5. Add an arrow from "OCR text, ≤300 characters" to the segment "OCR text" of the strip.
6. "Product context (title, brand, category)" has exactly one arrow, into the segment "product context". Remove its
   arrow into "VQ image tokens".
7. Keep the italic note "residual: at step 0 the model equals final-abt" pointing at the "+" circle.

Do not change anything else. Keep every label exactly as in the attached image, with identical spelling.
