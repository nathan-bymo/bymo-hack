# 06 — Correzione di `inference-pipeline.png` (necessaria)

**Output:** sovrascrivi `paper/figures/inference-pipeline.png`. Allega a ChatGPT l'immagine attuale.

Errori: refuso "Preaictions"; l'OCR è disegnato come lavoro CPU, ma docTR gira sulla GPU in un thread separato
(l'errore era nel mio prompt originale).

## Prompt

I am attaching a pipeline diagram. Recreate it **identically** (same title, lanes, boxes, colours, arrows, fonts and
positions) and apply only these changes:

1. Fix the spelling of the output box: it must read exactly "Predictions CSV (six dimensions)".
2. Rename the top lane label from "CPU threads" to "Parallel threads".
3. Change the text of the long box "OCR thread (docTR), one batch ahead" to "OCR thread (docTR on the GPU), one batch
   ahead", keep its orange "critical path" tag, and give this box the light blue fill used for frozen model weights.
4. Keep the boxes "Image decoding (thread pool, one batch ahead)" and "Sort by VQ grid size" light grey.

Do not change anything else. Keep every other label exactly as in the attached image, with identical spelling.
