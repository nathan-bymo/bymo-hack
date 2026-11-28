# Prompt per le figure del paper

Ogni file `NN-*.md` contiene un prompt da incollare in ChatGPT (generazione di immagini). Salva l'immagine
con il nome indicato in **Output** dentro `paper/figures/` e ricompila: `paper.tex` la include in automatico.
Finché il file manca, nel PDF compare un riquadro "Figure pending".

| Prompt | File da salvare | Dove va nel paper | Obbligatorio |
|---|---|---|---|
| [01-phase2-lineage.md](01-phase2-lineage.md) | `paper/figures/phase2-lineage.png` | Sezione 8, Figura 4 | sì |
| [02-mixed-architecture.md](02-mixed-architecture.md) | `paper/figures/mixed-architecture.png` | Sezione 9, Figura 5 | sì |
| [03-inference-pipeline.md](03-inference-pipeline.md) | `paper/figures/inference-pipeline.png` | Sezione 10, Figura 6 | sì |
| [04-three-strategies-revised.md](04-three-strategies-revised.md) | `paper/figures/three-strategies-revised.png` (se manca, il paper usa `apertus-training-phase-1.png`) | Sezione 6, Figura 3 | no |
| [05-fix-mixed-architecture.md](05-fix-mixed-architecture.md) | corregge `mixed-architecture.png` (collegamenti sbagliati) | Sezione 8 e report | **sì** |
| [06-fix-inference-pipeline.md](06-fix-inference-pipeline.md) | corregge `inference-pipeline.png` (refuso, OCR su GPU) | Sezione 9 | **sì** |
| [07-fix-three-strategies.md](07-fix-three-strategies.md) | corregge `three-strategies-revised.png` (origine dell'OCR) | Sezione 6 | consigliata |

## Regole di stile comuni (già incluse in ogni prompt)

Stesso stile di `paper/figures/apertus-training-phase-1.png`:
- sfondo bianco, diagramma piatto e vettoriale, nessuna prospettiva 3D, nessuna ombra marcata;
- riquadri arrotondati con bordo grigio scuro sottile, testo nero sans-serif (stile Inter o Helvetica);
- **azzurro chiaro (#CFE2F7)** per i componenti congelati, con "(frozen)" in blu;
- **rosa chiaro (#F6CFD3)** per quelli allenabili, con "(trainable)" in rosso scuro;
- **grigio chiarissimo (#F2F2F2)** per dati e token;
- frecce nere sottili; pannelli con barra del titolo grigia; legenda in basso;
- formato orizzontale, circa 1800 × 900 px, testo leggibile anche stampato a larghezza di pagina.

## Dopo la generazione

ChatGPT a volte sbaglia o inventa il testo dentro le immagini. Prima di salvare, controlla che **ogni etichetta e
ogni numero** corrisponda esattamente al prompt: niente numeri in più, nessuna parola storpiata. Se qualcosa non
torna, chiedi una correzione puntuale ("cambia solo l'etichetta X in Y"). I numeri dei risultati sono volutamente
nelle tabelle e nel grafico TikZ del paper, non nelle immagini generate.
