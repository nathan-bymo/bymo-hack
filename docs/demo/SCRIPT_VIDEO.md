# Script del video della demo (≤ 2 minuti)

Testo da dire in inglese, come l'interfaccia. Le indicazioni di regia sono in italiano, tra parentesi quadre.
Circa 280 parole: a ritmo calmo stanno in 1:55. Ogni numero viene da un report del repo: le fonti sono nelle
pagine della demo.

## Prima di registrare

- Avviare la demo: `make -C src/demo demo`, poi aprire `http://127.0.0.1:8010` in Chrome, finestra 1440 × 900
  (o schermo intero con zoom 90%).
- Sulla scrivania c'è la cartella **"Bymo demo - foto pronte"**: 13 foto del test held-out (`train.csv`, mai viste dal
  modello finale; la challenge permette di mostrarne poche). Per ognuna ci sono le risposte vere dei modelli allenati
  (finale, P13, P12, P11), calcolate su GPU durante le valutazioni e registrate nella demo. Trascinando una di queste
  foto, i dettagli prodotto si compilano da soli e i modelli allenati rispondono subito.
- Con una foto diversa i modelli allenati non hanno registrazioni: rispondono solo 8B e 70B dal vivo.
- Controllare che `LLM_API_KEY` sia nel `.env` (70B e 8B dal vivo).
- Fare una prova completa prima: la prima chiamata al gateway può essere lenta.

## Scena 2: la demo (0:15–0:55), pagina Demo

[Trascinare `07 monitor side view.jpg` dalla cartella sulla scrivania nel riquadro grigio.]

> I drop a product photo. On the right I pick the models: our final model, an earlier version of it, and Apertus
> 70B as it comes, without training.

[Sono già spuntati *Final model*, *LoRA + classifiers (P11)* e *Apertus 70B, no training*: le risposte arrivano da sole, senza pulsanti.]

> Each column is a model, each row is one of the six questions. Look at the perspective: the monitor is seen from
> its right side. The final model gets it right; the earlier version says left. Telling left from right was our
> hardest problem.

[Indicare la riga della prospettiva, rossa con "Models disagree".]

> A red percentage means the model is less than 60% sure, so a person should take a look. Product details are
> optional: in production they are often missing.

## Scena 3: i risultati (0:55–1:25), pagina Results

[Cliccare *Results*. Fermarsi sul grafico della classifica.]

> On the official Kaggle dev set our best submission scores 0.847: above both Gemini references, in the middle
> assessment group, behind only the organisers' own fine-tuned Qwen.

[Scorrere a *Question by question*, indicare la riga della prospettiva.]

> Question by question, on 1,193 photos it never saw: perspective goes from 0.198 for Apertus 8B without training
> to 0.755.

[Scorrere a *Speed and cost*.]

> It runs at 4.9 images per second on a single L40S: 100,000 images in under six hours, for about ten dollars.

## Scena 4: come funziona (1:25–1:50), pagina How it works

[Cliccare *How it works*. Lo schema è in alto.]

> Under the hood: SigLIP 2, Apertus' own image tokenizer and an OCR reader turn the photo into tokens and text.
> Apertus 8B reads them with a light adapter, and six small classifiers answer the six questions. The black boxes
> are what we trained: 0.37 gigabytes on top of the public Apertus.

[Scorrere a *What mattered*.]

> The biggest lesson: left and right were a coin flip, because a photo and its mirror look the same on average.
> Features that change sign when the image is mirrored fixed that.

## Chiusura (1:50–2:00)

[Tornare a *Demo* con la foto e le risposte visibili.]

> Only Apertus and open-weights models at runtime, on one GPU: ready for an on-premises or Swiss cloud
> deployment. Thank you.

## Numeri citati e fonti

| numero | dove si vede | fonte |
|---|---|---|
| 0.847 Kaggle dev (miglior invio del team), gruppo centrale | Results, classifica | `src/training/benchmark/README.md:26`, `challenge_description.md:143-145` |
| prospettiva 0.198 → 0.755 | Results, *Question by question* | `src/plain-benchmark/apertus-8B-benchmark/prompt_eval_coverage.md:69` e `RISULTATI.md:62` |
| 4.9 img/s su L40S, 5.7 h, $10.20 per 100.000 | Results, *Speed and cost* | `RISULTATI.md:15`, prezzi HF in `models.yaml` |
| 0.37 GB di pesi allenati | How it works, legenda dello schema | `weights/apertus-8B-vadapter-lora-heads-final/` (du -sh) |
| sinistra/destra a caso (55%) | How it works, *What mattered* | `RISULTATI.md:84-87` |
