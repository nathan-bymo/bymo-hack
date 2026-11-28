# Script del video della demo (≤ 2 minuti)

Testo da dire in inglese, come l'interfaccia. Le indicazioni di regia sono in italiano, tra parentesi quadre.
Circa 280 parole: a ritmo calmo stanno in 1:55. Ogni numero viene da un report del repo: le fonti sono nelle
pagine della demo.

## Prima di registrare

- Avviare la demo: `make -C src/demo demo`, poi aprire `http://127.0.0.1:8010` in Chrome, finestra 1440 × 900
  (o schermo intero con zoom 90%). Così la tabella delle risposte sta tutta nello schermo.
- Preparare **una** foto prodotto propria con testo stampato (per esempio una confezione con scritte in tedesco).
  Mai immagini Kaggle dev o final. Per la challenge si possono mostrare poche immagini di `train.csv`.
- Controllare che `LLM_API_KEY` sia nel `.env`: senza chiave i modelli senza training sono disattivati.
- **Versione A (consigliata):** se i modelli allenati sono stati registrati sulla GPU (opzione A, `scripts/record.py`),
  la casella *Final model* è attiva e risponde dalla registrazione. Registrare la foto usata nel video.
- **Versione B:** senza registrazioni, *Final model* mostra "Needs a GPU". Si usano 70B e 8B dal vivo e si legge la
  riga alternativa della scena 2.
- Fare una prova completa prima: la prima chiamata al gateway può essere lenta.

## Scena 1: il problema (0:00–0:15), pagina Demo vuota

> Galaxus shows millions of product images, and for every image it needs six answers: what it shows, the
> background, the perspective, whether a person is in it, which text is printed on it, and in which language.
> We built this with Apertus.

[Passare il mouse sul titolo, poi sulla zona della foto.]

## Scena 2: la demo (0:15–0:55), pagina Demo

[Trascinare la foto nel riquadro grigio.]

> I drop a product photo. On the right I pick the models.

**Versione A** [Lasciare spuntati *Final model* e *Apertus 70B, no training*, premere *Classify with 2 models*.]

> Our final model, and Apertus 70B as it comes, without training. Each column is a model, each row is one of the
> six questions. The final model answers in one pass, with a probability for every answer.

**Versione B** [Spuntare *Apertus 70B* e *Apertus 8B*, premere *Classify with 2 models*.]

> Here Apertus 8B and 70B answer live through the Hack Apertus API, without any training. Our trained model runs
> on a GPU, so on this laptop it is not connected; its scores are on the next page.

[Indicare una riga con "Models disagree" in rosso, se c'è; altrimenti una percentuale.]

> When two models give different answers, the row turns red. A red percentage means the model is less than 60%
> sure, so a person should take a look. Product details are optional: in production they are often missing, and
> the model still answers from the photo alone.

## Scena 3: i risultati (0:55–1:25), pagina Results

[Cliccare *Results*. Fermarsi sul grafico della classifica.]

> On the official Kaggle dev set our final model scores 0.832. That puts it in the middle assessment group: above
> Gemini 3.8 Flash, and level with Gemini 4 Argon.

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
| 0.832 Kaggle dev, gruppo centrale | Results, classifica | `src/training/apertus-8B-vadapter-lora-heads/RISULTATI.md:10`, `challenge_description.md:143-145` |
| prospettiva 0.198 → 0.755 | Results, *Question by question* | `src/plain-benchmark/apertus-8B-benchmark/prompt_eval_coverage.md:69` e `RISULTATI.md:62` |
| 4.9 img/s su L40S, 5.7 h, $10.20 per 100.000 | Results, *Speed and cost* | `RISULTATI.md:15`, prezzi HF in `models.yaml` |
| 0.37 GB di pesi allenati | How it works, legenda dello schema | `weights/apertus-8B-vadapter-lora-heads-final/` (du -sh) |
| sinistra/destra a caso (55%) | How it works, *What mattered* | `RISULTATI.md:84-87` |

"Level with Gemini 4 Argon": 0.832 contro 0.834, una differenza di 0.002. La challenge dice che differenze di pochi
decimi di punto non sono decisive (`challenge_description.md:211-212`).
