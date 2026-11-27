# Apertus 8B congelato + 6 teste di classificazione

Domanda: le teste sulle rappresentazioni interne di Apertus fanno meglio della risposta generata? Nessun LoRA in
questo esperimento. **Primo risultato con Apertus:** [results_small_h200.md](results_small_h200.md) (split piccolo,
H200). Il confronto con la risposta generata sulle stesse immagini manca ancora.

```
immagine + contesto -> Apertus 8B congelato -> feature di alcuni layer
feature di un layer -> strato condiviso (512) -> 6 teste -> 6 classi
```

- **Feature:** per ogni layer scelto (12, 16, 20, 24, 28, 32), stato dell'ultimo token del prompt concatenato alla
  media sui token immagine. Un solo forward per immagine, senza generare testo.
- **Focal loss** su ogni testa (γ = 2), calcolata solo sulle label umane in celle rilevanti.
- **GradNorm** (Chen et al., 2018; α = 1.5): impara un peso per ogni loss confrontando i gradienti sui pesi
  condivisi. Con Apertus congelato e teste separate non ci sarebbero pesi condivisi: per questo c'è lo strato comune.
- **Gradient clipping** a norma 1 in tutte le varianti.
- **Ablation:** `ce`, `focal`, `ce+gradnorm`, `focal+gradnorm`.
- **Dati (fase 1):** solo `training_rows`. Un decimo dei suoi prodotti, scelto con un hash, è la validazione interna
  per scegliere layer ed epoca (8.471 immagini per il training, 873 per la validazione). `coverage` (2.532) viene
  solo predetto alla fine. La standardizzazione delle feature usa solo le immagini di training.

## Esecuzione

### 1. Feature, su GPU

L'8B ha circa 8,9 miliardi di parametri: a 16 bit i soli pesi occupano circa 18 GB, quindi non entrano in una T4 da
16 GB. Usa una GPU più grande (per esempio L40S 48 GB), oppure `--device auto` su due T4 (**non provato**).

Prima accetta i termini del modello su Hugging Face e fai `hf auth login`. Servono il codice del repo e in `data/`
`train.csv` e `train_images/`.

```sh
python -m venv .venv-gpu && . .venv-gpu/bin/activate
pip install -r src/apertus-8B-heads/requirements-gpu.txt
python src/apertus-8B-heads/extract_features.py --out data/features/smoke --limit 16   # prova
python src/apertus-8B-heads/extract_features.py --out data/features/8b_512            # tutte le 11.876
```

Stampa immagini al secondo e memoria di picco. Se si interrompe, rilanciando riprende. Le feature (circa 2,3 GB)
restano in `data/`, fuori da git.

### 2. Teste e confronto, anche su CPU

Richiede `data/metric.py`. Dipendenze: `src/requirements.txt` e `torch`.

```sh
python src/apertus-8B-heads/train_heads.py --features data/features/8b_512 --out data/runs/heads_8b_512
```

Per ogni variante sceglie layer ed epoca sulla validazione interna, salva le predizioni su `coverage` in CSV e i pesi
delle teste, e stampa il comando di confronto con la v2 zero-shot. Riporta tutte le varianti. La scelta tra varianti
va fatta sulla validazione interna, non sul punteggio `coverage`.

## Prototipo rapido (split di Nathan)

Per capire in poco tempo se le teste hanno senso: `train_1500.csv` e `validation_450.csv` in `data/nathan_split/`
(file privati, passati nel team, **mai nel repository**). Secondo il suo README: entrambi escludono i prodotti di
`coverage` e non condividono prodotti; il train approssima la distribuzione originale, la validazione è bilanciata
per classe dove possibile e quindi **non** è rappresentativa. La macro F1 su queste 450 non è confrontabile con i
punteggi su `coverage`: ogni confronto va fatto sulle stesse 450.

Le 450 vengono da `training_rows`: un modello allenato su tutto `training_rows` le ha studiate e non può essere
valutato su di esse.

```sh
python src/apertus-8B-heads/extract_features.py --out data/features/8b_512_small \
    --csv data/nathan_split/train_1500.csv data/nathan_split/validation_450.csv
python src/apertus-8B-heads/train_heads.py --features data/features/8b_512_small --out data/runs/heads_small \
    --train-csv data/nathan_split/train_1500.csv --eval-csv data/nathan_split/validation_450.csv \
    --val-fraction 0.2
```

Layer ed epoca si scelgono sul 20% dei prodotti delle 1.500 (1.199 immagini per il training, 301 per la scelta); le
450 servono solo per la valutazione.

Verifiche del 27 novembre: `train_1500.csv` (1.500 immagini, 1.311 prodotti) e `validation_450.csv` (450 immagini di
450 prodotti) non hanno immagini né prodotti in comune tra loro o con `coverage`, hanno righe identiche a `train.csv`
e conteggi uguali a quelli dichiarati. Nella validazione non c'è nessun esempio di `Dutch`.

**Soglia con soli pixel** su questo split (miniature 24×24, 30 epoche, un seed; **non è un risultato di Apertus**):

| su validation_450 | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| ce | 0.242 | 0.336 | 0.170 | 0.403 | 0.130 | 0.495 | 0.103 | 0.149 |
| focal | 0.256 | 0.345 | 0.188 | 0.450 | 0.137 | 0.504 | 0.106 | 0.150 |
| ce+gradnorm | 0.242 | 0.335 | 0.172 | 0.408 | 0.137 | 0.486 | 0.103 | 0.145 |
| focal+gradnorm | 0.257 | 0.344 | 0.186 | 0.435 | 0.129 | 0.529 | 0.111 | 0.152 |

Validazione interna (sceglie l'epoca): ce 0.327, focal 0.338, ce+gradnorm 0.326, focal+gradnorm 0.332.

## Verifiche fatte finora (27 novembre 2026)

- Focal loss: con γ = 0 coincide con la cross-entropy; con γ = 2 coincide con la formula.
- Maschera: le celle non valutate non cambiano la loss.
- GradNorm: i pesi sommano a 6 (numero di task); un task con gradiente gonfiato perde peso.
- Estrazione: provata con le classi Apertus del fork su un modello **in miniatura con pesi casuali** e un processor
  simulato. Batch con padding e immagini singole coincidono. **Non ancora provata con pesi e processor veri.**
- Pipeline completa su tutto il train con feature finte (sotto).

### Controllo con soli pixel (non è un risultato di Apertus)

Al posto delle feature di Apertus: miniature RGB 24×24 dell'immagine, 8 epoche, stesse teste. Serve da **soglia
minima**: feature utili devono superarla chiaramente.

| variante | validazione interna, macro F1 |
|---|---|
| ce | 0.377 |
| focal | 0.387 |
| ce+gradnorm | 0.380 |
| focal+gradnorm | 0.383 |

| su coverage | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| soli pixel, ce | 0.362 | 0.676 | 0.275 | 0.589 | 0.298 | 0.623 | 0.197 | 0.189 |
| soli pixel, focal+gradnorm | 0.366 | 0.679 | 0.275 | 0.606 | 0.308 | 0.604 | 0.208 | 0.195 |
| 8B v2 zero-shot, per confronto | 0.528 | 0.705 | 0.471 | 0.580 | 0.203 | 0.904 | 0.527 | 0.485 |

In `perspective` i soli pixel superano la v2 zero-shot di 8B e 70B (0.203 e 0.274). Differenze piccole tra le quattro
varianti in questa prova non dicono nulla su quale sarà migliore con le feature di Apertus (un solo seed).

## Lancio su Hugging Face Jobs

`build_hf_job.py` genera in `data/hf_jobs/` un job autonomo (contiene gli ID dello split: non va nel repository). I
comandi sono nella sua docstring. `CHECK_ONLY=1` su CPU verifica accessi e processor vero prima di usare la GPU.
`SPLIT=full` allena su `training_rows` e valuta su `coverage` (**non ancora eseguito**).

## Cosa manca

- Confronto con l'8B v2 zero-shot sulle stesse 450 immagini.
- Test sul set completo (`SPLIT=full`).
- Predizioni su dev e final per una eventuale submission.
- Fase 2 (riallenamento su tutte le immagini), solo dopo aver scelto la struttura finale.
- Conferma degli organizzatori che teste su Apertus contano come "Apertus classifica".
