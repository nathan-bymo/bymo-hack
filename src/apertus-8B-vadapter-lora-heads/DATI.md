# Immagini usate per il training

Quali immagini di `train.csv` usa questo modello, in quale fase e con quale ruolo. Conteggi calcolati il 27 novembre
2026 sul `train.csv` della competizione (11.876 immagini, 7.260 prodotti). Le liste di ID non sono nel repository:
si ricostruiscono identiche su ogni macchina con il codice in fondo, perché tutti gli split dipendono solo da hash
del `product_id`.

## In breve

| fase | training | scelta del checkpoint | valutazione |
|---|---|---|---|
| **1. confronto** | **8.403** immagini (5.175 prodotti) | **941** immagini (572 prodotti) | `coverage`: 2.532 immagini (1.513 prodotti) |
| **2. modello finale** | tutte le **11.876** immagini (7.260 prodotti) | nessuna | solo submission sul dev di Kaggle |

- Le 8.403 + 941 sono `training_rows` (9.344 immagini): tutto `train.csv` tranne i prodotti di `coverage`.
- **Mai usate per il training:** le immagini `coverage` in fase 1, tutte le immagini dev (non etichettate) e final.
- Ogni prodotto sta in un solo gruppo: le immagini dello stesso prodotto non finiscono mai metà nel training e metà
  in validazione o in `coverage`.

## Fase 1

1. `coverage` (definito in `src/common/splits.py`) è il set di valutazione comune del team: i bucket di hash 0 e 1
   (`prompt-dev` + `prompt-test`, distribuzione originale) più i prodotti aggiunti per avere abbastanza esempi delle
   classi rare. In fase 1 questi prodotti sono esclusi da tutto: pesi, epoche, iperparametri.
2. Delle 9.344 immagini rimaste, il 10% dei prodotti diventa la **validazione interna** (hash `vlh-val:` +
   `product_id`, `--val-fraction 0.1`): 941 immagini. Serve solo a scegliere il checkpoint (2 valutazioni per epoca,
   macro F1 di `metric.py`).
3. Le altre **8.403** sono il training.
4. Alla fine `coverage` viene predetto una volta, end to end, con il checkpoint scelto.

## Fase 2

Training su tutte le 11.876 immagini con le stesse impostazioni e **lo stesso numero di epoche** scelto in fase 1;
si tiene l'ultimo checkpoint. Non ha un punteggio valido sul train: si controlla solo con una submission sul dev.

## Cosa entra nella loss

Si allena su ogni immagine, ma la loss di una dimensione conta solo le **celle valutate**: label umana
(`<dim>_source == "human"`) in una dimensione rilevante (`<dim>_relevant == "true"`). Le prelabel del modello e le
celle non rilevanti hanno label −1 e non cambiano la loss.

| dimensione | celle valutate nelle 8.403 (fase 1) | celle valutate nelle 11.876 (fase 2) |
|---|---|---|
| subject | 7.867 | 11.132 |
| background | 6.850 | 9.683 |
| perspective | 5.322 | 7.552 |
| individuals | 6.854 | 9.689 |
| text_classification | 7.561 | 10.707 |
| text_language | 3.584 | 5.135 |

## Augmentation e oversampling: item per epoca

Oltre a focal loss e alpha per classe (README), ogni epoca vede ogni immagine una volta più:

- **oversampling delle classi rare**: un'immagine è ripetuta finché la sua classe valutata più rara arriva a circa
  300 item per epoca, al massimo **4 volte** in tutto (l'originale + 3 ripetizioni);
- **ogni ripetizione è una variante diversa, non una copia**: l'immagine VQ a un'altra scala (lato 224, 288 o 240
  invece di 256, quindi altri token VQ), specchiata nelle ripetizioni dispari se l'immagine non ha testo. SigLIP
  vede sempre l'immagine intera. Niente ritagli: il prodotto tocca il bordo in quasi tutte le immagini e un ritaglio
  trasformerebbe un `full_product` in `detail_closeup` (linee guida). Nessun cambio di colore: renderebbe "color"
  uno sfondo "white";
- **mirror**: le immagini con perspective umana sinistra/destra e senza testo hanno anche una copia specchiata, con
  left e right scambiati nella label (equilibrio tra i due lati). Le prospettive simmetriche (`front_head_on`,
  `back_head_on`, `top_down`, `bottom_up`) restano uguali allo specchio.

| | immagini | ripetute 1× / 2× / 3× / 4× | altra scala | altra scala, specchiate | mirror | item per epoca |
|---|---|---|---|---|---|---|
| fase 1 | 8.403 | 5.841 / 1.777 / 517 / 268 | 2.584 | 1.031 | 1.372 | **13.390** |
| fase 2 | 11.876 | 9.780 / 1.638 / 245 / 213 | 1.886 | 881 | 1.917 | **16.560** |

Le varianti sono calcolate una volta da `encode.py` (fase 1: 14.331 tra immagini e varianti, validazione inclusa) e
sono identiche a ogni run. La validazione interna e `coverage` usano sempre l'immagine originale, preparata come in
inferenza.

**Correzione del 27 novembre (sera).** Il modello della fase 1 e la run `phase1-ocr` sono stati allenati con una
versione precedente, in cui le ripetizioni erano ritagli che tenevano il 90–100% di larghezza e altezza. Su 300
ritagli controllati, 231 tagliavano più dell'1% del contenuto e 129 più del 5%: in circa il 27% degli item di ogni
epoca un `full_product` poteva apparire come `detail_closeup`. Dalla versione `vq-scale-v2` le ripetizioni cambiano
solo la scala; `encode.py` scarta da solo dalla cache le varianti della versione vecchia.

## Classi in fase 1

Celle valutate per classe: nelle 8.403 immagini di training, negli item di un'epoca (con ripetizioni e mirror) e
nelle 941 di validazione interna (sempre originali).

| subject | training | item per epoca | validazione |
|---|---|---|---|
| full_product | 4.510 | 7.199 | 502 |
| detail_closeup | 590 | 674 | 72 |
| accessories | 159 | 361 | 8 |
| packaging | 281 | 595 | 40 |
| package_and_product | 171 | 374 | 22 |
| multipack_trade_unit | 99 | 418 | 9 |
| delivery_scope | 349 | 412 | 21 |
| product_grouping | 55 | 221 | 6 |
| multiview | 112 | 363 | 15 |
| energy_label | 312 | 312 | 38 |
| infographic | 701 | 959 | 96 |
| logo | 392 | 392 | 43 |
| placeholder | 136 | 408 | 9 |

| background | training | item per epoca | validazione |
|---|---|---|---|
| white | 5.636 | 9.439 | 642 |
| color | 689 | 1.177 | 53 |
| scene_lifestyle | 525 | 684 | 60 |

| perspective | training | item per epoca | validazione |
|---|---|---|---|
| front_head_on | 1.859 | 2.594 | 199 |
| front_angle_left | 755 | 1.378 | 84 |
| front_angle_right | 698 | 1.388 | 80 |
| back_head_on | 612 | 844 | 72 |
| back_angle_left | 177 | 388 | 19 |
| back_angle_right | 130 | 528 | 21 |
| left_side | 261 | 655 | 29 |
| right_side | 211 | 655 | 30 |
| top_down | 450 | 522 | 44 |
| bottom_up | 169 | 342 | 14 |

| individuals | training | item per epoca | validazione |
|---|---|---|---|
| individual_present | 458 | 727 | 57 |
| none | 6.396 | 10.577 | 698 |

| text_classification | training | item per epoca | validazione |
|---|---|---|---|
| certification_seal | 28 | 112 | 2 |
| technical_specs | 307 | 344 | 38 |
| promotional_text | 272 | 606 | 34 |
| nutritional_info | 394 | 571 | 44 |
| other | 246 | 536 | 26 |
| none | 6.314 | 10.107 | 707 |

| text_language | training | item per epoca | validazione |
|---|---|---|---|
| German | 938 | 1.328 | 103 |
| English | 1.100 | 1.676 | 132 |
| French | 146 | 440 | 19 |
| Italian | 17 | 68 | 0 |
| Dutch | 1 | 4 | 0 |
| other | 71 | 284 | 9 |
| none | 1.311 | 2.073 | 142 |

**Attenzione:** nella validazione interna non ci sono `Italian` né `Dutch` e ci sono solo 2 `certification_seal`.
La validazione basta per scegliere il checkpoint, ma il giudizio su queste classi va dato su `coverage`, che le
contiene apposta. `Dutch` ha un solo esempio valutato nel training: il modello non lo imparerà.

## Ricostruire le liste in locale

Con `data/train.csv` della competizione, dalla root del repository:

```python
import sys
sys.path[:0] = ["src/apertus-8B-vadapter-lora-heads", "src"]
import data as D

rows = D.read_csv("data/train.csv")
training = D.select(rows, "training")            # 9.344: fase 1, senza i prodotti di coverage
fit, validation = D.internal_split(training, 0.1)  # 8.403 per il training, 941 per scegliere il checkpoint
coverage = D.select(rows, "coverage")            # 2.532: valutazione della fase 1
everything = D.select(rows, "all")               # 11.876: fase 2
```

Le liste ottenute contengono ID della competizione: non vanno committate né condivise fuori dal team.
