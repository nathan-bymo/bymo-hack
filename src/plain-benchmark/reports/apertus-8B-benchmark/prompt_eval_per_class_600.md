# Risultati per classe: prime 600 immagini di prompt-dev

Elenco di tutte le prove: [RISULTATI.md](../../training/EXPERIMENTS.md).

## Scheda della prova P1 e P3: tutte le varianti di prompt sulle prime 600 immagini

| voce | |
|---|---|
| **Domanda** | P1: contano regole dettagliate e risposta a lettere (v1)? P3: conta il disegno a domande separate (v2)? |
| **Modelli** | Apertus 1.5 8B (7 varianti) e 70B (2 varianti) base, **nessun training**, via API Keera |
| **Valutazione** | 600 immagini (363 prodotti) di `prompt-dev`, dal train etichettato |
| **Impostazioni** | immagini a 512², contesto prodotto incluso, temperatura 0, risposta vincolata |
| **Cosa non dice** | la v2 è stata scelta guardando queste immagini: valori forse ottimistici; classi con 1-3 esempi non affidabili |


600 immagini di 363 prodotti, dal train etichettato. API Keera, immagini a 512², temperatura 0. Punteggio di `metric.py`,
solo sulle celle con label umana e rilevanti. Le varianti sono descritte in `prompt_eval.md` e in `src/training/common/prompt.py`.

**Come leggere:** `n` è il numero di esempi della classe nelle celle valutate. Con pochi esempi (indicati con ⚠ se `n < 10`)
il valore della classe è molto instabile. La macro F1 è la media delle classi, non la mediana. La v2 è stata scelta
guardando prompt-dev, quindi questi valori possono essere ottimistici. Non sono punteggi della classifica Kaggle.

## Riepilogo

| variante | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| 8B v1 breve JSON | 0.327 | 0.649 | 0.065 | 0.403 | 0.091 | 0.795 | 0.188 | 0.422 |
| 8B v1 breve lettere | 0.138 | 0.348 | 0.060 | 0.077 | 0.085 | 0.482 | 0.036 | 0.090 |
| 8B v1 completo JSON | 0.326 | 0.629 | 0.057 | 0.351 | 0.165 | 0.747 | 0.215 | 0.420 |
| 8B v1 completo lettere | 0.164 | 0.462 | 0.056 | 0.139 | 0.043 | 0.481 | 0.153 | 0.112 |
| 8B v2 separate nomi | 0.453 | 0.633 | 0.439 | 0.522 | 0.103 | 0.880 | 0.413 | 0.359 |
| 8B v2 separate lettere | 0.482 | 0.687 | 0.448 | 0.603 | 0.131 | 0.896 | 0.486 | 0.328 |
| 8B v2 descrizione | 0.521 | 0.719 | 0.474 | 0.575 | 0.228 | 0.895 | 0.511 | 0.446 |
| 70B v2 separate lettere | 0.454 | 0.682 | 0.427 | 0.521 | 0.135 | 0.783 | 0.528 | 0.332 |
| 70B v2 descrizione | 0.587 | 0.745 | 0.574 | 0.629 | 0.265 | 0.902 | 0.702 | 0.453 |

## F1 per classe, tutte le varianti

Un `-` indica una classe senza esempi e mai predetta da quella variante.

### subject

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| full_product | 336 | 0.75 | 0.75 | 0.75 | 0.71 | 0.81 | 0.79 | 0.72 | 0.82 | 0.81 |
| infographic | 44 | 0.00 | 0.00 | 0.00 | 0.00 | 0.37 | 0.45 | 0.36 | 0.14 | 0.37 |
| delivery_scope | 29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.13 | 0.39 |
| detail_closeup | 28 | 0.00 | 0.04 | 0.00 | 0.02 | 0.35 | 0.35 | 0.24 | 0.33 | 0.34 |
| logo | 24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.79 | 0.78 | 0.83 | 0.83 | 0.92 |
| packaging | 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.42 | 0.30 | 0.43 | 0.38 | 0.44 |
| multiview | 18 | 0.10 | 0.00 | 0.00 | 0.00 | 0.08 | 0.14 | 0.44 | 0.22 | 0.45 |
| energy_label | 17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.94 | 0.92 | 0.94 | 0.94 | 0.97 |
| package_and_product | 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.60 | 0.51 | 0.48 | 0.60 | 0.61 |
| accessories ⚠ | 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 |
| placeholder ⚠ | 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 0.84 | 0.84 | 0.67 | 0.89 |
| multipack_trade_unit ⚠ | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.57 | 0.75 | 0.75 | 0.00 | 0.40 |
| product_grouping ⚠ | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.67 |
| **macro F1** | 553 celle | **0.065** | **0.060** | **0.057** | **0.056** | **0.439** | **0.448** | **0.474** | **0.427** | **0.574** |

### background

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| white | 421 | 0.91 | 0.09 | 0.81 | 0.27 | 0.79 | 0.90 | 0.91 | 0.91 | 0.92 |
| scene_lifestyle | 47 | 0.26 | 0.00 | 0.15 | 0.00 | 0.67 | 0.76 | 0.69 | 0.62 | 0.77 |
| color | 38 | 0.04 | 0.15 | 0.09 | 0.15 | 0.11 | 0.15 | 0.12 | 0.03 | 0.19 |
| **macro F1** | 506 celle | **0.403** | **0.077** | **0.351** | **0.139** | **0.522** | **0.603** | **0.575** | **0.521** | **0.629** |

### perspective

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 148 | 0.57 | 0.50 | 0.58 | 0.01 | 0.58 | 0.57 | 0.70 | 0.59 | 0.74 |
| front_angle_left | 54 | 0.10 | 0.03 | 0.00 | 0.00 | 0.16 | 0.29 | 0.37 | 0.00 | 0.00 |
| front_angle_right | 47 | 0.21 | 0.26 | 0.00 | 0.21 | 0.12 | 0.00 | 0.04 | 0.22 | 0.35 |
| back_head_on | 45 | 0.04 | 0.00 | 0.32 | 0.00 | 0.16 | 0.15 | 0.33 | 0.12 | 0.57 |
| top_down | 31 | 0.00 | 0.06 | 0.00 | 0.20 | 0.00 | 0.00 | 0.26 | 0.12 | 0.29 |
| left_side | 23 | 0.00 | 0.00 | 0.22 | 0.00 | 0.00 | 0.12 | 0.18 | 0.00 | 0.15 |
| back_angle_right | 19 | 0.00 | 0.00 | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| right_side | 16 | 0.00 | 0.00 | 0.34 | 0.00 | 0.00 | 0.18 | 0.20 | 0.31 | 0.37 |
| bottom_up ⚠ | 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.18 |
| back_angle_left ⚠ | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **macro F1** | 400 celle | **0.091** | **0.085** | **0.165** | **0.043** | **0.103** | **0.131** | **0.228** | **0.135** | **0.265** |

### individuals

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 470 | 0.97 | 0.96 | 0.97 | 0.96 | 0.98 | 0.98 | 0.98 | 0.97 | 0.99 |
| individual_present | 36 | 0.62 | 0.00 | 0.52 | 0.00 | 0.78 | 0.81 | 0.81 | 0.59 | 0.82 |
| **macro F1** | 506 celle | **0.795** | **0.482** | **0.747** | **0.481** | **0.880** | **0.896** | **0.895** | **0.783** | **0.902** |

### text_classification

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 460 | 0.89 | 0.15 | 0.89 | 0.87 | 0.69 | 0.86 | 0.91 | 0.91 | 0.91 |
| technical_specs | 27 | 0.03 | 0.07 | 0.00 | 0.00 | 0.59 | 0.73 | 0.77 | 0.80 | 0.88 |
| nutritional_info | 26 | 0.14 | 0.00 | 0.40 | 0.00 | 0.72 | 0.63 | 0.62 | 0.73 | 0.78 |
| other | 20 | 0.00 | 0.00 | 0.00 | 0.05 | 0.11 | 0.29 | 0.34 | 0.23 | 0.19 |
| promotional_text | 16 | 0.06 | 0.00 | 0.00 | 0.00 | 0.34 | 0.38 | 0.42 | 0.22 | 0.44 |
| certification_seal ⚠ | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.03 | 0.00 | 0.29 | 1.00 |
| **macro F1** | 550 celle | **0.188** | **0.036** | **0.215** | **0.153** | **0.413** | **0.486** | **0.511** | **0.528** | **0.702** |

### text_language

| classe | n | 8B min JSON | 8B min lett. | 8B compl. JSON | 8B compl. lett. | 8B sep. nomi | 8B sep. lett. | 8B descr. | 70B sep. lett. | 70B descr. |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 107 | 0.76 | 0.55 | 0.49 | 0.25 | 0.00 | 0.12 | 0.27 | 0.09 | 0.27 |
| English | 89 | 0.41 | 0.02 | 0.55 | 0.11 | 0.54 | 0.54 | 0.64 | 0.40 | 0.62 |
| German | 65 | 0.66 | 0.06 | 0.53 | 0.42 | 0.50 | 0.50 | 0.61 | 0.43 | 0.50 |
| French | 10 | 0.46 | 0.00 | 0.71 | 0.00 | 0.47 | 0.47 | 0.73 | 0.40 | 0.84 |
| other ⚠ | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.44 |
| Dutch ⚠ | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 |
| Italian ⚠ | 1 | 0.67 | 0.00 | 0.67 | 0.00 | 1.00 | 0.67 | 0.67 | 1.00 | 0.50 |
| **macro F1** | 279 celle | **0.422** | **0.090** | **0.420** | **0.112** | **0.359** | **0.328** | **0.446** | **0.332** | **0.453** |

## Precision e recall delle tre varianti principali

Precision: quando il modello sceglie la classe, quante volte ha ragione. Recall: quanti esempi veri della classe trova.
Ogni cella è `precision / recall`.

### subject

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| full_product | 336 | 0.61 / 0.99 | 0.81 / 0.64 | 0.87 / 0.76 |
| infographic | 44 | 0.00 / 0.00 | 0.55 / 0.27 | 0.57 / 0.27 |
| delivery_scope | 29 | 0.00 / 0.00 | 1.00 / 0.07 | 1.00 / 0.24 |
| detail_closeup | 28 | 0.00 / 0.00 | 0.15 / 0.61 | 0.25 / 0.54 |
| logo | 24 | 0.00 / 0.00 | 0.71 / 1.00 | 0.88 / 0.96 |
| packaging | 20 | 0.00 / 0.00 | 0.41 / 0.45 | 0.33 / 0.70 |
| multiview | 18 | 0.33 / 0.06 | 0.50 / 0.39 | 0.34 / 0.67 |
| energy_label | 17 | 0.00 / 0.00 | 0.89 / 1.00 | 0.94 / 1.00 |
| package_and_product | 14 | 0.00 / 0.00 | 0.32 / 1.00 | 0.44 / 1.00 |
| accessories ⚠ | 9 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 0.11 |
| placeholder ⚠ | 9 | 0.00 / 0.00 | 0.80 / 0.89 | 0.89 / 0.89 |
| multipack_trade_unit ⚠ | 3 | 0.00 / 0.00 | 0.60 / 1.00 | 0.50 / 0.33 |
| product_grouping ⚠ | 2 | 0.00 / 0.00 | 0.00 / 0.00 | 0.50 / 1.00 |

### background

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| white | 421 | 0.86 / 0.97 | 0.89 / 0.93 | 0.92 / 0.92 |
| scene_lifestyle | 47 | 0.57 / 0.17 | 0.75 / 0.64 | 0.78 / 0.77 |
| color | 38 | 0.07 / 0.03 | 0.14 / 0.11 | 0.20 / 0.18 |

### perspective

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| front_head_on | 148 | 0.40 / 0.97 | 0.71 / 0.68 | 0.62 / 0.91 |
| front_angle_left | 54 | 0.33 / 0.06 | 0.25 / 0.70 | 0.00 / 0.00 |
| front_angle_right | 47 | 0.26 / 0.17 | 0.25 / 0.02 | 0.25 / 0.60 |
| back_head_on | 45 | 0.50 / 0.02 | 0.90 / 0.20 | 0.86 / 0.42 |
| top_down | 31 | 0.00 / 0.00 | 0.71 / 0.16 | 0.39 / 0.23 |
| left_side | 23 | 0.00 / 0.00 | 0.18 / 0.17 | 0.50 / 0.09 |
| back_angle_right | 19 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| right_side | 16 | 0.00 / 0.00 | 0.13 / 0.50 | 0.30 / 0.50 |
| bottom_up ⚠ | 9 | 0.00 / 0.00 | 1.00 / 0.11 | 0.50 / 0.11 |
| back_angle_left ⚠ | 8 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

### individuals

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| none | 470 | 0.97 / 0.98 | 1.00 / 0.96 | 0.99 / 0.99 |
| individual_present | 36 | 0.66 / 0.58 | 0.68 / 1.00 | 0.83 / 0.81 |

### text_classification

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| none | 460 | 0.86 / 0.93 | 0.98 / 0.86 | 0.97 / 0.86 |
| technical_specs | 27 | 0.03 / 0.04 | 0.90 / 0.67 | 0.96 / 0.81 |
| nutritional_info | 26 | 1.00 / 0.08 | 0.46 / 0.96 | 0.80 / 0.77 |
| other | 20 | 0.00 / 0.00 | 0.56 / 0.25 | 0.13 / 0.35 |
| promotional_text | 16 | 0.06 / 0.06 | 0.27 / 0.88 | 0.32 / 0.75 |
| certification_seal ⚠ | 1 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 |

### text_language

| classe | n | 8B v1 breve JSON | 8B v2 descrizione | 70B v2 descrizione |
|---|---|---|---|---|
| none | 107 | 0.65 / 0.91 | 1.00 / 0.16 | 1.00 / 0.16 |
| English | 89 | 0.83 / 0.27 | 0.52 / 0.84 | 0.65 / 0.58 |
| German | 65 | 0.55 / 0.82 | 0.51 / 0.75 | 0.35 / 0.88 |
| French | 10 | 1.00 / 0.30 | 0.67 / 0.80 | 0.89 / 0.80 |
| other ⚠ | 5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.50 / 0.40 |
| Dutch ⚠ | 2 | 0.00 / 0.00 | 0.12 / 0.50 | 0.00 / 0.00 |
| Italian ⚠ | 1 | 0.50 / 1.00 | 0.50 / 1.00 | 0.33 / 1.00 |
