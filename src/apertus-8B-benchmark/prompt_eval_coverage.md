# Confronto sul set coverage: 2.532 immagini (27 novembre 2026)

Cinque varianti zero-shot sul train etichettato, API Keera, modelli base, immagini a 512², temperatura 0, risposta
vincolata. Punteggio di `metric.py`, solo celle con label umana e rilevanti. Nessuna richiesta fallita e nessuna
risposta non valida. Split in `src/common/splits.py`.

## Varianti

- **v1 breve** (`minimal:json`): breve system prompt con i nomi delle classi, risposta JSON.
- **prompt condiviso** (`shared:json`): `src/docs/system_prompt.txt` come messaggio system, contesto prodotto nel
  messaggio utente, risposta JSON.
- **v2 descrizione** (`describe:codes`): descrizione dell'immagine, poi una domanda per dimensione con risposta a una
  lettera. Sono 7 richieste per immagine.

## Come riprodurre

```sh
.venv/bin/python src/scripts/eval_prompt.py 8b:minimal:json:512 8b:shared:json:512 8b:describe:codes:512 70b:shared:json:512 70b:describe:codes:512 \
    --split coverage --concurrency 32 --bootstrap 200 --classes
```

## Risultati: coverage

2532 immagini, 1513 prodotti.

| variante | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| 8B v1 breve | 0.316 | 0.632 | 0.076 | 0.411 | 0.087 | 0.734 | 0.192 | 0.397 |
| 8B prompt condiviso | 0.229 | 0.467 | 0.065 | 0.268 | 0.098 | 0.581 | 0.213 | 0.151 |
| 8B v2 descrizione | 0.528 | 0.705 | 0.471 | 0.580 | 0.203 | 0.904 | 0.527 | 0.485 |
| 70B prompt condiviso | 0.304 | 0.616 | 0.076 | 0.397 | 0.117 | 0.661 | 0.275 | 0.296 |
| 70B v2 descrizione | 0.583 | 0.739 | 0.546 | 0.631 | 0.274 | 0.899 | 0.618 | 0.529 |

## Risultati: base (distribuzione originale delle classi)

2363 immagini, 1438 prodotti.

| variante | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| 8B v1 breve | 0.310 | 0.636 | 0.079 | 0.416 | 0.087 | 0.733 | 0.190 | 0.357 |
| 8B prompt condiviso | 0.230 | 0.472 | 0.066 | 0.265 | 0.097 | 0.585 | 0.217 | 0.153 |
| 8B v2 descrizione | 0.513 | 0.708 | 0.473 | 0.576 | 0.205 | 0.911 | 0.494 | 0.417 |
| 70B prompt condiviso | 0.299 | 0.618 | 0.077 | 0.401 | 0.115 | 0.657 | 0.276 | 0.267 |
| 70B v2 descrizione | 0.587 | 0.742 | 0.545 | 0.630 | 0.276 | 0.909 | 0.625 | 0.537 |

## Risultati: prompt-test (non usato per scegliere il prompt)

1193 immagini, 713 prodotti.

| variante | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| 8B v1 breve | 0.291 | 0.631 | 0.073 | 0.421 | 0.086 | 0.665 | 0.196 | 0.306 |
| 8B prompt condiviso | 0.225 | 0.469 | 0.067 | 0.269 | 0.090 | 0.562 | 0.226 | 0.135 |
| 8B v2 descrizione | 0.502 | 0.703 | 0.464 | 0.545 | 0.198 | 0.921 | 0.476 | 0.407 |
| 70B prompt condiviso | 0.294 | 0.621 | 0.086 | 0.418 | 0.114 | 0.612 | 0.279 | 0.255 |
| 70B v2 descrizione | 0.590 | 0.740 | 0.543 | 0.611 | 0.276 | 0.922 | 0.618 | 0.569 |

## Differenze di macro F1

Intervallo al 95%, bootstrap appaiato per prodotto con 200 ricampionamenti.

| confronto (primo meno secondo) | coverage | base | prompt-test |
|---|---|---|---|
| 8B prompt condiviso − 8B v1 breve | -0.087 [-0.103, -0.073] | -0.080 [-0.100, -0.064] | -0.066 [-0.084, -0.048] |
| 8B v2 descrizione − 8B v1 breve | +0.212 [+0.193, +0.229] | +0.202 [+0.179, +0.222] | +0.211 [+0.175, +0.234] |
| 70B prompt condiviso − 8B v1 breve | -0.013 [-0.024, -0.001] | -0.012 [-0.026, +0.005] | +0.003 [-0.023, +0.022] |
| 70B v2 descrizione − 8B v1 breve | +0.267 [+0.245, +0.285] | +0.277 [+0.247, +0.297] | +0.299 [+0.251, +0.324] |
| 8B v2 descrizione − 8B prompt condiviso | +0.299 [+0.284, +0.315] | +0.282 [+0.263, +0.296] | +0.277 [+0.253, +0.297] |
| 70B v2 descrizione − 8B v2 descrizione | +0.055 [+0.041, +0.069] | +0.075 [+0.052, +0.089] | +0.088 [+0.059, +0.111] |
| 70B v2 descrizione − 70B prompt condiviso | +0.279 [+0.258, +0.300] | +0.288 [+0.257, +0.309] | +0.296 [+0.253, +0.329] |
| 70B prompt condiviso − 8B prompt condiviso | +0.074 [+0.059, +0.089] | +0.068 [+0.050, +0.088] | +0.069 [+0.044, +0.092] |

## Velocità

Misurata su gateway condiviso con 32 immagini in parallelo; non è la velocità di un'installazione propria. Le immagini
al secondo riguardano le richieste nuove di questo run (le altre venivano dalla cache).

| variante | richieste nuove (immagini) | immagini/s | latenza mediana per immagine (s) | token di prompt per immagine |
|---|---|---|---|---|
| 8B v1 breve | 1932 | 9.7 | 3.2 | 1386 |
| 8B prompt condiviso | 2522 | 9.6 | 3.3 | 2769 |
| 8B v2 descrizione | 1932 | 3.8 | 8.0 | 10011 |
| 70B prompt condiviso | 2522 | 4.1 | 7.8 | 2769 |
| 70B v2 descrizione | 1932 | 2.8 | 11.1 | 10084 |

La latenza mediana è calcolata su tutte le 2.532 risposte, incluse quelle salvate in run precedenti.

## F1 per classe

`n` è il numero di esempi della classe nelle celle valutate. `-`: classe senza esempi e mai predetta.

### coverage

#### subject

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| full_product | 1363 | 0.73 | 0.61 | 0.71 | 0.72 | 0.80 |
| infographic | 215 | 0.00 | 0.00 | 0.38 | 0.02 | 0.43 |
| detail_closeup | 158 | 0.00 | 0.00 | 0.27 | 0.00 | 0.38 |
| logo | 124 | 0.00 | 0.00 | 0.82 | 0.00 | 0.89 |
| delivery_scope | 109 | 0.00 | 0.00 | 0.14 | 0.00 | 0.33 |
| packaging | 88 | 0.04 | 0.04 | 0.49 | 0.02 | 0.47 |
| energy_label | 80 | 0.00 | 0.00 | 0.94 | 0.00 | 0.99 |
| package_and_product | 58 | 0.00 | 0.12 | 0.52 | 0.04 | 0.60 |
| multiview | 57 | 0.03 | 0.03 | 0.37 | 0.00 | 0.38 |
| accessories | 38 | 0.00 | 0.00 | 0.00 | 0.03 | 0.05 |
| placeholder | 33 | 0.00 | 0.00 | 0.81 | 0.00 | 0.85 |
| multipack_trade_unit | 31 | 0.13 | 0.05 | 0.50 | 0.06 | 0.37 |
| product_grouping | 30 | 0.06 | 0.00 | 0.18 | 0.11 | 0.55 |

#### background

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| white | 1730 | 0.91 | 0.06 | 0.91 | 0.91 | 0.91 |
| color | 204 | 0.05 | 0.18 | 0.18 | 0.09 | 0.24 |
| scene_lifestyle | 144 | 0.27 | 0.56 | 0.65 | 0.20 | 0.74 |

#### perspective

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| front_head_on | 614 | 0.58 | 0.56 | 0.65 | 0.58 | 0.73 |
| back_head_on | 211 | 0.03 | 0.04 | 0.34 | 0.34 | 0.56 |
| front_angle_left | 209 | 0.07 | 0.00 | 0.37 | 0.00 | 0.00 |
| front_angle_right | 199 | 0.16 | 0.06 | 0.05 | 0.00 | 0.42 |
| top_down | 120 | 0.03 | 0.02 | 0.14 | 0.09 | 0.32 |
| left_side | 82 | 0.00 | 0.08 | 0.18 | 0.00 | 0.10 |
| right_side | 65 | 0.00 | 0.17 | 0.25 | 0.00 | 0.40 |
| back_angle_right | 60 | 0.00 | 0.06 | 0.00 | 0.03 | 0.00 |
| bottom_up | 41 | 0.00 | 0.00 | 0.05 | 0.13 | 0.20 |
| back_angle_left | 37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

#### individuals

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1966 | 0.97 | 0.97 | 0.99 | 0.98 | 0.99 |
| individual_present | 114 | 0.50 | 0.19 | 0.82 | 0.34 | 0.81 |

#### text_classification

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1866 | 0.88 | 0.90 | 0.90 | 0.90 | 0.90 |
| nutritional_info | 127 | 0.17 | 0.38 | 0.67 | 0.75 | 0.81 |
| technical_specs | 103 | 0.01 | 0.00 | 0.65 | 0.00 | 0.79 |
| promotional_text | 86 | 0.09 | 0.00 | 0.44 | 0.00 | 0.44 |
| other | 83 | 0.00 | 0.00 | 0.18 | 0.00 | 0.22 |
| certification_seal | 30 | 0.00 | 0.00 | 0.32 | 0.00 | 0.55 |

#### text_language

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 391 | 0.72 | 0.54 | 0.14 | 0.07 | 0.20 |
| English | 331 | 0.47 | 0.02 | 0.63 | 0.45 | 0.61 |
| German | 288 | 0.65 | 0.31 | 0.66 | 0.54 | 0.54 |
| French | 56 | 0.44 | 0.19 | 0.68 | 0.25 | 0.64 |
| Italian | 30 | 0.41 | 0.00 | 0.69 | 0.46 | 0.71 |
| other | 30 | 0.00 | 0.00 | 0.06 | 0.05 | 0.38 |
| Dutch | 20 | 0.10 | 0.00 | 0.53 | 0.25 | 0.63 |

### base (distribuzione originale delle classi)

#### subject

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| full_product | 1280 | 0.73 | 0.61 | 0.72 | 0.72 | 0.80 |
| infographic | 193 | 0.00 | 0.00 | 0.38 | 0.02 | 0.41 |
| detail_closeup | 153 | 0.00 | 0.00 | 0.28 | 0.00 | 0.38 |
| logo | 124 | 0.00 | 0.00 | 0.85 | 0.00 | 0.90 |
| delivery_scope | 101 | 0.00 | 0.00 | 0.13 | 0.00 | 0.35 |
| energy_label | 80 | 0.00 | 0.00 | 0.96 | 0.00 | 0.99 |
| packaging | 75 | 0.00 | 0.04 | 0.47 | 0.02 | 0.46 |
| multiview | 52 | 0.03 | 0.03 | 0.36 | 0.00 | 0.40 |
| package_and_product | 51 | 0.00 | 0.12 | 0.49 | 0.04 | 0.59 |
| accessories | 38 | 0.00 | 0.00 | 0.00 | 0.03 | 0.05 |
| placeholder | 33 | 0.00 | 0.00 | 0.82 | 0.00 | 0.85 |
| multipack_trade_unit | 26 | 0.15 | 0.05 | 0.58 | 0.07 | 0.42 |
| product_grouping | 16 | 0.11 | 0.00 | 0.11 | 0.09 | 0.47 |

#### background

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| white | 1601 | 0.90 | 0.06 | 0.91 | 0.91 | 0.91 |
| color | 193 | 0.06 | 0.19 | 0.17 | 0.09 | 0.23 |
| scene_lifestyle | 137 | 0.29 | 0.55 | 0.65 | 0.21 | 0.75 |

#### perspective

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| front_head_on | 558 | 0.57 | 0.55 | 0.65 | 0.57 | 0.73 |
| front_angle_left | 196 | 0.08 | 0.00 | 0.36 | 0.00 | 0.00 |
| back_head_on | 189 | 0.03 | 0.04 | 0.34 | 0.33 | 0.57 |
| front_angle_right | 183 | 0.16 | 0.04 | 0.05 | 0.00 | 0.42 |
| top_down | 118 | 0.03 | 0.02 | 0.15 | 0.09 | 0.34 |
| left_side | 80 | 0.00 | 0.08 | 0.18 | 0.00 | 0.09 |
| right_side | 64 | 0.00 | 0.17 | 0.27 | 0.00 | 0.41 |
| back_angle_right | 60 | 0.00 | 0.06 | 0.00 | 0.03 | 0.00 |
| bottom_up | 41 | 0.00 | 0.00 | 0.05 | 0.13 | 0.20 |
| back_angle_left | 36 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

#### individuals

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1825 | 0.97 | 0.97 | 0.99 | 0.98 | 0.99 |
| individual_present | 108 | 0.49 | 0.20 | 0.83 | 0.34 | 0.83 |

#### text_classification

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1749 | 0.89 | 0.91 | 0.90 | 0.91 | 0.91 |
| nutritional_info | 114 | 0.15 | 0.39 | 0.66 | 0.75 | 0.82 |
| technical_specs | 98 | 0.01 | 0.00 | 0.66 | 0.00 | 0.80 |
| promotional_text | 77 | 0.10 | 0.00 | 0.43 | 0.00 | 0.44 |
| other | 77 | 0.00 | 0.00 | 0.16 | 0.00 | 0.22 |
| certification_seal | 11 | 0.00 | 0.00 | 0.14 | 0.00 | 0.56 |

#### text_language

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 362 | 0.73 | 0.55 | 0.13 | 0.06 | 0.18 |
| English | 310 | 0.47 | 0.02 | 0.64 | 0.44 | 0.61 |
| German | 271 | 0.68 | 0.33 | 0.67 | 0.56 | 0.56 |
| French | 52 | 0.40 | 0.18 | 0.69 | 0.27 | 0.67 |
| other | 16 | 0.00 | 0.00 | 0.00 | 0.05 | 0.57 |
| Italian ⚠ | 7 | 0.22 | 0.00 | 0.50 | 0.25 | 0.67 |
| Dutch ⚠ | 6 | 0.00 | 0.00 | 0.29 | 0.25 | 0.50 |

### prompt-test (non usato per scegliere il prompt)

#### subject

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| full_product | 653 | 0.73 | 0.63 | 0.73 | 0.73 | 0.81 |
| infographic | 99 | 0.00 | 0.00 | 0.37 | 0.02 | 0.39 |
| detail_closeup | 95 | 0.00 | 0.00 | 0.31 | 0.00 | 0.39 |
| logo | 67 | 0.00 | 0.00 | 0.84 | 0.00 | 0.89 |
| delivery_scope | 54 | 0.00 | 0.00 | 0.14 | 0.00 | 0.36 |
| energy_label | 40 | 0.00 | 0.00 | 0.96 | 0.00 | 0.99 |
| packaging | 37 | 0.00 | 0.04 | 0.38 | 0.00 | 0.45 |
| package_and_product | 28 | 0.00 | 0.14 | 0.54 | 0.05 | 0.66 |
| multiview | 20 | 0.00 | 0.07 | 0.43 | 0.00 | 0.42 |
| accessories | 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| placeholder | 13 | 0.00 | 0.00 | 0.71 | 0.00 | 0.83 |
| product_grouping | 10 | 0.00 | 0.00 | 0.18 | 0.14 | 0.55 |
| multipack_trade_unit ⚠ | 8 | 0.22 | 0.00 | 0.44 | 0.18 | 0.33 |

#### background

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| white | 819 | 0.91 | 0.05 | 0.90 | 0.91 | 0.91 |
| color | 103 | 0.06 | 0.19 | 0.16 | 0.09 | 0.26 |
| scene_lifestyle | 51 | 0.29 | 0.56 | 0.57 | 0.25 | 0.67 |

#### perspective

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| front_head_on | 275 | 0.57 | 0.56 | 0.64 | 0.57 | 0.72 |
| back_head_on | 101 | 0.04 | 0.04 | 0.39 | 0.38 | 0.63 |
| front_angle_left | 86 | 0.08 | 0.00 | 0.35 | 0.00 | 0.00 |
| front_angle_right | 80 | 0.12 | 0.06 | 0.02 | 0.00 | 0.40 |
| top_down | 69 | 0.06 | 0.03 | 0.10 | 0.12 | 0.34 |
| left_side | 44 | 0.00 | 0.11 | 0.18 | 0.00 | 0.04 |
| right_side | 32 | 0.00 | 0.05 | 0.29 | 0.00 | 0.42 |
| back_angle_right | 26 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 |
| back_angle_left | 25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| bottom_up | 25 | 0.00 | 0.00 | 0.00 | 0.07 | 0.21 |

#### individuals

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 920 | 0.97 | 0.97 | 0.99 | 0.97 | 0.99 |
| individual_present | 54 | 0.36 | 0.15 | 0.85 | 0.25 | 0.85 |

#### text_classification

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 879 | 0.90 | 0.91 | 0.91 | 0.91 | 0.90 |
| nutritional_info | 62 | 0.15 | 0.45 | 0.72 | 0.76 | 0.88 |
| technical_specs | 45 | 0.00 | 0.00 | 0.54 | 0.00 | 0.74 |
| promotional_text | 45 | 0.13 | 0.00 | 0.47 | 0.00 | 0.47 |
| other | 36 | 0.00 | 0.00 | 0.05 | 0.00 | 0.21 |
| certification_seal ⚠ | 6 | 0.00 | 0.00 | 0.17 | 0.00 | 0.50 |

#### text_language

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 183 | 0.71 | 0.55 | 0.06 | 0.08 | 0.16 |
| English | 156 | 0.44 | 0.03 | 0.64 | 0.47 | 0.60 |
| German | 131 | 0.66 | 0.31 | 0.67 | 0.53 | 0.56 |
| French | 30 | 0.33 | 0.06 | 0.72 | 0.24 | 0.59 |
| other | 10 | 0.00 | 0.00 | 0.00 | 0.07 | 0.59 |
| Italian ⚠ | 4 | 0.00 | 0.00 | 0.40 | 0.00 | 0.75 |
| Dutch ⚠ | 4 | 0.00 | 0.00 | 0.35 | 0.40 | 0.73 |

## Precision e recall per classe (coverage)

Ogni cella è `precision / recall`.

### subject

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| full_product | 1363 | 0.58 / 0.99 | 0.61 / 0.61 | 0.82 / 0.62 | 0.59 / 0.92 | 0.87 / 0.74 |
| infographic | 215 | 0.00 / 0.00 | 0.00 / 0.00 | 0.57 / 0.29 | 0.12 / 0.01 | 0.66 / 0.32 |
| detail_closeup | 158 | 0.00 / 0.00 | 0.00 / 0.00 | 0.17 / 0.64 | 0.00 / 0.00 | 0.28 / 0.58 |
| logo | 124 | 0.00 / 0.00 | 0.00 / 0.00 | 0.70 / 0.99 | 0.00 / 0.00 | 0.81 / 0.99 |
| delivery_scope | 109 | 0.00 / 0.00 | 0.00 / 0.00 | 0.89 / 0.07 | 0.00 / 0.00 | 0.96 / 0.20 |
| packaging | 88 | 0.20 / 0.02 | 0.02 / 0.09 | 0.46 / 0.52 | 0.12 / 0.01 | 0.34 / 0.75 |
| energy_label | 80 | 0.00 / 0.00 | 0.00 / 0.00 | 0.89 / 1.00 | 0.00 / 0.00 | 0.99 / 0.99 |
| package_and_product | 58 | 0.00 / 0.00 | 0.07 / 0.78 | 0.36 / 0.90 | 0.03 / 0.09 | 0.45 / 0.90 |
| multiview | 57 | 0.12 / 0.02 | 0.07 / 0.02 | 0.39 / 0.35 | 0.00 / 0.00 | 0.26 / 0.70 |
| accessories | 38 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.03 / 0.03 | 1.00 / 0.03 |
| placeholder | 33 | 0.00 / 0.00 | 0.00 / 0.00 | 0.74 / 0.88 | 0.00 / 0.00 | 0.78 / 0.94 |
| multipack_trade_unit | 31 | 0.19 / 0.10 | 0.08 / 0.03 | 0.48 / 0.52 | 0.25 / 0.03 | 0.67 / 0.26 |
| product_grouping | 30 | 0.50 / 0.03 | 0.00 / 0.00 | 0.75 / 0.10 | 0.29 / 0.07 | 0.57 / 0.53 |

### background

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| white | 1730 | 0.85 / 0.97 | 0.89 / 0.03 | 0.89 / 0.93 | 0.87 / 0.95 | 0.91 / 0.92 |
| color | 204 | 0.11 / 0.03 | 0.10 / 0.95 | 0.21 / 0.16 | 0.10 / 0.08 | 0.26 / 0.22 |
| scene_lifestyle | 144 | 0.66 / 0.17 | 0.65 / 0.49 | 0.73 / 0.59 | 0.61 / 0.12 | 0.73 / 0.76 |

### perspective

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| front_head_on | 614 | 0.41 / 0.98 | 0.39 / 0.99 | 0.68 / 0.63 | 0.41 / 0.99 | 0.62 / 0.88 |
| back_head_on | 211 | 0.50 / 0.01 | 0.50 / 0.02 | 0.74 / 0.22 | 0.51 / 0.25 | 0.80 / 0.43 |
| front_angle_left | 209 | 0.24 / 0.04 | 0.00 / 0.00 | 0.24 / 0.74 | 0.00 / 0.00 | 0.00 / 0.00 |
| front_angle_right | 199 | 0.23 / 0.13 | 0.16 / 0.04 | 0.29 / 0.03 | 0.00 / 0.00 | 0.31 / 0.66 |
| top_down | 120 | 0.67 / 0.02 | 0.25 / 0.01 | 0.53 / 0.08 | 0.21 / 0.06 | 0.41 / 0.27 |
| left_side | 82 | 0.00 / 0.00 | 0.25 / 0.05 | 0.16 / 0.20 | 0.00 / 0.00 | 0.33 / 0.06 |
| right_side | 65 | 0.00 / 0.00 | 0.37 / 0.11 | 0.16 / 0.60 | 0.00 / 0.00 | 0.31 / 0.57 |
| back_angle_right | 60 | 0.00 / 0.00 | 0.20 / 0.03 | 0.00 / 0.00 | 1.00 / 0.02 | 0.00 / 0.00 |
| bottom_up | 41 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 0.02 | 0.60 / 0.07 | 0.62 / 0.12 |
| back_angle_left | 37 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

### individuals

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1966 | 0.97 / 0.98 | 0.95 / 0.99 | 1.00 / 0.98 | 0.96 / 1.00 | 0.99 / 0.99 |
| individual_present | 114 | 0.55 / 0.45 | 0.57 / 0.11 | 0.72 / 0.96 | 0.81 / 0.22 | 0.87 / 0.75 |

### text_classification

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 1866 | 0.84 / 0.93 | 0.82 / 1.00 | 0.96 / 0.85 | 0.85 / 0.97 | 0.96 / 0.86 |
| nutritional_info | 127 | 1.00 / 0.09 | 0.84 / 0.24 | 0.51 / 0.98 | 0.74 / 0.75 | 0.87 / 0.76 |
| technical_specs | 103 | 0.01 / 0.01 | 0.00 / 0.00 | 0.90 / 0.51 | 0.00 / 0.00 | 0.93 / 0.69 |
| promotional_text | 86 | 0.09 / 0.08 | 0.00 / 0.00 | 0.30 / 0.78 | 0.00 / 0.00 | 0.32 / 0.71 |
| other | 83 | 0.00 / 0.00 | 0.00 / 0.00 | 0.36 / 0.12 | 0.00 / 0.00 | 0.15 / 0.41 |
| certification_seal | 30 | 0.00 / 0.00 | 0.00 / 0.00 | 0.21 / 0.63 | 0.00 / 0.00 | 0.86 / 0.40 |

### text_language

| classe | n | 8B v1 breve | 8B prompt condiviso | 8B v2 descrizione | 70B prompt condiviso | 70B v2 descrizione |
|---|---|---|---|---|---|---|
| none | 391 | 0.61 / 0.86 | 0.37 / 1.00 | 1.00 / 0.07 | 0.88 / 0.04 | 1.00 / 0.11 |
| English | 331 | 0.83 / 0.33 | 0.75 / 0.01 | 0.50 / 0.86 | 0.68 / 0.34 | 0.63 / 0.59 |
| German | 288 | 0.54 / 0.81 | 0.77 / 0.19 | 0.55 / 0.82 | 0.38 / 0.93 | 0.38 / 0.91 |
| French | 56 | 0.94 / 0.29 | 1.00 / 0.11 | 0.63 / 0.75 | 1.00 / 0.14 | 0.63 / 0.64 |
| Italian | 30 | 0.89 / 0.27 | 0.00 / 0.00 | 0.76 / 0.63 | 1.00 / 0.30 | 0.86 / 0.60 |
| other | 30 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 0.03 | 0.03 / 0.23 | 0.67 / 0.27 |
| Dutch | 20 | 1.00 / 0.05 | 0.00 / 0.00 | 0.42 / 0.70 | 0.75 / 0.15 | 0.67 / 0.60 |

## Limiti

- Tutte le varianti sono zero-shot: nessun modello è stato allenato.
- `coverage` aggiunge soprattutto classi rare: serve a confrontare i metodi, non a stimare il punteggio Kaggle.
- La v2 è stata scelta guardando prompt-dev. Il controllo pulito è la vista prompt-test; da ora prompt-test è stato
  usato una volta, con le varianti già fissate.
- In prompt-test e base alcune classi hanno pochi esempi (⚠ sotto 10).
- Nessuno di questi valori è un punteggio della classifica Kaggle.
