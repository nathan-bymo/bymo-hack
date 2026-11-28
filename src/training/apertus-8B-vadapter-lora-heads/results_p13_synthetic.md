# P13: modello finale + 1 epoca con immagini sintetiche (28 novembre 2026)

Elenco di tutte le prove: [../EXPERIMENTS.md](../EXPERIMENTS.md). Tabelle complete: [../benchmark/README.md](../benchmark/README.md).

## Scheda della prova

| voce | |
|---|---|
| **Domanda** | Continuare il modello finale con immagini sintetiche per lingue rare, sigilli e icone lo migliora? |
| **Partenza** | `final-abt/model/best` (SigLIP 2 + OCR + feature spaziali + attention pooling + LoRA + teste) |
| **Training** | `all_but_test` meno la validazione interna: 9.645 immagini reali + 939 sintetiche, 15,398 esempi, 1 epoca (963 passi, 22.9 min), batch 8×2, LR LoRA 5e-5, proiettore 2.5e-5, teste 2.5e-4, seed 3407 |
| **Sintetiche** | 1.040 generate da foto di training (`src/scripts/make_synthetic_text.py`); usate le 939 il cui prodotto è nel training |
| **Cosa cambia** | due cose insieme: un'epoca in più e le immagini sintetiche |
| **Esecuzione** | Modal, NVIDIA H200, job di circa 73 min; memoria di picco in training 77.2 GB |
| **Pesi** | repository privato del team, `synth-ft-1128-0243/model/best` |

## Punteggi

| set | `final-abt` | P13 | differenza |
|---|---|---|---|
| prompt-test (1.193) | 0.821 | 0.847 | +0.026 [IC 95% +0.004, +0.045] |
| prompt-test, weighted | 0.917 | 0.922 | +0.005 |
| **dev Kaggle pubblico** | **0.832** | **0.829** | −0.003 |

Su prompt-test il guadagno viene soprattutto dalle classi con meno di 10 esempi: senza di esse la differenza è circa +0.010.

Validazione interna di P13 (1.038 immagini): 0.824 a metà epoca, 0.838 a fine epoca.

## Per dimensione (prompt-test)

| dimensione | `final-abt` | P13 |
|---|---|---|
| subject | 0.846 | 0.845 |
| background | 0.881 | 0.870 |
| perspective | 0.755 | 0.783 |
| individuals | 0.959 | 0.965 |
| text_classification | 0.766 | 0.780 |
| text_language | 0.720 | 0.840 |

## Per classe (prompt-test)

### subject

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| full_product | 653 | 0.94 | 0.94 | +0.00 |
| infographic | 99 | 0.77 | 0.78 | +0.01 |
| detail_closeup | 95 | 0.83 | 0.85 | +0.02 |
| logo | 67 | 0.95 | 0.96 | +0.01 |
| delivery_scope | 54 | 0.84 | 0.87 | +0.03 |
| energy_label | 40 | 1.00 | 1.00 | +0.00 |
| packaging | 37 | 0.74 | 0.74 | +0.00 |
| package_and_product | 28 | 0.89 | 0.87 | -0.02 |
| multiview | 20 | 0.79 | 0.76 | -0.03 |
| accessories | 15 | 0.69 | 0.74 | +0.05 |
| placeholder | 13 | 0.96 | 0.96 | +0.00 |
| product_grouping | 10 | 0.95 | 0.90 | -0.05 |
| multipack_trade_unit | 8 | 0.67 | 0.62 | -0.04 |

### background

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| white | 819 | 0.98 | 0.98 | -0.00 |
| color | 103 | 0.81 | 0.79 | -0.01 |
| scene_lifestyle | 51 | 0.85 | 0.84 | -0.02 |

### perspective

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| front_head_on | 275 | 0.87 | 0.90 | +0.03 |
| back_head_on | 101 | 0.85 | 0.86 | +0.02 |
| front_angle_left | 86 | 0.64 | 0.71 | +0.07 |
| front_angle_right | 80 | 0.68 | 0.70 | +0.02 |
| top_down | 69 | 0.79 | 0.81 | +0.02 |
| left_side | 44 | 0.80 | 0.75 | -0.05 |
| right_side | 32 | 0.73 | 0.75 | +0.02 |
| back_angle_right | 26 | 0.65 | 0.73 | +0.08 |
| back_angle_left | 25 | 0.71 | 0.77 | +0.06 |
| bottom_up | 25 | 0.82 | 0.85 | +0.02 |

### individuals

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| none | 920 | 1.00 | 1.00 | +0.00 |
| individual_present | 54 | 0.92 | 0.93 | +0.01 |

### text_classification

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| none | 879 | 0.98 | 0.97 | -0.01 |
| nutritional_info | 62 | 0.96 | 0.95 | -0.01 |
| promotional_text | 45 | 0.84 | 0.82 | -0.02 |
| technical_specs | 45 | 0.86 | 0.86 | -0.00 |
| other | 36 | 0.55 | 0.58 | +0.02 |
| certification_seal | 6 | 0.40 | 0.50 | +0.10 |

### text_language

| classe | n | `final-abt` | P13 | differenza |
|---|---|---|---|---|
| none | 183 | 0.93 | 0.93 | +0.01 |
| English | 156 | 0.84 | 0.86 | +0.03 |
| German | 131 | 0.91 | 0.92 | +0.01 |
| French | 30 | 0.70 | 0.71 | +0.01 |
| other | 10 | 0.60 | 0.74 | +0.14 |
| Dutch | 4 | 0.40 | 0.86 | +0.46 |
| Italian | 4 | 0.67 | 0.86 | +0.19 |

## Velocità

Predizione dall'immagine al CSV su H200, OCR compreso, caricamento escluso: prompt-test 6,67 immagini/s; dev 7.48 immagini/s.
