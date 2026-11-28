# P12: LoRA + teste + feature spaziali + attention pooling, 512² (28 novembre 2026)

Elenco di tutte le prove: [../RISULTATI.md](../RISULTATI.md).

## Scheda della prova P12

| voce | |
|---|---|
| **Domanda** | Feature spaziali, attention pooling 2D e immagini a 512² migliorano LoRA + teste (P11)? |
| **Modello** | Apertus 1.5 8B, 32 layer, LoRA r32 + 6 teste + feature spaziali (sinistra−destra, alto−basso) + attention pooling 2D (8 query); 91,6 M parametri allenabili. Metodo di pooling di Nathan, senza SigLIP né OCR |
| **Training** | `training_rows` (9.344 immagini, 12.491 esempi per epoca), **2 epoche** (P11: 3), gradient checkpointing, seed 3407 |
| **Valutazione** | coverage (2.532), prompt-test (1.193), dev Kaggle |
| **Cosa cambia rispetto a P11** | tre cose insieme: risoluzione 512², pooling e feature spaziali, 2 epoche invece di 3. Gli effetti non si possono separare |
| **Esecuzione** | Hugging Face Jobs, H200: training 122 minuti, job completo 168 minuti, circa 14 $ |
| **Pesi** | repository privato del team, `archive/p12_lora_heads_pool_512_coverage0806` |

## Risultati

| set | P11 (256², 3 epoche) | P12 (512² + pooling, 2 epoche) | differenza [IC 95%] |
|---|---|---|---|
| coverage | 0.781 | 0.806 | +0.025 [+0.007, +0.044] |
| prompt-test | 0.791 | 0.791 | +0.000 [-0.022, +0.025] |
| **dev Kaggle pubblico** | 0.796 | **0.806** | +0.010 |

Riferimenti sul dev: modello di Nathan 0.832, Gemini 3.8 Flash 0.806, Gemini 4 Argon 0.834, Qwen 0.917.

## Per dimensione (coverage)

| dimensione | P11 | P12 |
|---|---|---|
| subject | 0.841 | 0.830 |
| background | 0.911 | 0.875 |
| perspective | 0.625 | 0.744 |
| individuals | 0.981 | 0.969 |
| text_classification | 0.806 | 0.805 |
| text_language | 0.521 | 0.611 |

## Per classe (coverage)

### subject

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| full_product | 1363 | 0.95 | 0.95 | -0.00 |
| infographic | 215 | 0.86 | 0.85 | -0.01 |
| detail_closeup | 158 | 0.86 | 0.84 | -0.02 |
| logo | 124 | 0.98 | 0.98 | +0.00 |
| delivery_scope | 109 | 0.83 | 0.85 | +0.02 |
| packaging | 88 | 0.88 | 0.88 | +0.00 |
| energy_label | 80 | 0.99 | 1.00 | +0.01 |
| package_and_product | 58 | 0.91 | 0.82 | -0.09 |
| multiview | 57 | 0.74 | 0.73 | -0.01 |
| accessories | 38 | 0.57 | 0.52 | -0.04 |
| placeholder | 33 | 0.95 | 0.97 | +0.01 |
| multipack_trade_unit | 31 | 0.82 | 0.77 | -0.05 |
| product_grouping | 30 | 0.59 | 0.62 | +0.04 |

### background

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| white | 1730 | 0.98 | 0.98 | -0.00 |
| color | 204 | 0.83 | 0.76 | -0.06 |
| scene_lifestyle | 144 | 0.92 | 0.88 | -0.04 |

### perspective

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| front_head_on | 614 | 0.88 | 0.89 | +0.01 |
| back_head_on | 211 | 0.87 | 0.88 | +0.01 |
| front_angle_left | 209 | 0.59 | 0.75 | +0.16 |
| front_angle_right | 199 | 0.52 | 0.71 | +0.19 |
| top_down | 120 | 0.76 | 0.72 | -0.03 |
| left_side | 82 | 0.54 | 0.74 | +0.20 |
| right_side | 65 | 0.47 | 0.68 | +0.21 |
| back_angle_right | 60 | 0.51 | 0.67 | +0.15 |
| bottom_up | 41 | 0.82 | 0.80 | -0.02 |
| back_angle_left | 37 | 0.28 | 0.61 | +0.33 |

### individuals

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| none | 1966 | 1.00 | 1.00 | -0.00 |
| individual_present | 114 | 0.96 | 0.94 | -0.02 |

### text_classification

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| none | 1866 | 0.98 | 0.98 | +0.00 |
| nutritional_info | 127 | 0.94 | 0.94 | -0.00 |
| technical_specs | 103 | 0.88 | 0.89 | +0.02 |
| promotional_text | 86 | 0.89 | 0.88 | -0.01 |
| other | 83 | 0.55 | 0.56 | +0.01 |
| certification_seal | 30 | 0.60 | 0.58 | -0.03 |

### text_language

| classe | n | P11 | P12 | differenza |
|---|---|---|---|---|
| none | 391 | 0.90 | 0.92 | +0.01 |
| English | 331 | 0.81 | 0.82 | +0.02 |
| German | 288 | 0.82 | 0.86 | +0.04 |
| French | 56 | 0.67 | 0.75 | +0.09 |
| Italian | 30 | 0.28 | 0.46 | +0.18 |
| other | 30 | 0.17 | 0.47 | +0.30 |
| Dutch | 20 | 0.00 | 0.00 | +0.00 |

## Cosa si vede

- **Destra e sinistra:** gli scambi di lato in `perspective` scendono da 200 a 85 (errori di vista da 440 a 318); tutte le viste laterali e angolate migliorano di 0,15–0,33.
- **Lingua:** 0.521 → 0.611 su coverage (Italian 0.28 → 0.46, other 0.17 → 0.47); `Dutch` resta 0 (2 celle umane nel training).
- **Peggiorano** `subject` (soprattutto `package_and_product`) e `background` (`color`). Ipotesi non verificata: due epoche invece di tre.
- Il guadagno sta soprattutto nelle classi rare: +0.025 su coverage, 0 su prompt-test, +0.010 sul dev.

## Velocità

| | P11 | P12 |
|---|---|---|
| predizione, immagini/s (H200) | 13,2 | 3,9 |
| training, esempi/s | 13,4 | 3,4 (con gradient checkpointing) |
| memoria di picco in training | 139 GB | 45 GB |
| memoria di picco in predizione (batch 32, coverage) | 22,0 GB | 32,5 GB |
