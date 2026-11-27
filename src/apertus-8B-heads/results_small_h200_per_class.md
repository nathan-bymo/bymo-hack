# Teste su H200, split piccolo: dettaglio per classe di tutte le varianti

Integra [results_small_h200.md](results_small_h200.md). Valutazione sulle 450 immagini bilanciate. Solo conteggi aggregati.

- **Esempi visti:** esempi valutati della classe nelle 1.199 immagini con cui si allenano i pesi delle teste.
- **Trovate:** esempi veri riconosciuti (recall). **Scelte giuste:** risposte corrette tra quelle con quella classe (precision).

## cross-entropy (macro F1 0.542)

### subject: macro F1 0.466, 402 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| full_product | 668 | 67 su 85 | 67 su 194 | 0.35 | 0.79 | 0.48 | detail_closeup (13), packaging (2), infographic (2) |
| packaging | 27 | 17 su 42 | 17 su 33 | 0.52 | 0.40 | 0.45 | full_product (14), package_and_product (6), infographic (3) |
| package_and_product | 18 | 4 su 38 | 4 su 13 | 0.31 | 0.11 | 0.16 | full_product (22), packaging (8), detail_closeup (3) |
| multiview | 13 | 8 su 36 | 8 su 8 | 1.00 | 0.22 | 0.36 | full_product (23), detail_closeup (2), delivery_scope (2) |
| detail_closeup | 93 | 7 su 30 | 7 su 33 | 0.21 | 0.23 | 0.22 | full_product (11), infographic (10), delivery_scope (2) |
| accessories | 18 | 1 su 29 | 1 su 1 | 1.00 | 0.03 | 0.07 | full_product (25), delivery_scope (2), packaging (1) |
| multipack_trade_unit | 10 | 4 su 28 | 4 su 5 | 0.80 | 0.14 | 0.24 | full_product (17), detail_closeup (3), packaging (2) |
| delivery_scope | 46 | 16 su 28 | 16 su 25 | 0.64 | 0.57 | 0.60 | full_product (6), infographic (5), detail_closeup (1) |
| infographic | 116 | 14 su 24 | 14 su 47 | 0.30 | 0.58 | 0.39 | full_product (6), packaging (2), logo (1) |
| product_grouping | 6 | 2 su 20 | 2 su 2 | 1.00 | 0.10 | 0.18 | infographic (9), full_product (3), delivery_scope (2) |
| energy_label | 42 | 14 su 14 | 14 su 14 | 1.00 | 1.00 | 1.00 | - |
| logo | 55 | 14 su 14 | 14 su 15 | 0.93 | 1.00 | 0.97 | - |
| placeholder | 20 | 12 su 14 | 12 su 12 | 1.00 | 0.86 | 0.92 | infographic (2) |

### background: macro F1 0.817, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| white | 806 | 204 su 209 | 204 su 249 | 0.82 | 0.98 | 0.89 | color (3), scene_lifestyle (2) |
| color | 92 | 74 su 127 | 74 su 78 | 0.95 | 0.58 | 0.72 | white (42), scene_lifestyle (11) |
| scene_lifestyle | 67 | 44 su 48 | 44 su 57 | 0.77 | 0.92 | 0.84 | white (3), color (1) |

### perspective: macro F1 0.230, 258 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| front_head_on | 281 | 31 su 43 | 31 su 69 | 0.45 | 0.72 | 0.55 | front_angle_right (8), front_angle_left (3), back_head_on (1) |
| back_head_on | 82 | 28 su 33 | 28 su 55 | 0.51 | 0.85 | 0.64 | front_head_on (4), front_angle_right (1) |
| front_angle_left | 102 | 5 su 30 | 5 su 54 | 0.09 | 0.17 | 0.12 | front_angle_right (18), front_head_on (6), left_side (1) |
| front_angle_right | 102 | 16 su 30 | 16 su 54 | 0.30 | 0.53 | 0.38 | front_angle_left (9), front_head_on (5) |
| top_down | 64 | 6 su 28 | 6 su 14 | 0.43 | 0.21 | 0.29 | front_head_on (14), front_angle_left (4), front_angle_right (4) |
| back_angle_left | 23 | 0 su 21 | 0 su 0 | 0.00 | 0.00 | 0.00 | front_angle_left (8), back_head_on (8), front_angle_right (2) |
| back_angle_right | 21 | 0 su 21 | 0 su 0 | 0.00 | 0.00 | 0.00 | front_angle_left (8), back_head_on (7), front_angle_right (5) |
| left_side | 39 | 3 su 19 | 3 su 7 | 0.43 | 0.16 | 0.23 | front_angle_left (10), front_head_on (4), top_down (1) |
| right_side | 27 | 1 su 17 | 1 su 5 | 0.20 | 0.06 | 0.09 | front_angle_left (7), front_head_on (3), left_side (3) |
| bottom_up | 12 | 0 su 16 | 0 su 0 | 0.00 | 0.00 | 0.00 | back_head_on (8), top_down (6), right_side (2) |

### individuals: macro F1 0.881, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 922 | 281 su 281 | 281 su 313 | 0.90 | 1.00 | 0.95 | - |
| individual_present | 44 | 71 su 103 | 71 su 71 | 1.00 | 0.69 | 0.82 | none (32) |

### text_classification: macro F1 0.493, 408 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 935 | 168 su 170 | 168 su 274 | 0.61 | 0.99 | 0.76 | nutritional_info (2) |
| other | 20 | 5 su 67 | 5 su 19 | 0.26 | 0.07 | 0.12 | none (55), promotional_text (4), technical_specs (3) |
| promotional_text | 26 | 48 su 65 | 48 su 55 | 0.87 | 0.74 | 0.80 | none (11), other (5), nutritional_info (1) |
| nutritional_info | 55 | 42 su 53 | 42 su 45 | 0.93 | 0.79 | 0.86 | none (11) |
| technical_specs | 43 | 12 su 41 | 12 su 15 | 0.80 | 0.29 | 0.43 | none (18), other (9), promotional_text (2) |
| certification_seal | 3 | 0 su 12 | 0 su 0 | 0.00 | 0.00 | 0.00 | none (11), promotional_text (1) |

### text_language: macro F1 0.363, 312 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| English | 136 | 77 su 85 | 77 su 162 | 0.48 | 0.91 | 0.62 | none (6), German (1), other (1) |
| German | 114 | 52 su 76 | 52 su 90 | 0.58 | 0.68 | 0.63 | English (23), none (1) |
| none | 176 | 42 su 65 | 42 su 51 | 0.82 | 0.65 | 0.72 | English (23) |
| French | 14 | 3 su 44 | 3 su 5 | 0.60 | 0.07 | 0.12 | German (21), English (20) |
| other | 7 | 0 su 35 | 0 su 1 | 0.00 | 0.00 | 0.00 | English (17), German (13), French (2) |
| Italian | 4 | 2 su 7 | 2 su 2 | 1.00 | 0.29 | 0.44 | German (3), English (2) |
| Dutch | 1 | 0 su 0 | 0 su 1 | 0.00 | 0.00 | 0.00 | - |

### Matrici di confusione: cross-entropy

Righe: label. Colonne: risposta delle teste.

#### subject

Colonne: c1 = full_product, c2 = detail_closeup, c3 = accessories, c4 = packaging, c5 = package_and_product, c6 = multipack_trade_unit, c7 = delivery_scope, c8 = product_grouping, c9 = multiview, c10 = energy_label, c11 = infographic, c12 = logo, c13 = placeholder

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_product | 67 | 13 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| detail_closeup | 11 | 7 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 10 | 0 | 0 |
| accessories | 25 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| packaging | 14 | 2 | 0 | 17 | 6 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| package_and_product | 22 | 3 | 0 | 8 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| multipack_trade_unit | 17 | 3 | 0 | 2 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| delivery_scope | 6 | 1 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 0 | 5 | 0 | 0 |
| product_grouping | 3 | 1 | 0 | 1 | 1 | 1 | 2 | 2 | 0 | 0 | 9 | 0 | 0 |
| multiview | 23 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 8 | 0 | 1 | 0 | 0 |
| energy_label | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| infographic | 6 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 1 | 0 |
| logo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| placeholder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 12 |

#### background

Colonne: c1 = white, c2 = color, c3 = scene_lifestyle

| label | c1 | c2 | c3 |
|---|---|---|---|
| white | 204 | 3 | 2 |
| color | 42 | 74 | 11 |
| scene_lifestyle | 3 | 1 | 44 |

#### perspective

Colonne: c1 = front_head_on, c2 = front_angle_left, c3 = front_angle_right, c4 = back_head_on, c5 = back_angle_left, c6 = back_angle_right, c7 = left_side, c8 = right_side, c9 = top_down, c10 = bottom_up

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 31 | 3 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| front_angle_left | 6 | 5 | 18 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| front_angle_right | 5 | 9 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_head_on | 4 | 0 | 1 | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_angle_left | 1 | 8 | 2 | 8 | 0 | 0 | 0 | 1 | 1 | 0 |
| back_angle_right | 1 | 8 | 5 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| left_side | 4 | 10 | 0 | 0 | 0 | 0 | 3 | 1 | 1 | 0 |
| right_side | 3 | 7 | 0 | 3 | 0 | 0 | 3 | 1 | 0 | 0 |
| top_down | 14 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| bottom_up | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 2 | 6 | 0 |

#### individuals

Colonne: c1 = individual_present, c2 = none

| label | c1 | c2 |
|---|---|---|
| individual_present | 71 | 32 |
| none | 0 | 281 |

#### text_classification

Colonne: c1 = certification_seal, c2 = technical_specs, c3 = promotional_text, c4 = nutritional_info, c5 = other, c6 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| certification_seal | 0 | 0 | 1 | 0 | 0 | 11 |
| technical_specs | 0 | 12 | 2 | 0 | 9 | 18 |
| promotional_text | 0 | 0 | 48 | 1 | 5 | 11 |
| nutritional_info | 0 | 0 | 0 | 42 | 0 | 11 |
| other | 0 | 3 | 4 | 0 | 5 | 55 |
| none | 0 | 0 | 0 | 2 | 0 | 168 |

#### text_language

Colonne: c1 = German, c2 = English, c3 = French, c4 = Italian, c5 = Dutch, c6 = other, c7 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| German | 52 | 23 | 0 | 0 | 0 | 0 | 1 |
| English | 1 | 77 | 0 | 0 | 0 | 1 | 6 |
| French | 21 | 20 | 3 | 0 | 0 | 0 | 0 |
| Italian | 3 | 2 | 0 | 2 | 0 | 0 | 0 |
| other | 13 | 17 | 2 | 0 | 1 | 0 | 2 |
| none | 0 | 23 | 0 | 0 | 0 | 0 | 42 |

## focal (macro F1 0.528)

### subject: macro F1 0.448, 402 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| full_product | 668 | 62 su 85 | 62 su 193 | 0.32 | 0.73 | 0.45 | detail_closeup (19), packaging (2), accessories (1) |
| packaging | 27 | 17 su 42 | 17 su 36 | 0.47 | 0.40 | 0.44 | full_product (15), package_and_product (6), infographic (4) |
| package_and_product | 18 | 3 su 38 | 3 su 12 | 0.25 | 0.08 | 0.12 | full_product (23), packaging (8), infographic (2) |
| multiview | 13 | 9 su 36 | 9 su 11 | 0.82 | 0.25 | 0.38 | full_product (23), infographic (2), detail_closeup (1) |
| detail_closeup | 93 | 7 su 30 | 7 su 35 | 0.20 | 0.23 | 0.22 | full_product (12), infographic (9), delivery_scope (1) |
| accessories | 18 | 2 su 29 | 2 su 4 | 0.50 | 0.07 | 0.12 | full_product (23), detail_closeup (2), delivery_scope (1) |
| multipack_trade_unit | 10 | 4 su 28 | 4 su 4 | 1.00 | 0.14 | 0.25 | full_product (16), detail_closeup (2), package_and_product (2) |
| delivery_scope | 46 | 13 su 28 | 13 su 18 | 0.72 | 0.46 | 0.57 | full_product (9), infographic (5), detail_closeup (1) |
| infographic | 116 | 15 su 24 | 15 su 50 | 0.30 | 0.62 | 0.41 | full_product (7), detail_closeup (1), packaging (1) |
| product_grouping | 6 | 0 su 20 | 0 su 0 | 0.00 | 0.00 | 0.00 | infographic (8), packaging (4), full_product (3) |
| energy_label | 42 | 14 su 14 | 14 su 14 | 1.00 | 1.00 | 1.00 | - |
| logo | 55 | 13 su 14 | 13 su 13 | 1.00 | 0.93 | 0.96 | infographic (1) |
| placeholder | 20 | 12 su 14 | 12 su 12 | 1.00 | 0.86 | 0.92 | infographic (2) |

### background: macro F1 0.764, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| white | 806 | 205 su 209 | 205 su 261 | 0.79 | 0.98 | 0.87 | scene_lifestyle (2), color (2) |
| color | 92 | 62 su 127 | 62 su 65 | 0.95 | 0.49 | 0.65 | white (50), scene_lifestyle (15) |
| scene_lifestyle | 67 | 41 su 48 | 41 su 58 | 0.71 | 0.85 | 0.77 | white (6), color (1) |

### perspective: macro F1 0.267, 258 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| front_head_on | 281 | 30 su 43 | 30 su 77 | 0.39 | 0.70 | 0.50 | front_angle_right (7), back_head_on (3), front_angle_left (2) |
| back_head_on | 82 | 31 su 33 | 31 su 56 | 0.55 | 0.94 | 0.70 | front_angle_right (1), front_head_on (1) |
| front_angle_left | 102 | 11 su 30 | 11 su 57 | 0.19 | 0.37 | 0.25 | front_head_on (10), front_angle_right (9) |
| front_angle_right | 102 | 12 su 30 | 12 su 36 | 0.33 | 0.40 | 0.36 | front_angle_left (11), front_head_on (7) |
| top_down | 64 | 9 su 28 | 9 su 19 | 0.47 | 0.32 | 0.38 | front_head_on (14), front_angle_right (4), left_side (1) |
| back_angle_left | 23 | 1 su 21 | 1 su 2 | 0.50 | 0.05 | 0.09 | front_angle_left (9), back_head_on (5), front_head_on (4) |
| back_angle_right | 21 | 0 su 21 | 0 su 0 | 0.00 | 0.00 | 0.00 | front_angle_left (11), back_head_on (6), front_angle_right (2) |
| left_side | 39 | 4 su 19 | 4 su 10 | 0.40 | 0.21 | 0.28 | front_angle_left (7), front_head_on (5), top_down (2) |
| right_side | 27 | 1 su 17 | 1 su 1 | 1.00 | 0.06 | 0.11 | front_angle_left (6), left_side (5), back_head_on (3) |
| bottom_up | 12 | 0 su 16 | 0 su 0 | 0.00 | 0.00 | 0.00 | back_head_on (8), top_down (5), front_head_on (3) |

### individuals: macro F1 0.885, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 922 | 281 su 281 | 281 su 312 | 0.90 | 1.00 | 0.95 | - |
| individual_present | 44 | 72 su 103 | 72 su 72 | 1.00 | 0.70 | 0.82 | none (31) |

### text_classification: macro F1 0.499, 408 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 935 | 169 su 170 | 169 su 271 | 0.62 | 0.99 | 0.77 | nutritional_info (1) |
| other | 20 | 9 su 67 | 9 su 24 | 0.38 | 0.13 | 0.20 | none (51), promotional_text (6), technical_specs (1) |
| promotional_text | 26 | 47 su 65 | 47 su 59 | 0.80 | 0.72 | 0.76 | none (11), other (6), nutritional_info (1) |
| nutritional_info | 55 | 39 su 53 | 39 su 41 | 0.95 | 0.74 | 0.83 | none (14) |
| technical_specs | 43 | 12 su 41 | 12 su 13 | 0.92 | 0.29 | 0.44 | none (14), other (9), promotional_text (6) |
| certification_seal | 3 | 0 su 12 | 0 su 0 | 0.00 | 0.00 | 0.00 | none (12) |

### text_language: macro F1 0.306, 312 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| English | 136 | 69 su 85 | 69 su 149 | 0.46 | 0.81 | 0.59 | none (11), German (5) |
| German | 114 | 51 su 76 | 51 su 98 | 0.52 | 0.67 | 0.59 | English (20), none (5) |
| none | 176 | 42 su 65 | 42 su 60 | 0.70 | 0.65 | 0.67 | English (23) |
| French | 14 | 1 su 44 | 1 su 3 | 0.33 | 0.02 | 0.04 | German (27), English (16) |
| other | 7 | 0 su 35 | 0 su 0 | 0.00 | 0.00 | 0.00 | English (18), German (12), French (2) |
| Italian | 4 | 1 su 7 | 1 su 1 | 1.00 | 0.14 | 0.25 | English (3), German (3) |
| Dutch | 1 | 0 su 0 | 0 su 1 | 0.00 | 0.00 | 0.00 | - |

### Matrici di confusione: focal

Righe: label. Colonne: risposta delle teste.

#### subject

Colonne: c1 = full_product, c2 = detail_closeup, c3 = accessories, c4 = packaging, c5 = package_and_product, c6 = multipack_trade_unit, c7 = delivery_scope, c8 = product_grouping, c9 = multiview, c10 = energy_label, c11 = infographic, c12 = logo, c13 = placeholder

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_product | 62 | 19 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| detail_closeup | 12 | 7 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 9 | 0 | 0 |
| accessories | 23 | 2 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| packaging | 15 | 0 | 0 | 17 | 6 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| package_and_product | 23 | 1 | 0 | 8 | 3 | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 |
| multipack_trade_unit | 16 | 2 | 0 | 2 | 2 | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| delivery_scope | 9 | 1 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 5 | 0 | 0 |
| product_grouping | 3 | 1 | 1 | 4 | 0 | 0 | 2 | 0 | 1 | 0 | 8 | 0 | 0 |
| multiview | 23 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 9 | 0 | 2 | 0 | 0 |
| energy_label | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| infographic | 7 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| logo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 | 0 |
| placeholder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 12 |

#### background

Colonne: c1 = white, c2 = color, c3 = scene_lifestyle

| label | c1 | c2 | c3 |
|---|---|---|---|
| white | 205 | 2 | 2 |
| color | 50 | 62 | 15 |
| scene_lifestyle | 6 | 1 | 41 |

#### perspective

Colonne: c1 = front_head_on, c2 = front_angle_left, c3 = front_angle_right, c4 = back_head_on, c5 = back_angle_left, c6 = back_angle_right, c7 = left_side, c8 = right_side, c9 = top_down, c10 = bottom_up

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 30 | 2 | 7 | 3 | 0 | 0 | 0 | 0 | 1 | 0 |
| front_angle_left | 10 | 11 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| front_angle_right | 7 | 11 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_head_on | 1 | 0 | 1 | 31 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_angle_left | 4 | 9 | 0 | 5 | 1 | 0 | 0 | 0 | 2 | 0 |
| back_angle_right | 1 | 11 | 2 | 6 | 1 | 0 | 0 | 0 | 0 | 0 |
| left_side | 5 | 7 | 1 | 0 | 0 | 0 | 4 | 0 | 2 | 0 |
| right_side | 2 | 6 | 0 | 3 | 0 | 0 | 5 | 1 | 0 | 0 |
| top_down | 14 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 9 | 0 |
| bottom_up | 3 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 5 | 0 |

#### individuals

Colonne: c1 = individual_present, c2 = none

| label | c1 | c2 |
|---|---|---|
| individual_present | 72 | 31 |
| none | 0 | 281 |

#### text_classification

Colonne: c1 = certification_seal, c2 = technical_specs, c3 = promotional_text, c4 = nutritional_info, c5 = other, c6 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| certification_seal | 0 | 0 | 0 | 0 | 0 | 12 |
| technical_specs | 0 | 12 | 6 | 0 | 9 | 14 |
| promotional_text | 0 | 0 | 47 | 1 | 6 | 11 |
| nutritional_info | 0 | 0 | 0 | 39 | 0 | 14 |
| other | 0 | 1 | 6 | 0 | 9 | 51 |
| none | 0 | 0 | 0 | 1 | 0 | 169 |

#### text_language

Colonne: c1 = German, c2 = English, c3 = French, c4 = Italian, c5 = Dutch, c6 = other, c7 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| German | 51 | 20 | 0 | 0 | 0 | 0 | 5 |
| English | 5 | 69 | 0 | 0 | 0 | 0 | 11 |
| French | 27 | 16 | 1 | 0 | 0 | 0 | 0 |
| Italian | 3 | 3 | 0 | 1 | 0 | 0 | 0 |
| other | 12 | 18 | 2 | 0 | 1 | 0 | 2 |
| none | 0 | 23 | 0 | 0 | 0 | 0 | 42 |

## cross-entropy + GradNorm (macro F1 0.537)

### subject: macro F1 0.462, 402 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| full_product | 668 | 69 su 85 | 69 su 192 | 0.36 | 0.81 | 0.50 | detail_closeup (11), packaging (2), infographic (2) |
| packaging | 27 | 16 su 42 | 16 su 28 | 0.57 | 0.38 | 0.46 | full_product (15), package_and_product (7), infographic (2) |
| package_and_product | 18 | 5 su 38 | 5 su 14 | 0.36 | 0.13 | 0.19 | full_product (21), packaging (6), infographic (3) |
| multiview | 13 | 8 su 36 | 8 su 9 | 0.89 | 0.22 | 0.36 | full_product (20), infographic (4), detail_closeup (2) |
| detail_closeup | 93 | 7 su 30 | 7 su 27 | 0.26 | 0.23 | 0.25 | infographic (13), full_product (9), delivery_scope (1) |
| accessories | 18 | 0 su 29 | 0 su 0 | 0.00 | 0.00 | 0.00 | full_product (26), delivery_scope (1), detail_closeup (1) |
| multipack_trade_unit | 10 | 4 su 28 | 4 su 6 | 0.67 | 0.14 | 0.24 | full_product (17), detail_closeup (2), package_and_product (2) |
| delivery_scope | 46 | 13 su 28 | 13 su 21 | 0.62 | 0.46 | 0.53 | infographic (9), full_product (6) |
| infographic | 116 | 17 su 24 | 17 su 63 | 0.27 | 0.71 | 0.39 | full_product (6), detail_closeup (1) |
| product_grouping | 6 | 2 su 20 | 2 su 2 | 1.00 | 0.10 | 0.18 | infographic (10), delivery_scope (3), full_product (3) |
| energy_label | 42 | 14 su 14 | 14 su 14 | 1.00 | 1.00 | 1.00 | - |
| logo | 55 | 14 su 14 | 14 su 14 | 1.00 | 1.00 | 1.00 | - |
| placeholder | 20 | 12 su 14 | 12 su 12 | 1.00 | 0.86 | 0.92 | infographic (2) |

### background: macro F1 0.756, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| white | 806 | 205 su 209 | 205 su 270 | 0.76 | 0.98 | 0.86 | scene_lifestyle (2), color (2) |
| color | 92 | 57 su 127 | 57 su 60 | 0.95 | 0.45 | 0.61 | white (59), scene_lifestyle (11) |
| scene_lifestyle | 67 | 41 su 48 | 41 su 54 | 0.76 | 0.85 | 0.80 | white (6), color (1) |

### perspective: macro F1 0.228, 258 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| front_head_on | 281 | 29 su 43 | 29 su 71 | 0.41 | 0.67 | 0.51 | front_angle_right (10), front_angle_left (3), back_head_on (1) |
| back_head_on | 82 | 28 su 33 | 28 su 55 | 0.51 | 0.85 | 0.64 | front_head_on (4), front_angle_right (1) |
| front_angle_left | 102 | 7 su 30 | 7 su 55 | 0.13 | 0.23 | 0.16 | front_angle_right (16), front_head_on (7) |
| front_angle_right | 102 | 12 su 30 | 12 su 50 | 0.24 | 0.40 | 0.30 | front_angle_left (10), front_head_on (7), right_side (1) |
| top_down | 64 | 6 su 28 | 6 su 12 | 0.50 | 0.21 | 0.30 | front_head_on (14), front_angle_left (4), front_angle_right (4) |
| back_angle_left | 23 | 0 su 21 | 0 su 0 | 0.00 | 0.00 | 0.00 | front_angle_left (8), back_head_on (8), front_angle_right (2) |
| back_angle_right | 21 | 0 su 21 | 0 su 0 | 0.00 | 0.00 | 0.00 | front_angle_left (9), back_head_on (7), front_angle_right (4) |
| left_side | 39 | 4 su 19 | 4 su 9 | 0.44 | 0.21 | 0.29 | front_angle_left (9), front_head_on (4), right_side (2) |
| right_side | 27 | 1 su 17 | 1 su 6 | 0.17 | 0.06 | 0.09 | front_angle_left (5), left_side (5), back_head_on (3) |
| bottom_up | 12 | 0 su 16 | 0 su 0 | 0.00 | 0.00 | 0.00 | back_head_on (8), top_down (5), front_head_on (2) |

### individuals: macro F1 0.902, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 922 | 281 su 281 | 281 su 308 | 0.91 | 1.00 | 0.95 | - |
| individual_present | 44 | 76 su 103 | 76 su 76 | 1.00 | 0.74 | 0.85 | none (27) |

### text_classification: macro F1 0.506, 408 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 935 | 168 su 170 | 168 su 274 | 0.61 | 0.99 | 0.76 | nutritional_info (2) |
| other | 20 | 5 su 67 | 5 su 11 | 0.45 | 0.07 | 0.13 | none (56), promotional_text (4), technical_specs (2) |
| promotional_text | 26 | 51 su 65 | 51 su 61 | 0.84 | 0.78 | 0.81 | none (12), other (1), nutritional_info (1) |
| nutritional_info | 55 | 44 su 53 | 44 su 47 | 0.94 | 0.83 | 0.88 | none (9) |
| technical_specs | 43 | 13 su 41 | 13 su 15 | 0.87 | 0.32 | 0.46 | none (17), promotional_text (6), other (5) |
| certification_seal | 3 | 0 su 12 | 0 su 0 | 0.00 | 0.00 | 0.00 | none (12) |

### text_language: macro F1 0.367, 312 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| English | 136 | 75 su 85 | 75 su 165 | 0.45 | 0.88 | 0.60 | none (7), German (2), other (1) |
| German | 114 | 50 su 76 | 50 su 83 | 0.60 | 0.66 | 0.63 | English (26) |
| none | 176 | 41 su 65 | 41 su 50 | 0.82 | 0.63 | 0.71 | English (23), German (1) |
| French | 14 | 5 su 44 | 5 su 10 | 0.50 | 0.11 | 0.19 | English (21), German (18) |
| other | 7 | 0 su 35 | 0 su 1 | 0.00 | 0.00 | 0.00 | English (19), German (11), French (2) |
| Italian | 4 | 2 su 7 | 2 su 2 | 1.00 | 0.29 | 0.44 | French (3), English (1), German (1) |
| Dutch | 1 | 0 su 0 | 0 su 1 | 0.00 | 0.00 | 0.00 | - |

### Matrici di confusione: cross-entropy + GradNorm

Righe: label. Colonne: risposta delle teste.

#### subject

Colonne: c1 = full_product, c2 = detail_closeup, c3 = accessories, c4 = packaging, c5 = package_and_product, c6 = multipack_trade_unit, c7 = delivery_scope, c8 = product_grouping, c9 = multiview, c10 = energy_label, c11 = infographic, c12 = logo, c13 = placeholder

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_product | 69 | 11 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| detail_closeup | 9 | 7 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 13 | 0 | 0 |
| accessories | 26 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| packaging | 15 | 1 | 0 | 16 | 7 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| package_and_product | 21 | 2 | 0 | 6 | 5 | 0 | 0 | 0 | 1 | 0 | 3 | 0 | 0 |
| multipack_trade_unit | 17 | 2 | 0 | 2 | 2 | 4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| delivery_scope | 6 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 0 | 9 | 0 | 0 |
| product_grouping | 3 | 0 | 0 | 1 | 0 | 1 | 3 | 2 | 0 | 0 | 10 | 0 | 0 |
| multiview | 20 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 8 | 0 | 4 | 0 | 0 |
| energy_label | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| infographic | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 0 | 0 |
| logo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| placeholder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 12 |

#### background

Colonne: c1 = white, c2 = color, c3 = scene_lifestyle

| label | c1 | c2 | c3 |
|---|---|---|---|
| white | 205 | 2 | 2 |
| color | 59 | 57 | 11 |
| scene_lifestyle | 6 | 1 | 41 |

#### perspective

Colonne: c1 = front_head_on, c2 = front_angle_left, c3 = front_angle_right, c4 = back_head_on, c5 = back_angle_left, c6 = back_angle_right, c7 = left_side, c8 = right_side, c9 = top_down, c10 = bottom_up

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 29 | 3 | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| front_angle_left | 7 | 7 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| front_angle_right | 7 | 10 | 12 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| back_head_on | 4 | 0 | 1 | 28 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_angle_left | 1 | 8 | 2 | 8 | 0 | 0 | 0 | 1 | 1 | 0 |
| back_angle_right | 1 | 9 | 4 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| left_side | 4 | 9 | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 |
| right_side | 2 | 5 | 1 | 3 | 0 | 0 | 5 | 1 | 0 | 0 |
| top_down | 14 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| bottom_up | 2 | 0 | 0 | 8 | 0 | 0 | 0 | 1 | 5 | 0 |

#### individuals

Colonne: c1 = individual_present, c2 = none

| label | c1 | c2 |
|---|---|---|
| individual_present | 76 | 27 |
| none | 0 | 281 |

#### text_classification

Colonne: c1 = certification_seal, c2 = technical_specs, c3 = promotional_text, c4 = nutritional_info, c5 = other, c6 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| certification_seal | 0 | 0 | 0 | 0 | 0 | 12 |
| technical_specs | 0 | 13 | 6 | 0 | 5 | 17 |
| promotional_text | 0 | 0 | 51 | 1 | 1 | 12 |
| nutritional_info | 0 | 0 | 0 | 44 | 0 | 9 |
| other | 0 | 2 | 4 | 0 | 5 | 56 |
| none | 0 | 0 | 0 | 2 | 0 | 168 |

#### text_language

Colonne: c1 = German, c2 = English, c3 = French, c4 = Italian, c5 = Dutch, c6 = other, c7 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| German | 50 | 26 | 0 | 0 | 0 | 0 | 0 |
| English | 2 | 75 | 0 | 0 | 0 | 1 | 7 |
| French | 18 | 21 | 5 | 0 | 0 | 0 | 0 |
| Italian | 1 | 1 | 3 | 2 | 0 | 0 | 0 |
| other | 11 | 19 | 2 | 0 | 1 | 0 | 2 |
| none | 1 | 23 | 0 | 0 | 0 | 0 | 41 |

## focal + GradNorm (macro F1 0.501)

### subject: macro F1 0.441, 402 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| full_product | 668 | 71 su 85 | 71 su 193 | 0.37 | 0.84 | 0.51 | detail_closeup (7), packaging (4), infographic (2) |
| packaging | 27 | 21 su 42 | 21 su 49 | 0.43 | 0.50 | 0.46 | full_product (12), package_and_product (5), infographic (3) |
| package_and_product | 18 | 2 su 38 | 2 su 8 | 0.25 | 0.05 | 0.09 | full_product (16), packaging (15), detail_closeup (3) |
| multiview | 13 | 8 su 36 | 8 su 8 | 1.00 | 0.22 | 0.36 | full_product (22), infographic (3), detail_closeup (3) |
| detail_closeup | 93 | 6 su 30 | 6 su 24 | 0.25 | 0.20 | 0.22 | full_product (13), infographic (10), delivery_scope (1) |
| accessories | 18 | 1 su 29 | 1 su 2 | 0.50 | 0.03 | 0.06 | full_product (26), delivery_scope (1), detail_closeup (1) |
| multipack_trade_unit | 10 | 3 su 28 | 3 su 4 | 0.75 | 0.11 | 0.19 | full_product (16), packaging (6), delivery_scope (1) |
| delivery_scope | 46 | 11 su 28 | 11 su 17 | 0.65 | 0.39 | 0.49 | full_product (9), infographic (7), detail_closeup (1) |
| infographic | 116 | 19 su 24 | 19 su 58 | 0.33 | 0.79 | 0.46 | full_product (3), packaging (1), detail_closeup (1) |
| product_grouping | 6 | 0 su 20 | 0 su 0 | 0.00 | 0.00 | 0.00 | infographic (9), full_product (5), packaging (2) |
| energy_label | 42 | 14 su 14 | 14 su 14 | 1.00 | 1.00 | 1.00 | - |
| logo | 55 | 13 su 14 | 13 su 13 | 1.00 | 0.93 | 0.96 | infographic (1) |
| placeholder | 20 | 12 su 14 | 12 su 12 | 1.00 | 0.86 | 0.92 | infographic (2) |

### background: macro F1 0.642, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| white | 806 | 206 su 209 | 206 su 297 | 0.69 | 0.99 | 0.81 | color (2), scene_lifestyle (1) |
| color | 92 | 37 su 127 | 37 su 39 | 0.95 | 0.29 | 0.45 | white (75), scene_lifestyle (15) |
| scene_lifestyle | 67 | 32 su 48 | 32 su 48 | 0.67 | 0.67 | 0.67 | white (16) |

### perspective: macro F1 0.203, 258 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| front_head_on | 281 | 19 su 43 | 19 su 65 | 0.29 | 0.44 | 0.35 | front_angle_right (18), front_angle_left (4), top_down (1) |
| back_head_on | 82 | 25 su 33 | 25 su 48 | 0.52 | 0.76 | 0.62 | front_head_on (6), front_angle_right (1), right_side (1) |
| front_angle_left | 102 | 8 su 30 | 8 su 59 | 0.14 | 0.27 | 0.18 | front_angle_right (15), front_head_on (5), left_side (1) |
| front_angle_right | 102 | 14 su 30 | 14 su 55 | 0.25 | 0.47 | 0.33 | front_angle_left (10), front_head_on (6) |
| top_down | 64 | 4 su 28 | 4 su 9 | 0.44 | 0.14 | 0.22 | front_head_on (19), front_angle_right (2), right_side (1) |
| back_angle_left | 23 | 0 su 21 | 0 su 1 | 0.00 | 0.00 | 0.00 | front_angle_left (8), back_head_on (6), left_side (3) |
| back_angle_right | 21 | 0 su 21 | 0 su 2 | 0.00 | 0.00 | 0.00 | front_angle_left (11), back_head_on (5), front_angle_right (3) |
| left_side | 39 | 4 su 19 | 4 su 14 | 0.29 | 0.21 | 0.24 | front_angle_left (11), front_head_on (1), back_angle_right (1) |
| right_side | 27 | 1 su 17 | 1 su 5 | 0.20 | 0.06 | 0.09 | left_side (6), front_angle_left (5), back_head_on (3) |
| bottom_up | 12 | 0 su 16 | 0 su 0 | 0.00 | 0.00 | 0.00 | back_head_on (7), front_head_on (5), top_down (3) |

### individuals: macro F1 0.881, 384 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 922 | 281 su 281 | 281 su 313 | 0.90 | 1.00 | 0.95 | - |
| individual_present | 44 | 71 su 103 | 71 su 71 | 1.00 | 0.69 | 0.82 | none (32) |

### text_classification: macro F1 0.472, 408 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| none | 935 | 168 su 170 | 168 su 276 | 0.61 | 0.99 | 0.75 | nutritional_info (2) |
| other | 20 | 8 su 67 | 8 su 24 | 0.33 | 0.12 | 0.18 | none (53), promotional_text (6) |
| promotional_text | 26 | 44 su 65 | 44 su 57 | 0.77 | 0.68 | 0.72 | none (13), other (8) |
| nutritional_info | 55 | 39 su 53 | 39 su 42 | 0.93 | 0.74 | 0.82 | none (14) |
| technical_specs | 43 | 9 su 41 | 9 su 9 | 1.00 | 0.22 | 0.36 | none (17), other (8), promotional_text (6) |
| certification_seal | 3 | 0 su 12 | 0 su 0 | 0.00 | 0.00 | 0.00 | none (11), promotional_text (1) |

### text_language: macro F1 0.364, 312 celle

| classe | esempi visti | trovate | scelte giuste | precision | recall | F1 | errori più frequenti |
|---|---|---|---|---|---|---|---|
| English | 136 | 77 su 85 | 77 su 160 | 0.48 | 0.91 | 0.63 | none (5), German (2), other (1) |
| German | 114 | 52 su 76 | 52 su 83 | 0.63 | 0.68 | 0.65 | English (20), none (3), other (1) |
| none | 176 | 41 su 65 | 41 su 51 | 0.80 | 0.63 | 0.71 | English (24) |
| French | 14 | 9 su 44 | 9 su 14 | 0.64 | 0.20 | 0.31 | English (20), German (15) |
| other | 7 | 0 su 35 | 0 su 2 | 0.00 | 0.00 | 0.00 | English (18), German (12), French (2) |
| Italian | 4 | 1 su 7 | 1 su 1 | 1.00 | 0.14 | 0.25 | French (3), German (2), English (1) |
| Dutch | 1 | 0 su 0 | 0 su 1 | 0.00 | 0.00 | 0.00 | - |

### Matrici di confusione: focal + GradNorm

Righe: label. Colonne: risposta delle teste.

#### subject

Colonne: c1 = full_product, c2 = detail_closeup, c3 = accessories, c4 = packaging, c5 = package_and_product, c6 = multipack_trade_unit, c7 = delivery_scope, c8 = product_grouping, c9 = multiview, c10 = energy_label, c11 = infographic, c12 = logo, c13 = placeholder

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 | c11 | c12 | c13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_product | 71 | 7 | 0 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 |
| detail_closeup | 13 | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 10 | 0 | 0 |
| accessories | 26 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| packaging | 12 | 1 | 0 | 21 | 5 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| package_and_product | 16 | 3 | 0 | 15 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| multipack_trade_unit | 16 | 1 | 0 | 6 | 1 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| delivery_scope | 9 | 1 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 7 | 0 | 0 |
| product_grouping | 5 | 0 | 1 | 2 | 0 | 1 | 2 | 0 | 0 | 0 | 9 | 0 | 0 |
| multiview | 22 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 3 | 0 | 0 |
| energy_label | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 0 | 0 |
| infographic | 3 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 0 | 0 |
| logo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 | 0 |
| placeholder | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 12 |

#### background

Colonne: c1 = white, c2 = color, c3 = scene_lifestyle

| label | c1 | c2 | c3 |
|---|---|---|---|
| white | 206 | 2 | 1 |
| color | 75 | 37 | 15 |
| scene_lifestyle | 16 | 0 | 32 |

#### perspective

Colonne: c1 = front_head_on, c2 = front_angle_left, c3 = front_angle_right, c4 = back_head_on, c5 = back_angle_left, c6 = back_angle_right, c7 = left_side, c8 = right_side, c9 = top_down, c10 = bottom_up

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 19 | 4 | 18 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| front_angle_left | 5 | 8 | 15 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| front_angle_right | 6 | 10 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| back_head_on | 6 | 0 | 1 | 25 | 0 | 0 | 0 | 1 | 0 | 0 |
| back_angle_left | 1 | 8 | 2 | 6 | 0 | 0 | 3 | 1 | 0 | 0 |
| back_angle_right | 1 | 11 | 3 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| left_side | 1 | 11 | 0 | 1 | 0 | 1 | 4 | 1 | 0 | 0 |
| right_side | 2 | 5 | 0 | 3 | 0 | 0 | 6 | 1 | 0 | 0 |
| top_down | 19 | 1 | 2 | 0 | 0 | 1 | 0 | 1 | 4 | 0 |
| bottom_up | 5 | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 3 | 0 |

#### individuals

Colonne: c1 = individual_present, c2 = none

| label | c1 | c2 |
|---|---|---|
| individual_present | 71 | 32 |
| none | 0 | 281 |

#### text_classification

Colonne: c1 = certification_seal, c2 = technical_specs, c3 = promotional_text, c4 = nutritional_info, c5 = other, c6 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| certification_seal | 0 | 0 | 1 | 0 | 0 | 11 |
| technical_specs | 0 | 9 | 6 | 1 | 8 | 17 |
| promotional_text | 0 | 0 | 44 | 0 | 8 | 13 |
| nutritional_info | 0 | 0 | 0 | 39 | 0 | 14 |
| other | 0 | 0 | 6 | 0 | 8 | 53 |
| none | 0 | 0 | 0 | 2 | 0 | 168 |

#### text_language

Colonne: c1 = German, c2 = English, c3 = French, c4 = Italian, c5 = Dutch, c6 = other, c7 = none

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---|---|---|---|---|---|---|
| German | 52 | 20 | 0 | 0 | 0 | 1 | 3 |
| English | 2 | 77 | 0 | 0 | 0 | 1 | 5 |
| French | 15 | 20 | 9 | 0 | 0 | 0 | 0 |
| Italian | 2 | 1 | 3 | 1 | 0 | 0 | 0 |
| other | 12 | 18 | 2 | 0 | 1 | 0 | 2 |
| none | 0 | 24 | 0 | 0 | 0 | 0 | 41 |
