# Teste su Apertus 8B: pesi per classe, focal e cross-entropy, 5 seed (27 novembre 2026)

Elenco di tutte le prove: [RISULTATI.md](../RISULTATI.md).

## Scheda della prova P9: pesi per classe per la focal e la cross-entropy

| voce | |
|---|---|
| **Domanda** | Dare più peso alle classi rare aiuta? A parità di pesi, la focal fa meglio della cross-entropy? |
| **Modello** | Apertus 1.5 8B base **congelato, senza LoRA**: feature salvate dalla prova P8 |
| **Cosa si allena** | solo strato condiviso e 6 teste |
| **Training** | `train_1500.csv`: 1.199 immagini per i pesi, 301 per scegliere layer ed epoca |
| **Valutazione** | `validation_450.csv` (450 immagini bilanciate) |
| **Varianti** | 6: cross-entropy e focal (γ=2), ciascuna senza pesi, con pesi moderati e con pesi forti. **GradNorm non è in questa prova** |
| **Pesi per classe** | proporzionali a (1/esempi di training)^p, con p=0.5 (moderati) o 1 (forti); media 1 sulle righe di training prima del limite; limite 10; uguali per le due loss |
| **Ripetizioni** | 5 seed (3407, 1, 2, 3, 4); 6 varianti × 6 layer × 5 seed = 180 training, massimo 30 epoche ciascuno |
| **Esecuzione** | CPU del Mac, circa 7,5 minuti, nessuna GPU |
| **Cosa non dice** | non confrontabile con `coverage`; l'intervallo ricampiona le immagini sulla media dei 5 seed e non copre tutta la variabilità tra training |

## Risultati sulle 450 (media di 5 seed)

| variante | macro F1 media | deviazione standard | min | max | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross-entropy, senza pesi | 0.539 | 0.006 | 0.533 | 0.546 | 0.609 | 0.466 | 0.777 | 0.236 | 0.893 | 0.501 | 0.359 |
| cross-entropy, pesi moderati | 0.538 | 0.006 | 0.532 | 0.545 | 0.614 | 0.443 | 0.774 | 0.219 | 0.903 | 0.513 | 0.376 |
| cross-entropy, pesi forti | 0.557 | 0.004 | 0.551 | 0.562 | 0.629 | 0.483 | 0.795 | 0.234 | 0.918 | 0.519 | 0.395 |
| focal, senza pesi | 0.532 | 0.013 | 0.519 | 0.547 | 0.601 | 0.448 | 0.759 | 0.225 | 0.912 | 0.488 | 0.363 |
| focal, pesi moderati | 0.538 | 0.007 | 0.532 | 0.551 | 0.611 | 0.449 | 0.782 | 0.227 | 0.904 | 0.504 | 0.365 |
| focal, pesi forti | 0.547 | 0.007 | 0.540 | 0.557 | 0.618 | 0.455 | 0.775 | 0.235 | 0.923 | 0.502 | 0.390 |

## Differenze

Media dei 5 seed. Intervallo al 95%, bootstrap appaiato per prodotto, 200 ricampionamenti.

| confronto (primo meno secondo) | differenza | intervallo |
|---|---|---|
| cross-entropy, pesi moderati − cross-entropy, senza pesi | -0.001 | [-0.006, +0.008] |
| cross-entropy, pesi forti − cross-entropy, senza pesi | +0.019 | [+0.010, +0.030] |
| focal, senza pesi − cross-entropy, senza pesi | -0.006 | [-0.013, +0.000] |
| focal, pesi moderati − cross-entropy, pesi moderati | +0.001 | [-0.008, +0.008] |
| focal, pesi forti − cross-entropy, pesi forti | -0.011 | [-0.018, -0.002] |
| focal, pesi moderati − focal, senza pesi | +0.006 | [-0.001, +0.017] |

## Scelta sulla validazione interna

Media dei 5 seed sulle 301 immagini (tra parentesi minimo e massimo). In grassetto il layer scelto per ogni variante.

| layer | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|
| 12 | 0.6232 (0.613-0.638) | 0.6287 (0.621-0.638) | 0.6189 (0.611-0.629) | 0.6254 (0.618-0.633) | 0.6273 (0.619-0.638) | 0.6211 (0.616-0.627) |
| 16 | 0.6487 (0.638-0.658) | 0.6435 (0.631-0.663) | 0.6397 (0.622-0.654) | 0.6364 (0.625-0.646) | 0.6410 (0.629-0.649) | 0.6385 (0.621-0.650) |
| 20 | 0.6540 (0.647-0.660) | 0.6581 (0.651-0.669) | 0.6506 (0.644-0.658) | 0.6536 (0.648-0.661) | 0.6520 (0.642-0.660) | 0.6405 (0.629-0.661) |
| 24 | **0.6613 (0.650-0.672)** | 0.6586 (0.652-0.665) | 0.6440 (0.636-0.662) | **0.6585 (0.644-0.668)** | 0.6537 (0.646-0.663) | 0.6424 (0.628-0.652) |
| 28 | 0.6587 (0.650-0.666) | 0.6616 (0.656-0.672) | **0.6559 (0.652-0.666)** | 0.6532 (0.644-0.664) | **0.6543 (0.644-0.665)** | 0.6364 (0.623-0.649) |
| 32 | 0.6587 (0.651-0.664) | **0.6643 (0.652-0.677)** | 0.6536 (0.640-0.667) | 0.6564 (0.638-0.671) | 0.6426 (0.635-0.660) | **0.6432 (0.633-0.658)** |

## Conclusioni

- **Regola dichiarata prima (scelta sulla validazione interna):** vince `cross-entropy, pesi moderati` (0.6643), che però sulle 450 vale 0.538, come senza pesi (0.539). Le differenze sulla validazione interna tra le varianti sono più piccole della variazione tra seed: quella validazione **non riesce a distinguerle**.
- **Pesi forti:** sulle 450 la cross-entropy con pesi forti ha il valore più alto (0.557, +0.019 [+0.010, +0.030] su quella senza pesi). Sceglierla per questo risultato significherebbe scegliere sul set di valutazione: va confermata su altre immagini.
- **Focal contro cross-entropy a parità di pesi:** la focal non fa meglio (senza pesi −0.006 [−0.013, +0.000]; moderati +0.001 [−0.008, +0.008]; forti −0.011 [−0.018, −0.002]).
- **Correzione rispetto a P8:** con un solo seed la focal era −0.013 sotto la cross-entropy; su 5 seed la differenza è −0.006 e l'intervallo tocca zero.
- Il guadagno dei pesi (circa 0.02) è piccolo rispetto al distacco dal LoRA sulle stesse 450 (vedi [confronto](../confronto_lora.md)).

## F1 per classe sulle 450 (media di 5 seed)

`n`: esempi nella validazione. Una classe senza esempi compare se almeno un seed l'ha predetta (F1 0).

### subject

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| full_product | 85 | 0.52 | 0.52 | 0.52 | 0.49 | 0.50 | 0.52 |
| packaging | 42 | 0.50 | 0.44 | 0.49 | 0.43 | 0.46 | 0.42 |
| package_and_product | 38 | 0.10 | 0.07 | 0.11 | 0.09 | 0.07 | 0.08 |
| multiview | 36 | 0.37 | 0.40 | 0.45 | 0.39 | 0.43 | 0.44 |
| detail_closeup | 30 | 0.24 | 0.26 | 0.26 | 0.17 | 0.23 | 0.26 |
| accessories | 29 | 0.00 | 0.03 | 0.08 | 0.01 | 0.04 | 0.06 |
| multipack_trade_unit | 28 | 0.35 | 0.25 | 0.41 | 0.27 | 0.24 | 0.27 |
| delivery_scope | 28 | 0.57 | 0.50 | 0.59 | 0.53 | 0.52 | 0.51 |
| infographic | 24 | 0.41 | 0.34 | 0.39 | 0.39 | 0.42 | 0.38 |
| product_grouping | 20 | 0.13 | 0.09 | 0.11 | 0.16 | 0.06 | 0.11 |
| energy_label | 14 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| logo | 14 | 0.95 | 0.94 | 0.94 | 0.96 | 0.94 | 0.93 |
| placeholder | 14 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |

### background

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| white | 209 | 0.87 | 0.87 | 0.88 | 0.86 | 0.87 | 0.87 |
| color | 127 | 0.65 | 0.68 | 0.70 | 0.62 | 0.67 | 0.69 |
| scene_lifestyle | 48 | 0.81 | 0.77 | 0.80 | 0.80 | 0.80 | 0.76 |

### perspective

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| front_head_on | 43 | 0.51 | 0.45 | 0.47 | 0.48 | 0.51 | 0.49 |
| back_head_on | 33 | 0.64 | 0.65 | 0.66 | 0.65 | 0.66 | 0.67 |
| front_angle_left | 30 | 0.21 | 0.21 | 0.19 | 0.21 | 0.20 | 0.21 |
| front_angle_right | 30 | 0.37 | 0.33 | 0.34 | 0.37 | 0.34 | 0.35 |
| top_down | 28 | 0.28 | 0.28 | 0.29 | 0.24 | 0.27 | 0.31 |
| back_angle_left | 21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| back_angle_right | 21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| left_side | 19 | 0.23 | 0.17 | 0.25 | 0.16 | 0.17 | 0.21 |
| right_side | 17 | 0.11 | 0.10 | 0.13 | 0.14 | 0.11 | 0.11 |
| bottom_up | 16 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### individuals

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| none | 281 | 0.95 | 0.95 | 0.96 | 0.96 | 0.96 | 0.96 |
| individual_present | 103 | 0.84 | 0.85 | 0.88 | 0.87 | 0.85 | 0.88 |

### text_classification

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| none | 170 | 0.77 | 0.78 | 0.78 | 0.77 | 0.77 | 0.78 |
| other | 67 | 0.10 | 0.11 | 0.16 | 0.12 | 0.14 | 0.12 |
| promotional_text | 65 | 0.78 | 0.82 | 0.81 | 0.75 | 0.78 | 0.77 |
| nutritional_info | 53 | 0.88 | 0.89 | 0.90 | 0.84 | 0.87 | 0.90 |
| technical_specs | 41 | 0.49 | 0.47 | 0.44 | 0.45 | 0.43 | 0.44 |
| certification_seal | 12 | 0.00 | 0.00 | 0.03 | 0.00 | 0.03 | 0.00 |

### text_language

| classe | n | cross-entropy, senza pesi | cross-entropy, pesi moderati | cross-entropy, pesi forti | focal, senza pesi | focal, pesi moderati | focal, pesi forti |
|---|---|---|---|---|---|---|---|
| English | 85 | 0.63 | 0.65 | 0.66 | 0.63 | 0.63 | 0.64 |
| German | 76 | 0.62 | 0.61 | 0.62 | 0.59 | 0.63 | 0.57 |
| none | 65 | 0.70 | 0.73 | 0.70 | 0.70 | 0.70 | 0.72 |
| French | 44 | 0.21 | 0.36 | 0.37 | 0.22 | 0.25 | 0.35 |
| other | 35 | 0.00 | 0.04 | 0.05 | 0.00 | 0.03 | 0.05 |
| Italian | 7 | 0.20 | 0.24 | 0.36 | 0.25 | 0.22 | 0.39 |
| Dutch | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Precision e recall: senza pesi e con pesi forti (cross-entropy, media di 5 seed)

| dimensione | classe | n | senza pesi P / R | pesi forti P / R |
|---|---|---|---|---|
| subject | full_product | 85 | 0.37 / 0.84 | 0.39 / 0.78 |
| subject | packaging | 42 | 0.50 / 0.50 | 0.48 / 0.50 |
| subject | package_and_product | 38 | 0.34 / 0.06 | 0.30 / 0.07 |
| subject | multiview | 36 | 0.82 / 0.24 | 0.80 / 0.32 |
| subject | detail_closeup | 30 | 0.28 / 0.22 | 0.28 / 0.25 |
| subject | accessories | 29 | 0.00 / 0.00 | 0.57 / 0.04 |
| subject | multipack_trade_unit | 28 | 0.86 / 0.24 | 0.75 / 0.29 |
| subject | delivery_scope | 28 | 0.72 / 0.48 | 0.62 / 0.56 |
| subject | infographic | 24 | 0.29 / 0.68 | 0.28 / 0.67 |
| subject | product_grouping | 20 | 0.70 / 0.07 | 0.70 / 0.06 |
| subject | energy_label | 14 | 1.00 / 1.00 | 1.00 / 1.00 |
| subject | logo | 14 | 0.94 / 0.97 | 0.92 / 0.97 |
| subject | placeholder | 14 | 1.00 / 0.86 | 1.00 / 0.86 |
| background | white | 209 | 0.78 / 0.99 | 0.80 / 0.97 |
| background | color | 127 | 0.94 / 0.50 | 0.92 / 0.57 |
| background | scene_lifestyle | 48 | 0.79 / 0.84 | 0.77 / 0.84 |
| perspective | front_head_on | 43 | 0.40 / 0.69 | 0.38 / 0.61 |
| perspective | back_head_on | 33 | 0.51 / 0.86 | 0.51 / 0.92 |
| perspective | front_angle_left | 30 | 0.15 / 0.32 | 0.15 / 0.28 |
| perspective | front_angle_right | 30 | 0.31 / 0.45 | 0.28 / 0.44 |
| perspective | top_down | 28 | 0.55 / 0.19 | 0.51 / 0.21 |
| perspective | back_angle_left | 21 | 0.00 / 0.00 | 0.00 / 0.00 |
| perspective | back_angle_right | 21 | 0.00 / 0.00 | 0.00 / 0.00 |
| perspective | left_side | 19 | 0.44 / 0.16 | 0.42 / 0.18 |
| perspective | right_side | 17 | 0.37 / 0.07 | 0.29 / 0.08 |
| perspective | bottom_up | 16 | 0.20 / 0.01 | 0.00 / 0.00 |
| individuals | none | 281 | 0.91 / 1.00 | 0.92 / 1.00 |
| individuals | individual_present | 103 | 1.00 / 0.72 | 1.00 / 0.78 |
| text_classification | none | 170 | 0.63 / 0.99 | 0.65 / 0.98 |
| text_classification | other | 67 | 0.26 / 0.06 | 0.43 / 0.10 |
| text_classification | promotional_text | 65 | 0.82 / 0.75 | 0.80 / 0.83 |
| text_classification | nutritional_info | 53 | 0.93 / 0.84 | 0.91 / 0.89 |
| text_classification | technical_specs | 41 | 0.88 / 0.34 | 0.85 / 0.30 |
| text_classification | certification_seal | 12 | 0.00 / 0.00 | 0.20 / 0.02 |
| text_language | English | 85 | 0.49 / 0.86 | 0.53 / 0.88 |
| text_language | German | 76 | 0.57 / 0.69 | 0.58 / 0.68 |
| text_language | none | 65 | 0.77 / 0.65 | 0.80 / 0.63 |
| text_language | French | 44 | 0.49 / 0.13 | 0.60 / 0.28 |
| text_language | other | 35 | 0.00 / 0.00 | 0.23 / 0.03 |
| text_language | Italian | 7 | 0.80 / 0.11 | 0.93 / 0.23 |
| text_language | Dutch | 0 | 0.00 / 0.00 | 0.00 / 0.00 |

## Come riprodurre

```sh
python src/apertus-8B-heads/train_heads.py --features <feature della prova P8> --out data/runs/heads_weights \
    --train-csv data/nathan_split/train_1500.csv --eval-csv data/nathan_split/validation_450.csv --val-fraction 0.2 \
    --variants "ce,ce+wsqrt,ce+winv,focal,focal+wsqrt,focal+winv" --seeds 3407,1,2,3,4
```
