# Confronto con il LoRA prototype di Kristian (27 novembre 2026)

Elenco di tutte le prove: [RISULTATI.md](../training/EXPERIMENTS.md).

## Scheda della prova P10: LoRA su Apertus 8B, split piccolo (eseguita da Kristian)

| voce | |
|---|---|
| **Domanda** | Quanto fa Apertus 8B con un LoRA allenato sulle 1.500 immagini? |
| **Modello** | Apertus 1.5 8B con **LoRA** (rank 32, alpha 32), learning rate 0.0002 |
| **Training** | `train_1500.csv`, tutte le 1.500 immagini; con ripetizioni delle classi rare e copie specchiate sono 3.952 esempi per epoca; 3 epoche, 741 passi, 15,9 minuti. Loss sulle sole label umane |
| **Valutazione** | `validation_450.csv` e `coverage` (2.532); nessuna delle due contiene prodotti di training |
| **Impostazioni** | immagini a **256²**, system prompt breve, risposta JSON generata in modo greedy; un seed |
| **Esecuzione** | Hugging Face Jobs, H200 |
| **Fonte dei numeri** | file in [micro-training/lora-generativo/results/prototype-256](lora-generativo/results/prototype-256/). **Non ricalcolati qui**: le predizioni non sono disponibili. Ho controllato che il numero di celle valutate coincida con i nostri set |
| **Cosa non dice** | un solo training; il confronto con le teste non è a parità di condizioni (vedi sotto) |

## Condizioni diverse

| | teste (P8) | LoRA (P10) |
|---|---|---|
| immagini per allenare i pesi | 1.199 (+301 per scegliere) | 1.500 |
| augmentation e ripetizioni | no | sì |
| risoluzione | 512² | 256² |
| Apertus | congelato | adattato con LoRA |
| uscita | 6 teste | JSON generato |

La differenza di punteggio **non si può attribuire solo all'uscita** (teste contro testo): cambiano più cose insieme.

## Sulle 450 di Nathan

| metodo | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| LoRA (P10) | 0.696 | 0.747 | 0.642 | 0.904 | 0.484 | 0.942 | 0.661 | 0.546 |
| teste, cross-entropy, H200, 1 seed (P8) | 0.542 | 0.612 | 0.466 | 0.817 | 0.230 | 0.881 | 0.493 | 0.363 |
| teste, cross-entropy con pesi forti, media 5 seed (P9) | 0.557 | 0.629 | 0.483 | 0.795 | 0.234 | 0.918 | 0.519 | 0.395 |

Risposte non valide del LoRA sulle 450: 9 (contano come errore).

## Su coverage (2.532 immagini)

| metodo | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| LoRA, 1.500 immagini, 256² (P10) | 0.734 | 0.879 | 0.712 | 0.910 | 0.586 | 0.953 | 0.679 | 0.565 |
| 8B zero-shot v2, 512² (P4) | 0.528 | 0.705 | 0.471 | 0.580 | 0.203 | 0.904 | 0.527 | 0.485 |
| 70B zero-shot v2, 512² (P4) | 0.583 | 0.739 | 0.546 | 0.631 | 0.274 | 0.899 | 0.618 | 0.529 |

LoRA sulle altre viste: base 0.721, prompt-test 0.711. Risposte non valide su coverage: 21. Le teste **non** sono ancora state valutate su coverage.

## F1 per classe su coverage

### subject

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| full_product | 1363 | 0.92 | 0.71 | 0.80 |
| infographic | 215 | 0.80 | 0.38 | 0.43 |
| detail_closeup | 158 | 0.76 | 0.27 | 0.38 |
| logo | 124 | 0.95 | 0.82 | 0.89 |
| delivery_scope | 109 | 0.77 | 0.14 | 0.33 |
| packaging | 88 | 0.70 | 0.49 | 0.47 |
| energy_label | 80 | 0.99 | 0.94 | 0.99 |
| package_and_product | 58 | 0.77 | 0.52 | 0.60 |
| multiview | 57 | 0.61 | 0.37 | 0.38 |
| accessories | 38 | 0.38 | 0.00 | 0.05 |
| placeholder | 33 | 0.94 | 0.81 | 0.85 |
| multipack_trade_unit | 31 | 0.41 | 0.50 | 0.37 |
| product_grouping | 30 | 0.26 | 0.18 | 0.55 |

### background

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| white | 1730 | 0.98 | 0.91 | 0.91 |
| color | 204 | 0.84 | 0.18 | 0.24 |
| scene_lifestyle | 144 | 0.91 | 0.65 | 0.74 |

### perspective

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| front_head_on | 614 | 0.83 | 0.65 | 0.73 |
| back_head_on | 211 | 0.82 | 0.34 | 0.56 |
| front_angle_left | 209 | 0.64 | 0.37 | 0.00 |
| front_angle_right | 199 | 0.59 | 0.05 | 0.42 |
| top_down | 120 | 0.50 | 0.14 | 0.32 |
| left_side | 82 | 0.52 | 0.18 | 0.10 |
| right_side | 65 | 0.56 | 0.25 | 0.40 |
| back_angle_right | 60 | 0.40 | 0.00 | 0.00 |
| bottom_up | 41 | 0.64 | 0.05 | 0.20 |
| back_angle_left | 37 | 0.36 | 0.00 | 0.00 |

### individuals

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| none | 1966 | 0.99 | 0.99 | 0.99 |
| individual_present | 114 | 0.91 | 0.82 | 0.81 |

### text_classification

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| none | 1866 | 0.97 | 0.90 | 0.90 |
| nutritional_info | 127 | 0.91 | 0.67 | 0.81 |
| technical_specs | 103 | 0.84 | 0.65 | 0.79 |
| promotional_text | 86 | 0.72 | 0.44 | 0.44 |
| other | 83 | 0.41 | 0.18 | 0.22 |
| certification_seal | 30 | 0.24 | 0.32 | 0.55 |

### text_language

| classe | n | LoRA | 8B v2 | 70B v2 |
|---|---|---|---|---|
| none | 391 | 0.90 | 0.14 | 0.20 |
| English | 331 | 0.76 | 0.63 | 0.61 |
| German | 288 | 0.75 | 0.66 | 0.54 |
| French | 56 | 0.60 | 0.68 | 0.64 |
| Italian | 30 | 0.55 | 0.69 | 0.71 |
| other | 30 | 0.06 | 0.06 | 0.38 |
| Dutch | 20 | 0.33 | 0.53 | 0.63 |

## F1 per classe sulle 450

### subject

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| full_product | 85 | 0.66 | 0.48 |
| packaging | 42 | 0.68 | 0.45 |
| package_and_product | 38 | 0.71 | 0.16 |
| multiview | 36 | 0.46 | 0.36 |
| detail_closeup | 30 | 0.62 | 0.22 |
| accessories | 29 | 0.06 | 0.07 |
| multipack_trade_unit | 28 | 0.79 | 0.24 |
| delivery_scope | 28 | 0.79 | 0.60 |
| infographic | 24 | 0.49 | 0.39 |
| product_grouping | 20 | 0.29 | 0.18 |
| energy_label | 14 | 1.00 | 1.00 |
| logo | 14 | 0.88 | 0.97 |
| placeholder | 14 | 0.92 | 0.92 |

### background

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| white | 209 | 0.94 | 0.89 |
| color | 127 | 0.89 | 0.72 |
| scene_lifestyle | 48 | 0.88 | 0.84 |

### perspective

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| front_head_on | 43 | 0.65 | 0.55 |
| back_head_on | 33 | 0.78 | 0.64 |
| front_angle_left | 30 | 0.55 | 0.12 |
| front_angle_right | 30 | 0.46 | 0.38 |
| top_down | 28 | 0.57 | 0.29 |
| back_angle_left | 21 | 0.46 | 0.00 |
| back_angle_right | 21 | 0.26 | 0.00 |
| left_side | 19 | 0.40 | 0.23 |
| right_side | 17 | 0.41 | 0.09 |
| bottom_up | 16 | 0.30 | 0.00 |

### individuals

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| none | 281 | 0.97 | 0.95 |
| individual_present | 103 | 0.92 | 0.82 |

### text_classification

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| none | 170 | 0.85 | 0.76 |
| other | 67 | 0.53 | 0.12 |
| promotional_text | 65 | 0.87 | 0.80 |
| nutritional_info | 53 | 0.92 | 0.86 |
| technical_specs | 41 | 0.65 | 0.43 |
| certification_seal | 12 | 0.15 | 0.00 |

### text_language

| classe | n | LoRA (P10) | teste cross-entropy (P8) |
|---|---|---|---|
| English | 85 | 0.70 | 0.62 |
| German | 76 | 0.66 | 0.63 |
| none | 65 | 0.86 | 0.72 |
| French | 44 | 0.46 | 0.12 |
| other | 35 | 0.00 | 0.00 |
| Italian | 7 | 0.60 | 0.44 |
| Dutch | 0 | - | 0.00 |

## Note

- Nel training del LoRA la loss finale è circa 0.001: il modello riproduce quasi perfettamente le immagini di training. Le valutazioni qui sopra sono su prodotti mai visti.
- Velocità riportata nei file di Kristian: 18,0 immagini/s su coverage (generazione, 256², H200). Non confrontabile con le 8 immagini/s dell'estrazione delle teste (512², condizioni diverse).
- Un commento in `lora/train.py` indica che l'**adapter finale di prototype-256 è andato perso** perché il caricamento è fallito. Da confermare con Kristian: se è così, per riusare quei pesi serve rifare il training.
