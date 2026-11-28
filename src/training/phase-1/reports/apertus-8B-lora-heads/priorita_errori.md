# Dove perdiamo: priorità per migliorare LoRA + teste (27 novembre 2026)

Elenco di tutte le prove: [RISULTATI.md](../EXPERIMENTS.md). Modello: P11, LoRA + teste (dev Kaggle **0.796**, coverage 0.781).

Kaggle mostra solo il totale; il dettaglio è sul set **coverage** (2.532 immagini di prodotti mai usati per il
training), che ha previsto bene il dev (0.781 contro 0.796). Obiettivo immediato: superare Gemini 3.8 Flash (dev
0.806, mancano circa **0.011**).

## Per dimensione

| dimensione | macro F1 | punti persi sul totale (se fosse perfetta) |
|---|---|---|
| text_language | 0.521 | 0.080 |
| perspective | 0.625 | 0.063 |
| text_classification | 0.806 | 0.032 |
| subject | 0.841 | 0.027 |
| background | 0.911 | 0.015 |
| individuals | 0.981 | 0.003 |

## Classi deboli, in ordine di guadagno possibile

Il guadagno è quanto salirebbe la macro F1 totale se la classe arrivasse a F1 0.80: (0.80 − F1) / classi della
dimensione / 6. Nelle dimensioni con poche classi ogni classe pesa di più. È una stima su coverage, non su Kaggle.

| # | dimensione | classe | F1 | trovate | scambiata soprattutto con | guadagno a 0.80 |
|---|---|---|---|---|---|---|
| 1 | text_language | Dutch | 0.00 | 0 su 20 | German (18), English (2) | +0.0190 |
| 2 | text_language | other | 0.17 | 4 su 30 | German (11), none (9) | +0.0150 |
| 3 | text_language | Italian | 0.28 | 5 su 30 | English (9), other (7) | +0.0124 |
| 4 | perspective | back_angle_left | 0.28 | 7 su 37 | back_angle_right (15), front_angle_right (7) | +0.0087 |
| 5 | text_classification | other | 0.55 | 39 su 83 | none (36), promotional_text (5) | +0.0069 |
| 6 | text_classification | certification_seal | 0.60 | 13 su 30 | none (11), other (5) | +0.0054 |
| 7 | perspective | right_side | 0.47 | 28 su 65 | left_side (24), front_head_on (4) | +0.0054 |
| 8 | perspective | back_angle_right | 0.51 | 29 su 60 | front_angle_right (14), front_angle_left (10) | +0.0048 |
| 9 | perspective | front_angle_right | 0.52 | 109 su 199 | front_angle_left (70), front_head_on (13) | +0.0047 |
| 10 | perspective | left_side | 0.54 | 43 su 82 | right_side (21), front_head_on (8) | +0.0044 |
| 11 | perspective | front_angle_left | 0.59 | 131 su 209 | front_angle_right (66), front_head_on (6) | +0.0036 |
| 12 | text_language | French | 0.67 | 38 su 56 | German (6), English (5) | +0.0032 |
| 13 | subject | accessories | 0.57 | 17 su 38 | full_product (17), delivery_scope (2) | +0.0030 |
| 14 | subject | product_grouping | 0.59 | 15 su 30 | full_product (5), multiview (5) | +0.0027 |
| 15 | subject | multiview | 0.74 | 39 su 57 | full_product (5), delivery_scope (4) | +0.0007 |
| 16 | perspective | top_down | 0.76 | 84 su 120 | front_head_on (28), bottom_up (2) | +0.0007 |

Somma se tutte queste classi arrivassero a 0.80: +0.101.

## Cosa dicono i numeri

- **Lingue rare** (Dutch, other, Italian): circa +0.046 se arrivassero a 0.80. Nel training di fase 1 ci sono solo
  2 celle umane di `Dutch` e 18 di `Italian`: quasi tutti gli esempi sono in coverage. Il modello finale su tutte le
  11.876 immagini li recupera (21 e 47 celle valutate), senza garanzia di miglioramento.
- **Destra e sinistra** in `perspective`: circa +0.03 sommando le viste laterali e angolate; il 45% degli errori di
  vista sono solo scambi di lato (200 su 440).
- **`other` e `certification_seal`** nel testo aggiunto: circa +0.012; spesso finiscono in `none`.

## Prove previste

| prova | colpisce | stato |
|---|---|---|
| modello finale su tutte le 11.876 immagini, resto invariato | lingue rare | da lanciare; si controlla solo con il dev di Kaggle |
| predizione anche sull'immagine specchiata (destra e sinistra scambiate e mediate) | destra/sinistra | da provare, senza training |
| attention pooling per dimensione | destra/sinistra, zone di testo | Nathan, in corso |
| 512² sul modello migliore | testi piccoli, sigilli | dopo l'attention pooling |
| calibrazione per classe | — | scartata: peggiora il gruppo di conferma (prompt-test) |
