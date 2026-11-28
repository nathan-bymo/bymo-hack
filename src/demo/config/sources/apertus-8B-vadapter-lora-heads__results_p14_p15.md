# P14 e P15: encoder di immagini più grande e training su tutti i dati (28 novembre 2026)

Elenco di tutte le prove: [../EXPERIMENTS.md](../EXPERIMENTS.md). Tabelle complete: [../benchmark/README.md](../benchmark/README.md).

## Prova preliminare senza training (solo encoder congelati)

Teste lineari sulle sole feature dell'encoder (media e griglia 4×4), 3.500 immagini di training, valutazione su prompt-test, stesso protocollo, weight decay scelto su una parte del training. **Non è un punteggio del modello.** Eseguita su un Mac (MPS).

| encoder | macro F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|
| SigLIP 2 base 224 | 0.682 | 0.725 | 0.756 | 0.616 | 0.861 | 0.624 | 0.511 |
| SigLIP 2 so400m 512 | 0.759 | 0.809 | 0.830 | 0.627 | 0.936 | 0.656 | 0.696 |

## P14: so400m a 512, 14×14 token

| voce | |
|---|---|
| **Domanda** | Un encoder di immagini più grande e a risoluzione più alta migliora il modello finale? |
| **Modello** | come `final-abt`, con `google/siglip2-so400m-patch16-512` (Apache-2.0) e 196 token continui al posto di SigLIP 2 base 224 con 49 |
| **Partenza** | LoRA, teste e attention pooling di `final-abt`; proiettore da zero |
| **Training** | `all_but_test` meno la validazione interna: 9,645 immagini, 14,459 esempi, 2 epoche (1808 passi, 52.0 min), batch 8×2, LR LoRA 1e-4, proiettore 2e-4, teste 5e-4, seed 3407; checkpoint scelto sulla validazione interna |
| **Esecuzione** | Modal H200; picco di memoria in training 89.4 GB. I codici VQ e il testo OCR sono riusati dalla cache |
| **Fermata** | dopo prompt-test, prima delle predizioni sul dev: nessun punteggio Kaggle |
| **Pesi** | repository privato del team, `eyes512-run1-1128-0431/model/best` |

Validazione interna (1.038 immagini): 0.784 (epoca 0.5), 0.786 (epoca 1.0), 0.826 (epoca 1.5), 0.825 (epoca 2.0). P13 sulle stesse immagini: 0.838.

| prompt-test (1.193) | macro F1 | senza le classi con meno di 10 esempi | weighted F1 |
|---|---|---|---|
| `final-abt` (dev 0.832) | 0.821 | 0.849 | 0.917 |
| P13 (dev 0.829) | 0.847 | 0.858 | 0.922 |
| P14 (dev non misurato) | 0.833 | 0.853 | 0.922 |

P14 meno `final-abt`: +0.012 [IC 95% -0.001, +0.027].

| dimensione | `final-abt` | P13 | P14 |
|---|---|---|---|
| subject | 0.846 | 0.845 | 0.868 |
| background | 0.881 | 0.870 | 0.886 |
| perspective | 0.755 | 0.783 | 0.768 |
| individuals | 0.959 | 0.965 | 0.965 |
| text_classification | 0.766 | 0.780 | 0.805 |
| text_language | 0.720 | 0.840 | 0.707 |

### Per classe (prompt-test)

#### subject

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| full_product | 653 | 0.94 | 0.94 | +0.01 |
| infographic | 99 | 0.77 | 0.80 | +0.03 |
| detail_closeup | 95 | 0.83 | 0.84 | +0.01 |
| logo | 67 | 0.95 | 0.99 | +0.04 |
| delivery_scope | 54 | 0.84 | 0.88 | +0.05 |
| energy_label | 40 | 1.00 | 1.00 | +0.00 |
| packaging | 37 | 0.74 | 0.81 | +0.08 |
| package_and_product | 28 | 0.89 | 0.91 | +0.02 |
| multiview | 20 | 0.79 | 0.76 | -0.03 |
| accessories | 15 | 0.69 | 0.74 | +0.05 |
| placeholder | 13 | 0.96 | 0.96 | +0.00 |
| product_grouping | 10 | 0.95 | 0.90 | -0.05 |
| multipack_trade_unit | 8 | 0.67 | 0.75 | +0.08 |

#### background

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| white | 819 | 0.98 | 0.98 | +0.00 |
| color | 103 | 0.81 | 0.82 | +0.01 |
| scene_lifestyle | 51 | 0.85 | 0.86 | +0.00 |

#### perspective

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| front_head_on | 275 | 0.87 | 0.89 | +0.02 |
| back_head_on | 101 | 0.85 | 0.87 | +0.02 |
| front_angle_left | 86 | 0.64 | 0.73 | +0.09 |
| front_angle_right | 80 | 0.68 | 0.74 | +0.06 |
| top_down | 69 | 0.79 | 0.78 | -0.02 |
| left_side | 44 | 0.80 | 0.75 | -0.05 |
| right_side | 32 | 0.73 | 0.79 | +0.05 |
| back_angle_right | 26 | 0.65 | 0.64 | -0.02 |
| back_angle_left | 25 | 0.71 | 0.67 | -0.04 |
| bottom_up | 25 | 0.82 | 0.85 | +0.02 |

#### individuals

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| none | 920 | 1.00 | 1.00 | +0.00 |
| individual_present | 54 | 0.92 | 0.93 | +0.01 |

#### text_classification

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| none | 879 | 0.98 | 0.98 | +0.00 |
| nutritional_info | 62 | 0.96 | 0.94 | -0.02 |
| promotional_text | 45 | 0.84 | 0.85 | +0.01 |
| technical_specs | 45 | 0.86 | 0.84 | -0.03 |
| other | 36 | 0.55 | 0.62 | +0.06 |
| certification_seal | 6 | 0.40 | 0.60 | +0.20 |

#### text_language

| classe | n | `final-abt` | P14 | differenza |
|---|---|---|---|---|
| none | 183 | 0.93 | 0.91 | -0.02 |
| English | 156 | 0.84 | 0.81 | -0.02 |
| German | 131 | 0.91 | 0.89 | -0.01 |
| French | 30 | 0.70 | 0.67 | -0.04 |
| other | 10 | 0.60 | 0.60 | +0.00 |
| Dutch | 4 | 0.40 | 0.40 | +0.00 |
| Italian | 4 | 0.67 | 0.67 | +0.00 |

Velocità su prompt-test (H200, immagine → CSV, OCR compreso, caricamento escluso): 6.23 immagini/s, memoria di picco 80.7 GB.

## P15: `final-abt` allenato su tutte le immagini etichettate

| voce | |
|---|---|
| **Domanda** | Aggiungere le 2.231 immagini reali mai usate (prompt-test e validazione interna) migliora il modello finale? |
| **Training** | tutte le 11,876 immagini, 16,560 esempi, 1 epoca (1035 passi, 23.0 min), LR LoRA 3e-5, proiettore 5e-5, teste 1.5e-4, seed 3407, ultimo checkpoint, senza immagini sintetiche |
| **Esecuzione** | Modal H200, job di circa 46 min; picco di memoria in training 77.2 GB |
| **Valutazione** | solo dev Kaggle: nessun set interno è valido perché il modello ha visto tutto il train |
| **Pesi** | repository privato del team, `base224-all-1128-0539/model/last` |

| dev Kaggle pubblico | punteggio |
|---|---|
| `final-abt` | 0.83177 |
| **P15** | **0.83144** |

Sul dev le due prove sono alla pari (−0.0003). Velocità sul dev: 7.49 immagini/s.
