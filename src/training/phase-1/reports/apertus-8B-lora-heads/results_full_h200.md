# LoRA + teste sul set di training completo (27 novembre 2026)

Elenco di tutte le prove: [RISULTATI.md](../EXPERIMENTS.md).

## Scheda della prova P11: LoRA + 6 teste, 9.344 immagini

| voce | |
|---|---|
| **Domanda** | Quanto fa Apertus 8B con LoRA e teste allenati insieme su tutto il training disponibile? |
| **Modello** | Apertus 1.5 8B, 32 layer, LoRA r32 (83,9 M parametri allenabili, LoRA e teste insieme), 6 teste; senza adapter visivo |
| **Training** | `training_rows`: 9.344 immagini. 12.491 esempi per epoca (ripetizioni fino a 3× e mirror). Loss sulle label umane (anche in celle non rilevanti) |
| **Valutazione** | `coverage`: 2.532 immagini di prodotti mai usati per il training; viste `base` e `prompt-test` |
| **Scelte** | nessuna durante il run: 3 epoche fissate prima, modello dopo l'ultimo passo, un solo seed |
| **Impostazioni** | 256², contesto prodotto incluso, cross-entropy senza pesi, batch 8 × 2, LR LoRA 2e-4, teste 1e-3 |
| **Esecuzione** | Hugging Face Jobs, 1× H200 |
| **Fonte dei numeri** | predizioni scaricate e punteggi ricalcolati con `metric.py`; coincidono con il log del job (0.7808) |
| **Cosa non dice** | non è un punteggio Kaggle; non dice se le teste sono meglio dell'uscita JSON a parità di dati; non misura il comportamento senza contesto |

## Risultato

| vista | immagini | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|---|
| coverage | 2532 | **0.781** | 0.905 | 0.841 | 0.911 | 0.625 | 0.981 | 0.806 | 0.521 |
| base (distribuzione originale) | 2363 | **0.786** | 0.913 | 0.848 | 0.914 | 0.628 | 0.990 | 0.817 | 0.521 |
| prompt-test | 1193 | **0.791** | 0.915 | 0.860 | 0.894 | 0.620 | 0.990 | 0.805 | 0.577 |

## Confronto su coverage

| metodo | training | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|---|
| LoRA + teste (P11) | 9.344 immagini | 0.781 | 0.905 | 0.841 | 0.911 | 0.625 | 0.981 | 0.806 | 0.521 |
| LoRA con JSON (P10, Kristian) | 1.500 immagini | 0.734 | 0.879 | 0.712 | 0.910 | 0.586 | 0.953 | 0.679 | 0.565 |
| 70B zero-shot v2 (P4) | nessuno | 0.583 | 0.739 | 0.546 | 0.631 | 0.274 | 0.899 | 0.618 | 0.529 |
| 8B zero-shot v2 (P4) | nessuno | 0.528 | 0.705 | 0.471 | 0.580 | 0.203 | 0.904 | 0.527 | 0.485 |

**Il confronto con P10 non è a parità di condizioni:** P11 ha circa sei volte più immagini di training. Il distacco
non si può attribuire alle teste. I numeri di P10 vengono dai file JSON del suo autore.

Differenze di macro F1 con intervallo al 95% (bootstrap appaiato per prodotto, 200 ricampionamenti; un solo seed di training):

| confronto | coverage | base | prompt-test |
|---|---|---|---|
| P11 − 70B v2 | +0.198 [+0.180, +0.220] | +0.199 [+0.179, +0.225] | +0.201 [+0.168, +0.246] |
| P11 − 8B v2 | +0.253 [+0.232, +0.270] | +0.274 [+0.252, +0.298] | +0.289 [+0.258, +0.322] |

## Per classe, coverage

`train`: celle della classe nelle 9.344 immagini che entrano nella loss; `item`: lo stesso negli esempi di un'epoca,
con ripetizioni e mirror. `n`: esempi valutati in coverage.

### subject: macro F1 0.841

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| full_product | 5210 | 7094 | 1363 | 0.94 | 0.96 | 0.95 | 0.92 | 0.80 | 0.71 |
| infographic | 805 | 909 | 215 | 0.83 | 0.89 | 0.86 | 0.80 | 0.43 | 0.38 |
| detail_closeup | 753 | 813 | 158 | 0.88 | 0.84 | 0.86 | 0.76 | 0.38 | 0.27 |
| logo | 435 | 435 | 124 | 0.98 | 0.98 | 0.98 | 0.95 | 0.89 | 0.82 |
| delivery_scope | 384 | 421 | 109 | 0.83 | 0.83 | 0.83 | 0.77 | 0.33 | 0.14 |
| packaging | 325 | 377 | 88 | 0.87 | 0.89 | 0.88 | 0.70 | 0.47 | 0.49 |
| energy_label | 350 | 352 | 80 | 0.99 | 1.00 | 0.99 | 0.99 | 0.99 | 0.94 |
| package_and_product | 196 | 407 | 58 | 0.94 | 0.88 | 0.91 | 0.77 | 0.60 | 0.52 |
| multiview | 127 | 283 | 57 | 0.81 | 0.68 | 0.74 | 0.61 | 0.38 | 0.37 |
| accessories | 170 | 384 | 38 | 0.77 | 0.45 | 0.57 | 0.38 | 0.05 | 0.00 |
| placeholder | 145 | 290 | 33 | 0.97 | 0.94 | 0.95 | 0.94 | 0.85 | 0.81 |
| multipack_trade_unit | 109 | 241 | 31 | 0.92 | 0.74 | 0.82 | 0.41 | 0.37 | 0.50 |
| product_grouping | 69 | 208 | 30 | 0.71 | 0.50 | 0.59 | 0.26 | 0.55 | 0.18 |

### background: macro F1 0.911

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| white | 6373 | 8916 | 1730 | 0.98 | 0.99 | 0.98 | 0.98 | 0.91 | 0.91 |
| color | 825 | 1155 | 204 | 0.88 | 0.78 | 0.83 | 0.84 | 0.24 | 0.18 |
| scene_lifestyle | 593 | 658 | 144 | 0.91 | 0.94 | 0.92 | 0.91 | 0.74 | 0.65 |

### perspective: macro F1 0.625

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 2233 | 2664 | 614 | 0.87 | 0.90 | 0.88 | 0.83 | 0.73 | 0.65 |
| back_head_on | 728 | 849 | 211 | 0.89 | 0.85 | 0.87 | 0.82 | 0.56 | 0.34 |
| front_angle_left | 934 | 1504 | 209 | 0.55 | 0.63 | 0.59 | 0.64 | 0.00 | 0.37 |
| front_angle_right | 814 | 1436 | 199 | 0.49 | 0.55 | 0.52 | 0.59 | 0.42 | 0.05 |
| top_down | 535 | 589 | 120 | 0.82 | 0.70 | 0.76 | 0.50 | 0.32 | 0.14 |
| left_side | 308 | 508 | 82 | 0.55 | 0.52 | 0.54 | 0.52 | 0.10 | 0.18 |
| right_side | 249 | 483 | 65 | 0.53 | 0.43 | 0.47 | 0.56 | 0.40 | 0.25 |
| back_angle_right | 159 | 431 | 60 | 0.55 | 0.48 | 0.51 | 0.40 | 0.00 | 0.00 |
| bottom_up | 185 | 371 | 41 | 0.80 | 0.85 | 0.82 | 0.64 | 0.20 | 0.05 |
| back_angle_left | 202 | 291 | 37 | 0.54 | 0.19 | 0.28 | 0.36 | 0.00 | 0.00 |

### individuals: macro F1 0.981

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 7204 | 9961 | 1966 | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| individual_present | 530 | 691 | 114 | 0.98 | 0.95 | 0.96 | 0.91 | 0.81 | 0.82 |

### text_classification: macro F1 0.806

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 7023 | 9713 | 1866 | 0.97 | 0.99 | 0.98 | 0.97 | 0.90 | 0.90 |
| nutritional_info | 438 | 547 | 127 | 0.94 | 0.94 | 0.94 | 0.91 | 0.81 | 0.67 |
| technical_specs | 345 | 364 | 103 | 0.89 | 0.86 | 0.88 | 0.84 | 0.79 | 0.65 |
| promotional_text | 306 | 377 | 86 | 0.86 | 0.92 | 0.89 | 0.72 | 0.44 | 0.44 |
| other | 273 | 325 | 83 | 0.67 | 0.47 | 0.55 | 0.41 | 0.22 | 0.18 |
| certification_seal | 30 | 90 | 30 | 1.00 | 0.43 | 0.60 | 0.24 | 0.55 | 0.32 |

### text_language: macro F1 0.521

| classe | train | item | n | precision | recall | F1 P11 | F1 P10 | F1 70B v2 | F1 8B v2 |
|---|---|---|---|---|---|---|---|---|---|
| none | 1753 | 2453 | 391 | 0.88 | 0.93 | 0.90 | 0.90 | 0.20 | 0.14 |
| English | 1241 | 1457 | 331 | 0.81 | 0.81 | 0.81 | 0.76 | 0.61 | 0.63 |
| German | 1060 | 1152 | 288 | 0.78 | 0.87 | 0.82 | 0.75 | 0.54 | 0.66 |
| French | 173 | 350 | 56 | 0.66 | 0.68 | 0.67 | 0.60 | 0.64 | 0.68 |
| Italian | 18 | 54 | 30 | 0.83 | 0.17 | 0.28 | 0.55 | 0.71 | 0.69 |
| other | 81 | 243 | 30 | 0.24 | 0.13 | 0.17 | 0.06 | 0.38 | 0.06 |
| Dutch | 2 | 6 | 20 | 0.00 | 0.00 | 0.00 | 0.33 | 0.63 | 0.53 |

## Matrice di confusione: perspective (coverage)

Righe: label. Colonne: c1 = front_head_on, c2 = front_angle_left, c3 = front_angle_right, c4 = back_head_on, c5 = back_angle_left, c6 = back_angle_right, c7 = left_side, c8 = right_side, c9 = top_down, c10 = bottom_up

| label | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---|---|---|---|---|---|---|---|---|---|
| front_head_on | 552 | 20 | 16 | 10 | 0 | 0 | 3 | 1 | 11 | 1 |
| front_angle_left | 6 | 131 | 66 | 0 | 0 | 3 | 2 | 1 | 0 | 0 |
| front_angle_right | 13 | 70 | 109 | 0 | 2 | 3 | 1 | 0 | 1 | 0 |
| back_head_on | 19 | 1 | 2 | 180 | 0 | 0 | 0 | 2 | 3 | 4 |
| back_angle_left | 0 | 2 | 7 | 4 | 7 | 15 | 2 | 0 | 0 | 0 |
| back_angle_right | 0 | 10 | 14 | 1 | 4 | 29 | 1 | 0 | 0 | 1 |
| left_side | 8 | 3 | 2 | 3 | 0 | 2 | 43 | 21 | 0 | 0 |
| right_side | 4 | 0 | 3 | 2 | 0 | 0 | 24 | 28 | 3 | 1 |
| top_down | 28 | 1 | 1 | 1 | 0 | 1 | 2 | 0 | 84 | 2 |
| bottom_up | 4 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 35 |

## Cosa si vede

- Il totale è il più alto misurato su coverage. Migliorano di molto le classi di `subject` poco frequenti, `top_down`,
  `bottom_up`, `promotional_text` e `certification_seal`.
- `text_language` (0.521) è più bassa che in P10 (0.565) e al livello del 70B zero-shot. Nel training di fase 1
  `Dutch` ha solo 2 celle con label umana (1 in una cella rilevante) e `Italian` 18, perché quasi tutti gli esempi di
  `Dutch` sono in coverage; `Dutch` non viene mai predetto e nessuno dei 20 veri viene trovato. È un limite di come è
  costruita la fase 1: la fase 2 userebbe tutti gli esempi, ma non si potrebbe misurare sul train.
- Destra e sinistra in `perspective` restano il punto debole (`front_angle_left` 0.59, `front_angle_right` 0.52).
- La memoria di training è arrivata a 139 GB su 141: margine quasi nullo con queste impostazioni.

## Punteggio Kaggle (dev, classifica pubblica)

Inviato il 27 novembre alle 21:30 CET (submission 59018375), dall'account di un membro del team. Righe final
costanti come in `sample_submission.csv` (le immagini final non sono ancora disponibili).

| modello | dev pubblico |
|---|---|
| Qwen 3.5 9B finetunato (baseline Galaxus) | 0.917 |
| Gemini 4 Argon zero-shot | 0.834 |
| Gemini 3.8 Flash zero-shot | 0.806 |
| **LoRA + teste (P11)** | **0.796** |

Il modello è **sotto Gemini 3.8 Flash** sul dev: oggi sarebbe nella fascia di performance più bassa.

Le predizioni dev vengono dal modello **riscaricato** dal repository privato (`MODE=predict` del job): su coverage dà
le stesse risposte del job di training per tutte le 2.532 immagini (macro F1 0.781), 13,35 immagini/s su H200.

## Velocità, memoria e costo

| fase | misura |
|---|---|
| codifica VQ delle immagini di training | 10.874 immagini (con mirror) a 19,4 immagini/s |
| training | 2.343 passi in 46,7 minuti, 13,4 esempi/s, picco di memoria riservata 139,0 GB |
| predizione di coverage, dall'immagine al CSV | 2.532 immagini in 191,5 s: **13,2 immagini/s**, batch 32, picco allocato **22,0 GB**; nessuna immagine illeggibile |
| job completo | 62,3 minuti, circa 5,19 $ (durata × 5 $/h; la fattura può differire) |

Stime ricavate dalla velocità di predizione, caricamento del modello escluso: dev + final (10.553 immagini) circa
13 minuti; 100.000 immagini circa 2,1 ore, circa 10,5 $ sulla H200. La codifica VQ elabora un'immagine alla volta:
è un candidato da ottimizzare, non ancora profilato. Attivazione `xielu` ottimizzata non installata.
La velocità di P10 (18,0 immagini/s) è misurata in modo diverso e non è confrontabile.

## Come riprodurre

```sh
python src/training/apertus-8B-lora-heads/build_hf_job.py
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN -e MODE=train -e SPLIT=full \
    -e "TRAIN_ARGS=--max-minutes 80" -e UPLOAD_REPO=<utente>/galaxus-lora-heads-private data/hf_jobs/hf_job_lora_heads.py
```

Seed 3407. Modello base `swiss-ai/Apertus-v1.5-8B`, revisione `1519de0ae7f3d114879291810a58259966f64c99`; fork di
Transformers al commit `3797303dda74844e3d1f8977ff5518bb91f818b4`; torch 2.8.0, peft 0.22.0. Codice: il commit che
aggiunge questo file (il job conteneva gli stessi file, non ancora in un commit al momento del lancio). I pesi sono in
un repository privato: **vanno pubblicati** se questo modello produce la submission finale.
