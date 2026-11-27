# Teste su Apertus 8B congelato: primo test su H200 (27 novembre 2026)

Apertus v1.5 8B base, **congelato, senza LoRA**; si allenano solo lo strato condiviso e le sei teste. Split di Nathan:
1.500 immagini di training (1.199 per allenare, 301 per scegliere layer ed epoca) e 450 di valutazione. Immagini a
512², contesto prodotto incluso. Un solo seed. Hugging Face Jobs, 1× H200, batch 16.

**Le 450 sono bilanciate per classe:** questi punteggi non si confrontano con quelli su `coverage`.

## Risultati sulle 450

| variante | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| soli pixel (focal+GradNorm) | 0.257 | 0.344 | 0.186 | 0.435 | 0.129 | 0.529 | 0.111 | 0.152 |
| cross-entropy | 0.542 | 0.612 | 0.466 | 0.817 | 0.230 | 0.881 | 0.493 | 0.363 |
| focal | 0.528 | 0.600 | 0.448 | 0.764 | 0.267 | 0.885 | 0.499 | 0.306 |
| cross-entropy + GradNorm | 0.537 | 0.605 | 0.462 | 0.756 | 0.228 | 0.902 | 0.506 | 0.367 |
| focal + GradNorm | 0.501 | 0.577 | 0.441 | 0.642 | 0.203 | 0.881 | 0.472 | 0.364 |

La riga dei soli pixel non è un risultato di Apertus: stesse teste su miniature 24×24, soglia minima.

## Differenze di macro F1

Intervallo al 95%, bootstrap appaiato per prodotto, 300 ricampionamenti. Non misura la variabilità tra seed.

| confronto (primo meno secondo) | differenza | intervallo |
|---|---|---|
| cross-entropy − soli pixel (focal+GradNorm) | +0.285 | [+0.266, +0.313] |
| focal + GradNorm − soli pixel (focal+GradNorm) | +0.244 | [+0.225, +0.270] |
| focal − cross-entropy | -0.013 | [-0.029, -0.002] |
| cross-entropy + GradNorm − cross-entropy | -0.005 | [-0.015, +0.005] |
| focal + GradNorm − cross-entropy | -0.041 | [-0.055, -0.026] |

## Scelta di layer ed epoca

Macro F1 sulla validazione interna (301 immagini). È la base della scelta; le 450 non sono usate.

| layer | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|
| 12 | 0.6138 | 0.6169 | 0.6134 | 0.6176 |
| 16 | 0.6406 | 0.6494 | 0.6450 | 0.6453 |
| 20 | 0.6489 | **0.6649** | 0.6485 | 0.6496 |
| 24 | **0.6732** | 0.6622 | **0.6630** | 0.6548 |
| 28 | 0.6508 | 0.6533 | 0.6563 | **0.6583** |
| 32 | 0.6488 | 0.6471 | 0.6516 | 0.6391 |

Con la cross-entropy, la variante con il valore più alto sulla validazione interna, il layer scelto è il 24 su 32.

Pesi finali di GradNorm (somma 6):

| variante | layer scelto | epoca | pesi |
|---|---|---|---|
| cross-entropy | 24 | 22 | - |
| focal | 20 | 16 | - |
| cross-entropy + GradNorm | 24 | 23 | subject 1.09, background 1.21, perspective 1.39, individuals 0.79, text_classification 0.74, text_language 0.79 |
| focal + GradNorm | 28 | 13 | subject 0.70, background 1.50, perspective 1.37, individuals 1.11, text_classification 0.61, text_language 0.71 |

## Configurazione del training delle teste

| impostazione | valore |
|---|---|
| Apertus | congelato, nessuna epoca: un forward per immagine |
| feature per layer | ultimo token del prompt e media dei token immagine, standardizzate sulle immagini di training |
| layer provati | 12, 16, 20, 24, 28, 32 |
| modello | strato condiviso da 512 (GELU, dropout 0.3) e una testa lineare per dimensione |
| modelli allenati | 24 (6 layer × 4 varianti) |
| epoche | massimo 30 per modello; si tiene quella migliore sulla validazione interna |
| batch | 256 su 1.199 immagini: 5 passi per epoca, al massimo 150 |
| ottimizzatore | AdamW, learning rate 0.001, weight decay 0.05, andamento a coseno |
| gradient clipping | norma 1.0, in tutte le varianti |
| focal | γ = 2, senza pesi per classe |
| GradNorm | α = 1.5, learning rate dei pesi 0.025 |
| seed | 3407, uno solo |

Epoche tenute: cross-entropy 22 (110 passi), focal 16, cross-entropy + GradNorm 23, focal + GradNorm 13. I passi sono
pochi, in particolare per GradNorm, che impara anche i pesi delle sei loss: l'effetto di questo **non è stato
verificato**. Con 30 epoche massime tutte le epoche tenute vengono prima dell'ultima; non è stato provato un massimo
diverso. Download, estrazione, training e valutazione: 7,4 minuti; training dei 24 modelli circa 1,4 minuti.

Dettaglio per classe e matrici di confusione di tutte le varianti: [results_small_h200_per_class.md](results_small_h200_per_class.md).

## Osservazioni sul dettaglio per classe (cross-entropy)

- Le classi più frequenti nel training vengono scelte spesso: `full_product` 194 volte (85 esempi veri), `none` del
  testo 274 (170), `English` 162 (85). Il training segue circa la distribuzione originale, le 450 sono bilanciate.
  **Ipotesi da verificare:** dare più peso alle classi rare può aiutare.
- Pochi esempi pesano ma non spiegano tutto: `certification_seal` (3 esempi visti) non viene mai trovata, `placeholder`
  (20) va bene, `front_angle_left` (102) viene scambiata per `front_angle_right` 18 volte su 30.

## F1 per classe

`n`: esempi della classe nelle celle valutate (⚠ sotto 10). `-`: classe senza esempi e mai predetta.

### subject

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| full_product | 85 | 0.37 | 0.48 | 0.45 | 0.50 | 0.51 |
| packaging | 42 | 0.00 | 0.45 | 0.44 | 0.46 | 0.46 |
| package_and_product | 38 | 0.00 | 0.16 | 0.12 | 0.19 | 0.09 |
| multiview | 36 | 0.00 | 0.36 | 0.38 | 0.36 | 0.36 |
| detail_closeup | 30 | 0.19 | 0.22 | 0.22 | 0.25 | 0.22 |
| accessories | 29 | 0.00 | 0.07 | 0.12 | 0.00 | 0.06 |
| multipack_trade_unit | 28 | 0.00 | 0.24 | 0.25 | 0.24 | 0.19 |
| delivery_scope | 28 | 0.00 | 0.60 | 0.57 | 0.53 | 0.49 |
| infographic | 24 | 0.25 | 0.39 | 0.41 | 0.39 | 0.46 |
| product_grouping | 20 | 0.00 | 0.18 | 0.00 | 0.18 | 0.00 |
| energy_label | 14 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| logo | 14 | 0.18 | 0.97 | 0.96 | 1.00 | 0.96 |
| placeholder | 14 | 0.42 | 0.92 | 0.92 | 0.92 | 0.92 |

### background

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| white | 209 | 0.75 | 0.89 | 0.87 | 0.86 | 0.81 |
| color | 127 | 0.19 | 0.72 | 0.65 | 0.61 | 0.45 |
| scene_lifestyle | 48 | 0.36 | 0.84 | 0.77 | 0.80 | 0.67 |

### perspective

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| front_head_on | 43 | 0.29 | 0.55 | 0.50 | 0.51 | 0.35 |
| back_head_on | 33 | 0.22 | 0.64 | 0.70 | 0.64 | 0.62 |
| front_angle_left | 30 | 0.26 | 0.12 | 0.25 | 0.16 | 0.18 |
| front_angle_right | 30 | 0.38 | 0.38 | 0.36 | 0.30 | 0.33 |
| top_down | 28 | 0.04 | 0.29 | 0.38 | 0.30 | 0.22 |
| back_angle_left | 21 | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 |
| back_angle_right | 21 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| left_side | 19 | 0.00 | 0.23 | 0.28 | 0.29 | 0.24 |
| right_side | 17 | 0.10 | 0.09 | 0.11 | 0.09 | 0.09 |
| bottom_up | 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### individuals

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| none | 281 | 0.86 | 0.95 | 0.95 | 0.95 | 0.95 |
| individual_present | 103 | 0.20 | 0.82 | 0.82 | 0.85 | 0.82 |

### text_classification

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| none | 170 | 0.59 | 0.76 | 0.77 | 0.76 | 0.75 |
| other | 67 | 0.03 | 0.12 | 0.20 | 0.13 | 0.18 |
| promotional_text | 65 | 0.00 | 0.80 | 0.76 | 0.81 | 0.72 |
| nutritional_info | 53 | 0.00 | 0.86 | 0.83 | 0.88 | 0.82 |
| technical_specs | 41 | 0.05 | 0.43 | 0.44 | 0.46 | 0.36 |
| certification_seal | 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### text_language

| classe | n | soli pixel | cross-entropy | focal | cross-entropy + GradNorm | focal + GradNorm |
|---|---|---|---|---|---|---|
| English | 85 | 0.28 | 0.62 | 0.59 | 0.60 | 0.63 |
| German | 76 | 0.29 | 0.63 | 0.59 | 0.63 | 0.65 |
| none | 65 | 0.33 | 0.72 | 0.67 | 0.71 | 0.71 |
| French | 44 | 0.00 | 0.12 | 0.04 | 0.19 | 0.31 |
| other | 35 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Italian ⚠ | 7 | 0.00 | 0.44 | 0.25 | 0.44 | 0.25 |
| Dutch ⚠ | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 |

## Precision e recall: cross-entropy

| dimensione | classe | n | precision | recall |
|---|---|---|---|---|
| subject | full_product | 85 | 0.35 | 0.79 |
| subject | packaging | 42 | 0.52 | 0.40 |
| subject | package_and_product | 38 | 0.31 | 0.11 |
| subject | multiview | 36 | 1.00 | 0.22 |
| subject | detail_closeup | 30 | 0.21 | 0.23 |
| subject | accessories | 29 | 1.00 | 0.03 |
| subject | delivery_scope | 28 | 0.64 | 0.57 |
| subject | multipack_trade_unit | 28 | 0.80 | 0.14 |
| subject | infographic | 24 | 0.30 | 0.58 |
| subject | product_grouping | 20 | 1.00 | 0.10 |
| subject | energy_label | 14 | 1.00 | 1.00 |
| subject | logo | 14 | 0.93 | 1.00 |
| subject | placeholder | 14 | 1.00 | 0.86 |
| background | white | 209 | 0.82 | 0.98 |
| background | color | 127 | 0.95 | 0.58 |
| background | scene_lifestyle | 48 | 0.77 | 0.92 |
| perspective | front_head_on | 43 | 0.45 | 0.72 |
| perspective | back_head_on | 33 | 0.51 | 0.85 |
| perspective | front_angle_left | 30 | 0.09 | 0.17 |
| perspective | front_angle_right | 30 | 0.30 | 0.53 |
| perspective | top_down | 28 | 0.43 | 0.21 |
| perspective | back_angle_left | 21 | 0.00 | 0.00 |
| perspective | back_angle_right | 21 | 0.00 | 0.00 |
| perspective | left_side | 19 | 0.43 | 0.16 |
| perspective | right_side | 17 | 0.20 | 0.06 |
| perspective | bottom_up | 16 | 0.00 | 0.00 |
| individuals | none | 281 | 0.90 | 1.00 |
| individuals | individual_present | 103 | 1.00 | 0.69 |
| text_classification | none | 170 | 0.61 | 0.99 |
| text_classification | other | 67 | 0.26 | 0.07 |
| text_classification | promotional_text | 65 | 0.87 | 0.74 |
| text_classification | nutritional_info | 53 | 0.93 | 0.79 |
| text_classification | technical_specs | 41 | 0.80 | 0.29 |
| text_classification | certification_seal | 12 | 0.00 | 0.00 |
| text_language | English | 85 | 0.48 | 0.91 |
| text_language | German | 76 | 0.58 | 0.68 |
| text_language | none | 65 | 0.82 | 0.65 |
| text_language | French | 44 | 0.60 | 0.07 |
| text_language | other | 35 | 0.00 | 0.00 |
| text_language | Italian | 7 | 1.00 | 0.29 |
| text_language | Dutch | 0 | 0.00 | 0.00 |

## Velocità e risorse

- Estrazione delle feature: 7.99 immagini/s su 1.950 immagini (batch 16, include lettura e preparazione delle immagini);
  memoria GPU di picco 27.4 GB. Nella prova da 16 immagini: 6,10 immagini/s.
- L'attivazione CUDA ottimizzata `xielu` non era installata: è stata usata la versione Python.
- Job completo (download, estrazione, training di 24 modelli di teste, valutazione): 7,4 minuti dall'avvio dello script.
- La velocità riguarda il forward completo a 32 layer; l'effetto di fermarsi a un layer intermedio non è stato misurato.

## Come riprodurre

```sh
python src/apertus-8B-heads/build_hf_job.py   # richiede data/nathan_split/
hf jobs uv run --flavor h200 --timeout 1h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e UPLOAD_REPO=<utente>/galaxus-heads-private -e BATCH=16 data/hf_jobs/hf_job_heads.py
```

Modello `swiss-ai/Apertus-v1.5-8B`, revisione `1519de0ae7f3d114879291810a58259966f64c99`; fork di Transformers al commit
`3797303dda74844e3d1f8977ff5518bb91f818b4`; torch 2.8.0. Codice del job: commit che aggiunge questo file.

## Limiti

- Manca il confronto con l'8B v2 zero-shot sulle stesse 450 immagini.
- Un solo seed, una sola configurazione di focal (γ = 2) e di GradNorm (α = 1.5), nessuna ricerca di iperparametri.
- Poco training (1.199 immagini); alcune classi hanno pochissimi esempi e nella validazione non c'è `Dutch`.
- Le conclusioni su focal e GradNorm valgono per questa prova.
