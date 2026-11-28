# LoRA su SigLIP, senza DINOv3 (run `siglip-lora-1128-0815`, 28 novembre)

Domanda: allenare anche l'encoder di immagini, finora congelato, migliora il modello? Solo numeri aggregati.

## Ricetta

- Partenza: `final-abt` (SigLIP 2 base 224 + VQ + OCR + LoRA r32 su 24 layer + teste + pooling).
- Nuovo: LoRA r16 su tutti i layer di SigLIP (2,65 M parametri, learning rate 5e-5), che parte da zero; le immagini
  stanno in memoria come pixel e SigLIP gira dentro il modello. In predizione la LoRA è fusa nell'encoder.
- Si allenano insieme anche LoRA di Apertus (3e-5), proiettore (5e-5), teste e pooling (1,5e-4).
- Dati: 10.683 immagini fuori da prompt-test, 11.204 esempi per epoca (ripetizioni solo per le classi di testo
  rare), 2 epoche, batch 8 × 2, nessuna immagine sintetica.
- **Il checkpoint è scelto su prompt-test:** il suo punteggio su prompt-test non è più una prova indipendente.
  La misura esterna è il dev di Kaggle.

## Risultati

| passo | macro F1 su prompt-test |
|---|---|
| partenza, con le immagini lette dentro il modello | 0,8207 (`final-abt`: 0,8214) |
| epoca 0,5 | 0,8280 |
| epoca 1 | 0,8351 |
| epoca 1,5 | 0,8355 |
| **epoca 2 (scelto)** | **0,8369** |
| checkpoint ricaricato, predizione dall'immagine | 0,8356 |

| ricaricato, prompt-test | macro | weighted | subject | background | perspective | individuals | text class. | language |
|---|---|---|---|---|---|---|---|---|
| LoRA su SigLIP | 0,836 | 0,923 | 0,860 | 0,873 | 0,777 | 0,970 | 0,793 | 0,740 |

| Kaggle dev (5.261 immagini) | punteggio |
|---|---|
| mixed (SigLIP + DINOv3 + VQ) | 0,84215 |
| **LoRA su SigLIP** (submission 59043423) | **0,83526** |
| `final-abt`, il punto di partenza | 0,83177 |
| P15: `final-abt` + 1 epoca su tutti i dati, encoder congelato | 0,83144 |

## Cosa dice

- Sul dev è +0,0035 sopra il punto di partenza e −0,0069 sotto il mixed. Il vantaggio sul punto di partenza è
  piccolo.
- **Non è un'ablazione:** insieme alla LoRA di SigLIP si sono allenate anche le altre parti, con dati e ripetizioni
  diversi da P15. Il guadagno non si può attribuire solo alla LoRA di SigLIP. Un controllo con la stessa ricetta ed
  encoder congelato non è stato fatto.
- Sul prompt-test +0,015 rispetto alla partenza diventa +0,0035 sul dev.

## Velocità, memoria, costo (H200, dall'immagine al CSV, OCR compreso)

| | prompt-test | dev |
|---|---|---|
| immagini/s | 6,75 | 7,43 |
| memoria di picco | 79,8 GB | 79,8 GB |

La LoRA fusa non aggiunge calcoli per immagine. Training: 37,1 minuti, picco 78,7 GB. Job intero (preparazione,
training, due predizioni): 63,9 minuti, circa 5,96 $ su Modal (differenza del report di spesa del giorno).
Nessuna immagine è stata sostituita.

## Pesi e riproduzione

- Pesi (con `vision_lora/`): repository privato `MANNIK14/galaxus-vlh-private/siglip-lora-1128-0815/model/best`,
  copia sul Mac con SHA-256 identico.
- Job: `build_hf_job.py` + `modal_run.py` con `PHASE=1 PART=all_but_test OCR=1 SIDE=256 TARGETED_CONTINUATION=1
  INIT_FROM=final-abt/model/best PREDICT_DEV=1` e `TRAIN_ARGS="--epochs 2 --batch-size 8 --accumulation 2
  --lr-lora 3e-5 --lr-projector 5e-5 --lr-heads 1.5e-4 --vision-lora-rank 16 --lr-vision 5e-5 --max-minutes 45"`.
