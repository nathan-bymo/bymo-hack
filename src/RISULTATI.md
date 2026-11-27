# Indice di tutti i risultati

Aggiornato al 27 novembre 2026. Serve per il report finale. **Ogni tabella vale sul proprio set di immagini: non
confrontare numeri di set diversi.** Nessuno di questi è un punteggio della classifica Kaggle.

## Set di valutazione

| set | immagini | note |
|---|---|---|
| coverage | 2.532 | prompt-dev + prompt-test + 169 immagini per le classi rare |
| base | 2.363 | prompt-dev + prompt-test, distribuzione originale |
| prompt-test | 1.193 | non usato per scegliere il prompt |
| prime 600 di prompt-dev | 600 | primi test dei prompt |
| validation_450 (Nathan) | 450 | bilanciata per classe, presa dalle immagini di training di coverage |

## Numeri principali (macro F1)

| metodo | set | macro F1 |
|---|---|---|
| 8B zero-shot, prompt breve v1 | coverage | 0.316 |
| 8B zero-shot, `docs/system_prompt.txt` | coverage | 0.229 |
| 70B zero-shot, `docs/system_prompt.txt` | coverage | 0.304 |
| 8B zero-shot, v2 con descrizione | coverage | 0.528 |
| 70B zero-shot, v2 con descrizione | coverage | 0.583 |
| solo pixel + teste (soglia, non Apertus) | coverage | 0.366 |
| solo pixel + teste (soglia, non Apertus) | validation_450 | 0.257 |
| **8B congelato + teste, cross-entropy** | validation_450 | **0.542** |
| 8B congelato + teste, cross-entropy + GradNorm | validation_450 | 0.537 |
| 8B congelato + teste, focal | validation_450 | 0.528 |
| 8B congelato + teste, focal + GradNorm | validation_450 | 0.501 |

La riga dei soli pixel su coverage usa tutto il training disponibile e 8 epoche; quella su validation_450 le 1.500
immagini e 30 epoche.

## Dove sono i dettagli

| file | contenuto |
|---|---|
| [apertus-8B-benchmark/prompt_eval_coverage.md](apertus-8B-benchmark/prompt_eval_coverage.md) | prompt zero-shot su coverage, base e prompt-test: dimensioni, classi, intervalli, velocità |
| [apertus-8B-benchmark/prompt_eval_per_class_600.md](apertus-8B-benchmark/prompt_eval_per_class_600.md) | le 9 varianti di prompt sulle prime 600, per classe |
| [apertus-8B-benchmark/prompt_eval.md](apertus-8B-benchmark/prompt_eval.md) | diagnosi del prompt, prove piccole, velocità del gateway |
| [apertus-8B-benchmark/teacher_student_zeroshot.md](apertus-8B-benchmark/teacher_student_zeroshot.md) | accordo 70B e 8B zero-shot (analisi preliminare alla distillazione) |
| [apertus-8B-heads/results_small_h200.md](apertus-8B-heads/results_small_h200.md) | teste su H200: risultati, intervalli, layer, configurazione, velocità |
| [apertus-8B-heads/results_small_h200_per_class.md](apertus-8B-heads/results_small_h200_per_class.md) | teste: classi e matrici di confusione delle 4 varianti |
| [apertus-8B-heads/README.md](apertus-8B-heads/README.md) | metodo, verifiche del codice, controlli con soli pixel |

## Non ancora misurato

- Prompt v2 sulle 450 di Nathan.
- Teste sul set completo (training su `training_rows`, valutazione su coverage).
- Pesi per le classi rare nelle teste.
- Velocità fermandosi a un layer intermedio.
- v2 completa senza contesto prodotto.
- Punteggio Kaggle sul dev.

## File grezzi, fuori dal repository

Predizioni, risposte del modello, feature, pesi delle teste e log contengono ID o derivano dai dati della
competizione: non vanno nel repository.

- `data/runs/`: risposte zero-shot dell'API e predizioni locali (solo sul Mac che ha eseguito i test).
- `data/hf_results/` e `data/hf_jobs/`: feature, predizioni, teste e log del job H200; una copia di feature,
  predizioni, teste e log è in un repository Hugging Face privato dell'account che ha lanciato il job.
- Vanno cancellati i dati della competizione dopo l'hackathon, come chiedono le regole.
