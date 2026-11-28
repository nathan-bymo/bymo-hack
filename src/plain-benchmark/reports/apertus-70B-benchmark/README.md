# Apertus 1.5 70B: risultati

I test del 70B sono stati fatti insieme a quelli dell'8B e i report completi stanno in `../apertus-8B-benchmark/`.
Questa pagina li riassume. Tutti i test sono **zero-shot** (nessun training), con il modello base via API Keera,
immagini a 512², temperatura 0. Nessuno di questi è un punteggio Kaggle. Indice di tutte le prove:
[../../training/EXPERIMENTS.md](../../training/EXPERIMENTS.md).

## Set coverage (2.532 immagini) — prova P4

| modello e prompt | coverage | base (2.363) | prompt-test (1.193) |
|---|---|---|---|
| **70B, prompt v2 con descrizione** | **0.583** | 0.587 | 0.590 |
| 70B, `shared JSON prompt` | 0.304 | 0.299 | 0.294 |
| 8B, prompt v2 con descrizione | 0.528 | 0.513 | 0.502 |
| 8B, `shared JSON prompt` | 0.229 | 0.230 | 0.225 |

Per dimensione, 70B v2 su coverage: subject 0.546, background 0.631, perspective 0.274, individuals 0.899,
text_classification 0.618, text_language 0.529. Velocità sul gateway condiviso: 2,8 immagini/s (v2, 7 richieste
per immagine), 4,1 immagini/s (prompt condiviso).

Differenza con intervallo al 95%: 70B v2 − 8B v2 = +0.055 [+0.041, +0.069] su coverage, +0.088 [+0.059, +0.111] su
prompt-test.

## Prime 600 immagini di prompt-dev — prova P3

| modello e prompt | macro F1 |
|---|---|
| 70B, v2 con descrizione | 0.587 |
| 70B, v2 a domande separate (senza descrizione) | 0.454 |

## Dove trovare i dettagli

| file | contenuto 70B |
|---|---|
| [prompt_eval_coverage.md](../apertus-8B-benchmark/prompt_eval_coverage.md) | tabelle per dimensione e **per classe** (coverage, base, prompt-test), precision/recall, intervalli, velocità |
| [prompt_eval_per_class_600.md](../apertus-8B-benchmark/prompt_eval_per_class_600.md) | per classe sulle 600, con precision e recall |
| [prompt_eval.md](../apertus-8B-benchmark/prompt_eval.md) | diagnosi del prompt su `subject` (8B e 70B), velocità iniziale del gateway |
| [teacher_student_zeroshot.md](../apertus-8B-benchmark/teacher_student_zeroshot.md) | accordo 70B–8B e affidabilità della confidenza del 70B (analisi per la distillazione) |

## Cosa ne abbiamo concluso

- Il 70B zero-shot è il migliore tra i metodi senza training, ma resta molto sotto i modelli allenati
  (LoRA + teste 256²: coverage 0.781, dev Kaggle 0.796).
- `perspective` (0.274) e la lingua `none` sono i punti più deboli del 70B.
- Come maestro per la distillazione è poco affidabile in `perspective`: anche con confidenza ≥ 0.90 ha ragione nel
  63% dei casi. Per questo la distillazione dal 70B zero-shot non è stata fatta.
- Il 70B a runtime non è sostenibile per la challenge (2,8 immagini/s via API condivisa, costo alto in proprio).
