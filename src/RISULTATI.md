# Registro delle prove e dei risultati

Aggiornato al 27 novembre 2026. Serve per il report finale. Ogni report ha in cima una **scheda della prova**.

**Regole di lettura**

1. Ogni numero vale sul proprio set di immagini: **non confrontare set diversi**.
2. "Zero-shot": nessun training. "Teste": Apertus congelato, si allenano solo le teste. "LoRA": Apertus adattato.
3. La macro F1 è la media delle sei dimensioni, ognuna media delle sue classi.
4. Nessuno di questi è un punteggio della classifica Kaggle, salvo la sezione "Kaggle" qui sotto.

## Set di valutazione

| set | immagini | cos'è |
|---|---|---|
| coverage | 2.532 | prompt-dev, prompt-test e 169 immagini per le classi rare; nessun modello va allenato sui suoi prodotti |
| base | 2.363 | prompt-dev + prompt-test, distribuzione originale delle classi |
| prompt-test | 1.193 | parte di coverage non usata per scegliere il prompt |
| prime 600 di prompt-dev | 600 | primi test dei prompt |
| validation_450 (Nathan) | 450 | **bilanciata** per classe; presa dalle immagini di training di coverage, quindi valida solo per modelli che non le hanno studiate |

## Elenco delle prove

| prova | domanda | modello e metodo | dati di training | valutata su | chi, dove | report |
|---|---|---|---|---|---|---|
| P0 | cosa supporta il gateway e quanto è veloce | 8B e 70B base, zero-shot | nessuno | immagine sintetica | API Keera | [prompt_eval](apertus-8B-benchmark/prompt_eval.md) |
| P1 | regole lunghe e lettere nel prompt v1 | 8B zero-shot, 4 varianti | nessuno | 600 | API Keera | [per classe 600](apertus-8B-benchmark/prompt_eval_per_class_600.md) |
| P2 | perché la v1 sbaglia `subject` | 8B e 70B zero-shot, 15 configurazioni | nessuno | 103 (solo subject) | API Keera | [prompt_eval](apertus-8B-benchmark/prompt_eval.md) |
| P3 | prompt v2 a domande separate | 8B e 70B zero-shot | nessuno | 600 | API Keera | [per classe 600](apertus-8B-benchmark/prompt_eval_per_class_600.md) |
| P4 | confronto dei prompt sul set grande | 8B e 70B zero-shot, 5 varianti | nessuno | coverage, base, prompt-test | API Keera | [coverage](apertus-8B-benchmark/prompt_eval_coverage.md) |
| P5 | il 70B può fare da maestro | risposte di P4 | nessuno | coverage | analisi locale | [maestro-studente](apertus-8B-benchmark/teacher_student_zeroshot.md) |
| P6 | la pipeline delle teste funziona; soglia minima | **pixel** 24×24 + teste (non Apertus), 8 epoche | 8.471 + 873 | coverage | Mac | [README teste](apertus-8B-heads/README.md) |
| P7 | soglia minima sullo split piccolo | **pixel** 24×24 + teste, 4 varianti, 30 epoche | 1.199 + 301 | validation_450 | Mac | [README teste](apertus-8B-heads/README.md) |
| P8 | le teste su Apertus funzionano; focal e GradNorm | 8B **congelato** + teste, 4 varianti, 1 seed | 1.199 + 301 | validation_450 | H200 | [teste H200](apertus-8B-heads/results_small_h200.md), [per classe](apertus-8B-heads/results_small_h200_per_class.md) |
| P9 | pesi per classe; focal contro cross-entropy | 8B congelato + teste, 6 varianti, 5 seed | 1.199 + 301 | validation_450 | Mac, feature di P8 | [pesi per classe](apertus-8B-heads/results_weights_small.md) |
| P10 | LoRA sullo split piccolo | 8B + **LoRA**, 256², JSON | 1.500 | validation_450, coverage | Kristian, H200 | [confronto LoRA](confronto_lora.md) |
| P11 | LoRA e teste insieme, training completo | 8B + **LoRA + teste**, 256² | 9.344 | coverage, base, prompt-test | H200 | [LoRA + teste](apertus-8B-lora-heads/results_full_h200.md) |

Per le teste, "1.199 + 301" vuol dire: immagini per allenare i pesi, più immagini per scegliere layer ed epoca.

## Risultati principali

### Su validation_450

| prova | metodo | macro F1 |
|---|---|---|
| P7 | soli pixel + teste (focal + GradNorm) | 0.257 |
| P8 | 8B congelato + teste, focal + GradNorm | 0.501 |
| P8 | 8B congelato + teste, focal | 0.528 |
| P8 | 8B congelato + teste, cross-entropy + GradNorm | 0.537 |
| P8 | 8B congelato + teste, cross-entropy | 0.542 |
| P9 | cross-entropy, media 5 seed (senza pesi / pesi moderati / pesi forti) | 0.539 / 0.538 / 0.557 |
| P9 | focal, media 5 seed (senza pesi / pesi moderati / pesi forti) | 0.532 / 0.538 / 0.547 |
| **P10** | **8B + LoRA** | **0.696** |

### Su coverage

| prova | metodo | macro F1 |
|---|---|---|
| P4 | 8B zero-shot, `docs/system_prompt.txt` | 0.229 |
| P4 | 70B zero-shot, `docs/system_prompt.txt` | 0.304 |
| P4 | 8B zero-shot, prompt breve v1 | 0.316 |
| P6 | soli pixel + teste (non Apertus) | 0.366 |
| P4 | 8B zero-shot, v2 | 0.528 |
| P4 | 70B zero-shot, v2 | 0.583 |
| P10 | 8B + LoRA con JSON, allenato su 1.500 immagini | 0.734 |
| **P11** | **8B + LoRA + teste, allenato su 9.344 immagini** | **0.781** (base 0.786, prompt-test 0.791) |

## Kaggle (dev, classifica pubblica)

| modello | dev pubblico |
|---|---|
| Qwen 3.5 9B finetunato (baseline) | 0.917 |
| Gemini 4 Argon zero-shot | 0.834 |
| Gemini 3.8 Flash zero-shot | 0.806 |
| **P11, LoRA + teste** | **0.796** |

## Cosa sappiamo

- La v2 zero-shot migliora molto sulla v1, anche su prompt-test (P4).
- Le teste su Apertus congelato superano di molto la soglia dei pixel (P7, P8).
- La focal non fa meglio della cross-entropy, anche a parità di pesi per classe (P9). GradNorm è stato provato solo in
  P8, con un seed e pochi passi di training: lì non ha aiutato.
- I pesi forti per classe danno circa +0.02 sulle 450 (P9), ma la scelta sulla validazione interna non li indica:
  da confermare su altre immagini.
- **LoRA + teste sul training completo (P11) è il metodo migliore misurato finora su coverage.** Il distacco da P10 non si
  attribuisce alle teste: P11 ha circa sei volte più immagini.
- Tra i metodi sullo split piccolo, sulle 450 il migliore è il LoRA (P10): +0.154 sulle 450 rispetto alle teste di P8. **Le condizioni non
  sono uguali** (risoluzione, numero di immagini, augmentation), quindi il distacco non si attribuisce solo al metodo.

## Non ancora misurato

- Prompt v2 sulle 450.
- Teste con Apertus congelato sul set completo.
- LoRA con uscita JSON sul set completo: serve per confrontare teste e testo a parità di dati.
- Teste sopra il modello con LoRA, e confronto con le teste congelate a parità di condizioni.
- Velocità fermandosi a un layer intermedio.
- v2 completa senza contesto prodotto.
- Punteggio Kaggle sul dev.

## File grezzi, fuori dal repository

Predizioni, risposte del modello, feature, pesi delle teste e log contengono ID o derivano dai dati della
competizione: non vanno nel repository.

- `data/runs/`: risposte zero-shot dell'API, predizioni e teste di P9 (solo sul Mac che ha eseguito i test).
- `data/hf_results/`, `data/hf_jobs/`: feature, predizioni, teste e log di P8; una copia di feature, predizioni,
  teste e log è in un repository Hugging Face privato dell'account che ha lanciato il job.
- `data/share_heads/`: pesi delle teste in un formato che si carica con `torch.load(weights_only=True)`.
- Dopo l'hackathon i dati della competizione vanno cancellati, come chiedono le regole.
