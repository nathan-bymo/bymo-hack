# Analisi preliminare maestro–studente: 70B e 8B zero-shot (27 novembre 2026)

**Non è un test di distillazione:** nessun modello è stato allenato. Serve a vedere dove il 70B potrebbe aiutare o
danneggiare. Le conclusioni riguardano l'8B zero-shot; dopo un LoRA i suoi errori saranno diversi. Entrambi usano il
prompt v2 con descrizione. Le percentuali sono sulle celle valutate e sono dominate dalle classi frequenti: non
sostituiscono la macro F1.

## Accordo: coverage

| dimensione | celle | entrambi giusti | solo 70B giusto | solo 8B giusto | entrambi sbagliati, stessa risposta | entrambi sbagliati, risposte diverse |
|---|---|---|---|---|---|---|
| subject | 2384 | 53.3% | 13.9% | 4.9% | 19.4% | 8.5% |
| background | 2078 | 80.7% | 3.4% | 2.3% | 13.3% | 0.3% |
| perspective | 1638 | 28.1% | 23.4% | 12.0% | 13.4% | 23.1% |
| individuals | 2080 | 96.6% | 1.4% | 1.1% | 0.9% | 0.0% |
| text_classification | 2295 | 72.2% | 9.3% | 8.8% | 6.2% | 3.4% |
| text_language | 1146 | 43.5% | 6.6% | 10.9% | 25.9% | 13.0% |

## Accordo: base (distribuzione originale delle classi)

| dimensione | celle | entrambi giusti | solo 70B giusto | solo 8B giusto | entrambi sbagliati, stessa risposta | entrambi sbagliati, risposte diverse |
|---|---|---|---|---|---|---|
| subject | 2222 | 54.4% | 13.8% | 5.0% | 19.1% | 7.7% |
| background | 1931 | 80.5% | 3.5% | 2.2% | 13.5% | 0.4% |
| perspective | 1525 | 27.8% | 23.7% | 11.9% | 13.4% | 23.2% |
| individuals | 1933 | 96.7% | 1.4% | 1.1% | 0.7% | 0.0% |
| text_classification | 2126 | 72.9% | 9.3% | 8.5% | 5.9% | 3.4% |
| text_language | 1024 | 44.1% | 6.4% | 10.4% | 26.9% | 12.1% |

## Accordo: prompt-test (non usato per scegliere il prompt)

| dimensione | celle | entrambi giusti | solo 70B giusto | solo 8B giusto | entrambi sbagliati, stessa risposta | entrambi sbagliati, risposte diverse |
|---|---|---|---|---|---|---|
| subject | 1139 | 55.1% | 12.9% | 4.9% | 19.5% | 7.6% |
| background | 973 | 78.7% | 4.2% | 2.7% | 13.9% | 0.5% |
| perspective | 763 | 28.6% | 22.8% | 10.7% | 14.8% | 23.1% |
| individuals | 974 | 97.2% | 1.2% | 0.9% | 0.6% | 0.0% |
| text_classification | 1073 | 73.0% | 9.1% | 9.1% | 5.2% | 3.5% |
| text_language | 518 | 42.9% | 7.3% | 10.6% | 27.2% | 12.0% |

## Il 70B sa quando è affidabile? (coverage)

Confidenza: probabilità della classe scelta. Ogni probabilità è condizionata alla descrizione generata dal 70B.

| dimensione | confidenza < 0.60: celle, accuratezza | 0.60–0.90 | ≥ 0.90 |
|---|---|---|---|
| subject | 167, 32.9% | 446, 42.8% | 1771, 76.6% |
| background | 22, 63.6% | 93, 55.9% | 1963, 85.7% |
| perspective | 243, 28.4% | 455, 39.3% | 940, 63.4% |
| individuals | 5, 20.0% | 12, 41.7% | 2063, 98.5% |
| text_classification | 184, 42.9% | 475, 60.8% | 1636, 91.9% |
| text_language | 88, 30.7% | 209, 31.1% | 849, 56.9% |
