# Registro delle verifiche della demo

Solo conteggi, esiti e decisioni: nessun ID di immagine, nessuna predizione, nessun dato della competizione.

## Decisioni e default usati

| data | decisione | fonte |
|---|---|---|
| 27/11 | Q1 screenshot assenti → palette di ripiego (D.1.2); token da misurare in M2.0 solo se arrivano | risposta del team |
| 27/11 | Q2 `LLM_API_KEY` assente → zero-shot `unavailable`; il team va avvisato quando M1.6 la richiede | risposta del team |
| 27/11 | Q3 accesso ai pesi P11 sconosciuto → P11 solo metriche riportate | risposta del team |
| 27/11 | Q4 full FT: `phase2/model/last` se esiste, altrimenti `epoch-1` etichettato "intermedio" | risposta del team |
| 27/11 | Q6 nessuna GPU per la presentazione → modo 3 (zero-shot live + predizioni registrate) | risposta del team |
| 27/11 | Q12 video ≤ 2 min registrato entro le 12:45 di sabato + segmento live di 60–90 s | risposta del team |
| 27/11 | Q16 installazioni `pip`/`npm` approvate (M0.3, M0.4); download di pesi **non** approvati | risposta del team |
| 27/11 | Q23 interfaccia in inglese | risposta del team |
| 27/11 | Altre domande: default del piano (§16) | risposta del team |

## Riverifica dei fatti sensibili al tempo (27/11, ~21:30 UTC, solo elenco dei file)

| cosa | piano (20:08 UTC) | riverificato | azione |
|---|---|---|---|
| `StoryofLegend/apertus-8b-galaxus-lora` | `be79276…`, adapter `final` senza pesi, snapshot `step-00123…00738` | invariato | nessuna |
| `StoryofLegend/apertus-8b-galaxus-fullft` | `167c04f`, solo `epoch-1` | `711018b` (20:52 UTC): **`epoch-2` aggiunto** (stesse dimensioni: `decoder.safetensors` 10.469.432.256 B, `heads.pt` 16.934.087 B); `last/` e `dev_submission.csv` ancora assenti | `models.yaml` (B.7) e A.4 aggiornati alla nuova revisione; checkpoint resta `epoch-1` "intermedio" per Q4 |
| job HF `6b09cdb46c5d61fff189842a` | RUNNING, a100-large | RUNNING, a100-large, creato 18:36 UTC | nessuna |

Nota per il team: con `epoch-2` disponibile, Q4 potrebbe preferire `epoch-2` come checkpoint intermedio; la scelta resta aperta.
