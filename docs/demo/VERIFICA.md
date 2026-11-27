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

## M0 — setup (27/11)

| cosa | esito |
|---|---|
| Branch `feat/demo-app` da `main` (2ea85b1); `git diff --name-only main` tocca solo `src/demo/`, `docs/demo/`, `.gitignore` | ok |
| `.gitignore` + `check_no_data.py`: `git check-ignore -v` stampa una regola per i 3 percorsi di prova; lo script fallisce su CSV, PNG, ID e chiave finti messi in stage (4 casi su 4), passa sullo stage pulito | ok |
| `check_no_data.py --all` segnala `challenge_description.md` (2 occorrenze del formato ID, file del template già tracciato, non nostro): non modificato | nota |
| M0.3: installati nel `.venv` con `uv` (il venv non ha `pip`): fastapi 0.165.0, uvicorn 0.54.0, python-multipart 0.0.34, pytest 9.2.0, pytest-asyncio 1.5.0. pandas 3.1.1, pillow 13.0.0, pydantic 2.14.4 invariati (verificato con `--dry-run` e dopo l'installazione) | ok |
| M0.4: Vite 8 + React 19 + TypeScript + Tailwind 4 (`@tailwindcss/vite`), versioni esatte nel lockfile; `npm run build` ok, `dist/` ignorato; `npm run dev` serve la pagina (HTTP 200, `DemoBadge` presente nel modulo e nel bundle) | ok |
| Verifica **visiva** del badge nel browser | **non fatta**: estensione Chrome non connessa; Playwright non ha i browser scaricati (download non autorizzato) |
| Token di design | tutti di **ripiego** Tailwind (Q1: nessuno screenshot); in `tokens.css` solo 5 token per ora, il resto in M2.0 |
