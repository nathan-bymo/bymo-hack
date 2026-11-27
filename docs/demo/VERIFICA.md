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

## M2.0 — token di design (27/11)

Tutti i token di `src/demo/frontend/src/styles/tokens.css` sono di **ripiego** (palette Tailwind, dichiarata riga per
riga con commento `ripiego:`); nessuno è misurato perché non ci sono screenshot (Q1). I colori dei chip di confidenza
e del badge "invalid" sono nostri (non Galaxus). Procedura di misura: D.1.1, da eseguire se gli screenshot arrivano.
Nota tecnica: Tailwind 4 non emette le variabili di tema non usate, quindi i ripiego sono esadecimali letterali.

## Stato al 27/11 (sera) — M1…M5, M7 (senza GPU)

Commit di riferimento: ramo `feat/demo-app`. Nessun push, job, download di pesi o azione Kaggle eseguiti.

### Test (solo conteggi ed esiti)
| cosa | esito |
|---|---|
| pytest backend (`make -C src/demo test`) | **106 passati**, 0 falliti |
| E2E Playwright (Chrome installato, fixture sintetiche, registrazioni **segnaposto** solo nel test) | **1 test, 9 passi, passato** (~7 s): badge, menu, upload di 3 immagini + 1 file non valido (rifiuto `UNSUPPORTED_MEDIA_TYPE` mostrato), run modello A, cambio modello B **senza nuovo `POST /api/images`**, confronto con disaccordo evidenziato e filtro, dettaglio con correzione → `Corrected` nella coda, anteprima storefront (regola illustrativa, valori rivisti), dashboard (banner, linea 0.917 con etichetta, badge rosso del full FT, Kaggle dev solo per P11, matrice, calibrazione n/a, costo) |
| `npm run build` (tsc + vite) | ok |
| `make -C src/demo demo`-equivalente: API su :8010 serve il build; `/`, `/image-classification`, `/image-classification/performance` rispondono 200 HTML; `/api/*` inesistente → JSON `NOT_FOUND` | ok |
| Guardia dev/final: copia identica di un file di `dev_images` → 422 `FORBIDDEN_IMAGE`, 0 file salvati; indice di 5.260 hash unici (5.261 file) | ok |
| Parità dei corpi di richiesta zero-shot con `eval_prompt.run_variant` (7 richieste, con e senza contesto, gateway finto) | **identici** |
| Import metriche riportate: 20 righe; valori rilevati da file riga per riga e confrontati con `models.yaml`; discrepanza → errore | ok |
| Controllo costo: P11 13,2 img/s su H200 a 5 $/h → 0,105 $ per 1.000, 10,5 $ per 100.000 (come `results_full_h200.md:159-160`) | ok |

### Decisioni e scostamenti dal piano (piano corretto dove serve)
1. **Kaggle dev di P11 = 0.796** è già nel repo (`results_full_h200.md:150-162`, commit 5e66333 del 27/11 alle 21:31): il piano lo
   dava assente. Inserito in `models.yaml` con fonte e nota "da confermare su Kaggle"; piano A/§3/§9 aggiornati.
2. `eval_results.provenance` ammette anche `na` (righe "seen during training" senza numeri); schema §8 aggiornato.
3. La chiave di cache delle predizioni include anche `image_id` (la stessa immagine in due sessioni non collide).
4. Upload: se **nessun** file è accettato la richiesta fallisce con lo stato del primo rifiuto (413/415/422) e
   `details.rejected`; se almeno uno è accettato → 201 con `rejected[]`.
5. Interfaccia `ModelAdapter` semplificata: `predict(images, use_context, batch_size, on_result, on_error, should_cancel)`
   (un solo metodo bloccante per il job manager) invece di `preprocess` + `predict`.
6. Le card della dashboard compaiono solo per i modelli con almeno una riga sul set scelto (P8/P9 solo su `validation_450`, Q20).
7. P11: `validation_450` è in `seen_sets` (è dentro `training_rows`) → "seen during training — invalid".
8. Default usati (§16): Q5 worktree, Q7 70B incluso, Q8–Q11, Q13 `auto`/`none → -`, Q14, Q17 non eseguito, Q18–Q22.
9. Moduli `src/` non toccati: `git diff --name-only main` solo in `src/demo/`, `docs/demo/`, `.gitignore`.

### Verifica visiva (M6.4) — parziale, honesta
Nessuno screenshot di riferimento (Q1): **nessun confronto fedele** possibile. Controllati a occhio sugli screenshot
Playwright (1440×900, densità Normale): barra titolo e linea d'accento, sidebar con voce attiva e pulsante viola inattivo,
toolbar con menu, tabella dense, pannello "Image type" azzurro con robot e 6 tendine su due colonne (Cancel a sinistra, Save
a destra), badge sempre visibile in tutte le schermate, nessun logo Galaxus, chip di confidenza con numero + icona.
**Non verificati:** contrasto AA con uno strumento, densità Presentazione su TV. La schermata 6 (storefront, regola illustrativa) è stata guardata sullo screenshot e coperta dall'E2E.

### Non verificato / bloccato
| cosa | bloccato da |
|---|---|
| Zero-shot **live** end-to-end (M1.6 manuale) | `LLM_API_KEY` non in `.env`: codice e parità dei corpi verificati solo con gateway finto |
| Modelli allenati in modalità **registrato** con risultati veri; piano B (M7.4) | serve un host GPU (o un collega) per `record.py` e un ok |
| Esecuzione reale di worker, `record.py`, `evaluate.py`, `parity.py`, misure di velocità (M6.5) | GPU/gateway + ok esplicito (scope C). Verificati solo con la famiglia finta: protocollo worker↔proxy (token, load/predict/unload, errori), LRU/budget VRAM, rifiuti di `evaluate.py` (dev/final/contaminati), catena export→record→rescan→predict, calcolo metriche su un dataset sintetico |
| P11 pesi (Q3), repo HF non noto | accesso |
| Confidenza e matrici di confusione ricalcolate; calibrazione | predizioni per immagine su set etichettato |
| Adapter reali P10/P11/full FT (scritti dai file del repo, **mai eseguiti**) | GPU; restano "parity not verified" |
| `parity.py` per P11/full FT (scritto, non eseguito); P10 non implementato | GPU + ok |
