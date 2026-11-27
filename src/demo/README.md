# Demo "Image classification (AI)" — Team Bymo · Hack Apertus

Applicazione dimostrativa (frontend React + backend FastAPI) che mostra come i modelli Apertus del team
pre-compilerebbero il pannello **"Image type"** della pagina "Processed image" dell'ERP di Galaxus: una persona
rivede, corregge e salva. Piano completo: `docs/demo/PIANO_DEMO.md`; verifiche: `docs/demo/VERIFICA.md`.

**Cosa non è:** nessun collegamento all'ERP reale (guscio statico con testi segnaposto, nessun logo o asset
Galaxus). Le correzioni umane sono **solo dati della demo**: nessun endpoint, script o formato le trasforma in dati
di training.

## 1. Requisiti
- Python del `.venv` del repo (3.12) e Node 22. Chrome installato (solo per i test E2E).
- Modelli locali (non servono per la presentazione in modalità 3): GPU ≥ 24 GB [STIMA] e le versioni di
  `backend/requirements-gpu.txt`. Su questa macchina non c'è GPU.

## 2. Setup
```sh
uv pip install --python .venv/bin/python -r src/demo/backend/requirements.txt   # il .venv non ha pip
cd src/demo/frontend && npm ci
```

## 3. Credenziali
- `LLM_API_KEY` (gateway Keera) in `.env` alla radice (ignorato da git, vedi `.env.example`) o nell'ambiente. Senza,
  i modelli zero-shot risultano `unavailable` con il motivo. La chiave non viene mai stampata né loggata.
- `hf auth login` serve solo per scaricare pesi privati (non fatto dalla demo senza conferma esplicita).

## 4. Modelli e pesi
`src/demo/models.yaml` è il registro: ogni valore ha una fonte, i campi ignoti valgono `[DA VERIFICARE]` e rendono il
modello `unavailable`. **Nessun peso viene scaricato in automatico.** Il full fine-tune si legge da un worktree
(`git worktree add ../hack-apertus-fullft feat/apertus-8b-fullft-heads`, variabile `DEMO_FULLFT_SRC`). I worker GPU
(`demo_worker`, uno per famiglia) non sono ancora implementati: richiedono un host GPU e un ok esplicito.

## 5. `kaggle_dev.score`
Si compila **solo a mano** in `models.yaml`, con `source`, data e nota. Oggi solo P11 (0.796, da
`results_full_h200.md:150-162`, **da confermare su Kaggle**); il full fine-tune non ha metriche locali valide.

## 6. Avvio
```sh
make -C src/demo demo   # build del frontend + API su http://127.0.0.1:8010 (presentazione)
make -C src/demo dev    # API :8010 + Vite :5173
```
Modi di esecuzione: (1) tutto su un host GPU; (2) laptop + worker remoto via tunnel SSH
(`DEMO_WORKER_URL_<FAMIGLIA>`); (3) **laptop senza GPU** (default): zero-shot live via Keera + modelli allenati in
modalità *registrato*.

## 7. Modalità registrato
`scripts/export_session.py --session <id>` crea `data/demo/export/<id>/` (immagini + `manifest.json` senza nomi) →
su un host GPU si producono `recorded/<modello>/<run>/{run.json,predictions.jsonl}` (script `record.py`: **da fare**
con la GPU) → si copiano in `data/demo/recorded/` e si fa `POST /api/recorded/rescan` (o si riavvia). La UI mostra
"Recorded <data> on <GPU>" ovunque compaia un risultato registrato. Lookup per sha256 + hash del contesto: se manca,
`RECORDED_MISSING`.

## 8. Metriche
Numeri riportati dai file del repo, riletti con la loro riga (`scripts/import_reported.py`); ogni numero ha set,
fonte, dati di training e badge (validità + provenienza). `dev` e `final` sono rifiutati. Il full fine-tune mostra
"seen during training — invalid". `evaluate.py` (ricalcolo con `metric.py` ufficiale) **non è ancora scritto**:
richiede GPU o chiave e un ok esplicito.

## 9. Immagini
Mai immagini dev o final (guardia sugli sha256: `scripts/build_guard_index.py`). Test con immagini sintetiche.
Nessuna immagine, CSV, database o screenshot in git: `scripts/check_no_data.py` prima di ogni commit.

## 10. Test
```sh
make -C src/demo test    # pytest del backend
make -C src/demo e2e     # Playwright (Chrome installato), fixture sintetiche e registrazioni segnaposto
make -C src/demo check   # controllo anti-dati sullo stage
```

## 11. Problemi comuni
- `MODEL_UNAVAILABLE`: chiave mancante, pesi assenti o nessun worker GPU (il motivo è nella risposta).
- `GPU_OOM`: un solo modello residente per worker; ≥ 24 GB.
- Gateway lento (≈ 8–11 s/immagine, condiviso): usare le predizioni in cache o registrate.
- HEIC/GIF/TIFF rifiutati: esportare le foto in JPEG.
- Porta occupata: `PORT=8011 make -C src/demo demo`.

## 12. Pulizia
`make -C src/demo clean-data` cancella `data/demo/` (database, upload, registrazioni): da fare dopo l'hackathon
insieme ai dati della competizione.
