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
modello `unavailable`. **Nessun peso viene scaricato in automatico.** Pesi nel repo (Git LFS, `local_path`): P11 in
`weights/lora_heads_256_p11/`, modello finale di Nathan in `weights/apertus-8B-vadapter-lora-heads-final/`. Il full fine-tune si legge da un worktree
(`git worktree add ../hack-apertus-fullft feat/apertus-8b-fullft-heads`, variabile `DEMO_FULLFT_SRC`). I worker GPU
(`python -m demo_worker.server --family <lora_generative|lora_heads|fullft_heads> --port <p>`, uno per famiglia) sono
scritti e il **protocollo** è testato con una famiglia finta (`--dry`, solo test); gli adapter reali **non sono stati
mai eseguiti** (nessuna GPU in sviluppo): restano "parity not verified" finché `scripts/parity.py` non passa su un host
GPU. Pesi: `scripts/fetch_weights.py --model <id>` stampa il piano e scarica solo con `--yes`. Il modello finale ha la famiglia
`vadapter_heads` (SigLIP 2 + OCR docTR, `requirements-gpu.txt` include `python-doctr`).

## 5. `kaggle_dev.score`
Si compila **solo a mano** in `models.yaml`, con `source` (file del repo), `source_label` e nota. Oggi: modello finale
0.83177, full fine-tune 0.80473, P11 0.796 (submission del team su Kaggle, 27/11). Riferimenti dello stesso set in
`defaults.kaggle_references` (Qwen 0.917, Gemini 4 Argon 0.834, Gemini 3.8 Flash 0.806).

## 6. Avvio
```sh
make -C src/demo demo   # build del frontend + API su http://127.0.0.1:8010 (presentazione)
make -C src/demo dev    # API :8010 + Vite :5173
```
Modi di esecuzione: (1) tutto su un host GPU; (2) laptop + worker remoto via tunnel SSH
(`DEMO_WORKER_URL_<FAMIGLIA>`); (3) **laptop senza GPU** (default): zero-shot live via Keera + modelli allenati in
modalità *registrato*.

### Pagine (28/11)
- `/shop`, `/shop/:id` — **Storefront** (lato cliente): negozio di prova costruito dal catalogo; mostra solo le foto
  pubblicate e applica le regole scelte nel catalogo.
- `/catalog`, `/catalog/:id` — **Catalog** (lato admin): prodotti, foto, pubblicazione e regole della vetrina.
- `/` — **Demo**: foto a sinistra; a destra i modelli (riquadri con punteggio e stato) e una tabella con le sei
  domande in righe e i modelli in colonne. A 1440 × 900 sta tutto in uno schermo. Rosso solo per "incerto" (< 60%) e
  "modelli in disaccordo". Dettagli prodotto facoltativi sotto la foto; in fondo il pannello per pubblicare la foto nel
  catalogo. Zero-shot live; modelli allenati live (worker GPU) o registrati.
- Le pagine **Results** e **How it works** sono state tolte (28/11, i contenuti stanno nel paper e nella
  presentazione): `/results`, `/how` e `/models` portano a `/`. Le API delle metriche del backend sono rimaste.
- Script parlato del video: [docs/demo/SCRIPT_VIDEO.md](../../docs/demo/SCRIPT_VIDEO.md).
- Foto di esempio con risposte registrate: `scripts/import_probs.py` trasforma i file `*.probs.pt` dei job di
  valutazione in registrazioni (`data/demo/recorded/`, fuori da git). All'upload di una foto registrata la demo
  ricompila i dettagli prodotto con cui il modello ha risposto. In Demo compaiono solo i modelli che rispondono.

## 7. Modalità registrato
`scripts/export_session.py --session <id>` crea `data/demo/export/<id>/` (immagini + `manifest.json` senza nomi) →
su un host GPU `scripts/record.py --model <id> --export data/demo/export/<id>` produce
`recorded/<modello>/<run>/{run.json,predictions.jsonl}` (catena export → record → rescan → predict verificata con la
famiglia finta) → si copiano in `data/demo/recorded/` e si fa `POST /api/recorded/rescan` (o si riavvia). La UI mostra
"Recorded <data> on <GPU>" ovunque compaia un risultato registrato. Lookup per sha256 + hash del contesto: se manca,
`RECORDED_MISSING`.

### Foto di esempio (per il video)
La pagina **Classify** mostra una fila "Sample photos": un clic e la foto entra come se fosse caricata; se il modello
scelto ha una risposta registrata per quella foto compare l'etichetta "Ready" e la risposta arriva subito. Le foto sono i
file di `data/demo/samples/` (fuori da git; solo foto di `prompt-test`, mai dev o final, senza persone). Con una foto
propria la demo funziona come prima: il modello gira dal vivo (worker GPU o gateway) e ci vuole qualche secondo.
Per avere le risposte registrate di ogni modello serve una GPU:
```sh
python src/demo/scripts/export_session.py --session <id della sessione con le foto di esempio>   # laptop
python src/demo/scripts/record.py --model <id> --export data/demo/export/<id sessione>           # host GPU, un modello alla volta
# copiare data/demo/recorded/<modello>/<run>/ sul laptop e POST /api/recorded/rescan
```
`record.py` copre i modelli con un adapter (`vadapter_heads`, `lora_heads`, `fullft_heads`); il modello `mixed` non ne ha
uno: per quello si usa `import_probs.py` con le probabilità di un job di valutazione. Le foto sono state scelte dalle
righe di prompt-test, quindi final-abt e P13 non le hanno viste; P15 e il full fine-tune sono allenati su tutto il train
e le hanno viste (la demo lo dice nella nota della risposta registrata solo se il report del modello lo riporta).

## 8. Metriche
Numeri riportati dai file del repo, riletti con la loro riga (`scripts/import_reported.py`); ogni numero ha set,
fonte, dati di training e badge (validità + provenienza). `dev` e `final` sono rifiutati. Il full fine-tune mostra
"seen during training — invalid". `scripts/evaluate.py --model <id> --set <set>` ricalcola con il `metric.py` ufficiale (rifiuta dev/final, rifiuta
i set visti in training salvo `--allow-contaminated`); l'API legge `data/demo/eval/` all'avvio e mostra questi numeri
per primi (badge "recomputed"). Usa la GPU o il gateway: **solo con ok esplicito**, mai 13:00–14:30 di sabato.

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
