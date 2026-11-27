# Allegato E — Piano di lavoro passo per passo

Parte di [PIANO_DEMO.md](../PIANO_DEMO.md) (sezione 13). Task piccoli (30–90 minuti), raggruppati in milestone.

Marcatori: **[CRITICO]** sul percorso critico · **[GPU]** serve un host con GPU · **[DECISIONE]** serve una scelta
del team (domanda in [§16](../PIANO_DEMO.md#16-domande-aperte-per-il-team)) · **[OK UTENTE]** serve un'autorizzazione
esplicita (installazioni, download, job a pagamento, uso di dati).

Regole valide per ogni task:

- Branch `feat/demo-app`; nessuna modifica fuori da `src/demo/`, `docs/demo/` e `.gitignore`.
- Prima di ogni commit: `git status` + `python src/demo/scripts/check_no_data.py` (M0.2). Commit in italiano.
- Le stime sono per una persona che conosce lo stack; una sessione di Claude Code può essere più veloce, ma la
  pianificazione usa questi numeri.

---

## E.1 Agenda proposta (27–28 novembre, CET)

| quando | flusso A (backend) | flusso B (frontend) | note |
|---|---|---|---|
| ven 21:45–22:30 | M0.1, M0.2, M0.3 | M0.4 | **[OK UTENTE]** per le installazioni |
| ven 22:30–02:00 | M1.1 → M1.5, M1.7 | M2.0 (ripiego), M2.1, M2.2 | M1.6 se c'è la chiave Keera |
| sab 08:00–11:30 | M1.6, M4.1–M4.3, M3.* se c'è un host GPU | M2.3, M2.4, M2.5, M4.4, M5.1 | |
| sab 11:30 | **freeze del codice** della demo | | solo correzioni bloccanti dopo |
| sab 11:30–12:00 | M6.1, M6.3 rapidi | M7.3 | |
| sab 12:00–12:45 | registrazione del **video demo (≤ 2 min, README.md:82-84)** | | il video va consegnato entro le 14:30 |
| sab 13:00–14:30 | **nessun lavoro sulla demo che usi GPU, HF, Kaggle o la rete del team** | | finestra delle predizioni final |
| sab 14:30–15:00 | pre-riscaldamento: modello caricato, sessione pronta, piano B registrato | | pitch alle 15:00 |

## E.2 Tagli di ambito

| taglio | task | durata stimata (una persona) |
|---|---|---|
| **A — minimo presentabile** | M0.1–M0.4, M1.1–M1.5, M1.6 **o** M1.7, M2.0–M2.4, M4.1–M4.3, M4.4 (solo card + barre), M7.3 | ~16 h; ~9 h con due flussi in parallelo |
| **B — completo senza GPU** | A + M1.4b (guardia), M2.5, M5.1–M5.3, M6.1, M6.3, M6.4, M7.1, M7.2 | +~8 h |
| **C — con GPU** | B + M3.1–M3.5, M4.5, M6.2, M6.5, M7.4 | +~8 h, **[GPU]** |
| stretch | M5.4 (storefront), confidenza metodo B per P10, cache VQ condivisa | +~3 h |

**Percorso critico:** M0.1 → M0.3 → M1.1 → M1.3 → M1.4 → M1.5 → (M1.6 | M1.7) → M2.1 → M2.2 → M2.3 → M2.4 →
M4.2 → M4.4 → M7.3. Il flusso B parte da M0.4 e si aggancia a M2.2 quando l'API risponde.

---

## M0 — Setup e branch

### M0.1 — Branch e scheletro **[CRITICO]**
- **Obiettivo:** branch isolato e cartelle vuote.
- **File:** `src/demo/README.md` (stub), `src/demo/backend/demo_api/__init__.py`, `src/demo/backend/demo_worker/__init__.py`,
  `src/demo/scripts/`, `src/demo/config/`, `docs/demo/` (il piano è già qui).
- **Dipende da:** —
- **Passi:** 1) `git switch main && git pull --ff-only` (solo con ok a usare la rete); 2) `git switch -c feat/demo-app`;
  3) creare le cartelle; 4) **nessuna** cartella chiamata `data`.
- **Accettazione:** `git diff --stat main` mostra solo file sotto `src/demo/` e `docs/demo/`; `git branch --show-current`
  = `feat/demo-app`.
- **Stima:** 15 min · **Commit:** `chore: crea lo scheletro della demo`

### M0.2 — Regole `.gitignore` e controllo anti-dati **[CRITICO]**
- **Obiettivo:** impossibile committare dati, screenshot, database o predizioni.
- **File:** `.gitignore` (righe aggiunte in fondo, testo esatto in [§18](../PIANO_DEMO.md#18-gitignore-e-igiene-dei-dati)),
  `src/demo/scripts/check_no_data.py`.
- **Dipende da:** M0.1
- **Passi:** 1) aggiungere le regole; 2) script che esamina `git diff --cached --name-only` e fallisce se un file
  staged ha estensione immagine (`.jpg .jpeg .png .webp .gif`) fuori da `src/demo/frontend/src/assets/icons/`
  (solo SVG ammessi lì), oppure `.csv .jsonl .sqlite .db .pt .safetensors`, oppure se il contenuto contiene
  `img_\d{5}` o una chiave che inizia con `hf_`.
- **Accettazione:** `git check-ignore -v docs/demo/reference/x.png data/demo/demo.sqlite src/demo/frontend/test-results/a.png`
  stampa una regola per ognuno; lo script fallisce su un file finto `prova.csv` messo in stage (poi rimosso).
- **Stima:** 30 min · **Commit:** `chore: escludi da git i dati della demo e aggiungi il controllo anti-dati`

### M0.3 — Ambiente Python del backend **[CRITICO] [OK UTENTE]**
- **Obiettivo:** dipendenze dell'API, senza toccare le versioni del repo.
- **File:** `src/demo/backend/requirements.txt` (`fastapi`, `uvicorn[standard]`, `python-multipart`, `pyyaml`,
  `httpx`, `pytest`, `pytest-asyncio`; pandas e pillow alle versioni di `src/requirements.txt:1-2`),
  `src/demo/backend/requirements-gpu.txt` (stesse versioni dei job: torch==2.8.0, torchvision==0.23.0, fork
  transformers @3797303dda74844e3d1f8977ff5518bb91f818b4, peft==0.22.0, accelerate==1.16.0, safetensors,
  huggingface_hub; fonte src/apertus-8B-lora-heads/build_hf_job.py:40-55).
- **Dipende da:** M0.1
- **Passi:** 1) chiedere l'ok all'installazione; 2) installare **nel `.venv` del repo** (ha già fork transformers,
  pandas, pillow, pydantic) i soli pacchetti mancanti; 3) fissare le versioni installate in `requirements.txt` con
  `==`.
- **Accettazione:** `.venv/bin/python -c "import fastapi, uvicorn, multipart, yaml, httpx, pytest"` senza errori;
  `pip`/`uv` non ha cambiato la versione di pandas (3.1.1) né di pillow (13.0.0).
- **Stima:** 20 min · **Commit:** `chore: aggiungi le dipendenze del backend della demo`

### M0.4 — Scaffold del frontend **[CRITICO] [OK UTENTE]**
- **Obiettivo:** progetto Vite + React + TypeScript con lo stack scelto.
- **File:** `src/demo/frontend/` (`package.json`, `package-lock.json`, `vite.config.ts` con proxy `/api` →
  `http://127.0.0.1:8010`, `tsconfig.json`, Tailwind, `src/main.tsx`, `src/styles/tokens.css`).
- **Dipende da:** M0.1
- **Passi:** 1) ok per `npm install`; 2) `npm create vite@latest` (template react-ts) dentro `src/demo/frontend`;
  3) aggiungere `tailwindcss`, `@tanstack/react-query`, `react-router-dom`, `recharts`, `@visx/heatmap`,
  `@visx/scale`, `react-dropzone`, `@playwright/test` (dev); 4) fissare le versioni esatte (lockfile committato).
- **Accettazione:** `npm run build` crea `dist/` (ignorato); `npm run dev` mostra una pagina con il `DemoBadge`.
- **Stima:** 45 min · **Commit:** `chore: crea il frontend della demo con Vite, React e Tailwind`

### M0.5 — Collegamenti locali per gli script del repo (facoltativo)
- **Obiettivo:** far girare `src/scripts/eval_prompt.py` senza modificarlo (gap G12).
- **Passi:** `ln -s hackapertus-galaxus/train.csv data/train.csv`, idem `metric.py`, `train_images` (tutto dentro
  `data/`, ignorato).
- **Accettazione:** `.venv/bin/python src/scripts/eval_prompt.py --help` funziona; `git status` pulito.
- **Stima:** 10 min · **Commit:** nessuno (solo file locali)

---

## M1 — Backend: scheletro, tassonomia, upload, registro

### M1.1 — Configurazione, errori, salute **[CRITICO]**
- **Obiettivo:** app FastAPI con schema di errore unico.
- **File:** `demo_api/main.py`, `settings.py` (env: `DEMO_REPO_ROOT`, `DEMO_DATA_DIR`=`data/demo`,
  `DEMO_KAGGLE_DIR`=`data/hackapertus-galaxus`, `DEMO_METRIC_PATH`, limiti upload, `DEMO_FULLFT_SRC`,
  `DEMO_WORKER_URL_*`, `DEMO_WORKER_TOKEN`), `errors.py`, `routers/health.py`.
- **Dipende da:** M0.3
- **Passi:** 1) handler che traduce eccezioni ed errori Pydantic nello schema di [C.1](C_api.md#c1-schema-di-errore-unificato);
  2) `X-Request-Id`; 3) caricamento di `.env` senza `sys.exit` e senza mai stampare valori; 4) `GET /api/health`.
- **Accettazione:** `uvicorn demo_api.main:app --port 8010` parte; `curl /api/health` restituisce `status: ok`; una
  rotta inesistente restituisce `NOT_FOUND` nello schema unico; i log non contengono `LLM_API_KEY`.
- **Stima:** 45 min · **Commit:** `feat: avvia il backend della demo con errori uniformi e controllo di salute`

### M1.2 — Tassonomia e regole di applicabilità **[CRITICO]**
- **File:** `demo_api/taxonomy.py`, `demo_api/relevance.py`, `src/demo/config/erp_labels.yaml`, `routers/taxonomy.py`,
  `tests/test_taxonomy.py`, `tests/test_relevance.py`.
- **Dipende da:** M1.1
- **Passi:** 1) importare `common.prompt.TAXONOMY` (nessuna copia delle classi); 2) etichette ERP da YAML (4 note +
  `auto`); 3) `applicable(labels) -> dict[dim, bool]` con le 5 regole di `challenge_description.md:102-107`;
  4) `GET /api/taxonomy`.
- **Accettazione:** test: 41 stringhe, uguali a `prompt.TAXONOMY` e (se `metric.py` è presente) a
  `metric.VALID_CLASSES`; un caso di test per ogni regola (es. subject `logo` → background, perspective,
  individuals, text_classification non applicabili; text_language applicabile con subject `packaging` anche senza
  overlay).
- **Stima:** 45 min · **Commit:** `feat: tassonomia e regole di applicabilità per la demo`

### M1.3 — Database e archivio file **[CRITICO]**
- **File:** `demo_api/db.py` (sqlite3 della libreria standard, WAL, `CREATE TABLE IF NOT EXISTS` dello schema in
  [§8](../PIANO_DEMO.md#8-modello-dati)), `demo_api/storage.py`, `routers/sessions.py`, `tests/test_db.py`.
- **Dipende da:** M1.1
- **Accettazione:** all'avvio esiste `data/demo/demo.sqlite` con le 8 tabelle; `POST /api/sessions` e
  `GET /api/sessions` funzionano; un test crea e rilegge una sessione in un DB temporaneo.
- **Stima:** 60 min · **Commit:** `feat: database e archivio locali della demo`

### M1.4 — Upload validato **[CRITICO]**
- **File:** `demo_api/images_io.py`, `routers/images.py`, `tests/test_upload.py`.
- **Dipende da:** M1.3
- **Passi:** i 7 passi di [C.4](C_api.md#c4-sessioni-e-immagini) tranne la guardia (M1.4b); miniature; endpoint
  `GET /api/images`, `…/file`, `…/thumbnail`, `PUT …/context`.
- **Accettazione (test con immagini sintetiche generate in memoria da Pillow):** JPEG/PNG/WebP validi → 201; un file
  `.png` con contenuto testo → 415; un JPEG troncato → 422 `CORRUPT_IMAGE`; GIF animata → 415; file 21 MB → 413;
  JPEG con EXIF `Orientation=6` → salvato in PNG con larghezza e altezza scambiate e `exif_transposed: true`; stesso
  file caricato due volte → `duplicate: true`; un JPEG senza EXIF → byte salvati identici a quelli inviati.
- **Stima:** 90 min · **Commit:** `feat: caricamento delle immagini con validazione completa`

### M1.4b — Guardia dev/final
- **File:** `src/demo/scripts/build_guard_index.py`, `demo_api/guard.py`, test.
- **Dipende da:** M1.4
- **Passi:** sha256 di ogni file in `data/hackapertus-galaxus/dev_images/` (e `final_images/` se esiste) →
  `data/demo/guard/sha256.txt`; l'upload rifiuta con `FORBIDDEN_IMAGE`.
- **Accettazione:** caricare una copia identica di un file di `dev_images` → 422 `FORBIDDEN_IMAGE` (test manuale, il
  file resta fuori dal repo); l'indice non è tracciato da git.
- **Stima:** 30 min · **Commit:** `feat: blocca il caricamento delle immagini dev e final`

### M1.5 — Registro dei modelli **[CRITICO]**
- **File:** `src/demo/models.yaml` (bozza di [B.7](B_registro_modelli.md#b7-bozza-completa-di-srcdemomodelsyaml)),
  `demo_api/registry.py`, `routers/models.py`, `tests/test_registry.py`.
- **Dipende da:** M1.2
- **Passi:** schema Pydantic; stato calcolato (chiave presente, pesi in cache con `local_files_only=True`, worker
  raggiungibile, registrazioni presenti, campi `[DA VERIFICARE]`); `GET /api/models`, `GET /api/models/{id}`.
- **Accettazione:** su questo laptop senza `.env`: i due zero-shot `unavailable` con motivo "manca LLM_API_KEY", P10,
  P11 e full FT `unavailable` con motivo leggibile, P8/P9 `unavailable` "solo metriche"; un `source` che punta a un
  file inesistente fa fallire l'avvio con un messaggio chiaro.
- **Stima:** 60 min · **Commit:** `feat: registro dei modelli della demo da models.yaml`

### M1.6 — Zero-shot 8B e 70B, predizioni e job **[CRITICO] [DECISIONE] [OK UTENTE]**
- **Obiettivo:** primo modello vero end-to-end (il più economico eseguibile da qui).
- **File:** `demo_api/adapters/base.py`, `adapters/zeroshot.py`, `demo_api/jobs.py` (coda asyncio, SSE),
  `routers/predict.py`, `tests/test_zeroshot_payload.py`.
- **Dipende da:** M1.4, M1.5; **chiave Keera configurata dall'utente in `.env`** (domanda Q2)
- **Passi:** B.2 passo per passo; cache per chiave (C.5); `GET /api/jobs/{id}`, `…/events`, `GET /api/predictions`.
- **Accettazione:** test senza rete: i 7 corpi di richiesta costruiti dall'adapter sono **identici** a quelli che
  costruirebbe `eval_prompt.run_variant.predict` per la stessa riga e lo stesso URL (confronto di dizionari);
  test manuale con la chiave: un'immagine nostra → 6 label valide e confidenze, tempo misurato mostrato; una
  seconda richiesta uguale torna `cached: true` senza chiamate di rete.
- **Stima:** 90 min · **Commit:** `feat: classificazione zero-shot con Apertus 8B e 70B tramite gateway`

### M1.7 — Modalità registrato
- **File:** `adapters/recorded.py`, `src/demo/scripts/export_session.py`, `routers/recorded.py`, test.
- **Dipende da:** M1.5
- **Accettazione:** un `predictions.jsonl` di prova **generato dal test** (valori segnaposto, mai committato) viene
  letto; `POST /api/predict` con `mode: recorded` restituisce `mode: "recorded"` e i dati di `run.json`; immagine
  assente → `RECORDED_MISSING`.
- **Stima:** 45 min · **Commit:** `feat: modalità registrato per i modelli senza GPU in presentazione`

---

## M2 — Frontend: guscio ERP, coda, dettaglio

### M2.0 — Token di design **[DECISIONE]**
- **Obiettivo:** token misurati (se gli screenshot ci sono) o ripiego dichiarato.
- **File:** `src/demo/frontend/src/styles/tokens.css`, config Tailwind.
- **Passi:** procedura di [D.1.1](D_interfaccia.md#d11-procedura-di-misura-da-eseguire-quando-gli-screenshot-ci-sono).
- **Accettazione:** ogni token ha un commento `misurato: …` oppure `ripiego: tailwind <nome>`; `VERIFICA.md` elenca
  quali sono di ripiego.
- **Stima:** 30 min (misura) / 10 min (ripiego) · **Commit:** `feat: token di design della demo`

### M2.1 — Guscio ERP **[CRITICO]**
- **File:** `src/components/shell/*` (`ErpShell`, `ErpHeader`, `Wordmark`, `DemoBadge`, `AccentLine`, `Sidebar`,
  `SwitchNowButton`, `TitleBar`, `SubLinks`, `ActionToolbar`, `CollapsibleSection`, `LabelValueGrid`, `WidgetCard`),
  `src/routes.tsx`, pagina `/product/DEMO-0001`.
- **Dipende da:** M0.4, M2.0
- **Accettazione:** schermata 1 come in [D.4](D_interfaccia.md#schermata-1--guscio-erp-e-pagina-prodotto-demo); badge
  visibile su ogni route; voci inattive con tooltip; `Enrich Product ▾ → Image classification (AI)` porta alla coda.
- **Stima:** 75 min · **Commit:** `feat: guscio in stile ERP con badge della demo`

### M2.2 — Client API e stati comuni **[CRITICO]**
- **File:** `src/api/client.ts` (fetch tipizzato, schema d'errore), `src/api/hooks.ts` (TanStack Query),
  `src/api/sse.ts` (EventSource con ripiego a polling), `EmptyState`, `ErrorState`, `ModelUnavailableState`, `Toast`.
- **Dipende da:** M1.1, M0.4
- **Accettazione:** con backend spento appare il banner "Backend non raggiungibile"; un errore 415 dell'upload
  mostra il messaggio del server; i tipi TypeScript corrispondono agli esempi di [allegato C](C_api.md).
- **Stima:** 45 min · **Commit:** `feat: client delle API e stati di errore nel frontend`

### M2.3 — Coda di classificazione **[CRITICO]**
- **File:** `src/pages/QueuePage.tsx`, `UploadDropzone`, `ModelSelect`, `RunButton`, `ProgressBar`,
  `ThroughputBadge`, `ResultsTable`, `ConfidenceChip`, `StatusPill`, `ModeBadge`, `ContextWarning`.
- **Dipende da:** M2.1, M2.2, M1.4, (M1.6 o M1.7)
- **Accettazione:** caricare 5 immagini nostre → righe con miniature; `Run classification` → barra che avanza da
  SSE, celle che si riempiono, img/s misurati con la GPU o "gateway remoto"; ordinamento per min conf; stati vuoto,
  in coda, errore, non applicabile visibili.
- **Stima:** 90 min · **Commit:** `feat: coda di classificazione con caricamento e avanzamento in tempo reale`

### M2.4 — Dettaglio con pannello "Image type" **[CRITICO]**
- **File:** `src/pages/ImageDetailPage.tsx`, `ImagePreview`, `MetadataTable`, `ImageTypePanel`,
  `DimensionDropdown`, `TopKPopover`, `EditedMarker`, `RawOutputDisclosure`, `ConfidenceMethodNote`, `ContextForm`;
  backend `routers/reviews.py` (`PUT /api/images/{id}/review`, `GET …/reviews`).
- **Dipende da:** M2.3
- **Accettazione:** schermata 3 come in D.4; modificare Perspective → `EditedMarker`; `Save` → stato `Corrected` nella
  coda; `Cancel` → valori del modello; output non valido → pannello rosso con testo grezzo; scorciatoie J/K/1–6/S/Esc.
- **Stima:** 90 min · **Commit:** `feat: dettaglio immagine con Image type precompilato e revisione umana`

### M2.5 — Cambio modello senza ricaricare
- **Dipende da:** M2.4
- **Accettazione:** con due modelli disponibili (anche uno registrato), cambiare modello nella coda e nel dettaglio
  mostra i risultati dell'altro senza un nuovo `POST /api/images`; le predizioni già fatte tornano `cached`.
- **Stima:** 45 min · **Commit:** `feat: cambio di modello sulle stesse immagini`

---

## M3 — Adapter GPU **[GPU]**

### M3.1 — Worker e proxy
- **File:** `demo_worker/server.py`, `demo_worker/families/__init__.py`, `demo_api/adapters/worker_proxy.py`,
  gestione LRU/VRAM in `demo_api/registry.py`.
- **Dipende da:** M1.5
- **Accettazione:** su una macchina senza GPU il worker parte in modalità `--dry` solo per i test del protocollo;
  su GPU: `GET /worker/info` riporta GPU e memoria; `load`/`unload` cambiano lo stato in SSE; un secondo modello che
  non entra nel budget scarica il primo (se libero).
- **Stima:** 90 min · **Commit:** `feat: worker GPU per i modelli locali con gestione della memoria`

### M3.2 — `HeadsAdapter` P11 **[DECISIONE]**
- **File:** `demo_worker/families/lora_heads.py`.
- **Dipende da:** M3.1; accesso ai pesi P11 (domanda Q3); base 8B in cache (download **[OK UTENTE]**, 18,4 GB).
- **Passi:** B.4.1 passo per passo.
- **Accettazione:** parità con `predict.py` (M6.2) su 20 immagini di coverage; predizione live dalla UI.
- **Stima:** 60 min · **Commit:** `feat: adapter per Apertus 8B LoRA con teste (P11)`

### M3.3 — `HeadsAdapter` full FT **[DECISIONE]**
- **File:** `demo_worker/families/fullft_heads.py`; worktree del branch (B.6).
- **Dipende da:** M3.1; scelta del checkpoint (domanda Q4); download 10,49 GB **[OK UTENTE]**.
- **Accettazione:** parità con `ft_predict.py` su 20 immagini; la scheda modello mostra "nessuna metrica locale
  valida" e il punteggio Kaggle solo se inserito.
- **Stima:** 60 min · **Commit:** `feat: adapter per Apertus 8B in fine-tuning completo con teste`

### M3.4 — `LoraGenerativeAdapter` P10 + confidenza (metodo A)
- **File:** `demo_worker/families/lora_generative.py`.
- **Dipende da:** M3.1; download adapter 0,319 GB + base **[OK UTENTE]**.
- **Accettazione:** parità con la logica di `evaluate.py` su 20 immagini (stesso testo generato); testo identico con
  e senza `output_scores`; avviso "adapter step-00738" visibile.
- **Stima:** 90 min · **Commit:** `feat: adapter per Apertus 8B LoRA generativo (P10) con confidenza`

### M3.5 — Registrazione batch
- **File:** `src/demo/scripts/record.py`.
- **Dipende da:** M1.7, una tra M3.2–M3.4
- **Accettazione:** sull'host GPU, `record.py --model apertus-8b-lora-heads-p11 --session <id>` crea `run.json` e
  `predictions.jsonl` per ogni immagine della sessione; copiati sul laptop in `data/demo/recorded/`, la UI li mostra
  come "Recorded … on <GPU>".
- **Stima:** 60 min · **Commit:** `feat: registrazione delle predizioni su GPU per la presentazione`

---

## M4 — Metriche e dashboard

### M4.1 — Calcolo delle metriche **[CRITICO]**
- **File:** `demo_api/metrics/compute.py`, `tests/test_metrics.py`.
- **Passi:** wrapper di `metric.evaluate` caricato da `DEMO_METRIC_PATH`; soluzione con `common.splits.scored_cells`;
  accuratezza per dimensione; matrice di confusione sulle celle valutate (colonna "vuota/non valida").
- **Accettazione:** l'esempio calcolato a mano di [§9.2](../PIANO_DEMO.md#92-calcolo-esatto) dà esattamente
  0,4444 / 0,25 / 0,3472 (macro) e 0,5208 (weighted); se `metric.py` manca, il test viene saltato con un messaggio e
  una seconda implementazione di controllo (solo nei test) dà gli stessi numeri.
- **Stima:** 45 min · **Commit:** `feat: calcolo delle metriche con metric.py ufficiale`

### M4.2 — Import dei risultati riportati **[CRITICO]**
- **File:** `src/demo/scripts/import_reported.py`, `tests/test_import_reported.py`.
- **Passi:** leggere `evals` di `models.yaml`; per P10 caricare i JSON completi (per dimensione e per classe); per
  zero-shot e P11 leggere dalle tabelle markdown indicate (`per_class_source`) e la matrice di perspective di P11;
  ogni riga di `eval_results` salva `source` = `file:riga`.
- **Accettazione:** test: P11 coverage macro 0.781 e perspective 0.625; zero-shot 8B coverage 0.528; P10 coverage
  0.7341… dal JSON; la matrice di P11 ha 10×10 celle e righe che sommano al supporto di classe del report (es.
  `front_head_on` = 614).
- **Stima:** 90 min · **Commit:** `feat: importa le metriche riportate con la loro fonte`

### M4.3 — API delle metriche e badge **[CRITICO]**
- **File:** `routers/metrics.py`, `demo_api/metrics/badges.py`, test.
- **Accettazione:** `GET /api/metrics/apertus-8b-fullft-heads?set=coverage` restituisce zero risultati validi e la
  nota di contaminazione; `GET /api/metrics/compare` esclude di default i risultati `seen_in_training`; `set=dev` →
  `VALIDATION_ERROR`.
- **Stima:** 45 min · **Commit:** `feat: API delle metriche con badge di validità`

### M4.4 — Dashboard **[CRITICO]**
- **File:** `src/pages/PerformancePage.tsx`, `ModelCard`, `MetricBadge`, `DimensionBarChart`, `PerClassTable`,
  `ConfusionMatrix`, `ConfidenceHistogram`, `ReliabilityDiagram`, `CostPanel`, `CoverageWarningBanner`, `SourceFootnote`.
- **Dipende da:** M4.3, M2.2
- **Accettazione:** banner coverage sempre visibile; linea 0.917 con l'etichetta esatta e la nota "set diverso";
  ogni grafico ha la fonte `file:riga`; il full FT appare con badge rosso e "Kaggle dev: n/a" finché il valore non è
  in `models.yaml`; matrice disponibile solo per P11/perspective; calibrazione "n/a" con il motivo.
- **Stima:** 120 min (taglio A: 45 min solo card + barre) · **Commit:** `feat: dashboard delle prestazioni con metriche oneste`

### M4.5 — Script di valutazione offline **[GPU] [OK UTENTE]**
- **File:** `src/demo/scripts/evaluate.py` (spec in [§9.1](../PIANO_DEMO.md#91-fonti-di-verità-in-ordine-di-priorità)).
- **Accettazione:** rifiuta `--set dev`/`final` e qualsiasi CSV fuori dal train; rifiuta i modelli con il set in
  `training.seen_sets` salvo `--allow-contaminated` (risultato marcato `seen_in_training`); su 20 immagini scrive
  `metrics.json` e `run.json` in `data/demo/eval/<id>/<set>/`. Esecuzione completa solo con ok.
- **Stima:** 60 min · **Commit:** `feat: script di valutazione offline sui set interni`

---

## M5 — Confronto, revisione, velocità e costo

### M5.1 — Confronto modelli
- **File:** `src/pages/ComparePage.tsx`, `ComparisonGrid`, `ModelToggleList`, `LatencyStrip`.
- **Accettazione:** schermata 4 di D.4; disaccordi evidenziati; filtro "solo disaccordi"; "Calcola mancanti" lancia
  i job per i modelli disponibili.
- **Stima:** 75 min · **Commit:** `feat: confronto tra modelli sulle stesse immagini`

### M5.2 — Rifinitura della revisione ed export
- **Accettazione:** stati `AI suggested`/`Reviewed`/`Corrected` coerenti in coda, dettaglio e export; `GET /api/export.csv`
  con valori rivisti; avviso se la sessione contiene immagini `train_coverage`.
- **Stima:** 45 min · **Commit:** `feat: export CSV con le revisioni`

### M5.3 — Pannello velocità e costo
- **Accettazione:** `CostPanel` calcola $/1.000 immagini solo da img/s misurati o riportati con GPU nota; per il
  gateway mostra "n/a (costo API non pubblico)"; linea obiettivo 2 img/s.
- **Stima:** 30 min · **Commit:** `feat: velocità misurata e costo per mille immagini`

### M5.4 — Anteprima storefront (stretch)
- **Accettazione:** schermata 6 con la regola YAML e la scritta "regola illustrativa"; spiegazione per ogni posizione.
- **Stima:** 60 min · **Commit:** `feat: anteprima della galleria in stile storefront con regola illustrativa`

---

## M6 — Test, parità, E2E, verifica visiva

### M6.1 — Suite pytest
- **Accettazione:** `cd src/demo/backend && ../../../.venv/bin/python -m pytest -q` verde; elenco dei test in
  [§14.1](../PIANO_DEMO.md#141-test-unitari-pytest).
- **Stima:** 45 min · **Commit:** `chore: completa i test unitari del backend della demo`

### M6.2 — Parità con il codice del repo **[GPU] [OK UTENTE]**
- **File:** `src/demo/scripts/parity.py`.
- **Accettazione:** [§14.2](../PIANO_DEMO.md#142-test-di-parità); risultato scritto in `docs/demo/VERIFICA.md` (solo
  conteggi e esito, nessun ID).
- **Stima:** 60 min · **Commit:** `chore: verifica di parità tra demo e codice del repo`

### M6.3 — E2E Playwright
- **File:** `src/demo/frontend/e2e/demo.spec.ts`.
- **Accettazione:** [§14.3](../PIANO_DEMO.md#143-e2e-playwright); screenshot in `test-results/` (ignorato).
- **Stima:** 60 min · **Commit:** `chore: percorso completo della demo con Playwright`

### M6.4 — Verifica visiva
- **Accettazione:** checklist di [§14.4](../PIANO_DEMO.md#144-verifica-visiva) compilata in `VERIFICA.md`.
- **Stima:** 30 min · **Commit:** `docs: verifica visiva della demo`

### M6.5 — Misura del throughput **[GPU]**
- **Accettazione:** per ogni modello eseguito: img/s su 20 e su 100 immagini (batch 1 e batch 8), GPU, picco di
  memoria, in `VERIFICA.md`; confronto con l'obiettivo ≥ 2 img/s; solo numeri misurati.
- **Stima:** 30 min · **Commit:** `docs: velocità misurata dei modelli nella demo`

---

## M7 — Documentazione e copione

### M7.1 — `src/demo/README.md`
- Schema in [§17](../PIANO_DEMO.md#17-documentazione-da-produrre-in-fase-di-implementazione). **Stima:** 30 min ·
  **Commit:** `docs: istruzioni per avviare la demo`

### M7.2 — `docs/demo/VERIFICA.md`
- **Stima:** 30 min · **Commit:** `docs: registro delle verifiche della demo`

### M7.3 — `docs/demo/SCRIPT_PRESENTAZIONE.md` **[CRITICO]**
- Versione 3–5 min, taglio da 2 min per il video, segmento live di 60–90 s nel pitch. **Stima:** 30 min ·
  **Commit:** `docs: copione della presentazione della demo`

### M7.4 — Piano B registrato
- **Accettazione:** sessione di presentazione con 15–20 immagini, predizioni registrate per ogni modello allenato
  usato, avviata da `make -C src/demo demo` senza rete; verificata a mano il giorno stesso.
- **Stima:** 30 min · **Commit:** nessuno (solo dati locali)
