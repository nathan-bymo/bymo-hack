# Allegato C — API del backend

Parte di [PIANO_DEMO.md](../PIANO_DEMO.md) (sezione 7). Base URL: `http://127.0.0.1:8010/api` (FastAPI). In sviluppo
il frontend Vite (porta 5173) inoltra `/api` a 8010; in modalità presentazione FastAPI serve anche il build del
frontend sulla stessa porta.

> **Tutti gli esempi JSON di questo allegato usano valori segnaposto** (`"<…>"`, `0.0`, `0`). Non sono risultati,
> metriche o predizioni reali e non vanno copiati in test, UI o documentazione come se lo fossero.

Convenzioni:

- JSON UTF-8, chiavi `snake_case`; date ISO 8601 UTC (`"2026-11-28T09:00:00Z"`).
- ID generati dalla demo: `ses_<uuid4>`, `img-demo-<uuid4>`, `pred_<uuid4>`, `rev_<uuid4>`, `job_<uuid4>`. Un
  `image_id` della competizione (forma `img_` seguito da cinque cifre, challenge_description.md:185) non diventa mai un ID: se il nome
  del file caricato lo contiene, resta solo in `original_filename`, che l'export include soltanto su richiesta
  esplicita e con un avviso.
- Ogni risposta ha l'header `X-Request-Id`.

---

## C.1 Schema di errore unificato

```json
{
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "<English, user-readable text>",
    "details": {"model_id": "<id>", "reason": "<motivo>"},
    "request_id": "<uuid>"
  }
}
```

| `code` | HTTP | quando |
|---|---|---|
| `VALIDATION_ERROR` | 422 | corpo o parametri non validi (Pydantic) |
| `NOT_FOUND` | 404 | sessione, immagine, predizione o job inesistente |
| `MODEL_NOT_FOUND` | 404 | `model_id` non in `models.yaml` |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | MIME o magic byte non tra JPEG, PNG, WebP; immagine animata |
| `FILE_TOO_LARGE` | 413 | file > `DEMO_MAX_UPLOAD_MB` (default 20) o > 50 file per richiesta |
| `IMAGE_TOO_LARGE` | 413 | più di `DEMO_MAX_PIXELS` (default 50.000.000) pixel |
| `CORRUPT_IMAGE` | 422 | Pillow non decodifica (`verify()` o `load()` falliscono) |
| `FORBIDDEN_IMAGE` | 422 | sha256 uguale a un'immagine di `dev_images/` (o `final_images/` se presente): vedi C.4 |
| `MODEL_UNAVAILABLE` | 409 | stato `unavailable` (chiave mancante, pesi mancanti, GPU assente, configurazione incompleta) |
| `MODEL_LOADING` | 409 | il modello è in caricamento: riprovare o seguire gli eventi |
| `MODEL_BUSY` | 409 | budget VRAM pieno e nessun modello scaricabile (B.8) |
| `CONFIRMATION_REQUIRED` | 409 | download o azione costosa senza `confirm: true` |
| `GPU_OOM` | 507 | memoria GPU esaurita anche a batch 1 |
| `UPSTREAM_ERROR` | 502 | errore del gateway Keera / server OpenAI-compatibile / worker remoto |
| `TIMEOUT` | 504 | caricamento > 15 min o predizione > 120 s per immagine |
| `RECORDED_MISSING` | 409 | modalità registrato senza registrazione per quell'immagine/contesto |
| `INTERNAL_ERROR` | 500 | eccezione non prevista (messaggio generico; dettagli solo nel log locale) |

Nessun messaggio d'errore contiene chiavi, token o percorsi della home dell'utente.

---

## C.2 Salute, tassonomia, impostazioni

### `GET /api/health`
```json
{"status": "ok", "version": "<versione demo>", "git_commit": "<sha corto>",
 "gpu_local": false, "workers": [{"family": "<famiglia>", "url": "<url>", "reachable": false}]}
```

### `GET /api/taxonomy`
Fonte: `common.prompt.TAXONOMY` (prompt.py:26-42) + `src/demo/config/erp_labels.yaml`.
```json
{
  "dimensions": [
    {"key": "subject", "erp_label": "Subject", "classes": [
      {"index": 0, "code": "A", "value": "full_product", "erp_label": "<etichetta>", "erp_label_source": "prompt|auto"}
    ]}
  ],
  "relevance_rules": [
    {"if": {"subject": ["logo", "placeholder", "energy_label"]},
     "not_applicable": ["background", "perspective", "individuals", "text_classification"],
     "source": "challenge_description.md:102"}
  ],
  "confidence_bins": {"low_below": 0.60, "high_from": 0.90, "source": "src/plain-benchmark/apertus-8B-benchmark/teacher_student_zeroshot.md:57"}
}
```

### `GET /api/settings` · `PUT /api/settings`
```json
{"gpu_label": "<es. A100 80GB>", "usd_per_hour": 0.0, "presentation_mode": false,
 "show_ground_truth": false, "default_use_context": true, "zero_shot_parallel_images": 8}
```
`PUT` accetta lo stesso oggetto (campi parziali). `show_ground_truth` è `false` di default (domanda aperta Q9).

---

## C.3 Modelli

### `GET /api/models`
```json
{
  "models": [
    {
      "model_id": "apertus-8b-lora-heads-p11",
      "display_name": "<nome>",
      "family": "lora_heads",
      "runtime": "worker_gpu",
      "status": "unavailable",
      "status_reason": "<es. weights missing: repo da verificare>",
      "modes": ["live", "recorded"],
      "recorded_runs": [{"run_id": "<id>", "recorded_at": "<data>", "gpu_name": "<gpu>", "images": 0}],
      "confidence": {"kind": "full_distribution", "method": "head_softmax"},
      "training": {"data": "<dati>", "phase": 1},
      "weights": {"repo": "<repo>", "revision": "<sha>", "subfolder": "<cartella>", "size_gb": 0.0},
      "vram_estimate_gb": 0.0,
      "vram_estimate_kind": "measured|estimate",
      "headline_eval": {"set": "coverage", "macro_f1": 0.0, "validity": "phase1_internal",
                        "provenance": "reported_only", "source": "<file:riga>"},
      "kaggle_dev": {"score": null}
    }
  ]
}
```

### `GET /api/models/{model_id}`
Come un elemento della lista, più `evals` completi, `measured`, `caveats`, `contamination_note`.

### `POST /api/models/{model_id}/load`
Richiesta: `{}`. Risposta `202`:
```json
{"job_id": "job_<uuid>", "model_id": "<id>", "state": "loading"}
```
Errori: `MODEL_NOT_FOUND`, `MODEL_UNAVAILABLE`, `MODEL_BUSY`, `GPU_OOM`, `TIMEOUT`. Per i zero-shot e i registrati
risponde `200 {"state": "ready"}` senza job.

### `POST /api/models/{model_id}/unload`
Risposta `200 {"model_id": "<id>", "state": "not_loaded"}`. `409 MODEL_BUSY` se un job usa il modello.

### `POST /api/models/{model_id}/download`
Richiesta `{"confirm": false}` → `409 CONFIRMATION_REQUIRED` con `details.download_gb`. Con `{"confirm": true}` →
`202 {"job_id": …}`. Disabilitato (`409 MODEL_UNAVAILABLE`, motivo "nessun worker GPU") se non c'è un worker.

---

## C.4 Sessioni e immagini

### `POST /api/sessions` · `GET /api/sessions` · `GET /api/sessions/{session_id}`
```json
{"session_id": "ses_<uuid>", "name": "<es. Demo 28/11>", "created_at": "<data>", "image_count": 0}
```

### `POST /api/images` (multipart)
Campi: `session_id` (obbligatorio), `files` (1–50), `source_kind` per file o globale (`own_photo` |
`train_coverage` | `other`), `context` opzionale (JSON per file: chiavi di `CONTEXT_FIELDS`).

Validazione, in ordine (si ferma al primo errore per file; gli altri file proseguono):

1. Dimensione ≤ `DEMO_MAX_UPLOAD_MB` → altrimenti `FILE_TOO_LARGE`.
2. MIME dichiarato in {`image/jpeg`, `image/png`, `image/webp`} **e** magic byte coerenti: JPEG `FF D8 FF`, PNG
   `89 50 4E 47 0D 0A 1A 0A`, WebP `RIFF????WEBP` → altrimenti `UNSUPPORTED_MEDIA_TYPE`. HEIC, GIF, TIFF, SVG
   rifiutati (le foto da telefono vanno esportate in JPEG).
3. `PIL.Image.open` + `verify()`, poi riapertura + `load()`; `n_frames > 1` → `UNSUPPORTED_MEDIA_TYPE`;
   pixel > `DEMO_MAX_PIXELS` → `IMAGE_TOO_LARGE`; eccezioni → `CORRUPT_IMAGE`.
4. Orientamento EXIF: se il tag `Orientation` è presente e diverso da 1, l'immagine viene raddrizzata con
   `ImageOps.exif_transpose` e salvata in **PNG** (senza perdita); altrimenti si salvano **i byte originali senza
   modifiche**. Il preprocessing del modello resta quello del repo (`common/images.py`, che non legge l'EXIF).
5. `sha256` dei byte salvati; se esiste già nella sessione, si restituisce l'immagine esistente (`duplicate: true`).
6. **Guardia dev/final**: se lo sha256 è nell'indice degli hash di `data/hackapertus-galaxus/dev_images/` (e di
   `final_images/` se esiste) → `FORBIDDEN_IMAGE`. Indice costruito una volta da
   `python src/demo/scripts/build_guard_index.py` in `data/demo/guard/sha256.txt` (ignorato da git). Copre solo i
   file identici: è una rete di sicurezza, non un sostituto della regola.
7. Miniatura JPEG 256 px in `data/demo/thumbs/`.

Risposta `201`:
```json
{
  "images": [
    {"image_id": "img-demo-<uuid>", "session_id": "ses_<uuid>", "original_filename": "<nome>",
     "sha256": "<hash>", "format": "JPEG", "mime": "image/jpeg", "width": 0, "height": 0, "bytes": 0,
     "has_alpha": false, "exif_transposed": false, "source_kind": "own_photo",
     "context": {}, "duplicate": false,
     "file_url": "/api/images/img-demo-<uuid>/file", "thumbnail_url": "/api/images/img-demo-<uuid>/thumbnail"}
  ],
  "rejected": [{"original_filename": "<nome>", "error": {"code": "CORRUPT_IMAGE", "message": "<testo>"}}]
}
```

### `GET /api/images?session_id=…` · `GET /api/images/{image_id}`
Come sopra; `GET /api/images/{image_id}` aggiunge `predictions` (ultima per modello) e `review` (ultima).

### `GET /api/images/{image_id}/file` · `GET /api/images/{image_id}/thumbnail`
Byte dell'immagine salvata / miniatura, `Cache-Control: private, max-age=3600`.

### `PUT /api/images/{image_id}/context`
```json
{"product_title": "<testo>", "product_type": "<testo>", "brand": "<testo>", "category": "<testo>", "subcategory": "<testo>"}
```
Campi opzionali; stringhe vuote = campo assente. Cambiare il contesto **non** cancella le predizioni vecchie: la
chiave di cache include l'hash del contesto, quindi la prossima predizione è nuova.

---

## C.5 Predizioni e job

### `POST /api/predict`
```json
{"model_id": "apertus-8b-lora-heads-p11", "image_ids": ["img-demo-<uuid>"], "use_context": true,
 "mode": "live", "batch_size": 8}
```
`mode`: `live` (default) o `recorded`. `batch_size` facoltativo (default per famiglia: teste 8, P10 8, zero-shot =
parallelismo). Chiave di cache: `(model_id, model_revision_or_fingerprint, image_sha256, context_hash,
use_context, prompt_version, mode)`. Le immagini già in cache tornano subito (`cached: true`); le altre vanno in un
job.

Risposta `202` (o `200` se tutto è in cache):
```json
{"job_id": "job_<uuid>", "model_id": "<id>", "total": 0, "cached": 0, "queued": 0,
 "events_url": "/api/jobs/job_<uuid>/events"}
```

### `GET /api/jobs/{job_id}`
```json
{
  "job_id": "job_<uuid>", "kind": "predict", "model_id": "<id>", "status": "running|done|error|canceled",
  "done": 0, "total": 0, "started_at": "<data>", "finished_at": null,
  "results": [
    {
      "image_id": "img-demo-<uuid>", "prediction_id": "pred_<uuid>", "cached": false, "mode": "live",
      "valid": true,
      "dimensions": {
        "subject": {"label": "<classe>", "erp_label": "<etichetta>", "confidence": 0.0,
                    "top_k": [{"label": "<classe>", "p": 0.0}], "confidence_method": "head_softmax",
                    "applicable": true},
        "background": {"label": "<classe>", "erp_label": "<etichetta>", "confidence": null, "top_k": null,
                       "confidence_method": null, "applicable": true}
      },
      "raw_output": null,
      "timing": {"preprocess_ms": 0.0, "model_ms": 0.0, "total_ms": 0.0},
      "recorded": null
    }
  ],
  "batch": {"images": 0, "seconds": 0.0, "images_per_second": 0.0, "ms_per_image": 0.0,
            "gpu_name": "<gpu o 'gateway remoto'>", "batch_size": 0, "peak_memory_gb": null},
  "error": null
}
```
(Nell'esempio solo 2 dimensioni per brevità: la risposta vera ha sempre tutte e 6.) Per `mode: recorded`,
`recorded` = `{"run_id", "recorded_at", "gpu_name", "images_per_second"}` e `batch` riporta i valori **della
registrazione**, marcati come tali.

### `GET /api/jobs/{job_id}/events` (SSE, `text/event-stream`)
Eventi:

| evento | dati |
|---|---|
| `state` | `{"model_id", "state", "message"}` (caricamento) |
| `progress` | `{"done", "total", "images_per_second"}` |
| `result` | un elemento di `results` |
| `error` | `{"image_id" | null, "error": {…schema C.1}}` |
| `done` | `{"status", "batch"}` |

Heartbeat `: ping` ogni 15 s. Il client ricade su polling di `GET /api/jobs/{id}` ogni 1 s se SSE si chiude.

### `GET /api/events` (SSE globale)
Stati dei modelli (`state`) e avvio/fine dei job, per aggiornare badge e `ModelSelect` senza ricaricare.

### `GET /api/predictions?session_id=&model_id=&image_id=`
Lista di predizioni (stesso formato di `results`), più `model_revision`, `created_at`.

### `POST /api/jobs/{job_id}/cancel`
`200 {"status": "canceled"}`; le immagini già completate restano.

---

## C.6 Revisione umana

### `PUT /api/images/{image_id}/review`
```json
{"prediction_id": "pred_<uuid>", "model_id": "<id>",
 "values": {"subject": "<classe>", "background": "<classe>", "perspective": "<classe>",
            "individuals": "<classe>", "text_classification": "<classe>", "text_language": "<classe>"},
 "note": "<facoltativa>"}
```
Valida ogni valore contro `prompt.TAXONOMY`. Il server calcola `changed_dimensions` confrontando con la
predizione e imposta `status`: `reviewed` (nessuna modifica) o `corrected` (almeno una). Risposta `200`:
```json
{"review_id": "rev_<uuid>", "image_id": "img-demo-<uuid>", "status": "corrected",
 "changed_dimensions": ["<dimensione>"], "created_at": "<data>"}
```
Le revisioni sono **dati della demo**: non esiste alcun endpoint che le esporti in un formato di training.

### `GET /api/images/{image_id}/reviews`
Storico delle revisioni dell'immagine.

---

## C.7 Metriche, throughput, export

### `GET /api/metrics/{model_id}?set=coverage`
```json
{
  "model_id": "<id>", "set": "coverage",
  "results": [
    {"eval_id": "<id>", "validity": "phase1_internal", "provenance": "reported_only",
     "badges": ["phase-1 internal set (coverage)", "reported only"],
     "source": "<file:riga>", "checkpoint": "<nome o null>", "trained_on": "<dati>",
     "macro_f1": 0.0, "weighted_f1": 0.0, "accuracy": null,
     "per_dimension": {"subject": {"macro_f1": 0.0, "weighted_f1": null, "support": null}},
     "per_class": {"subject": {"<classe>": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}}},
     "confusion": null,
     "confidence_calibration": null,
     "computed_at": "<data>", "notes": ["<nota>"]}
  ],
  "kaggle_dev": {"score": null, "badge": "Kaggle dev (public)"}
}
```
`set` ∈ {`coverage`, `base`, `prompt-test`, `prompt-dev-600`, `validation_450`}; **`dev` e `final` sono rifiutati
(`VALIDATION_ERROR`)**: il dev non ha label e il final non si valuta. `confusion` (se presente):
`{"dimension": {"labels": […], "matrix": [[0]], "source": "<file:riga o recomputed>"}}`.

### `GET /api/metrics/compare?models=a,b,c&set=coverage`
Stessa struttura, un blocco per modello; i risultati con `validity = seen_in_training` sono **esclusi di default**
(`include_invalid=true` per mostrarli, sempre con badge rosso).

### `GET /api/throughput?model_id=`
```json
{"model_id": "<id>",
 "measured_in_demo": [{"job_id": "<id>", "images": 0, "images_per_second": 0.0, "ms_per_image": 0.0,
                        "gpu_name": "<gpu>", "batch_size": 0, "measured_at": "<data>"}],
 "reported": [{"value": 0.0, "what": "<cosa>", "conditions": "<condizioni>", "source": "<file:riga>"}],
 "cost_per_1k_images_usd": null, "cost_basis": {"gpu_label": "<gpu>", "usd_per_hour": 0.0}}
```
`cost_per_1k_images_usd = usd_per_hour / (images_per_second × 3600) × 1000`, calcolato **solo** da img/s misurati
o riportati con GPU nota; per il gateway Keera `null` ("costo dell'API non pubblico").

### `GET /api/export.csv?session_id=&model_id=&include_reviews=true`
CSV con `image_id` (della demo), `original_filename` (solo se `include_filenames=true`), le 6 dimensioni (valore
rivisto se c'è, altrimenti predetto), `status`, `min_confidence`, `model_id`, `mode`. Header
`Content-Disposition: attachment`. Il file finisce nei download del browser, **non** nel repo; se la sessione
contiene immagini `train_coverage`, la UI avvisa che il CSV contiene predizioni su dati della competizione.

### `POST /api/recorded/rescan`
Rilegge `data/demo/recorded/`; `200 {"runs": 0}`.
