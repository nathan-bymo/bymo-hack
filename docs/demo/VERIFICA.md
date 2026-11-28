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
| 28/11 | **Deroga alla guardia dev/final**: 7 foto di dev ammesse come sample della demo (`data/demo/guard/allow.txt`, gli hash soltanto, fuori da git); tutte le altre restano rifiutate. Scelte dalle predizioni del full fine-tune (niente persone, soggetto prodotto, una per vista), non etichettate a mano. Motivo: servono risposte vere dei modelli senza GPU, e per dev esistono già le probabilità delle submission. Le regole ufficiali permettono di mostrare poche foto della competizione (`challenge_description.md` e `FAQ.md`) | richiesta del team |
| 28/11 | Sample: risposte registrate solo per `apertus-8b-fullft-heads` (probabilità in `data/share/fullft_3ep_dev_probs.pt`, 7 foto su 7, importate con `import_probs.py`); gli altri modelli restano senza risposta finché non arrivano i loro file di probabilità su dev | verifica |

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
| Token di design | prima di ripiego Tailwind; ora **descritti** dalla descrizione degli screenshot (vedi l'ultimo aggiornamento) |

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


## Aggiornamento — token e stile dalla descrizione degli screenshot (27/11, sera)

L'utente ha incollato una **descrizione** degli screenshot A–D (foto da telefono di un televisore: colori ±10–15%,
verde/viola/rosso i meno affidabili; risoluzione 1920×1080 non confermata). Applicata a `tokens.css` (ogni riga
`descritto:`; **non** misurata con D.1.1) e al guscio: linea d'accento di 3 px in alto, header bianco di 56 px con ricerca
arrotondata, pannello principale incorniciato di verde con barra titolo `Products` (18 px, sottolineato), toolbar
allineata a destra, sezioni separate da linea scura di 2 px, colonna destra a schede quadrate, pannello "Image type"
con bordo superiore scuro di 3 px e righe `#A9BCD6`, cornice grigia scura di 8 px attorno all'anteprima, `Cancel`
con gradiente, tendine da 26 px con raggio 2 px; storefront con linea multicolore e pillola nera.
Etichette ERP: confermate come "descritte" le 4 note (Full Product View, Lifestyle Scene, Front – Angled Left,
Person(s) Present) e `-` per Overlay/Language; **ancora ignoto** il valore `none` di Individuals (resta `-`, proposta).
Non applicabile: il pulsante di salvataggio ha etichetta illeggibile nello screenshot (usato "Save"); nessuna voce di
sidebar risulta evidenziata nell'originale (da noi evidenziata quella nuova). Test: 106 pytest e E2E passati dopo la modifica.
La verifica visiva resta **contro una descrizione**, non contro gli screenshot originali.

## Aggiornamento 28/11 notte — modelli del team, pagine Models e Try (branch `feat/demo-modelli`)

| cosa | esito |
|---|---|
| `models.yaml`: modello finale di Nathan (Kaggle dev 0.83177, prompt-test 0.821 held-out, 6,2 img/s H200, 4,9 img/s L40S), P11 con `local_path` ai pesi LFS, full fine-tune `phase2/model/last` (Kaggle dev 0.80473, 9,55 img/s e 18,7 GB su A100), riferimenti Kaggle; righe di fonte P11 corrette (velocità :173, prezzo H200 :174) | ok: l'import riga per riga dei report non segnala discrepanze |
| Costo per 100.000 immagini dal registro: finale 10,20 $ (L40S) / 22,40 $ (H200), full FT 7,27 $ (A100), P11 10,52 $ (H200); zero-shot n/a (prezzo del gateway non pubblico); P10 n/a (velocità non end to end) | ok, test `test_scores_speeds_and_cost_per_100k` |
| pytest backend | **103 passati**, 6 saltati |
| `npm run build` | ok; `oxlint` senza avvisi nuovi nelle pagine aggiunte |
| E2E Playwright (Chrome) esteso: Models (classifica Kaggle con 6 righe, 2 modelli segnati non validi su coverage, costo 7,27 $, filtro esperimenti secondari) e Try (6 dimensioni, disaccordo su perspective tra due registrazioni **segnaposto**) | **1 test, 10 passi, passato** |
| `make -C src/demo demo` | build + API su :8010; `/`, `/image-classification/try`, `/image-classification/models`, `/api/models` → 200 |
| Zero-shot **live** con la chiave in `.env` (mai stampata), su un'immagine sintetica disegnata (nessun dato della competizione) | **ok**: 8B 7,0 s, 70B 23,5 s; 6 dimensioni valide con probabilità; i due modelli non concordano su subject |
| Console del browser sulle pagine nuove | nessun errore |
| Guardia sha256 dev/final | **non costruita su questo Mac**: le cartelle `dev_images`/`final_images` non ci sono (rischio basso: niente immagini dev qui) |
| Worker `vadapter_heads` (modello finale) e `parity.py` esteso | scritti; **mai eseguiti su GPU**: "parity not verified" |
| Modelli allenati in modalità registrato con predizioni **vere** | **non fatto**: serve un host GPU (proposta e costo da approvare) |

## Aggiornamento 28/11 — controllo sui requisiti della challenge

| requisito (challenge_description.md) | nella demo | verifica |
|---|---|---|
| metodo che funziona anche senza contesto prodotto | Classify: "Product details (optional)", risposte segnate "photo only" se il contesto manca | E2E: salvataggio di 2 campi su 5 |
| ≥ 2 img/s, > 100.000 immagini al giorno, memoria, costo per 100.000 | Models: colonne ✓ 2 img/s e "100,000 images take" (ore) | E2E: full fine-tune 2,9 h |
| Insights (15%) e bonus prospettiva | Models: "What mattered" (5) e "What did not help" (4), ognuno con file:riga | fonti rilette riga per riga |
| weighted F1 | Models, test interno: weighted F1 sotto ogni modello | E2E |
| solo Apertus + modelli open-weights a runtime, deployabile on-premises | Models: "What runs at inference" | fonti: RISULTATI.md:119-120, AGENTS.md:58-65 |
| trasparenze composte su bianco | già in `common/images.py` (usato dallo zero-shot) | prova live con PNG trasparente |
| prova live dal browser, chiave reale | 8B + 70B insieme su un PNG sintetico: 9,5 s, nessun errore in pagina né in console | Playwright su :8010 |

## Aggiornamento 28/11 mattina — frontend per il video

- Tre pagine nell'ordine del video: Demo, Results, How it works (prima: Classify e Models).
- Demo: tabella domande × modelli accanto alla foto, tutta visibile a 1440 × 900 senza scorrere (screenshot
  `02b-video-frame` del test).
- Results: gruppo di valutazione della challenge, tabella per domanda sul test held-out (perspective 0.198 → 0.755).
- How it works: schema della pipeline, passi di training con tempi e costi (`RISULTATI.md:45-47, 60-62`).
- Verificato: build, lint (solo avvisi già presenti), test E2E passato, pytest 103 passati e 6 saltati dopo il pull
  di main; prova dal vivo con la chiave vera su una foto sintetica, 70B + 8B in 9,4 s, nessun errore in console.
- Non verificato: modelli allenati su GPU e registrazioni vere (opzione A ancora da lanciare).

## Aggiornamento 28/11 mattina — tutti i modelli rispondono nella demo

- Registrate le risposte vere di modello finale, P13, P12 e P11 su 13 foto del test held-out, dalle probabilità che
  i modelli hanno scritto nei job di valutazione su GPU (`src/demo/scripts/import_probs.py`; dati in
  `data/demo/recorded/`, fuori da git). Ordine delle classi controllato ricalcolando le metriche (prospettiva del
  modello finale 0.754 contro 0.755 del report).
- Upload di una foto registrata: i dettagli prodotto usati dal modello si compilano da soli (test
  `test_upload_of_a_recorded_photo_gets_the_recorded_context`).
- La pagina Demo mostra solo i modelli che rispondono: niente più "Needs a GPU". Il full fine-tune ha probabilità
  solo sulle immagini dev, che nella demo non si usano: compare in Results e How it works, non in Demo.
- P12, P13, P14, P15 e il modello mixed nel registro (P14 anche registrato in Demo); Results con tutti i modelli; How it works con tutte le prove (P0–P15, full fine-tune, mixed,
  modello finale) divise in quattro fasi, ciascuna con il suo set.
- Verificato: pytest 104 passati e 6 saltati, E2E passato, prova sul server vero: foto presa dalla scrivania,
  modello finale + P11 registrati e 70B dal vivo in 4,3 s, nessun errore in console.
