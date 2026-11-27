# Allegato A — Inventario verificato del repository

Parte di [PIANO_DEMO.md](../PIANO_DEMO.md) (sezione 3). Verifiche eseguite il **27/11/2026 tra 20:45 e 21:30 CET**
sulla macchina del piano (laptop). Ogni fatto cita la fonte come `percorso:riga` o `branch:percorso`. Le righe dei
file su `main` si riferiscono al commit `be6ae08`; quelle sul branch `feat/apertus-8b-fullft-heads` al commit
`9145fca`. Lo stato degli artefatti su Hugging Face è quello letto alle **20:08 UTC** (21:08 CET).

Legenda: **[VERIFICATO]** letto nel file o con un comando · **[STIMA]** calcolo nostro, non misurato ·
**[DA VERIFICARE]** non verificabile da qui, vedi [§16 del piano](../PIANO_DEMO.md#16-domande-aperte-per-il-team).

---

## A.1 Albero del repository (solo le parti rilevanti)

```
.                                  (main = be6ae08 = origin/main)
├── README.md                      regole del template; video demo max 2 min (README.md:82-84); src/ = "Your code" (README.md:52)
├── challenge_description.md       task, celle valutate (100-107), punteggio (112-120), classi (126-133), baseline (139-145),
│                                  immagini mostrabili nel pitch (235-236), criteri di valutazione (216-222)
├── FAQ.md                         crediti HF, L40S ~1,80 $/h (FAQ.md:18-22), immagini mostrabili (FAQ.md:114-116)
├── technical_report.md            report da compilare; campo "Demo" (technical_report.md:10)
├── reference/                     notebook Qwen 3.5 9B degli organizzatori
├── .env.example                   LLM_NAME, LLM_BASE_URL, LLM_API_KEY (.env.example:2-4); **.env assente** su questa macchina
├── .gitignore                     dati e pesi esclusi (.gitignore:1-41); data/ alla riga 30, *.csv alla riga 36
├── data/                          IGNORATO (.gitignore:30). Contenuto locale:
│   ├── hackapertus-galaxus/       train.csv, dev.csv, metric.py, labeling_guideline.*, sample_submission.csv,
│   │                              train_images/ (11.876 file, 2,9 GB), dev_images/ (5.261 file, 1,3 GB), final_encrypted.bin
│   ├── prototype_splits/          train_1500.csv, validation_450.csv, distribution.json, README.md
│   ├── runs/hf-lora/prototype-256/ settings.json, train_summary.json, eval_coverage.json, eval_prototype_validation.json (P10)
│   └── hf_jobs/hf_job_fullft.py   job generato del full fine-tune (contiene ID: mai committare)
└── src/
    ├── AGENTS.md                  regole vincolanti del team (prompt v2, due fasi, Keera, Kaggle)
    ├── RISULTATI.md               registro delle prove P0–P11 con i numeri principali
    ├── README.md                  strategia del team (idee, non risultati)
    ├── confronto_lora.md          scheda P10, confronto con P8/P9, F1 per classe
    ├── requirements.txt           pandas==3.1.1, pillow==13.0.0 (src/requirements.txt:1-2)
    ├── common/                    prompt.py (tassonomia e prompt), images.py (preparazione immagini), splits.py (split per hash)
    ├── scripts/                   eval_prompt.py (zero-shot su Keera + scorer di CSV), keera_classify.py (v1, path rotto),
    │                              db_download.py, requirements.txt (+ openai==3.60.0, tqdm)
    ├── docs/system_prompt.txt     prompt condiviso iniziale, solo termine di confronto (src/AGENTS.md:26)
    ├── apertus-8B-benchmark/      report zero-shot P0–P5 (*.md) e lora/ (P10: apertus_lora.py, encode.py, evaluate.py,
    │                              train.py, build_hf_job.py, results/prototype-256/*.json)
    ├── apertus-8B-heads/          P6–P9: teste su Apertus congelato (extract_features.py, heads.py, train_heads.py, report)
    ├── apertus-8B-lora-heads/     P11: LoRA + teste (data.py, encoders.py, model.py, predict.py, make_submission.py, test_pipeline.py)
    ├── apertus-8B-fullft-heads/   VUOTA su main (solo __pycache__ non tracciato). Codice sul branch feat/apertus-8b-fullft-heads
    ├── apertus-8B-vadapter-lora-heads/ VUOTA su main. Codice su origin/feat/vadapter-lora-heads, incluso nel branch del full FT
    ├── apertus-70B-benchmark/     solo .gitkeep
    └── apertus-vadapter-benchmark/ solo .gitkeep
```

**Branch** [VERIFICATO con `git branch -a`, `git log`]:

| branch | commit | contenuto rispetto a main |
|---|---|---|
| `main` = `origin/main` = `origin/feature/sviluppo-nikola` | `be6ae08` | base |
| `apertus-8b-lora` | `4e3b342` | antenato di main, nessuna differenza |
| `origin/feat/vadapter-lora-heads` | `e5130bb` | aggiunge `src/apertus-8B-vadapter-lora-heads/` |
| `feat/apertus-8b-fullft-heads` = origin | `9145fca` | aggiunge `CLAUDE.md`, `docs/context/*.md`, `src/apertus-8B-fullft-heads/`, `src/apertus-8B-vadapter-lora-heads/`; **diverge da main**: non contiene `src/apertus-8B-lora-heads/` né gli ultimi report di main |

Conseguenza per la demo: i moduli del full FT non esistono su main. Un merge dei due branch dà un conflitto banale
in `src/AGENTS.md:10` (le due righe aggiunte nello stesso punto). Il piano propone un `git worktree` del branch del
full FT invece di un merge (vedi [allegato B, B.6](B_registro_modelli.md#b6-come-il-worker-importa-il-codice-del-repo-senza-modificarlo)).

**File citati nel prompt ma assenti su main** [VERIFICATO]: `CLAUDE.md` e `docs/context/*.md` esistono solo sul
branch `feat/apertus-8b-fullft-heads` (`git show feat/apertus-8b-fullft-heads:CLAUDE.md`). `docs/demo/reference/`
**non esiste**: gli screenshot non sono stati copiati (nessun PNG/JPG recente trovato nemmeno in `~/Downloads`,
`~/Pictures`, `~/Desktop`).

---

## A.2 Tassonomia: 41 classi

**Fonte di verità:** `src/common/prompt.py:26-42` (`TAXONOMY`). L'ordine fissa l'indice usato dalle teste
(`src/apertus-8B-lora-heads/predict.py:145`: `D.TAXONOMY[d][int(argmax)]`; stesso schema in
`feat/apertus-8b-fullft-heads:src/apertus-8B-vadapter-lora-heads/data.py:159-164`) e la lettera del prompt v2
(`src/common/prompt.py:44`, `CODES`). Le stesse classi sono in `challenge_description.md:126-133`.
[VERIFICATO con un one-liner il 27/11]: `prompt.DIMENSIONS == metric.DIMENSIONS` e, per ogni dimensione,
`set(prompt.TAXONOMY[d]) == metric.VALID_CLASSES[d]` (`data/hackapertus-galaxus/metric.py:18-20`); totale 41.

Colonna **Etichetta ERP**: la demo mostra le stringhe interne come valore di verità e un'etichetta "stile ERP" come
testo. Solo 4 etichette e i nomi delle 6 dimensioni sono noti, dal prompt del team che descrive lo screenshot C
(gli screenshot non sono disponibili: **[DA VERIFICARE]** su `docs/demo/reference/`). Per le altre 37 la demo usa una
**regola automatica dichiarata** (`auto`: snake_case → Title Case, per esempio `detail_closeup` → "Detail Closeup"),
mostrata come tale nel file `src/demo/config/erp_labels.yaml`. Nessuna etichetta inventata viene presentata come
etichetta Galaxus.

| dimensione (dataset) | nome ERP | idx | lettera v2 | classe (stringa esatta) | etichetta ERP | fonte classe |
|---|---|---|---|---|---|---|
| `subject` | Subject | 0 | A | `full_product` | Full Product View *(prompt, da verificare)* | prompt.py:28 |
| | | 1 | B | `detail_closeup` | auto | prompt.py:28 |
| | | 2 | C | `accessories` | auto | prompt.py:28 |
| | | 3 | D | `packaging` | auto | prompt.py:28 |
| | | 4 | E | `package_and_product` | auto | prompt.py:28 |
| | | 5 | F | `multipack_trade_unit` | auto | prompt.py:29 |
| | | 6 | G | `delivery_scope` | auto | prompt.py:29 |
| | | 7 | H | `product_grouping` | auto | prompt.py:29 |
| | | 8 | I | `multiview` | auto | prompt.py:29 |
| | | 9 | J | `energy_label` | auto | prompt.py:29 |
| | | 10 | K | `infographic` | auto | prompt.py:30 |
| | | 11 | L | `logo` | auto | prompt.py:30 |
| | | 12 | M | `placeholder` | auto | prompt.py:30 |
| `background` | Background | 0 | A | `white` | auto | prompt.py:32 |
| | | 1 | B | `color` | auto | prompt.py:32 |
| | | 2 | C | `scene_lifestyle` | Lifestyle Scene *(prompt, da verificare)* | prompt.py:32 |
| `perspective` | Perspective | 0 | A | `front_head_on` | auto | prompt.py:34 |
| | | 1 | B | `front_angle_left` | Front – Angled Left *(trattino lungo "–"; prompt, da verificare)* | prompt.py:34 |
| | | 2 | C | `front_angle_right` | auto | prompt.py:34 |
| | | 3 | D | `back_head_on` | auto | prompt.py:34 |
| | | 4 | E | `back_angle_left` | auto | prompt.py:34 |
| | | 5 | F | `back_angle_right` | auto | prompt.py:35 |
| | | 6 | G | `left_side` | auto | prompt.py:35 |
| | | 7 | H | `right_side` | auto | prompt.py:35 |
| | | 8 | I | `top_down` | auto | prompt.py:35 |
| | | 9 | J | `bottom_up` | auto | prompt.py:35 |
| `individuals` | Individuals | 0 | A | `individual_present` | Person(s) Present *(prompt, da verificare)* | prompt.py:37 |
| | | 1 | B | `none` | `-` *(proposta, da verificare)* | prompt.py:37 |
| `text_classification` | Overlay | 0 | A | `certification_seal` | auto | prompt.py:39 |
| | | 1 | B | `technical_specs` | auto | prompt.py:39 |
| | | 2 | C | `promotional_text` | auto | prompt.py:39 |
| | | 3 | D | `nutritional_info` | auto | prompt.py:39 |
| | | 4 | E | `other` | auto | prompt.py:39 |
| | | 5 | F | `none` | `-` *(prompt: "Overlay … `-`")* | prompt.py:39 |
| `text_language` | Language | 0 | A | `German` | auto (= German) | prompt.py:41 |
| | | 1 | B | `English` | auto | prompt.py:41 |
| | | 2 | C | `French` | auto | prompt.py:41 |
| | | 3 | D | `Italian` | auto | prompt.py:41 |
| | | 4 | E | `Dutch` | auto | prompt.py:41 |
| | | 5 | F | `other` | auto | prompt.py:41 |
| | | 6 | G | `none` | `-` *(prompt: "Language … `-`")* | prompt.py:41 |

Conteggi: subject 13, background 3, perspective 10, individuals 2, text_classification 6, text_language 7 = **41**.

### A.2.1 `none`, `relevant` e `-`

- **`none` è una classe vera**, valutata come le altre (`challenge_description.md:135`). Nel file di submission è
  sempre minuscola: `None` con la N maiuscola è letta come cella vuota (`challenge_description.md:188-190`).
  Esiste in `individuals`, `text_classification` e `text_language`.
- **`relevant`** è una colonna del dataset (`<dim>_relevant`, `train.csv`); una cella conta solo se
  `<dim>_source == "human"` **e** `<dim>_relevant == "true"` (`src/common/splits.py:33-37`, `scored_cells`;
  stessa regola in `src/scripts/eval_prompt.py:197-203`). Le regole che la producono sono applicate **alle label
  umane**, mai alle predizioni (`challenge_description.md:109-110`):

| regola ufficiale (`challenge_description.md:102-107`) | dimensioni non valutate |
|---|---|
| subject `logo`, `placeholder` o `energy_label` | background, perspective, individuals, text_classification |
| subject `infographic` | background, perspective, individuals |
| subject `delivery_scope`, `detail_closeup` o `product_grouping` | perspective |
| background `scene_lifestyle` | subject, perspective |
| text_language | valutata **solo** se c'è un overlay di testo, oppure se subject è `infographic`, `packaging`, `package_and_product` o `full_product` |

- **Nella demo** il campo `applicable` di ogni dimensione è calcolato applicando **le stesse regole alla
  predizione** (subject, background e text_classification predetti). È una convenzione di interfaccia, dichiarata
  in UI ("regola ufficiale applicata alla predizione: non è la regola di punteggio"). Interpretazione di "overlay
  di testo presente": `text_classification != "none"` **[DA VERIFICARE con il team; default proposto]**.
- **`-` in ERP**: secondo il prompt del team, Overlay e Language appaiono come `-` quando non applicabili. La demo
  mostra `-` (a) per il valore `none` di Overlay/Language/Individuals e (b) per le dimensioni con
  `applicable = false`, distinguendo i due casi con un'icona e un tooltip; il valore interno resta visibile nel
  popover e nell'export.

---

## A.3 Codice condiviso riusato dalla demo

| funzione / costante | file:riga | cosa fa | usata da |
|---|---|---|---|
| `TAXONOMY`, `DIMENSIONS`, `CODES` | src/common/prompt.py:26-44 | classi, ordine, lettere | tutto |
| `context_block(row)` | src/common/prompt.py:205-208 | blocco `<context>` sanificato (≤200 caratteri per campo); senza contesto produce `<context>\nnot available\n</context>` [VERIFICATO] | tutti i modelli |
| `DESCRIBE`, `DESCRIBE_MAX_TOKENS` | src/common/prompt.py:273-279 | richiesta di descrizione (≤160 token) | zero-shot v2 |
| `describe_messages(url)` | src/common/prompt.py:358-359 | messaggio immagine + descrizione | zero-shot v2 |
| `question_messages(row, url, d, answer, use_context, description)` | src/common/prompt.py:362-368 | domanda per dimensione dopo la descrizione | zero-shot v2 |
| `question_constraint(d, "codes")` | src/common/prompt.py:371-375 | regex di una lettera, `max_tokens` 2, `top_logprobs` 20 | zero-shot v2 |
| `parse_question(choice, d, "codes")` | src/common/prompt.py:389-396 | (classe, distribuzione) o (None, None) | zero-shot v2 |
| `_distribution(d, token)` | src/common/prompt.py:378-386 | probabilità per classe dalle top-20 logprob, rinormalizzate | zero-shot v2 |
| `opaque_rgb_on_white(image)` | src/common/images.py:13-20 | trasparenza su bianco | tutti |
| `target_size(w, h, side)` | src/common/images.py:23-26 | area ≈ side², proporzioni (cap 8), lati multipli di 16 | tutti |
| `prepared_jpeg(path, side, mirrored)` | src/common/images.py:29-37 | bianco → resize LANCZOS → JPEG q92 | tutti |
| `prepared_image(path, side, mirrored)` | src/common/images.py:40-42 | JPEG preparato ridecodificato (modelli locali) | P10, P11, full FT |
| `data_url(path, side)` | src/common/images.py:45-47 | JPEG preparato come data URL base64 | zero-shot (API) |
| `scored_cells(rows)` | src/common/splits.py:33-37 | maschera human ∧ relevant | metriche |
| `split_rows(rows, name)` | src/common/splits.py:64-73 | `dev`, `test`, `base`, `coverage` (per hash del prodotto) | script di valutazione |
| `training_rows(rows)` | src/common/splits.py:76-79 | train meno i prodotti di coverage | controlli |
| `complete(body)` | src/scripts/eval_prompt.py:72-92 | POST `/chat/completions` con retry; legge `LLM_BASE_URL`, `LLM_API_KEY` dall'ambiente | zero-shot |
| `solution_frame(rows)` | src/scripts/eval_prompt.py:197-203; src/apertus-8B-lora-heads/data.py:157-163 | soluzione per metric.py (celle non valutate vuote) | metriche |
| `metric.score`, `metric.evaluate` | data/hackapertus-galaxus/metric.py:68-94 (file Kaggle, **non** nel repo) | punteggio ufficiale e dettaglio per classe | metriche |

**Funzione di punteggio esistente:** il repo non ha una propria implementazione della metrica: usa sempre
`metric.py` dei dati Kaggle, caricato con `importlib` da un percorso locale (`src/scripts/eval_prompt.py:59-63`,
`src/apertus-8B-lora-heads/data.py:166-172`). La demo fa lo stesso (mai copiarlo nel repo).

Comportamento di `metric.evaluate` [VERIFICATO leggendo `data/hackapertus-galaxus/metric.py:73-180`]:
classi per dimensione = classi nelle label ∪ classi **valide** predette; una predizione vuota è sbagliata ma non crea
una classe; `f1_weighted` della dimensione pesa le F1 per supporto; `overall.f1_macro` = media delle dimensioni con
almeno una cella valutata; `overall.f1_weighted` = media delle weighted F1 pesata per supporto. Una classe
sconosciuta nella submission solleva `ParticipantVisibleError` (`metric.py:122-130`).

---

## A.4 Schede modello

Formato comune: ID · famiglia · training · codice · pesi · base · preprocessing · output · confidenza · VRAM ·
velocità · eseguibile qui? · stato. "Qui" = questo laptop (vedi A.6: nessuna GPU utilizzabile).

### A.4.1 `apertus-8b-zeroshot-v2` — Apertus 8B zero-shot, prompt v2 (prova P4)

| voce | valore |
|---|---|
| famiglia | zero-shot generativo, remoto (API OpenAI-compatibile) |
| training | nessuno |
| endpoint | gateway Keera `LLM_BASE_URL=https://gateway.keera.ch/api/v1`, modello `apertus-v1-5-8b` (src/AGENTS.md:58-65; nome costruito in src/scripts/eval_prompt.py:98) |
| codice | orchestrazione in `run_variant.predict` (src/scripts/eval_prompt.py:122-146, funzione interna: **non importabile**, va ricomposta), parsing in `run_variant.parse` (eval_prompt.py:148-155) |
| spec | `8b:describe:codes:512` (src/AGENTS.md:16-24) |
| preprocessing | 1) `images.data_url(path, 512)` (images.py:45) → JPEG q92, area ≈ 512², su bianco. 2) Richiesta di descrizione: `describe_messages(url)` con `{"max_tokens": DESCRIBE_MAX_TOKENS}` (eval_prompt.py:135-137). 3) Per ognuna delle 6 dimensioni, in ordine `DIMENSIONS`: `question_messages(row, url, d, "codes", use_context, description)` con `question_constraint(d, "codes")` (eval_prompt.py:140-143). Corpo di ogni richiesta: `{"model", "temperature": 0, "messages", **constraint}` (eval_prompt.py:118-120). Nessun messaggio system (src/AGENTS.md:16). 7 richieste per immagine. |
| output | `parse_question(choice, d, "codes")` → classe o `None` per dimensione (prompt.py:389-396) |
| confidenza | distribuzione completa per dimensione dalle top-20 logprob della lettera (`_distribution`, prompt.py:378-386), condizionata alla descrizione generata |
| VRAM | nessuna in locale |
| velocità misurata | 3,8 img/s con 32 immagini in parallelo, latenza mediana 8,0 s/immagine, ~10.011 token di prompt per immagine, gateway condiviso (src/apertus-8B-benchmark/prompt_eval_coverage.md:97) |
| metriche | coverage 0.528 (prompt_eval_coverage.md:45), base 0.513 (:57), prompt-test 0.502 (:69); per classe in prompt_eval_coverage.md:107-178, P/R in :326-400 |
| eseguibile qui | **sì, se** `LLM_API_KEY` è configurata: `.env` assente su questa macchina [VERIFICATO] |
| stato | `unavailable: missing LLM_API_KEY` finché non c'è la chiave; poi `ready` |

### A.4.2 `apertus-70b-zeroshot-v2` — Apertus 70B zero-shot, prompt v2 (P4)

Come A.4.1 con modello `apertus-v1-5-70b` (src/AGENTS.md:64). Velocità misurata 2,8 img/s, latenza mediana
11,1 s/immagine (prompt_eval_coverage.md:99). Metriche: coverage 0.583 (:47), base 0.587 (:59), prompt-test
0.590 (:71). Eseguibile solo da remoto (Keera o server proprio); in locale servirebbero circa 140 GB di VRAM solo per
i pesi bf16 **[STIMA: 70·10⁹ × 2 byte]**. Stato: come l'8B.

Nota su entrambi i zero-shot: la v2 è stata **scelta sulle prime 600 di prompt-dev**, che fanno parte di coverage
(docs/context/EXPERIMENTS.md:80-82 sul branch del full FT; `src/apertus-8B-benchmark/prompt_eval_coverage.md:61`
indica prompt-test come vista "non usata per scegliere il prompt"). La vista più pulita è **prompt-test**.

### A.4.3 `apertus-8b-lora-p10` — Apertus 8B + LoRA, risposta JSON (P10)

| voce | valore |
|---|---|
| famiglia | generativo con LoRA, locale su GPU |
| training | `data/prototype_splits/train_1500.csv` (1.500 immagini dentro `training_rows`), 3.952 esempi/epoca, 3 epoche, 741 passi (data/runs/hf-lora/prototype-256/train_summary.json; src/confronto_lora.md:11) |
| LoRA | r 32, alpha 32, dropout 0, target `.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj)$`, peft 0.22.0 (adapter_config.json su HF, letto il 27/11) |
| job | HF Job `6b09964d6c5d61fff1893dfe`, h200, **COMPLETED** (letto con `HfApi().inspect_job`) |
| pesi | repo HF **privato** `StoryofLegend/apertus-8b-galaxus-lora`, revisione `be79276586ea5610d3a96a081988da8d977e7af3`. `prototype-256/adapters/final/` contiene **solo** `adapter_config.json`: **l'adapter finale è perso** (src/apertus-8B-benchmark/lora/train.py:171; src/confronto_lora.md:200). Disponibili gli snapshot `prototype-256/adapters/step-00123 … step-00738`, 0,319 GB ciascuno [VERIFICATO con `repo_info`]. Proposta: **`step-00738`** (3 passi prima della fine, lr ≈ 0: docs/context/EXPERIMENTS.md:149-150 sul branch del full FT) |
| base | `swiss-ai/Apertus-v1.5-8B` — **revisione non fissata nel codice P10** (docs/context/PROJECT_CONTEXT.md:174-175 sul branch); la demo usa `1519de0ae7f3d114879291810a58259966f64c99`, l'unica presente nella cache locale [VERIFICATO in `~/.cache/huggingface/hub`] **[DA VERIFICARE che sia quella del job]** |
| codice | `src/apertus-8B-benchmark/lora/apertus_lora.py` (libreria) + `evaluate.py:94-193` (logica inline in `main()`, **non importabile come funzione**) |
| preprocessing | 1) `L.load_vision_tokenizer(model_dir, device)` (apertus_lora.py:162-192): VQ fp32, TF32 disattivato, processor. 2) `images.prepared_image(path, 256)` (images.py:40). 3) `L.encode_image(vq, processor, image)` → (h, w, codici int32) (apertus_lora.py:195-206). 4) `L.Sequences(tokenizer, config.image_token_offset)` (:212); `prompt_ids(row, h, w, codes, use_context=True)` (:253-255): chat template con `SYSTEM_PROMPT` breve (:31-35) e `context_block` come testo utente, `enable_thinking=False` (:236-238). 5) `L.collate([{"input_ids": p, "labels": p}], pad_id)` (:313-327, padding a sinistra; uso in evaluate.py:148) |
| modello | `Apertus1p5ForConditionalGeneration.from_pretrained(dir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=L.ATTENTION)`; `vision_tokenizer = None`, `audio_tokenizer = None`; `PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()` (evaluate.py:136-142) |
| decoding | `generate(max_new_tokens=96, do_sample=False, eos_token_id=sequences.end, pad_token_id=tokenizer.pad_token_id)`, decode con `skip_special_tokens=True` (evaluate.py:150-159); batch 32 ordinati per lunghezza (evaluate.py:144-147) |
| output | `L.parse_answer(text)` (apertus_lora.py:283-290): JSON con le 6 classi o `None` se non valido (anche una sola dimensione fuori tassonomia invalida tutto) |
| confidenza | **non estratta nel repo**; vedi metodo A/B nel piano §6 |
| VRAM | non misurata in inferenza. [STIMA] ~20–24 GB: pesi bf16 del modello completo 18,4 GB (`model.safetensors.index.json`, total_size 18.396.023.436 byte) + VQ separato 0,87 GB + cache KV per batch 32 |
| velocità | 18,0 img/s su coverage, H200, batch 32 (eval_coverage.json:6) — **solo `generate`, codifica VQ esclusa** (docs/context/EXPERIMENTS.md:148 sul branch); non confrontabile con P11 |
| metriche | **dell'adapter `final` perso**: coverage 0.734 (src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json; src/confronto_lora.md:44), validation_450 0.696 (confronto_lora.md:34); 21 risposte non valide su coverage (eval_coverage.json:5). `step-00738` non è mai stato valutato |
| eseguibile qui | no (serve GPU) |
| stato | `needs GPU`; metriche `reported only` riferite a un adapter diverso da quello disponibile |

### A.4.4 `apertus-8b-lora-heads-p11` — Apertus 8B + LoRA + 6 teste, training completo di fase 1 (P11)

Non elencato nel prompt di partenza: è il **migliore misurato su coverage** (src/RISULTATI.md:67).

| voce | valore |
|---|---|
| famiglia | teste di classificazione su decoder con LoRA (fusa), locale su GPU |
| training | `training_rows`: 9.344 immagini, 12.491 esempi/epoca, 3 epoche, cross-entropy senza pesi, label umane, contesto prodotto incluso, seed 3407 (src/apertus-8B-lora-heads/results_full_h200.md:9-16, 172) |
| architettura | 32 layer, LoRA r32 su q,k,v,o,up,down; feature = [ultimo token, media token VQ] 2×4096; teste LayerNorm → 512 → GELU → dropout → 6 lineari (src/apertus-8B-lora-heads/model.py:25-37, 59-64) |
| pesi | repo HF **privato** del collega che ha lanciato il job (Nikola Manev, autore del commit be6ae08); nome indicato solo come `<utente>/galaxus-lora-heads-private` (results_full_h200.md:169), cartella attesa `full-train/model/final/` con `lora/`, `heads.pt`, `settings.json` (PREFIX = `{SPLIT}-{MODE}`, build_hf_job.py:77, 182, 207-209; model.py:129-139). **[DA VERIFICARE]** nome del repo, accesso in lettura per l'account `StoryofLegend` (oggi `list_models(author="StoryofLegend")` mostra solo i due repo P10 e full FT) |
| base | `swiss-ai/Apertus-v1.5-8B` @ `1519de0ae7f3d114879291810a58259966f64c99` (src/apertus-8B-lora-heads/encoders.py:19-20) |
| codice | `predict.py:65-179` (comando end-to-end); funzioni: `model.load(directory, folder, device, merge=True)` (model.py:142-156), `encoders.ImageEncoder(directory, device)` (encoders.py:31), `.encode(paths, variants, side)` (:69), `encoders.PromptBuilder(tokenizer, image_token_offset)` (:90), `.ids(row, grid, codes, use_context)` (:125-128), `encoders.collate(sequences, pad_id)` (:131-139), `predict.fingerprint(checkpoint, use_context)` (predict.py:38-44) |
| preprocessing | `images.prepared_image(path, settings["side"])` = 256 (encoders.py:83-87) → VQ fp32 → sequenza: chat template senza system, utente = `[immagine] + context_block(row) + "\n\n" + TASK` (encoders.py:21-22, 125-128), generation prompt, `enable_thinking=False` (:104-113) → padding a sinistra |
| inferenza | `torch.autocast(bf16)`; `classifier(input_ids, attention)` → 6 tensori di logit (predict.py:139-141) |
| output | `softmax` per testa, label = argmax (predict.py:142-147); mai celle vuote né classi inventate (README.md:14) |
| confidenza | softmax delle teste, distribuzione completa; già salvata in `<out>.probs.pt` e `.partial.jsonl` (predict.py:144-147, 160-162) |
| VRAM | **misurata**: picco allocato 22,0 GB a batch 32 su H200 (results_full_h200.md:156) |
| velocità | **misurata**: 13,2 img/s dall'immagine al CSV, batch 32, H200 (results_full_h200.md:156); stima 100.000 immagini ≈ 2,1 h ≈ 10,5 $ a 5 $/h (:159-160) |
| metriche | coverage 0.781, base 0.786, prompt-test 0.791; per classe (P/R/F1) su coverage; **matrice di confusione di perspective su coverage** (results_full_h200.md:21-25, 46-136) |
| limite noto | comportamento **senza contesto prodotto non misurato** (results_full_h200.md:17) |
| eseguibile qui | no (serve GPU ≥ 24 GB **[STIMA]** dal picco misurato) |
| stato | `weights missing` (accesso da verificare) → `needs GPU` |

### A.4.5 `apertus-8b-fullft-heads` — Apertus 8B full fine-tune + 6 teste, fase 2

| voce | valore |
|---|---|
| famiglia | teste su decoder interamente allenato, locale su GPU |
| architettura | decoder **troncato al layer 24**: i layer 25–32 sono **eliminati**, non congelati; tutti i pesi dei 24 layer e la norm finale allenati (5,2 B parametri); embedding e VQ congelati (feat/apertus-8b-fullft-heads:src/apertus-8B-fullft-heads/README.md:3-15, 45-49). Correzione al prompt di partenza ("24 of 32 layers trained") |
| training | **fase 2**: tutte le 11.876 immagini, 16.560 item/epoca, focal γ=2 + alpha `effective`, solo celle valutate, lr 1e-5 backbone / 1e-3 teste, batch 16, AdamW 8-bit, seed 3407 (README.md:31-58 del branch). Job: `PHASE=2`, `EPOCHS=3`, flavor `a100-large` [VERIFICATO con `inspect_job`] |
| job | HF Job `6b09cdb46c5d61fff189842a`, **RUNNING** alle 20:08 UTC (creato 18:36 UTC) |
| pesi | repo HF **privato** `StoryofLegend/apertus-8b-galaxus-fullft`, revisione `711018b7a25dc130d04074331f1aa6946f727e23` (ultima modifica 20:52 UTC; riverificato il 27/11, in precedenza `167c04f` delle 19:57). **Presenti** `phase2/model/epoch-1/` ed `epoch-2/` (dalle 20:52 UTC), ciascuno con `decoder.safetensors` 10,469 GB, `heads.pt` 0,017 GB, `settings.json` [VERIFICATO]. Il percorso è `phase2/model/…`, non `phase2/epoch-1/` come nel prompt; `last/` e `phase2/predictions/dev_submission.csv` **non esistono ancora** (attesi da ft_train.py:250-256, build_hf_job.py:265-284 del branch) |
| settings.json (epoch-1) | `layers` 24, `head_hidden` 512, `head_dropout` 0.3, `side` 256, `model` swiss-ai/Apertus-v1.5-8B, `revision` 1519de0…, `task` = stesso TASK di P11, `features` "last token + mean of the VQ image tokens", `tiny` false [VERIFICATO scaricando solo il JSON] |
| codice (solo sul branch) | `ft_predict.py:34-109`; funzioni: `ft_model.load(directory, folder, device)` (ft_model.py:153-156) → `build` (:109-114) + `load_weights` (:139-150); `ft_encode.VQEncoder(directory, device)` (ft_encode.py:55-79), `.encode(paths, variants, side)` (:81-90); `ft_encode.model_dir(full=True)` (:42-45); `encoders.PromptBuilder(tokenizer, offset, prefix_tokens=0)` del vadapter (vadapter encoders.py:158-197; uso in ft_predict.py:64); `encoders.collate` (vadapter encoders.py:200-208); `ft_train.autocast(device)` (ft_train.py:68-69); `data.prediction_frame` (vadapter data.py:159-164) |
| preprocessing | identico a P11 nei passi (prepared_image 256 → VQ → `context_block + "\n\n" + TASK`), ma con i moduli del vadapter e `prefix_tokens=0` (nessuno slot SigLIP) |
| output | softmax per testa, argmax (ft_predict.py:86-95) |
| confidenza | softmax delle teste; **addestrate con focal + alpha**: probabilità non paragonabili a quelle di P11 (cross-entropy) |
| VRAM | non misurata. [STIMA] picco ~24 GB al caricamento: il modello completo bf16 viene caricato sul device prima del taglio (vadapter model.py:113-123), poi `load_file(..., device=device)` porta 10,5 GB di pesi sul device (ft_model.py:145-146) |
| velocità | non misurata |
| metriche | **nessuna valida in locale**: ha visto tutto il train, coverage compreso. Unico numero valido: punteggio Kaggle dev, **non ancora disponibile** (src/RISULTATI.md:90) |
| eseguibile qui | no (serve GPU) |
| stato | `needs GPU`; checkpoint consigliato: `last` quando esiste, altrimenti `epoch-1` (solo come anteprima, etichettata) |

### A.4.6 `apertus-8b-frozen-heads-p8` / `apertus-8b-frozen-heads-p9` — Apertus 8B congelato + teste

| voce | valore |
|---|---|
| training | `train_1500` di Nathan: 1.199 per i pesi + 301 per scegliere layer ed epoca; 512²; P8 un seed, P9 5 seed (src/apertus-8B-heads/results_small_h200.md:5-17; src/RISULTATI.md:34-35) |
| codice | `extract_features.py` (feature per layer, stato dell'ultimo token + media token immagine) + `train_heads.py` (teste, standardizzazione con media/std del fit: train_heads.py:173-176, salvate nel checkpoint :194). **Non esiste un predittore end-to-end immagine → classi** |
| pesi | `data/share_heads/`, `data/hf_results/` su un'altra macchina (src/RISULTATI.md:97-100) e un repo privato `<utente>/galaxus-heads-private` (results_small_h200.md:240): **assenti qui** |
| metriche | solo validation_450 (bilanciato, non confrontabile con coverage): P8 cross-entropy 0.542, P9 pesi forti 0.557 (src/RISULTATI.md:47-53) |
| stato | `unavailable: code missing + weights missing`. Nella demo: solo metriche `reported only`, fuori dall'MVP |

---

## A.5 Artefatti di valutazione

"Visto in training?" significa: le immagini del set erano nel training di quel modello.

| set | modello | file (fonte) | visto in training? | per classe? | predizioni per immagine? |
|---|---|---|---|---|---|
| coverage / base / prompt-test | 8B e 70B zero-shot v2 | src/apertus-8B-benchmark/prompt_eval_coverage.md:37-72 (per dimensione), :107-325 (F1 per classe), :326-400 (P/R) | no training; prompt scelto su prompt-dev ⊂ coverage | sì (markdown) | no: risposte in `data/runs/` di un'altra macchina (docs/context/EXPERIMENTS.md:47-49 sul branch) |
| prime 600 di prompt-dev | 8B/70B zero-shot (v1, v2) | src/apertus-8B-benchmark/prompt_eval.md:39-55; prompt_eval_per_class_600.md | prompt scelto qui: ottimistico | sì | no |
| 103 immagini (solo subject) | zero-shot, 15 config | src/apertus-8B-benchmark/prompt_eval.md (P2) | — | parziale | no |
| coverage / base / test | P10 (adapter `final` perso) | src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json (JSON completo: f1_macro, f1_weighted, precision/recall macro, per_class con P/R/F1/support) | no (held-out) | **sì (JSON)** | no (predictions_*.csv non caricato su HF né presente qui) |
| validation_450 | P10 | …/prototype-256/eval_prototype_validation.json | no | sì (JSON) | no |
| coverage / base / prompt-test | P11 | src/apertus-8B-lora-heads/results_full_h200.md:21-25 (per dimensione), :46-120 (per classe solo coverage), :122-136 (confusione perspective) | no (held-out, fase 1) | sì per coverage | no: "predizioni scaricate" sulla macchina del collega (:16) |
| validation_450 | P8, P9 teste congelate | src/apertus-8B-heads/results_small_h200.md, results_small_h200_per_class.md, results_weights_small.md; src/confronto_lora.md:30-36 | no | sì (markdown) | no |
| coverage | full FT fase 2 | nessun file | **sì: contaminato** | — | — |
| Kaggle dev | tutti | nessuno: nessun punteggio Kaggle registrato (src/RISULTATI.md:90) | no | no | no |
| riferimento | Qwen 3.5 9B FT | challenge_description.md:143 (dev 0.917); holdout 600 del notebook src/README.md:133-143 (set diverso) | — | no | no |

Riepilogo: **nessuna predizione per immagine** è disponibile su questa macchina per nessun modello. Le uniche
metriche leggibili da macchina sono i JSON di P10; tutto il resto è markdown. L'unica matrice di confusione è
quella di perspective di P11.

---

## A.6 Ambiente

| voce | valore | fonte |
|---|---|---|
| GPU | NVIDIA GeForce MX130 (GM108M, 2 GB, Maxwell) + Intel UHD 620; `nvidia-smi` fallisce (driver non caricato) | `lspci`, `nvidia-smi` |
| RAM / CPU | 7 GB / 8 thread | `free -g`, `nproc` |
| Python del progetto | `.venv` con Python 3.12.15 | `.venv/bin/python -c "import sys"` |
| librerie nel `.venv` | torch 2.15.0+cpu (CUDA non disponibile), transformers 5.14.0.dev0 dal fork swiss-ai al commit `3797303dda74844e3d1f8977ff5518bb91f818b4`, peft 0.22.0, accelerate 1.16.0, safetensors 0.8.0, pandas 3.1.1, pillow 13.0.0, numpy 2.5.5, huggingface_hub 1.33.0, pydantic 2.14.4, openai 3.60.0, tokenizers 0.22.2; **assenti**: fastapi, uvicorn, pytest, bitsandbytes, vllm | `importlib.metadata` |
| versioni dei job GPU | torch==2.8.0, torchvision==0.23.0, transformers fork @3797303, peft==0.22.0, accelerate==1.16.0 (src/apertus-8B-lora-heads/build_hf_job.py:40-55); full FT anche bitsandbytes==0.48.2 (requirements-gpu.txt del branch), non necessario in inferenza (import pigro in ft_train.py:110) | |
| Python di sistema | `/usr/bin/python3.11` (3.11.18) | `python3.11 --version` |
| Node | node v22.22.2, npm 10.9.7, pnpm presente | `node --version` |
| frontend esistente | **nessuno** (nessun package.json nel repo) | `find` |
| cache HF | `swiss-ai/Apertus-v1.5-8B` @1519de0: solo config, tokenizer, chat template, processor e `model-vision_tokenizer-model.safetensors` (875 MB in tutto); **pesi del LM assenti** (18,4 GB in totale). Token HF presente in `~/.cache/huggingface/token` (account `StoryofLegend`, nessuna organizzazione) | `ls`, `HfApi().whoami()` |
| config Apertus | 32 layer, hidden 4096, vocab 266.752, `image_token_offset` 131.272, codebook VQ 131.072, `audio_token_offset` 262.344; processor `min_pixels` 65.536 (=256²), `max_pixels` 1.960.000, `spatial_factor` 16 | config.json, processor_config.json nella cache |
| `.gitignore` | vedi .gitignore:1-41; `data/` (riga 30) ignora **qualsiasi** cartella chiamata `data`, ovunque | `git check-ignore -v` |

---

## A.7 Gap trovati e gestione proposta

| # | gap | impatto | gestione proposta |
|---|---|---|---|
| G1 | screenshot di riferimento assenti (`docs/demo/reference/` non esiste) | token di design non misurabili | procedura di misura in [allegato D, D.1](D_interfaccia.md#d1-design-token); palette di ripiego dichiarata finché mancano |
| G2 | `CLAUDE.md` e `docs/context/` solo sul branch del full FT | contesto non su main | letti dal branch; nessuna azione per la demo |
| G3 | codice del full FT solo su branch divergente | import impossibile da main | `git worktree` del branch (B.6) o attesa del merge del team |
| G4 | full FT: solo `epoch-1`, job in corso; percorsi diversi dal prompt | checkpoint finale non ancora esistente | `models.yaml` con revisione fissata; aggiornare a `last` quando esiste |
| G5 | adapter `final` di P10 perso | metriche riferite a pesi non disponibili | usare `step-00738` con badge "reported only (adapter final)"; rivalutare solo con ok |
| G6 | pesi di P11 in un repo privato di un collega | il modello migliore non è caricabile | chiedere accesso o modalità "registrato" prodotta dal collega |
| G7 | pesi e predittore end-to-end di P8/P9 assenti | non eseguibili | solo metriche riportate, fuori MVP |
| G8 | nessuna predizione per immagine in locale | niente confusioni (tranne perspective P11), niente calibrazione | `evaluate.py` offline su GPU, solo con ok |
| G9 | nessun punteggio Kaggle dev | l'unico numero valido del full FT manca | campo inserito a mano dall'utente in `models.yaml`, mai stimato |
| G10 | `.env` assente | zero-shot non eseguibile | l'utente configura `LLM_API_KEY` localmente |
| G11 | nessuna GPU utilizzabile qui | nessun modello locale | worker su host GPU o modalità "registrato" |
| G12 | `eval_prompt.py` cerca `data/train.csv`, `data/metric.py`, `data/train_images/` ma i file sono in `data/hackapertus-galaxus/` (docs/context/PROJECT_CONTEXT.md:220-223 sul branch) | lo script del repo non gira com'è | symlink locali in `data/` (ignorati) oppure percorsi configurabili nella demo |
| G13 | etichette ERP note solo per 4 classi | testo UI incompleto | regola `auto` dichiarata; domanda al team |
| G14 | comportamento senza contesto prodotto non misurato (results_full_h200.md:17) | upload senza contesto = condizione non misurata | form di contesto opzionale + avviso in UI |
| G15 | moduli con lo stesso nome (`data`, `encoders`, `model`) in `apertus-8B-lora-heads/` e `apertus-8B-vadapter-lora-heads/` | collisione in `sys.modules` nello stesso processo | un processo worker per famiglia (B.5) |
| G16 | velocità del full FT non misurata; quella di P10 esclude la codifica VQ | confronto di throughput incompleto | misurare in demo (§14); mostrare solo numeri misurati con condizioni |
| G17 | `phase1-256/settings.json` nel repo del LoRA senza adapter | un training di fase 1 del LoRA forse mai partito | domanda al team; non usato |
| G18 | revisione del modello base non fissata nel codice P10 | possibile mismatch base/adapter | usare 1519de0 e verificarlo nel test di parità |
