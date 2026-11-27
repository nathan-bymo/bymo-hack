# Allegato B — Registro modelli, adapter e `models.yaml`

Parte di [PIANO_DEMO.md](../PIANO_DEMO.md) (sezione 5). Le schede con i fatti verificati sono in
[allegato A, A.4](A_inventario.md#a4-schede-modello); qui c'è **come** la demo li usa.

Principio: **gli adapter non reimplementano il preprocessing**. Ogni passo che trasforma immagine e contesto in
input del modello chiama la funzione del repo citata. Quando la logica del repo è inline in un `main()` (zero-shot in
`eval_prompt.py`, P10 in `evaluate.py`), l'adapter ricompone **le stesse chiamate nello stesso ordine**, con un test
di parità che lo dimostra ([§14](../PIANO_DEMO.md#14-piano-di-test-e-verifica)).

---

## B.1 Interfaccia `ModelAdapter`

File: `src/demo/backend/demo_api/adapters/base.py`. Pseudo-codice (Python 3.11+, dataclass e `typing.Protocol`).

```python
ConfidenceKind = Literal["full_distribution", "chosen_only", "none"]
ConfidenceMethod = Literal["head_softmax", "letter_logprob", "sequence_prob", "constrained_scoring"]
AdapterState = Literal["unavailable", "not_loaded", "loading", "ready", "unloading", "error"]

@dataclass(frozen=True)
class StoredImage:
    image_id: str               # uuid della demo ("img-demo-…"), MAI un image_id della competizione
    sha256: str                 # hash dei byte salvati
    path: Path                  # data/demo/uploads/<aa>/<sha256>.<ext>
    context: dict[str, str]     # chiavi di prompt.CONTEXT_FIELDS (prompt.py:48-54); {} se assente

@dataclass
class DimensionPrediction:
    label: str | None                      # stringa esatta di prompt.TAXONOMY[d], None = output non valido
    confidence: float | None               # None → "n/a" in UI
    top_k: list[tuple[str, float]] | None  # max 3, ordinati; None se il metodo non dà alternative
    confidence_method: ConfidenceMethod | None

@dataclass
class ImagePrediction:
    image_id: str
    valid: bool                            # False se almeno una dimensione è None
    dimensions: dict[str, DimensionPrediction]   # chiavi = prompt.DIMENSIONS, sempre tutte e 6
    raw_output: str | None                 # testo generato (P10) o descrizione (zero-shot); None per le teste
    timing_ms: dict[str, float]            # {"preprocess", "model", "total"} per immagine

@dataclass
class ModelInfo:                           # dati statici da models.yaml + stato runtime
    model_id: str; display_name: str; family: str; runtime: str
    training: dict; weights: dict; prompt: dict; vram_estimate_gb: float | None
    confidence: ConfidenceKind; status: AdapterState; status_reason: str | None

class ModelAdapter(Protocol):
    model_id: str
    def info(self) -> ModelInfo: ...
    def state(self) -> tuple[AdapterState, str | None]: ...          # (stato, motivo leggibile)
    def supports_confidence(self) -> ConfidenceKind: ...
    def load(self, on_progress: Callable[[str], None]) -> None: ...   # bloccante; lanciato dal job manager
    def unload(self) -> None: ...
    def preprocess(self, images: list[StoredImage], use_context: bool) -> "Prepared": ...  # delega a src/
    def predict(self, prepared: "Prepared", batch_size: int) -> list[ImagePrediction]: ...
```

Regole comuni:

- `applicable` **non** lo calcola l'adapter: lo aggiunge l'API con `demo_api/relevance.py` a partire dalle label
  ([allegato A, A.2.1](A_inventario.md#a21-none-relevant-e--)).
- Un adapter non scarica mai pesi: risolve i percorsi con `huggingface_hub.snapshot_download(...,
  local_files_only=True)`. Se mancano, `state()` restituisce `("unavailable", "weights missing: <repo>/<subfolder>")`.
  **Non chiamare** `encoders.model_dir()` (src/apertus-8B-lora-heads/encoders.py:25-28) né `ft_encode.model_dir()`:
  senza `local_files_only` scaricherebbero 18,4 GB.
- `use_context=True` è il comportamento di training di tutti i modelli; senza contesto si usa
  `prompt.context_block({})` (produce "not available"), condizione **non misurata** per i modelli allenati.
- Ordine delle dimensioni sempre `prompt.DIMENSIONS`.

---

## B.2 `ZeroShotGenerativeAdapter` (`apertus-8b-zeroshot-v2`, `apertus-70b-zeroshot-v2`)

Dove gira: **nel processo dell'API** (solo HTTP verso Keera o un server OpenAI-compatibile; nessun torch).

| passo | chiamata al repo | fonte |
|---|---|---|
| import | `from common import images, prompt` con `<repo>/src` in `sys.path`; `eval_prompt` caricato con `importlib.util.spec_from_file_location("bymo_eval_prompt", "<repo>/src/scripts/eval_prompt.py")` | eval_prompt.py:38-41 |
| credenziali | l'API carica `.env` con la stessa logica di `load_env` ma **senza** `sys.exit` e scrive `LLM_BASE_URL`/`LLM_API_KEY` in `os.environ` (letti da `complete` a ogni chiamata) | eval_prompt.py:47-56, 75-77 |
| immagine | `url = images.data_url(path, 512)` | images.py:45-47 |
| riga | `row = {campo: valore}` con le chiavi di `CONTEXT_FIELDS` | prompt.py:48-54 |
| descrizione | `body = {"model": name, "temperature": 0, "messages": prompt.describe_messages(url), "max_tokens": prompt.DESCRIBE_MAX_TOKENS}` → `choice = complete(body)["choices"][0]`; `description = choice["message"]["content"]` | eval_prompt.py:118-120, 135-137 |
| 6 domande | per `d` in `prompt.DIMENSIONS`: `messages = prompt.question_messages(row, url, d, "codes", use_context, description)`; `body = {"model", "temperature": 0, "messages", **prompt.question_constraint(d, "codes")}` | eval_prompt.py:140-143; prompt.py:362-375 |
| parsing | `label, probs = prompt.parse_question(choice, d, "codes")` | eval_prompt.py:151; prompt.py:389-396 |
| nome modello | `apertus-v1-5-8b` / `apertus-v1-5-70b` | eval_prompt.py:98; src/AGENTS.md:62-64 |

- Le 7 richieste di un'immagine sono **in sequenza** (la seconda dipende dalla descrizione); più immagini in
  parallelo con un `ThreadPoolExecutor` (default 8, massimo 32 come in P4: prompt_eval_coverage.md:31-34).
- `confidence` = `probs[label]`; `top_k` = 3 classi con probabilità più alta; `confidence_method = "letter_logprob"`.
  Se `parse_question` restituisce probabilità `None` (logprob mancanti), confidenza `None`.
- `raw_output` = la descrizione generata (utile da mostrare: è il "ragionamento" visibile del modello).
- Errori: `RuntimeError` di `complete` (HTTP non ritentabile o 5 tentativi falliti, eval_prompt.py:85-92) →
  errore `UPSTREAM_ERROR` per quell'immagine, le altre continuano (come eval_prompt.py:163-165).
- Latenza attesa (misurata in P4, gateway condiviso, 32 in parallelo): mediana 8,0 s/immagine (8B), 11,1 s (70B)
  (prompt_eval_coverage.md:97, 99). La UI lo dice prima di lanciare.
- `state()`: `unavailable` se `LLM_API_KEY` manca; `ready` altrimenti (nessun `load`). Opzionale: ping con una
  richiesta `max_tokens: 1` su un'immagine sintetica al primo uso, risultato mostrato nel registro.

---

## B.3 `LoraGenerativeAdapter` (`apertus-8b-lora-p10`)

Dove gira: **nel worker** della famiglia `lora_generative` (GPU). Codice di riferimento:
`src/apertus-8B-benchmark/lora/apertus_lora.py` e `evaluate.py:94-193`.

| passo | chiamata al repo | fonte |
|---|---|---|
| import | `sys.path.insert(0, "<repo>/src/apertus-8B-benchmark/lora")`; `import apertus_lora as L` (trova `common` risalendo le cartelle) | apertus_lora.py:18-24 |
| base | `base_dir = snapshot_download("swiss-ai/Apertus-v1.5-8B", revision=<models.yaml>, local_files_only=True)` | — |
| adapter | `snapshot_download("StoryofLegend/apertus-8b-galaxus-lora", revision="be79276586ea5610d3a96a081988da8d977e7af3", allow_patterns=["prototype-256/adapters/step-00738/*"], local_files_only=True)` + sottocartella | HF, A.4.3 |
| tokenizer | `tokenizer = AutoTokenizer.from_pretrained(base_dir)`; `config = AutoConfig.from_pretrained(base_dir)`; `sequences = L.Sequences(tokenizer, config.image_token_offset)` | evaluate.py:114-116 |
| VQ | `vq, processor, _ = L.load_vision_tokenizer(base_dir, "cuda")` (fp32, TF32 off) | apertus_lora.py:162-192 |
| modello | `Apertus1p5ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=L.ATTENTION)`; `model.model.vision_tokenizer = None`; `model.model.audio_tokenizer = None`; `PeftModel.from_pretrained(model, adapter_dir).merge_and_unload().eval()` | evaluate.py:136-142 |
| immagine | `image = L.images.prepared_image(path, 256)`; `h, w, codes = L.encode_image(vq, processor, image)` | apertus_lora.py:195-206, 346 |
| prompt | `ids = sequences.prompt_ids(row, h, w, codes, use_context)` | apertus_lora.py:253-255 |
| batch | ordinare per lunghezza (`sorted(..., key=len)`), lotti di `batch_size` (32 nel repo); `L.collate([{"input_ids": p, "labels": p} for p in lotto], tokenizer.pad_token_id)` | evaluate.py:144-148; apertus_lora.py:313-327 |
| decoding | `model.generate(input_ids, attention_mask, max_new_tokens=96, do_sample=False, eos_token_id=sequences.end, pad_token_id=tokenizer.pad_token_id)`; testo = `tokenizer.decode(generated, skip_special_tokens=True)` | evaluate.py:150-159 |
| parsing | `L.parse_answer(text)` → dict o `None` | apertus_lora.py:283-290 |

- Output non valido (`None`): tutte e 6 le dimensioni con `label=None`, `valid=False`, `raw_output` = testo
  generato. Nessuna riparazione del JSON. In P10 è successo 21 volte su 2.532 (eval_coverage.json:5).
- Confidenza: metodo A di default (`return_dict_in_generate=True, output_scores=True`), vedi
  [§6 del piano](../PIANO_DEMO.md#6-confidenza). **Il test di parità verifica che le sequenze generate siano
  identiche con e senza `output_scores`.**
- Avviso permanente nella scheda modello: "metriche riportate dell'adapter `final` (perso); pesi caricati:
  `step-00738`, 3 passi prima della fine, mai valutato".

---

## B.4 `HeadsAdapter`

Due varianti, una per famiglia di worker. Entrambe: un solo forward per immagine, softmax per testa, label = argmax.

### B.4.1 Variante `lora_heads` (`apertus-8b-lora-heads-p11`)

| passo | chiamata al repo | fonte |
|---|---|---|
| import | `sys.path.insert(0, "<repo>/src/apertus-8B-lora-heads")`; `import data as D, encoders as E, model as M, predict as P` | predict.py:29-33 |
| caricamento | `classifier, settings = M.load(base_dir, ckpt_dir, device, merge=True)` | model.py:142-156 |
| encoder | `encoder = E.ImageEncoder(base_dir, device)` | encoders.py:31-56 |
| prompt | `tokenizer = AutoTokenizer.from_pretrained(base_dir)`; `builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(base_dir).image_token_offset)` | predict.py:107-109 |
| identità | `P.fingerprint(ckpt_dir, use_context)` → parte della chiave di cache | predict.py:38-44 |
| immagini | `encoded = encoder.encode([img.path for img in lotto], [""] * len(lotto), settings["side"])` | encoders.py:69-80; predict.py:115-116 |
| sequenze | `seqs = [builder.ids(row, e["grid"], e["codes"], use_context) for row, e in zip(righe, encoded)]` | encoders.py:125-128; predict.py:137-138 |
| forward | `input_ids, attention = E.collate(seqs, tokenizer.pad_token_id)`; `with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): logits = classifier(input_ids.to(device), attention.to(device))` | predict.py:130, 139-141 |
| output | `probs = [x.float().softmax(-1).cpu() for x in logits]`; label `D.TAXONOMY[d][int(probs[t][i].argmax())]` | predict.py:142-147 |

Immagine illeggibile: come predict.py:117-126, si ripete l'encode una per una e si segnala `CORRUPT_IMAGE` solo per
quella (in pratica non succede: l'upload ha già decodificato l'immagine).

### B.4.2 Variante `fullft_heads` (`apertus-8b-fullft-heads`)

Codice solo sul branch `feat/apertus-8b-fullft-heads` (`9145fca`), letto da un worktree (B.6).

| passo | chiamata al repo (branch) | fonte |
|---|---|---|
| import | `sys.path.insert(0, "<fullft_src>/apertus-8B-fullft-heads")`; `import ft_model as M` (aggiunge da solo `apertus-8B-vadapter-lora-heads` al path); poi `import data as D, encoders as E`; `from ft_encode import VQEncoder`; `from ft_train import autocast` | ft_predict.py:23-29; ft_model.py:23-29 |
| caricamento | `classifier, settings = M.load(base_dir, ckpt_dir, device)` (bf16 decoder 24 layer + teste fp32) | ft_model.py:153-156 |
| encoder | `encoder = VQEncoder(base_dir, device)` (solo VQ, niente SigLIP) | ft_encode.py:55-79 |
| prompt | `builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(base_dir).image_token_offset, prefix_tokens=0)` | ft_predict.py:63-64 |
| immagini | `encoded = encoder.encode(paths, [""] * n, settings["side"])` | ft_encode.py:81-90; ft_predict.py:70-71 |
| sequenze | `builder.ids(row, e["grid"], e["codes"], use_context)` | vadapter encoders.py:194-197 |
| forward | `input_ids, attention = E.collate(seqs, tokenizer.pad_token_id)`; `with torch.inference_mode(), autocast(device): logits = classifier(...)` | ft_predict.py:82-85 |
| output | softmax per testa; `D.prediction_frame(ids, probs)` per l'argmax | ft_predict.py:86-95; vadapter data.py:159-164 |

Controllo di caricamento: `ft_model.load_weights` rifiuta un checkpoint di forma diversa (ft_model.py:145-148);
l'adapter mostra quel messaggio così com'è.

---

## B.5 Worker GPU, `WorkerProxy` e `RecordedAdapter`

### B.5.1 Perché un processo worker per famiglia

1. **Collisione di nomi**: `data`, `encoders`, `model` esistono sia in `src/apertus-8B-lora-heads/` sia in
   `src/apertus-8B-vadapter-lora-heads/` (usato dal full FT). Nello stesso processo Python il secondo `import data`
   restituirebbe il primo modulo da `sys.modules`: preprocessing sbagliato **senza errori**.
2. **Liberare la VRAM**: terminare il processo è l'unico modo sicuro per restituire tutta la memoria CUDA.
3. **Remoto o locale con lo stesso codice**: il worker è un piccolo server HTTP; può girare come sottoprocesso
   dell'API (host con GPU) o su un altro host raggiunto con un tunnel SSH.

Ogni worker, all'avvio, verifica che i moduli importati vengano dalla cartella attesa
(`assert Path(data.__file__).parent == atteso`) e registra il commit del codice usato (`git -C <root> rev-parse HEAD`).

### B.5.2 Protocollo del worker (`src/demo/backend/demo_worker/server.py`)

Server FastAPI minimale, porta 8011 (un worker per famiglia: 8011, 8012, 8013), header obbligatorio
`X-Demo-Worker-Token: $DEMO_WORKER_TOKEN` quando non è su `127.0.0.1`.

| metodo e percorso | richiesta | risposta |
|---|---|---|
| `GET /worker/info` | — | `{"family", "gpu_name", "total_gb", "free_gb", "torch", "transformers_commit", "code_commit", "loaded": [model_id…]}` |
| `POST /worker/load` | `{"model_id", "base_dir", "checkpoint_dir", "settings"}` | stream NDJSON di stati `{"state": "loading"|"ready"|"error", "message", "peak_memory_gb"}` |
| `POST /worker/unload` | `{"model_id"}` | `{"state": "not_loaded"}` |
| `POST /worker/predict` | `{"model_id", "use_context", "batch_size", "images": [{"image_id", "sha256", "ext", "bytes_b64", "context"}]}` | `{"results": [ImagePrediction…], "batch": {"seconds", "images_per_second", "peak_memory_gb", "gpu_name", "batch_size"}}` |

Il worker scrive i byte ricevuti in una cartella temporanea con l'estensione originale e passa **il percorso** alle
funzioni del repo (che leggono da file): i byte sono quelli salvati dall'API, quindi lo stesso input.

### B.5.3 `WorkerProxy` (lato API)

Implementa `ModelAdapter` inoltrando a `DEMO_WORKER_URL_<FAMILY>` (default: sottoprocesso locale avviato con
`python -m demo_worker.server --family <family> --port <p>` se `torch.cuda.is_available()` è vero nell'ambiente GPU).
Timeout: caricamento 15 min, predizione 120 s per immagine (configurabili).

### B.5.4 `RecordedAdapter` (modalità "registrato")

Per presentare un modello allenato senza GPU durante la presentazione, con risultati **veri prodotti prima**.

- Produzione: `python src/demo/scripts/record.py --model <id> --session <session_id> --out data/demo/recorded/<id>/<run_id>/`
  eseguito **sull'host GPU**, che usa gli stessi adapter (B.3, B.4) in modalità batch. Prima:
  `python src/demo/scripts/export_session.py --session <id>` crea `data/demo/export/<session_id>/` con `images/` e
  `manifest.json` (sha256, estensione, contesto). Nessun ID della competizione nel manifest.
- Formato: `run.json` = `{model_id, model_revision, checkpoint, code_commit, gpu_name, batch_size,
  images_per_second, peak_memory_gb, started_at, finished_at, use_context}`; `predictions.jsonl` = una riga per
  immagine `{image_sha256, context_hash, use_context, dimensions{…DimensionPrediction}, valid, raw_output, timing_ms}`.
- Uso: l'API scansiona `data/demo/recorded/` all'avvio (o `POST /api/recorded/rescan`); una predizione registrata ha
  `mode: "recorded"` e la UI mostra il badge **"Recorded <date, time> on <GPU>"** ovunque compaia.
- Non usarlo mai per immagini diverse da quelle registrate (lookup per sha256 + hash del contesto: se manca, il
  modello risulta `unavailable` per quell'immagine, con il motivo).

---

## B.6 Come il worker importa il codice del repo senza modificarlo

| processo | `sys.path` aggiunto | moduli importati |
|---|---|---|
| API | `<repo>/src` | `common.prompt`, `common.images`, `common.splits`; `bymo_eval_prompt` da file |
| worker `lora_generative` | `<repo>/src/apertus-8B-benchmark/lora` | `apertus_lora` |
| worker `lora_heads` | `<repo>/src/apertus-8B-lora-heads` | `data`, `encoders`, `model`, `predict` |
| worker `fullft_heads` | `<fullft_src>/apertus-8B-fullft-heads` (+ automatico `apertus-8B-vadapter-lora-heads`) | `ft_model`, `data`, `encoders`, `ft_encode`, `ft_train` |

- `<fullft_src>` = variabile `DEMO_FULLFT_SRC`. Finché il branch del full FT non è in main:
  `git worktree add ../hack-apertus-fullft feat/apertus-8b-fullft-heads` e
  `DEMO_FULLFT_SRC=../hack-apertus-fullft/src`. Nel worktree **non si committa**. Quando il team unisce il branch in
  main, `DEMO_FULLFT_SRC` torna a `<repo>/src`.
- Il codice della demo non viene installato come pacchetto dentro `src/`; ha il proprio pacchetto
  `src/demo/backend/demo_api` (e `demo_worker`). **Nessuna cartella si chiama `data`** (sarebbe ignorata da
  `.gitignore:30`).
- Vietato: copiare funzioni del repo nella demo, modificare file sotto `src/` fuori da `src/demo/`.

---

## B.7 Bozza completa di `src/demo/models.yaml`

```yaml
# src/demo/models.yaml — registro dei modelli della demo (bozza dal piano del 27/11/2026, verifiche 20:08 UTC).
# Regole: ogni valore ha una fonte; i campi ignoti valgono "[DA VERIFICARE]" e rendono il modello "unavailable".
# Nessuna metrica è stimata. kaggle_dev.score si compila SOLO a mano con il punteggio letto su Kaggle.
schema_version: 1
verified_at: "2026-11-27T20:08:00Z"

defaults:
  base_model:
    repo: swiss-ai/Apertus-v1.5-8B
    revision: 1519de0ae7f3d114879291810a58259966f64c99      # src/apertus-8B-lora-heads/encoders.py:19-20
    size_gb: 18.4                                             # model.safetensors.index.json total_size 18396023436
  transformers_fork: "git+https://github.com/swiss-ai/transformers.git@3797303dda74844e3d1f8977ff5518bb91f818b4"
  reference_line:
    value: 0.917
    label: "Organisers' reference (Qwen 3.5 9B FT, dev)"
    source: "challenge_description.md:143"
  throughput_target_images_per_second: 2                      # challenge_description.md:67
  cost:
    default: {gpu_label: "A100 80GB (HF a100-large)", usd_per_hour: 2.50,
              source: "feat/apertus-8b-fullft-heads:src/apertus-8B-fullft-heads/README.md:71"}
    alternatives:
      - {gpu_label: "H200 (HF)", usd_per_hour: 5.00, source: "src/apertus-8B-lora-heads/results_full_h200.md:157"}
      - {gpu_label: "RTX PRO 6000 (HF)", usd_per_hour: 2.75, source: "feat/apertus-8b-fullft-heads:src/apertus-8B-fullft-heads/README.md:70"}
      - {gpu_label: "L40S (HF)", usd_per_hour: 1.80, source: "FAQ.md:22"}

models:

  - id: apertus-8b-zeroshot-v2
    display_name: "Apertus 8B · zero-shot · prompt v2"
    family: zeroshot_generative
    adapter: ZeroShotGenerativeAdapter
    runtime: remote_openai
    endpoint: {base_url_env: LLM_BASE_URL, api_key_env: LLM_API_KEY, model_name: apertus-v1-5-8b,
               default_base_url: "https://gateway.keera.ch/api/v1"}   # src/AGENTS.md:58-65
    prompt: {version: v2, spec: "8b:describe:codes:512", layout: describe, answer: codes, side: 512,
             use_context_default: true, system_message: false}        # src/AGENTS.md:16-24
    confidence: {kind: full_distribution, method: letter_logprob}
    training: {data: none, phase: null, seen_sets: []}
    vram_estimate_gb: null                                            # remoto
    measured:
      - {what: "img/s", value: 3.8, conditions: "gateway Keera condiviso, 32 in parallelo, coverage",
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:97"}
      - {what: "latenza mediana s/immagine", value: 8.0, source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:97"}
    evals:
      - {set: coverage, validity: phase1_internal, provenance: reported_only, macro_f1: 0.528, weighted_f1: 0.705,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:45",
         note: "prompt scelto sulle prime 600 di prompt-dev, che sono dentro coverage"}
      - {set: base, validity: phase1_internal, provenance: reported_only, macro_f1: 0.513, weighted_f1: 0.708,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:57"}
      - {set: prompt-test, validity: held_out, provenance: reported_only, macro_f1: 0.502, weighted_f1: 0.703,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:69", note: "non usato per scegliere il prompt"}
    per_class_source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:107-400"
    kaggle_dev: {score: null, submitted_at: null, note: ""}

  - id: apertus-70b-zeroshot-v2
    display_name: "Apertus 70B · zero-shot · prompt v2"
    family: zeroshot_generative
    adapter: ZeroShotGenerativeAdapter
    runtime: remote_openai
    endpoint: {base_url_env: LLM_BASE_URL, api_key_env: LLM_API_KEY, model_name: apertus-v1-5-70b,
               default_base_url: "https://gateway.keera.ch/api/v1"}
    prompt: {version: v2, spec: "70b:describe:codes:512", layout: describe, answer: codes, side: 512,
             use_context_default: true, system_message: false}
    confidence: {kind: full_distribution, method: letter_logprob}
    training: {data: none, phase: null, seen_sets: []}
    vram_estimate_gb: null
    measured:
      - {what: "img/s", value: 2.8, conditions: "gateway Keera condiviso, 32 in parallelo, coverage",
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:99"}
      - {what: "latenza mediana s/immagine", value: 11.1, source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:99"}
    evals:
      - {set: coverage, validity: phase1_internal, provenance: reported_only, macro_f1: 0.583, weighted_f1: 0.739,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:47", note: "prompt scelto su prompt-dev ⊂ coverage"}
      - {set: base, validity: phase1_internal, provenance: reported_only, macro_f1: 0.587, weighted_f1: 0.742,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:59"}
      - {set: prompt-test, validity: held_out, provenance: reported_only, macro_f1: 0.590, weighted_f1: 0.740,
         source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:71"}
    per_class_source: "src/apertus-8B-benchmark/prompt_eval_coverage.md:107-400"
    kaggle_dev: {score: null, submitted_at: null, note: ""}

  - id: apertus-8b-lora-p10
    display_name: "Apertus 8B · LoRA (P10, 1.500 immagini) · JSON"
    family: lora_generative
    adapter: LoraGenerativeAdapter
    runtime: worker_gpu
    weights:
      repo: StoryofLegend/apertus-8b-galaxus-lora          # privato
      revision: be79276586ea5610d3a96a081988da8d977e7af3
      subfolder: prototype-256/adapters/step-00738         # "final" perso: solo adapter_config.json
      size_gb: 0.319
      requires_base: true
      base_revision_note: "[DA VERIFICARE] revisione base non fissata nel codice P10; si usa quella di defaults"
    code: {root: "src/apertus-8B-benchmark/lora", reference: "evaluate.py:94-193"}
    prompt: {version: "lora-json-short", system_prompt_source: "src/apertus-8B-benchmark/lora/apertus_lora.py:31-35",
             side: 256, use_context_default: true, decoding: {max_new_tokens: 96, do_sample: false}}
    dtype: bfloat16
    confidence: {kind: chosen_only, method: sequence_prob, stretch_method: constrained_scoring}
    training: {data: "prototype train_1500 (dentro training_rows)", phase: 1, seen_sets: [train_1500]}
    vram_estimate_gb: 24          # [STIMA] 18,4 GB pesi + VQ 0,87 GB + cache KV; da misurare
    measured:
      - {what: "img/s (solo generate, VQ escluso)", value: 18.0, conditions: "H200, batch 32, coverage",
         source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json:6"}
    evals:   # tutti riferiti all'adapter "final", NON a step-00738
      - {set: coverage, validity: phase1_internal, provenance: reported_only, macro_f1: 0.734, weighted_f1: 0.879,
         source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json", checkpoint: final,
         note: "21 risposte non valide su 2.532"}
      - {set: base, validity: phase1_internal, provenance: reported_only, macro_f1: 0.721, weighted_f1: 0.885,
         source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json", checkpoint: final}
      - {set: prompt-test, validity: phase1_internal, provenance: reported_only, macro_f1: 0.711, weighted_f1: 0.881,
         source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json", checkpoint: final,
         note: "valore esatto 0.7115; docs/context/EXPERIMENTS.md sul branch scrive 0.712"}
      - {set: validation_450, validity: held_out, provenance: reported_only, macro_f1: 0.696, weighted_f1: 0.747,
         source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_prototype_validation.json", checkpoint: final,
         note: "set bilanciato, non rappresentativo"}
    per_class_source: "src/apertus-8B-benchmark/lora/results/prototype-256/eval_coverage.json"
    kaggle_dev: {score: null, submitted_at: null, note: ""}

  - id: apertus-8b-lora-heads-p11
    display_name: "Apertus 8B · LoRA + 6 teste (P11, 9.344 immagini)"
    family: lora_heads
    adapter: HeadsAdapter
    runtime: worker_gpu
    weights:
      repo: "[DA VERIFICARE] <utente>/galaxus-lora-heads-private"     # results_full_h200.md:169
      revision: "[DA VERIFICARE]"
      subfolder: "full-train/model/final"                            # [DA VERIFICARE] da build_hf_job.py:77,182,207-209
      local_path: null                                               # alternativa: cartella final/ copiata a mano
      files: ["lora/adapter_model.safetensors", "lora/adapter_config.json", "heads.pt", "settings.json"]  # model.py:129-139
      requires_base: true
    code: {root: "src/apertus-8B-lora-heads", reference: "predict.py:65-179"}
    prompt: {version: "heads-task", task_source: "src/apertus-8B-lora-heads/encoders.py:21-22", side: 256,
             use_context_default: true, system_message: false}
    dtype: bfloat16
    confidence: {kind: full_distribution, method: head_softmax, training_loss: cross_entropy}
    training: {data: "training_rows (9.344)", phase: 1, seen_sets: [training_rows]}
    vram_estimate_gb: 22.0        # MISURATO: picco allocato, batch 32, H200 (results_full_h200.md:156)
    measured:
      - {what: "img/s (immagine → CSV)", value: 13.2, conditions: "H200, batch 32, coverage, caricamento escluso",
         source: "src/apertus-8B-lora-heads/results_full_h200.md:156"}
    evals:
      - {set: coverage, validity: phase1_internal, provenance: reported_only, macro_f1: 0.781, weighted_f1: 0.905,
         source: "src/apertus-8B-lora-heads/results_full_h200.md:23"}
      - {set: base, validity: phase1_internal, provenance: reported_only, macro_f1: 0.786, weighted_f1: 0.913,
         source: "src/apertus-8B-lora-heads/results_full_h200.md:24"}
      - {set: prompt-test, validity: phase1_internal, provenance: reported_only, macro_f1: 0.791, weighted_f1: 0.915,
         source: "src/apertus-8B-lora-heads/results_full_h200.md:25"}
    per_class_source: "src/apertus-8B-lora-heads/results_full_h200.md:46-120"
    confusion_source: {dimension: perspective, set: coverage, source: "src/apertus-8B-lora-heads/results_full_h200.md:122-136"}
    caveats: ["comportamento senza contesto prodotto non misurato (results_full_h200.md:17)"]
    kaggle_dev: {score: null, submitted_at: null, note: ""}

  - id: apertus-8b-fullft-heads
    display_name: "Apertus 8B · full fine-tune 24 layer + 6 teste (fase 2, tutto il train)"
    family: fullft_heads
    adapter: HeadsAdapter
    runtime: worker_gpu
    weights:
      repo: StoryofLegend/apertus-8b-galaxus-fullft         # privato
      revision: 711018b7a25dc130d04074331f1aa6946f727e23     # 20:52 UTC (riverificato 27/11); cambierà con last
      subfolder: phase2/model/epoch-1                       # Q4: `last` se esiste, altrimenti epoch-1 "intermedio"; epoch-2 esiste dalle 20:52 UTC (stessa dimensione, file di epoch-1 invariati)
      files: ["decoder.safetensors", "heads.pt", "settings.json"]
      size_gb: 10.486                                       # 10,469 + 0,017
      requires_base: true                                   # embedding congelati dal modello base (ft_model.py:117-122)
      job: {id: 6b09cdb46c5d61fff189842a, flavor: a100-large, status_at_verification: RUNNING, env: {PHASE: "2", EPOCHS: "3"}}
    code: {root_env: DEMO_FULLFT_SRC, branch: feat/apertus-8b-fullft-heads, commit: 9145fca,
           reference: "src/apertus-8B-fullft-heads/ft_predict.py:34-109"}
    prompt: {version: "heads-task", task_source: "src/apertus-8B-vadapter-lora-heads/encoders.py:30-31 (branch)",
             side: 256, use_context_default: true, prefix_tokens: 0}
    dtype: bfloat16
    confidence: {kind: full_distribution, method: head_softmax, training_loss: "focal gamma=2 + alpha effective"}
    training: {data: "tutto train.csv (11.876)", phase: 2, seen_sets: [train_all, coverage, base, prompt-test, validation_450, train_1500]}
    vram_estimate_gb: 24          # [STIMA] picco al caricamento; da misurare
    measured: []
    evals: []                     # nessun set locale valido: tutto il train è stato visto
    contamination_note: "Allenato su tutte le 11.876 immagini: ogni metrica su sottoinsiemi del train è 'seen during training — invalid'."
    kaggle_dev: {score: null, submitted_at: null, note: "unico numero valido per questo modello"}

  - id: apertus-8b-frozen-heads-p8
    display_name: "Apertus 8B congelato + teste (P8, cross-entropy, 1 seed)"
    family: frozen_heads
    adapter: null
    runtime: none
    status_override: {status: unavailable, reason: "manca un predittore end-to-end e i pesi non sono su questa macchina"}
    training: {data: "train_1500: 1.199 fit + 301 scelta", phase: 1, seen_sets: [train_1500]}
    evals:
      - {set: validation_450, validity: held_out, provenance: reported_only, macro_f1: 0.542, weighted_f1: 0.612,
         source: "src/confronto_lora.md:35", note: "set bilanciato, non confrontabile con coverage"}
    kaggle_dev: {score: null}

  - id: apertus-8b-frozen-heads-p9
    display_name: "Apertus 8B congelato + teste (P9, cross-entropy pesi forti, media 5 seed)"
    family: frozen_heads
    adapter: null
    runtime: none
    status_override: {status: unavailable, reason: "manca un predittore end-to-end e i pesi non sono su questa macchina"}
    training: {data: "train_1500: 1.199 fit + 301 scelta", phase: 1, seen_sets: [train_1500]}
    evals:
      - {set: validation_450, validity: held_out, provenance: reported_only, macro_f1: 0.557, weighted_f1: 0.629,
         source: "src/confronto_lora.md:36", note: "media di 5 seed; set bilanciato"}
    kaggle_dev: {score: null}
```

Validazione all'avvio (`demo_api/registry.py`): schema Pydantic; un valore che contiene `[DA VERIFICARE]` in
`weights.*` rende il modello `unavailable` con motivo "configurazione incompleta: <campo>"; `kaggle_dev.score` deve
essere `null` o un numero in [0, 1]; un `evals[*].source` deve puntare a un file esistente del repo.

---

## B.8 Gestione della memoria

| regola | dettaglio |
|---|---|
| caricamento pigro | nessun modello si carica all'avvio; `POST /api/models/{model_id}/load` o il primo `POST /api/predict` che lo richiede (con conferma in UI se il caricamento supera 30 s) |
| un modello residente di default | `DEMO_MAX_RESIDENT=1`; con più VRAM si alza |
| budget VRAM | `DEMO_VRAM_BUDGET_GB` = memoria totale della GPU del worker meno 2 GB (letta da `/worker/info`); ogni modello dichiara `vram_estimate_gb` (misurato o [STIMA]); dopo il primo caricamento si salva il picco misurato e si usa quello |
| LRU | se il nuovo modello non entra nel budget, si scarica il modello usato meno di recente **che non ha job in corso**; se tutti sono occupati → `409 MODEL_BUSY` |
| condivisione della base | **non prevista**: P10 fonde il LoRA nel CausalLM, P11 lo fonde nel decoder (`merge=True`), il full FT sostituisce 24 layer e taglia gli altri. Si condividono solo i file su disco (cache HF) |
| cache dei codici VQ | opzionale: P10, P11 e full FT usano `side` 256 con la stessa preparazione e lo stesso calcolo dei codici (apertus_lora.py:195-206; encoders.py:58-67; vadapter encoders.py:72-81). Una cache `data/demo/cache/vq/256/<sha256>.pt` condivisa tra modelli è ammessa **solo dopo** che il test di parità ha mostrato codici identici tra le tre implementazioni |
| stati | `unavailable` → (`not_loaded` ⇄ `loading` → `ready` ⇄ `unloading`) e `error`; pubblicati in SSE |
| OOM in caricamento | il worker cattura `torch.cuda.OutOfMemoryError`, risponde `GPU_OOM` e termina; l'API riporta lo stato `error` con il picco osservato |
| OOM in predizione | un solo nuovo tentativo dimezzando il batch (fino a 1); se fallisce ancora → `GPU_OOM` per quel job |
| unload | `POST /api/models/{model_id}/unload` → il worker libera; se la famiglia non ha altri modelli residenti, il processo termina |

---

## B.9 Strategia hardware (senza finzioni)

| modo | dove | cosa gira | quando usarlo |
|---|---|---|---|
| **1. Tutto su un host GPU** | API + worker + frontend sulla stessa macchina con GPU ≥ 24 GB [STIMA] (consigliati 40–48 GB per tenere due modelli) | tutti i modelli locali + zero-shot remoto | se il team ha una workstation o una VM con GPU per la presentazione |
| **2. Laptop + worker remoto** | API e frontend sul laptop; worker su host GPU raggiunto con `ssh -L 8012:127.0.0.1:8012 <host>`; `DEMO_WORKER_URL_LORA_HEADS=http://127.0.0.1:8012` | come sopra | se esiste un host GPU ma si presenta dal laptop |
| **3. Laptop senza GPU** (questa macchina) | API e frontend sul laptop | zero-shot 8B/70B **live** via Keera; modelli allenati in modalità **registrato** (B.5.4) | default consigliato per la presentazione, con modalità 1 o 2 se disponibili |

- 70B: solo remoto (Keera o un server proprio OpenAI-compatibile indicato in `LLM_BASE_URL`). Supporto vLLM per
  `Apertus1p5ForConditionalGeneration` **non verificato** (docs/context/PROJECT_CONTEXT.md:201 sul branch).
- Quantizzazione 4-bit: **non prevista** nell'MVP (bitsandbytes non installato; nessun modello allenato è stato
  provato quantizzato). Se il team la vuole, la variante va registrata con un ID diverso e il badge
  "quantized — metriche calcolate in bf16".
- Pesi privati: letti con il token HF dell'utente (`~/.cache/huggingface/token`), mai scritto in log, config o UI.
- HF Jobs come host del worker: lo stato del job espone un campo `expose_urls` (visto in `inspect_job`), ma l'uso
  di un job come server **non è verificato** e costa: fuori piano salvo richiesta.

## B.10 Download dei pesi

- Mai automatici. `POST /api/models/{model_id}/download` richiede `{"confirm": true}` e prima risponde con la
  dimensione (`409 CONFIRMATION_REQUIRED` + `{"download_gb": …}`); equivalente CLI:
  `python src/demo/scripts/fetch_weights.py --model <id> --yes`.
- Dimensioni: base 8B 18,4 GB (condivisa); P10 +0,319 GB; full FT +10,49 GB; P11 [DA VERIFICARE].
- `snapshot_download(repo, revision=<sha>, allow_patterns=[f"{subfolder}/*"])`: si scarica solo il checkpoint scelto,
  alla revisione fissata.
- Sul laptop (7 GB di RAM, nessuna GPU) il download non serve: la UI disabilita il pulsante con il motivo.
