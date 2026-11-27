# Vision adapter + Apertus 8B (LoRA) + teste di classificazione

**Stato (27 novembre, sera): modello finale su prompt-test, 1.193 immagini mai viste, macro F1 0,821, weighted 0,917,
perspective 0,755.** Pipeline di training, risultati e cosa abbiamo imparato: [RISULTATI.md](RISULTATI.md). Oltre a
quanto descritto sotto, il modello finale usa il testo OCR nel prompt (docTR), feature spaziali e un attention pooling
con posizioni 2D; `predict.py` gira anche su 2 × T4 in fp16 (`build_kaggle_notebook.py`).

Unisce le tre prove che finora hanno dato segnali positivi, in un solo modello allenato end to end:

- `apertus-vadapter-benchmark/`: SigLIP 2 congelato → projector, 49 token visivi continui accanto ai token VQ;
- `apertus-8B-benchmark/lora/`: LoRA su Apertus con i codici VQ precalcolati, ripetizione delle classi rare, mirror;
- `apertus-8B-heads/`: sei teste sugli stati interni di Apertus, focal loss sulle sole celle valutate, layer 24.

```
immagine ─┬─► SigLIP 2 (congelato) ─► pooling 7×7 ─► projector ─► 49 soft token ─┐
          └─► tokenizer VQ di Apertus (congelato), circa 256² ─► token immagine ──┤
contesto prodotto + richiesta ────────────────────────────────────────────────────┤
                                                                                  ▼
                               Apertus 8B, layer 1–24 (LoRA r32), gli altri 8 eliminati
                                                                                  │
                  [ultimo token, media token VQ, media soft token] ─► 6 teste ─► 6 classi
```

**Perché le teste invece del JSON.** Un solo forward per immagine, nessun decoding, nessun JSON da validare (mai
celle vuote o classi inventate), probabilità per classe già pronte per calibrazione ed ensemble. Il troncamento al
layer 24 toglie un quarto del calcolo. Il generativo produce circa 60 token di JSON uno alla volta.

## File

| file | contenuto |
|---|---|
| `DATI.md` | quali immagini vanno nel training, in validazione e in valutazione, con i conteggi per classe |
| `data.py` | parti di `train.csv`, validazione interna per prodotto, label (−1 sulle celle non valutate), mirror, ripetizioni, alpha per classe |
| `encoders.py` | immagine → codici VQ e feature SigLIP; sequenza di token con gli slot dei soft token. Uguale in training e inferenza |
| `model.py` | projector (stesso formato del vadapter), Apertus troncato + LoRA, teste, focal loss, salvataggio e caricamento |
| `encode.py` | cache dei due encoder congelati sulle immagini di training e sui mirror (una sola volta, non a ogni epoca) |
| `train.py` | training end to end, scelta del checkpoint sulla validazione interna con `metric.py` |
| `predict.py` | inferenza end to end da immagini a CSV, con probabilità e immagini/s (caricamento immagini incluso) |
| `build_hf_job.py` | genera il job autonomo per Hugging Face Jobs (H200) |
| `test_pipeline.py` | test su CPU dell'impianto |

## Dati (src/AGENTS.md)

| fase | training | scelta del checkpoint | valutazione |
|---|---|---|---|
| 1 | `training_rows` (9.344) meno la validazione interna: **8.403** immagini, 13.390 item per epoca con oversampling augmentato e mirror | validazione interna: 10% dei prodotti per hash, **941** immagini | `coverage` (2.532), predetto solo alla fine |
| 2 | tutte le 11.876 immagini etichettate | nessuna: stesse impostazioni ed epoche della fase 1, si tiene l'ultimo checkpoint | solo submission sul dev |

Conteggi verificati sul `train.csv` locale il 27 novembre: nessun prodotto in comune tra training e validazione
interna, né tra training e `coverage`. La loss vede solo le label umane in celle rilevanti.
**Dettaglio completo (classi, ripetizioni, mirror, come ricostruire le liste): [DATI.md](DATI.md).**

## Esecuzione su Hugging Face (H200) con 20 $

H200 costa **5 $/h** (0,083 $/min, `hf jobs hardware`, 27 novembre 2026). Il `--timeout` del job è il tetto
massimo di spesa: alla scadenza HF ferma il job. `--max-minutes` adatta il training a un budget di minuti: dopo 30
passi misura la durata di un passo e accorcia lo schedule, così il learning rate scende comunque a zero e la run
finisce con il checkpoint migliore già salvato.

```sh
python src/apertus-8B-vadapter-lora-heads/build_hf_job.py
export HF_TOKEN=$(cat ~/.cache/huggingface/token) KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
# 0. controllo su CPU (~0,01 $): segreti, dati, modello gated, tokenizer e template reali
hf jobs uv run --flavor cpu-upgrade --timeout 20m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e CHECK_ONLY=1 data/hf_jobs/hf_job_vlh.py
# 1. fase 1: 2 epoche, training al massimo 45 min; tetto di spesa 1 h 30 = 7,50 $
hf jobs uv run --flavor h200 --timeout 1h30m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e UPLOAD_REPO=<utente>/galaxus-vlh-private -e PREDICT_DEV=1 \
    -e TRAIN_ARGS="--epochs 2 --max-minutes 45" data/hf_jobs/hf_job_vlh.py
# 2. fase 2: EPOCHS = epoca del checkpoint migliore della fase 1 (anche frazionaria, da train_summary.json);
#    gli item per epoca sono il 24% in più, quindi il budget di minuti cresce di conseguenza; tetto 1 h 45 = 8,75 $
hf jobs uv run --flavor h200 --timeout 1h45m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e PHASE=2 -e EPOCHS=<epoca> -e TRAIN_ARGS="--max-minutes 60" \
    -e UPLOAD_REPO=<utente>/galaxus-vlh-private data/hf_jobs/hf_job_vlh.py
```

Il job della fase 1: cache degli encoder → smoke test (3 passi) → training → predizione end to end di `coverage`
→ `eval_prompt.py --classes` → dev se `PREDICT_DEV=1`. Il checkpoint migliore va nel repository privato appena
salvato, così un timeout non perde nulla. Altre variabili: `TRAIN_ARGS` (argomenti di `train.py`), `SIDE` (lato
VQ, 256), `NO_CONTEXT_CHECK=1` (anche `coverage` senza contesto, ~3 min in più). Projector e LoRA partono **da
zero**; `INIT_PROJECTOR` resta solo come opzione.

### Quanto dura e quanto costa (stima)

Nessuna misura diretta di questo modello. Riferimento misurato: le teste su Apertus congelato facevano **8,0
immagini/s** su H200 (forward a 32 layer, VQ 512² ≈ 1.150 token, batch 16, lettura immagini inclusa, xIELU in
Python). Qui una sequenza ha circa 460 token (256 VQ + 49 soft + testo) e 24 layer, ma il training costa circa 3
forward (forward + backward): circa **0,9 volte** il costo per immagine di quella prova, quindi ~8–12 item/s, cioè
19–28 min per epoca in fase 1 (13.390 item) e 23–35 min in fase 2 (16.560 item). Con il budget di 45 min la fase 1
fa quindi 1,6–2 epoche: lo schedule si adatta da solo.

| passo | fase 1 | fase 2 |
|---|---|---|
| installazione, download dati e modello | 5–10 min | 5–10 min |
| cache VQ + SigLIP (14.331 / 16.560 tra immagini e varianti, preparazione in 16 thread) | 7–13 min | 8–15 min |
| smoke test (3 passi, con caricamento del modello) | 2–3 min | 2–3 min |
| training (2 epoche, validazione inclusa) | ≤ 45 min | ≤ 60 min |
| predizione `coverage` (2.532) | 2–4 min | – |
| predizione dev (5.261) | 4–7 min | 4–7 min |
| **totale** | **~70–85 min, circa 6–7 $** | **~80–95 min, circa 6,50–8 $** |

Con le due fasi restano circa 5–6 $ di margine, per un errore o per **una** ablation (per esempio `--gamma 0`).
Il margine sparisce se si rilanciano job interi: per questo prima c'è il controllo su CPU, e lo smoke test ferma il
job dopo ~15 min se qualcosa non va. Il log stampa item/s ogni 10 passi: se la velocità è molto sotto la stima,
conviene fermare il job (`hf jobs cancel <id>`) e rivedere il piano.

Se il credito non basta per entrambe le fasi, il modello della fase 1 è già un candidato valido per la submission
(src/AGENTS.md): allenato sul 71% delle immagini, con un punteggio misurato su `coverage`.

Per lanciare a mano su una GPU: `encode.py`, `train.py`, `predict.py` hanno i comandi nella docstring.

## Impostazioni predefinite

| impostazione | valore | da dove viene |
|---|---|---|
| layer tenuti | 24 su 32 | migliore sulla validazione interna con le teste su Apertus congelato |
| immagine VQ | circa 256², proporzioni mantenute | LoRA; l'adapter compensa la risoluzione |
| SigLIP 2 | `siglip2-base-patch16-224`, immagine originale, 49 token | vadapter |
| posizione soft token | subito prima del blocco immagine VQ, nel turno utente | **diversa dal vadapter** (che li metteva prima di tutto il prompt) |
| prompt | template di chat senza system, contesto + richiesta delle teste | teste |
| LoRA | nuova, da zero, r32, α32, dropout 0.05, attention e MLP dei 24 layer, LR 2e-4 | LoRA (solo le impostazioni) |
| projector | nuovo, da zero, LR 1e-4 | vadapter (solo l'architettura) |
| teste | LayerNorm → 512 → GELU → dropout 0.3 → 6 lineari, LR 1e-3 | teste (LayerNorm al posto della standardizzazione: le feature cambiano con la LoRA) |
| loss | focal γ = 2 per testa, solo celle valutate; `--class-alpha effective` per i pesi per classe | teste; motivazione sotto |
| augmentation e oversampling | classi rare fino a 4× per epoca (obiettivo 300 item), ogni ripetizione l'immagine VQ a un'altra scala (224/288/240), specchiata se senza testo; mirror left/right senza testo | LoRA, esteso: dettaglio in [DATI.md](DATI.md) |
| batch | 16, AdamW, warmup 5% e coseno, clipping 1.0, bf16, 2 epoche | |
| validazione | 2 volte per epoca, si tiene il checkpoint migliore | |

**Focal loss e sbilanciamento.** Lo sbilanciamento è forte (in fase 1, ad esempio, 4.510 `full_product` contro 55
`product_grouping`, 28 `certification_seal`, 1 `Dutch`) e la macro F1 pesa ogni classe allo stesso modo. Tre leve,
tutte attive:

- **oversampling augmentato** fino a 4× per epoca: ogni ripetizione è una variante diversa (altra scala dell'immagine
  VQ, mirror se senza testo; niente ritagli, vedi DATI.md), non una copia identica, quindi il modello non può memorizzare i token esatti;
- **focal loss** γ = 2: riduce il peso degli esempi già facili, quasi tutti delle classi frequenti;
- **alpha per classe** (`--class-alpha effective`, predefinito): numero effettivo di campioni (Cui et al. 2019,
  β = 0,999) calcolato sugli item dell'epoca, cioè **dopo** le ripetizioni, così lo sbilanciamento non viene
  compensato due volte. Peso medio per esempio pari a 1 e rapporto massimo 10 tra i pesi di una dimensione: senza
  questo limite l'unico esempio di `Dutch` schiaccerebbe a ~0,01 il peso delle altre lingue.

Pesi risultanti in fase 1, ad esempio: `full_product` 0,57, `product_grouping` 2,88; `white` 0,90,
`scene_lifestyle` 1,82; `certification_seal` 7,11, `none` (testo) 0,75; `Italian` e `Dutch` 7,11, `English` 0,76.
Il training li stampa all'inizio.

Unica misura finora sui nostri dati: teste su Apertus **congelato**, 1.199 immagini, un seed; lì la focal senza
alpha ha fatto −0,013 di macro F1 rispetto alla cross-entropy. Le condizioni qui sono diverse; se resta credito,
la prima ablation è `--gamma 0 --class-alpha none`. Va controllata anche la weighted F1, che gli organizzatori
guardano: troppa compensazione la abbassa.

Non incluso: GradNorm (non ha aiutato con le teste su Apertus congelato, `apertus-8B-heads/results_small_h200.md`).

## Verifiche fatte (27 novembre 2026)

- `test_pipeline.py`, 12 test su CPU: focal loss (γ = 0 coincide con la cross-entropy, formula con alpha), celle non
  valutate senza effetto sulla loss, soft token negli slot giusti, risultato identico con e senza padding, gradienti
  su projector e teste ma non sugli embedding congelati, medie separate per token VQ e soft token, label dei mirror,
  split per prodotto, alpha con peso medio 1 per esempio e rapporto massimo 10, variante senza adapter, varianti di augmentation (originale identica all'inferenza, altra scala VQ senza tagli, varianti vecchie scartate dalla cache, mirror solo senza testo, label left/right scambiate).
- Uno dei test usa la **vera classe del decoder Apertus del fork** in miniatura con pesi casuali, troncata da 4 a 2
  layer: padding e troncamento funzionano. Non prova nulla sui pesi reali né sulla qualità.
- Split sui dati reali (sopra).
- **Non ancora eseguito su GPU.** Tokenizer, template, caricamento del modello reale, PEFT e memoria si verificano
  con `CHECK_ONLY=1` e con lo smoke test all'inizio del job.

```sh
python -m pytest -q src/apertus-8B-vadapter-lora-heads/test_pipeline.py
```

## Ablation previste (per il report)

Tutte con `TRAIN_ARGS`, confrontate sulla validazione interna e poi su `coverage`:

- senza adapter: `--no-adapter` (solo token VQ; la media dei soft token resta a zero);
- senza LoRA: `--lora-rank 0` (Apertus congelato + adapter + teste);
- loss: `--gamma 0 --class-alpha none` (cross-entropy), `--class-alpha none` (focal senza pesi per classe);
- layer: `--layers 20 / 28 / 32`; risoluzione: `SIDE=512`;
- contesto: `--context-dropout 0.2`; il job misura già sempre `coverage` senza contesto.
