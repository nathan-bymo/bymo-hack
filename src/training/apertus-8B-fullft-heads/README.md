# Apertus 8B in fine-tuning completo + teste di classificazione

Apertus v1.5 8B troncato al layer 24, con **tutti i pesi dei layer allenati** (niente LoRA) e sei teste di
classificazione. Niente SigLIP: il modello legge solo i token immagine VQ di Apertus. Dati, split, oversampling,
augmentation e loss sono quelli di `apertus-8B-vadapter-lora-heads/` ([DATI.md](../apertus-8B-vadapter-lora-heads/DATI.md)),
**con lo stesso codice** (`data.py`, `encoders.py`, teste e focal loss di `model.py`, importati da quella cartella).

```
immagine ─► tokenizer VQ di Apertus (congelato), circa 256² ─► token immagine ─┐
contesto prodotto + richiesta ─────────────────────────────────────────────────┤
                                                                               ▼
              Apertus 8B, layer 1–24, TUTTI i pesi allenati (fp32), gli altri 8 eliminati
                                                                               │
                         [ultimo token, media token VQ] ─► 6 teste ─► 6 classi
```

Confronti già possibili sugli stessi dati: teste su Apertus **congelato** (`micro-training/apertus-8B-heads/`), LoRA generativa
(`micro-training/lora-generativo/`) e vision adapter + LoRA + teste (`apertus-8B-vadapter-lora-heads/`).

## File

| file | contenuto |
|---|---|
| `ft_model.py` | classificatore: decoder troncato, layer e norm finale allenabili in fp32, embedding congelati, teste; salvataggio dei soli layer allenati (safetensors) |
| `ft_encode.py` | cache dei codici VQ di tutte le varianti del training (originali, mirror, ritagli delle classi rare); più processi sulla stessa GPU |
| `ft_train.py` | training, scelta del checkpoint sulla validazione interna con `metric.py`; `--tiny` per una prova su CPU |
| `ft_predict.py` | inferenza end to end da immagini a CSV, con probabilità e immagini/s |
| `build_hf_job.py` | genera il job autonomo per Hugging Face Jobs (`data/hf_jobs/hf_job_fullft.py`, non committato) |
| `test_fullft.py` | 10 test su CPU con la vera classe del decoder Apertus in miniatura |

## Dati

Identici a [DATI.md](../apertus-8B-vadapter-lora-heads/DATI.md), con lo stesso codice:

| fase | training | scelta del checkpoint | valutazione |
|---|---|---|---|
| 1 | `training_rows` meno la validazione interna: **8.403** immagini, **13.390** item per epoca | **941** immagini (10% dei prodotti, hash `vlh-val:`) | `coverage` (2.532), predetto una volta alla fine |
| 2 | tutte le **11.876** immagini, 16.560 item per epoca | nessuna: stesse impostazioni, epoche della fase 1, ultimo checkpoint | solo submission sul dev |

I conteggi 8.403 / 941 / 13.390 sono verificati da `test_fullft.py` sul `train.csv` locale. La loss vede solo le
celle valutate (label umana in una dimensione rilevante).

## Impostazioni

| impostazione | valore | perché |
|---|---|---|
| layer tenuti | 24 su 32 | migliore con le teste su Apertus congelato (P8); un quarto di calcolo in meno |
| cosa si allena | tutti i pesi dei 24 layer (attention, MLP, xIELU, norm) e la norm finale: **5,2 B parametri** | richiesta: fine-tuning completo |
| congelato | embedding (1,1 B, token immagine inclusi), tokenizer VQ | circa 9 GB in meno tra pesi fp32, gradienti e stati |
| immagine | VQ circa 256², proporzioni mantenute, nessun SigLIP | come LoRA P10 e vlh |
| feature delle teste | ultimo token + media dei token VQ (2 × 4096) | come vlh, senza la media dei soft token |
| teste | LayerNorm → 512 → GELU → dropout 0,3 → 6 lineari | come vlh |
| loss | focal γ = 2 per testa, solo celle valutate, alpha per classe `effective` | come vlh (scelta del 27 novembre) |
| learning rate | backbone 1e-5, teste 1e-3; warmup 5% e coseno; weight decay 0,01 sulle matrici | valori tipici per un fine-tuning completo |
| ottimizzatore | AdamW 8-bit (bitsandbytes), pesi master in fp32, forward in autocast bf16 | in bf16 puro gli aggiornamenti da ~1e-5 si perdono; la ricetta Qwen usa anch'essa AdamW 8-bit |
| batch | 16, clipping 1,0, gradient checkpointing | |
| epoche | **3**, validazione 2 volte per epoca, si tiene il checkpoint migliore | |
| seed | 3407 | |

**Memoria (stima).** Pesi fp32 21 GB + gradienti fp32 21 GB + stati AdamW 8-bit 10,5 GB + embedding bf16 2,2 GB + attivazioni
con checkpointing: **circa 60 GB**. Serve quindi una GPU da 80 GB o più. Lo smoke test all'inizio del job stampa il picco reale.

**Checkpoint.** Layer allenati + norm in bf16 (`decoder.safetensors`, circa 10,5 GB), teste (`heads.pt`) e
`settings.json`. Gli embedding non si salvano: sono quelli del modello base, alla revisione scritta in
`settings.json`. Il job carica il checkpoint nel repository privato una volta sola, alla fine del training:
un upload da 10 GB a ogni miglioramento fermerebbe il training per minuti.

## Esecuzione su Hugging Face

Hardware: **`rtx-pro-6000`** (96 GB, 2,75 $/h, `hf jobs hardware`, 27 novembre 2026), invece della H200 (5 $/h).
Funziona anche `a100-large` (80 GB, 2,50 $/h), con meno margine di memoria. Il `--timeout` è il tetto di spesa.
`--max-minutes` è solo una sicurezza: se 3 epoche non ci stanno, lo schedule si accorcia e il log lo dice al passo 30.

```sh
python src/training/apertus-8B-fullft-heads/build_hf_job.py
export HF_TOKEN=$(cat ~/.cache/huggingface/token) KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
# 0. controllo su CPU (~0,01 $): segreti, dati, modello gated, tokenizer e template reali
hf jobs uv run --flavor cpu-upgrade --timeout 20m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e CHECK_ONLY=1 data/hf_jobs/hf_job_fullft.py
# 1. fase 1: 3 epoche, tetto di spesa 5 h = 13,75 $
hf jobs uv run --flavor rtx-pro-6000 --timeout 5h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e UPLOAD_REPO=<utente>/apertus-8b-galaxus-fullft -e TRAIN_ARGS="--max-minutes 230" \
    data/hf_jobs/hf_job_fullft.py
# 2. fase 2: EPOCHS dalla fase 1 (train_summary.json); 24% di item in più
hf jobs uv run --flavor rtx-pro-6000 --timeout 6h --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e PHASE=2 -e EPOCHS=<epoche> -e TRAIN_ARGS="--max-minutes 290" \
    -e UPLOAD_REPO=<utente>/apertus-8b-galaxus-fullft data/hf_jobs/hf_job_fullft.py
```

Passi del job della fase 1: download dei dati e del modello → cache VQ con 3 processi → smoke test (3 passi, con
salvataggio di un checkpoint) → training → caricamento del checkpoint migliore → predizione di `coverage` →
`eval_prompt.py --classes --bootstrap 200`. Con `PREDICT_DEV=1` predice anche il dev, senza inviare nulla. Il log del
training stampa item/s e memoria ogni 10 passi. Al passo 30 stampa anche la durata stimata: se è molto sopra le
attese, conviene fermare il job (`hf jobs cancel <id>`).

### Quanto dura e quanto costa (stima, non misurata)

Riferimento misurato: la LoRA P10 su H200 faceva 12,4 esempi/s, con 32 layer e senza checkpointing. Il fine-tuning
completo calcola anche i gradienti dei pesi e, con il checkpointing, ripete il forward: circa 2 volte il costo per
token, su 24 layer invece di 32. La RTX PRO 6000 ha circa metà della potenza bf16 di una H200 (valore di targa, non
misurato qui). Stima: **4–8 item/s**.

| passo | fase 1 | fase 2 |
|---|---|---|
| installazione, download dati e modello | 10 min | 10 min |
| cache VQ (14.331 / 16.560 varianti, 3 processi) | 15–25 min | 15–30 min |
| smoke test | ~5 min | ~5 min |
| training, 3 epoche | 40.170 item: 85–170 min | 49.680 item: 105–205 min |
| upload del checkpoint + predizione `coverage` | ~15 min | – |
| predizione dev | (opzionale, ~15 min) | ~15 min |
| **totale** | **~2,2–3,7 h, circa 6–10 $** | **~2,5–4,4 h, circa 7–12 $** |

Il costo per run è simile a quello di una H200 (metà del prezzo orario, circa il doppio del tempo). Il vantaggio è il
tetto di spesa più basso per ogni ora di margine.

## Verifiche fatte (27 novembre 2026)

- `test_fullft.py`, 10 test su CPU con la vera classe del decoder Apertus del fork, in miniatura:
  - si allenano solo layer, norm e teste, in fp32; gli embedding restano congelati in bf16;
  - i gradienti arrivano a ogni layer e mai agli embedding; una dimensione non valutata non dà gradienti alla sua testa;
  - il gradient checkpointing dà gli stessi gradienti;
  - le feature sono ultimo token + media dei token immagine, e il padding non cambia il risultato;
  - salvando e ricaricando i logit sono identici in fp32 e vicini in bf16, e un checkpoint di un'altra forma viene rifiutato;
  - qualche passo di AdamW abbassa la loss;
  - i worker dell'encode si dividono le varianti una volta sola;
  - i conteggi di DATI.md tornano sul `train.csv` reale.
- Prova completa su CPU con `--tiny` (decoder casuale in miniatura, tokenizer, template e tokenizer VQ veri, immagini vere):
  `ft_encode.py` con 2 worker (39 varianti) → `ft_train.py` (2 epoche, validazione, checkpoint) → `ft_predict.py`
  su 6 immagini di `coverage`, con punteggio di `metric.py`. I numeri non contano (pesi casuali).
- `CHECK_ONLY` del job generato, eseguito in locale: bundle, accesso Kaggle, modello gated, tokenizer e template ok.
- **Non ancora eseguito su GPU**: memoria reale, velocità, bitsandbytes sulla RTX PRO 6000. Li controllano lo smoke
  test e il log dei primi 30 passi.

```sh
python src/training/apertus-8B-fullft-heads/test_fullft.py      # oppure: python -m pytest -q src/training/apertus-8B-fullft-heads/test_fullft.py
```

## Rischi

- **Overfitting**: 5,2 B parametri su 8.403 immagini. La LoRA P10 era già a loss 0,003 dalla seconda epoca. La
  validazione interna sceglie il checkpoint fra 6 valutazioni; un lr del backbone di 1e-5 limita il danno.
- **Teste nuove con un backbone che si muove**: le teste partono da zero con un lr 100 volte più alto del backbone;
  nel primo 5% dei passi (warmup) gli aggiornamenti al backbone sono piccoli.
- **Velocità**: la stima è ampia. Il log al passo 30 la misura.

## Ablation possibili

`--lr-backbone 2e-5`, `--gamma 0 --class-alpha none` (cross-entropy), `--layers 20 / 28`, `--context-dropout 0.2`,
`--optimizer adamw` (stati fp32, +31,5 GB, solo su H200), `NO_CONTEXT_CHECK=1`.
