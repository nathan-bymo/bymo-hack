# Apertus 8B con LoRA + teste di classificazione

Parte dal codice di Nathan (`apertus-8B-vadapter-lora-heads/`, ramo `feat/vadapter-lora-heads`), senza visual adapter e
projector. **Risultato sul training completo:** macro F1 0.781 su coverage, vedi
[results_full_h200.md](results_full_h200.md).

```
immagine ─► tokenizer VQ di Apertus (congelato, ~256²) ─► token immagine ──┐
contesto prodotto + richiesta ─────────────────────────────────────────────┤
                                                                           ▼
                      Apertus 8B, 32 layer, LoRA r32  ─► [ultimo token, media dei token immagine] ─► 6 teste
```

Un solo forward per immagine, nessun testo generato: mai celle vuote o classi inventate, probabilità già pronte.

## Cosa cambia rispetto al codice di partenza

| cosa | prima | ora | perché |
|---|---|---|---|
| `warmup` in `train.py` | variabile non definita: il training si fermava prima del primo passo | corretto; il training è eseguito dai test | |
| ripetizioni delle classi rare | ritagli del 90–100% | copie identiche | un prodotto tagliato è `detail_closeup`, ma la label restava la stessa |
| ripetizioni e mirror | fino a 4×, obiettivo 300 | fino a 3×, obiettivo 200, come il LoRA del team | confronto alla pari (3.952 esempi per epoca sul set piccolo) |
| adapter | SigLIP 2 + projector | rimossi | il modello usa solo Apertus |
| layer | 24 | 32 | con 5 seed, sulle teste congelate, 24 e 32 sono praticamente pari |
| loss | focal γ=2 con pesi per classe | cross-entropy, label umane (`--label-cells human`) | la focal non ha aiutato (prova P9); stesse label del LoRA |
| epoche e scelta | 2, checkpoint sulla validazione interna | 3, modello finale, nessuna scelta (`--val-fraction 0`) | come il LoRA |
| batch | 16 | 8 × 2 accumuli; se la memoria finisce, il job riprova con gradient checkpointing | il LoRA del team ha riservato 106,7 GB con batch 8 |
| split piccolo | non supportato | `--train-csv`; il job usa `data/nathan_split/` | |
| salvataggio | un upload fallito non fermava il job | il modello finale viene controllato file per file | un adapter è già andato perso |
| predizioni | niente ripresa | riprende, salta le immagini illeggibili; `make_submission.py` crea il file Kaggle | |

`--gamma`, `--class-alpha`, `--layers`, `--val-fraction` restano disponibili per le ablation.

## Confronto con il LoRA (stesse condizioni, per quanto possibile)

Uguali: immagini di training, ripetizioni e mirror, risoluzione 256², LoRA (rank, moduli, learning rate, weight
decay), 3 epoche, 16 immagini per passo, clipping, warmup e schedule, label umane, contesto, seed.
Diversi: **uscita** (teste invece di JSON), **prompt** (il LoRA usa un system prompt breve; qui una richiesta dopo il
contesto, senza system prompt nostro), ordine casuale dei batch, dettagli dell'implementazione.

## File

| file | contenuto |
|---|---|
| `data.py` | split, label, ripetizioni e mirror |
| `encoders.py`, `encode.py` | immagine → codici VQ, sequenza di token; cache dei codici di training |
| `model.py` | Apertus + LoRA + teste, salvataggio e caricamento |
| `train.py` | training |
| `predict.py` | **comando per le predizioni**, dalle immagini al CSV |
| `make_submission.py` | file Kaggle da `sample_submission.csv` e dai CSV di predizione; **non invia nulla** |
| `build_hf_job.py` | job per Hugging Face: `check`, `smoke`, `train` |
| `test_pipeline.py` | test su CPU |

## Predizioni e file Kaggle

```sh
python src/apertus-8B-lora-heads/predict.py --checkpoint <modello> --csv data/dev.csv --out data/runs/x/dev.csv
python src/apertus-8B-lora-heads/predict.py --checkpoint <modello> --csv data/final/final.csv --out data/runs/x/final.csv
python src/apertus-8B-lora-heads/make_submission.py --predictions data/runs/x/dev.csv data/runs/x/final.csv \
    --require data/dev.csv data/final/final.csv --out data/submissions/submission.csv
```

Richiede GPU, `data/dev_images/` e, dopo le 13:00 del 28 novembre, le immagini final estratte in `data/final/`. La
passphrase non va nel repository. `make_submission.py` accetta CSV di qualsiasi metodo con `image_id` e le sei
dimensioni. **Non è incluso** un comando che produca su dev e final le risposte di un LoRA generativo.

## Job su Hugging Face

I comandi sono nella docstring di `build_hf_job.py`. Ordine: `MODE=check` (CPU), `MODE=smoke` (3 passi, upload,
download in una cartella nuova, ricaricamento, stesse predizioni), `MODE=train`. Il job non esegue le predizioni su
dev e final.

## Verifiche fatte (27 novembre 2026)

- `test_pipeline.py`, 10 test su CPU, tra cui la catena completa encode → train → salvataggio → ricaricamento →
  predizioni con ripresa → file Kaggle, con la **classe vera del decoder Apertus** del fork e PEFT su un modello in
  miniatura con pesi casuali. Tokenizer e tokenizer VQ sono sostituti.
- Esempi identici a quelli del LoRA su `training_rows` e sul set piccolo.
- `make_submission.py` sul formato vero di `sample_submission.csv` e `dev.csv`, con risposte finte.
- `MODE=check` su Hugging Face (CPU): accesso ai dati e al modello, tokenizer, chat template e image processor veri.
- `MODE=smoke` su H200: 3 passi, upload, download, ricaricamento, stesse predizioni su 16 immagini.
- Training completo e predizione di coverage su H200: vedi [results_full_h200.md](results_full_h200.md).
- `MODE=predict` su H200: modello riscaricato, stesse risposte su coverage, 5.261 immagini dev predette; dev pubblico Kaggle 0.796.
- **Non ancora provate:** predizioni sulle immagini final, ripresa di un job interrotto.

```sh
python -m pytest -q src/apertus-8B-lora-heads/test_pipeline.py
```
