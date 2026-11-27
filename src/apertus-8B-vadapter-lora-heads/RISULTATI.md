# Risultati e pipeline di training

Stato del 27 novembre 2026, sera. Solo metriche aggregate: nessuna immagine, label o predizione individuale.
**Ogni confronto vale sullo stesso set di immagini.** Nessuno di questi numeri è un punteggio Kaggle.

## Modello finale

```
immagine ─┬─► SigLIP 2 (congelato) ─► projector ─► 49 soft token ──────────────┐
          ├─► tokenizer VQ di Apertus (congelato), ~256 token ──────────────────┤
          └─► OCR docTR (congelato) ─► blocco <ocr> nel prompt ─────────────────┤
contesto prodotto + richiesta ─────────────────────────────────────────────────┤
                                                                               ▼
                       Apertus 8B, 24 layer su 32, LoRA r32
                                                                               ▼
   [ultimo token, media token VQ, media soft token]  +  feature spaziali (sinistra−destra, alto−basso)
                                                      +  attention pooling con posizioni 2D
                                                                               ▼
                                     6 teste ─► 6 classi
```

Pesi: repository Hugging Face **privato** `nathan-monzani-bymo/galaxus-vlh-private`, cartella `final-abt/model/best`
(LoRA, projector, teste, attention pooling e `settings.json`). Non sono su git (`.gitignore`, regole del team).

## Pipeline di training effettuata

Ogni passo riparte dal checkpoint del precedente. Hugging Face Jobs, 1× H200, bf16. Comandi in `build_hf_job.py`.

| passo | checkpoint | dati di training | cosa cambia | epoche | tempo | costo |
|---|---|---|---|---|---|---|
| 1 | `phase1/model/best` | `training_rows` meno validazione interna (8.403 immagini) | da zero: projector, LoRA, teste; augmentation con ritagli (poi corretta, vedi sotto) | 2 | 31 min training, 58 min job | ~4,80 $ |
| 2 | `phase1-ocr/model/best` | come il passo 1 | + testo OCR nel prompt, augmentation corretta (`vq-scale-v2`), batch 8 × accumulo 2 | +1 | 19 min training | ~2,70 $ (più ~2,40 $ di un tentativo andato in OOM) |
| 3 | **`final-abt/model/best`** | tutto tranne prompt-test (9.645 immagini, 14.459 item per epoca) | + feature spaziali, + attention pooling 2D (strati nuovi inizializzati a zero) | +3 | 63 min training, 73 min job | ~6,10 $ |

Impostazioni comuni: focal loss γ = 2 con alpha per classe (numero effettivo di campioni, calcolato dopo
l'oversampling), oversampling delle classi rare fino a 4× con varianti diverse (scala VQ 224/288/240, mirror solo
senza testo), learning rate a coseno adattato al budget di minuti, gradient clipping 1,0. Passi 2 e 3: learning rate
dimezzati (LoRA 1e-4, projector 5e-5, teste 5e-4). Dettagli dei dati in [DATI.md](DATI.md).

## Prompt-test: 1.193 immagini, distribuzione naturale

Mai viste da nessun checkpoint (né in training né per scegliere epoche). È il numero più onesto disponibile.

| modello | macro F1 | weighted F1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|
| passo 1 | 0,750 | 0,887 | 0,828 | 0,803 | 0,586 | 0,960 | 0,772 | 0,550 |
| passo 2 (+ OCR) | 0,741 | 0,891 | 0,816 | 0,808 | 0,544 | 0,955 | 0,771 | 0,554 |
| **passo 3, finale** | **0,821** | **0,917** | **0,846** | **0,881** | **0,755** | 0,959 | 0,766 | **0,720** |

Classi più deboli del modello finale: `certification_seal` 0,40 (6 esempi), `other` (testo) 0,55, `Dutch` 0,40 e
`Italian` 0,67 (4 esempi ciascuna: valori molto rumorosi), `front_angle_left` 0,64, `back_angle_right` 0,65.

Il passo 3 cambia tre cose insieme (più dati, feature spaziali + attention pooling, più epoche): il guadagno non è
separabile tra le tre.

## Coverage (2.532 immagini, ricco di classi rare) — solo passi 1 e 2

Il passo 3 ha visto parte di coverage in training: non ha un punteggio valido qui.

| modello | macro F1 | weighted F1 | perspective | text_language |
|---|---|---|---|---|
| passo 1 | 0,742 | 0,879 | 0,576 | 0,533 |
| passo 2 | 0,748 | 0,885 | 0,553 | 0,575 |
| passo 2 + flip TTA | 0,749 | 0,885 | 0,572 | 0,575 |

Riferimenti sullo stesso set: 8B zero-shot v2 0,528, 70B zero-shot v2 0,583.

## Cosa abbiamo imparato

- **Sinistra/destra era a caso.** Con le sole medie dei token immagine il modello sceglieva il lato giusto nel 55%
  dei casi, con il 92% delle probabilità tra 35/65 e 65/35: medie uguali per un'immagine e il suo specchio, e il
  mirror (label scambiate) spingeva al 50/50. Feature che cambiano segno con lo specchio e un attention pooling che
  vede le posizioni 2D portano perspective da 0,54 a 0,76 su prompt-test.
- **La calibrazione post-hoc non aiuta.** Bias per classe scelti sulla validazione: coverage 0,743 → 0,738
  (overfitting); logit adjustment con un parametro per dimensione: 0,743 → 0,745. Lo sbilanciamento è già corretto
  in training (focal + alpha + oversampling); gli errori residui sono di percezione.
- **Una validazione senza classi rare inganna.** Nella validazione del passo 2 mancavano Italian e Dutch:
  predirne uno solo aggiunge una classe a F1 zero alla macro. 0,746 → 0,801 in validazione, solo +0,006 su coverage.
- **L'OCR aiuta le lingue con dati** (German 0,79 → 0,86, French 0,63 → 0,69 su coverage), non quelle quasi
  assenti dal training (Dutch: 1 esempio).
- **Augmentation: niente ritagli.** Il prodotto tocca il bordo in 269 immagini su 300: un ritaglio del 10% tagliava
  il contenuto in 231 casi su 300, trasformando `full_product` in `detail_closeup`. Corretto dal passo 2.
- **Flip TTA:** +0,019 su perspective ma −0,009 su individuals e −17% di velocità: non adottato.

## Velocità (H200, end to end, lettura immagini inclusa)

| modello | img/s | dettaglio |
|---|---|---|
| passo 1 (senza OCR) | 13,8 | |
| finale (con OCR) | 6,2 | su prompt-test, per 1.193 immagini: preparazione 7,6 s, SigLIP 27,4 s, VQ 80,7 s, OCR 75,5 s, Apertus + teste 83,5 s |

Apertus con le teste: circa 30 ms per immagine nel passo 2 (~33 img/s), circa 70 ms nel finale, dove feature
spaziali e attention pooling sono calcolati con cicli Python per immagine (vettorizzabili). Il resto del tempo va a
VQ (fp32) e OCR, in sequenza nello stesso thread. Qwen 3.5 9B di riferimento: ~29 img/s su H100 con vLLM. Requisito della challenge:
2 img/s.

## Riproducibilità e inferenza

- Training: `build_hf_job.py` genera il job (variabili `PART`, `INIT_FROM`, `OCR`, `RUN_NAME`, `TRAIN_ARGS`).
- Inferenza H200: `predict.py --checkpoint <cartella>`; su 2 × T4 (Kaggle): `--dtype float16 --encoder-device cuda:1`
  con il controllo overflow fp16 di `model.py`. Notebook Kaggle: `build_kaggle_notebook.py`.
- Test: `python -m pytest -q src/apertus-8B-vadapter-lora-heads/test_pipeline.py` (30 test, CPU).

Modelli usati a runtime (da elencare nel report): Apertus v1.5 8B, SigLIP 2 (`google/siglip2-base-patch16-224`),
docTR (`fast_base` + `Felix92/doctr-torch-parseq-multilingual-v1`).
