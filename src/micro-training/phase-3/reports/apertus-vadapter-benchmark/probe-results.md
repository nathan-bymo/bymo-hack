# SigLIP 2: sonda lineare rapida

Esecuzione locale del 27 novembre 2026: `google/siglip2-base-patch16-224` congelato, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, feature medie delle patch e una testa lineare per dimensione. La sonda è allenata sulle etichette umane e rilevanti. Lo split di `train.csv` è per `product_id` (80/20, seed 42), seguito da un campione casuale di 256 immagini di training e 64 di holdout. Il confronto usa la classe più frequente del campione di training per dimensione.

| Dimensione | Macro F1 sonda | Macro F1 classe maggioritaria | Celle holdout valutate |
| --- | ---: | ---: | ---: |
| subject | 0.472 | 0.071 | 63 |
| background | 0.780 | 0.304 | 55 |
| perspective | 0.363 | 0.063 | 45 |
| individuals | 1.000 | 0.491 | 55 |
| text_classification | 0.660 | 0.174 | 60 |
| text_language | 0.496 | 0.060 | 32 |
| Media delle sei dimensioni | **0.628** | **0.194** | — |

La macro F1 considera le classi presenti nelle etichette o nelle predizioni, come la metrica della challenge. Sono escluse le celle non rilevanti e quelle non etichettate da umani. Il campione è piccolo e le classi rare hanno poche osservazioni: il risultato, incluso `individuals = 1.000`, è instabile. Non confrontarlo direttamente con i punteggi leaderboard, calcolati su immagini e distribuzioni diverse.

**Questo non è il benchmark di Apertus con visual adapter**: non sono stati usati Apertus, il projector o il system prompt. Il test misura solo quanto le feature di SigLIP 2 siano utilizzabili da un classificatore lineare addestrato sul piccolo campione. Per valutare l'adapter servono il training del projector e la valutazione di Apertus sulle stesse immagini.

Comando eseguito dalla root del repository, con i pesi già nella cache temporanea e un ambiente Python con PyTorch e Transformers:

```bash
HF_HOME=/private/tmp/apertus-vadapter-hf /private/tmp/apertus-vadapter-venv/bin/python \
  src/micro-training/apertus-vadapter-benchmark/siglip_probe.py \
  --train-size 256 --holdout-size 64 --device mps
```

Mac Apple M1 Pro, 16 GB; 100 epoche sulle teste lineari, AdamW, learning rate 0.01. Estrazione feature e training delle teste hanno richiesto circa 11 secondi, escluso il caricamento del modello. Non è una misura di velocità dell'intera soluzione Apertus.
