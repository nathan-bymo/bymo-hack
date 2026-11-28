# Benchmark dei tre modelli Apertus 8B su H200

## Sintesi

**final-abt è il riferimento di qualità sul dev Kaggle. P12 è il più veloce su H200. P11 offre il checkpoint più piccolo dei tre e un throughput vicino a P12.** La priorità comune è la distinzione sinistra/destra in `perspective`. Per isolare il contributo dell’architettura, il prossimo esperimento dovrebbe tenere fissi dati, epoche e budget e variare prima le feature spaziali e poi l’OCR.

## Protocollo

- **Dati.** Le passate a batch 32 hanno elaborato tutte le 5.261 immagini di `dev.csv`. Il runner verifica l’hash SHA-256 del CSV scaricato da Kaggle contro quello del file locale e controlla la copertura delle predizioni per ID. Il batch 1 usa le prime 128 righe del CSV, un campione deterministico.
- **Ambiente.** Hugging Face Jobs, una NVIDIA H200, modelli eseguiti in sequenza con contesto prodotto e BF16 per il decoder. Ogni checkpoint mantiene la propria pipeline di preprocessing. Il vision adapter ordina le immagini per dimensione VQ e gira con flip TTA disattivato. Tutti partono da Apertus v1.5 8B, revisione `1519de0ae7f3d114879291810a58259966f64c99`.
- **Tempi.** La finestra di inferenza include lettura e preparazione effettiva delle immagini, encoder e classificazione. Caricamento del modello, ordinamento preliminare per VQ e scrittura finale sono fuori dalla finestra dove applicabile. P11 include la scrittura incrementale delle predizioni. I ms/immagine sono la durata della passata divisa per il numero di immagini. Anche batch 1 usa il prefetch, quindi il dato rappresenta il ritmo medio del flusso e non la latenza isolata di una richiesta.
- **Qualità.** Il CSV dev contiene immagini e contesto, con etichette riservate alla valutazione Kaggle. I punteggi della colonna Kaggle sono storici. La diagnostica sul dev usa le confidenze softmax e i disaccordi, senza applicare il filtro ufficiale sulle celle umane e rilevanti. La confidenza è grezza e può avere calibrazione diversa tra modelli.
- **Esecuzione.** Il primo job si è fermato durante il caricamento dell’OCR. Il job con il controllo OpenCV corretto ha salvato tutti i tempi e si è poi fermato per un import del runner. La passata diagnostica successiva ha completato l’aggregazione. Le sue velocità figurano come replica a batch 32.

Runner: `src/scripts/benchmark_dev.py` (removed in the clean-up, see git history) · [Job dei tempi](https://huggingface.co/jobs/StoryofLegend/6b0a153e54072e8b716b2b36) · [Job diagnostico completato](https://huggingface.co/jobs/StoryofLegend/6b0a1cd454072e8b716b458a) · [Regole di scoring](../challenge_description.md#evaluation)

## Confronto operativo

| Modello | Secondi su dev | img/s batch 32 | ms/img batch 32 | img/s replica | ms/img batch 1 | Picco GPU batch 32 | Kaggle dev storico |
|---|---:|---:|---:|---:|---:|---:|---:|
| P12 · Full fine-tuning + 6 teste | 359,7 | 14,63 | 68,37 | 14,36 | 85,16 | 18,7 GiB | 0,80473 |
| P11 · LoRA + 6 teste | 396,3 | 13,27 | 75,33 | 13,17 | 90,62 | 22,2 GiB | 0,796 |
| final-abt · Vision adapter + LoRA + OCR + 6 teste | 760,9 | 6,91 | 144,63 | 6,88 | 174,22 | 79,8 GiB | 0,83177 |

I picchi sono memoria **allocata** da PyTorch. P12 e final-abt azzerano il contatore dopo il caricamento. P11 lo riporta dall’avvio del processo. A 5 USD/ora, il costo stimato di sola inferenza per 100.000 immagini è 9,49 USD per P12, 10,47 per P11 e 20,10 per final-abt. La stima usa il throughput batch 32 e lascia fuori avvio, lavoro di preparazione del job e trasferimenti.

| Modello | Secondi batch 1 | img/s batch 1 | Picco GPU batch 1 | Durata processo batch 32 | Durata processo batch 1 |
|---|---:|---:|---:|---:|---:|
| P12 | 10,9 | 11,72 | 13,1 GiB | 383,48 s | 34,61 s |
| P11 | 11,6 | 11,07 | 17,1 GiB | 422,05 s | 37,64 s |
| final-abt | 22,3 | 5,74 | 13,9 GiB | 823,29 s | 49,60 s |

La durata del processo include il caricamento e la scrittura dei risultati. Il job aveva già preparato il dataset e scaricato il checkpoint remoto di P12, mentre la cache condivisa del modello base si è popolata durante la prima esecuzione di final-abt.

## Risultati con etichette

| Modello | Set | Macro F1 | Weighted F1 | Subject | Background | Perspective | Individuals | Text classification | Text language |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P11 | coverage · 2.532 | 0,781 | 0,905 | 0,841 | 0,911 | 0,625 | 0,981 | 0,806 | 0,521 |
| P11 | base · 2.363 | 0,786 | 0,913 | 0,848 | 0,914 | 0,628 | 0,990 | 0,817 | 0,521 |
| P11 | prompt-test · 1.193 | 0,791 | 0,915 | 0,860 | 0,894 | 0,620 | 0,990 | 0,805 | 0,577 |
| final-abt | prompt-test · 1.193 | 0,821 | 0,917 | 0,846 | 0,881 | 0,755 | 0,959 | 0,766 | 0,720 |

P12 ha utilizzato tutte le 11.876 immagini etichettate e il suo risultato valido disponibile è il Kaggle dev storico. Final-abt ha visto parte di `coverage` in training e la sua valutazione locale valida è `prompt-test`. Il confronto su `prompt-test` usa le stesse immagini per P11 e final-abt. Cambiano quantità di dati, numero di epoche, loss e architettura, quindi il guadagno complessivo di 0,030 macro F1 non isola un singolo contributo. P12 · P11 · Final-abt

## Come leggere la diagnostica per classe

**N** è il numero di volte in cui il modello ha predetto la classe. **Conf. media** è la media della probabilità softmax della classe vincente. **<0,60** è la quota di queste predizioni sotto la soglia. **Disaccordo** indica che almeno uno degli altri modelli predice una classe diversa. **Alternativa comune** è la quota per cui gli altri due concordano su una stessa classe diversa. L’ultima colonna mostra l’alternativa più frequente in quel sottoinsieme, con il suo conteggio. Ogni dimensione totalizza 5.261 predizioni per modello. Le percentuali sono rapportate a N. Una riga con N = 0 indica assenza di predizioni e non fornisce un denominatore per le percentuali.

## P12 · Full fine-tuning + 6 teste

### Checkpoint, architettura e training

- Checkpoint privato `StoryofLegend/apertus-8b-galaxus-fullft`, revisione `aac1375d3efb93460a564faaef00a60430d84443`, percorso `phase2/model/last`. Decoder da circa 10,47 GB, più teste e configurazione.
- 24 layer del decoder addestrati con norma finale e sei teste. Embedding e tokenizer VQ congelati. Immagini VQ a 256². Le teste usano l’ultimo token e la media dei token immagine. Il riepilogo riporta 5.238.930.521 parametri allenati.
- Training su 11.876 immagini, 16.560 esempi per epoca con ripetizioni e mirror, 3 epoche, 3.105 passi e seed 3407. Focal loss γ = 2 con pesi di classe effective sulle celle umane e rilevanti. Batch 16, LR decoder 1e-5 e LR teste 1e-3, AdamW a 8 bit.
- Training su A100 da 80 GB. 165,3 min di addestramento e 57,1 GiB di picco riportato.

La difficoltà principale sul dev è `perspective`: 44,4% di predizioni sotto 0,60. `right_side` è a bassa confidenza nel 99,2% dei 120 casi e gli altri modelli concordano su un’alternativa nel 32,5%. `Dutch` compare 110 volte con il 90,0% sotto soglia. Per capire se il problema dipende dal checkpoint o dalla ricetta, serve una replica con holdout per prodotto fissato prima del training e checkpoint intermedi valutabili.

Report P12 · Riepilogo training · [Inferenza](../src/training/phase-1/src/training/apertus-8B-fullft-heads/ft_predict.py)

### Diagnostica dev per dimensione

#### subject

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `full_product` | 3 096 | 89,7% | 7,6% | 8,5% | 1,5% | `detail_closeup` (16) |
| `detail_closeup` | 430 | 78,3% | 21,2% | 31,4% | 7,9% | `full_product` (33) |
| `accessories` | 60 | 85,4% | 16,7% | 46,7% | 25,0% | `full_product` (12) |
| `packaging` | 173 | 85,6% | 12,1% | 20,8% | 11,0% | `full_product` (15) |
| `package_and_product` | 134 | 84,1% | 14,2% | 32,1% | 9,7% | `full_product` (6) |
| `multipack_trade_unit` | 59 | 87,9% | 10,2% | 25,4% | 3,4% | `full_product` (1) |
| `delivery_scope` | 223 | 89,2% | 9,4% | 15,7% | 4,5% | `full_product` (7) |
| `product_grouping` | 37 | 85,2% | 10,8% | 37,8% | 16,2% | `multipack_trade_unit` (3) |
| `multiview` | 80 | 81,2% | 17,5% | 40,0% | 11,2% | `full_product` (8) |
| `energy_label` | 203 | 99,5% | 0,5% | 1,0% | 0,5% | `infographic` (1) |
| `infographic` | 482 | 83,0% | 13,9% | 26,6% | 6,4% | `full_product` (20) |
| `logo` | 230 | 96,3% | 3,0% | 3,0% | 1,3% | `infographic` (2) |
| `placeholder` | 54 | 96,8% | 1,9% | 7,4% | 0,0% | — |

#### background

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `white` | 4 459 | 93,3% | 1,5% | 2,5% | 0,2% | `color` (9) |
| `color` | 451 | 83,7% | 15,3% | 18,2% | 4,7% | `white` (16) |
| `scene_lifestyle` | 351 | 88,7% | 11,7% | 16,8% | 2,6% | `color` (6) |

#### perspective

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `front_head_on` | 2 313 | 80,4% | 15,7% | 11,4% | 1,9% | `top_down` (22) |
| `front_angle_left` | 959 | 50,5% | 93,8% | 68,6% | 23,4% | `front_angle_right` (156) |
| `front_angle_right` | 438 | 47,3% | 92,5% | 70,3% | 17,8% | `front_angle_left` (56) |
| `back_head_on` | 516 | 85,1% | 10,3% | 24,4% | 7,0% | `front_head_on` (33) |
| `back_angle_left` | 128 | 50,9% | 77,3% | 82,8% | 27,3% | `back_angle_right` (14) |
| `back_angle_right` | 148 | 50,0% | 83,1% | 70,3% | 15,5% | `back_angle_left` (10) |
| `left_side` | 197 | 50,6% | 93,4% | 61,4% | 18,8% | `right_side` (27) |
| `right_side` | 120 | 48,9% | 99,2% | 73,3% | 32,5% | `left_side` (23) |
| `top_down` | 316 | 75,9% | 23,1% | 34,8% | 12,3% | `front_head_on` (29) |
| `bottom_up` | 126 | 85,1% | 14,3% | 18,3% | 2,4% | `back_head_on` (2) |

#### individuals

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `individual_present` | 281 | 95,6% | 1,1% | 6,0% | 1,4% | `none` (4) |
| `none` | 4 980 | 99,0% | 0,1% | 0,3% | 0,1% | `individual_present` (7) |

#### text_classification

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `certification_seal` | 27 | 78,3% | 18,5% | 77,8% | 29,6% | `none` (7) |
| `technical_specs` | 304 | 78,3% | 26,6% | 48,0% | 24,3% | `none` (71) |
| `promotional_text` | 170 | 85,2% | 12,9% | 17,6% | 7,1% | `none` (7) |
| `nutritional_info` | 309 | 90,1% | 4,5% | 7,8% | 1,3% | `none` (4) |
| `other` | 159 | 72,3% | 28,3% | 35,8% | 15,7% | `none` (15) |
| `none` | 4 292 | 93,5% | 2,5% | 2,8% | 0,8% | `other` (15) |

#### text_language

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `German` | 757 | 82,0% | 19,2% | 21,7% | 4,0% | `English` (15) |
| `English` | 1 245 | 73,9% | 24,4% | 43,0% | 17,6% | `none` (187) |
| `French` | 104 | 85,5% | 13,5% | 35,6% | 15,4% | `German` (7) |
| `Italian` | 24 | 87,5% | 8,3% | 87,5% | 16,7% | `English` (1) |
| `Dutch` | 110 | 47,0% | 90,0% | 100,0% | 4,5% | `German` (4) |
| `other` | 52 | 72,5% | 28,8% | 69,2% | 19,2% | `English` (4) |
| `none` | 2 969 | 81,5% | 9,1% | 10,0% | 1,5% | `English` (40) |

## P11 · LoRA + 6 teste

### Checkpoint, architettura e training

- Checkpoint locale in `weights/lora_heads_256_p11`, con adapter PEFT, teste e configurazione. I due file tensoriali totalizzano circa 336 MB.
- 32 layer, immagini VQ a 256², LoRA r32 e α = 32 sulle proiezioni q, k, v, o, up e down. Sei teste su ultimo token e media dei token immagine. Il report riporta 83,9 M di parametri allenabili. Il checkpoint usa la configurazione senza encoder aggiuntivo, OCR, feature spaziali e attention pooling.
- Training su 9.344 immagini di `training_rows` e 12.491 esempi per epoca con ripetizioni fino a 3× e mirror. Tre epoche fissate, seed 3407. Cross-entropy senza pesi, batch 8 con accumulo 2, LR LoRA 2e-4 e LR teste 1e-3. Il report P11 descrive supervisione su label umane anche in celle non rilevanti.
- Training su H200, 2.343 passi e circa 46,7 minuti di addestramento. Per il confronto locale sono disponibili `coverage`, `base` e `prompt-test`.

Sul dev, `right_side` e `left_side` sono sotto soglia rispettivamente nel 92,4% e nel 73,6% delle predizioni. `Dutch` compare zero volte. Su `coverage`, i punti più deboli documentati sono `Dutch` (F1 0,00, 20 esempi), `other` in `text_language` (0,17, 30), `Italian` (0,28, 30) e `back_angle_left` (0,28, 37). Su `product_grouping`, il recall è 0,50 su 30 esempi.

Report P11 · [Checkpoint](../weights/lora_heads_256_p11/README.md) · [Inferenza](../src/training/phase-1/src/training/apertus-8B-lora-heads/predict.py)

### Diagnostica dev per dimensione

#### subject

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `full_product` | 3 231 | 97,0% | 2,2% | 12,3% | 4,3% | `detail_closeup` (67) |
| `detail_closeup` | 369 | 92,2% | 7,3% | 20,1% | 5,7% | `full_product` (15) |
| `accessories` | 45 | 90,6% | 4,4% | 28,9% | 13,3% | `full_product` (2) |
| `packaging` | 157 | 93,6% | 4,5% | 12,7% | 3,8% | `full_product` (3) |
| `package_and_product` | 114 | 91,6% | 4,4% | 20,2% | 4,4% | `full_product` (4) |
| `multipack_trade_unit` | 62 | 90,6% | 6,5% | 29,0% | 16,1% | `full_product` (6) |
| `delivery_scope` | 245 | 91,1% | 9,0% | 23,3% | 7,8% | `full_product` (11) |
| `product_grouping` | 45 | 79,8% | 24,4% | 48,9% | 20,0% | `full_product` (3) |
| `multiview` | 72 | 87,5% | 15,3% | 33,3% | 12,5% | `full_product` (3) |
| `energy_label` | 201 | 100,0% | 0,0% | 0,0% | 0,0% | — |
| `infographic` | 430 | 90,8% | 7,4% | 17,7% | 5,6% | `full_product` (18) |
| `logo` | 233 | 98,1% | 2,6% | 4,3% | 0,4% | `infographic` (1) |
| `placeholder` | 57 | 97,1% | 5,3% | 12,3% | 1,8% | `infographic` (1) |

#### background

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `white` | 4 413 | 98,6% | 0,6% | 1,5% | 0,4% | `color` (8) |
| `color` | 522 | 93,2% | 5,0% | 29,3% | 16,5% | `white` (69) |
| `scene_lifestyle` | 326 | 95,1% | 4,0% | 10,4% | 1,8% | `white` (3) |

#### perspective

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `front_head_on` | 2 468 | 92,3% | 6,3% | 16,9% | 5,0% | `back_head_on` (65) |
| `front_angle_left` | 822 | 57,1% | 68,6% | 63,4% | 13,3% | `front_angle_right` (78) |
| `front_angle_right` | 581 | 58,0% | 68,3% | 77,6% | 26,5% | `front_angle_left` (120) |
| `back_head_on` | 452 | 92,9% | 6,0% | 13,7% | 4,6% | `front_head_on` (12) |
| `back_angle_left` | 53 | 55,7% | 60,4% | 58,5% | 15,1% | `back_angle_right` (7) |
| `back_angle_right` | 136 | 62,0% | 50,0% | 67,6% | 14,7% | `back_angle_left` (15) |
| `left_side` | 174 | 54,7% | 73,6% | 56,3% | 14,4% | `right_side` (12) |
| `right_side` | 118 | 51,6% | 92,4% | 72,9% | 25,4% | `left_side` (27) |
| `top_down` | 335 | 84,0% | 17,9% | 38,5% | 14,9% | `front_head_on` (39) |
| `bottom_up` | 122 | 94,2% | 2,5% | 15,6% | 3,3% | `top_down` (4) |

#### individuals

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `individual_present` | 284 | 98,8% | 0,7% | 7,0% | 1,1% | `none` (3) |
| `none` | 4 977 | 99,9% | 0,0% | 0,2% | 0,1% | `individual_present` (3) |

#### text_classification

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `certification_seal` | 8 | 68,6% | 50,0% | 25,0% | 0,0% | — |
| `technical_specs` | 176 | 94,2% | 4,5% | 10,2% | 0,6% | `none` (1) |
| `promotional_text` | 174 | 93,3% | 4,0% | 19,5% | 4,0% | `none` (3) |
| `nutritional_info` | 300 | 97,0% | 0,7% | 5,0% | 1,3% | `none` (3) |
| `other` | 164 | 84,5% | 11,0% | 37,8% | 15,2% | `none` (19) |
| `none` | 4 439 | 98,9% | 0,7% | 6,1% | 2,2% | `technical_specs` (60) |

#### text_language

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `German` | 771 | 92,7% | 5,8% | 23,1% | 7,5% | `English` (40) |
| `English` | 1 021 | 90,0% | 8,1% | 30,5% | 11,4% | `none` (94) |
| `French` | 99 | 92,1% | 8,1% | 32,3% | 8,1% | `English` (5) |
| `Italian` | 3 | 94,8% | 0,0% | 0,0% | 0,0% | — |
| `Dutch` | 0 | — | — | — | — | — |
| `other` | 192 | 80,7% | 15,6% | 91,7% | 15,1% | `English` (13) |
| `none` | 3 175 | 95,7% | 2,1% | 15,8% | 4,1% | `English` (113) |

## final-abt · Vision adapter + LoRA + OCR + 6 teste

### Checkpoint, architettura e training

- Checkpoint `final-abt/model/best` materializzato in `src/outputs/apertus-8B-vadapter-lora-heads-final`. Adapter LoRA e tensori allenabili totalizzano circa 392 MB.
- 24 layer Apertus con LoRA r32, α = 32 e dropout LoRA 0,05. Input con VQ a 256², 49 soft token da SigLIP 2 congelato e projector, testo docTR e contesto prodotto. Le sei teste combinano ultimo token, medie VQ e soft token, differenze spaziali orizzontali e verticali e attention pooling con posizioni 2D.
- Tre passaggi su H200 in BF16. Fase iniziale su 8.403 immagini di fit per 2 epoche. Continuazione con OCR e augmentation corretta sullo stesso fit per 1 epoca. Passo finale su 9.645 immagini, escludendo `prompt-test`, per altre 3 epoche. Il passo finale aggiunge feature spaziali e attention pooling.
- Focal loss con γ = 2 e alpha effective. Oversampling fino a 4× con varianti di scala VQ e mirror nelle condizioni previste dalla ricetta. Nelle due continuazioni, LR LoRA 1e-4, projector 5e-5 e teste 5e-4. Il passo finale riporta 14.459 esempi per epoca e 63 minuti di training.
- Profilo a batch 32 sul dev, in secondi: lettura e preparazione 33,8, SigLIP 116,0, VQ 281,3, OCR 329,1, forward 399,5. Le fasi si sovrappongono. Il picco allocato è 79,8 GiB a batch 32 e 13,9 GiB a batch 1, su insiemi di immagini di dimensione diversa.

Il 25,9% delle predizioni di `perspective` è sotto 0,60. Per `front_angle_right` sono il 55,4% su 659 casi. In 166 casi entrambi gli altri modelli scelgono `front_angle_left`. Sul `prompt-test`, `certification_seal` e `Dutch` hanno F1 0,40, rispettivamente su 6 e 4 esempi. Il supporto ridotto rende questi valori sensibili a pochi casi.

Report final-abt · [Impostazioni checkpoint](../weights/apertus-8B-vadapter-lora-heads-final/settings.json) · [Inferenza](../src/training/phase-2/src/training/apertus-8B-vadapter-lora-heads/predict.py)

### Diagnostica dev per dimensione

#### subject

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `full_product` | 3 085 | 91,3% | 6,2% | 8,2% | 2,2% | `detail_closeup` (20) |
| `detail_closeup` | 469 | 82,7% | 13,6% | 37,1% | 15,4% | `full_product` (68) |
| `accessories` | 55 | 78,5% | 23,6% | 41,8% | 20,0% | `full_product` (10) |
| `packaging` | 158 | 89,6% | 5,1% | 13,3% | 5,1% | `package_and_product` (4) |
| `package_and_product` | 112 | 86,7% | 14,3% | 18,8% | 6,2% | `full_product` (6) |
| `multipack_trade_unit` | 60 | 89,2% | 8,3% | 26,7% | 11,7% | `full_product` (5) |
| `delivery_scope` | 226 | 90,3% | 8,4% | 16,8% | 5,3% | `full_product` (9) |
| `product_grouping` | 34 | 89,6% | 5,9% | 32,4% | 0,0% | — |
| `multiview` | 77 | 82,2% | 19,5% | 37,7% | 15,6% | `full_product` (6) |
| `energy_label` | 202 | 99,9% | 0,0% | 0,5% | 0,0% | — |
| `infographic` | 487 | 81,2% | 15,0% | 27,3% | 8,6% | `full_product` (30) |
| `logo` | 239 | 96,3% | 2,9% | 6,7% | 3,8% | `full_product` (4) |
| `placeholder` | 57 | 95,6% | 5,3% | 12,3% | 8,8% | `infographic` (5) |

#### background

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `white` | 4 490 | 94,5% | 1,9% | 3,1% | 0,9% | `color` (32) |
| `color` | 420 | 83,7% | 13,6% | 12,1% | 6,0% | `white` (15) |
| `scene_lifestyle` | 351 | 87,3% | 13,4% | 16,8% | 6,0% | `color` (11) |

#### perspective

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `front_head_on` | 2 548 | 83,5% | 15,4% | 19,5% | 4,8% | `front_angle_left` (67) |
| `front_angle_left` | 611 | 62,1% | 51,1% | 50,7% | 11,0% | `front_angle_right` (41) |
| `front_angle_right` | 659 | 60,5% | 55,4% | 80,3% | 29,0% | `front_angle_left` (166) |
| `back_head_on` | 490 | 89,4% | 8,4% | 20,4% | 3,1% | `front_head_on` (15) |
| `back_angle_left` | 102 | 72,4% | 27,5% | 78,4% | 35,3% | `back_angle_right` (33) |
| `back_angle_right` | 101 | 63,5% | 46,5% | 56,4% | 14,9% | `back_angle_left` (8) |
| `left_side` | 180 | 70,2% | 28,9% | 57,8% | 12,8% | `right_side` (17) |
| `right_side` | 132 | 67,9% | 39,4% | 75,8% | 24,2% | `left_side` (26) |
| `top_down` | 315 | 79,8% | 21,9% | 34,6% | 14,9% | `front_head_on` (38) |
| `bottom_up` | 123 | 93,4% | 4,9% | 16,3% | 3,3% | `top_down` (4) |

#### individuals

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `individual_present` | 278 | 93,7% | 2,5% | 5,0% | 1,4% | `none` (4) |
| `none` | 4 983 | 99,4% | 0,3% | 0,3% | 0,2% | `individual_present` (10) |

#### text_classification

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `certification_seal` | 38 | 69,7% | 39,5% | 84,2% | 21,1% | `none` (8) |
| `technical_specs` | 254 | 85,0% | 17,3% | 37,8% | 6,7% | `none` (15) |
| `promotional_text` | 168 | 89,0% | 6,0% | 16,7% | 3,0% | `none` (3) |
| `nutritional_info` | 324 | 90,7% | 4,3% | 12,0% | 5,2% | `none` (16) |
| `other` | 157 | 76,4% | 24,2% | 35,0% | 11,5% | `none` (13) |
| `none` | 4 320 | 95,1% | 2,8% | 3,5% | 0,4% | `other` (10) |

#### text_language

| Classe | N | Conf. media | <0,60 | Disaccordo | Alternativa comune | Alternativa più frequente |
|---|---:|---:|---:|---:|---:|---|
| `German` | 745 | 88,6% | 9,4% | 20,4% | 5,5% | `English` (27) |
| `English` | 1 269 | 79,0% | 19,5% | 44,1% | 13,6% | `none` (138) |
| `French` | 90 | 89,4% | 10,0% | 25,6% | 12,2% | `English` (6) |
| `Italian` | 19 | 90,2% | 10,5% | 84,2% | 10,5% | `German` (1) |
| `Dutch` | 9 | 89,3% | 11,1% | 100,0% | 55,6% | `German` (5) |
| `other` | 59 | 77,8% | 23,7% | 72,9% | 22,0% | `none` (5) |
| `none` | 3 070 | 86,1% | 7,3% | 13,0% | 2,8% | `English` (83) |

## Prossimi esperimenti

1. **Isolare la geometria.** Confrontare P11 invariato con P11 più feature spaziali e attention pooling, partendo dallo stesso checkpoint e usando per entrambi gli stessi prodotti, step ed epoche aggiuntive. Misurare F1, precision, recall, supporto e matrice di confusione sulle coppie sinistra/destra. Aggiungere un confronto con solo pooling se la prima variante migliora.
2. **Misurare il contributo dell’OCR.** Ripetere la variante geometrica con OCR, mantenendo gli altri fattori fissi. Su `prompt-test` final-abt guadagna 0,143 F1 in `text_language` rispetto a P11, ma perde 0,039 in `text_classification`. Monitorare in particolare `Dutch`, `Italian`, `certification_seal` e `other`, con supporti espliciti. Per prove di soglie o pesi di classe usare la validazione interna e verificare poi su un holdout separato dal fit.
3. **Misurare qualità e costo insieme.** Ripetere throughput e picco GPU a batch 1, 8, 16 e 32 con un campione fisso. Nel vision adapter profilare VQ, OCR e forward. Il confronto attuale mostra circa 2,1× di costo di sola inferenza per final-abt rispetto a P12 sulla H200. Misurare prima eventuali ottimizzazioni della pipeline, poi decidere se aumentare ulteriormente le epoche.
4. **Rendere valutabile il full fine-tuning.** Se si vuole investire ancora in P12, rifare una fase di confronto con prodotti esclusi dal fit. Confrontare i checkpoint intermedi sulle stesse classi e sullo stesso holdout usato per la variante LoRA. Questo separa il contributo dell’addestramento completo da quello delle 2.532 immagini aggiuntive viste dal P12 attuale.
