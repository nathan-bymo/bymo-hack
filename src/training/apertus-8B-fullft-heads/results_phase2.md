# Fine-tuning completo di Apertus 8B + teste: analisi del run su tutti i dati (27 novembre 2026)

Elenco di tutte le prove: [RISULTATI.md](../EXPERIMENTS.md). Numero di prova provvisorio **P12** (P11 è il LoRA + teste
del team, submission delle 21:30): da allineare con RISULTATI.md.

## Scheda della prova P12

| voce | |
|---|---|
| **Domanda** | Quanto fa Apertus 8B con tutti i pesi dei layer allenati (niente LoRA) e 6 teste, su tutti i dati? |
| **Modello** | Apertus 1.5 8B troncato al layer 24; 5,24 B parametri allenati (master fp32, autocast bf16, AdamW 8-bit), embedding e tokenizer VQ congelati; 6 teste su [ultimo token, media dei token VQ] |
| **Training** | **tutte** le 11.876 immagini di `train.csv` (fase 2 di the data split definition, senza fase 1); 16.560 item per epoca con le ripetizioni delle classi rare e i mirror di [DATI.md](../apertus-8B-vadapter-lora-heads/DATI.md); 3 epoche, 3.105 passi, 165 minuti. Focal loss γ = 2 sulle sole celle valutate, alpha per classe `effective`; lr 1e-5 per il backbone e 1e-3 per le teste; batch 16; seed 3407 |
| **Valutazione** | solo il dev su Kaggle (classifica pubblica): il modello ha visto tutte le immagini etichettate, quindi **nessun punteggio locale valido** |
| **Impostazioni** | immagini VQ a 256², nessun SigLIP; prompt = template di chat + contesto prodotto + frase fissa `encoders.TASK`, identico in training e in inferenza |
| **Esecuzione** | Hugging Face Jobs `6b09cdb46c5d61fff189842a`, `a100-large`, 3 h 21 min, circa 8,40 $; codice del commit `93ca561` più checkpoint per epoca e scadenza (commit successivo) |
| **Pesi** | repo privato `StoryofLegend/apertus-8b-galaxus-fullft`, revisione `aac1375d3efb93460a564faaef00a60430d84443`: `phase2/model/{epoch-1,epoch-2,last}` |
| **Fonte dei numeri** | score: submission Kaggle `59021011`; il resto da `phase2/logs/train.log`, `phase2/predictions/dev_submission.{csv,probs.pt,report.json}` e `train.csv`, ricalcolato in locale |
| **Cosa non dice** | nessuna F1 per dimensione o per classe (Kaggle dà un solo numero); un solo training e un solo seed; il confronto con P11 non è a parità di dati |

## Risultato

| modello | dati di training | dev Kaggle (pubblico) |
|---|---|---|
| Qwen 3.5 9B finetunato (organizzatori) | train | 0,917 |
| Gemini 4 Argon zero-shot | nessuno | 0,834 |
| Gemini 3.8 Flash zero-shot | nessuno | 0,806 |
| **P12, fine-tuning completo + teste** | **11.876 immagini** | **0,805** |
| P11, LoRA + teste (team) | 9.344 immagini (`training_rows`) | 0,796 |

- Miglior punteggio Apertus del team finora, **0,0013 sotto Gemini Flash**. La valutazione delle final è a fasce
  (sotto Flash / tra Flash e Qwen / sopra Qwen), e siamo sul confine tra le prime due.
- **+0,009 su P11, ma non a parità di condizioni**: P12 ha il 27% di immagini in più e un altro metodo di
  adattamento. Il guadagno non si può attribuire al fine-tuning completo: con questi dati il fine-tuning completo
  non si stacca dalla LoRA + teste, mentre costa una GPU da 80 GB e pesi da 10,5 GB invece di un adapter.

## Cosa è andato bene

| punto | evidenza |
|---|---|
| Run stabile e senza sprechi | nessuna loss non finita; memoria 57,1 GB su 80 per tutto il run (stima 60); 3 epoche intere, dentro il budget calcolato al passo 30 (166 min stimati, 165 effettivi) |
| Protezione dai timeout | checkpoint salvati e caricati a fine epoca 1 e 2 (19 s per il salvataggio, upload in background senza fermare il training) |
| Costo | circa 8,40 $ per training + predizione del dev su `a100-large`, la GPU più economica con almeno 80 GB |
| Velocità di inferenza | **9,55 img/s** end to end su A100 (VQ incluso), 18,7 GB di memoria: le 5.292 final in circa 9 minuti, quasi 5 volte il minimo di 2 img/s. Circa **7,30 $ per 100.000 immagini** (2,9 h di A100 a 2,50 $/h) |
| Output sempre valido | 5.261 righe, nessuna cella vuota, nessuna classe inventata: le teste non possono generare testo sbagliato (il LoRA P10 aveva 21 risposte JSON non valide su 2.532) |
| Distribuzione delle classi credibile | per subject, background, individuals e text_classification la distribuzione predetta sul dev è vicina a quella delle celle valutate del train (`full_product` 58,8% contro 57,3%, `white` 84,8% contro 82,7%, `individual_present` 5,4% contro 6,5%, classi rare presenti in proporzioni simili) |
| Classi facili molto sicure | probabilità media della classe scelta: `energy_label` 1,00, `placeholder` 0,97, `logo` 0,96, `full_product` 0,90 |
| Coerenza tra dimensioni | le combinazioni che sembravano strane sono nelle stesse proporzioni delle label umane: infografiche senza testo aggiunto 32% (train 39,6%), testo aggiunto con lingua `none` 20% (train 27,2%) |

## Cosa si può migliorare

### 1. Perspective: il modello vede l'angolo ma non il lato

La loss dell'epoca 1, quando ogni immagine è vista per la prima volta, è la più alta di tutte: 1,02, contro 0,57 di
subject. Sul dev il 45% delle immagini ha la classe scelta sotto 0,6 di probabilità.

| classe predetta | n | probabilità media | p media del lato opposto |
|---|---|---|---|
| `front_head_on` | 2.310 | 0,80 | – |
| `back_head_on` | 516 | 0,85 | – |
| `front_angle_left` | 957 | 0,51 | **0,40** (`front_angle_right`) |
| `front_angle_right` | 440 | 0,47 | **0,38** (`front_angle_left`) |
| `left_side` / `right_side` | 198 / 119 | 0,51 / 0,49 | 0,41 / 0,41 |

- Le viste frontali e posteriori sono sicure. Sulle viste angolate e laterali il modello mette quasi la stessa
  probabilità sui due lati: riconosce "angolato" ma non "a sinistra" o "a destra".
- La massa di probabilità totale è quasi pari (772 contro 705 per `front_angle_*`), ma l'argmax pende a sinistra:
  sulle immagini dove perspective conta, `front_angle_left` 18,4% e `front_angle_right` 8,3%, contro 13,9% e 12,9%
  nel train. Non è uno scambio sistematico delle label (il mirror di `data.py` scambia correttamente), è
  un'incertezza che si risolve sempre dallo stesso lato.
- **Intervento**: flip TTA solo per perspective. Si predice anche l'immagine specchiata, si scambiano le probabilità
  left↔right e si fa la media. La decisione diventa simmetrica per costruzione e la preferenza per la sinistra si
  annulla. Costo: una seconda passata (circa 5 img/s, sempre sopra i 2 richiesti), nessun training.

### 2. text_language: `Dutch` predetto troppo spesso

- 108 immagini predette `Dutch`, con probabilità media 0,47 e il 75% sotto 0,5. Per 90 di queste la seconda
  scelta è `German`. Nel train le celle valutate `Dutch` sono 21 su 5.135 (0,4%): sul dev ce ne si aspetta una
  ventina, non 108.
- Causa probabile: peso `effective` 6,68 di `Dutch` (il più alto) e oversampling fino a 4× su 21 esempi.
- Con la macro F1 ogni falso positivo su una classe rara costa molto, e qui toglie anche recall a `German`.
- **Intervento**: soglia o bias per classe sulle probabilità già salvate (`dev_submission.probs.pt`), per esempio
  `Dutch` solo sopra 0,6. Si prova in pochi minuti, senza GPU, con una submission. `Italian` (24, probabilità
  0,88) e `French` (104, 0,86) sembrano sani.
- La loss dell'epoca 1 di text_language (0,97) è la seconda più alta. Il 41% delle celle rilevanti di
  text_language (3.541 su 8.676) sono prelabel non controllate e restano fuori dalla loss: la supervisione è scarsa.

### 3. Nessuna misura locale

Saltare la fase 1 ha risparmiato circa 7 $, ma lascia senza:

- F1 per dimensione e per classe, quindi senza la conferma dei punti 1 e 2;
- una scelta dell'epoca: nelle epoche 2 e 3 la loss scende da 0,97 a 0,08 (text_language) e da 1,02 a 0,18
  (perspective), cioè per lo più memorizzazione. Non sappiamo se `epoch-1` o `epoch-2` andrebbero meglio di `last`;
- un set su cui calibrare bias e soglie in modo valido.

Opzioni, dalla più economica:

| intervento | costo | cosa dà |
|---|---|---|
| soglia su `Dutch` dalle probabilità salvate | 0 $, 1 submission | verifica del punto 2 sul dev |
| predizione del dev con `epoch-1` ed `epoch-2` | circa 1,50 $, 2 submission | scelta dell'epoca sul dev |
| flip TTA per perspective sul dev | circa 0,70 $, 1 submission | verifica del punto 1 |
| fase 1 (training su `training_rows`, valutazione su `coverage`) | circa 7 $, 2,7 h | F1 per classe valide, confronto pulito con P10/P11, set per calibrare |

Le prime tre usano il dev pubblico per scegliere tra poche alternative: è ammesso, ma con al massimo 3 o 4 scelte
per non adattarsi alla classifica pubblica.

### 4. Limiti strutturali (il distacco da Qwen, 0,11)

- **Risoluzione e token discreti** (ipotesi). A 256² Apertus vede 256 token VQ discreti, mentre Qwen usa un encoder
  visivo continuo a 512². I dettagli fini, l'orientamento e il testo piccolo nell'immagine ne soffrono: sono
  proprio perspective e text_language, le due dimensioni più deboli qui. 512² darebbe 1.024 token, circa 4 volte il
  costo del training: non sta nel credito rimasto per un fine-tuning completo.
- **Fine-tuning completo contro LoRA**: a parità di dati non abbiamo ancora un confronto. Visto il costo, la LoRA +
  teste è probabilmente il compromesso migliore per le prossime prove.
- **Velocità**: xIELU gira in Python (manca il kernel CUDA) e il VQ codifica un'immagine alla volta. Sono margini
  di velocità, non di qualità.

## Riproduzione

```sh
python src/training/apertus-8B-fullft-heads/build_hf_job.py
hf jobs uv run --flavor a100-large --timeout 420m --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN \
    -e PHASE=2 -e EPOCHS=3 -e JOB_TIMEOUT_MINUTES=420 -e UPLOAD_REPO=<utente>/apertus-8b-galaxus-fullft \
    data/hf_jobs/hf_job_fullft.py
```

Le predizioni e le probabilità contengono gli ID delle immagini: restano nel repo privato e in `data/`, mai in git.
