# Bymo × Hack Apertus Zurich — Challenge Galaxus

Guida interna del team: cosa chiede la challenge, come viene valutata e idee concrete per superare
la baseline Qwen 3.5 9B finetunata (dev score **0.917**).

> Fonti: [challenge_description.md](../challenge_description.md), [FAQ.md](../FAQ.md),
> [reference/finetune_qwen35_reference.ipynb](../reference/finetune_qwen35_reference.ipynb).
> In caso di dubbio valgono quelle e le [regole Kaggle](https://www.kaggle.com/competitions/hackapertus-galaxus/rules).

---

## 1. Cosa chiede la challenge

Galaxus ha milioni di immagini prodotto. Oggi le classifica un **Qwen 3.5 9B finetunato**. Dobbiamo costruire un
metodo basato su **Apertus v1.5** (8B o 70B, anche finetunato) che lo possa sostituire: altrettanto accurato, veloce
ed economico per **più di 100.000 immagini al giorno**, e installabile on-premise, air-gapped o in un cloud svizzero.

Per ogni immagine si predice **una classe per ciascuna delle 6 dimensioni** (41 classi in totale):

| Dimensione | Domanda | Classi |
| --- | --- | --- |
| `subject` | Cosa mostra l'immagine? | full_product, detail_closeup, accessories, packaging, package_and_product, multipack_trade_unit, delivery_scope, product_grouping, multiview, energy_label, infographic, logo, placeholder |
| `background` | Cosa c'è dietro il prodotto? | white, color, scene_lifestyle |
| `perspective` | Da dove si vede il prodotto? | front_head_on, front_angle_left, front_angle_right, back_head_on, back_angle_left, back_angle_right, left_side, right_side, top_down, bottom_up |
| `individuals` | C'è una persona o un animale reale? | individual_present, none |
| `text_classification` | Che testo sovrapposto c'è? | certification_seal, technical_specs, promotional_text, nutritional_info, other, none |
| `text_language` | Quale lingua domina il testo? | German, English, French, Italian, Dutch, other, none |

Ogni immagine ha anche un **contesto prodotto** (titolo, tipo, brand, categoria, sottocategoria). In produzione può
mancare o essere scarso: il metodo deve funzionare anche senza.

### Dati (su Kaggle, mai nel repo)

- `train.csv` + immagini: circa 11.900 immagini etichettate. Circa 1/5 delle celle sono **prelabel di un modello**
  (`<dim>_source == "model"`), non verificate da umani. La colonna `<dim>_relevant` dice se la cella viene valutata.
- `dev.csv`: 5.261 immagini senza label → leaderboard pubblica.
- `final_encrypted.bin`: 5.292 immagini, passphrase pubblicata **sabato 28/11 alle 13:00** → leaderboard privata.
- `labeling_guideline.md/.pdf`: definizioni delle classi e convenzioni. **Da leggere tutti.**
- `metric.py`: la metrica ufficiale, da usare in locale.

### Regole sui modelli

- **Apertus v1.5 deve fare la classificazione.**
- **In sviluppo** si può usare qualsiasi cosa, anche modelli chiusi (Gemini, GPT, Claude) per creare target di
  training o **pseudo-label sulle immagini dev** (consentito).
- **A runtime** solo Apertus + modelli **open-weights** per compiti di supporto (image encoder, OCR, embedding).
  Niente API proprietarie.
- **Vietato**: etichettare a mano dev/final, cercare i prodotti nello shop Galaxus o con reverse image search,
  allenarsi o fare tuning sulle immagini final.
- Il train invece è nostro: possiamo pulirlo, anche a mano.

### Valutazione

**Media delle macro F1 delle 6 dimensioni.** Conseguenze pratiche:

- Le **classi rare pesano quanto quelle frequenti**: `certification_seal`, `multiview`, `bottom_up` contano come
  `white` o `none`.
- La F1 è calcolata su tutte le classi presenti nelle label **più ogni classe valida che predici**: predire una
  classe che non compare mai costa.
- Si valutano **solo le label umane** e solo dove la dimensione è significativa:
  - subject `logo` / `placeholder` / `energy_label` → valutato solo subject (e text_language se c'è testo);
  - subject `infographic` → background, perspective, individuals non valutati;
  - subject `delivery_scope` / `detail_closeup` / `product_grouping` → perspective non valutata;
  - background `scene_lifestyle` → subject e perspective non valutati;
  - text_language valutato solo se c'è overlay testuale o subject è infographic, packaging, package_and_product,
    full_product.
- Il file di submission non può avere celle vuote; `none` sempre minuscolo (`None` = cella vuota).

### Come vengono assegnati i punti

| Parte | Peso |
| --- | --- |
| Performance sulle immagini final (a fasce: sopra Qwen 0.917 / tra Gemini 3.8 Flash e Qwen / sotto Gemini Flash) | ~30% |
| Innovazione: quanto andiamo oltre "allenare più a lungo" | 20% |
| Velocità e risorse: immagini/s, memoria, costo per 100k immagini | 20% |
| Insight: evidenze chiare di cosa ha funzionato e cosa no (ablation) | 15% |
| Qualità del codice e praticità | 15% |

**Bonus:** ottimi punteggi di `perspective` e una lista di errori di label in `train.csv`
(csv `image_id,dimension,label,suggested_label`, da mandare in DM su Discord, **mai committare**).

### Baseline da battere (dev)

| Modello | Dev score |
| --- | --- |
| Qwen 3.5 9B finetunato (produzione Galaxus) | **0.917** |
| Gemini 4 Argon zero-shot | 0.834 |
| Gemini 3.8 Flash zero-shot | 0.806 |

Holdout del notebook di riferimento (600 immagini), utile per capire **dove Qwen è debole**:

| Dimensione | Macro F1 | Weighted F1 |
| --- | --- | --- |
| subject | 0.875 | 0.927 |
| background | 0.911 | 0.967 |
| perspective | 0.859 | 0.880 |
| individuals | 0.989 | 0.998 |
| text_classification | 0.842 | 0.970 |
| text_language | 0.916 | 0.940 |
| **overall** | **0.899** | 0.951 |

Classi più deboli: `certification_seal` 0.67, `accessories` 0.69, `top_down` 0.71, `multiview` 0.71,
text `other` 0.75, `multipack_trade_unit` 0.77, `promotional_text` 0.78, `left_side` 0.80.
**Il margine sta tutto in subject, perspective e text_classification, e nelle classi rare.**

### Scadenze (CET)

| Quando | Cosa |
| --- | --- |
| Ven 27/11 dopo la deep dive | Iscrizione Kaggle, **team Kaggle creato prima della prima submission**, download dati, verifica SHA-256 di `final_encrypted.bin` e che `7zz x` chieda la passphrase |
| Durante l'hack | Training, max **20 submission/giorno** (giorno UTC: sabato inizia all'01:00 CET) |
| Sab 28/11 13:00 | Passphrase final. Inferenza a **≥ 2 img/s** (5.292 immagini in ~45 min) |
| Sab 28/11 ~14:15 | Upload su Kaggle e **selezione** della submission finale |
| Sab 28/11 14:30 | Deadline Kaggle + sprintd (repo pubblico, video ≤ 2 min, deck ≤ 10 slide, report PDF ≤ 6 pagine, pesi pubblicati su HF) |
| Sab 28/11 15:00 | Pitch 7 min + 2 min di domande |

---

## 2. Punto di partenza: portare il recipe Qwen su Apertus

Apertus v1.5 è multimodale (`image-text-to-text`, `Apertus1p5ForConditionalGeneration`), quindi il notebook di
riferimento si porta quasi tutto. Elementi da tenere perché valgono punti:

1. **Immagini**: compositing dei PNG trasparenti su bianco (funzione nella pagina dati Kaggle; mai
   `convert("RGB")` nudo), resize a ~512×512 di area mantenendo l'aspect ratio.
2. **Holdout per `product_id`**, non per immagine (altrimenti il punteggio locale è gonfiato).
3. **Loss solo sulle label umane**: le prelabel restano nel JSON ma i loro token sono mascherati a `-100`.
4. **Mirror** delle immagini con perspective left/right **senza testo**, scambiando left↔right nella label.
5. **Ripetizione delle classi rare** in ogni epoca.
6. **LoRA su LM e vision tower**, con LR della vision a 1/4 di quella del LM (2e-4 / 5e-5), rank 32, 3 epoche,
   batch effettivo 16.
7. **Inferenza con vLLM** (il notebook fa ~13–29 img/s su H100).

Questo ci porta vicino a ~0.90. Per andare oltre, e per prendere i punti di innovazione e velocità, servono le idee
sotto.

---

## 3. Idee per battere Qwen 3.5 9B

Ordinate per rapporto impatto/sforzo. Le prime 4 sono quelle su cui puntare.

### 3.1 Ottimizzare direttamente la macro F1 (impatto alto, sforzo basso)

La generazione greedy massimizza l'accuratezza, non la macro F1. Invece di leggere solo il token generato:

- **Leggere i logprob di tutte le classi** per ogni dimensione (scoring delle classi o constrained decoding).
- **Calibrare un bias per classe** sull'holdout per massimizzare la macro F1 (coordinate ascent sui bias di
  ciascuna dimensione). Alza il recall delle classi rare a scapito di quelle frequenti: è esattamente quello che
  premia la metrica.
- **Non predire mai classi che non esistono** nelle label di quella dimensione.
- Sfruttare le regole di rilevanza: nelle celle non valutate la risposta è indifferente, quindi non serve
  ottimizzare lì (ma la cella non deve essere vuota).

### 3.2 Pseudo-label sul dev + distillazione (impatto alto)

Consentito dalle regole. Le 5.261 immagini dev quasi **raddoppiano** i dati:

- Etichettare dev con un **ensemble di insegnanti**: il nostro Apertus finetunato, Apertus 70B, Gemini 4 Argon /
  altri modelli chiusi (permessi in sviluppo) con la labeling guideline nel prompt.
- Tenere solo le pseudo-label **ad alta confidenza o su cui gli insegnanti concordano**, poi riallenare.
- Usare gli stessi insegnanti per aumentare le **classi rare** del train.
- Variante: **soft label** (distribuzione degli insegnanti) invece di hard label, se scriviamo una loss custom.

Attenzione: non si può fare nulla del genere sulle immagini **final**.

### 3.3 Specializzarsi su perspective (bonus + debolezza di Qwen)

- **TTA con flip orizzontale**: predire su immagine e mirror, scambiare left↔right nelle probabilità del mirror e
  fare la media (solo se non c'è testo, che si può stimare dalla stessa predizione di `text_classification`).
- Risoluzione più alta solo per perspective (es. 768) se il budget di velocità lo consente.
- Le convenzioni di perspective nella guideline sono sottili (front_angle_left vs left_side, top_down):
  includere le definizioni nel prompt oppure in un'analisi errori mirata.

### 3.4 OCR + language ID come supporto per i testi (impatto medio-alto, sforzo basso)

Testo e lingua sono difficili da leggere a 512 px per un VLM. Modelli open-weights a runtime sono consentiti:

- **OCR open** (PaddleOCR, docTR, Tesseract) → testo estratto passato nel prompt di Apertus.
- **Language ID** (fastText `lid.176`, CLD3) sul testo OCR → suggerimento per `text_language`.
- Apertus resta il classificatore: riceve l'OCR come contesto aggiuntivo. Aiuta `certification_seal`,
  `promotional_text`, `nutritional_info` e le lingue.
- Costo: OCR leggero è veloce, ma va misurato rispetto al vincolo di ≥ 2 img/s.

### 3.5 Teste di classificazione invece della generazione (velocità + innovazione)

- Estrarre gli hidden state di Apertus (ultimo token o pooling) e allenare **6 teste lineari** con cross-entropy
  pesata per classe (o loss sensibili alla macro F1, es. focal loss / logit adjustment).
- Una sola forward pass, zero decoding → molto più veloce e cheap; le probabilità sono native per la calibrazione
  di 3.1.
- **Da chiedere su Discord** se conta ancora come "Apertus classifica" (la FAQ lo suggerisce esplicitamente per
  design di questo tipo).
- Alternativa più sicura: output ultra-compatto (un token per classe, 6 token totali) invece del JSON.

### 3.6 Robustezza al contesto mancante

- **Context dropout** in training: nel 20–30% degli esempi togliere tutti o parte dei campi prodotto.
- Ablation da riportare: score con e senza contesto (vale come insight e risponde a un requisito esplicito).

### 3.7 Pulizia delle label del train (bonus + qualità)

- **Cross-validation a k fold** sul train: le immagini dove il modello è molto sicuro e diverso dalla label umana
  sono candidate errori.
- Verifica con un modello chiuso (in sviluppo) e, sul train, anche a mano.
- Esportare il csv per il bonus (**fuori dal repo**) e riallenare con le label corrette.

### 3.8 Retrieval come hint (innovazione)

- Un image encoder open (SigLIP 2, DINOv2) indicizza il train; per ogni immagine si recuperano i k vicini e le loro
  label vanno nel prompt di Apertus come esempi.
- Utile per le classi rare e per gli stili ricorrenti dei fornitori. Anche questo va confermato su Discord.

### 3.9 Velocità e costo (20% del punteggio)

- vLLM con **prefix caching** del system prompt, batch grandi, `max_tokens` minimo.
- Meno vision token (provare 384 vs 512 vs 640 e riportare l'ablation velocità/score).
- **Quantizzazione** FP8 / AWQ dei pesi merged dopo il finetune.
- Riportare: img/s, VRAM di picco, **costo per 100k immagini** (es. L40S ~1,80 $/h: a 20 img/s sono ~1,4 h →
  ~2,5 $ per 100k).
- 8B sempre come modello di runtime; 70B solo come insegnante offline.

### Cosa probabilmente **non** conviene

- Allenare più epoche/più a lungo e basta: non dà punti di innovazione e rischia overfitting sulle classi frequenti.
- Apertus 70B a runtime: troppo lento e costoso per ≥ 2 img/s con il nostro budget.
- Prompt lunghissimi dopo il finetune: il modello impara le classi dalle label, il prompt serve solo a inquadrare.

---

## 4. Idee dalla ricerca recente: oltre il semplice fine-tuning

Il 50% del punteggio non dipende dall'accuratezza: innovazione (20%), velocità e costo (20%), insight (15%).
Un fine-tuning ben fatto ci mette solo nella fascia giusta della performance. Per vincere serve **un'architettura
con una storia**: più veloce di Qwen, almeno altrettanto accurata, con ablation chiare.

Premessa importante per scegliere: il nostro compito è **percezione + classificazione**, non ragionamento a
più passi. Le tecniche nate per il ragionamento (looping, test-time compute, RL con chain-of-thought) vanno adattate,
non copiate.

### 4.1 Apertus "troncato" con teste di classificazione ed early exit (raccomandata)

**L'idea.** Diversi lavori mostrano che nei VLM l'informazione per classificare un'immagine è spesso **più leggibile
linearmente dai layer intermedi** che dall'ultimo, e che c'è un divario tra ciò che il modello "sa" internamente e ciò
che genera ([Yu & Lee, COLM 2025](https://arxiv.org/abs/2508.20279);
[arXiv 2604.04411](https://arxiv.org/pdf/2604.04411)).

**Cosa facciamo.**
1. **Probing layer per layer**: estraiamo gli hidden state di ogni layer di Apertus 8B (ultimo token o media sui
   token immagine) e alleniamo 6 sonde lineari per layer. Otteniamo un grafico "macro F1 per layer e per
   dimensione": **è già un insight da report e da pitch**.
2. **Tronchiamo il modello** al layer migliore (es. metà dei layer): stessa accuratezza, metà del compute.
3. **LoRA + teste** sul modello troncato, con cross-entropy pesata o logit adjustment per le classi rare.
4. **Early exit per dimensione**: `individuals` e `background` probabilmente si risolvono presto, `perspective` e i
   testi più tardi. Ogni dimensione esce dal suo layer.

**Perché vince.** Una forward pass sola, niente decoding, metà dei layer → img/s molto più alti di Qwen e costo per
100k immagini più basso. Probabilità native per la calibrazione della macro F1 (3.1). Apertus resta il modello che
classifica: va solo confermato su Discord che le teste su Apertus contano.

### 4.2 Cascata: classificatore veloce + Apertus completo solo sui casi difficili (raccomandata)

**L'idea.** Il 4.1 gestisce tutte le immagini. Solo quelle con **bassa confidenza** (es. il 10–20%, scelto
sull'holdout) passano al modello completo: Apertus generativo finetunato, risoluzione più alta, OCR nel prompt,
TTA con flip.

**Perché vince.** Si spende compute solo dove serve: accuratezza del modello grande, velocità media del modello
piccolo. È la soluzione **più pratica da adottare per Galaxus**, che nei casi in bilico è il criterio decisivo.
Per il report: curva accuratezza/velocità al variare della soglia.

### 4.3 Retrofitting di un looped transformer su Apertus (esperimento ad alto rischio)

**L'idea.** I looped transformer riusano lo stesso blocco di layer più volte (profondità ricorrente).
[McLeish et al., NeurIPS 2025](https://arxiv.org/abs/2511.07384) convertono un LLM pre-addestrato in un modello
ricorrente: tengono i primi layer come *prelude*, un blocco centrale come *recurrent block* da ripetere e gli ultimi
come *coda*, **eliminano i layer in mezzo**, poi aumentano gradualmente il numero di iterazioni durante il training
(curriculum di ricorrenze) dopo una fase di "guarigione". Lavori del 2026 fanno lo stesso con LoRA rank 16 sul solo
blocco ricorrente ([arXiv 2608.11233](https://arxiv.org/pdf/2608.11233)) o con loop adattivi
([arXiv 2609.35748](https://arxiv.org/pdf/2609.35748)).

**Come lo adattiamo.** *Loop-to-shrink*: partiamo dall'Apertus troncato del 4.1, togliamo altri layer e li
sostituiamo con k iterazioni di un blocco centrale condiviso, con una LoRA diversa per iterazione (pochi MB).
Il numero di loop diventa una **manopola a runtime**: 1 loop per le immagini facili, 3–4 per quelle incerte, con
uscita quando le teste sono abbastanza sicure.

**Valutazione onesta.**
- Pro: massima innovazione, meno parametri unici in memoria, compute adattivo per immagine, storia forte per il pitch.
- Contro: i benefici documentati riguardano soprattutto il **ragionamento** (matematica), non la percezione. Un
  confronto del 2026 riporta che le varianti looped restano **meno accurate** del modello standard a parità di
  compute ([arXiv 2609.35748](https://arxiv.org/pdf/2609.35748)). La fase di guarigione richiede dati e GPU che
  in 30 ore non abbiamo in abbondanza.
- **Decisione suggerita**: ramo sperimentale per 1 persona, **solo dopo** che 4.1 funziona. Anche se non batte la
  versione troncata, l'ablation "loop vs no-loop a parità di compute" è un insight valido (15% del punteggio).

### 4.4 Reinforcement fine-tuning con reward allineato alla macro F1 (rischio medio)

**L'idea.** [Visual-RFT (ICCV 2025)](https://arxiv.org/abs/2503.01785) applica GRPO ai VLM con reward verificabili
per la classificazione e riporta grandi guadagni con pochi esempi.

**Come lo adattiamo.** Dopo l'SFT, un breve passaggio GRPO con reward = correttezza per dimensione, **pesata
dall'inverso della frequenza della classe** e azzerata sulle celle non valutate. In questo modo il reward
approssima la macro F1, cosa che la cross-entropy non fa. Si allena solo sui casi su cui l'SFT sbaglia o è incerto.

**Valutazione.** Utile soprattutto per le classi rare; costoso in GPU (più campioni per immagine). Da provare solo
con budget residuo.

### 4.5 Compressione dei token visivi e pruning dei layer (velocità)

- **Meno token visivi**: la letteratura 2026 sul token pruning (SwiftVLM, MVP, DiFiT) mostra che molti token
  visivi sono ridondanti. Le nostre immagini prodotto hanno spesso grandi aree di sfondo bianco: scartare i token
  di sfondo uniforme prima del LLM è semplice e quasi gratis.
- **Pruning dei layer guidato dalla metrica**, sul modello di [GM-Skip](https://arxiv.org/html/2508.18227v1) e
  [INTERLACE](https://arxiv.org/html/2511.19676): togliere i blocchi la cui rimozione peggiora meno la macro F1
  sull'holdout, poi recuperare con LoRA.
- Si combina con 4.1: tronco + token compressi + quantizzazione FP8.

### 4.6 Distillazione da un ensemble di insegnanti (accuratezza)

Estensione del 3.2: Apertus 70B + modelli chiusi (permessi solo in sviluppo) + il nostro 8B producono
**soft label** sul train e sulle immagini dev; lo studente (Apertus troncato) impara la distribuzione, non solo la
classe. È il modo più diretto per portare l'accuratezza di modelli grandi in un modello veloce.

### Architettura che proponiamo di portare in finale

```
immagine ──► compositing bianco + resize ──► Apertus 8B troncato (+ LoRA, token di sfondo scartati)
                                              │
                                              ├─► 6 teste + flip TTA + bias calibrati per la macro F1
                                              │        │
                                              │        ├─ confidenza alta (~85%) ──► risposta
                                              │        └─ confidenza bassa ──┐
                                              │                              ▼
                                              └──────────── Apertus 8B completo + OCR/langID ──► risposta
```

Da confrontare, nel report, con: Qwen 3.5 9B (baseline), Apertus zero-shot, Apertus LoRA generativo (port del
notebook), e opzionalmente la variante looped. Per ogni passaggio: macro F1, weighted F1, img/s, VRAM, costo per
100k immagini.

---

## 5. Piano di lavoro proposto

| Fase | Chi | Output |
| --- | --- | --- |
| 0. Setup | tutti | Kaggle team, dati scaricati, accesso HF ad Apertus (gated), split holdout per prodotto, `metric.py` funzionante |
| 1. Baseline zero-shot | 1 persona | Apertus 8B zero-shot con la guideline nel prompt → primo punteggio e prima submission valida |
| 2. LoRA baseline | 1–2 persone | Port del notebook Qwen su Apertus 8B → punteggio holdout + dev |
| 3. Quick win | in parallelo | Calibrazione bias (3.1), TTA flip (3.3), OCR + langID (3.4) |
| 3b. Ricerca | 1–2 persone | Probing per layer → Apertus troncato con teste (4.1), poi cascata (4.2); ramo looped (4.3) solo se avanza tempo |
| 4. Dati | in parallelo | Pseudo-label dev (3.2), lista errori train (3.7) |
| 5. Retrain + velocità | 1–2 persone | Modello finale, merge, quantizzazione, benchmark img/s |
| 6. Dry run | tutti | Venerdì sera / sabato mattina: run completa sul dev cronometrata, pipeline final pronta |
| 7. Consegna | tutti | Report (sezione 7!), deck, video, pesi su HF, commit referenziato nel report |

**Compute**: 20 $ HF per team + 10 $ HF a testa (Perks su sprintd) + 30 h GPU Kaggle a settimana per account.
Un L40S (48 GB) basta per il LoRA bf16 dell'8B (Qwen ha usato 38 GB di picco); il notebook ha impiegato 73 min su H100.

---

## 6. Struttura del codice (proposta)

```
src/
├── README.md          # questo file
├── config.py          # path, iperparametri, seed
├── data.py            # caricamento csv, compositing su bianco, resize, split per product_id
├── prompts.py         # system prompt e user prompt con contesto opzionale
├── train.py           # LoRA su Apertus v1.5 8B, loss solo label umane, mirror, repeat
├── predict.py         # inferenza vLLM, logprob per classe, TTA
├── probe.py           # probing per layer, troncamento e teste di classificazione (4.1)
├── cascade.py         # routing per confidenza tra modello veloce e completo (4.2)
├── calibrate.py       # bias per classe per massimizzare la macro F1 sull'holdout
├── ocr.py             # OCR + language ID (supporto a runtime)
├── evaluate.py        # wrapper attorno a metric.py, report per classe
└── make_submission.py # submission.csv (dev + final)
```

Variabili d'ambiente richieste dal template (vedi [../.env.example](../.env.example)):
`LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`.

**Mai committare**: immagini, csv, label, passphrase, pesi, la lista errori del train. Il `.gitignore` del template
li esclude già, ma controllare sempre `git status` prima di un commit.
