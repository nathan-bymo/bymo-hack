# Bymo × Hack Apertus Zurich — Challenge Galaxus

Guida interna del team: cosa chiede la challenge, come viene valutata e quali sono le opzioni più forti per
superare nettamente la baseline Qwen 3.5 9B finetunata.

> Fonti: [challenge_description.md](../challenge_description.md), [FAQ.md](../FAQ.md),
> [reference/finetune_qwen35_reference.ipynb](../reference/finetune_qwen35_reference.ipynb), deep dive degli
> organizzatori del 27/11. In caso di dubbio valgono i documenti ufficiali e le
> [regole Kaggle](https://www.kaggle.com/competitions/hackapertus-galaxus/rules).

**Obiettivo:** Qwen 3.5 9B fa **0.917** sul dev secondo la challenge description; nella deep dive gli organizzatori
hanno parlato di **0.910**. Puntiamo a **≥ 0.93** sul dev per stare chiaramente sopra la fascia di Qwen, perché
differenze di pochi decimi di punto percentuale non contano.

**In breve: le 5 opzioni più forti** (dettagli nella [sezione 4](#4-le-opzioni-più-forti-per-battere-qwen)):
1. **Adapter visivo continuo** (SigLIP 2 / DINOv3 → projector → Apertus): colma il punto debole di Apertus, che
   vede le immagini come token discreti.
2. **Distillazione dal 70B all'8B**: stesso tokenizer, si distillano le distribuzioni sulle classi, anche sulle
   immagini dev non etichettate.
3. **Training e decisione orientati alla macro F1**: logit adjustment, sampling bilanciato, bias per classe
   calibrati sull'holdout.
4. **Pipeline testo**: OCR + language ID + crop ad alta risoluzione delle zone di testo.
5. **Ensemble a costo zero**: media dei pesi di più LoRA (soup) + TTA con flip per perspective.

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

### 1.1 Note dalla deep dive degli organizzatori (27/11)

**Le label, in sintesi:**
- **Subject**: cosa mostra l'immagine (13 classi, vedi tabella).
- **Background**: cosa c'è dietro il prodotto (white, color, scene_lifestyle).
- **Perspective**: da quale angolo si vede il prodotto (10 classi).
- **Individuals**: se c'è una persona o un animale reale (none, individual_present).
- **Text classification**: che tipo di testo c'è nell'immagine (nutritional_info, technical_specs, promotional_text,
  certification_seal, other). Attenzione: secondo la challenge description esiste anche **`none`**, che viene valutata
  come le altre classi.
- **Text language**: classificata solo se c'è testo sovrapposto o se il subject è infographic, packaging,
  package_and_product o full_product.

**I dati:**
- **22.429 immagini** in totale (≈ 11.876 train + 5.261 dev + 5.292 final), **14.118 prodotti**,
  **1.619 tipi di prodotto**, **41 classi in 6 dimensioni**.
- Le immagini sono state **pre-classificate con un LLM**, poi **revisionate e allineate da persone**. Le label umane
  sono quindi abbastanza affidabili: pulire le label conta meno che bilanciare le classi, anche se la lista errori
  vale comunque un bonus.
- **Forte sbilanciamento** su alcune classi → servono metodi di bilanciamento (sezione 4.3).
- Con 14.118 prodotti su 22.429 immagini, molti prodotti hanno più immagini: l'holdout va **sempre** separato per
  `product_id`.

**Idea del team (Nathan):** distillare il **70B** (eventualmente finetunato) nell'**8B**. È sviluppata nella
[sezione 4.2](#42-distillazione-dal-70b-all8b-idea-del-team).

### 1.2 Dati (su Kaggle, mai nel repo)

- `train.csv` + immagini: ≈ 11.900 immagini etichettate. Circa 1/5 delle celle sono **prelabel di un modello**
  (`<dim>_source == "model"`), non verificate da umani, per lo più in celle non valutate. La colonna
  `<dim>_relevant` dice se la cella viene valutata.
- `dev.csv`: 5.261 immagini senza label → leaderboard pubblica.
- `final_encrypted.bin`: 5.292 immagini, passphrase pubblicata **sabato 28/11 alle 13:00** → leaderboard privata.
- `labeling_guideline.md/.pdf`: definizioni delle classi e convenzioni. **Da leggere tutti.**
- `metric.py`: la metrica ufficiale, da usare in locale.

### 1.3 Regole sui modelli

- **Apertus v1.5 deve fare la classificazione.**
- **In sviluppo** si può usare qualsiasi cosa, anche modelli chiusi (Gemini, GPT, Claude) per creare target di
  training o **pseudo-label sulle immagini dev** (consentito).
- **A runtime** solo Apertus + modelli **open-weights** per compiti di supporto (image encoder, OCR, embedding).
  Niente API proprietarie.
- **Vietato**: etichettare a mano dev/final, cercare i prodotti nello shop Galaxus o con reverse image search,
  allenarsi o fare tuning sulle immagini final.
- Il train invece è nostro: possiamo pulirlo, anche a mano.

### 1.4 Valutazione

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

### 1.5 Come vengono assegnati i punti

| Parte | Peso |
| --- | --- |
| Performance sulle immagini final (a fasce: sopra Qwen / tra Gemini 3.8 Flash e Qwen / sotto Gemini Flash) | ~30% |
| Innovazione: quanto andiamo oltre "allenare più a lungo" | 20% |
| Velocità e risorse: immagini/s, memoria, costo per 100k immagini | 20% |
| Insight: evidenze chiare di cosa ha funzionato e cosa no (ablation) | 15% |
| Qualità del codice e praticità | 15% |

Guardano anche la **weighted F1** e i punteggi per dimensione, perché le classi frequenti sono la maggior parte delle
immagini dello shop. **Bonus:** ottimi punteggi di `perspective` e una lista di errori di label in `train.csv`
(csv `image_id,dimension,label,suggested_label`, da mandare in DM su Discord, **mai committare**).

### 1.6 Baseline da battere (dev)

| Modello | Dev score |
| --- | --- |
| Qwen 3.5 9B finetunato (produzione Galaxus) | **0.917** (0.910 secondo la deep dive) |
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
**Il margine sta in subject, perspective e text_classification, e nelle classi rare.**

### 1.7 Scadenze (CET)

| Quando | Cosa |
| --- | --- |
| Ven 27/11 dopo la deep dive | Iscrizione Kaggle, **team Kaggle creato prima della prima submission**, download dati, verifica SHA-256 di `final_encrypted.bin` e che `7zz x` chieda la passphrase |
| Durante l'hack | Training, max **20 submission/giorno** (giorno UTC: sabato inizia all'01:00 CET) |
| Sab 28/11 13:00 | Passphrase final. Inferenza a **≥ 2 img/s** (5.292 immagini in ~45 min) |
| Sab 28/11 ~14:15 | Upload su Kaggle e **selezione** della submission finale |
| Sab 28/11 14:30 | Deadline Kaggle + sprintd (repo pubblico, video ≤ 2 min, deck ≤ 10 slide, report PDF ≤ 6 pagine, pesi pubblicati su HF) |
| Sab 28/11 15:00 | Pitch 7 min + 2 min di domande |

---

## 2. Come Apertus 1.5 vede le immagini (fondamentale per le scelte)

Dalla `config.json`, dal `processor_config.json` e dal
[technical report di Apertus 1.5](https://arxiv.org/abs/2610.15542):

- **Nessun vision encoder continuo.** A differenza di Qwen (ViT + projector), in Apertus 1.5 *"la multimodalità
  entra dal tokenizer"*: un **tokenizer visivo VQ** (codebook di 131.072 codici) trasforma l'immagine in **token
  discreti**, che entrano nel LLM come normali token del vocabolario. L'architettura del LLM è quella di Apertus 1.0
  (32 layer, hidden 4096, attivazione xIELU) e la testa di output produce solo testo.
- **16 pixel per token.** Immagini tra 256×256 (**256 token**) e ~1,96 MP. A 512×512 sono **1.024 token**, contro i
  256 di Qwen alla stessa risoluzione. **A parità di risoluzione Apertus è circa 4 volte più costoso nel prefill.**
- **La percezione è il punto debole.** Il report stesso dice che sulla percezione i modelli visivi specializzati
  restano avanti; la letteratura sui tokenizer discreti mostra un divario ampio soprattutto su **testo nell'immagine
  e dettagli fini** ([ViQ](https://arxiv.org/html/2606.27313);
  [Kelix](https://arxiv.org/html/2602.09843v1): OCRBench ~70 per i modelli a token discreti contro 85+ dei VLM con
  encoder continuo).
- **Thinking è un flag del chat template**: va disattivato per la classificazione, come nel notebook di Qwen.

**Conseguenze per noi:**
1. La parte "LoRA sulla vision tower" del recipe Qwen **non si porta**: il tokenizer VQ è discreto e i gradienti non
   lo attraversano. L'equivalente è allenare le **righe di embedding dei token immagine** nel LLM (sono parametri
   normali del modello) oppure aggiungere un encoder continuo (4.1).
2. Le dimensioni a rischio sono proprio quelle dove Qwen è già debole: **text_classification, text_language,
   perspective, dettagli di subject**. Lì serve aiuto esterno (OCR, encoder continuo).
3. La **risoluzione** è un compromesso forte tra velocità e accuratezza: va misurato (256² / 384² / 512²) e
   riportato come ablation.

---

## 3. Punto di partenza: portare il recipe Qwen su Apertus

Il notebook di riferimento si porta quasi tutto. Elementi da tenere perché valgono punti:

1. **Immagini**: compositing dei PNG trasparenti su bianco (funzione nella pagina dati Kaggle; mai
   `convert("RGB")` nudo), resize mantenendo l'aspect ratio, lati multipli di 16.
2. **Holdout per `product_id`**, non per immagine (altrimenti il punteggio locale è gonfiato).
3. **Loss solo sulle label umane**: le prelabel restano nel JSON ma i loro token sono mascherati a `-100`.
4. **Mirror** delle immagini con perspective left/right **senza testo**, scambiando left↔right nella label.
5. **Ripetizione delle classi rare** in ogni epoca.
6. **LoRA sul LLM** (rank 32, LR 2e-4, 3 epoche, batch effettivo 16). Al posto della LoRA sulla vision tower:
   provare a sbloccare gli **embedding dei token immagine** con LR bassa.
7. **Inferenza con vLLM** (Qwen faceva ~13–29 img/s su H100). Verificare subito che vLLM supporti
   `Apertus1p5ForConditionalGeneration`; in caso contrario serve un piano B con transformers in batch.

Questo dovrebbe portarci nella zona di Qwen. Per superarlo nettamente servono le opzioni della sezione 4.

---

## 4. Le opzioni più forti per battere Qwen

Ordinate per impatto atteso sul punteggio finale (accuratezza + innovazione + velocità). Le prime tre sono il
nucleo della nostra soluzione.

### 4.1 Adapter visivo continuo: dare ad Apertus "occhi migliori" (impatto massimo)

**Perché.** Il divario più grande tra Apertus e Qwen sta nella percezione (sezione 2). Le regole consentono
esplicitamente **image encoder open-weights come supporto**, mentre la classificazione resta di Apertus.

**Come.** È un'architettura in stile LLaVA costruita sopra Apertus:
1. Un encoder continuo congelato estrae le feature dell'immagine. Candidati:
   - **SigLIP 2** (Apache-2.0): semantica allineata al testo, variante NaFlex per aspect ratio nativi;
   - **DINOv3**: feature geometriche forti, potenzialmente le migliori per **perspective** (verificare la licenza).
   Si possono anche concatenare.
2. Un **projector MLP** a 2 layer mappa le feature (es. 256 token dopo pooling 2×2) nello spazio degli embedding di
   Apertus. Questi token vengono inseriti nel prompt come "soft token", **insieme** ai token VQ a bassa risoluzione
   (256² = 256 token) oppure al loro posto.
3. Training in due fasi: (a) solo projector con Apertus congelato, veloce; (b) projector + LoRA, end-to-end sul
   nostro task.

**Perché può vincere.** Attacca direttamente il punto debole di Apertus, permette di **abbassare la risoluzione VQ**
(meno token → più veloce) e racconta bene la storia di innovazione: *"abbiamo reso Apertus un VLM di percezione
migliore con un encoder open svizzero-compatibile"*.

**Rischi.** Il projector parte da zero con ~12k immagini etichettate (+ dev pseudo-etichettato): per un output a
classi fisse è sufficiente, ma va verificato presto con un esperimento piccolo. **Confermare su Discord** che
"image encoder + projector + Apertus che classifica" rientra nelle regole: la FAQ cita l'image encoder come esempio
consentito.

**Ablation da riportare:** solo VQ vs solo encoder vs entrambi; SigLIP 2 vs DINOv3 vs concatenati; macro F1 per
dimensione e img/s.

### 4.2 Distillazione dal 70B all'8B (idea del team)

**Perché.** Il 70B ha più capacità percettiva e di conoscenza, ma è troppo lento per il runtime (≥ 2 img/s, costo per
100k immagini). La distillazione porta la sua accuratezza in un modello veloce. **8B e 70B condividono il tokenizer**
(testo e VQ visivo, da verificare sul `config.json` del 70B), quindi si possono distillare **direttamente le
distribuzioni di probabilità** sui token delle classi, non solo le risposte.

**Come.**
1. **Insegnante.** Tre livelli, dal più economico al più forte:
   - 70B zero/few-shot con la labeling guideline, tramite l'API di inferenza di Hack Apertus (se espone il 70B e i
     logprob) o self-hosted;
   - **70B finetunato con QLoRA** sul train: pesi a 4 bit ≈ 40 GB, entra su una H100 80 GB. A 256 token immagine le
     sequenze sono corte (~400 token). Stima grezza: **3–5 ore per epoca** su H100 → 1 epoca è nel budget del team
     se uniamo i coupon;
   - ensemble di insegnanti (70B finetunato + più 8B finetunati con seed diversi + modelli chiusi in sviluppo).
2. **Target soft.** L'insegnante produce, per ogni immagine e dimensione, la distribuzione sulle classi.
   Dove usarla:
   - **immagini dev** (non etichettate, ma dello stesso dominio delle final): il guadagno più grande, perché il
     70B finetunato non le ha viste;
   - **celle prelabel del train** (~1/5): le sostituiamo con le distribuzioni del 70B invece di scartarle;
   - **non** sulle celle umane del train su cui il 70B è stato allenato (le avrebbe memorizzate): lì resta la label
     umana.
3. **Studente 8B.** Loss = cross-entropy sulle label umane + **KL con temperatura** (T ≈ 2) verso l'insegnante,
   pesata con un coefficiente α. Lo studente può essere l'8B generativo, quello con adapter (4.1) o quello troncato
   (5.1).

**Varianti da valutare.**
- **Self-distillation** (born-again): se il 70B costa troppo, si distilla un ensemble di 8B in un singolo 8B. Più
  economico, guadagno di solito più piccolo ma affidabile.
- **Filtro di confidenza** sulle pseudo-label dev: tenere solo le immagini dove insegnanti diversi concordano.

**Ablation da riportare:** studente senza distillazione vs con soft label del 70B zero-shot vs del 70B finetunato;
quante immagini dev servono. Ottimo materiale per la sezione insight.

### 4.3 Training e decisione orientati alla macro F1 (sbilanciamento)

Gli organizzatori confermano un forte sbilanciamento: con la macro F1 le classi rare pesano quanto le frequenti.
Tre leve, da combinare:

1. **Durante il training**
   - **Logit-adjusted loss / Balanced Softmax**: si aggiunge alla logit di ogni classe il logaritmo della sua
     frequenza a priori, così la loss smette di favorire le classi frequenti. La versione generalizzata ha un
     parametro β (0 = softmax normale, 1 = Balanced Softmax) da scegliere sull'holdout
     ([Generalised Balanced Softmax](https://arxiv.org/pdf/2607.22258)).
   - **Sampling bilanciato**: ripetizione delle classi rare (come il notebook) + mirror per perspective.
   - Nel modello generativo: **peso per classe** sui token di risposta.
2. **Training disaccoppiato** (con le teste di classificazione, 5.1): prima si allena la rappresentazione con il
   sampling naturale, poi si ri-allenano **solo le teste** con sampling bilanciato per classe. È economico e di solito
   migliora molto le classi rare.
3. **Al momento della decisione**
   - Leggere i **logprob di tutte le classi** (scoring delle classi o constrained decoding).
   - **Bias per classe calibrati sull'holdout** per massimizzare la macro F1 (coordinate ascent per dimensione).
   - **Non predire mai classi che non esistono** nelle label di quella dimensione.
   - Le celle non valutate (regole di rilevanza) non vanno ottimizzate, ma non devono restare vuote.

Attenzione all'altro lato: guardano anche la **weighted F1**, quindi non bisogna sacrificare troppo le classi
frequenti. I bias vanno scelti guardando entrambe.

### 4.4 Pipeline testo: OCR + language ID + crop ad alta risoluzione

Con i token VQ il testo è la cosa più difficile per Apertus (sezione 2). Modelli open-weights a runtime sono
consentiti:

- **OCR open** (PaddleOCR, Apache-2.0; docTR) → testo estratto e posizione dei box nel prompt di Apertus.
- **Language ID** (fastText `lid.176`) sul testo OCR → suggerimento per `text_language`.
- **Crop ad alta risoluzione** delle zone di testo trovate dall'OCR, passati come seconda immagine: Apertus legge
  il dettaglio senza pagare l'alta risoluzione sull'intera immagine.
- Apertus resta il classificatore: OCR e lingua sono contesto in più. Aiuta `certification_seal`,
  `promotional_text`, `nutritional_info`, `other` e le lingue.
- Costo: l'OCR leggero è veloce, ma va misurato rispetto al vincolo di ≥ 2 img/s. Si può lanciare solo quando un
  rilevatore veloce dice che c'è testo.

### 4.5 Ensemble a costo zero e TTA

- **Model soup di LoRA**: allenare 2–3 LoRA con seed o mix di dati diversi e **fare la media dei pesi**. Il guadagno
  è simile a un ensemble, ma il costo di inferenza resta quello di un solo modello.
- **TTA con flip per perspective**: predire su immagine e mirror, scambiare left↔right nelle probabilità del mirror e
  fare la media (solo senza testo). È la leva diretta per il bonus perspective.
- **TTA multi-risoluzione** solo nella cascata (5.2), sui casi incerti.

---

## 5. Altre idee dalla ricerca (velocità e innovazione)

Il 50% del punteggio non dipende dall'accuratezza: innovazione (20%), velocità e costo (20%), insight (15%).
Premessa: il nostro compito è **percezione + classificazione**, non ragionamento a più passi. Le tecniche nate per il
ragionamento (looping, test-time compute, RL con chain-of-thought) vanno adattate, non copiate.

### 5.1 Apertus troncato con teste di classificazione ed early exit (raccomandata)

**L'idea.** Nei VLM l'informazione per classificare è spesso **più leggibile linearmente dai layer intermedi** che
dall'ultimo, e c'è un divario tra ciò che il modello "sa" e ciò che genera
([Yu & Lee, COLM 2025](https://arxiv.org/abs/2508.20279); [arXiv 2604.04411](https://arxiv.org/pdf/2604.04411)).

**Cosa facciamo.**
1. **Probing layer per layer** sui 32 layer di Apertus 8B (ultimo token o media sui token immagine): grafico "macro
   F1 per layer e per dimensione". **È già un insight da report e da pitch.**
2. **Troncamento** al layer migliore: stessa accuratezza, meno compute.
3. **LoRA + 6 teste** sul modello troncato, con logit adjustment e training disaccoppiato (4.3).
4. **Early exit per dimensione**: `individuals` e `background` probabilmente si risolvono presto, `perspective` e i
   testi più tardi.

**Perché vince.** Una sola forward pass, niente decoding, meno layer → molti più img/s e costo per 100k più basso.
Probabilità native per la calibrazione. Da confermare su Discord che le teste su Apertus contano come "Apertus
classifica"; alternativa più sicura: output di 6 token (un token per classe) invece del JSON.

### 5.2 Cascata: classificatore veloce + modello completo sui casi difficili (raccomandata)

Il modello veloce (5.1) gestisce tutte le immagini. Solo quelle con **bassa confidenza** (es. 10–20%, soglia scelta
sull'holdout) passano al modello completo: Apertus con adapter, risoluzione più alta, OCR, TTA. Accuratezza del
modello grande, velocità media del piccolo: è la soluzione **più pratica da adottare per Galaxus**. Per il report:
curva accuratezza/velocità al variare della soglia.

### 5.3 Retrofitting di un looped transformer (esperimento ad alto rischio)

**L'idea.** Un looped transformer riusa lo stesso blocco di layer più volte.
[McLeish et al., NeurIPS 2025](https://arxiv.org/abs/2511.07384) convertono un LLM pre-addestrato in un modello
ricorrente: primi layer come *prelude*, un blocco centrale ripetuto, ultimi layer come *coda*, **eliminando i layer
in mezzo**. Poi aumentano gradualmente le iterazioni durante il training (curriculum di ricorrenze) dopo una fase di
"guarigione". Lavori del 2026 usano LoRA rank 16 sul solo blocco ricorrente
([arXiv 2608.11233](https://arxiv.org/pdf/2608.11233)) o loop adattivi
([arXiv 2609.35748](https://arxiv.org/pdf/2609.35748)).

**Come lo adattiamo.** *Loop-to-shrink*: dall'Apertus troncato (5.1) togliamo altri layer e li sostituiamo con k
iterazioni di un blocco condiviso, con una LoRA diversa per iterazione. Il numero di loop diventa una **manopola a
runtime**: 1 loop per le immagini facili, 3–4 per quelle incerte.

**Valutazione onesta.** Massima innovazione e compute adattivo, ma i benefici documentati riguardano soprattutto il
**ragionamento**, e un confronto del 2026 trova le varianti looped **meno accurate** del modello standard a parità di
compute. **Solo come ramo sperimentale dopo 5.1**: anche un risultato negativo, ben misurato, vale come insight.

### 5.4 Reinforcement fine-tuning con reward allineato alla macro F1 (rischio medio)

[Visual-RFT (ICCV 2025)](https://arxiv.org/abs/2503.01785) applica GRPO ai VLM con reward verificabili per la
classificazione. Adattamento: dopo l'SFT, un breve passaggio GRPO con reward = correttezza per dimensione,
**pesata dall'inverso della frequenza della classe** e azzerata sulle celle non valutate, in modo da approssimare la
macro F1. Costoso in GPU: solo con budget residuo.

### 5.5 Compressione dei token visivi e pruning dei layer

- **Scartare i token VQ dello sfondo uniforme**: le immagini prodotto hanno spesso grandi aree bianche. Con 1.024
  token a 512², la riduzione può essere grande e quasi gratis (letteratura 2026 sul token pruning: SwiftVLM, MVP,
  DiFiT).
- **Pruning dei layer guidato dalla metrica** ([GM-Skip](https://arxiv.org/html/2508.18227v1),
  [INTERLACE](https://arxiv.org/html/2511.19676)): togliere i blocchi la cui rimozione peggiora meno la macro F1,
  poi recuperare con LoRA.
- **Quantizzazione** FP8 / AWQ dei pesi merged.

### 5.6 Altre leve pratiche

- **Context dropout**: nel 20–30% degli esempi togliere tutti o parte dei campi prodotto (requisito esplicito di
  robustezza). Ablation con e senza contesto.
- **Lista errori del train** (bonus): cross-validation a k fold, le immagini dove il modello è molto sicuro e in
  disaccordo con la label umana sono candidate. Verifica con un modello chiuso o a mano (sul train è consentito).
  Csv **fuori dal repo**.
- **Retrieval come hint**: un encoder open indicizza il train, i k vicini e le loro label vanno nel prompt. Da
  confermare su Discord.
- **Velocità**: vLLM con prefix caching del system prompt, batch grandi, `max_tokens` minimo. Riportare img/s, VRAM
  di picco e **costo per 100k immagini** (es. L40S ~1,80 $/h: a 20 img/s sono ~1,4 h → ~2,5 $ per 100k).

### Cosa probabilmente non conviene

- Allenare più epoche e basta: nessun punto di innovazione, rischio di overfitting sulle classi frequenti.
- Apertus 70B a runtime: troppo lento e costoso per ≥ 2 img/s. Il 70B serve come **insegnante** (4.2).
- Alzare la risoluzione VQ su tutta l'immagine per leggere il testo: il costo cresce col quadrato; meglio i crop
  mirati (4.4) o l'encoder continuo (4.1).

---

## 6. Architettura che proponiamo di portare in finale

```
                         ┌──────────────── SigLIP 2 / DINOv3 (congelato) ──► projector ──┐
immagine ──► bianco + ───┤                                                               ▼
             resize      └──► token VQ a 256² ────────────────────────────► Apertus 8B studente
                                                                            (LoRA, distillato dal 70B,
                                                                             troncato + 6 teste)
                                                                                     │
                                         bias calibrati per macro F1 + flip TTA ◄────┘
                                                        │
                              confidenza alta (~85%) ───┴─── confidenza bassa
                                      │                            │
                                   risposta        OCR + langID + crop testo ──► Apertus completo ──► risposta
```

Da confrontare nel report con: Qwen 3.5 9B (baseline), Apertus zero-shot, Apertus LoRA generativo (port del
notebook), poi ogni componente aggiunto uno alla volta. Per ogni passaggio: macro F1, weighted F1, F1 per dimensione,
img/s, VRAM, costo per 100k immagini.

---

## 7. Piano di lavoro proposto

| Fase | Chi | Output |
| --- | --- | --- |
| 0. Setup | tutti | Kaggle team, dati scaricati, accesso HF ad Apertus 8B e 70B (gated), split holdout per prodotto, `metric.py` funzionante, verifica supporto vLLM |
| 1. Baseline | 1 persona | Apertus 8B zero-shot con la guideline nel prompt → prima submission valida; test 256² vs 512² |
| 2. LoRA baseline | 1 persona | Port del notebook su Apertus 8B → punteggio holdout + dev |
| 3. Adapter visivo (4.1) | 1 persona | Esperimento piccolo SigLIP 2 → projector → Apertus; se funziona, diventa il modello principale |
| 4. Insegnante 70B (4.2) | 1 persona | 70B zero-shot subito (pseudo-label dev), QLoRA 1 epoca se il budget lo consente, soft label su dev e prelabel |
| 5. Quick win | in parallelo | Bias calibrati (4.3), flip TTA (4.5), OCR + langID (4.4) |
| 6. Studente finale | 1–2 persone | 8B con adapter + distillazione + logit adjustment; soup di 2–3 LoRA; teste/troncamento (5.1) |
| 7. Velocità | 1 persona | Merge, quantizzazione, cascata (5.2), benchmark img/s e costo |
| 8. Dry run | tutti | Venerdì sera / sabato mattina: run completa sul dev cronometrata, pipeline final pronta |
| 9. Consegna | tutti | Report (sezione 7!), deck, video, pesi su HF, commit referenziato nel report |

**Compute**: 20 $ HF per team + 10 $ HF a testa (Perks su sprintd) + 30 h GPU Kaggle a settimana per account.
Un L40S (48 GB, ~1,80 $/h) basta per il LoRA bf16 dell'8B. Il QLoRA del 70B richiede una H100 80 GB (prezzo HF da
verificare): è la voce di costo più grande, quindi si decide dopo aver visto il 70B zero-shot.

---

## 8. Struttura del codice (proposta)

```
src/
├── README.md          # questo file
├── config.py          # path, iperparametri, seed
├── data.py            # caricamento csv, compositing su bianco, resize, split per product_id
├── prompts.py         # system prompt e user prompt con contesto opzionale
├── adapter.py         # encoder continuo + projector verso gli embedding di Apertus (4.1)
├── train.py           # LoRA su Apertus v1.5 8B, loss solo label umane, mirror, repeat, logit adjustment
├── teacher.py         # 70B: pseudo-label e soft label su dev e prelabel (4.2)
├── distill.py         # studente: CE + KL verso l'insegnante (4.2)
├── predict.py         # inferenza vLLM, logprob per classe, TTA
├── probe.py           # probing per layer, troncamento e teste di classificazione (5.1)
├── cascade.py         # routing per confidenza tra modello veloce e completo (5.2)
├── calibrate.py       # bias per classe per massimizzare la macro F1 sull'holdout
├── ocr.py             # OCR + language ID + crop del testo (4.4)
├── evaluate.py        # wrapper attorno a metric.py, report per classe
└── make_submission.py # submission.csv (dev + final)
```

Variabili d'ambiente richieste dal template (vedi [../.env.example](../.env.example)):
`LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`.

**Mai committare**: immagini, csv, label, passphrase, pesi, la lista errori del train. Il `.gitignore` del template
li esclude già, ma controllare sempre `git status` prima di un commit.

---

## Fonti

- [Apertus 1.5 technical report](https://arxiv.org/abs/2610.15542)
- [ViQ: Text-Aligned Visual Quantized Representations](https://arxiv.org/html/2606.27313)
- [Kelix: Closing the Understanding Gap of Discrete Tokens](https://arxiv.org/html/2602.09843v1)
- [Generalised Balanced Softmax](https://arxiv.org/pdf/2607.22258)
- [Generalized Logit Adjustment for fine-tuned foundation models](https://arxiv.org/pdf/2310.08106)
- [A Survey on Efficient Vision-Language Models](https://arxiv.org/pdf/2504.09724)
- [How Multimodal LLMs Solve Image Tasks (Yu & Lee, COLM 2025)](https://arxiv.org/abs/2508.20279)
- [Probe vs generation gap, arXiv 2604.04411](https://arxiv.org/pdf/2604.04411)
- [Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384)
- [Retrofitting Recurrent Depth into a Pretrained LM](https://arxiv.org/pdf/2608.11233)
- [Adaptive Looped Transformers](https://arxiv.org/pdf/2609.35748)
- [Visual-RFT](https://arxiv.org/abs/2503.01785)
- [GM-Skip](https://arxiv.org/html/2508.18227v1)
- [INTERLACE](https://arxiv.org/html/2511.19676)
