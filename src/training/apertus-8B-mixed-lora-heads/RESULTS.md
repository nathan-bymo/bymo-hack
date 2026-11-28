# Mixed: risultati e lavoro del 28 novembre

Modello: SigLIP 2 + DINOv3 + VQ di Apertus + OCR, con un solo decoder Apertus 8B (24 layer, LoRA) e sei teste. Architettura e lancio sono nel [README](README.md). Elenco di tutte le prove: [../EXPERIMENTS.md](../EXPERIMENTS.md).

Il file riporta solo numeri aggregati: nessun image_id, label o file della competizione. Ogni confronto vale sullo stesso set di immagini.

## 1. Training del mixed (run `mixed-modal-20261128-033303`)

Modal, 1 H200. Partenza da `final-abt/model/best` (SigLIP + VQ). Il ramo DINO entra con un residuo inizializzato a zero, quindi al passo 0 il modello riproduce esattamente `final-abt`. Dati: 10.683 immagini reali più 1.040 sintetiche per il testo. La selezione del checkpoint si fa sulla validazione interna (1.038 immagini), e prompt-test resta sempre fuori.

| Stadio | Validazione interna, macro F1 | Note |
|---|---|---|
| partenza (`final-abt`) | 0,82116 | |
| solo fusione DINO, 0,5 epoche | 0,82872 | |
| joint, epoca 1 | 0,83321 | |
| joint, epoca 2 | 0,83661 | |
| **joint, epoca 3** | **0,84308** | checkpoint scelto |
| joint, epoca 4 | 0,83661 | la loss di training scende e quella di validazione sale da 0,305 a 0,316: overfitting |

- **Tempi:** circa 18 minuti per epoca joint sulla cache di feature e 23 minuti per costruire la cache. Lo stadio top2 (sblocco degli ultimi due blocchi) non è partito.

| Set | Macro F1 | Weighted F1 | Velocità |
|---|---|---|---|
| prompt-test (1.193 immagini) | 0,84282 | 0,9235 | 5,93 img/s |
| **dev Kaggle pubblico** (submission 59038887) | **0,84215** | | 5,98 img/s |

**Contesto della classifica pubblica il 28/11 mattina:**
- aperture 0,91151, PrismAI 0,87477, BYMO 0,84700 (ensemble SigLIP+DINO, submission 59032216);
- baseline degli organizzatori: Qwen 3.5 9B 0,917, Gemini 4 Argon 0,834, Gemini 3.8 Flash 0,806.

## 2. Profilo per dimensione e combinazioni (prompt-test)

| Modello | Macro F1 | Subject | Background | Perspective | Individuals | Text class. | Text lang. |
|---|---|---|---|---|---|---|---|
| SigLIP+VQ (`final-abt`) | 0,8214 | 0,846 | 0,881 | 0,755 | 0,959 | 0,766 | 0,720 |
| DINO 81 token | 0,8077 | 0,772 | 0,847 | 0,783 | 0,944 | 0,719 | 0,781 |
| Mixed | 0,8428 | 0,868 | 0,875 | 0,776 | 0,970 | 0,769 | 0,798 |
| Mixed + DINO 50/50 | 0,8557 | 0,841 | 0,897 | 0,833 | 0,970 | 0,776 | 0,819 |
| **Teacher** (subject dal solo mixed, resto mixed+DINO) | **0,8603** | 0,868 | 0,897 | 0,833 | 0,970 | 0,776 | 0,819 |

- **Teacher contro mixed:** +0,017 macro F1, con IC 95% [+0,006, +0,030] dal bootstrap appaiato per prodotto. Il valore è un po' ottimista, perché la scelta di subject è stata fatta su prompt-test, confermandola su entrambe le metà per prodotto.
- **Mediare con DINO peggiora subject** (da 0,868 a 0,841). Per questo nel teacher la combinazione cambia dimensione per dimensione.
- **P13 e P14** (rispettivamente forti su text_language e text_classification) non hanno probabilità nel repository e restano fuori.

## 3. Strategia decisa

**Criteri della giuria:**
- La performance pesa circa il 30% ed è valutata **a gruppi**: sotto Gemini 3.8 Flash, tra Flash e Qwen, sopra Qwen. "Differences of a few tenths of a percent are not decisive."
- Velocità e risorse pesano il 20%. A parità di risultato la giuria preferisce la soluzione più pratica da adottare.
- È previsto un bonus per una perspective molto buona e per una lista di errori di label.

**Decisione:**
- **A runtime gira un solo decoder.** Un ensemble di più modelli guadagnerebbe +0,005–0,01, senza cambiare gruppo, ma dimezzerebbe la velocità. Gli ensemble servono solo come teacher in sviluppo.
- **Consegna di default:** mixed joint 3 con il percorso di inferenza veloce più TF32 (sezione 5, 14,3 img/s sul dev) e la TTA selettiva di perspective (sezione 6). Lo studente distillato (sezione 4) la sostituisce solo se supera il proprio criterio di promozione.

**Scartato:**

| Idea | Motivo |
|---|---|
| Ensemble a runtime | velocità |
| Qwen come teacher allenato da zero | non entra nei tempi |
| top2 o più epoche | già in overfitting all'epoca 4 |
| Soglie o bias per classe | già provati, in overfitting |
| Teacher VLM chiuso | Gemini Argon fa 0,834, meno del nostro ensemble |
| Classificatore testuale sull'OCR, modello sul contesto prodotto | poco guadagno atteso rispetto al tempo |

## 4. Distillazione sulle immagini dev (F1): target pronti, training non lanciato

**Target:**
- Le probabilità soft del teacher della sezione 2 sulle 5.261 immagini dev, permesse dalle regole: "Pseudo-labels on the unlabeled dev images are allowed".
- File nel repository privato: `teacher/dev_teacher_v1.pt` (report in `teacher/dev_teacher_v1.json`), SHA-256 `6d02871f2c4024b00dcf7e96491ac1ce45a0adcf37ce62ad2b455e033e202cf0`.
- Si rigenera con [build_teacher_targets.py](../../scripts/build_teacher_targets.py).
- Contenuto: `image_ids` nell'ordine di dev.csv, `probabilities` (6 tensori float32 in ordine TAXONOMY) e `mask` (bool [5261, 6]).

**Maschera:**
1. Le regole di scoring si applicano alle predizioni del teacher. Esempi: niente perspective per un infographic previsto, niente text_language senza overlay previsto o senza un subject che lo richiede.
2. Soglia di confidenza per dimensione, con precisione del teacher ≥ 0,90 su prompt-test. Il vincolo agisce solo su perspective, con soglia 0,51.
3. Celle dev tenute: subject 93%, background 81%, perspective 59%, individuals 81%, text_classification 91%, text_language 76%.

**Codice:**
- [distillation.py](distillation.py) controlla checksum e coerenza dei target e calcola la loss soft.
- [train_mixed.py](train_mixed.py) ha una modalità pilot: warm start dal joint 3, mezza epoca con la loss soft a peso 0,5, e un controllo solo supervisionato con lo stesso budget di esempi.
- Il codice è testato in [test_mixed.py](test_mixed.py), ma **non è mai stato lanciato**.

**Lancio** (descritto anche nel [README](README.md#distillation-pilot)):

```sh
python3 src/training/apertus-8B-mixed-lora-heads/build_hf_job.py
MIXED_MINUTES=150 modal run --detach src/training/apertus-8B-mixed-lora-heads/modal_app.py \
  --output-repo nathan-monzani-bymo/galaxus-vlh-private \
  --data-repo nathan-monzani-bymo/galaxus-data-private \
  --source-path mixed-modal-20261128-033303/model/best \
  --source-revision 7d5feac45b8ee2f73c0d0a7854cd995bdfabdc6b \
  --teacher-file teacher/dev_teacher_v1.pt \
  --teacher-sha256 6d02871f2c4024b00dcf7e96491ac1ce45a0adcf37ce62ad2b455e033e202cf0 \
  --synthetic-file synthetic/dino_text_1040.tar --local-hf-auth
```

- **Revisione:** `7d5feac…` è una revisione del repository privato che contiene sia il checkpoint sia il teacher.
- **Accesso:** chi lancia deve avere un token Hugging Face con accesso ai due repository privati del team.
- **Budget:** `MIXED_MINUTES` è il timeout della funzione Modal, cioè il tetto di costo, su una H200. Il training dispone di `MIXED_MINUTES − 25` minuti, oppure del valore di `--wall-minutes`. In quel tempo il job deve costruire la cache di circa 21.000 viste, contando anche il dev (prima erano 23 minuti per circa 16.000), allenare due rami da mezza epoca ciascuno (circa 9 minuti l'uno a 18 minuti per epoca) e fare gli export. Con 150 minuti il margine c'è.
- **Verifiche fatte:** dopo l'integrazione con le modifiche dei colleghi (LoRA su SigLIP, modalità final) passano i 7 test di [test_mixed.py](test_mixed.py) su CPU. Il pilot non è ancora stato eseguito su GPU.

**Criterio di promozione dello studente:**
- +0,008 macro F1 su prompt-test rispetto al joint 3, con bootstrap appaiato positivo;
- weighted F1 entro 0,002;
- almeno il 90% del throughput.

Il dev pubblico non può confermarlo, perché lo studente ha visto quelle immagini.

## 5. Velocità dell'inferenza (F2)

Script: [fast_infer.py](fast_infer.py). Si lancia con [modal_infer.py](modal_infer.py) dopo aver generato il bundle con [build_infer_job.py](build_infer_job.py).

Fa le stesse operazioni dell'export di `train_mixed.py`, ma organizzate per la velocità:
- LoRA fusa nei pesi;
- immagini ordinate per dimensione VQ e batch 32;
- immagini decodificate in un pool di thread un batch avanti;
- OCR in un thread separato, anche lui un batch avanti;
- un'immagine illeggibile viene sostituita da una bianca, così ogni riga riceve una predizione.

Misure su H200, dall'immagine al CSV, caricamento dei modelli escluso (run `infer-bench-20261128-072424` e `infer-dev-20261128-072424`):

| Configurazione | Set | img/s | Macro F1 | Classe identica all'export originale |
|---|---|---|---|---|
| come l'export originale (batch 16) | prompt-test | 6,23 | 0,84282 | 100% |
| **percorso veloce** | prompt-test | 7,51 | 0,84318 | 99,6–100% per dimensione |
| **percorso veloce** | dev | **8,53** (contro 5,98, +43%) | — | 99,5–99,9% per dimensione, uguale al confronto col CSV inviato |

- **Tempo per fase**, export originale su prompt-test (192 s): VQ 84 s, OCR 62 s, Apertus 38 s, SigLIP 6 s, DINO 1 s.
- **Collo di bottiglia:** il tokenizer VQ, in fp32 senza TF32 (i codici sono un argmax). Ordinare e allargare i batch non lo accelera, perché è limitato dal calcolo. Nel percorso veloce sul dev il thread OCR lavora 589 s su 617, quindi diventa lui il percorso critico.
- **Parità:** le piccole differenze vengono dall'arrotondamento bf16 della LoRA fusa e dalla composizione dei batch, e la macro F1 non peggiora. Il criterio adottato è accordo ≥ 99,5% e F1 non peggiore. La parità bit a bit si ottiene solo con il percorso originale.
- **Memoria di picco:** 47 GB nell'export originale, 79–88 GB nel percorso veloce (batch 32 e OCR in parallelo).
### TF32 (`--tf32`, run `infer-tf32-20261128-075937`)

Il flag attiva il TF32 per le convoluzioni e le matmul in fp32 di VQ, SigLIP e OCR. Le H200 erano esaurite e la misura è stata fatta su una **H100 80GB**. Sotto i 100 GB di memoria lo script riduce da solo il batch a 16.

| Configurazione | Set | img/s | Macro F1 | Classe identica all'export originale | Memoria di picco |
|---|---|---|---|---|---|
| veloce + TF32 | prompt-test | 13,83 | 0,84242 | 99,3–100% | 23 GB |
| veloce + TF32, senza LoRA fusa | prompt-test | 12,99 | 0,84231 | 99,3–100% | 22 GB |
| **veloce + TF32** | **dev** | **14,34** | — | **99,1–100%** (uguale al confronto col CSV inviato) | 23 GB |

- **Codici VQ:** con TF32 cambia lo 0,7–0,9% dei token, e solo il 20–23% delle immagini ha tutti i codici identici. L'effetto sulla F1 è −0,0008 rispetto al percorso veloce e −0,0004 rispetto all'export originale. È rumore al livello di prompt-test e viene dal TF32, non dalla fusione della LoRA.
- **Tempo per fase**, veloce + TF32 su prompt-test: il VQ scende da 84 a 18 s. Ora il percorso critico è l'OCR: sul dev il suo thread lavora 359 s su 367.
- **Velocità complessiva:** da 5,98 a 14,34 img/s sul dev, ×2,4, su una GPU più piccola e con 23 GB di memoria invece di 47.

**Decisione:** il TF32 entra nella configurazione di consegna, perché la velocità pesa il 20% della valutazione e la perdita è trascurabile. Il preset `final_safe` resta come riserva senza TF32.

## 6. TTA di perspective (F3a): adottata nella variante selettiva

**Come funziona:**
- Si specchia l'immagine e se ne predice la perspective.
- Le classi sinistra/destra si scambiano (`front_angle_left↔right`, `back_angle_left↔right`, `left_side↔right_side`) e si fa la media con la vista originale.
- Le altre cinque dimensioni e l'OCR vengono sempre dalla vista originale.
- **Variante selettiva:** si specchia solo se le due classi più probabili sono una coppia sinistra/destra con margine < 0,15, soglia fissata prima della prova.

| Variante | Immagini specchiate | Perspective | Accuratezza sinistra/destra | Macro F1 | Delta contro nessuna TTA (IC 95%) | img/s |
|---|---|---|---|---|---|---|
| nessuna | 0% | 0,7781 | 0,744 | 0,84318 | | 7,51 |
| tutte | 100% | 0,7892 | 0,768 | 0,84503 | +0,0019 [−0,0031, +0,0071] | 4,16 |
| **selettiva** | **3,1%** | 0,7887 | 0,765 | 0,84496 | **+0,0017 [+0,0001, +0,0039]** | circa −3% |

**Per classe (selettiva rispetto a nessuna TTA):**
- front_angle_left da 0,732 a 0,745, front_angle_right da 0,731 a 0,776;
- left_side da 0,814 a 0,833, right_side da 0,772 a 0,814;
- back_angle_right invariata a 0,694, back_angle_left da 0,696 a 0,681 (circa un'immagine su 25);
- front_head_on da 0,874 a 0,876.

Valutazione offline con [eval_tta.py](../../scripts/eval_tta.py).

## 7. Lista degli errori di label: risultato negativo, non inviata

**Metodo:** [label_errors.py](../../scripts/label_errors.py) cerca le celle del train in cui almeno 2 modelli che non hanno visto il prodotto concordano con alta confidenza contro la label umana. Ne ha trovate 57, salvate in `data/label_errors/`, mai nel repository.

**Controllo visivo:** ho guardato il caso più estremo per ogni dimensione. **Nessuno su 6 è un errore certo.** In 4 casi aveva ragione la label umana:
- una tinta unita è `placeholder`;
- persone piccole sullo sfondo sono `individual_present`;
- quote dimensionali sono `technical_specs`;
- un fondale da studio sfumato è `color`.

Gli altri 2 sono ambigui.

**Insight:** i disaccordi confidenti dei modelli sono **punti ciechi condivisi**, non errori di label. Nessun ensemble li corregge, ed è coerente con il limite di circa 0,90 su prompt-test di un selettore oracolo dei checkpoint esistenti.

## 8. Pipeline finale (13:00)

**Configurazione di consegna:** mixed joint 3 (`mixed-modal-20261128-033303/model/best`, gli stessi pesi di `weights/apertus-8B-mixed-lora-heads-final/`), con percorso veloce, TF32 e TTA selettiva di perspective. È il preset `final` di [modal_infer.py](modal_infer.py). Riserva: `final_safe`, senza TF32.

**Lancio:**

```sh
python3 src/training/apertus-8B-mixed-lora-heads/build_infer_job.py
FINAL_PASSPHRASE=… modal run src/training/apertus-8B-mixed-lora-heads/modal_infer.py --runs final --local-hf-auth
```

Per una prova sul dev con la stessa configurazione: `--runs dev_final`.

- **Archivio:** `final_encrypted.bin` è nel dataset privato, con lo SHA-256 pubblicato verificato (`4b37a260…6465`). Il job lo scarica, controlla di nuovo l'hash e lo decifra sul proprio disco.
- **Passphrase:** passata come secret Modal temporaneo, mai in un file né come argomento.
- **Output:** `final_fast.csv` e le probabilità finiscono nel repository privato. Il file Kaggle si compone con [final_submission.py](../../scripts/final_submission.py) (righe dev più righe final).
- **GPU:** il job usa una H200 e, se sono esaurite come la mattina del 28/11, ripiega su H100 o A100 da 80 GB.
- **Prove completate:** percorso veloce sul dev (sezione 5) e percorso veloce + TF32 sul dev (sezione 5, 14,34 img/s).
- **Prova non ancora fatta:** la combinazione completa del preset `final`, cioè veloce + TF32 + TTA selettiva, non è stata eseguita in un unico run. La TTA è stata misurata solo su prompt-test, senza TF32. Prima delle 12:30 conviene lanciare `--runs dev_final`.
