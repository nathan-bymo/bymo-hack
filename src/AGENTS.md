# Istruzioni per gli agenti

Questo progetto partecipa alla competizione Kaggle `hackapertus-galaxus`. Leggi `src/README.md` e le regole della competizione prima di preparare esperimenti o usare i dati.

## Organizzazione

- `apertus-vadapter-benchmark/`: esperimenti e notebook per il benchmark con visual adapter.
- `apertus-70B-benchmark/`: esperimenti e notebook per il benchmark Apertus 70B.
- `apertus-8B-benchmark/`: esperimenti e notebook per il benchmark Apertus 8B.
- `apertus-8B-lora-heads/`: LoRA su Apertus 8B e 6 teste allenati insieme, senza adapter; comando per le predizioni, file Kaggle e job H200. Risultato sul training completo in `results_full_h200.md`.
- `scripts/`: script condivisi. Conserva ogni esperimento nella propria cartella e documenta dipendenze, configurazione, dati, comandi e risultati in modo riproducibile.
- `common/`: codice condiviso per prompt (`prompt.py`), preparazione delle immagini (`images.py`) e split dei dati (`splits.py`).

## Prompt condiviso: v2

Per i benchmark zero-shot di 8B e 70B usa la **v2** di `common/prompt.py`, variante `describe:codes`, a 512². Non ha un messaggio system:

1. immagine e richiesta di descriverla (`describe_messages`);
2. per ognuna delle sei dimensioni, una richiesta separata che riparte da quella descrizione, con il contesto prodotto e una riga per classe (`question_messages` con `answer="codes"`);
3. risposta di una sola lettera, vincolata con `question_constraint` e letta con `parse_question`, che restituisce anche le probabilità.

```sh
.venv/bin/python src/scripts/eval_prompt.py 8b:describe:codes:512 70b:describe:codes:512 --split coverage
```

Sul set `coverage`, in zero-shot, la v2 ha macro F1 0.528 (8B) e 0.583 (70B), contro 0.229 e 0.304 di `docs/system_prompt.txt` (dettagli in `apertus-8B-benchmark/prompt_eval_coverage.md`). `docs/system_prompt.txt` resta nel repository solo come termine di confronto (variante `shared:json`).

Per i modelli allenati (LoRA, visual adapter) il prompt non è ancora stato provato: la v2 richiede 7 richieste per immagine. Scegli il formato in fase 1, usa **lo stesso formato in training e in inferenza** e scrivilo nei risultati. Ogni modifica al testo della v2 va misurata contro la versione condivisa e registrata nei risultati.

## Dati per training e valutazione

Il training del modello si fa in due fasi. Split e filtro sono in `common/splits.py`; non committare liste di ID.

| Fase | Dati di training (da `train.csv`) | A cosa serve |
| --- | --- | --- |
| 1. Confronto | solo `training_rows` (9.344 immagini) | scegliere metodo, epoche e impostazioni e confrontare i modelli sulle 2.532 immagini del set `coverage` |
| 2. Modello finale | tutte le 11.876 immagini etichettate | riallenare con metodo e impostazioni già scelti nella fase 1, senza cambiarli |

```python
import pandas as pd
from common.splits import training_rows  # con src nel PYTHONPATH

tutto = pd.read_csv("data/train.csv", dtype=str, keep_default_na=False)
train = training_rows(tutto)  # fase 1
```

- In fase 1 non usare le immagini di `coverage` in nessun modo per addestrare: né pesi, né epoche, né iperparametri, né calibrazione sui dati di training.
- Un modello non si valuta mai su immagini che ha studiato. Il modello della fase 2 non ha quindi un punteggio valido sul train: si controlla solo con una submission sul dev di Kaggle. Se peggiora in modo chiaro rispetto a quello della fase 1, o non è pronto in tempo (prima delle 13:00 CET del 28 novembre), si usa quello della fase 1.
- Nei risultati e nel report indica per ogni modello su quali dati è stato addestrato. I punteggi della fase 1 non vanno presentati come punteggi del modello finale.
- Le immagini final non si usano per il training in nessuna fase.

Confronto con `scripts/eval_prompt.py --split coverage`: mostra i risultati su `coverage`, su `base` (distribuzione originale delle classi) e su `test`. Le predizioni di modelli eseguiti fuori dall'API si passano come CSV con `image_id` e le sei dimensioni; il CSV non va nel repository.

## Esecuzione dei benchmark

### Baseline 8B e 70B: API Keera

Esegui localmente il codice di valutazione delle versioni standard di Apertus 1.5 8B e 70B; chiama il gateway Keera fornito da CSCS per l'inferenza. Imposta `LLM_BASE_URL=https://gateway.keera.ch/api/v1` e seleziona l'identificatore esatto in `LLM_NAME`:

| Modello | Funzione |
| --- | --- |
| `apertus-v1-5-8b` | 8B standard, supporta tool call |
| `apertus-v1-5-8b-thinking` | 8B in modalità reasoning, senza tool call |
| `apertus-v1-5-70b` | 70B standard, supporta tool call |
| `apertus-v1-5-70b-thinking` | 70B in modalità reasoning, senza tool call |

Usa le versioni standard per la baseline, salvo richiesta diversa. Prima di eseguire, ogni collega deve configurare la propria chiave in `LLM_API_KEY`, tramite variabile d'ambiente o file locale `.env` ignorato da Git. Il progetto usa anche `LLM_BASE_URL` e `LLM_NAME` come documentato nel README principale. Non chiedere di incollare la chiave in chat; non stamparla nei log, negli errori o nei notebook. Esempio di configurazione senza chiave:

```sh
export LLM_BASE_URL="https://gateway.keera.ch/api/v1"
export LLM_NAME="apertus-v1-5-8b"
# Impostare separatamente LLM_API_KEY con la propria chiave.
```

Usa la documentazione e le credenziali fornite per il gateway per verificare autenticazione, formato delle richieste e supporto alle immagini prima di implementare il client. Se manca la chiave, chiedi all'utente di configurarla localmente e segnala che la prova live è bloccata. Non assumere che accesso alle immagini, logprob, batch o altre funzionalità siano disponibili senza verificarle. L'inferenza avviene sul server remoto: misura latenza, throughput, errori e limiti dell'API. Verifica costi, limiti e disponibilità del servizio.

### Architetture personalizzate e training

Per il visual adapter e le altre architetture, fine-tuning o distillazioni descritti in `src/README.md`, verifica se servono accesso ai pesi, gradienti, embedding o componenti non esposti dall'API e scegli l'ambiente in base alle risorse richieste. Il team dispone anche di crediti Hugging Face indicati come 20 per persona: verifica saldo, idoneità e costo effettivo prima dell'uso. Valuta Kaggle in alternativa, senza presumere che una configurazione sia sufficiente per ogni modello. Per notebook Kaggle, usa il server MCP ufficiale `https://www.kaggle.com/mcp` e configura **GPU T4 x2**, verificando che entrambe siano visibili prima del carico e che il codice le utilizzi quando necessario. Due T4 hanno 16 GB ciascuna; verifica in particolare che il 70B e la tecnica scelta possano funzionare nella memoria disponibile.

Controlla stato, log ed errori fino alla conclusione e riporta ambiente, configurazione, metriche e risultati effettivi; per Kaggle includi link e versione del notebook. Se accesso, quota, disponibilità GPU, permessi o memoria impediscono il lancio, segnala il blocco e non dichiarare un'esecuzione riuscita. Non ripiegare silenziosamente su CPU, una GPU o un modello diverso.

Ogni collega configura e autorizza il proprio client MCP, ad esempio con `codex mcp add kaggle --url https://www.kaggle.com/mcp` e `codex mcp login kaggle`. L'accesso del client Kaggle CLI e quello MCP possono richiedere autorizzazioni separate. Riavvia la sessione Codex se il server aggiunto non compare tra gli strumenti. Se serve, ogni membro aderisce alla competizione e ne accetta le regole con il proprio account.

Non inserire credenziali, token, dati della competizione, pesi o output sensibili nel repository o nei notebook pubblici. Rispetta `.gitignore` e le regole della competizione, incluse quelle sull'uso dei dati finali. Non inviare submission né rendere pubblici notebook o risultati senza una richiesta esplicita.
