# Istruzioni per gli agenti

Questo progetto partecipa alla competizione Kaggle `hackapertus-galaxus`. Leggi `src/README.md` e le regole della competizione prima di preparare esperimenti o usare i dati.

## Organizzazione

- `apertus-vadapter-benchmark/`: esperimenti e notebook per il benchmark con visual adapter.
- `apertus-70B-benchmark/`: esperimenti e notebook per il benchmark Apertus 70B.
- `apertus-8B-benchmark/`: esperimenti e notebook per il benchmark Apertus 8B.
- `scripts/`: script condivisi. Conserva ogni esperimento nella propria cartella e documenta dipendenze, configurazione, dati, comandi e risultati in modo riproducibile.
- `scripts/system_prompt.txt`: prompt di classificazione condiviso per i benchmark standard 8B e 70B e, dove compatibile, per il visual adapter. Usalo come messaggio system; passa immagine e contesto prodotto nel messaggio utente. Confronta eventuali modifiche con la versione condivisa e registrale nei risultati.

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
