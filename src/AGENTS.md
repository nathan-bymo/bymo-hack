# Istruzioni per gli agenti

Questo progetto partecipa alla competizione Kaggle `hackapertus-galaxus`. Leggi `src/README.md` e le regole della competizione prima di preparare esperimenti o usare i dati.

## Organizzazione

- `apertus-vadapter-benchmark/`: esperimenti e notebook per il benchmark con visual adapter.
- `apertus-70B-benchmark/`: esperimenti e notebook per il benchmark Apertus 70B.
- `apertus-8B-benchmark/`: esperimenti e notebook per il benchmark Apertus 8B.
- `scripts/`: script condivisi. Conserva ogni esperimento nella propria cartella e documenta dipendenze, configurazione, dati, comandi e risultati in modo riproducibile.

## Esecuzione su Kaggle

Quando il lavoro richiesto comprende un benchmark o l'esecuzione di un notebook, prepara e lancia automaticamente il notebook su Kaggle tramite il server MCP ufficiale `https://www.kaggle.com/mcp`, usando l'account autorizzato del collega. Configura l'acceleratore **GPU T4 x2** e verifica nel notebook che siano visibili due NVIDIA Tesla T4 prima di avviare il carico. Configura esplicitamente il codice per usare entrambe le GPU quando necessario: renderle disponibili non distribuisce automaticamente il modello.

Controlla stato, log ed errori fino alla conclusione e riporta link, versione del notebook, configurazione, metriche e risultati effettivi. Se MCP, accesso, quota, disponibilità GPU, permessi o memoria impediscono il lancio, segnala il blocco e non dichiarare un'esecuzione riuscita. Due T4 hanno 16 GB ciascuna; in particolare, verifica che il 70B e la tecnica scelta possano funzionare realmente nella memoria disponibile. Non ripiegare silenziosamente su CPU, una GPU o un modello diverso.

Ogni collega configura e autorizza il proprio client MCP, ad esempio con `codex mcp add kaggle --url https://www.kaggle.com/mcp` e `codex mcp login kaggle`. L'accesso del client Kaggle CLI e quello MCP possono richiedere autorizzazioni separate. Riavvia la sessione Codex se il server aggiunto non compare tra gli strumenti. Se serve, ogni membro aderisce alla competizione e ne accetta le regole con il proprio account.

Non inserire credenziali, token, dati della competizione, pesi o output sensibili nel repository o nei notebook pubblici. Rispetta `.gitignore` e le regole della competizione, incluse quelle sull'uso dei dati finali. Non inviare submission né rendere pubblici notebook o risultati senza una richiesta esplicita.
