# Esperimenti SigLIP 2 + Apertus su Hugging Face

Stato del 27 novembre 2026. Questa pagina riporta metriche aggregate e configurazioni; non contiene immagini, etichette, predizioni individuali o segreti.

## Metodo e dati

- Apertus `swiss-ai/Apertus-v1.5-8B` e `google/siglip2-base-patch16-224` congelati; solo il projector a due layer è addestrato. Nessuna LoRA.
- H200 singola, Apertus in BF16, SigLIP a 224×224, immagine VQ nativa di Apertus con area circa 256×256 e rapporto d'aspetto conservato. Il projector aggiunge 49 token visivi continui.
- `data/prototype_splits/train_1500.csv`: 1.500 immagini con distribuzione il più possibile vicina all'originale. `validation_450.csv`: 450 immagini intenzionalmente più bilanciate. Nessun prodotto in comune e nessun prodotto del set `coverage` del team nel training o in questa validazione.
- Prompt lungo condiviso `src/training/common/prompt.py (SHARED_JSON_PROMPT)`; risposta JSON a sei classi. Cross-entropy sui token supervisionati della risposta, con etichette non umane o irrilevanti mascherate. Learning rate 0,0001, AdamW, seed 42. Nessuna focal loss o pesatura delle classi.
- La prima epoca ha usato batch 1. La seconda è ripartita dai pesi della prima con batch 4 e un nuovo ottimizzatore: lo stato del primo ottimizzatore non era stato salvato.

## Risultati sulla validazione prototipo

Macro F1 con `metric.py` della challenge, sulle stesse 450 immagini. Le celle non umane e non rilevanti sono escluse. Il set è bilanciato intenzionalmente e non rappresenta la distribuzione naturale o la leaderboard.

| Metrica | Dopo epoca 1 | Dopo epoca 2 |
| --- | ---: | ---: |
| Media macro F1 | 0,467 | 0,516 |
| subject | 0,358 | 0,431 |
| background | 0,661 | 0,740 |
| perspective | 0,160 | 0,244 |
| individuals | 0,973 | 0,948 |
| text_classification | 0,306 | 0,365 |
| text_language | 0,346 | 0,368 |
| Risposte JSON con almeno una dimensione invalida | 3/450 | 9/450 |

La differenza complessiva è circa +0,049, osservata su questo set; batch e stato dell'ottimizzatore sono cambiati tra le due epoche. Non attribuire l'effetto esclusivamente al numero di epoche e non confrontare questi valori con benchmark ottenuti su altre immagini o con altri prompt.

Il training della prima epoca ha richiesto 1.188,62 s (batch 1); la seconda 1.114,61 s (batch 4), escludendo valutazione e caricamento. Picchi di memoria CUDA riportati rispettivamente: 32,552 e 81,535 GiB. La prova batch 4, in queste condizioni, non ha prodotto un'accelerazione proporzionale alla dimensione del batch.

## Checkpoint e job

Checkpoint locali, esclusi da Git in `data/checkpoints/apertus-vadapter/`:

| Epoca | File | SHA-256 |
| --- | --- | --- |
| 1 | `projector_epoch1.pt` | `65088af2e42e6ca17178a38bcd3189a2eaef7a2108cb3dcc87aa40b422901acf` |
| 2 | `projector_epoch2.pt` | `0fd9cc2a88b3330c9b3f5c2219d330e9449fb6c1a4d16db1d36506d127241132` |

Ogni file pesa circa 76 MiB e contiene solo il projector, non i pesi di Apertus o SigLIP né lo stato dell'ottimizzatore. Per ricostruire il modello servono anche i due modelli base e la configurazione descritta sopra.

- [Job epoca 1](https://huggingface.co/jobs/nathan-monzani-bymo/6b098e2354072e8b7169f354): completato.
- [Job epoca 2](https://huggingface.co/jobs/nathan-monzani-bymo/6b09959e54072e8b7169ffc8): completato.
- [Valutazione describe:codes v2](https://huggingface.co/jobs/nathan-monzani-bymo/6b099d006c5d61fff18944d2): cancellata su richiesta dopo almeno 530 immagini su 600, usando il checkpoint epoca 2 sulle prime immagini di `prompt-dev` del collega. Non ha addestrato pesi e non ha prodotto una metrica finale. Generava una descrizione, poi sei risposte a una lettera vincolate alle classi valide; ha salvato predizioni parziali nel volume privato.

## Passaggi e verifiche

1. Una sonda lineare su SigLIP congelato è stata valutata su un piccolo split separato per prodotto: vedi [probe-results.md](probe-results.md). Non includeva Apertus o il projector.
2. Il primo notebook Kaggle partiva senza GPU; dopo l'abilitazione delle T4 lo smoke test è riuscito, ma il training è stato interrotto al passo 15 su richiesta. Nessun risultato Kaggle completo.
3. Il primo tentativo H200 è fallito per un controllo del device non compatibile con la GPU singola. Dopo la correzione, uno smoke test e la prima epoca completa sono riusciti. Il risultato della prima epoca su 450 immagini è riportato sopra.
4. Una seconda epoca è partita dal checkpoint della prima, con batch 4 e ottimizzatore reinizializzato; smoke test, training e valutazione sono terminati. I pesi dei due projector sono stati scaricati localmente, verificati tramite SHA-256 e caricati con successo in modalità `weights_only`.
5. La valutazione v2 è stata interrotta su richiesta; nessun job HF resta attivo. Non scambiare le predizioni parziali per risultati finali.

Il benchmark v2 del collega usa immagini a 512² attraverso l'API Keera; questa prova usa VQ 256² e inferenza locale BF16 con SigLIP. Le immagini sono le stesse, ma backend, risoluzione e impostazioni differiscono: la differenza di punteggio non isolerà da sola l'effetto dell'adapter.
