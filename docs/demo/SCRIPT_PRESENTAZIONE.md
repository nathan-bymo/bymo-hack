# Copione della presentazione — Demo "Image classification (AI)"

Team Bymo · Hack Apertus. Pitch sabato 28/11 alle 15:00; video di massimo 2 minuti da consegnare entro le 14:30
(registrazione 12:00–12:45; **dalle 13:00 alle 14:30 nessun uso di GPU, HF, Kaggle o rete del team per la demo**).
L'interfaccia è in inglese; il copione è in italiano. Ogni risultato registrato mostra il badge "Recorded …".

**Prima di iniziare (14:30–15:00):** `make -C src/demo demo`; `.env` con `LLM_API_KEY`; sessione pronta con 15–20 immagini
(foto nostre + poche di coverage, mai dev/final, nessuna persona senza consenso); predizioni dei modelli allenati
registrate (piano B, avviabile senza rete); video di riserva a portata di mano.

## Versione 3–5 minuti

| tempo | scena | cosa si dice | cosa si vede |
|---|---|---|---|
| 0:00 | pagina prodotto demo nel guscio ERP | "Questo è il punto del flusso Galaxus dove oggi lavora il modello di classificazione." | badge "Demo — Team Bymo · Hack Apertus" |
| 0:20 | `Enrich Product ▾ → Image classification (AI)` | "Abbiamo aggiunto una voce: classificazione delle immagini con Apertus v1.5." | coda vuota |
| 0:35 | trascinare 10–15 immagini, `Run classification` | "Velocità misurata in tempo reale, contro l'obiettivo di almeno 2 immagini al secondo." | barra di avanzamento, img/s e GPU o "gateway remoto" |
| 1:05 | ordinare per **Min conf**, aprire la prima | "Il revisore parte dai casi dubbi." | colonna Min conf, chip di confidenza |
| 1:20 | dettaglio: "Image type" pre-compilato, top-3, correggere Perspective, `Save` | "Persona nel ciclo: il modello suggerisce, la persona decide." | `AI suggested` → `Corrected` |
| 1:50 | cambio modello sulle stesse immagini; `Compare models` | "Zero-shot contro modelli allenati, senza ricaricare le immagini; qui i disaccordi." | celle evidenziate, latenza per modello |
| 2:30 | dashboard | "Coverage non è la classifica Kaggle. Il full fine-tune ha visto tutto il train: badge rosso, l'unico numero valido è il Kaggle dev." | banner, barre con linea 0.917 "set diverso", badge |
| 3:15 | velocità e costo | "Velocità misurate o riportate con la loro GPU; costo per 1.000 e per 100.000 immagini." | pannello costi (P11: 10,5 $ per 100.000 su H200, da `results_full_h200.md`) |
| 3:45 | chiusura | "Apertus, persona nel ciclo, numeri onesti con la loro fonte." | — |

## Taglio da 2 minuti (video)

0:00–0:35 guscio e voce di menu → coda (20 s) → dettaglio con correzione (35 s) → confronto (20 s) → dashboard e
velocità (25 s). Non superare 2:00.

## Cose da dire con onestà
- Il contesto prodotto manca sulle nostre foto: condizione non misurata per i modelli allenati (avviso in UI).
- Le confidenze di modelli diversi non sono confrontabili (metodi diversi).
- P11 ha Kaggle dev 0.796 (dev pubblico), sotto il riferimento degli organizzatori 0.917.
- Nessun collegamento all'ERP reale: guscio statico a scopo dimostrativo.
