# Prompt zero-shot: valutazione del 2026-11-27

API Keera, modelli base, immagini a 512², temperatura 0. Codice: `src/common/prompt.py`.

## Come riprodurre

Python 3.11, dati della competizione in `data/`, chiave in `.env`.

```sh
python3.11 -m venv .venv && .venv/bin/pip install -r src/requirements.txt
.venv/bin/python src/scripts/eval_prompt.py 8b:minimal:json:512 8b:split:names:512 8b:split:codes:512 \
    8b:describe:codes:512 --split dev --limit 600 --concurrency 24 --classes
.venv/bin/python src/scripts/eval_prompt.py 8b:describe:codes:512 70b:split:codes:512 70b:describe:codes:512 \
    --split dev --limit 600 --concurrency 24 --classes
```

## Risultati: 600 immagini, 363 prodotti dello split prompt-dev

| variante | macro F1 | weighted | subject | background | perspective | individuals | text_class. | language | img/s |
|---|---|---|---|---|---|---|---|---|---|
| 8b minimal json (v1) | 0.327 | 0.649 | 0.065 | 0.403 | 0.091 | 0.795 | 0.188 | 0.422 | 8.0 |
| 8b minimal codes (v1) | 0.138 | 0.348 | 0.060 | 0.077 | 0.085 | 0.482 | 0.036 | 0.090 | 9.8 |
| 8b full json (v1) | 0.326 | 0.629 | 0.057 | 0.351 | 0.165 | 0.747 | 0.215 | 0.420 | 8.1 |
| 8b full codes (v1) | 0.164 | 0.462 | 0.056 | 0.139 | 0.043 | 0.481 | 0.153 | 0.112 | 9.8 |
| 8b split names (v2) | 0.453 | 0.633 | 0.439 | 0.522 | 0.103 | 0.880 | 0.413 | 0.359 | 6.8 |
| 8b split codes (v2) | 0.482 | 0.687 | 0.448 | 0.603 | 0.131 | 0.896 | 0.486 | 0.328 | 6.4 |
| **8b describe codes (v2)** | **0.521** | 0.719 | 0.474 | 0.575 | 0.228 | 0.895 | 0.511 | 0.446 | 3.8 |
| 70b split codes (v2) | 0.454 | 0.682 | 0.427 | 0.521 | 0.135 | 0.783 | 0.528 | 0.332 | 3.2 |
| **70b describe codes (v2)** | **0.587** | 0.745 | 0.574 | 0.629 | 0.265 | 0.902 | 0.702 | 0.453 | 2.3 |

Differenze di macro F1 con intervallo al 95% (bootstrap appaiato per prodotto, 200 ricampionamenti):

- 8b describe codes contro 8b minimal json: +0.194 [+0.154, +0.231]
- 70b describe codes contro 8b describe codes: +0.066 [+0.028, +0.099]

Le img/s sono state misurate sul gateway condiviso (16 richieste in parallelo per la v1, 24 per la v2). Nella v2
le richieste di una stessa immagine sono in sequenza.

## Diagnosi che ha portato alla v2

Solo `subject`, 103 immagini dello split prompt-dev con al massimo 8 per classe (non confrontabile con la tabella
sopra). Script usato una tantum, non incluso nel repo. Macro F1:

| configurazione | 8B | 70B |
|---|---|---|
| v1: sei dimensioni con regole nel system prompt, JSON | 0.01 | 0.07 |
| solo subject con regole, dopo l'immagine | 0.18 | 0.06 |
| solo i nomi delle classi, dopo l'immagine | 0.39 | 0.34 |
| una riga per classe | 0.41 | 0.48 |
| descrizione, poi una riga per classe | 0.54 | 0.63 |

## Limiti

- La v2 è stata scelta guardando lo split prompt-dev. Lo split prompt-test non è ancora stato usato.
- Alcune classi hanno da 1 a 3 esempi nel campione: i loro valori per classe sono molto rumorosi.
- `perspective` resta molto debole. In `text_language` la classe `none` ha recall 0.16.
- Non ancora provati: contesto assente nella v2 completa, altre risoluzioni, calibrazione delle probabilità.
