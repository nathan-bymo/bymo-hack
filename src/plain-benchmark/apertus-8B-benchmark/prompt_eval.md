# Prompt zero-shot: valutazione del 2026-11-27

Elenco di tutte le prove: [RISULTATI.md](../../training/EXPERIMENTS.md).

## Scheda della prova P0 e P2: velocità iniziale del gateway, diagnosi del prompt e prove di funzionamento

| voce | |
|---|---|
| **Domanda** | P0: il gateway cosa supporta e quanto è veloce? P2: perché la v1 risponde quasi sempre `full_product`? |
| **Modelli** | Apertus 1.5 8B e 70B base, **nessun training**, via API Keera |
| **Dati** | P0: immagine sintetica. P2: 103 immagini di `prompt-dev`, al massimo 8 per classe di `subject`, solo `subject`. Prove di funzionamento: 10, 12 e 16 immagini |
| **Contenuto del file** | riepilogo di P1 e P3, diagnosi completa P2, prove piccole, velocità P0 |
| **Cosa non dice** | il campione di P2 è piccolo e bilanciato: serve a trovare la causa, non a misurare la qualità |


API Keera, modelli base, immagini a 512², temperatura 0. Codice: `src/training/common/prompt.py`.

**Tutti i risultati:**

- [Confronto sul set coverage, 2.532 immagini](prompt_eval_coverage.md): il più completo, con prompt-test e prompt condiviso.
- [Analisi preliminare maestro–studente](teacher_student_zeroshot.md).
- [Risultati per classe sulle prime 600 immagini](prompt_eval_per_class_600.md).
- In questo file: riepilogo sulle 600 immagini, diagnosi e prove piccole.

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

- La v2 è stata scelta guardando lo split prompt-dev. Quando sono stati prodotti i numeri di questo file, prompt-test
  non era ancora stato usato: il controllo è in `prompt_eval_coverage.md`.
- Alcune classi hanno da 1 a 3 esempi nel campione: i loro valori per classe sono molto rumorosi.
- `perspective` resta molto debole. In `text_language` la classe `none` ha recall 0.16.
- Non ancora provati: contesto assente nella v2 completa, altre risoluzioni, calibrazione delle probabilità.

## Diagnosi completa su `subject`

Tutte le configurazioni provate sulle 103 immagini descritte sopra. Ogni cella è `accuratezza / macro F1`. Il campione
è piccolo e bilanciato per classe: **non è confrontabile** con le altre tabelle. Script usati una tantum, non nel repo.

Primo giro (risposta JSON, tranne dove indicato):

| configurazione | 8B | 70B |
|---|---|---|
| 1. v1: sei dimensioni nel system prompt | 0.08 / 0.01 | 0.13 / 0.07 |
| 2. come 1, senza contesto prodotto | 0.09 / 0.03 | 0.26 / 0.22 |
| 3. sei dimensioni dopo l'immagine | 0.08 / 0.01 | 0.15 / 0.10 |
| 4. solo subject con regole, nel system prompt | 0.08 / 0.01 | 0.10 / 0.04 |
| 5. solo subject con regole, dopo l'immagine | 0.20 / 0.18 | 0.16 / 0.06 |
| 6. come 5, senza regole | 0.45 / 0.39 | 0.39 / 0.34 |
| 7. come 5, risposta a una lettera | 0.41 / 0.34 | 0.27 / 0.20 |

Secondo giro (domanda dopo l'immagine, senza regole lunghe):

| configurazione | 8B | 70B |
|---|---|---|
| nomi, senza contesto, testo libero | 0.51 / 0.45 | 0.48 / 0.44 |
| nomi, con contesto, testo libero | 0.47 / 0.40 | 0.44 / 0.38 |
| nomi, senza contesto, vincolato | 0.47 / 0.39 | 0.42 / 0.40 |
| nomi, con contesto, vincolato | 0.46 / 0.39 | 0.42 / 0.37 |
| una riga per classe, senza contesto | 0.47 / 0.41 | 0.50 / 0.48 |
| una riga per classe, con contesto | 0.45 / 0.42 | 0.50 / 0.47 |
| descrizione e una riga per classe, senza contesto | 0.59 / 0.54 | 0.65 / 0.63 |
| descrizione e una riga per classe, con contesto | 0.53 / 0.49 | 0.64 / 0.62 |

In questo campione il contesto prodotto non ha aiutato, e con l'8B a due passaggi il risultato era più basso. La v2
usa comunque il contesto: l'ablation senza contesto sull'intera v2 **non è ancora stata fatta**.

## Prove piccole di funzionamento

Servivano a controllare parsing e probabilità. Con così poche immagini **i punteggi non permettono confronti**.

| prova | variante | macro F1 | weighted F1 |
|---|---|---|---|
| 16 immagini di prompt-dev | 8B v1 breve JSON | 0.311 | 0.675 |
| 16 immagini di prompt-dev | 8B v1 breve lettere | 0.285 | 0.421 |
| 16 immagini di prompt-dev | 8B v1 completo JSON | 0.331 | 0.685 |
| 16 immagini di prompt-dev | 8B v1 completo lettere | 0.324 | 0.480 |
| 12 immagini di prompt-dev | 8B v2 descrizione, nomi | 0.463 | 0.771 |
| 12 immagini di prompt-dev | 8B v2 separate, lettere | 0.465 | 0.752 |
| 12 immagini di prompt-dev | 8B v2 descrizione, lettere | 0.540 | 0.798 |
| 10 immagini di coverage | 8B prompt condiviso | 0.344 | 0.425 |
| 10 immagini di coverage | 70B prompt condiviso | 0.443 | 0.607 |

## Velocità del gateway, prova iniziale

48 richieste per riga, un'immagine sintetica 512² ripetuta (la cache può aver aiutato), risposta breve.

| modello | richieste in parallelo | richieste/s | latenza mediana (s) |
|---|---|---|---|
| 8B | 4 | 10.7 | 0.30 |
| 8B | 16 | 22.5 | 0.51 |
| 70B | 4 | 6.8 | 0.50 |
| 70B | 16 | 19.2 | 0.64 |
