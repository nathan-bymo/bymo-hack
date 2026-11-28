# Statistiche di tutti i modelli, per i benchmark

Solo numeri aggregati: nessuna immagine, label o predizione individuale. Stato del 28 novembre, ore 04:00 CET.

| file | contenuto |
|---|---|
| [overall.md](overall.md) | una riga per prova e set: macro F1, weighted F1 e macro F1 delle sei dimensioni |
| [per_class.md](per_class.md) | per ogni prova, set, dimensione e classe: precision, recall, F1, numero di esempi |

Si rigenerano con `python src/scripts/benchmark_stats.py --out src/training/benchmark` dalle predizioni salvate in
`data/` (fuori da git); le stesse tabelle in CSV restano in `data/benchmark/`. Non servono GPU né chiamate API. Una prova ha un punteggio su un set solo se ne ha predetto
tutte le immagini. **Confrontare solo righe dello stesso set.** Descrizione delle prove: [../EXPERIMENTS.md](../EXPERIMENTS.md).

Attenzione:
- il modello di Nathan e gli ensemble hanno solo `prompt-test` (il modello finale ha visto parte di coverage);
- le regole degli ensemble sono state scelte guardando `prompt-test`: quei due numeri sono ottimistici;
- le righe "Baseline pixel" sono controlli senza Apertus, non modelli del team;
- il fine-tuning completo di Kristian ha visto tutto il train: ha solo il punteggio Kaggle;
- il LoRA generativo di Kristian (P10) ha i suoi risultati in
  `../../micro-training/lora-generativo/results/` ed è fuori dalle due tabelle.

## Kaggle dev (leaderboard pubblica, 5.261 immagini)

| modello | punteggio | inviato (UTC) |
|---|---|---|
| Nathan: LoRA + teste + SigLIP 2 + OCR + pooling (`final-abt`) | **0,83177** | 27/11 22:40 |
| P13: `final-abt` + 1 epoca con immagini sintetiche | 0,82872 | 28/11 02:57 |
| P12: LoRA + teste + pooling, 512² | 0,80566 | 28/11 01:04 |
| Kristian: full fine-tuning 24 layer + teste | 0,80473 | 27/11 22:02 |
| P11: LoRA + teste, 256² | 0,79573 | 27/11 20:30 |

Le righe delle immagini final in questi invii hanno i valori di `sample_submission.csv`.

## Velocità, memoria e costo

| modello | immagini/s | memoria | costo del training | fonte |
|---|---|---|---|---|
| P11 | 13,2 (H200) | 22,0 GB in predizione, 139 GB in training | circa 5,19 $ | `../apertus-8B-lora-heads/results_full_h200.md` |
| P12 | 3,9 (H200) | 32,5 GB in predizione, 45,4 GB in training | circa 14 $ (168 min × 5 $/h) | `../apertus-8B-lora-heads/results_p12_512_pooling.md` |
| Nathan `final-abt` | 6,2 (H200); 4,9 (L40S) | non riportata | circa 16 $ (tre passi più un tentativo fallito) | `../apertus-8B-vadapter-lora-heads/RISULTATI.md` |
| P13 | 6,7 (H200, prompt-test) | non confrontabile | 1 epoca, job di circa 73 min su Modal | `../apertus-8B-vadapter-lora-heads/results_p13_synthetic.md` |
| 8B zero-shot v2 | 3,8 (API Keera condivisa) | remota | nessuno | report in `plain-benchmark` |
| 70B zero-shot v2 | 2,8 (API Keera condivisa) | remota | nessuno | report in `plain-benchmark` |

Le velocità hanno condizioni diverse (batch, API condivisa, GPU): non sono un confronto controllato. Requisito della
challenge: almeno 2 immagini/s.

## Dove sono i pesi

I repository Hugging Face sono **privati**. Il 28/11 alle 02:49 i 33 file di pesi presenti in `data/` e `weights/`
del Mac di Nikola avevano tutti una copia online identica (stesso SHA-256).

| modello | Git LFS (`weights/`) | Hugging Face |
|---|---|---|
| Nathan `final-abt` | `apertus-8B-vadapter-lora-heads-final/` | `MANNIK14/galaxus-vlh-private/final-abt/model/best` |
| P13 | — | `MANNIK14/galaxus-vlh-private/synth-ft-1128-0243/model/best` |
| P11 | `lora_heads_256_p11/` | `MANNIK14/galaxus-lora-heads-private/archive/p11_lora_heads_256_coverage0781` |
| P12 | — | `MANNIK14/galaxus-lora-heads-private/archive/p12_lora_heads_pool_512_coverage0806` |
| teste su Apertus congelato, pesi per classe | — | `MANNIK14/galaxus-heads-private` (`small/`, `archive/`) |
| baseline pixel e smoke test | — | `MANNIK14/galaxus-heads-private/archive/solo_mac_20261128` |

I pesi del full fine-tuning di Kristian sono nel suo repository: non sono stati verificati qui.
