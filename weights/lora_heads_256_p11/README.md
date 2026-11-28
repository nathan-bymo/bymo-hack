# LoRA + 6 teste su Apertus 1.5 8B, 256² (prova P11)

- **Punteggi:** dev Kaggle pubblico 0.796; coverage 0.781 (macro F1, `metric.py`). Risultati e fonti:
  [phase-1 results](../../src/training/phase-1/RESULTS.md).
- **Training:** `training_rows` (9.344 immagini), 3 epoche, seed 3407, nessun prodotto di coverage.
- **Base:** `swiss-ai/Apertus-v1.5-8B` alla revisione `1519de0ae7f3d114879291810a58259966f64c99`.
- **Codice:** `src/training/apertus-8B-lora-heads/`, commit `be6ae08` (training) e successivi compatibili.
- **File:** `lora/` adapter PEFT (LoRA r32 su q, k, v, o, up, down di tutti i 32 layer); `heads.pt` teste
  (`torch.load(..., weights_only=True)`); `settings.json` impostazioni per ricaricare il modello.

```sh
python src/training/apertus-8B-lora-heads/predict.py --checkpoint weights/lora_heads_256_p11 --csv data/dev.csv \
    --out data/runs/p11/dev.csv
```

Nessun dato della competizione in questi file. Copia identica nel repository Hugging Face privato del team
(`archive/p11_lora_heads_256_coverage0781`).
