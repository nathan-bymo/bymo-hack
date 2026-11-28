# Phase 2 — results

The generative LoRA prototype scored 0.696 macro F1 on the balanced 450-image set and 0.734 on the 2,532-image coverage set. It used 1,500 training images and different settings from the frozen-head prototype, so the gap cannot be assigned to LoRA alone. It generated JSON rather than using classification heads. The historical 256² run used H200 and seed 3407.

Source: `confronto_lora.md` (original report, git history), `train_summary.json` (original report, git history).
