# Phase 1 — results

Scores are macro F1 unless stated otherwise. The evaluation sets and training configurations differ, so the rows are not a controlled architectural ablation. Coverage contains 2,532 images including rare-class supplements; prompt-test is its 1,193-image subset. Public dev is a separate unlabeled set scored by Kaggle.

## Three initial directions

| Direction | Training | Coverage | Prompt-test | Public dev |
|---|---:|---:|---:|---:|
| 24 decoder layers + heads | 11,876 | Not held out | Not held out | 0.80473 |
| LoRA + heads, P11 | 9,344 | 0.7808 | 0.7909 | 0.796 |
| SigLIP + LoRA + heads, first checkpoint | 8,403 | 0.742 | 0.750 | Not recorded here |

The full decoder run trained approximately 5.24 billion parameters, retaining frozen VQ and embeddings. It used focal loss and effective-number weights, 3 epochs, seed 3407 and batch 16. Training took about 165 minutes on A100 80 GB, with 57.1 GB peak memory. Since all labeled images were used, neither coverage nor training data is a held-out evaluation. Public dev submission: 59021011. Dev prediction ran at 9.55 images/s end to end on the A100 (18.7 GB); the full job took 3 h 21 min, about $8.40 (estimate). The roughly 0.009 gain over P11 is not a matched comparison (27% more images, different adaptation). Source: `RESULTS.md` (original report, git history), `results_phase2.md` (original report, git history).

P11 used 32 decoder layers, rank-32 LoRA, six heads, 256² images, unweighted cross-entropy, 3 epochs and seed 3407. Its 9,344 training images became 12,491 items per epoch through repetition and mirroring. It trained for 46.7 minutes on H200, with 139 GB peak reserved memory; coverage inference was 13.2 images/s and 22 GB peak allocated. Public dev submission: 59018375. On coverage, the six dimension scores were 0.841 subject, 0.911 background, 0.625 perspective, 0.981 individuals, 0.806 text classification and 0.521 text language. Dutch had only two human-labeled training cells in this split and none of 20 coverage examples was detected. Source: `RESULTS.md` (original report, git history), `results_full_h200.md` (original report, git history).

P11 per dimension on prompt-test (1,193 images, four decimals): macro F1 0.7909, weighted F1 0.9151, subject 0.8605, background 0.8939, perspective 0.6197, individuals 0.9900, text classification 0.8050, text language 0.5765. The technical report rounds these to 0.791 / 0.861 / 0.894 / 0.620 / 0.990 / 0.805 / 0.577; some older reports print subject as 0.860 (same value 0.8605, rounded down). Source: `overall.md` (original report, git history); `RESULTS.md` (original report, git history).

The first visual-adapter checkpoint used frozen SigLIP 2 and VQ, 49 projected visual tokens, a 24-layer Apertus decoder with LoRA and six heads. The source records 8,403 training images and 941 internal validation images. HF job `6b09af0354072e8b716a1d36` confirms 13,390 augmented items per epoch, two epochs, 1,674 steps and a best internal validation score of 0.7460. Its public-dev inference processed 5,261 images at 13.8 images/s on H200. The reported prompt-test score was 0.750, weighted F1 0.887. Prompt-test dimensions of this first checkpoint: 0.828 subject, 0.803 background, 0.586 perspective, 0.960 individuals, 0.772 text classification, 0.550 language; end-to-end throughput without OCR 13.8 images/s on H200. Training: 31 min (58 min job), about $4.80 (estimate). Source: `RESULTS.md` (original report, git history), `RISULTATI.md` (original report, git history).

## H200 benchmark of the three 8B models on public dev

All 5,261 dev images, one H200 on HF Jobs, batch 32, product context, BF16 decoder, model loading excluded; peaks are PyTorch allocated memory. The benchmark labels the full-decoder model "P12"; it is the decoder fine-tuning run above, not the phase-2 P12 (LoRA + heads at 512²).

| Model | Seconds on dev | Images/s (batch 32) | Images/s (batch 1) | Peak GPU (batch 32) | Inference cost per 100,000 images at $5/h | Public dev |
|---|---:|---:|---:|---:|---:|---:|
| 24 decoder layers + heads | 359.7 | 14.63 | 11.72 | 18.7 GiB | $9.49 | 0.80473 |
| LoRA + heads (P11) | 396.3 | 13.27 | 11.07 | 22.2 GiB | $10.47 | 0.796 |
| SigLIP 2 + VQ + OCR + LoRA + heads (`final-abt`, phase 2) | 760.9 | 6.91 | 5.74 | 79.8 GiB | $20.10 | 0.83177 |

The 6.91 images/s of `final-abt` is the "earlier single-encoder model" speed quoted in the technical report. Source: `docs/benchmark_dev_h200.md`.

## Reproducibility evidence

The current phase code is a maintained copy, not a verified snapshot of each historical run. The initial adapter log confirms its `training` split, 256-side encoding, two epochs and `--max-minutes 45`; it does not by itself prove that the copied source and environment are identical. The recovered bundle differs from the current phase in 8 of its 10 bundled files. Dependency revisions and all input bytes still need verification before claiming exact replication. Remote account access and historical artifacts may be needed. See [README](README.md) for launch and split information.
