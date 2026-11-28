# Phase 1 — results

Scores are macro F1 unless stated otherwise. The evaluation sets and training configurations differ, so the rows are not a controlled architectural ablation. Coverage contains 2,532 images including rare-class supplements; prompt-test is its 1,193-image subset. Public dev is a separate unlabeled set scored by Kaggle.

## Three initial directions

| Direction | Training | Coverage | Prompt-test | Public dev |
|---|---:|---:|---:|---:|
| 24 decoder layers + heads | 11,876 | Not held out | Not held out | 0.80473 |
| LoRA + heads, P11 | 9,344 | 0.7808 | 0.7909 | 0.796 |
| SigLIP + LoRA + heads, first checkpoint | 8,403 | 0.742 | 0.750 | Not recorded here |

The full decoder run trained approximately 5.24 billion parameters, retaining frozen VQ and embeddings. It used focal loss and effective-number weights, 3 epochs, seed 3407 and batch 16. Training took about 165 minutes on A100 80 GB, with 57.1 GB peak memory. Since all labeled images were used, neither coverage nor training data is a held-out evaluation. Public dev submission: 59021011. [Recorded full fine-tuning report](../apertus-8B-fullft-heads/RESULTS.md), [detailed run](../apertus-8B-fullft-heads/results_phase2.md).

P11 used 32 decoder layers, rank-32 LoRA, six heads, 256² images, unweighted cross-entropy, 3 epochs and seed 3407. Its 9,344 training images became 12,491 items per epoch through repetition and mirroring. It trained for 46.7 minutes on H200, with 139 GB peak reserved memory; coverage inference was 13.2 images/s and 22 GB peak allocated. Public dev submission: 59018375. On coverage, the six dimension scores were 0.841 subject, 0.911 background, 0.625 perspective, 0.981 individuals, 0.806 text classification and 0.521 text language. Dutch had only two human-labeled training cells in this split and none of 20 coverage examples was detected. [Recorded P11 report, including per-class results](../apertus-8B-lora-heads/results_full_h200.md).

The first visual-adapter checkpoint used frozen SigLIP 2 and VQ, 49 projected visual tokens, a 24-layer Apertus decoder with LoRA and six heads. The source records 8,403 training images and 941 internal validation images. HF job `6b09af0354072e8b716a1d36` confirms 13,390 augmented items per epoch, two epochs, 1,674 steps and a best internal validation score of 0.7460. Its public-dev inference processed 5,261 images at 13.8 images/s on H200. The reported prompt-test score was 0.750, weighted F1 0.887. [Visual-adapter chronology and per-dimension results](../apertus-8B-vadapter-lora-heads/RESULTS.md).

## Reproducibility evidence

The current phase code is a maintained copy, not a verified snapshot of each historical run. The initial adapter log confirms its `training` split, 256-side encoding, two epochs and `--max-minutes 45`; it does not by itself prove that the copied source and environment are identical. The recovered bundle differs from the current phase in 8 of its 10 bundled files. Dependency revisions and all input bytes still need verification before claiming exact replication. Remote account access and historical artifacts may be needed. See [README](README.md) for launch and split information.
