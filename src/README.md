# Source code — BYMO, Hack Apertus × Galaxus

We adapt **Apertus 1.5 8B** to classify Galaxus product images along six dimensions (subject, background,
perspective, individuals, text classification, text language), scored as the mean of per-dimension macro F1.

The delivered model keeps Apertus' native VQ image tokens. It adds two frozen open-weights encoders: **SigLIP 2**, and
**DINOv3** fused into the SigLIP stream through a zero-initialised residual. Both reach the decoder as 49 continuous
tokens. OCR text (docTR) goes into the prompt. Six classification heads with spatial readouts sit on a 24-layer
Apertus decoder trained with LoRA.

| | Value |
|---|---|
| Public dev macro F1 | **0.84215** (organiser references: Gemini 3.8 Flash 0.806, fine-tuned Qwen 3.5 9B 0.917) |
| Delivery inference | 11.7 images/s on one H200 (merged LoRA, TF32, selective perspective TTA) |
| Weights | [`weights/apertus-8B-mixed-lora-heads-final/`](../weights/apertus-8B-mixed-lora-heads-final/) (Git LFS) |

The full story is in the [paper](../paper/paper.pdf) and the [technical report](../technical_report.md).

## How the work is organised

The work went in stages, each motivated by the previous one. Every training stage is a **standalone working root**:
its own `src/` holds the code exactly as needed for that stage, plus a `README.md` with launch commands and a
`RESULTS.md` with the measured numbers and their sources. Run commands from inside the stage folder; put the
authorised competition data in its `data/` folder (a `data` link to the repository's `data/` works and is
git-ignored).

| Folder | What it contains |
|---|---|
| [`db_exploration/`](db_exploration/) | Dataset analyses that shaped the design and the evaluation: rarity of classes and resolution of the local test split, a context-only baseline, label consistency within a product. |
| [`plain-benchmark/`](plain-benchmark/) | Zero-shot benchmark of Apertus 8B and 70B through the hackathon API (prompt design, results). |
| [`micro-training/phase-1`](micro-training/phase-1/) | Classification heads on frozen Apertus features (layer choice, loss choice). |
| [`micro-training/phase-2`](micro-training/phase-2/) | Generative LoRA on a 1,500-image split. |
| [`micro-training/phase-3`](micro-training/phase-3/) | Prototype of a continuous SigLIP 2 visual adapter next to the VQ tokens. |
| [`training/phase-1`](training/phase-1/) | Three directions: decoder fine-tuning with heads, LoRA with heads (P11), LoRA with heads and a SigLIP 2 visual adapter. |
| [`training/phase-2`](training/phase-2/) | Spatial readouts and 2D attention pooling (P12), OCR, the `final-abt` visual adapter, synthetic text (P13), larger encoder (P14), all data (P15), trainable SigLIP LoRA. |
| [`training/phase-3`](training/phase-3/) | DINOv3 and the **delivered mixed model**, its training and its fast inference path. |
| [`inference/phase-3-model/`](inference/phase-3-model/) | Scripts that predict the final images with the delivered model and build the Kaggle file: on Modal (`final_modal.sh`) or on Hugging Face Jobs with your own account (`final_hf.sh`). |
| [`scripts/`](scripts/) | Shared tools: Kaggle file assembly (`final_submission.py`), ensembles, test-time augmentation evaluation, label-error search, teacher targets, data download, report PDF. |
| [`demo/`](demo/) | Local web demo (FastAPI + React) that pre-fills the "image type" panel of a product page with the models' answers. |

[`training/REPRODUCIBILITY.md`](training/REPRODUCIBILITY.md) lists what is needed to reproduce each stage and the
evidence that is still missing.

## Quick start: final predictions with the delivered model

```sh
git lfs pull
export FINAL_PASSPHRASE='...'                        # never write it into a file
bash src/inference/phase-3-model/final_modal.sh      # Modal, one H200
# or, with your own Hugging Face account:
export HF_TOKEN=... KAGGLE_API_TOKEN=... UPLOAD_REPO=you/private-repo
bash src/inference/phase-3-model/final_hf.sh
```

## Rules we follow

- Only Apertus and open-weights supporting models run at inference time; no closed model or API.
- Competition images, labels, image ids and predictions never enter the repository; tokens and passphrases live
  only in environment variables or job secrets.
