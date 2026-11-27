# CLAUDE.md: Bymo × Hack Apertus Zurich, Galaxus image classification

Start every session here. Load the detailed context only when the task needs it:

| File | When to read |
| --- | --- |
| `docs/context/PROJECT_CONTEXT.md` | task, rules, data, metric, splits, code map, known discrepancies |
| `docs/context/EXPERIMENTS.md` | any question about results, baselines or which set a number comes from |
| `docs/context/RESEARCH_DIRECTION.md` | planning the next experiment, priorities and open questions |
| `src/AGENTS.md` (Italian) | **binding team rules**: shared prompt, two-phase training, Keera and Kaggle usage |
| `src/README.md` (Italian) | the team's original strategy doc (ideas, not results) |

## Objective

Build an **Apertus v1.5** method (8B or 70B, may be finetuned or adapted) that labels each Galaxus product image
with one class in each of 6 dimensions (41 classes). It should match or beat the finetuned Qwen 3.5 9B (dev **0.917**)
and stay fast and cheap (≥ 2 img/s on final; cost per 100k images is judged). Kaggle competition
`hackapertus-galaxus`. Kaggle team = Hack Apertus team name **Bymo**.

## Hard constraints (official: `challenge_description.md`, `FAQ.md`, Kaggle rules)

- **Apertus does the classification at runtime.** Other runtime models must be open-weights and only do
  supporting tasks (encoder, OCR). No closed or proprietary API at runtime. Anything is allowed during
  development, including closed models and pseudo-labels on dev.
- **Never** label dev/final images by hand, look products up in the Galaxus shop or by reverse image search, or
  train/tune on final images. Train labels may be cleaned, also by hand.
- **Data never enters git or published weights**: no images, csv, labels, ID lists, passphrase or label-error list.
  `.gitignore` excludes `data/` and `*.csv`; check `git status` before every commit.
- Do **not** decrypt or touch `final_encrypted.bin` before the passphrase is published (Sat 28 Nov 13:00 CET).
- Deadline Sat 28 Nov **14:30 CET** (Kaggle + sprintd). Upload by ~14:15 and **select** the submission. 20
  submissions per UTC day. Publish the weights, then record the commit and weights revision in
  `technical_report.md` §7.
- Submission: one row per dev+final image, exact class names, lowercase `none`, no empty cells.
- Do not submit to Kaggle, make notebooks public, push or launch paid jobs without an explicit request.

## Metric in one line

For each dimension, macro F1 over (classes present in the labels ∪ valid classes you predicted). The score is the
mean over the 6 dimensions. Only `human` + `relevant` cells count. Predicting a class that never occurs costs F1;
rare classes weigh as much as frequent ones. Judges also look at weighted F1. Score locally with
`data/hackapertus-galaxus/metric.py` (`score`, `evaluate`).

## Current status (2026-11-27 evening)

- Zero-shot is measured but far off the target: on coverage, 8B v2 scores 0.528 and 70B v2 0.583, while the Qwen
  holdout reference is 0.899. Details in `EXPERIMENTS.md`.
- First 8B LoRA (prototype: 1,500 images, 256², H200) scores **0.734** on coverage (zero-shot 8B 0.528); see EXPERIMENTS L1.
- The 70B and visual-adapter benchmark folders are empty.
- Branch `apertus-8b-lora`; `main` holds the zero-shot work.

## Evaluation principles

- Splits come from `src/common/splits.py` (hash-based, by product). Phase 1: train on `training_rows` (9,344),
  compare on `coverage` (2,532). Phase 2: retrain on all 11,876 with the frozen recipe, checked only on the Kaggle dev
  LB. See `src/AGENTS.md`.
- Never evaluate on images a model trained on. For each score, state the training data, the evaluation set and
  its source file.
- `coverage` is enriched with rare classes: use it to compare methods, not to predict the LB. Use `base` /
  `prompt-test` views for the natural distribution.
- Report macro F1, weighted F1, per-dimension and per-class F1, plus speed (img/s, hardware). Use a paired
  product bootstrap for differences (`eval_prompt.py --bootstrap`).
- Use the same prompt/answer format and the same image preparation (`src/common/images.py`) in training and
  inference.

## Conventions

- Team docs, result reports and commit messages are in **Italian**, in the form `type: descrizione`
  (`feat`, `docs`, `chore`). Write result reports as `.md` in the matching benchmark folder.
- Put shared code in `src/common/` and experiments in `src/apertus-{8B,70B,vadapter}-benchmark/`. Document the
  dependencies, config, data, command and results of each experiment so it can be reproduced.
- Secrets: `LLM_BASE_URL`, `LLM_NAME`, `LLM_API_KEY` from env or `.env` (git-ignored). Never print keys.
- Local `.venv`: CPU-only torch plus the swiss-ai transformers fork. GPU work runs on Kaggle (T4×2) or HF.
- Do not run training, model downloads, dependency installs or paid jobs unless explicitly asked.
- Local data lives in `data/hackapertus-galaxus/`, but `eval_prompt.py` expects `data/train.csv` (see the
  discrepancies in PROJECT_CONTEXT §8).

## Maintaining this context

- New result → add an entry to `EXPERIMENTS.md` (template at the bottom) with its source file. Never write a
  number without a source.
- Decision taken → move it from "recommendation" to "decision" in `RESEARCH_DIRECTION.md`, with the date.
- Discrepancy fixed → update PROJECT_CONTEXT §8.
- Keep this file under ~120 lines and put detail in the files under `docs/context/`.
