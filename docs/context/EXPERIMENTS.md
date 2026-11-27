# Experiments knowledge base

Status as of 2026-11-27 (evening). Every number below is copied from the source file named next to it. Nothing
here is a Kaggle leaderboard score unless it says so. **No Apertus model has been trained or submitted yet.**

Legend: **[measured]** run happened and is reported in a repo file · **[reference]** organizer material ·
**[not run]** code or plan exists, no result.

## 0. Evaluation sets used so far

All are subsets of `train.csv`, split by `product_id` (definitions in `src/common/splits.py`; sizes recomputed
locally on 2026-11-27).

| Set | Images | Products | Origin | Use |
| --- | --- | --- | --- | --- |
| prompt-dev | 1,170 | – | hash bucket 0/10 | prompt iteration (already "spent") |
| prompt-test | 1,193 | 713 | bucket 1/10 | confirmation; used once for the v2 check |
| base | 2,363 | 1,438 | prompt-dev + prompt-test | natural class distribution |
| coverage | 2,532 | 1,513 | base + extra products for rare classes | **team's main comparison set** (not representative of leaderboard) |
| training_rows | 9,344 | – | train minus coverage products | phase-1 training pool |
| "first 600 of prompt-dev" | 600 | 363 | first 600 rows in split order | early prompt runs |
| Qwen-notebook holdout | 600 | 378 | seed-3407 product shuffle | Qwen reference + `keera_classify.py`; 291 of its 378 products are in training_rows, 87 in coverage, so it is **not** a clean holdout for phase-1 models |
| prototype train_1500 / validation_450 | 1,500 / 450 | 1,311 / 450 | drawn from training_rows (seed 42) | smoke tests; validation is label-balanced, not representative |

Numbers on different sets are **not comparable** with each other.

## 1. Reference: Qwen 3.5 9B finetune  [reference]

Source: `reference/finetune_qwen35_reference.ipynb` (outputs of the organizers' run), `challenge_description.md`.

- Dev (public LB): finetuned Qwen 3.5 9B **0.917**; Gemini 4 Argon zero-shot 0.834; Gemini 3.8 Flash zero-shot 0.806.
  `src/README.md` §1.6 records that organizers said 0.910 in the deep dive (oral, unverified).
- Notebook holdout (600 img): overall macro F1 **0.899**, weighted 0.951. Per dim macro: subject 0.875,
  background 0.911, perspective 0.859, individuals 0.989, text_classification 0.842, text_language 0.916.
- Weakest classes (holdout, tiny n): certification_seal 0.67 (n=2), accessories 0.69 (n=17), top_down 0.71 (n=30),
  multiview 0.71 (n=7), text other 0.75 (n=18), multipack_trade_unit 0.77 (n=8), promotional_text 0.78 (n=10),
  left_side 0.80 (n=12).
- Recipe: `unsloth/Qwen3.5-9B` bf16, LoRA r=32 α=32 all-linear on vision+language, LR 2e-4 (vision 5e-5),
  AdamW8bit, cosine, warmup 0.05, 3 epochs, eff. batch 16, seed 3407; 512² area, JSON answer, value tokens of
  `model`-source labels masked to -100; mirror augmentation for left/right perspective images without text (+1,834
  rows); rare-class row repetition min(3, ceil(200/count)) (+1,196 rows). 73 min, 38.2 GB peak on 1×H100.
- Inference: vLLM 0.26.0, greedy, JSON-schema guided, prefix caching: 13.3 → 28.7 img/s on H100.

## 2. Apertus zero-shot prompt experiments  [measured]

Setup common to all: Keera gateway (CSCS), base `apertus-v1-5-8b` / `apertus-v1-5-70b`, 512² images prepared by
`src/common/images.py`, temperature 0, constrained output. Code: `src/scripts/eval_prompt.py` +
`src/common/prompt.py`. Raw answers were written to `data/runs/` on the machine that ran them (**not present on
this machine**).

### 2.1 Variants

| Spec | Description |
| --- | --- |
| `minimal:json` (v1 short) | short system prompt with class names, one JSON answer |
| `full:json` / `full:codes` (v1) | long decision rules (`prompt.RULES`) in system prompt |
| `shared:json` | team prompt `src/docs/system_prompt.txt` as system message, JSON |
| `split:names` / `split:codes` (v2) | one question per dimension after the image, one line per class |
| `describe:codes` (**v2, shared default**) | 1 description request (≤160 tok) + 6 per-dimension 1-letter questions → 7 requests/image; letter logprobs give class probabilities |

### 2.2 Coverage set, 2,532 images — main comparison
Source: `src/apertus-8B-benchmark/prompt_eval_coverage.md`.

| Variant | Macro F1 | Weighted | subj | bg | persp | indiv | text_cls | lang | img/s (gateway) |
|---|---|---|---|---|---|---|---|---|---|
| 8B v1 short | 0.316 | 0.632 | 0.076 | 0.411 | 0.087 | 0.734 | 0.192 | 0.397 | 9.7 |
| 8B shared prompt | 0.229 | 0.467 | 0.065 | 0.268 | 0.098 | 0.581 | 0.213 | 0.151 | 9.6 |
| **8B v2 describe** | **0.528** | 0.705 | 0.471 | 0.580 | 0.203 | 0.904 | 0.527 | 0.485 | 3.8 |
| 70B shared prompt | 0.304 | 0.616 | 0.076 | 0.397 | 0.117 | 0.661 | 0.275 | 0.296 | 4.1 |
| **70B v2 describe** | **0.583** | 0.739 | 0.546 | 0.631 | 0.274 | 0.899 | 0.618 | 0.529 | 2.8 |

Same variants on prompt-test (clean view, 1,193 img): 8B v2 0.502, 70B v2 0.590. On base: 0.513 / 0.587.

Paired product bootstrap (200 resamples), coverage: 8B v2 − 8B v1 short +0.212 [+0.193, +0.229];
70B v2 − 8B v2 +0.055 [+0.041, +0.069]; 8B shared − 8B v1 short −0.087 [−0.103, −0.073].

Prompt tokens per image: v1 short 1,386; shared 2,769; v2 describe ~10,011 (7 requests).

### 2.3 Earlier run, first 600 prompt-dev images
Source: `src/apertus-8B-benchmark/prompt_eval.md`, `prompt_eval_per_class_600.md`. Best: 8B v2 describe 0.521,
70B v2 describe 0.587. v1 `codes` variants were much worse than v1 `json` (0.138–0.164 vs 0.326–0.327). v2 was
**chosen on this split**, so these numbers are optimistic.

### 2.4 Subject-only diagnosis (103 images, ≤8 per class; not comparable)
Source: `prompt_eval.md`. Long rules in the system prompt collapsed subject to full_product (8B macro 0.01).
Removing rules → 0.39; one line per class → 0.41; describe-then-answer → 0.54 (8B) / 0.63 (70B). Product context
did **not** help here (8B describe: 0.54 without vs 0.49 with). The no-context ablation on the full v2 has **not**
been run.

### 2.5 Per-class observations (coverage, 8B v2 / 70B v2 F1)
Source: `prompt_eval_coverage.md`.

- Easy for zero-shot: energy_label 0.94/0.99, logo 0.82/0.89, placeholder 0.81/0.85, individuals none 0.99,
  background white 0.91.
- Near zero: accessories 0.00/0.05; back_angle_left 0.00/0.00; back_angle_right 0.00/0.00; bottom_up 0.05/0.20;
  text_language other 0.06/0.38.
- Systematic biases:
  - text_language `none` recall 0.07 (8B) / 0.11 (70B): models almost never say "no language".
  - 70B gets front_angle_left completely wrong (precision 0.00, recall 0.00), while 8B over-predicts it (recall
    0.74, precision 0.24). Both confuse left and right, which may mean the models read the clock-face convention
    mirrored (hypothesis).
  - background `color` 0.18/0.24 (confused with white).
  - delivery_scope recall 0.07/0.20.
- In 8B and 70B v2, nutritional_info, promotional_text and certification_seal have high recall and low precision
  (they are over-predicted).

### 2.6 Teacher–student agreement, zero-shot 70B vs 8B  [measured, no training]
Source: `src/apertus-8B-benchmark/teacher_student_zeroshot.md`.

- "Only 70B right" vs "only 8B right" (coverage): subject 13.9% vs 4.9%; perspective 23.4% vs 12.0%;
  text_language 6.6% vs **10.9%** (8B better); text_classification 9.3% vs 8.8%.
- Perspective: only 28.1% of cells both right; 23.1% both wrong with different answers.
- 70B confidence is informative: subject accuracy 76.6% when p≥0.9 vs 32.9% when p<0.6; text_cls 91.9% vs 42.9%;
  perspective is only 63.4% even at p≥0.9.
- Implication (hypothesis): zero-shot 70B soft labels are a weak teacher for perspective/text_language; any
  distillation needs a finetuned teacher or confidence filtering.

### 2.7 Gateway throughput (synthetic, possibly cached)
Source: `prompt_eval.md`. 8B: 10.7 req/s @4 parallel, 22.5 @16; 70B: 6.8 @4, 19.2 @16. Shared gateway, not a
self-hosted measurement.

## 3. Apertus 8B LoRA

Code: `src/apertus-8B-benchmark/lora/` (commit `ca6a327`, plus uncommitted prototype support of 2026-11-27).
Pipeline details: `PROJECT_CONTEXT.md` §6.

### L1 prototype-256  [measured]   2026-11-27, user's track

Source: `src/apertus-8B-benchmark/lora/results/prototype-256/*.json` (copied from the HF repo), job log, `prototype-256/eval_{coverage,prototype_validation}.json` in the private repo
`StoryofLegend/apertus-8b-galaxus-lora`. The final adapter was evaluated inside the job.

| Set | Macro F1 | Weighted | subj | bg | persp | indiv | text_cls | lang |
|---|---|---|---|---|---|---|---|---|
| coverage (2,532) | **0.734** | 0.879 | 0.712 | 0.910 | 0.586 | 0.953 | 0.679 | 0.565 |
| base (2,363) | 0.721 | 0.885 | 0.703 | 0.909 | 0.588 | 0.960 | 0.680 | 0.489 |
| prompt-test (1,193) | 0.712 | 0.881 | 0.672 | 0.902 | 0.583 | 0.947 | 0.691 | 0.473 |
| validation_450 (balanced) | 0.696 | 0.747 | 0.642 | 0.904 | 0.484 | 0.942 | 0.661 | 0.546 |

- On coverage it beats 8B zero-shot v2 by +0.21 (0.528 → 0.734) and 70B zero-shot v2 by +0.15. This is with 16%
  of the phase-1 data and 256². No bootstrap interval yet.
- Weakest classes on coverage: text_language other 0.06 (n=30), certification_seal 0.24, product_grouping 0.26,
  Dutch 0.33, back_angle_left 0.36, accessories 0.38, back_angle_right 0.40, multipack 0.41, text other 0.41,
  top_down 0.50.
- 21/2,532 invalid answers on coverage and 9/450 on validation (unconstrained JSON). They count as wrong.
- The training loss reached ~0.003 from epoch 2, which means memorization. Snapshots are every 123 steps (half an
  epoch) for an epoch ablation.
- Speed (H200): VQ encode ~6.3 img/s per process, ×3 processes ≈ 19 img/s. Training: 741 steps in 16 min.
  Inference with HF `generate` at batch 32: 18 img/s (not optimized, excludes encoding).
- The upload of `adapters/final` failed (PEFT README metadata); fixed afterwards in `train.py`. Use
  `adapters/step-00738`, 3 steps before the end with lr ≈ 0.

Run details:
- **Trained on**: `data/prototype_splits/train_1500.csv` (1,500 images, 1,311 products, inside training_rows):
  3,952 examples per epoch with 251 mirrors and rare-class repeats. Note that `RARE_CLASS_TARGET=200` was sized for
  11k rows, so on 1,500 rows most rare-class rows repeat 3×.
- **Evaluated on**: `validation_450` (label-balanced, not representative) and `coverage` (2,532; comparable with
  zero-shot 8B 0.528 and 70B 0.583).
- **Setup**: HF Job `StoryofLegend/6b09964d6c5d61fff1893dfe` (`apertus8b-lora-prototype-256`, 27/11 14:40, 26m 38s; an earlier launch `6b0994fa…3c7f` failed in training after 4 min), H200, bf16 LoRA
  r=32 α=32 on q,k,v,o,up,down of the LM; no gradient checkpointing; side 256 (~256 VQ tokens); JSON answer;
  human-only value loss; lr 2e-4 cosine; 3 epochs at most, cut to fit `TRAIN_HOURS=2`; batch 8 × accumulation 2;
  seed 3407. Outputs go to the private model repo `StoryofLegend/apertus-8b-galaxus-lora`, prefix
  `prototype-256/`.


## 4. Apertus 70B, visual adapter  [not run]

`src/apertus-70B-benchmark/` and `src/apertus-vadapter-benchmark/` contain only `.gitkeep`. The 70B has only the
zero-shot runs of §2.

## 5. Ideas listed in src/README.md with no experiment yet  [not run]

Continuous visual adapter (SigLIP 2 / DINOv3 → projector), 70B→8B distillation, logit adjustment / balanced
sampling / per-class bias calibration, OCR + language ID, LoRA soup + flip TTA, layer probing + truncated model
with heads, confidence cascade, looped transformer, GRPO with a macro-F1 reward, VQ token pruning, context dropout.

## 6. Reproducibility notes

- Zero-shot runs: `.venv/bin/python src/scripts/eval_prompt.py <variants> --split coverage --concurrency 32
  --bootstrap 200 --classes`. Needs `LLM_API_KEY` for Keera, and `data/train.csv`, `data/metric.py` and
  `data/train_images/` **directly under `data/`**. On this machine the files are in `data/hackapertus-galaxus/`, so
  the script would not find them as is.
- Results are cached by spec + prompt-wording hash in `data/runs/`. Changing prompt wording invalidates the cache.
- Gateway speeds depend on the load of the shared service, so they are not hardware numbers.
- Prototype splits: `data/prototype_splits/distribution.json` gives seed 42, source sha256 `936a330d…` (it matches
  the local train.csv) and the selection rules. The script that generated them is **not in the repo**.

## Template for new entries

```
### <id> <short name>  [measured|not run]   date, owner
Trained on: <training_rows | all train | prototype_1500 | none>   Evaluated on: <coverage | dev LB | …>
Model / adapter / prompt format / resolution / key hyperparameters / commit
Result: macro, weighted, per-dim (source file)   Speed: img/s on <hardware>
Takeaway (one line), caveats
```
