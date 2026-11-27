# Project context

Consolidated on 2026-11-27 from the official docs, the team docs (Italian), the code and a local data profile.
Official sources win over team docs: `challenge_description.md` > `FAQ.md` > Kaggle rules >
`data/hackapertus-galaxus/{README.md,labeling_guideline.md,metric.py}` > `src/AGENTS.md` > `src/README.md`.
Results live in `EXPERIMENTS.md`, plans in `RESEARCH_DIRECTION.md`.

## 1. Problem formulation

Multi-task image classification: for each product image, choose 1 class in each of 6 dimensions, given optional
catalog context (title, product_type, brand, category, subcategory). Production context can be missing or poor, so
the method must work without it.

| Dimension | Classes (n) | Scored train cells (human ∧ relevant) | Imbalance (largest : smallest) |
| --- | --- | --- | --- |
| subject | 13 | 11,132 | full_product 6,375 : product_grouping 91 |
| background | 3 | 9,683 | white 8,008 : scene_lifestyle 729 |
| perspective | 10 | 7,552 | front_head_on 2,672 : back_angle_right 211 |
| individuals | 2 | 9,689 | none 9,060 : individual_present 629 |
| text_classification | 6 | 10,707 | none 8,887 : certification_seal 60 |
| text_language | 7 | 5,135 | none 1,844 : Dutch 21 |

Full per-class counts: run the profile in §9 or see `data/prototype_splits/distribution.json` ("original").

Key labeling conventions (paraphrased from `labeling_guideline.md`; read the original for edge cases):
- subject: "what does the customer buy?" Food/drink/supplements in their own pack → full_product. Any cut-off
  edge → detail_closeup. multiview vs multipack vs product_grouping vs delivery_scope is decided by identity,
  orientation and size of the objects.
- background: white = no backdrop or ≥ 250 brightness. When unsure between color and scene_lifestyle → scene_lifestyle.
- perspective: find the front face first, then read a clock face from above with the camera at 6 o'clock. There
  are per-group front faces (handle = back, so handle left → right_side; screens; shoes; clothing never
  top_down/bottom_up). This is the noisiest dimension.
- text_classification: only digitally added overlays count, plus the nutrition label (which always wins).
  Printed packaging text → none. On an infographic, use the dominant kind of text.
- text_language: brand names, technical terms and numbers do not count. If tied → first of
  German > English > French > Italian > Dutch. none = no words that count.

## 2. Challenge requirements and constraints

- **Models**: Apertus v1.5 8B/70B (also finetuned or adapted) performs the classification. At runtime it may be
  helped only by open-weights models (encoder, OCR, embeddings). Any model or API (including closed) is allowed in
  development and for pseudo-labels on dev. Apertus can run self-hosted or through the Hack Apertus API (Keera).
  The report must list every model used.
- **Forbidden**: hand-labeling dev or final, shop lookups or reverse image search, training or tuning on final.
- **Data privacy**: no images, csv, labels, passphrase or label-error csv in the repo or in the weights. Delete the
  data after the hackathon.
- **Deployability**: on-prem, air-gapped or sovereign Swiss cloud. The report states which, plus build-time vs
  runtime dependencies.
- **Reproducibility**: the report names the commit and the weights revision. Weights are published by 14:30.
  Organizers may rerun the code, and a score that does not reproduce within its performance group can be
  disqualified.
- **Speed**: plan for ≥ 2 img/s (5,292 final images in ~45 min).
- **Timeline (CET)**: final passphrase Sat 28 Nov 13:00. Upload by ~14:15 and select the submission. The deadline
  is 14:30 (Kaggle + sprintd: public repo, ≤ 2 min video, ≤ 10-slide deck, ≤ 6-page report PDF
  `TeamName_Report.pdf` at the repo root). Pitch at 15:00.
- **Hand-in content**: `technical_report.md` §7 (scores, what mattered, models, resources, commit and weights).
  Setup, training and inference go in the README or report §6, with pinned deps, configs and seeds.
- **Assessment**: performance on final ~30%, scored in bands: above Qwen 9B / between Gemini 3.8 Flash and Qwen /
  below Flash; differences of a few tenths of a percent are not decisive. Innovation 20%, speed and resources 20%,
  insights 15%, code quality 15%. Weighted F1 and per-dimension scores are also looked at. **Bonus**: very good
  perspective, and a csv of suspected train label errors (`image_id,dimension,label,suggested_label`) sent by
  Discord DM and never committed.
- **Compute**: HF $20 per team + $10 coupon per member; Kaggle 30 GPU-h/week per account (T4×2); L40S on HF
  ≈ $1.80/h.

## 3. Evaluation methodology

- Score = mean over the 6 dimensions of the macro F1 (`metric.score`). `metric.evaluate` adds per-class P/R/F1 and
  weighted F1.
- The class set for each dimension is the classes in the labels ∪ the *valid* classes predicted. A predicted class
  that is absent from the labels adds an F1 = 0 term. An empty prediction is wrong but adds no class.
- Only cells with a human label and relevant = true are scored. Relevance rules (applied to human labels only):
  logo, placeholder and energy_label score only subject (+ text_language); infographic → no bg, persp or indiv;
  delivery_scope, detail_closeup and product_grouping → no perspective; scene_lifestyle → no subject or
  perspective; text_language only with an overlay or subject ∈ {infographic, packaging, package_and_product,
  full_product}.
- Submission: header `image_id,subject,…,text_language`, 10,553 rows (dev + final), as in `sample_submission.csv`.
  Before 13:00 Sat, keep the constant final rows.
- Local scoring: build the solution with non-scored cells set to empty (see `challenge_description.md` "Scoring
  locally", or `eval_prompt.solution_frame`). metric.py needs Python ≥ 3.10 and pandas ≥ 2.1.

## 4. Data facts (verified locally 2026-11-27)

- `train.csv` 11,876 images / 7,260 products (median 1 image per product, max 26). `dev.csv` 5,261 / 3,371.
  Final 5,292 / 3,487. No products are shared between parts, but some identical pictures (logos, placeholders,
  reused photos) recur.
- Columns: `image_id, product_id, image_path, product_title, product_type, brand, category, subcategory`, and in
  train, for each dim, `{d}`, `{d}_source` (human/model) and `{d}_relevant`.
- Context fields are almost never empty (title empty in 0.1% of train). The "missing context" robustness
  requirement must therefore be simulated (context dropout); the data will not exercise it.
- 1,594 product types in train. Only 18 dev rows (12 types) have a type that is unseen in train.
- Images: 10,532 jpg + 1,344 png in train. In a 600-image sample: median 0.88 MP, 10% under 0.16 MP, ~10% have
  a min side < 256 px, ~11% have an alpha channel. Docs: ~4% truly transparent; some are 1×1 or thin strips.
  **Always** composite onto white (`images.opaque_rgb_on_white`).
- Label sources: model prelabels sit mostly in non-relevant cells, except **text_language**, where 3,541 relevant
  cells (41% of relevant) are model prelabels (mostly none/English), so they are not scored. A few hundred
  human-labeled cells per dimension are non-relevant (for example 552 in perspective).
- Leakage and eval risks:
  - Split by `product_id` always (multi-image products).
  - Duplicate pictures across parts (logos, placeholders) can inflate local scores slightly.
  - Dev and final labels follow the same human-only scoring, so their distribution is presumably similar to the
    scored train cells (assumption).

## 5. Splits

`src/common/splits.py`: hash-based by product, deterministic, no committed ID lists.

- prompt-dev = bucket 0/10 (1,170 img); prompt-test = bucket 1 (1,193); base = both (2,363).
- coverage = base + "extra" products added (rarest classes first) until each class reaches
  `class_goal = min(30, max(20, 2·total//3), total)` → 2,532 images, 1,513 products.
- `training_rows` = train minus coverage products → 9,344 images.
- Docstring rule: "Train, tune and select checkpoints on these rows only." → any tuning needs an inner split of
  training_rows.
- Side effect: for classes with ≤ ~30 examples the floor of 20 takes nearly all of them. Training keeps
  **Dutch 1**, Italian 17, certification_seal 30, product_grouping 61, text_language other 80, multipack 108.
- `data/prototype_splits/` (not in git; generator script not in repo): `train_1500.csv` (1,311 products, chosen
  from 200 random draws to match the original distribution, TV distance 0.025) and `validation_450.csv` (one
  image per product, label-balanced with subject weighted 3×, **not representative**). Both come from
  training_rows (verified: 0 coverage products, 0 shared products). Seed 42, source sha256 matches the local
  train.csv.
- Qwen-notebook holdout (600 img, seed 3407), reused by `keera_classify.py`: it is mixed (291 of its products are
  in training_rows, 87 in coverage), so it is not a valid holdout for phase-1 models.

## 6. Implementation map

### Shared: `src/common/`
- `prompt.py`: `TAXONOMY` (order fixes the letter codes A…), `CODES`, `context_block` (sanitized, ≤ 200 chars per
  field, "not available" if empty). v1 builders `system_prompt`/`build_messages`/`output_constraint`/`parse_choice`.
  v2 `DESCRIBE`, `QUESTIONS`, `describe_messages`, `question_messages`, `question_constraint` (regex for one
  letter, top-20 logprobs), `parse_question` → (class, probability distribution). Gateway note: use
  `structured_outputs`; `guided_regex` is silently ignored.
- `images.py`: `opaque_rgb_on_white`; `target_size` (area ≈ side², aspect ratio kept and capped at 8, sides
  multiples of 16); `prepared_jpeg`/`prepared_image` (JPEG q92 round-trip, optional mirror); `data_url`.
- `splits.py`: see §5.

### Scripts: `src/scripts/`
- `eval_prompt.py`: zero-shot variant runner on Keera plus a scorer for external prediction CSVs. Specs look like
  `model:layout:answer:side[:nocontext]` or `path.csv`. It caches in `data/runs/`, does a paired product bootstrap,
  and shows coverage/base/test views. Expects `data/train.csv` and `data/metric.py`.
- `keera_classify.py`: older v1 JSON zero-shot runner (holdout or dev, writes a submission). **Broken path**: it
  reads `src/scripts/system_prompt.txt`, which was moved to `src/docs/`.
- `db_download.py`: Kaggle API download into `data/`.
- Deps: `src/requirements.txt` (pandas 3.1.1, pillow 13.0.0); `src/scripts/requirements.txt` adds openai 3.60.0
  and tqdm.

### LoRA pipeline: `src/apertus-8B-benchmark/lora/` (commit ca6a327, never run)
- `apertus_lora.py` (library):
  - `select_rows` ("training" = phase 1, "all" = phase 2).
  - Target = a single JSON with all 6 class names, after a short custom `SYSTEM_PROMPT` + image + `context_block`.
    This is **not** the v2 prompt (allowed: AGENTS lets phase 1 pick the trained format).
  - Value tokens of non-human dimensions are masked to -100, while keys and punctuation stay in the loss.
    `_relevant` is ignored in training.
  - Mirror copies for human left/right perspective images with both text fields = none.
  - `repeat_counts` = min(3, ceil(200/rarest class count)).
  - VQ tokenizer encode, `Sequences`, left-pad `collate`.
- `encode.py`: precomputes VQ codes on GPU, sharded. The cache is `codes_{part}_{side}_shard{i}of{n}.pkl` → dict
  `image_id[|m]` → (h, w, int32 codes).
- `train.py`:
  - torchrun DDP; HF Trainer subclass with answer-only logits.
  - Kaggle `fp16-nf4`: QLoRA nf4 + double quant, fp16 compute, xIELU in fp32, power-of-2 `down_proj` scaling
    calibrated on 8 examples, abort if activations exceed 6e4.
  - HF `bf16`: plain LoRA.
  - LoRA r=32 α=32, dropout 0, on `language_model.*(q|k|v|o|up|down)_proj`; image-token embeddings are not trained.
  - lr 2e-4, cosine, warmup 5%, ≤ 3 epochs, eff. batch 16, seed 3407.
  - `BudgetCallback` shrinks the schedule to fit a wall-clock deadline, so the step count is time-dependent.
  - Snapshots every ~half epoch to `adapters/step-N`.
- `evaluate.py`: bf16 base + `merge_and_unload`, unconstrained greedy `generate` (max 96 tokens), scores
  coverage/base/test with metric.py, writes `eval_*.json` + `predictions_*.csv`.
- `build_notebook.py` → `lora_train.ipynb` (Kaggle T4×2, side 256, `HF_TOKEN` secret, 5.4 h session budget,
  preflight, encode → train; **no eval step**).
- `build_hf_job.py` → `hf_job.py` (`hf jobs uv run --flavor a100-large`, side 384, downloads the Kaggle data
  onto HF compute, trains bf16, evaluates, uploads adapters and logs to a private HF repo).
- Pins: swiss-ai transformers fork `3797303dda74844e3d1f8977ff5518bb91f818b4`, peft 0.22.0, accelerate 1.16.0
  (HF job also pins torch 2.8.0, kaggle 2.2.5). Kaggle torch and bitsandbytes are unpinned. The model revision is
  unpinned.

### Prototype support (added 2026-11-27, uncommitted)
- `select_rows` accepts `prototype_train` / `prototype_validation`. Ids come from `$PROTOTYPE_IDS` (gzip + base64
  JSON) or `$PROTOTYPE_DIR` (the local `data/prototype_splits`). An assert guarantees no coverage product is touched.
- `evaluate.py --part {coverage,prototype_validation}` and `--name`.
- `build_hf_job.py`: H200 in the launch example, `PART=prototype_train` evaluates on prototype_validation and then
  coverage, and `--pack-prototype DIR` prints the secret value. `hf_job.py` was rebuilt.
- Tested on CPU with `--tiny`: encode → train → evaluate on both parts.
- hub 2.4 copies local job scripts to the private bucket `StoryofLegend/jobs-artifacts`. That is why the ids go as
  a secret and not inside the script.

### Not implemented anywhere
Dev/final prediction + submission writer, final decryption flow, checkpoint averaging, flip TTA, calibration or
logit adjustment, logprob class scoring for the trained model, context dropout, 70B or adapter code, a self-hosted
speed benchmark.

## 7. Apertus v1.5 facts that shape design

From `src/README.md` §2 and the memory note; verify against the model card before relying on them.
- No continuous vision encoder. A VQ image tokenizer (codebook 131,072) turns the image into discrete tokens in the
  LM vocabulary (total vocab 266,752; output head 131,072, text only).
- 16 px per token, so 256² → 256 tokens, 384² → 576, 512² → 1,024 (Qwen: 256 at 512²). Accepted range about 256²
  to ~1.96 MP.
- The LM is Apertus 1.0: 32 layers, hidden 4096, xIELU activation. Thinking is a chat-template flag and must be
  off.
- Needs the swiss-ai transformers fork. vLLM support is unverified. CPU VQ encoding takes 6–17 s per image.
- Keera gateway model ids: `apertus-v1-5-8b`, `apertus-v1-5-70b` (+ `-thinking`). It serves base models only.

## 8. Contradictions and discrepancies (unresolved, listed so they are not silently fixed)

1. **Qwen dev score**: 0.917 (challenge_description) vs 0.910 (deep dive, reported orally in `src/README.md`).
   The official doc wins.
2. **Shared prompt vs LoRA format**: AGENTS names v2 `describe:codes` as the shared prompt (zero-shot). The LoRA
   uses a short JSON prompt. This is permitted, but no results doc records the choice yet, although the
   `apertus_lora.py` comment says "the results note it".
3. **HF Jobs path vs team decision**: the memory note records "no HF Jobs, no competition data on HF". `hf_job.py`
   downloads the data onto HF compute and uploads adapters there. One of the two needs confirming.
4. **Stale notebook**: `lora_train.ipynb` is out of sync with `train.py`/`apertus_lora.py` (missing
   `--precision`, `--no-grad-ckpt`, upload flags and the TF32 change). The fp16 path is said to be equivalent.
   Regenerate with `build_notebook.py` before use.
5. **Resolution mismatch**: Kaggle trains at side 256, while `evaluate.py` defaults to 384 and the HF job uses 384.
6. **No eval for the Kaggle adapter**: `evaluate.py` needs bf16 (T4 has none) and merges into a bf16 base. The
   adapter was trained on an nf4/fp16 base with MLP scaling, which may cause a train/inference mismatch
   (suspicion).
7. **Data layout**: `eval_prompt.py`, `AGENTS.md` and `prototype_splits/README.md` assume `data/train.csv` and
   `data/train_images/`. Locally the files are in `data/hackapertus-galaxus/`, which `keera_classify.py` and the
   LoRA `find_data_dir` handle. The prototype README says paths are "relative to data/", but they only resolve
   from `data/hackapertus-galaxus/`.
8. **AGENTS.md phase-1 wording**: "choose method, epochs and settings by comparing on coverage" vs "do not use
   coverage in any way for … epochs, hyperparameters, calibration". `splits.py` says to tune and select
   checkpoints on training_rows. The likely intended reading is: tune on an inner split, then use coverage only to
   compare finished methods. Needs confirming.
9. **Rare-class starvation** in training_rows (§5) works against the README's goal of rare-class balancing in
   phase 1.
10. `keera_classify.py` reads a prompt path that no longer exists.
11. `src/README.md` §8 proposes a file layout (`src/config.py`, `train.py`, …) that differs from the actual
    per-benchmark layout. Treat it as aspirational.
12. Remote branch `feature/sviluppo-nikola` has no commits beyond main (as of 2026-11-27).

## 9. Reproducing the data profile

The scripts used on 2026-11-27 were throwaway. To recompute: load train.csv with `dtype=str,
keep_default_na=False`, use `splits.scored_cells` for the scored mask, and use `split_rows`/`training_rows` for
the splits. Never write ID lists into the repo.
