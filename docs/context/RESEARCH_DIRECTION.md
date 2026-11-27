# Research direction

Labels used here: **[decision]** agreed by the team, with source · **[recommendation]** proposed in this context
file, not agreed · **[hypothesis]** untested belief · **[open]** needs an answer.

Time budget: about 20 h remain from the evening of 2026-11-27. Final inference runs Sat 13:00–14:15. This
favours few, decisive experiments over a broad search.

## 1. Confirmed decisions

| Decision | Source |
| --- | --- |
| Apertus v1.5 classifies; open-weights helpers only at runtime | challenge rules |
| v2 `describe:codes` at 512² is the shared **zero-shot** prompt; `system_prompt.txt` is kept only for comparison | `src/AGENTS.md` |
| Two-phase training: phase 1 on `training_rows`, compared on `coverage`; phase 2 retrained on all train with the frozen recipe, checked only on dev LB; fall back to phase 1 if phase 2 is clearly worse or late | `src/AGENTS.md` |
| Same prompt format in training and inference, chosen in phase 1 | `src/AGENTS.md` |
| ~~Kaggle T4×2 only, no HF Jobs~~ → **superseded 2026-11-27 evening**: the user's LoRA track (README §4.5) runs on **HF Jobs, H200, bf16 LoRA, 256²**. First a single training on `data/prototype_splits` (train_1500 → validation_450 + coverage). Checkpoint averaging rather than 3 separate trainings; flip TTA and inference later | chat with the user, 2026-11-27 |
| Prototype ids reach the job only as the HF secret `PROTOTYPE_IDS` (ids only, packed by `build_hf_job.py --pack-prototype`); no csv or id list in the repo or the bucket | implementation, 2026-11-27 |
| Answer format of the first LoRA: the committed JSON of class names (unchanged; letter codes are still an open option) | implicit, the run uses the existing format |

## 2. Where we stand: the gap

- Best measured Apertus result: 70B zero-shot v2 **0.583**, 8B **0.528** (coverage). The Qwen reference holdout is
  **0.899**. Zero-shot prompting cannot close a 0.3–0.4 gap: **supervised finetuning is required** (inference from
  the gap, not tested).
- The biggest unknown on the critical path: **how close a LoRA on Apertus 8B (discrete VQ image tokens) gets to
  Qwen**. Every other idea is tuning around this number, and it does not exist yet.

## 3. Known bottlenecks and risks

1. **Perception through discrete VQ tokens** [hypothesis, supported by literature in `src/README.md` §2]. Fine
   detail and in-image text may stay weak after a LoRA. This would hurt text_classification, text_language and
   perspective most.
2. **Inference throughput on our own hardware** [open, high risk]. A finetuned model cannot run on the Keera
   gateway, so it must be self-hosted. At 512², Apertus uses 1,024 image tokens per image, 4× more than Qwen's 256.
   VQ encoding is slow on CPU (6–17 s/image, memory note), so it must run on GPU. vLLM support for
   `Apertus1p5ForConditionalGeneration` is unverified. We have not measured that ≥ 2 img/s for 5,292 images is
   reachable on T4×2 / L40S with transformers. **Time a dry run on dev before Saturday 13:00.**
3. **fp16 on T4** [measured, bad sign]: T4 has no bf16. In the bf16 H200 run of 2026-11-27, the calibration
   probe measured max |hidden| **119,808**, above the fp16 maximum of 65,504. The MLP scaling of `patch_mlp` only
   rescales the MLP input, and `train.py` aborts in fp16 when hidden > 6e4. The Kaggle T4 fp16 path is therefore
   expected to fail as is, for training and for inference. On T4, inference needs a fix (for example residual-stream
   scaling) or other hardware.
4. **Rare classes nearly absent from phase-1 training** [measured]. Coverage takes almost every example:
   text_language Dutch keeps **1** training example (20 of 21 are in coverage), Italian 17, certification_seal 30,
   product_grouping 61. Phase-1 scores on these classes underestimate phase 2. Phase-1 recipe choices for rare
   classes rest on almost no data.
5. **text_language supervision** [measured]: 3,541 relevant cells (41%) are unchecked `model` prelabels, mostly
   none/English. They are not scored. Masking them, as the Qwen recipe does, leaves 5,135 scored examples. The `none`
   class has very low recall in zero-shot (0.07–0.11).
6. **Perspective is noisy by design** (guideline: judgment for uncovered product groups). Zero-shot models confuse
   left and right. Flip TTA and mirror augmentation must swap left↔right labels correctly (front_angle_left ↔
   front_angle_right, left_side ↔ right_side, back_angle_left ↔ back_angle_right). Mirroring images with text
   corrupts text cues.
7. **Inner validation for tuning** [open]: `splits.py` says to "train, tune and select checkpoints" on
   training_rows only. Checkpoint choice, calibration biases and thresholds therefore need an inner split of
   training_rows. `prototype_splits/validation_450` is label-balanced, which makes it a poor target for
   calibration (it would push predictions toward rare classes).
8. **LoRA pipeline gaps** [measured by code review, PROJECT_CONTEXT §6/§8]:
   - The trained format is an unconstrained JSON of class names, so there are no class probabilities. Calibration,
     TTA averaging and cascades need constrained decoding or per-class scoring added.
   - A Kaggle-trained adapter has no evaluation or prediction path: `evaluate.py` assumes bf16.
   - There is no dev/final submission writer.
   - The notebook is stale.
   - Train resolution (256) and eval default (384) differ.
9. **Compute budget**: HF $20 team + $10 per member; Kaggle 30 GPU-h/week per account, 12 h per session. A 70B
   QLoRA (~H100) is the largest possible cost.

## 4. Unresolved technical questions

1. LoRA 8B quality: macro F1 on coverage at 256² vs 512²? [open, highest priority]
2. Answer format for the trained model: one prompt with 6 letters (one generation) vs 6 questions vs classification
   heads? This affects speed, calibration (class probabilities) and the "Apertus classifies" rule.
3. Does product context help after finetuning? Context dropout for robustness is a stated production requirement.
   Zero-shot evidence says context did not help subject (103-image probe only).
4. Does training on text_language prelabels help or hurt?
5. Is the 70B worth it as a finetuned teacher, given cost and time? The zero-shot 70B is a weak teacher for
   perspective and text_language (`teacher_student_zeroshot.md`).
6. Can a continuous open encoder (SigLIP 2 / DINOv3) be wired into Apertus 1p5 in the time left? Does it count as
   "Apertus classifies"? **Ask on Discord** before building (README §4.1 and §5.1 say so).
7. Achievable img/s and VRAM for the final pipeline on the hardware we will use Saturday.
8. Kaggle-only vs HF Jobs: the memory note says Kaggle only, while `hf_job.py` exists for HF. Which is allowed?
   This also decides whether bf16 training and 384² are available.
9. Does training an nf4/fp16 QLoRA and then merging into a bf16 base for inference degrade results? Alternative:
   serve the unmerged adapter on the same quantized base.

## 5. Candidate approaches, ranked for the remaining time  [recommendation]

Ranked by (expected score gain × probability of working by Saturday 13:00) and then judging value. Not decided.

| # | Approach | Why | Cost / risk | Judging axis |
| --- | --- | --- | --- | --- |
| A | **8B QLoRA baseline**, the pipeline as committed (single-pass JSON of class names, human-only loss, mirror + rare-class repeat; a port of the Qwen recipe). A 6-letter-code target is a cheap variant that yields class probabilities | creates the number every other decision depends on; it is the fallback submission | the pipeline exists but is untested; fp16/T4 risk | performance |
| B | **End-to-end timed dry run on dev** (VQ encode + inference + submission writer) | protects the deadline; speed is 20% of the grade | low cost, high value | speed, practicality |
| C | **Decision-time calibration**: read class probabilities, per-class log-prior / bias tuned for macro F1 on an inner validation split; never predict classes absent from a dimension | cheap, typically several points on macro F1 with imbalance [hypothesis] | needs a representative inner validation | performance, insight |
| D | **Flip TTA for perspective** (+ mirror training) with correct label swaps | targets the perspective bonus; zero added training | ×2 inference cost unless only applied to images without text or on uncertain cases | performance, bonus |
| E | **Resolution ablation** 256² / 384² / 512² for the LoRA | 4× speed difference between 256² and 512² | one extra training or eval per point | speed, insight |
| F | **Pseudo-labels on dev** from the best development model (the finetuned Apertus, or an ensemble), then retrain | dev has the same distribution as final; explicitly allowed | one extra training cycle; may confirm its own errors | performance, innovation |
| G | **OCR + language ID** hints for text_language / text_classification | attacks the weakest dimensions and the discrete-token weakness | adds runtime models (must be open) and latency | performance, innovation |
| H | **Checkpoint averaging / LoRA soup** | free at inference | small gain [hypothesis] | performance |
| I | **Layer probing → truncated model + heads, early exit** | strong speed and innovation story; probing is also a cheap insight plot | needs rule confirmation; engineering time | innovation, speed, insight |
| J | Continuous visual adapter, 70B teacher distillation, GRPO, looped transformer, VQ token pruning | high innovation | unlikely to finish and validate by Saturday; also requires rule checks | innovation |

Notes:
- A teacher for F could also be the organizers' open-weights Qwen recipe, retrained by us. Rules allow any model
  in development to create training targets. Whether the team wants this, and how it reads for "innovation", is a
  team decision [open].
- "Insights" counts 15%. Every step above should be logged as an ablation with a bootstrap interval.

## 6. Missing experiments (cheapest first)

0. Regenerate `lora_train.ipynb` from `build_notebook.py`, then add an inference path that works on the hardware
   we have (fp16 on T4, or bf16 on L40S/A100), with constrained or per-class scoring and a submission writer.
1. Smoke test of the LoRA pipeline: fp16 NaN, step time, memory, VQ cache throughput on T4.
2. Inference speed of the base 8B on our own GPU: VQ encode + prefill at 256² and 512², batch size sweep.
3. First LoRA run on training_rows, then score on coverage, base and prompt-test.
4. v2 zero-shot without context (the no-context ablation that was planned and never run).
5. Calibration with and without per-class biases, on an inner validation split.
6. Flip TTA on perspective.
7. Layer-wise linear probe on frozen 8B features (an insight plot, which also tests idea I).
8. Label-error candidates: cross-validated high-confidence disagreements on train (bonus csv, never committed).

## 7. Suggested critical path to Saturday  [recommendation]

1. Tonight: A smoke test → first full phase-1 LoRA (Kaggle) → coverage score. In parallel: B with the base model
   (speed only), plus the inner-validation split and the calibration code (C).
2. Overnight/morning: phase-2 retrain on all train with the frozen recipe, and a dev LB submission of phase 1 to
   anchor the real score.
3. Morning: C + D on top, a second dev submission, final dry run timed on dev.
4. 13:00: decrypt, run, write the submission (dev rows = best dev file), upload by 14:15, select it, publish the
   weights, then fill `technical_report.md` §7.
