# Zero-shot benchmark results

These are historical API results, not outputs of the reconstructed notebook or Kaggle leaderboard scores.

| Model and prompt | Coverage macro F1 | Prompt-test macro F1 |
|---|---:|---:|
| 8B, minimal JSON | 0.316 | 0.291 |
| 8B, shared JSON | 0.229 | 0.225 |
| 8B, description then codes | 0.528 | 0.502 |
| 70B, shared JSON | 0.304 | 0.294 |
| 70B, description then codes | 0.583 | 0.590 |

On coverage, the 70B description prompt scored 0.546 subject, 0.631 background, 0.274 perspective, 0.899 individuals, 0.618 text classification and 0.529 language. Its reported throughput was 2.8 images/s on the shared gateway; v2 uses seven sequential requests per image. The prompt was selected on prompt-dev, with prompt-test held out from prompt selection. The 8B description prompt ran at 3.8 images/s on the same gateway.

Is the 70B reliable when confident? On coverage, its perspective answers are right 28.4% of the time below confidence 0.60, 39.3% between 0.60 and 0.90 and **63.4%** at ≥ 0.90; for comparison subject reaches 76.6% and individuals 98.5% at ≥ 0.90. Distilling the 70B was therefore not pursued.

Four-decimal coverage scores: 70B description 0.5829, 8B description 0.5282, 70B shared JSON 0.3036, 8B shared JSON 0.2293. Source: `src/training/benchmark/overall.md`, `src/training/benchmark/README.md` (speeds).

Source: `src/plain-benchmark/apertus-8B-benchmark/prompt_eval_coverage.md`, `src/plain-benchmark/apertus-8B-benchmark/prompt_eval.md`, `src/plain-benchmark/apertus-8B-benchmark/teacher_student_zeroshot.md` and `src/plain-benchmark/apertus-70B-benchmark/README.md`; these files are deleted in the current working tree and can be read with `git show 36d3aff:<path>`.
