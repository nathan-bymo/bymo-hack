# Zero-shot benchmark results

These are historical API results, not outputs of the reconstructed notebook or Kaggle leaderboard scores.

| Model and prompt | Coverage macro F1 | Prompt-test macro F1 |
|---|---:|---:|
| 8B, minimal JSON | 0.316 | 0.291 |
| 8B, shared JSON | 0.229 | 0.225 |
| 8B, description then codes | 0.528 | 0.502 |
| 70B, shared JSON | 0.304 | 0.294 |
| 70B, description then codes | 0.583 | 0.590 |

On coverage, the 70B description prompt scored 0.546 subject, 0.631 background, 0.274 perspective, 0.899 individuals, 0.618 text classification and 0.529 language. Its reported throughput was 2.8 images/s on the shared gateway; v2 uses seven sequential requests per image. The prompt was selected on prompt-dev, with prompt-test held out from prompt selection. [Coverage and per-class results](apertus-8B-benchmark/prompt_eval_coverage.md); [70B summary](apertus-70B-benchmark/README.md); [prompt diagnostics](apertus-8B-benchmark/prompt_eval.md).
