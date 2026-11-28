# Phase 1 — results

On the intentionally balanced 450-image validation set, frozen Apertus 8B with heads scored 0.542 using cross-entropy, 0.537 with cross-entropy and GradNorm, 0.528 with focal loss, and 0.501 with focal loss and GradNorm. The pixel-only baseline scored approximately 0.257. These P8 results used one seed. In a subsequent five-seed frozen-feature study, strongly weighted cross-entropy averaged 0.557 and strongly weighted focal loss 0.547. This does not establish a general advantage of focal loss.

Source: `results_small_h200.md` (original report, git history), `results_small_h200_per_class.md` (original report, git history), `results_weights_small.md` (original report, git history); four-decimal values in `overall.md` (original report, git history).
