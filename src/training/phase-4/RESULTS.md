# Phase 4 — results

**Pending: not yet run.** The job is prepared and CPU-tested (see [README](README.md)); no GPU run, prompt-test sanity
check or public dev score exists yet.

Decision rule (from the [README](README.md)): phase 4 replaces the phase-3 model only if its public dev score is
**above 0.84215**. The 13:00 final run uses the phase-3 model regardless, so a valid submission exists; if phase 4
wins, the final images are predicted again with the phase-4 checkpoint and its weights are published in `weights/`
before 14:30. Its prompt-test score is measured on images it trained on and is a sanity check only.
