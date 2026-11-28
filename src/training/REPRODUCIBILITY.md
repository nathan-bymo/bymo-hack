# Reproduction guide and evidence status

This guide distinguishes **running the current code**, **reproducing a documented experiment**, and **independently verifying the final submission**. The latter two require the exact historical source bundle, data, configurations, checkpoints and logs. Current code and phase code copies are not snapshots of old runs. Remote training incurs compute charges; none of the commands below were launched as part of this documentation check.

## Common requirements

1. Work from the repository root. Obtain the competition data through an authorized source and place `train.csv` under `data/` (or use the documented private dataset route). Follow the competition's data and model terms; do not publish data, credentials or generated bundles containing image IDs. The generators embed the SHA-256 of the local `train.csv`, which the remote job checks against its data.
2. Install Python and the Hugging Face CLI, authenticate with `hf auth login`, and accept access terms for Apertus and any gated supporting models. Confirm the active account with `hf auth whoami`. For GPU runs arrange access to the specified GPU and an authorized private output repository. Use environment variables or the job provider's secret store for tokens, not source control. HF commands below use `--secrets HF_TOKEN` and, when applicable, `--secrets KAGGLE_API_TOKEN`; export those credentials in the shell first. Consult the competition rules for data access and usage.
3. For Modal runs, install and authenticate the Modal CLI separately and configure the secret or local Hugging Face authentication described in each experiment's README. The Modal CLI was unavailable in the environment used to write this guide.
4. Review budgets and timeouts before launching. The guides give launch patterns, not a guarantee of GPU availability, cost, permissions or identical results. Verify that outputs stay private.

## Phase guides

- [Phase 1: initial training](phase-1/README.md)
- [Phase 2: branch improvements](phase-2/README.md)
- [Phase 3: DINOv3 and mixed](phase-3/README.md)

Recorded results, with the source of every number: [phase 1](phase-1/RESULTS.md), [phase 2](phase-2/RESULTS.md),
and [phase 3](phase-3/RESULTS.md).

## What has been checked

Each phase now has an independent copy of its supporting source. In standalone copies under `/private/tmp` with the authorized `train.csv` and `metric.py` supplied, all eight applicable job generators succeeded and their generated Python parsed. The phase test files were run in separate processes: 124 passed and two skipped (one in each visual-adapter test suite). These are CPU tests, not an end-to-end GPU or remote-provider verification.


On 28 November 2026 the five current job generators ran locally with `data/train.csv` present, and their generated Python files passed `py_compile`. The LoRA builder reported that the optional small split was unavailable, so its generated bundle supports `SPLIT=full` only. The generators read the current working tree, including any uncommitted changes. They do not recreate the exact historical job scripts. Separately, the five training test files passed when run in isolated processes (73 passed, one skipped); running them together caused Python module-name collisions. GPU execution, remote preflights and complete historical reruns were **not** tested during this check.

HF job history for the currently authenticated account records, among others, successful visual-adapter jobs `6b09af0354072e8b716a1d36` (initial phase) and `6b09e0f06c5d61fff189ab6a` (later final-abt run). The latter's accessible log confirms completion and upload of predictions and logs. This does not by itself establish the exact source revision, complete effective configuration, or provenance of a competition submission. Other team members' accounts may hold further jobs.

## Evidence required for an exact reproduction

Record source commit and dirty state; immutable job bundle and checksum; full resolved config and environment/dependency versions; base model and encoder revisions; authorized data version and split manifests with checksums; random seeds, hardware, and training checkpoints; launch command, job ID, logs and outputs; inference flags, prediction file checksum and submission ID. Store data-derived IDs, logs and artifacts only where authorized. For the final submission, link the actual selected submission to its source commit and published checkpoint revision and rerun the full inference pipeline. These links are not yet established in the repository.

## Independent historical audit

The locally available `train.csv` has SHA-256 `936a330d207f4ded5cf8e3834682ce8f6f219f0372ab8dafaceed2245b8c15ab`; that digest is embedded in retained visual-adapter, DINO and mixed bundles. This verifies the local CSV against those bundles, not image bytes, other data files, or whether those bundles were actually executed. A retained older adapter bundle differs in 9 of 11 sources from the current phase copy, and a retained DINO bundle differs in six visual-adapter files. A local mixed bundle matches the current 17 sources but is later than the historical run. Consequently exact historical parity is not established.

## Original HF artifacts recovered

Job metadata associates the original visual-adapter scripts with the private bucket paths in the phase 1 and phase 2 READMEs. The scripts have been retained under ignored `data/historical_hf/`; their embedded code differs from the current copies. For the micro visual-adapter prototype, the epoch-1 bucket manifests match the local prototype manifests byte-for-byte, and the HF metadata confirms both H200 job images and mounts. P8 heads manifests, the P11/P12 teammate job bundles, and the mixed Modal source image have not been recovered.
