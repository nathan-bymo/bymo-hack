# Final predictions with the delivered phase-3 model

The delivered model is the phase-3 mixed checkpoint (SigLIP 2 + DINOv3 + VQ, one Apertus 8B decoder; weights in
[`weights/apertus-8B-mixed-lora-heads-final/`](../../../weights/apertus-8B-mixed-lora-heads-final/), public dev
0.84215). Both scripts predict the 5,292 final images and build the Kaggle file with
[`final_submission.py`](../../scripts/final_submission.py): the dev rows of the best dev file plus the final rows,
every cell checked against `sample_submission.csv`. Nothing is submitted automatically. Both scripts require
`DEV_CSV` (your local dev predictions csv, git-ignored and not in the repository) and the authorised `data/` folder.

| Script | Where | Path | Time |
|---|---|---|---|
| [final_modal.sh](final_modal.sh) | Modal, one H200 (`GPU=H100` if H200s are sold out) | fast path: merged LoRA, TF32, selective perspective TTA | about 8 min of inference + start-up |
| [final_hf.sh](final_hf.sh) | Hugging Face Jobs, one H200, your own account | original export (`train_mixed.py --predict-csv`) | about 15 min + start-up |

The passphrase of `final_encrypted.bin` is read only from the `FINAL_PASSPHRASE` environment variable (a transient
Modal secret or an HF Jobs secret). It is never written into a file, an argument list or a log. The archive is checked
against its published SHA-256 and decrypted on the job's disk; only predictions leave the job.

After the upload, select the submission with the final predictions on Kaggle's submissions page: otherwise Kaggle
ranks the best public score, which may be a file with the constant final rows.
