#!/usr/bin/env bash
# Final images with the delivered phase-3 mixed model on Hugging Face Jobs (one H200), for anyone with their own
# HF account: uploads the published weights to YOUR private repo, decrypts the final archive on the job's disk and
# predicts it with train_mixed.py --predict-csv (original export path, about 6 images/s: ~15 minutes for 5,292 images).
#
#   export HF_TOKEN=...            # your token: gated access to swiss-ai/Apertus-v1.5-8B and DINOv3 accepted
#   export KAGGLE_API_TOKEN=...    # a Kaggle account in the competition (final_encrypted.bin is downloaded from Kaggle)
#   export FINAL_PASSPHRASE='...'  # never write it into a file
#   export UPLOAD_REPO=you/galaxus-final-private    # a PRIVATE model repository of yours
#   export DEV_CSV=path/to/dev_predictions.csv      # required: dev rows of your best dev file (local, git-ignored)
#   bash src/inference/phase-3-model/final_hf.sh
#
# Needs the authorised data/train.csv locally (the job generator hashes it) ; the weights are downloaded from the public nathan-monzani-bymo/apertus-8B-mixed repository.
set -euo pipefail
: "${HF_TOKEN:?}" "${KAGGLE_API_TOKEN:?}" "${FINAL_PASSPHRASE:?}" "${UPLOAD_REPO:?a private HF model repository}"

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$ROOT"
PHASE="$ROOT/src/training/phase-3"            # standalone working root of the delivered model
MIXED=src/training/apertus-8B-mixed-lora-heads  # relative to $PHASE
CHECKPOINT=phase3-mixed/model/best
RUN_NAME=${RUN_NAME:-final-hf-$(date -u +%Y%m%d-%H%M%S)}
: "${DEV_CSV:?export DEV_CSV=<your dev predictions csv (image_id + six dimensions, the dev rows of your best dev submission); it is git-ignored, not in the repo>}"

hf repo create "$UPLOAD_REPO" --private --exist-ok
hf download nathan-monzani-bymo/apertus-8B-mixed --revision 49f4bd8f479f53f1fc232aaee8028e7c2b44161d --local-dir weights/apertus-8B-mixed-lora-heads-final
hf upload "$UPLOAD_REPO" weights/apertus-8B-mixed-lora-heads-final "$CHECKPOINT" --private \
  --include "settings.json" "trainable.pt" "mixed_fusion.pt" "lora/*"
cd "$PHASE"
[ -e data ] || ln -s ../../../data data          # the generator hashes data/train.csv (git-ignored link)
python3 "$MIXED/build_hf_job.py"                 # writes data/hf_jobs/hf_job_mixed.py
hf jobs uv run --flavor h200 --timeout 60m \
  --secrets HF_TOKEN --secrets KAGGLE_API_TOKEN --secrets FINAL_PASSPHRASE \
  -e FINAL=1 -e UPLOAD_REPO="$UPLOAD_REPO" -e PREDICT_FROM="$CHECKPOINT" -e RUN_NAME="$RUN_NAME" \
  data/hf_jobs/hf_job_mixed.py
# When the job prints ALL DONE (hf jobs logs <id>):
echo "Then: hf download $UPLOAD_REPO $RUN_NAME/predictions/final.csv --local-dir data/runs/hf"
echo "      python3 src/scripts/final_submission.py --dev $DEV_CSV --final data/runs/hf/$RUN_NAME/predictions/final.csv --out data/submissions/final_hf.csv"
