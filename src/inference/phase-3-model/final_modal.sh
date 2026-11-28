#!/usr/bin/env bash
# Final images with the delivered phase-3 mixed model on Modal: one H200 (GPU=H100 if H200s are sold out),
# fast path + TF32 + selective perspective TTA (preset "final", about 8 minutes of inference), then the Kaggle file.
#
#   export FINAL_PASSPHRASE='...'          # never write it into a file
#   export DEV_CSV=path/to/dev_predictions.csv   # required: dev rows of your best dev file (local, git-ignored)
#   bash src/inference/phase-3-model/final_modal.sh
#
# Needs: Modal CLI logged in, a Hugging Face token (~/.cache/huggingface/token or HF_TOKEN) with read access to the
# gated Apertus and DINOv3 and to the private repositories below, and the authorised data/train.csv locally
# (the job generator hashes it). The final archive is read from DATA_REPO, checked against its SHA-256 and decrypted
# on the job's disk; only predictions leave the job.
set -euo pipefail
: "${FINAL_PASSPHRASE:?export FINAL_PASSPHRASE first (it is passed to Modal as a transient secret)}"

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PHASE="$ROOT/src/training/phase-3"            # standalone working root of the delivered model
MIXED=src/training/apertus-8B-mixed-lora-heads  # relative to $PHASE
OUTPUT_REPO=${OUTPUT_REPO:-nathan-monzani-bymo/galaxus-vlh-private}    # private: checkpoint in, predictions out
DATA_REPO=${DATA_REPO:-nathan-monzani-bymo/galaxus-data-private}        # private: competition data + final archive
CHECKPOINT=${CHECKPOINT:-mixed-modal-20261128-033303/model/best}        # = weights/apertus-8B-mixed-lora-heads-final
: "${DEV_CSV:?export DEV_CSV=<your dev predictions csv (image_id + six dimensions, the dev rows of your best dev submission); it is git-ignored, not in the repo>}"
MODAL=${MODAL:-modal}
NAME=${NAME:-final-$(date -u +%Y%m%d-%H%M%S)}

GPU=${GPU:-H200}                                 # H100 if H200s are sold out
cd "$PHASE"
[ -e data ] || ln -s ../../../data data          # the generator hashes data/train.csv (git-ignored link)
python3 "$MIXED/build_infer_job.py"
"$MODAL" run "$MIXED/modal_infer.py" --runs final --local-hf-auth --name "$NAME" --gpu "$GPU" \
  --output-repo "$OUTPUT_REPO" --data-repo "$DATA_REPO" --checkpoint "$CHECKPOINT"
cd "$ROOT"

hf download "$OUTPUT_REPO" "$NAME/infer/final.csv" "$NAME/infer/final.json" --local-dir data/runs/modal
python3 src/scripts/final_submission.py --dev "$DEV_CSV" --final "data/runs/modal/$NAME/infer/final.csv" \
  --out "data/submissions/final_${NAME}.csv"
echo "Upload data/submissions/final_${NAME}.csv to Kaggle and SELECT it for the final ranking."
