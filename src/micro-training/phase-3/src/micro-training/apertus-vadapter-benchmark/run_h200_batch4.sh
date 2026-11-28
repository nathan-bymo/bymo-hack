#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir 'accelerate>=1.10,<2' 'pandas==3.1.1' 'pillow==13.0.0' 'torchvision==0.21.0' 'transformers @ https://github.com/swiss-ai/transformers/archive/3797303dda74844e3d1f8977ff5518bb91f818b4.zip'

workspace=${WORKSPACE_ROOT:-/runner}
data="${DATA_ROOT:-/dataset/data}"
root="$workspace/src/apertus-vadapter-benchmark"
prompt="$workspace/src/docs/system_prompt.txt"
initial="${INITIAL_PROJECTOR:-/dataset/outputs/run-256/projector.pt}"
train_manifest="$data/prototype_splits/train_1500.csv"
validation_manifest="$data/prototype_splits/validation_450.csv"
common=(--backend hf-h200 --data-dir "$data" --system-prompt "$prompt" --train-manifest "$train_manifest" --validation-manifest "$validation_manifest" --vq-size 256 --initial-projector "$initial" --batch-size 4)

echo 'Testing the previous checkpoint with batch size 4'
python "$root/train_projector.py" "${common[@]}" --train-size 8 --holdout-size 4 --output-dir "$workspace"/outputs/smoke-batch4
echo 'Smoke test passed; training one additional epoch from the original checkpoint'
python "$root/train_projector.py" "${common[@]}" --train-size 1500 --holdout-size 450 --output-dir "$workspace"/outputs/run-batch4
