#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir 'accelerate>=1.10,<2' 'pandas==3.1.1' 'pillow==13.0.0' 'torchvision==0.21.0' 'transformers @ https://github.com/swiss-ai/transformers/archive/3797303dda74844e3d1f8977ff5518bb91f818b4.zip'

code=/workspace/src/apertus-vadapter-benchmark/eval_describe_codes.py
checkpoint=/checkpoint/projector.pt
echo 'Running four-image smoke test of describe:codes v2'
python "$code" --data-dir /workspace/data --checkpoint "$checkpoint" --output-dir /workspace/outputs/smoke-v2 --vq-size 256 --limit 4
echo 'Smoke test passed; evaluating 600 prompt-dev images'
python "$code" --data-dir /workspace/data --checkpoint "$checkpoint" --output-dir /workspace/outputs/eval-v2 --vq-size 256 --limit 600
