#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
name="dino-no-vq-$(date -u +%Y%m%d-%H%M%S)"
: "${OUTPUT_REPO:?Set OUTPUT_REPO to an existing private repository on the colleague account}"
: "${SOURCE_REPO:=$OUTPUT_REPO}"
: "${SOURCE_PATH:=final-abt/model/best}"
if [[ -z ${HF_TOKEN:-} ]]; then
    HF_TOKEN=$(<"${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}")
fi
export HF_TOKEN

source_args=(--env "SOURCE_REPO=$SOURCE_REPO")
if [[ -n ${SOURCE_HF_TOKEN:-} ]]; then
    export SOURCE_HF_TOKEN
    source_args+=(--secrets SOURCE_HF_TOKEN)
fi
if [[ -n ${DATA_REPO:-} ]]; then
    source_args+=(--env "DATA_REPO=$DATA_REPO")
elif [[ -n ${KAGGLE_USERNAME:-} && -n ${KAGGLE_KEY:-} ]]; then
    export KAGGLE_USERNAME KAGGLE_KEY
    source_args+=(--secrets KAGGLE_USERNAME --secrets KAGGLE_KEY)
else
    : "${KAGGLE_API_TOKEN:?Set DATA_REPO (preferred), KAGGLE_API_TOKEN or the Kaggle username and key}"
    export KAGGLE_API_TOKEN
    source_args+=(--secrets KAGGLE_API_TOKEN)
fi

job=data/hf_jobs/hf_job_dinov3.py
python3 src/training/apertus-8B-dinov3-lora-heads/build_hf_job.py --out "$job"
hf jobs uv run --flavor h200 --timeout 180m --attempts 1 --detach --name "$name" \
    --secrets HF_TOKEN "${source_args[@]}" \
    --env "UPLOAD_REPO=$OUTPUT_REPO" \
    --env "RUN_NAME=$name" --env "INIT_FROM=$SOURCE_PATH" "$@" "$job"
