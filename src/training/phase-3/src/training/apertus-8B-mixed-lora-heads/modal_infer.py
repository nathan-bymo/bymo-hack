"""Fast inference of a mixed checkpoint on one H200 (benchmark, dev predictions, final images).

  python3 src/training/apertus-8B-mixed-lora-heads/build_infer_job.py
  modal run --detach src/training/apertus-8B-mixed-lora-heads/modal_infer.py --runs bench --local-hf-auth
  FINAL_PASSPHRASE=... modal run src/training/apertus-8B-mixed-lora-heads/modal_infer.py --runs final --local-hf-auth

--runs picks a preset below or takes a JSON list. The passphrase is read from the environment, never from a file.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

REMOTE = "/opt/galaxus/hf_job_mixed_infer.py"
app = modal.App("galaxus-apertus-mixed-infer")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    bundle = Path(__file__).resolve().parents[3] / "data/hf_jobs/hf_job_mixed_infer.py"
    image = (image.apt_install("git", "libgl1", "libglib2.0-0", "p7zip-full").pip_install("uv")
             .env({"UV_CACHE_DIR": "/opt/galaxus/uv-cache"})
             .add_local_file(bundle, REMOTE, copy=True)
             .run_commands(f"uv lock --script {REMOTE}", f"uv sync --locked --script {REMOTE}"))

FAST = ["--merge-lora", "--sort", "--batch", "32", "--ocr-thread"]
PRESETS = {
    # Speed and parity on prompt-test: the export path as it was, then the fast path, then the fast path + flip TTA.
    "bench": [
        {"name": "pt_baseline", "part": "prompt_test", "args": ["--batch", "16"]},
        {"name": "pt_fast", "part": "prompt_test", "args": FAST},
        {"name": "pt_fast_tta", "part": "prompt_test", "args": [*FAST, "--tta-perspective", "all"]},
    ],
    # Rehearsal of the final pipeline on dev: CSV to compare with the submitted one.
    "dev": [{"name": "dev_fast", "part": "dev", "args": FAST}],
    # TF32 for the fp32 VQ tokenizer (the bottleneck), OCR and SigLIP: speed, code agreement and parity.
    "tf32": [
        {"name": "pt_fast_tf32", "part": "prompt_test", "args": [*FAST, "--tf32"]},
        {"name": "pt_tf32_nomerge", "part": "prompt_test", "args": ["--sort", "--batch", "32", "--ocr-thread", "--tf32"]},
        {"name": "dev_fast_tf32", "part": "dev", "args": [*FAST, "--tf32"]},
    ],
    # Delivery configuration: fast path + TF32 + selective perspective TTA (RESULTS.md, sections 5-6 and 8).
    "final": [{"name": "final", "part": "final", "args": [*FAST, "--tf32", "--tta-perspective", "selective"]}],
    "dev_final": [{"name": "dev_final", "part": "dev", "args": [*FAST, "--tf32", "--tta-perspective", "selective"]}],
    # Fallback without TF32: VQ codes identical to training, about half the speed.
    "final_safe": [{"name": "final_safe", "part": "final", "args": [*FAST, "--tta-perspective", "selective"]}],
}


@app.function(image=image, gpu="H200",
              cpu=16, memory=65536, timeout=90 * 60, retries=0)
def infer(output_repo, data_repo, checkpoint, reference, runs, name):
    env = {**os.environ, "UPLOAD_REPO": output_repo, "DATA_REPO": data_repo, "CHECKPOINT": checkpoint,
           "REFERENCE": reference, "RUNS": runs, "RUN_NAME": name, "WORK": "/tmp/galaxus-infer",
           "PYTHONUNBUFFERED": "1"}
    subprocess.run(["uv", "run", "--locked", "--script", REMOTE], env=env, check=True)
    return f"{output_repo}/{name}"


@app.local_entrypoint()
def main(runs: str = "bench", output_repo: str = "",
         data_repo: str = "",
         checkpoint: str = "mixed-modal-20261128-033303/model/best", reference: str = "mixed-modal-20261128-033303",
         secret: str = "galaxus-dino", local_hf_auth: bool = False, name: str = "", gpu: str = "H200"):
    if not output_repo or not data_repo:
        raise ValueError("Pass --output-repo and --data-repo explicitly.")
    spec = PRESETS.get(runs) or json.loads(runs)
    passphrase = os.environ.get("FINAL_PASSPHRASE", "")
    if any(run["part"] == "final" for run in spec) and not passphrase:
        raise ValueError("Set FINAL_PASSPHRASE in the environment for the final images.")
    if local_hf_auth:
        path = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
        credentials = [modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN") or path.read_text().strip()})]
    else:
        credentials = [modal.Secret.from_name(secret)]
    if passphrase:  # a transient secret: the passphrase is never a function argument or a file
        credentials.append(modal.Secret.from_dict({"FINAL_PASSPHRASE": passphrase}))
    name = name or f"infer-{runs if runs in PRESETS else 'custom'}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Starting one {gpu} GPU. Results: {output_repo}/{name}", flush=True)
    print(infer.with_options(secrets=credentials, gpu=gpu).remote(output_repo, data_repo, checkpoint, reference,
                                                        json.dumps(spec), name))
