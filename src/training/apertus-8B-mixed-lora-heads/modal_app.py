"""Run the mixed encoder experiment on one H200."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

REMOTE = "/opt/galaxus/hf_job_mixed.py"
app = modal.App("galaxus-apertus-mixed")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    bundle = Path(__file__).resolve().parents[3] / "data/hf_jobs/hf_job_mixed.py"
    image = (image.apt_install("git", "libgl1", "libglib2.0-0").pip_install("uv")
             .env({"UV_CACHE_DIR": "/opt/galaxus/uv-cache"})
             .add_local_file(bundle, REMOTE, copy=True)
             .run_commands(f"uv lock --script {REMOTE}", f"uv sync --locked --script {REMOTE}"))


@app.function(image=image, gpu="H200", cpu=8, memory=49152, timeout=150 * 60, retries=0, max_containers=1)
def train(output_repo, data_repo, source_repo, source_path, synthetic_file, dino_revision, name):
    env = {
        **os.environ, "UPLOAD_REPO": output_repo, "DATA_REPO": data_repo, "SOURCE_REPO": source_repo or output_repo,
        "INIT_FROM": source_path, "MIXED_SYNTHETIC_FILE": synthetic_file, "DINO_REVISION": dino_revision,
        "RUN_NAME": name, "WORK": "/tmp/galaxus-mixed", "PYTHONUNBUFFERED": "1",
    }
    subprocess.run(["uv", "run", "--locked", "--script", REMOTE], env=env, check=True)
    return f"{output_repo}/{name}"


@app.local_entrypoint()
def main(output_repo: str, data_repo: str, source_repo: str = "", source_path: str = "final-abt/model/best",
         secret: str = "galaxus-dino", synthetic_file: str = "", dino_revision: str = "",
         local_hf_auth: bool = False):
    if local_hf_auth:
        path = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
        token = os.environ.get("HF_TOKEN") or path.read_text().strip()
        if not token:
            raise ValueError("Missing HF token.")
        credentials = [modal.Secret.from_dict({"HF_TOKEN": token})]
    else:
        credentials = [modal.Secret.from_name(secret)]
    name = f"mixed-modal-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Starting one H200. Results: {output_repo}/{name}", flush=True)
    print(train.with_options(secrets=credentials).remote(output_repo, data_repo, source_repo, source_path,
                                                        synthetic_file, dino_revision, name))
