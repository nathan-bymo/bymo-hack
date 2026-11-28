"""Run the mixed encoder experiment on one H200."""

import getpass
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
    image = (image.apt_install("git", "libgl1", "libglib2.0-0", "p7zip-full").pip_install("uv")
             .env({"UV_CACHE_DIR": "/opt/galaxus/uv-cache"})
             .add_local_file(bundle, REMOTE, copy=True)
             .run_commands(f"uv lock --script {REMOTE}", f"uv sync --locked --script {REMOTE}"))


# MIXED_MINUTES is read when the command starts: it is the timeout of the function, so the cap on the cost.
MINUTES = int(os.environ.get("MIXED_MINUTES", "150"))


@app.function(image=image, gpu="H200", cpu=8, memory=65536, timeout=MINUTES * 60, retries=0, max_containers=1)
def train(output_repo, data_repo, source_repo, source_path, synthetic_file, dino_revision, name, extra):
    env = {
        **os.environ, "UPLOAD_REPO": output_repo, "DATA_REPO": data_repo, "SOURCE_REPO": source_repo or output_repo,
        "INIT_FROM": source_path, "MIXED_SYNTHETIC_FILE": synthetic_file, "DINO_REVISION": dino_revision,
        "RUN_NAME": name, "WORK": "/tmp/galaxus-mixed", "PYTHONUNBUFFERED": "1", **extra,
    }
    subprocess.run(["uv", "run", "--locked", "--script", REMOTE], env=env, check=True)
    return f"{output_repo}/{name}"


@app.local_entrypoint()
def main(output_repo: str, data_repo: str, source_repo: str = "", source_path: str = "final-abt/model/best",
         secret: str = "galaxus-dino", synthetic_file: str = "", dino_revision: str = "",
         local_hf_auth: bool = False, vision_lora: int = 0, vision_epochs: float = 2.0, wall_minutes: int = 0,
         final: bool = False, final_archive: str = "", teacher_file: str = "", teacher_sha256: str = "",
         source_revision: str = ""):
    pilot = bool(teacher_file)  # distillation pilot on the dev pseudo-labels (RESULTS.md, section 4)
    if pilot != bool(teacher_sha256) or pilot != bool(source_revision):
        raise ValueError("Provide the teacher file, its SHA-256 and the pinned source revision together.")
    if pilot and (vision_lora or final):
        raise ValueError("The distillation pilot runs alone: no --vision-lora or --final.")
    if local_hf_auth:
        path = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
        token = os.environ.get("HF_TOKEN") or path.read_text().strip()
        if not token:
            raise ValueError("Missing HF token.")
        credentials = [modal.Secret.from_dict({"HF_TOKEN": token})]
    else:
        credentials = [modal.Secret.from_name(secret)]
    extra = {"WALL_MINUTES": str(wall_minutes or max(20, MINUTES - 25))}
    if vision_lora:
        if source_path == "final-abt/model/best":
            raise ValueError("--vision-lora continues a mixed checkpoint: pass its --source-path.")
        extra.update(VISION_LORA=str(vision_lora), VISION_EPOCHS=str(vision_epochs))
    prefix = "mixed-modal"
    if pilot:
        extra.update(MIXED_TEACHER_FILE=teacher_file, MIXED_TEACHER_SHA256=teacher_sha256,
                     MIXED_SOURCE_REVISION=source_revision)
        prefix = "distill-modal"
    if final:
        # The passphrase is typed (or taken from the environment) at launch: it is never written to a file.
        secrets = {"FINAL_PASSPHRASE": os.environ.get("FINAL_PASSPHRASE") or getpass.getpass("Passphrase: ")}
        if local_hf_auth:  # a named secret already holds the Kaggle token
            token_file = Path.home() / ".kaggle/access_token"
            kaggle = os.environ.get("KAGGLE_API_TOKEN") or (token_file.read_text().strip() if token_file.exists() else "")
            if kaggle:
                secrets["KAGGLE_API_TOKEN"] = kaggle
            elif not final_archive:
                raise ValueError("Downloading the final images needs a Kaggle token.")
        credentials.append(modal.Secret.from_dict(secrets))
        extra = {"FINAL": "1", "PREDICT_FROM": source_path, "FINAL_ARCHIVE": final_archive}
        prefix = "final-mixed"
    name = f"{prefix}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Starting one H200. Results: {output_repo}/{name}", flush=True)
    print(train.with_options(secrets=credentials).remote(output_repo, data_repo, source_repo, source_path,
                                                        synthetic_file, dino_revision, name, extra))
