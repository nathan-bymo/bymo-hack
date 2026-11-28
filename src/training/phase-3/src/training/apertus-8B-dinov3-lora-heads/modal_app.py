"""Run the existing DINOv3 experiment on Modal with the same dependencies and training configuration."""

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

REMOTE_BUNDLE = "/opt/galaxus/hf_job_dinov3.py"
LONG_TIMEOUT = 255 * 60

app = modal.App("galaxus-apertus-dinov3")
base_image = modal.Image.debian_slim(python_version="3.12")
training_image = base_image
if modal.is_local():
    bundle = Path(__file__).resolve().parents[3] / "data/hf_jobs/hf_job_dinov3.py"
    training_image = (
        base_image.apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install("uv")
        .env({"UV_CACHE_DIR": "/opt/galaxus/uv-cache"})
        .add_local_file(bundle, REMOTE_BUNDLE, copy=True)
        .run_commands(f"uv lock --script {REMOTE_BUNDLE}", f"uv sync --locked --script {REMOTE_BUNDLE}")
    )
access_image = base_image.pip_install("huggingface_hub", "kaggle==2.2.5")


def job_environment(output_repo, source_repo, data_repo, source_path, name):
    return {
        **os.environ,
        "UPLOAD_REPO": output_repo,
        "SOURCE_REPO": source_repo or output_repo,
        "INIT_FROM": source_path,
        "RUN_NAME": name,
        "WORK": "/tmp/galaxus-work",
        "DATA_REPO": data_repo or os.environ.get("DATA_REPO", ""),
        "PYTHONUNBUFFERED": "1",
    }


@app.function(image=access_image, cpu=1, memory=1024, timeout=120)
def check_access(output_repo, source_repo, data_repo, source_path):
    from huggingface_hub import HfApi, hf_hub_download

    if not os.environ.get("HF_TOKEN"):
        raise ValueError("The Modal secret must contain HF_TOKEN.")
    env = job_environment(output_repo, source_repo, data_repo, source_path, "preflight")
    api = HfApi()
    source_api = HfApi(token=env.get("SOURCE_HF_TOKEN"))
    if not api.repo_info(output_repo).private or not source_api.repo_info(env["SOURCE_REPO"]).private:
        raise ValueError("Both model repositories must be private.")
    artifacts = ("settings.json", "trainable.pt", "lora/adapter_model.safetensors")
    for artifact in artifacts:
        if not source_api.file_exists(env["SOURCE_REPO"], f"{source_path}/{artifact}"):
            raise FileNotFoundError(f"Missing source checkpoint file: {artifact}")
    hf_hub_download("facebook/dinov3-vitb16-pretrain-lvd1689m", "config.json")
    hf_hub_download("swiss-ai/Apertus-v1.5-8B", "config.json", revision="1519de0ae7f3d114879291810a58259966f64c99")

    if env["DATA_REPO"]:
        data = api.repo_info(env["DATA_REPO"], repo_type="dataset")
        if not data.private:
            raise ValueError("The competition dataset repository must be private.")
        required = {"train.csv", "dev.csv", "metric.py", "train_images.tar", "dev_images.tar"}
        if not required.issubset(api.list_repo_files(env["DATA_REPO"], repo_type="dataset")):
            raise ValueError("The private dataset is missing required files.")
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        kaggle = KaggleApi()
        kaggle.authenticate()
        if not kaggle.competition_list_files("hackapertus-galaxus").files:
            raise ValueError("Could not access the Kaggle competition data.")
    return "Read access verified for the source checkpoint, private output repository, models and dataset."


@app.function(image=training_image, gpu="H200", cpu=8, memory=49152,
              timeout=3 * 60 * 60, retries=0, max_containers=1)
def train_on_gpu(output_repo, source_repo, data_repo, source_path, name, long_run, synthetic_file, progressive):
    env = job_environment(output_repo, source_repo, data_repo, source_path, name)
    if long_run:
        env.update(DINO_EPOCHS="10", DINO_MAX_MINUTES="205", DINO_TOTAL_MINUTES="230", DINO_EVAL_ONCE="1")
    if synthetic_file:
        env.update(DINO_SYNTHETIC_FILE=synthetic_file, DINO_EPOCHS="4", DINO_MAX_MINUTES="80",
                   DINO_TOTAL_MINUTES="100", DINO_EVAL_ONCE="1")
    if progressive:
        env.update(DINO_PROGRESSIVE="1", DINO_MAX_MINUTES="75", DINO_TOTAL_MINUTES="95")
    subprocess.run(["uv", "run", "--locked", "--script", REMOTE_BUNDLE], env=env, check=True)
    return f"{output_repo}/{name}"


@app.local_entrypoint()
def main(output_repo: str, source_repo: str = "", data_repo: str = "",
         source_path: str = "final-abt/model/best", secret: str = "galaxus-dino",
         gpu: str = "H200", preflight: bool = False, long_run: bool = False,
         local_hf_auth: bool = False, synthetic_file: str = "", progressive: bool = False):
    if gpu not in {"H100", "H200"}:
        raise ValueError("Select H100 or H200.")
    if progressive and (gpu != "H200" or not synthetic_file):
        raise ValueError("Progressive training requires H200 and the original private synthetic archive.")
    if local_hf_auth:
        token_path = Path(os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache/huggingface/token")))
        token = os.environ.get("HF_TOKEN") or token_path.read_text().strip()
        if not token or not data_repo:
            raise ValueError("Local HF auth requires an HF token and a private dataset repo.")
        credentials = [modal.Secret.from_dict({"HF_TOKEN": token})]
    else:
        credentials = [modal.Secret.from_name(secret)]
    args = (output_repo, source_repo, data_repo, source_path)
    if preflight:
        print(check_access.with_options(secrets=credentials).remote(*args))
        return

    prefix = "dino-progressive-modal" if progressive else "dino-synthetic-modal" if synthetic_file else "dino-no-vq-modal"
    name = f"{prefix}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    print(f"Starting one {gpu} container. Results: {output_repo}/{name}", flush=True)
    timeout = 120 * 60 if synthetic_file else LONG_TIMEOUT if long_run else 3 * 60 * 60
    print(train_on_gpu.with_options(gpu=gpu, secrets=credentials, timeout=timeout).remote(
        *args, name, long_run, synthetic_file, progressive))
