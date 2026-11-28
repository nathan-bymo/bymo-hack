"""Runs the job script of build_hf_job.py on Modal instead of Hugging Face Jobs.

The job is the same: the same environment variables, private Hugging Face repository and logs. Needs a Modal secret
named "galaxus" with HF_TOKEN and KAGGLE_API_TOKEN. From the repository root:

  python src/training/apertus-8B-vadapter-lora-heads/build_hf_job.py
  modal run src/training/apertus-8B-vadapter-lora-heads/modal_run.py --cpu-check --env 'CHECK_ONLY=1 UPLOAD_REPO=...'
  MODAL_HOURS=2 modal run --detach src/training/apertus-8B-vadapter-lora-heads/modal_run.py --env 'UPLOAD_REPO=... ...'

MODAL_GPU (default H200) and MODAL_HOURS (default 2) are read when the command starts. MODAL_HOURS is the timeout of
the function, so the cap on the cost. With --detach the job goes on if the computer disconnects.
"""

import os
import shlex
import subprocess

import modal

JOB = "data/hf_jobs/hf_job_vlh.py"  # written by build_hf_job.py (git-ignored)
GPU = os.environ.get("MODAL_GPU", "H200")
HOURS = float(os.environ.get("MODAL_HOURS", "2"))

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "libgl1", "libglib2.0-0", "p7zip-full")
         .pip_install("uv")
         .add_local_file(JOB, "/job/hf_job_vlh.py"))
app = modal.App("galaxus-vlh", image=image)
SECRETS = [modal.Secret.from_name("galaxus")]


def execute(settings):
    subprocess.run(["uv", "run", "/job/hf_job_vlh.py"], env={**os.environ, **settings}, check=True)


@app.function(gpu=GPU, cpu=8, memory=65536, timeout=int(HOURS * 3600), retries=0, secrets=SECRETS)
def gpu_job(settings: dict):
    execute(settings)


@app.function(cpu=4, memory=16384, timeout=20 * 60, retries=0, secrets=SECRETS)
def cpu_job(settings: dict):
    execute(settings)


@app.local_entrypoint()
def main(env: str = "", cpu_check: bool = False):
    settings = dict(item.split("=", 1) for item in shlex.split(env))
    if cpu_check and not settings.get("CHECK_ONLY"):
        raise SystemExit("--cpu-check needs CHECK_ONLY=1 in --env")
    (cpu_job if cpu_check else gpu_job).remote(settings)
