"""Build data/hf_jobs/hf_job_mixed_infer.py: fast inference of a mixed checkpoint (benchmark, dev, final).

Environment of the job:
  UPLOAD_REPO   private model repo with the checkpoint; results go to <RUN_NAME>/
  DATA_REPO     private dataset with train.csv, dev.csv, metric.py, images and final_encrypted.bin
  CHECKPOINT    folder of the checkpoint in UPLOAD_REPO (default: the mixed joint-3 best)
  REFERENCE     prefix in UPLOAD_REPO whose predictions/{prompt_test,dev}.probs.pt are the parity reference
  RUNS          JSON list of runs: {"name", "part": prompt_test|dev|final, "args": [fast_infer.py flags]}
  FINAL_PASSPHRASE  only for part final; never printed
"""

import argparse
import base64
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FOLDER = HERE.relative_to(ROOT).as_posix()
SIGLIP = "src/training/apertus-8B-vadapter-lora-heads"
DINO = "src/training/apertus-8B-dinov3-lora-heads"

ENTRY = '''
FINAL_REPO_FILE = "final_encrypted.bin"


def final_from_repo():
    """Decrypts final_encrypted.bin from the private dataset (SHA-256 checked) on the job's disk."""
    from huggingface_hub import hf_hub_download

    passphrase = os.environ.get("FINAL_PASSPHRASE", "")
    assert passphrase, "part final needs FINAL_PASSPHRASE"
    archive = Path(hf_hub_download(DATA_REPO, FINAL_REPO_FILE, repo_type="dataset", local_dir=DATA / "final_download"))
    digest = hashlib.sha256()
    with open(archive, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    assert digest.hexdigest() == FINAL_SHA256, "final_encrypted.bin does not match the published SHA-256"
    if not shutil.which("7z"):
        subprocess.run("apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null", shell=True, check=True)
    target = DATA / "final"
    result = subprocess.run(["7z", "x", f"-p{passphrase}", f"-o{target}", "-y", str(archive)], capture_output=True,
                            text=True)
    if result.returncode:
        raise SystemExit("7z could not open the archive: " + (result.stderr or result.stdout)[-300:].replace(passphrase, "***"))
    tables = sorted(target.rglob("final.csv"))
    assert tables, "the archive holds no final.csv"
    log("final images decrypted")
    return tables[0].parent


def run_infer():
    import json
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    unpack()
    sys.path.insert(0, str(WORK / "@@MIXED@@"))
    from mixed_model import E

    runs = json.loads(os.environ["RUNS"])
    checkpoint_path = os.environ.get("CHECKPOINT", "mixed-modal-20261128-033303/model/best")
    reference = os.environ.get("REFERENCE", "mixed-modal-20261128-033303")
    api = HfApi()
    info = api.repo_info(UPLOAD_REPO)
    assert info.private, "the model repository must be private"
    parts = {run["part"] for run in runs}
    if "prompt_test" not in parts:
        os.environ["DEV_ONLY"] = "1"
    ensure_opencv()
    with ThreadPoolExecutor(4) as pool:
        data = pool.submit(download_data)
        base = pool.submit(E.model_dir)
        final = pool.submit(final_from_repo) if "final" in parts else None
        checkpoint = pool.submit(snapshot_download, UPLOAD_REPO, revision=info.sha,
                                 allow_patterns=[f"{checkpoint_path}/*", f"{checkpoint_path}/lora/*"],
                                 local_dir=WORK / "source")
        data.result()
        base.result()
        checkpoint = Path(checkpoint.result()) / checkpoint_path
        final_dir = final.result() if final else None
    log("checkpoint", checkpoint_path, "at revision", info.sha)
    out = RUN / "infer"
    out.mkdir(parents=True, exist_ok=True)
    for spec in runs:
        part = spec["part"]
        command = ["@@MIXED@@/fast_infer.py", "--checkpoint", checkpoint, "--out", out / spec["name"], *spec["args"]]
        if part == "prompt_test":
            command += ["--data-dir", DATA, "--csv", DATA / "train.csv", "--part", "test", "--labels"]
        elif part == "dev":
            command += ["--data-dir", DATA, "--csv", DATA / "dev.csv"]
        else:
            command += ["--data-dir", final_dir, "--csv", final_dir / "final.csv"]
        if part in ("prompt_test", "dev") and spec.get("parity", True):
            try:
                command += ["--reference", hf_hub_download(UPLOAD_REPO, f"{reference}/predictions/{part}.probs.pt",
                                                           revision=info.sha)]
            except Exception as error:  # noqa: BLE001
                log("no parity reference for", part, type(error).__name__)
        run(command, "infer.log")
        upload(out, "infer", [f"{spec['name']}.*"])
    upload(RUN, "logs", ["*.log"])
    log("ALL DONE")


if __name__ == "__main__":
    run_infer()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/hf_jobs/hf_job_mixed_infer.py")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("siglip_hf_builder", ROOT / SIGLIP / "build_hf_job.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    members = [*base.FILES, *(f"{DINO}/{name}" for name in ("pipeline.py", "experiment.py", "base_weights.py")),
               *(f"{FOLDER}/{name}" for name in ("mixed_model.py", "train_mixed.py", "fast_infer.py"))]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in members:
            contents = (ROOT / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    template, old = base.TEMPLATE.rsplit('if __name__ == "__main__":', 1)
    if old.strip() != "main()":
        raise ValueError("The original HF job entry point has changed.")
    script = (template + ENTRY).replace("@@FOLDER@@", SIGLIP).replace("@@MIXED@@", FOLDER)
    script = script.replace("@@TRAIN_SHA256@@", hashlib.sha256((ROOT / "data/train.csv").read_bytes()).hexdigest())
    script = script.replace("@@BUNDLE@@", base64.b64encode(buffer.getvalue()).decode())
    compile(script, str(args.out), "exec")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(script)
    print(f"Generated {args.out} ({len(members)} bundled sources).")


if __name__ == "__main__":
    main()
