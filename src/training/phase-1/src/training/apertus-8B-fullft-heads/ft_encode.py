"""Caches the Apertus VQ codes of every image variant ft_train.py will see (no SigLIP in this model).

The variants are exactly those of apertus-8B-vadapter-lora-heads (data.training_items on the same internal split,
see its DATI.md): the original of every image, the mirrored copies of left/right perspectives without text and the
augmented repeats of rare classes (mild crop, mirrored when there is no text). The VQ tokenizer is frozen, so it
runs once per variant instead of once per epoch. Shards go to data/ (git-ignored); run again to resume.

Several processes can share one GPU (the VQ tokenizer encodes one image at a time and leaves most of it idle):
--worker i --workers n takes every n-th variant and writes its own shards.

  python src/training/apertus-8B-fullft-heads/ft_encode.py --part training --out data/fullft_cache/training_256
  for i in 0 1 2; do python src/training/apertus-8B-fullft-heads/ft_encode.py --part training \
      --out data/fullft_cache/training_256 --worker $i --workers 3 & done; wait
  python src/training/apertus-8B-fullft-heads/ft_encode.py --part all --val-fraction 0 --out data/fullft_cache/all_256
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
VLH = HERE.parent / "apertus-8B-vadapter-lora-heads"
for path in (VLH, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import data as D  # noqa: E402
import encoders as E  # noqa: E402  (apertus-8B-vadapter-lora-heads/encoders.py)
from common import images  # noqa: E402

ROOT = HERE.parents[2]
# Files enough for tokenizer, chat template, config and the VQ tokenizer: no 17 GB download for a CPU test.
LIGHT_FILES = ["*.json", "*.jinja", "model-vision_tokenizer-*"]
MAX_FAILURES = 10  # per worker


def model_dir(full=True):
    from huggingface_hub import snapshot_download

    return snapshot_download(E.MODEL_ID, revision=E.MODEL_REVISION, allow_patterns=None if full else LIGHT_FILES)


def vq_image(path, variant, side, name=""):
    """The image the VQ tokenizer sees: common.images preparation, or a training variant of encoders.py."""
    if variant in ("", "m"):
        return images.prepared_image(path, side, variant == "m")
    return E.variant_images(path, variant, side, name)[0]


class VQEncoder(E.ImageEncoder):
    """The frozen VQ tokenizer of Apertus only (fp32, no TF32: codes are an argmax); vq_codes is inherited."""

    def __init__(self, directory, device="cuda", threads=8):  # noqa: super().__init__ would also load SigLIP
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VisionTokenizerModel

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        directory = Path(directory)
        vq = Apertus1p5VisionTokenizerModel(AutoConfig.from_pretrained(directory).vision_tokenizer_config)
        weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        state = {}
        for name in sorted({f for k, f in weight_map.items() if "vision_tokenizer." in k}):
            with safe_open(str(directory / name), framework="pt") as handle:
                for key in handle.keys():
                    if "vision_tokenizer." in key:
                        state[key.split("vision_tokenizer.", 1)[1]] = handle.get_tensor(key)
        missing, _ = vq.load_state_dict(state, strict=False)
        assert not missing, f"vision tokenizer weights missing: {missing[:5]}"
        self.device = torch.device(device)
        self.vq = vq.to(self.device, torch.float32).eval()
        self.apertus_processor = AutoProcessor.from_pretrained(directory).image_processor
        self.pool = ThreadPoolExecutor(threads)  # PIL releases the GIL while decoding and resizing

    def encode(self, paths, variants, side, names=None):
        """One {"grid", "codes"} per image; images are prepared in threads, codes computed one at a time."""
        names = names or [str(p) for p in paths]
        prepared = list(self.pool.map(lambda job: vq_image(job[0], job[1], side, job[2]),
                                     zip(paths, variants, names)))
        results = []
        for image in prepared:
            grid_h, grid_w, codes = self.vq_codes(image)
            results.append({"grid": (grid_h, grid_w), "codes": codes})
        return results


def load_cache(folder):
    """{key: {"grid", "codes"}} from the shards of every worker."""
    cache = {}
    for shard in sorted(Path(folder).glob("shard_*.pt")):
        cache.update(torch.load(shard, weights_only=False))
    return cache


def variant_jobs(rows, fit, items):
    """{cache key: (image id, image path, variant)}: every original image, then the training variants."""
    jobs = {D.key(i): (i, p, "") for i, p in zip(rows.image_id, rows.image_path)}
    jobs |= {D.key(fit.image_id[i], v): (fit.image_id[i], fit.image_path[i], v) for i, v in items}
    return jobs


def worker_jobs(jobs, worker, workers):
    """Every workers-th job, starting at worker: the workers cover all jobs, each exactly once."""
    return dict(list(jobs.items())[worker::workers])


def write_meta(folder, settings):
    meta = folder / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != settings:
        sys.exit(f"{folder} holds a cache with other settings: choose another folder")
    temporary = folder / f"meta.{os.getpid()}.tmp"  # atomic: several workers write the same content
    temporary.write_text(json.dumps(settings, indent=2))
    temporary.replace(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--part", default="training", choices=("training", "all"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=256, help="area of the VQ image, aspect ratio kept")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="the same as ft_train.py")
    parser.add_argument("--no-repeats", action="store_true")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32, help="images prepared together in threads")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows, for a test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert 0 <= args.worker < args.workers

    args.out.mkdir(parents=True, exist_ok=True)
    write_meta(args.out, {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "side": args.side})
    rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    fit, validation = D.internal_split(rows, args.val_fraction)
    items = D.training_items(fit, D.label_matrix(fit), repeats=not args.no_repeats, mirrors=not args.no_mirrors)
    jobs = worker_jobs(variant_jobs(rows, fit, items), args.worker, args.workers)
    prefix = f"shard_w{args.worker}of{args.workers}_"
    done = set()
    for shard in args.out.glob(f"{prefix}*.pt"):
        done.update(torch.load(shard, weights_only=False))
    todo = [job for name, job in jobs.items() if name not in done]
    print(f"worker {args.worker}/{args.workers}: {len(rows)} images ({args.part}), {len(jobs)} variants for this "
          f"worker, {len(done)} cached, {len(todo)} to do", flush=True)
    if not todo:
        return

    encoder = VQEncoder(model_dir(full=False), args.device)
    next_shard = len(list(args.out.glob(f"{prefix}*.pt")))
    started, processed, failures = time.time(), 0, []
    for start in range(0, len(todo), args.shard_size):
        chunk, shard = todo[start:start + args.shard_size], {}
        for begin in range(0, len(chunk), args.batch_size):
            batch = chunk[begin:begin + args.batch_size]
            try:
                results = encoder.encode([args.data_dir / p for _, p, _ in batch], [v for _, _, v in batch],
                                         args.side, [i for i, _, _ in batch])
            except Exception:  # find the unreadable image(s): one bad file must not cost the whole batch
                results = []
                for job in batch:
                    try:
                        results += encoder.encode([args.data_dir / job[1]], [job[2]], args.side, [job[0]])
                    except Exception as error:
                        failures.append(f"{D.key(job[0], job[2])}: {type(error).__name__}: {error}")
                        results.append(None)
            shard.update({D.key(i, v): r for (i, _, v), r in zip(batch, results) if r is not None})
        temporary = args.out / f"{prefix}writing.tmp"
        torch.save(shard, temporary)
        temporary.rename(args.out / f"{prefix}{next_shard:05d}.pt")
        next_shard += 1
        processed += len(chunk)
        print(f"worker {args.worker}: {processed}/{len(todo)} variants, "
              f"{processed / (time.time() - started):.1f} variants/s", flush=True)
    if failures:  # ft_train.py drops a few missing variants (MAX_MISSING) and refuses more
        (args.out / f"failures_w{args.worker}.json").write_text(json.dumps(failures, indent=2))
        print(f"worker {args.worker}: {len(failures)} variants failed, e.g. {failures[:3]}", flush=True)
        if len(failures) > MAX_FAILURES:
            sys.exit(f"worker {args.worker}: too many failures, not an unreadable image or two")


if __name__ == "__main__":
    main()
