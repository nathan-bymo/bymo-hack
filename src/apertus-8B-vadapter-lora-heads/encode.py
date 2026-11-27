"""Caches the frozen inputs of every training image: Apertus VQ codes and pooled SigLIP 2 features.

Both encoders are frozen, so they run once per image variant instead of once per epoch. The variants are exactly
those train.py will use (data.training_items on the same internal split, hence the same --val-fraction): the
original of every image, the mirrored copies and the augmented repeats of rare classes. Shards go to data/
(git-ignored). Run again to resume; changed repeat settings only add the missing variants.

  python src/apertus-8B-vadapter-lora-heads/encode.py --part training --out data/vlh_cache/training_256
  python src/apertus-8B-vadapter-lora-heads/encode.py --part all --val-fraction 0 --out data/vlh_cache/all_256
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import encoders as E  # noqa: E402

ROOT = HERE.parents[1]


def load_cache(folder):
    """{key: {"grid", "codes", "siglip"}} from all shards of a folder."""
    cache = {}
    for shard in sorted(Path(folder).glob("shard_*.pt")):
        cache.update(torch.load(shard, weights_only=False))
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--part", default="training", choices=("training", "all", "coverage"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=256, help="area of the VQ image, aspect ratio kept")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="the same as train.py")
    parser.add_argument("--no-repeats", action="store_true")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "siglip": E.SIGLIP_ID, "side": args.side,
                "pooled_side": E.POOLED_SIDE}
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != settings:
        sys.exit(f"{args.out} holds a cache with other settings: choose another folder")
    meta.write_text(json.dumps(settings, indent=2))

    rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    fit, validation = D.internal_split(rows, args.val_fraction)
    items = D.training_items(fit, D.label_matrix(fit), repeats=not args.no_repeats, mirrors=not args.no_mirrors)
    jobs = {D.key(i, ""): (i, p, "") for i, p in zip(rows.image_id, rows.image_path)}
    jobs |= {D.key(fit.image_id[i], v): (fit.image_id[i], fit.image_path[i], v) for i, v in items}
    done = set()
    for shard in args.out.glob("shard_*.pt"):
        done.update(torch.load(shard, weights_only=False))
    todo = [job for name, job in jobs.items() if name not in done]
    print(f"{len(rows)} images ({args.part}), {len(jobs)} variants with mirrors and augmented repeats, "
          f"{len(done)} cached, {len(todo)} to do", flush=True)
    if not todo:
        return

    encoder = E.ImageEncoder(E.model_dir())
    next_shard = len(list(args.out.glob("shard_*.pt")))
    started, processed, failures = time.time(), 0, []
    for start in range(0, len(todo), args.shard_size):
        chunk, shard = todo[start:start + args.shard_size], {}
        for begin in range(0, len(chunk), args.batch_size):
            batch = chunk[begin:begin + args.batch_size]
            try:
                results = encoder.encode([args.data_dir / p for _, p, _ in batch], [v for _, _, v in batch], args.side,
                                         [i for i, _, _ in batch])
            except Exception as error:  # one unreadable image must not stop the run; reported at the end
                failures.append(f"{batch[0][0]}..: {type(error).__name__}: {error}")
                continue
            shard.update({D.key(i, v): result for (i, _, v), result in zip(batch, results)})
        temporary = args.out / "writing.tmp"
        torch.save(shard, temporary)
        temporary.rename(args.out / f"shard_{next_shard:05d}.pt")
        next_shard += 1
        processed += len(chunk)
        print(f"{processed}/{len(todo)} images, {processed / (time.time() - started):.1f} images/s", flush=True)
    if failures:
        sys.exit(f"{len(failures)} batches failed: {failures[:3]}")


if __name__ == "__main__":
    main()
