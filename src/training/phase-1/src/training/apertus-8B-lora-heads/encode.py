"""Caches the Apertus VQ codes of every training image and of its mirrored copy.

The VQ tokenizer is frozen, so it runs once per image instead of once per epoch. Shards go to data/ (git-ignored).
Run again to resume.

  python src/training/apertus-8B-lora-heads/encode.py --train-csv data/heads_split/train_1500.csv --out data/lh_cache/small_256
  python src/training/apertus-8B-lora-heads/encode.py --part training --out data/lh_cache/training_256
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

ROOT = HERE.parents[2]


def load_cache(folder):
    """{key: {"grid", "codes"}} from all shards of a folder."""
    cache = {}
    for shard in sorted(Path(folder).glob("shard_*.pt")):
        cache.update(torch.load(shard, weights_only=True))
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--part", default="training", choices=("training", "all"))
    parser.add_argument("--train-csv", type=Path, help="a smaller split instead of --part")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=256, help="area of the VQ image, aspect ratio kept")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    settings = {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "side": args.side}
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != settings:
        sys.exit(f"{args.out} holds a cache with other settings: choose another folder")
    meta.write_text(json.dumps(settings, indent=2))

    rows = D.read_csv(args.train_csv) if args.train_csv else D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    items = D.training_items(rows, mirrors=not args.no_mirrors)
    jobs = {D.key(rows.image_id[i], v): (rows.image_path[i], v) for i, v in items}
    done = set()
    for shard in args.out.glob("shard_*.pt"):
        done.update(torch.load(shard, weights_only=True))
    todo = [(name, *job) for name, job in jobs.items() if name not in done]
    print(f"{len(rows)} images, {len(jobs)} with mirrored copies, {len(done)} cached, {len(todo)} to do", flush=True)
    if not todo:
        return

    encoder = E.ImageEncoder(E.model_dir(), args.device)
    next_shard = len(list(args.out.glob("shard_*.pt")))
    started, processed, failures = time.time(), 0, []
    for start in range(0, len(todo), args.shard_size):
        chunk, shard = todo[start:start + args.shard_size], {}
        for begin in range(0, len(chunk), args.batch_size):
            batch = chunk[begin:begin + args.batch_size]
            try:
                results = encoder.encode([args.data_dir / p for _, p, _ in batch], [v for _, _, v in batch], args.side)
            except Exception as error:  # one unreadable image must not stop the run; reported at the end
                failures.append(f"{batch[0][0]}..: {type(error).__name__}: {error}")
                continue
            shard.update({name: result for (name, _, _), result in zip(batch, results)})
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
