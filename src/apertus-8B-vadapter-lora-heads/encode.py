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
    parser.add_argument("--part", default="training", choices=("training", "all", "all_but_test", "coverage"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", type=int, default=256, help="area of the VQ image, aspect ratio kept")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="the same as train.py")
    parser.add_argument("--no-repeats", action="store_true")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ocr", action="store_true", help="also the OCR text of every original image (ocr.json)")
    args = parser.parse_args()

    settings = {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "siglip": E.SIGLIP_ID, "side": args.side,
                "pooled_side": E.POOLED_SIDE, "augmentation": E.AUGMENTATION}
    args.out.mkdir(parents=True, exist_ok=True)
    meta = args.out / "meta.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        if {k: v for k, v in old.items() if k != "augmentation"} != {k: v for k, v in settings.items()
                                                                       if k != "augmentation"}:
            sys.exit(f"{args.out} holds a cache with other settings: choose another folder")
        if old.get("augmentation") != E.AUGMENTATION:
            purge_augmented(args.out)
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
    todo = E.sort_by_vq_size(todo, lambda job: args.data_dir / job[1], args.side)  # full VQ batches
    print(f"{len(rows)} images ({args.part}), {len(jobs)} variants with mirrors and augmented repeats, "
          f"{len(done)} cached, {len(todo)} to do", flush=True)
    if todo:
        encode_images(args, todo)
    if args.ocr:
        read_text(args, rows)


def purge_augmented(folder):
    """Drops the augmented variants ("|a...") of another augmentation version; originals and mirrors stay valid."""
    dropped = 0
    for shard in sorted(Path(folder).glob("shard_*.pt")):
        content = torch.load(shard, weights_only=False)
        kept = {k: v for k, v in content.items() if "|a" not in k}
        dropped += len(content) - len(kept)
        torch.save(kept, shard)
    print(f"augmentation changed: {dropped} cached variants dropped, they will be recomputed", flush=True)


def read_text(args, rows):
    """OCR of the original images only: every variant of an image gets the text of its original."""
    from concurrent.futures import ThreadPoolExecutor

    settings = {"detector": E.OCR_DETECTOR, "recognizer": E.OCR_RECOGNIZER, "revision": E.OCR_RECOGNIZER_REVISION,
                "min_confidence": E.OCR_MIN_CONFIDENCE, "max_chars": E.OCR_MAX_CHARS}
    meta, path = args.out / "ocr_meta.json", args.out / "ocr.json"
    if meta.exists() and json.loads(meta.read_text()) != settings:
        sys.exit(f"{path} was read with other OCR settings: delete it or choose another folder")
    meta.write_text(json.dumps(settings, indent=2))
    texts = json.loads(path.read_text()) if path.exists() else {}
    todo = [(i, p) for i, p in zip(rows.image_id, rows.image_path) if i not in texts]
    print(f"OCR: {len(texts)} images read, {len(todo)} to do", flush=True)
    if not todo:
        return
    reader, started = E.TextReader(), time.time()
    with ThreadPoolExecutor(16) as pool:
        for start in range(0, len(todo), 64):
            batch = todo[start:start + 64]
            pictures = list(pool.map(lambda job: E.full_resolution(args.data_dir / job[1]), batch))
            texts.update(zip([i for i, _ in batch], reader.read(pictures)))
            done = start + len(batch)
            if done % 1024 < 64 or done == len(todo):
                temporary = args.out / "ocr.tmp"
                temporary.write_text(json.dumps(texts, ensure_ascii=False))
                temporary.rename(path)
                found = sum(bool(texts[i]) for i, _ in todo[:done])
                print(f"OCR {done}/{len(todo)} images, {done / (time.time() - started):.1f} images/s, "
                      f"text found in {found / done:.0%}", flush=True)


def encode_images(args, todo):
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
                if processed == 0 and begin == 0:
                    raise  # the very first batch: a bug, not a bad image; stop before wasting GPU time
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
