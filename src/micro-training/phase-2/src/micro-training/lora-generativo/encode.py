"""Precomputes the Apertus VQ codes of the training images (and their mirrored copies) on one GPU.

Run one process per GPU with --shard i --num-shards n; each writes its own file, and train.py reads them all.

    python encode.py --model-dir M --data-dir D --side 256 --part training --shard 0 --num-shards 2 --out codes
"""

import argparse
import pickle
import time
from pathlib import Path

import apertus_lora as L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--part", default="training", choices=("training", "all", "coverage", *L.PROTOTYPE_PARTS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows, for a quick test")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = L.select_rows(L.read_train(args.data_dir), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    records = rows.to_dict("records")
    jobs = L.image_jobs(records)
    if args.no_mirrors:
        jobs = [job for job in jobs if not job[2]]
    jobs = jobs[args.shard :: args.num_shards]

    out_path = Path(args.out) / f"codes_{args.part}_{args.side}_shard{args.shard}of{args.num_shards}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codes = pickle.loads(out_path.read_bytes()) if out_path.exists() else {}
    todo = [job for job in jobs if L.code_key(job[0], job[2]) not in codes]

    vision_tokenizer, processor, _ = L.load_vision_tokenizer(args.model_dir, args.device)
    started, failures = time.time(), []
    for number, (image_id, image_file, mirrored) in enumerate(todo, 1):
        try:
            image = L.images.prepared_image(image_file, args.side, mirrored)
            codes[L.code_key(image_id, mirrored)] = L.encode_image(vision_tokenizer, processor, image)
        except Exception as error:  # one unreadable image must not stop the run; it is reported below
            failures.append(f"{image_id}: {type(error).__name__}: {error}")
        if number % 500 == 0 or number == len(todo):
            rate = number / (time.time() - started)
            print(f"shard {args.shard}: {number}/{len(todo)} images, {rate:.2f} images/s", flush=True)
            out_path.write_bytes(pickle.dumps(codes))
    out_path.write_bytes(pickle.dumps(codes))
    print(f"shard {args.shard}: {len(codes)} codes in {out_path}, {len(failures)} failures {failures[:3]}", flush=True)


if __name__ == "__main__":
    main()
