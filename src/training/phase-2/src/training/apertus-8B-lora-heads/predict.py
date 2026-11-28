"""End-to-end inference from images: VQ codes -> Apertus (LoRA merged) -> heads.

Writes a CSV with image_id and the six dimensions, the class probabilities (<out>.probs.pt) and the speed
(<out>.report.json): images/s over the whole pipeline, image loading included. Run the same command again after a
stop: images already predicted are not computed again. An image that cannot be read is reported and left out;
make_submission.py then keeps the constant answer for it.

  python src/training/apertus-8B-lora-heads/predict.py --checkpoint data/runs/lh_small/final --part coverage \\
      --out data/runs/lh_small/coverage.csv
  python src/training/apertus-8B-lora-heads/predict.py --checkpoint data/runs/lh_small/final --csv data/dev.csv \\
      --out data/runs/lh_small/dev.csv
  python src/training/apertus-8B-lora-heads/predict.py --checkpoint ... --csv data/final/final.csv --out .../final.csv

image_path is relative to the folder of the csv (--images-root changes it). With label columns in the csv and
data/metric.py, it also prints the macro F1.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import encoders as E  # noqa: E402
import model as M  # noqa: E402

ROOT = HERE.parents[2]


def fingerprint(checkpoint, use_context):
    """Identifies what produced a partial result, so that two models are never mixed."""
    digest = hashlib.sha256()
    for path in sorted(Path(checkpoint).rglob("*")):
        if path.is_file() and path.suffix in (".pt", ".safetensors", ".json") and path.name != "train_summary.json":
            digest.update(path.relative_to(checkpoint).as_posix().encode() + path.read_bytes())
    return f"{digest.hexdigest()[:16]}|context={use_context}"


def read_partial(path, expected):
    if not path.exists():
        path.write_text(json.dumps({"fingerprint": expected}) + "\n")
        return {}
    lines = path.read_text().splitlines()
    if json.loads(lines[0])["fingerprint"] != expected:
        sys.exit(f"{path} comes from another model or setting: choose another --out or delete it")
    done = {}
    for line in lines[1:]:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a line cut by a stop: that image is predicted again
        done[record["image_id"]] = record
    path.write_text("".join([lines[0] + "\n", *(json.dumps(r) + "\n" for r in done.values())]))
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder saved by train.py (final/ or best/)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="train.csv and metric.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=("coverage",), help="a part of train.csv")
    source.add_argument("--csv", type=Path, help="any csv with the columns of dev.csv")
    parser.add_argument("--images-root", type=Path, help="folder that image_path starts from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-context", action="store_true", help="robustness check: product context left out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible")
    device = torch.device(args.device)

    if args.part:
        rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
        root = args.images_root or args.data_dir
    else:
        rows = D.read_csv(args.csv)
        root = args.images_root or args.csv.parent
    if args.limit:
        rows = rows.head(args.limit)
    if rows.image_id.duplicated().any():
        sys.exit("the csv repeats an image_id")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(".partial.jsonl")
    done = read_partial(partial, fingerprint(args.checkpoint, not args.no_context))
    todo = [r for r in rows.to_dict("records") if r["image_id"] not in done]
    print(f"{len(rows)} images, {len(rows) - len(todo)} already predicted, {len(todo)} to do", flush=True)

    failed, seconds = [], 0.0
    if todo:
        from transformers import AutoConfig, AutoTokenizer

        directory = E.model_dir()
        loading = time.time()
        classifier, settings = M.load(directory, args.checkpoint, device)
        encoder = E.ImageEncoder(directory, args.device)
        tokenizer = AutoTokenizer.from_pretrained(directory)
        builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset)
        print(f"model loaded in {time.time() - loading:.0f} s; VQ side {settings['side']}", flush=True)

        def encode(batch):
            """(rows, encodings) of the images that can be read; the others are listed in failed."""
            try:
                return batch, encoder.encode([root / r["image_path"] for r in batch], [""] * len(batch),
                                             settings["side"])
            except Exception:
                good_rows, encodings = [], []
                for row in batch:
                    try:
                        encodings += encoder.encode([root / row["image_path"]], [""], settings["side"])
                        good_rows.append(row)
                    except Exception as error:
                        failed.append(row["image_id"])
                        print(f"{row['image_id']} cannot be read: {type(error).__name__}: {error}", flush=True)
                return good_rows, encodings

        batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
        started, count = time.time(), 0
        with ThreadPoolExecutor(1) as pool, torch.inference_mode(), partial.open("a") as keep:
            pending = pool.submit(encode, batches[0])  # encoding runs one batch ahead of the GPU forward
            for number in range(len(batches)):
                batch, encoded = pending.result()
                pending = pool.submit(encode, batches[number + 1]) if number + 1 < len(batches) else None
                if not batch:
                    continue
                sequences = [builder.ids(r, e["grid"], e["codes"], not args.no_context)
                             for r, e in zip(batch, encoded)]
                input_ids, attention = E.collate(sequences, tokenizer.pad_token_id)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = classifier(input_ids.to(device), attention.to(device))
                probabilities = [x.float().softmax(-1).cpu() for x in logits]
                for index, row in enumerate(batch):
                    record = {"image_id": row["image_id"],
                              "labels": {d: D.TAXONOMY[d][int(probabilities[t][index].argmax())]
                                         for t, d in enumerate(D.DIMENSIONS)},
                              "probabilities": [probabilities[t][index].tolist() for t in range(len(D.DIMENSIONS))]}
                    done[row["image_id"]] = record
                    keep.write(json.dumps(record) + "\n")
                keep.flush()
                count += len(batch)
                if (number + 1) % 20 == 0:
                    print(f"{count}/{len(todo)} images, {count / (time.time() - started):.2f} images/s", flush=True)
        seconds = time.time() - started

    predicted = [i for i in rows.image_id if i in done]
    predictions = pd.DataFrame([{"image_id": i, **done[i]["labels"]} for i in predicted],
                               columns=["image_id", *D.DIMENSIONS])
    predictions.to_csv(args.out, index=False)
    torch.save({"image_ids": predicted, "dimensions": list(D.DIMENSIONS),
                "probabilities": [torch.tensor([done[i]["probabilities"][t] for i in predicted])
                                  for t in range(len(D.DIMENSIONS))]}, args.out.with_suffix(".probs.pt"))
    new = len(todo) - len(failed)
    report = {"images": len(rows), "predicted": len(predicted), "unreadable": len(rows) - len(predicted),
              "new_in_this_run": new, "seconds": round(seconds, 1),
              "images_per_second": round(new / seconds, 2) if seconds else None,
              "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1)
              if torch.cuda.is_available() else None,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
              "batch_size": args.batch_size, "context": not args.no_context}
    labeled = all(f"{d}_source" in rows for d in D.DIMENSIONS)
    if labeled and (args.data_dir / "metric.py").exists():
        results = D.load_metric(args.data_dir).evaluate(D.solution_frame(rows), predictions, "image_id") \
            if len(predicted) == len(rows) else None
        if results:
            report["macro_f1"] = results["overall"]["f1_macro"]
            report["dimensions"] = {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
