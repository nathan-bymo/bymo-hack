"""End-to-end inference from images: VQ codes -> fully finetuned Apertus (24 layers) -> 6 heads.

Writes a CSV with image_id and the six dimensions (argmax, never an empty cell), the class probabilities (for bias
calibration or ensembles) and the speed: images/s over the whole pipeline, image loading and VQ encoding included.

  python src/apertus-8B-fullft-heads/ft_predict.py --checkpoint data/runs/fullft_phase1/best --part coverage \
      --out data/runs/fullft_phase1/coverage.csv
  python src/apertus-8B-fullft-heads/ft_predict.py --checkpoint data/runs/fullft_phase1/best \
      --csv data/dev.csv --out data/runs/fullft_phase1/dev_submission.csv

With labels (--part coverage, or a csv with label columns) it also prints the macro F1 of metric.py.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ft_model as M  # noqa: E402  (puts apertus-8B-vadapter-lora-heads on the path)
import data as D  # noqa: E402
import encoders as E  # noqa: E402
from ft_encode import VQEncoder, model_dir  # noqa: E402
from ft_train import autocast  # noqa: E402

ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder saved by ft_train.py (best/ or last/)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="image_path starts from here")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=("coverage",), help="a part of train.csv")
    source.add_argument("--csv", type=Path, help="any csv with the columns of dev.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-context", action="store_true", help="robustness check: product context left out")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    settings = json.loads((args.checkpoint / "settings.json").read_text())
    if settings.get("tiny"):
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        sys.exit("no CUDA GPU is visible")
    rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part) if args.part else D.read_csv(args.csv)
    if args.limit:
        rows = rows.head(args.limit)
    from transformers import AutoConfig, AutoTokenizer

    directory = model_dir(full=not settings.get("tiny"))
    loading = time.time()
    classifier, settings = M.load(directory, args.checkpoint, device)
    encoder = VQEncoder(directory, device)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset, prefix_tokens=0)
    print(f"model loaded in {time.time() - loading:.0f} s; {len(rows)} images, VQ side {settings['side']}", flush=True)

    records = rows.to_dict("records")
    batches = [records[i:i + args.batch_size] for i in range(0, len(records), args.batch_size)]

    def encode(batch):  # runs one batch ahead of the decoder forward
        return encoder.encode([args.data_dir / r["image_path"] for r in batch], [""] * len(batch), settings["side"])

    parts = [[] for _ in D.DIMENSIONS]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with ThreadPoolExecutor(1) as pool, torch.inference_mode():
        pending = pool.submit(encode, batches[0]) if batches else None
        for number, batch in enumerate(batches):
            encoded = pending.result()
            pending = pool.submit(encode, batches[number + 1]) if number + 1 < len(batches) else None
            sequences = [builder.ids(r, e["grid"], e["codes"], not args.no_context) for r, e in zip(batch, encoded)]
            input_ids, attention = E.collate(sequences, tokenizer.pad_token_id)
            with autocast(device):
                logits = classifier(input_ids.to(device), attention.to(device))
            for task, task_logits in enumerate(logits):
                parts[task].append(task_logits.float().softmax(-1).cpu())
            if (number + 1) % 20 == 0:
                done = sum(len(b) for b in batches[:number + 1])
                print(f"{done}/{len(records)} images, {done / (time.time() - started):.2f} images/s", flush=True)
    seconds = time.time() - started
    probabilities = [torch.cat(p) for p in parts]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions = D.prediction_frame(rows.image_id, probabilities)
    predictions.to_csv(args.out, index=False)
    torch.save({"image_ids": list(rows.image_id), "dimensions": D.DIMENSIONS, "probabilities": probabilities},
               args.out.with_suffix(".probs.pt"))
    report = {"images": len(rows), "seconds": round(seconds, 1), "images_per_second": round(len(rows) / seconds, 2),
              "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1) if device.type == "cuda" else 0,
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "batch_size": args.batch_size, "context": not args.no_context, "checkpoint": str(args.checkpoint)}
    if all(f"{d}_source" in rows for d in D.DIMENSIONS):
        results = D.load_metric(args.data_dir).evaluate(D.solution_frame(rows), predictions, "image_id")
        report["macro_f1"] = results["overall"]["f1_macro"]
        report["weighted_f1"] = results["overall"].get("f1_weighted")
        report["dimensions"] = {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
