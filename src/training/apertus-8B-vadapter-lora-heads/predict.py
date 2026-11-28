"""End-to-end inference from images: VQ codes + SigLIP -> projector -> Apertus (LoRA merged) -> heads.

Writes a CSV with image_id and the six dimensions (never an empty cell), the class probabilities (for bias
calibration or ensembles) and the speed: images/s over the whole pipeline, image loading included.

  python src/training/apertus-8B-vadapter-lora-heads/predict.py --checkpoint data/runs/vlh_phase1/best --part coverage \
      --out data/runs/vlh_phase1/coverage.csv
  python src/training/apertus-8B-vadapter-lora-heads/predict.py --checkpoint data/runs/vlh_phase1/best \
      --csv data/dev.csv --out data/runs/vlh_phase1/dev_submission.csv

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
import data as D  # noqa: E402
import encoders as E  # noqa: E402
import model as M  # noqa: E402

ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder saved by train.py (best/ or last/)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="image_path starts from here")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--part", choices=("coverage", "test", "validation"),
                        help="a part of train.csv; validation = internal validation of train.py (--val-fraction)")
    source.add_argument("--csv", type=Path, help="any csv with the columns of dev.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1, help="as in train.py, for --part validation")
    parser.add_argument("--train-part", default="training", help="the --part of train.py, for --part validation")
    parser.add_argument("--bias", type=Path, help="per-class logit biases of calibrate.py, added before the argmax")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-context", action="store_true", help="robustness check: product context left out")
    parser.add_argument("--no-merge", action="store_true", help="keep LoRA unmerged")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"),
                        help="float16 on GPUs without bf16 (T4): MLPs guarded and calibrated against overflow")
    parser.add_argument("--encoder-device", default="cuda", help="GPU of VQ, SigLIP and OCR, e.g. cuda:1 on 2 x T4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tta-flip", action="store_true",
                        help="images predicted without text: also the mirrored image (left/right swapped), averaged")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible")

    if args.part == "validation":
        rows = D.internal_split(D.select(D.read_csv(args.data_dir / "train.csv"), args.train_part), args.val_fraction)[1]
    elif args.part:
        rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
    else:
        rows = D.read_csv(args.csv)
    if args.limit:
        rows = rows.head(args.limit)
    from transformers import AutoConfig, AutoTokenizer

    directory = E.model_dir()
    loading = time.time()
    dtype = getattr(torch, args.dtype)
    classifier, settings = M.load(directory, args.checkpoint, "cuda:0", merge=not args.no_merge, dtype=dtype)
    encoder = E.ImageEncoder(directory, args.encoder_device)
    reader = E.TextReader(args.encoder_device) if settings.get("ocr") else None
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset,
                              E.POOLED_SIDE**2 if settings.get("adapter", True) else 0, settings.get("language_hint", False))
    print(f"model loaded in {time.time() - loading:.0f} s; {len(rows)} images, VQ side {settings['side']}", flush=True)

    original_order = list(rows.image_id)
    records = E.sort_by_vq_size(rows.to_dict("records"), lambda r: args.data_dir / r["image_path"], settings["side"])
    rows = rows.set_index("image_id", drop=False).loc[[r["image_id"] for r in records]].reset_index(drop=True)
    batches = [records[i:i + args.batch_size] for i in range(0, len(records), args.batch_size)]

    stage_seconds = {"ocr": 0.0, "forward": 0.0, "tta": 0.0}

    def encode(batch):  # runs one batch ahead of the GPU forward
        paths = [args.data_dir / r["image_path"] for r in batch]
        encoded = encoder.encode(paths, [""] * len(batch), settings["side"])
        clock = time.perf_counter()
        texts = reader.read([E.full_resolution(path) for path in paths]) if reader else [None] * len(batch)
        stage_seconds["ocr"] += time.perf_counter() - clock
        for entry, text in zip(encoded, texts):
            entry["ocr"] = text
        return encoded

    def forward(batch, encoded):
        sequences = [builder.ids(r, e["grid"], e["codes"], not args.no_context, e["ocr"])
                     for r, e in zip(batch, encoded)]
        input_ids, attention = E.collate(sequences, tokenizer.pad_token_id)
        siglip = torch.stack([e["siglip"] for e in encoded])
        with torch.autocast("cuda", dtype=dtype):
            logits = classifier(input_ids.to("cuda:0"), attention.to("cuda:0"), siglip.to("cuda:0"))
        probs = [task_logits.float().softmax(-1).cpu() for task_logits in logits]
        if any(not torch.isfinite(p).all() for p in probs):
            if args.dtype != "float16" or getattr(classifier, "safe_fp32", False):
                raise RuntimeError("non-finite probabilities")
            with torch.autocast("cuda", dtype=dtype):
                where = M.first_non_finite(classifier, input_ids.to("cuda:0"), attention.to("cuda:0"),
                                           siglip.to("cuda:0"))
            print("fp16 overflow; first non-finite modules:\n  " + "\n  ".join(where or ["(none found by hooks)"]),
                  flush=True)
            print(f"switching to the safe mode: {M.fp32_compute(classifier.language_model)} linear layers compute in "
                  "float32 (weights stay float16 in memory)", flush=True)
            classifier.safe_fp32 = True
            fallback["used"], fallback["where"] = True, where
            return forward(batch, encoded)
        return probs

    fallback = {"used": False, "where": []}
    text_none = [D.TAXONOMY[d].index("none") for d in ("text_classification", "text_language")]
    text_tasks = [D.DIMENSIONS.index(d) for d in ("text_classification", "text_language")]

    def flip_average(batch, encoded, probs):
        """Mirror test-time augmentation, only where the model sees no text (a mirror would garble it): the
        mirrored image gives a second opinion; its perspective probabilities have left and right swapped."""
        chosen = [i for i in range(len(batch))
                  if all(probs[t][i].argmax() == none for t, none in zip(text_tasks, text_none))]
        if not chosen:
            return probs
        mirrored = encoder.encode([args.data_dir / batch[i]["image_path"] for i in chosen], ["m"] * len(chosen),
                                  settings["side"])
        for entry, i in zip(mirrored, chosen):
            entry["ocr"] = encoded[i]["ocr"]  # no text was seen: the OCR text of the original stays
        flipped = forward([batch[i] for i in chosen], mirrored)
        flipped[D.PERSPECTIVE] = flipped[D.PERSPECTIVE][:, D.MIRROR_PERMUTATION]
        flip_counter[0] += len(chosen)
        result = [p.clone() for p in probs]
        for task in range(len(result)):
            result[task][chosen] = (probs[task][chosen] + flipped[task]) / 2
        return result

    flip_counter = [0]
    parts = [[] for _ in D.DIMENSIONS]
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with ThreadPoolExecutor(1) as pool, torch.inference_mode():
        pending = pool.submit(encode, batches[0]) if batches else None
        for number, batch in enumerate(batches):
            encoded = pending.result()
            pending = pool.submit(encode, batches[number + 1]) if number + 1 < len(batches) else None
            clock = time.perf_counter()
            probs = forward(batch, encoded)
            stage_seconds["forward"] += time.perf_counter() - clock
            if args.tta_flip:
                clock = time.perf_counter()
                probs = flip_average(batch, encoded, probs)
                stage_seconds["tta"] += time.perf_counter() - clock
            for task, task_probs in enumerate(probs):
                parts[task].append(task_probs)
            if (number + 1) % 20 == 0:
                done = sum(len(b) for b in batches[:number + 1])
                print(f"{done}/{len(records)} images, {done / (time.time() - started):.2f} images/s", flush=True)
    seconds = time.time() - started
    probabilities = [torch.cat(p) for p in parts]
    if args.bias:  # the saved probabilities stay uncalibrated; only the decision moves
        import calibrate

        decision = calibrate.apply(probabilities, json.loads(args.bias.read_text()))
    else:
        decision = probabilities

    args.out.parent.mkdir(parents=True, exist_ok=True)
    predictions = D.prediction_frame(rows.image_id, decision)
    predictions = predictions.set_index("image_id", drop=False).loc[original_order].reset_index(drop=True)
    predictions.to_csv(args.out, index=False)
    torch.save({"image_ids": list(rows.image_id), "dimensions": D.DIMENSIONS, "probabilities": probabilities},
               args.out.with_suffix(".probs.pt"))
    report = {"images": len(rows), "seconds": round(seconds, 1), "images_per_second": round(len(rows) / seconds, 2),
              "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
              "gpu": torch.cuda.get_device_name(0), "gpus": torch.cuda.device_count(), "dtype": args.dtype,
              "fp32_safe_mode": fallback["used"], "first_non_finite": fallback["where"],
              "batch_size": args.batch_size, "context": not args.no_context,
              "checkpoint": str(args.checkpoint), "tta_flip": args.tta_flip, "flipped_images": flip_counter[0],
              "seconds_per_stage": {k: round(v, 1) for k, v in {**encoder.seconds, **stage_seconds}.items()},
              "note": "images, siglip, vq and ocr run in a thread one batch ahead of forward: stages overlap"}
    if all(f"{d}_source" in rows for d in D.DIMENSIONS):
        results = D.load_metric(args.data_dir).evaluate(D.solution_frame(rows), predictions, "image_id")
        report["macro_f1"] = results["overall"]["f1_macro"]
        report["dimensions"] = {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
