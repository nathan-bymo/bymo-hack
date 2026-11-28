"""Small frozen-SigLIP linear probe; this is not an Apertus adapter benchmark."""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

from supervision import DIMENSIONS, split_by_product

MODEL = "google/siglip2-base-patch16-224"


def eligible(row, dimension):
    return row[f"{dimension}_source"] == "human" and row[f"{dimension}_relevant"].lower() == "true"


def opaque_image(path):
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(white, rgba).convert("RGB")


def score(truth, predicted):
    classes = sorted(set(truth) | set(predicted))
    if not classes:
        return None
    f1s = []
    for label in classes:
        tp = sum(a == label and b == label for a, b in zip(truth, predicted, strict=True))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted, strict=True))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(f1s) / len(f1s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--holdout-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; explicitly select --device cpu if desired")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    with (args.data_dir / "train.csv").open(newline="", encoding="utf-8") as handle:
        train, holdout = split_by_product(list(csv.DictReader(handle)), seed=args.seed)
    rng = random.Random(args.seed)
    train = rng.sample(train, min(args.train_size, len(train)))
    holdout = rng.sample(holdout, min(args.holdout_size, len(holdout)))
    all_rows = train + holdout
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    vision = SiglipVisionModel.from_pretrained(MODEL, local_files_only=True).to(args.device).eval()
    features = []
    begin = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(all_rows), args.batch_size):
            rows = all_rows[start:start + args.batch_size]
            images = [opaque_image(args.data_dir / r["image_path"]) for r in rows]
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(args.device)
            features.append(vision(pixel_values=pixels).last_hidden_state.mean(dim=1).cpu())
            print(f"Features: {min(start + len(rows), len(all_rows))}/{len(all_rows)}", flush=True)
    features = torch.cat(features).float()
    n_train = len(train)
    dims = {}
    for dimension in DIMENSIONS:
        train_indices = [i for i, row in enumerate(train) if eligible(row, dimension)]
        valid_indices = [i for i, row in enumerate(holdout) if eligible(row, dimension)]
        counts = Counter(train[i][dimension] for i in train_indices)
        if not counts or not valid_indices:
            dims[dimension] = {"error": "No eligible training or validation cells"}
            continue
        classes = sorted(counts)
        class_ids = {name: i for i, name in enumerate(classes)}
        labels = torch.tensor([class_ids[train[i][dimension]] for i in train_indices])
        x = features[train_indices]
        head = torch.nn.Linear(x.shape[1], len(classes))
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=0.01)
        for _ in range(args.epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(x), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            values = head(features[[n_train + i for i in valid_indices]]).argmax(-1).tolist()
        predicted = [classes[i] for i in values]
        truth = [holdout[i][dimension] for i in valid_indices]
        majority = counts.most_common(1)[0][0]
        dims[dimension] = {
            "train_scored_cells": len(train_indices), "holdout_scored_cells": len(valid_indices),
            "train_classes": len(classes), "holdout_classes": len(set(truth)),
            "probe_macro_f1": score(truth, predicted),
            "majority_macro_f1": score(truth, [majority] * len(truth)),
        }
    present = [v for v in dims.values() if "probe_macro_f1" in v]
    result = {
        "experiment": "frozen_siglip2_linear_probe_not_apertus", "model": MODEL,
        "seed": args.seed, "train_images": len(train), "holdout_images": len(holdout),
        "elapsed_seconds": round(time.monotonic() - begin, 2), "device": args.device,
        "mean_macro_f1": sum(v["probe_macro_f1"] for v in present) / len(present) if present else None,
        "mean_majority_macro_f1": sum(v["majority_macro_f1"] for v in present) / len(present) if present else None,
        "dimensions": dims,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
