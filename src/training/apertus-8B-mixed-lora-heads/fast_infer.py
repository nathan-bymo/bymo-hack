"""Fast end-to-end inference of the mixed model (SigLIP 2 + DINOv3 + VQ + OCR, one Apertus decoder).

Same inputs as train_mixed.export, organized for speed:
  - LoRA merged into the decoder weights (--merge-lora): no adapter matmuls per layer;
  - images sorted by VQ size (--sort) and larger batches (--batch): the VQ tokenizer batches only images of the same
    size, so small unsorted batches ran it almost one image at a time. Codes stay those of one image at a time;
  - image decoding in a thread pool one batch ahead, and OCR in its own thread one batch ahead (--ocr-thread);
  - unreadable images are replaced by a blank image, so every row gets a prediction.

Optional test-time augmentation for perspective only (--tta-perspective): the mirrored image goes through the model,
its perspective probabilities are remapped left<->right and averaged with the original; the other five dimensions
always come from the original view, and OCR is always read from the original. "all" stores both views so that the
selective variant can be evaluated offline; "selective" mirrors only rows whose two most likely perspective classes
are a left/right pair with a margin below --margin.

  python fast_infer.py --checkpoint <mixed best> --data-dir <data> --csv <rows.csv> --out <dir>/<name> \
      [--reference <probs.pt>] [--labels] --merge-lora --sort --batch 32 --ocr-thread

Writes <name>.csv, <name>.probs.pt (and <name>.flip.probs.pt with --tta-perspective all) and <name>.json (speed per
stage, peak memory, parity against --reference, metrics with --labels).
"""

import argparse
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mixed_model import D, DINO_SIDE, E, load_mixed  # noqa: E402
from pipeline import DinoEncoder, probabilities  # noqa: E402
from common import images  # noqa: E402

SIDE = 256
PERSPECTIVE = D.DIMENSIONS.index("perspective")
PAIRS = {frozenset(pair) for pair in (("front_angle_left", "front_angle_right"), ("back_angle_left", "back_angle_right"),
                                       ("left_side", "right_side"))}


def views(path, mirrored=False):
    """(VQ image, original on white for SigLIP and OCR, DINO image), exactly as MixedEncoder prepares them."""
    original = E.full_resolution(path, mirrored)
    return (images.prepared_image(path, SIDE, mirrored), original,
            original.resize((DINO_SIDE, DINO_SIDE), Image.Resampling.BICUBIC))


def safe_views(path, blank):
    try:
        return views(path), False
    except Exception as error:  # noqa: BLE001 - every row must get a prediction
        print(f"unreadable image replaced by a blank one: {type(error).__name__}: {str(error)[:120]}", flush=True)
        return views(blank), True


class Timer:
    def __init__(self):
        self.seconds = {}

    def add(self, name, start, sync=True):
        if sync:
            torch.cuda.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def producer(batches, data_dir, blank, out, pool):
    for records in batches:
        loaded = list(pool.map(lambda row: safe_views(data_dir / row["image_path"], blank), records))
        out.put((records, [v for v, _ in loaded], [r["image_id"] for r, (_, bad) in zip(records, loaded) if bad]))
    out.put(None)


def ocr_stage(reader, source, out, timer):
    while (item := source.get()) is not None:
        records, loaded, replaced = item
        start = time.perf_counter()
        texts = reader.read([original for _, original, _ in loaded])
        timer.add("ocr", start, sync=False)
        out.put((records, loaded, replaced, texts))
    out.put(None)


@torch.inference_mode()
def forward(model, builder, encoder, dino, records, loaded, texts, timer):
    start = time.perf_counter()
    siglip = encoder.siglip_features([original for _, original, _ in loaded])
    timer.add("siglip", start)
    start = time.perf_counter()
    codes = encoder.vq_codes_batch([vq for vq, _, _ in loaded])
    timer.add("vq", start)
    start = time.perf_counter()
    dino_features = dino.encode([d for _, _, d in loaded], DINO_SIDE)
    timer.add("dino", start)
    start = time.perf_counter()
    sequences = [builder.ids(row, (h, w), c, ocr=text)
                 for row, (h, w, c), text in zip(records, codes, texts, strict=True)]
    ids, mask = E.collate(sequences, builder.tokenizer.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids.cuda(), mask.cuda(), siglip.cuda(), dino_features.cuda())
    probs = probabilities(logits)
    timer.add("apertus", start)
    return probs


def mirrored_perspective(probs):
    return probs[:, D.MIRROR_PERMUTATION]


def needs_flip(perspective, margin):
    top = perspective.topk(2, dim=-1)
    names = D.TAXONOMY["perspective"]
    return [frozenset((names[a], names[b])) in PAIRS and float(pa - pb) < margin
            for (a, b), (pa, pb) in zip(top.indices.tolist(), top.values.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True, help="rows with image_id, image_path and the context")
    parser.add_argument("--part", default="", help="test: only the prompt-test rows of --csv (train.csv)")
    parser.add_argument("--out", type=Path, required=True, help="output prefix")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--ocr-thread", action="store_true")
    parser.add_argument("--tta-perspective", choices=["off", "all", "selective"], default="off")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--reference", type=Path, help="probs.pt of the same rows, for the parity check")
    parser.add_argument("--labels", action="store_true", help="score with metric.py (rows carry labels)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tf32", action="store_true",
                        help="TF32 for every fp32 convolution and matmul (VQ, SigLIP, OCR); codes are checked first")
    args = parser.parse_args()

    loading = time.perf_counter()
    directory = E.model_dir()
    model, settings, builder = load_mixed(directory, args.checkpoint)
    if not settings.get("mixed"):
        raise ValueError("Expected a trained mixed checkpoint.")
    if args.merge_lora and settings.get("lora_rank"):
        model.aperture.language_model = model.aperture.language_model.merge_and_unload()
    model.eval()
    encoder = E.ImageEncoder(directory, "cuda")
    if settings.get("vision_lora_rank"):  # a checkpoint trained with a LoRA on SigLIP: merged into the encoder
        encoder.add_vision_lora(args.checkpoint / "vision_lora")
    dino = DinoEncoder(settings["dino_revision"], device="cuda", grid_side=E.POOLED_SIDE)
    reader = E.TextReader("cuda")
    load_seconds = time.perf_counter() - loading

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gib < 100 and args.batch > 16:  # 80 GB GPUs (H100, A100): the batch-32 path peaks near 88 GB
        print(f"{total_gib:.0f} GiB GPU: batch {args.batch} -> 16", flush=True)
        args.batch = 16
    rows = D.read_csv(args.csv)
    if args.part:
        rows = D.select(rows, args.part)
    if args.limit:
        rows = rows.head(args.limit)
    records = rows.to_dict("records")
    if args.sort:
        records = E.sort_by_vq_size(records, lambda row: args.data_dir / row["image_path"], SIDE)
    batches = [records[i:i + args.batch] for i in range(0, len(records), args.batch)]
    blank = args.out.parent / "blank.png"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), (255, 255, 255)).save(blank)

    vq_check = None
    if args.tf32:
        sample = [views(args.data_dir / row["image_path"]) for row in records[:64]]
        exact = encoder.vq_codes_batch([vq for vq, _, _ in sample])
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        fast = encoder.vq_codes_batch([vq for vq, _, _ in sample])
        same = [(a[2] == b[2]).float().mean().item() for a, b in zip(exact, fast, strict=True)]
        vq_check = {"images": len(same), "token_agreement": sum(same) / len(same),
                    "images_identical": sum(v == 1.0 for v in same) / len(same)}
        print("TF32 VQ check:", vq_check, flush=True)

    torch.cuda.reset_peak_memory_stats()
    timer = Timer()
    started = time.perf_counter()
    loaded_queue, ocr_queue = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    pool = ThreadPoolExecutor(16)
    threads = [threading.Thread(target=producer, args=(batches, args.data_dir, blank, loaded_queue, pool), daemon=True)]
    if args.ocr_thread:
        threads.append(threading.Thread(target=ocr_stage, args=(reader, loaded_queue, ocr_queue, timer), daemon=True))
    for thread in threads:
        thread.start()
    source = ocr_queue if args.ocr_thread else loaded_queue

    ids, parts, flips, replaced, flipped_rows = [], [[] for _ in D.DIMENSIONS], [], [], 0
    while True:
        wait = time.perf_counter()
        item = source.get()
        timer.add("waiting", wait, sync=False)
        if item is None:
            break
        if args.ocr_thread:
            batch, loaded, bad, texts = item
        else:
            batch, loaded, bad = item
            start = time.perf_counter()
            texts = reader.read([original for _, original, _ in loaded])
            timer.add("ocr", start, sync=False)
        replaced += bad
        probs = forward(model, builder, encoder, dino, batch, loaded, texts, timer)
        if args.tta_perspective != "off":
            chosen = (list(range(len(batch))) if args.tta_perspective == "all"
                      else [i for i, flag in enumerate(needs_flip(probs[PERSPECTIVE], args.margin)) if flag])
            flip = torch.full_like(probs[PERSPECTIVE], float("nan"))
            if chosen:
                start = time.perf_counter()
                mirrored = list(pool.map(lambda i: views(args.data_dir / batch[i]["image_path"], True)
                                         if batch[i]["image_id"] not in bad else loaded[i], chosen))
                timer.add("tta_images", start, sync=False)
                mirror_probs = forward(model, builder, encoder, dino, [batch[i] for i in chosen], mirrored,
                                       [texts[i] for i in chosen], timer)
                flip[chosen] = mirrored_perspective(mirror_probs[PERSPECTIVE])
                flipped_rows += len(chosen)
                if args.tta_perspective == "selective":
                    probs[PERSPECTIVE][chosen] = (probs[PERSPECTIVE][chosen] + flip[chosen]) / 2
            flips.append(flip)
        ids.extend(row["image_id"] for row in batch)
        for part, values in zip(parts, probs, strict=True):
            part.append(values)
    seconds = time.perf_counter() - started
    for thread in threads:
        thread.join()

    order = {image_id: i for i, image_id in enumerate(ids)}
    original_ids = [row["image_id"] for row in rows.to_dict("records")]
    index = torch.tensor([order[i] for i in original_ids])
    probs = [torch.cat(part)[index] for part in parts]
    predictions = D.prediction_frame(original_ids, probs)
    predictions.to_csv(args.out.with_suffix(".csv"), index=False)
    torch.save({"image_ids": original_ids, "dimensions": D.DIMENSIONS, "probabilities": probs},
               args.out.with_suffix(".probs.pt"))
    if args.tta_perspective == "all":
        torch.save({"image_ids": original_ids, "perspective_flip": torch.cat(flips)[index]},
                   args.out.with_suffix(".flip.probs.pt"))

    report = {"images": len(original_ids), "seconds": seconds, "images_per_second": len(original_ids) / seconds,
              "load_seconds": load_seconds, "stage_seconds": timer.seconds,
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3, "gpu": torch.cuda.get_device_name(),
              "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "replaced_images": len(replaced), "tta_rows": flipped_rows, "tf32_vq_check": vq_check}
    if args.reference:
        reference = torch.load(args.reference, weights_only=True, map_location="cpu")
        position = {image_id: i for i, image_id in enumerate(reference["image_ids"])}
        pick = torch.tensor([position[i] for i in original_ids])
        parity = {}
        for name, mine, theirs in zip(D.DIMENSIONS, probs, reference["probabilities"], strict=True):
            theirs = theirs[pick].float()
            parity[name] = {"argmax_agreement": float((mine.argmax(1) == theirs.argmax(1)).float().mean()),
                            "max_abs_diff": float((mine - theirs).abs().max())}
        report["parity"] = parity
    if args.labels:
        metric = D.load_metric(args.data_dir)
        results = metric.evaluate(D.solution_frame(rows), predictions, "image_id")
        report["macro_f1"] = results["overall"]["f1_macro"]
        report["weighted_f1"] = results["overall"]["f1_weighted"]
        report["dimensions"] = {d: results[d]["f1_macro"] for d in D.DIMENSIONS}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
