"""Train only the SigLIP-to-Apertus projector and evaluate a product holdout."""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig, SiglipVisionModel

sys.path.insert(0, str(next(q for p in Path(__file__).resolve().parents for q in (p, p / "training") if (q / "common" / "prompt.py").is_file())))
from common.splits import split_rows, training_rows
from adapter import ContinuousVisionAdapter, prepend_visual_tokens
from supervision import DIMENSIONS, supervised_answer

APERTUS = "swiss-ai/Apertus-v1.5-8B"
SIGLIP = "google/siglip2-base-patch16-224"


def select_split_rows(rows, train_size, holdout_size, seed):
    train = training_rows(rows).to_dict("records")
    holdout = split_rows(rows, "dev").head(holdout_size).to_dict("records")
    train = random.Random(seed).sample(train, min(train_size, len(train)))
    if not holdout or {row["product_id"] for row in train} & {row["product_id"] for row in holdout}:
        raise RuntimeError("Empty holdout or overlap between training and evaluation products")
    return train, holdout


def select_manifest_rows(rows, train_manifest, validation_manifest, train_size, holdout_size, seed):
    """Use the exact private manifests and reject altered rows or product leakage."""
    train_frame = pd.read_csv(train_manifest, dtype=str, keep_default_na=False)
    valid_frame = pd.read_csv(validation_manifest, dtype=str, keep_default_na=False)
    source = rows.set_index("image_id", verify_integrity=True)
    for name, frame in (("training", train_frame), ("validation", valid_frame)):
        if frame.empty or frame.image_id.duplicated().any() or set(frame.columns) != set(rows.columns):
            raise ValueError(f"Invalid {name} manifest")
        if not set(frame.image_id).issubset(source.index):
            raise ValueError(f"Unknown image in {name} manifest")
        expected = source.loc[frame.image_id, [column for column in rows.columns if column != "image_id"]]
        actual = frame.set_index("image_id").loc[:, expected.columns]
        if not expected.equals(actual):
            raise ValueError(f"{name} manifest differs from original data")
    eligible = set(training_rows(rows).product_id)
    train_products, valid_products = set(train_frame.product_id), set(valid_frame.product_id)
    if train_products & valid_products or not (train_products | valid_products).issubset(eligible):
        raise ValueError("Prototype manifests overlap by product or contain reserved coverage products")
    train = train_frame.to_dict("records")
    train = random.Random(seed).sample(train, min(train_size, len(train)))
    return train, valid_frame.head(holdout_size).to_dict("records")


def white_rgb(path):
    with Image.open(path) as source:
        image = source.convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(white, image).convert("RGB")


def resize_for_vq(image, target=256):
    scale = target / (image.width * image.height) ** 0.5
    scale *= min(1.0, 8 * target / (max(image.width, image.height) * scale))
    width = max(16, round(image.width * scale / 16) * 16)
    height = max(16, round(image.height * scale / 16) * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def prompt_inputs(processor, image, row, system_prompt, vq_size=256):
    fields = (("Product", "product_title"), ("Brand", "brand"), ("Product type", "product_type"),
              ("Category", "category"), ("Subcategory", "subcategory"))
    context = "\n".join(f"{name}: {row[key]}" for name, key in fields if row.get(key))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image", "image": resize_for_vq(image, vq_size)},
                                       {"type": "text", "text": context + "\nClassify the image."}]},
    ]
    return processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                         add_generation_prompt=True, enable_thinking=False)


def make_training_batch(rows, data_dir, processor, prompt, vq_size, end_id):
    """Right-pad token sequences and native VQ images without mixing examples."""
    images, ids_list, labels_list, native, sizes, prompt_lengths = [], [], [], [], [], []
    for row in rows:
        image = white_rgb(data_dir / row["image_path"])
        item = prompt_inputs(processor, image, row, prompt, vq_size)
        _, answer_ids, answer_labels = supervised_answer(row, processor.tokenizer)
        answer_ids.append(end_id)
        answer_labels.append(end_id)
        prefix_ids = item["input_ids"][0]
        prompt_lengths.append(len(prefix_ids))
        ids_list.append(torch.cat((prefix_ids, torch.tensor(answer_ids, dtype=torch.long))))
        labels_list.append(torch.cat((torch.full_like(prefix_ids, -100), torch.tensor(answer_labels, dtype=torch.long))))
        native.append(item["pixel_values"][0])
        sizes.append(item["image_sizes"][0])
        images.append(image)
    ids = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(value) for value in ids_list])
    mask = torch.arange(ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    height = max(value.shape[-2] for value in native)
    width = max(value.shape[-1] for value in native)
    pixels = torch.stack([torch.nn.functional.pad(value, (0, width - value.shape[-1], 0, height - value.shape[-2]))
                          for value in native])
    return images, ids, mask.long(), labels, pixels, torch.stack(sizes), prompt_lengths


def sparse_causal_loss(hidden, labels, lm_head):
    """Project supervised positions only; avoids enormous full-prompt logits."""
    targets = labels[:, 1:]
    keep = targets.ne(-100)
    if not keep.any():
        raise ValueError("No supervised answer tokens")
    selected = hidden[:, :-1, :][keep.to(hidden.device)]
    logits = lm_head(selected).float()
    return torch.nn.functional.cross_entropy(logits, targets[keep].to(logits.device))


def parse_prediction(text, valid_classes):
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return {dimension: "" for dimension in DIMENSIONS}
    if not isinstance(obj, dict):
        return {dimension: "" for dimension in DIMENSIONS}
    return {dimension: obj.get(dimension, "") if isinstance(obj.get(dimension), str) and obj[dimension] in valid_classes[dimension] else ""
            for dimension in DIMENSIONS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--holdout-size", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--initial-projector", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--vq-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--backend", choices=("kaggle-t4", "hf-h200"), default="kaggle-t4")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    args = parser.parse_args()
    if min(args.train_size, args.holdout_size, args.epochs, args.batch_size) < 1:
        parser.error("training size, holdout size, epochs and batch size must be positive")
    if (args.train_manifest is None) != (args.validation_manifest is None):
        parser.error("provide both prototype manifests or neither")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if args.backend == "kaggle-t4" and (len(gpu_names) != 2 or any("T4" not in name.upper() for name in gpu_names)):
        raise RuntimeError(f"Expected exactly two Tesla T4 GPUs; got {gpu_names}")
    if args.backend == "hf-h200" and (len(gpu_names) != 1 or "H200" not in gpu_names[0].upper()):
        raise RuntimeError(f"Expected exactly one H200 GPU; got {gpu_names}")
    rows = pd.read_csv(args.data_dir / "train.csv", dtype=str, keep_default_na=False)
    if args.train_manifest is not None:
        train, holdout = select_manifest_rows(rows, args.train_manifest, args.validation_manifest,
                                              args.train_size, args.holdout_size, args.seed)
    else:
        train, holdout = select_split_rows(rows, args.train_size, args.holdout_size, args.seed)
    prompt = args.system_prompt.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location("competition_metric", args.data_dir / "metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    processor = AutoProcessor.from_pretrained(APERTUS)
    end_id = processor.tokenizer.convert_tokens_to_ids("<|assistant_end|>")
    if end_id is None or end_id == processor.tokenizer.unk_token_id:
        raise RuntimeError("Apertus assistant end token is unavailable")
    vision_processor = AutoImageProcessor.from_pretrained(SIGLIP)
    if args.backend == "kaggle-t4":
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForMultimodalLM.from_pretrained(APERTUS, device_map="balanced", quantization_config=quantization,
                                                         dtype=torch.float16)
        vision_dtype = torch.float16
    else:
        model = AutoModelForMultimodalLM.from_pretrained(APERTUS, device_map={"": 0}, dtype=torch.bfloat16)
        vision_dtype = torch.bfloat16
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in ("cpu", "disk") for device in device_map.values()):
        raise RuntimeError("Apertus was offloaded to CPU or disk; cannot run the requested GPU experiment")
    model.requires_grad_(False)
    if args.backend == "kaggle-t4":
        model.gradient_checkpointing_enable()
        model.model.language_model.train()  # checkpointing trades compute for memory on T4
    else:
        model.eval()
    embed_device = model.get_input_embeddings().weight.device
    if embed_device.type != "cuda":
        raise RuntimeError(f"Apertus embeddings are on {embed_device}, not CUDA")
    vision_index = max(range(len(gpu_names)), key=lambda device: torch.cuda.mem_get_info(device)[0])
    vision_device = torch.device(f"cuda:{vision_index}")
    encoder = SiglipVisionModel.from_pretrained(SIGLIP, dtype=vision_dtype).to(vision_device)
    adapter = ContinuousVisionAdapter(encoder, encoder.config.hidden_size, model.config.text_config.hidden_size).to(vision_device)
    initial_projector_sha256 = None
    if args.initial_projector is not None:
        with args.initial_projector.open("rb") as checkpoint_file:
            initial_projector_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        adapter.projector.load_state_dict(torch.load(args.initial_projector, map_location="cpu", weights_only=True), strict=True)
        print(f"Loaded initial projector, sha256={initial_projector_sha256}", flush=True)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.projector.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(gpu_names)):
        torch.cuda.reset_peak_memory_stats(index)
    begin = time.monotonic()
    system_n_tokens = None
    for epoch in range(args.epochs):
        rng.shuffle(train)
        total_batches = (len(train) + args.batch_size - 1) // args.batch_size
        for step, start in enumerate(range(0, len(train), args.batch_size), 1):
            batch = train[start:start + args.batch_size]
            images, input_ids, mask, labels, native_pixels, sizes, prompt_lengths = make_training_batch(
                batch, args.data_dir, processor, prompt, args.vq_size, end_id
            )
            input_ids, mask, labels = input_ids.to(embed_device), mask.to(embed_device), labels.to(embed_device)
            pixels = vision_processor(images=images, return_tensors="pt")["pixel_values"].to(vision_device, dtype=vision_dtype)
            visual = adapter(pixels)
            prepared = prepend_visual_tokens(model, input_ids, mask, visual, labels)
            hidden = model.model(inputs_embeds=prepared["inputs_embeds"], attention_mask=prepared["attention_mask"],
                                 pixel_values=native_pixels.to(embed_device), image_sizes=sizes.to(embed_device),
                                 use_cache=False).last_hidden_state
            loss = sparse_causal_loss(hidden, prepared["labels"], model.lm_head)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                temporary = args.output_dir / "projector.tmp"
                torch.save(adapter.projector.state_dict(), temporary)
                temporary.replace(args.output_dir / "projector.pt")
            if system_n_tokens is None:
                system_n_tokens = prompt_lengths[0]
            print(f"epoch={epoch + 1} step={step}/{total_batches} images={min(start + len(batch),len(train))}/{len(train)} loss={loss.item():.4f} elapsed={time.monotonic()-begin:.1f}s", flush=True)
    torch.save(adapter.projector.state_dict(), args.output_dir / "projector.pt")
    training_seconds = time.monotonic() - begin
    model.eval()
    adapter.eval()
    predictions = []
    invalid = 0
    with torch.inference_mode():
        for row in holdout:
            image = white_rgb(args.data_dir / row["image_path"])
            inputs = prompt_inputs(processor, image, row, prompt, args.vq_size)
            ids = inputs["input_ids"].to(embed_device)
            pixels = vision_processor(images=image, return_tensors="pt")["pixel_values"].to(vision_device, dtype=vision_dtype)
            prepared = prepend_visual_tokens(model, ids, inputs["attention_mask"].to(embed_device), adapter(pixels))
            generated = model.generate(**prepared, pixel_values=inputs["pixel_values"].to(embed_device),
                                       image_sizes=inputs["image_sizes"].to(embed_device), max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True)
            answer = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
            parsed = parse_prediction(answer, metric.VALID_CLASSES)
            invalid += any(not value for value in parsed.values())
            predictions.append({"image_id": row["image_id"], **parsed})
    solution = pd.DataFrame([{key: row[key] for key in ("image_id", *DIMENSIONS)} for row in holdout])
    for dimension in DIMENSIONS:
        for index, row in enumerate(holdout):
            if row[f"{dimension}_source"] != "human" or row[f"{dimension}_relevant"].lower() != "true":
                solution.loc[index, dimension] = None
    results = metric.evaluate(solution, pd.DataFrame(predictions), "image_id")
    summary = {"train_images": len(train), "holdout_images": len(holdout),
               "evaluation_split": "prototype-balanced-450" if args.validation_manifest else "prompt-dev",
               "training_split": "prototype-train-1500" if args.train_manifest else "outside-coverage",
               "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
               "initial_projector_sha256": initial_projector_sha256,
               "learning_rate": args.learning_rate, "gpu_names": gpu_names, "invalid_outputs": invalid, "vq_size": args.vq_size, "backend": args.backend,
               "elapsed_seconds": round(time.monotonic() - begin, 2), "prompt_tokens_including_image": system_n_tokens,
               "training_seconds": round(training_seconds, 2),
               "peak_allocated_gib": [round(torch.cuda.max_memory_allocated(i) / 1024**3, 3) for i in range(len(gpu_names))],
               "overall_macro_f1": results["overall"]["f1_macro"],
               "dimensions": {dimension: results[dimension]["f1_macro"] for dimension in DIMENSIONS if dimension in results}}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
