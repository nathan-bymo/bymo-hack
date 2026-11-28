"""Evaluate the frozen visual-adapter checkpoint with the team's v2 prompt."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModelForMultimodalLM, AutoProcessor, LogitsProcessor, SiglipVisionModel

sys.path.insert(0, str(next(q for p in Path(__file__).resolve().parents for q in (p, p / "training") if (q / "common" / "prompt.py").is_file())))
from adapter import ContinuousVisionAdapter, prepend_visual_tokens
from common import prompt as team_prompt
from common.splits import split_rows
from train_projector import DIMENSIONS, resize_for_vq, white_rgb


class RestrictTokens(LogitsProcessor):
    def __init__(self, allowed):
        self.allowed = allowed

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed] = scores[:, self.allowed]
        return mask


def local_messages(messages, image):
    """Translate the API's image_url blocks to local processor image blocks."""
    translated = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = [{"type": "image", "image": image} if block.get("type") == "image_url" else block
                       for block in content]
        translated.append({"role": message["role"], "content": content})
    return translated


def allowed_code_ids(tokenizer, dimension):
    result = {}
    for code in team_prompt.CODES[dimension]:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != code:
            raise RuntimeError(f"Code {code!r} is not a single reversible token")
        result[ids[0]] = code
    return result


def evaluate_metrics(metric, rows, records):
    solution = rows[["image_id", *DIMENSIONS]].copy()
    for dim in DIMENSIONS:
        ok = (rows[f"{dim}_source"] == "human") & (rows[f"{dim}_relevant"].str.lower() == "true")
        solution.loc[~ok, dim] = None
    predictions = pd.DataFrame(records)[["image_id", *DIMENSIONS]]
    return metric.evaluate(solution, predictions, "image_id")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--vq-size", type=int, choices=(256, 512), default=256)
    args = p.parse_args()
    if args.limit < 1:
        p.error("limit must be positive")
    if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Expected exactly one H200 GPU")
    rows = pd.read_csv(args.data_dir / "train.csv", dtype=str, keep_default_na=False)
    rows = split_rows(rows, "dev").head(args.limit).reset_index(drop=True)
    if rows.empty:
        raise RuntimeError("No evaluation images")
    spec = importlib.util.spec_from_file_location("competition_metric", args.data_dir / "metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    processor = AutoProcessor.from_pretrained("swiss-ai/Apertus-v1.5-8B")
    image_processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model = AutoModelForMultimodalLM.from_pretrained("swiss-ai/Apertus-v1.5-8B", device_map={"": 0}, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    encoder = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224", dtype=torch.bfloat16).cuda()
    adapter = ContinuousVisionAdapter(encoder, encoder.config.hidden_size, model.config.text_config.hidden_size).cuda().eval()
    with args.checkpoint.open("rb") as f:
        checkpoint_sha = hashlib.file_digest(f, "sha256").hexdigest()
    adapter.projector.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    code_ids = {dim: allowed_code_ids(processor.tokenizer, dim) for dim in DIMENSIONS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, started = [], time.monotonic()
    print(f"Loaded projector sha256={checkpoint_sha}; split=prompt-dev; limit={len(rows)}; vq_size={args.vq_size}", flush=True)
    with torch.inference_mode():
        for index, row in enumerate(rows.to_dict("records"), 1):
            image = white_rgb(args.data_dir / row["image_path"])
            native_image = resize_for_vq(image, args.vq_size)
            pixels = image_processor(images=image, return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.bfloat16)
            visual_tokens = adapter(pixels)
            initial = processor.apply_chat_template(local_messages(team_prompt.describe_messages("local"), native_image),
                tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
            native_features = torch.cat(model.model.get_image_features(initial["pixel_values"].cuda(), initial["image_sizes"].cuda()).pooler_output, dim=0)

            def generate(messages, maximum, allowed=None):
                inputs = processor.apply_chat_template(local_messages(messages, native_image), tokenize=True,
                    return_dict=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)
                ids = inputs["input_ids"].cuda()
                prepared = prepend_visual_tokens(model, ids, inputs["attention_mask"].cuda(), visual_tokens)
                visual_mask = ids.eq(model.config.image_token_id)
                if visual_mask.sum().item() != native_features.shape[0]:
                    raise RuntimeError("Native image placeholder count does not match VQ features")
                prefix_length = visual_tokens.shape[1]
                mask = torch.nn.functional.pad(visual_mask, (prefix_length, 0)).unsqueeze(-1)
                prepared["inputs_embeds"] = prepared["inputs_embeds"].masked_scatter(mask, native_features.to(prepared["inputs_embeds"].dtype))
                kwargs = {"max_new_tokens": maximum, "do_sample": False, "use_cache": True}
                if allowed is not None:
                    kwargs["logits_processor"] = [RestrictTokens(list(allowed))]
                output = model.generate(**prepared, **kwargs)
                return output[0].tolist()

            description_ids = generate(team_prompt.describe_messages("local"), team_prompt.DESCRIBE_MAX_TOKENS)
            description = processor.tokenizer.decode(description_ids, skip_special_tokens=True)
            prediction = {"image_id": row["image_id"]}
            for dim in DIMENSIONS:
                messages = team_prompt.question_messages(row, "local", dim, answer="codes", description=description)
                token_ids = generate(messages, 1, code_ids[dim])
                prediction[dim] = team_prompt.CODES[dim].get(code_ids[dim].get(token_ids[-1], ""), "") if token_ids else ""
            records.append(prediction)
            if index % 10 == 0 or index == len(rows):
                elapsed = time.monotonic() - started
                print(f"evaluated={index}/{len(rows)} elapsed={elapsed:.1f}s images_per_second={index/elapsed:.3f}", flush=True)
                pd.DataFrame(records).to_csv(args.output_dir / "predictions.partial.csv", index=False)
    predictions = pd.DataFrame(records)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    result = evaluate_metrics(metric, rows, records)
    summary = {"checkpoint_sha256": checkpoint_sha, "split": "prompt-dev", "images": len(rows),
        "vq_size": args.vq_size, "vision_encoder": "siglip2-base-patch16-224", "prompt": "describe:codes:v2",
        "decoding": "greedy description; constrained greedy one-token class codes", "elapsed_seconds": time.monotonic() - started,
        "overall_macro_f1": result["overall"]["f1_macro"],
        "dimensions": {dim: result[dim]["f1_macro"] for dim in DIMENSIONS if dim in result}}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
