"""One forward pass of the frozen Apertus v1.5 backbone per image; saves the features the heads train on.

For every chosen layer the feature is the hidden state of the last prompt token (it has seen image and text)
concatenated with the mean over the image tokens (what the model holds about the image alone: the image comes
first, so these positions do not see the text). Text is never generated.

Needs a GPU, access to the gated model (accept its terms on Hugging Face, then `hf auth login`) and the
Transformers branch in requirements-gpu.txt. Output goes to data/, which git ignores. Run again to resume.

  python src/apertus-8B-heads/extract_features.py --out data/features/8b_512
  python src/apertus-8B-heads/extract_features.py --out data/features/8b_512_small \
      --csv data/nathan_split/train_1500.csv data/nathan_split/validation_450.csv
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import images, prompt  # noqa: E402

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
MODEL_REVISION = "1519de0ae7f3d114879291810a58259966f64c99"
TASK = ("Classify this product image: what it shows, what is behind the product, from where the camera sees it, "
        "whether a real person or animal is visible, which text was added, and the language of the text.")


def instruction(row, use_context):
    return (prompt.context_block(row) + "\n\n" if use_context else "") + TASK


@torch.inference_mode()
def batch_features(model, processor, conversations, layers):
    """Tensor [batch, layers, 2 * hidden] for a list of single-turn conversations."""
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to(model.device)
    attention = inputs["attention_mask"]
    if not attention[:, -1].all():
        raise RuntimeError("the last position is padding: left padding is required")
    image_positions = inputs["input_ids"] == model.config.image_token_id
    if not image_positions.any(dim=1).all():
        raise RuntimeError("a prompt has no image tokens")
    output = model.model(  # the backbone without the text generation head
        **inputs, position_ids=(attention.cumsum(1) - 1).clamp(min=0), use_cache=False, output_hidden_states=True,
    )
    weights = image_positions.float()
    features = []
    for layer in layers:
        hidden = output.hidden_states[layer].float()
        mask = weights.to(hidden.device)  # with several GPUs the layers live on different devices
        image_mean = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        features.append(torch.cat([hidden[:, -1], image_mean], dim=-1).cpu())
    return torch.stack(features, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for the feature shards, inside data/")
    parser.add_argument("--csv", type=Path, nargs="+", default=[ROOT / "data" / "train.csv"],
                        help="one or more csv files with the columns of train.csv or dev.csv")
    parser.add_argument("--images-root", type=Path, default=ROOT / "data", help="folder that image_path starts from")
    parser.add_argument("--layers", default="12,16,20,24,28,32", help="0 is the embedding, 32 the last layer")
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda", help='"cuda", or "auto" to split over several GPUs; never uses the CPU on its own')
    parser.add_argument("--no-context", action="store_true")
    arguments = parser.parse_args()
    if (arguments.device == "auto" or arguments.device.startswith("cuda")) and not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible")
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    layers = [int(value) for value in arguments.layers.split(",")]
    arguments.out.mkdir(parents=True, exist_ok=True)
    settings = {"model": MODEL_ID, "revision": MODEL_REVISION, "layers": layers, "side": arguments.side,
                "context": not arguments.no_context, "task": TASK}
    meta = arguments.out / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != settings:
        sys.exit(f"{arguments.out} holds features with other settings: choose another folder")
    meta.write_text(json.dumps(settings, indent=2))

    rows = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False) for path in arguments.csv])
    rows = rows.drop_duplicates("image_id")
    if arguments.limit:
        rows = rows.head(arguments.limit)
    done = set()
    for shard in arguments.out.glob("shard_*.pt"):
        done.update(torch.load(shard, weights_only=True)["image_ids"])
    todo = rows[~rows.image_id.isin(done)].to_dict("records")
    print(f"{len(rows)} images, {len(done)} already done, {len(todo)} to do", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype="auto", device_map=arguments.device
    ).eval()
    offloaded = {str(d) for d in getattr(model, "hf_device_map", {}).values()} & {"cpu", "disk"}
    if arguments.device == "auto" and offloaded:
        sys.exit(f"the model does not fit on the GPUs, part of it went to {sorted(offloaded)}")

    def conversation(row):
        image = images.prepared_jpeg(arguments.images_root / row["image_path"], arguments.side)
        return [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction(row, not arguments.no_context)},
        ]}]

    next_shard = len(list(arguments.out.glob("shard_*.pt")))
    started, processed = time.time(), 0
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(todo), arguments.shard_size):
            chunk = todo[start : start + arguments.shard_size]
            conversations = list(pool.map(conversation, chunk))
            parts = [
                batch_features(model, processor, conversations[i : i + arguments.batch_size], layers)
                for i in range(0, len(chunk), arguments.batch_size)
            ]
            temporary = arguments.out / "writing.tmp"
            torch.save({"image_ids": [row["image_id"] for row in chunk], "features": torch.cat(parts)}, temporary)
            temporary.rename(arguments.out / f"shard_{next_shard:05d}.pt")
            next_shard += 1
            processed += len(chunk)
            memory = f", peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB" if torch.cuda.is_available() else ""
            print(f"{processed}/{len(todo)} images, {processed / (time.time() - started):.2f} images/s{memory}",
                  flush=True)


if __name__ == "__main__":
    main()
