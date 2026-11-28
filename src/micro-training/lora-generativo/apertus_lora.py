"""Shared pieces of the Apertus v1.5 8B LoRA: training rows, prompt, token sequences and the VQ image codes.

Apertus v1.5 turns an image into discrete codes with its VQ tokenizer. The language model then embeds those codes
like text tokens (code + config.image_token_offset). So the codes are computed once per image (encode.py) and the
training sequences are plain token ids: no image passes through the model during training.

Layout of one image inside the sequence, as Apertus1p5Processor writes it:
    <|img_start|> "H*W" <|img_token_start|> row 1 <|img_end_of_row|> row 2 ... row H <|img_end|>
where each row holds W codes.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, *(p / "training" for p in HERE.parents), *HERE.parents):
    if (candidate / "common" / "prompt.py").exists():
        sys.path.insert(0, str(candidate))
        break

from common import images, prompt, splits  # noqa: E402

DIMENSIONS = prompt.DIMENSIONS
TAXONOMY = prompt.TAXONOMY

# Short on purpose: after finetuning the model knows the classes from the labels, and every prompt token costs
# compute on the T4. This deviates from src/training/common/prompt.py (SHARED_JSON_PROMPT) (1,641 tokens); the results note it.
SYSTEM_PROMPT = (
    "You classify one product image of an online shop along six dimensions: "
    + ", ".join(DIMENSIONS)
    + ". Answer with a JSON object with one class name per dimension, in this order."
)

MIRRORED_PERSPECTIVES = {
    "front_angle_left": "front_angle_right",
    "front_angle_right": "front_angle_left",
    "back_angle_left": "back_angle_right",
    "back_angle_right": "back_angle_left",
    "left_side": "right_side",
    "right_side": "left_side",
}

# The VQ and audio tokenizers do not support sdpa and are dropped after loading (the codes are precomputed): only
# the language model asks for sdpa. A plain "sdpa" for the whole model raises a ValueError at load time.
ATTENTION = {"text_config": "sdpa", "vision_tokenizer_config": "eager", "audio_tokenizer_config": "eager"}

RARE_CLASS_TARGET = 200
MAX_REPEATS = 3


# ---------------------------------------------------------------- rows, targets, repeats, mirrors


def read_train(data_dir):
    import pandas as pd

    rows = pd.read_csv(Path(data_dir) / "train.csv", dtype=str, keep_default_na=False)
    rows["image_file"] = [str(Path(data_dir) / path) for path in rows["image_path"]]
    return rows


PROTOTYPE_PARTS = {"prototype_train": "train", "prototype_validation": "validation"}
PROTOTYPE_FILES = {"train": "train_1500.csv", "validation": "validation_450.csv"}


def pack_prototype_ids(folder):
    """The image ids of data/prototype_splits, gzipped and base64 encoded: the value of the PROTOTYPE_IDS secret.

    Only ids travel, never labels or images; the job reads the rows from its own copy of train.csv.
    """
    import base64
    import gzip

    import pandas as pd

    ids = {name: pd.read_csv(Path(folder) / file, dtype=str, usecols=["image_id"]).image_id.tolist()
           for name, file in PROTOTYPE_FILES.items()}
    return base64.b64encode(gzip.compress(json.dumps(ids).encode())).decode()


def prototype_ids(name):
    """Ids of the prototype split `name`, from $PROTOTYPE_IDS (packed) or the csv files in $PROTOTYPE_DIR."""
    import base64
    import gzip
    import os

    if os.environ.get("PROTOTYPE_IDS"):
        return json.loads(gzip.decompress(base64.b64decode(os.environ["PROTOTYPE_IDS"])))[name]
    folder = os.environ.get("PROTOTYPE_DIR")
    assert folder, "set PROTOTYPE_IDS (packed ids) or PROTOTYPE_DIR (folder of data/prototype_splits)"
    return json.loads(gzip.decompress(base64.b64decode(pack_prototype_ids(folder))))[name]


def select_rows(rows, part):
    """"training" (phase 1, src/training/common/splits.py), "all" (phase 2), "coverage" (evaluation), or the prototype splits
    "prototype_train" / "prototype_validation" (data/prototype_splits, both inside training_rows)."""
    if part == "training":
        return splits.training_rows(rows)
    if part == "coverage":
        return splits.split_rows(rows, "coverage")
    if part == "all":
        return rows.reset_index(drop=True)
    if part in PROTOTYPE_PARTS:
        ids = prototype_ids(PROTOTYPE_PARTS[part])
        picked = rows.set_index("image_id").loc[ids].reset_index()  # keeps the order of the split
        reserved = set(splits.split_rows(rows, "coverage").product_id)
        assert not picked.product_id.isin(reserved).any(), f"{part} touches the coverage set"
        return picked
    raise ValueError(part)


def target_of(row):
    return {d: row[d] for d in DIMENSIONS}


def human_dimensions(row):
    """Dimensions whose label a person set or confirmed: only these reach the loss."""
    return [d for d in DIMENSIONS if row[f"{d}_source"] == "human"]


def is_mirrorable(row):
    """A human left/right perspective on an image without text, which a mirror would garble."""
    return (
        row["perspective_source"] == "human"
        and row["perspective"] in MIRRORED_PERSPECTIVES
        and row["text_classification"] == "none"
        and row["text_language"] == "none"
    )


def mirrored_target(target):
    return {**target, "perspective": MIRRORED_PERSPECTIVES[target["perspective"]]}


def repeat_counts(records):
    """How often each row is shown per epoch: enough to lift its rarest human class towards RARE_CLASS_TARGET."""
    counts = Counter((d, r[d]) for r in records for d in human_dimensions(r))
    result = []
    for r in records:
        rarest = min((counts[(d, r[d])] for d in human_dimensions(r)), default=RARE_CLASS_TARGET)
        result.append(min(MAX_REPEATS, max(1, math.ceil(RARE_CLASS_TARGET / rarest))))
    return result


def image_jobs(records):
    """(image_id, image_file, mirrored) of every image the training needs: all rows plus the mirrorable ones."""
    jobs = [(r["image_id"], r["image_file"], False) for r in records]
    jobs += [(r["image_id"], r["image_file"], True) for r in records if is_mirrorable(r)]
    return jobs


def code_key(image_id, mirrored):
    return f"{image_id}|m" if mirrored else image_id


# ---------------------------------------------------------------- VQ codes


def load_vision_tokenizer(model_dir, device):
    """The VQ tokenizer alone (0.87 GB, fp32) and the image processor, without the 17 GB language model."""
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VisionTokenizerModel

    config = AutoConfig.from_pretrained(model_dir)
    model = Apertus1p5VisionTokenizerModel(config.vision_tokenizer_config)
    from safetensors import safe_open

    index_path = Path(model_dir) / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files = sorted({f for k, f in weight_map.items() if "vision_tokenizer." in k})
    else:
        files = [p.name for p in Path(model_dir).glob("*vision_tokenizer*.safetensors")]
    state = {}
    for name in files:
        with safe_open(str(Path(model_dir) / name), framework="pt") as handle:
            for key in handle.keys():
                if "vision_tokenizer." in key:
                    state[key.split("vision_tokenizer.", 1)[1]] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"vision tokenizer weights missing: {missing[:5]}"
    # Codes are an argmax over 131,072 codebook entries: keep true fp32 (no TF32) so that every GPU gives the
    # same codes in training and in inference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.to(device=device, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor, config


def encode_image(vision_tokenizer, processor, image):
    """(grid_height, grid_width, codes) of one prepared PIL image; codes are raw codebook indices, row major."""
    import torch

    batch = processor.image_processor(images=[image], return_tensors="pt")
    height, width = (int(v) for v in batch["image_sizes"][0])
    grid_h, grid_w = (int(v) for v in batch["image_grids"][0])
    pixels = batch["pixel_values"][:1, :, :height, :width].to(next(vision_tokenizer.parameters()).device)
    with torch.inference_mode():
        codes = vision_tokenizer.encode(pixels)[0]
    assert tuple(codes.shape) == (grid_h, grid_w), (codes.shape, grid_h, grid_w)
    return grid_h, grid_w, codes.flatten().to("cpu", torch.int32).numpy()


# ---------------------------------------------------------------- token sequences


class Sequences:
    """Builds the token ids of prompts and answers with the Apertus tokenizer and chat template."""

    def __init__(self, tokenizer, image_token_offset):
        self.tokenizer = tokenizer
        self.offset = image_token_offset
        ids = tokenizer.convert_tokens_to_ids
        self.boi, self.wrapper, self.eol, self.eoi = (
            ids(t) for t in ("<|img_start|>", "<|img_token_start|>", "<|img_end_of_row|>", "<|img_end|>")
        )
        self.end = ids("<|assistant_end|>")
        self.image_placeholder = "<|image|>"
        self._prompt_parts = {}

    def _text_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def prompt_parts(self, user_text):
        """Token ids before and after the image placeholder of the generation prompt."""
        if user_text not in self._prompt_parts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            assert rendered.count(self.image_placeholder) == 1, rendered
            before, after = rendered.split(self.image_placeholder)
            self._prompt_parts[user_text] = (self._text_ids(before), self._text_ids(after))
        return self._prompt_parts[user_text]

    def image_ids(self, grid_h, grid_w, codes):
        ids = [self.boi, *self._text_ids(f"{grid_h}*{grid_w}"), self.wrapper]
        for row in range(grid_h):
            if row:
                ids.append(self.eol)
            ids.extend(int(c) + self.offset for c in codes[row * grid_w : (row + 1) * grid_w])
        ids.append(self.eoi)
        return ids

    def prompt_ids(self, row, grid_h, grid_w, codes, use_context=True):
        before, after = self.prompt_parts(prompt.context_block(row if use_context else {}))
        return before + self.image_ids(grid_h, grid_w, codes) + after

    def answer(self, target, trained_dimensions):
        """(ids, labels) of the JSON answer plus <|assistant_end|>; values of untrained dimensions are -100."""
        text, spans = serialize_answer(target)
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoding["input_ids"])
        labels = list(ids)
        for dimension, (start, end) in spans.items():
            if dimension in trained_dimensions:
                continue
            for index, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
                if token_start < end and token_end > start:
                    labels[index] = -100
        return ids + [self.end], labels + [self.end]


def serialize_answer(target):
    """The answer JSON, with the character span of each class value inside its quotes."""
    text, spans = "{", {}
    for index, dimension in enumerate(DIMENSIONS):
        text += ("" if index == 0 else ", ") + json.dumps(dimension) + ": "
        encoded = json.dumps(target[dimension])
        spans[dimension] = (len(text) + 1, len(text) + len(encoded) - 1)
        text += encoded
    return text + "}", spans


def parse_answer(text):
    try:
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict) or any(parsed.get(d) not in TAXONOMY[d] for d in DIMENSIONS):
        return None
    return {d: parsed[d] for d in DIMENSIONS}


def build_examples(records, codes, sequences, use_context=True):
    """Training examples {input_ids, labels} for every row, with repeats of rare classes and mirrored copies."""
    examples = []
    for record, repeats in zip(records, repeat_counts(records)):
        trained = human_dimensions(record)
        variants = [(False, target_of(record), repeats)]
        if is_mirrorable(record):
            variants.append((True, mirrored_target(target_of(record)), 1))
        for mirrored, target, count in variants:
            grid_h, grid_w, image_codes = codes[code_key(record["image_id"], mirrored)]
            prompt_ids = sequences.prompt_ids(record, grid_h, grid_w, image_codes, use_context)
            answer_ids, answer_labels = sequences.answer(target, trained)
            example = {
                "input_ids": prompt_ids + answer_ids,
                "labels": [-100] * len(prompt_ids) + answer_labels,
            }
            examples.extend([example] * count)
    return examples


def collate(examples, pad_id):
    """Left padding, so that every answer ends at the last position and the loss needs only the last logits."""
    import torch

    length = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    attention = torch.zeros((len(examples), length), dtype=torch.long)
    for row, example in enumerate(examples):
        n = len(example["input_ids"])
        input_ids[row, length - n :] = torch.tensor(example["input_ids"])
        labels[row, length - n :] = torch.tensor(example["labels"])
        attention[row, length - n :] = 1
    position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
    return {"input_ids": input_ids, "attention_mask": attention, "position_ids": position_ids, "labels": labels}


def find_data_dir(root="/kaggle/input"):
    """The folder of the competition files on Kaggle (its mount path has changed over time)."""
    hits = sorted(Path(root).glob("**/train.csv"), key=lambda p: len(p.parts))
    assert hits, f"train.csv not found under {root}: is the competition attached to the notebook?"
    return hits[0].parent


def load_metric(data_dir):
    import importlib.util

    spec = importlib.util.spec_from_file_location("metric", Path(data_dir) / "metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


images = images  # re-exported for encode.py
