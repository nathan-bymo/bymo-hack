"""Image -> model inputs, identical in encode.py (cache for training) and predict.py (end to end).

The image goes through the frozen VQ tokenizer of Apertus, which turns it into discrete image tokens. Sequence of
one example (chat template of Apertus, generation prompt at the end):
    <template> <|img_start|> "H*W" <|img_token_start|> VQ rows ... <|img_end|> <context>...</context> TASK <generation>
"""

import json
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from common import images, prompt  # noqa: E402

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
MODEL_REVISION = "1519de0ae7f3d114879291810a58259966f64c99"
TASK = ("Classify this product image: what it shows, what is behind the product, from where the camera sees it, "
        "whether a real person or animal is visible, which text was added, and the language of the text.")


def model_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


class ImageEncoder:
    """The frozen VQ tokenizer of Apertus, in fp32 and without TF32: the codes are an argmax."""

    def __init__(self, directory, device="cuda"):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VisionTokenizerModel

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        directory = Path(directory)
        config = AutoConfig.from_pretrained(directory)
        vq = Apertus1p5VisionTokenizerModel(config.vision_tokenizer_config)
        index = directory / "model.safetensors.index.json"
        files = sorted({f for k, f in json.loads(index.read_text())["weight_map"].items() if "vision_tokenizer." in k})
        state = {}
        for name in files:
            with safe_open(str(directory / name), framework="pt") as handle:
                for key in handle.keys():
                    if "vision_tokenizer." in key:
                        state[key.split("vision_tokenizer.", 1)[1]] = handle.get_tensor(key)
        missing, _ = vq.load_state_dict(state, strict=False)
        assert not missing, f"vision tokenizer weights missing: {missing[:5]}"
        self.device = torch.device(device)
        self.vq = vq.to(self.device, torch.float32).eval()
        self.apertus_processor = AutoProcessor.from_pretrained(directory).image_processor

    @torch.inference_mode()
    def vq_codes(self, image):
        """(grid_h, grid_w, int32 codes row major) of one prepared PIL image."""
        batch = self.apertus_processor(images=[image], return_tensors="pt")
        height, width = (int(v) for v in batch["image_sizes"][0])
        grid_h, grid_w = (int(v) for v in batch["image_grids"][0])
        pixels = batch["pixel_values"][:1, :, :height, :width].to(self.device)
        codes = self.vq.encode(pixels)[0]
        assert tuple(codes.shape) == (grid_h, grid_w), (codes.shape, grid_h, grid_w)
        return grid_h, grid_w, codes.flatten().to("cpu", torch.int32)

    def encode(self, paths, variants, side):
        """One dict per image: grid and codes. Images are decoded and resized in threads. Variant "m" is mirrored."""
        if not hasattr(self, "pool"):
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(16)  # PIL releases the GIL while decoding and resizing
        prepared = list(self.pool.map(lambda job: prepared_image(job[0], job[1], side), zip(paths, variants)))
        results = []
        for image in prepared:
            grid_h, grid_w, codes = self.vq_codes(image)
            results.append({"grid": (grid_h, grid_w), "codes": codes})
        return results


def prepared_image(path, variant, side):
    """The same preparation as everywhere else in the project (common.images)."""
    if variant not in ("", "m"):
        raise ValueError(f"unknown image variant {variant!r}")
    return images.prepared_image(path, side, variant == "m")


class PromptBuilder:
    """Token ids of one example with the Apertus tokenizer and chat template."""

    def __init__(self, tokenizer, image_token_offset):
        self.tokenizer = tokenizer
        self.offset = image_token_offset
        ids = tokenizer.convert_tokens_to_ids
        self.boi, self.wrapper, self.eol, self.eoi = (
            ids(t) for t in ("<|img_start|>", "<|img_token_start|>", "<|img_end_of_row|>", "<|img_end|>"))
        self._parts = {}

    def _text_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def parts(self, user_text):
        """Token ids before and after the image placeholder of the generation prompt (cached per text)."""
        if user_text not in self._parts:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
            assert rendered.count("<|image|>") == 1, rendered
            before, after = rendered.split("<|image|>")
            self._parts[user_text] = (self._text_ids(before), self._text_ids(after))
        return self._parts[user_text]

    def image_ids(self, grid_h, grid_w, codes):
        ids = [self.boi, *self._text_ids(f"{grid_h}*{grid_w}"), self.wrapper]
        codes = codes.tolist()
        for row in range(grid_h):
            if row:
                ids.append(self.eol)
            ids.extend(c + self.offset for c in codes[row * grid_w:(row + 1) * grid_w])
        ids.append(self.eoi)
        return ids

    def ids(self, row, grid, codes, use_context=True):
        text = prompt.context_block(row if use_context else {}) + "\n\n" + TASK
        before, after = self.parts(text)
        return before + self.image_ids(*grid, codes) + after


def collate(sequences, pad_id):
    """Left padding: the last position of every row is its last prompt token."""
    length = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, length - len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        attention[row, length - len(sequence):] = 1
    return input_ids, attention
