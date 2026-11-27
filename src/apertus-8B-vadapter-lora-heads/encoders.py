"""Image -> model inputs, identical in encode.py (cache for training) and predict.py (end to end).

Two views of every image, both frozen:
  - Apertus VQ codes: the discrete image tokens Apertus was trained on (prepared image, about side x side);
  - SigLIP 2 patch features, average-pooled to POOLED_SIDE x POOLED_SIDE (49 x 768): the projector turns them into
    continuous soft tokens. Pooling comes before the projector, so caching the pooled features is exact.

Sequence of one example (chat template of Apertus, no system message, generation prompt at the end):
    <template> [49 soft-token slots] <|img_start|> "H*W" <|img_token_start|> VQ rows ... <|img_end|>
    <context> ... </context> TASK <generation prompt>
Soft-token slots hold PREFIX_SLOT (-1) in input_ids; the model writes the projected SigLIP tokens there.
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
SIGLIP_ID = "google/siglip2-base-patch16-224"
POOLED_SIDE = 7
PREFIX_SLOT = -1
TASK = ("Classify this product image: what it shows, what is behind the product, from where the camera sees it, "
        "whether a real person or animal is visible, which text was added, and the language of the text.")


def model_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION)


# ---------------------------------------------------------------- images


class ImageEncoder:
    """Frozen VQ tokenizer of Apertus (fp32, no TF32: codes are an argmax) and frozen SigLIP 2."""

    def __init__(self, directory, device="cuda"):
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, SiglipVisionModel
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5VisionTokenizerModel
        from safetensors import safe_open

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
        self.siglip = SiglipVisionModel.from_pretrained(SIGLIP_ID, dtype=torch.float32).to(self.device).eval()
        self.siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_ID)

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

    @torch.inference_mode()
    def siglip_features(self, pil_images):
        """[batch, 49, 768] float16: SigLIP patch grid average-pooled to POOLED_SIDE x POOLED_SIDE."""
        pixels = self.siglip_processor(images=pil_images, return_tensors="pt")["pixel_values"].to(self.device)
        patches = self.siglip(pixel_values=pixels).last_hidden_state
        batch, count, channels = patches.shape
        side = int(count**0.5)
        assert side * side == count, f"expected a square patch grid, got {count} patches"
        grid = patches.transpose(1, 2).reshape(batch, channels, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, POOLED_SIDE)
        return pooled.flatten(2).transpose(1, 2).to("cpu", torch.float16)

    def encode(self, paths, variants, side, names=None):
        """One dict per image: grid, codes and siglip features. Images are decoded and resized in threads.

        variants: "" for the image as it is (always at inference), or a training variant (see variant_images);
        names seed the crops (the image ids), so that a variant is the same on every run.
        """
        if not hasattr(self, "pool"):
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(16)  # PIL releases the GIL while decoding and resizing
        names = names or [str(p) for p in paths]
        pairs = list(self.pool.map(lambda job: variant_images(*job, side), zip(paths, variants, names)))
        vq_images, siglip_images = [vq for vq, _ in pairs], [full for _, full in pairs]
        features = self.siglip_features(siglip_images)
        results = []
        for image, feature in zip(vq_images, features):
            grid_h, grid_w, codes = self.vq_codes(image)
            results.append({"grid": (grid_h, grid_w), "codes": codes, "siglip": feature})
        return results


CROP_KEEP = (0.9, 1.0)  # share of width and of height kept by a training crop


def variant_images(path, variant, side, name=""):
    """(image for the VQ tokenizer, image for SigLIP) of one variant.

    "" and "m": exactly the inference preparation (common.images), plain or mirrored. "a<k>" and "a<k>m": a mild
    crop keeping 90-100% of width and height at a position fixed by (name, k), mirrored with "m". No colour change:
    it would turn white backgrounds into "color".
    """
    mirrored = variant.endswith("m")
    crop = variant[:-1] if mirrored else variant
    if not crop:
        return images.prepared_image(path, side, mirrored), full_resolution(path, mirrored)
    import io
    import random

    from PIL import Image

    rgb = full_resolution(path, mirrored)
    rng = random.Random(f"{name}|{crop}")
    width, height = (round(size * rng.uniform(*CROP_KEEP)) for size in rgb.size)
    left, top = rng.randint(0, rgb.width - width), rng.randint(0, rgb.height - height)
    rgb = rgb.crop((left, top, left + width, top + height))
    buffer = io.BytesIO()  # the same resize and JPEG round trip as common.images.prepared_jpeg
    rgb.resize(images.target_size(*rgb.size, side), Image.LANCZOS).save(buffer, format="JPEG",
                                                                         quality=images.JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), rgb


def full_resolution(path, mirrored=False):
    """The original image on white for SigLIP (its processor resizes to 224 x 224)."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = images.opaque_rgb_on_white(image)
    return ImageOps.mirror(rgb) if mirrored else rgb


# ---------------------------------------------------------------- token sequences


class PromptBuilder:
    """Token ids of one example with the Apertus tokenizer and chat template."""

    def __init__(self, tokenizer, image_token_offset, prefix_tokens=POOLED_SIDE**2):
        self.tokenizer = tokenizer
        self.offset = image_token_offset
        self.prefix_tokens = prefix_tokens
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
        text = (prompt.context_block(row) if use_context else prompt.context_block({})) + "\n\n" + TASK
        before, after = self.parts(text)
        return before + [PREFIX_SLOT] * self.prefix_tokens + self.image_ids(*grid, codes) + after


def collate(sequences, pad_id):
    """Left padding: the last position of every row is its last prompt token."""
    length = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, length - len(sequence):] = torch.tensor(sequence, dtype=torch.long)
        attention[row, length - len(sequence):] = 1
    return input_ids, attention
