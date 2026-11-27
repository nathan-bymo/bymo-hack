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

    VQ_BATCH = 32  # images of the same size encoded together

    def vq_pixels(self, image):
        """Pixel tensor [3, H, W] of one prepared PIL image, cropped to its true size, and its code grid."""
        batch = self.apertus_processor(images=[image], return_tensors="pt")
        height, width = (int(v) for v in batch["image_sizes"][0])
        grid = tuple(int(v) for v in batch["image_grids"][0])
        return batch["pixel_values"][0, :, :height, :width], grid

    @torch.inference_mode()
    def vq_codes_batch(self, pil_images):
        """[(grid_h, grid_w, int32 codes row major)] of prepared PIL images.

        Images of the same size go through the tokenizer together: no padding, and every operation of the
        tokenizer (convolutions, GroupNorm, attention) works per image, so the codes are those of one image at a
        time (test_pipeline.py checks it on the real tokenizer class), only faster.
        """
        prepared = [self.vq_pixels(image) for image in pil_images]
        results = [None] * len(prepared)
        groups = {}
        for index, (pixels, _) in enumerate(prepared):
            groups.setdefault(tuple(pixels.shape), []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), self.VQ_BATCH):
                chunk = indices[start:start + self.VQ_BATCH]
                pixels = torch.stack([prepared[i][0] for i in chunk]).to(self.device)
                codes = self.vq.encode(pixels)
                for i, grid_codes in zip(chunk, codes):
                    grid = prepared[i][1]
                    assert tuple(grid_codes.shape) == grid, (grid_codes.shape, grid)
                    results[i] = (*grid, grid_codes.flatten().to("cpu", torch.int32))
        return results

    def vq_codes(self, image):
        """(grid_h, grid_w, int32 codes row major) of one prepared PIL image."""
        return self.vq_codes_batch([image])[0]

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
        import time

        if not hasattr(self, "pool"):
            from concurrent.futures import ThreadPoolExecutor

            self.pool = ThreadPoolExecutor(16)  # PIL releases the GIL while decoding and resizing
            self.seconds = {"images": 0.0, "siglip": 0.0, "vq": 0.0}  # time per stage, for the speed report
        names = names or [str(p) for p in paths]
        clock = time.perf_counter()
        pairs = list(self.pool.map(lambda job: variant_images(job[0], job[1], side, job[2]),
                                   zip(paths, variants, names)))
        vq_images, siglip_images = [vq for vq, _ in pairs], [full for _, full in pairs]
        self.seconds["images"] += time.perf_counter() - clock
        clock = time.perf_counter()
        features = self.siglip_features(siglip_images)
        self.seconds["siglip"] += time.perf_counter() - clock
        clock = time.perf_counter()
        codes = self.vq_codes_batch(vq_images)
        self.seconds["vq"] += time.perf_counter() - clock
        return [{"grid": (grid_h, grid_w), "codes": c, "siglip": feature}
                for (grid_h, grid_w, c), feature in zip(codes, features)]


AUGMENTATION = "vq-scale-v2"  # stored in the cache metadata: variants of another version are recomputed
VQ_SCALES = {1: 0.875, 2: 1.125, 3: 0.9375}  # repeat k: VQ image side = side * scale


def variant_images(path, variant, side, name=""):
    """(image for the VQ tokenizer, image for SigLIP) of one variant.

    "" and "m": exactly the inference preparation (common.images), plain or mirrored. "a<k>" and "a<k>m": the VQ
    image at another scale (side 224, 288 or 240 instead of 256: other VQ tokens), mirrored with "m"; SigLIP gets the
    whole image as at inference. Nothing is cut away: products touch the image border in most images, and any crop
    would turn a full_product into a detail_closeup (labeling guideline). No colour change either: it would turn
    white backgrounds into "color". `name` is kept for the call signature; the variants need no randomness.
    """
    mirrored = variant.endswith("m")
    repeat = variant[:-1] if mirrored else variant
    scale = VQ_SCALES[int(repeat[1:])] if repeat else 1.0
    return (images.prepared_image(path, round(side * scale), mirrored), full_resolution(path, mirrored))


def vq_size(path, side):
    """Size of the prepared VQ image, from the file header only: sorting by it fills the VQ batches."""
    from PIL import Image

    with Image.open(path) as image:
        return images.target_size(*image.size, side)


def sort_by_vq_size(items, path_of, side):
    """items reordered so that images of the same VQ size are neighbours (headers read in threads)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(lambda item: vq_size(path_of(item), side), items))
    return [item for _, item in sorted(zip(sizes, items), key=lambda pair: pair[0])]


def full_resolution(path, mirrored=False):
    """The original image on white for SigLIP (its processor resizes to 224 x 224)."""
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = images.opaque_rgb_on_white(image)
    return ImageOps.mirror(rgb) if mirrored else rgb


# ---------------------------------------------------------------- text in the image (OCR)

OCR_DETECTOR = "fast_base"  # docTR, Apache-2.0
OCR_RECOGNIZER = "Felix92/doctr-torch-parseq-multilingual-v1"  # Apache-2.0, multilingual vocabulary (äöüß, accents)
OCR_RECOGNIZER_REVISION = "8bde31c864bcfa47489140b9dddb25ec70f4fd54"
OCR_MIN_CONFIDENCE = 0.5
OCR_MAX_CHARS = 300  # about 80-100 tokens


class TextReader:
    """Frozen docTR OCR on the original image: its text goes into the prompt as a hint; Apertus still decides."""

    def __init__(self, device="cuda"):
        from doctr.models import from_hub, ocr_predictor

        recognizer = from_hub(OCR_RECOGNIZER, revision=OCR_RECOGNIZER_REVISION)
        self.predictor = ocr_predictor(det_arch=OCR_DETECTOR, reco_arch=recognizer, pretrained=True,
                                       assume_straight_pages=True, det_bs=16, reco_bs=256).to(device).eval()

    @torch.inference_mode()
    def read(self, pil_images):
        import numpy as np

        return [page_text(page) for page in self.predictor([np.asarray(image) for image in pil_images]).pages]


def keep_word(value):
    """Drops the symbol noise of OCR on photos: at least 2 letters or digits, and mostly letters or digits."""
    alnum = sum(c.isalnum() for c in value)
    return alnum >= 2 and alnum >= 0.6 * len(value)


def page_text(page):
    """Confident words, line by line, the largest text first (it is the most prominent), cut at OCR_MAX_CHARS."""
    lines = []
    for block in page.blocks:
        for line in block.lines:
            words = [w.value for w in line.words if w.confidence >= OCR_MIN_CONFIDENCE and keep_word(w.value)]
            if words:
                (_, top), (_, bottom) = line.geometry
                lines.append((bottom - top, " ".join(words)))
    lines.sort(key=lambda pair: -pair[0])
    return " | ".join(text for _, text in lines)[:OCR_MAX_CHARS]


LANGUAGE_NAMES = {"de": "German", "en": "English", "fr": "French", "it": "Italian", "nl": "Dutch"}


def language_hint(text):
    """A guess of the language of the OCR text by langdetect (Apache-2.0, offline, a docTR dependency).

    Only a hint in the prompt: Apertus decides, and must still apply the labeling rules (brands, technical terms and
    numbers do not count, the language with the most area wins). Deterministic (fixed seed)."""
    words = [w for w in text.replace("|", " ").split() if sum(c.isalpha() for c in w) >= 3]
    if len(words) < 3:
        return "not enough words"
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        best = detect_langs(" ".join(words))[0]
    except LangDetectException:
        return "not enough words"
    name = LANGUAGE_NAMES.get(best.lang, f"other ({best.lang})")
    return f"{name} {best.prob:.2f}"


def ocr_block(text, hint=False):
    """The OCR text as a prompt block; "<|" and "|>" are broken so that no OCR text can form a special token.
    hint: add the language guess of language_hint (models trained with --language-hint)."""
    clean = (text or "no text found").replace("<|", "< |").replace("|>", "| >")
    block = "<ocr>\n" + clean
    if hint:
        block += "\nlanguage guess: " + (language_hint(text) if text else "no text")
    return block + "\n</ocr>"


# ---------------------------------------------------------------- token sequences


class PromptBuilder:
    """Token ids of one example with the Apertus tokenizer and chat template."""

    def __init__(self, tokenizer, image_token_offset, prefix_tokens=POOLED_SIDE**2, language_hint=False):
        self.tokenizer = tokenizer
        self.offset = image_token_offset
        self.prefix_tokens = prefix_tokens
        self.language_hint = language_hint
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
            if len(self._parts) > 50_000:  # OCR texts are almost all different: keep the cache bounded
                self._parts.clear()
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

    def ids(self, row, grid, codes, use_context=True, ocr=None):
        """ocr: the OCR text of the image (None: model without OCR, no block at all)."""
        text = prompt.context_block(row if use_context else {})
        if ocr is not None:
            text += "\n" + ocr_block(ocr, self.language_hint)
        text += "\n\n" + TASK
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
