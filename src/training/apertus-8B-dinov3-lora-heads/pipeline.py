"""DINOv3 image features and the Apertus classifier without native VQ tokens."""

import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apertus-8B-vadapter-lora-heads"))
import data as D  # noqa: E402
import encoders as E  # noqa: E402
import model as M  # noqa: E402
from common import images, prompt  # noqa: E402

DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_SIDE, AUGMENTED_SIDE, GRID_SIDE, PATCH_SIZE = 288, 320, 9, 16


def fit_image(path, side):
    """Keep the whole product and its aspect ratio, including transparent margins."""
    with Image.open(path) as image:
        image = images.opaque_rgb_on_white(ImageOps.exif_transpose(image))
    return ImageOps.pad(image, (side, side), method=Image.Resampling.LANCZOS, color="white")


def pool_dino(hidden, side, register_count):
    width = side // PATCH_SIZE
    if hidden.shape[1] != 1 + register_count + width * width:
        raise ValueError(f"Unexpected DINO patch grid for side {side}: {tuple(hidden.shape)}")
    patches = hidden[:, 1 + register_count:]
    grid = patches.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[-1], width, width)
    return F.adaptive_avg_pool2d(grid.float(), GRID_SIDE).flatten(2).transpose(1, 2).contiguous()


class DinoEncoder:
    def __init__(self, revision, device="cuda", model_folder=None):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device)
        source = model_folder or DINO_ID
        self.processor = AutoImageProcessor.from_pretrained(source, revision=revision)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(source, revision=revision, dtype=dtype)
        self.model.requires_grad_(False).eval().to(self.device)
        self.revision = revision
        cfg = self.model.config
        if cfg.patch_size != PATCH_SIZE or cfg.hidden_size != 768:
            raise ValueError("This run expects DINOv3 ViT-B/16 with a hidden size of 768.")
        self.register_count = cfg.num_register_tokens

    @torch.inference_mode()
    def encode(self, pil_images, side):
        pixels = self.processor(images=pil_images, do_resize=False, do_center_crop=False,
                                return_tensors="pt")["pixel_values"]
        if pixels.shape[-2:] != (side, side):
            raise ValueError(f"DINO preprocessing changed the requested {side}px image size.")
        hidden = self.model(pixel_values=pixels.to(self.device, self.model.dtype)).last_hidden_state
        features = pool_dino(hidden, side, self.register_count).to("cpu", torch.float16)
        if not features.isfinite().all():
            raise ValueError("DINO produced non-finite features.")
        return features


class DinoPrompt(E.PromptBuilder):
    def ids(self, row, text):
        context = prompt.context_block(row) + "\n" + E.ocr_block(text, self.language_hint)
        before, after = self.parts(context + "\n\n" + E.TASK)
        return before + [E.PREFIX_SLOT] * self.prefix_tokens + after


def make_model(directory, folder, device="cuda", warm_start=False):
    """Reuse the older decoder, heads and pool. Only the input projector is newly initialized."""
    from transformers import AutoTokenizer

    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())
    if warm_start:
        if (settings.get("model") != E.MODEL_ID or settings.get("revision") != E.MODEL_REVISION
                or not all(settings.get(key) for key in ("adapter", "ocr", "spatial", "attention_pool"))
                or settings.get("vision_dim") != 768 or settings.get("layers") != 24):
            raise ValueError("Warm start requires the matching 24-layer SigLIP + OCR checkpoint.")
    elif (settings.get("vision_encoder") != DINO_ID or settings.get("vq") is not False
          or settings.get("soft_side") != GRID_SIDE):
        raise ValueError("Expected a saved DINOv3 checkpoint without VQ.")

    classifier = M.build(directory, settings, device)
    M.load_weights(classifier, folder, settings)
    if warm_start:
        classifier.projector = M.Projector(768, classifier.language_model.config.hidden_size).to(device)
        settings.pop("siglip", None)
        settings.update(vision_encoder=DINO_ID, vq=False, side=IMAGE_SIDE,
                        soft_side=GRID_SIDE, init_checkpoint=str(folder))
    classifier.soft_side = GRID_SIDE
    # Keep the old head widths for weight transfer. The absent VQ features are zeros.
    classifier.image_range = (0, 0)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = DinoPrompt(tokenizer, image_token_offset=0, prefix_tokens=GRID_SIDE**2,
                         language_hint=settings.get("language_hint", False))
    return classifier, settings, builder


def encode_groups(encoder, rows, root, text, batches, workers=8, reader=None):
    """Encode each batch at a uniform image size. OCR is always read from the original image."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(workers) as pool:
        for indices, side in batches:
            records = [rows[index] for index in indices]
            paths = [root / row["image_path"] for row in records]
            fitted = list(pool.map(lambda path: fit_image(path, side), paths))
            features = encoder.encode(fitted, side)
            missing = [(row["image_id"], path) for row, path in zip(records, paths)
                       if row["image_id"] not in text]
            if missing:
                if reader is None:
                    raise ValueError("OCR text is missing from the cache.")
                originals = list(pool.map(lambda item: E.full_resolution(item[1]), missing))
                recognized = reader.read(originals)
                text.update((key, value) for (key, _), value in zip(missing, recognized, strict=True))
            yield records, features


def model_inputs(builder, records, features, ocr, device):
    sequences = [builder.ids(row, ocr[row["image_id"]]) for row in records]
    ids, mask = E.collate(sequences, builder.tokenizer.pad_token_id)
    return ids.to(device), mask.to(device), features.to(device)


def probabilities(logits):
    parts = [task.float().softmax(-1).cpu() for task in logits]
    if any(not task.isfinite().all() for task in parts):
        raise ValueError("Classifier produced non-finite probabilities.")
    return parts
