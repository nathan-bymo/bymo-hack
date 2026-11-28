"""One Apertus decoder with aligned SigLIP and DINOv3 features."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apertus-8B-dinov3-lora-heads"))
from base_weights import PATCH_NAME, restore_blocks  # noqa: E402
from pipeline import D, DINO_ID, E, M, DinoEncoder  # noqa: E402

FUSION_NAME = "mixed_fusion.pt"
DINO_SIDE = 288
DINO_AUGMENTED_SIDE = 320


class MixedClassifier(nn.Module):
    def __init__(self, aperture, vision_size=768):
        super().__init__()
        self.aperture = aperture
        self.fusion = nn.Sequential(nn.LayerNorm(vision_size), nn.Linear(vision_size, vision_size, bias=False))
        nn.init.zeros_(self.fusion[-1].weight)

    def forward(self, input_ids, mask, siglip, dino):
        if siglip.shape != dino.shape:
            raise ValueError("SigLIP and DINO require matching patch grids and feature widths.")
        mixed = siglip.float() + self.fusion(dino.float()).float()
        return self.aperture(input_ids, mask, mixed)


def load_mixed(directory, folder, dino_revision=None, device="cuda"):
    from transformers import AutoConfig, AutoTokenizer

    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())
    if (settings.get("model") != E.MODEL_ID or settings.get("revision") != E.MODEL_REVISION
            or settings.get("siglip") != E.SIGLIP_ID or settings.get("side") != 256
            or settings.get("vision_dim") != 768 or settings.get("layers") != 24
            or not all(settings.get(name) for name in ("adapter", "ocr", "spatial", "attention_pool"))):
        raise ValueError("Expected the 24-layer SigLIP + VQ + OCR checkpoint or a continuation of it.")
    tokenizer = AutoTokenizer.from_pretrained(directory)
    config = AutoConfig.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, config.image_token_offset, prefix_tokens=E.POOLED_SIDE**2,
                              language_hint=settings.get("language_hint", False))
    if settings.get("row_end_id") not in (None, builder.eol):
        raise ValueError("The VQ row delimiter does not match the Apertus tokenizer.")
    settings["row_end_id"] = builder.eol
    aperture = M.build(directory, settings, device)
    M.load_weights(aperture, folder, settings)
    restore_blocks(aperture.language_model, folder, settings)
    model = MixedClassifier(aperture).to(device)
    if settings.get("mixed"):
        fusion = torch.load(folder / FUSION_NAME, map_location="cpu", weights_only=True)
        model.fusion.load_state_dict(fusion)
        if dino_revision and settings["dino_revision"] != dino_revision:
            raise ValueError("DINO revision does not match the mixed checkpoint.")
    else:
        if not dino_revision:
            raise ValueError("A DINO revision is required when starting from SigLIP alone.")
        settings.update(mixed=True, dino=DINO_ID, dino_revision=dino_revision,
                        dino_side=DINO_SIDE, visual_tokens=E.POOLED_SIDE**2)
    return model, settings, builder


def save_mixed(model, settings, folder, masters=None):
    if settings.get("unfrozen_blocks") and (masters is None or not masters.master):
        raise ValueError("A mixed checkpoint with unfrozen Apertus blocks requires FP32 master weights.")
    M.save(model.aperture, settings, folder)
    torch.save(model.fusion.state_dict(), folder / FUSION_NAME)
    if settings.get("unfrozen_blocks"):
        masters.save(folder)
        if not (folder / PATCH_NAME).is_file():
            raise RuntimeError("The base weight patch was not written.")


class MixedEncoder:
    def __init__(self, directory, dino_revision, device="cuda"):
        self.siglip_vq = E.ImageEncoder(directory, device)
        self.dino = DinoEncoder(dino_revision, device=device, grid_side=E.POOLED_SIDE)
        self.reader = E.TextReader(device)

    def encode(self, records, variants, data_dir, ocr, workers):
        paths = [data_dir / record["image_path"] for record in records]
        encoded = self.siglip_vq.encode(paths, variants, 256)
        originals = list(workers.map(E.full_resolution, paths))
        dino = [None] * len(paths)
        for variant, side in (("", DINO_SIDE), ("a2", DINO_AUGMENTED_SIDE)):
            positions = [i for i, name in enumerate(variants) if name == variant]
            if positions:
                fitted = [originals[i].resize((side, side), Image.Resampling.BICUBIC) for i in positions]
                features = self.dino.encode(fitted, side)
                for i, value in zip(positions, features, strict=True):
                    dino[i] = value
        missing = [i for i, record in enumerate(records) if record["image_id"] not in ocr]
        if missing:
            read = self.reader.read([originals[i] for i in missing])
            ocr.update((records[i]["image_id"], text) for i, text in zip(missing, read, strict=True))
        for entry, features in zip(encoded, dino, strict=True):
            if features is None:
                raise ValueError("Unknown variant for DINO features.")
            entry["dino"] = features
        return encoded

    def move(self, device):
        self.siglip_vq.vq.to(device)
        self.siglip_vq.siglip.to(device)
        self.dino.model.to(device)
        self.reader.predictor.to(device)


def batch_inputs(builder, records, features, ocr, device):
    sequences = [builder.ids(row, vision["grid"], vision["codes"], ocr=ocr[row["image_id"]])
                 for row, vision in zip(records, features, strict=True)]
    ids, mask = E.collate(sequences, builder.tokenizer.pad_token_id)
    sources = (torch.stack([entry[key] for entry in features]).to(device) for key in ("siglip", "dino"))
    return ids.to(device), mask.to(device), *sources


def prepare_features(encoder, rows, items, data_dir, ocr, batch_size):
    from experiment import chunks

    result = {}
    with ThreadPoolExecutor(8) as workers:
        for number, part in enumerate(chunks(items, batch_size), start=1):
            records = [rows[i] for i, _ in part]
            variants = [variant for _, variant in part]
            features = encoder.encode(records, variants, data_dir, ocr, workers)
            for row, variant, entry in zip(records, variants, features, strict=True):
                result[D.key(row["image_id"], variant)] = entry
            if number % 50 == 0:
                print(f"Encoded {len(result)}/{len(items)} mixed views", flush=True)
    if len(result) != len(items):
        raise ValueError("Mixed feature cache contains missing or duplicate views.")
    return result
