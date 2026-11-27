"""P11: Apertus 8B + LoRA fuso + 6 teste (`src/apertus-8B-lora-heads`). Piano B.4.1. NON verificata su GPU."""

from __future__ import annotations

from contextlib import nullcontext

from .base import REPO_ROOT, add_path, assert_module_origin
from .heads_common import HeadsFamilyBase

FOLDER = REPO_ROOT / "src" / "apertus-8B-lora-heads"


class LoraHeadsFamily(HeadsFamilyBase):
    name = "lora_heads"

    def _modules(self):
        add_path(REPO_ROOT / "src")
        add_path(FOLDER)
        import data as D
        import encoders as E
        import model as M

        for module in (D, E, M):
            assert_module_origin(module, FOLDER)
        return {"D": D, "E": E, "M": M}

    def _load_classifier(self, modules, base_dir, checkpoint, device):
        return modules["M"].load(base_dir, checkpoint, device)  # LoRA fuso nel decoder (merge=True)

    def _make_encoder(self, modules, base_dir, device):
        return modules["E"].ImageEncoder(base_dir, "cuda")

    def _make_builder(self, modules, tokenizer, config):
        return modules["E"].PromptBuilder(tokenizer, config.image_token_offset)

    def _autocast(self, modules, device):
        import torch

        return torch.autocast(device.type, dtype=torch.bfloat16)
