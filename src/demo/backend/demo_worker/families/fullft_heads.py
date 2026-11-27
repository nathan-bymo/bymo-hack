"""Full fine-tune: 24 layer allenati + 6 teste (branch `feat/apertus-8b-fullft-heads`). Piano B.4.2. NON verificata.

Il codice è letto da `DEMO_FULLFT_SRC` (worktree del branch) finché il team non lo unisce in main.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import REPO_ROOT, WorkerError, add_path
from .heads_common import HeadsFamilyBase


class FullFtHeadsFamily(HeadsFamilyBase):
    name = "fullft_heads"

    def _modules(self):
        src = Path(os.environ.get("DEMO_FULLFT_SRC") or REPO_ROOT / "src").resolve()
        folder = src / "apertus-8B-fullft-heads"
        if not (folder / "ft_model.py").exists():
            raise WorkerError("MODEL_UNAVAILABLE", "full fine-tune code not found: set DEMO_FULLFT_SRC to the "
                              "`src` folder of a worktree of feat/apertus-8b-fullft-heads")
        add_path(REPO_ROOT / "src")
        add_path(folder)
        import ft_model as FM  # noqa: F401 - aggiunge da solo la cartella apertus-8B-vadapter-lora-heads al path
        import data as D
        import encoders as E
        import ft_encode as FE
        import ft_train as FT

        return {"D": D, "E": E, "FM": FM, "FE": FE, "FT": FT, "folder": folder}

    def _load_classifier(self, modules, base_dir, checkpoint, device):
        return modules["FM"].load(base_dir, checkpoint, device)

    def _make_encoder(self, modules, base_dir, device):
        return modules["FE"].VQEncoder(base_dir, device)

    def _make_builder(self, modules, tokenizer, config):
        return modules["E"].PromptBuilder(tokenizer, config.image_token_offset, prefix_tokens=0)

    def _autocast(self, modules, device):
        return modules["FT"].autocast(device)
