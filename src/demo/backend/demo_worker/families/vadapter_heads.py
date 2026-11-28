"""Modello finale: SigLIP 2 + OCR docTR + Apertus 8B (24 layer, LoRA fuso) + 6 teste (`src/training/apertus-8B-vadapter-lora-heads`).

Ripete passo per passo il ciclo di `predict.py` (encode → OCR → prompt → forward → softmax per testa), senza TTA e
senza bias di calibrazione: come la submission del dev. NON VERIFICATA su GPU in questa sessione (nessuna GPU
disponibile): resta "parity not verified" finché `scripts/parity.py` non confronta le probabilità con `predict.py`.
"""

from __future__ import annotations

import time

from .base import DIMENSIONS, REPO_ROOT, WorkerError, add_path, assert_module_origin, head_prediction
from .heads_common import HeadsFamilyBase

PHASE_SRC = REPO_ROOT / "src" / "training" / "phase-2" / "src"
FOLDER = PHASE_SRC / "training" / "apertus-8B-vadapter-lora-heads"


class VadapterHeadsFamily(HeadsFamilyBase):
    name = "vadapter_heads"

    def _modules(self):
        add_path(PHASE_SRC)
        add_path(FOLDER)
        import data as D
        import encoders as E
        import model as M

        for module in (D, E, M):
            assert_module_origin(module, FOLDER)
        return {"D": D, "E": E, "M": M}

    def load(self, model_id: str, cfg: dict, on_state) -> dict:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from .base import resolve_weights

        if not torch.cuda.is_available():
            raise WorkerError("MODEL_UNAVAILABLE", "no CUDA GPU is visible to the worker")
        on_state("loading", "resolving local weights")
        base_dir, checkpoint = resolve_weights(cfg["weights"], cfg["base"])
        modules = self._modules()
        E, M = modules["E"], modules["M"]
        torch.cuda.reset_peak_memory_stats()
        on_state("loading", "loading decoder, LoRA, projector, pooling and heads")
        classifier, settings = M.load(str(base_dir), checkpoint, "cuda:0", merge=True, dtype=torch.bfloat16)
        on_state("loading", "loading VQ tokenizer and SigLIP 2")
        encoder = E.ImageEncoder(str(base_dir), "cuda")
        reader = None
        if settings.get("ocr"):
            on_state("loading", "loading docTR OCR")
            reader = E.TextReader("cuda")
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        prefix = E.POOLED_SIDE ** 2 if settings.get("adapter", True) else 0
        builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(base_dir).image_token_offset, prefix,
                                  settings.get("language_hint", False))
        self.models[model_id] = {"classifier": classifier, "settings": settings, "encoder": encoder, "reader": reader,
                                 "tokenizer": tokenizer, "builder": builder, "modules": modules,
                                 "device": torch.device("cuda:0")}
        return {"peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "side": settings["side"]}

    def _run_batch(self, m: dict, items: list[dict], paths: list, use_context: bool) -> list[dict]:
        import torch

        E, D = m["modules"]["E"], m["modules"]["D"]
        t0 = time.perf_counter()
        encoded = m["encoder"].encode(paths, [""] * len(paths), m["settings"]["side"])
        texts = m["reader"].read([E.full_resolution(p) for p in paths]) if m["reader"] else [None] * len(paths)
        for entry, text in zip(encoded, texts):
            entry["ocr"] = text
        t1 = time.perf_counter()
        sequences = [m["builder"].ids(item.get("context") or {}, e["grid"], e["codes"], use_context, e["ocr"])
                     for item, e in zip(items, encoded)]
        input_ids, attention = E.collate(sequences, m["tokenizer"].pad_token_id)
        siglip = torch.stack([e["siglip"] for e in encoded])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m["classifier"](input_ids.to(m["device"]), attention.to(m["device"]), siglip.to(m["device"]))
        probabilities = [x.float().softmax(-1).cpu() for x in logits]
        if any(not torch.isfinite(p).all() for p in probabilities):
            raise WorkerError("INTERNAL_ERROR", "non-finite probabilities")
        t2 = time.perf_counter()
        per = len(items)
        out = []
        for index, item in enumerate(items):
            probs = [probabilities[t][index].tolist() for t in range(len(DIMENSIONS))]
            out.append({"image_id": item["image_id"], "dimensions": head_prediction(D.TAXONOMY, probs),
                        "raw_output": f"OCR: {texts[index]}" if texts[index] else None,
                        "timing_ms": {"preprocess": (t1 - t0) * 1000 / per, "model": (t2 - t1) * 1000 / per,
                                      "total": (t2 - t0) * 1000 / per}})
        return out

    # predict() resta quello della classe base: scrive i file, divide in lotti, dimezza il lotto in caso di OOM.
