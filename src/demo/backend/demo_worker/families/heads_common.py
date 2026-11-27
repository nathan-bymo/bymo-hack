"""Logica comune delle due famiglie con teste (P11 e full fine-tune): un forward per immagine, softmax per testa.

NON VERIFICATA su GPU in questa sessione (nessuna GPU disponibile): il test di parità (`scripts/parity.py`) va
eseguito su un host GPU prima di fidarsi dei risultati. Il modello resta "parity not verified" finché non passa.
"""

from __future__ import annotations

import time
from typing import Callable

from .base import DIMENSIONS, Family, WorkerError, head_prediction, written_images


class HeadsFamilyBase(Family):
    """Le sottoclassi forniscono `_modules()`, `_load_classifier`, `_make_encoder`, `_make_builder`, `_autocast`."""

    def __init__(self):
        self.models: dict[str, dict] = {}

    # --- hook ---------------------------------------------------------------------------------------------
    def _modules(self):
        raise NotImplementedError

    def _load_classifier(self, modules, base_dir, checkpoint, device):
        raise NotImplementedError

    def _make_encoder(self, modules, base_dir, device):
        raise NotImplementedError

    def _make_builder(self, modules, tokenizer, config):
        raise NotImplementedError

    def _autocast(self, modules, device):
        raise NotImplementedError

    # --- protocollo ---------------------------------------------------------------------------------------
    def info(self) -> dict:
        from .base import gpu_info

        return {**gpu_info(), "transformers_commit": None, "code_commit": None, "loaded": self.loaded(),
                "family": self.name}

    def loaded(self) -> list[str]:
        return list(self.models)

    def load(self, model_id: str, cfg: dict, on_state: Callable[[str, str], None]) -> dict:
        import torch
        from transformers import AutoConfig, AutoTokenizer

        from .base import resolve_weights

        if not torch.cuda.is_available():
            raise WorkerError("MODEL_UNAVAILABLE", "no CUDA GPU is visible to the worker")
        on_state("loading", "resolving local weights")
        base_dir, checkpoint = resolve_weights(cfg["weights"], cfg["base"])
        modules = self._modules()
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        on_state("loading", "loading decoder and heads")
        classifier, settings = self._load_classifier(modules, base_dir, checkpoint, device)
        on_state("loading", "loading the VQ tokenizer")
        encoder = self._make_encoder(modules, base_dir, device)
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        builder = self._make_builder(modules, tokenizer, AutoConfig.from_pretrained(base_dir))
        self.models[model_id] = {"classifier": classifier, "settings": settings, "encoder": encoder,
                                 "tokenizer": tokenizer, "builder": builder, "modules": modules, "device": device}
        return {"peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "side": settings["side"]}

    def unload(self, model_id: str) -> None:
        import gc

        import torch

        self.models.pop(model_id, None)
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, model_id: str, items: list[dict], use_context: bool, batch_size: int):
        import torch

        if model_id not in self.models:
            raise WorkerError("MODEL_UNAVAILABLE", "model not loaded in this worker")
        m = self.models[model_id]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results: list[dict] = []
        with written_images(items) as paths:
            position, size = 0, max(1, batch_size)
            while position < len(items):
                chunk = slice(position, position + size)
                try:
                    results += self._run_batch(m, items[chunk], paths[chunk], use_context)
                    position += size
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise WorkerError("GPU_OOM", "GPU out of memory even at batch size 1") from None
                    size = max(1, size // 2)  # un solo nuovo tentativo con lotto dimezzato, poi ancora se serve
        seconds = time.perf_counter() - started
        return results, {"seconds": seconds, "images_per_second": len(items) / seconds if seconds else None,
                         "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                         "gpu_name": torch.cuda.get_device_name(0), "batch_size": batch_size}

    def _run_batch(self, m: dict, items: list[dict], paths: list, use_context: bool) -> list[dict]:
        import torch

        E, D = m["modules"]["E"], m["modules"]["D"]
        t0 = time.perf_counter()
        encoded = m["encoder"].encode(paths, [""] * len(paths), m["settings"]["side"])
        t1 = time.perf_counter()
        sequences = [m["builder"].ids(item.get("context") or {}, e["grid"], e["codes"], use_context)
                     for item, e in zip(items, encoded)]
        input_ids, attention = E.collate(sequences, m["tokenizer"].pad_token_id)
        with torch.inference_mode(), self._autocast(m["modules"], m["device"]):
            logits = m["classifier"](input_ids.to(m["device"]), attention.to(m["device"]))
        probabilities = [x.float().softmax(-1).cpu() for x in logits]
        t2 = time.perf_counter()
        per = len(items)
        out = []
        for index, item in enumerate(items):
            probs = [probabilities[t][index].tolist() for t in range(len(DIMENSIONS))]
            out.append({"image_id": item["image_id"], "dimensions": head_prediction(D.TAXONOMY, probs),
                        "raw_output": None,
                        "timing_ms": {"preprocess": (t1 - t0) * 1000 / per, "model": (t2 - t1) * 1000 / per,
                                      "total": (t2 - t0) * 1000 / per}})
        return out
