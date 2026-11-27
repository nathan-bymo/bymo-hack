"""Famiglia finta, SOLO per i test del protocollo (`--dry`): nessun modello, nessuna GPU, risposte deterministiche.

Si abilita solo con DEMO_WORKER_ALLOW_DRY=1 e non va mai usata per mostrare risultati: i valori sono segnaposto.
"""

from __future__ import annotations

import hashlib
import os

from .base import DIMENSIONS, Family, WorkerError, head_prediction, timer

TAXONOMY = {
    "subject": ("full_product", "detail_closeup"), "background": ("white", "color"),
    "perspective": ("front_head_on", "left_side"), "individuals": ("none", "individual_present"),
    "text_classification": ("none", "other"), "text_language": ("none", "German"),
}


class DryFamily(Family):
    def __init__(self, name: str):
        if os.environ.get("DEMO_WORKER_ALLOW_DRY") != "1":
            raise WorkerError("MODEL_UNAVAILABLE", "dry mode is only for protocol tests (DEMO_WORKER_ALLOW_DRY=1)")
        self.name = name
        self._loaded: dict[str, dict] = {}

    def info(self) -> dict:
        return {"family": self.name, "gpu_name": "DRY (no GPU)", "total_gb": 80.0, "free_gb": 80.0 - 8.0 * len(self._loaded),
                "torch": None, "transformers_commit": None, "code_commit": "dry", "loaded": self.loaded()}

    def loaded(self) -> list[str]:
        return list(self._loaded)

    def load(self, model_id, cfg, on_state):
        on_state("loading", "dry load")
        if cfg.get("fail"):
            raise WorkerError(cfg["fail"], "dry failure")
        self._loaded[model_id] = cfg
        return {"peak_memory_gb": 8.0}

    def unload(self, model_id):
        self._loaded.pop(model_id, None)

    def predict(self, model_id, items, use_context, batch_size):
        if model_id not in self._loaded:
            raise WorkerError("MODEL_UNAVAILABLE", "model not loaded")
        clock = timer()
        results = []
        for item in items:
            h = int(hashlib.sha256((item["sha256"] + model_id).encode()).hexdigest(), 16)
            probs = [[0.75, 0.25] if (h >> i) % 2 == 0 else [0.25, 0.75] for i, _ in enumerate(DIMENSIONS)]
            results.append({"image_id": item["image_id"], "dimensions": head_prediction(TAXONOMY, probs),
                            "raw_output": None, "timing_ms": {"preprocess": 1.0, "model": 1.0, "total": 2.0}})
        seconds = clock() / 1000
        return results, {"seconds": seconds, "images_per_second": len(items) / seconds if seconds else None,
                         "peak_memory_gb": 8.0, "gpu_name": "DRY (no GPU)", "batch_size": batch_size}
