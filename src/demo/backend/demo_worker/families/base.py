"""Base comune delle famiglie del worker. Il worker non scarica mai pesi: risolve percorsi solo dalla cache HF."""

from __future__ import annotations

import base64
import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")
REPO_ROOT = Path(__file__).resolve().parents[5]


class WorkerError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def add_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def assert_module_origin(module, expected_dir: Path) -> None:
    """Le cartelle del repo hanno moduli con lo stesso nome (`data`, `encoders`, `model`): si controlla da dove arrivano."""
    actual = Path(module.__file__).resolve().parent
    if actual != Path(expected_dir).resolve():
        raise WorkerError("INTERNAL_ERROR", f"module {module.__name__} imported from the wrong folder",
                          {"expected": str(expected_dir), "actual": str(actual)})


def resolve_weights(spec: dict, base: dict) -> tuple[Path, Path]:
    """(cartella del modello base, cartella del checkpoint) dalla sola cache locale (`local_files_only=True`)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        base_dir = Path(snapshot_download(base["repo"], revision=base["revision"], local_files_only=True))
        if spec.get("local_path"):
            checkpoint = Path(spec["local_path"])
            if not checkpoint.is_absolute():   # in models.yaml i percorsi locali partono dalla radice del repo
                checkpoint = REPO_ROOT / checkpoint
        else:
            root = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                          allow_patterns=[f"{spec['subfolder']}/*"], local_files_only=True))
            checkpoint = root / spec["subfolder"]
    except LocalEntryNotFoundError as error:
        raise WorkerError("MODEL_UNAVAILABLE", "weights not found in the local cache (the worker never downloads)",
                          {"reason": str(error)[:200]}) from None
    if not checkpoint.exists():
        raise WorkerError("MODEL_UNAVAILABLE", "checkpoint folder not found", {"folder": checkpoint.name})
    return base_dir, checkpoint


@contextlib.contextmanager
def written_images(items: list[dict]):
    """Scrive i byte ricevuti in file temporanei con l'estensione originale e restituisce i percorsi (le funzioni del
    repo leggono da file: i byte sono quelli salvati dall'API, quindi lo stesso input)."""
    with tempfile.TemporaryDirectory(prefix="demo-worker-") as folder:
        paths = []
        for index, item in enumerate(items):
            path = Path(folder) / f"{index}.{item['ext']}"
            path.write_bytes(base64.b64decode(item["bytes_b64"]))
            paths.append(path)
        yield paths


def head_prediction(taxonomy: dict, probabilities: list[list[float]]) -> dict:
    """Per ogni dimensione: label = argmax, confidenza = probabilità massima, top-3 (softmax della testa)."""
    out = {}
    for dimension, probs in zip(DIMENSIONS, probabilities):
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        classes = taxonomy[dimension]
        out[dimension] = {"label": classes[order[0]], "confidence": float(probs[order[0]]),
                          "top_k": [{"label": classes[i], "p": float(probs[i])} for i in order[:3]],
                          "confidence_method": "head_softmax"}
    return out


class Family:
    name = ""

    def info(self) -> dict:  # pragma: no cover - sovrascritto
        raise NotImplementedError

    def loaded(self) -> list[str]:
        return []

    def load(self, model_id: str, cfg: dict, on_state: Callable[[str, str], None]) -> dict:
        raise NotImplementedError

    def unload(self, model_id: str) -> None:
        raise NotImplementedError

    def predict(self, model_id: str, items: list[dict], use_context: bool, batch_size: int) -> tuple[list[dict], dict]:
        raise NotImplementedError


def gpu_info() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return {"gpu_name": torch.cuda.get_device_name(0), "total_gb": round(total / 1024**3, 2),
                    "free_gb": round(free / 1024**3, 2), "torch": torch.__version__}
        return {"gpu_name": None, "total_gb": 0.0, "free_gb": 0.0, "torch": torch.__version__}
    except Exception:  # noqa: BLE001 - torch assente (macchina senza GPU)
        return {"gpu_name": None, "total_gb": 0.0, "free_gb": 0.0, "torch": None}


def timer() -> Callable[[], float]:
    start = time.perf_counter()
    return lambda: (time.perf_counter() - start) * 1000
