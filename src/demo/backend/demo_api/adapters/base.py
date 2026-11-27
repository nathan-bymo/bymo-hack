"""Interfaccia degli adapter (piano B.1). Gli adapter chiamano il codice del repo: non reimplementano il preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol

ConfidenceMethod = Literal["head_softmax", "letter_logprob", "sequence_prob", "constrained_scoring"]


@dataclass(frozen=True)
class StoredImage:
    image_id: str               # id della demo (img-demo-…), mai un id della competizione
    sha256: str
    path: Path                  # data/demo/uploads/<aa>/<sha256>.<ext>
    context: dict               # chiavi di prompt.CONTEXT_FIELDS; {} se assente
    context_hash: str = ""


@dataclass
class DimensionPrediction:
    label: str | None                              # stringa esatta di prompt.TAXONOMY[d]; None = output non valido
    confidence: float | None = None                # None -> "n/a"
    top_k: list[tuple[str, float]] | None = None   # max 3, ordinati
    confidence_method: ConfidenceMethod | None = None

    def to_json(self) -> dict:
        return {"label": self.label, "confidence": self.confidence,
                "top_k": [{"label": l, "p": p} for l, p in self.top_k] if self.top_k else None,
                "confidence_method": self.confidence_method}


@dataclass
class ImagePrediction:
    image_id: str
    dimensions: dict[str, DimensionPrediction]     # sempre tutte e 6, nell'ordine di prompt.DIMENSIONS
    raw_output: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(d.label is not None for d in self.dimensions.values())


@dataclass
class BatchInfo:
    images: int
    seconds: float
    images_per_second: float | None
    gpu_name: str
    batch_size: int
    peak_memory_gb: float | None = None

    def to_json(self) -> dict:
        ms = (self.seconds / self.images * 1000) if self.images else None
        return {"images": self.images, "seconds": self.seconds, "images_per_second": self.images_per_second,
                "ms_per_image": ms, "gpu_name": self.gpu_name, "batch_size": self.batch_size,
                "peak_memory_gb": self.peak_memory_gb}


class ModelAdapter(Protocol):
    model_id: str
    model_revision: str
    prompt_version: str

    def predict(self, images: list[StoredImage], use_context: bool, batch_size: int,
                on_result: Callable[[ImagePrediction], None],
                on_error: Callable[[str, Exception], None],
                should_cancel: Callable[[], bool]) -> BatchInfo: ...


def top_k(probabilities: dict[str, float] | None, k: int = 3) -> list[tuple[str, float]] | None:
    if not probabilities:
        return None
    return sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
