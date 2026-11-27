"""Gestione della memoria dei modelli sull'host GPU (piano B.8): budget VRAM, LRU, mai scaricare modelli con job attivi."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable

from .errors import ApiError


class ResidentSet:
    """Modelli residenti in ordine di uso (il meno recente per primo).

    `budget_gb` = memoria totale della GPU meno 2 GB (letta dal worker); `max_resident` = DEMO_MAX_RESIDENT (default 1).
    """

    def __init__(self, busy: Callable[[str], bool], max_resident: int = 1):
        self.busy = busy
        self.max_resident = max_resident
        self.resident: OrderedDict[str, float] = OrderedDict()
        self.lock = threading.RLock()

    def touch(self, model_id: str) -> None:
        with self.lock:
            if model_id in self.resident:
                self.resident.move_to_end(model_id)

    def plan(self, model_id: str, needed_gb: float, budget_gb: float) -> list[str]:
        """Modelli da scaricare (dal meno recente) perché `model_id` entri; `MODEL_BUSY` se non si può liberare spazio."""
        with self.lock:
            if model_id in self.resident:
                return []
            if needed_gb > budget_gb:
                raise ApiError("MODEL_BUSY", "The model does not fit in the GPU memory budget.",
                               {"needed_gb": needed_gb, "budget_gb": budget_gb})
            victims: list[str] = []
            remaining = OrderedDict(self.resident)
            while len(remaining) >= self.max_resident or sum(remaining.values()) + needed_gb > budget_gb:
                candidate = next((m for m in remaining if not self.busy(m)), None)
                if candidate is None:
                    raise ApiError("MODEL_BUSY", "GPU memory is full and every resident model has a running job.",
                                   {"resident": list(remaining)})
                victims.append(candidate)
                del remaining[candidate]
            return victims

    def add(self, model_id: str, gb: float) -> None:
        with self.lock:
            self.resident[model_id] = gb
            self.resident.move_to_end(model_id)

    def remove(self, model_id: str) -> None:
        with self.lock:
            self.resident.pop(model_id, None)
