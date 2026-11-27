"""Adapter lato API per i modelli che girano nel worker GPU (piano B.5.3): inoltra via HTTP, con token se serve."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Callable

import httpx

from ..errors import STATUS, ApiError
from ..memory import ResidentSet
from ..registry import ModelEntry, Registry
from ..settings import Settings
from ..taxonomy import dimensions
from .base import BatchInfo, DimensionPrediction, ImagePrediction, StoredImage

LOAD_TIMEOUT = 15 * 60
PER_IMAGE_TIMEOUT = 120


def _api_error(payload: dict, default: str = "UPSTREAM_ERROR") -> ApiError:
    error = payload.get("error", payload)
    code = error.get("code") if error.get("code") in STATUS else default
    return ApiError(code, error.get("message", "The GPU worker returned an error."), error.get("details") or {})


class WorkerProxy:
    def __init__(self, entry: ModelEntry, registry: Registry, settings: Settings, residents: ResidentSet,
                 unload_other: Callable[[str], None]):
        raw = entry.raw
        self.entry, self.registry, self.settings, self.residents = entry, registry, settings, residents
        self.unload_other = unload_other
        self.model_id, self.family = entry.id, raw["family"]
        self.url = settings.worker_urls[self.family].rstrip("/")
        self.headers = {"X-Demo-Worker-Token": settings.worker_token} if settings.worker_token else {}
        weights = raw.get("weights") or {}
        self.weights = weights
        self.base = registry.defaults.get("base_model", {})
        where = weights.get("local_path") or f"{weights.get('revision')}/{weights.get('subfolder')}"
        self.model_revision = str(where)
        prompt = raw.get("prompt") or {}
        self.side = int(prompt.get("side", 256))
        self.prompt_version = f"{prompt.get('version', 'heads-task')}:side{self.side}"
        self.vram_gb = float(raw.get("vram_estimate_gb") or 24)

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, timeout: float, **kwargs) -> httpx.Response:
        try:
            response = httpx.request(method, self.url + path, headers=self.headers, timeout=timeout, **kwargs)
        except httpx.TimeoutException:
            raise ApiError("TIMEOUT", "The GPU worker did not answer in time.", {"model_id": self.model_id}) from None
        except httpx.HTTPError:
            raise ApiError("UPSTREAM_ERROR", "The GPU worker is not reachable.", {"model_id": self.model_id}) from None
        if response.status_code == 401:
            raise ApiError("UPSTREAM_ERROR", "The GPU worker rejected the token.", {"model_id": self.model_id})
        if response.status_code >= 400:
            raise _api_error(response.json() if response.content else {})
        return response

    def info(self) -> dict:
        return self._request("GET", "/worker/info", 10).json()

    def _set_state(self, state: str) -> None:
        self.registry.worker_state[self.family] = state

    # ------------------------------------------------------------------ caricamento e memoria
    def ensure_loaded(self, on_state: Callable[[str, str], None] | None = None) -> None:
        info = self.info()
        if self.model_id in info.get("loaded", []):
            self.residents.touch(self.model_id)
            self._set_state("ready")
            return
        budget = max(0.0, float(info.get("total_gb") or 0) - 2.0)  # memoria della GPU meno 2 GB
        for victim in self.residents.plan(self.model_id, self.vram_gb, budget):
            self.unload_other(victim)
        self._set_state("loading")
        body = {"model_id": self.model_id, "weights": self.weights, "base": self.base, "side": self.side}
        final = None
        try:
            with httpx.stream("POST", self.url + "/worker/load", json=body, headers=self.headers,
                              timeout=LOAD_TIMEOUT) as stream:
                if stream.status_code >= 400:
                    stream.read()
                    raise _api_error(json.loads(stream.text or "{}"))
                for line in stream.iter_lines():
                    if not line.strip():
                        continue
                    final = json.loads(line)
                    if on_state:
                        on_state(final["state"], final.get("message", ""))
        except httpx.TimeoutException:
            self._set_state("error")
            raise ApiError("TIMEOUT", "Model loading took longer than 15 minutes.", {"model_id": self.model_id}) from None
        except httpx.HTTPError:
            self._set_state("error")
            raise ApiError("UPSTREAM_ERROR", "The GPU worker is not reachable.", {"model_id": self.model_id}) from None
        if not final or final["state"] != "ready":
            self._set_state("error")
            error = final or {}
            code = error.get("code") if error.get("code") in STATUS else "UPSTREAM_ERROR"
            raise ApiError(code, error.get("message", "Model loading failed."), error.get("details") or {})
        self._set_state("ready")
        self.residents.add(self.model_id, float(final.get("peak_memory_gb") or self.vram_gb))

    def unload(self) -> None:
        self._request("POST", "/worker/unload", 60, json={"model_id": self.model_id})
        self.residents.remove(self.model_id)
        self._set_state("not_loaded")

    # ------------------------------------------------------------------ predizione
    def predict(self, images: list[StoredImage], use_context: bool, batch_size: int,
                on_result: Callable[[ImagePrediction], None], on_error: Callable[[str, Exception], None],
                should_cancel: Callable[[], bool]) -> BatchInfo:
        self.ensure_loaded()
        size = max(1, batch_size or 8)
        started, done, ips, gpu, peak = time.perf_counter(), 0, None, "worker GPU", None
        for start in range(0, len(images), size):
            if should_cancel():
                break
            chunk = images[start:start + size]
            payload = [{"image_id": i.image_id, "sha256": i.sha256, "ext": Path(i.path).suffix.lstrip("."),
                        "bytes_b64": base64.b64encode(Path(i.path).read_bytes()).decode(), "context": i.context}
                       for i in chunk]
            self.residents.touch(self.model_id)
            body = self._request("POST", "/worker/predict", PER_IMAGE_TIMEOUT * len(chunk), json={
                "model_id": self.model_id, "use_context": use_context, "batch_size": size, "images": payload}).json()
            batch = body["batch"]
            gpu, peak = batch.get("gpu_name") or gpu, batch.get("peak_memory_gb") or peak
            for result in body["results"]:
                dims = {}
                for d in dimensions():
                    x = result["dimensions"][d]
                    dims[d] = DimensionPrediction(x["label"], x["confidence"],
                                                  [(t["label"], t["p"]) for t in x["top_k"]] if x.get("top_k") else None,
                                                  x["confidence_method"])
                on_result(ImagePrediction(result["image_id"], dims, result.get("raw_output"), result.get("timing_ms") or {}))
                done += 1
        seconds = time.perf_counter() - started
        ips = done / seconds if done and seconds else None  # dall'immagine salvata alla risposta, trasferimento incluso
        return BatchInfo(done, seconds, ips, gpu, size, peak)
