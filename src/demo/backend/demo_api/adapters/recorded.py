"""Modalità registrato (piano B.5.4): risultati veri prodotti prima su GPU, riletti da `data/demo/recorded/`.

Layout: `recorded/<model_id>/<run_id>/{run.json, predictions.jsonl}`. Lookup per (sha256, hash del contesto,
use_context). Nessuna predizione inventata: se manca la registrazione si risponde `RECORDED_MISSING`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..db import Database
from ..errors import ApiError
from ..taxonomy import dimensions
from .base import BatchInfo, DimensionPrediction, ImagePrediction, StoredImage


def scan(db: Database, recorded_dir: Path) -> int:
    """Rilegge le registrazioni e riallinea la tabella `recorded_runs`. Restituisce il numero di run."""
    db.execute("DELETE FROM recorded_runs")
    count = 0
    if not recorded_dir.exists():
        return 0
    for run_json in sorted(recorded_dir.glob("*/*/run.json")):
        run_dir = run_json.parent
        try:
            meta = json.loads(run_json.read_text())
        except ValueError:
            continue
        preds = run_dir / "predictions.jsonl"
        if not preds.exists():
            continue
        images = sum(1 for line in preds.read_text().splitlines() if line.strip())
        db.execute("INSERT OR REPLACE INTO recorded_runs (run_id, model_id, path, recorded_at, gpu_name, images, "
                   "meta_json) VALUES (?,?,?,?,?,?,?)",
                   (run_dir.name, run_json.parent.parent.name, str(run_dir),
                    meta.get("finished_at") or meta.get("started_at"), meta.get("gpu_name"), images,
                    json.dumps(meta)))
        count += 1
    return count


def recorded_context(db: Database, sha256: str) -> dict | None:
    """Contesto prodotto con cui una foto è stata registrata (se la registrazione lo riporta).

    Serve all'upload: la stessa foto, caricata senza dettagli, riceve quelli usati dal modello, così la risposta
    registrata corrisponde davvero all'input mostrato nella pagina.
    """
    for row in db.all("SELECT path FROM recorded_runs ORDER BY recorded_at DESC"):
        preds = Path(row["path"]) / "predictions.jsonl"
        if not preds.exists():
            continue
        for line in preds.read_text().splitlines():
            if line.strip() and sha256 in line:
                rec = json.loads(line)
                if rec.get("image_sha256") == sha256 and rec.get("context"):
                    return rec["context"]
    return None


def models_recorded_for(db: Database, sha256: str) -> list[str]:
    """Modelli che hanno una risposta registrata per questa foto (qualunque contesto): quelli che rispondono subito."""
    found: list[str] = []
    for row in db.all("SELECT model_id, path FROM recorded_runs ORDER BY model_id, recorded_at DESC"):
        if row["model_id"] in found:
            continue
        preds = Path(row["path"]) / "predictions.jsonl"
        if preds.exists() and any(sha256 in line and json.loads(line).get("image_sha256") == sha256
                                  for line in preds.read_text().splitlines() if line.strip()):
            found.append(row["model_id"])
    return found


def run_info(row: dict) -> dict:
    meta = json.loads(row.get("meta_json") or "{}")
    return {"run_id": row["run_id"], "recorded_at": row["recorded_at"], "gpu_name": row["gpu_name"],
            "images_per_second": meta.get("images_per_second")}


class RecordedAdapter:
    def __init__(self, model_id: str, db: Database):
        self.model_id, self.db = model_id, db
        runs = db.all("SELECT * FROM recorded_runs WHERE model_id = ? ORDER BY recorded_at DESC, run_id DESC",
                      (model_id,))
        if not runs:
            raise ApiError("MODEL_UNAVAILABLE", "No recorded predictions for this model.", {"model_id": model_id})
        self.runs = runs
        latest = json.loads(runs[0]["meta_json"] or "{}")
        self.model_revision = f"recorded:{latest.get('model_revision', runs[0]['run_id'])}"
        self.prompt_version = f"recorded:{runs[0]['run_id']}"
        self._index: dict[str, dict] = {}

    def _load(self, run: dict) -> dict:
        if run["run_id"] not in self._index:
            table = {}
            for line in (Path(run["path"]) / "predictions.jsonl").read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    table[(rec["image_sha256"], rec.get("context_hash", ""), bool(rec.get("use_context", True)))] = rec
            self._index[run["run_id"]] = table
        return self._index[run["run_id"]]

    def find(self, image: StoredImage, use_context: bool):
        key = (image.sha256, image.context_hash, use_context)
        for run in self.runs:
            rec = self._load(run).get(key)
            if rec is not None:
                return run, rec
        return None, None

    def predict(self, images: list[StoredImage], use_context: bool, batch_size: int,
                on_result: Callable[[ImagePrediction], None], on_error: Callable[[str, Exception], None],
                should_cancel: Callable[[], bool]) -> BatchInfo:
        started, done, used = time.perf_counter(), 0, None
        for image in images:
            if should_cancel():
                break
            run, rec = self.find(image, use_context)
            if rec is None:
                on_error(image.image_id, ApiError(
                    "RECORDED_MISSING", "No recorded prediction for this image and context.",
                    {"model_id": self.model_id}))
                continue
            used = used or run
            dims = {d: _dim(rec["dimensions"][d]) for d in dimensions()}
            prediction = ImagePrediction(image.image_id, dims, rec.get("raw_output"), rec.get("timing_ms") or {})
            prediction.recorded_run_id = run["run_id"]  # type: ignore[attr-defined]
            on_result(prediction)
            done += 1
        info = run_info(used) if used else {}
        meta = json.loads(used["meta_json"]) if used else {}
        batch = BatchInfo(done, time.perf_counter() - started, info.get("images_per_second"),
                          info.get("gpu_name") or "recorded", int(meta.get("batch_size") or 0),
                          meta.get("peak_memory_gb"))
        batch.recorded = info  # type: ignore[attr-defined]
        return batch


def _dim(d: dict) -> DimensionPrediction:
    top = [(t["label"], t["p"]) for t in d["top_k"]] if d.get("top_k") else None
    return DimensionPrediction(d.get("label"), d.get("confidence"), top, d.get("confidence_method"))
