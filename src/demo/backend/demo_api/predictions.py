"""Salvataggio e formato delle predizioni (allegato C, C.5)."""

from __future__ import annotations

import json
import uuid

from .adapters.base import ImagePrediction
from .db import Database, now_iso
from .relevance import applicable
from .taxonomy import dimensions, erp_label


def cache_key_where() -> str:
    return ("image_id = ? AND model_id = ? AND model_revision = ? AND image_sha256 = ? AND context_hash = ? "
            "AND use_context = ? AND prompt_version = ? AND mode = ?")


def find_cached(db: Database, image: dict, model_id: str, revision: str, prompt_version: str, use_context: bool,
                mode: str) -> dict | None:
    return db.one(f"SELECT * FROM predictions WHERE {cache_key_where()}",
                  (image["image_id"], model_id, revision, image["sha256"], image["context_hash"], int(use_context),
                   prompt_version, mode))


def store(db: Database, image: dict, model_id: str, revision: str, prompt_version: str, use_context: bool,
          mode: str, prediction: ImagePrediction, job_id: str | None, recorded_run_id: str | None = None) -> dict:
    prediction_id = f"pred_{uuid.uuid4()}"
    dims = {d: prediction.dimensions[d].to_json() for d in dimensions()}
    db.execute(
        "INSERT OR REPLACE INTO predictions (prediction_id, image_id, image_sha256, model_id, model_revision, "
        "prompt_version, use_context, context_hash, mode, recorded_run_id, valid, dimensions_json, raw_output, "
        "timing_json, job_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (prediction_id, image["image_id"], image["sha256"], model_id, revision, prompt_version, int(use_context),
         image["context_hash"], mode, recorded_run_id, int(prediction.valid),
         json.dumps(dims, ensure_ascii=False), prediction.raw_output, json.dumps(prediction.timing_ms), job_id,
         now_iso()))
    return db.one("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))


def to_result(row: dict, cached: bool = False, recorded: dict | None = None) -> dict:
    """Una riga di `predictions` nel formato di `results` (C.5), con etichette ERP e applicabilità."""
    dims = json.loads(row["dimensions_json"])
    labels = {d: dims[d]["label"] for d in dimensions()}
    ok = applicable(labels)
    out = {}
    for d in dimensions():
        label_erp, source = erp_label(d, dims[d]["label"])
        out[d] = {**dims[d], "erp_label": label_erp, "erp_label_source": source, "applicable": ok[d]}
    return {
        "image_id": row["image_id"], "prediction_id": row["prediction_id"], "model_id": row["model_id"],
        "model_revision": row["model_revision"], "cached": cached, "mode": row["mode"], "valid": bool(row["valid"]),
        "dimensions": out, "raw_output": row["raw_output"], "timing": _timing(row), "recorded": recorded,
        "use_context": bool(row["use_context"]), "created_at": row["created_at"],
    }


def _timing(row: dict) -> dict:
    t = json.loads(row["timing_json"] or "{}")
    return {"preprocess_ms": t.get("preprocess"), "model_ms": t.get("model"), "total_ms": t.get("total")}


def latest_per_model(db: Database, image_id: str) -> list[dict]:
    """Ultima predizione dell'immagine per ogni modello (C.4: `GET /api/images/{id}`)."""
    rows = db.all("SELECT * FROM predictions WHERE image_id = ? ORDER BY created_at DESC, rowid DESC", (image_id,))
    seen, out = set(), []
    for row in rows:
        if row["model_id"] not in seen:
            seen.add(row["model_id"])
            out.append(to_result(row))
    return out
