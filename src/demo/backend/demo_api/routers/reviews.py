"""Revisione umana (C.6). Le revisioni sono dati della demo: nessun endpoint le esporta come dati di training."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..db import now_iso
from ..errors import ApiError
from ..taxonomy import dimensions, is_valid
from .images import get_image_row

router = APIRouter()


class ReviewBody(BaseModel):
    prediction_id: str | None = None
    model_id: str | None = None
    values: dict[str, str]
    note: str | None = None


def review_dict(row: dict) -> dict:
    return {"review_id": row["review_id"], "image_id": row["image_id"], "prediction_id": row["prediction_id"],
            "model_id": row["model_id"], "status": row["status"], "values": json.loads(row["values_json"]),
            "changed_dimensions": json.loads(row["changed_json"]), "note": row["note"],
            "created_at": row["created_at"]}


def latest_review(db, image_id: str) -> dict | None:
    row = db.one("SELECT * FROM reviews WHERE image_id = ? ORDER BY created_at DESC, rowid DESC", (image_id,))
    return review_dict(row) if row else None


@router.put("/images/{image_id}/review")
def put_review(image_id: str, body: ReviewBody, request: Request):
    db = request.app.state.db
    get_image_row(db, image_id)
    dims = dimensions()
    missing = [d for d in dims if d not in body.values]
    extra = [k for k in body.values if k not in dims]
    if missing or extra:
        raise ApiError("VALIDATION_ERROR", "`values` must contain exactly the six dimensions.",
                       {"missing": missing, "unknown": extra})
    invalid = {d: v for d, v in body.values.items() if not is_valid(d, v)}
    if invalid:
        raise ApiError("VALIDATION_ERROR", "Some values are not valid classes.", {"invalid": invalid})
    predicted: dict[str, str | None] = {}
    if body.prediction_id:
        pred = db.one("SELECT * FROM predictions WHERE prediction_id = ? AND image_id = ?",
                      (body.prediction_id, image_id))
        if pred is None:
            raise ApiError("NOT_FOUND", "Prediction not found for this image.", {"prediction_id": body.prediction_id})
        predicted = {d: v["label"] for d, v in json.loads(pred["dimensions_json"]).items()}
        model_id = pred["model_id"]
    else:
        model_id = body.model_id
    changed = [d for d in dims if predicted.get(d) != body.values[d]]
    status = "corrected" if changed else "reviewed"
    review_id = f"rev_{uuid.uuid4()}"
    created = now_iso()
    db.execute("INSERT INTO reviews (review_id, image_id, prediction_id, model_id, values_json, changed_json, "
               "status, note, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
               (review_id, image_id, body.prediction_id, model_id,
                json.dumps({d: body.values[d] for d in dims}), json.dumps(changed), status, body.note, created))
    return {"review_id": review_id, "image_id": image_id, "status": status, "changed_dimensions": changed,
            "created_at": created}


@router.get("/images/{image_id}/reviews")
def list_reviews(image_id: str, request: Request):
    db = request.app.state.db
    get_image_row(db, image_id)
    rows = db.all("SELECT * FROM reviews WHERE image_id = ? ORDER BY created_at, rowid", (image_id,))
    return {"reviews": [review_dict(r) for r in rows]}
