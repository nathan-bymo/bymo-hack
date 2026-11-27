"""Metriche (C.7): numeri con set, fonte, dati di training e badge. `dev` e `final` non si valutano mai."""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..errors import ApiError
from ..metrics.badges import SEEN_TOOLTIP, badges
from ..taxonomy import dimensions

router = APIRouter()
SETS = ("coverage", "base", "prompt-test", "prompt-dev-600", "validation_450")


def _check_set(name: str) -> str:
    if name in ("dev", "final"):
        raise ApiError("VALIDATION_ERROR", "`dev` has no labels and `final` is never evaluated: not a valid set.",
                       {"set": name, "allowed": list(SETS)})
    if name not in SETS:
        raise ApiError("VALIDATION_ERROR", "Unknown set.", {"set": name, "allowed": list(SETS)})
    return name


def result_dict(row: dict) -> dict:
    per_class = json.loads(row["per_class_json"]) if row["per_class_json"] else None
    notes = json.loads(row["notes_json"]) if row["notes_json"] else []
    return {
        "eval_id": row["eval_id"], "validity": row["validity"], "provenance": row["provenance"],
        "badges": badges(row["validity"], row["provenance"]),
        "badge_tooltip": SEEN_TOOLTIP if row["validity"] == "seen_in_training" else None,
        "source": row["source"], "checkpoint": row["checkpoint"], "trained_on": row["trained_on"],
        "macro_f1": row["macro_f1"], "weighted_f1": row["weighted_f1"],
        "accuracy": json.loads(row["accuracy_json"]) if row["accuracy_json"] else None,
        "per_dimension": json.loads(row["per_dimension_json"]) if row["per_dimension_json"] else {},
        "per_class": per_class["data"] if per_class else None,
        "per_class_source": per_class["source"] if per_class else None,
        "confusion": json.loads(row["confusion_json"]) if row["confusion_json"] else None,
        "confidence_calibration": json.loads(row["calibration_json"]) if row["calibration_json"] else None,
        "computed_at": row["computed_at"], "notes": notes,
    }


def _kaggle(raw: dict) -> dict:
    k = raw.get("kaggle_dev") or {}
    return {"score": k.get("score"), "badge": "Kaggle dev (public)", "submitted_at": k.get("submitted_at"),
            "source": k.get("source"), "note": k.get("note")}


def block(request: Request, model_id: str, set_name: str, include_invalid: bool) -> dict:
    registry, db = request.app.state.registry, request.app.state.db
    entry = registry.get(model_id)
    if entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
    rows = db.all("SELECT * FROM eval_results WHERE model_id = ? AND set_name = ? ORDER BY validity, rowid",
                  (model_id, set_name))
    valid = [r for r in rows if r["validity"] != "seen_in_training"]
    shown = rows if include_invalid else valid
    return {
        "model_id": model_id, "set": set_name, "display_name": entry.raw["display_name"],
        "results": [result_dict(r) for r in shown], "excluded_invalid": 0 if include_invalid else len(rows) - len(valid),
        "contamination_note": entry.raw.get("contamination_note"),
        "training": entry.raw.get("training"), "kaggle_dev": _kaggle(entry.raw),
    }


@router.get("/metrics/compare")
def compare(request: Request, models: str, set: str = "coverage", include_invalid: bool = False):
    name = _check_set(set)
    ids = [m for m in models.split(",") if m]
    return {"set": name, "models": {m: block(request, m, name, include_invalid) for m in ids}}


@router.get("/metrics/{model_id}")
def model_metrics(model_id: str, request: Request, set: str = "coverage", include_invalid: bool = False):
    return block(request, model_id, _check_set(set), include_invalid)


# ---------------------------------------------------------------------------------------------- velocità e costo
def cost_per_1k(usd_per_hour: float | None, images_per_second: float | None) -> float | None:
    if not usd_per_hour or not images_per_second:
        return None
    return usd_per_hour / (images_per_second * 3600) * 1000


def _price_table(registry) -> dict[str, float]:
    cost = registry.defaults.get("cost", {})
    table = {}
    for item in [cost.get("default"), *cost.get("alternatives", [])]:
        if item:
            table[item["gpu_label"]] = item["usd_per_hour"]
    return table


@router.get("/throughput")
def throughput(request: Request, model_id: str | None = None):
    from .settings import read_settings

    registry, db = request.app.state.registry, request.app.state.db
    settings = read_settings(request)
    prices = _price_table(registry)
    target = registry.defaults.get("throughput_target_images_per_second")
    entry = registry.get(model_id) if model_id else None
    if model_id and entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})

    sql = "SELECT * FROM jobs WHERE kind = 'predict' AND status = 'done' AND images_per_second IS NOT NULL"
    rows = db.all(sql + (" AND model_id = ?" if model_id else "") + " ORDER BY finished_at DESC", [model_id] if model_id else [])
    measured = []
    for r in rows:
        recorded = db.one("SELECT 1 FROM predictions WHERE job_id = ? AND mode = 'recorded' LIMIT 1", (r["job_id"],)) is not None
        gateway = (r["gpu_name"] or "").startswith("gateway")
        note, cost = None, None
        if gateway:
            note = "API cost not public"
        else:
            cost = cost_per_1k(settings["usd_per_hour"], r["images_per_second"])
            note = (f"uses the price set in the cost panel ({settings['gpu_label']}); "
                    "valid only if it is the GPU the speed was measured on")
        measured.append({"job_id": r["job_id"], "model_id": r["model_id"], "images": r["done"],
                         "images_per_second": r["images_per_second"],
                         "ms_per_image": 1000 / r["images_per_second"] if r["images_per_second"] else None,
                         "gpu_name": r["gpu_name"], "batch_size": r["batch_size"], "measured_at": r["finished_at"],
                         "mode": "recorded" if recorded else "live", "cost_per_1k_images_usd": cost, "cost_note": note})

    reported = []
    for e in ([entry] if entry else registry.models.values()):
        for m in e.raw.get("measured", []):
            if not str(m.get("what", "")).startswith("img/s"):
                continue
            gpu = m.get("gpu_label")
            price = prices.get(gpu) if gpu else None
            eligible = m.get("cost_eligible", False)
            reported.append({"model_id": e.id, "value": m["value"], "what": m["what"], "conditions": m.get("conditions"),
                             "source": m["source"], "gpu_label": gpu, "usd_per_hour": price if eligible else None,
                             "cost_per_1k_images_usd": cost_per_1k(price, m["value"]) if eligible else None,
                             "cost_basis": f"{gpu} at {price} $/h" if eligible and price else None})
    headline = next((r for r in reported if r["cost_per_1k_images_usd"] is not None), None)
    live_gpu = next((m for m in measured if m["cost_per_1k_images_usd"] is not None), None)
    pick = headline or live_gpu
    return {
        "model_id": model_id, "target_images_per_second": target,
        "measured_in_demo": measured, "reported": reported,
        "cost_per_1k_images_usd": pick["cost_per_1k_images_usd"] if pick else None,
        "cost_basis": ({"gpu_label": headline["gpu_label"], "usd_per_hour": headline["usd_per_hour"], "source": headline["source"]}
                       if headline else ({"gpu_label": settings["gpu_label"], "usd_per_hour": settings["usd_per_hour"],
                                          "source": "cost panel setting"} if live_gpu else None)),
        "cost_note": None if pick else "No measured speed with a known GPU: cost not computed (gateway API cost is not public).",
        "settings": {"gpu_label": settings["gpu_label"], "usd_per_hour": settings["usd_per_hour"]},
        "gpu_prices": prices,
    }


# ---------------------------------------------------------------------------------------------- export
@router.get("/export.csv")
def export_csv(request: Request, session_id: str, model_id: str | None = None, include_reviews: bool = True,
               include_filenames: bool = False):
    db = request.app.state.db
    if db.one("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)) is None:
        raise ApiError("NOT_FOUND", "Session not found.", {"session_id": session_id})
    images = db.all("SELECT * FROM images WHERE session_id = ? ORDER BY created_at, rowid", (session_id,))
    dims = dimensions()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["image_id", *(["original_filename"] if include_filenames else []), *dims, "status",
                     "min_confidence", "model_id", "mode"])
    has_coverage = any(i["source_kind"] == "train_coverage" for i in images)
    for image in images:
        pred = db.one("SELECT * FROM predictions WHERE image_id = ?" + (" AND model_id = ?" if model_id else "")
                      + " ORDER BY created_at DESC, rowid DESC", [image["image_id"], *([model_id] if model_id else [])])
        review = db.one("SELECT * FROM reviews WHERE image_id = ? ORDER BY created_at DESC, rowid DESC", (image["image_id"],)) \
            if include_reviews else None
        if pred is None and review is None:
            writer.writerow([image["image_id"], *([image["original_filename"]] if include_filenames else []),
                             *[""] * len(dims), "no_prediction", "", model_id or "", ""])
            continue
        values = {d: "" for d in dims}
        confs = []
        if pred:
            predicted = json.loads(pred["dimensions_json"])
            for d in dims:
                values[d] = predicted[d]["label"] or ""
                if predicted[d]["confidence"] is not None:
                    confs.append(predicted[d]["confidence"])
        status = "ai_suggested"
        if review and (not pred or review["prediction_id"] == pred["prediction_id"]):
            values = json.loads(review["values_json"])
            status = review["status"]
        writer.writerow([image["image_id"], *([image["original_filename"]] if include_filenames else []),
                         *[values[d] for d in dims], status, f"{min(confs):.4f}" if confs else "",
                         (pred or {}).get("model_id", review["model_id"] if review else ""),
                         (pred or {}).get("mode", "")])
    headers = {"Content-Disposition": f'attachment; filename="demo-export-{session_id[4:12]}.csv"'}
    if has_coverage:
        headers["X-Demo-Warning"] = "The session contains competition (train/coverage) images: do not share this CSV."
    return Response(out.getvalue(), media_type="text/csv", headers=headers)
