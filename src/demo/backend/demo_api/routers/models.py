from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..errors import ApiError

router = APIRouter()


def registry_of(request: Request):
    return request.app.state.registry


@router.get("/models")
def list_models(request: Request):
    reg = registry_of(request)
    return {"models": [reg.summary(e) for e in reg.models.values()],
            "reference_line": reg.defaults.get("reference_line"),
            "kaggle_references": reg.defaults.get("kaggle_references", []),
            "throughput_target_images_per_second": reg.defaults.get("throughput_target_images_per_second")}


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    reg = registry_of(request)
    entry = reg.get(model_id)
    if entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
    return reg.detail(entry)


# ------------------------------------------------------------------------------------------ caricamento e download
import threading  # noqa: E402
import uuid  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from ..db import now_iso  # noqa: E402
from ..registry import _contains_placeholder  # noqa: E402


class ConfirmBody(BaseModel):
    confirm: bool = False


def _entry(request: Request, model_id: str):
    entry = request.app.state.registry.get(model_id)
    if entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
    return entry


def _worker_entry(request: Request, model_id: str):
    entry = _entry(request, model_id)
    status, reason = request.app.state.registry.live_status(entry)
    if status == "unavailable":
        raise ApiError("MODEL_UNAVAILABLE", "This model is not available.", {"model_id": model_id, "reason": reason})
    return entry


@router.post("/models/{model_id}/load")
def load_model(model_id: str, request: Request):
    entry = _entry(request, model_id)
    if entry.raw["family"] == "zeroshot_generative":
        _worker_entry(request, model_id)
        return {"model_id": model_id, "state": "ready"}   # nessun caricamento: modello remoto
    _worker_entry(request, model_id)
    db, proxy = request.app.state.db, request.app.state.worker_adapter(entry)
    job_id = f"job_{uuid.uuid4()}"
    db.execute("INSERT INTO jobs (job_id, kind, model_id, status, total, done, started_at) VALUES (?,?,?,?,?,?,?)",
               (job_id, "load", model_id, "running", 1, 0, now_iso()))

    def run():
        try:
            proxy.ensure_loaded()
            db.execute("UPDATE jobs SET status='done', done=1, finished_at=? WHERE job_id=?", (now_iso(), job_id))
        except ApiError as error:
            db.execute("UPDATE jobs SET status='error', finished_at=?, error_json=? WHERE job_id=?",
                       (now_iso(), json.dumps({"code": error.code, "message": error.message}), job_id))

    threading.Thread(target=run, daemon=True).start()
    return JSONResponse({"job_id": job_id, "model_id": model_id, "state": "loading"}, status_code=202)


@router.post("/models/{model_id}/unload")
def unload_model(model_id: str, request: Request):
    entry = _entry(request, model_id)
    if entry.raw["family"] == "zeroshot_generative":
        return {"model_id": model_id, "state": "not_loaded"}
    if request.app.state.jobs.is_busy(model_id):
        raise ApiError("MODEL_BUSY", "A job is using this model.", {"model_id": model_id})
    _worker_entry(request, model_id)
    request.app.state.worker_adapter(entry).unload()
    return {"model_id": model_id, "state": "not_loaded"}


@router.post("/models/{model_id}/download")
def download_model(model_id: str, body: ConfirmBody, request: Request):
    entry = _entry(request, model_id)
    weights = entry.raw.get("weights")
    if not weights or _contains_placeholder(weights):
        raise ApiError("MODEL_UNAVAILABLE", "No downloadable weights are configured for this model.",
                       {"model_id": model_id})
    family = entry.raw["family"]
    if family not in request.app.state.settings.worker_urls:
        raise ApiError("MODEL_UNAVAILABLE", "No GPU worker is connected: weights are downloaded on the GPU host.",
                       {"model_id": model_id, "reason": "no GPU worker"})
    gb = weights.get("size_gb")
    if not body.confirm:
        raise ApiError("CONFIRMATION_REQUIRED", "Downloading weights needs explicit confirmation.",
                       {"download_gb": gb, "base_gb": request.app.state.registry.defaults.get("base_model", {}).get("size_gb")})
    raise ApiError("MODEL_UNAVAILABLE", "Run the download on the GPU host: "
                   "python src/demo/scripts/fetch_weights.py --model <id> --yes", {"model_id": model_id})
