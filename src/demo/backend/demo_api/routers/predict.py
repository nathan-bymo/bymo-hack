from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import predictions as P
from ..adapters.zeroshot import ZeroShotGenerativeAdapter
from ..errors import ApiError

router = APIRouter()


class PredictBody(BaseModel):
    model_id: str
    image_ids: list[str] = Field(min_length=1, max_length=500)
    use_context: bool = True
    mode: str = Field(default="live", pattern="^(live|recorded)$")
    batch_size: int | None = Field(default=None, ge=1, le=64)


def get_adapter(request: Request, model_id: str, mode: str):
    registry = request.app.state.registry
    entry = registry.get(model_id)
    if entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
    if mode == "recorded":
        return request.app.state.recorded_adapter(entry)  # RecordedAdapter (M1.7)
    status, reason = registry.live_status(entry)
    if status == "unavailable":
        raise ApiError("MODEL_UNAVAILABLE", "This model is not available.", {"model_id": model_id, "reason": reason})
    if entry.raw["family"] == "zeroshot_generative":
        cache = request.app.state.adapters
        if model_id not in cache:
            cache[model_id] = ZeroShotGenerativeAdapter(entry.raw, parallel=request.app.state.zero_shot_parallel)
        return cache[model_id]
    proxy = request.app.state.worker_adapter(entry)  # WorkerProxy (M3.1)
    return proxy


@router.post("/predict", status_code=202)
def predict(body: PredictBody, request: Request):
    db = request.app.state.db
    adapter = get_adapter(request, body.model_id, body.mode)
    images = []
    for image_id in body.image_ids:
        row = db.one("SELECT * FROM images WHERE image_id = ?", (image_id,))
        if row is None:
            raise ApiError("NOT_FOUND", "Image not found.", {"image_id": image_id})
        images.append(row)
    job, cached, queued = request.app.state.jobs.submit_predict(
        adapter=adapter, model_id=body.model_id, images=images, use_context=body.use_context, mode=body.mode,
        batch_size=body.batch_size or 0)
    payload = {"job_id": job.job_id, "model_id": body.model_id, "total": job.total, "cached": cached,
               "queued": queued, "events_url": f"/api/jobs/{job.job_id}/events"}
    from fastapi.responses import JSONResponse

    return JSONResponse(payload, status_code=202 if queued else 200)


def _job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise ApiError("NOT_FOUND", "Job not found.", {"job_id": job_id})
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    return _job(request, job_id).snapshot()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    _job(request, job_id)
    job = request.app.state.jobs.cancel(job_id)
    return {"status": "canceled" if job.cancel.is_set() else job.status}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    job = _job(request, job_id)

    async def stream():
        index, idle = 0, 0.0
        while True:
            with job.lock:
                pending = job.events[index:]
            for name, data in pending:
                index += 1
                yield f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if name == "done":
                    return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.05)
            idle += 0.05
            if idle >= 15:
                idle = 0.0
                yield ": ping\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/predictions")
def list_predictions(request: Request, session_id: str | None = None, model_id: str | None = None,
                     image_id: str | None = None):
    db = request.app.state.db
    sql, params = ("SELECT p.* FROM predictions p JOIN images i ON i.image_id = p.image_id WHERE 1=1", [])
    if session_id:
        sql += " AND i.session_id = ?"
        params.append(session_id)
    if model_id:
        sql += " AND p.model_id = ?"
        params.append(model_id)
    if image_id:
        sql += " AND p.image_id = ?"
        params.append(image_id)
    rows = db.all(sql + " ORDER BY p.created_at, p.rowid", params)
    return {"predictions": [P.to_result(r) for r in rows]}
