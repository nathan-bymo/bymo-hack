from __future__ import annotations

from fastapi import APIRouter, Request

from ..errors import ApiError

router = APIRouter()


def registry_of(request: Request):
    return request.app.state.registry


@router.get("/models")
def list_models(request: Request):
    reg = registry_of(request)
    return {"models": [reg.summary(e) for e in reg.models.values()],
            "reference_line": reg.defaults.get("reference_line"),
            "throughput_target_images_per_second": reg.defaults.get("throughput_target_images_per_second")}


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    reg = registry_of(request)
    entry = reg.get(model_id)
    if entry is None:
        raise ApiError("MODEL_NOT_FOUND", "Model not found.", {"model_id": model_id})
    return reg.detail(entry)
