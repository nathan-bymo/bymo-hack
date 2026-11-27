from __future__ import annotations

import json

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter()


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_label: str | None = None
    usd_per_hour: float | None = Field(default=None, ge=0)
    presentation_mode: bool | None = None
    show_ground_truth: bool | None = None
    default_use_context: bool | None = None
    zero_shot_parallel_images: int | None = Field(default=None, ge=1, le=32)


def defaults(request: Request) -> dict:
    cost = request.app.state.registry.defaults.get("cost", {}).get("default", {})
    return {"gpu_label": cost.get("gpu_label", ""), "usd_per_hour": cost.get("usd_per_hour", 0.0),
            "presentation_mode": False, "show_ground_truth": False, "default_use_context": True,
            "zero_shot_parallel_images": 8}


def read_settings(request: Request) -> dict:
    out = defaults(request)
    for row in request.app.state.db.all("SELECT key, value_json FROM settings"):
        if row["key"] in out:
            out[row["key"]] = json.loads(row["value_json"])
    return out


@router.get("/settings")
def get_settings(request: Request):
    return read_settings(request)


@router.put("/settings")
def put_settings(body: SettingsBody, request: Request):
    db = request.app.state.db
    for key, value in body.model_dump(exclude_none=True).items():
        db.execute("INSERT INTO settings (key, value_json) VALUES (?, ?) "
                   "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json", (key, json.dumps(value)))
    current = read_settings(request)
    request.app.state.zero_shot_parallel = current["zero_shot_parallel_images"]
    return current
