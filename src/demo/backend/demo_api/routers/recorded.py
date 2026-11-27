from __future__ import annotations

from fastapi import APIRouter, Request

from ..adapters.recorded import scan

router = APIRouter()


@router.post("/recorded/rescan")
def rescan(request: Request):
    return {"runs": scan(request.app.state.db, request.app.state.storage.recorded_dir)}
