from __future__ import annotations

from fastapi import APIRouter, Request

from ..settings import VERSION, git_commit

router = APIRouter()


@router.get("/health")
def health(request: Request):
    settings = request.app.state.settings
    workers = [{"family": family, "url": url, "reachable": False} for family, url in settings.worker_urls.items()]
    return {"status": "ok", "version": VERSION, "git_commit": git_commit(settings.repo_root),
            "gpu_local": False, "workers": workers,
            "guard": {"loaded": request.app.state.guard is not None,
                      "entries": len(request.app.state.guard) if request.app.state.guard else 0}}
