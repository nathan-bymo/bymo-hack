"""Foto di esempio per la presentazione: file in `data/demo/samples/` con le risposte già registrate dei modelli.

Servono a mostrare subito una risposta senza aspettare l'inferenza. Le foto non entrano in git (cartella `data/`).
Il caricamento nella pagina passa dal normale upload, quindi vale anche la guardia su dev e final.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from .. import images_io
from ..adapters.recorded import models_recorded_for
from ..errors import ApiError

router = APIRouter()
MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_sha_cache: dict[tuple[str, float], str | None] = {}


def _title(stem: str) -> str:
    text = re.sub(r"^\d+[\s_.-]*", "", stem).replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:] if text else stem


def _sha(path: Path, request: Request) -> str | None:
    key = (str(path), path.stat().st_mtime)
    if key not in _sha_cache:
        settings = request.app.state.settings
        try:
            _sha_cache[key] = images_io.ingest(path.read_bytes(), MIME_BY_SUFFIX[path.suffix.lower()],
                                               settings.max_upload_mb * 1024 * 1024, settings.max_pixels).sha256
        except ApiError:
            _sha_cache[key] = None
    return _sha_cache[key]


def _files(request: Request) -> list[Path]:
    folder = request.app.state.storage.samples_dir
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MIME_BY_SUFFIX) if folder.exists() else []


@router.get("/samples")
def list_samples(request: Request):
    db = request.app.state.db
    out = []
    for path in _files(request):
        sha = _sha(path, request)
        if sha is None:
            continue
        out.append({"sample_id": path.stem, "title": _title(path.stem), "filename": path.name,
                    "file_url": f"/api/samples/{path.stem}/file", "sha256": sha,
                    "recorded_models": models_recorded_for(db, sha)})
    return {"samples": out}


@router.get("/samples/{sample_id}/file")
def sample_file(sample_id: str, request: Request):
    path = next((p for p in _files(request) if p.stem == sample_id), None)   # solo file dell'elenco: nessun percorso libero
    if path is None:
        raise ApiError("NOT_FOUND", "Sample photo not found.", {"sample_id": sample_id})
    return FileResponse(path, media_type=MIME_BY_SUFFIX[path.suffix.lower()], headers={"Cache-Control": "private, max-age=3600"})
