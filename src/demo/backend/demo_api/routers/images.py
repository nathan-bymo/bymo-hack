from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from .. import images_io
from ..db import now_iso
from ..errors import ApiError

router = APIRouter()
SOURCE_KINDS = {"own_photo", "train_coverage", "other"}
MIME_BY_EXT = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def image_dict(row: dict, duplicate: bool = False) -> dict:
    image_id = row["image_id"]
    return {
        "image_id": image_id, "session_id": row["session_id"], "original_filename": row["original_filename"],
        "sha256": row["sha256"], "format": row["format"], "mime": row["mime"], "width": row["width"],
        "height": row["height"], "bytes": row["bytes"], "has_alpha": bool(row["has_alpha"]),
        "exif_transposed": bool(row["exif_transposed"]), "source_kind": row["source_kind"],
        "context": json.loads(row["context_json"] or "{}"), "context_hash": row["context_hash"],
        "duplicate": duplicate, "created_at": row["created_at"],
        "file_url": f"/api/images/{image_id}/file", "thumbnail_url": f"/api/images/{image_id}/thumbnail",
    }


def get_image_row(db, image_id: str) -> dict:
    row = db.one("SELECT * FROM images WHERE image_id = ?", (image_id,))
    if row is None:
        raise ApiError("NOT_FOUND", "Image not found.", {"image_id": image_id})
    return row


def _json_list(raw: str | None, name: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ApiError("VALIDATION_ERROR", f"`{name}` must be valid JSON.") from None


@router.post("/images", status_code=201)
def upload_images(
    request: Request,
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
    source_kind: str = Form("own_photo"),
    source_kinds: str | None = Form(None),
    context: str | None = Form(None),
):
    settings, db, storage = request.app.state.settings, request.app.state.db, request.app.state.storage
    if db.one("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)) is None:
        raise ApiError("NOT_FOUND", "Session not found.", {"session_id": session_id})
    if len(files) > settings.max_files_per_request:
        raise ApiError("FILE_TOO_LARGE", f"At most {settings.max_files_per_request} files per request.")
    if source_kind not in SOURCE_KINDS:
        raise ApiError("VALIDATION_ERROR", "Invalid `source_kind`.", {"allowed": sorted(SOURCE_KINDS)})
    kinds = _json_list(source_kinds, "source_kinds") or []
    contexts = _json_list(context, "context")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    guard = getattr(request.app.state, "guard", None)

    accepted, rejected = [], []
    for index, upload in enumerate(files):
        name = upload.filename or f"file-{index}"
        try:
            data = upload.file.read(max_bytes + 1)
            item = images_io.ingest(data, upload.content_type, max_bytes, settings.max_pixels)
            if guard is not None:
                guard.check(item.sha256)
            kind = kinds[index] if index < len(kinds) else source_kind
            if kind not in SOURCE_KINDS:
                raise ApiError("VALIDATION_ERROR", "Invalid `source_kind`.", {"allowed": sorted(SOURCE_KINDS)})
            raw_ctx = contexts[index] if isinstance(contexts, list) and index < len(contexts) else (
                contexts if isinstance(contexts, dict) else None)
            ctx = images_io.clean_context(raw_ctx)
            existing = db.one("SELECT * FROM images WHERE session_id = ? AND sha256 = ?", (session_id, item.sha256))
            if existing:
                accepted.append(image_dict(existing, duplicate=True))
                continue
            path = storage.upload_path(item.sha256, item.ext)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(item.data)
            thumb = storage.thumb_path(item.sha256)
            if not thumb.exists():
                thumb.write_bytes(images_io.make_thumbnail(item.data))
            image_id = f"img-demo-{uuid.uuid4()}"
            db.execute(
                "INSERT INTO images (image_id, session_id, original_filename, sha256, mime, format, width, height, "
                "bytes, has_alpha, exif_transposed, stored_path, thumb_path, source_kind, context_json, "
                "context_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (image_id, session_id, Path(name).name, item.sha256, item.mime, item.format, item.width,
                 item.height, len(item.data), int(item.has_alpha), int(item.exif_transposed), str(path),
                 str(thumb), kind, json.dumps(ctx, ensure_ascii=False), images_io.context_hash(ctx), now_iso()))
            accepted.append(image_dict(get_image_row(db, image_id)))
        except ApiError as error:
            rejected.append({"original_filename": Path(name).name,
                             "error": {"code": error.code, "message": error.message, "details": error.details}})
    if rejected and not accepted:
        # Nessun file accettato: la richiesta fallisce con lo stato del primo rifiuto (413/415/422).
        first = rejected[0]["error"]
        raise ApiError(first["code"], first["message"], {**first["details"], "rejected": rejected})
    return {"images": accepted, "rejected": rejected}


@router.get("/images")
def list_images(request: Request, session_id: str):
    db = request.app.state.db
    rows = db.all("SELECT * FROM images WHERE session_id = ? ORDER BY created_at, rowid", (session_id,))
    return {"images": [image_dict(r) for r in rows]}


@router.get("/images/{image_id}")
def get_image(image_id: str, request: Request):
    db = request.app.state.db
    out = image_dict(get_image_row(db, image_id))
    out["predictions"], out["review"] = [], None
    return out


def _file_response(path: str | Path, media_type: str) -> FileResponse:
    if not Path(path).exists():
        raise ApiError("NOT_FOUND", "Image file not found on disk.")
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/images/{image_id}/file")
def image_file(image_id: str, request: Request):
    row = get_image_row(request.app.state.db, image_id)
    ext = Path(row["stored_path"]).suffix.lstrip(".")
    return _file_response(row["stored_path"], MIME_BY_EXT.get(ext, "application/octet-stream"))


@router.get("/images/{image_id}/thumbnail")
def image_thumbnail(image_id: str, request: Request):
    row = get_image_row(request.app.state.db, image_id)
    return _file_response(row["thumb_path"], "image/jpeg")


class ContextBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_title: str | None = None
    product_type: str | None = None
    brand: str | None = None
    category: str | None = None
    subcategory: str | None = None


@router.put("/images/{image_id}/context")
def put_context(image_id: str, body: ContextBody, request: Request):
    db = request.app.state.db
    get_image_row(db, image_id)
    ctx = images_io.clean_context(body.model_dump())
    db.execute("UPDATE images SET context_json = ?, context_hash = ? WHERE image_id = ?",
               (json.dumps(ctx, ensure_ascii=False), images_io.context_hash(ctx), image_id))
    return image_dict(get_image_row(db, image_id))
