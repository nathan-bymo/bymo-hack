from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..db import now_iso
from ..errors import ApiError

router = APIRouter()


class SessionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)


def session_dict(db, session_id: str) -> dict | None:
    row = db.one(
        "SELECT s.session_id, s.name, s.created_at, s.notes, "
        "(SELECT COUNT(*) FROM images i WHERE i.session_id = s.session_id) AS image_count "
        "FROM sessions s WHERE s.session_id = ?", (session_id,))
    return row


@router.post("/sessions", status_code=201)
def create_session(body: SessionCreate, request: Request):
    db = request.app.state.db
    session_id = f"ses_{uuid.uuid4()}"
    db.execute("INSERT INTO sessions (session_id, name, created_at, notes) VALUES (?, ?, ?, ?)",
               (session_id, body.name.strip(), now_iso(), body.notes))
    return session_dict(db, session_id)


@router.get("/sessions")
def list_sessions(request: Request):
    db = request.app.state.db
    rows = db.all(
        "SELECT s.session_id, s.name, s.created_at, s.notes, "
        "(SELECT COUNT(*) FROM images i WHERE i.session_id = s.session_id) AS image_count "
        "FROM sessions s ORDER BY s.created_at DESC, s.rowid DESC")
    return {"sessions": rows}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    row = session_dict(request.app.state.db, session_id)
    if row is None:
        raise ApiError("NOT_FOUND", "Session not found.", {"session_id": session_id})
    return row
