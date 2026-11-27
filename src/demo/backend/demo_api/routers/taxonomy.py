from __future__ import annotations

from fastapi import APIRouter

from ..taxonomy import taxonomy_payload

router = APIRouter()


@router.get("/taxonomy")
def get_taxonomy():
    return taxonomy_payload()
