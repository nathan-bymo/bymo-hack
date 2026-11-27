"""Schema di errore unico (allegato C, C.1). I messaggi sono in inglese e non contengono segreti né percorsi."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("demo_api")

STATUS = {
    "VALIDATION_ERROR": 422, "NOT_FOUND": 404, "MODEL_NOT_FOUND": 404, "UNSUPPORTED_MEDIA_TYPE": 415,
    "FILE_TOO_LARGE": 413, "IMAGE_TOO_LARGE": 413, "CORRUPT_IMAGE": 422, "FORBIDDEN_IMAGE": 422,
    "MODEL_UNAVAILABLE": 409, "MODEL_LOADING": 409, "MODEL_BUSY": 409, "CONFIRMATION_REQUIRED": 409,
    "GPU_OOM": 507, "UPSTREAM_ERROR": 502, "TIMEOUT": 504, "RECORDED_MISSING": 409, "INTERNAL_ERROR": 500,
}


class ApiError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}

    @property
    def status_code(self) -> int:
        return STATUS[self.code]

    def body(self, request_id: str | None = None) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details,
                          "request_id": request_id}}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _json(error: ApiError, request: Request) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(error.body(rid), status_code=error.status_code, headers={"X-Request-Id": rid})


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _json(exc, request)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        fields = [{"loc": [str(p) for p in e["loc"]], "msg": e["msg"]} for e in exc.errors()]
        return _json(ApiError("VALIDATION_ERROR", "The request is not valid.", {"fields": fields}), request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        mapping = {404: ("NOT_FOUND", "Not found."), 413: ("FILE_TOO_LARGE", "The request is too large."),
                   415: ("UNSUPPORTED_MEDIA_TYPE", "Unsupported media type."),
                   422: ("VALIDATION_ERROR", "The request is not valid.")}
        code, message = mapping.get(exc.status_code, ("INTERNAL_ERROR", "Unexpected error."))
        if exc.status_code == 405:
            code, message = "VALIDATION_ERROR", "Method not allowed for this path."
            rid = _request_id(request)
            return JSONResponse(ApiError(code, message).body(rid), status_code=405, headers={"X-Request-Id": rid})
        return _json(ApiError(code, message), request)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error", exc_info=exc)  # i dettagli restano nel log locale
        return _json(ApiError("INTERNAL_ERROR", "Unexpected error. Details are in the local log."), request)
