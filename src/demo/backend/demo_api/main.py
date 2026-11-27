"""Applicazione FastAPI della demo (porta 8010)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import errors
from .settings import VERSION, Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from . import db as db_module
        from .storage import Storage

        app.state.storage = Storage(settings.data_dir)
        app.state.storage.ensure()
        handler = logging.FileHandler(settings.data_dir / "logs" / "api.log")
        handler.setLevel(logging.INFO)
        logging.getLogger("demo_api").addHandler(handler)
        app.state.db = db_module.Database(settings.data_dir / "demo.sqlite")
        app.state.db.init()
        from .guard import Guard

        app.state.guard = Guard.load(app.state.storage.guard_index)
        from .registry import Registry

        app.state.registry = Registry(settings, app.state.db)  # fallisce all'avvio se models.yaml non è valido
        yield
        app.state.db.close()

    app = FastAPI(title="Demo Image classification (AI)", version=VERSION, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings = settings
    errors.install(app)

    from .routers import health, images, models, sessions, taxonomy

    for module in (health, taxonomy, sessions, images, models):
        app.include_router(module.router, prefix="/api")

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
    async def unknown_api(path: str):
        raise errors.ApiError("NOT_FOUND", "Unknown API path.", {"path": "/api/" + path})

    dist = Path(settings.frontend_dist)
    if (dist / "index.html").exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


def __getattr__(name: str):
    # `uvicorn demo_api.main:app` crea l'app alla prima richiesta dell'attributo.
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(name)
