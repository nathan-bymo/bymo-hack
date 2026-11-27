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


def _no_worker(entry):
    raise errors.ApiError("MODEL_UNAVAILABLE", "No GPU worker is available for this model.", {"model_id": entry.id})


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
        from .jobs import JobManager

        app.state.jobs = JobManager(app.state.db)
        from .adapters import recorded as recorded_module

        recorded_module.scan(app.state.db, app.state.storage.recorded_dir)
        from .metrics.reported import import_reported

        try:
            import_reported(app.state.db, {k: v.raw for k, v in app.state.registry.models.items()}, settings.repo_root)
        except Exception:  # noqa: BLE001 - la demo parte comunque, senza metriche riportate
            logging.getLogger("demo_api").exception("import delle metriche riportate non riuscito")
        app.state.recorded_adapter = lambda entry: recorded_module.RecordedAdapter(entry.id, app.state.db)
        yield
        app.state.db.close()

    app = FastAPI(title="Demo Image classification (AI)", version=VERSION, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.adapters = {}
    app.state.zero_shot_parallel = 8
    app.state.worker_adapter = _no_worker
    errors.install(app)

    from .routers import (
        health, images, metrics, models, predict, recorded, reviews, sessions, settings as settings_router, taxonomy,
    )

    for module in (health, taxonomy, sessions, images, models, predict, recorded, reviews, settings_router, metrics):
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
