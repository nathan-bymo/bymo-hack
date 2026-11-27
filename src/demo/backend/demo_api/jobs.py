"""Coda dei job di predizione: thread per job, eventi in memoria per l'SSE, stato nel database."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import predictions as P
from .adapters.base import BatchInfo, StoredImage
from .db import Database, now_iso
from .errors import ApiError


@dataclass
class Job:
    job_id: str
    model_id: str
    total: int
    kind: str = "predict"
    status: str = "running"
    done: int = 0
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    results: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    batch: dict | None = None
    error: dict | None = None
    events: list[tuple[str, dict]] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, name: str, data: dict) -> None:
        with self.lock:
            self.events.append((name, data))

    def snapshot(self) -> dict:
        with self.lock:
            return {"job_id": self.job_id, "kind": self.kind, "model_id": self.model_id, "status": self.status,
                    "done": self.done, "total": self.total, "started_at": self.started_at,
                    "finished_at": self.finished_at, "results": list(self.results), "errors": list(self.errors),
                    "batch": self.batch, "error": self.error}


class JobManager:
    def __init__(self, db: Database):
        self.db = db
        self.jobs: dict[str, Job] = {}
        self.active_models: dict[str, int] = {}   # per non scaricare modelli con job attivi (M3)
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def is_busy(self, model_id: str) -> bool:
        return self.active_models.get(model_id, 0) > 0

    def submit_predict(self, *, adapter: Any, model_id: str, images: list[dict], use_context: bool, mode: str,
                       batch_size: int, gpu_hint: str | None = None) -> tuple[Job, int, int]:
        """Crea il job; le immagini già in cache sono restituite subito. Ritorna (job, cached, queued)."""
        job = Job(f"job_{uuid.uuid4()}", model_id, total=len(images))
        todo: list[dict] = []
        cached = 0
        for image in images:
            row = P.find_cached(self.db, image, model_id, adapter.model_revision, adapter.prompt_version,
                                use_context, mode)
            if row:
                cached += 1
                job.done += 1
                result = P.to_result(row, cached=True, db=self.db)
                job.results.append(result)
                job.emit("result", result)
            else:
                todo.append(image)
        self.jobs[job.job_id] = job
        self.db.execute("INSERT INTO jobs (job_id, kind, model_id, status, total, done, started_at) "
                        "VALUES (?,?,?,?,?,?,?)", (job.job_id, "predict", model_id, "running", job.total,
                                                   job.done, job.started_at))
        if not todo:
            self._finish(job, "done", None)
            return job, cached, 0
        with self._lock:
            self.active_models[model_id] = self.active_models.get(model_id, 0) + 1
        thread = threading.Thread(target=self._run, daemon=True, name=job.job_id,
                                  args=(job, adapter, todo, use_context, mode, batch_size))
        thread.start()
        return job, cached, len(todo)

    # ------------------------------------------------------------------ esecuzione
    def _run(self, job: Job, adapter: Any, todo: list[dict], use_context: bool, mode: str, batch_size: int) -> None:
        by_id = {i["image_id"]: i for i in todo}
        stored = [StoredImage(i["image_id"], i["sha256"], _path(i), json.loads(i["context_json"] or "{}"),
                              i["context_hash"]) for i in todo]

        def on_result(prediction) -> None:
            row = P.store(self.db, by_id[prediction.image_id], job.model_id, adapter.model_revision,
                          adapter.prompt_version, use_context, mode, prediction, job.job_id,
                          getattr(prediction, "recorded_run_id", None))
            result = P.to_result(row, cached=False, db=self.db)
            with job.lock:
                job.done += 1
                job.results.append(result)
                done = job.done
            job.emit("result", result)
            job.emit("progress", {"done": done, "total": job.total, "images_per_second": None})
            self.db.execute("UPDATE jobs SET done = ? WHERE job_id = ?", (done, job.job_id))

        def on_error(image_id: str, error: Exception) -> None:
            api = error if isinstance(error, ApiError) else ApiError("INTERNAL_ERROR", "Unexpected error.")
            payload = {"image_id": image_id, "error": {"code": api.code, "message": api.message,
                                                       "details": api.details}}
            with job.lock:
                job.errors.append(payload)
            job.emit("error", payload)

        try:
            info: BatchInfo = adapter.predict(stored, use_context, batch_size, on_result, on_error,
                                              job.cancel.is_set)
            job.batch = info.to_json()
            status = "canceled" if job.cancel.is_set() else ("error" if info.images == 0 and job.errors else "done")
            self._finish(job, status, info)
        except ApiError as error:
            job.error = {"code": error.code, "message": error.message, "details": error.details}
            job.emit("error", {"image_id": None, "error": job.error})
            self._finish(job, "error", None)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("demo_api").exception("job %s failed", job.job_id)
            job.error = {"code": "INTERNAL_ERROR", "message": "Unexpected error. Details are in the local log.",
                         "details": {}}
            job.emit("error", {"image_id": None, "error": job.error})
            self._finish(job, "error", None)
        finally:
            with self._lock:
                self.active_models[job.model_id] = max(0, self.active_models.get(job.model_id, 1) - 1)

    def _finish(self, job: Job, status: str, info: BatchInfo | None) -> None:
        job.status = status
        job.finished_at = now_iso()
        if info is not None:
            job.batch = info.to_json()
        job.emit("done", {"status": status, "batch": job.batch})
        self.db.execute(
            "UPDATE jobs SET status=?, done=?, finished_at=?, seconds=?, images_per_second=?, gpu_name=?, "
            "batch_size=?, peak_memory_gb=?, error_json=? WHERE job_id=?",
            (status, job.done, job.finished_at, info.seconds if info else None,
             info.images_per_second if info else None, info.gpu_name if info else None,
             info.batch_size if info else None, info.peak_memory_gb if info else None,
             json.dumps(job.error) if job.error else None, job.job_id))

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status == "running":
            job.cancel.set()
        return job


def _path(image: dict):
    from pathlib import Path

    return Path(image["stored_path"])
