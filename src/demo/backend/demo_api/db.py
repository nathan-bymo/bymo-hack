"""SQLite della demo (`data/demo/demo.sqlite`, WAL), sqlite3 della libreria standard, schema del piano §8."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, notes TEXT);
CREATE TABLE IF NOT EXISTS images (
  image_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(session_id),
  original_filename TEXT, sha256 TEXT NOT NULL, mime TEXT, format TEXT, width INTEGER, height INTEGER,
  bytes INTEGER, has_alpha INTEGER, exif_transposed INTEGER, stored_path TEXT, thumb_path TEXT,
  source_kind TEXT CHECK (source_kind IN ('own_photo','train_coverage','other')),
  context_json TEXT NOT NULL DEFAULT '{}', context_hash TEXT NOT NULL, created_at TEXT,
  UNIQUE (session_id, sha256));
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, kind TEXT CHECK (kind IN ('predict','load','download','evaluate')), model_id TEXT,
  status TEXT, total INTEGER, done INTEGER, started_at TEXT, finished_at TEXT, seconds REAL,
  images_per_second REAL, gpu_name TEXT, batch_size INTEGER, peak_memory_gb REAL, error_json TEXT);
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id TEXT PRIMARY KEY, image_id TEXT REFERENCES images(image_id), image_sha256 TEXT, model_id TEXT,
  model_revision TEXT, prompt_version TEXT, use_context INTEGER, context_hash TEXT,
  mode TEXT CHECK (mode IN ('live','recorded')), recorded_run_id TEXT, valid INTEGER, dimensions_json TEXT,
  raw_output TEXT, timing_json TEXT, job_id TEXT, created_at TEXT,
  UNIQUE (image_id, model_id, model_revision, image_sha256, context_hash, use_context, prompt_version, mode));
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, image_id TEXT REFERENCES images(image_id),
  prediction_id TEXT REFERENCES predictions(prediction_id), model_id TEXT, values_json TEXT, changed_json TEXT,
  status TEXT CHECK (status IN ('reviewed','corrected')), note TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS eval_results (
  eval_id TEXT PRIMARY KEY, model_id TEXT, set_name TEXT, checkpoint TEXT,
  validity TEXT CHECK (validity IN ('held_out','phase1_internal','seen_in_training','na')),
  provenance TEXT CHECK (provenance IN ('recomputed','offline_script','reported_only','kaggle_user','na')),
  source TEXT, trained_on TEXT, macro_f1 REAL, weighted_f1 REAL, accuracy_json TEXT, per_dimension_json TEXT,
  per_class_json TEXT, confusion_json TEXT, calibration_json TEXT, computed_at TEXT, notes_json TEXT);
CREATE TABLE IF NOT EXISTS recorded_runs (
  run_id TEXT PRIMARY KEY, model_id TEXT, path TEXT, recorded_at TEXT, gpu_name TEXT, images INTEGER,
  meta_json TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT);
CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_image ON predictions(image_id, model_id);
"""

TABLES = ("sessions", "images", "jobs", "predictions", "reviews", "eval_results", "recorded_runs", "settings")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def init(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, tuple(params)).fetchone()
            return dict(row) if row else None

    def tables(self) -> list[str]:
        rows = self.all("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return sorted(r["name"] for r in rows)
