"""Fixture comuni. Le immagini dei test sono generate a runtime con Pillow: nessun file di dati nel repo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from demo_api.main import create_app  # noqa: E402
from demo_api.repo_code import ensure_src_on_path  # noqa: E402
from demo_api.settings import Settings  # noqa: E402

ensure_src_on_path()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "demo", frontend_dist=tmp_path / "no-dist",
                    kaggle_dir=tmp_path / "kaggle", metric_path=tmp_path / "kaggle" / "metric.py")


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as c:
        yield c
