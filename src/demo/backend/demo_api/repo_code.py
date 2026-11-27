"""Accesso al codice del repo senza modificarlo né copiarlo (piano B.6): si aggiunge `<repo>/src` a `sys.path`."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

from .settings import REPO_ROOT


def ensure_src_on_path(src_dir: Path | None = None) -> None:
    path = str(src_dir or REPO_ROOT / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=1)
def prompt_module():
    ensure_src_on_path()
    from common import prompt

    return prompt


@lru_cache(maxsize=1)
def images_module():
    ensure_src_on_path()
    from common import images

    return images


@lru_cache(maxsize=1)
def splits_module():
    ensure_src_on_path()
    from common import splits

    return splits


@lru_cache(maxsize=1)
def eval_prompt_module():
    """`src/scripts/eval_prompt.py` caricato da file con il nome `bymo_eval_prompt` (piano B.2)."""
    ensure_src_on_path()
    path = REPO_ROOT / "src" / "scripts" / "eval_prompt.py"
    spec = importlib.util.spec_from_file_location("bymo_eval_prompt", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bymo_eval_prompt"] = module
    spec.loader.exec_module(module)
    return module


def load_metric(metric_path: Path):
    """Il `metric.py` ufficiale dei dati Kaggle, caricato da file: mai copiato né reimplementato."""
    spec = importlib.util.spec_from_file_location("bymo_metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
