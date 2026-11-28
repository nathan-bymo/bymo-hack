"""Configurazione della demo: variabili d'ambiente e `.env` ignorato da git. Non stampa mai i valori dei segreti."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION = "0.1.0"


def load_dotenv(path: Path) -> None:
    """Come `eval_prompt.load_env`, ma senza `sys.exit` e senza richiedere la chiave: non sovrascrive l'ambiente."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


@dataclass
class Settings:
    repo_root: Path = REPO_ROOT
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "demo")
    kaggle_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "hackapertus-galaxus")
    metric_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "hackapertus-galaxus" / "metric.py")
    models_yaml: Path = field(default_factory=lambda: REPO_ROOT / "src" / "demo" / "models.yaml")
    fullft_src: Path = field(default_factory=lambda: REPO_ROOT / "src" / "training")
    frontend_dist: Path = field(default_factory=lambda: REPO_ROOT / "src" / "demo" / "frontend" / "dist")
    max_upload_mb: int = 20
    max_files_per_request: int = 50
    max_pixels: int = 50_000_000
    worker_token: str | None = None
    worker_urls: dict[str, str] = field(default_factory=dict)

    @property
    def src_dir(self) -> Path:
        return self.repo_root / "src"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env")
        kaggle = _path("DEMO_KAGGLE_DIR", REPO_ROOT / "data" / "hackapertus-galaxus")
        worker_urls = {
            key.removeprefix("DEMO_WORKER_URL_").lower(): value
            for key, value in os.environ.items()
            if key.startswith("DEMO_WORKER_URL_")
        }
        return cls(
            repo_root=_path("DEMO_REPO_ROOT", REPO_ROOT),
            data_dir=_path("DEMO_DATA_DIR", REPO_ROOT / "data" / "demo"),
            kaggle_dir=kaggle,
            metric_path=_path("DEMO_METRIC_PATH", kaggle / "metric.py"),
            models_yaml=_path("DEMO_MODELS_YAML", REPO_ROOT / "src" / "demo" / "models.yaml"),
            fullft_src=_path("DEMO_FULLFT_SRC", REPO_ROOT / "src" / "training"),
            frontend_dist=_path("DEMO_FRONTEND_DIST", REPO_ROOT / "src" / "demo" / "frontend" / "dist"),
            max_upload_mb=int(os.environ.get("DEMO_MAX_UPLOAD_MB", "20")),
            max_pixels=int(os.environ.get("DEMO_MAX_PIXELS", "50000000")),
            worker_token=os.environ.get("DEMO_WORKER_TOKEN"),
            worker_urls=worker_urls,
        )


def git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - informazione facoltativa
        return None
