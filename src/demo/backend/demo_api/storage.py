"""Archivio file locale in `data/demo/` (tutto ignorato da git)."""

from __future__ import annotations

from pathlib import Path

EXT_BY_FORMAT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


class Storage:
    def __init__(self, root: Path):
        self.root = Path(root)

    def ensure(self) -> None:
        for sub in ("uploads", "thumbs", "recorded", "export", "eval", "guard", "logs", "samples"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def upload_path(self, sha256: str, ext: str) -> Path:
        return self.root / "uploads" / sha256[:2] / f"{sha256}.{ext}"

    def thumb_path(self, sha256: str) -> Path:
        return self.root / "thumbs" / f"{sha256}.jpg"

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    @property
    def recorded_dir(self) -> Path:
        return self.root / "recorded"

    @property
    def guard_index(self) -> Path:
        return self.root / "guard" / "sha256.txt"
