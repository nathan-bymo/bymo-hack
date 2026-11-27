"""Guardia dev/final: rifiuta le copie identiche di immagini che la demo non deve mai usare (C.4, passo 6)."""

from __future__ import annotations

from pathlib import Path

from .errors import ApiError


class Guard:
    def __init__(self, hashes: set[str]):
        self.hashes = hashes

    @classmethod
    def load(cls, path: Path) -> "Guard | None":
        if not path.exists():
            return None
        return cls({line.strip() for line in path.read_text().splitlines() if line.strip()})

    def __len__(self) -> int:
        return len(self.hashes)

    def check(self, sha256: str) -> None:
        if sha256 in self.hashes:
            raise ApiError("FORBIDDEN_IMAGE",
                           "This image belongs to the competition dev/final set and cannot be used in the demo.")
