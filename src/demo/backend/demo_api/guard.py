"""Guardia dev/final: rifiuta le copie identiche di immagini che la demo non deve mai usare (C.4, passo 6)."""

from __future__ import annotations

from pathlib import Path

from .errors import ApiError


class Guard:
    """Hash vietati (`sha256.txt`) meno quelli ammessi a mano (`allow.txt`): le poche foto di dev usate come sample.

    Le regole della challenge permettono di mostrare poche foto della competizione in demo e video; per tutto il
    resto di dev e final la guardia resta chiusa.
    """

    def __init__(self, hashes: set[str], allowed: set[str] | None = None):
        self.hashes = hashes
        self.allowed = allowed or set()

    @classmethod
    def load(cls, path: Path) -> "Guard | None":
        if not path.exists():
            return None
        read = lambda p: {line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")}
        allow = path.with_name("allow.txt")
        return cls(read(path), read(allow) if allow.exists() else set())

    def __len__(self) -> int:
        return len(self.hashes)

    def check(self, sha256: str) -> None:
        if sha256 in self.hashes and sha256 not in self.allowed:
            raise ApiError("FORBIDDEN_IMAGE",
                           "This image belongs to the competition dev/final set and cannot be used in the demo.")
