#!/usr/bin/env python3
"""Costruisce l'indice degli sha256 delle immagini dev (e final, se esiste la cartella) della competizione.

L'indice finisce in `data/demo/guard/sha256.txt` (ignorato da git) e contiene SOLO hash: niente nomi di file né ID.
L'API lo usa per rifiutare l'upload di una copia identica (`FORBIDDEN_IMAGE`). È una rete di sicurezza: copre i
file identici, non sostituisce la regola. Non tocca mai `final_encrypted.bin`.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(kaggle_dir: Path, out: Path) -> dict[str, int]:
    hashes: set[str] = set()
    counts: dict[str, int] = {}
    for folder in ("dev_images", "final_images"):
        directory = kaggle_dir / folder
        if not directory.is_dir():
            counts[folder] = 0
            continue
        files = [p for p in directory.rglob("*") if p.is_file()]
        hashes.update(sha256_file(p) for p in files)
        counts[folder] = len(files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(hashes)) + ("\n" if hashes else ""))
    counts["unique_hashes"] = len(hashes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle-dir", type=Path, default=REPO / "data" / "hackapertus-galaxus")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "demo" / "guard" / "sha256.txt")
    args = parser.parse_args()
    counts = build(args.kaggle_dir, args.out)
    print("indice scritto:", ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
