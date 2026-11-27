#!/usr/bin/env python3
"""Controllo anti-dati: da lanciare prima di ogni commit della demo.

Esamina i file in stage (`git diff --cached --name-only`) e fallisce se ne trova uno
che non dovrebbe stare in git: immagini, CSV, JSONL, database, pesi, screenshot, cartelle
di dati, file `.env`, oppure contenuto con ID della competizione (`img_` + 5 cifre) o
chiavi che iniziano con `hf_`.

Uso:
    python src/demo/scripts/check_no_data.py            # controlla lo stage
    python src/demo/scripts/check_no_data.py --all      # controlla tutti i file tracciati

Non stampa mai il contenuto trovato, solo il percorso e il motivo (niente ID o chiavi nei log).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
DATA_EXTENSIONS = {".csv", ".jsonl", ".sqlite", ".sqlite3", ".db", ".pt", ".safetensors", ".parquet", ".bin"}
# Le sole icone ammesse nel frontend sono SVG nostri (non sono nell'elenco delle immagini sopra).
ICON_DIR = "src/demo/frontend/src/assets/icons/"
FORBIDDEN_PREFIXES = ("data/", "demo_data/", "docs/demo/reference/")
FORBIDDEN_SEGMENTS = ("test-results", "playwright-report", "blob-report")

COMPETITION_ID = re.compile(rb"img_\d{5}")
HF_KEY = re.compile(rb"hf_[A-Za-z0-9]{20,}")
MAX_SCAN_BYTES = 5_000_000


def name_problems(path: str) -> list[str]:
    """Motivi per cui il solo nome del file è vietato."""
    problems: list[str] = []
    p = PurePosixPath(path)
    suffix = p.suffix.lower()
    if path.startswith(FORBIDDEN_PREFIXES):
        problems.append("cartella di dati o screenshot di riferimento")
    if any(part in FORBIDDEN_SEGMENTS for part in p.parts):
        problems.append("artefatto di test (screenshot, report)")
    if suffix in IMAGE_EXTENSIONS:
        problems.append("immagine (nella demo solo SVG in " + ICON_DIR + ")")
    if suffix in DATA_EXTENSIONS:
        problems.append(f"estensione di dati o pesi ({suffix})")
    if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
        problems.append("file di segreti")
    return problems


def content_problems(data: bytes) -> list[str]:
    """Motivi per cui il contenuto è vietato (senza riportare il testo trovato)."""
    problems: list[str] = []
    if b"\0" in data[:8192]:
        return problems  # binario: già coperto dal controllo sul nome
    data = data[:MAX_SCAN_BYTES]
    if COMPETITION_ID.search(data):
        problems.append("contiene un ID di immagine della competizione")
    if HF_KEY.search(data):
        problems.append("contiene una chiave che inizia con hf_")
    return problems


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def staged_paths() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.decode().split("\0") if p]


def tracked_paths() -> list[str]:
    return [p for p in _git("ls-files", "-z").decode().split("\0") if p]


def check(paths: list[str], read) -> dict[str, list[str]]:
    """Restituisce {percorso: [motivi]} per i file vietati. `read(path)` dà i byte da esaminare."""
    found: dict[str, list[str]] = {}
    for path in paths:
        problems = name_problems(path)
        try:
            problems += content_problems(read(path))
        except (OSError, subprocess.CalledProcessError):
            pass
        if problems:
            found[path] = problems
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="controlla tutti i file tracciati, non solo lo stage")
    args = parser.parse_args()

    if args.all:
        paths, read = tracked_paths(), lambda p: _git("show", f"HEAD:{p}")
        where = "tracciati"
    else:
        paths, read = staged_paths(), lambda p: _git("show", f":{p}")
        where = "in stage"

    found = check(paths, read)
    if not found:
        print(f"check_no_data: OK ({len(paths)} file {where})")
        return 0
    print(f"check_no_data: FALLITO — {len(found)} file {where} non ammessi:", file=sys.stderr)
    for path, problems in sorted(found.items()):
        print(f"  {path}: {'; '.join(problems)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
