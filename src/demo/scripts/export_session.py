#!/usr/bin/env python3
"""Esporta una sessione della demo per `record.py` sull'host GPU: `data/demo/export/<session_id>/`.

Contiene `images/<sha256>.<ext>` (i byte salvati dall'API) e `manifest.json` con sha256, estensione, contesto e
hash del contesto di ogni immagine. Nessun ID della competizione e nessun nome di file originale nel manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def export(data_dir: Path, session_id: str) -> Path:
    con = sqlite3.connect(data_dir / "demo.sqlite")
    con.row_factory = sqlite3.Row
    if con.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone() is None:
        sys.exit(f"sessione non trovata: {session_id}")
    rows = con.execute("SELECT * FROM images WHERE session_id = ? ORDER BY created_at, rowid", (session_id,)).fetchall()
    out = data_dir / "export" / session_id
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        source = Path(row["stored_path"])
        ext = source.suffix.lstrip(".")
        shutil.copy2(source, out / "images" / f"{row['sha256']}.{ext}")
        manifest.append({"sha256": row["sha256"], "ext": ext, "context": json.loads(row["context_json"] or "{}"),
                         "context_hash": row["context_hash"]})
    (out / "manifest.json").write_text(json.dumps({"session_id": session_id, "images": manifest}, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data" / "demo")
    args = parser.parse_args()
    out = export(args.data_dir, args.session)
    print(f"esportate {len(json.loads((out / 'manifest.json').read_text())['images'])} immagini in {out}")


if __name__ == "__main__":
    main()
