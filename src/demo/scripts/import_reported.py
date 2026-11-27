#!/usr/bin/env python3
"""Importa nel database della demo i punteggi riportati nei file del repo (con la loro fonte `file:riga`).

Non serve rete né GPU: rilegge i file indicati in `src/demo/models.yaml` e li confronta con i valori lì scritti.
L'API lo fa comunque all'avvio; questo script serve per controllarlo a mano.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

from demo_api.db import Database  # noqa: E402
from demo_api.metrics.reported import import_reported  # noqa: E402
from demo_api.registry import Registry  # noqa: E402
from demo_api.settings import Settings  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    db = Database(settings.data_dir / "demo.sqlite")
    db.init()
    registry = Registry(settings, db)
    n = import_reported(db, {k: v.raw for k, v in registry.models.items()}, settings.repo_root)
    print(f"importate {n} righe in eval_results")


if __name__ == "__main__":
    main()
