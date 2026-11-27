#!/usr/bin/env python3
"""Registra le predizioni di un modello su una sessione esportata (da lanciare sull'host GPU).

    python src/demo/scripts/export_session.py --session <id>          # sul laptop
    python src/demo/scripts/record.py --model apertus-8b-lora-heads-p11 --export data/demo/export/<id>   # su GPU
    # poi copiare data/demo/recorded/<modello>/<run>/ sul laptop e POST /api/recorded/rescan

Usa gli stessi adapter della demo (nessun peso viene scaricato). Scrive `run.json` e `predictions.jsonl`; il manifest
non contiene ID della competizione né nomi di file. Mai tra le 13:00 e le 14:30 di sabato 28/11 (GPU del team).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

from demo_api.offline import now, run_model, write_json  # noqa: E402
from demo_api.registry import Registry  # noqa: E402
from demo_api.repo_code import ensure_src_on_path  # noqa: E402
from demo_api.settings import Settings  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--export", type=Path, required=True, help="cartella creata da export_session.py")
    parser.add_argument("--out", type=Path, help="default: data/demo/recorded/<modello>/<run>/")
    parser.add_argument("--run-id")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--dry", action="store_true", help="solo test dello script (DEMO_WORKER_ALLOW_DRY=1)")
    args = parser.parse_args(argv)
    ensure_src_on_path()
    settings = Settings.from_env()
    registry = Registry(settings)
    manifest = json.loads((args.export / "manifest.json").read_text())
    items = [{"image_id": m["sha256"], "sha256": m["sha256"], "path": args.export / "images" / f"{m['sha256']}.{m['ext']}",
              "context": m["context"]} for m in manifest["images"]]
    context_of = {m["sha256"]: m for m in manifest["images"]}
    started = now()
    results, batch, info = run_model(settings, registry, args.model, items, not args.no_context, args.batch_size, args.dry)
    finished = now()
    run_id = args.run_id or f"run-{finished.replace(':', '').replace('-', '')}"
    out = args.out or settings.data_dir / "recorded" / args.model / run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as handle:
        for r in results:
            handle.write(json.dumps({
                "image_sha256": r["image_id"], "context_hash": context_of[r["image_id"]]["context_hash"],
                "use_context": not args.no_context, "dimensions": r["dimensions"],
                "valid": all(d["label"] is not None for d in r["dimensions"].values()),
                "raw_output": r.get("raw_output"), "timing_ms": r.get("timing_ms") or {}}) + "\n")
    write_json(out / "run.json", {
        "model_id": args.model, "model_revision": info["model_revision"], "checkpoint": info.get("checkpoint"),
        "code_commit": None, "gpu_name": batch.get("gpu_name"), "batch_size": batch.get("batch_size"),
        "images_per_second": batch.get("images_per_second"), "peak_memory_gb": batch.get("peak_memory_gb"),
        "started_at": started, "finished_at": finished, "use_context": not args.no_context,
        "prompt_version": info.get("prompt_version")})
    print(f"registrate {len(results)}/{len(items)} immagini in {out}")


if __name__ == "__main__":
    main()
