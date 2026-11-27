#!/usr/bin/env python3
"""Valuta un modello su un set interno con il `metric.py` ufficiale e scrive `data/demo/eval/<modello>/<set>/`.

    python src/demo/scripts/evaluate.py --model <id> --set {coverage|base|prompt-test|validation_450} [--limit N]
                                        [--batch-size 32] [--allow-contaminated]

Rifiuta `dev` e `final` (il dev non ha label, il final non si valuta) e qualsiasi set non derivato da `train.csv`.
Rifiuta i set in `training.seen_sets` del modello, salvo `--allow-contaminated` (risultato `seen_in_training`).
Si lancia SOLO con l'ok esplicito dell'utente (usa la GPU o il gateway) e mai tra le 13:00 e le 14:30 di sabato 28/11.
Output: `predictions.csv`, `probs.json`, `metrics.json` (metric.evaluate + accuratezza + confusioni + calibrazione),
`run.json` (GPU, img/s, revisione dei pesi, data, validità).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

import pandas as pd  # noqa: E402

from demo_api.metrics.compute import calibration, compute, solution_frame  # noqa: E402
from demo_api.metrics.offline import validity_for  # noqa: E402
from demo_api.offline import now, run_model, write_json  # noqa: E402
from demo_api.registry import Registry  # noqa: E402
from demo_api.repo_code import ensure_src_on_path, load_metric, prompt_module, splits_module  # noqa: E402
from demo_api.settings import Settings  # noqa: E402

SETS = ("coverage", "base", "prompt-test", "prompt-dev-600", "validation_450")
SPLIT_NAME = {"coverage": "coverage", "base": "base", "prompt-test": "test", "prompt-dev-600": "dev"}


def select_rows(settings: Settings, set_name: str) -> pd.DataFrame:
    """Righe del set, sempre derivate da train.csv (mai dev/final)."""
    train = pd.read_csv(settings.kaggle_dir / "train.csv", dtype=str, keep_default_na=False)
    if set_name == "validation_450":
        ids_path = settings.repo_root / "data" / "prototype_splits" / "validation_450.csv"
        ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)["image_id"]
        if not set(ids) <= set(train.image_id):
            raise SystemExit("validation_450 contiene ID che non sono in train.csv: rifiutato")
        return train[train.image_id.isin(set(ids))].reset_index(drop=True)
    rows = splits_module().split_rows(train, SPLIT_NAME[set_name])
    return rows.head(600) if set_name == "prompt-dev-600" else rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-contaminated", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--dry", action="store_true", help="solo test dello script (famiglia finta)")
    args = parser.parse_args(argv)

    if args.set_name in ("dev", "final"):
        raise SystemExit(f"set '{args.set_name}' rifiutato: il dev non ha label e il final non si valuta mai")
    if args.set_name not in SETS:
        raise SystemExit(f"set sconosciuto '{args.set_name}'. Ammessi: {', '.join(SETS)}")
    ensure_src_on_path()
    settings = Settings.from_env()
    registry = Registry(settings)
    entry = registry.get(args.model)
    if entry is None:
        raise SystemExit(f"modello sconosciuto: {args.model}")
    validity = validity_for(entry.raw, args.set_name, args.allow_contaminated)
    if validity == "seen_in_training" and not args.allow_contaminated:
        raise SystemExit(f"{args.model} è stato allenato su '{args.set_name}': valutazione non valida. "
                         "Usa --allow-contaminated per ottenere un risultato marcato 'seen during training — invalid'.")

    rows = select_rows(settings, args.set_name)
    if args.limit:
        rows = rows.head(args.limit).reset_index(drop=True)
    p = prompt_module()
    fields = [k for k, _ in p.CONTEXT_FIELDS]
    items = [{"image_id": r["image_id"], "sha256": r["image_id"], "path": settings.kaggle_dir / r["image_path"],
              "context": {k: r[k] for k in fields if r.get(k)}} for r in rows.to_dict("records")]
    results, batch, info = run_model(settings, registry, args.model, items, not args.no_context, args.batch_size, args.dry)

    by_id = {r["image_id"]: r for r in results}
    dims = list(p.DIMENSIONS)
    predictions = pd.DataFrame([{"image_id": i, **{d: (by_id[i]["dimensions"][d]["label"] if i in by_id else None) for d in dims}}
                                for i in rows.image_id])
    confidences = pd.DataFrame([{"image_id": i, **{d: (by_id[i]["dimensions"][d]["confidence"] if i in by_id else None) for d in dims}}
                                for i in rows.image_id])
    metric = load_metric(settings.metric_path)
    solution = solution_frame(rows)
    metrics = compute(metric, solution, predictions)
    if confidences[dims].notna().any().any():
        metrics["calibration"] = calibration(solution, predictions, confidences.astype({d: float for d in dims}))

    out = settings.data_dir / "eval" / args.model / args.set_name
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    write_json(out / "probs.json", {i: {d: by_id[i]["dimensions"][d] for d in dims} for i in by_id})
    write_json(out / "metrics.json", metrics)
    write_json(out / "run.json", {
        "model_id": args.model, "set": args.set_name, "images": len(rows), "predicted": len(by_id), "validity": validity,
        "provenance": "offline_script", "checkpoint": info.get("checkpoint"), "model_revision": info["model_revision"],
        "gpu_name": batch.get("gpu_name"), "images_per_second": batch.get("images_per_second"),
        "batch_size": batch.get("batch_size"), "peak_memory_gb": batch.get("peak_memory_gb"),
        "trained_on": (entry.raw.get("training") or {}).get("data"), "finished_at": now(), "context": not args.no_context,
        "notes": (["Result on a set the model was trained on: not a valid evaluation."] if validity == "seen_in_training" else [])})
    print(f"{args.model} su {args.set_name}: {len(by_id)}/{len(rows)} immagini, macro F1 "
          f"{metrics['overall']['macro_f1']:.4f}, weighted {metrics['overall']['weighted_f1']:.4f} [{validity}] → {out}")


if __name__ == "__main__":
    main()
