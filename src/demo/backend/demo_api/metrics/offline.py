"""Importa i risultati di `evaluate.py` (priorità 2, `provenance = offline_script`) da `data/demo/eval/<modello>/<set>/`."""

from __future__ import annotations

import json
from pathlib import Path

from ..db import Database, now_iso

HELD_OUT_FOR_UNTRAINED = {"prompt-test"}   # zero-shot: il prompt non è stato scelto su prompt-test


def validity_for(model_raw: dict, set_name: str, allow_contaminated: bool = False) -> str:
    """Validità del set per il modello: contaminato se in `seen_sets`, altrimenti held-out o set interno di fase 1."""
    seen = set((model_raw.get("training") or {}).get("seen_sets") or [])
    if set_name in seen:
        return "seen_in_training"
    if set_name == "validation_450":
        return "held_out"
    if (model_raw.get("training") or {}).get("data") == "none" and set_name in HELD_OUT_FOR_UNTRAINED:
        return "held_out"
    return "phase1_internal"


def import_offline(db: Database, eval_dir: Path) -> int:
    db.execute("DELETE FROM eval_results WHERE provenance = 'offline_script'")
    count = 0
    for metrics_path in sorted(eval_dir.glob("*/*/metrics.json")):
        run_path = metrics_path.parent / "run.json"
        if not run_path.exists():
            continue
        metrics, run = json.loads(metrics_path.read_text()), json.loads(run_path.read_text())
        per_dimension, per_class = {}, {}
        for d, v in metrics["per_dimension"].items():
            per_dimension[d] = {"macro_f1": v["macro_f1"], "weighted_f1": v["weighted_f1"], "support": v["support"],
                                "accuracy": v.get("accuracy")}
            per_class[d] = v["per_class"]
        model_id, set_name = metrics_path.parent.parent.name, metrics_path.parent.name
        db.execute(
            "INSERT OR REPLACE INTO eval_results (eval_id, model_id, set_name, checkpoint, validity, provenance, source, "
            "trained_on, macro_f1, weighted_f1, accuracy_json, per_dimension_json, per_class_json, confusion_json, "
            "calibration_json, computed_at, notes_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"{model_id}|{set_name}|offline|{run['validity']}", model_id, set_name, run.get("checkpoint"),
             run["validity"], "offline_script", f"data/demo/eval/{model_id}/{set_name}/metrics.json",
             run.get("trained_on"), metrics["overall"]["macro_f1"], metrics["overall"]["weighted_f1"],
             json.dumps({d: v.get("accuracy") for d, v in metrics["per_dimension"].items()}),
             json.dumps(per_dimension), json.dumps({"source": f"data/demo/eval/{model_id}/{set_name}/metrics.json",
                                                    "data": per_class}), json.dumps(metrics["confusion"]),
             json.dumps(metrics.get("calibration")) if metrics.get("calibration") else None,
             run.get("finished_at") or now_iso(), json.dumps(run.get("notes") or [])))
        count += 1
    return count
