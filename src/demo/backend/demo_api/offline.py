"""Esecuzione dei modelli fuori dall'API (script `record.py`, `evaluate.py`, `parity.py`): stessi adapter della demo.

I modelli con GPU girano **in questo processo** tramite le famiglie del worker (si lancia sull'host GPU); lo zero-shot
usa l'adapter del gateway. `dry=True` usa la famiglia finta e serve solo ai test degli script.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from .adapters.base import StoredImage
from .adapters.zeroshot import ZeroShotGenerativeAdapter
from .registry import Registry
from .settings import Settings


def run_model(settings: Settings, registry: Registry, model_id: str, items: list[dict], use_context: bool,
              batch_size: int, dry: bool = False, log=print) -> tuple[list[dict], dict, dict]:
    """items: [{image_id, sha256, path, context}]. Restituisce (risultati, batch, informazioni sul modello)."""
    entry = registry.get(model_id)
    if entry is None:
        raise SystemExit(f"modello sconosciuto: {model_id}")
    raw = entry.raw
    if raw["family"] == "zeroshot_generative":
        if not os.environ.get(raw["endpoint"]["api_key_env"]):
            raise SystemExit(f"manca {raw['endpoint']['api_key_env']}: impostala in .env o nell'ambiente")
        adapter = ZeroShotGenerativeAdapter(raw, parallel=8)
        results, errors = [], []
        stored = [StoredImage(i["image_id"], i["sha256"], Path(i["path"]), i.get("context") or {}) for i in items]
        batch = adapter.predict(stored, use_context, batch_size or 8, results.append,
                                lambda image_id, error: errors.append((image_id, str(error))), lambda: False)
        if errors:
            log(f"{len(errors)} immagini non riuscite (prima: {errors[0][1][:120]})")
        out = [{"image_id": r.image_id, "dimensions": {d: v.to_json() for d, v in r.dimensions.items()},
                "raw_output": r.raw_output, "timing_ms": r.timing_ms} for r in results]
        return out, batch.to_json(), {"model_revision": adapter.model_revision, "prompt_version": adapter.prompt_version,
                                      "checkpoint": None, "errors": len(errors)}
    from demo_worker.families import get_family

    family = get_family(raw["family"], dry)
    weights = raw.get("weights") or {}
    log(f"carico {model_id} (solo da cache locale, nessun download)…")
    loaded = family.load(model_id, {"model_id": model_id, "weights": weights, "base": registry.defaults.get("base_model", {}),
                                    "side": (raw.get("prompt") or {}).get("side", 256), **({"fail": weights["fail"]} if "fail" in weights else {})},
                         lambda state, message: log(f"  {state}: {message}"))
    payload = [{"image_id": i["image_id"], "sha256": i["sha256"], "ext": Path(i["path"]).suffix.lstrip("."),
                "bytes_b64": base64.b64encode(Path(i["path"]).read_bytes()).decode(), "context": i.get("context") or {}}
               for i in items]
    results, batch = family.predict(model_id, payload, use_context, batch_size or 8)
    batch = {"images": len(results), "ms_per_image": batch["seconds"] * 1000 / max(1, len(results)), **batch}
    return results, batch, {"model_revision": weights.get("local_path") or f"{weights.get('revision')}/{weights.get('subfolder')}",
                            "prompt_version": (raw.get("prompt") or {}).get("version"), "checkpoint": weights.get("subfolder"),
                            "load": loaded}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
