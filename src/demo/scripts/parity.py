#!/usr/bin/env python3
"""Test di parità: l'adapter della demo contro il codice originale del repo, su ~20 immagini di coverage (piano §14.2).

    python src/demo/scripts/parity.py --model apertus-8b-lora-heads-p11 --n 20      # su host GPU, solo con ok esplicito

- P11 / full fine-tune / modello finale: lancia `predict.py` / `ft_predict.py` (`--part coverage --limit N --batch-size N`) e confronta
  con l'adapter sulle stesse immagini: label identiche su tutte le celle, probabilità con differenza massima ≤ 1e-3.
- zero-shot: confronto dei corpi di richiesta con `eval_prompt.run_variant` (senza rete, gateway finto).
- P10: serve il confronto con `evaluate.py` e i codici di `encode.py`: NON ancora implementato (esce con un errore).

Stampa solo conteggi ed esiti (mai ID di immagini). Mai dev o final.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

from demo_api.registry import Registry  # noqa: E402
from demo_api.repo_code import ensure_src_on_path  # noqa: E402
from demo_api.settings import Settings  # noqa: E402

TOLERANCE = 1e-3


def zeroshot(registry: Registry, model_id: str) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_zeroshot_payload.py"]
    done = subprocess.run(cmd, cwd=REPO / "src" / "demo" / "backend", capture_output=True, text=True)
    return {"model": model_id, "check": "request bodies identical to eval_prompt.run_variant", "passed": done.returncode == 0,
            "detail": done.stdout.strip().splitlines()[-1] if done.stdout else ""}


def heads(settings: Settings, registry: Registry, model_id: str, n: int) -> dict:
    import pandas as pd
    import torch

    from demo_api.offline import run_model
    from demo_api.repo_code import prompt_module, splits_module
    from demo_worker.families.base import resolve_weights

    raw = registry.get(model_id).raw
    base_dir, checkpoint = resolve_weights(raw["weights"], registry.defaults["base_model"])
    train = pd.read_csv(settings.kaggle_dir / "train.csv", dtype=str, keep_default_na=False)
    rows = splits_module().split_rows(train, "coverage").head(n)
    folder = (settings.fullft_src if model_id == "apertus-8b-fullft-heads" else settings.repo_root / "src" / "training" / "phase-1" / "src" / "training")
    script = folder / {"fullft_heads": "apertus-8B-fullft-heads/ft_predict.py",
                       "vadapter_heads": "apertus-8B-vadapter-lora-heads/predict.py"}.get(
        raw["family"], "apertus-8B-lora-heads/predict.py")
    extra = ["--data-dir", str(settings.kaggle_dir)] if raw["family"] == "vadapter_heads" else []
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.csv"
        subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint), "--part", "coverage",
                        "--limit", str(n), "--batch-size", str(n), "--out", str(ref), *extra], check=True)
        reference = torch.load(ref.with_suffix(".probs.pt"))
    fields = [k for k, _ in prompt_module().CONTEXT_FIELDS]
    items = [{"image_id": r["image_id"], "sha256": r["image_id"], "path": settings.kaggle_dir / r["image_path"],
              "context": {k: r[k] for k in fields if r.get(k)}} for r in rows.to_dict("records")]
    results, batch, _ = run_model(settings, registry, model_id, items, True, n)
    by_id = {r["image_id"]: r for r in results}
    cells = same = 0
    worst = 0.0
    for t, d in enumerate(prompt_module().DIMENSIONS):
        for index, image_id in enumerate(reference["image_ids"]):
            probs = reference["probabilities"][t][index].tolist()
            ours = by_id[image_id]["dimensions"][d]
            label = prompt_module().TAXONOMY[d][max(range(len(probs)), key=probs.__getitem__)]
            cells += 1
            same += int(ours["label"] == label)
            worst = max(worst, abs(ours["confidence"] - max(probs)))
    return {"model": model_id, "images": len(reference["image_ids"]), "cells": cells, "identical_labels": same,
            "max_probability_difference": worst, "passed": same == cells and worst <= TOLERANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    ensure_src_on_path()
    settings = Settings.from_env()
    registry = Registry(settings)
    entry = registry.get(args.model)
    if entry is None:
        raise SystemExit(f"modello sconosciuto: {args.model}")
    family = entry.raw["family"]
    if family == "zeroshot_generative":
        out = zeroshot(registry, args.model)
    elif family in ("lora_heads", "fullft_heads", "vadapter_heads"):
        out = heads(settings, registry, args.model, args.n)
    else:
        raise SystemExit("parità di P10 non ancora implementata: serve il confronto con evaluate.py e i codici di encode.py")
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
