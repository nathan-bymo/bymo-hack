#!/usr/bin/env python3
"""Scarica i pesi di un modello alla revisione fissata in `models.yaml` (SOLO l'host GPU, SOLO con `--yes`).

La demo non scarica mai nulla da sola. Senza `--yes` lo script stampa cosa scaricherebbe e quanto pesa, poi esce.
Dimensioni: base 8B 18,4 GB (condivisa), P10 +0,319 GB, full fine-tune +10,49 GB; P11 e modello finale sono già
nel repo (Git LFS, `local_path`): per loro si scarica solo la base.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

from demo_api.registry import Registry, _contains_placeholder  # noqa: E402
from demo_api.settings import Settings  # noqa: E402


def plan(registry: Registry, model_id: str) -> list[dict]:
    entry = registry.get(model_id)
    if entry is None:
        raise SystemExit(f"modello sconosciuto: {model_id}")
    weights = entry.raw.get("weights")
    if not weights:
        raise SystemExit("questo modello non ha pesi da scaricare (remoto o solo metriche)")
    if _contains_placeholder(weights):
        raise SystemExit("la configurazione dei pesi è incompleta ([DA VERIFICARE]): nessun download possibile")
    base = registry.defaults["base_model"]
    items = []
    if weights.get("requires_base"):
        items.append({"repo": base["repo"], "revision": base["revision"], "patterns": None, "gb": base.get("size_gb")})
    if weights.get("local_path"):   # adapter weights in a local folder (published on Hugging Face, not in git)
        if not (REPO / weights["local_path"]).exists():
            source = weights.get("hf_repo", "the team's Hugging Face page")
            print(f"local weights missing: {weights['local_path']} (hf download {source} --local-dir {weights['local_path']})")
        return items
    items.append({"repo": weights["repo"], "revision": weights["revision"],
                  "patterns": [f"{weights['subfolder']}/*"], "gb": weights.get("size_gb")})
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--yes", action="store_true", help="conferma il download (altrimenti si stampa solo il piano)")
    args = parser.parse_args()
    items = plan(Registry(Settings.from_env()), args.model)
    total = sum(i["gb"] or 0 for i in items)
    for i in items:
        print(f"- {i['repo']} @ {i['revision'][:7]} {i['patterns'] or '(tutto)'}  ~{i['gb']} GB")
    print(f"totale ≈ {total:.2f} GB")
    if not args.yes:
        print("Nessun download eseguito. Rilancia con --yes per scaricare.")
        return
    from huggingface_hub import snapshot_download

    for i in items:
        print("scarico", i["repo"], "…", flush=True)
        snapshot_download(i["repo"], revision=i["revision"], allow_patterns=i["patterns"])
    print("fatto")


if __name__ == "__main__":
    main()
