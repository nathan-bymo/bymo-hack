#!/usr/bin/env python3
"""Trasforma le probabilità già calcolate su GPU (file `*.probs.pt` dei job di valutazione) in registrazioni della demo.

    python src/demo/scripts/import_probs.py --model apertus-8b-vadapter-lora-heads-final \
        --probs data/ensemble/vlh_final-abt_prompt_test_probs.pt --rows data/train.csv --ids ids.json \
        --gpu-name "H200 (HF job)" --images-per-second 6.2 --source "RISULTATI.md:62"

Nessuna predizione inventata: ogni risposta è l'argmax delle probabilità scritte dal modello, con le sue alternative.
Le immagini sono lette dal disco e passano per lo stesso `ingest` dell'upload, così lo sha256 coincide con quello che
la demo calcola quando la stessa foto viene caricata. Il contesto prodotto con cui il modello ha risposto viene
salvato nella registrazione: all'upload della stessa foto la demo lo ricompila.
L'output resta in `data/demo/recorded/` (fuori da git): contiene contesti prodotto della competizione.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "demo" / "backend"))

from demo_api import images_io  # noqa: E402
from demo_api.repo_code import ensure_src_on_path  # noqa: E402
from demo_api.settings import Settings  # noqa: E402
from demo_api.taxonomy import dimensions  # noqa: E402

TOP_K = 3


def load_probs(path: Path):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    dims = list(data["dimensions"])
    probs = data["probabilities"]
    if isinstance(probs, dict):
        probs = [probs[d] for d in dims]
    index = {image_id: k for k, image_id in enumerate(data["image_ids"])}
    return dims, probs, index, data.get("classes")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--classes-from", type=Path, help="file .probs.pt con l'elenco delle classi, se --probs non lo ha")
    parser.add_argument("--rows", type=Path, required=True, help="CSV con image_id, image_path e i campi del contesto")
    parser.add_argument("--ids", type=Path, required=True, help="JSON con l'elenco degli image_id da registrare")
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--images-per-second", type=float)
    parser.add_argument("--source", required=True, help="report o job da cui vengono le probabilità")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    ensure_src_on_path()
    settings = Settings.from_env()
    dims, probs, index, classes = load_probs(args.probs)
    if classes is None:
        if not args.classes_from:
            raise SystemExit("--probs non contiene le classi: usare --classes-from")
        classes = load_probs(args.classes_from)[3]
    if list(dimensions()) != dims:
        raise SystemExit(f"ordine delle dimensioni diverso: {dims}")
    rows = {r["image_id"]: r for r in csv.DictReader(args.rows.open(encoding="utf-8"))}
    wanted = json.loads(args.ids.read_text())
    root = args.rows.parent

    # il run_id è la chiave della tabella recorded_runs: deve essere unico tra tutti i modelli
    run_id = args.run_id or f"{args.model}-probs-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = settings.data_dir / "recorded" / args.model / run_id
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for image_id in wanted:
            if image_id not in index:
                print(f"saltata: non presente nel file di probabilità ({args.probs.name})")
                continue
            row = rows[image_id]
            data = (root / row["image_path"]).read_bytes()
            ingested = images_io.ingest(data, images_io.sniff(data), settings.max_upload_mb * 1024 * 1024, settings.max_pixels)
            context = images_io.clean_context(row)
            result = {}
            for d, tensor in zip(dims, probs):
                p = tensor[index[image_id]].float()
                order = p.argsort(descending=True)[:TOP_K].tolist()
                top = [{"label": classes[d][k], "p": round(float(p[k]), 4)} for k in order]
                result[d] = {"label": top[0]["label"], "confidence": top[0]["p"], "top_k": top,
                             "confidence_method": "head_softmax"}
            handle.write(json.dumps({
                "image_sha256": ingested.sha256, "context_hash": images_io.context_hash(context), "use_context": True,
                "context": context, "dimensions": result, "valid": True, "raw_output": None, "timing_ms": {}},
                ensure_ascii=False) + "\n")
            written += 1
    (out / "run.json").write_text(json.dumps({
        "model_id": args.model, "model_revision": args.probs.stem, "gpu_name": args.gpu_name,
        "images_per_second": args.images_per_second, "batch_size": 32, "peak_memory_gb": None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "use_context": True,
        "source": args.source, "note": "probabilities written by the model during its GPU evaluation job"},
        indent=2) + "\n")
    print(f"{args.model}: {written}/{len(wanted)} immagini registrate in {out}")


if __name__ == "__main__":
    main()
