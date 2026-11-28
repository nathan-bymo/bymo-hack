"""Recomputes the metrics of every saved run into two tables, for the benchmarks. Aggregates only, no API calls.

Reads predictions saved under data/ (git-ignored): CSV files and the cached zero-shot answers of eval_prompt.py.
A run is scored on a set only if it predicted every image of that set. Writes two tables,
  overall     one row per (run, set): macro and weighted F1, then macro F1 of each dimension
  per_class   one row per (run, set, dimension, class): support and the scores of metric.py
as markdown in --out (for the repository, which ignores *.csv) and as csv in data/benchmark/ (local).

  python src/scripts/benchmark_stats.py --out src/training/benchmark
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "training")]
import eval_prompt as E  # noqa: E402
from common import prompt, splits  # noqa: E402

DATA = HERE.parents[1] / "data"

CSV_RUNS = {  # name: file under data/
    "P11 LoRA+teste 256": "hf_results_lh/full-predict/predictions/coverage.csv",
    "P12 LoRA+teste+pooling 512": "hf_results_lh512/full-train/predictions/coverage.csv",
    "Nathan vlh finale (final-abt)": "ensemble/pt_nathan.csv",
    "P13 final-abt + sintetici": "hf_results_synth/synth-ft-1128-0243/predictions/test.csv",
    "P14 SigLIP so400m 512": "hf_results_eyes/eyes512-run1-1128-0431/predictions/test.csv",
    "Ensemble P11+Nathan media": "ensemble/pt_media-50-50.csv",
    "Ensemble P11+Nathan, lingua da Nathan": "ensemble/pt_mix.csv",
}
CSV_FOLDERS = {  # prefix: folder of CSV files under data/
    "Teste su Apertus congelato": "hf_results/small/heads",
    "Teste con pesi per classe": "runs/heads_weights",
    "Baseline pixel, 450": "runs/heads_pixels_small",
    "Baseline pixel, coverage": "runs/smoke_heads",
}


def zero_shot_predictions(folder):
    """{image_id: labels} from the cached answers of one zero-shot variant (folder name of eval_prompt.py)."""
    size, style, answer = folder.name.split("_")[:3]
    result = {}
    for line in (folder / "answers.jsonl").read_text().splitlines():
        record = json.loads(line)
        if style in ("split", "describe"):
            labels = {d: prompt.parse_question(record["choices"][d], d, answer)[0] for d in prompt.DIMENSIONS}
        else:
            labels = prompt.parse_choice(record["choice"], answer)
            labels = labels[0] if isinstance(labels, tuple) else labels
        if labels:
            result[record["image_id"]] = labels
        else:
            result[record["image_id"]] = {d: None for d in prompt.DIMENSIONS}  # invalid answer: scored as wrong
    return result


def all_runs(everything):
    for name, relative in CSV_RUNS.items():
        yield name, DATA / relative, None
    for prefix, relative in CSV_FOLDERS.items():
        for path in sorted((DATA / relative).glob("*.csv")):
            yield f"{prefix}: {path.stem}", path, None
    for folder in sorted((DATA / "runs").glob("*b_*")):
        if (folder / "answers.jsonl").exists():
            size, style, answer = folder.name.split("_")[:3]
            yield f"Zero-shot {size.upper()}: {style}, {answer}", None, zero_shot_predictions(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metric = E.load_metric()
    everything = pd.read_csv(DATA / "train.csv", dtype=str, keep_default_na=False)
    coverage = E.load_split("coverage", 0)
    buckets = coverage.product_id.map(splits.bucket)
    named = {"coverage": coverage, "base (prompt-dev + prompt-test)": coverage[buckets.isin([0, 1])],
             "prompt-test": coverage[buckets == 1], "prompt-dev, prime 600": E.load_split("dev", 600)}
    overall, per_class = [], []
    for name, path, predictions in all_runs(everything):
        if path is not None:
            if not path.exists():
                print("manca:", name)
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            predictions = {r["image_id"]: {d: r[d] or None for d in prompt.DIMENSIONS} for r in frame.to_dict("records")}
        ids = set(predictions)
        sets = {label: rows for label, rows in named.items() if set(rows.image_id) <= ids}
        if not sets and ids <= set(everything.image_id):  # e.g. the balanced set of 450 images
            sets = {f"set di {len(ids)} immagini": everything[everything.image_id.isin(ids)]}
        for label, rows in sets.items():
            rows = rows.reset_index(drop=True)
            results = metric.evaluate(E.solution_frame(rows), E.submission_frame(rows, predictions), "image_id")
            overall.append({"run": name, "set": label, "images": len(rows),
                            "macro_f1": round(results["overall"]["f1_macro"], 4),
                            "weighted_f1": round(results["overall"]["f1_weighted"], 4),
                            **{d: round(results[d]["f1_macro"], 4) for d in prompt.DIMENSIONS if d in results}})
            for dimension in prompt.DIMENSIONS:
                for klass, values in results.get(dimension, {}).get("per_class", {}).items():
                    per_class.append({"run": name, "set": label, "dimension": dimension, "class": klass,
                                      **{k: round(v, 4) if isinstance(v, float) else v for k, v in values.items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    local = DATA / "benchmark"
    local.mkdir(parents=True, exist_ok=True)
    for name, title, rows in (("overall", "Punteggi per prova e set", overall),
                              ("per_class", "Punteggi per classe", per_class)):
        frame = pd.DataFrame(rows)
        frame.to_csv(local / f"{name}.csv", index=False)
        (args.out / f"{name}.md").write_text(markdown(title, frame))
    print(f"{len(overall)} righe in overall, {len(per_class)} in per_class")


def markdown(title, frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [f"# {title}", "", "Generato da `src/scripts/benchmark_stats.py`. Solo numeri aggregati. "
             "Leggere prima le avvertenze in [README.md](README.md).", "",
             "| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
