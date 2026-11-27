"""Importa i punteggi riportati nei file del repo (priorità 3, `provenance = reported_only`).

Ogni numero viene riletto dal file indicato in `models.yaml` e confrontato con il valore scritto lì: se non
coincidono l'import fallisce. Nessun numero è stimato; ogni riga di `eval_results` porta la sua fonte `file:riga`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..db import Database, now_iso
from ..repo_code import prompt_module

SETS = ("coverage", "base", "prompt-test", "prompt-dev-600", "validation_450")
JSON_VIEWS = {"coverage": "coverage", "base (natural distribution)": "base", "test only": "prompt-test",
              "prototype_validation": "validation_450"}


class ReportedImportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- markdown
def _cells(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def num(text: str) -> float | None:
    try:
        return float(text.replace("**", "").strip())
    except ValueError:
        return None


def parse_tables(path: Path) -> list[dict]:
    """Tabelle markdown con intestazioni di sezione e numero di riga (da 1) di ogni riga."""
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: dict[int, str] = {}
    tables, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = m.group(2).strip()
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s\-|:]+\|?\s*$", lines[i + 1]):
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append((j + 1, _cells(lines[j])))
                j += 1
            tables.append({"line": i + 1, "headings": dict(headings), "header": _cells(line), "rows": rows})
            i = j
            continue
        i += 1
    return tables


def row_at(tables: list[dict], line: int) -> tuple[list[str], list[str]]:
    for table in tables:
        for number, cells in table["rows"]:
            if number == line:
                return table["header"], cells
    raise ReportedImportError(f"nessuna riga di tabella alla riga {line}")


def dimension_values(header: list[str], cells: list[str]) -> dict[str, float]:
    dims = prompt_module().DIMENSIONS
    return {d: v for d, name in zip(dims, dims) if name in header
            and (v := num(cells[header.index(name)])) is not None}


# ---------------------------------------------------------------------------------------------- per classe
def zeroshot_per_class(tables: list[dict], variant: str, rel: str) -> tuple[dict, str]:
    """F1 + precision/recall per classe di una variante zero-shot, su coverage (prompt_eval_coverage.md)."""
    dims = prompt_module().DIMENSIONS
    out: dict[str, dict] = {d: {} for d in dims}
    lines: list[int] = []
    for t in tables:
        h = t["headings"]
        dimension = h.get(4) if h.get(4) in dims else h.get(3) if h.get(3) in dims else None
        if dimension is None or variant not in t["header"]:
            continue
        col = t["header"].index(variant)
        if str(h.get(3, "")).startswith("coverage") and h.get(2, "").startswith("F1 per classe"):
            for number, cells in t["rows"]:
                f1, n = num(cells[col]), num(cells[1])
                if f1 is not None and n is not None:
                    out[dimension].setdefault(cells[0], {})["f1"] = f1
                    out[dimension][cells[0]]["support"] = int(n)
            lines += [t["line"], t["rows"][-1][0]]
        elif h.get(2, "").startswith("Precision e recall per classe (coverage)"):
            for number, cells in t["rows"]:
                pr = cells[col].split("/")
                if len(pr) == 2 and num(pr[0]) is not None:
                    out[dimension].setdefault(cells[0], {}).update(precision=num(pr[0]), recall=num(pr[1]))
            lines += [t["line"], t["rows"][-1][0]]
    if not any(out.values()):
        raise ReportedImportError(f"{rel}: nessuna tabella per classe per '{variant}'")
    return out, f"{rel}:{min(lines)}-{max(lines)}"


def p11_per_class(tables: list[dict], rel: str) -> tuple[dict, str]:
    dims = prompt_module().DIMENSIONS
    out: dict[str, dict] = {}
    lines: list[int] = []
    for t in tables:
        h = t["headings"]
        if not str(h.get(2, "")).startswith("Per classe, coverage"):
            continue
        dimension = str(h.get(3, "")).split(":")[0].strip()
        if dimension not in dims or "F1 P11" not in t["header"]:
            continue
        idx = {k: t["header"].index(k) for k in ("n", "precision", "recall", "F1 P11")}
        out[dimension] = {c[0]: {"precision": num(c[idx["precision"]]), "recall": num(c[idx["recall"]]),
                                 "f1": num(c[idx["F1 P11"]]), "support": int(num(c[idx["n"]]))}
                          for _, c in t["rows"]}
        lines += [t["line"], t["rows"][-1][0]]
    if set(out) != set(dims):
        raise ReportedImportError(f"{rel}: tabelle per classe incomplete {sorted(out)}")
    return out, f"{rel}:{min(lines)}-{max(lines)}"


def p11_confusion(path: Path, tables: list[dict], rel: str) -> dict:
    text = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(text) if l.startswith("## Matrice di confusione: perspective"))
    mapping = dict(re.findall(r"c(\d+) = ([a-z_]+)", " ".join(text[start:start + 6])))
    columns = [mapping[str(i)] for i in range(1, len(mapping) + 1)]
    table = next(t for t in tables if t["headings"].get(2, "").startswith("Matrice di confusione: perspective"))
    labels = [c[0] for _, c in table["rows"]]
    matrix = [[int(float(v)) for v in c[1:]] for _, c in table["rows"]]
    if labels != columns or any(len(r) != len(columns) for r in matrix):
        raise ReportedImportError(f"{rel}: matrice di confusione non quadrata o con ordine diverso")
    return {"perspective": {"labels": labels, "columns": columns, "matrix": matrix, "set": "coverage",
                            "source": f"{rel}:{table['line']}-{table['rows'][-1][0]}"}}


# ---------------------------------------------------------------------------------------------- JSON (P10)
def json_views(path: Path, rel: str) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {JSON_VIEWS.get(view, view): payload for view, payload in data["views"].items()}


def _json_row(payload: dict) -> dict:
    dims = prompt_module().DIMENSIONS
    per_dimension, per_class = {}, {}
    for d in dims:
        if d in payload:
            per_dimension[d] = {"macro_f1": payload[d]["f1_macro"], "weighted_f1": payload[d]["f1_weighted"],
                                "support": payload[d]["support"]}
            per_class[d] = payload[d]["per_class"]
    return {"macro_f1": payload["overall"]["f1_macro"], "weighted_f1": payload["overall"]["f1_weighted"],
            "per_dimension": per_dimension, "per_class": per_class}


# ---------------------------------------------------------------------------------------------- import
def _near(a: float | None, b: float | None, places: int = 3) -> bool:
    return a is not None and b is not None and round(a, places) == round(b, places)


def build_rows(models: dict[str, dict], repo_root: Path) -> list[dict]:
    """Righe `eval_results` (come dict) per tutti i modelli di `models.yaml`."""
    rows: list[dict] = []
    cache: dict[Path, list[dict]] = {}

    def tables_of(rel: str) -> list[dict]:
        path = repo_root / rel
        if path not in cache:
            cache[path] = parse_tables(path)
        return cache[path]

    for model_id, raw in models.items():
        training = (raw.get("training") or {}).get("data")
        yaml_sets = set()
        for ev in raw.get("evals", []):
            rel, _, line = str(ev["source"]).partition(":")
            row = {"model_id": model_id, "set_name": ev["set"], "checkpoint": ev.get("checkpoint"),
                   "validity": ev["validity"], "provenance": ev["provenance"], "source": ev["source"],
                   "trained_on": training, "macro_f1": ev.get("macro_f1"), "weighted_f1": ev.get("weighted_f1"),
                   "per_dimension": {}, "per_class": None, "confusion": None,
                   "notes": [n for n in [ev.get("note")] if n]}
            if rel.endswith(".json"):
                path = repo_root / rel
                views = json_views(path, rel)
                payload = views.get(ev["set"])
                if payload is None:
                    raise ReportedImportError(f"{rel}: vista '{ev['set']}' non trovata")
                parsed = _json_row(payload)
                if not (_near(parsed["macro_f1"], ev.get("macro_f1")) and _near(parsed["weighted_f1"], ev.get("weighted_f1"))):
                    raise ReportedImportError(f"{rel}: i valori in models.yaml non coincidono con il JSON ({model_id}, {ev['set']})")
                row.update(macro_f1=parsed["macro_f1"], weighted_f1=parsed["weighted_f1"],
                           per_dimension=parsed["per_dimension"], per_class={"source": rel, "data": parsed["per_class"]})
            elif line:
                header, cells = row_at(tables_of(rel), int(line))
                macro = num(cells[header.index("macro F1")]) if "macro F1" in header else None
                weighted = num(cells[header.index("weighted F1")]) if "weighted F1" in header else None
                if not (_near(macro, ev.get("macro_f1")) and _near(weighted, ev.get("weighted_f1"))):
                    raise ReportedImportError(f"{ev['source']}: i valori in models.yaml non coincidono con la tabella")
                row["per_dimension"] = {d: {"macro_f1": v, "weighted_f1": None, "support": None}
                                        for d, v in dimension_values(header, cells).items()}
            if ev["set"] == "coverage" and raw["family"] == "zeroshot_generative":
                variant = "8B v2 descrizione" if "8b" in model_id else "70B v2 descrizione"
                data, src = zeroshot_per_class(tables_of(raw["per_class_source"].split(":")[0]), variant,
                                               raw["per_class_source"].split(":")[0])
                row["per_class"] = {"source": src, "data": data}
            if ev["set"] == "coverage" and model_id == "apertus-8b-lora-heads-p11":
                rel_pc = raw["per_class_source"].split(":")[0]
                data, src = p11_per_class(tables_of(rel_pc), rel_pc)
                row["per_class"] = {"source": src, "data": data}
                cs = raw["confusion_source"]["source"].split(":")[0]
                row["confusion"] = p11_confusion(repo_root / cs, tables_of(cs), cs)
            yaml_sets.add(ev["set"])
            rows.append(row)
        # insiemi visti in training senza una riga propria: marcati "seen during training — invalid", senza numeri
        seen = set((raw.get("training") or {}).get("seen_sets") or [])
        for s in SETS:
            if s in seen and s not in yaml_sets:
                note = raw.get("contamination_note") or "Set used for training: not a valid evaluation."
                rows.append({"model_id": model_id, "set_name": s, "checkpoint": None, "validity": "seen_in_training",
                             "provenance": "na", "source": "models.yaml (training.seen_sets)", "trained_on": training,
                             "macro_f1": None, "weighted_f1": None, "per_dimension": {}, "per_class": None,
                             "confusion": None, "notes": [note]})
    return rows


def import_reported(db: Database, models: dict[str, dict], repo_root: Path) -> int:
    """Riscrive le righe `reported_only`/`na` di `eval_results` (quelle ricalcolate non si toccano)."""
    rows = build_rows(models, repo_root)
    db.execute("DELETE FROM eval_results WHERE provenance IN ('reported_only', 'na')")
    for r in rows:
        eval_id = f"{r['model_id']}|{r['set_name']}|{r['checkpoint'] or 'default'}|{r['validity']}"
        db.execute(
            "INSERT OR REPLACE INTO eval_results (eval_id, model_id, set_name, checkpoint, validity, provenance, "
            "source, trained_on, macro_f1, weighted_f1, accuracy_json, per_dimension_json, per_class_json, "
            "confusion_json, calibration_json, computed_at, notes_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (eval_id, r["model_id"], r["set_name"], r["checkpoint"], r["validity"], r["provenance"], r["source"],
             r["trained_on"], r["macro_f1"], r["weighted_f1"], None, json.dumps(r["per_dimension"]),
             json.dumps(r["per_class"]) if r["per_class"] else None,
             json.dumps(r["confusion"]) if r["confusion"] else None, None, now_iso(), json.dumps(r["notes"])))
    return len(rows)
