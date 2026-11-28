"""Tassonomia: le 41 classi vengono da `common.prompt.TAXONOMY` (mai copiate); qui solo etichette ERP e indici."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .repo_code import prompt_module

ERP_LABELS_PATH = Path(__file__).resolve().parents[2] / "config" / "erp_labels.yaml"

# Fasce di confidenza del chip: le stesse dell'analisi del 70B (teacher_student_zeroshot.md:57). Solo aiuto visivo.
CONFIDENCE_BINS = {"low_below": 0.60, "high_from": 0.90,
                   "source": "src/plain-benchmark/reports/apertus-8B-benchmark/teacher_student_zeroshot.md:57"}


def dimensions() -> tuple[str, ...]:
    return tuple(prompt_module().DIMENSIONS)


def classes(dimension: str) -> tuple[str, ...]:
    return tuple(prompt_module().TAXONOMY[dimension])


def is_valid(dimension: str, value: str | None) -> bool:
    return dimension in prompt_module().TAXONOMY and value in prompt_module().TAXONOMY[dimension]


@lru_cache(maxsize=1)
def _erp() -> dict:
    return yaml.safe_load(ERP_LABELS_PATH.read_text(encoding="utf-8"))


def _auto(value: str) -> str:
    return value.replace("_", " ").title() if value.islower() else value


def erp_dimension_label(dimension: str) -> str:
    return _erp()["dimensions"][dimension]


def erp_label(dimension: str, value: str | None) -> tuple[str, str]:
    """(etichetta, fonte) con fonte in {prompt, proposal, auto}. `None` = output non valido."""
    if value is None:
        return "Invalid output", "none"
    known = (_erp()["labels"].get(dimension) or {}).get(value)
    if known:
        return known["label"], known["source"]
    return _auto(value), "auto"


def taxonomy_payload() -> dict:
    from .relevance import RULES

    codes = prompt_module().CODES
    out = []
    for dim in dimensions():
        letters = {v: k for k, v in codes[dim].items()}
        items = []
        for index, value in enumerate(classes(dim)):
            label, source = erp_label(dim, value)
            items.append({"index": index, "code": letters[value], "value": value,
                          "erp_label": label, "erp_label_source": source})
        out.append({"key": dim, "erp_label": erp_dimension_label(dim), "classes": items})
    return {"dimensions": out, "relevance_rules": RULES, "confidence_bins": CONFIDENCE_BINS}
