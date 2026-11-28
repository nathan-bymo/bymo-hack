"""Registro dei modelli da `src/demo/models.yaml`: validazione all'avvio e stato calcolato (piano B.7, B.8)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .settings import Settings

PLACEHOLDER = "[DA VERIFICARE]"
REQUIRED = ("id", "display_name", "family", "runtime")
ZEROSHOT = "zeroshot_generative"


class RegistryError(RuntimeError):
    """Configurazione dei modelli non valida: l'avvio fallisce con un messaggio chiaro."""


def _contains_placeholder(value: Any) -> str | None:
    if isinstance(value, str):
        return value if PLACEHOLDER in value else None
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).endswith("_note"):
                continue  # le note descrittive non rendono il modello inutilizzabile
            if _contains_placeholder(item):
                return str(key)
    if isinstance(value, list):
        for item in value:
            if _contains_placeholder(item):
                return "files"
    return None


@dataclass
class ModelEntry:
    raw: dict
    defaults: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.raw["id"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


class Registry:
    def __init__(self, settings: Settings, db=None):
        self.settings = settings
        self.db = db
        self.models: dict[str, ModelEntry] = {}
        self.defaults: dict = {}
        self.verified_at: str | None = None
        self.worker_state: dict[str, str] = {}   # famiglia -> stato riportato dal worker (M3)
        self.load()

    # ------------------------------------------------------------------ caricamento e validazione
    def load(self) -> None:
        path = self.settings.models_yaml
        if not path.exists():
            raise RegistryError(f"models.yaml non trovato: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("schema_version") != 1:
            raise RegistryError("models.yaml: schema_version deve essere 1")
        self.defaults, self.verified_at = data.get("defaults", {}), data.get("verified_at")
        seen: set[str] = set()
        for raw in data.get("models", []):
            missing = [k for k in REQUIRED if not raw.get(k)]
            if missing:
                raise RegistryError(f"models.yaml: modello {raw.get('id', '?')}: campi mancanti {missing}")
            if raw["id"] in seen:
                raise RegistryError(f"models.yaml: id duplicato {raw['id']}")
            seen.add(raw["id"])
            self._validate(raw)
            self.models[raw["id"]] = ModelEntry(raw, self.defaults)

    def _validate(self, raw: dict) -> None:
        model_id = raw["id"]
        score = (raw.get("kaggle_dev") or {}).get("score")
        if score is not None and not (isinstance(score, (int, float)) and 0 <= score <= 1):
            raise RegistryError(f"models.yaml: {model_id}: kaggle_dev.score deve essere null o un numero in [0, 1]")
        for item in raw.get("evals", []):
            for source in str(item.get("source", "")).split(";"):
                relative = source.strip().split(":")[0] if source.strip() else ""
                if not relative:
                    raise RegistryError(f"models.yaml: {model_id}: un risultato non ha `source`")
                if not (self.settings.repo_root / relative).exists():
                    raise RegistryError(
                        f"models.yaml: {model_id}: la fonte '{relative}' non esiste nel repo (evals[*].source)")
        kaggle = raw.get("kaggle_dev") or {}
        if kaggle.get("score") is not None:
            relative = str(kaggle.get("source", "")).split(":")[0]
            if not relative or not (self.settings.repo_root / relative).exists():
                raise RegistryError(f"models.yaml: {model_id}: kaggle_dev.score richiede `source` (file esistente nel repo)")
        for key in ("per_class_source",):
            if raw.get(key):
                relative = str(raw[key]).split(":")[0]
                if not (self.settings.repo_root / relative).exists():
                    raise RegistryError(f"models.yaml: {model_id}: {key} '{relative}' non esiste nel repo")

    # ------------------------------------------------------------------ stato
    def get(self, model_id: str) -> ModelEntry | None:
        return self.models.get(model_id)

    def recorded_runs(self, model_id: str) -> list[dict]:
        if self.db is None:
            return []
        rows = self.db.all("SELECT run_id, recorded_at, gpu_name, images FROM recorded_runs WHERE model_id = ? "
                           "ORDER BY recorded_at DESC", (model_id,))
        return rows

    def live_status(self, entry: ModelEntry) -> tuple[str, str | None]:
        raw = entry.raw
        override = raw.get("status_override")
        if override:
            return override["status"], override.get("reason")
        family = raw["family"]
        if family == ZEROSHOT:
            env = (raw.get("endpoint") or {}).get("api_key_env", "LLM_API_KEY")
            if not os.environ.get(env):
                return "unavailable", f"missing {env} (set it in .env or the environment)"
            return "ready", None
        placeholder = _contains_placeholder(raw.get("weights", {}))
        if placeholder:
            return "unavailable", f"incomplete configuration: weights.{placeholder} still to be verified"
        url = self.settings.worker_urls.get(family)
        if not url:
            return "unavailable", "no GPU worker configured (laptop mode: use recorded runs)"
        return self.worker_state.get(family, "not_loaded"), None

    def status(self, entry: ModelEntry) -> dict:
        status, reason = self.live_status(entry)
        runs = self.recorded_runs(entry.id)
        modes = (["live"] if status != "unavailable" else []) + (["recorded"] if runs else [])
        if status == "unavailable" and runs:
            reason = (reason or "") + "; recorded predictions available"
        return {"status": status, "status_reason": reason, "modes": modes, "recorded_runs": runs}

    # ------------------------------------------------------------------ presentazione (C.3)
    @staticmethod
    def headline_eval(raw: dict) -> dict | None:
        evals = raw.get("evals") or []
        pick = next((e for e in evals if e.get("set") == "coverage"), evals[0] if evals else None)
        if pick is None:
            return None
        return {"set": pick["set"], "macro_f1": pick.get("macro_f1"), "validity": pick["validity"],
                "provenance": pick["provenance"], "source": pick["source"]}

    def prices(self) -> dict[str, float]:
        cost = self.defaults.get("cost", {})
        return {item["gpu_label"]: item["usd_per_hour"] for item in [cost.get("default"), *cost.get("alternatives", [])]
                if item}

    def speeds(self, raw: dict) -> list[dict]:
        """Velocità riportate (img/s) con il costo per 100.000 immagini, solo se end to end e su una GPU nota."""
        prices = self.prices()
        out = []
        for m in raw.get("measured", []):
            if not str(m.get("what", "")).startswith("img/s"):
                continue
            gpu = m.get("gpu_label")
            price = prices.get(gpu) if gpu and m.get("cost_eligible") else None
            cost = round(price / (m["value"] * 3600) * 100_000, 2) if price and m["value"] else None
            reason = None
            if cost is None:
                reason = ("shared gateway: API cost not public" if raw["family"] == ZEROSHOT
                          else "not end to end: cost not computed")
            out.append({"images_per_second": m["value"], "what": m["what"], "conditions": m.get("conditions"),
                        "gpu_label": gpu or ("gateway" if raw["family"] == ZEROSHOT else None),
                        "usd_per_hour": price, "cost_per_100k_usd": cost, "cost_note": reason,
                        "source": m["source"]})
        return out

    @staticmethod
    def scores(raw: dict) -> list[dict]:
        """Ogni punteggio con il suo set: Kaggle dev prima, poi i set interni. Mai mescolati in una classifica."""
        out = []
        kaggle = raw.get("kaggle_dev") or {}
        if kaggle.get("score") is not None:
            out.append({"set": "kaggle_dev", "macro_f1": kaggle["score"], "validity": "kaggle_public",
                        "source": kaggle.get("source"), "source_label": kaggle.get("source_label"),
                        "note": kaggle.get("note")})
        for ev in raw.get("evals", []):
            out.append({"set": ev["set"], "macro_f1": ev.get("macro_f1"), "weighted_f1": ev.get("weighted_f1"),
                        "validity": ev["validity"], "source": ev["source"], "note": ev.get("note"),
                        "checkpoint": ev.get("checkpoint")})
        return out

    def summary(self, entry: ModelEntry) -> dict:
        raw = entry.raw
        vram = raw.get("vram_estimate_gb")
        measured = any(m.get("what", "").startswith("VRAM") for m in raw.get("measured", []))
        weights = raw.get("weights")
        memory = raw.get("memory") or ({"gb": vram, "kind": "estimate", "what": "estimate", "source": "models.yaml"}
                                       if vram else None)
        out = {
            "model_id": entry.id, "display_name": raw["display_name"],
            "short_name": raw.get("short_name") or raw["display_name"], "featured": bool(raw.get("featured")),
            "description": raw.get("description"),
            "family": raw["family"], "runtime": raw["runtime"], **self.status(entry),
            "confidence": raw.get("confidence"), "training": raw.get("training"),
            "training_label": raw.get("training_label") or (raw.get("training") or {}).get("data"),
            "weights": {k: weights.get(k) for k in ("repo", "revision", "subfolder", "size_gb", "local_path",
                                                    "location_label")} if weights else None,
            "vram_estimate_gb": vram,
            "vram_estimate_kind": "measured" if (raw["id"] == "apertus-8b-lora-heads-p11" or measured) else "estimate",
            "memory": memory,
            "speeds": self.speeds(raw),
            "scores": self.scores(raw),
            "seen_sets": (raw.get("training") or {}).get("seen_sets") or [],
            "contamination_note": raw.get("contamination_note"),
            "headline_eval": self.headline_eval(raw),
            "kaggle_dev": raw.get("kaggle_dev") or {"score": None},
            "parity": raw.get("parity") or {"verified": False, "scope": "parity test not run (needs a GPU host: scripts/parity.py)"},
        }
        return out

    def detail(self, entry: ModelEntry) -> dict:
        raw = entry.raw
        out = self.summary(entry)
        out.update({"evals": raw.get("evals", []), "measured": raw.get("measured", []),
                    "caveats": raw.get("caveats", []), "contamination_note": raw.get("contamination_note"),
                    "prompt": raw.get("prompt"), "code": raw.get("code"),
                    "per_class_source": raw.get("per_class_source"), "confusion_source": raw.get("confusion_source")})
        return out
