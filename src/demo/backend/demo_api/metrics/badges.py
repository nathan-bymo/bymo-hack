"""Badge su due assi, sempre visibili accanto a ogni numero (piano §9.3)."""

from __future__ import annotations

VALIDITY = {
    "held_out": "held-out",
    "phase1_internal": "phase-1 internal set (coverage)",
    "seen_in_training": "seen during training — invalid",
    "na": "n/a",
}
PROVENANCE = {
    "recomputed": "recomputed",
    "offline_script": "recomputed",
    "reported_only": "reported only",
    "kaggle_user": "Kaggle dev (public)",
}
SEEN_TOOLTIP = "This model was trained on these images: not a valid evaluation."


def badges(validity: str, provenance: str) -> list[str]:
    out = [VALIDITY[validity]]
    if provenance in PROVENANCE:
        out.append(PROVENANCE[provenance])
    return out
