"""Applicabilità delle dimensioni: le regole di challenge_description.md:102-107 applicate alla **predizione**.

È una convenzione di interfaccia (A.2.1), non la regola di punteggio, che si applica alle label umane.
Con una label mancante (output non valido) una regola non scatta: nel dubbio la dimensione resta applicabile.
"""

from __future__ import annotations

SOURCE = "challenge_description.md:102"

RULES = [
    {"if": {"subject": ["logo", "placeholder", "energy_label"]},
     "not_applicable": ["background", "perspective", "individuals", "text_classification"], "source": "challenge_description.md:102"},
    {"if": {"subject": ["infographic"]},
     "not_applicable": ["background", "perspective", "individuals"], "source": "challenge_description.md:103"},
    {"if": {"subject": ["delivery_scope", "detail_closeup", "product_grouping"]},
     "not_applicable": ["perspective"], "source": "challenge_description.md:104"},
    {"if": {"background": ["scene_lifestyle"]},
     "not_applicable": ["subject", "perspective"], "source": "challenge_description.md:105"},
    {"if": {"text_language": "only when a text overlay is present, or subject is infographic, packaging, "
                              "package_and_product or full_product"},
     "not_applicable": ["text_language"], "source": "challenge_description.md:106-107"},
]

LANGUAGE_SUBJECTS = {"infographic", "packaging", "package_and_product", "full_product"}


def applicable(labels: dict[str, str | None]) -> dict[str, bool]:
    """`labels`: classe predetta (o None) per ognuna delle 6 dimensioni. Restituisce dimensione -> applicabile."""
    result = {d: True for d in ("subject", "background", "perspective", "individuals", "text_classification",
                                "text_language")}
    subject, background = labels.get("subject"), labels.get("background")
    overlay = labels.get("text_classification")
    for rule in RULES[:4]:
        (dimension, values), = rule["if"].items()
        if labels.get(dimension) in values:
            for target in rule["not_applicable"]:
                result[target] = False
    if subject is not None and overlay is not None:
        if not (overlay != "none" or subject in LANGUAGE_SUBJECTS):
            result["text_language"] = False
    return result
