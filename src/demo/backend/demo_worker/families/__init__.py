"""Famiglie di modelli del worker GPU: una per processo (piano B.5.1)."""

from __future__ import annotations


def get_family(name: str, dry: bool):
    if dry:
        from .dry import DryFamily

        return DryFamily(name)
    if name == "lora_heads":
        from .lora_heads import LoraHeadsFamily

        return LoraHeadsFamily()
    if name == "fullft_heads":
        from .fullft_heads import FullFtHeadsFamily

        return FullFtHeadsFamily()
    if name == "lora_generative":
        from .lora_generative import LoraGenerativeFamily

        return LoraGenerativeFamily()
    raise ValueError(f"famiglia sconosciuta: {name}")
