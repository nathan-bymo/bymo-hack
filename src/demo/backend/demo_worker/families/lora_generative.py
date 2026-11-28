"""P10: Apertus 8B + LoRA (JSON generato), `src/micro-training/lora-generativo`. Piano B.3. NON verificata su GPU.

Confidenza (metodo A): prodotto delle probabilità dei token del valore scelto, dagli score della generazione
(`output_scores=True`). Il test di parità deve mostrare che le sequenze generate sono identiche con e senza score.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable

from .base import DIMENSIONS, REPO_ROOT, Family, WorkerError, add_path, gpu_info, written_images

PHASE_SRC = REPO_ROOT / "src" / "micro-training" / "phase-2" / "src"
FOLDER = PHASE_SRC / "micro-training" / "lora-generativo"


def value_token_confidence(tokenizer, generated_ids: list[int], scores_prob: list[float], text: str) -> dict:
    """{dimensione: prodotto delle probabilità dei token che coprono il valore della dimensione nel JSON generato}."""
    ends, previous = [], 0
    for k in range(len(generated_ids)):
        length = len(tokenizer.decode(generated_ids[: k + 1], skip_special_tokens=True))
        ends.append((previous, length))
        previous = length
    out = {}
    for dimension in DIMENSIONS:
        match = re.search(rf'"{dimension}"\s*:\s*"([^"]*)"', text)
        if not match:
            continue
        start, end = match.span(1)
        probs = [p for (s, e), p in zip(ends, scores_prob) if s < end and e > start]
        if probs:
            confidence = 1.0
            for p in probs:
                confidence *= p
            out[dimension] = confidence
    return out


class LoraGenerativeFamily(Family):
    name = "lora_generative"

    def __init__(self):
        self.models: dict[str, dict] = {}

    def info(self) -> dict:
        return {**gpu_info(), "transformers_commit": None, "code_commit": None, "loaded": self.loaded(), "family": self.name}

    def loaded(self) -> list[str]:
        return list(self.models)

    def load(self, model_id: str, cfg: dict, on_state: Callable[[str, str], None]) -> dict:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

        from .base import resolve_weights

        if not torch.cuda.is_available():
            raise WorkerError("MODEL_UNAVAILABLE", "no CUDA GPU is visible to the worker")
        on_state("loading", "resolving local weights")
        base_dir, adapter_dir = resolve_weights(cfg["weights"], cfg["base"])
        add_path(PHASE_SRC)
        add_path(FOLDER)
        import apertus_lora as L

        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        config = AutoConfig.from_pretrained(base_dir)
        sequences = L.Sequences(tokenizer, config.image_token_offset)
        on_state("loading", "loading the VQ tokenizer")
        vq, processor, _ = L.load_vision_tokenizer(base_dir, "cuda")
        on_state("loading", "loading the 8B model and merging the LoRA")
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            base_dir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=L.ATTENTION)
        model.model.vision_tokenizer = None
        model.model.audio_tokenizer = None
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload().eval()
        self.models[model_id] = {"L": L, "tokenizer": tokenizer, "sequences": sequences, "vq": vq,
                                 "processor": processor, "model": model, "side": int(cfg.get("side", 256))}
        return {"peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "side": int(cfg.get("side", 256))}

    def unload(self, model_id: str) -> None:
        import gc

        import torch

        self.models.pop(model_id, None)
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, model_id: str, items: list[dict], use_context: bool, batch_size: int):
        import torch

        if model_id not in self.models:
            raise WorkerError("MODEL_UNAVAILABLE", "model not loaded in this worker")
        m = self.models[model_id]
        L, tokenizer, sequences, model = m["L"], m["tokenizer"], m["sequences"], m["model"]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results: list[dict] = []
        with written_images(items) as paths:
            prompts, prep_ms = {}, {}
            for item, path in zip(items, paths):
                t0 = time.perf_counter()
                image = L.images.prepared_image(path, m["side"])
                h, w, codes = L.encode_image(m["vq"], m["processor"], image)
                prompts[item["image_id"]] = sequences.prompt_ids(item.get("context") or {}, h, w, codes, use_context)
                prep_ms[item["image_id"]] = (time.perf_counter() - t0) * 1000
            order = sorted(prompts, key=lambda i: len(prompts[i]))
            by_id = {i["image_id"]: i for i in items}
            position, size = 0, max(1, batch_size)
            while position < len(order):
                ids = order[position:position + size]
                batch = L.collate([{"input_ids": prompts[i], "labels": prompts[i]} for i in ids], tokenizer.pad_token_id)
                try:
                    t1 = time.perf_counter()
                    with torch.inference_mode():
                        out = model.generate(input_ids=batch["input_ids"].to("cuda"),
                                             attention_mask=batch["attention_mask"].to("cuda"), max_new_tokens=96,
                                             do_sample=False, eos_token_id=sequences.end,
                                             pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                                             output_scores=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if size == 1:
                        raise WorkerError("GPU_OOM", "GPU out of memory even at batch size 1") from None
                    size = max(1, size // 2)
                    continue
                position += size
                model_ms = (time.perf_counter() - t1) * 1000 / len(ids)
                start = batch["input_ids"].shape[1]
                for row, image_id in enumerate(ids):
                    generated = out.sequences[row, start:]
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    parsed = L.parse_answer(text)
                    token_probs = [float(torch.softmax(out.scores[k][row].float(), -1)[int(tok)])
                                   for k, tok in enumerate(generated)]
                    confs = value_token_confidence(tokenizer, [int(t) for t in generated], token_probs, text) if parsed else {}
                    dims = {d: {"label": parsed[d] if parsed else None, "confidence": confs.get(d) if parsed else None,
                                "top_k": None, "confidence_method": "sequence_prob" if confs.get(d) is not None else None}
                            for d in DIMENSIONS}
                    results.append({"image_id": image_id, "dimensions": dims, "raw_output": text,
                                    "timing_ms": {"preprocess": prep_ms[image_id], "model": model_ms,
                                                  "total": prep_ms[image_id] + model_ms}})
        results.sort(key=lambda r: [i["image_id"] for i in items].index(r["image_id"]))
        seconds = time.perf_counter() - started
        return results, {"seconds": seconds, "images_per_second": len(items) / seconds if seconds else None,
                         "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                         "gpu_name": torch.cuda.get_device_name(0), "batch_size": batch_size}
