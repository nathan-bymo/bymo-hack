"""Zero-shot Apertus 8B/70B via gateway OpenAI-compatibile (piano B.2). Gira nel processo dell'API.

Le sette richieste di un'immagine (descrizione + 6 domande) sono costruite con le funzioni di `common.prompt` e
spedite con `eval_prompt.complete`: stesso corpo, stessi tentativi, stessa lettura delle logprob del repo.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..errors import ApiError
from ..repo_code import eval_prompt_module, images_module, prompt_module
from ..taxonomy import dimensions
from .base import BatchInfo, DimensionPrediction, ImagePrediction, StoredImage, top_k

GATEWAY_NAME = "gateway remoto"
MAX_PARALLEL = 32


class ZeroShotGenerativeAdapter:
    def __init__(self, entry: dict, parallel: int = 8):
        self.entry = entry
        self.model_id = entry["id"]
        endpoint = entry["endpoint"]
        self.model_name = endpoint["model_name"]
        self.default_base_url = endpoint.get("default_base_url")
        prompt_spec = entry["prompt"]
        self.side = int(prompt_spec["side"])
        self.answer = prompt_spec["answer"]
        self.parallel = max(1, min(MAX_PARALLEL, parallel))
        p = prompt_module()
        wording = "".join(p.question_text(d, self.answer) for d in p.DIMENSIONS) + p.DESCRIBE
        from common.splits import digest  # noqa: PLC0415 - dopo ensure_src_on_path

        self.prompt_version = f"{p.SPLIT_VERSION}:{prompt_spec['spec']}:{digest(wording)[:8]}"
        self.model_revision = self.model_name  # il gateway non espone una revisione dei pesi

    # ---------------------------------------------------------------- richieste (identiche a eval_prompt.run_variant)
    def describe_body(self, url: str) -> dict:
        p = prompt_module()
        return {"model": self.model_name, "temperature": 0, "messages": p.describe_messages(url),
                "max_tokens": p.DESCRIBE_MAX_TOKENS}

    def question_body(self, row: dict, url: str, dimension: str, use_context: bool, description: str) -> dict:
        p = prompt_module()
        messages = p.question_messages(row, url, dimension, self.answer, use_context, description)
        return {"model": self.model_name, "temperature": 0, "messages": messages,
                **p.question_constraint(dimension, self.answer)}

    # ---------------------------------------------------------------- una immagine
    def _ensure_env(self) -> None:
        if not os.environ.get("LLM_BASE_URL") and self.default_base_url:
            os.environ["LLM_BASE_URL"] = self.default_base_url

    def predict_one(self, image: StoredImage, use_context: bool) -> ImagePrediction:
        self._ensure_env()
        complete = eval_prompt_module().complete
        p = prompt_module()
        started = time.perf_counter()
        url = images_module().data_url(image.path, self.side)
        preprocess_ms = (time.perf_counter() - started) * 1000
        try:
            choice = complete(self.describe_body(url))["choices"][0]
            description = choice["message"]["content"]
            dims: dict[str, DimensionPrediction] = {}
            for dimension in p.DIMENSIONS:
                body = self.question_body(image.context, url, dimension, use_context, description)
                answer = complete(body)["choices"][0]
                label, probs = p.parse_question(answer, dimension, self.answer)
                if label is None:
                    dims[dimension] = DimensionPrediction(None)
                elif probs:
                    dims[dimension] = DimensionPrediction(label, probs.get(label), top_k(probs), "letter_logprob")
                else:
                    dims[dimension] = DimensionPrediction(label)  # logprob assenti -> confidenza n/a
        except RuntimeError as error:  # errore HTTP non ritentabile o tentativi esauriti (eval_prompt.complete)
            raise ApiError("UPSTREAM_ERROR", "The gateway returned an error for this image.",
                           {"reason": _safe(str(error))}) from None
        total_ms = (time.perf_counter() - started) * 1000
        return ImagePrediction(image.image_id, {d: dims[d] for d in dimensions()}, description,
                               {"preprocess": preprocess_ms, "model": total_ms - preprocess_ms, "total": total_ms})

    # ---------------------------------------------------------------- lotto
    def predict(self, images: list[StoredImage], use_context: bool, batch_size: int,
                on_result: Callable[[ImagePrediction], None], on_error: Callable[[str, Exception], None],
                should_cancel: Callable[[], bool]) -> BatchInfo:
        parallel = max(1, min(MAX_PARALLEL, batch_size or self.parallel))
        started = time.perf_counter()
        done = 0

        def work(image: StoredImage):
            if should_cancel():
                return None
            try:
                return image, self.predict_one(image, use_context), None
            except Exception as error:  # noqa: BLE001 - un'immagine sbagliata non ferma le altre
                return image, None, error

        with ThreadPoolExecutor(parallel) as pool:
            for future in as_completed([pool.submit(work, image) for image in images]):
                outcome = future.result()
                if outcome is None:
                    continue
                image, prediction, error = outcome
                if error is not None:
                    on_error(image.image_id, error)
                else:
                    done += 1
                    on_result(prediction)
        seconds = time.perf_counter() - started
        return BatchInfo(done, seconds, (done / seconds) if done and seconds else None, GATEWAY_NAME, parallel)


def _safe(text: str) -> str:
    """Niente chiavi né percorsi nei messaggi d'errore."""
    key = os.environ.get("LLM_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text[:300]
