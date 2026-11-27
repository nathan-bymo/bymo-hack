"""Worker GPU della demo: un piccolo server HTTP per famiglia di modelli (piano B.5.2).

    python -m demo_worker.server --family lora_heads --port 8012

Su `127.0.0.1` non serve il token; su altri host è obbligatorio (`DEMO_WORKER_TOKEN`, header `X-Demo-Worker-Token`).
Il worker non scarica mai pesi. `--dry` serve solo ai test del protocollo e richiede DEMO_WORKER_ALLOW_DRY=1.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .families import get_family
from .families.base import REPO_ROOT, WorkerError


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def create_app(family_name: str, dry: bool = False, token: str | None = None) -> FastAPI:
    family = get_family(family_name, dry)
    lock = threading.Lock()  # un solo lavoro alla volta sulla GPU
    app = FastAPI(title=f"demo worker {family_name}", docs_url=None, openapi_url=None)

    def check(provided: str | None) -> None:
        if token and provided != token:
            raise HTTPException(status_code=401, detail="bad token")

    @app.exception_handler(WorkerError)
    async def _worker_error(request: Request, exc: WorkerError):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, status_code=409)

    @app.get("/worker/info")
    def info(x_demo_worker_token: str | None = Header(default=None)):
        check(x_demo_worker_token)
        return {**family.info(), "family": family_name, "code_commit": _commit(), "loaded": family.loaded()}

    @app.post("/worker/load")
    def load(body: dict, x_demo_worker_token: str | None = Header(default=None)):
        check(x_demo_worker_token)
        messages: queue.Queue = queue.Queue()

        def run():
            with lock:
                try:
                    out = family.load(body["model_id"], body, lambda s, m: messages.put({"state": s, "message": m}))
                    messages.put({"state": "ready", "message": "loaded", **out})
                except WorkerError as error:
                    messages.put({"state": "error", "message": error.message, "code": error.code, "details": error.details})
                except Exception as error:  # noqa: BLE001
                    code = "GPU_OOM" if type(error).__name__ == "OutOfMemoryError" else "INTERNAL_ERROR"
                    messages.put({"state": "error", "message": f"{type(error).__name__}", "code": code})
                messages.put(None)

        threading.Thread(target=run, daemon=True).start()

        def stream():
            while (item := messages.get()) is not None:
                yield json.dumps(item) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/worker/unload")
    def unload(body: dict, x_demo_worker_token: str | None = Header(default=None)):
        check(x_demo_worker_token)
        with lock:
            family.unload(body["model_id"])
        return {"state": "not_loaded"}

    @app.post("/worker/predict")
    def predict(body: dict, x_demo_worker_token: str | None = Header(default=None)):
        check(x_demo_worker_token)
        with lock:
            results, batch = family.predict(body["model_id"], body["images"], bool(body.get("use_context", True)),
                                            int(body.get("batch_size") or 8))
        return {"results": results, "batch": batch}

    return app


def main() -> None:
    import os

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", required=True, choices=("lora_generative", "lora_heads", "fullft_heads", "vadapter_heads"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dry", action="store_true", help="solo test del protocollo (DEMO_WORKER_ALLOW_DRY=1)")
    args = parser.parse_args()
    token = os.environ.get("DEMO_WORKER_TOKEN")
    if args.host not in ("127.0.0.1", "localhost") and not token:
        raise SystemExit("DEMO_WORKER_TOKEN è obbligatorio quando il worker non è su 127.0.0.1")
    uvicorn.run(create_app(args.family, args.dry, token), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
