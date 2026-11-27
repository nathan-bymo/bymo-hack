"""Zero-shot classification of the Galaxus images with an Apertus v1.5 model behind the Keera gateway.

The script sends one chat request per image (system prompt from system_prompt.txt, the image plus the product
context in the user message), writes one JSON line per answer, and scores a holdout with the official metric.py.

    .venv/bin/python src/scripts/keera_classify.py --probe                 # one image: auth, images, JSON, logprobs
    .venv/bin/python src/scripts/keera_classify.py --part holdout --limit 50
    .venv/bin/python src/scripts/keera_classify.py --part holdout          # 600 images, same split as the Qwen notebook
    .venv/bin/python src/scripts/keera_classify.py --part dev              # 5,261 images + submission.csv (not uploaded)

LLM_BASE_URL, LLM_NAME and LLM_API_KEY come from the environment or from the .env at the repository root.
Runs are resumable: image ids already in the output file are skipped. Outputs go to data/runs/, which is gitignored.
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
from openai import APIStatusError, AsyncOpenAI, BadRequestError
from PIL import Image, ImageOps
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "hackapertus-galaxus"
RUNS_DIR = REPO_ROOT / "data" / "runs"
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")

sys.path.insert(0, str(DATA_DIR))
import metric  # noqa: E402  the leaderboard metric, from the competition files

DIMENSIONS = metric.DIMENSIONS
CLASSES = {dimension: sorted(metric.VALID_CLASSES[dimension]) for dimension in DIMENSIONS}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {dimension: {"type": "string", "enum": CLASSES[dimension]} for dimension in DIMENSIONS},
    "required": list(DIMENSIONS),
    "additionalProperties": False,
}

# The holdout of the reference notebook, so the scores are comparable with its Qwen 3.5 9B run (0.899).
HOLDOUT_IMAGES = 600
SEED = 3407

# Apertus v1.5 makes one visual token per 16 x 16 pixels.
PATCH_SIZE = 16
MAX_SIDE_FACTOR = 8
JPEG_QUALITY = 92


def load_env():
    """Fill os.environ from the .env at the repository root, without overriding variables already set."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def read_part(csv_path):
    """A csv of the competition as strings, with the full path of each image."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame["image_file"] = [str(csv_path.parent / image_path) for image_path in frame["image_path"]]
    return frame


def holdout_rows():
    """The 600-image holdout of whole products, split exactly as in the reference notebook."""
    labeled = read_part(DATA_DIR / "train.csv")
    images_per_product = labeled["product_id"].value_counts()
    products = sorted(images_per_product.index)
    random.Random(SEED).shuffle(products)
    chosen, count = set(), 0
    for product_id in products:
        if count >= HOLDOUT_IMAGES:
            break
        chosen.add(product_id)
        count += images_per_product[product_id]
    return labeled[labeled["product_id"].isin(chosen)].reset_index(drop=True)


def opaque_rgb_on_white(image):
    """RGB, with any transparency composited onto white (the function of the Kaggle data page)."""
    if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def resized(image, image_size):
    """The image scaled to about image_size squared pixels, aspect ratio kept, each side a multiple of 16."""
    scale = image_size / (image.width * image.height) ** 0.5
    scale *= min(1.0, MAX_SIDE_FACTOR * image_size / (max(image.width, image.height) * scale))
    width = max(PATCH_SIZE, round(image.width * scale / PATCH_SIZE) * PATCH_SIZE)
    height = max(PATCH_SIZE, round(image.height * scale / PATCH_SIZE) * PATCH_SIZE)
    return image.resize((width, height), Image.LANCZOS)


def image_data_url(image_file, image_size, mirrored=False):
    """The prepared image as a base64 JPEG data URL."""
    with Image.open(image_file) as image:
        rgb = opaque_rgb_on_white(image)
    if mirrored:
        rgb = ImageOps.mirror(rgb)
    buffer = io.BytesIO()
    resized(rgb, image_size).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def user_text(row, with_context):
    """The product context of an image, or a bare instruction without it."""
    if not with_context:
        return "No product context is available. Classify the image."
    return (
        f"Product: {row['product_title']}\nBrand: {row['brand']}\nProduct type: {row['product_type']}\n"
        f"Category: {row['category']} / {row['subcategory']}"
    )


def parse_answer(answer):
    """The class per dimension of a model answer, or None when it is not a valid answer."""
    if not answer:
        return None
    try:
        parsed = json.loads(answer[answer.find("{") : answer.rfind("}") + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict) or any(parsed.get(dimension) not in CLASSES[dimension] for dimension in DIMENSIONS):
        return None
    return {dimension: parsed[dimension] for dimension in DIMENSIONS}


class Classifier:
    """Sends the chat requests, with structured output when the gateway accepts it."""

    def __init__(self, client, model, system_prompt, args):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.args = args
        self.use_schema = not args.no_schema

    def messages(self, row):
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(row["image_file"], self.args.image_size)}},
                    {"type": "text", "text": user_text(row, not self.args.no_context)},
                ],
            },
        ]

    async def complete(self, messages, **extra):
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.args.max_tokens,
            **extra,
        }
        if self.use_schema:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "image_classes", "schema": RESPONSE_SCHEMA, "strict": True},
            }
        try:
            return await self.client.chat.completions.create(**request)
        except BadRequestError as error:
            message = str(error).lower()
            if self.use_schema and any(word in message for word in ("response_format", "json_schema", "structured")):
                tqdm.write(f"the gateway refused response_format, continuing without it: {error}")
                self.use_schema = False
                return await self.complete(messages, **extra)
            raise

    async def classify(self, row):
        """One output record for an image: the answer, its parse, timing, usage or the error."""
        record = {"image_id": row["image_id"]}
        started = time.perf_counter()
        try:
            messages = await asyncio.to_thread(self.messages, row)
            response = await self.complete(messages)
            choice = response.choices[0]
            record["answer"] = choice.message.content
            record["finish_reason"] = choice.finish_reason
            if response.usage:
                record["prompt_tokens"] = response.usage.prompt_tokens
                record["completion_tokens"] = response.usage.completion_tokens
        except Exception as error:  # noqa: BLE001  one bad image or network failure must not stop a 5,000-image run
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        record["latency_s"] = round(time.perf_counter() - started, 3)
        record["prediction"] = parse_answer(record.get("answer"))
        return record


async def probe(classifier, rows):
    """One request on one image, printed in full, plus a check whether logprobs come back."""
    row = rows.iloc[0]
    messages = await asyncio.to_thread(classifier.messages, row)
    started = time.perf_counter()
    response = await classifier.complete(messages)
    print(f"model {response.model}, {time.perf_counter() - started:.2f} s, structured output: {classifier.use_schema}")
    print(f"usage: {response.usage}")
    print(f"finish_reason: {response.choices[0].finish_reason}")
    print(f"answer: {response.choices[0].message.content}")
    print(f"parsed: {parse_answer(response.choices[0].message.content)}")
    try:
        with_logprobs = await classifier.complete(messages, logprobs=True, top_logprobs=5)
        content = with_logprobs.choices[0].logprobs and with_logprobs.choices[0].logprobs.content
        print(f"logprobs: {'returned, ' + str(len(content)) + ' tokens' if content else 'not returned'}")
    except APIStatusError as error:
        print(f"logprobs: refused ({error.status_code}: {str(error)[:200]})")


async def run(classifier, rows, output_path, concurrency):
    """Classifies every row not yet in output_path and appends the records. Returns the wall time in seconds."""
    done = set()
    if output_path.exists():
        done = {json.loads(line)["image_id"] for line in output_path.read_text().splitlines() if line.strip()}
    todo = [row for row in rows.to_dict("records") if row["image_id"] not in done]
    print(f"{len(rows)} images, {len(done)} already done, {len(todo)} to classify, concurrency {concurrency}")
    if not todo:
        return 0.0
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(row):
        async with semaphore:
            return await classifier.classify(row)

    started = time.perf_counter()
    errors = 0
    with output_path.open("a") as output, tqdm(total=len(todo), desc="images") as progress:
        for future in asyncio.as_completed([limited(row) for row in todo]):
            record = await future
            output.write(json.dumps(record) + "\n")
            output.flush()
            errors += "error" in record
            progress.update()
            progress.set_postfix(errors=errors)
    return time.perf_counter() - started


def read_records(output_path, image_ids):
    """The last record per image id of a run file, for the given image ids."""
    records = {}
    for line in output_path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["image_id"]] = record
    return {image_id: records[image_id] for image_id in image_ids if image_id in records}


def summarize(records, wall_seconds, concurrency):
    """Throughput, latency, tokens and failures of a run."""
    latencies = sorted(record["latency_s"] for record in records.values() if "error" not in record)
    errors = [record for record in records.values() if "error" in record]
    invalid = [record for record in records.values() if "error" not in record and record["prediction"] is None]
    print(f"\n{len(records)} records: {len(errors)} errors, {len(invalid)} answers that do not parse")
    if latencies:
        p95 = latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)]
        print(f"latency per request: median {statistics.median(latencies):.2f} s, p95 {p95:.2f} s")
    if wall_seconds:
        print(f"this session: {len(latencies)} ok in {wall_seconds:.0f} s at concurrency {concurrency}, "
              f"{len(latencies) / wall_seconds:.2f} images/s")
    prompt_tokens = [record["prompt_tokens"] for record in records.values() if record.get("prompt_tokens")]
    if prompt_tokens:
        completion = [record.get("completion_tokens") or 0 for record in records.values() if record.get("prompt_tokens")]
        print(f"tokens per image: prompt median {statistics.median(prompt_tokens):.0f}, "
              f"completion median {statistics.median(completion):.0f}")
    for record in errors[:3]:
        print(f"  error {record['image_id']}: {record['error']}")
    for record in invalid[:3]:
        print(f"  invalid {record['image_id']} ({record.get('finish_reason')}): {str(record.get('answer'))[:200]!r}")


def score_holdout(rows, records):
    """metric.py on the holdout: human labels in relevant cells only, as on Kaggle."""
    solution = pd.DataFrame({"image_id": rows["image_id"]})
    for dimension in DIMENSIONS:
        scored = (rows[f"{dimension}_source"] == "human") & (rows[f"{dimension}_relevant"] == "true")
        solution[dimension] = rows[dimension].where(scored)
    submission = pd.DataFrame(
        [
            {"image_id": image_id, **((records.get(image_id) or {}).get("prediction") or dict.fromkeys(DIMENSIONS))}
            for image_id in rows["image_id"]
        ]
    )
    metrics = metric.evaluate(solution, submission, "image_id")
    missing = sum((records.get(image_id) or {}).get("prediction") is None for image_id in rows["image_id"])
    print(f"\nholdout, {len(rows)} images, {missing} without a valid answer (counted as wrong)\n")
    print(f"{'':20} {'F1 macro':>9} {'F1 weighted':>12} {'cells':>6}")
    for dimension in [dimension for dimension in DIMENSIONS if dimension in metrics] + ["overall"]:
        values = metrics[dimension]
        print(f"{dimension:20} {values['f1_macro']:9.3f} {values['f1_weighted']:12.3f} {values['support']:6}")
    weakest = sorted(
        (values["f1"], f"{dimension}/{class_name}", values["support"])
        for dimension in DIMENSIONS
        if dimension in metrics
        for class_name, values in metrics[dimension]["per_class"].items()
    )[:10]
    print("\nweakest classes:", ", ".join(f"{name} {f1:.2f} (n={support})" for f1, name, support in weakest))
    return metrics


def write_submission(records, output_path):
    """submission.csv: sample_submission.csv with the predicted dev rows replaced. Not uploaded."""
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", dtype=str, keep_default_na=False)
    submission = sample.set_index("image_id")
    predicted = {image_id: record["prediction"] for image_id, record in records.items() if record.get("prediction")}
    ids = [image_id for image_id in submission.index if image_id in predicted]
    submission.loc[ids, list(DIMENSIONS)] = [[predicted[image_id][dimension] for dimension in DIMENSIONS] for image_id in ids]
    submission = submission.reset_index()
    for dimension in DIMENSIONS:
        assert submission[dimension].isin(CLASSES[dimension]).all(), f"{dimension} has an empty or unknown class"
    submission.to_csv(output_path, index=False)
    print(f"\n{output_path}: {len(ids)} predicted rows, the others keep the constants of sample_submission.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=("holdout", "dev"), default="holdout")
    parser.add_argument("--limit", type=int, help="only the first N images of the part, for a quick test")
    parser.add_argument("--image-size", type=int, default=512, help="about N x N pixels, sides multiples of 16")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--no-context", action="store_true", help="leave the product context out of the prompt")
    parser.add_argument("--no-schema", action="store_true", help="do not ask for structured JSON output")
    parser.add_argument("--probe", action="store_true", help="one request, printed in full, and a logprobs check")
    parser.add_argument("--tag", default="", help="suffix of the run name")
    args = parser.parse_args()

    load_env()
    missing = [name for name in ("LLM_BASE_URL", "LLM_NAME", "LLM_API_KEY") if not os.environ.get(name)]
    if missing:
        sys.exit(f"missing {', '.join(missing)}: set them in the environment or in {REPO_ROOT / '.env'}")
    model = os.environ["LLM_NAME"]
    client = AsyncOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"], max_retries=4, timeout=120)
    classifier = Classifier(client, model, SYSTEM_PROMPT_PATH.read_text().strip(), args)

    rows = holdout_rows() if args.part == "holdout" else read_part(DATA_DIR / "dev.csv")
    if args.limit:
        rows = rows.iloc[: args.limit].reset_index(drop=True)

    if args.probe:
        asyncio.run(probe(classifier, rows))
        return

    run_name = f"{args.part}_{args.image_size}px" + ("_noctx" if args.no_context else "") + (f"_{args.tag}" if args.tag else "")
    run_dir = RUNS_DIR / model
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_name}.jsonl"
    print(f"model {model} at {os.environ['LLM_BASE_URL']}, run {output_path}")

    wall_seconds = asyncio.run(run(classifier, rows, output_path, args.concurrency))
    records = read_records(output_path, rows["image_id"])
    summarize(records, wall_seconds, args.concurrency)
    if args.part == "holdout":
        metrics = score_holdout(rows, records)
        (run_dir / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    else:
        write_submission(records, run_dir / f"{run_name}_submission.csv")


if __name__ == "__main__":
    main()
