"""Compare prompt variants on labeled train images through the Apertus API, scored with metric.py.

Splits are by product_id through a hash, so they are stable on every machine:
  prompt-dev  (bucket 0 of 10): iterate on the prompt here.
  prompt-test (bucket 1 of 10): confirm a decision here, rarely.
Together they are 20% of the products. Keep both out of any finetuning.

Variants are written model:layout:answer:side[:nocontext], for example 8b:describe:names:512. Layouts "split" and
"describe" take the answers "names" or "codes"; "minimal" and "full" take "json" or "codes". Differences to the
first variant come with a paired bootstrap interval over products.

A variant that ends in .csv is a file of predictions made elsewhere, for example by a finetuned model or a visual
adapter on a GPU: columns image_id plus the six dimensions, with a row for every image of the split. It is scored
on the same images and compared like any other variant. Finetuning must leave out buckets 0 and 1.

Raw answers go to data/runs/ (ignored by git) and are reused when the script runs again.

  python src/scripts/eval_prompt.py 8b:minimal:json:512 8b:full:codes:512 --split dev --limit 600
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import images, prompt  # noqa: E402

DATA = ROOT / "data"
BUCKETS = {"dev": 0, "test": 1}


def load_env():
    """Settings from the environment, completed by the git-ignored .env file."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    if not os.environ.get("LLM_API_KEY"):
        sys.exit("LLM_API_KEY is missing: set it in the environment or in .env")


def load_metric():
    spec = importlib.util.spec_from_file_location("metric", DATA / "metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_split(split, limit):
    """The rows of a split. With a limit, whole products are taken in a fixed pseudo-random order."""
    rows = pd.read_csv(DATA / "train.csv", dtype=str, keep_default_na=False)
    rows = rows[rows.product_id.map(lambda p: int(digest(p), 16) % 10 == BUCKETS[split])]
    rows = rows.assign(order=rows.product_id.map(lambda p: digest("order:" + p))).sort_values(["order", "image_id"])
    return (rows.head(limit) if limit else rows).reset_index(drop=True)


def complete(body, attempts=5):
    """One chat completion, with retries on rate limits, server errors and network errors."""
    request = urllib.request.Request(
        os.environ.get("LLM_BASE_URL", "").rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json",
                 "User-Agent": "eval-client/0.1"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429) and error.code < 500:
                raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')[:300]}") from None
            reason = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            reason = type(error).__name__
        if attempt + 1 < attempts:
            time.sleep(2**attempt + random.random())
    raise RuntimeError(f"gave up after {attempts} attempts: {reason}")


def run_variant(spec, rows, concurrency):
    """Predictions per image_id for one variant, calling the API only for images not answered yet."""
    size, style, answer, side, *flags = spec.split(":")
    model, use_context = f"apertus-v1-5-{size}", "nocontext" not in flags
    split_layout = style in ("split", "describe")
    # The folder name changes with the prompt text, so cached answers never mix prompt revisions.
    if split_layout:
        wording = "".join(prompt.question_text(d, answer) for d in prompt.DIMENSIONS)
        wording += prompt.DESCRIBE if style == "describe" else ""
        version = prompt.SPLIT_VERSION
    else:
        wording, version = prompt.system_prompt(style, answer), prompt.PROMPT_VERSION
    folder = DATA / "runs" / f"{spec.replace(':', '_')}_{version}_{digest(wording)[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / "answers.jsonl"
    cached = {}
    if cache_path.exists():
        cached = {r["image_id"]: r for r in map(json.loads, cache_path.read_text().splitlines())}

    def ask(messages, constraint):
        response = complete({"model": model, "temperature": 0, "messages": messages, **constraint})
        return response["choices"][0], response["usage"]

    def predict(row):
        url = images.data_url(DATA / row["image_path"], int(side))
        started = time.time()
        record = {"image_id": row["image_id"]}
        if not split_layout:
            messages = prompt.build_messages(row, url, style, answer, use_context)
            record["choice"], record["usage"] = ask(messages, prompt.output_constraint(answer))
        else:
            usages, description = [], None
            if style == "describe":
                choice, usage = ask(prompt.describe_messages(url), {"max_tokens": prompt.DESCRIBE_MAX_TOKENS})
                description = record["description"] = choice["message"]["content"]
                usages.append(usage)
            record["choices"] = {}
            for dimension in prompt.DIMENSIONS:
                messages = prompt.question_messages(row, url, dimension, answer, use_context, description)
                record["choices"][dimension], usage = ask(messages, prompt.question_constraint(dimension, answer))
                usages.append(usage)
            record["usage"] = {"prompt_tokens": sum(u["prompt_tokens"] for u in usages)}
        record["seconds"] = time.time() - started
        return record

    def parse(record):
        if not split_layout:
            return prompt.parse_choice(record["choice"], answer)
        pairs = {d: prompt.parse_question(record["choices"][d], d, answer) for d in prompt.DIMENSIONS}
        # An invalid dimension stays empty (scored as wrong) without losing the other five.
        labels = {d: label for d, (label, _) in pairs.items()}
        probabilities = {d: p for d, (_, p) in pairs.items()}
        return labels, (probabilities if all(probabilities.values()) else None)

    todo = [row for row in rows.to_dict("records") if row["image_id"] not in cached]
    failures, started = [], time.time()
    with ThreadPoolExecutor(concurrency) as pool, cache_path.open("a") as cache:
        for future in as_completed([pool.submit(predict, row) for row in todo]):
            try:
                record = future.result()
            except Exception as error:  # one bad image must not lose the run; it is reported below
                failures.append(str(error))
                continue
            cached[record["image_id"]] = record
            cache.write(json.dumps(record) + "\n")
    elapsed = time.time() - started
    records = [cached[i] for i in rows.image_id if i in cached]
    parsed = {r["image_id"]: parse(r) for r in records}
    stats = {
        "requested": len(todo), "failed": len(failures),
        "invalid": sum(labels is None or None in labels.values() for labels, _ in parsed.values()),
        "no_probabilities": sum(p is None for _, p in parsed.values()) if answer == "codes" else None,
        "images_per_second": (len(todo) - len(failures)) / elapsed if todo else None,
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records) / max(1, len(records)),
        "latency": sorted(r["seconds"] for r in records)[len(records) // 2] if records else None,
    }
    if failures:
        print(f"{spec}: {len(failures)} failed requests, first error: {failures[0]}")
    return {i: labels for i, (labels, _) in parsed.items() if labels}, stats


def read_predictions(path, rows):
    """Predictions per image_id from a csv made outside this script. Every image of the split must be there."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = {"image_id", *prompt.DIMENSIONS} - set(frame.columns)
    if missing_columns:
        sys.exit(f"{path}: missing columns {sorted(missing_columns)}")
    frame = frame[frame.image_id.isin(set(rows.image_id))]
    missing = len(rows) - frame.image_id.nunique()
    if missing or frame.image_id.duplicated().any():
        sys.exit(f"{path}: {missing} images of the split are missing, or some image_id is repeated")
    return {r["image_id"]: {d: r[d] or None for d in prompt.DIMENSIONS} for r in frame.to_dict("records")}


def solution_frame(rows):
    """The scored cells only: human labels in relevant cells, as described in challenge_description.md."""
    solution = rows[["image_id"]].copy()
    for dimension in prompt.DIMENSIONS:
        scored = (rows[f"{dimension}_source"] == "human") & (rows[f"{dimension}_relevant"].str.lower() == "true")
        solution[dimension] = rows[dimension].where(scored)
    return solution


def submission_frame(rows, predictions):
    return pd.DataFrame(
        [{"image_id": i, **(predictions.get(i) or dict.fromkeys(prompt.DIMENSIONS))} for i in rows.image_id]
    )


def bootstrap_difference(metric, rows, solution, base, other, samples):
    """95% interval of (other - base) in overall macro F1, resampling whole products."""
    rng = random.Random(0)
    positions = list(rows.groupby("product_id").indices.values())
    differences = []
    for _ in range(samples):
        picked = [i for _ in positions for i in positions[rng.randrange(len(positions))]]
        ids = [f"b{n}" for n in range(len(picked))]
        resampled = [frame.iloc[picked].assign(image_id=ids) for frame in (solution, base, other)]
        differences.append(metric.score(resampled[0], resampled[2], "image_id")
                           - metric.score(resampled[0], resampled[1], "image_id"))
    differences.sort()
    return differences[int(0.025 * samples)], differences[int(0.975 * samples)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--split", choices=BUCKETS, default="dev")
    parser.add_argument("--limit", type=int, default=0, help="number of images, 0 for the whole split")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--classes", action="store_true", help="also print the per class F1 of every variant")
    arguments = parser.parse_args()

    if not all(spec.endswith(".csv") for spec in arguments.variants):
        load_env()
    metric = load_metric()
    rows = load_split(arguments.split, arguments.limit)
    solution = solution_frame(rows)
    print(f"split {arguments.split}: {len(rows)} images, {rows.product_id.nunique()} products")

    results = {}
    for spec in arguments.variants:
        if spec.endswith(".csv"):
            predictions, stats = read_predictions(spec, rows), {"source": "csv", "failed": 0}
        else:
            predictions, stats = run_variant(spec, rows, arguments.concurrency)
        if stats["failed"] > 0.01 * len(rows):
            sys.exit(f"{spec}: too many failed requests, not scoring. Run again to retry the missing images.")
        submission = submission_frame(rows, predictions)
        results[spec] = (submission, metric.evaluate(solution, submission, "image_id"), stats)
        print(f"{spec}: {json.dumps(stats)}")

    short = {"subject": "subj", "background": "backg", "perspective": "persp", "individuals": "indiv",
             "text_classification": "textcl", "text_language": "lang"}
    print("\n| variant | macro F1 | weighted | " + " | ".join(short.values()) + " |")
    print("|---" * (3 + len(short)) + "|")
    for spec, (_, scores, _) in results.items():
        cells = [scores["overall"]["f1_macro"], scores["overall"]["f1_weighted"]]
        cells += [scores[d]["f1_macro"] for d in short]
        print(f"| {spec} | " + " | ".join(f"{value:.3f}" for value in cells) + " |")

    first = arguments.variants[0]
    for spec in arguments.variants[1:]:
        if arguments.bootstrap:
            low, high = bootstrap_difference(metric, rows, solution, results[first][0], results[spec][0],
                                             arguments.bootstrap)
            delta = results[spec][1]["overall"]["f1_macro"] - results[first][1]["overall"]["f1_macro"]
            print(f"{spec} minus {first}: {delta:+.3f}, 95% interval [{low:+.3f}, {high:+.3f}]")

    if arguments.classes:
        for spec, (_, scores, _) in results.items():
            print(f"\n{spec}")
            for dimension in short:
                values = sorted(scores[dimension]["per_class"].items(), key=lambda item: item[1]["f1"])
                print(f"  {dimension}: " + ", ".join(
                    f"{name} {v['f1']:.2f} (P {v['precision']:.2f} R {v['recall']:.2f} n {v['support']})"
                    for name, v in values))


if __name__ == "__main__":
    main()
