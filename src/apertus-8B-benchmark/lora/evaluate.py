"""Scores a LoRA adapter on the coverage set (or the prototype validation set) of train.csv with the competition
metric (greedy decoding).

    python evaluate.py --model-dir M --data-dir D --codes codes --adapter run/adapters/final --side 384 --out run

The codes come from encode.py --part <part> --no-mirrors. On coverage it prints the macro F1 on coverage, on "base"
(the natural class distribution) and on "test", as src/scripts/eval_prompt.py does. On prototype_validation
(label-balanced, not representative) it prints that set alone. Writes eval_<name>.json (scores only, no labels) and
predictions_<name>.csv (model answers; never commit it).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import torch

import apertus_lora as L


def solution_frame(rows):
    """The scored cells only: human labels in relevant cells, as in challenge_description.md."""
    solution = rows[["image_id"]].copy()
    for d in L.DIMENSIONS:
        scored = (rows[f"{d}_source"] == "human") & (rows[f"{d}_relevant"].str.lower() == "true")
        solution[d] = rows[d].where(scored)
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--side", type=int, default=384)
    parser.add_argument("--part", default="coverage", choices=("coverage", "prototype_validation"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name", default="final")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tiny", action="store_true", help="random miniature model on CPU, to test the code")
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir)
    sequences = L.Sequences(tokenizer, config.image_token_offset)
    codes = {}
    for path in sorted(Path(args.codes).glob(f"codes_{args.part}_{args.side}_shard*.pkl")):
        codes.update(pickle.loads(path.read_bytes()))
    rows = L.select_rows(L.read_train(args.data_dir), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    records = rows.to_dict("records")
    prompts = {}
    for record in records:
        if record["image_id"] in codes:
            grid_h, grid_w, image_codes = codes[record["image_id"]]
            prompts[record["image_id"]] = sequences.prompt_ids(record, grid_h, grid_w, image_codes, not args.no_context)
    print(f"{len(records)} {args.part} rows, {len(prompts)} with codes", flush=True)

    if args.tiny:
        from train import load_model

        model = load_model(argparse.Namespace(tiny=True, model_dir=args.model_dir), 0)
    else:
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=L.ATTENTION
        )
        model.model.vision_tokenizer = None
        model.model.audio_tokenizer = None
    device = "cpu" if args.tiny else "cuda"
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    order = sorted(prompts, key=lambda i: len(prompts[i]))
    answers, started = {}, time.time()
    for start in range(0, len(order), args.batch):
        ids = order[start : start + args.batch]
        batch = L.collate([{"input_ids": prompts[i], "labels": prompts[i]} for i in ids], tokenizer.pad_token_id)
        with torch.inference_mode():
            output = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=96,
                do_sample=False,
                eos_token_id=sequences.end,
                pad_token_id=tokenizer.pad_token_id,
            )
        for image_id, generated in zip(ids, output[:, batch["input_ids"].shape[1] :]):
            answers[image_id] = tokenizer.decode(generated, skip_special_tokens=True)
    seconds = time.time() - started
    print(f"{len(answers)} answers in {seconds:.0f} s, {len(answers) / seconds:.1f} images/s (HF generate, batch "
          f"{args.batch}, not optimised for speed)", flush=True)

    predictions = {i: L.parse_answer(a) for i, a in answers.items()}
    invalid = sum(p is None for p in predictions.values()) + len(records) - len(predictions)
    submission = pd.DataFrame(
        [{"image_id": r["image_id"], **(predictions.get(r["image_id"]) or dict.fromkeys(L.DIMENSIONS))} for r in records]
    )
    metric = L.load_metric(args.data_dir)
    views = {args.part: rows.index}
    if args.part == "coverage":
        buckets = rows.product_id.map(L.splits.bucket)
        views["base (natural distribution)"] = rows.index[buckets.isin(L.splits.BUCKETS["base"])]
        views["test only"] = rows.index[buckets.isin(L.splits.BUCKETS["test"])]
    report = {"adapter": args.adapter, "part": args.part, "side": args.side, "invalid_answers": invalid,
              "images_per_second":
              len(answers) / seconds, "views": {}}
    for view, index in views.items():
        part = rows.loc[index].reset_index(drop=True)
        scores = metric.evaluate(solution_frame(part), submission.loc[index].reset_index(drop=True), "image_id")
        report["views"][view] = scores
        cells = " | ".join(f"{d} {scores[d]['f1_macro']:.3f}" for d in L.DIMENSIONS if d in scores)
        print(f"{view}: {len(part)} images, macro F1 {scores['overall']['f1_macro']:.4f}, weighted "
              f"{scores['overall']['f1_weighted']:.4f} | {cells}", flush=True)
    first = report["views"][args.part]
    weakest = sorted((v["f1"], f"{d}/{c}", v["support"]) for d in L.DIMENSIONS if d in first
                     for c, v in first[d]["per_class"].items())[:10]
    print("weakest classes:", ", ".join(f"{n} {f:.2f} (n={s})" for f, n, s in weakest), flush=True)
    print(f"invalid answers: {invalid}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.name}.json").write_text(json.dumps(report, indent=2, default=str))
    submission.to_csv(out / f"predictions_{args.name}.csv", index=False)


if __name__ == "__main__":
    main()
