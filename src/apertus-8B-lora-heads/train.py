"""Trains LoRA + 6 heads together on the cached inputs of encode.py (one GPU, bf16).

  small split   python src/apertus-8B-lora-heads/train.py --train-csv data/nathan_split/train_1500.csv \\
                    --cache data/lh_cache/small_256 --out data/runs/lh_small
  phase 1       python src/apertus-8B-lora-heads/train.py --part training --cache data/lh_cache/training_256 \\
                    --out data/runs/lh_phase1
  phase 2       python src/apertus-8B-lora-heads/train.py --part all --cache data/lh_cache/all_256 --out ...

The defaults copy the settings of the LoRA baseline (apertus-8B-benchmark/lora), so that the main difference is the
output: heads instead of generated JSON. Items (rare class repeats and mirrors), LoRA rank 32 with the same target
modules, learning rate 2e-4, weight decay 0.01, 3 epochs, 8 x 2 images per step, clipping 1.0, the same warmup and
cosine schedule, the loss on human labels. The heads learn at 1e-3 with the plain cross entropy.

Nothing is selected during the run: the model after the last step is the result. With --val-fraction > 0 a part of
the training products is held out, the best checkpoint on it is saved too, and the result is that one. The coverage
set is never used here: predict.py scores it afterwards.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import encoders as E  # noqa: E402
import model as M  # noqa: E402
from encode import load_cache  # noqa: E402

ROOT = HERE.parents[1]
PROBE = (10, 30)  # steps between which the step time is measured (after the first, slower, steps)


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def peak_memory_gb():
    return torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0


def upload(repo, folder, path_in_repo, required=False):
    """Copies a folder to a private HF repo and checks that every file arrived.

    Snapshots are not fatal. The final model is (required=True): a job disk disappears with the job.
    """
    if not repo:
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo,
                          ignore_patterns=["README.md"], commit_message=f"upload {path_in_repo}")
        remote = set(api.list_repo_files(repo))
        local = {f"{path_in_repo}/{p.relative_to(folder).as_posix()}" for p in Path(folder).rglob("*")
                 if p.is_file() and p.name != "README.md"}
        if local - remote:
            raise RuntimeError(f"files missing after the upload: {sorted(local - remote)[:3]}")
        log(f"uploaded and checked {path_in_repo} ({len(local)} files)")
    except Exception as error:
        message = f"upload of {path_in_repo} failed: {type(error).__name__}: {str(error)[:200]}"
        if required:
            raise RuntimeError(message) from None
        log(message)


class Batches:
    """Turns (row index, variant) items into model inputs from the cache."""

    def __init__(self, records, labels, cache, builder, pad_id, context_dropout=0.0, seed=0):
        self.records, self.labels, self.cache = records, labels, cache
        self.builder, self.pad_id = builder, pad_id
        self.context_dropout, self.rng = context_dropout, random.Random(seed)

    def __call__(self, items, train):
        sequences, labels = [], []
        for index, variant in items:
            entry = self.cache[D.key(self.records[index]["image_id"], variant)]
            use_context = not (train and self.rng.random() < self.context_dropout)
            sequences.append(self.builder.ids(self.records[index], entry["grid"], entry["codes"], use_context))
            labels.append(D.mirror_labels(self.labels[index]) if D.is_mirrored(variant) else self.labels[index])
        input_ids, attention = E.collate(sequences, self.pad_id)
        return input_ids, attention, torch.stack(labels)


@torch.inference_mode()
def evaluate(classifier, batches, rows, metric, batch_size, device):
    classifier.eval()
    parts = [[] for _ in D.DIMENSIONS]
    items = [(i, "") for i in range(len(rows))]
    for start in range(0, len(items), batch_size):
        input_ids, attention, _ = batches(items[start:start + batch_size], train=False)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = classifier(input_ids.to(device), attention.to(device))
        for task, task_logits in enumerate(logits):
            parts[task].extend(task_logits.argmax(-1).tolist())
    classifier.train()
    predictions = rows[["image_id"]].copy()
    for task, dimension in enumerate(D.DIMENSIONS):
        predictions[dimension] = [D.TAXONOMY[dimension][i] for i in parts[task]]
    results = metric.evaluate(D.solution_frame(rows), predictions, "image_id")
    return results["overall"]["f1_macro"], {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--cache", type=Path, required=True, help="folder written by encode.py")
    p.add_argument("--out", type=Path, required=True, help="inside data/ (git-ignored)")
    p.add_argument("--part", default="training", choices=("training", "all"))
    p.add_argument("--train-csv", type=Path, help="a smaller split instead of --part; never touches coverage")
    p.add_argument("--val-fraction", type=float, default=0.0, help="training products held out to pick a checkpoint")
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="only the first N rows, for a smoke test")
    p.add_argument("--max-steps", type=int, default=0, help="for a smoke test")
    p.add_argument("--max-minutes", type=float, default=0,
                   help="training budget: the schedule is shortened to fit it, so the learning rate still reaches 0")
    p.add_argument("--layers", type=int, default=32, help="Apertus layers kept (of 32)")
    p.add_argument("--lora-rank", type=int, default=32, help="0: no LoRA (frozen Apertus)")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--head-hidden", type=int, default=512)
    p.add_argument("--head-dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=float, default=3.0, help="may be fractional")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-heads", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--label-cells", default="human", choices=D.LABEL_CELLS,
                   help="human: as the LoRA baseline; scored: human labels in relevant cells only")
    p.add_argument("--gamma", type=float, default=0.0, help="focal loss; 0 = cross entropy")
    p.add_argument("--class-alpha", default="none", choices=("none", "effective"))
    p.add_argument("--no-repeats", action="store_true")
    p.add_argument("--no-mirrors", action="store_true")
    p.add_argument("--context-dropout", type=float, default=0.0, help="share of examples without context")
    p.add_argument("--grad-ckpt", action="store_true", help="less memory, slower")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda")
    p.add_argument("--upload-repo", default="", help="private HF repo that receives the checkpoints")
    p.add_argument("--upload-prefix", default="lora-heads")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    all_rows = D.read_csv(args.data_dir / "train.csv")
    rows = D.read_csv(args.train_csv) if args.train_csv else D.select(all_rows, args.part)
    if args.train_csv or args.part == "training":
        D.assert_clean(rows, all_rows)
    if args.limit:
        rows = rows.head(args.limit)
    fit, validation = D.internal_split(rows, args.val_fraction)
    fit_labels = D.label_matrix(fit, args.label_cells)
    items = D.training_items(fit, repeats=not args.no_repeats, mirrors=not args.no_mirrors)
    cache = load_cache(args.cache)
    needed = {D.key(fit.image_id[i], v) for i, v in items} | {D.key(i) for i in validation.image_id}
    missing = needed - set(cache)
    if missing:
        sys.exit(f"{len(missing)} images have no cached codes: run encode.py on the same rows first")
    log(f"{len(fit)} images to train, {len(items)} items per epoch with repeats and mirrors, "
        f"{len(validation)} held out; loss on {args.label_cells} cells")

    from transformers import AutoConfig, AutoTokenizer

    directory = E.model_dir()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset)
    settings = {name: getattr(args, name) for name in
                ("layers", "lora_rank", "lora_alpha", "lora_dropout", "head_hidden", "head_dropout")}
    settings |= {"side": json.loads((args.cache / "meta.json").read_text())["side"], "model": E.MODEL_ID,
                 "revision": E.MODEL_REVISION, "task": E.TASK}
    classifier = M.build(directory, settings, device)
    if args.grad_ckpt:
        classifier.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        classifier.language_model.enable_input_require_grads()
    classifier.train()

    lora = [p for p in classifier.language_model.parameters() if p.requires_grad]
    groups = [{"params": list(classifier.heads.parameters()), "lr": args.lr_heads, "weight_decay": 0.05}]
    if lora:
        groups.insert(0, {"params": lora, "lr": args.lr_lora, "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(groups)
    trainable = [p for group in groups for p in group["params"]]
    log(f"{args.layers} layers, LoRA r{args.lora_rank}: {sum(p.numel() for p in trainable) / 1e6:.1f} M trainable "
        "parameters")

    steps_per_epoch = math.ceil(len(items) / (args.batch_size * args.accumulation))
    total = args.max_steps or max(1, math.ceil(steps_per_epoch * args.epochs))
    plan = {"total": total, "warmup": max(1, round(args.warmup_ratio * min(total, 2 * steps_per_epoch)))}

    def factor(step):
        if step < plan["warmup"]:
            return (step + 1) / plan["warmup"]
        progress = min(1.0, (step - plan["warmup"]) / max(1, plan["total"] - plan["warmup"]))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    alphas = D.class_alpha(D.item_labels(fit_labels, items)) if args.class_alpha == "effective" else None
    batches = Batches(fit.to_dict("records"), fit_labels, cache, builder, tokenizer.pad_token_id,
                      args.context_dropout, args.seed)
    held_out = Batches(validation.to_dict("records"), D.label_matrix(validation), cache, builder, tokenizer.pad_token_id)
    metric = D.load_metric(args.data_dir) if len(validation) else None
    eval_every = max(1, steps_per_epoch // max(1, args.evals_per_epoch))
    log(f"{steps_per_epoch} steps per epoch, {total} steps, warmup {plan['warmup']}")

    args.out.mkdir(parents=True, exist_ok=True)
    history, best, step, stop, times = [], None, 0, False, {}
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(args.seed)
    epoch = 0
    while not stop:
        epoch += 1
        order = [items[i] for i in torch.randperm(len(items), generator=generator).tolist()]
        micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for start in range(0, len(micro), args.accumulation):
            group, seen = micro[start:start + args.accumulation], []
            for chunk in group:
                input_ids, attention, labels = batches(chunk, train=True)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = classifier(input_ids.to(device), attention.to(device))
                losses = M.task_losses(logits, labels.to(device), args.gamma, alphas)
                if all(loss is None for loss in losses):
                    continue  # no labeled cell in this micro-batch
                (sum(loss for loss in losses if loss is not None) / len(group)).backward()
                seen.append([None if loss is None else loss.item() for loss in losses])
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
            if not torch.isfinite(norm):
                sys.exit(f"step {step + 1}: the gradient norm is {norm.item()}, stopping")
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            times[step] = time.time()
            if step == PROBE[1] and args.max_minutes and not args.max_steps:
                per_step = (times[PROBE[1]] - times[PROBE[0]]) / (PROBE[1] - PROBE[0])
                left = args.max_minutes * 60 - (time.time() - started)
                fitted = max(step + 1, min(plan["total"], step + int(left / per_step)))
                log(f"{per_step:.2f} s per step: {fitted} of {plan['total']} steps fit in {args.max_minutes} min")
                plan["total"] = total = fitted
            if step % 10 == 0 or step == 1:
                per_task = [sum(v[t] for v in seen if v[t] is not None) / max(1, sum(v[t] is not None for v in seen))
                            for t in range(len(D.DIMENSIONS))]
                rate = step * args.batch_size * args.accumulation / (time.time() - started)
                log(f"epoch {epoch} step {step}/{total} loss " + " ".join(f"{x:.3f}" for x in per_task)
                    + f" | {rate:.1f} items/s, peak {peak_memory_gb():.1f} GB")
            stop = step >= total or bool(args.max_minutes and time.time() - started > args.max_minutes * 60 * 1.05)
            last_of_epoch = start + args.accumulation >= len(micro)
            if len(validation) and (step % eval_every == 0 or last_of_epoch or stop):
                score, per_dimension = evaluate(classifier, held_out, validation, metric, 2 * args.batch_size, device)
                history.append({"step": step, "macro_f1": score, "dimensions": per_dimension})
                log(f"held out, step {step}: macro F1 {score:.4f}")
                if best is None or score > best["macro_f1"]:
                    best = history[-1]
                    M.save(classifier, settings, args.out / "best")
                    upload(args.upload_repo, args.out / "best", f"{args.upload_prefix}/best")
            if stop:
                break
        if not len(validation) and not stop:
            M.save(classifier, settings, args.out / "snapshot")  # a stopped job still has the last full epoch
            upload(args.upload_repo, args.out / "snapshot", f"{args.upload_prefix}/snapshot")

    result = args.out / ("best" if len(validation) else "final")
    if not len(validation):
        M.save(classifier, settings, result)
    summary = {"rows": len(rows), "fit_images": len(fit), "held_out_images": len(validation),
               "items_per_epoch": len(items), "steps": step, "steps_per_epoch": steps_per_epoch,
               "minutes": round((time.time() - started) / 60, 1), "best": best, "history": history,
               "peak_memory_gb": round(peak_memory_gb(), 1), "settings": settings,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (result / "train_summary.json").write_text(json.dumps(summary, indent=2))
    upload(args.upload_repo, result, f"{args.upload_prefix}/{result.name}", required=True)
    log(f"done: {step} steps in {summary['minutes']} min, model in {result}")
    return result


if __name__ == "__main__":
    main()
