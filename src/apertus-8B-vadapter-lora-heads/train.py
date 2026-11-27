"""Trains projector + LoRA + 6 heads end to end on the cached inputs of encode.py (one GPU, bf16: H200/H100/L40S).

  phase 1  python src/apertus-8B-vadapter-lora-heads/train.py --cache data/vlh_cache/training_256 \
               --out data/runs/vlh_phase1
  phase 2  python src/apertus-8B-vadapter-lora-heads/train.py --part all --val-fraction 0 --epochs <best epoch> \
               --cache data/vlh_cache/all_256 --out data/runs/vlh_phase2

Phase 1 trains on common.splits.training_rows minus the internal validation (--val-fraction of the products, by
hash). The validation picks the checkpoint (macro F1 of metric.py, evaluated --evals-per-epoch times per epoch);
the coverage set is never seen here: predict.py scores it at the end. Phase 2 trains on every labeled image with
the settings and the number of epochs chosen in phase 1, and keeps the last checkpoint (nothing to select on).

Loss: per head, focal loss over the scored cells only (human label in a relevant cell), optional per-class alpha.
Rare classes are repeated (up to 4x per epoch), each repeat a different variant (mild crop, mirrored when there is
no text); left/right perspectives without text get one more mirrored copy. See DATI.md.
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


def upload(repo, folder, path_in_repo, patterns=None):
    """Copies a checkpoint to a private HF repo, so that a stopped job loses nothing. Never fatal."""
    if not repo:
        return
    try:
        from huggingface_hub import HfApi

        # README.md: the card PEFT writes holds a local base-model path, which the Hub rejects as metadata.
        HfApi().upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo,
                              allow_patterns=patterns, ignore_patterns=["README.md"],
                              commit_message=f"upload {path_in_repo}")
        log(f"uploaded {path_in_repo} to {repo}")
    except Exception as error:
        log(f"upload of {path_in_repo} failed: {type(error).__name__}: {str(error)[:200]}")


class Batches:
    """Turns (row index, variant) items into model inputs from the cache."""

    def __init__(self, records, labels, cache, builder, pad_id, context_dropout=0.0, seed=0):
        self.records, self.labels, self.cache = records, labels, cache
        self.builder, self.pad_id = builder, pad_id
        self.context_dropout, self.rng = context_dropout, random.Random(seed)

    def __call__(self, items, train):
        sequences, siglip, labels = [], [], []
        for index, variant in items:
            entry = self.cache[D.key(self.records[index]["image_id"], variant)]
            use_context = not (train and self.rng.random() < self.context_dropout)
            sequences.append(self.builder.ids(self.records[index], entry["grid"], entry["codes"], use_context))
            siglip.append(entry["siglip"])
            labels.append(D.mirror_labels(self.labels[index]) if D.is_mirrored(variant) else self.labels[index])
        input_ids, attention = E.collate(sequences, self.pad_id)
        return input_ids, attention, torch.stack(siglip), torch.stack(labels)


@torch.inference_mode()
def evaluate(classifier, batches, rows, metric, batch_size, device):
    classifier.eval()
    items = [(i, "") for i in range(len(rows))]
    parts = [[] for _ in D.DIMENSIONS]
    for start in range(0, len(items), batch_size):
        input_ids, attention, siglip, _ = batches(items[start:start + batch_size], train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = classifier(input_ids.to(device), attention.to(device), siglip.to(device))
        for task, task_logits in enumerate(logits):
            parts[task].append(task_logits.float().softmax(-1).cpu())
    classifier.train()
    probabilities = [torch.cat(p) for p in parts]
    results = metric.evaluate(D.solution_frame(rows), D.prediction_frame(rows.image_id, probabilities), "image_id")
    return results["overall"]["f1_macro"], {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, required=True, help="folder written by encode.py")
    parser.add_argument("--out", type=Path, required=True, help="inside data/ (git-ignored)")
    parser.add_argument("--part", default="training", choices=("training", "all"))
    parser.add_argument("--val-fraction", type=float, default=0.1, help="training products that pick the checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows, for a smoke test")
    parser.add_argument("--max-steps", type=int, default=0, help="for a smoke test")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="training budget: after a few steps the schedule is fitted into it, so the learning "
                             "rate still decays to 0 and the run ends with its best checkpoint saved")
    # model
    parser.add_argument("--layers", type=int, default=24, help="Apertus layers kept (of 32)")
    parser.add_argument("--lora-rank", type=int, default=32, help="0: no LoRA (frozen Apertus)")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--init-projector", type=Path, help="projector.pt of apertus-vadapter-benchmark")
    parser.add_argument("--no-adapter", action="store_true", help="ablation: VQ tokens only, no SigLIP soft tokens")
    # training
    parser.add_argument("--epochs", type=float, default=2, help="may be fractional (phase 2 repeats phase 1)")
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss; 0 = cross entropy")
    parser.add_argument("--class-alpha", default="effective", choices=("none", "effective"),
                        help="per-class focal alpha from the effective number of samples")
    parser.add_argument("--no-repeats", action="store_true")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--context-dropout", type=float, default=0.0, help="share of examples without context")
    parser.add_argument("--grad-ckpt", action="store_true", help="less memory, slower")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--upload-repo", default="", help="private HF repo that receives every saved checkpoint")
    parser.add_argument("--upload-prefix", default="vlh")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU is visible")
    if args.part == "all" and args.val_fraction:
        sys.exit("phase 2 (--part all) has nothing to validate on: pass --val-fraction 0 and the phase 1 epochs")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    rows = D.select(D.read_csv(args.data_dir / "train.csv"), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    if args.part == "training":
        reserved = set(D.select(D.read_csv(args.data_dir / "train.csv"), "coverage").product_id)
        assert not rows.product_id.isin(reserved).any(), "training rows touch the coverage set"
    fit, validation = D.internal_split(rows, args.val_fraction)
    fit_labels, val_labels = D.label_matrix(fit), D.label_matrix(validation)
    items = D.training_items(fit, fit_labels, repeats=not args.no_repeats, mirrors=not args.no_mirrors)
    cache = load_cache(args.cache)
    needed = {D.key(fit.image_id[i], v) for i, v in items} | {D.key(i) for i in validation.image_id}
    missing = needed - set(cache)
    if missing and args.limit:  # smoke test: the repeats of a few rows differ from those of the full part
        items = [(i, v) for i, v in items if D.key(fit.image_id[i], v) in cache]
        missing = {k for k in missing if "|" not in k}
    if missing:
        sys.exit(f"{len(missing)} image variants have no cached inputs (e.g. {sorted(missing)[:3]}): run encode.py "
                 "with the same --part, --val-fraction, --no-repeats and --no-mirrors")
    log(f"{args.part}: {len(fit)} images to train ({len(items)} items per epoch with augmented repeats and mirrors), "
        f"{len(validation)} for validation")

    from transformers import AutoConfig, AutoTokenizer

    directory = E.model_dir()
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset,
                              0 if args.no_adapter else E.POOLED_SIDE**2)
    settings = {key_: value for key_, value in vars(args).items()
                if key_ in ("layers", "lora_rank", "lora_alpha", "lora_dropout", "head_hidden", "head_dropout")}
    first = next(iter(cache.values()))
    settings |= {"adapter": not args.no_adapter, "vision_dim": first["siglip"].shape[-1], "side": json.loads((args.cache / "meta.json").read_text())["side"],
                 "model": E.MODEL_ID, "revision": E.MODEL_REVISION, "siglip": E.SIGLIP_ID, "task": E.TASK}
    classifier = M.build(directory, settings, device)
    if args.init_projector:
        classifier.projector.load_state_dict(torch.load(args.init_projector, map_location="cpu", weights_only=True))
        log(f"projector initialised from {args.init_projector}")
    if args.grad_ckpt:
        classifier.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    classifier.train()

    lora = [p for n, p in classifier.language_model.named_parameters() if p.requires_grad]
    groups = [{"params": classifier.projector.parameters(), "lr": args.lr_projector, "weight_decay": 0.0},
              {"params": classifier.heads.parameters(), "lr": args.lr_heads, "weight_decay": 0.05}]
    if lora:
        groups.insert(0, {"params": lora, "lr": args.lr_lora, "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(groups)
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log(f"{args.layers} layers kept, LoRA r{args.lora_rank}: {trainable / 1e6:.1f} M trainable parameters")

    steps_per_epoch = math.ceil(len(items) / (args.batch_size * args.accumulation))
    total = args.max_steps or max(1, round(steps_per_epoch * args.epochs))
    plan = {"total": total, "warmup": max(1, round(args.warmup_ratio * total))}  # the budget may lower "total"

    def factor(s):
        progress = min(1.0, max(0, s - plan["warmup"]) / max(1, plan["total"] - plan["warmup"]))
        return min(1.0, (s + 1) / plan["warmup"]) * 0.5 * (1 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    alphas = D.class_alpha(D.item_labels(fit_labels, items)) if args.class_alpha == "effective" else None
    if alphas:
        log("focal alpha per class: " + " | ".join(
            f"{d} " + " ".join(f"{n}={w:.2f}" for n, w in zip(D.TAXONOMY[d], a.tolist()))
            for d, a in zip(D.DIMENSIONS, alphas)))
    batches = Batches(fit.to_dict("records"), fit_labels, cache, builder, tokenizer.pad_token_id,
                      args.context_dropout, args.seed)
    val_batches = Batches(validation.to_dict("records"), val_labels, cache, builder, tokenizer.pad_token_id)
    metric = D.load_metric(args.data_dir)
    eval_every = max(1, steps_per_epoch // max(1, args.evals_per_epoch))
    log(f"{steps_per_epoch} steps per epoch, {total} steps, warmup {warmup}, validation every {eval_every} steps")

    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], {"score": -1.0}
    started, step, stop, times = time.time(), 0, False, {}
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, math.ceil(args.epochs) + 1):
        order = [items[i] for i in torch.randperm(len(items), generator=generator).tolist()]
        micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for start in range(0, len(micro), args.accumulation):
            group, losses_seen = micro[start:start + args.accumulation], []
            for chunk in group:
                input_ids, attention, siglip, labels = batches(chunk, train=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = classifier(input_ids.to(device), attention.to(device), siglip.to(device))
                losses = M.task_losses(logits, labels.to(device), args.gamma, alphas)
                if all(l is None for l in losses):
                    continue  # no scored cell in this micro-batch
                loss = sum(l for l in losses if l is not None) / len(group)
                loss.backward()
                losses_seen.append([None if l is None else l.item() for l in losses])
            torch.nn.utils.clip_grad_norm_([p for p in classifier.parameters() if p.requires_grad], args.clip)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            times[step] = time.time()
            if step == PROBE[1] and args.max_minutes and not args.max_steps:
                per_step = (times[PROBE[1]] - times[PROBE[0]]) / (PROBE[1] - PROBE[0])
                per_eval_step = len(validation) * per_step / (args.batch_size * args.accumulation) / 3 / eval_every
                left = args.max_minutes * 60 - (time.time() - started)
                fitted = max(step + 1, min(plan["total"], step + int(left / (per_step + per_eval_step))))
                if fitted < plan["total"]:
                    log(f"{per_step:.2f} s per step: {fitted} of {plan['total']} steps fit in {args.max_minutes} "
                        f"min ({fitted / steps_per_epoch:.2f} epochs); the schedule ends there")
                    plan["total"] = total = fitted
                else:
                    log(f"{per_step:.2f} s per step: all {plan['total']} steps fit in {args.max_minutes} min")
            if step % 10 == 0:
                per_task = [sum(v[t] for v in losses_seen if v[t] is not None) / max(1, sum(v[t] is not None for v in losses_seen))
                            for t in range(len(D.DIMENSIONS))]
                rate = step * args.batch_size * args.accumulation / (time.time() - started)
                log(f"epoch {epoch} step {step}/{total} loss " + " ".join(f"{x:.3f}" for x in per_task)
                    + f" | {rate:.1f} items/s, peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
            last_of_epoch = start + args.accumulation >= len(micro)
            timeout = args.max_minutes and time.time() - started > args.max_minutes * 60 * 1.05
            stop = step >= total or bool(timeout)
            if len(validation) and (step % eval_every == 0 or last_of_epoch or stop):
                score, per_dimension = evaluate(classifier, val_batches, validation, metric, 2 * args.batch_size, device)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "macro_f1": score,
                                "dimensions": per_dimension})
                log(f"validation step {step} (epoch {step / steps_per_epoch:.2f}): macro F1 {score:.4f} | "
                    + " ".join(f"{d} {v:.3f}" for d, v in per_dimension.items()))
                if score > best["score"]:
                    best = history[-1] | {"score": score}
                    M.save(classifier, settings, args.out / "best")
                    upload(args.upload_repo, args.out / "best", f"{args.upload_prefix}/best")
            if stop:
                break
        if not len(validation):
            M.save(classifier, settings, args.out / "last")
            upload(args.upload_repo, args.out / "last", f"{args.upload_prefix}/last")
        if stop:
            break

    summary = {"part": args.part, "fit_images": len(fit), "validation_images": len(validation),
               "items_per_epoch": len(items), "steps": step, "steps_per_epoch": steps_per_epoch,
               "minutes": round((time.time() - started) / 60, 1), "best": best, "history": history,
               "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
               "gpu": torch.cuda.get_device_name(0), "settings": settings,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    upload(args.upload_repo, args.out, args.upload_prefix, ["train_summary.json"])
    if len(validation):
        log(f"done: {step} steps in {summary['minutes']} min; best validation macro F1 {best['score']:.4f} "
            f"at epoch {best.get('epoch')}, checkpoint in {args.out / 'best'}")
    else:
        log(f"done: {step} steps in {summary['minutes']} min, checkpoint in {args.out / 'last'}")


if __name__ == "__main__":
    main()
