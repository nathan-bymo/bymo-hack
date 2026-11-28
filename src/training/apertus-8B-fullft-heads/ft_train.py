"""Full finetuning of Apertus 8B (24 layers kept, every weight trained, no LoRA) + 6 heads, on the cache of
ft_encode.py. One GPU with bf16 and >= 80 GB (RTX PRO 6000 96 GB, A100 80 GB, H100/H200).

  phase 1  python src/training/apertus-8B-fullft-heads/ft_train.py --cache data/fullft_cache/training_256 \
               --out data/runs/fullft_phase1
  phase 2  python src/training/apertus-8B-fullft-heads/ft_train.py --part all --val-fraction 0 --epochs <phase 1> \
               --cache data/fullft_cache/all_256 --out data/runs/fullft_phase2
  CPU test python src/training/apertus-8B-fullft-heads/ft_train.py --tiny --limit 24 --epochs 1 --cache ... --out ...

Data (apertus-8B-vadapter-lora-heads/DATI.md, same code): phase 1 trains on common.splits.training_rows minus the
internal validation (10% of the products by hash), which picks the checkpoint (macro F1 of metric.py, twice per
epoch); coverage is never seen here, ft_predict.py scores it at the end. Phase 2 trains on every labeled image with
the settings and epochs of phase 1 and keeps the last checkpoint. Rare classes are repeated (up to 4x per epoch,
each repeat a different mild crop, mirrored when there is no text); left/right perspectives without text get one
more mirrored copy.

Loss: per head, focal loss (gamma 2) over the scored cells only (human label in a relevant cell), per-class alpha
from the effective number of samples of the epoch items.

Memory (24 layers, 5.2 B trained parameters): fp32 weights 21 GB + fp32 gradients 21 GB + 8-bit AdamW states
10.5 GB + frozen bf16 embeddings 2.2 GB, plus activations (gradient checkpointing by default): about 60 GB.
"""

import argparse
import json
import math
import random
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ft_model as M  # noqa: E402  (puts apertus-8B-vadapter-lora-heads on the path)
import data as D  # noqa: E402
import encoders as E  # noqa: E402
from ft_encode import load_cache, model_dir  # noqa: E402

ROOT = HERE.parents[2]
PROBE = (10, 30)  # steps between which the step time is measured (after the first, slower, steps)
MAX_MISSING = 20  # variants the cache may lack (unreadable images) before training refuses to start
MAX_BAD_STEPS = 5  # steps with a non-finite loss that are skipped before the run stops
SAVE_RESERVE = 300  # seconds kept free before the budget ends: final save of ~10 GB and slack


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def upload(repo, folder, path_in_repo, patterns=None):
    """Copies a folder to a private HF repo. Never fatal: the checkpoint stays on the job disk."""
    if not repo:
        return
    try:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo,
                              allow_patterns=patterns, commit_message=f"upload {path_in_repo}")
        log(f"uploaded {path_in_repo} to {repo}")
    except Exception as error:
        log(f"upload of {path_in_repo} failed: {type(error).__name__}: {str(error)[:200]}")


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


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
    items = [(i, "") for i in range(len(rows))]
    parts = [[] for _ in D.DIMENSIONS]
    for start in range(0, len(items), batch_size):
        input_ids, attention, _ = batches(items[start:start + batch_size], train=False)
        with autocast(device):
            logits = classifier(input_ids.to(device), attention.to(device))
        for task, task_logits in enumerate(logits):
            parts[task].append(task_logits.float().softmax(-1).cpu())
    classifier.train()
    probabilities = [torch.cat(p) for p in parts]
    results = metric.evaluate(D.solution_frame(rows), D.prediction_frame(rows.image_id, probabilities), "image_id")
    return results["overall"]["f1_macro"], {d: results[d]["f1_macro"] for d in D.DIMENSIONS if d in results}


def make_optimizer(groups, name):
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, betas=(0.9, 0.999))
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=torch.cuda.is_available())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, required=True, help="folder written by ft_encode.py")
    parser.add_argument("--out", type=Path, required=True, help="inside data/ (git-ignored)")
    parser.add_argument("--part", default="training", choices=("training", "all"))
    parser.add_argument("--val-fraction", type=float, default=0.1, help="training products that pick the checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows, for a smoke test")
    parser.add_argument("--max-steps", type=int, default=0, help="for a smoke test")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="safety budget: after a few steps the schedule is fitted into it, so the learning rate "
                             "still decays to 0 (the log says how many epochs fit)")
    parser.add_argument("--deadline", type=float, default=0,
                        help="unix time by which training must have ended and saved (the job timeout minus the time "
                             "for upload and prediction); the schedule is fitted into it like --max-minutes")
    parser.add_argument("--tiny", action="store_true", help="random miniature Apertus on the CPU, to test the code")
    # model
    parser.add_argument("--layers", type=int, default=24, help="Apertus layers kept (of 32)")
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    # training
    parser.add_argument("--epochs", type=float, default=3, help="may be fractional (phase 2 repeats phase 1)")
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="decoder matrices only")
    parser.add_argument("--optimizer", default="adamw8bit", choices=("adamw8bit", "adamw"),
                        help="adamw: fp32 states, 31.5 GB more (H200 only)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss; 0 = cross entropy")
    parser.add_argument("--class-alpha", default="effective", choices=("none", "effective"),
                        help="per-class focal alpha from the effective number of samples")
    parser.add_argument("--no-repeats", action="store_true")
    parser.add_argument("--no-mirrors", action="store_true")
    parser.add_argument("--context-dropout", type=float, default=0.0, help="share of examples without context")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="faster, needs much more memory")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--upload-repo", default="", help="private HF repo that receives the final checkpoint")
    parser.add_argument("--upload-prefix", default="fullft")
    args = parser.parse_args()
    if args.tiny:
        device = torch.device("cpu")
        args.layers, args.optimizer = min(args.layers, M.TINY_TEXT["num_hidden_layers"]), "adamw"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        sys.exit("no CUDA GPU is visible (use --tiny for a CPU test)")
    if args.part == "all" and args.val_fraction:
        sys.exit("phase 2 (--part all) has nothing to validate on: pass --val-fraction 0 and the phase 1 epochs")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    every_row = D.read_csv(args.data_dir / "train.csv")
    rows = D.select(every_row, args.part)
    if args.limit:
        rows = rows.head(args.limit)
    if args.part == "training":
        reserved = set(D.select(every_row, "coverage").product_id)
        assert not rows.product_id.isin(reserved).any(), "training rows touch the coverage set"
    fit, validation = D.internal_split(rows, args.val_fraction)
    fit_labels, val_labels = D.label_matrix(fit), D.label_matrix(validation)
    items = D.training_items(fit, fit_labels, repeats=not args.no_repeats, mirrors=not args.no_mirrors)
    cache = load_cache(args.cache)
    meta = json.loads((args.cache / "meta.json").read_text())
    missing_items = [(i, v) for i, v in items if D.key(fit.image_id[i], v) not in cache]
    missing_val = [i for i in validation.image_id if D.key(i) not in cache]
    if len(missing_items) + len(missing_val) > (len(items) if args.limit else MAX_MISSING):
        sys.exit(f"{len(missing_items) + len(missing_val)} image variants have no cached codes: run ft_encode.py with "
                 "the same --part, --val-fraction, --no-repeats and --no-mirrors")
    if missing_items or missing_val:  # unreadable images (ft_encode failures_*.json) or a --limit smoke test
        log(f"dropping {len(missing_items)} training variants and {len(missing_val)} validation images without codes")
        dropped = set(missing_items)
        items = [item for item in items if item not in dropped]
        keep = ~validation.image_id.isin(missing_val)
        validation, val_labels = validation[keep].reset_index(drop=True), val_labels[torch.tensor(keep.values)]
    log(f"{args.part}: {len(fit)} images to train ({len(items)} items per epoch with augmented repeats and mirrors), "
        f"{len(validation)} for validation")

    from transformers import AutoConfig, AutoTokenizer

    directory = model_dir(full=not args.tiny)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    builder = E.PromptBuilder(tokenizer, AutoConfig.from_pretrained(directory).image_token_offset, prefix_tokens=0)
    settings = {"layers": args.layers, "head_hidden": args.head_hidden, "head_dropout": args.head_dropout,
                "side": meta["side"], "model": E.MODEL_ID, "revision": E.MODEL_REVISION, "task": E.TASK,
                "features": "last token + mean of the VQ image tokens", "tiny": args.tiny}
    classifier = M.build(directory, settings, device)
    M.make_trainable(classifier.language_model)
    grad_ckpt = not args.no_grad_ckpt and not args.tiny
    if grad_ckpt:
        classifier.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    classifier.train()
    groups = M.parameter_groups(classifier, args.lr_backbone, args.lr_heads, args.weight_decay)
    optimizer = make_optimizer(groups, args.optimizer)
    trainable = sum(p.numel() for g in groups for p in g["params"])
    frozen = sum(p.numel() for p in classifier.parameters() if not p.requires_grad)
    log(f"{args.layers} layers kept, full finetuning: {trainable / 1e6:,.0f} M trained parameters (fp32), "
        f"{frozen / 1e6:,.0f} M frozen; optimizer {args.optimizer}; gradient checkpointing {grad_ckpt}")

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
    log(f"{steps_per_epoch} steps per epoch, {total} steps, warmup {plan['warmup']}, validation every {eval_every} "
        "steps")

    args.out.mkdir(parents=True, exist_ok=True)
    history, best, bad_steps = [], {"score": -1.0}, 0
    started, step, stop, times = time.time(), 0, False, {}
    budget = args.max_minutes * 60 if args.max_minutes else math.inf  # seconds of training, from `started`
    if args.deadline:
        budget = min(budget, args.deadline - started - SAVE_RESERVE)
        log(f"deadline: at most {budget / 60:.0f} min of training, then save and upload")
        if budget <= 0:
            sys.exit("no time left before the deadline")
    pending_upload = None  # epoch checkpoints are uploaded in the background while training goes on

    def save_epoch(epoch):
        """No validation (phase 2): a checkpoint per epoch, so a stopped job keeps the epochs already done."""
        nonlocal pending_upload
        saving = time.time()
        folder = args.out / f"epoch-{epoch}"
        M.save(classifier, settings, folder)
        log(f"epoch {epoch} checkpoint saved in {time.time() - saving:.0f} s")
        if args.upload_repo:
            if pending_upload:
                pending_upload.join()
            pending_upload = threading.Thread(target=upload, args=(args.upload_repo, folder,
                                                                   f"{args.upload_prefix}/epoch-{epoch}"))
            pending_upload.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def peak():
        return torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0

    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, math.ceil(args.epochs) + 1):
        order = [items[i] for i in torch.randperm(len(items), generator=generator).tolist()]
        micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for start in range(0, len(micro), args.accumulation):
            group, losses_seen, finite = micro[start:start + args.accumulation], [], True
            for chunk in group:
                input_ids, attention, labels = batches(chunk, train=True)
                with autocast(device):
                    logits = classifier(input_ids.to(device), attention.to(device))
                losses = M.task_losses(logits, labels.to(device), args.gamma, alphas)
                if all(l is None for l in losses):
                    continue  # no scored cell in this micro-batch
                loss = sum(l for l in losses if l is not None) / len(group)
                if not torch.isfinite(loss):
                    finite = False
                    break
                loss.backward()
                losses_seen.append([None if l is None else l.item() for l in losses])
            if finite:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.clip)
                optimizer.step()
            else:
                bad_steps += 1
                log(f"step {step + 1}: non-finite loss, step skipped ({bad_steps}/{MAX_BAD_STEPS})")
                if bad_steps >= MAX_BAD_STEPS:
                    if not len(validation):
                        M.save(classifier, settings, args.out / "last")
                        upload(args.upload_repo, args.out / "last", f"{args.upload_prefix}/last")
                    sys.exit("too many non-finite losses: stopping (the best or last checkpoint is kept)")
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            times[step] = time.time()
            if step == PROBE[1] and math.isfinite(budget) and not args.max_steps:
                per_step = (times[PROBE[1]] - times[PROBE[0]]) / (PROBE[1] - PROBE[0])
                per_eval_step = len(validation) * per_step / (args.batch_size * args.accumulation) / 3 / eval_every
                saves = 2 * 60 * (math.ceil(args.epochs) - 1) if not len(validation) else 0  # epoch checkpoints
                left = budget - (time.time() - started) - saves
                fitted = max(step + 1, min(plan["total"], step + int(left / (per_step + per_eval_step))))
                if fitted < plan["total"]:
                    log(f"{per_step:.2f} s per step: only {fitted} of {plan['total']} steps fit in the budget of "
                        f"{budget / 60:.0f} min ({fitted / steps_per_epoch:.2f} epochs); the schedule ends there")
                    plan["total"] = total = fitted
                else:
                    log(f"{per_step:.2f} s per step: all {plan['total']} steps fit in the budget of {budget / 60:.0f} "
                        f"min (about {per_step * plan['total'] / 60:.0f} min of training)")
            if step % 10 == 0 and losses_seen:
                per_task = [sum(v[t] for v in losses_seen if v[t] is not None)
                            / max(1, sum(v[t] is not None for v in losses_seen)) for t in range(len(D.DIMENSIONS))]
                rate = step * args.batch_size * args.accumulation / (time.time() - started)
                log(f"epoch {epoch} step {step}/{total} loss " + " ".join(f"{x:.3f}" for x in per_task)
                    + f" | lr {schedule.get_last_lr()[0]:.2e} | {rate:.1f} items/s, peak {peak():.1f} GB")
            last_of_epoch = start + args.accumulation >= len(micro)
            timeout = time.time() - started > budget
            stop = step >= total or bool(timeout)
            if len(validation) and (step % eval_every == 0 or last_of_epoch or stop):
                score, per_dimension = evaluate(classifier, val_batches, validation, metric, 2 * args.batch_size,
                                                device)
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "macro_f1": score,
                                "dimensions": per_dimension})
                log(f"validation step {step} (epoch {step / steps_per_epoch:.2f}): macro F1 {score:.4f} | "
                    + " ".join(f"{d} {v:.3f}" for d, v in per_dimension.items()))
                if score > best["score"]:
                    best = history[-1] | {"score": score}
                    saving = time.time()
                    M.save(classifier, settings, args.out / "best")
                    log(f"best checkpoint saved in {time.time() - saving:.0f} s")
            if stop:
                break
        if stop:
            break
        if not len(validation) and epoch < math.ceil(args.epochs):
            save_epoch(epoch)
    if not len(validation):
        M.save(classifier, settings, args.out / "last")

    summary = {"part": args.part, "fit_images": len(fit), "validation_images": len(validation),
               "items_per_epoch": len(items), "steps": step, "steps_per_epoch": steps_per_epoch,
               "epochs_done": round(step / steps_per_epoch, 2), "skipped_steps": bad_steps,
               "minutes": round((time.time() - started) / 60, 1), "best": best, "history": history,
               "peak_memory_gb": round(peak(), 1), "trained_parameters": trainable,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "settings": settings,
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    checkpoint = "best" if len(validation) else "last"
    if pending_upload:
        pending_upload.join()
    upload(args.upload_repo, args.out / checkpoint, f"{args.upload_prefix}/{checkpoint}")
    upload(args.upload_repo, args.out, args.upload_prefix, ["train_summary.json"])
    if len(validation):
        log(f"done: {step} steps in {summary['minutes']} min; best validation macro F1 {best['score']:.4f} "
            f"at epoch {best.get('epoch')}, checkpoint in {args.out / 'best'}; peak memory {peak():.1f} GB")
    else:
        log(f"done: {step} steps in {summary['minutes']} min, checkpoint in {args.out / 'last'}; "
            f"peak memory {peak():.1f} GB")


if __name__ == "__main__":
    main()
