"""Fit DINO fusion into the pretrained SigLIP + VQ classifier, then unfreeze on a plateau."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from mixed_model import D, E, M, MixedEncoder, batch_inputs, load_mixed, prepare_features, save_mixed
from base_weights import BaseMasters
from experiment import add_synthetic, chunks, losses_with_synthetic, training_rows
from pipeline import probabilities
from train import log, upload


def forward(model, builder, rows, features, ocr):
    inputs = batch_inputs(builder, rows, features, ocr, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(*inputs)


def cached_batches(rows, cache, ocr, batch_size):
    for records in chunks(rows, batch_size):
        yield records, [cache[row["image_id"]] for row in records], ocr


def live_batches(rows, encoder, data_dir, batch_size):
    ocr = {}
    with ThreadPoolExecutor(8) as workers:
        for records in chunks(rows, batch_size):
            features = encoder.encode(records, [""] * len(records), data_dir, ocr, workers)
            yield records, features, ocr


@torch.inference_mode()
def predict_batches(model, builder, batches):
    model.eval()
    parts = [[] for _ in D.DIMENSIONS]
    ids = []
    for records, features, ocr in batches:
        ids.extend(row["image_id"] for row in records)
        for part, values in zip(parts, probabilities(forward(model, builder, records, features, ocr)), strict=True):
            part.append(values)
    return ids, [torch.cat(part) for part in parts]


def validation(model, builder, val, cache, ocr, metric, batch_size, stage, epoch):
    ids, probs = predict_batches(model, builder, cached_batches(val.to_dict("records"), cache, ocr, batch_size))
    results = metric.evaluate(D.solution_frame(val), D.prediction_frame(ids, probs), "image_id")
    labels = D.label_matrix(val)
    losses = []
    for i, probability in enumerate(probs):
        keep = labels[:, i] >= 0
        if keep.any():
            losses.append(-probability[keep, labels[keep, i]].clamp_min(1e-12).log().mean().item())
    row = {"stage": stage, "epoch": epoch, "macro_f1": results["overall"]["f1_macro"],
           "weighted_f1": results["overall"]["f1_weighted"], "validation_loss": sum(losses) / len(losses),
           "dimensions": {d: results[d]["f1_macro"] for d in D.DIMENSIONS}}
    log(f"validation {stage} epoch {epoch:g}: macro F1 {row['macro_f1']:.5f} | "
        f"weighted F1 {row['weighted_f1']:.5f} | loss {row['validation_loss']:.4f} | "
        + " ".join(f"{d}={value:.3f}" for d, value in row["dimensions"].items()))
    return row


def configure(model, settings, stage):
    aperture = model.aperture
    model.requires_grad_(False)
    model.fusion.requires_grad_(True)
    groups = [{"params": list(model.fusion.parameters()), "lr": 2e-4 if stage == "fusion" else 6e-5}]
    masters = BaseMasters(aperture.language_model, settings["layers"])
    if stage == "fusion":
        return groups, masters

    for module in (aperture.projector, aperture.heads, aperture.pool):
        module.requires_grad_(True)
    adapters = []
    for name, parameter in aperture.language_model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
            adapters.append(parameter)
    groups.extend([
        {"params": [*aperture.projector.parameters(), *aperture.heads.parameters(), *aperture.pool.parameters()],
         "lr": 4e-5},
        {"params": adapters, "lr": 1.5e-5},
    ])
    if stage == "top2":
        groups.append({"params": masters.expand(2), "lr": 2e-6, "weight_decay": 0.})
    return groups, masters


def train_stage(stage, epochs, model, settings, builder, fit, labels, items, cache, ocr, args, deadline, validate):
    groups, masters = configure(model, settings, stage)
    rates = [group["lr"] for group in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
    rows = fit.to_dict("records")
    val_state = validate.best
    meaningful_best, plateau, regressions = val_state["macro_f1"], 0, 0
    previous_train_loss = None
    generator = torch.Generator().manual_seed({"fusion": 3407, "joint": 3408, "top2": 3409}[stage])
    step = 0
    for epoch in range(1, math.ceil(epochs) + 1):
        order = torch.randperm(len(items), generator=generator).tolist()
        if epochs < 1:
            order = order[:math.ceil(len(order) * epochs)]
        if stage == "fusion":
            model.eval()
            model.fusion.train()
        else:
            model.train()
        train_loss = []
        for positions in chunks(order, 2 * args.train_batch):
            if time.monotonic() >= deadline:
                return "time_limit"
            optimizer.zero_grad(set_to_none=True)
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * min(1., (step + 1) / 20)
            microbatches = list(chunks(positions, args.train_batch))
            for microbatch in microbatches:
                views = [items[i] for i in microbatch]
                indices = [i for i, _ in views]
                records = [rows[i] for i in indices]
                features = [cache[D.key(row["image_id"], variant)] for row, (_, variant) in zip(records, views, strict=True)]
                target = labels[indices].to("cuda")
                synthetic = torch.tensor([i >= args.real_count for i in indices], device="cuda")
                losses = losses_with_synthetic(forward(model, builder, records, features, ocr), target, synthetic, args.alphas)
                active = [loss for loss in losses if loss is not None]
                if active:
                    value = sum(active)
                    (value / len(microbatches)).backward()
                    train_loss.append(value.detach())
            masters.transfer_gradients()
            if stage == "top2" and not step:
                witness = next((p for p in masters.master.values()
                                if p.grad is not None and p.grad.flatten()[:4096].abs().max().item() > 0), None)
                if witness is None:
                    return "no_base_gradient"
                old = witness.detach().flatten()[:4096].clone()
            active = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(active, 1.)):
                return "nonfinite_gradient"
            optimizer.step()
            if stage == "top2" and not step:
                if torch.equal(old, witness.detach().flatten()[:4096]):
                    return "no_master_update"
                log("Verified an FP32 master weight update in the two unfrozen Apertus blocks")
            masters.sync()
            step += 1
            if step % 100 == 0:
                log(f"{stage} step {step}")
        if not train_loss:
            raise RuntimeError("No training loss was computed.")
        mean_loss = torch.stack(train_loss).mean().item()
        best_before = validate.best
        row = validate(stage, min(epoch, epochs), model, settings, masters, mean_loss)
        score = row["macro_f1"]
        if score >= meaningful_best + .002:
            meaningful_best, plateau = score, 0
        else:
            plateau += 1
        if (previous_train_loss is not None and mean_loss < previous_train_loss
                and score <= best_before["macro_f1"] - .003
                and row["validation_loss"] > best_before["validation_loss"]):
            regressions += 1
        else:
            regressions = 0
        previous_train_loss = mean_loss
        if regressions >= 2:
            return "validation_regression"
        if plateau >= 2:
            return "validation_plateau"
    return "epoch_limit"


class Selection:
    def __init__(self, args, val, cache, ocr, metric, builder):
        self.args, self.val, self.cache, self.ocr, self.metric, self.builder = args, val, cache, ocr, metric, builder
        self.best = None
        self.history = []
        self.replay = None

    def __call__(self, stage, epoch, model, settings, masters=None, train_loss=None):
        row = validation(model, self.builder, self.val, self.cache, self.ocr, self.metric,
                         2 * self.args.train_batch, stage, epoch)
        row["training_loss"] = train_loss
        row["unfrozen_blocks"] = 2 if stage == "top2" else 0
        self.history.append(row)
        if self.best is None or row["macro_f1"] > self.best["macro_f1"]:
            self.best = row
            subset = self.val.head(4).to_dict("records")
            _, self.replay = predict_batches(model, self.builder, cached_batches(subset, self.cache, self.ocr, 4))
            save_mixed(model, {**settings, "unfrozen_blocks": row["unfrozen_blocks"]},
                       self.args.out / "best", masters)
            upload(self.args.upload_repo, self.args.out / "best", f"{self.args.prefix}/model/best")
            log(f"New best: {stage}, macro F1 {row['macro_f1']:.5f}")
        return row


def export(model, builder, frame, encoder, args, name, metric=None):
    started = time.monotonic()
    ids, probs = predict_batches(model, builder, live_batches(frame.to_dict("records"), encoder,
                                                            args.data_dir, args.encode_batch))
    seconds = time.monotonic() - started
    predictions = D.prediction_frame(ids, probs)
    predictions.to_csv(args.out / f"{name}.csv", index=False)
    torch.save({"image_ids": ids, "dimensions": D.DIMENSIONS, "probabilities": probs},
               args.out / f"{name}.probs.pt")
    report = {"images": len(frame), "images_per_second": len(frame) / seconds, "seconds": seconds,
              "gpu": torch.cuda.get_device_name(), "visual_tokens": E.POOLED_SIDE**2}
    if metric is not None:
        report["metrics"] = metric.evaluate(D.solution_frame(frame), predictions, "image_id")
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=2))
    upload(args.upload_repo, args.out, f"{args.prefix}/predictions", [f"{name}.*"])
    log(f"{name}: {len(frame)} images at {report['images_per_second']:.2f} img/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dino-revision")
    parser.add_argument("--source-id")
    parser.add_argument("--source-revision")
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--ocr-cache", type=Path)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--unfreeze-epochs", type=int, default=2)
    parser.add_argument("--train-batch", type=int, default=6)
    parser.add_argument("--encode-batch", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, default=95 * 60)
    parser.add_argument("--upload-repo", default="")
    parser.add_argument("--prefix", default="mixed")
    parser.add_argument("--predict-dev", action="store_true")
    args = parser.parse_args()
    if min(args.wall_seconds, args.joint_epochs, args.train_batch, args.encode_batch) <= 0 or args.unfreeze_epochs < 0:
        parser.error("Invalid time, batch size or epoch count.")
    if args.initial and not args.dino_revision:
        parser.error("Training requires an exact DINO revision.")
    deadline = time.monotonic() + args.wall_seconds
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    directory = E.model_dir()
    rows = D.read_csv(args.data_dir / "train.csv")
    metric = D.load_metric(args.data_dir)
    model, settings, builder = load_mixed(directory, args.initial or args.checkpoint, args.dino_revision)
    encoder = MixedEncoder(directory, settings["dino_revision"])
    val = D.select(rows, "test")

    if args.initial:
        if settings.get("unfrozen_blocks") or any(p.count_nonzero().item() for p in model.fusion[-1].parameters()):
            raise ValueError("--initial expects the original SigLIP + VQ checkpoint.")
        settings.update(init_checkpoint=args.source_id or str(args.initial), init_revision=args.source_revision)
        fit, val, labels, items = training_rows(rows)
        args.real_count = len(fit)
        if args.synthetic:
            fit, labels, items = add_synthetic(fit, val, labels, items, D.read_csv(args.synthetic), args.data_dir)
        args.alphas = D.class_alpha(D.item_labels(labels, [(i, v) for i, v in items if i < args.real_count]))
        ocr = json.loads(args.ocr_cache.read_text()) if args.ocr_cache else {}
        sample = fit.head(2).to_dict("records")
        with ThreadPoolExecutor(2) as workers:
            features = encoder.encode(sample, ["", "a2"], args.data_dir, ocr, workers)
        inputs = batch_inputs(builder, sample, features, ocr, "cuda")
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            direct = model.aperture(*inputs[:3])
            mixed = model(*inputs)
            for expected, actual in zip(direct, mixed, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        model.requires_grad_(False)
        model.fusion.requires_grad_(True)
        smoke = sum(part.float().square().mean() for part in forward(model, builder, sample, features, ocr))
        smoke.backward()
        grad = model.fusion[-1].weight.grad
        if grad is None or not grad.isfinite().all() or grad.abs().sum().item() == 0:
            raise RuntimeError("The GPU smoke test did not propagate gradients to DINO fusion.")
        model.zero_grad(set_to_none=True)
        del mixed, direct, inputs, smoke
        log("GPU smoke passed: source logits unchanged, gradients reach DINO fusion")

        combined = fit.to_dict("records") + val.to_dict("records")
        cache = prepare_features(encoder, combined, items + [(len(fit) + i, "") for i in range(len(val))],
                                 args.data_dir, ocr, args.encode_batch)
        encoder.move("cpu")
        torch.cuda.empty_cache()
        choose = Selection(args, val, cache, ocr, metric, builder)
        choose("baseline", 0., model, settings)
        reasons = {}
        for stage, epochs in (("fusion", .5), ("joint", args.joint_epochs), ("top2", args.unfreeze_epochs)):
            if time.monotonic() >= deadline or not epochs:
                break
            if stage == "top2" and reasons.get("joint") != "validation_plateau":
                break
            if stage != "fusion":
                del model
                torch.cuda.empty_cache()
                model, settings, builder = load_mixed(directory, args.out / "best", args.dino_revision)
                choose.builder = builder
            reasons[stage] = train_stage(stage, epochs, model, settings, builder, fit, labels, items,
                                         cache, ocr, args, deadline, choose)
            log(f"{stage} complete: {reasons[stage]}")
            if reasons[stage] in {"validation_regression", "nonfinite_gradient", "time_limit"}:
                break
        summary = {"best": choose.best, "history": choose.history, "phase_outcomes": reasons,
                   "real_fit": args.real_count, "synthetic_fit": len(fit) - args.real_count,
                   "validation": len(val), "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
        (args.out / "training.json").write_text(json.dumps(summary, indent=2))
        upload(args.upload_repo, args.out, f"{args.prefix}/model", ["training.json"])
        del model
        torch.cuda.empty_cache()
        model, settings, builder = load_mixed(directory, args.out / "best", args.dino_revision)
        _, replay = predict_batches(model, builder, cached_batches(val.head(4).to_dict("records"), cache, ocr, 4))
        for actual, expected in zip(replay, choose.replay, strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        log(f"Reloaded winner reproduced predictions: {choose.best['stage']} | {choose.best['macro_f1']:.5f}")
        del choose, cache
        encoder.move("cuda")

    export(model, builder, val, encoder, args, "prompt_test", metric)
    if args.predict_dev:
        export(model, builder, D.read_csv(args.data_dir / "dev.csv"), encoder, args, "dev")


if __name__ == "__main__":
    main()
