"""Fit DINO fusion into the pretrained SigLIP + VQ classifier, then unfreeze on a plateau."""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from mixed_model import D, E, M, MixedEncoder, batch_inputs, load_mixed, prepare_features, save_mixed
from base_weights import BaseMasters
from distillation import load_teacher, paired_macro_intervals, soft_losses
from experiment import add_synthetic, chunks, losses_with_synthetic, training_rows
from pipeline import probabilities
from train import load_pixels, log, upload


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


def guarded_batches(records, encoder, data_dir, batch_size, replaced):
    """live_batches that never stops on an unreadable image: it is read as a blank picture, so every row still gets
    a prediction. The ids of those images are appended to `replaced`."""
    from predict import blank_image, with_placeholders

    blank = {"image_id": "__blank__", "image_path": str(blank_image())}
    ocr = {}
    with ThreadPoolExecutor(8) as workers:
        for part in chunks(records, batch_size):
            def read(items):
                return encoder.encode(items, [""] * len(items), data_dir, ocr, workers)

            features, bad = with_placeholders(read, part, blank)
            for index in bad:
                ocr[part[index]["image_id"]] = ""
                replaced.append(part[index]["image_id"])
            yield part, features, ocr


def predict_csv(model, builder, encoder, args):
    """Any csv with the columns of dev.csv (the final images) -> <out>/final.csv and its report; no labels needed."""
    rows = D.read_csv(args.predict_csv)
    replaced, started = [], time.monotonic()
    ids, probs = predict_batches(model, builder, guarded_batches(rows.to_dict("records"), encoder, args.data_dir,
                                                               args.encode_batch, replaced))
    seconds = time.monotonic() - started
    if ids != list(rows.image_id):
        raise RuntimeError("The predictions do not match the rows of the csv.")
    D.prediction_frame(ids, probs).to_csv(args.out / "final.csv", index=False)
    torch.save({"image_ids": ids, "dimensions": D.DIMENSIONS, "probabilities": probs}, args.out / "final.probs.pt")
    report = {"images": len(rows), "seconds": round(seconds, 1), "images_per_second": round(len(rows) / seconds, 2),
              "images_replaced_by_blank": len(replaced)}
    (args.out / "final.report.json").write_text(json.dumps(report, indent=2))
    log(f"final: {len(rows)} images at {report['images_per_second']} img/s, {len(replaced)} replaced by a blank one")


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
    return row, probs


def configure(model, settings, stage, vision_lr=5e-5):
    aperture = model.aperture
    model.requires_grad_(False)
    model.fusion.requires_grad_(True)
    groups = [{"params": list(model.fusion.parameters()), "lr": 2e-4 if stage == "fusion" else 6e-5}]
    if stage == "vision":
        seeing = [p for name, p in aperture.vision.named_parameters() if "lora_" in name]
        if not seeing:
            raise ValueError("The vision stage needs a LoRA on SigLIP.")
        for parameter in seeing:
            parameter.requires_grad_(True)
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
    if stage == "vision":
        groups.append({"params": seeing, "lr": vision_lr})
    return groups, masters


def train_stage(stage, epochs, model, settings, builder, fit, labels, items, cache, ocr, args, deadline, validate,
                *, teacher=None, sample_count=None):
    groups, masters = configure(model, settings, stage, args.lr_vision if stage == "vision" else 5e-5)
    pilot = stage in ("control", "distill")
    rates = [group["lr"] * (.5 if pilot else 1.) for group in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
    rows = fit.to_dict("records")
    val_state = validate.best
    meaningful_best, plateau, regressions = val_state["macro_f1"], 0, 0
    previous_train_loss = None
    seed = {"fusion": 3407, "joint": 3408, "top2": 3409, "vision": 3410, "control": 3410, "distill": 3410}[stage]
    generator = torch.Generator().manual_seed(seed)
    step = 0
    for epoch in range(1, math.ceil(epochs) + 1):
        order = torch.randperm(len(items), generator=generator).tolist()
        if sample_count is not None:
            if not 0 < sample_count <= len(order):
                raise ValueError("The pilot budget must fit in a single pass over each branch's examples.")
            order = order[:sample_count]
        elif epochs < 1:
            order = order[:math.ceil(len(order) * epochs)]
        if stage == "fusion":
            model.eval()
            model.fusion.train()
        else:
            model.train()
        train_loss, timed_out = [], False
        for positions in chunks(order, 2 * args.train_batch):
            if time.monotonic() >= deadline:
                if stage != "vision" or not train_loss:
                    return "time_limit"
                timed_out = True  # the vision stage scores what it has learned before it stops
                break
            optimizer.zero_grad(set_to_none=True)
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * min(1., (step + 1) / 20)
            microbatches = list(chunks(positions, args.train_batch))
            for microbatch in microbatches:
                views = [items[i] for i in microbatch]
                indices = [i for i, _ in views]
                records = [rows[i] for i in indices]
                features = [cache[D.key(row["image_id"], variant)] for row, (_, variant) in zip(records, views, strict=True)]
                logits = forward(model, builder, records, features, ocr)
                device = logits[0].device
                target = labels[indices].to(device)
                supervised_count = teacher[2] if teacher else len(fit)
                synthetic = torch.tensor([args.real_count <= i < supervised_count for i in indices], device=device)
                losses = losses_with_synthetic(logits, target, synthetic, args.alphas)
                active = [loss for loss in losses if loss is not None]
                if teacher:
                    positions = torch.tensor([k for k, i in enumerate(indices) if i >= supervised_count],
                                             dtype=torch.long)
                    if positions.numel():
                        teacher_rows = torch.tensor([indices[k] - supervised_count for k in positions.tolist()])
                        active.extend(.5 * loss for loss in soft_losses(logits, *teacher[:2], positions, teacher_rows))
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
                log(f"{stage} step {step} | loss {torch.stack(train_loss[-100:]).mean().item():.3f}")
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
        if timed_out:
            return "time_limit"
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
        self.predictions = None

    def __call__(self, stage, epoch, model, settings, masters=None, train_loss=None):
        row, probs = validation(model, self.builder, self.val, self.cache, self.ocr, self.metric,
                                2 * self.args.train_batch, stage, epoch)
        row["training_loss"] = train_loss
        row["unfrozen_blocks"] = 2 if stage == "top2" else 0
        self.history.append(row)
        if self.best is None or row["macro_f1"] > self.best["macro_f1"]:
            self.best = row
            self.predictions = probs
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
    return report


def run_pilot(args, directory, rows, metric, deadline, dev, teacher):
    model, settings, builder = load_mixed(directory, args.warm_start, args.dino_revision)
    settings.update(distill_source=args.source_id or str(args.warm_start), distill_revision=args.source_revision)
    encoder = MixedEncoder(directory, settings["dino_revision"])
    fit, val, labels, items = training_rows(rows)
    args.real_count = len(fit)
    if args.synthetic:
        fit, labels, items = add_synthetic(fit, val, labels, items, D.read_csv(args.synthetic), args.data_dir)
    args.alphas = D.class_alpha(D.item_labels(labels, [(i, v) for i, v in items if i < args.real_count]))
    supervised_count = len(fit)
    probabilities, mask = teacher
    eligible = mask.any(1)
    dev = dev.loc[eligible.numpy()].reset_index(drop=True)
    probabilities = [values[eligible] for values in probabilities]
    mask = mask[eligible]
    fit_distill = pd.concat([fit, dev], ignore_index=True)
    labels_distill = torch.cat([labels, labels.new_full((len(dev), len(D.DIMENSIONS)), -1)])
    distill_items = [*items, *((supervised_count + i, "") for i in range(len(dev)))]
    budget = math.ceil(len(distill_items) * .5)
    if budget > len(items):
        raise ValueError("Not enough supervised views for a matched control.")

    # Compare actual image-to-CSV throughput using the same hardware, batches and OCR path.
    baseline_live = export(model, builder, val, encoder, args, "baseline_prompt_test", metric)
    ocr = json.loads(args.ocr_cache.read_text()) if args.ocr_cache else {}
    combined = pd.concat([fit_distill, val], ignore_index=True).to_dict("records")
    cache = prepare_features(encoder, combined, distill_items + [(len(fit_distill) + i, "") for i in range(len(val))],
                             args.data_dir, ocr, args.encode_batch)
    encoder.move("cpu")
    torch.cuda.empty_cache()
    control_args = argparse.Namespace(**(vars(args) | {"out": args.out / "control", "prefix": f"{args.prefix}/control"}))
    control = Selection(control_args, val, cache, ocr, metric, builder)
    control("baseline", 0., model, settings)
    baseline_probs = control.predictions
    torch.manual_seed(3410)
    outcome = train_stage("control", .5, model, settings, builder, fit, labels, items, cache, ocr, args, deadline,
                          control, sample_count=budget)
    if outcome != "epoch_limit":
        raise RuntimeError(f"Control did not finish the pilot budget: {outcome}")
    del model
    torch.cuda.empty_cache()

    model, _, builder = load_mixed(directory, args.warm_start, args.dino_revision)
    choose = Selection(args, val, cache, ocr, metric, builder)
    choose("baseline", 0., model, settings)
    for actual, expected in zip(choose.predictions, baseline_probs, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    torch.manual_seed(3410)
    settings["teacher_sha256"] = args.teacher_sha256
    outcome = train_stage("distill", .5, model, settings, builder, fit_distill, labels_distill, distill_items,
                          cache, ocr, args, deadline, choose, teacher=(probabilities, mask, supervised_count),
                          sample_count=budget)
    if outcome != "epoch_limit":
        raise RuntimeError(f"Distillation did not finish the pilot budget: {outcome}")
    intervals = paired_macro_intervals(val, baseline_probs, control.predictions, choose.predictions)
    control_history, history, best = control.history, choose.history, choose.best
    del model
    torch.cuda.empty_cache()
    model, _, builder = load_mixed(directory, args.out / "best", args.dino_revision)
    _, replay = predict_batches(model, builder, cached_batches(val.head(4).to_dict("records"), cache, ocr, 4))
    for actual, expected in zip(replay, choose.replay, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    del choose, control, cache
    encoder.move("cuda")
    student_live = export(model, builder, val, encoder, args, "prompt_test", metric)
    first, last = (entry["metrics"]["overall"] for entry in (baseline_live, student_live))
    macro_change = last["f1_macro"] - first["f1_macro"]
    weighted_change = last["f1_weighted"] - first["f1_weighted"]
    speed_ratio = student_live["images_per_second"] / baseline_live["images_per_second"]
    summary = {"best": best, "history": history, "control_history": control_history, "matched_views": budget,
               "matched_updates": math.ceil(budget / (2 * args.train_batch)), "teacher_images": len(dev),
               "teacher_sha256": args.teacher_sha256, "source_revision": args.source_revision,
               "paired_cached_macro_ci95": intervals,
               "macro_f1_change_live": macro_change, "weighted_f1_change_live": weighted_change,
               "live_throughput_ratio": speed_ratio, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "passes_gate": (best["stage"] == "distill" and macro_change >= .008 and weighted_change >= -.002
                               and min(interval[0] for interval in intervals.values()) > 0 and speed_ratio >= .90)}
    (args.out / "training.json").write_text(json.dumps(summary, indent=2))
    upload(args.upload_repo, args.out, f"{args.prefix}/model", ["training.json"])
    log(f"Pilot gate: {summary['passes_gate']} | macro delta {macro_change:+.5f} | "
        f"weighted delta {weighted_change:+.5f} | throughput ratio {speed_ratio:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial", type=Path)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--warm-start", type=Path,
                        help="Mixed checkpoint for training. Requires --teacher and --teacher-sha256.")
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
    parser.add_argument("--vision-lora-rank", type=int, default=0,
                        help="with --checkpoint: continue a mixed checkpoint training a new LoRA on SigLIP too")
    parser.add_argument("--vision-epochs", type=float, default=2)
    parser.add_argument("--lr-vision", type=float, default=5e-5)
    parser.add_argument("--predict-csv", type=Path,
                        help="with --checkpoint: only predict this csv (image paths start from --data-dir)")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-sha256")
    args = parser.parse_args()
    if min(args.wall_seconds, args.joint_epochs, args.train_batch, args.encode_batch) <= 0 or args.unfreeze_epochs < 0:
        parser.error("Invalid time, batch size or epoch count.")
    if args.initial and not args.dino_revision:
        parser.error("Training requires an exact DINO revision.")
    if args.warm_start or args.teacher or args.teacher_sha256:
        if not (args.warm_start and args.teacher and args.teacher_sha256) or args.predict_dev:
            parser.error("Pilot requires a mixed checkpoint, --teacher and --teacher-sha256, without --predict-dev.")
        settings = json.loads((args.warm_start / "settings.json").read_text())
        if not settings.get("mixed") or settings.get("unfrozen_blocks"):
            parser.error("The pilot expects a mixed checkpoint without unfrozen base blocks.")
    vision = args.vision_lora_rank > 0
    if vision and not args.checkpoint:
        parser.error("--vision-lora-rank continues a mixed checkpoint: use --checkpoint.")
    deadline = time.monotonic() + args.wall_seconds
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    directory = E.model_dir()
    if args.warm_start:
        rows = D.read_csv(args.data_dir / "train.csv")
        metric = D.load_metric(args.data_dir)
        dev = D.read_csv(args.data_dir / "dev.csv")
        teacher = load_teacher(args.teacher, args.teacher_sha256, dev, rows)
        run_pilot(args, directory, rows, metric, deadline, dev, teacher)
        return
    if args.predict_csv:
        if not args.checkpoint or vision:
            parser.error("--predict-csv needs --checkpoint and trains nothing.")
        model, settings, builder = load_mixed(directory, args.checkpoint, args.dino_revision)
        encoder = MixedEncoder(directory, settings["dino_revision"])
        if settings.get("vision_lora_rank"):
            encoder.use_vision_lora(args.checkpoint)
        return predict_csv(model, builder, encoder, args)
    rows = D.read_csv(args.data_dir / "train.csv")
    metric = D.load_metric(args.data_dir)
    model, settings, builder = load_mixed(directory, args.initial or args.checkpoint, args.dino_revision,
                                          vision_lora_rank=args.vision_lora_rank)
    encoder = MixedEncoder(directory, settings["dino_revision"])
    val = D.select(rows, "test")
    trains = bool(args.initial) or vision
    final_checkpoint = args.out / "best" if trains else args.checkpoint

    if trains:
        if args.initial:
            if settings.get("unfrozen_blocks") or any(p.count_nonzero().item() for p in model.fusion[-1].parameters()):
                raise ValueError("--initial expects the original SigLIP + VQ checkpoint.")
        elif not settings.get("mixed") or settings.get("unfrozen_blocks"):
            raise ValueError("The vision stage continues a mixed checkpoint without unfrozen Apertus blocks.")
        settings.update(init_checkpoint=args.source_id or str(args.initial or args.checkpoint),
                        init_revision=args.source_revision)
        fit, val, labels, items = training_rows(rows)
        args.real_count = len(fit)
        if args.synthetic:
            fit, labels, items = add_synthetic(fit, val, labels, items, D.read_csv(args.synthetic), args.data_dir)
        args.alphas = D.class_alpha(D.item_labels(labels, [(i, v) for i, v in items if i < args.real_count]))
        ocr = json.loads(args.ocr_cache.read_text()) if args.ocr_cache else {}
        sample = fit.head(2).to_dict("records")
        with ThreadPoolExecutor(2) as workers:
            features = encoder.encode(sample, ["", "a2"], args.data_dir, ocr, workers)
        if args.initial:
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

        else:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(settings["siglip"])
            pictures = load_pixels(args.data_dir, {(r["image_id"], False): r["image_path"] for r in sample}, processor)
            seen = torch.stack([pictures[r["image_id"], False] for r in sample]).to("cuda")
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                live = model.aperture.vision(seen)
            frozen = encoder.siglip_vq.siglip_features([E.full_resolution(args.data_dir / r["image_path"])
                                                        for r in sample]).float().to("cuda")
            gap = (live.float() - frozen).abs().max().item()
            if gap > 0.02:
                raise RuntimeError(f"The new LoRA is zero, yet SigLIP inside the model differs by {gap:.4f}.")
            configure(model, settings, "vision")
            model.train()
            inputs = batch_inputs(builder, sample, [{**f, "siglip": pictures[r["image_id"], False]}
                                                    for r, f in zip(sample, features, strict=True)], ocr, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                smoke = sum(part.float().square().mean() for part in model(*inputs))
            smoke.backward()
            reached = sum(p.grad.abs().sum().item() for n, p in model.aperture.vision.named_parameters()
                          if "lora_B" in n and p.grad is not None)
            if not 0 < reached < float("inf"):
                raise RuntimeError("The GPU smoke test did not propagate gradients to the SigLIP LoRA.")
            model.zero_grad(set_to_none=True)
            del inputs, smoke, live, seen
            log(f"GPU smoke passed: same SigLIP features at the start (max gap {gap:.5f}), "
                "gradients reach its LoRA")

        combined = fit.to_dict("records") + val.to_dict("records")
        cache = prepare_features(encoder, combined, items + [(len(fit) + i, "") for i in range(len(val))],
                                 args.data_dir, ocr, args.encode_batch)
        if vision:  # SigLIP learns: the cache keeps the pixels, the model computes the features
            from transformers import AutoImageProcessor

            views = [(combined[i], variant) for i, variant in
                     items + [(len(fit) + i, "") for i in range(len(val))]]
            wanted = {(row["image_id"], D.is_mirrored(v)): row["image_path"] for row, v in views}
            pictures = load_pixels(args.data_dir, wanted, AutoImageProcessor.from_pretrained(settings["siglip"]))
            for row, variant in views:
                name = D.key(row["image_id"], variant)
                cache[name] = {**cache[name], "siglip": pictures[row["image_id"], D.is_mirrored(variant)]}
            log(f"{len(pictures)} pictures in memory for the SigLIP LoRA")
        encoder.move("cpu")
        torch.cuda.empty_cache()
        choose = Selection(args, val, cache, ocr, metric, builder)
        choose("baseline", 0., model, settings)
        reasons = {}
        stages = ((("vision", args.vision_epochs),) if vision else
                  (("fusion", .5), ("joint", args.joint_epochs), ("top2", args.unfreeze_epochs)))
        for stage, epochs in stages:
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

    if settings.get("vision_lora_rank"):  # the live encoder must read the images as the trained model saw them
        encoder.use_vision_lora(final_checkpoint)
        log("SigLIP LoRA merged into the encoder for prediction")
    export(model, builder, val, encoder, args, "prompt_test", metric)
    if args.predict_dev:
        export(model, builder, D.read_csv(args.data_dir / "dev.csv"), encoder, args, "dev")


if __name__ == "__main__":
    main()
