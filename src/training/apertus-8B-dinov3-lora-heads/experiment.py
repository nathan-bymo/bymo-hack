"""Train and evaluate the DINOv3 replacement from the existing SigLIP + OCR checkpoint."""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd
import torch

from base_weights import BaseMasters
from pipeline import (
    AUGMENTED_SIDE, D, E, IMAGE_SIDE, M, DinoEncoder, encode_groups, make_model,
    model_inputs, probabilities,
)
from train import log, upload


def chunks(values, size):
    return (values[start:start + size] for start in range(0, len(values), size))


def training_rows(rows):
    fit, validation = D.fit_and_validation(rows, D.select(rows, "all_but_test"), "all_but_test", 0, True)
    labels = D.label_matrix(fit)
    items = D.training_items(fit, labels, rare_text_only=True)
    return fit, validation, labels, items


def add_synthetic(fit, val, labels, items, synthetic, data_dir):
    if (synthetic.empty or not synthetic.image_id.is_unique or synthetic.image_id.isin(fit.image_id).any()
            or synthetic.image_id.isin(val.image_id).any()
            or not synthetic.product_id.isin(fit.product_id).all()):
        raise ValueError("Synthetic rows must have new image ids and come from real training products only.")
    for value in synthetic.image_path:
        path = Path(value)
        if not path.parts or path.parts[0] != "synthetic" or ".." in path.parts or not (data_dir / path).is_file():
            raise ValueError(f"Invalid synthetic image path: {value}")
    targets = D.label_matrix(synthetic)
    for task, name in enumerate(D.DIMENSIONS):
        if name not in ("text_classification", "text_language"):
            targets[:, task] = -1
    if not targets.ge(0).any(dim=1).all():
        raise ValueError("Every synthetic image needs at least one text label.")
    offset = len(fit)
    return (pd.concat([fit, synthetic], ignore_index=True), torch.cat([labels, targets]),
            items + [(i, "") for i in range(offset, offset + len(synthetic))])


def losses_with_synthetic(logits, labels, synthetic, alphas):
    if not synthetic.any():
        return M.task_losses(logits, labels, 2., alphas)
    losses = []
    for task, logits_for_task in enumerate(logits):
        known = labels[:, task] >= 0
        count = int(known.sum())
        parts = []
        for mask, weight in ((known & ~synthetic, 1.), (known & synthetic, .25)):
            size = int(mask.sum())
            if size:
                parts.append(weight * size / count * M.focal_loss(logits_for_task[mask], labels[mask, task], 2., alphas[task]))
        losses.append(sum(parts) if parts else None)
    return losses


def cached_features(encoder, reader, data_dir, rows, items, ocr, batch_size):
    cache = {}
    for variant, side in (("", IMAGE_SIDE), ("a2", AUGMENTED_SIDE)):
        indices = [index for index, key in items if key == variant]
        for records, features in encode_groups(encoder, rows, data_dir, ocr,
                                               ((batch, side) for batch in chunks(indices, batch_size)), reader=reader):
            cache.update((D.key(row["image_id"], variant), feature) for row, feature in zip(records, features, strict=True))
            if len(cache) % (batch_size * 50) == 0:
                log(f"DINO encoded {len(cache)}/{len(items)} views")
    if len(cache) != len(items):
        raise ValueError("DINO feature cache does not cover the requested images and variants.")
    return cache


def forward(model, builder, rows, features, ocr):
    inputs = model_inputs(builder, rows, features, ocr, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(*inputs)


def evaluate(classifier, builder, frame, cache, ocr, metric, batch_size=16):
    classifier.eval()
    rows = frame.to_dict("records")
    parts = [[] for _ in D.DIMENSIONS]
    with torch.inference_mode():
        for records in chunks(rows, batch_size):
            feature = torch.stack([cache[row["image_id"]] for row in records])
            for target, prob in zip(parts, probabilities(forward(classifier, builder, records, feature, ocr)), strict=True):
                target.append(prob)
    probs = [torch.cat(part) for part in parts]
    metrics = metric.evaluate(D.solution_frame(frame), D.prediction_frame(frame.image_id, probs), "image_id")
    matrix = D.label_matrix(frame)
    loss = []
    for task, prob in enumerate(probs):
        labels = matrix[:, task]
        known = labels >= 0
        if known.any():
            loss.append(-prob[known, labels[known]].clamp_min(1e-12).log().mean().item())
    return metrics, sum(loss) / len(loss)


class EarlyStop:
    def __init__(self):
        self.significant_best = -math.inf
        self.plateau = self.regression = 0
        self.previous_loss = None

    def update(self, score, val_loss, train_loss, best):
        if score >= self.significant_best + 0.002:
            self.significant_best, self.plateau = score, 0
        else:
            self.plateau += 1
        falling = self.previous_loss is not None and train_loss < self.previous_loss
        worse = best is not None and score <= best["macro_f1"] - 0.003 and val_loss > best["validation_loss"]
        self.regression = self.regression + 1 if falling and worse else 0
        self.previous_loss = train_loss
        if self.regression >= 2:
            return "validation_regression"
        if self.plateau >= 3:
            return "validation_plateau"
        return None


def train(args, classifier, settings, builder, fit, val, labels, items, cache, ocr, metric, real_count):
    rows = fit.to_dict("records")
    lora = [p for p in classifier.language_model.parameters() if p.requires_grad]
    heads = [*classifier.heads.parameters(), *classifier.pool.parameters()]
    project = list(classifier.projector.parameters())
    if not lora:
        raise ValueError("The source checkpoint has no trainable LoRA parameters.")
    learning_rates = (3e-5, 8e-5, 2e-5) if args.resume_dino else (1e-4, 2.5e-4, 5e-5)
    groups = [{"params": group, "lr": rate, "weight_decay": decay}
              for group, rate, decay in zip((project, heads, lora), learning_rates, (0., 0.05, 0.01), strict=True)]
    optimizer = torch.optim.AdamW(groups)
    params = project + heads + lora
    alphas = D.class_alpha(D.item_labels(labels, [(i, v) for i, v in items if i < real_count]))
    batch_size, accumulation = args.train_batch, 2
    per_epoch = math.ceil(len(items) / (batch_size * accumulation))
    warm_steps = 0 if args.resume_dino else max(1, per_epoch // 2)
    limit = per_epoch * args.epochs
    frequency = max(1, per_epoch // 2)
    stop = EarlyStop()
    started = time.monotonic()
    deadline = min(started + 60 * args.max_minutes, args.deadline or math.inf)
    generator = torch.Generator().manual_seed(3407)
    best, history, window = None, [], []
    step = 0
    reason = "epoch_limit"
    if args.resume_dino:
        results, val_loss = evaluate(classifier, builder, val, cache, ocr, metric)
        score = results["overall"]["f1_macro"]
        best = {"step": 0, "epoch": 0., "macro_f1": score, "validation_loss": val_loss,
                "dimensions": {dim: results[dim]["f1_macro"] for dim in D.DIMENSIONS}}
        history.append(best)
        stop.significant_best = score
        M.save(classifier, settings, args.out / "best")
        upload(args.upload_repo, args.out / "best", f"{args.prefix}/model/best")
        log(f"epoch 0.00 validation: macro F1 {score:.5f} | warm-start checkpoint retained")

    for epoch in range(args.epochs):
        order = torch.randperm(len(items), generator=generator).tolist()
        for positions in chunks(order, batch_size * accumulation):
            projector_phase = step < warm_steps
            if step in (0, warm_steps):
                for parameter in lora:
                    parameter.requires_grad_(not projector_phase)
                log("projector and heads adaptation" if projector_phase else "joint training, LoRA unfrozen")
            classifier.train()
            if projector_phase:
                classifier.language_model.eval()

            # Warm the newly initialized projector and the newly enabled LoRA independently.
            progress = max(0, step - warm_steps) / max(1, limit - warm_steps)
            decay = 0.5 * (1 + math.cos(math.pi * progress))
            warmup = min(1., (step + 1) / 30)
            factors = (warmup * decay, warmup * decay, min(1., max(0, step - warm_steps + 1) / 30) * decay)
            for group, rate, factor in zip(optimizer.param_groups, learning_rates, factors, strict=True):
                group["lr"] = rate * factor

            microbatches = list(chunks(positions, batch_size))
            optimizer.zero_grad(set_to_none=True)
            for microbatch in microbatches:
                selected = [items[i] for i in microbatch]
                indices = [index for index, _ in selected]
                records = [rows[index] for index in indices]
                features = torch.stack([cache[D.key(row["image_id"], variant)]
                                        for row, (_, variant) in zip(records, selected, strict=True)])
                target = torch.stack([labels[i] for i in indices]).to("cuda")
                logits = forward(classifier, builder, records, features, ocr)
                if real_count == len(fit):
                    losses = M.task_losses(logits, target, 2., alphas)
                else:
                    synthetic = torch.tensor([i >= real_count for i in indices], device="cuda")
                    losses = losses_with_synthetic(logits, target, synthetic, alphas)
                active = [value for value in losses if value is not None]
                if not active:
                    continue
                loss = sum(active)
                (loss / len(microbatches)).backward()
                window.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1

            end = step % per_epoch == 0 or step == limit
            time_up = time.monotonic() >= deadline
            if step in (warm_steps, warm_steps + 30) or step % 50 == 0:
                log(f"step {step}/{limit}, epoch {step / per_epoch:.2f}, elapsed {(time.monotonic() - started) / 60:.1f} min")
            mid_epoch = step % per_epoch == frequency and (step <= per_epoch or not args.eval_once_per_epoch)
            if mid_epoch or end or time_up:
                results, val_loss = evaluate(classifier, builder, val, cache, ocr, metric)
                train_loss = torch.stack(window).mean().item() if window else math.inf
                window.clear()
                score = results["overall"]["f1_macro"]
                record = {"step": step, "epoch": step / per_epoch, "macro_f1": score,
                          "training_loss": train_loss, "validation_loss": val_loss,
                          "dimensions": {d: results[d]["f1_macro"] for d in D.DIMENSIONS}}
                history.append(record)
                log(f"epoch {step / per_epoch:.2f} validation: macro F1 {score:.5f} | weighted F1 "
                    f"{results['overall']['f1_weighted']:.5f} | val loss {val_loss:.4f} | train loss {train_loss:.4f} | "
                    + " ".join(f"{dim}={value:.3f}" for dim, value in record["dimensions"].items()))
                decision = stop.update(score, val_loss, train_loss, best)
                if best is None or score > best["macro_f1"]:
                    best = record
                    M.save(classifier, settings, args.out / "best")
                    upload(args.upload_repo, args.out / "best", f"{args.prefix}/model/best")
                if time_up or decision:
                    reason = "time_limit" if time_up else decision
                    break
        else:
            continue
        break

    summary = {"fit_images": len(fit), "synthetic_images": len(fit) - real_count,
               "validation_images": len(val), "items_per_epoch": len(items),
               "seconds": time.monotonic() - started, "stop_reason": reason, "best": best, "history": history}
    (args.out / "training.json").write_text(json.dumps(summary, indent=2))
    upload(args.upload_repo, args.out, f"{args.prefix}/model", ["training.json"])
    log("training finished:", reason, "best:", best)


def predict(classifier, builder, encoder, reader, frame, args, name, metric=None):
    rows = frame.to_dict("records")
    parts = [[] for _ in D.DIMENSIONS]
    start = time.monotonic()
    text = {}
    with torch.inference_mode():
        for records, features in encode_groups(
                encoder, rows, args.data_dir, text,
                ((batch, IMAGE_SIDE) for batch in chunks(range(len(rows)), args.encode_batch)), reader=reader):
            for part, probs in zip(parts, probabilities(forward(classifier, builder, records, features, text)), strict=True):
                part.append(probs)
    elapsed = time.monotonic() - start
    probs = [torch.cat(part) for part in parts]
    predictions = D.prediction_frame(frame.image_id, probs)
    predictions.to_csv(args.out / f"{name}.csv", index=False)
    torch.save({"image_ids": list(frame.image_id), "dimensions": D.DIMENSIONS, "probabilities": probs},
               args.out / f"{name}.probs.pt")
    report = {"images": len(frame), "seconds": elapsed, "images_per_second": len(frame) / elapsed,
              "visual_tokens": encoder.grid_side**2,
              "gpu": torch.cuda.get_device_name(), "encoder_batch_size": args.encode_batch,
              "note": "Wall time includes reading images, DINOv3, OCR on originals and Apertus."}
    if metric is not None:
        report["metrics"] = metric.evaluate(D.solution_frame(frame), predictions, "image_id")
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=2))
    upload(args.upload_repo, args.out, f"{args.prefix}/predictions", [f"{name}.*"])
    log(f"{name}: {len(frame)} images, {len(frame) / elapsed:.2f} img/s end to end.")


def probe(classifier, builder, records, features, ocr):
    classifier.eval()
    with torch.inference_mode():
        return probabilities(forward(classifier, builder, records, features, ocr))


def train_progressive(args, classifier, settings, builder, fit, val, labels, items,
                      cache, ocr, metric, real_count, baseline, baseline_features, baseline_probe):
    rows = fit.to_dict("records")
    check_rows = val.head(4).to_dict("records")
    check_features = torch.stack([cache[row["image_id"]] for row in check_rows])
    best_path = args.out / "best"
    shutil.copytree(args.initial, best_path)
    best, history = baseline, [baseline]
    best_probe, best_features = baseline_probe, baseline_features
    upload(args.upload_repo, best_path, f"{args.prefix}/model/best")

    rates = (3e-5, 8e-5, 2e-5)
    groups = [list(classifier.projector.parameters()),
              [*classifier.heads.parameters(), *classifier.pool.parameters()],
              [p for p in classifier.language_model.parameters() if p.requires_grad]]
    optimizer = torch.optim.AdamW([{"params": parameters, "lr": rate, "weight_decay": decay}
                                 for parameters, rate, decay in zip(groups, rates, (0., .05, .01), strict=True)],
                                foreach=False)
    masters = BaseMasters(classifier.language_model, settings["layers"])
    alphas = D.class_alpha(D.item_labels(labels, [(i, v) for i, v in items if i < real_count]))
    batch_size = args.train_batch
    steps_per_stage = math.ceil(len(items) / (4 * batch_size))
    generator = torch.Generator().manual_seed(3407)
    started = time.monotonic()
    deadline = min(started + 60 * args.max_minutes, args.deadline or math.inf)
    step, reason = 0, "planned_depth"
    previous_stage = baseline
    for blocks, rate in ((0, None), (2, 2e-6), (4, 1e-6), (6, 6e-7)):
        if time.monotonic() >= deadline:
            reason = "time_limit"
            break
        if blocks:
            new = masters.expand(blocks)
            optimizer.add_param_group({"params": new, "lr": rate, "weight_decay": 0.})
        log(f"stage 144 tokens / {blocks} unfrozen base blocks: {steps_per_stage} steps planned")
        classifier.train()
        stage_steps, failed = 0, False
        window = []
        order = torch.randperm(len(items), generator=generator).tolist()
        for positions in chunks(order[:steps_per_stage * 2 * batch_size], 2 * batch_size):
            if time.monotonic() >= deadline:
                break
            if blocks:
                optimizer.param_groups[-1]["lr"] = rate * min(1., (stage_steps + 1) / 20)
            microbatches = list(chunks(positions, batch_size))
            optimizer.zero_grad(set_to_none=True)
            for microbatch in microbatches:
                views = [items[i] for i in microbatch]
                indices = [index for index, _ in views]
                records = [rows[index] for index in indices]
                pixel_features = torch.stack([cache[D.key(row["image_id"], variant)]
                                              for row, (_, variant) in zip(records, views, strict=True)])
                target = labels[indices].to("cuda")
                synthetic = torch.tensor([i >= real_count for i in indices], device="cuda")
                parts = losses_with_synthetic(forward(classifier, builder, records, pixel_features, ocr),
                                             target, synthetic, alphas)
                parts = [loss for loss in parts if loss is not None]
                if parts:
                    loss = sum(parts)
                    (loss / len(microbatches)).backward()
                    window.append(loss.detach())
            masters.transfer_gradients()
            if blocks and not stage_steps:
                sample = next((p for p in optimizer.param_groups[-1]["params"]
                               if p.grad is not None and p.grad.flatten()[:4096].abs().max().item() > 0), None)
                if sample is None:
                    log(f"No gradient reached the newly unfrozen blocks ({blocks}); retaining the best checkpoint.")
                    reason, failed = "no_base_gradient", True
                    break
                old_sample = sample.detach().flatten()[:4096].clone()
            trainable = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(grad_norm):
                log(f"Non-finite gradient with {blocks} blocks; retaining the best checkpoint.")
                reason, failed = "nonfinite_gradient", True
                break
            optimizer.step()
            if blocks and not stage_steps:
                if torch.equal(sample.detach().flatten()[:4096], old_sample):
                    log("FP32 masters did not change; retaining the best checkpoint.")
                    reason, failed = "no_master_update", True
                    break
                log(f"FP32 master update verified with {blocks} blocks active")
            masters.sync()
            step += 1
            stage_steps += 1
            if stage_steps % 100 == 0:
                log(f"stage {blocks} blocks: {stage_steps}/{steps_per_stage} steps")

        if failed:
            break
        if not stage_steps:
            reason = "time_limit"
            break
        results, val_loss = evaluate(classifier, builder, val, cache, ocr, metric)
        record = {"step": step, "grid_side": 12, "unfrozen_blocks": blocks,
                  "macro_f1": results["overall"]["f1_macro"], "weighted_f1": results["overall"]["f1_weighted"],
                  "validation_loss": val_loss, "training_loss": torch.stack(window).mean().item(),
                  "dimensions": {d: results[d]["f1_macro"] for d in D.DIMENSIONS}}
        history.append(record)
        log(f"stage validation: 144 tokens, {blocks} blocks | macro F1 {record['macro_f1']:.5f} | "
            + " ".join(f"{d}={value:.3f}" for d, value in record["dimensions"].items()))
        if record["macro_f1"] > best["macro_f1"]:
            best = record
            settings["unfrozen_blocks"] = blocks
            best_probe, best_features = probe(classifier, builder, check_rows, check_features, ocr), check_features
            M.save(classifier, settings, best_path)
            masters.save(best_path)
            upload(args.upload_repo, best_path, f"{args.prefix}/model/best")
        if stage_steps < steps_per_stage:
            reason = "time_limit"
            break
        if blocks and (record["macro_f1"] - previous_stage["macro_f1"] < .002
                       or record["macro_f1"] < baseline["macro_f1"] - .005):
            reason = "unfreezing_did_not_help"
            break
        previous_stage = record

    summary = {"fit_images": len(fit), "synthetic_images": len(fit) - real_count,
               "validation_images": len(val), "stop_reason": reason, "best": best, "history": history,
               "seconds": time.monotonic() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
    (args.out / "training.json").write_text(json.dumps(summary, indent=2))
    upload(args.upload_repo, args.out, f"{args.prefix}/model", ["training.json"])
    log("training finished:", reason, "best:", best)
    return check_rows, best_features, best_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initial", type=Path, help="the original SigLIP + OCR checkpoint")
    source.add_argument("--checkpoint", type=Path, help="a trained DINO checkpoint, for inference")
    parser.add_argument("--dino-revision", help="resolved DINO revision, required for training")
    parser.add_argument("--source-id", help="repository and path of the source checkpoint")
    parser.add_argument("--source-revision", help="resolved revision of the source checkpoint")
    parser.add_argument("--ocr-cache", type=Path)
    parser.add_argument("--resume-dino", action="store_true", help="keep the trained DINO projector, LoRA, pooling and heads")
    parser.add_argument("--progressive", action="store_true", help="144 visual tokens, then progressively unfreeze the final base blocks")
    parser.add_argument("--synthetic", type=Path, help="synthetic manifest for training only, based on the same fit products")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-once-per-epoch", action="store_true", help="keep the first mid-epoch check, then validate only at epoch ends")
    parser.add_argument("--max-minutes", type=float, default=100)
    parser.add_argument("--wall-seconds", type=float, default=0, help="remaining time allocated to training and encoding")
    parser.add_argument("--encode-batch", type=int, default=16)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--upload-repo", default="")
    parser.add_argument("--prefix", default="dino")
    parser.add_argument("--predict-dev", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_minutes <= 0 or min(args.encode_batch, args.train_batch) < 1:
        parser.error("Epochs, training time and batch sizes must be positive.")
    if args.initial and not args.dino_revision:
        parser.error("--dino-revision is required for training.")
    if (args.synthetic or args.resume_dino) and not args.initial:
        parser.error("Synthetic training and continuation require --initial.")
    if args.progressive and not (args.resume_dino and args.synthetic):
        parser.error("Progressive unfreezing requires the existing DINO checkpoint and synthetic recipe.")
    args.deadline = time.monotonic() + args.wall_seconds if args.wall_seconds > 0 else 0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    directory = E.model_dir()
    metric = D.load_metric(args.data_dir)
    rows = D.read_csv(args.data_dir / "train.csv")

    if args.initial:
        fit, val, labels, items = training_rows(rows)
        real_count = len(fit)
        real_variants = len(items) - real_count
        if args.synthetic:
            fit, labels, items = add_synthetic(fit, val, labels, items, D.read_csv(args.synthetic), args.data_dir)
        ocr = json.loads(args.ocr_cache.read_text()) if args.ocr_cache else {}
        for frame in (fit, val):
            if not frame.image_id.is_unique:
                raise ValueError("Duplicate image identifiers in the data.")
        log(f"training {real_count} real, {len(fit) - real_count} synthetic, {real_variants} scale variants; validating {len(val)} real images")
        encoder, reader = DinoEncoder(args.dino_revision), E.TextReader()
        classifier, settings, builder = make_model(directory, args.initial, warm_start=not args.resume_dino)
        settings["dino_revision"] = args.dino_revision
        if args.source_id:
            settings["init_checkpoint"] = args.source_id
            settings["init_revision"] = args.source_revision
        if args.progressive:
            if settings["soft_side"] != 9 or settings.get("unfrozen_blocks", 0):
                raise ValueError("Progressive training expects the original 81-token DINO + LoRA checkpoint.")
            validation_records = val.to_dict("records")
            baseline_cache = cached_features(encoder, reader, args.data_dir, validation_records,
                                             [(i, "") for i in range(len(val))], ocr, args.encode_batch)
            results, val_loss = evaluate(classifier, builder, val, baseline_cache, ocr, metric)
            baseline = {"step": 0, "grid_side": 9, "unfrozen_blocks": 0, "validation_loss": val_loss,
                        "macro_f1": results["overall"]["f1_macro"], "weighted_f1": results["overall"]["f1_weighted"],
                        "dimensions": {d: results[d]["f1_macro"] for d in D.DIMENSIONS}}
            baseline_features = torch.stack([baseline_cache[row["image_id"]] for row in validation_records[:4]])
            baseline_probe = probe(classifier, builder, validation_records[:4], baseline_features, ocr)
            del baseline_cache
            log(f"baseline validation: 81 tokens, macro F1 {baseline['macro_f1']:.5f}")
            encoder.grid_side = classifier.soft_side = settings["soft_side"] = 12
            builder.prefix_tokens = 144
        # Check the real models and data before preparing the full split.
        smoke = fit.head(2).to_dict("records")
        for side in (IMAGE_SIDE, AUGMENTED_SIDE):
            records, features = next(encode_groups(encoder, smoke, args.data_dir, ocr, [([0, 1], side)], reader=reader))
            with torch.no_grad():
                probabilities(forward(classifier.eval(), builder, records, features, ocr))
        # Exercise backward through the decoder with LoRA enabled before encoding the entire dataset.
        classifier.train()
        logits = forward(classifier, builder, records, features.clone(), ocr)
        target = labels[:len(records)].to("cuda")
        loss = sum(value for value in M.task_losses(logits, target, 2.) if value is not None)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0, error_if_nonfinite=True)
        classifier.zero_grad(set_to_none=True)
        del logits, loss
        log("GPU smoke passed at both resolutions, including backward")
        all_records = fit.to_dict("records") + val.to_dict("records")
        views = items + [(len(fit) + i, "") for i in range(len(val))]
        cache = cached_features(encoder, reader, args.data_dir, all_records, views, ocr, args.encode_batch)
        if args.deadline and time.monotonic() >= args.deadline:
            raise TimeoutError("Image encoding exhausted the training allocation.")
        # The encoders stay in RAM during optimization only when needed for final inference.
        encoder.model.to("cpu")
        reader.predictor.to("cpu")
        torch.cuda.empty_cache()
        if args.progressive:
            check_rows, best_features, best_probe = train_progressive(
                args, classifier, settings, builder, fit, val, labels, items, cache, ocr, metric,
                real_count, baseline, baseline_features, baseline_probe)
        else:
            train(args, classifier, settings, builder, fit, val, labels, items, cache, ocr, metric, real_count)
        del classifier, cache
        torch.cuda.empty_cache()
        classifier, settings, builder = make_model(directory, args.out / "best")
        if args.progressive:
            restored_probe = probe(classifier, builder, check_rows, best_features, ocr)
            for actual, expected in zip(restored_probe, best_probe, strict=True):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            log("Reloaded best checkpoint reproduces the saved probabilities")
        encoder.grid_side = settings["soft_side"]
        encoder.model.to("cuda")
        reader.predictor.to("cuda")
    else:
        classifier, settings, builder = make_model(directory, args.checkpoint)
        encoder, reader = DinoEncoder(settings["dino_revision"], grid_side=settings["soft_side"]), E.TextReader()
        val = D.select(rows, "test")

    classifier.eval()
    predict(classifier, builder, encoder, reader, val, args, "prompt_test", metric)
    if args.predict_dev:
        predict(classifier, builder, encoder, reader, D.read_csv(args.data_dir / "dev.csv"), args, "dev")


if __name__ == "__main__":
    main()
