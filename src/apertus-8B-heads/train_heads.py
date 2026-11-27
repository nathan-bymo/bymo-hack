"""Train the six heads on frozen features (phase 1) and write predictions for an evaluation set.

Data: by default common.splits.training_rows for training and the coverage set for evaluation. --train-csv and
--eval-csv replace them, for example with a smaller prototype split. A part of the training products, chosen by
hash, is the internal validation used to pick the layer and the epoch (--val-fraction). The evaluation set is never
used for either; it is only predicted at the end. Training never contains a product of the coverage set or of the
evaluation set. The loss sees only human labels in relevant cells.

Four variants show what each ingredient adds: ce, focal, ce+gradnorm, focal+gradnorm.

  python src/apertus-8B-heads/train_heads.py --features data/features/8b_512 --out data/runs/heads_8b_512
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT / "src" / "scripts")]
from common import prompt, splits  # noqa: E402
from eval_prompt import load_metric, solution_frame  # noqa: E402
from heads import GradNorm, SharedHeads, task_losses  # noqa: E402

VARIANTS = {"ce": (False, False), "focal": (True, False), "ce+gradnorm": (False, True),
            "focal+gradnorm": (True, True)}


def load_features(folder):
    ids, parts = [], []
    for shard in sorted(folder.glob("shard_*.pt")):
        content = torch.load(shard, weights_only=True)
        ids += content["image_ids"]
        parts.append(content["features"])
    if not ids:
        sys.exit(f"no feature shards in {folder}")
    return {image_id: index for index, image_id in enumerate(ids)}, torch.cat(parts)


def label_tensor(rows):
    """[rows, 6] class indices, -1 where the cell does not count."""
    scored = splits.scored_cells(rows)
    columns = []
    for dimension in prompt.DIMENSIONS:
        index = {name: i for i, name in enumerate(prompt.TAXONOMY[dimension])}
        columns.append([index[value] if keep else -1 for value, keep in zip(rows[dimension], scored[dimension])])
    return torch.tensor(columns).T


@torch.inference_mode()
def predict(model, features, rows):
    model.eval()
    logits = model(features)
    frame = {"image_id": list(rows.image_id)}
    for dimension, task_logits in zip(prompt.DIMENSIONS, logits):
        frame[dimension] = [prompt.TAXONOMY[dimension][i] for i in task_logits.argmax(-1).tolist()]
    return pd.DataFrame(frame)


def train_one(variant, fit, validation, metric, validation_rows, settings, device):
    """The best model on the internal validation over all epochs, its score, epoch and task weights."""
    use_focal, use_gradnorm = VARIANTS[variant]
    torch.manual_seed(settings.seed)
    features, labels = fit
    model = SharedHeads(features.shape[1], [len(prompt.TAXONOMY[d]) for d in prompt.DIMENSIONS],
                        settings.hidden, settings.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, settings.epochs * -(-len(features) // settings.batch_size))
    balancer = GradNorm(len(prompt.DIMENSIONS), model.shared.weight, settings.alpha, settings.weight_lr) \
        if use_gradnorm else None
    solution = solution_frame(validation_rows)
    generator = torch.Generator().manual_seed(settings.seed)
    best = {"score": -1.0}
    for epoch in range(1, settings.epochs + 1):
        model.train()
        for batch in torch.randperm(len(features), generator=generator).split(settings.batch_size):
            losses = task_losses(model(features[batch].to(device)), labels[batch].to(device),
                                 settings.gamma if use_focal else 0.0)
            if balancer:
                balancer.step(losses)
                total = balancer.combine(losses)
            else:
                total = sum(loss for loss in losses if loss is not None)
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.clip)
            optimizer.step()
            schedule.step()
        score = metric.score(solution, predict(model, validation, validation_rows), "image_id")
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "model": copy.deepcopy(model.state_dict()),
                    "weights": balancer.weights.detach().cpu().tolist() if balancer else None}
    model.load_state_dict(best["model"])
    return model, best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="folder for predictions and heads, inside data/")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--train-csv", type=Path, help="training images instead of training_rows")
    parser.add_argument("--eval-csv", type=Path, help="evaluation images instead of the coverage set")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="training products used to pick layer and epoch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0, help="focal loss focusing parameter")
    parser.add_argument("--alpha", type=float, default=1.5, help="GradNorm asymmetry")
    parser.add_argument("--weight-lr", type=float, default=0.025, help="learning rate of the GradNorm weights")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient norm clipping, all variants")
    parser.add_argument("--seed", type=int, default=3407)
    settings = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # the heads are tiny; CPU is enough
    metric = load_metric()
    index, features = load_features(settings.features)
    def read(path):
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    rows = read(ROOT / "data" / "train.csv")
    coverage = splits.split_rows(rows, "coverage")
    training = read(settings.train_csv) if settings.train_csv else splits.training_rows(rows)
    evaluation = read(settings.eval_csv) if settings.eval_csv else coverage
    for name, other in (("the evaluation set", evaluation), ("the coverage set", coverage)):
        shared = set(training.product_id) & set(other.product_id)
        if shared:
            sys.exit(f"{len(shared)} training products are also in {name}: the scores would not be valid")
    is_validation = training.product_id.map(
        lambda p: int(splits.digest("heads-val:" + p), 16) % 1000 < 1000 * settings.val_fraction)
    parts = {"fit": training[~is_validation].reset_index(drop=True),
             "validation": training[is_validation].reset_index(drop=True),
             "evaluation": evaluation.reset_index(drop=True)}
    for name, part in parts.items():
        missing = sum(i not in index for i in part.image_id)
        if missing:
            sys.exit(f"{missing} images of '{name}' have no features: finish extract_features.py first")
    print(" | ".join(f"{name}: {len(part)} images" for name, part in parts.items()))

    positions = {name: torch.tensor([index[i] for i in part.image_id]) for name, part in parts.items()}
    labels = label_tensor(parts["fit"])
    settings.out.mkdir(parents=True, exist_ok=True)
    results = {variant: [] for variant in settings.variants.split(",")}
    chosen = {}
    for layer in range(features.shape[1]):
        # standardise with the statistics of the fit rows only
        fit = features[positions["fit"], layer].float()
        mean, std = fit.mean(0), fit.std(0).clamp(min=1e-5)
        matrices = {name: (features[positions[name], layer].float() - mean) / std for name in parts}
        for variant in results:
            model, best = train_one(variant, (matrices["fit"], labels), matrices["validation"].to(device), metric,
                                    parts["validation"], settings, device)
            results[variant].append(best["score"])
            print(f"layer {layer} {variant:15} validation {best['score']:.4f} at epoch {best['epoch']}", flush=True)
            if variant not in chosen or best["score"] > chosen[variant]["score"]:
                chosen[variant] = {
                    "score": best["score"], "layer": layer, "epoch": best["epoch"], "weights": best["weights"],
                    "predictions": predict(model, matrices["evaluation"].to(device), parts["evaluation"]),
                    "checkpoint": {"model": model.state_dict(), "mean": mean, "std": std, "layer": layer,
                                   "settings": vars(settings) | {"features": str(settings.features),
                                                                 "out": str(settings.out)}},
                }

    print("\ninternal validation macro F1 (picks layer and epoch; not the evaluation score)")
    print("| variant | chosen layer position | epoch | macro F1 | GradNorm weights |\n|---|---|---|---|---|")
    for variant, best in chosen.items():
        weights = ", ".join(f"{d} {w:.2f}" for d, w in zip(prompt.DIMENSIONS, best["weights"] or [])) or "-"
        print(f"| {variant} | {best['layer']} | {best['epoch']} | {best['score']:.4f} | {weights} |")
        best["predictions"].to_csv(settings.out / f"{variant}.csv", index=False)
        torch.save(best["checkpoint"], settings.out / f"{variant}.pt")
    csv_files = " ".join(str(settings.out / f"{variant}.csv") for variant in chosen)
    target = f"--rows {settings.eval_csv}" if settings.eval_csv else "--split coverage"
    print("\nScore on the evaluation set, next to the zero-shot baseline:\n"
          f"  python src/scripts/eval_prompt.py 8b:describe:codes:512 {csv_files} {target} --classes")


if __name__ == "__main__":
    main()
