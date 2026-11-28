"""Averages the class probabilities of several models, dimension by dimension, and writes the predicted classes.

Each input is a .pt file with image_ids and probabilities, in one of the two formats of the team:
  - src/training/apertus-8B-lora-heads/predict.py: {"image_ids", "dimensions", "probabilities": [tensor per dimension]},
    classes in the order of common.prompt.TAXONOMY;
  - the other folders: {"image_ids", "classes": {dimension: [names]}, "probabilities": {dimension: tensor}}.
Classes are matched by name, so the order inside a file does not matter. Files are loaded with weights_only=True.

  python src/scripts/ensemble.py --model p11=data/ensemble/p11_dev_probs.pt --model vlh=data/ensemble/vlh_dev.pt \\
      --weights "p11=1,vlh=1" --dimension-weights "text_language:vlh=1" --out data/ensemble/dev_ensemble.csv

--weights gives the default weight of every model; --dimension-weights replaces them for one dimension (models left
out get 0). With --rows (a csv with the columns of train.csv) the result is also scored with data/metric.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "training"))
from common import prompt  # noqa: E402

DIMENSIONS, TAXONOMY = prompt.DIMENSIONS, prompt.TAXONOMY


def load_probabilities(path):
    """(image ids, {dimension: array [images, classes] in TAXONOMY order})."""
    content = torch.load(path, weights_only=True, map_location="cpu")
    ids = list(content["image_ids"])
    result = {}
    for index, dimension in enumerate(DIMENSIONS):
        if isinstance(content["probabilities"], dict):
            values = torch.as_tensor(content["probabilities"][dimension]).double().numpy()
            names = list(content["classes"][dimension])
        else:
            values = torch.as_tensor(content["probabilities"][index]).double().numpy()
            names = list(TAXONOMY[dimension])
        if sorted(names) != sorted(TAXONOMY[dimension]):
            raise ValueError(f"{path}: the classes of {dimension} differ from the taxonomy")
        if values.shape != (len(ids), len(names)) or not np.allclose(values.sum(1), 1, atol=1e-3):
            raise ValueError(f"{path}: {dimension} is not a probability table for every image")
        result[dimension] = values[:, [names.index(name) for name in TAXONOMY[dimension]]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: repeated image ids")
    return ids, result


def parse_weights(text, models):
    weights = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        name, value = part.split("=")
        if name not in models:
            raise ValueError(f"unknown model {name!r} in the weights")
        weights[name] = float(value)
    return weights


def combine(models, weights, dimension_weights):
    """(image ids, {dimension: averaged probabilities}) over the images that every model predicted."""
    ids = sorted(set.intersection(*(set(model_ids) for model_ids, _ in models.values())))
    if not ids:
        raise ValueError("the models have no image in common")
    combined = {}
    for dimension in DIMENSIONS:
        chosen = dimension_weights.get(dimension, weights)
        total = sum(chosen.get(name, 0.0) for name in models)
        if total <= 0:
            raise ValueError(f"{dimension}: every weight is zero")
        mixture = 0
        for name, (model_ids, probabilities) in models.items():
            weight = chosen.get(name, 0.0)
            if weight:
                position = {image_id: i for i, image_id in enumerate(model_ids)}
                mixture = mixture + weight * probabilities[dimension][[position[i] for i in ids]]
        combined[dimension] = mixture / total
    return ids, combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="name=path of a probability file")
    parser.add_argument("--weights", default="", help="name=weight,...; default 1 for every model")
    parser.add_argument("--dimension-weights", action="append", default=[],
                        help="dimension:name=weight,... replaces --weights for that dimension")
    parser.add_argument("--rows", type=Path, help="csv with labels, to score the result with data/metric.py")
    parser.add_argument("--out", type=Path, required=True, help="csv with image_id and the six dimensions")
    args = parser.parse_args(argv)

    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        models[name] = load_probabilities(path)
    weights = {name: 1.0 for name in models} | parse_weights(args.weights, models)
    dimension_weights = {}
    for item in args.dimension_weights:
        dimension, text = item.split(":", 1)
        if dimension not in DIMENSIONS:
            raise ValueError(f"unknown dimension {dimension!r}")
        dimension_weights[dimension] = parse_weights(text, models)
    ids, combined = combine(models, weights, dimension_weights)
    counts = {name: len(model_ids) for name, (model_ids, _) in models.items()}
    print(f"{len(ids)} images in common; images per model: {counts}")

    frame = pd.DataFrame({"image_id": ids,
                          **{d: [TAXONOMY[d][i] for i in combined[d].argmax(1)] for d in DIMENSIONS}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    torch.save({"image_ids": ids, "dimensions": list(DIMENSIONS),
                "probabilities": [torch.tensor(combined[d]) for d in DIMENSIONS]}, args.out.with_suffix(".probs.pt"))
    print(f"written {args.out}")
    if args.rows:
        import importlib.util

        sys.path.insert(0, str(ROOT / "src" / "scripts"))
        from eval_prompt import solution_frame

        spec = importlib.util.spec_from_file_location("metric", ROOT / "data" / "metric.py")
        metric = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)
        rows = pd.read_csv(args.rows, dtype=str, keep_default_na=False)
        rows = rows[rows.image_id.isin(set(ids))].reset_index(drop=True)
        results = metric.evaluate(solution_frame(rows), frame, "image_id")
        print(f"macro F1 {results['overall']['f1_macro']:.4f}, weighted {results['overall']['f1_weighted']:.4f} | "
              + " ".join(f"{d} {results[d]['f1_macro']:.3f}" for d in DIMENSIONS if d in results))


if __name__ == "__main__":
    main()
