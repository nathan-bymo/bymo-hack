"""Validate dev teacher targets and compare the pilot on held-out products."""

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from mixed_model import D


def load_teacher(path, sha256, dev, train):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("Teacher checksum does not match.")
    if (not dev.image_id.is_unique or dev.image_id.isin(train.image_id).any()
            or dev.product_id.isin(train.product_id).any()
            or not dev.image_path.str.fullmatch(r"dev_images/[^/]+").all()):
        raise ValueError("Teacher images must be unique dev images from products outside train.")
    targets = torch.load(path, map_location="cpu", weights_only=True)
    if list(targets["image_ids"]) != dev.image_id.tolist() or tuple(targets["dimensions"]) != tuple(D.DIMENSIONS):
        raise ValueError("Teacher IDs or dimensions do not match dev.csv and the task taxonomy.")
    mask = targets["mask"]
    probs = targets["probabilities"]
    if mask.dtype != torch.bool or mask.shape != (len(dev), len(D.DIMENSIONS)) or not mask.any():
        raise ValueError("Teacher mask must contain at least one active cell.")
    for values, size in zip(probs, D.CLASS_COUNTS, strict=True):
        if (values.dtype != torch.float32 or values.shape != (len(dev), size)
                or not torch.isfinite(values).all() or (values < 0).any()
                or not torch.allclose(values.sum(1), torch.ones(len(dev)), atol=1e-5, rtol=0.)):
            raise ValueError("Teacher values must be normalized float32 probabilities for every dev image.")
    return probs, mask


def soft_losses(logits, probabilities, mask, positions, rows):
    """One loss per dimension, normalized over that dimension's eligible dev cells."""
    losses = []
    for column, (prediction, targets) in enumerate(zip(logits, probabilities, strict=True)):
        keep = mask[rows, column]
        if keep.any():
            device = prediction.device
            log_probs = F.log_softmax(prediction[positions[keep].to(device)].float(), dim=-1)
            losses.append(-(targets[rows[keep]].to(device) * log_probs).sum(dim=-1).mean())
    return losses


def paired_macro_intervals(val, baseline, control, student, samples=2000):
    labels = D.label_matrix(val).numpy()
    preds = [np.stack([p.argmax(1).numpy() for p in branch], axis=1)
             for branch in (baseline, control, student)]
    groups = list(val.groupby("product_id", sort=True).indices.values())
    rng = np.random.default_rng(3407)

    def macro(indices, predicted):
        scores = []
        for column, size in enumerate(D.CLASS_COUNTS):
            actual = labels[indices, column]
            valid = actual >= 0
            actual, guess = actual[valid], predicted[indices, column][valid]
            if actual.size:
                support = np.bincount(actual, minlength=size) + np.bincount(guess, minlength=size)
                hits = np.bincount(actual[actual == guess], minlength=size)
                present = support > 0
                scores.append((2. * hits[present] / support[present]).mean())
        return np.mean(scores)

    differences = []
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        before, supervised, distilled = [macro(indices, prediction) for prediction in preds]
        differences.append((distilled - before, distilled - supervised))
    limits = np.quantile(differences, [.025, .975], axis=0).T
    return dict(zip(("student_minus_baseline", "student_minus_control"), limits.tolist(), strict=True))
