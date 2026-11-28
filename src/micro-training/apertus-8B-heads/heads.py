"""Six classification heads on frozen Apertus features, with focal loss and GradNorm.

    features -> shared layer (trainable) -> one linear head per dimension

GradNorm balances the six losses by looking at the gradients they send into weights that all tasks share. With a
frozen backbone and six separate linear heads there would be no such weights, hence the small shared layer.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SharedHeads(nn.Module):
    def __init__(self, input_size, class_counts, hidden_size=512, dropout=0.2):
        super().__init__()
        self.shared = nn.Linear(input_size, hidden_size)
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList(nn.Linear(hidden_size, count) for count in class_counts)

    def forward(self, features):
        hidden = self.body(self.shared(features))
        return [head(hidden) for head in self.heads]


MAX_CLASS_WEIGHT = 10.0


def class_weights(labels, class_count, power):
    """Weight of each class, proportional to (1 / training examples) ** power. This is the alpha of the focal loss.

    labels: the labeled training targets of one task. power 0 gives no weighting, 0.5 a moderate one, 1 the full
    inverse frequency. The weights are scaled so that their mean over the training rows is 1 (the loss keeps its
    scale), then capped so that a class with one or two examples cannot dominate. Classes without examples get 0.
    """
    counts = torch.bincount(labels, minlength=class_count).float()
    weights = torch.where(counts > 0, counts.clamp(min=1) ** -power, torch.zeros_like(counts))
    weights *= counts.sum() / (weights * counts).sum()
    return weights.clamp(max=MAX_CLASS_WEIGHT)


def focal_loss(logits, targets, gamma, weights=None):
    """Mean of -alpha_y * (1 - p_t)^gamma * log(p_t) over the rows. gamma = 0 is the (weighted) cross entropy."""
    log_pt = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    losses = -((1 - log_pt.exp()).clamp(min=0) ** gamma) * log_pt
    if weights is not None:
        losses = losses * weights[targets]
    return losses.mean()


def task_losses(logits, labels, gamma, weights=None):
    """One focal loss per dimension over its labeled rows only (label -1: not scored). None if a batch has none."""
    losses = []
    for task, task_logits in enumerate(logits):
        known = labels[:, task] >= 0
        losses.append(focal_loss(task_logits[known], labels[known, task], gamma,
                                 None if weights is None else weights[task]) if known.any() else None)
    return losses


class GradNorm:
    """GradNorm (Chen et al., ICML 2018): learns one weight per task so that all tasks train at a similar rate.

    For task i with weight w_i: G_i = || grad of w_i L_i on the shared weights ||. Tasks that learned less so far
    (high L_i / L_i(0) compared with the others) get a larger target gradient norm: mean(G) * rate_i ** alpha.
    The weights move to bring G_i to its target, and are rescaled to sum to the number of tasks.
    """

    def __init__(self, tasks, shared_weight, alpha=1.5, learning_rate=0.025):
        self.shared_weight, self.alpha = shared_weight, alpha
        self.weights = torch.ones(tasks, device=shared_weight.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        self.initial_losses = None

    def step(self, losses):
        """Update the weights from this batch. Call before the backward pass of the weighted loss."""
        if any(loss is None for loss in losses):
            return  # a task without labels in the batch has no gradient to compare
        norms = torch.stack([
            torch.autograd.grad(loss, self.shared_weight, retain_graph=True)[0].norm() for loss in losses
        ]).detach()
        values = torch.stack([loss.detach() for loss in losses])
        if self.initial_losses is None:
            self.initial_losses = values.clamp(min=1e-8)
        weighted = self.weights * norms
        progress = values / self.initial_losses
        rate = progress / progress.mean()
        target = (weighted.mean() * rate**self.alpha).detach()
        self.optimizer.zero_grad()
        self.weights.grad = torch.autograd.grad((weighted - target).abs().sum(), self.weights)[0]
        self.optimizer.step()
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            self.weights *= len(losses) / self.weights.sum()

    def combine(self, losses):
        return sum(self.weights[i].detach() * loss for i, loss in enumerate(losses) if loss is not None)
