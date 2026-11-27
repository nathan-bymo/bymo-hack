"""Apertus 8B with LoRA -> 6 classification heads, trained together.

    hidden    = Apertus layers 1..L (LoRA) on [template, VQ image tokens, context, task]
    features  = [last token, mean of the VQ image tokens]      2 x 4096
    + optional spatial features: (left half - right half), (top half - bottom half) of the VQ tokens
    + optional attention pooling: learned queries over the VQ tokens, which carry their 2D position
    logits    = heads(features, spatial, pooled)               one linear head per dimension

One forward pass per image, no text generation. With L < 32 the layers after L are deleted.

Spatial features and attention pooling follow apertus-8B-vadapter-lora-heads/model.py (Nathan): with the means alone
an image and its mirror give the same features, so left and right cannot be told apart.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from data import CLASS_COUNTS

# The same modules as the LoRA baseline: attention and MLP projections of the language model.
LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(up_proj|down_proj))$"
# The VQ and audio tokenizers do not support sdpa; only the language model is used here.
ATTENTION = {"text_config": "sdpa", "vision_tokenizer_config": "eager", "audio_tokenizer_config": "eager"}


class Heads(nn.Module):
    """LayerNorm (the features move while LoRA trains), shared layer, one linear head per dimension.

    Optional extra inputs (spatial differences, attention pooling) each go through their own LayerNorm and linear
    layer into the same hidden layer.
    """

    def __init__(self, input_size, class_counts, hidden_size=512, dropout=0.3, spatial_size=0, pooled_size=0):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.shared = nn.Linear(input_size, hidden_size)
        self.spatial_size, self.pooled_size = spatial_size, pooled_size
        for name, size in (("spatial", spatial_size), ("pooled", pooled_size)):
            if size:
                setattr(self, f"norm_{name}", nn.LayerNorm(size))
                setattr(self, f"shared_{name}", nn.Linear(size, hidden_size))
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList(nn.Linear(hidden_size, count) for count in class_counts)

    def forward(self, features, spatial=None, pooled=None):
        hidden = self.shared(self.norm(features))
        if self.spatial_size:
            hidden = hidden + self.shared_spatial(self.norm_spatial(spatial))
        if self.pooled_size:
            hidden = hidden + self.shared_pooled(self.norm_pooled(pooled))
        hidden = self.body(hidden)
        return [head(hidden) for head in self.heads]


def focal_loss(logits, targets, gamma=0.0, alpha=None):
    """Mean of -alpha_t (1 - p_t)^gamma log p_t. gamma 0 and no alpha: plain cross entropy (the default)."""
    log_pt = F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    loss = -((1 - log_pt.exp()).clamp(min=0) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * alpha.to(loss.device)[targets]
    return loss.mean()


def task_losses(logits, labels, gamma=0.0, alphas=None):
    """One loss per dimension over its labeled rows (label -1: left out). None if a batch has none."""
    losses = []
    for task, task_logits in enumerate(logits):
        known = labels[:, task] >= 0
        alpha = alphas[task] if alphas is not None else None
        losses.append(focal_loss(task_logits[known], labels[known, task], gamma, alpha) if known.any() else None)
    return losses


def pooled_features(hidden, input_ids, attention, image_range):
    """[batch, 2 * hidden]: the last token and the mean of the VQ image tokens."""
    hidden = hidden.float()
    low, high = image_range
    mask = ((input_ids >= low) & (input_ids < high) & (attention > 0)).to(hidden.dtype)[..., None]
    return torch.cat([hidden[:, -1], (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)], dim=-1)


def image_grids(input_ids, image_range, row_end_id):
    """Per row of the batch: (positions of the VQ tokens, grid width). The width is the number of VQ tokens before
    the first end-of-row token (the whole run when there is a single row)."""
    low, high = image_range
    result = []
    for ids in input_ids:
        vq = ((ids >= low) & (ids < high)).nonzero().squeeze(1)
        width = vq.numel()
        ends = (ids == row_end_id).nonzero().squeeze(1)
        if vq.numel() and ends.numel():
            width = int((vq < ends[0]).sum())
        result.append((vq, max(1, width)))
    return result


def grid_halves(count, width, device=None):
    """Masks (left, right, top, bottom) over `count` tokens of a row-major grid `width` wide; the middle column or
    row of an odd grid belongs to neither half."""
    index = torch.arange(count, device=device)
    column, row = index % width, index // width
    height = max(1, -(-count // width))
    return column < (width - 1) / 2, column > (width - 1) / 2, row < (height - 1) / 2, row > (height - 1) / 2


def spatial_features(hidden, grids):
    """[batch, 2 * hidden]: (left half - right half) and (top half - bottom half) of the VQ tokens. They change sign
    for a mirrored image, where the means stay the same."""
    hidden = hidden.float()
    zero = hidden.new_zeros(hidden.shape[-1])
    rows = []
    for b, (positions, width) in enumerate(grids):
        if positions.numel() == 0:
            rows.append(torch.cat([zero, zero]))
            continue
        tokens = hidden[b, positions]
        left, right, top, bottom = grid_halves(positions.numel(), width, hidden.device)
        mean = lambda mask: tokens[mask].mean(0) if mask.any() else zero
        rows.append(torch.cat([mean(left) - mean(right), mean(top) - mean(bottom)]))
    return torch.stack(rows)


class AttentionPool(nn.Module):
    """Learned queries over the VQ tokens, which carry their 2D position (Fourier features of row and column).
    Without the positions an attention pooling would be as blind to left and right as the mean."""

    def __init__(self, hidden_size, width=256, queries=8, heads=4, frequencies=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, width)
        self.value = nn.Linear(hidden_size, width)
        self.position = nn.Linear(4 * frequencies, width)
        self.register_buffer("frequencies", torch.pi * 2.0 ** torch.arange(frequencies), persistent=False)
        self.queries = nn.Parameter(torch.randn(queries, width) / width**0.5)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output_size = queries * width

    def encode_positions(self, count, width, device):
        index = torch.arange(count, device=device)
        height = max(1, -(-count // width))
        coords = torch.stack([(index % width) / max(1, width - 1), (index // width) / max(1, height - 1)], 1)
        angles = coords[..., None] * self.frequencies.to(device)
        return torch.cat([angles.sin().flatten(1), angles.cos().flatten(1)], 1)

    def forward(self, hidden, grids):
        hidden = hidden.float()
        length = max(1, max(positions.numel() for positions, _ in grids))
        tokens = hidden.new_zeros(len(grids), length, hidden.shape[-1])
        codes = hidden.new_zeros(len(grids), length, self.position.in_features)
        padding = torch.ones(len(grids), length, dtype=torch.bool, device=hidden.device)
        for b, (positions, width) in enumerate(grids):
            n = positions.numel()
            if n:
                tokens[b, :n] = hidden[b, positions]
                codes[b, :n] = self.encode_positions(n, width, hidden.device)
                padding[b, :n] = False
            else:
                padding[b, 0] = False  # an image without tokens attends to a zero vector instead of nothing
        normed, place = self.norm(tokens), self.position(codes)
        queries = self.queries.unsqueeze(0).expand(len(grids), -1, -1)
        pooled, _ = self.attention(queries, self.key(normed) + place, self.value(normed) + place,
                                   key_padding_mask=padding, need_weights=False)
        return pooled.flatten(1)


class Classifier(nn.Module):
    """Wraps a decoder with forward(input_ids=, attention_mask=, position_ids=).last_hidden_state."""

    def __init__(self, language_model, heads, image_range, row_end_id=None, pool=None, spatial=False):
        super().__init__()
        self.language_model, self.heads, self.image_range = language_model, heads, image_range
        self.row_end_id, self.pool, self.spatial = row_end_id, pool, spatial

    def forward(self, input_ids, attention_mask):
        if not attention_mask[:, -1].all():
            raise ValueError("the last position is padding: left padding is required")
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        hidden = self.language_model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, use_cache=False).last_hidden_state
        features = pooled_features(hidden, input_ids, attention_mask, self.image_range)
        spatial = pooled = None
        if self.spatial or self.pool is not None:
            grids = image_grids(input_ids, self.image_range, self.row_end_id)
            if self.spatial:
                spatial = spatial_features(hidden, grids)
            if self.pool is not None:
                pooled = self.pool(hidden, grids)
        return self.heads(features, spatial, pooled)


# ---------------------------------------------------------------- Apertus


def load_language_model(directory, layers, device="cuda"):
    """The text decoder of Apertus v1.5 8B in bf16, cut after `layers` layers; returns (decoder, image id range)."""
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

    full = Apertus1p5ForConditionalGeneration.from_pretrained(
        directory, dtype=torch.bfloat16, device_map={"": device}, attn_implementation=ATTENTION)
    config = full.config
    decoder = full.model.language_model
    del full  # the VQ/audio tokenizers and the lm_head are not needed
    low = config.image_token_offset
    image_range = (low, low + config.vision_tokenizer_config.codebook_size)
    return truncate(decoder, layers), image_range


def truncate(decoder, layers):
    total = decoder.config.num_hidden_layers
    if not 1 <= layers <= total:
        raise ValueError(f"layers must be in 1..{total}")
    decoder.layers = decoder.layers[:layers]
    decoder.config.num_hidden_layers = layers
    decoder.requires_grad_(False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decoder


def add_lora(decoder, rank, alpha, dropout=0.0):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=LORA_TARGETS)
    return get_peft_model(decoder, config)


def make_classifier(decoder, image_range, settings):
    """Heads (and the optional pooling) on a decoder, from settings; checkpoints without the new keys load as before."""
    hidden = decoder.config.hidden_size
    spatial, attention = settings.get("spatial", False), settings.get("attention_pool", False)
    if (spatial or attention) and settings.get("row_end_id") is None:
        raise ValueError("spatial features and attention pooling need row_end_id in the settings")
    pool = AttentionPool(hidden) if attention else None
    heads = Heads(2 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"],
                  2 * hidden if spatial else 0, pool.output_size if pool is not None else 0)
    return Classifier(decoder, heads, image_range, settings.get("row_end_id"), pool, spatial)


def build(directory, settings, device="cuda"):
    """Classifier from settings (a dict, the same in training and in the saved checkpoint)."""
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    if settings["lora_rank"]:
        decoder = add_lora(decoder, settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"])
    return make_classifier(decoder, image_range, settings).to(device)


def save(classifier, settings, folder):
    """LoRA adapter (PEFT format), heads and settings; never the base weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if settings["lora_rank"]:
        classifier.language_model.save_pretrained(str(folder / "lora"))
        # The card PEFT writes holds a local base-model path, which the Hub rejects as invalid metadata: the
        # first LoRA prototype lost its adapter this way.
        (folder / "lora" / "README.md").unlink(missing_ok=True)
    trainable = {"heads": classifier.heads.state_dict()}
    if classifier.pool is not None:
        trainable["pool"] = classifier.pool.state_dict()
    torch.save(trainable, folder / "heads.pt")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))


def load(directory, folder, device="cuda", merge=True):
    """A saved checkpoint for inference; LoRA merged into the weights by default (no extra cost per image)."""
    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    if settings["lora_rank"]:
        from peft import PeftModel

        decoder = PeftModel.from_pretrained(decoder, str(folder / "lora"))
        if merge:
            decoder = decoder.merge_and_unload()
    classifier = make_classifier(decoder, image_range, settings)
    trainable = torch.load(folder / "heads.pt", map_location="cpu", weights_only=True)
    classifier.heads.load_state_dict(trainable["heads"])
    if classifier.pool is not None:
        classifier.pool.load_state_dict(trainable["pool"])
    return classifier.to(device).eval(), settings
