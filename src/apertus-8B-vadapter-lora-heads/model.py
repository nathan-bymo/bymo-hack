"""SigLIP 2 features -> projector -> Apertus 8B (truncated, LoRA) -> 6 classification heads.

    soft tokens = projector(pooled SigLIP features)            49 x 4096, trainable
    hidden      = Apertus layers 1..L (LoRA) on [template, soft tokens, VQ image tokens, context, task]
    features    = [last token, mean of VQ image tokens, mean of soft tokens]    3 x 4096
    logits      = heads(features)                              one linear head per dimension

One forward pass per image, no text generation, and the layers after L are deleted (heads on frozen Apertus: layer
24 of 32 was the best on the internal validation, apertus-8B-heads/results_small_h200.md).
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from encoders import PREFIX_SLOT

LORA_TARGETS = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(up_proj|down_proj))$"
# The VQ and audio tokenizers do not support sdpa; only the language model is used here.
ATTENTION = {"text_config": "sdpa", "vision_tokenizer_config": "eager", "audio_tokenizer_config": "eager"}


class Projector(nn.Sequential):
    """Same layout as apertus-vadapter-benchmark (keys 0.* and 2.*), so its checkpoints load as initialisation."""

    def __init__(self, vision_dim, text_dim):
        super().__init__(nn.Linear(vision_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))


class Heads(nn.Module):
    """LayerNorm (the features move while LoRA trains), shared layer, one linear head per dimension.

    With spatial_size > 0 a second input, the spatial differences (see spatial_features), goes through its own
    LayerNorm and linear layer into the same hidden layer. That layer starts at zero: heads loaded from a checkpoint
    without spatial features give exactly the same outputs at the start, and training adds the spatial part.
    """

    def __init__(self, input_size, class_counts, hidden_size=512, dropout=0.3, spatial_size=0, pooled_size=0):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.shared = nn.Linear(input_size, hidden_size)
        self.spatial_size, self.pooled_size = spatial_size, pooled_size
        for name, size in (("spatial", spatial_size), ("pooled", pooled_size)):
            if size:
                setattr(self, f"norm_{name}", nn.LayerNorm(size))
                layer = nn.Linear(size, hidden_size)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                setattr(self, f"shared_{name}", layer)
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


def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """Mean of -alpha_t (1 - p_t)^gamma log p_t. gamma 0 and no alpha: plain cross entropy."""
    log_pt = F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    loss = -((1 - log_pt.exp()).clamp(min=0) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * alpha.to(loss.device)[targets]
    return loss.mean()


def task_losses(logits, labels, gamma, alphas=None):
    """One focal loss per dimension over its scored rows (label -1: not scored). None if a batch has none."""
    losses = []
    for task, task_logits in enumerate(logits):
        known = labels[:, task] >= 0
        alpha = alphas[task] if alphas is not None else None
        losses.append(focal_loss(task_logits[known], labels[known, task], gamma, alpha) if known.any() else None)
    return losses


def pooled_features(hidden, input_ids, attention, image_range):
    """[batch, 3 * hidden]: last token, mean of the VQ image tokens, mean of the soft tokens."""
    hidden = hidden.float()
    low, high = image_range
    masks = [((input_ids >= low) & (input_ids < high) & (attention > 0)), (input_ids == PREFIX_SLOT)]
    means = []
    for mask in masks:
        weights = mask.to(hidden.dtype)[..., None]
        means.append((hidden * weights).sum(1) / weights.sum(1).clamp(min=1))
    return torch.cat([hidden[:, -1], *means], dim=-1)


def grid_halves(count, width):
    """Masks (left, right, top, bottom) over `count` tokens of a row-major grid `width` wide; the middle column
    or row of an odd grid belongs to neither half."""
    index = torch.arange(count)
    column, row = index % width, index // width
    height = max(1, -(-count // width))
    return (column < (width - 1) / 2, column > (width - 1) / 2, row < (height - 1) / 2, row > (height - 1) / 2)


def image_grids(input_ids, image_range, row_end_id, soft_side):
    """Per row of the batch: [(positions, width), ...] for the soft tokens (soft_side wide) and the VQ tokens (as
    wide as their first row, which ends at the first end-of-row token)."""
    low, high = image_range
    result = []
    for ids in input_ids:
        soft = (ids == PREFIX_SLOT).nonzero().squeeze(1)
        vq = ((ids >= low) & (ids < high)).nonzero().squeeze(1)
        width = vq.numel()
        if vq.numel():
            ends = (ids == row_end_id).nonzero().squeeze(1)
            if ends.numel():
                width = int((vq < ends[0]).sum())
        result.append([(soft, soft_side), (vq, max(1, width))])
    return result


def spatial_features(hidden, input_ids, image_range, row_end_id, soft_side):
    """[batch, 4 * hidden]: (left half - right half) and (top half - bottom half) of the soft tokens and of the VQ
    tokens. Means are the same for an image and its mirror; these differences change sign, so the heads can tell
    left from right (left/right were a coin toss with the means alone)."""
    hidden = hidden.float()
    rows = []
    for b, grids in enumerate(image_grids(input_ids, image_range, row_end_id, soft_side)):
        parts = []
        for positions, width in grids:
            if positions.numel() == 0:
                parts += [hidden.new_zeros(hidden.shape[-1])] * 2
                continue
            left, right, top, bottom = (mask.to(hidden.device) for mask in grid_halves(positions.numel(), width))
            tokens = hidden[b, positions]
            mean = lambda mask: tokens[mask].mean(0) if mask.any() else hidden.new_zeros(hidden.shape[-1])
            parts += [mean(left) - mean(right), mean(top) - mean(bottom)]
        rows.append(torch.cat(parts))
    return torch.stack(rows)


class AttentionPool(nn.Module):
    """Learned queries over the image tokens, which carry their 2D position (Fourier features of row and column,
    plus a learned flag soft/VQ). Plain attention pooling would not see where a token is (positions only enter
    Apertus through RoPE, not into the hidden states), so it would stay blind to left/right like the means."""

    def __init__(self, hidden_size, width=256, queries=8, heads=4, frequencies=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, width)
        self.value = nn.Linear(hidden_size, width)
        self.position = nn.Linear(4 * frequencies + 2, width)
        self.register_buffer("frequencies", torch.pi * 2.0 ** torch.arange(frequencies), persistent=False)
        self.queries = nn.Parameter(torch.randn(queries, width) / width**0.5)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output_size = queries * width

    def encode_positions(self, count, width, kind, device):
        index = torch.arange(count, device=device)
        height = max(1, -(-count // width))
        coords = torch.stack([(index % width) / max(1, width - 1), (index // width) / max(1, height - 1)], 1)
        angles = coords[..., None] * self.frequencies.to(device)  # [count, 2, frequencies]
        flag = torch.zeros(count, 2, device=device)
        flag[:, kind] = 1.0
        return torch.cat([angles.sin().flatten(1), angles.cos().flatten(1), flag], 1)

    def forward(self, hidden, grids):
        hidden = hidden.float()
        tokens, codes = [], []
        for b, rows in enumerate(grids):
            picked, positions = [], []
            for kind, (where, width) in enumerate(rows):
                if where.numel():
                    picked.append(hidden[b, where])
                    positions.append(self.encode_positions(where.numel(), width, kind, hidden.device))
            tokens.append(torch.cat(picked) if picked else hidden.new_zeros(1, hidden.shape[-1]))
            codes.append(torch.cat(positions) if positions else hidden.new_zeros(1, self.position.in_features))
        length = max(t.shape[0] for t in tokens)
        padding = torch.ones(len(tokens), length, dtype=torch.bool, device=hidden.device)
        batch_tokens = hidden.new_zeros(len(tokens), length, hidden.shape[-1])
        batch_codes = hidden.new_zeros(len(tokens), length, self.position.in_features)
        for b, (t, c) in enumerate(zip(tokens, codes)):
            batch_tokens[b, :t.shape[0]], batch_codes[b, :t.shape[0]], padding[b, :t.shape[0]] = t, c, False
        normed, place = self.norm(batch_tokens), self.position(batch_codes)
        keys, values = self.key(normed) + place, self.value(normed) + place
        queries = self.queries.unsqueeze(0).expand(len(tokens), -1, -1)
        pooled, _ = self.attention(queries, keys, values, key_padding_mask=padding, need_weights=False)
        return pooled.flatten(1)


class Classifier(nn.Module):
    """Wraps any decoder with .embed_tokens and forward(inputs_embeds=, attention_mask=, position_ids=)."""

    def __init__(self, language_model, projector, heads, image_range, row_end_id=None, soft_side=7, pool=None):
        super().__init__()
        self.language_model, self.projector, self.heads, self.pool = language_model, projector, heads, pool
        self.image_range, self.row_end_id, self.soft_side = image_range, row_end_id, soft_side
        self.residual_dtype = None  # float32 on T4: fp16 weights, but a residual stream beyond the fp16 range

    def embed(self, input_ids, siglip):
        embeddings = self.language_model.get_input_embeddings()(input_ids.clamp(min=0))
        slots = input_ids == PREFIX_SLOT
        if slots.any():
            soft = self.projector(siglip.to(self.projector[0].weight.dtype))
            if int(slots.sum()) != soft.shape[0] * soft.shape[1]:
                raise ValueError("every row needs exactly one block of soft-token slots")
            embeddings = embeddings.masked_scatter(slots[..., None], soft.to(embeddings.dtype))
        if self.residual_dtype is not None:  # under fp16 autocast the matmuls stay fp16, sums and norms run in fp32
            embeddings = embeddings.to(self.residual_dtype)
        return embeddings

    def forward(self, input_ids, attention_mask, siglip):
        if not attention_mask[:, -1].all():
            raise ValueError("the last position is padding: left padding is required")
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        embeds = self.embed(input_ids, siglip)
        with torch.autocast("cuda", enabled=not getattr(self, "safe_fp32", False)):
            hidden = self.language_model(inputs_embeds=embeds, attention_mask=attention_mask,
                                         position_ids=position_ids, use_cache=False).last_hidden_state
        features = pooled_features(hidden, input_ids, attention_mask, self.image_range)
        spatial = pooled = None
        if self.heads.spatial_size:
            spatial = spatial_features(hidden, input_ids, self.image_range, self.row_end_id, self.soft_side)
        if self.pool is not None:
            pooled = self.pool(hidden, image_grids(input_ids, self.image_range, self.row_end_id, self.soft_side))
        return self.heads(features, spatial, pooled)


# ---------------------------------------------------------------- Apertus


def load_language_model(directory, layers, device="cuda", dtype=torch.bfloat16):
    """The text decoder of Apertus v1.5 8B (bf16; fp16 for T4), cut after `layers` layers; (decoder, image id range)."""
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

    full = Apertus1p5ForConditionalGeneration.from_pretrained(
        directory, dtype=dtype, device_map={"": device}, attn_implementation=ATTENTION)
    config = full.config
    decoder = full.model.language_model
    total = decoder.config.num_hidden_layers
    if not 1 <= layers <= total:
        raise ValueError(f"layers must be in 1..{total}")
    decoder.layers = decoder.layers[:layers]
    decoder.config.num_hidden_layers = layers
    del full  # the VQ/audio tokenizers and the 1 GB lm_head are not needed
    import gc

    gc.collect()  # model objects can hold reference cycles: free them before measuring memory
    torch.cuda.empty_cache()
    decoder.requires_grad_(False)
    if dtype == torch.float16:  # xIELU keeps its scalars in the bf16 of the config; T4 has no bf16
        for module in decoder.modules():
            if type(module).__name__ == "XIELUActivation":
                module.to(torch.float16)
    low = config.image_token_offset
    return decoder, (low, low + config.vision_tokenizer_config.codebook_size)


def add_lora(decoder, rank, alpha, dropout=0.0):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=LORA_TARGETS)
    return get_peft_model(decoder, config)


def build(directory, settings, device="cuda"):
    """Classifier from settings (a dict, the same in training and in the saved checkpoint)."""
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    if settings["lora_rank"]:
        decoder = add_lora(decoder, settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"])
    return assemble(decoder, image_range, settings).to(device)


def assemble(decoder, image_range, settings):
    """Projector, heads and optional attention pooling around a decoder, as the settings describe them."""
    from data import CLASS_COUNTS

    hidden = decoder.config.hidden_size
    pool = AttentionPool(hidden) if settings.get("attention_pool") else None
    heads = Heads(3 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"],
                  4 * hidden if settings.get("spatial") else 0, pool.output_size if pool else 0)
    return Classifier(decoder, Projector(settings["vision_dim"], hidden), heads, image_range,
                      settings.get("row_end_id"), pool=pool)


def save(classifier, settings, folder):
    """LoRA adapter (PEFT format), projector + heads and settings; never the 16 GB base weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if settings["lora_rank"]:
        classifier.language_model.save_pretrained(str(folder / "lora"))
    state = {"projector": classifier.projector.state_dict(), "heads": classifier.heads.state_dict()}
    if classifier.pool is not None:
        state["pool"] = classifier.pool.state_dict()
    torch.save(state, folder / "trainable.pt")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))


def load_weights(classifier, folder, settings):
    """Trained LoRA, projector and heads of a checkpoint into a freshly built classifier (to continue training)."""
    folder = Path(folder)
    report = []
    if settings["lora_rank"]:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(folder / "lora" / "adapter_model.safetensors"))
        result = set_peft_model_state_dict(classifier.language_model, state)
        unexpected = list(getattr(result, "unexpected_keys", []))
        if unexpected:
            raise ValueError(f"LoRA weights that fit nowhere: {unexpected[:5]}")
        report.append(f"{len(state)} LoRA tensors")
    trainable = torch.load(folder / "trainable.pt", map_location="cpu", weights_only=True)
    classifier.projector.load_state_dict(trainable["projector"])
    result = classifier.heads.load_state_dict(trainable["heads"], strict=False)
    new = [k for k in result.missing_keys if "spatial" not in k and "pooled" not in k]
    if new or result.unexpected_keys:
        raise ValueError(f"heads do not match the checkpoint: missing {new}, unexpected {result.unexpected_keys}")
    if result.missing_keys:
        report.append("new head layers for " + "/".join(sorted({k.split(".")[0].split("_")[1] for k in result.missing_keys}))
                      + " (start at zero)")
    if classifier.pool is not None:
        if "pool" in trainable:
            classifier.pool.load_state_dict(trainable["pool"])
            report.append("attention pooling")
        else:
            report.append("new attention pooling")
    return ", ".join(report + ["projector", "heads"])


FP16_SAFE = 2.0e4  # largest MLP activation let into an fp16 matmul (fp16 max is 65504)


def guard_fp16(decoder):
    """xIELU in fp32 and the MLP output as scale * down(act / scale), with scale a power of two chosen at every
    call from the largest activation: no fp16 overflow inside the MLP (Apertus activations exceed the fp16 range;
    same idea as apertus-8B-benchmark/lora/train.py, there with a fixed scale). down_proj is linear, so the result
    equals down(act) up to rounding. No .item(): the GPU is never synchronised."""
    import types

    def forward(self, x):
        activation = self.act_fn(self.up_proj(x).float())
        scale = torch.exp2(torch.ceil(torch.log2(activation.abs().amax().clamp(min=1.0) / FP16_SAFE))).clamp(min=1.0)
        out = self.down_proj((activation / scale).to(x.dtype))
        return (out.float() * scale).to(x.dtype)

    count = 0
    for module in decoder.modules():
        if type(module).__name__ == "Apertus1p5TextMLP":
            module.forward = types.MethodType(forward, module)
            count += 1
    if not count:
        raise RuntimeError("no Apertus1p5TextMLP found to guard")
    return count


def fp32_compute(decoder):
    """Safe mode for T4: weights stay float16 in memory, every linear layer of the decoder computes in float32
    (weights upcast on the fly) and autocast is off inside it. No fp16 overflow is possible; slower."""
    import types

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            bias = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), bias)

    count = 0
    for module in decoder.modules():
        if isinstance(module, nn.Linear):
            module.forward = types.MethodType(forward, module)
            count += 1
    return count


@torch.no_grad()
def first_non_finite(classifier, *inputs):
    """Names of the first modules whose output is not finite, with the largest |input| they got (diagnostics)."""
    found, hooks = [], []

    def hook(name):
        def check(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(out) and out.is_floating_point() and not torch.isfinite(out).all() and len(found) < 8:
                tensors = [a for a in args if torch.is_tensor(a) and a.is_floating_point()]
                peak = max((float(a.float().abs().max()) for a in tensors), default=float("nan"))
                found.append(f"{name} ({type(module).__name__}, input dtype {tensors[0].dtype if tensors else '-'}, "
                             f"max |input| {peak:.3g})")
        return check

    for name, module in classifier.named_modules():
        hooks.append(module.register_forward_hook(hook(name)))
    try:
        classifier(*inputs)
    finally:
        for h in hooks:
            h.remove()
    return found


def load(directory, folder, device="cuda", merge=True, dtype=torch.bfloat16):
    """A saved checkpoint for inference; LoRA merged into the weights by default (no extra cost per image).
    dtype float16 (T4): the MLPs are guarded against overflow (guard_fp16)."""
    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())
    # float16 means a 16 GB T4: the whole Apertus (about 18 GB) does not fit. Load it into CPU memory, cut it to
    # `layers`, merge LoRA there, and only then move the decoder (about 12.7 GB) to the GPU.
    staging = "cpu" if dtype == torch.float16 else device
    decoder, image_range = load_language_model(directory, settings["layers"], staging, dtype)
    if settings["lora_rank"]:
        from peft import PeftModel

        decoder = PeftModel.from_pretrained(decoder, str(folder / "lora"))
        if merge:
            decoder = decoder.merge_and_unload()
    if staging != device:
        decoder = decoder.to(device)
        import gc

        gc.collect()
    trainable = torch.load(folder / "trainable.pt", map_location="cpu", weights_only=True)
    classifier = assemble(decoder, image_range, settings)
    classifier.projector.load_state_dict(trainable["projector"])
    classifier.heads.load_state_dict(trainable["heads"])
    if classifier.pool is not None:
        classifier.pool.load_state_dict(trainable["pool"])
    if dtype == torch.float16:
        guard_fp16(decoder)
        classifier.residual_dtype = torch.float32  # the Apertus residual stream exceeds 65504 (seen on T4)
    return classifier.to(device).eval(), settings
