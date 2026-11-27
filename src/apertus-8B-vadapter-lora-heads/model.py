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
    """LayerNorm (the features move while LoRA trains), shared layer, one linear head per dimension."""

    def __init__(self, input_size, class_counts, hidden_size=512, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.shared = nn.Linear(input_size, hidden_size)
        self.body = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList(nn.Linear(hidden_size, count) for count in class_counts)

    def forward(self, features):
        hidden = self.body(self.shared(self.norm(features)))
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


class Classifier(nn.Module):
    """Wraps any decoder with .embed_tokens and forward(inputs_embeds=, attention_mask=, position_ids=)."""

    def __init__(self, language_model, projector, heads, image_range):
        super().__init__()
        self.language_model, self.projector, self.heads = language_model, projector, heads
        self.image_range = image_range

    def embed(self, input_ids, siglip):
        embeddings = self.language_model.get_input_embeddings()(input_ids.clamp(min=0))
        slots = input_ids == PREFIX_SLOT
        if slots.any():
            soft = self.projector(siglip.to(self.projector[0].weight.dtype))
            if int(slots.sum()) != soft.shape[0] * soft.shape[1]:
                raise ValueError("every row needs exactly one block of soft-token slots")
            embeddings = embeddings.masked_scatter(slots[..., None], soft.to(embeddings.dtype))
        return embeddings

    def forward(self, input_ids, attention_mask, siglip):
        if not attention_mask[:, -1].all():
            raise ValueError("the last position is padding: left padding is required")
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        hidden = self.language_model(inputs_embeds=self.embed(input_ids, siglip), attention_mask=attention_mask,
                                     position_ids=position_ids, use_cache=False).last_hidden_state
        return self.heads(pooled_features(hidden, input_ids, attention_mask, self.image_range))


# ---------------------------------------------------------------- Apertus


def load_language_model(directory, layers, device="cuda"):
    """The text decoder of Apertus v1.5 8B in bf16, cut after `layers` layers; returns (decoder, image id range)."""
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

    full = Apertus1p5ForConditionalGeneration.from_pretrained(
        directory, dtype=torch.bfloat16, device_map={"": device}, attn_implementation=ATTENTION)
    config = full.config
    decoder = full.model.language_model
    total = decoder.config.num_hidden_layers
    if not 1 <= layers <= total:
        raise ValueError(f"layers must be in 1..{total}")
    decoder.layers = decoder.layers[:layers]
    decoder.config.num_hidden_layers = layers
    del full  # the VQ/audio tokenizers and the 1 GB lm_head are not needed
    torch.cuda.empty_cache()
    decoder.requires_grad_(False)
    low = config.image_token_offset
    return decoder, (low, low + config.vision_tokenizer_config.codebook_size)


def add_lora(decoder, rank, alpha, dropout=0.0):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=LORA_TARGETS)
    return get_peft_model(decoder, config)


def build(directory, settings, device="cuda"):
    """Classifier from settings (a dict, the same in training and in the saved checkpoint)."""
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    hidden = decoder.config.hidden_size
    if settings["lora_rank"]:
        decoder = add_lora(decoder, settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"])
    from data import CLASS_COUNTS

    projector = Projector(settings["vision_dim"], hidden)
    heads = Heads(3 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"])
    return Classifier(decoder, projector, heads, image_range).to(device)


def save(classifier, settings, folder):
    """LoRA adapter (PEFT format), projector + heads and settings; never the 16 GB base weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if settings["lora_rank"]:
        classifier.language_model.save_pretrained(str(folder / "lora"))
    torch.save({"projector": classifier.projector.state_dict(), "heads": classifier.heads.state_dict()},
               folder / "trainable.pt")
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
    from data import CLASS_COUNTS

    hidden = decoder.config.hidden_size
    trainable = torch.load(folder / "trainable.pt", map_location="cpu", weights_only=True)
    projector = Projector(settings["vision_dim"], hidden)
    projector.load_state_dict(trainable["projector"])
    heads = Heads(3 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"])
    heads.load_state_dict(trainable["heads"])
    return Classifier(decoder, projector, heads, image_range).to(device).eval(), settings
