"""Apertus 8B with LoRA -> 6 classification heads, trained together.

    hidden    = Apertus layers 1..L (LoRA) on [template, VQ image tokens, context, task]
    features  = [last token, mean of the VQ image tokens]      2 x 4096
    logits    = heads(features)                                one linear head per dimension

One forward pass per image, no text generation. With L < 32 the layers after L are deleted.
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


class Classifier(nn.Module):
    """Wraps a decoder with forward(input_ids=, attention_mask=, position_ids=).last_hidden_state."""

    def __init__(self, language_model, heads, image_range):
        super().__init__()
        self.language_model, self.heads, self.image_range = language_model, heads, image_range

    def forward(self, input_ids, attention_mask):
        if not attention_mask[:, -1].all():
            raise ValueError("the last position is padding: left padding is required")
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        hidden = self.language_model(input_ids=input_ids, attention_mask=attention_mask,
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


def build(directory, settings, device="cuda"):
    """Classifier from settings (a dict, the same in training and in the saved checkpoint)."""
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    hidden = decoder.config.hidden_size
    if settings["lora_rank"]:
        decoder = add_lora(decoder, settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"])
    heads = Heads(2 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"])
    return Classifier(decoder, heads, image_range).to(device)


def save(classifier, settings, folder):
    """LoRA adapter (PEFT format), heads and settings; never the base weights."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if settings["lora_rank"]:
        classifier.language_model.save_pretrained(str(folder / "lora"))
        # The card PEFT writes holds a local base-model path, which the Hub rejects as invalid metadata: the
        # first LoRA prototype lost its adapter this way.
        (folder / "lora" / "README.md").unlink(missing_ok=True)
    torch.save({"heads": classifier.heads.state_dict()}, folder / "heads.pt")
    (folder / "settings.json").write_text(json.dumps(settings, indent=2))


def load(directory, folder, device="cuda", merge=True):
    """A saved checkpoint for inference; LoRA merged into the weights by default (no extra cost per image)."""
    folder = Path(folder)
    settings = json.loads((folder / "settings.json").read_text())
    decoder, image_range = load_language_model(directory, settings["layers"], device)
    hidden = decoder.config.hidden_size
    if settings["lora_rank"]:
        from peft import PeftModel

        decoder = PeftModel.from_pretrained(decoder, str(folder / "lora"))
        if merge:
            decoder = decoder.merge_and_unload()
    heads = Heads(2 * hidden, CLASS_COUNTS, settings["head_hidden"], settings["head_dropout"])
    heads.load_state_dict(torch.load(folder / "heads.pt", map_location="cpu", weights_only=True)["heads"])
    return Classifier(decoder, heads, image_range).to(device).eval(), settings
