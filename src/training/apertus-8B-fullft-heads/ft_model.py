"""Apertus v1.5 8B, fully finetuned (no LoRA, no SigLIP), cut after L layers -> 6 classification heads.

    hidden   = Apertus layers 1..L on [template, VQ image tokens, context, task]     every weight trained
    features = [last token, mean of the VQ image tokens]                            2 x 4096
    logits   = heads(features)                                                      one linear head per dimension

Trained: all weights of the kept decoder layers and of the final norm, held in fp32 (master weights, the forward
runs under bf16 autocast), and the heads. Frozen: the token embeddings (1.1 B parameters, image tokens included)
and the VQ tokenizer. The lm_head and the layers after L are deleted.

Heads, focal loss and the loading of the truncated decoder come from apertus-8B-vadapter-lora-heads/model.py, so the
two models differ only in what is trained (all weights here, LoRA + projector there) and in the SigLIP soft tokens.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
VLH = HERE.parent / "apertus-8B-vadapter-lora-heads"
for path in (VLH, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import data as D  # noqa: E402
import model as VM  # noqa: E402  (apertus-8B-vadapter-lora-heads/model.py)

Heads, focal_loss, task_losses = VM.Heads, VM.focal_loss, VM.task_losses
FEATURES = 2  # last token, mean of the image tokens
TINY_TEXT = {"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 16, "num_hidden_layers": 4}


def pooled_features(hidden, input_ids, attention, image_range):
    """[batch, 2 * hidden]: state of the last prompt token and mean state of the VQ image tokens."""
    hidden = hidden.float()
    low, high = image_range
    image = ((input_ids >= low) & (input_ids < high) & (attention > 0)).to(hidden.dtype)[..., None]
    mean = (hidden * image).sum(1) / image.sum(1).clamp(min=1)
    return torch.cat([hidden[:, -1], mean], dim=-1)


class Classifier(nn.Module):
    """Wraps a decoder with .get_input_embeddings() and forward(inputs_embeds=, attention_mask=, position_ids=)."""

    def __init__(self, language_model, heads, image_range):
        super().__init__()
        self.language_model, self.heads, self.image_range = language_model, heads, image_range

    def forward(self, input_ids, attention_mask):
        if not attention_mask[:, -1].all():
            raise ValueError("the last position is padding: left padding is required")
        position_ids = (attention_mask.cumsum(1) - 1).clamp(min=0)
        embeddings = self.language_model.get_input_embeddings()(input_ids)
        hidden = self.language_model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                     position_ids=position_ids, use_cache=False).last_hidden_state
        return self.heads(pooled_features(hidden, input_ids, attention_mask, self.image_range))


def make_trainable(decoder):
    """Kept layers + final norm trainable, in fp32 (Adam updates of ~1e-5 vanish in bf16); embeddings frozen."""
    decoder.requires_grad_(False)
    for module in (*decoder.layers, decoder.norm):
        module.float().requires_grad_(True)
    return [p for p in decoder.parameters() if p.requires_grad]


def parameter_groups(classifier, lr_backbone, lr_heads, weight_decay, head_weight_decay=0.05):
    """AdamW groups: decoder matrices (weight decay), decoder vectors (norms, xIELU: none), heads."""
    matrices, vectors = [], []
    for parameter in classifier.language_model.parameters():
        if parameter.requires_grad:
            (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    return [{"params": matrices, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": vectors, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": list(classifier.heads.parameters()), "lr": lr_heads, "weight_decay": head_weight_decay}]


# ---------------------------------------------------------------- Apertus


def tiny_language_model(directory, layers):
    """A random miniature of the real Apertus decoder class (same tokenizer ids), to test the code on a CPU."""
    from transformers import AutoConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    config = AutoConfig.from_pretrained(directory)
    config.text_config.update(TINY_TEXT)
    torch.manual_seed(1234)  # the same "base weights" in training and in prediction
    decoder = Apertus1p5TextModel(config.text_config).float()
    decoder.layers = decoder.layers[:layers]
    decoder.config.num_hidden_layers = layers
    decoder.requires_grad_(False)
    low = config.image_token_offset
    return decoder, (low, low + config.vision_tokenizer_config.codebook_size)


def base(directory, settings, device):
    """(decoder with the base weights cut after settings["layers"], image id range)."""
    if settings.get("tiny"):
        decoder, image_range = tiny_language_model(directory, settings["layers"])
        return decoder.to(device), image_range
    return VM.load_language_model(directory, settings["layers"], device)


def build(directory, settings, device="cuda"):
    """Classifier with the base weights and new heads (bf16 decoder: make_trainable turns it into fp32)."""
    decoder, image_range = base(directory, settings, device)
    heads = Heads(FEATURES * decoder.config.hidden_size, D.CLASS_COUNTS, settings["head_hidden"],
                  settings["head_dropout"])
    return Classifier(decoder, heads, image_range).to(device)


def save(classifier, settings, folder, dtype=torch.bfloat16):
    """Kept decoder layers + norm (safetensors, bf16: ~10.5 GB for 24 layers), heads and settings.

    The frozen embeddings are not saved: they are those of the base model (settings hold its revision). Written to
    a temporary folder first, so a stopped job never leaves a half-written checkpoint.
    """
    from safetensors.torch import save_file

    folder = Path(folder)
    temporary = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    state = {name: value.detach().to("cpu", dtype).contiguous()
             for name, value in classifier.language_model.state_dict().items()
             if not name.startswith("embed_tokens.")}
    save_file(state, str(temporary / "decoder.safetensors"), metadata={"format": "pt"})
    torch.save(classifier.heads.state_dict(), temporary / "heads.pt")
    (temporary / "settings.json").write_text(json.dumps(settings, indent=2))
    shutil.rmtree(folder, ignore_errors=True)
    temporary.rename(folder)


def load_weights(classifier, folder):
    """Writes a saved checkpoint into a classifier built from the same base model and settings."""
    from safetensors.torch import load_file

    folder = Path(folder)
    device = next(classifier.parameters()).device
    missing, unexpected = classifier.language_model.load_state_dict(
        load_file(str(folder / "decoder.safetensors"), device=str(device)), strict=False)
    if unexpected or set(missing) - {"embed_tokens.weight"}:
        raise ValueError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    classifier.heads.load_state_dict(torch.load(folder / "heads.pt", map_location=device, weights_only=True))
    return classifier


def load(directory, folder, device="cuda"):
    """A saved checkpoint for inference (bf16 decoder, fp32 heads)."""
    settings = json.loads((Path(folder) / "settings.json").read_text())
    return load_weights(build(directory, settings, device), folder).eval(), settings
