"""Continuous image prefix for Apertus 1.5; training orchestration is separate."""

import torch
from torch import nn
from torch.nn import functional as F


class ContinuousVisionAdapter(nn.Module):
    """Frozen patch encoder and trainable 2-layer projector.

    Accepts square patch grids. DINOv3 includes a CLS token and register tokens;
    pass 1 + encoder.config.num_register_tokens as num_prefix_tokens for it.
    """

    def __init__(self, vision_encoder, vision_dim, text_dim, pooled_side=7, num_prefix_tokens=0):
        super().__init__()
        if pooled_side < 1:
            raise ValueError("pooled_side must be positive")
        if num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens cannot be negative")
        self.vision_encoder = vision_encoder
        self.vision_encoder.requires_grad_(False)
        self.vision_encoder.eval()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim)
        )
        self.pooled_side = pooled_side
        self.num_prefix_tokens = num_prefix_tokens

    def train(self, mode=True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def forward(self, pixel_values, **vision_inputs):
        with torch.no_grad():
            features = self.vision_encoder(pixel_values=pixel_values, **vision_inputs).last_hidden_state
        if features.ndim != 3:
            raise ValueError("vision encoder must return [batch, patches, channels]")
        if features.shape[1] <= self.num_prefix_tokens:
            raise ValueError("encoder output contains no patch tokens")
        features = features[:, self.num_prefix_tokens:, :]
        batch, count, channels = features.shape
        spatial_shapes = vision_inputs.get("spatial_shapes")
        patch_mask = vision_inputs.get("pixel_attention_mask")
        if spatial_shapes is None:
            side = int(count**0.5)
            if side * side != count:
                raise ValueError("expected a square patch grid without a CLS token")
            shapes = [(side, side)] * batch
        else:
            if spatial_shapes.shape != (batch, 2):
                raise ValueError("spatial_shapes must contain height and width for each image")
            shapes = spatial_shapes.tolist()
        if patch_mask is not None and patch_mask.shape != (batch, count):
            raise ValueError("pixel_attention_mask must match the patch sequence")
        pooled_rows = []
        for index, (height, width) in enumerate(shapes):
            valid = int(height) * int(width)
            if min(height, width) < self.pooled_side or valid > count:
                raise ValueError("invalid patch grid or pooled_side too large")
            if patch_mask is not None:
                mask = patch_mask[index].bool()
                if not mask[:valid].all() or mask[valid:].any():
                    raise ValueError("expected valid patches followed by padding")
            grid = features[index, :valid].transpose(0, 1).reshape(1, channels, height, width)
            pooled_rows.append(F.adaptive_avg_pool2d(grid, (self.pooled_side, self.pooled_side)))
        pooled = torch.cat(pooled_rows, dim=0)
        tokens = pooled.flatten(2).transpose(1, 2)
        return self.projector(tokens.to(dtype=self.projector[0].weight.dtype))


def prepend_visual_tokens(model, input_ids, attention_mask, visual_tokens, labels=None):
    """Prepare input embeddings, attention mask and optional causal-LM labels.

    Native image placeholders remain unchanged; Apertus can replace them with
    its VQ image embeddings when its pixel_values and image_sizes are supplied.
    The caller must mask prompt, padding, non-human and irrelevant label tokens.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have equal [batch, sequence] shapes")
    if visual_tokens.ndim != 3 or visual_tokens.shape[0] != input_ids.shape[0]:
        raise ValueError("visual_tokens must have shape [batch, prefix, hidden]")
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")
    embeddings = model.get_input_embeddings()(input_ids)
    if visual_tokens.shape[-1] != embeddings.shape[-1]:
        raise ValueError("visual token width must equal language embedding width")
    prefix = visual_tokens.to(device=embeddings.device, dtype=embeddings.dtype)
    prefix_mask = attention_mask.new_ones((input_ids.shape[0], prefix.shape[1]))
    result = {
        "inputs_embeds": torch.cat((prefix, embeddings), dim=1),
        "attention_mask": torch.cat((prefix_mask, attention_mask), dim=1),
    }
    if labels is not None:
        ignored = labels.new_full((input_ids.shape[0], prefix.shape[1]), -100)
        result["labels"] = torch.cat((ignored, labels), dim=1)
    return result
