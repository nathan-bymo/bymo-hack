"""Check pretrained vision encoders and their adapter shapes on a synthetic image."""

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

from adapter import ContinuousVisionAdapter

ENCODERS = {
    "siglip2": "google/siglip2-base-patch16-224",
    "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder", choices=ENCODERS)
    args = parser.parse_args()
    model_id = ENCODERS[args.encoder]
    processor = AutoImageProcessor.from_pretrained(model_id)
    if args.encoder == "siglip2":
        # The fixed-resolution SigLIP 2 checkpoint uses the SigLIP architecture;
        # Siglip2VisionModel is for the NaFlex-style patch interface.
        encoder = SiglipVisionModel.from_pretrained(model_id)
        prefix_tokens = 0
    else:
        encoder = AutoModel.from_pretrained(model_id)
        prefix_tokens = 1 + encoder.config.num_register_tokens
    # A synthetic image avoids uploading or saving any competition data.
    image = Image.new("RGB", (224, 224), "white")
    vision_inputs = processor(images=image, return_tensors="pt")
    adapter = ContinuousVisionAdapter(encoder, encoder.config.hidden_size, 64, pooled_side=7, num_prefix_tokens=prefix_tokens)
    tokens = adapter(**vision_inputs)
    assert tokens.shape == (1, 49, 64)
    tokens.square().mean().backward()
    assert adapter.projector[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())
    print(f"{args.encoder}: pretrained encoder forward and projector backward passed; shape={tuple(tokens.shape)}")


if __name__ == "__main__":
    main()
