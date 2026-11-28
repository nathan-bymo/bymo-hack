from types import SimpleNamespace

import pytest
import torch
from torch import nn

from adapter import ContinuousVisionAdapter, prepend_visual_tokens


class FakeVision(nn.Module):
    def __init__(self, patches=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.patches = patches

    def forward(self, pixel_values, **kwargs):
        batch = pixel_values.shape[0]
        values = torch.arange(self.patches * 3, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=values.reshape(1, self.patches, 3).repeat(batch, 1, 1) * self.weight)


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 5)
        self.embedding.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding


def test_projector_receives_gradient_with_frozen_encoder_and_language():
    adapter = ContinuousVisionAdapter(FakeVision(), 3, 5, pooled_side=2)
    model = FakeLanguage()
    adapter.train()
    tokens = adapter(torch.zeros(2, 3, 4, 4))
    result = prepend_visual_tokens(model, torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2), tokens)
    assert tokens.shape == (2, 4, 5)
    result["inputs_embeds"].sum().backward()
    assert adapter.projector[0].weight.grad is not None
    assert adapter.vision_encoder.weight.grad is None
    assert not adapter.vision_encoder.training
    assert model.embedding.weight.grad is None


def test_prefix_masks_and_labels_preserve_padding_and_original_tokens():
    model = FakeLanguage()
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 1, 0]])
    labels = torch.tensor([[-100, 2, -100]])
    result = prepend_visual_tokens(model, ids, mask, torch.ones(1, 2, 5), labels)
    assert result["attention_mask"].tolist() == [[1, 1, 1, 1, 0]]
    assert result["labels"].tolist() == [[-100, -100, -100, 2, -100]]
    assert torch.equal(result["inputs_embeds"][:, 2:], model.get_input_embeddings()(ids))


def test_non_square_encoder_output_fails_loudly():
    adapter = ContinuousVisionAdapter(FakeVision(patches=15), 3, 5)
    with pytest.raises(ValueError, match="square patch grid"):
        adapter(torch.zeros(1, 3, 4, 4))


def test_dinov3_cls_and_register_tokens_are_excluded():
    encoder = FakeVision(patches=21)
    adapter = ContinuousVisionAdapter(encoder, 3, 5, pooled_side=2, num_prefix_tokens=5)
    seen = []
    adapter.projector.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    assert adapter(torch.zeros(1, 3, 4, 4)).shape == (1, 4, 5)
    patches = torch.arange(21 * 3, dtype=torch.float32).reshape(1, 21, 3)[:, 5:]
    expected = torch.nn.functional.avg_pool2d(patches.transpose(1, 2).reshape(1, 3, 4, 4), 2).flatten(2).transpose(1, 2)
    torch.testing.assert_close(seen[0], expected)


def test_siglip_padded_patch_tokens_do_not_affect_pooling():
    adapter = ContinuousVisionAdapter(FakeVision(patches=20), 3, 5, pooled_side=2)
    seen = []
    adapter.projector.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    adapter(torch.zeros(1, 3, 4, 4), spatial_shapes=torch.tensor([[4, 4]]), pixel_attention_mask=torch.tensor([[1] * 16 + [0] * 4]))
    patches = torch.arange(20 * 3, dtype=torch.float32).reshape(1, 20, 3)[:, :16]
    expected = torch.nn.functional.avg_pool2d(patches.transpose(1, 2).reshape(1, 3, 4, 4), 2).flatten(2).transpose(1, 2)
    torch.testing.assert_close(seen[0], expected)


def test_real_transformers_vision_architectures_with_random_weights():
    pytest.importorskip("transformers")
    from transformers import DINOv3ViTConfig, DINOv3ViTModel, Siglip2VisionConfig, Siglip2VisionModel

    dino = DINOv3ViTModel(DINOv3ViTConfig(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_register_tokens=2))
    adapter = ContinuousVisionAdapter(dino, 32, 16, pooled_side=2, num_prefix_tokens=3)
    assert adapter(torch.randn(1, 3, 32, 32)).shape == (1, 4, 16)

    siglip = Siglip2VisionModel(Siglip2VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, patch_size=8, num_patches=16))
    adapter = ContinuousVisionAdapter(siglip, 32, 16, pooled_side=2)
    assert adapter(torch.randn(1, 16, 3 * 8 * 8), spatial_shapes=torch.tensor([[4, 4]]), pixel_attention_mask=torch.ones(1, 16, dtype=torch.long)).shape == (1, 4, 16)


def test_rejects_mismatched_label_shapes():
    with pytest.raises(ValueError, match="labels"):
        prepend_visual_tokens(FakeLanguage(), torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3), torch.zeros(1, 2, 5), torch.ones(1, 2))


def test_apertus_forward_keeps_native_image_placeholder(monkeypatch):
    pytest.importorskip("transformers")
    from transformers import Apertus1p5Config, Apertus1p5TextConfig
    from transformers.models.apertus1p5 import modeling_apertus1p5 as aperture

    config = Apertus1p5Config(
        text_config=Apertus1p5TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        ),
        vision_tokenizer_config={"codebook_size": 4},
        audio_tokenizer_config={"codebook_size": 4},
        image_token_id=10, audio_token_id=11, image_token_offset=32, audio_token_offset=36,
    )
    original = aperture.AutoModel.from_config
    monkeypatch.setattr(aperture, "Apertus1p5VisionTokenizerModel", lambda config: nn.Identity())
    monkeypatch.setattr(
        aperture.AutoModel, "from_config",
        lambda config, **kwargs: original(config, **kwargs) if config.model_type == "apertus1p5_text" else nn.Identity(),
    )
    model = aperture.Apertus1p5ForConditionalGeneration(config)
    model.requires_grad_(False)
    image_features = torch.full((1, 32), 0.25)
    monkeypatch.setattr(model.model, "get_image_features", lambda *args: SimpleNamespace(pooler_output=(image_features,)))
    adapter = ContinuousVisionAdapter(FakeVision(), 3, 32, pooled_side=2)
    tokens = adapter(torch.zeros(1, 3, 4, 4))
    ids = torch.tensor([[1, 10, 4, 5]])
    prepared = prepend_visual_tokens(model, ids, torch.ones_like(ids), tokens, torch.tensor([[-100, -100, -100, 5]]))
    result = model(**prepared, pixel_values=torch.zeros(1, 3, 4, 4), image_sizes=torch.tensor([[4, 4]]), use_cache=False)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert adapter.projector[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
    prepared.pop("labels")
    model.eval()
    with torch.no_grad():
        original = model(**prepared, pixel_values=torch.zeros(1, 3, 4, 4),
                         image_sizes=torch.tensor([[4, 4]]), use_cache=False).logits
        image_mask = torch.nn.functional.pad(ids.eq(config.image_token_id), (tokens.shape[1], 0)).unsqueeze(-1)
        cached = dict(prepared)
        cached["inputs_embeds"] = cached["inputs_embeds"].masked_scatter(image_mask, image_features)
        cached_logits = model(**cached, use_cache=False).logits
        torch.testing.assert_close(original, cached_logits)
        generated = model.generate(**prepared, pixel_values=torch.zeros(1, 3, 4, 4),
                                   image_sizes=torch.tensor([[4, 4]]), max_new_tokens=2,
                                   pad_token_id=0, eos_token_id=2)
    assert generated.ndim == 2 and generated.shape[0] == 1
