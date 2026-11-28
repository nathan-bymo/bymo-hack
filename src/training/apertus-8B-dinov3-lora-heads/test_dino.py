"""Regression checks for the DINO patch grid, warm start and checkpoint selection."""

import pytest
import torch
from PIL import Image, ImageDraw

from base_weights import BaseMasters, block_parameters, restore_blocks
from experiment import EarlyStop
from pipeline import D, E, M, DinoPrompt, fit_image, make_model, model_inputs, pool_dino


def test_patch_pool_removes_class_and_register_tokens():
    for side in (288, 320):
        hidden = torch.cat((torch.zeros(2, 5, 8), torch.ones(2, (side // 16)**2, 8)), dim=1)
        pooled = pool_dino(hidden, side, 4)
        assert pooled.shape == (2, 81, 8)
        assert torch.equal(pooled, torch.ones_like(pooled))
        larger = pool_dino(hidden, side, 4, grid_side=12)
        assert larger.shape == (2, 144, 8) and torch.equal(larger, torch.ones_like(larger))
    with pytest.raises(ValueError, match="patch grid"):
        pool_dino(hidden, 288, 4)


def test_scale_variant_preserves_entire_wide_image(tmp_path):
    path = tmp_path / "wide.png"
    image = Image.new("RGB", (100, 20), "red")
    ImageDraw.Draw(image).rectangle((90, 0, 99, 19), fill="blue")
    image.save(path)
    for side in (288, 320):
        fitted = fit_image(path, side)
        assert fitted.size == (side, side)
        assert fitted.getpixel((1, side // 2)) == (255, 0, 0)
        assert fitted.getpixel((side - 2, side // 2)) == (0, 0, 255)
        assert fitted.getpixel((side // 2, 1)) == (255, 255, 255)


class Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 5

    def apply_chat_template(self, messages, **kwargs):
        return "x<|image|>" + messages[0]["content"][1]["text"]

    def __call__(self, value, **kwargs):
        return {"input_ids": [ord(c) for c in value]}


def test_no_native_vq_and_real_decoder_warm_start_roundtrip(tmp_path, monkeypatch):
    from transformers import AutoTokenizer
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    def build(directory, settings, device):
        config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=2, head_dim=8)
        torch.manual_seed(311)
        decoder = Apertus1p5TextModel(config).requires_grad_(False)
        decoder = M.add_lora(decoder, 4, 4)
        return M.assemble(decoder, (200, 255), settings)

    monkeypatch.setattr(M, "build", build)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    original = {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "layers": 24, "vision_dim": 768,
                "adapter": True, "ocr": True, "spatial": True, "attention_pool": True,
                "lora_rank": 4, "lora_alpha": 4, "head_hidden": 16, "head_dropout": 0.}
    source = build(None, original, "cpu")
    with torch.no_grad():
        source.projector[0].weight.fill_(0.5)
        source.heads.heads[0].weight.fill_(0.3)
        for name, parameter in source.language_model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    M.save(source, original, tmp_path / "source")
    model, settings, builder = make_model(None, tmp_path / "source", device="cpu", warm_start=True)
    assert isinstance(builder, DinoPrompt)
    assert settings["vq"] is False and model.image_range == (0, 0) and model.soft_side == 9
    assert torch.equal(source.heads.heads[0].weight, model.heads.heads[0].weight)
    assert not torch.equal(source.projector[0].weight, model.projector[0].weight)
    for (name, src), loaded in zip(source.language_model.named_parameters(),
                                   model.language_model.parameters(), strict=True):
        if "lora_" in name:
            assert torch.equal(src, loaded)
    records, features, ocr = [{"image_id": "one"}], torch.randn(1, 81, 768), {"one": "50 cm"}
    ids, mask, features = model_inputs(builder, records, features, ocr, "cpu")
    assert ids.eq(E.PREFIX_SLOT).sum().item() == 81
    assert all(grids[1][0].numel() == 0 for grids in M.image_grids(ids, model.image_range, None, 9))
    logits = model(ids, mask, features)
    assert [x.shape[-1] for x in logits] == D.CLASS_COUNTS
    loss = sum(x.square().mean() for x in logits)
    loss.backward()
    assert model.projector[0].weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.language_model.named_parameters() if "lora_" in name)

    settings.update(dino_revision="test", soft_side=12)
    model.soft_side = 12
    builder.prefix_tokens = 144
    ids, mask, features = model_inputs(builder, records, torch.randn(1, 144, 768), ocr, "cpu")
    M.save(model.eval(), settings, tmp_path / "dino")
    restored, _, _ = make_model(None, tmp_path / "dino", device="cpu")
    assert restored.soft_side == 12 and ids.eq(E.PREFIX_SLOT).sum() == 144
    for before, after in zip(model(ids, mask, features), restored.eval()(ids, mask, features), strict=True):
        torch.testing.assert_close(before, after)


def test_progressive_blocks_take_fp32_steps_and_reload(tmp_path):
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    config = Apertus1p5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    torch.manual_seed(6)
    decoder = M.add_lora(Apertus1p5TextModel(config).bfloat16().requires_grad_(False), 4, 4)
    weights = BaseMasters(decoder, 6)
    for blocks in (2, 4, 6):
        newly_trainable = weights.expand(blocks)
        assert newly_trainable and all(p.dtype == torch.float32 for p in newly_trainable)
        assert all(not p.requires_grad for p in block_parameters(decoder, 6, blocks, adapters=True).values())
        assert set(weights.base) == set(block_parameters(decoder, 6, blocks))
    name = next(iter(weights.base))
    base, master = weights.base[name], weights.master[name]
    original = master.detach().clone()
    optimizer = torch.optim.SGD([master], lr=2e-6)
    for _ in range(40):
        base.grad = torch.ones_like(base)
        weights.transfer_gradients()
        assert base.grad is None and master.grad.dtype == torch.float32
        optimizer.step()
        weights.sync()
    assert not torch.equal(master, original)
    assert torch.equal(base, master.bfloat16())
    weights.save(tmp_path)
    with torch.no_grad():
        base.zero_()
    restore_blocks(decoder, tmp_path, {"layers": 6, "unfrozen_blocks": 6})
    assert torch.equal(base, master.bfloat16())


def test_early_stop_uses_multiple_observations():
    stop = EarlyStop()
    best = {"macro_f1": .82, "validation_loss": .6}
    assert stop.update(.82, .6, 1., None) is None
    assert stop.update(.81, .65, .9, best) is None
    assert stop.update(.80, .7, .8, best) == "validation_regression"
    stop = EarlyStop()
    assert stop.update(.82, .6, 1., None) is None
    for _ in range(2):
        assert stop.update(.821, .6, .9, best) is None
    assert stop.update(.821, .6, .9, best) == "validation_plateau"
