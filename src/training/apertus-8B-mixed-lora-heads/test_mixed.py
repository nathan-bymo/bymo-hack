"""Verify zero-residual parity, staged gradients and checkpoint reload."""

from types import SimpleNamespace

import torch

from mixed_model import D, E, M, MixedClassifier, load_mixed, save_mixed
from train_mixed import configure


def architecture():
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    torch.manual_seed(199)
    config = Apertus1p5TextConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    decoder = M.add_lora(Apertus1p5TextModel(config).requires_grad_(False), 4, 4)
    return M.assemble(decoder, (100, 200), settings())


def settings():
    return {"model": E.MODEL_ID, "revision": E.MODEL_REVISION, "siglip": E.SIGLIP_ID,
            "side": 256, "vision_dim": 768, "layers": 24, "adapter": True, "ocr": True,
            "spatial": True, "attention_pool": True, "row_end_id": 9, "head_hidden": 16, "head_dropout": 0.,
            "lora_rank": 4, "lora_alpha": 4}


def inputs():
    ids = torch.tensor([[1, *([-1] * 49), 100, 101, 9, 102, 103, 4]])
    return ids, torch.ones_like(ids, dtype=torch.bool), torch.randn(1, 49, 768), torch.randn(1, 49, 768)


def test_zero_fusion_exactly_matches_source_and_only_fusion_learns():
    model = MixedClassifier(architecture()).eval()
    sample = inputs()
    with torch.no_grad():
        expected = model.aperture(*sample[:3])
        initial = model(*sample)
        for reference, actual in zip(expected, initial, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0., atol=0.)
    groups, masters = configure(model, {"layers": 3}, "fusion")
    assert not masters.master and len(groups) == 1
    initial = model(*sample)
    assert [part.shape[-1] for part in initial] == D.CLASS_COUNTS
    sum(t.square().mean() for t in initial).backward()
    assert model.fusion[-1].weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.aperture.parameters())


def test_stage_selection_and_mixed_checkpoint_roundtrip(tmp_path, monkeypatch):
    from transformers import AutoConfig, AutoTokenizer

    monkeypatch.setattr(M, "build", lambda *args: architecture())
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args: SimpleNamespace(image_token_offset=100))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args: SimpleNamespace(convert_tokens_to_ids=lambda _: 9))
    model = MixedClassifier(architecture())
    cfg = settings()
    M.save(model.aperture, cfg, tmp_path / "source")
    model, cfg, _ = load_mixed(None, tmp_path / "source", "test-revision", device="cpu")
    groups, masters = configure(model, {"layers": 3}, "joint")
    assert len(groups) == 3 and groups[2]["params"] and not masters.master
    groups, masters = configure(model, {"layers": 3}, "top2")
    assert len(groups) == 4 and groups[-1]["params"]
    assert all(p.dtype == torch.float32 for p in groups[-1]["params"])
    assert all("layers.0." not in name for name in masters.base)
    for name, parameter in model.aperture.language_model.named_parameters():
        if ("layers.1." in name or "layers.2." in name) and "lora_" in name:
            assert not parameter.requires_grad

    # Exercise save/load of all mixed state, including the FP32 base patch.
    sample = inputs()
    with torch.no_grad():
        model.fusion[-1].weight[0, 0] = .2
        next(iter(masters.master.values())).mul_(1.1)
        masters.sync()
    cfg.update(layers=3, unfrozen_blocks=2)
    save_mixed(model.eval(), cfg, tmp_path / "mixed", masters)
    # The compatibility check expects the production depth but the tensors in this test belong to a 3-layer decoder.
    # Load the mixed state and the base patch with the same production helpers on the miniature model.
    from base_weights import restore_blocks
    from mixed_model import FUSION_NAME

    rebuilt = MixedClassifier(architecture()).eval()
    M.load_weights(rebuilt.aperture, tmp_path / "mixed", cfg)
    restore_blocks(rebuilt.aperture.language_model, tmp_path / "mixed", cfg)
    rebuilt.fusion.load_state_dict(torch.load(tmp_path / "mixed" / FUSION_NAME, weights_only=True))
    with torch.inference_mode():
        for expected, actual in zip(model(*sample), rebuilt(*sample), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
