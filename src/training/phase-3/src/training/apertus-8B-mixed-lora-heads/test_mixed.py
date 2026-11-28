"""Verify zero-residual parity, staged gradients and checkpoint reload."""

import time
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from distillation import load_teacher, soft_losses
from mixed_model import D, E, M, MixedClassifier, load_mixed, save_mixed
import train_mixed
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


def tiny_vision_tower(rank=4):
    from peft import LoraConfig, get_peft_model
    from transformers import SiglipVisionConfig, SiglipVisionModel

    torch.manual_seed(7)
    config = SiglipVisionConfig(hidden_size=768, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                                image_size=28, patch_size=4)
    encoder = SiglipVisionModel(config).requires_grad_(False)
    lora = LoraConfig(r=rank, lora_alpha=rank, bias="none", target_modules=E.VISION_LORA_TARGETS)
    return M.VisionTower(get_peft_model(encoder, lora), 7, 1 / 255, [0.5] * 3, [0.5] * 3)


def test_siglip_lora_starts_unchanged_learns_and_survives_a_reload(tmp_path):
    torch.manual_seed(5)
    plain = MixedClassifier(architecture()).eval()
    ids, mask, _, dino = inputs()
    pixels = torch.randint(0, 256, (1, 3, 28, 28), dtype=torch.uint8)
    with pytest.raises(ValueError, match="trainable image encoder"):
        plain(ids, mask, pixels, dino)

    tower = tiny_vision_tower().eval()
    values = (pixels.float() * (1 / 255) - 0.5) / 0.5  # the arithmetic of VisionTower.forward
    with torch.no_grad():  # what the frozen encoder gives, rounded to float16 like the cached features
        frozen = E.pool_patches(tower.siglip.base_model.model(pixel_values=values).last_hidden_state, 7).half().float()
        with tower.siglip.disable_adapter():
            torch.testing.assert_close(tower(pixels), frozen, rtol=0., atol=0.)
        torch.testing.assert_close(tower(pixels), frozen, rtol=0., atol=0.)  # a new LoRA changes nothing

    plain.aperture.vision = tower
    with torch.no_grad():
        torch.testing.assert_close(plain(ids, mask, pixels, dino)[0], plain(ids, mask, frozen, dino)[0],
                                   rtol=0., atol=0.)
    groups, _ = configure(plain, {"layers": 3}, "vision", 1e-3)
    assert len(groups) == 4 and groups[-1]["lr"] == 1e-3
    trained = {n for n, p in tower.named_parameters() if p.requires_grad}
    assert trained and all("lora_" in n for n in trained)
    plain.train()
    optimizer = torch.optim.AdamW(groups)
    sum(part.square().mean() for part in plain(ids, mask, pixels, dino)).backward()
    assert sum(p.grad.abs().sum() for n, p in tower.named_parameters() if "lora_B" in n) > 0
    optimizer.step()
    plain.eval()
    with torch.no_grad():
        after = tower(pixels)
    assert not torch.equal(after, frozen)

    cfg = settings() | {"vision_lora_rank": 4, "vision_lora_alpha": 4}
    M.save(plain.aperture, cfg, tmp_path / "saved")
    assert (tmp_path / "saved" / "vision_lora" / "adapter_model.safetensors").is_file()
    rebuilt = architecture()
    rebuilt.vision = tiny_vision_tower().eval()
    assert "vision LoRA" in M.load_weights(rebuilt, tmp_path / "saved", cfg)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt.vision(pixels), after)

    # prediction: the LoRA merged into the frozen encoder reads the images as the trained model did
    encoder = E.ImageEncoder.__new__(E.ImageEncoder)
    encoder.siglip = tiny_vision_tower().siglip.unload()
    encoder.device = torch.device("cpu")
    with torch.no_grad():
        assert torch.allclose(encoder.siglip(pixel_values=values).last_hidden_state,
                              tiny_vision_tower().siglip(pixel_values=values).last_hidden_state)
    encoder.add_vision_lora(tmp_path / "saved" / "vision_lora")
    with torch.no_grad():
        merged = E.pool_patches(encoder.siglip(pixel_values=values).last_hidden_state, 7)
    torch.testing.assert_close(merged, after, atol=2e-3, rtol=2e-3)
    with pytest.raises(FileNotFoundError):
        encoder.add_vision_lora(tmp_path / "nothing")


def test_one_broken_image_never_stops_the_mixed_prediction(tmp_path):
    from train_mixed import guarded_batches

    class Encoder:
        def encode(self, records, variants, data_dir, ocr, workers):
            if any((data_dir / r["image_path"]).name == "broken.png" for r in records):
                raise OSError("cannot read")
            for record in records:
                ocr.setdefault(record["image_id"], "text")
            return [{"path": str(data_dir / r["image_path"])} for r in records]

    names = ["a.png", "broken.png", "c.png"]
    records = [{"image_id": n, "image_path": n} for n in names]
    replaced = []
    batches = list(guarded_batches(records, Encoder(), tmp_path, 2, replaced))
    assert [r["image_id"] for part, _, _ in batches for r in part] == names
    assert replaced == ["broken.png"] and batches[0][2]["broken.png"] == ""
    read = [entry["path"] for _, features, _ in batches for entry in features]
    assert len(read) == 3 and read[1].endswith("blank.png") and read[2].endswith("c.png")


def test_teacher_requires_matching_split_checksum_and_order(tmp_path):
    import hashlib

    dev = pd.DataFrame({"image_id": ["one", "two"], "product_id": ["p1", "p2"],
                        "image_path": ["dev_images/one.jpg", "dev_images/two.jpg"]})
    train = pd.DataFrame({"image_id": ["three"], "product_id": ["p3"]})
    artifact = tmp_path / "teacher.pt"
    probabilities = [torch.full((2, size), 1 / size) for size in D.CLASS_COUNTS]
    mask = torch.ones((2, len(D.DIMENSIONS)), dtype=torch.bool)
    torch.save({"image_ids": dev.image_id.tolist(), "dimensions": D.DIMENSIONS,
                "probabilities": probabilities, "mask": mask}, artifact)
    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _, actual = load_teacher(artifact, sha, dev, train)
    assert torch.equal(actual, mask)
    with pytest.raises(ValueError, match="checksum"):
        load_teacher(artifact, "invalid", dev, train)
    with pytest.raises(ValueError, match="IDs"):
        load_teacher(artifact, sha, dev.iloc[::-1], train)
    with pytest.raises(ValueError, match="outside train"):
        load_teacher(artifact, sha, dev, train.assign(product_id="p1"))


def test_soft_loss_masks_gradients_and_uses_full_distribution():
    logits = torch.tensor([[2., -2.], [-3., 3.], [1., -1.]], requires_grad=True)
    probabilities = [torch.tensor([[.2, .8], [.6, .4]])]
    mask = torch.tensor([[False], [True]])
    losses = soft_losses([logits], probabilities, mask, torch.tensor([2, 0]), torch.tensor([0, 1]))
    assert len(losses) == 1
    direct = -(probabilities[0][1] * logits[0].log_softmax(0)).sum()
    torch.testing.assert_close(losses[0], direct)
    losses[0].backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1:].count_nonzero() == 0


def test_control_and_student_complete_matched_optimizer_updates(monkeypatch):
    model = MixedClassifier(architecture())
    fit = pd.DataFrame({"image_id": ["a", "b", "c", "d", "e", "f"]})
    labels = torch.zeros((6, len(D.DIMENSIONS)), dtype=torch.long)
    labels[5:] = -1
    args = SimpleNamespace(real_count=5, train_batch=1, alphas=[None] * len(D.DIMENSIONS))
    sample = inputs()
    monkeypatch.setattr(train_mixed, "forward",
                        lambda model, builder, rows, features, ocr: model(
                            *(part.expand(len(rows), *part.shape[1:]) for part in sample)))
    updates = []
    soft_calls = []
    actual_adamw = torch.optim.AdamW
    actual_soft_losses = train_mixed.soft_losses

    class CountSteps(actual_adamw):
        def step(self, closure=None):
            updates.append(self)
            return super().step(closure)

    class Validate:
        best = {"macro_f1": .5, "validation_loss": .5}

        def __call__(self, *args):
            return self.best

    def count_soft_loss(*args):
        soft_calls.append(True)
        return actual_soft_losses(*args)

    monkeypatch.setattr(torch.optim, "AdamW", CountSteps)
    monkeypatch.setattr(train_mixed, "soft_losses", count_soft_loss)
    cache = {image_id: None for image_id in fit.image_id}
    probabilities = [torch.full((1, size), 1 / size) for size in D.CLASS_COUNTS]
    teacher = (probabilities, torch.ones((1, len(D.DIMENSIONS)), dtype=torch.bool), 5)
    for stage, frame, targets in (("control", fit.iloc[:5], None), ("distill", fit, teacher)):
        count_before = len(updates)
        result = train_mixed.train_stage(stage, .5, model, {"layers": 3}, None, frame, labels[:len(frame)],
                                         [(i, "") for i in range(len(frame))], cache, {}, args,
                                         time.monotonic() + 30, Validate(), teacher=targets, sample_count=5)
        assert result == "epoch_limit"
        assert len(updates) - count_before == 3
    assert len(soft_calls) == 1
