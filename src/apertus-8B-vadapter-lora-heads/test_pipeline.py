"""CPU tests of the plumbing, with a tiny stand-in decoder instead of Apertus (they prove nothing on quality).

  python -m pytest -q src/apertus-8B-vadapter-lora-heads/test_pipeline.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402
import encoders as E  # noqa: E402
import model as M  # noqa: E402

IMAGE_RANGE = (100, 200)


class TinyDecoder(nn.Module):
    """Causal, padding-aware: hidden_t = W * mean of the valid embeddings up to t."""

    def __init__(self, vocab=256, hidden=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache=False):
        mask = attention_mask[..., None].float()
        running = (inputs_embeds * mask).cumsum(1) / mask.cumsum(1).clamp(min=1)
        return SimpleNamespace(last_hidden_state=self.mix(running))


def classifier(prefix=4, vision_dim=6):
    torch.manual_seed(0)
    decoder = TinyDecoder()
    return M.Classifier(decoder, M.Projector(vision_dim, 8), M.Heads(24, D.CLASS_COUNTS, 16, 0.0), IMAGE_RANGE)


def sequence(prefix, image_tokens, text):
    return [1, 2] + [E.PREFIX_SLOT] * prefix + image_tokens + text


def test_focal_matches_cross_entropy_and_formula():
    logits, targets = torch.randn(7, 5), torch.randint(0, 5, (7,))
    assert torch.allclose(M.focal_loss(logits, targets, 0.0), F.cross_entropy(logits, targets))
    p = logits.softmax(-1)[torch.arange(7), targets]
    assert torch.allclose(M.focal_loss(logits, targets, 2.0), (-(1 - p) ** 2 * p.log()).mean(), atol=1e-6)
    alpha = torch.arange(1.0, 6.0)
    expected = (-(alpha[targets]) * (1 - p) ** 2 * p.log()).mean()
    assert torch.allclose(M.focal_loss(logits, targets, 2.0, alpha), expected, atol=1e-6)


def test_unscored_cells_do_not_change_the_loss():
    logits = [torch.randn(4, n) for n in D.CLASS_COUNTS]
    labels = torch.zeros(4, 6, dtype=torch.long)
    labels[2:, 0] = -1
    changed = [l.clone() for l in logits]
    changed[0][2:] += 100
    assert torch.allclose(M.task_losses(logits, labels, 2.0)[0], M.task_losses(changed, labels, 2.0)[0])
    labels[:, 1] = -1
    assert M.task_losses(logits, labels, 2.0)[1] is None


def test_soft_tokens_fill_their_slots_and_padding_does_not_matter():
    model = classifier().eval()
    short = sequence(4, [150, 151, 152], [3])
    long = sequence(4, [150, 151, 152, 153, 154], [3, 4, 5])
    siglip = torch.randn(2, 4, 6)
    ids, attention = E.collate([short, long], pad_id=0)
    together = model(ids, attention, siglip)
    for row, seq in enumerate((short, long)):
        alone_ids, alone_attention = E.collate([seq], pad_id=0)
        alone = model(alone_ids, alone_attention, siglip[row:row + 1])
        for a, b in zip(together, alone):
            assert torch.allclose(a[row], b[0], atol=1e-5)
    embedded = model.embed(ids, siglip)
    slots = ids == E.PREFIX_SLOT
    assert torch.allclose(embedded[slots].reshape(2, 4, 8), model.projector(siglip), atol=1e-6)


def test_gradients_reach_projector_and_heads_not_frozen_embeddings():
    model = classifier()
    model.language_model.embed_tokens.requires_grad_(False)
    ids, attention = E.collate([sequence(4, [150, 160], [3, 4])], pad_id=0)
    labels = torch.zeros(1, 6, dtype=torch.long)
    sum(l for l in M.task_losses(model(ids, attention, torch.randn(1, 4, 6)), labels, 2.0)).backward()
    assert model.projector[0].weight.grad.abs().sum() > 0
    assert model.heads.shared.weight.grad.abs().sum() > 0
    assert model.language_model.embed_tokens.weight.grad is None


def test_pooled_features_separate_image_and_soft_tokens():
    hidden = torch.arange(6.0)[None, :, None].repeat(1, 1, 2)
    ids = torch.tensor([[0, E.PREFIX_SLOT, E.PREFIX_SLOT, 150, 151, 7]])
    attention = torch.tensor([[0, 1, 1, 1, 1, 1]])
    features = M.pooled_features(hidden, ids, attention, IMAGE_RANGE)
    assert features.tolist() == [[5.0, 5.0, 3.5, 3.5, 1.5, 1.5]]


def test_left_padding_is_required():
    model = classifier()
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]]), torch.randn(1, 0, 6))


def rows_frame():
    base = {"image_path": "x", "product_title": "", "product_type": "", "brand": "", "category": "", "subcategory": ""}
    rows = []
    for n, (perspective, text) in enumerate([("left_side", "none"), ("front_head_on", "none"),
                                            ("right_side", "promotional_text"), ("left_side", "none")]):
        row = {**base, "image_id": f"i{n}", "product_id": f"p{n % 3}"}
        for d in D.DIMENSIONS:
            row |= {d: D.TAXONOMY[d][0], f"{d}_source": "human", f"{d}_relevant": "true"}
        row |= {"perspective": perspective, "text_classification": text,
                "text_language": "none" if text == "none" else "German"}
        rows.append(row)
    rows[3]["subject_source"] = "model"
    return pd.DataFrame(rows)


def test_labels_mirrors_and_items():
    rows = rows_frame()
    labels = D.label_matrix(rows)
    assert labels[3, 0] == -1 and (labels[:3, 0] >= 0).all()
    mirrored = D.mirror_labels(labels)
    names = D.TAXONOMY["perspective"]
    assert names[mirrored[0, D.PERSPECTIVE]] == "right_side"
    assert names[mirrored[1, D.PERSPECTIVE]] == "front_head_on"
    items = D.training_items(rows, labels, repeats=False)
    assert (0, "m") in items and (3, "m") in items and (2, "m") not in items and (1, "m") not in items
    assert len([v for _, v in items if v == ""]) == len(rows)


def test_repeats_are_distinct_variants_and_mirror_only_without_text():
    rows = rows_frame()
    labels = D.label_matrix(rows)
    D_counts = D.repeat_counts
    try:
        D.repeat_counts = lambda labels: [4] * len(labels)
        items = D.training_items(rows, labels)
    finally:
        D.repeat_counts = D_counts
    variants = {index: [v for i, v in items if i == index] for index in range(len(rows))}
    assert variants[1] == ["", "a1m", "a2", "a3m"]          # no text: odd repeats mirrored
    assert variants[2] == ["", "a1", "a2", "a3"]            # text: never mirrored
    assert variants[0] == ["", "a1m", "a2", "a3m", "m"]     # left/right without text: one more mirror
    item = D.item_labels(labels, items)
    names = D.TAXONOMY["perspective"]
    assert [names[l[D.PERSPECTIVE]] for (i, v), l in zip(items, item) if i == 0] == \
        ["left_side", "right_side", "left_side", "right_side", "right_side"]


def test_variant_images(tmp_path):
    from PIL import Image, ImageOps
    from common import images

    path = tmp_path / "x.png"
    picture = Image.new("RGBA", (300, 200), (255, 255, 255, 0))
    picture.paste((200, 30, 30, 255), (10, 20, 120, 180))
    picture.save(path)
    vq, full = E.variant_images(path, "", 256, "img")
    assert vq.tobytes() == images.prepared_image(path, 256).tobytes()
    assert full.size == (300, 200) and full.getpixel((0, 0)) == (255, 255, 255)
    _, mirrored = E.variant_images(path, "m", 256, "img")
    assert mirrored.tobytes() == ImageOps.mirror(full).tobytes()
    first, crop = E.variant_images(path, "a1", 256, "img")
    again, crop_again = E.variant_images(path, "a1", 256, "img")
    assert crop.tobytes() == crop_again.tobytes() and first.tobytes() == again.tobytes()
    assert 270 <= crop.width <= 300 and 180 <= crop.height <= 200
    assert all(side % images.PATCH == 0 for side in first.size)
    _, other = E.variant_images(path, "a2", 256, "img")
    assert (other.size, other.tobytes()) != (crop.size, crop.tobytes())


def test_internal_split_is_by_product_and_alpha_has_mean_one():
    rows = pd.DataFrame({"product_id": [f"p{n // 2}" for n in range(400)], "image_id": [str(n) for n in range(400)]})
    fit, validation = D.internal_split(rows, 0.2)
    assert not set(fit.product_id) & set(validation.product_id) and 0 < len(validation) < len(rows)
    assert len(D.internal_split(rows, 0.0)[1]) == 0
    labels = torch.tensor([[0] * 6] * 30 + [[1] * 6] * 3)
    for alpha in D.class_alpha(labels):
        per_example = alpha[labels[:, 0]].mean()  # the loss keeps its scale
        assert torch.isclose(per_example, torch.tensor(1.0)) and alpha[1] > alpha[0]
    one_example = torch.tensor([[0] * 6] * 500 + [[1] * 6])
    alpha = D.class_alpha(one_example)[0]
    assert torch.isclose(alpha[1] / alpha[0], torch.tensor(10.0)) and alpha[0] > 0.9  # capped, frequent class kept


def test_real_apertus_decoder_in_miniature():
    """The decoder class of the Apertus fork (random weights, 4 layers cut to 2): padding and truncation."""
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig

    torch.manual_seed(0)
    config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    decoder = text.Apertus1p5TextModel(config).float().eval()
    decoder.layers = decoder.layers[:2]
    decoder.config.num_hidden_layers = 2
    model = M.Classifier(decoder, M.Projector(6, 32), M.Heads(96, D.CLASS_COUNTS, 16, 0.0), IMAGE_RANGE).eval()
    short, long = sequence(4, [150, 151], [3]), sequence(4, [150, 151, 152, 153], [3, 4, 5, 6])
    siglip = torch.randn(2, 4, 6)
    ids, attention = E.collate([short, long], pad_id=0)
    together = model(ids, attention, siglip)
    for row, seq in enumerate((short, long)):
        alone = model(*E.collate([seq], pad_id=0), siglip[row:row + 1])
        for a, b in zip(together, alone):
            assert torch.allclose(a[row], b[0], atol=1e-4)
    assert len(decoder.layers) == 2


def test_without_adapter_the_soft_token_mean_is_zero():
    model = classifier().eval()
    ids, attention = E.collate([sequence(0, [150, 151], [3])], pad_id=0)
    features = M.pooled_features(model.language_model(inputs_embeds=model.embed(ids, torch.randn(1, 0, 6)),
                                                      attention_mask=attention, position_ids=None).last_hidden_state,
                                 ids, attention, IMAGE_RANGE)
    assert torch.all(features[:, 16:] == 0)
    model(ids, attention, torch.randn(1, 0, 6))
