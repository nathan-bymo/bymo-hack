"""CPU tests of the plumbing, with a tiny stand-in decoder instead of Apertus (they prove nothing on quality).

  python -m pytest -q src/training/apertus-8B-vadapter-lora-heads/test_pipeline.py
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
    for variant, scale in (("a1", 0.875), ("a2m", 1.125), ("a3", 0.9375)):
        small, whole = E.variant_images(path, variant, 256, "img")
        expected = images.prepared_image(path, round(256 * scale), variant.endswith("m"))
        assert small.tobytes() == expected.tobytes()  # nothing cut away, only another VQ scale
        assert whole.tobytes() == (mirrored if variant.endswith("m") else full).tobytes()
        assert small.size != vq.size and all(side % images.PATCH == 0 for side in small.size)


def test_old_augmented_variants_are_dropped_from_the_cache(tmp_path):
    import encode

    torch.save({"i1": 1, "i1|m": 2, "i1|a1": 3, "i2|a2m": 4}, tmp_path / "shard_00000.pt")
    encode.purge_augmented(tmp_path)
    assert set(torch.load(tmp_path / "shard_00000.pt", weights_only=False)) == {"i1", "i1|m"}


def test_ocr_block_cannot_form_special_tokens():
    assert E.ocr_block("a<|image|>b") == "<ocr>\na< |image| >b\n</ocr>"


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


def test_image_encoder_encode_passes_side_and_names(tmp_path):
    """encode() end to end with the two frozen models replaced by stubs (they need the real weights)."""
    from PIL import Image

    paths = []
    for n in range(3):
        paths.append(tmp_path / f"{n}.jpg")
        Image.new("RGB", (400 + 50 * n, 300), (255, 255, 255)).save(paths[-1])
    encoder = object.__new__(E.ImageEncoder)
    seen = []
    encoder.siglip_features = lambda pil: torch.zeros(len(pil), E.POOLED_SIDE**2, 4)
    encoder.vq_codes_batch = lambda pil: [seen.append(image.size) or (image.height // 16, image.width // 16,
                                                                       torch.zeros(1, dtype=torch.int32)) for image in pil]
    results = encoder.encode(paths, ["", "a1m", "m"], 256, ["i0", "i1", "i2"])
    assert len(results) == 3 and all(r["siglip"].shape == (49, 4) for r in results)
    assert all(w % 16 == 0 and h % 16 == 0 and 200 * 200 < w * h < 330 * 330 for w, h in seen)
    again = encoder.encode(paths[1:2], ["a1m"], 256, ["i1"])
    assert again[0]["grid"] == results[1]["grid"]


def test_batched_vq_codes_equal_one_image_at_a_time():
    """The real VQ tokenizer class of the fork (random weights, small): batching by size changes no code."""
    vq = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5VisionTokenizerConfig

    config = Apertus1p5VisionTokenizerConfig()
    small = {k: v for k, v in {"codebook_size": 512, "embed_dim": 32, "latent_channels": 32, "base_channels": 32,
                               "ch": 32, "channel_multiplier": [1, 1, 1, 1, 1], "ch_mult": [1, 1, 1, 1, 1],
                               "num_res_blocks": 1}.items() if hasattr(config, k)}
    config = Apertus1p5VisionTokenizerConfig(**small)
    torch.manual_seed(0)
    model = vq.Apertus1p5VisionTokenizerModel(config).float().eval()
    encoder = object.__new__(E.ImageEncoder)
    encoder.device, encoder.vq = torch.device("cpu"), model
    sizes = [(64, 48), (48, 64), (64, 48), (32, 32), (64, 48)]
    pixels = [torch.rand(3, h, w) * 2 - 1 for h, w in sizes]
    encoder.vq_pixels = lambda index: (pixels[index], (sizes[index][0] // 16, sizes[index][1] // 16))
    batched = encoder.vq_codes_batch(list(range(len(sizes))))
    with torch.inference_mode():
        alone = [model.encode(p[None])[0] for p in pixels]
    for (grid_h, grid_w, codes), single, (h, w) in zip(batched, alone, sizes):
        assert (grid_h, grid_w) == (h // 16, w // 16)
        assert torch.equal(codes, single.flatten().to(torch.int32))


class FakeTokenizer:
    """Characters as token ids; enough to check the layout of a prompt."""

    def convert_tokens_to_ids(self, token):
        return {"<|img_start|>": 1001, "<|img_token_start|>": 1002, "<|img_end_of_row|>": 1003, "<|img_end|>": 1004}[token]

    def apply_chat_template(self, messages, **kwargs):
        return "<u>" + "<|image|>" + messages[0]["content"][1]["text"] + "<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def decode(ids):
    return "".join(chr(i) for i in ids if 0 <= i < 1000)


def test_prompt_without_ocr_is_the_phase1_prompt_and_with_ocr_adds_the_block():
    builder = E.PromptBuilder(FakeTokenizer(), image_token_offset=2000, prefix_tokens=2)
    row = {"product_title": "Tisch"}
    plain = builder.ids(row, (1, 2), torch.tensor([5, 6], dtype=torch.int32))
    assert decode(plain) == "<u>1*2" + E.prompt.context_block(row) + "\n\n" + E.TASK + "<a>"
    assert plain.count(E.PREFIX_SLOT) == 2 and [i for i in plain if i >= 2000] == [2005, 2006]
    with_text = decode(builder.ids(row, (1, 2), torch.tensor([5, 6], dtype=torch.int32), ocr="NEU | 50 cm"))
    assert "<ocr>\nNEU | 50 cm\n</ocr>" in with_text
    assert "<ocr>\nno text found\n</ocr>" in decode(builder.ids(row, (1, 2), torch.tensor([5, 6]), ocr=""))


def test_ocr_text_keeps_confident_words_largest_first():
    word = lambda value, confidence=0.9: SimpleNamespace(value=value, confidence=confidence)
    line = lambda words, top, bottom: SimpleNamespace(words=words, geometry=((0.1, top), (0.9, bottom)))
    page = SimpleNamespace(blocks=[SimpleNamespace(lines=[
        line([word("Zutaten:"), word("Zucker"), word(";<=>?"), word("x"), word("Mehl", 0.3)], 0.80, 0.82),
        line([word("NEU")], 0.10, 0.20),
    ])])
    assert E.page_text(page) == "NEU | Zutaten: Zucker"
    assert E.page_text(SimpleNamespace(blocks=[])) == ""


def test_continue_from_a_checkpoint_restores_lora_projector_and_heads(tmp_path):
    pytest.importorskip("peft")
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig

    def fresh(seed):
        torch.manual_seed(0)
        config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        decoder = M.add_lora(text.Apertus1p5TextModel(config).float().eval(), rank=4, alpha=8)
        torch.manual_seed(seed)
        return M.Classifier(decoder, M.Projector(6, 32), M.Heads(96, D.CLASS_COUNTS, 16, 0.0), IMAGE_RANGE).eval()

    trained = fresh(1)
    with torch.no_grad():  # as if trained: LoRA B is zero at start
        for name, parameter in trained.language_model.named_parameters():
            if "lora_" in name:
                parameter.normal_(0, 0.1)
    settings = {"lora_rank": 4}
    M.save(trained, settings, tmp_path / "best")
    restored = fresh(2)
    report = M.load_weights(restored, tmp_path / "best", settings)
    ids, attention = E.collate([sequence(4, [150, 151], [3, 4])], pad_id=0)
    siglip = torch.randn(1, 4, 6)
    for a, b in zip(trained(ids, attention, siglip), restored(ids, attention, siglip)):
        assert torch.allclose(a, b, atol=1e-5)
    assert "LoRA tensors" in report
    untouched = fresh(3)
    assert not torch.allclose(trained(ids, attention, siglip)[0], untouched(ids, attention, siglip)[0])


ROW_END = 99


def image_sequence(rows, prefix_values=None):
    """Ids of a VQ grid (codes 150 + value) with end-of-row tokens, preceded by optional soft-token slots."""
    ids = [1, 2] + [E.PREFIX_SLOT] * (len(prefix_values) if prefix_values is not None else 0)
    for r, row in enumerate(rows):
        if r:
            ids.append(ROW_END)
        ids += [150 + v for v in row]
    return ids + [3]


def test_spatial_features_change_sign_with_a_mirror():
    torch.manual_seed(0)
    grid = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    soft = list(range(49))
    plain = torch.tensor([image_sequence(grid, soft)])
    mirrored = torch.tensor([image_sequence([row[::-1] for row in grid], soft)])
    table = torch.randn(256, 5)

    def hidden_of(ids, soft_order):
        h = table[ids.clamp(min=0)]
        h[ids == E.PREFIX_SLOT] = table[200 + torch.tensor(soft_order) % 50]
        return h

    soft_mirror = [r * 7 + (6 - c) for r in range(7) for c in range(7)]
    a = M.spatial_features(hidden_of(plain, soft), plain, IMAGE_RANGE, ROW_END, 7)
    b = M.spatial_features(hidden_of(mirrored, soft_mirror), mirrored, IMAGE_RANGE, ROW_END, 7)
    width = 5
    soft_lr, soft_tb, vq_lr, vq_tb = a[0].split(width)
    m_soft_lr, m_soft_tb, m_vq_lr, m_vq_tb = b[0].split(width)
    assert torch.allclose(m_soft_lr, -soft_lr, atol=1e-6) and torch.allclose(m_vq_lr, -vq_lr, atol=1e-6)
    assert torch.allclose(m_soft_tb, soft_tb, atol=1e-6) and torch.allclose(m_vq_tb, vq_tb, atol=1e-6)
    assert soft_lr.abs().sum() > 0 and vq_lr.abs().sum() > 0


def test_spatial_heads_start_as_the_checkpoint_heads(tmp_path):
    old = classifier()
    torch.save({"projector": old.projector.state_dict(), "heads": old.heads.state_dict()}, tmp_path / "trainable.pt")
    new = M.Classifier(old.language_model, M.Projector(6, 8), M.Heads(24, D.CLASS_COUNTS, 16, 0.0, spatial_size=32),
                       IMAGE_RANGE, ROW_END).eval()
    report = M.load_weights(new, tmp_path, {"lora_rank": 0})
    assert "spatial" in report
    ids, attention = E.collate([image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])], pad_id=0)
    siglip = torch.randn(1, 4, 6)
    old.soft_side = new.soft_side = 2
    for a, b in zip(old.eval()(ids, attention, siglip), new(ids, attention, siglip)):
        assert torch.allclose(a, b, atol=1e-6)


def test_language_hint():
    pytest.importorskip("langdetect")
    assert E.language_hint("Prodotto nello stabilimento di via Monte | zucchero fecola di patate").startswith("Italian")
    assert E.language_hint("Dit product is gemaakt van gerecycled materiaal en geschikt voor de vaatwasser").startswith("Dutch")
    assert E.language_hint("USB-C | 50 cm | Pro") == "not enough words"
    text = "Zutaten Zucker Weizenmehl Butter Eier Salz"
    assert E.language_hint(text) == E.language_hint(text) and E.language_hint(text).startswith("German")
    assert "language guess: no text" in E.ocr_block("", hint=True) and "language guess" not in E.ocr_block("abc")


def test_spatial_classifier_is_padding_invariant_on_the_real_decoder():
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig

    torch.manual_seed(0)
    config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    heads = M.Heads(96, D.CLASS_COUNTS, 16, 0.0, spatial_size=128)
    torch.nn.init.normal_(heads.shared_spatial.weight, std=0.1)
    model = M.Classifier(text.Apertus1p5TextModel(config).float(), M.Projector(6, 32), heads, IMAGE_RANGE,
                         ROW_END, soft_side=2).eval()
    short = image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])
    long = image_sequence([[1, 2, 5], [3, 4, 6], [7, 8, 9]], [0, 1, 2, 3]) + [4, 5]
    siglip = torch.randn(2, 4, 6)
    ids, attention = E.collate([short, long], pad_id=0)
    together = model(ids, attention, siglip)
    for row, seq in enumerate((short, long)):
        alone = model(*E.collate([seq], pad_id=0), siglip[row:row + 1])
        for a, b in zip(together, alone):
            assert torch.allclose(a[row], b[0], atol=1e-4)


def full_classifier(decoder=None, hidden=8, seed=0):
    torch.manual_seed(seed)
    decoder = decoder or TinyDecoder(hidden=hidden)
    pool = M.AttentionPool(hidden, width=16, queries=2, heads=2, frequencies=2)
    heads = M.Heads(3 * hidden, D.CLASS_COUNTS, 16, 0.0, spatial_size=4 * hidden, pooled_size=pool.output_size)
    return M.Classifier(decoder, M.Projector(6, hidden), heads, IMAGE_RANGE, ROW_END, soft_side=2, pool=pool).eval()


def test_new_layers_start_from_the_checkpoint_outputs(tmp_path):
    old = classifier()
    torch.save({"projector": old.projector.state_dict(), "heads": old.heads.state_dict()}, tmp_path / "trainable.pt")
    new = full_classifier()
    new.language_model = old.language_model
    report = M.load_weights(new, tmp_path, {"lora_rank": 0})
    assert "pooled" in report and "spatial" in report and "new attention pooling" in report
    ids, attention = E.collate([image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])], pad_id=0)
    siglip = torch.randn(1, 4, 6)
    for a, b in zip(old.eval()(ids, attention, siglip), new(ids, attention, siglip)):
        assert torch.allclose(a, b, atol=1e-6)


def test_attention_pool_sees_where_tokens_are():
    torch.manual_seed(0)
    pool = M.AttentionPool(8, width=16, queries=2, heads=2, frequencies=2).eval()
    grid = [[1, 2, 3], [4, 5, 6]]
    plain = torch.tensor([image_sequence(grid)])
    mirrored = torch.tensor([image_sequence([row[::-1] for row in grid])])
    table = torch.randn(256, 8)
    out = {}
    for name, ids in (("plain", plain), ("mirrored", mirrored)):
        hidden = table[ids.clamp(min=0)]
        out[name] = pool(hidden, M.image_grids(ids, IMAGE_RANGE, ROW_END, 7))
        out[name + "_mean"] = hidden[0][(ids[0] >= 100) & (ids[0] < 200)].mean(0)
    assert torch.allclose(out["plain_mean"], out["mirrored_mean"])  # what the old features saw
    assert not torch.allclose(out["plain"], out["mirrored"], atol=1e-4)  # the pooling tells them apart


def test_gradients_reach_the_attention_pool_after_one_step():
    model = full_classifier().train()
    groups = [{"params": [*model.heads.parameters(), *model.pool.parameters()], "lr": 0.1}]
    optimizer = torch.optim.SGD(groups)
    ids, attention = E.collate([image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])] * 2, pad_id=0)
    labels = torch.zeros(2, 6, dtype=torch.long)
    for _ in range(2):
        optimizer.zero_grad()
        sum(l for l in M.task_losses(model(ids, attention, torch.randn(2, 4, 6)), labels, 2.0)).backward()
        optimizer.step()
    assert model.pool.key.weight.grad.abs().sum() > 0 and model.heads.shared_pooled.weight.abs().sum() > 0


def test_attention_pool_is_saved_and_restored(tmp_path):
    trained = full_classifier(seed=1)
    with torch.no_grad():
        for parameter in [*trained.pool.parameters(), *trained.heads.parameters()]:
            parameter.normal_(0, 0.1)
    M.save(trained, {"lora_rank": 0}, tmp_path)
    restored = full_classifier(seed=2)
    restored.language_model = trained.language_model
    assert "attention pooling" in M.load_weights(restored, tmp_path, {"lora_rank": 0})
    ids, attention = E.collate([image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])], pad_id=0)
    siglip = torch.randn(1, 4, 6)
    for a, b in zip(trained(ids, attention, siglip), restored(ids, attention, siglip)):
        assert torch.allclose(a, b, atol=1e-6)


def test_full_head_is_padding_invariant_on_the_real_decoder():
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig

    torch.manual_seed(0)
    config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = full_classifier(text.Apertus1p5TextModel(config).float(), hidden=32)
    for layer in (model.heads.shared_spatial, model.heads.shared_pooled):
        torch.nn.init.normal_(layer.weight, std=0.1)
    short = image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])
    long = image_sequence([[1, 2, 5], [3, 4, 6], [7, 8, 9]], [0, 1, 2, 3]) + [4, 5]
    siglip = torch.randn(2, 4, 6)
    together = model(*E.collate([short, long], pad_id=0), siglip)
    for row, seq in enumerate((short, long)):
        alone = model(*E.collate([seq], pad_id=0), siglip[row:row + 1])
        for a, b in zip(together, alone):
            assert torch.allclose(a[row], b[0], atol=1e-4)


def test_training_loop_pieces_end_to_end():
    """train.Batches + losses + optimizer groups as in train.py, on a tiny model: one epoch must run and learn."""
    import train

    rows = rows_frame()
    labels = D.label_matrix(rows)
    items = D.training_items(rows, labels)
    cache = {D.key(r["image_id"], v): {"grid": (2, 2), "codes": torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                                       "siglip": torch.randn(4, 6)} for r in rows.to_dict("records")
             for v in {v for _, v in items} | {""}}
    class SmallTokenizer(FakeTokenizer):  # special ids inside the 256 ids of TinyDecoder, above ASCII text
        def convert_tokens_to_ids(self, token):
            return {"<|img_start|>": 250, "<|img_token_start|>": 251, "<|img_end_of_row|>": 252, "<|img_end|>": 253}[token]

    builder = E.PromptBuilder(SmallTokenizer(), image_token_offset=200, prefix_tokens=4)
    batches = train.Batches(rows.to_dict("records"), labels, cache, builder, pad_id=0,
                            ocr={i: "NEU" for i in rows.image_id})
    model = full_classifier().train()
    model.image_range, model.row_end_id = (200, 205), 252
    optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-2}])
    alphas = D.class_alpha(D.item_labels(labels, items))
    first = last = None
    for _ in range(30):
        input_ids, attention, siglip, target = batches(items, train=True)
        loss = sum(l for l in M.task_losses(model(input_ids, attention, siglip), target, 2.0, alphas) if l is not None)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first * 0.5, (first, last)


def test_fp16_guard_prevents_mlp_overflow_on_the_real_decoder():
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    import copy

    torch.manual_seed(0)
    config = Apertus1p5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    reference = text.Apertus1p5TextModel(config).float().eval()
    with torch.no_grad():  # activations far beyond the fp16 range, as in the real Apertus
        for layer in reference.layers:
            layer.mlp.up_proj.weight.mul_(5000.0)
            layer.mlp.down_proj.weight.div_(100.0)  # stays in the normal fp16 range, like real weights
    ids = torch.randint(0, 64, (1, 12))
    attention = torch.ones_like(ids)

    def run(model, dtype):
        with torch.no_grad():
            embeds = model.embed_tokens(ids).to(dtype)
            return model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).last_hidden_state.float()

    expected = run(reference, torch.float32)
    plain = copy.deepcopy(reference).half()
    guarded = copy.deepcopy(reference).half()
    assert M.guard_fp16(guarded) == 2
    assert not torch.isfinite(run(plain, torch.float16)).all()  # the overflow the guard must prevent
    result = run(guarded, torch.float16)
    assert torch.isfinite(result).all()
    assert torch.allclose(result, expected, atol=5e-2, rtol=5e-2), (result - expected).abs().max()


def test_fp32_residual_stream_with_fp16_weights_on_the_real_decoder():
    """On T4 the Apertus residual stream exceeds the fp16 range: fp16 weights with an fp32 residual must hold."""
    text = pytest.importorskip("transformers.models.apertus1p5.modeling_apertus1p5")
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    import copy

    torch.manual_seed(0)
    config = Apertus1p5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    reference = text.Apertus1p5TextModel(config).float().eval()
    ids, attention = torch.randint(0, 64, (1, 12)), torch.ones(1, 12, dtype=torch.long)

    def run(model, dtype, autocast):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16, enabled=autocast):
            embeds = (model.embed_tokens(ids).float() * 1e7).to(dtype)  # residual ~2e5, beyond 65504
            return model(inputs_embeds=embeds, attention_mask=attention, use_cache=False).last_hidden_state.float()

    half = copy.deepcopy(reference).half()
    M.guard_fp16(half)
    assert not torch.isfinite(run(half, torch.float16, True)).all()
    result = run(half, torch.float32, True)
    assert torch.isfinite(result).all() and torch.allclose(result, run(reference, torch.float32, False), atol=1e-2)


def test_safe_mode_and_overflow_diagnostics():
    """The T4 safe mode (fp32 linears over fp16 weights) gives finite outputs where fp16 overflows, and the
    diagnostics name the first module that overflowed."""
    model = full_classifier(hidden=8)
    model.language_model.half()
    with torch.no_grad():
        model.language_model.mix.weight.fill_(3e4)  # the tiny decoder's linear overflows in fp16
    ids, attention = E.collate([image_sequence([[1, 2], [3, 4]], [0, 1, 2, 3])], pad_id=0)
    siglip = torch.randn(1, 4, 6)
    with torch.no_grad():
        hidden = model.language_model.mix(torch.ones(1, 8, dtype=torch.float16))
    assert not torch.isfinite(hidden).all()
    assert M.fp32_compute(model.language_model) == 1
    with torch.no_grad():
        safe = model.language_model.mix(torch.ones(1, 8, dtype=torch.float16))
    assert torch.isfinite(safe).all() and safe.dtype == torch.float32
    broken = full_classifier(hidden=8).float()
    with torch.no_grad():
        broken.heads.shared.weight.fill_(float("inf"))
    where = M.first_non_finite(broken, ids, attention, siglip)
    assert where and where[0].startswith("heads.shared")
