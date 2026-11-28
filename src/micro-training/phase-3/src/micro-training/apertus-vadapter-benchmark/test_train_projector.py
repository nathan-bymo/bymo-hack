import json

import torch
from PIL import Image
from types import SimpleNamespace

import train_projector
from train_projector import make_training_batch, parse_prediction, sparse_causal_loss
from supervision import DIMENSIONS


def test_sparse_loss_matches_full_causal_loss_and_gradient():
    torch.manual_seed(5)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    head = torch.nn.Linear(4, 7, bias=False)
    labels = torch.tensor([[-100, -100, 4, 2, -100], [-100, 3, 1, -100, 5]])
    actual = sparse_causal_loss(hidden, labels, head)
    logits = head(hidden[:, :-1, :]).reshape(-1, 7)
    expected = torch.nn.functional.cross_entropy(logits, labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_prediction_parser_treats_invalid_values_as_missing():
    allowed = {name: {"none"} for name in DIMENSIONS}
    response = {name: "none" for name in DIMENSIONS}
    response["subject"] = ["not", "a", "class"]
    parsed = parse_prediction("prefix " + json.dumps(response), allowed)
    assert parsed["subject"] == ""
    assert all(parsed[key] == "none" for key in DIMENSIONS if key != "subject")
    assert all(not value for value in parse_prediction("no json", allowed).values())


def test_training_batch_pads_ids_labels_and_images_without_losing_boundaries(tmp_path, monkeypatch):
    Image.new("RGB", (3, 2), "white").save(tmp_path / "one.png")
    Image.new("RGB", (2, 4), "white").save(tmp_path / "two.png")
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    rows = [{"image_path": "one.png", "n": 2}, {"image_path": "two.png", "n": 4}]

    def inputs(processor, image, row, prompt, vq_size):
        return {"input_ids": torch.arange(1, row["n"] + 1).unsqueeze(0),
                "pixel_values": torch.ones(1, 3, image.height, image.width),
                "image_sizes": torch.tensor([[image.height, image.width]])}

    monkeypatch.setattr(train_projector, "prompt_inputs", inputs)
    monkeypatch.setattr(train_projector, "supervised_answer", lambda row, tok: ("x", [7, 8], [7, -100]))
    images, ids, mask, labels, pixels, sizes, prompt_lengths = make_training_batch(rows, tmp_path, processor, "p", 256, 9)
    assert ids.tolist() == [[1, 2, 7, 8, 9, 0, 0], [1, 2, 3, 4, 7, 8, 9]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    assert labels.tolist() == [[-100, -100, 7, -100, 9, -100, -100], [-100, -100, -100, -100, 7, -100, 9]]
    assert sizes.tolist() == [[2, 3], [4, 2]] and prompt_lengths == [2, 4]
    assert pixels.shape == (2, 3, 4, 3) and pixels[0, :, 2:, :].sum() == 0
    assert len(images) == 2
