"""Tests on a CPU. The end-to-end test runs encode -> train -> save -> reload -> predict (with a resume) ->
submission, with the real Apertus decoder class and real PEFT on a random miniature model. The tokenizer and the
VQ tokenizer are stand-ins. It proves the plumbing, nothing about the real weights or the quality.

  python -m pytest -q src/training/apertus-8B-lora-heads/test_pipeline.py

Needs torch, the Transformers fork of requirements-gpu.txt, peft and pytest. Two tests use data/ when present.
"""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import data as D  # noqa: E402
import encode  # noqa: E402
import encoders as E  # noqa: E402
import make_submission  # noqa: E402
import model as M  # noqa: E402
import predict  # noqa: E402
import train  # noqa: E402

REAL = ROOT / "data" / "train.csv"


def row(index, **values):
    base = {"image_id": f"x{index}", "product_id": f"p{index}", "image_path": f"images/{index}.png",
            "product_title": "Thing", "product_type": "", "brand": "", "category": "", "subcategory": ""}
    labels = {"subject": "full_product", "background": "white", "perspective": "front_angle_left",
              "individuals": "none", "text_classification": "none", "text_language": "none"}
    labels.update({k: v for k, v in values.items() if k in labels})
    for dimension, value in labels.items():
        base |= {dimension: value, f"{dimension}_source": "human", f"{dimension}_relevant": "true"}
    base.update({k: v for k, v in values.items() if k not in labels})
    return base


# ---------------------------------------------------------------- data


def test_labels_human_and_scored_cells():
    rows = pd.DataFrame([row(1, background_relevant="false", individuals_source="model")])
    human, scored = D.label_matrix(rows, "human")[0], D.label_matrix(rows, "scored")[0]
    background, individuals = D.DIMENSIONS.index("background"), D.DIMENSIONS.index("individuals")
    assert human[background] >= 0 and scored[background] == -1
    assert human[individuals] == -1 and scored[individuals] == -1


def test_repeats_are_identical_copies_and_mirrors_swap_labels():
    rows = pd.DataFrame([row(1), row(2, perspective="front_head_on"), row(3, text_language="German")])
    items = D.training_items(rows)
    assert {variant for _, variant in items} <= {"", "m"}, "no crops: a cut product is not a full_product"
    assert [v for i, v in items if i == 0].count("m") == 1
    assert "m" not in [v for i, v in items if i in (1, 2)], "no mirror without left/right or with text"
    assert [v for i, v in items if i == 0].count("") == D.MAX_REPEATS
    labels = D.label_matrix(rows, "human")
    mirrored = D.item_labels(labels, [(0, "m")])[0]
    assert D.TAXONOMY["perspective"][mirrored[D.PERSPECTIVE]] == "front_angle_right"


@pytest.mark.skipif(not REAL.exists(), reason="competition data not on this machine")
def test_same_items_as_the_lora_baseline():
    sys.path.insert(0, str(ROOT / "src" / "micro-training" / "lora-generativo"))
    baseline = pytest.importorskip("apertus_lora")
    all_rows = D.read_csv(REAL)
    splits_to_check = [D.select(all_rows, "training")]
    small = ROOT / "data" / "nathan_split" / "train_1500.csv"
    if small.exists():
        splits_to_check.append(D.read_csv(small))
    for rows in splits_to_check:
        records = rows.to_dict("records")
        assert D.repeat_counts(records) == baseline.repeat_counts(records)
        assert [D.is_mirrorable(r) for r in records] == [baseline.is_mirrorable(r) for r in records]
        expected = sum(baseline.repeat_counts(records)) + sum(map(baseline.is_mirrorable, records))
        assert len(D.training_items(rows)) == expected
        labels = D.label_matrix(rows, "human")
        for index, record in enumerate(records[:300]):
            trained = [d for t, d in enumerate(D.DIMENSIONS) if labels[index, t] >= 0]
            assert trained == baseline.human_dimensions(record)
    if small.exists():
        assert len(D.training_items(D.read_csv(small))) == 3952  # as in its train_summary.json


@pytest.mark.skipif(not REAL.exists(), reason="competition data not on this machine")
def test_training_splits_never_touch_coverage():
    all_rows = D.read_csv(REAL)
    D.assert_clean(D.select(all_rows, "training"), all_rows)
    with pytest.raises(ValueError):
        D.assert_clean(D.select(all_rows, "coverage").head(3), all_rows)


# ---------------------------------------------------------------- losses and features


def test_default_loss_is_cross_entropy_and_unlabeled_cells_do_nothing():
    torch.manual_seed(0)
    logits, targets = torch.randn(16, 7), torch.randint(0, 7, (16,))
    assert torch.allclose(M.focal_loss(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)
    defaults = train.parser().parse_args(["--cache", "x", "--out", "y"])
    assert (defaults.gamma, defaults.class_alpha, defaults.layers, defaults.epochs) == (0.0, "none", 32, 3.0)
    assert (defaults.batch_size, defaults.accumulation, defaults.label_cells) == (8, 2, "human")
    heads = [torch.randn(4, n, requires_grad=True) for n in D.CLASS_COUNTS]
    labels = torch.zeros(4, 6, dtype=torch.long)
    labels[:, 2] = -1
    losses = M.task_losses(heads, labels)
    assert losses[2] is None
    sum(x for x in losses if x is not None).backward()
    assert heads[2].grad is None


def test_features_are_last_token_and_image_mean():
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    ids, attention = torch.tensor([[0, 100, 101, 5]]), torch.tensor([[0, 1, 1, 1]])
    features = M.pooled_features(hidden, ids, attention, (100, 200))
    assert torch.equal(features[0, :6], hidden[0, 3]) and torch.equal(features[0, 6:], hidden[0, 1:3].mean(0))


# ---------------------------------------------------------------- submission


def sample_frame():
    constant = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
                "individuals": "none", "text_classification": "none", "text_language": "none"}
    return pd.DataFrame([{"image_id": f"x{i}", **constant} for i in range(5)])


def test_submission_keeps_order_and_refuses_bad_files():
    sample = sample_frame()
    good = sample.iloc[[3, 1]].assign(subject="logo")
    submission, counts = make_submission.build(sample, [good], {"x1", "x3"})
    assert list(submission.image_id) == list(sample.image_id)
    assert list(submission.subject) == ["full_product", "logo", "full_product", "logo", "full_product"]
    assert counts["predicted"] == 2 and counts["kept_from_sample"] == 3
    for bad, required in [(good.assign(subject="Logo"), {"x1"}), (good.assign(individuals=""), {"x1"}),
                          (pd.concat([good, good]), {"x1"}), (good, {"x0"}), (good.assign(image_id=["zz", "x1"]), set())]:
        with pytest.raises(ValueError):
            make_submission.build(sample, [bad], required)


def test_final_upload_failure_is_not_silent(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / "heads.pt").write_text("x")

    class Incomplete:
        def upload_folder(self, **_):
            pass

        def list_repo_files(self, _):
            return []

    monkeypatch.setattr(huggingface_hub, "HfApi", Incomplete)
    train.upload("a/b", tmp_path, "model/snapshot")  # a snapshot only logs the problem
    with pytest.raises(RuntimeError):
        train.upload("a/b", tmp_path, "model/final", required=True)


# ---------------------------------------------------------------- end to end


class FakeTokenizer:
    pad_token_id = 0
    special = {"<|img_start|>": 3, "<|img_token_start|>": 4, "<|img_end_of_row|>": 5, "<|img_end|>": 6}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 80) for c in text]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def apply_chat_template(self, messages, **_):
        return "<s>user " + "<|image|>" + messages[0]["content"][1]["text"] + " assistant"


class FakeImageEncoder:
    """Prepares the real pixels, then makes deterministic codes from them."""

    def __init__(self, directory, device="cuda"):
        pass

    def encode(self, paths, variants, side):
        results = []
        for path, variant in zip(paths, variants):
            image = E.prepared_image(path, variant, 32)
            pixels = torch.tensor(list(image.resize((2, 2)).tobytes()), dtype=torch.int32)
            results.append({"grid": (2, 2), "codes": (pixels.reshape(4, 3).sum(1) % 60).to(torch.int32)})
        return results


def tiny_decoder(layers):
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    torch.manual_seed(123)  # the same "base weights" at every load
    config = Apertus1p5TextConfig(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, pad_token_id=0,
                                  max_position_embeddings=512)
    return M.truncate(Apertus1p5TextModel(config), layers)


@pytest.fixture
def tiny_world(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Apertus1p5ForConditionalGeneration"):
        pytest.skip("needs the Transformers fork with Apertus 1.5")
    from PIL import Image

    (tmp_path / "images").mkdir()
    subjects = ["full_product", "logo", "infographic"]
    rows = []
    for i in range(24):
        Image.new("RGB", (40 + i, 30), (10 * i % 255, 255 - 7 * i, 90)).save(tmp_path / "images" / f"{i}.png")
        rows.append(row(i, subject=subjects[i % 3], perspective=["front_angle_left", "front_head_on"][i % 2]))
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(rows[:9]).to_csv(tmp_path / "small.csv", index=False)
    if (ROOT / "data" / "metric.py").exists():
        shutil.copy(ROOT / "data" / "metric.py", tmp_path / "metric.py")
    monkeypatch.setattr(E, "model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(E, "ImageEncoder", FakeImageEncoder)
    monkeypatch.setattr(M, "load_language_model", lambda directory, layers, device="cpu": (tiny_decoder(layers), (100, 160)))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(image_token_offset=100))
    monkeypatch.setattr(D, "assert_clean", lambda *a: None)  # checked on the real data above
    return tmp_path


def test_end_to_end(tiny_world):
    world, cache, out = tiny_world, tiny_world / "cache", tiny_world / "run"
    encode.main(["--data-dir", str(world), "--train-csv", str(world / "small.csv"), "--out", str(cache),
                 "--side", "256", "--device", "cpu"])
    result = train.main(["--data-dir", str(world), "--train-csv", str(world / "small.csv"), "--cache", str(cache),
                         "--out", str(out), "--layers", "3", "--lora-rank", "4", "--lora-alpha", "4",
                         "--epochs", "3", "--batch-size", "2", "--accumulation", "2", "--lr-lora", "0.01",
                         "--device", "cpu"])
    summary = json.loads((result / "train_summary.json").read_text())
    assert result == out / "final" and summary["steps"] == 3 * summary["steps_per_epoch"] > 3
    assert (result / "heads.pt").exists() and (result / "lora" / "adapter_model.safetensors").exists()
    assert not (result / "lora" / "README.md").exists()
    assert (out / "snapshot" / "heads.pt").exists()

    # LoRA learned something, and the reloaded model is the trained one (merged and unmerged agree)
    from safetensors.torch import load_file

    adapter = load_file(result / "lora" / "adapter_model.safetensors")
    assert any("lora_B" in name and tensor.abs().sum() > 0 for name, tensor in adapter.items())
    assert len(adapter) == 3 * 6 * 2, "LoRA on q, k, v, o, up and down of every layer"
    ids, attention = E.collate([[3, 10, 101, 102, 20], [101, 30]], 0)
    outputs = []
    for merge in (True, False):
        classifier, _ = M.load(str(world), result, "cpu", merge=merge)
        with torch.inference_mode():
            outputs.append(torch.cat(classifier(ids, attention), dim=1))
    assert torch.allclose(outputs[0], outputs[1], atol=1e-5)
    with torch.inference_mode():  # padding does not change the answer of the shorter row
        alone = torch.cat(classifier(ids[1:, -2:], attention[1:, -2:]), dim=1)
    assert torch.allclose(alone, outputs[1][1:], atol=1e-5)

    # predictions: stop after 8 images, resume, an unreadable image is left out
    rows = pd.read_csv(world / "train.csv", dtype=str, keep_default_na=False)
    (world / "images" / "20.png").write_bytes(b"not an image")
    rows.to_csv(world / "dev.csv", index=False)
    arguments = ["--checkpoint", str(result), "--data-dir", str(world), "--csv", str(world / "dev.csv"),
                 "--out", str(world / "pred" / "dev.csv"), "--batch-size", "4", "--device", "cpu"]
    predict.main([*arguments, "--limit", "8"])
    first = pd.read_csv(world / "pred" / "dev.csv", dtype=str, keep_default_na=False)
    predict.main(arguments)
    full = pd.read_csv(world / "pred" / "dev.csv", dtype=str, keep_default_na=False)
    report = json.loads((world / "pred" / "dev.report.json").read_text())
    assert len(first) == 8 and len(full) == 23 and report["unreadable"] == 1 and report["new_in_this_run"] == 15
    assert full.head(8).equals(first), "resumed predictions are not recomputed differently"
    for dimension in D.DIMENSIONS:
        assert full[dimension].isin(D.TAXONOMY[dimension]).all()
    with pytest.raises(SystemExit):  # another setting must not be mixed into the same partial file
        predict.main([*arguments, "--no-context"])

    # the submission: the unreadable dev image keeps the sample answer only when allowed
    constant = {d: D.TAXONOMY[d][0] for d in D.DIMENSIONS}
    pd.DataFrame([{"image_id": i, **constant} for i in rows.image_id]).to_csv(world / "sample.csv", index=False)
    command = ["--sample", str(world / "sample.csv"), "--predictions", str(world / "pred" / "dev.csv"),
               "--require", str(world / "dev.csv"), "--out", str(world / "submission.csv")]
    with pytest.raises(SystemExit):
        make_submission.main(command)
    make_submission.main([*command, "--allow-missing", "1"])
    submission = pd.read_csv(world / "submission.csv", dtype=str, keep_default_na=False)
    assert len(submission) == 24 and not (submission == "").any().any()


def test_held_out_fraction_keeps_the_best_checkpoint(tiny_world):
    if not (tiny_world / "metric.py").exists():
        pytest.skip("metric.py comes with the competition data")
    world, cache = tiny_world, tiny_world / "cache"
    encode.main(["--data-dir", str(world), "--train-csv", str(world / "train.csv"), "--out", str(cache),
                 "--device", "cpu"])
    result = train.main(["--data-dir", str(world), "--train-csv", str(world / "train.csv"), "--cache", str(cache),
                         "--out", str(world / "run"), "--layers", "2", "--lora-rank", "4", "--epochs", "1",
                         "--val-fraction", "0.3", "--batch-size", "4", "--accumulation", "1", "--device", "cpu",
                         "--grad-ckpt"])  # the fallback of the job when the GPU runs out of memory
    assert result == world / "run" / "best" and json.loads((result / "train_summary.json").read_text())["best"]


# ---------------------------------------------------------------- spatial features and attention pooling


def mirrored_grid(hidden, height, width):
    """The same tokens with every row reversed: what the VQ grid of a mirrored image looks like."""
    order = torch.arange(height * width).reshape(height, width).flip(1).flatten()
    return hidden[:, order]


def test_spatial_features_change_sign_with_the_mirror_and_means_do_not():
    torch.manual_seed(0)
    height, width = 3, 4
    hidden = torch.randn(1, height * width, 8)
    grids = [(torch.arange(height * width), width)]
    plain, mirror = M.spatial_features(hidden, grids), M.spatial_features(mirrored_grid(hidden, height, width), grids)
    assert torch.allclose(plain[0, :8], -mirror[0, :8], atol=1e-6), "left - right flips"
    assert torch.allclose(plain[0, 8:], mirror[0, 8:], atol=1e-6), "top - bottom does not"
    assert torch.allclose(hidden.mean(1), mirrored_grid(hidden, height, width).mean(1)), "the mean cannot tell"


def test_grid_width_comes_from_the_first_end_of_row():
    ids = torch.tensor([[1, 100, 101, 102, 5, 103, 104, 105, 6, 7]])
    (positions, width), = M.image_grids(ids, (100, 200), row_end_id=5)
    assert positions.tolist() == [1, 2, 3, 5, 6, 7] and width == 3


def test_attention_pool_sees_positions():
    torch.manual_seed(0)
    pool = M.AttentionPool(8, width=16, queries=2, heads=2).eval()
    hidden = torch.randn(1, 12, 8)
    grids = [(torch.arange(12), 4)]
    with torch.no_grad():
        plain, mirror = pool(hidden, grids), pool(mirrored_grid(hidden, 3, 4), grids)
        padded = pool(torch.cat([hidden, torch.randn(1, 5, 8)], 1), grids)  # extra tokens outside the grid
    assert plain.shape == (1, 32) and not torch.allclose(plain, mirror)
    assert torch.allclose(plain, padded, atol=1e-6)


def test_old_checkpoint_settings_still_build_plain_heads():
    decoder = tiny_decoder_for_unit_tests()
    classifier = M.make_classifier(decoder, (100, 160), {"head_hidden": 16, "head_dropout": 0.0})
    assert classifier.pool is None and not classifier.spatial and classifier.heads.spatial_size == 0
    with pytest.raises(ValueError):
        M.make_classifier(decoder, (100, 160), {"head_hidden": 16, "head_dropout": 0.0, "spatial": True})


def tiny_decoder_for_unit_tests():
    pytest.importorskip("transformers")
    try:
        return tiny_decoder(2)
    except ImportError:
        pytest.skip("needs the Transformers fork with Apertus 1.5")


def test_end_to_end_with_spatial_features_and_attention_pooling(tiny_world):
    world, cache, out = tiny_world, tiny_world / "cache", tiny_world / "run_pool"
    encode.main(["--data-dir", str(world), "--train-csv", str(world / "small.csv"), "--out", str(cache),
                 "--device", "cpu"])
    result = train.main(["--data-dir", str(world), "--train-csv", str(world / "small.csv"), "--cache", str(cache),
                         "--out", str(out), "--layers", "2", "--lora-rank", "4", "--epochs", "1", "--batch-size", "4",
                         "--accumulation", "1", "--spatial", "--attention-pool", "--device", "cpu"])
    settings = json.loads((result / "settings.json").read_text())
    assert settings["spatial"] and settings["attention_pool"] and settings["row_end_id"] == FakeTokenizer.special["<|img_end_of_row|>"]
    saved = torch.load(result / "heads.pt", weights_only=True)
    assert "pool" in saved and any("shared_spatial" in k for k in saved["heads"])
    # a sequence with a 2 x 2 image: the reloaded model predicts like itself, merged or not
    ids, attention = E.collate([[3, 10, 101, 102, 5, 103, 104, 6, 20], [3, 101, 102, 5, 103, 104, 6, 30]], 0)
    outputs = []
    for merge in (True, False):
        classifier, _ = M.load(str(world), result, "cpu", merge=merge)
        assert classifier.pool is not None and classifier.spatial
        with torch.inference_mode():
            outputs.append(torch.cat(classifier(ids, attention), dim=1))
    assert torch.allclose(outputs[0], outputs[1], atol=1e-5)
    predict.main(["--checkpoint", str(result), "--data-dir", str(world), "--csv", str(world / "small.csv"),
                  "--out", str(world / "pred_pool" / "small.csv"), "--batch-size", "4", "--device", "cpu"])
    assert len(pd.read_csv(world / "pred_pool" / "small.csv")) == 9
