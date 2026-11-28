"""CPU tests of the full-finetuning plumbing, with the real Apertus decoder class in miniature (random weights).
They prove nothing on quality.

  python -m pytest -q src/training/apertus-8B-fullft-heads/test_fullft.py      (or without pytest:)
  python src/training/apertus-8B-fullft-heads/test_fullft.py
"""

import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ft_model as M  # noqa: E402
import data as D  # noqa: E402
import encoders as E  # noqa: E402
from ft_encode import variant_jobs, worker_jobs  # noqa: E402

ROOT = HERE.parents[2]
IMAGE_RANGE = (150, 250)


def mini_classifier(layers=2, seed=0):
    from transformers.models.apertus1p5.configuration_apertus1p5 import Apertus1p5TextConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5TextModel

    torch.manual_seed(seed)
    config = Apertus1p5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    decoder = Apertus1p5TextModel(config).to(torch.bfloat16)  # as loaded from the checkpoint
    decoder.layers = decoder.layers[:layers]
    decoder.config.num_hidden_layers = layers
    decoder.requires_grad_(False)
    return M.Classifier(decoder, M.Heads(64, D.CLASS_COUNTS, 16, 0.0), IMAGE_RANGE)


def batch(*lengths):
    sequences = [[1, 2] + list(range(150, 150 + n)) + [3, 4] for n in lengths]
    return E.collate(sequences, pad_id=0)


def labels(rows, missing_task=None):
    result = torch.stack([torch.tensor([i % c for c in D.CLASS_COUNTS]) for i in range(rows)])
    if missing_task is not None:
        result[:, missing_task] = -1
    return result


def test_only_layers_norm_and_heads_train_in_fp32():
    model = mini_classifier()
    trained = M.make_trainable(model.language_model)
    decoder = model.language_model
    assert all(p.dtype == torch.float32 and p.requires_grad for p in decoder.layers.parameters())
    assert decoder.norm.weight.requires_grad and decoder.norm.weight.dtype == torch.float32
    assert not decoder.embed_tokens.weight.requires_grad and decoder.embed_tokens.weight.dtype == torch.bfloat16
    assert len(trained) == len(list(decoder.layers.parameters())) + 1
    groups = M.parameter_groups(model, 1e-5, 1e-3, 0.01)
    grouped = [p for g in groups for p in g["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(trained) + len(list(model.heads.parameters()))
    assert all(p.ndim == 1 for p in groups[1]["params"]) and groups[1]["weight_decay"] == 0.0


def test_gradients_reach_every_layer_and_heads_not_embeddings():
    model = mini_classifier()
    M.make_trainable(model.language_model)
    ids, attention = batch(3, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(ids, attention)
    sum(l for l in M.task_losses(logits, labels(2), 2.0) if l is not None).backward()
    for name, parameter in model.language_model.named_parameters():
        if name.startswith("embed_tokens"):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(p.grad is not None for p in model.heads.parameters())


def test_gradient_checkpointing_gives_the_same_gradients():
    ids, attention = batch(3, 5)
    gradients = []
    for checkpointing in (False, True):
        model = mini_classifier()
        M.make_trainable(model.language_model)
        model.train()
        if checkpointing:
            model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        sum(l for l in M.task_losses(model(ids, attention), labels(2), 2.0) if l is not None).backward()
        gradients.append([p.grad.clone() for p in model.language_model.layers.parameters()])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*gradients))


def test_an_unscored_dimension_gives_no_gradient_to_its_head():
    model = mini_classifier()
    M.make_trainable(model.language_model)
    ids, attention = batch(3, 5)
    task = D.DIMENSIONS.index("text_language")
    sum(l for l in M.task_losses(model(ids, attention), labels(2, task), 2.0) if l is not None).backward()
    assert model.heads.heads[task].weight.grad is None
    assert model.heads.heads[0].weight.grad.abs().sum() > 0


def test_features_are_last_token_and_image_mean_and_padding_does_not_matter():
    model = mini_classifier()
    M.make_trainable(model.language_model)
    model.eval()
    ids, attention = batch(2, 6)
    hidden = model.language_model(inputs_embeds=model.language_model.embed_tokens(ids), attention_mask=attention,
                                  position_ids=(attention.cumsum(1) - 1).clamp(min=0)).last_hidden_state.float()
    features = M.pooled_features(hidden, ids, attention, IMAGE_RANGE)
    assert features.shape == (2, 64)
    image = (ids[0] >= 150) & (ids[0] < 250)
    assert torch.allclose(features[0, :32], hidden[0, -1]) and torch.allclose(features[0, 32:], hidden[0, image].mean(0))
    together = model(ids, attention)
    for row, length in enumerate((2, 6)):
        alone = model(*batch(length))
        for a, b in zip(together, alone):
            assert torch.allclose(a[row], b[0], atol=2e-2), (a[row], b[0])


def test_save_and_load_give_the_same_logits():
    trained = mini_classifier()
    M.make_trainable(trained.language_model)
    with torch.no_grad():  # move the weights away from the base, as training would
        for parameter in trained.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05 * torch.randn_like(parameter))
    trained.eval()
    ids, attention = batch(3, 4)
    expected = trained(ids, attention)
    with tempfile.TemporaryDirectory() as folder:
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 5e-2)):
            M.save(trained, {"layers": 2}, Path(folder) / "best", dtype=dtype)
            names = sorted(p.name for p in (Path(folder) / "best").iterdir())
            assert names == ["decoder.safetensors", "heads.pt", "settings.json"], names
            fresh = mini_classifier()  # base weights again (same seed), other heads
            if dtype == torch.float32:
                M.make_trainable(fresh.language_model)
            M.load_weights(fresh, Path(folder) / "best").eval()
            for a, b in zip(expected, fresh(ids, attention)):
                assert torch.allclose(a, b, atol=tolerance), (a - b).abs().max()


def test_a_wrong_checkpoint_is_refused():
    trained = mini_classifier(layers=2)
    with tempfile.TemporaryDirectory() as folder:
        M.save(trained, {"layers": 2}, Path(folder) / "best")
        try:
            M.load_weights(mini_classifier(layers=1), Path(folder) / "best")
        except ValueError:
            return
    raise AssertionError("a checkpoint with more layers was accepted")


def test_a_few_steps_lower_the_loss():
    model = mini_classifier()
    M.make_trainable(model.language_model)
    model.train()
    optimizer = torch.optim.AdamW(M.parameter_groups(model, 1e-3, 1e-2, 0.0))
    ids, attention = batch(3, 5, 4, 6)
    target = labels(4)

    def loss():
        return sum(l for l in M.task_losses(model(ids, attention), target, 2.0) if l is not None)

    first = loss().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    assert loss().item() < 0.5 * first


def test_workers_share_the_variants_exactly_once():
    import pandas as pd

    rows = pd.DataFrame({"image_id": [f"i{n}" for n in range(7)], "image_path": [f"p{n}" for n in range(7)]})
    items = [(0, ""), (0, "a1"), (3, "m"), (5, "a1m"), (5, "a2")]
    jobs = variant_jobs(rows, rows, items)
    assert len(jobs) == 7 + 4
    shares = [worker_jobs(jobs, w, 3) for w in range(3)]
    assert sum(len(s) for s in shares) == len(jobs) and set().union(*shares) == set(jobs)


def test_recipe_matches_dati_md_on_the_real_train_csv():
    """Counts of DATI.md (phase 1): 8,403 images to train, 941 to validate, 13,390 items per epoch."""
    found = [p for p in (ROOT / "data" / "train.csv", ROOT / "data" / "hackapertus-galaxus" / "train.csv") if p.exists()]
    if not found:
        print("skipped: no train.csv")
        return
    rows = D.select(D.read_csv(found[0]), "training")
    fit, validation = D.internal_split(rows, 0.1)
    assert (len(fit), len(validation)) == (8403, 941)
    assert len(D.training_items(fit, D.label_matrix(fit))) == 13390
    assert not set(fit.product_id) & set(validation.product_id)


if __name__ == "__main__":
    tests = [(name, test) for name, test in sorted(globals().items()) if name.startswith("test_")]
    for name, test in tests:
        test()
        print("ok", name)
    print(f"{len(tests)} tests passed")
