import json

import pandas as pd
import pytest

from demo_api.adapters.base import StoredImage
from demo_api.adapters.zeroshot import ZeroShotGenerativeAdapter
from demo_api.errors import ApiError
from demo_api.registry import Registry
from demo_api.repo_code import eval_prompt_module, prompt_module
from fakes import EXPECTED_A, fake_complete
from imgs import make


@pytest.fixture
def adapter(settings):
    entry = Registry(settings).get("apertus-8b-zeroshot-v2").raw
    return ZeroShotGenerativeAdapter(entry, parallel=2)


@pytest.fixture
def image_file(tmp_path):
    path = tmp_path / "pics" / "x.jpg"
    path.parent.mkdir()
    path.write_bytes(make("JPEG", size=(120, 80)))
    return path


@pytest.mark.parametrize("use_context", [True, False])
def test_seven_request_bodies_identical_to_run_variant(adapter, image_file, tmp_path, monkeypatch, use_context):
    mod = eval_prompt_module()
    ctx = {"product_title": "Test kettle", "brand": "Acme"}
    # riferimento: eval_prompt.run_variant, con DATA sulla cartella temporanea e un gateway finto
    reference_calls = []
    monkeypatch.setattr(mod, "DATA", tmp_path)
    monkeypatch.setattr(mod, "complete", fake_complete(reference_calls))
    row = {"image_id": "ref-1", "image_path": str(image_file.relative_to(tmp_path)), **ctx}
    spec = "8b:describe:codes:512" + ("" if use_context else ":nocontext")
    mod.run_variant(spec, pd.DataFrame([row]), 1)
    assert len(reference_calls) == 7
    # demo: stessa immagine, stesso contesto
    demo_calls = []
    monkeypatch.setattr(mod, "complete", fake_complete(demo_calls))
    adapter.predict_one(StoredImage("img-demo-1", "0" * 64, image_file, ctx), use_context)
    assert demo_calls == reference_calls


def test_prediction_fields_and_confidence(adapter, image_file, monkeypatch):
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete())
    result = adapter.predict_one(StoredImage("img-demo-1", "0" * 64, image_file, {}), True)
    assert list(result.dimensions) == list(prompt_module().DIMENSIONS) and result.valid
    subject = result.dimensions["subject"]
    assert subject.label == "full_product" and subject.confidence == pytest.approx(EXPECTED_A)
    assert subject.confidence_method == "letter_logprob" and [l for l, _ in subject.top_k][0] == "full_product"
    assert len(subject.top_k) == 3 and subject.top_k[0][1] >= subject.top_k[1][1] >= subject.top_k[2][1]
    assert result.raw_output == "A synthetic test description."
    assert result.timing_ms["total"] >= result.timing_ms["preprocess"]


def test_missing_logprobs_give_na_confidence(adapter, image_file, monkeypatch):
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete(with_logprobs=False))
    d = adapter.predict_one(StoredImage("i", "0" * 64, image_file, {}), True).dimensions["background"]
    assert d.label == "white" and d.confidence is None and d.top_k is None and d.confidence_method is None


def test_upstream_error_hides_key(adapter, image_file, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret-key")
    monkeypatch.setattr(eval_prompt_module(), "complete", fake_complete(fail="HTTP 500: super-secret-key leaked"))
    with pytest.raises(ApiError) as err:
        adapter.predict_one(StoredImage("i", "0" * 64, image_file, {}), True)
    assert err.value.code == "UPSTREAM_ERROR" and "super-secret-key" not in json.dumps(err.value.body())


def test_prompt_version_and_model_name(adapter):
    assert adapter.model_name == "apertus-v1-5-8b" and adapter.prompt_version.startswith("v2:8b:describe:codes:512:")
