import torch
from PIL import Image

from eval_describe_codes import RestrictTokens, allowed_code_ids, local_messages
from common import prompt


def test_logits_are_restricted_to_available_codes():
    scores = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    result = RestrictTokens([0, 2])(torch.tensor([[0]]), scores)
    assert result.argmax(-1).item() == 2
    assert torch.isneginf(result[0, 1]) and torch.isneginf(result[0, 3])


def test_v2_conversation_keeps_description_and_questions():
    image = Image.new("RGB", (32, 32))
    messages = prompt.question_messages({}, "local", "perspective", answer="codes", description="a product")
    translated = local_messages(messages, image)
    assert translated[0]["content"][0]["image"] is image
    assert translated[0]["content"][1]["text"] == prompt.DESCRIBE
    assert translated[1] == {"role": "assistant", "content": "a product"}
    assert prompt.question_text("perspective", "codes") in translated[2]["content"]


def test_codes_are_mapped_back_to_token_ids():
    class Tokenizer:
        def encode(self, token, add_special_tokens=False):
            return [ord(token)]

        def decode(self, ids):
            return chr(ids[0])

    assert allowed_code_ids(Tokenizer(), "individuals") == {ord("A"): "A", ord("B"): "B"}
