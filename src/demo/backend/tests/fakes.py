"""Gateway finto, solo per i test: nessuna rete."""

import math

QUESTION_LETTER = "A"


def fake_complete(calls=None, letter=QUESTION_LETTER, with_logprobs=True, fail=None):
    def complete(body, attempts=5):
        if calls is not None:
            calls.append(body)
        if fail:
            raise RuntimeError(fail)
        if "structured_outputs" in body:
            choice = {"message": {"content": letter}}
            if with_logprobs:
                choice["logprobs"] = {"content": [{"token": letter, "top_logprobs": [
                    {"token": letter, "logprob": -0.1}, {"token": "B", "logprob": -2.4}]}]}
            return {"choices": [choice], "usage": {"prompt_tokens": 10}}
        return {"choices": [{"message": {"content": "A synthetic test description."}}],
                "usage": {"prompt_tokens": 10}}

    return complete


EXPECTED_A = math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.4))
