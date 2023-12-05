import token_transfer as tt
import unittest
import json
import numpy as np


class TestTokenTransfer(unittest.TestCase):
    def setUp(self) -> None:
        with open("test-data.json", "r") as f:
            self.data = json.load(f)
        return super().setUp()

    def test_token_transfer(self):
        source_tokens = ["whe", "rever", "_cou", "ld", "<eos>"]
        target_tokens = ["wh", "er", "ever", "_co", "uld", "<eos>"]
        probs = [
            [("rever", 0.4), ("never", 0.2), ("re", 0.2), ("lp", 0.05), ("lm", 0.05)],
            [("_cou", 0.2), ("_can", 0.1), ("_will", 0.05), ("_shall", 0.05)],
            [("ld", 0.8), ("ldve", 0.1), ("th", 0.025), ("pe", 0.05), ("nt", 0.025)],
            [
                ("<eos>", 0.75),
                ("_have", 0.1),
                ("_you", 0.05),
                ("_your", 0.05),
                ("'nt", 0.05),
            ],
        ]
        source_log_probs = [0.0]
        for i, source_token_probs in enumerate(probs):
            source_token = source_tokens[i + 1]
            stp = [st[1] for st in source_token_probs if st[0] == source_token]
            assert len(stp) == 1
            source_log_probs.append(np.log(stp[0]))
        target_token_log_probs = tt.token_transfer(source_tokens, target_tokens, probs)
        assert np.allclose(np.sum(target_token_log_probs), np.sum(source_log_probs))

    def test_from_openai_identity(self):
        target_token_logp = tt.from_openai_response(self.data, self.data["tokens"])
        self.assertTrue(np.allclose(self.data["token_logprobs"], target_token_logp[1:]))

    def test_from_openai(self):
        assert self.data["tokens"] == [
            "\n\n",
            "This",
            " is",
            " a",
            " test",
            ".",
        ], "this test depends on the inputs"
        target_tokens = ["\n", "\n", "This is", " a", " test."]
        target_token_logp = tt.from_openai_response(self.data, target_tokens)

        self.assertTrue(
            np.allclose(sum(self.data["token_logprobs"]), sum(target_token_logp[1:]))
        )


if __name__ == "__main__":
    unittest.main()
