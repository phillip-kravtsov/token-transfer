from token_transfer import token_transfer, from_openai
import unittest
import numpy as np
data = {
    "tokens": ["\n\n", "This", " is", " a", " test", "."],
    "token_logprobs": [
        -0.86915743,
        -0.35828337,
        -0.008103862,
        -0.00550173,
        -0.0014032064,
        -0.06300423,
    ],
    "top_logprobs": [
        {
            "\n\n": -0.86915743,
            "\n": -1.5560223,
            " string": -3.156324,
            " sentence": -3.2896361,
            " of": -3.7075934,
        },
        {
            "This": -0.35828337,
            "\n": -1.847051,
            "\n\n": -3.0329373,
            " This": -4.944918,
            "I": -4.9773717,
        },
        {
            " is": -0.008103862,
            " sentence": -5.2877955,
            "\n": -7.06451,
            " statement": -7.886844,
            "\n\n": -8.535557,
        },
        {
            " a": -0.00550173,
            " just": -6.2743807,
            " an": -6.3107214,
            " only": -7.745223,
            " not": -8.000508,
        },
        {
            " test": -0.0014032064,
            " sentence": -7.599525,
            " ": -8.814065,
            " te": -8.949878,
            " sample": -9.242383,
        },
        {
            ".": -0.06300423,
            ".\n": -3.285281,
            "<|endoftext|>": -4.3656716,
            " ": -5.737307,
            ",": -5.822476,
        },
    ],
    "text_offset": [18, 20, 24, 27, 29, 34],
}

class TestTokenTransfer(unittest.TestCase):
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
        target_token_log_probs = token_transfer(source_tokens, target_tokens, probs)
        assert np.allclose(np.sum(target_token_log_probs), np.sum(source_log_probs))

    def test_from_openai(self):
        target_token_logp = from_openai(data, data['tokens'])
        self.assertTrue(np.allclose(data['token_logprobs'], target_token_logp[1:]))

if __name__ == '__main__':
    unittest.main()
