import token_transfer as tt
import unittest
import json
import numpy as np


class TestTokenTransfer(unittest.TestCase):
    def setUp(self) -> None:
        with open("test-data-1.json", "r") as f:
            self.data_short = json.load(f)
        with open("test-data-2.json", "r") as f:
            self.data_long = json.load(f)
        with open("test-merge.json", "r") as f:
            self.merge_data = json.load(f)

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
        target_token_logp = tt.from_openai_response(
            self.data_long, self.data_long["tokens"]
        )
        expected = self.data_long["token_logprobs"]
        target = target_token_logp[1:]
        self.assertTrue(
            np.allclose(sum(expected), sum(target), atol=0.001, rtol=0.0001)
        )

    def test_merging_simple_openai(self):
        target_token_logp = tt.from_openai_response(
            self.merge_data, self.merge_data["tokens"]
        )
        original = self.merge_data["token_logprobs"]
        self.assertTrue(np.allclose(sum(original), sum(target_token_logp[1:])))

    def test_merge(self):
        root = tt.TrieNode(None)
        b = tt.TrieNode("b", prob=0.9)
        a0 = tt.TrieNode("a", prob=0.5)
        c0 = tt.TrieNode("c", prob=0.3)
        d0 = tt.TrieNode("d", prob=0.7)
        a0.add_child(c0)
        a0.add_child(d0)
        b.add_child(a0)

        empty = tt.TrieNode(tt.EMPTY_NODE_KEY, prob=0.5)
        b.add_child(empty)
        a1 = tt.TrieNode("a", prob=0.9)
        c1 = tt.TrieNode("c", prob=0.8)
        g1 = tt.TrieNode("g", prob=0.2)
        f1 = tt.TrieNode("f", prob=0.1)
        a1.add_child(c1)
        a1.add_child(g1)
        empty.add_child(a1)
        empty.add_child(f1)
        root.add_child(b)
        root.pprint()
        tt._fold_empty_nodes(root)
        root.pprint()
        b = root.children["b"]
        a = b.children["a"]
        f = b.children["f"]
        c = a.children["c"]
        d = a.children["d"]
        g = a.children["g"]

        assert np.allclose(b.prob, 0.9)
        assert np.allclose(a.prob, 0.95)
        assert np.allclose(f.prob, 0.05)
        assert np.allclose(c.prob, (0.3 * 0.5 + 0.5 * 0.9 * 0.8) / 0.95)
        assert np.allclose(d.prob, 0.5 * 0.7 / 0.95)
        assert np.allclose(g.prob, (0.2 * 0.45) / 0.95)

    def test_from_openai(self):
        if self.data_long["tokens"] != [
            "\n\n",
            "This",
            " is",
            " a",
            " test",
            ".",
        ]:
            print("this test depends on the inputs, skipping.")
            return
        target_tokens = ["\n", "\n", "This is", " a", " test."]
        target_token_logp = tt.from_openai_response(self.data_long, target_tokens)

        self.assertTrue(
            np.allclose(
                sum(self.data_long["token_logprobs"]), sum(target_token_logp[1:])
            )
        )


if __name__ == "__main__":
    unittest.main()
