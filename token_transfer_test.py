import token_transfer as tt
import unittest
import json
import numpy as np


class TestTokenTransfer(unittest.TestCase):
    def setUp(self) -> None:
        with open("test-data/test-data-1.json", "r") as f:
            self.data_short = json.load(f)
        with open("test-data/test-data-2.json", "r") as f:
            self.data_long = json.load(f)
        with open("test-data/test-merge.json", "r") as f:
            self.merge_data = json.load(f)

        return super().setUp()

    @unittest.skip("")
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
        assert np.allclose(target_token_log_probs[1:], source_log_probs)

    def test_from_openai_identity(self):
        #vocab = set().union(*self.data_long["top_logprobs"])
        partial_vocab = [set(d.keys()) for d in self.data_long['top_logprobs']]
        target_token_logp = tt.token_transfer_from_openai_response(
            self.data_long, self.data_long["tokens"], target_vocab=[set()] + partial_vocab
        )
        print([sum([np.exp(v) for v in d.values()]) for d in self.data_long['top_logprobs']])
        expected = np.array(self.data_long["token_logprobs"])
        actual = np.array(target_token_logp[1:])
        print("DIFF IS")
        diff = abs(np.exp(actual) - np.exp(expected))
        diff[diff < 1e-6] = 0
        print(diff)

        self.assertTrue(np.allclose(sum(expected), sum(actual)))

    @unittest.skip("")
    def test_merging_simple_openai(self):
        target_token_logp = tt.token_transfer_from_openai_response(
            self.merge_data, self.merge_data["tokens"]
        )
        original = self.merge_data["token_logprobs"]
        self.assertTrue(np.allclose(sum(original), sum(target_token_logp[1:])))

    def test_fold_empty_nodes(self):
        root = tt.TrieNode(None)
        b = tt.TrieNode("b", prob=0.9)
        a0 = tt.TrieNode("a", prob=0.5)
        c0 = tt.TrieNode("c", prob=0.3)
        d0 = tt.TrieNode("d", prob=0.7)
        e0 = tt.TrieNode("e", prob=0.5)
        h0 = tt.TrieNode("h", prob=0.33)
        c0.add_child(e0)
        c0.add_child(h0)
        a0.add_child(c0)
        a0.add_child(d0)
        b.add_child(a0)

        empty = tt.TrieNode(tt.EMPTY_NODE_KEY, prob=0.5)
        b.add_child(empty)
        a1 = tt.TrieNode("a", prob=0.9)
        c1 = tt.TrieNode("c", prob=0.8)
        e1 = tt.TrieNode("e", prob=0.8)
        g1 = tt.TrieNode("g", prob=0.2)
        f1 = tt.TrieNode("f", prob=0.1)
        k1 = tt.TrieNode("k", prob=0.6)
        a1.add_child(c1)
        c1.add_child(e1)
        c1.add_child(k1)
        a1.add_child(g1)
        empty.add_child(a1)
        empty.add_child(f1)
        root.add_child(b)

        tt._fold_empty_nodes(root)

        b = root.children["b"]
        a = b.children["a"]
        f = b.children["f"]
        c = a.children["c"]
        d = a.children["d"]
        g = a.children["g"]
        e = c.children["e"]
        h = c.children["h"]
        k = c.children["k"]

        assert np.allclose(b.prob, 0.9)
        assert np.allclose(a.prob, 0.95)
        assert np.allclose(f.prob, 0.05)
        assert np.allclose(c.prob, (0.3 * 0.5 + 0.5 * 0.9 * 0.8) / a.prob)
        assert np.allclose(d.prob, 0.5 * 0.7 / a.prob)
        assert np.allclose(g.prob, (0.2 * 0.45) / a.prob)
        assert np.allclose(
            e.prob, (0.5 * 0.3 * 0.5 + 0.5 * 0.9 * 0.8 * 0.8) / (c.prob * a.prob)
        )
        assert np.allclose(h.prob, (0.5 * 0.3 * 0.33) / (c.prob * a.prob))
        assert np.allclose(k.prob, (0.5 * 0.9 * 0.8 * 0.6) / (c.prob * a.prob))

    @unittest.skip("")
    def test_from_openai(self):
        if self.data_short["tokens"] != [
            "\n\n",
            "This",
            " is",
            " a",
            " test",
            ".",
        ]:
            print("this test depends on the inputs, skipping.")
            return
        target_tokens = ["\n", "\n", "This is", " a", " test", "."]
        target_token_logp = tt.token_transfer_from_openai_response(
            self.data_short, target_tokens, verbose=False
        )
        actual = self.data_short["token_logprobs"]
        expected = target_token_logp[1:]
        self.assertTrue(np.allclose(sum(actual[:-1]), sum(expected[:-1])))
    @unittest.skip("")
    def test_from_openai_with_vocab(self):
        target_tokens = self.data_short["tokens"]
        vocab = set().union(*self.data_short["top_logprobs"])
        print(vocab)
        target_token_logp = tt.token_transfer_from_openai_response(
            self.data_short, target_tokens, verbose=False, target_vocab=vocab
        )
        actual = self.data_short["token_logprobs"]
        expected = target_token_logp[1:]
        print(actual)
        print(expected)
        self.assertTrue(np.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
