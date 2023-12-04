from typing import List, Tuple, Optional, Dict
import numpy as np

SPECIAL_TOKENS = {"<eos>"}
EMPTY_NODE_KEY = "<EMPTY>"


def _chars_or_special(string):
    if string in SPECIAL_TOKENS:
        return [string]
    return string


class TrieNode:
    def __init__(
        self,
        char: Optional[str] = None,
        is_word: bool = False,
        prob: Optional[float] = None,
    ):
        self.char = char
        self.is_word = is_word
        self.prob = prob
        self.children: Dict[str, TrieNode] = {}

    @property
    def log_prob(self) -> float:
        assert self.prob is not None
        return np.log(self.prob)

    def add_child(self, child: "TrieNode"):
        assert child.char not in self.children and child.char is not None
        self.children[child.char] = child

    def pprint_node(self, indents: Optional[List[int]] = None, flag: bool = False):
        if indents is None:
            indents = [0]

        indent_string = "|".join([" " * indent for indent in indents])
        info_string = f"{self.prob:.4f}" if self.prob is not None else ""
        char_string = self.char
        if self.char == "\n":
            char_string = "\\n"
        print(indent_string + f"{char_string} {info_string}")

        indents = indents[:]
        if flag:
            indents.append(1)
        else:
            indents[-1] += 2
        for child in self.children.values():
            child.pprint_node(indents, len(self.children) > 1)

    def __contains__(self, string: str) -> bool:
        if not len(string):
            return False
        if string in SPECIAL_TOKENS or len(string) == 1:
            return string in self.children and self.children[string].is_word
        c, rest = string[0], string[1:]
        if c not in self.children:
            return False
        return rest in self.children[c]

    def add_string(self, string):
        p = self
        for c in _chars_or_special(string):
            if c in p.children:
                p = p.children[c]
            else:
                child = TrieNode(c)
                p.add_child(child)
                p = child
        p.is_word = True

    def get_string(self, string: str) -> "TrieNode":
        """
        Returns the leaf node of the trie representing the string.
        Note that this does not handle the case where string spans
        across an empty node.
        """
        p = self
        for char in _chars_or_special(string):
            p = p.children[char]
        assert p.is_word
        return p


def _add_probs(root: TrieNode, probs: List[Tuple[str, float]]):
    tokens = [t[0] for t in probs]
    for token, prob in probs:
        if not len(token):
            root.add_child(TrieNode(EMPTY_NODE_KEY, prob=prob))
    tokens = [token for token in tokens if len(token)]

    if not all([token in root for token in tokens]):
        print([token for token in tokens if token not in root])
        root.pprint_node()
        raise AssertionError

    chars = set([_chars_or_special(token)[0] for token in tokens])
    for char in chars:
        assert char in root.children
        subset = [(token, prob) for token, prob in probs if token.startswith(char)]
        total_prob = sum(x[1] for x in subset)
        root.children[char].prob = total_prob
        cut_subset = [
            (token[token.index(char) + len(char) :], prob / total_prob)
            for token, prob in subset
        ]
        _add_probs(root.children[char], cut_subset)


def _fold_empty_nodes(node: TrieNode):
    if EMPTY_NODE_KEY in node.children:
        empty_node = node.children[EMPTY_NODE_KEY]
        empty_node_prob = empty_node.prob
        assert empty_node_prob is not None
        for key, child in empty_node.children.items():
            assert key not in node.children and child.prob is not None
            node.add_child(child)
            child.prob *= empty_node_prob
        del node.children[EMPTY_NODE_KEY]
    for child in node.children.values():
        _fold_empty_nodes(child)


def _combine_tries(
    root: TrieNode, tries: List[TrieNode], source_tokens: List[str]
) -> None:
    # Attach the tries together according to the actual sequence.
    cur = root.get_string(source_tokens[0])
    for i, token in enumerate(source_tokens[1:]):
        trie = tries[i]
        # Bypass the root nodes of each trie
        for child in trie.children.values():
            cur.add_child(child)
        leaf = trie.get_string(token)
        if EMPTY_NODE_KEY in leaf.children:
            leaf = leaf.children[EMPTY_NODE_KEY]
        cur = leaf


def token_transfer(
    source_tokens: List[str],
    target_tokens: List[str],
    all_probs: List[List[Tuple[str, float]]],
) -> List[float]:
    """
    Returns logprobs for every target token.
    """
    assert "".join(source_tokens) == "".join(target_tokens)
    string = "".join(source_tokens)
    assert len(all_probs) == len(source_tokens) - 1
    print("Source tokens", "|".join(source_tokens))
    print("Target tokens", "|".join(target_tokens))
    print(f"{string=}")

    root = TrieNode()
    root.add_string(source_tokens[0])
    _add_probs(root, [(source_tokens[0], 1.0)])

    # Make tries out of each top-k tokens.
    tries = []
    for i, token_probs in enumerate(all_probs):
        tokens = [tp[0] for tp in token_probs]
        assert any(token == source_tokens[i + 1] for token in tokens)

        trie = TrieNode()
        for token in tokens:
            trie.add_string(token)
        _add_probs(trie, token_probs)
        tries.append(trie)

    _combine_tries(root, tries, source_tokens)
    _fold_empty_nodes(root)

    root.pprint_node()

    cur = root
    target_token_log_probs = []
    for token in target_tokens:
        log_p = 0.0
        for s in _chars_or_special(token):
            cur = cur.children[s]
            assert cur.log_prob is not None
            log_p += cur.log_prob
        target_token_log_probs.append(log_p)
    return target_token_log_probs


def test():
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


def from_openai(response_logprobs, target_tokens):
    # add implicit bos token
    source_tokens: List[str] = ["bos"] + response_logprobs["tokens"]
    target_tokens = ["bos"] + target_tokens[:]
    probs_struct: List[List[Tuple[str, float]]] = []
    for response_token_logprobs in response_logprobs["top_logprobs"]:
        probs_struct.append(
            [(tok, np.exp(lp)) for tok, lp in response_token_logprobs.items()]
        )

    target_logp = token_transfer(source_tokens, target_tokens, probs_struct)
    print(np.sum(target_logp))
    print(np.sum(response_logprobs["token_logprobs"]))
    return target_logp


if __name__ == "__main__":
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

    target_tokens = ["\n\nThis", " is a", " test."]
    from_openai(data, target_tokens)
