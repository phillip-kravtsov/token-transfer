from typing import List, Tuple, Optional, Dict
import numpy as np

SPECIAL_TOKENS = {"<eos>", "<bos>", "<|endoftext|>"}
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

    def pprint(
        self,
        indents: Optional[List[int]] = None,
        flag: bool = False,
        compact=True,
        depth=None,
    ):
        if depth == 0:
            return
        if indents is None:
            indents = [0]

        indent_string = "|".join([" " * indent for indent in indents])
        info_string = f"{self.prob:.4f}" if self.prob is not None else ""
        char_string = {"\n": "\\n", "\t": "\\t", " ": "' '"}.get(
            self.char, str(self.char)
        )
        print(indent_string + f"{char_string} {info_string}")
        indents = indents[:]
        width = 1 if compact else len(char_string)
        if flag:
            indents.append(width)
        else:
            indents[-1] += 1 + width
        for child in sorted(self.children.values(), key=lambda x: -(x.prob or 0)):
            child.pprint(
                indents,
                len(self.children) > 1,
                depth=depth - 1 if depth is not None else None,
            )

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

    assert all([token in root for token in tokens])

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


def _merge_trees(
    first: TrieNode,
    second: TrieNode,
    first_base_log_prob: float,
    second_base_log_prob: float,
) -> None:
    """
    Mutates `first` to be the merge of the trie nodes.
    first_base_prob and second_base_prob indicate the likelihood of the sequence before
    the first and second nodes respectively.
    Caller is responsible for cleaning up `second`.
    """
    assert first.char == second.char
    assert second.prob is not None
    assert first.prob is not None
    original_first_log_prob = first.log_prob
    first.prob = np.exp(first_base_log_prob + first.log_prob) + np.exp(
        second_base_log_prob + second.log_prob
    )

    for key in set().union(first.children, second.children):
        if key in first.children and key in second.children:
            _merge_trees(
                first.children[key],
                second.children[key],
                original_first_log_prob + first_base_log_prob,
                second_base_log_prob + second.log_prob,
            )
            first.children[key].prob /= first.prob

        elif key in first.children:
            child = first.children[key]
            assert child.prob is not None
            child.prob = np.exp(
                child.log_prob
                + original_first_log_prob
                + first_base_log_prob
                - first.log_prob
            )

        elif key in second.children:
            child = second.children[key]
            assert child.prob is not None
            child.prob = np.exp(
                child.log_prob + second.log_prob + second_base_log_prob - first.log_prob
            )
            first.add_child(child)


def _fold_empty_nodes(node: TrieNode):
    if EMPTY_NODE_KEY in node.children:
        empty_node = node.children[EMPTY_NODE_KEY]
        empty_node_prob = empty_node.prob
        assert empty_node_prob is not None
        if empty_node_prob == 1.0:
            for key, empty_node_child in empty_node.children.items():
                # assert key not in node.children and empty_node_child.prob is not None
                node.add_child(empty_node_child)
        else:
            keys_to_merge = []
            for key, empty_node_child in empty_node.children.items():
                if key in node.children:
                    keys_to_merge.append(key)

            for key in keys_to_merge:
                _merge_trees(
                    node.children[key],
                    empty_node.children[key],
                    0.0,
                    empty_node.log_prob,
                )
                del empty_node.children[key]

            for key, child in empty_node.children.items():
                if key not in keys_to_merge:
                    child.prob *= empty_node_prob
                    node.add_child(child)
        del node.children[EMPTY_NODE_KEY]
    for child in node.children.values():
        _fold_empty_nodes(child)


def _combine_tries(
    root: TrieNode, tries: List[TrieNode], source_tokens: List[str]
) -> None:
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


def build_prob_trie(
    source_tokens: List[str], all_probs: List[List[Tuple[str, float]]]
) -> TrieNode:
    root = TrieNode()
    root.add_string(source_tokens[0])
    _add_probs(root, [(source_tokens[0], 1.0)])

    tries: List[TrieNode] = []
    for i, token_probs in enumerate(all_probs):
        tokens = [tp[0] for tp in token_probs]
        assert any(token == source_tokens[i + 1] for token in tokens)
        trie = TrieNode()
        for token in tokens:
            trie.add_string(token)
        _add_probs(trie, token_probs)
        tries.append(trie)

    _combine_tries(root, tries, source_tokens)
    return root


def token_transfer(
    source_tokens: List[str],
    target_tokens: List[str],
    all_probs: List[List[Tuple[str, float]]],
    verbose=False,
) -> List[float]:
    """
    This function returns the log probabilities for each target token given the source tokens and
    the top-k log probabilities at each position.

    Args:
        source_tokens (List[str]): The source tokens.
        target_tokens (List[str]): The target tokens.
        all_probs (List[List[Tuple[str, float]]]): The top-k log probabilities at each position.
        verbose (bool, optional): If True, prints verbose output. Defaults to False.

    Returns:
        List[float]: The log probabilities for each target token.

    Note:
        The first element will always be 0.0, as we do not have likelihoods over the beginning of
        sentence (bos) token. The function expects the beginning of sentence/initial tokens to be
        passed in since they could, in theory, be tokenized differently by the tokenizer.
        The all_probs[i] refers to the likelihoods at the source[i+1].
    """
    assert "".join(source_tokens) == "".join(target_tokens)
    string = "".join(source_tokens)
    assert len(all_probs) == len(source_tokens) - 1

    root = build_prob_trie(source_tokens, all_probs)
    if verbose:
        print("Source tokens", "|".join(source_tokens))
        print("Target tokens", "|".join(target_tokens))
        print(f"{string=}")
        root.pprint()
    _fold_empty_nodes(root)
    if verbose:
        root.pprint()

    target_token_log_probs = []
    cur = root
    for token in target_tokens:
        log_p = 0.0
        for s in _chars_or_special(token):
            cur = cur.children[s]
            log_p += cur.log_prob
        target_token_log_probs.append(log_p)
    return target_token_log_probs


def from_openai_response(
    response_logprobs, target_tokens: List[str], verbose: bool = False
) -> List[float]:
    source_tokens: List[str] = ["<bos>"] + response_logprobs["tokens"]
    target_tokens = ["<bos>"] + target_tokens[:]
    probs_struct: List[List[Tuple[str, float]]] = []
    for response_token_logprobs in response_logprobs["top_logprobs"]:
        probs_struct.append(
            [(tok, np.exp(lp)) for tok, lp in response_token_logprobs.items()]
        )
    target_logp = token_transfer(
        source_tokens, target_tokens, probs_struct, verbose=verbose
    )
    return target_logp
