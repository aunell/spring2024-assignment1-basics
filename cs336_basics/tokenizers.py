import regex as re
from collections import Counter

def bpe_tokenizer_training(input_path: str, 
                           vocab_size: int, 
                           special_tokens: list[str]):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
                """
    with open(input_path, 'r', encoding='utf-8') as file:
            corpus = file.read()
    vocab = init_vocab(special_tokens)
    pretokenized_vocab = pretokenize_vocab(corpus, vocab)
    merges = []
    # pairs = get_pairs(pretokenized_vocab)
    while len(vocab.keys()) < vocab_size:
        pairs = get_pairs(pretokenized_vocab)
        best_pair = max(pairs, key=lambda pair: (pairs.get(pair), max(pair))) 
        pretokenized_vocab= merge_vocab(pretokenized_vocab, best_pair)
        merges.append(best_pair)
        vocab[len(vocab)]= best_pair[0]+best_pair[1]
    return (vocab, merges)

def init_vocab(special_tokens: list[str]):
    vocab={}
    vocab.update({i: (c.encode("utf-8")) for i, c in enumerate(special_tokens)})
    vocab.update({i+1: bytes([i]) for i in range(0, 256)})
    return vocab

def pretokenize_vocab(corpus: str, vocab: dict[int, bytes]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = re.findall(PAT, corpus)
    return {(tuple(vocab[ord(char)] for char in token if ord(char) in vocab)): count for token, count in Counter(tokens).items()}

def get_pairs(pretokenized_vocab: dict[tuple[int], int]):
    pairs = Counter()
    for token, count in pretokenized_vocab.items():
        for i in range(len(token) - 1):
            pairs[token[i], token[i + 1]] += count
    return pairs

def merge_vocab(pretokenize_dict: dict[tuple[bytes], int], pair: tuple[bytes, bytes]):
    new_dict = {}
    for key, value in pretokenize_dict.items():
            new_key = []
            i = 0
            while i < len(key):
                if key[i:i+2] == pair:
                    new_key.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_dict[tuple(new_key)] = value
    return new_dict

def reformat_vocab(vocab):
    for key, value in vocab.items():    
        print(value.decode("utf-8"))
    return {value.decode("utf-8"): key for key, value in vocab.items()}