def bpe_tokenizer_training(input_path: str, 
                           vocab_size: int, 
                           special_tokens: list[str], 
                           vocab: dict[int, bytes], 
                           merges: list[tuple[bytes, bytes]]):
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
    #string to unicode character to byte with list(encoded)
    #150,000 unicode characters, represented with 256 bytes so not 1:1
    with open(input_path, 'r') as file:
        corpus = file.read()
    vocab = {i: bytes([ord(c)]) for i, c in enumerate(special_tokens)}
    #how do we initialize beyond the special tokens? should have 256 entries?