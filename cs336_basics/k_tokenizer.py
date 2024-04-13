import regex as re
from collections import Counter
from tqdm import tqdm

# class Tokenizer():

def merge_token(token, pair, new_merged_bp_id, vocab):
    #TODO
    id =len(vocab)-1
    new_merged_token = list(token)  # Convert to list if not already, ensures a new list is created
    i = 0  # Start from the first token

    while i < len(new_merged_token) - 1:  # Iterate until the second last token
        token_1, token_2 = new_merged_token[i], new_merged_token[i+1]

        # Check if the current and next token form the most frequent pair
        if (token_1, token_2) == pair:
            # Replace the two tokens with the new merged byte pair ID
            new_merged_token = new_merged_token[:i] + [id] + new_merged_token[i+2:]
            
            # Do not increment i, as the next token now shifts to the current position
        else:
            # If no merge, move to the next token
            i += 1
    # print(f"{token_ids_to_string(current_merged_token, vocab)} -> {token_ids_to_string(new_merged_token, vocab)}")
    return new_merged_token

def update_pairs_and_tokens(pairs, pretokenized_vocab,  impacted_tokens, new_token, token, merged_id1, merged_id2, id_pre, vocab):
    #TODO
    best_pair = merged_id1, merged_id2
    updates = []
    id = len(vocab) - 1
     # Define the variable "merged_token"
    # breakpoint()
    for i in range(len(new_token)):
        if new_token[i] == id:
            preceeding_byte = new_token[i-1] if i > 0 else None
            following_byte = new_token[i+1] if i < len(new_token) - 1 else None

            freq = pretokenized_vocab.get(token, 0)

            if preceeding_byte:
                pairs[(preceeding_byte, best_pair[0])] -= freq
                pairs[(preceeding_byte, id)] += freq
                
                if (preceeding_byte, id) in impacted_tokens.keys():
                    if token not in impacted_tokens[(preceeding_byte, id)]:
                        impacted_tokens[(preceeding_byte, id)].append(token)
                else:
                    impacted_tokens[(preceeding_byte, id)] = [token]
            if following_byte:
                pairs[(best_pair[1], following_byte)] -= freq
                pairs[(id, following_byte)] += freq

                if (id, following_byte) in impacted_tokens.keys():
                    if token not in impacted_tokens[(id, following_byte)]:
                        impacted_tokens[(id, following_byte)].append(token)
                else:
                    impacted_tokens[(id, following_byte)] = [token]

    return pairs, impacted_tokens

def return_best_pair(pairs, vocab):
    max_value = max(pairs.values())
    best_pairs = [pair for pair, value in pairs.items() if value == max_value]
    if len(best_pairs) == 1:
        return best_pairs[0]
    else: #tie break
        best_byte_pairs_byte = [(vocab[pair[0]], vocab[pair[1]]) for pair in best_pairs] #byte pairs in bytes instead of ints
        best_byte_pair  = max(best_byte_pairs_byte) #find best best byte pair lexically
        best_byte_pair_final = next(pair for pair, byte_pair in zip(best_pairs, best_byte_pairs_byte) if byte_pair == best_byte_pair)
        return best_byte_pair_final

def add_adjacent_token(impacted_tokens, pair, token):
    if pair in impacted_tokens.keys():
        if token not in impacted_tokens[pair]:
            impacted_tokens[pair].append(token)
            #add that the given token will be impacted by the merge
    else:
        impacted_tokens[pair] = [token]
    return impacted_tokens

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
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens) if token.encode("utf-8") not in vocab.values()})

    # Initialize pre-tokenized frequencies from the training corpus
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_frequencies = Counter()
    print(f"")
    print(f"Beginning pretokenization of {input_path}...")
    with open(input_path, 'rb') as f:
        for line in tqdm(f,desc="Pretokenizing"):
            decoded_line = line.decode("utf-8") # decode line so we can pretokenize using string regex
            tokens = re.findall(pretokenization_pattern, decoded_line)
            tokens_as_bytes = [token.encode() for token in tokens]
            token_frequencies.update(tokens_as_bytes) # this should store tuple[byte]:count
    print(f"Pretokenizing done. Found {len(token_frequencies)} unique tokens.")
    
    unmerged_token_to_merged_token_mapping = {token: token for token in token_frequencies.keys()} # maps token to its current merged token

    # We will now go through these pretokenized counts to compute our byte pair counts
    byte_pair_frequencies = Counter()
    byte_pair_associated_tokens = {}
    total_tokens = len(token_frequencies.items())
    for token, freq in tqdm(token_frequencies.items(), total=total_tokens, desc="Computing byte pair frequencies"):
        byte_pairs_in_token = [(token[i], token[i+1]) for i in range(len(token) - 1)]
        for pair in byte_pairs_in_token:
            byte_pair_frequencies[pair] += freq
            byte_pair_associated_tokens = add_adjacent_token(byte_pair_associated_tokens, pair, token)
    print(f"Byte pair frequencies computed. Found {len(byte_pair_frequencies)} unique byte pairs.")

    # Now, we begin the process of merging
    merges = []
    total_merges_allowed = vocab_size - len(vocab)
    for _ in tqdm(range(total_merges_allowed), desc="Merging byte pairs"):  # Wrap the range with tqdm for progress bar

        # Get most frequent byte pair that is also lexically greatest
        most_frequent_pair = return_best_pair(byte_pair_frequencies, vocab)

        # Merge the pair and add this merge to our vocab
        new_merged_bytes = (vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]])
        vocab[len(vocab)] = new_merged_bytes
        merges.append((vocab[most_frequent_pair[0]],vocab[most_frequent_pair[1]]))

        # Remove this pair from our dictionary
        byte_pair_frequencies[most_frequent_pair] = 0

        # Get the words containing this pair so that we can update frequencies accordingly
        associated_word_tokens = byte_pair_associated_tokens[most_frequent_pair]

        # Update the dictionary frequencies to contain pairs containing the merged tokens
        for associated_word_token in associated_word_tokens:

            # First, go through and merge all instances of the most frequent pair in our associated token
            associated_merged_token = unmerged_token_to_merged_token_mapping[associated_word_token]
            new_merged_token = merge_token(associated_merged_token, most_frequent_pair, len(vocab)-1, vocab)
            unmerged_token_to_merged_token_mapping[associated_word_token] = new_merged_token

            # Now that we've merged our token properly, we can compute new byte pair frequencies
            byte_pair_frequencies, byte_pair_associated_tokens = update_pairs_and_tokens(byte_pair_frequencies, token_frequencies, byte_pair_associated_tokens, new_merged_token, associated_word_token, most_frequent_pair[0], most_frequent_pair[1], len(vocab)-1, vocab)

    return vocab, merges

def bpe_tokenizer2_training(input_path: str, 
                           vocab_size: int, 
                           special_tokens: list[str]):
        #init vocab
    vocab={}
    vocab.update({i: bytes([i]) for i in range(256)})
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens) if token.encode("utf-8") not in vocab.values()})

    #pretokenize vocab
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenized_vocab = Counter()
    with open(input_path, 'rb') as f:
        for line in f:
            d_line = line.decode("utf-8")
            untokenized_words = re.findall(PAT, d_line)
            tokenized_words = [token.encode() for token in untokenized_words]
            pretokenized_vocab.update(tokenized_words)

    unmerged_to_merged_map = {token: token for token in pretokenized_vocab.keys()} #byte to byte

    #get pairs
    pairs = Counter() #mapping of byte pairs to frequency
    impacted_tokens = {} #mapping of byte pairs to list of tokens that will be impacted by the merge
    for token, count in pretokenized_vocab.items():
        byte_pair_list_per_token = [(token[i], token[i+1]) for i in range(len(token) - 1)]
        for pair in byte_pair_list_per_token:
            pairs[pair] += count
            impacted_tokens = add_adjacent_token(impacted_tokens, pair, token)
            #pairs is a mapping of int pairs to their frequency
            #impacted tokens is a mapping of int pairs to the byte tokens that will be impacted by the merge
    #merge pairs
    merges = []
    while len(vocab.keys()) < vocab_size:
        best_pair = return_best_pair(pairs, vocab)
        pairs[best_pair] = 0 #set frequency to 0
        merges.append(best_pair)
        vocab[len(vocab)]= best_pair[0]+best_pair[1]
        #TODO, update pairs, pretokenized_vocab 
        updated_tokens = impacted_tokens[best_pair]
        for token in updated_tokens:
            token = unmerged_to_merged_map[token]
            new_token = merge_token(token, best_pair, 0, vocab)
            unmerged_to_merged_map[token] = new_token
            pairs, impacted_tokens = update_pairs_and_tokens(pairs, pretokenized_vocab, new_token, impacted_tokens, token, best_pair, 0, 0, vocab)

    return vocab, merges