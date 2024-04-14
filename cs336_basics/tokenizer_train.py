import regex as re
from collections import Counter
import time

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
    #init vocab
    time1 = time.time()
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens) if token.encode("utf-8") not in vocab.values()})
    time2 = time.time()
    print('init vocab', time2-time1)

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
    time3 = time.time()
    print('pretokenize vocab', time3-time2)
    #get pairs
    pairs = Counter() #mapping of byte pairs to frequency
    impacted_tokens = {} #mapping of byte pairs to list of tokens that will be impacted by the merge
    for token, count in pretokenized_vocab.items():
        byte_pair_list_per_token = [(token[i], token[i+1]) for i in range(len(token) - 1)]
        for pair in byte_pair_list_per_token:
            pairs[pair] += count
            impacted_tokens = add_adjacent_token(impacted_tokens, pair, token)
    merges = []
    time4 = time.time()
    print('get pairs', time4-time3)
    while len(vocab.keys()) < vocab_size:
        # # Get most frequent byte pair that is also lexically greatest
        best_pair = return_best_pair(pairs, vocab) #ints
        pairs[best_pair] = 0 #set frequency to 0
        merges.append((vocab[best_pair[0]],vocab[best_pair[1]]))
        vocab[len(vocab)]= (vocab[best_pair[0]]+vocab[best_pair[1]]) #vocab[best_pair] is byte assoiated with best pair index
        updated_tokens = impacted_tokens[best_pair]
        for adj_token in updated_tokens:
            # breakpoint()
            unmerged_token = unmerged_to_merged_map[adj_token]
            time5 = time.time()
            new_merged_token = merge_token(unmerged_token, best_pair, vocab)
            time6 = time.time()
            unmerged_to_merged_map[adj_token] = new_merged_token #update the mapping of unmerged to merged tokens, mapped to int representation
            time7 = time.time()
            pairs, impacted_tokens = update_pairs_and_tokens(pairs, pretokenized_vocab, impacted_tokens, new_merged_token, adj_token, best_pair, vocab)
            time8 = time.time()
    time9 = time.time()
    print(time2-time1, time3-time2, time4-time3, time9-time4, time6-time5, time8-time7 )
    return vocab, merges

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

def merge_token(token, pair, vocab):
    '''
    merges the token based on the most frequent pair of byte indices
    token is byte pair, pair is tuple of indices, vocab is dictionary mapping int to bytes'''
    id =len(vocab)-1 #id of the new token that is formed from the pair merging
    token_merge = list(token) #turns bytes into their int representation
    i = 0 
    while i<len(token_merge)-1:
        t1, t2 = token_merge[i], token_merge[i+1]
        if (t1, t2) == pair:
            token_merge = token_merge[:i] + [id] + token_merge[i+2:] #new list of indices with the pair merged and replaced by last elem of vocab
        else:
            i+=1
    return token_merge

def update_pairs_and_tokens(pairs, pretokenized_vocab,  impacted_tokens, new_token, token, best_pair, vocab):
    '''
    updates the pairs and impacted tokens after a merge
    pairs is a mapping of byte pair ints to their frequency
    pretokenized_vocab is a mapping of byte tokens to their frequency
    impacted tokens is a mapping of ints to the byte tokens that will be impacted by the merge
        ie (78, 73): [b' UNIVERSE', b' NI']
    new_token is the merged token ints
    token is the byte representaiton of the token impacted by the merge
    best_pair is the indices of the pair of bytes that was merged
    vocab is a mapping of int to bytes'''
    # breakpoint() #refactor
    id = len(vocab) - 1
    for i in range(len(new_token)):
        if new_token[i] == id: #if we are at the part of the new token that is merging
            count = pretokenized_vocab.get(token, 0)
            if i>0:
                byte_before_merge = new_token[i-1]
                pairs[(byte_before_merge, best_pair[0])]-= count #decrement pair (which is inside the adj token) by how many times the merged token appears in the adjacent token
                pairs[(byte_before_merge, id)] +=count #add pair that accounts for the adjacent token + new merged token presence
                if (byte_before_merge, id) in impacted_tokens.keys():
                    if token not in impacted_tokens[(byte_before_merge, id)]:
                        impacted_tokens[(byte_before_merge, id)].append(token)
                else:
                    impacted_tokens[(byte_before_merge, id)] = [token]
            if i<len(new_token)-1:
                byte_after_merge = new_token[i+1]
                pairs[(best_pair[1], byte_after_merge)] -= count
                pairs[(id, byte_after_merge)]+= count
                if ((id, byte_after_merge)) in impacted_tokens.keys():
                    if token not in impacted_tokens[(id, byte_after_merge)]:
                        impacted_tokens[(id, byte_after_merge)].append(token)
                else:
                    impacted_tokens[(id, byte_after_merge)] = [token]

    return pairs, impacted_tokens