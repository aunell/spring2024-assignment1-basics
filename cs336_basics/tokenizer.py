import regex as re
import json
from typing import Iterable, List, Optional, Iterator

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.inverse_vocab = {v: k for k, v in vocab.items()} #mapping byte to int
    
    def encode(self, text: str):
        tokens = self.tokenize(text)

        merged_tokens = []
        for token in tokens:
            if token in self.special_tokens:    
                breakpoint()
                merged_tokens.append(self.inverse_vocab[token.encode()])
            else:
                merged_word = self.get_merged_word(token)
                merged_tokens.append(merged_word)
        return merged_tokens
    
    def get_merged_word(self, token: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        #token is a string
        #return list of token ids post merge

        #tokenize word
        token_ids_pre_merge = []
        for char in token:
            if char not in self.special_tokens:
                byte_str = char.encode("utf-8")
                for byte in byte_str:
                    byte_format = bytes([byte])
                    token_ids_pre_merge.append(self.inverse_vocab[byte_format])
        merges_remaining = 1
        while merges_remaining!=0:
            token_ids_pre_merge, merges_remaining = self.find_best_merge(token_ids_pre_merge, vocab, merges)
        return token_ids_pre_merge
    
    def find_best_merge(self, token_ids, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        merges_remaining = 0
        token_ids = list(token_ids)
        i=0
        all_merges = []
        while i<len(token_ids)-1:
            if (token_ids[i], token_ids[i+1]) in merges:
                token_ids[i] = vocab[(token_ids[i], token_ids[i+1])]
                token_ids.pop(i+1)
                merges_remaining+=1
            i+=1






    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] :
        pass


    def decode(self, tokens: list[int]):
        return "".join([self.vocab[token].decode("utf-8") for token in tokens])
    
    def tokenize(self, text: str):
        return None
    
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as file:
            vocab = json.load(file)
        with open(merges_filepath, 'r') as file:    
            merges = file.readlines()
        return Tokenizer(vocab, merges, special_tokens)
    