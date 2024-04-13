import regex as re
import json
from typing import Iterable, List, Optional, Iterator

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab.copy()
        self.merges = {tuple(merge): i for i, merge in enumerate(merges)}
        self.merges_dict = {k:v for v, k in enumerate(merges)}
        self.special_tokens = special_tokens
        self.inverse_vocab = {v: k for k, v in vocab.items()} #mapping byte to int
        self.special_tokens = special_tokens or []
    
    def encode(self, text: str):
        tokens = self.tokenize(text)

        merged_tokens = []
        for token in tokens:
            if token in self.special_tokens: 
                merged_tokens.extend([self.inverse_vocab[token.encode()]])
            else:
                merged_word = self.get_merged_word(token)
                merged_tokens.extend(merged_word)
        return merged_tokens
    
    def get_merged_word(self, token: str):
        token_ids_pre_merge = []
        for char in token:
            if char not in self.special_tokens:
                byte_str = char.encode("utf-8")
                for byte in byte_str:
                    byte_format = bytes([byte])
                    token_ids_pre_merge.append(self.inverse_vocab[byte_format])
        merges_remaining = 1
        while merges_remaining!=0:
            token_ids_pre_merge, merges_remaining = self.find_best_merge(token_ids_pre_merge)
        return token_ids_pre_merge
    
    def find_best_merge(self, token_ids):
        merges_remaining = 0
        token_ids= list(token_ids)
        i=0
        all_merges = []

        while i<len(token_ids)-1:
            token_1, token_2 = token_ids[i], token_ids[i+1]
            if (self.vocab[token_1], self.vocab[token_2]) in self.merges_dict:
                all_merges.append({
                    "index": i,
                    "merge": (token_1, token_2),
                    "merge_order": self.merges_dict[(self.vocab[token_1], self.vocab[token_2])]
                })
            i += 1
        if len(all_merges)>0:
            best_merge = min(all_merges, key=lambda x: x["merge_order"])
            i, t1, t2 = best_merge['index'], best_merge['merge'][0], best_merge['merge'][1]
            token_ids = token_ids[:i] + [self.inverse_vocab[self.vocab[t1]+self.vocab[t2]]] + token_ids[i+2:]
            merges_remaining += 1
        return token_ids, merges_remaining

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] :
        buffer = ""
        for section in iterable:
            buffer += section
            lines = buffer.split('\n')
            for line in lines[:-1]:
                yield from self.encode(line+"\n")
            buffer = lines[-1]
        if buffer:
            yield from self.encode(buffer)


    def decode(self, tokens: list[int]):
        bytes = [self.vocab.get(token_id, '�') for token_id in tokens]
        bytes_str = b''.join(bytes)
        return bytes_str.decode('utf-8', errors='replace')
    
    def tokenize(self, text: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_sorted = sorted(self.special_tokens, key = len, reverse=True)
        temp = {}
        for i, special_token in enumerate(special_sorted):
            text = text.replace(special_token, f' {i}111')
            temp[f' {i}111'] = special_token
        split_text = re.findall(PAT, text)
        final_tokens = [temp.get(token, token) for token in split_text]
        return final_tokens
    
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r') as vocab_file:
            vocab = {line.strip(): i for i, line in enumerate(vocab_file)}
        with open(merges_filepath, 'r') as merges_file:
            merges = [tuple(line.strip().split()) for line in merges_file]
        return Tokenizer(vocab, merges, special_tokens)
    