from cs336_basics.tokenizer_train import *
import json

# input_path = '/data/TinyStoriesV2-GPT4-train.txt'
input_path = '/data/owt_train.txt'
vocab_size = 32000
special_tokens = ['<|endoftext|> ']

vocab, merges = bpe_tokenizer_training(input_path, vocab_size, special_tokens)
print(vocab)
print(merges)
with open('owt_vocab.json', 'w') as json_file:
    json.dump(vocab, json_file)

with open('owt_merge.txt', 'w') as file:
    for item in merges:
        file.write("%s\n" % item)