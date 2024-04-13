from cs336_basics.tokenizers import *
import json

input_path = '/data/TinyStoriesV2-GPT4-train.txt'
vocab_size = 10000
special_tokens = ['<eos>']

vocab, merges = bpe_tokenizer_training(input_path, vocab_size, special_tokens)
# print(vocab)
# print(merges)
with open('tinystories_vocab.json', 'w') as json_file:
    json.dump(vocab, json_file)

with open('merge.txt', 'w') as file:
    for item in merges:
        file.write("%s\n" % item)