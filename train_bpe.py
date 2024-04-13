from cs336_basics.tokenizers import *
input_path = '/data/TinyStoriesV2-GPT4-train.txt'
vocab_size = 10000
special_tokens = ['<eos>']

vocab, merges = bpe_tokenizer_training(input_path, vocab_size, special_tokens)
print(vocab)
print(merges)
