
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.data import Field, BucketIterator
from torchtext.datasets import IWSLT

import random
import numpy as np
import math

# this time the model will expect to get the batch dimension first
src_spec = Field(
    tokenize = "spacy",
    tokenizer_language="en",
    init_token = '<sos>',
    eos_token = '<eos>',
    lower = True,
    batch_first = True)

trg_spec = Field(tokenize = "spacy",
    tokenizer_language="xx", # no language-specific tokenizer available for cz
    init_token = '<sos>',
    eos_token = '<eos>',
    lower = True,
    batch_first = True)
            
train_data, valid_data, test_data = IWSLT.splits(exts = ('.en', '.cs'), fields = (src_spec, trg_spec),
  test='IWSLT16.TED.tst2013') # 2014 does not exist

len(train_data.examples), len(valid_data.examples), len(test_data.examples)

vars(train_data.examples[111])
vars(train_data.examples[11111])
vars(train_data.examples[111111])

src_spec.build_vocab(train_data, min_freq = 2)
trg_spec.build_vocab(train_data, min_freq = 2)

len(src_spec.vocab), len(trg_spec.vocab)

src_spec.vocab.stoi["hi"], trg_spec.vocab.stoi["ahoj"]

src_spec.vocab.itos[0], src_spec.vocab.itos[1], src_spec.vocab.itos[2], src_spec.vocab.itos[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 8

# Defines an iterator that batches examples of similar lengths together.
# Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch.
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    device = device)
    
src_len_in_tokens = (np.array([len(el.src) for el in train_iterator.data()]))
np.min(src_len_in_tokens), np.median(src_len_in_tokens), np.max(src_len_in_tokens)


trg_len_in_tokens = (np.array([len(el.trg) for el in train_iterator.data()]))
np.min(trg_len_in_tokens), np.median(trg_len_in_tokens), np.max(trg_len_in_tokens)
np.quantile(trg_len_in_tokens, 0.95)

batch = next(iter(train_iterator))
batch.src.shape, batch.trg.shape


######################################################################################

class Encoder(nn.Module):
    def __init__(self, num_input_features, embedding_dim, n_heads, hidden_dim, n_layers, max_length, dropout):
        super(Encoder, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_input_features, embedding_dim)
        # learn positional encoding
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        # one module of multihead attention, feedforward networks and layer norms
        # d_model: number of expected features in the input (here: embedding dimension)
        encoder_layers = TransformerEncoderLayer(embedding_dim, n_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # bs * src len
        # repeat vector 0 ... 35 once for every batch item
        # input for pos_embedding
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # bs * src len * hidden dim
        src = (self.embedding(src) * math.sqrt(self.embedding_dim)) + self.pos_embedding(pos)
        # apply transformer stack
        output = self.transformer_encoder(src)
        # bs * src len * hidden dim
        return output

num_input_features = len(src_spec.vocab)
embedding_dim = 256 # embedding dimension
max_length = 100 # max number of positions to encode
hidden_dim = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
n_heads = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
encoder = Encoder(num_input_features, embedding_dim, n_heads, hidden_dim, n_layers, max_length, dropout).to(device)

src = batch.src
encoder(batch.src)











##################################################################################

# now do it using nn.Transformer
torch.nn.Transformer(d_model=512,
n_heads=8,
num_encoder_layers=6,
num_decoder_layers=6,
dim_feedforward=2048,
dropout=0.1,
activation='relu',
custom_encoder=None,
custom_decoder=None)

