
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
    batch_first = True,
    fix_length=100
    )

trg_spec = Field(tokenize = "spacy",
    tokenizer_language="xx", # no language-specific tokenizer available for cz
    init_token = '<sos>',
    eos_token = '<eos>',
    lower = True,
    batch_first = True,
    fix_length=100
    )
            
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

src_pad_idx = src_spec.vocab.stoi["<pad>"]
trg_pad_idx = trg_spec.vocab.stoi["<pad>"]

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
    def forward(self, src, src_key_padding_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # bs * src len
        # repeat vector 0 ... 35 once for every batch item
        # input for pos_embedding
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # bs * src len * hidden dim
        src = (self.embedding(src) * math.sqrt(self.embedding_dim)) + self.pos_embedding(pos)
        # apply transformer stack
        src = torch.transpose(src, 1, 0)
        output = self.transformer_encoder(src, src_key_padding_mask = src_key_padding_mask)
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
encoder_outputs = encoder(batch.src, src_key_padding_mask = src != src_pad_idx)
encoder_outputs.size()


class Decoder(nn.Module):
    def __init__(self, num_output_features, embedding_dim, n_heads, hidden_dim, n_layers, max_length, dropout):
        super(Decoder, self).__init__()
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_output_features, embedding_dim)
        # learn positional encoding
        self.pos_embedding = nn.Embedding(max_length, embedding_dim)
        decoder_layers = TransformerDecoderLayer(embedding_dim, n_heads, hidden_dim, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers)
        self.fc = nn.Linear(hidden_dim, num_output_features)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.pos_embedding.weight.data.uniform_(-initrange, initrange)
    def forward(self, trg, encoder_outputs, tgt_mask, tgt_key_padding_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # bs * trg len
        # input for pos_embedding
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # bs * trg len * hidden dim
        trg = (self.embedding(trg) * math.sqrt(self.embedding_dim)) + self.pos_embedding(pos)
        # apply transformer stack
        # bs * trg len * hidden dim
        trg = torch.transpose(trg, 1, 0)
        output = self.transformer_decoder(trg, encoder_outputs,
          tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        output = self.fc(output)
        return output


num_output_features = len(trg_spec.vocab)
decoder = Decoder(num_output_features, embedding_dim, n_heads, hidden_dim, n_layers, max_length, dropout).to(device)

trg = batch.trg
decoded = decoder(trg, encoder_outputs, tgt_mask = torch.tril(torch.ones((trg.size()[1], trg.size()[1]), device = device)).bool(), tgt_key_padding_mask = trg != trg_pad_idx)
decoded.size()

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def make_src_key_padding_mask(self, src):
        # bs * src_len
        src_mask = src == src_pad_idx
        return src_mask
    def make_trg_key_padding_mask(self, trg):
        # bs * trg_len
        trg_mask = trg == trg_pad_idx
        return trg_mask
    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        mask = (torch.triu(torch.ones(trg_len, trg_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    def forward(self, src, trg):
        encoded = self.encoder(src, self.make_src_key_padding_mask(src))
        output = self.decoder(trg, encoded,  self.make_trg_mask(trg), self.make_trg_key_padding_mask(trg))
        return output

model = Seq2Seq(encoder, decoder, device).to(device)
model(src, trg)


learning_rate = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
pad_idx = trg_spec.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        # bs * (trg len - 1) * num_output_features
        output = torch.transpose(output, 1, 0)
        output_dim = output.shape[-1]
        # (bs * (trg len - 1)) * num_output_features
        output = output.contiguous().view(-1, output_dim)
        # (bs * trg len)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:,:-1])
            output = torch.transpose(output, 1, 0)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_key_padding_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_key_padding_mask = model.make_trg_key_padding_mask(trg_tensor)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, trg_key_padding_mask)
            output = torch.transpose(output, 1, 0)
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]: break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]

n_epochs = 9
clip = 1

example_idx = [11, 77, 133, 241, 333, 477, 555, 777]

for epoch in range(n_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluate(model, valid_iterator, criterion)
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    for i in range(8):
        example_src = vars(train_data.examples[example_idx[i]])['src']
        example_trg = vars(train_data.examples[example_idx[i]])['trg']
        translation = translate_sentence(example_src, src_spec, trg_spec, model, device)
        src_sentence = " ".join(i for i in example_src)
        target_sentence = " ".join(i for i in example_trg)
        translated_sentence = " ".join(i for i in translation)
        print("Source: " + src_sentence)
        print("Target: " + target_sentence)
        print("Predicted: " + translated_sentence + "\n")


    
    

