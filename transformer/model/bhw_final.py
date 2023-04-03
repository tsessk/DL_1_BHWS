from collections import Counter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer
from torch import Tensor
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
import io
import os
from torch import Tensor
import math
import wandb
from tqdm import tqdm

# https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html - отсюда код обработки данных
# https://pytorch.org/tutorials/beginner/translation_transformer.html - отсюда код обучения
# Окружение:
# pytorch-ignite==0.4.11
# pytorch-lightning==1.9.3
# torch==1.13.0+cpu
# torchinfo==1.7.2
# torchmetrics==0.11.1
# torchtext==0.14.0
# wandb==0.13.10
# tqdm==4.64.1

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


def build_vocab(filepath, M_F = 1):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(string_.split())
    vocab_ = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=M_F)
    vocab_.set_default_index(vocab_['<unk>'])
    return vocab_


def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token] for token in raw_de.split()],
                                  dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in raw_en.split()],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))

    return data


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, train_loader):
    model.train()
    losses = 0

    for src, tgt in tqdm(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_loader))


def evaluate(model, val_loader):
    model.eval()
    losses = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_loader))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = torch.tensor([de_vocab[token] for token in src_sentence.split()],
                       dtype=torch.long).view(-1, 1)

    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(en_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


if __name__ == "__main__":
    input_path = './data/'
    train_p = ('train.de-en.de', 'train.de-en.en')
    val_p = ('val.de-en.de', 'val.de-en.en')
    test_p = 'test1.de-en.de'
    MIN_FREQUENCY = 7

    de_vocab = build_vocab(input_path + train_p[0], MIN_FREQUENCY)
    en_vocab = build_vocab(input_path + train_p[1], MIN_FREQUENCY)

    train_data = data_process([input_path + t for t in train_p])
    val_data = data_process([input_path + t for t in val_p])

    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<bos>']
    EOS_IDX = de_vocab['<eos>']

    BATCH_SIZE = 64

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=generate_batch, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=generate_batch, num_workers=2)

    torch.manual_seed(0)

    SRC_VOCAB_SIZE = len(de_vocab)
    TGT_VOCAB_SIZE = len(en_vocab)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    NUM_EPOCHS = 18

    os.environ["WANDB_API_KEY"] = '9c87219b80917d112c22c20a605d8dc6d6befdfd'
    run = wandb.init(project="bhw-2")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(transformer, optimizer, train_loader)
        val_loss = evaluate(transformer, val_loader)
        wandb.log({'train_loss': train_loss,
                   'val_loss': val_loss})
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}"))

    res_file = io.open('test1.de-en.en.txt', 'w')
    with io.open(input_path + "test1.de-en.de", encoding="utf8") as f:
        for string_ in f:
            res_file.write(translate(transformer, string_) + '\n')
