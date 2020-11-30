import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import random

version = 'tf4'

class TransTransformer(nn.Module):
    def __init__(self, vocab, d_model, max_length=16, h=4, num_layers=2, dropout=0.01):
        super(TransTransformer, self).__init__()
        self.encoder = TransformerEn(vocab, d_model, max_length, h, num_layers, dropout)
        self.decoder = TransformerDe(vocab, d_model, max_length, h, num_layers, dropout)
    
    def forward(self, data):
        memory = self.encoder(data)
        output = self.decoder(memory)
        return output
    
    def loss(self, data, prod):
        vocb_size = prod.shape[-1]
        # loss = F.cross_entropy(prod.reshape(-1,vocb_size), data.view(-1), reduction="none")
        loss = F.nll_loss(prod.transpose(1, 2), data, reduction="none")
        loss = loss.sum()
        return loss
    
    def argmax(self, prod):
        return prod.argmax(2)


def get_lengths(tokens, eos_idx=3):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

class TransformerEn(nn.Module):
    def __init__(self, vocab, d_model, max_length, h=4, num_layers=4, dropout=0.01):
        super(TransformerEn, self).__init__()
        self.max_length = max_length
        self.eos_idx = 3
        self.pad_idx = 0
        self.embed = EmbeddingLayer(vocab, d_model, max_length, self.pad_idx)
        self.sos_token = nn.Parameter(torch.randn(d_model))
        self.encoder = Encoder(num_layers, d_model, vocab, h, dropout)
    
    def forward(self, inp_tokens):
        # inp_lengths = get_lengths(inp_tokens).to(inp_tokens.device)
        inp_lengths = torch.Tensor([self.max_length]*inp_tokens.shape[0]).to(inp_tokens.device)
        batch_size = inp_tokens.size(0)

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :self.max_length] >= inp_lengths.unsqueeze(-1)
        src_mask = src_mask.view(batch_size, 1, 1, self.max_length)

        enc_input = self.embed(inp_tokens, pos_idx[:, :self.max_length])
        memory = self.encoder(enc_input, src_mask)

        # return memory.view(batch_size, -1)
        return memory

class TransformerDe(nn.Module):
    def __init__(self, vocab, d_model, max_length, h=4, num_layers=4, dropout=0.01, encoder_as_decoder=False):
        super(TransformerDe, self).__init__()
        self.max_length = max_length
        self.eos_idx = 3
        self.pad_idx = 0
        self.embed = EmbeddingLayer(vocab, d_model, max_length, self.pad_idx)
        self.sos_token = nn.Parameter(torch.randn(d_model))
        if encoder_as_decoder:
            self.decoder = Encoder(num_layers, d_model, vocab, h, dropout)
        else:
            self.decoder = Decoder(num_layers, d_model, vocab, h, dropout)
    
    def forward(self, memory, gold_tokens=None, temperature=1.0):
        generate = gold_tokens is None
        batch_size = memory.size(0)
        # memory = memory.view(batch_size, self.max_length, -1)
        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(memory.device)
        tgt_mask = torch.ones((self.max_length, self.max_length)).to(memory.device)
        tgt_mask = (tgt_mask.tril() == 0).view(1, 1, self.max_length, self.max_length)
        sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)
        
        if not generate:
            dec_input = gold_tokens[:, :-1]
            max_dec_len = gold_tokens.size(1)
            dec_input_emb = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_len - 1])), 1)
            log_probs = self.decoder(
                dec_input_emb, memory, tgt_mask[:, :, :max_dec_len, :max_dec_len],
                temperature
            )
        else:
            
            log_probs = []
            next_token = sos_token
            prev_states = None
            
            for k in range(self.max_length):
                log_prob, prev_states = self.decoder.incremental_forward(
                    next_token, memory, tgt_mask[:, :, k:k+1, :k+1],
                    temperature,
                    prev_states
                )

                log_probs.append(log_prob)
                next_token = self.embed(log_prob.argmax(-1), pos_idx[:, k:k+1])
            log_probs = torch.cat(log_probs, 1)
            
        return log_probs
    
    def loss(self, data, prod):
        vocb_size = prod.shape[-1]
        loss = F.nll_loss(prod.transpose(1, 2), data, reduction="none")
        loss = loss.sum()
        return loss





class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)
            
        for layer in self.layers:
            y = layer(y, mask)

        return self.norm(y)
        
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, x, memory, tgt_mask, temperature):
        y = x

        assert y.size(1) == tgt_mask.size(-1)
        
        for layer in self.layers:
            y = layer(y, memory, tgt_mask)

        return self.generator(self.norm(y), temperature)
    def incremental_forward(self, x, memory, tgt_mask, temperature, prev_states=None):
        y = x

        new_states = []

                                            
        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, tgt_mask,
                prev_states[i] if prev_states else None
            )

            new_states.append(new_sub_states)
        
        new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]
        
        return self.generator(y, temperature), new_states
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = Linear(d_model, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)

class EmbeddingLayer0(nn.Module):
    def __init__(self, vocab, d_model, max_length, pad_idx):
        super(EmbeddingLayer0, self).__init__()
        self.token_embed = Embedding(vocab, d_model)
        self.pos_embed = Embedding(max_length, d_model)
        self.vocab_size = vocab
    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)
        return y
class EmbeddingLayer1(nn.Module):
    def __init__(self, vocab, d_model, max_length, pad_idx):
        super(EmbeddingLayer1, self).__init__()
        self.token_embed = Embedding(vocab, d_model-max_length)
        self.max_length = max_length
        self.d_model = d_model
        self.vocab_size = vocab
        self.norm = LayerNorm(d_model)
        self.pos_embed = self.pos_embed_onehot
    def pos_embed_onehot(self, pos):
        # embd = torch.zeros(list(pos.shape)+[self.max_length], device=pos.device)
        embd = F.one_hot(pos, self.max_length).float()
        return embd

    def forward(self, x, pos):
        # print("\n=========",x.shape, pos.shape) # torch.Size([200, 16]) torch.Size([200, 16])
        if len(x.size()) == 2:
            y = torch.cat([self.token_embed(x), self.pos_embed(pos)], -1)
        else:
            y = torch.cat([torch.matmul(x, self.token_embed.weight), self.pos_embed(pos)], -1)
        return y

EmbeddingLayer = EmbeddingLayer1
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer =  nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.pw_ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.src_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        x = self.sublayer[2](x, self.pw_ffn)
        return x

    def incremental_forward(self, x, memory, tgt_mask, prev_states=None):
        new_states = []
        m = memory

        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask))
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states  
   
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([Linear(d_model, d_model) for _ in range(3)])
        self.fc = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for x, l in zip((query, key, value), self.head_projs)]

        # query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        attn_feature, _ = self.scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)
        # return attn_concated

    def scaled_attention(self, query, key, value, mask):
        d_k = query.size(-1)
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))
        attn_weight = F.softmax(scores, -1)
        # scores = query.matmul(key.transpose(-2, -1))
        # scores.masked_fill_(mask, 0.)
        # attn_weight = scores

        attn_feature = attn_weight.matmul(value)

        return attn_feature, attn_weight
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )
        
    def forward(self, x):
        return self.mlp(x)

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)
    
def Linear(in_features, out_features, bias=False, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps,  elementwise_affine=False)
    return m
