import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
# from ResBlock import ResBlock_v2 as ResBlock
from LayerTools import *

version = 'lstm4'

class TransLstm(nn.Module):
    def __init__(self, vocb_size, hidden_size):
        super(TransLstm, self).__init__()
        self.hidden_size = hidden_size
        self.vocb_size = vocb_size
        self.n_layer = 4
        self.n_direction = 2
        self.embedding = nn.Embedding(vocb_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, self.n_layer, batch_first=True, bidirectional=self.n_direction==2)
        self.decoder = nn.LSTM(hidden_size, hidden_size, self.n_layer, batch_first=True, bidirectional=self.n_direction==2)
        self.debedding = nn.Linear(hidden_size*self.n_direction, vocb_size, bias=False)
    
    def forward(self, data):
        b_size = data.shape[0]
        length = data.shape[1]
        device = data.device
        hidden = torch.zeros(self.n_layer*2, b_size, self.hidden_size).to(device)
        cell = torch.zeros(self.n_layer*2, b_size, self.hidden_size).to(device)

        embd = self.embedding(data)
        _, (hidden, cell) = self.encoder(embd, (hidden, cell))

        input = embd[:,0:1,:]
        sos = torch.zeros(b_size,1,self.vocb_size).to(device)
        sos[:,0,data[0,0]] = 1.
        outputs = [sos]
        for t in range(1, length):
            output, (hidden, cell) = self.decoder(input, (hidden, cell))
            output = self.debedding(output)
            outputs.append(output)
            input = self.embedding(output.argmax(2))
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def loss(self, data, prod):
        vocb_size = prod.shape[-1]
        loss = F.cross_entropy(prod.reshape(-1,vocb_size), data.view(-1), reduction="none")
        loss = loss.sum()
        return loss
    
    def argmax(self, prod):
        return prod.argmax(2)
