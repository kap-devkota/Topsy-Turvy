from __future__ import print_function,division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

class LastHundredEmbed(nn.Module):
    
    def forward(self, x):
        return x[:,:,-100:]

class IdentityEmbed(nn.Module):
    
    def forward(self, x):
        return x

class FullyConnectedEmbed(nn.Module):
    def __init__(self, nin, nout, dropout=0.5, activation=nn.ReLU()):
        super(FullyConnectedEmbed, self).__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout
        
        self.transform = nn.Linear(nin, nout)
        self.drop = nn.Dropout(p = self.dropout_p)
        self.activation = activation
        
    def forward(self, x):
        t = self.transform(x)
        t = self.activation(t)
        t = self.drop(t)
        return t
    
class LSTMEmbed(nn.Module):
    def __init__(self, nout, activation='ReLU', sparse=False, p=0.5):
        super(LSTMEmbed, self).__init__()
        self.activation = activation
        self.sparse = sparse
        self.p = p
        
        self.embedding = SkipLSTM(21, nout, 1024, 3)
        self.embedding.load_state_dict(torch.load(EMBEDDING_STATE_DICT))
        
        for param in self.embedding.parameters():
            param.requires_grad = False
        torch.nn.init.normal_(self.embedding.proj.weight)
        torch.nn.init.uniform_(self.embedding.proj.bias, 0, 0)
        self.embedding.proj.weight.requires_grad = True
        self.embedding.proj.bias.requires_grad = True
        
        self.activationDict = nn.ModuleDict({
            'None': IdentityEmbed(),
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid()
        })
        self.dropout = nn.Dropout(p=self.p)
        
        
    def forward(self, x):
        
        t = self.embedding(x)
        if self.activation: t = self.activationDict[self.activation](t)
        if self.sparse: t = self.dropout(t)

        return t
    
    def long_embed(self, x):
        return self.embedding.transform(x)

class SkipLSTM(nn.Module):
    def __init__(self, nin, nout, hidden_dim, num_layers, dropout=0, bidirectional=True):
        super(SkipLSTM, self).__init__()

        self.nin = nin
        self.nout = nout

        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()
        dim = nin
        for i in range(num_layers):
            f = nn.LSTM(dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
            self.layers.append(f)
            if bidirectional:
                dim = 2*hidden_dim
            else:
                dim = hidden_dim

        n = hidden_dim*num_layers + nin
        if bidirectional:
            n = 2*hidden_dim*num_layers + nin

        self.proj = nn.Linear(n, nout)

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def transform(self, x):
        one_hot = self.to_one_hot(x)
        hs =  [one_hot] # []
        h_ = one_hot
        for f in self.layers:
            h,_ = f(h_)
            #h = self.dropout(h)
            hs.append(h)
            h_ = h
        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
        return h

    def forward(self, x):
        one_hot = self.to_one_hot(x)
        hs = [one_hot]
        h_ = one_hot

        for f in self.layers:
            h,_ = f(h_)
            #h = self.dropout(h)
            hs.append(h)
            h_ = h

        if type(x) is PackedSequence:
            h = torch.cat([z.data for z in hs], 1)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = torch.cat([z for z in hs], 2)
            z = self.proj(h.view(-1,h.size(2)))
            z = z.view(x.size(0), x.size(1), -1)

        return z
