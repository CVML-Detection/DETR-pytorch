import torch
import torch.nn.functional as F
from torch import nn, Tensor

import copy


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(num_layers=6, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = Decoder(num_layers=6, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1)
        

    def forward(self, src, query_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2,0,1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        # a = self.encoder(src)
        # b = self.decoder(tgt, a, query_embed)
        return 0


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers        
        self.layers = _get_clones(EncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self):

        return 0


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu


    def forward(self):
        return 0


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(DecoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self):

        return 0

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu


    def forward(self):
        return 0

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])