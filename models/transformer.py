import torch
import torch.nn.functional as F
from torch import nn, Tensor

import copy


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(num_layers=6, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.decoder = Decoder(num_layers=6, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1)

    def forward(self, src, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)                           # (b,256,38,38)->(1444,b,256)   src
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)               # (b,256,38,38)->(1444,b,256)   pos
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)         # (100,256)->(100,b,256)        query
        tgt = torch.zeros_like(query_embed)                             # (100,b,256)                   query(tensor)
        out_encoder = self.encoder(src, pos_embed)
        out_decoder = self.decoder(tgt, out_encoder, pos_embed, query_embed)
        return out_decoder.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers        
        self.layers = _get_clones(EncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        output = self.norm(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu


    def forward(self, src, pos):
        q = src + pos
        k = src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(DecoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, pos, query_pos):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)
            intermediate.append(self.norm(output))
        output = self.norm(output)
        intermediate.pop()
        intermediate.append(output)

        return torch.stack(intermediate)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

    def forward(self, tgt, memory, pos, query_pos):
        q = tgt + query_pos
        k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        tgt2 = self.multihead_attn(query=tgt+query_pos, key=memory+pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])