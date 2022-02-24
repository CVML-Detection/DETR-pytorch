import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def forward(self):
        return 0