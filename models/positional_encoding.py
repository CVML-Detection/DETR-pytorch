import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_pos_feats = 128    # (hidden dim // 2)
        self.temperature = 10000
        self.normalize = True
        self.scale = 2 * math.pi

    def forward(self):
        return 0