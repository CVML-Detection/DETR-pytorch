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

    def forward(self, feat):
        pos = torch.rand(2, 256, 38, 38).to(feat.device)    # FIXME) Positional Encoding은 기존 코드에서 mask 에만 적용, 의미 파악 필요
        return pos