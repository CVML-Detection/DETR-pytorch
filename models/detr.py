import torch
import torch.nn.functional as F
from torch import nn

from .backbone import Backbone


class DETR(nn.Module):
    def __init__(self, num_classes=81, num_queries=100):
        super().__init__()
        # self.backbone = Backbone()
        # self.transformer = Transformer
        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self):
        return 0